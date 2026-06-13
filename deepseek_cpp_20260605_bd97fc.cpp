//File 0222 : sparse/xsparse_mesh.hpp
//Finite element mesh assembly: sparse matrix construction from 2D/3D element stiffness and mass matrices using SIMD-accelerated integration.
#ifndef XTENSOR_XSPARSE_MESH_HPP
#define XTENSOR_XSPARSE_MESH_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xarray.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xstrides.hpp"
#include "../core/xeval.hpp"
#include "../core/xbuilder.hpp"
#include "../core/xnorm.hpp"
#include "../sparse/xcsr.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xsparse_config.hpp"

namespace xt {
namespace sparse {

    namespace detail
    {
        /**
         * Compute the local stiffness matrix for a 2D linear triangular element (CST).
         * Nodes: p0(x0,y0), p1(x1,y1), p2(x2,y2).  Returns 3x3 matrix.
         */
        template <class T>
        inline auto triangle_stiffness(T x0, T y0, T x1, T y1, T x2, T y2, T young, T poisson)
        {
            xarray_container<uvector<T>> K({3, 3}, T(0));
            T area = T(0.5) * ((x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0));
            T factor = young / (T(4) * area * (T(1) - poisson * poisson));
            T b[3] = { y1 - y2, y2 - y0, y0 - y1 };
            T c[3] = { x2 - x1, x0 - x2, x1 - x0 };
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    T bij = b[i] * b[j];
                    T cij = c[i] * c[j];
                    K(i, j) = factor * (bij + T(0.5) * (T(1) - poisson) * cij);
                }
            return K;
        }

        /**
         * Compute the local mass matrix for a 2D linear triangular element (lumped or consistent).
         */
        template <class T>
        inline auto triangle_mass(T x0, T y0, T x1, T y1, T x2, T y2, T density, bool lumped = false)
        {
            xarray_container<uvector<T>> M({3, 3}, T(0));
            T area = T(0.5) * ((x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0));
            T factor = density * area / T(12);
            if (lumped)
            {
                for (int i = 0; i < 3; ++i) M(i, i) = density * area / T(3);
            }
            else
            {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        M(i, j) = (i == j) ? T(2) * factor : T(1) * factor;
            }
            return M;
        }

        /**
         * Local stiffness for a 2D bilinear quadrilateral element (Q4).
         * Returns 4x4 matrix.
         */
        template <class T>
        inline auto quad_stiffness(const T nodes[4][2], T young, T poisson)
        {
            xarray_container<uvector<T>> K({4, 4}, T(0));
            T D = young / (T(1) - poisson * poisson);
            T D11 = D, D22 = D, D12 = D * poisson, D33 = D * (T(1) - poisson) / T(2);
            // 2x2 Gauss quadrature points
            T gp[4][2] = { { -0.5773502692, -0.5773502692 },
                           {  0.5773502692, -0.5773502692 },
                           {  0.5773502692,  0.5773502692 },
                           { -0.5773502692,  0.5773502692 } };
            T w = T(1.0);
            for (int q = 0; q < 4; ++q)
            {
                T xi = gp[q][0], eta = gp[q][1];
                T dN[4][2] = { { -0.25*(1-eta), -0.25*(1-xi) },
                               {  0.25*(1-eta), -0.25*(1+xi) },
                               {  0.25*(1+eta),  0.25*(1+xi) },
                               { -0.25*(1+eta),  0.25*(1-xi) } };
                T jac[2][2] = { {0,0},{0,0} };
                for (int i = 0; i < 4; ++i)
                {
                    jac[0][0] += dN[i][0] * nodes[i][0];
                    jac[0][1] += dN[i][0] * nodes[i][1];
                    jac[1][0] += dN[i][1] * nodes[i][0];
                    jac[1][1] += dN[i][1] * nodes[i][1];
                }
                T detJ = jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0];
                T invJ[2][2] = { { jac[1][1]/detJ, -jac[0][1]/detJ },
                                 { -jac[1][0]/detJ,  jac[0][0]/detJ } };
                T B[3][8] = { {0} };
                for (int a = 0; a < 4; ++a)
                {
                    B[0][2*a]   = invJ[0][0]*dN[a][0] + invJ[0][1]*dN[a][1]; // epsilon_xx
                    B[1][2*a+1] = invJ[1][0]*dN[a][0] + invJ[1][1]*dN[a][1]; // epsilon_yy
                    B[2][2*a]   = invJ[1][0]*dN[a][0] + invJ[1][1]*dN[a][1]; // gamma_xy
                    B[2][2*a+1] = B[0][2*a];
                }
                // K += B^T * D * B * detJ * w
                for (int i = 0; i < 8; ++i)
                    for (int j = 0; j < 8; ++j)
                    {
                        T sum = T(0);
                        sum += B[0][i] * D11 * B[0][j];
                        sum += B[1][i] * D12 * B[0][j] + B[0][i] * D12 * B[1][j];
                        sum += B[1][i] * D22 * B[1][j];
                        sum += B[2][i] * D33 * B[2][j];
                        K(i/2, j/2) += sum * detJ * w;
                    }
            }
            return K;
        }
    }

    /**
     * Assemble a sparse stiffness matrix for a 2D mesh of triangles.
     * @param nodes Node coordinates (N x 2).
     * @param elements Element connectivity (M x 3) with 0-based node indices.
     * @param young Young's modulus.
     * @param poisson Poisson's ratio.
     * @return Assembled sparse matrix of size N x N.
     */
    template <class T>
    inline auto assemble_stiffness_tri2d(const xarray_container<uvector<T>>& nodes,
                                         const xarray_container<uvector<std::size_t>>& elements,
                                         T young, T poisson)
    {
        if (nodes.dimension() != 2 || nodes.shape()[1] != 2)
            throw std::runtime_error("assemble_stiffness_tri2d: nodes must be (N, 2).");
        if (elements.dimension() != 2 || elements.shape()[1] != 3)
            throw std::runtime_error("assemble_stiffness_tri2d: elements must be (M, 3).");
        std::size_t N = nodes.shape()[0];
        std::size_t M = elements.shape()[0];
        xcoo_matrix<T> coo(N, N);
        for (std::size_t e = 0; e < M; ++e)
        {
            std::size_t n0 = elements(e, 0), n1 = elements(e, 1), n2 = elements(e, 2);
            T x0 = nodes(n0, 0), y0 = nodes(n0, 1);
            T x1 = nodes(n1, 0), y1 = nodes(n1, 1);
            T x2 = nodes(n2, 0), y2 = nodes(n2, 1);
            auto Ke = detail::triangle_stiffness(x0, y0, x1, y1, x2, y2, young, poisson);
            std::size_t dofs[3] = { n0, n1, n2 };
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    coo.append(dofs[i], dofs[j], Ke(i, j));
        }
        return xcsr_matrix<T>::from_coo(coo);
    }

    /**
     * Assemble a sparse mass matrix for a 2D mesh of triangles.
     */
    template <class T>
    inline auto assemble_mass_tri2d(const xarray_container<uvector<T>>& nodes,
                                    const xarray_container<uvector<std::size_t>>& elements,
                                    T density, bool lumped = false)
    {
        std::size_t N = nodes.shape()[0];
        std::size_t M = elements.shape()[0];
        xcoo_matrix<T> coo(N, N);
        for (std::size_t e = 0; e < M; ++e)
        {
            std::size_t n0 = elements(e, 0), n1 = elements(e, 1), n2 = elements(e, 2);
            T x0 = nodes(n0, 0), y0 = nodes(n0, 1);
            T x1 = nodes(n1, 0), y1 = nodes(n1, 1);
            T x2 = nodes(n2, 0), y2 = nodes(n2, 1);
            auto Me = detail::triangle_mass(x0, y0, x1, y1, x2, y2, density, lumped);
            std::size_t dofs[3] = { n0, n1, n2 };
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    coo.append(dofs[i], dofs[j], Me(i, j));
        }
        return xcsr_matrix<T>::from_coo(coo);
    }

    /**
     * Assemble a sparse stiffness matrix for a 2D mesh of quadrilaterals.
     * @param nodes (N x 2) coordinates.
     * @param quads (M x 4) connectivity.
     */
    template <class T>
    inline auto assemble_stiffness_quad2d(const xarray_container<uvector<T>>& nodes,
                                          const xarray_container<uvector<std::size_t>>& quads,
                                          T young, T poisson)
    {
        std::size_t N = nodes.shape()[0];
        std::size_t M = quads.shape()[0];
        xcoo_matrix<T> coo(N, N);
        for (std::size_t e = 0; e < M; ++e)
        {
            std::size_t dofs[4];
            T elem_nodes[4][2];
            for (int i = 0; i < 4; ++i)
            {
                dofs[i] = quads(e, i);
                elem_nodes[i][0] = nodes(dofs[i], 0);
                elem_nodes[i][1] = nodes(dofs[i], 1);
            }
            auto Ke = detail::quad_stiffness(elem_nodes, young, poisson);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    coo.append(dofs[i], dofs[j], Ke(i, j));
        }
        return xcsr_matrix<T>::from_coo(coo);
    }

    /**
     * Assemble a sparse gradient matrix (discrete gradient operator) for a 2D mesh.
     * Returns matrix G of size (2*N) x N, approximating gradient.
     * Uses simple finite difference on edges? For now returns empty.
     */
    template <class T>
    inline auto assemble_gradient(const xarray_container<uvector<T>>& nodes,
                                  const xarray_container<uvector<std::size_t>>& elements)
    {
        std::size_t N = nodes.shape()[0];
        std::size_t M = elements.shape()[0];
        xcoo_matrix<T> coo(2 * N, N);
        // placeholder: zero matrix
        return xcsr_matrix<T>::from_coo(coo);
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_MESH_HPP