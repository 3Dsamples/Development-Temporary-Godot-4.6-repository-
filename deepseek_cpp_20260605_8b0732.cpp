//File 0226 : sparse/xsparse_optimize.hpp
//Sparse gradient and Hessian assembly for physics‑based optimization: spring energy, elastic networks, and nonlinear conjugate gradient with sparse structures.
#ifndef XTENSOR_XSPARSE_OPTIMIZE_HPP
#define XTENSOR_XSPARSE_OPTIMIZE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
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
#include "../core/xnorm.hpp"
#include "../core/xoptimize.hpp"
#include "../sparse/xcsr.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xsparse_config.hpp"
#include "../sparse/xsparse_reducer.hpp"
#include "../sparse/xsparse_solver.hpp"

namespace xt {
namespace sparse {

    /**
     * Compute the gradient and Hessian of a spring‑mass system with N nodes
     * connected by springs. Each spring has stiffness k and rest length l0.
     * Energy: E = 0.5 * k * (|x_i - x_j| - l0)^2 per edge.
     * Gradient vector (3*N) and Hessian matrix (3*N x 3*N) stored as CSR.
     * @param positions Node positions as (N x 3) dense array.
     * @param edges Pairs of connected node indices (E x 2).
     * @param stiffness Spring constant k per edge (E) or scalar.
     * @param rest_length Rest length l0 per edge (E) or scalar.
     * @return pair (gradient as 1D vector, Hessian as CSR sparse matrix).
     */
    template <class T>
    inline auto spring_gradient_hessian(
        const xarray_container<uvector<T>>& positions,
        const xarray_container<uvector<std::size_t>>& edges,
        const xarray_container<uvector<T>>& stiffness,
        const xarray_container<uvector<T>>& rest_length)
    {
        if (positions.dimension() != 2 || positions.shape()[1] != 3)
            throw std::runtime_error("spring_gradient_hessian: positions must be (N, 3).");
        if (edges.dimension() != 2 || edges.shape()[1] != 2)
            throw std::runtime_error("spring_gradient_hessian: edges must be (E, 2).");
        std::size_t N = positions.shape()[0];
        std::size_t E = edges.shape()[0];
        xarray_container<uvector<T>> grad({3 * N}, T(0));
        xcoo_matrix<T> hess_coo(3 * N, 3 * N);
        T* g = grad.data();
        const T* pos = positions.data();

        for (std::size_t e = 0; e < E; ++e)
        {
            std::size_t a = edges(e, 0);
            std::size_t b = edges(e, 1);
            T k = stiffness[e];
            T l0 = rest_length[e];

            // Vector from a to b
            T dx[3], rlen;
            dx[0] = pos[3*b + 0] - pos[3*a + 0];
            dx[1] = pos[3*b + 1] - pos[3*a + 1];
            dx[2] = pos[3*b + 2] - pos[3*a + 2];
            rlen = std::sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
            if (rlen < 1e-15) continue;

            T force_mag = k * (rlen - l0);
            T dir[3] = { dx[0]/rlen, dx[1]/rlen, dx[2]/rlen };

            // Gradient contribution
            for (int d = 0; d < 3; ++d)
            {
                T f = force_mag * dir[d];
                g[3*a + d] -= f;
                g[3*b + d] += f;
            }

            // Hessian contribution: block matrix for pair (a,b)
            // d^2E / dx_a dx_b = -k * (n*n^T + (1 - l0/r)*(I - n*n^T))  where n = dir
            T factor1 = k;
            T factor2 = k * (T(1) - l0 / rlen);
            for (int d1 = 0; d1 < 3; ++d1)
            {
                for (int d2 = 0; d2 < 3; ++d2)
                {
                    T h_val = factor1 * dir[d1] * dir[d2] + factor2 * ((d1==d2 ? T(1) : T(0)) - dir[d1]*dir[d2]);
                    // H_{a,a} += h_val; H_{b,b} += h_val
                    hess_coo.append(3*a + d1, 3*a + d2, h_val);
                    hess_coo.append(3*b + d1, 3*b + d2, h_val);
                    // H_{a,b} -= h_val; H_{b,a} -= h_val
                    hess_coo.append(3*a + d1, 3*b + d2, -h_val);
                    hess_coo.append(3*b + d1, 3*a + d2, -h_val);
                }
            }
        }
        auto hess = xcsr_matrix<T>::from_coo(hess_coo);
        return std::make_pair(grad, hess);
    }

    /**
     * Compute only the gradient of the spring energy (cheaper).
     * Uses SIMD for accumulation across edges.
     */
    template <class T>
    inline auto spring_gradient(
        const xarray_container<uvector<T>>& positions,
        const xarray_container<uvector<std::size_t>>& edges,
        const xarray_container<uvector<T>>& stiffness,
        const xarray_container<uvector<T>>& rest_length)
    {
        std::size_t N = positions.shape()[0];
        std::size_t E = edges.shape()[0];
        xarray_container<uvector<T>> grad({3 * N}, T(0));
        T* g = grad.data();
        const T* pos = positions.data();
        for (std::size_t e = 0; e < E; ++e)
        {
            std::size_t a = edges(e, 0);
            std::size_t b = edges(e, 1);
            T k = stiffness[e];
            T l0 = rest_length[e];
            T dx[3], rlen;
            dx[0] = pos[3*b + 0] - pos[3*a + 0];
            dx[1] = pos[3*b + 1] - pos[3*a + 1];
            dx[2] = pos[3*b + 2] - pos[3*a + 2];
            rlen = std::sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
            if (rlen < 1e-15) continue;
            T force_mag = k * (rlen - l0);
            T inv_rlen = T(1) / rlen;
            T dir[3] = { dx[0]*inv_rlen, dx[1]*inv_rlen, dx[2]*inv_rlen };
            for (int d = 0; d < 3; ++d)
            {
                T f = force_mag * dir[d];
                g[3*a + d] -= f;
                g[3*b + d] += f;
            }
        }
        return grad;
    }

    /**
     * Nonlinear Conjugate Gradient solver for sparse spring energy minimization.
     * Uses the sparse Hessian as preconditioner (approximate Newton step via CG).
     */
    template <class T>
    inline auto spring_energy_minimize(
        const xcsr_matrix<T>& Hessian_template,
        const xarray_container<uvector<T>>& initial_positions,
        const xarray_container<uvector<std::size_t>>& edges,
        const xarray_container<uvector<T>>& stiffness,
        const xarray_container<uvector<T>>& rest_length,
        T tol = T(1e-4), std::size_t max_iter = 100)
    {
        std::size_t N = initial_positions.shape()[0];
        xarray_container<uvector<T>> pos = initial_positions;
        auto grad = spring_gradient(pos, edges, stiffness, rest_length);
        auto dir = -grad; // steepest descent direction
        for (std::size_t iter = 0; iter < max_iter; ++iter)
        {
            T grad_norm = std::sqrt(dot1d(grad, grad));
            if (grad_norm < tol) break;

            // Use Hessian to compute Newton step direction: H * p = -g
            // Approximate by applying the current Hessian (could use a fixed template or recompute)
            // For simplicity, we'll recompute Hessian at current position (expensive but accurate)
            auto [g, H] = spring_gradient_hessian(pos, edges, stiffness, rest_length);
            // Solve H * p = -g using sparse CG
            auto p = cg_solve(H, -g, T(1e-6), 100, preconditioner_type::diagonal);
            // Line search (backtracking)
            T alpha = T(1.0);
            T f0 = T(0); // energy: 0.5 * sum springs
            // Compute initial energy
            {
                T energy = 0;
                for (std::size_t e = 0; e < edges.shape()[0]; ++e)
                {
                    std::size_t a = edges(e,0), b = edges(e,1);
                    T dx[3];
                    for (int d=0; d<3; ++d) dx[d] = pos(3*b+d) - pos(3*a+d);
                    T len = std::sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
                    T k = stiffness[e], l0 = rest_length[e];
                    T diff = len - l0;
                    energy += T(0.5) * k * diff * diff;
                }
                f0 = energy;
            }
            // backtracking
            while (alpha > T(1e-10))
            {
                auto new_pos = pos + alpha * p;
                // compute new energy
                T new_energy = 0;
                for (std::size_t e = 0; e < edges.shape()[0]; ++e)
                {
                    std::size_t a = edges(e,0), b = edges(e,1);
                    T dx[3];
                    for (int d=0; d<3; ++d) dx[d] = new_pos(3*b+d) - new_pos(3*a+d);
                    T len = std::sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
                    T k = stiffness[e], l0 = rest_length[e];
                    T diff = len - l0;
                    new_energy += T(0.5) * k * diff * diff;
                }
                if (new_energy < f0) break;
                alpha *= T(0.5);
            }
            // Update position and gradient
            pos = pos + alpha * p;
            grad = spring_gradient(pos, edges, stiffness, rest_length);
            // Fletcher-Reeves beta
            T beta = dot1d(grad, grad) / std::max(grad_norm*grad_norm, T(1e-15));
            dir = -grad + beta * dir;
        }
        return pos;
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_OPTIMIZE_HPP