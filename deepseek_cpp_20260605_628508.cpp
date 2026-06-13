//File 0224 : sparse/xsparse_tensor3d.hpp
//3D structured grid sparse assembly using Kronecker products and tensor-product operators for finite difference Laplacian, gradient, and divergence.
#ifndef XTENSOR_XSPARSE_TENSOR3D_HPP
#define XTENSOR_XSPARSE_TENSOR3D_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
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
#include "../core/xmanipulation.hpp"
#include "../sparse/xcsr.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xsparse_config.hpp"
#include "../sparse/xsparse_xbuilder.hpp"
#include "../sparse/xsparse_block.hpp"

namespace xt {
namespace sparse {

    namespace detail
    {
        /**
         * Compute the 1D finite difference matrix (tridiagonal) for a given grid size n.
         * Boundary conditions: Dirichlet (default), or periodic.
         */
        template <class T>
        inline auto fd1d_laplacian(std::size_t n, T dx, bool periodic = false)
        {
            xcoo_matrix<T> coo(n, n);
            T inv_dx2 = T(1) / (dx * dx);
            for (std::size_t i = 0; i < n; ++i)
            {
                coo.append(i, i, T(2) * inv_dx2);
                if (i > 0) coo.append(i, i - 1, -inv_dx2);
                if (i + 1 < n) coo.append(i, i + 1, -inv_dx2);
            }
            if (periodic)
            {
                coo.append(0, n - 1, -inv_dx2);
                coo.append(n - 1, 0, -inv_dx2);
            }
            return xcsr_matrix<T>::from_coo(coo);
        }

        /**
         * Compute the 1D finite difference gradient matrix (forward difference, size (n-1) x n).
         * For periodic, size n x n.
         */
        template <class T>
        inline auto fd1d_gradient(std::size_t n, T dx, bool periodic = false)
        {
            std::size_t m = periodic ? n : n - 1;
            xcoo_matrix<T> coo(m, n);
            T inv_dx = T(1) / dx;
            if (periodic)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    coo.append(i, i, -inv_dx);
                    coo.append(i, (i + 1) % n, inv_dx);
                }
            }
            else
            {
                for (std::size_t i = 0; i < m; ++i)
                {
                    coo.append(i, i, -inv_dx);
                    coo.append(i, i + 1, inv_dx);
                }
            }
            return xcsr_matrix<T>::from_coo(coo);
        }

        /**
         * Kronecker product of two CSR matrices: C = kron(A, B).
         */
        template <class T>
        inline auto kron_csr(const xcsr_matrix<T>& A, const xcsr_matrix<T>& B)
        {
            return kron(A, B);
        }

        /**
         * Sum of two CSR matrices with same sparsity pattern (faster than general add).
         * Assumes row_ptr and col_idx are identical; only adds values.
         */
        template <class T>
        inline auto sum_same_pattern(const xcsr_matrix<T>& A, const xcsr_matrix<T>& B)
        {
            if (A.rows() != B.rows() || A.cols() != B.cols() || A.row_ptr() != B.row_ptr() || A.col_idx() != B.col_idx())
                throw std::runtime_error("sum_same_pattern: sparsity patterns differ.");
            std::vector<T> sum_values(A.values().size());
            const T* a_vals = A.values().data();
            const T* b_vals = B.values().data();
            if constexpr (is_simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t n = A.values().size();
                std::size_t vec_count = n / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type va = simd_type::load_unaligned(a_vals + i * simd_size);
                    simd_type vb = simd_type::load_unaligned(b_vals + i * simd_size);
                    (va + vb).store_unaligned(sum_values.data() + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    sum_values[i] = a_vals[i] + b_vals[i];
            }
            else
            {
                for (std::size_t i = 0; i < A.values().size(); ++i)
                    sum_values[i] = a_vals[i] + b_vals[i];
            }
            return xcsr_matrix<T>(A.rows(), A.cols(),
                                  std::vector<std::size_t>(A.row_ptr()),
                                  std::vector<std::size_t>(A.col_idx()),
                                  std::move(sum_values));
        }
    }

    /**
     * 3D Laplacian operator on a structured grid using tensor products.
     * L = I_z ⊗ I_y ⊗ L_x + I_z ⊗ L_y ⊗ I_x + L_z ⊗ I_y ⊗ I_x
     * where L_x, L_y, L_z are 1D Laplacian matrices along each axis.
     * The total matrix size is (nx*ny*nz) x (nx*ny*nz).
     * This approach is more memory efficient than assembling via element assembly.
     */
    template <class T>
    inline auto laplacian3d_tensor(std::size_t nx, std::size_t ny, std::size_t nz,
                                    T dx, T dy, T dz, bool periodic_x = false,
                                    bool periodic_y = false, bool periodic_z = false)
    {
        // 1D Laplacian matrices
        auto Lx = detail::fd1d_laplacian(nx, dx, periodic_x);
        auto Ly = detail::fd1d_laplacian(ny, dy, periodic_y);
        auto Lz = detail::fd1d_laplacian(nz, dz, periodic_z);

        // Identity matrices
        auto Ix = eye_sparse<T>(nx);
        auto Iy = eye_sparse<T>(ny);
        auto Iz = eye_sparse<T>(nz);

        // Term 1: Iz ⊗ Iy ⊗ Lx
        auto Iy_kron_Lx = kron_csr(Iy, Lx);
        auto term1 = kron_csr(Iz, Iy_kron_Lx);

        // Term 2: Iz ⊗ Ly ⊗ Ix
        auto Ly_kron_Ix = kron_csr(Ly, Ix);
        auto term2 = kron_csr(Iz, Ly_kron_Ix);

        // Term 3: Lz ⊗ Iy ⊗ Ix
        auto Iy_kron_Ix = kron_csr(Iy, Ix);
        auto term3 = kron_csr(Lz, Iy_kron_Ix);

        // Sum the three terms using general sparse addition
        auto temp = xcsr_matrix<T>::add(term1, term2);
        auto result = xcsr_matrix<T>::add(temp, term3);
        return result;
    }

    /**
     * 3D Gradient operator (forward difference) returning a sparse matrix of size
     * (3 * nx*ny*nz) x (nx*ny*nz) if staggered, or simpler structure.
     * For convenience, we return separate components: Gx, Gy, Gz.
     */
    template <class T>
    inline auto gradient3d_tensor(std::size_t nx, std::size_t ny, std::size_t nz,
                                   T dx, T dy, T dz)
    {
        auto Gx_1d = detail::fd1d_gradient(nx, dx);
        auto Gy_1d = detail::fd1d_gradient(ny, dy);
        auto Gz_1d = detail::fd1d_gradient(nz, dz);

        auto Ix = eye_sparse<T>(nx);
        auto Iy = eye_sparse<T>(ny);
        auto Iz = eye_sparse<T>(nz);

        // Gx = Iz ⊗ Iy ⊗ Gx_1d   (size (nx-1)*ny*nz x nx*ny*nz for non-periodic)
        auto Iy_kron_Gx = kron_csr(Iy, Gx_1d);
        auto Gx = kron_csr(Iz, Iy_kron_Gx);

        // Gy = Iz ⊗ Gy_1d ⊗ Ix
        auto Gy_kron_Ix = kron_csr(Gy_1d, Ix);
        auto Gy = kron_csr(Iz, Gy_kron_Ix);

        // Gz = Gz_1d ⊗ Iy ⊗ Ix
        auto Iy_kron_Ix = kron_csr(Iy, Ix);
        auto Gz = kron_csr(Gz_1d, Iy_kron_Ix);

        return std::make_tuple(Gx, Gy, Gz);
    }

    /**
     * 3D Divergence operator as negative adjoint of gradient (for uniform grid).
     * Dx = -Gx^T, etc. Returns Dx, Dy, Dz.
     */
    template <class T>
    inline auto divergence3d_tensor(std::size_t nx, std::size_t ny, std::size_t nz,
                                     T dx, T dy, T dz)
    {
        auto [Gx, Gy, Gz] = gradient3d_tensor(nx, ny, nz, dx, dy, dz);
        return std::make_tuple(sptranspose(Gx), sptranspose(Gy), sptranspose(Gz));
    }

    /**
     * Assemble the full gradient operator as a single (3*n) x n block matrix.
     * [ Gx ]
     * [ Gy ]
     * [ Gz ]
     */
    template <class T>
    inline auto gradient3d_full(const xcsr_matrix<T>& Gx, const xcsr_matrix<T>& Gy, const xcsr_matrix<T>& Gz)
    {
        std::size_t rows = Gx.rows() + Gy.rows() + Gz.rows();
        std::size_t cols = Gx.cols();
        xcoo_matrix<T> coo(rows, cols);
        auto append = [&](const xcsr_matrix<T>& G, std::size_t row_offset) {
            for (std::size_t r = 0; r < G.rows(); ++r)
                for (std::size_t j = G.row_ptr()[r]; j < G.row_ptr()[r + 1]; ++j)
                    coo.append(row_offset + r, G.col_idx()[j], G.values()[j]);
        };
        append(Gx, 0);
        append(Gy, Gx.rows());
        append(Gz, Gx.rows() + Gy.rows());
        return xcsr_matrix<T>::from_coo(coo);
    }

    /**
     * 3D anisotropic diffusion operator with variable coefficients (scalar per cell).
     * For simplicity, we assemble a diagonal tensor coefficient and multiply with Laplacian.
     * K * L, where K is diagonal.
     */
    template <class T>
    inline auto anisotropic_diffusion3d(const xcsr_matrix<T>& L, const xarray_container<uvector<T>>& kappa)
    {
        // Multiply sparse matrix L by diagonal matrix diag(kappa) from the right: A = L * diag(kappa)
        // A(i,j) = L(i,j) * kappa(j)
        xcoo_matrix<T> coo(L.rows(), L.cols());
        for (std::size_t r = 0; r < L.rows(); ++r)
        {
            for (std::size_t j = L.row_ptr()[r]; j < L.row_ptr()[r + 1]; ++j)
            {
                std::size_t c = L.col_idx()[j];
                T val = L.values()[j] * kappa[c];
                if (val != T(0))
                    coo.append(r, c, val);
            }
        }
        return xcsr_matrix<T>::from_coo(coo);
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_TENSOR3D_HPP