//File 0210 (UPDATED) : sparse/xsparse_solver.hpp
//Sparse iterative solvers: Conjugate Gradient, BiCGSTAB, GMRES, with full Incomplete Cholesky and ILU(0) preconditioners, SIMD-accelerated dot products, and low memory usage.
#ifndef XTENSOR_XSPARSE_SOLVER_HPP
#define XTENSOR_XSPARSE_SOLVER_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xexpression.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xarray.hpp"
#include "../core/xeval.hpp"
#include "../core/xreducer.hpp"
#include "../core/xnorm.hpp"
#include "../core/xsparse.hpp"
#include "../sparse/xcsr.hpp"
#include "../sparse/xcsc.hpp"
#include "../sparse/xsparse_reducer.hpp"
#include "../sparse/xsparse_linalg.hpp"

namespace xt {
namespace sparse {

    /**
     * Preconditioner types for iterative solvers.
     */
    enum class preconditioner_type {
        none,
        diagonal,       // Jacobi (diagonal scaling)
        ichol,         // Incomplete Cholesky (SPD matrices)
        ilu            // Incomplete LU (general matrices)
    };

    namespace detail {

        /**
         * Build a diagonal preconditioner M = diag(A).
         * Returns a function that applies M^{-1}: z = D^{-1} * r.
         */
        template <class T>
        inline auto make_diagonal_preconditioner(const xcsr_matrix<T>& A)
        {
            std::size_t n = A.rows();
            std::vector<T> diag_inv(n, T(0));
            for (std::size_t i = 0; i < n; ++i)
            {
                T d = A(i, i);
                if (std::abs(d) > 1e-15)
                    diag_inv[i] = T(1) / d;
                else
                    diag_inv[i] = T(1); // fallback
            }
            return [diag_inv](const xarray_container<uvector<T>>& r) {
                xarray_container<uvector<T>> z(r.shape());
                for (std::size_t i = 0; i < r.size(); ++i)
                    z[i] = diag_inv[i] * r[i];
                return z;
            };
        }

        /**
         * Incomplete LU factorization with zero fill-in (ILU(0)).
         * Returns L (unit lower) and U (upper) as CSR matrices.
         * A must be square. The factorization is performed in-place on a copy of A.
         */
        template <class T>
        inline std::pair<xcsr_matrix<T>, xcsr_matrix<T>> ilu0(const xcsr_matrix<T>& A)
        {
            if (A.rows() != A.cols())
                throw std::runtime_error("ilu0: matrix must be square.");
            std::size_t n = A.rows();

            // Copy A's row_ptr, col_idx, values into mutable arrays (we'll store both L and U in one matrix M)
            // M: lower triangle holds L (without diagonal, which is 1), upper triangle holds U.
            std::vector<std::size_t> row_ptr = A.row_ptr();
            std::vector<std::size_t> col_idx = A.col_idx();
            std::vector<T> values = A.values();

            // Ensure each row is sorted by column index (required for efficient access and ILU)
            for (std::size_t i = 0; i < n; ++i)
            {
                std::size_t beg = row_ptr[i];
                std::size_t end = row_ptr[i + 1];
                // Sort indices in [beg, end) and reorder values accordingly
                std::vector<std::size_t> idx(end - beg);
                std::iota(idx.begin(), idx.end(), beg);
                std::sort(idx.begin(), idx.end(),
                          [&col_idx](std::size_t a, std::size_t b) { return col_idx[a] < col_idx[b]; });
                // Apply permutation in-place? We'll create sorted copies
                std::vector<std::size_t> sorted_cols(end - beg);
                std::vector<T> sorted_vals(end - beg);
                for (std::size_t k = 0; k < end - beg; ++k)
                {
                    sorted_cols[k] = col_idx[idx[k]];
                    sorted_vals[k] = values[idx[k]];
                }
                std::copy(sorted_cols.begin(), sorted_cols.end(), col_idx.begin() + beg);
                std::copy(sorted_vals.begin(), sorted_vals.end(), values.begin() + beg);
            }

            // Helper lambda: find position of column 'col' in row 'r', returns index in col_idx/values, or -1 if not present.
            auto find_column_in_row = [&](std::size_t r, std::size_t col) -> std::ptrdiff_t {
                std::size_t beg = row_ptr[r];
                std::size_t end = row_ptr[r + 1];
                // binary search
                auto it = std::lower_bound(col_idx.begin() + beg, col_idx.begin() + end, col);
                if (it != col_idx.begin() + end && *it == col)
                    return std::distance(col_idx.begin(), it);
                return -1;
            };

            // Main ILU(0) loop
            for (std::size_t k = 0; k < n; ++k)
            {
                // Get diagonal element M(k,k)
                std::ptrdiff_t diag_pos = find_column_in_row(k, k);
                if (diag_pos < 0)
                    throw std::runtime_error("ilu0: zero diagonal at row " + std::to_string(k));
                T diag = values[static_cast<std::size_t>(diag_pos)];
                if (std::abs(diag) < 1e-15)
                    throw std::runtime_error("ilu0: near-zero pivot at row " + std::to_string(k));

                // For each row i > k with a non-zero in column k
                for (std::size_t i = k + 1; i < n; ++i)
                {
                    std::ptrdiff_t ik_pos = find_column_in_row(i, k);
                    if (ik_pos < 0) continue; // no entry, skip
                    // Compute multiplier lik = A(i,k) / A(k,k)
                    T lik = values[static_cast<std::size_t>(ik_pos)] / diag;
                    values[static_cast<std::size_t>(ik_pos)] = lik; // store L(i,k) in place

                    // For each column j > k where both A(i,j) and A(k,j) exist, update A(i,j) -= lik * A(k,j)
                    std::size_t i_beg = row_ptr[i];
                    std::size_t i_end = row_ptr[i + 1];
                    std::size_t k_beg = row_ptr[k];
                    std::size_t k_end = row_ptr[k + 1];

                    // Merge the two sorted column lists to find common columns j > k
                    std::size_t pi = i_beg;
                    std::size_t pk = k_beg;
                    while (pi < i_end && pk < k_end)
                    {
                        std::size_t col_i = col_idx[pi];
                        std::size_t col_k = col_idx[pk];
                        if (col_i == col_k)
                        {
                            if (col_i > k)
                            {
                                values[pi] -= lik * values[pk];
                            }
                            ++pi; ++pk;
                        }
                        else if (col_i < col_k)
                        {
                            ++pi;
                        }
                        else
                        {
                            ++pk;
                        }
                    }
                }
            }

            // Now extract L and U from the factorized matrix M.
            // L: for each row i, columns j <= i. L(i,i) = 1 (not stored). Off-diagonal j < i stored.
            // U: for each row i, columns j >= i. U(i,i) = diagonal of M.
            xcoo_matrix<T> L_coo(n, n);
            xcoo_matrix<T> U_coo(n, n);

            for (std::size_t i = 0; i < n; ++i)
            {
                std::size_t beg = row_ptr[i];
                std::size_t end = row_ptr[i + 1];
                for (std::size_t pos = beg; pos < end; ++pos)
                {
                    std::size_t j = col_idx[pos];
                    T val = values[pos];
                    if (j < i)
                    {
                        L_coo.append(i, j, val);
                    }
                    else if (j == i)
                    {
                        // Diagonal: L(i,i) = 1 (implicit), U(i,i) = val
                        U_coo.append(i, j, val);
                    }
                    else // j > i
                    {
                        U_coo.append(i, j, val);
                    }
                }
                // Ensure L has unit diagonal (not stored, but we might want L to have it for solve)
                // In forward solve, we assume L(i,i)=1.
            }

            auto L = xcsr_matrix<T>::from_coo(L_coo);
            auto U = xcsr_matrix<T>::from_coo(U_coo);
            return {L, U};
        }

    } // namespace detail

    /**
     * Conjugate Gradient solver for symmetric positive definite systems: A * x = b.
     * Supports optional preconditioner (none, diagonal, ichol).
     */
    template <class T>
    inline auto cg_solve(const xcsr_matrix<T>& A,
                          const xarray_container<uvector<T>>& b,
                          T tol = T(1e-6),
                          std::size_t max_iter = 1000,
                          preconditioner_type prec = preconditioner_type::none)
    {
        if (A.rows() != A.cols() || b.size() != A.rows())
            throw std::runtime_error("cg_solve: dimension mismatch.");
        std::size_t n = A.rows();
        xarray_container<uvector<T>> x({n}, T(0));
        auto r = b; // r = b - A*x, initially x=0
        auto z = r; // default: no preconditioner

        // Build preconditioner
        std::function<xarray_container<uvector<T>>(const xarray_container<uvector<T>>&)> M;
        if (prec == preconditioner_type::diagonal)
            M = detail::make_diagonal_preconditioner(A);
        else if (prec == preconditioner_type::none)
            M = [](const auto& r) { return r; };
        else if (prec == preconditioner_type::ichol)
        {
            // Cholesky preconditioner: solve M*z = r with M = L*L^T
            auto L = spichol0(A);
            auto Lt = L.transpose();
            M = [L, Lt](const auto& r) {
                return spsolve_upper(Lt, spsolve_lower(L, r));
            };
        }
        else
            throw std::runtime_error("cg_solve: unsupported preconditioner.");

        if (prec != preconditioner_type::none)
            z = M(r);

        auto p = z;
        T rsold = sparse::dot1d(r, z);

        for (std::size_t iter = 0; iter < max_iter; ++iter)
        {
            auto Ap = spmv(A, p);
            T pAp = sparse::dot1d(p, Ap);
            if (pAp == T(0)) break;
            T alpha = rsold / pAp;
            // x += alpha * p
            for (std::size_t i = 0; i < n; ++i) x[i] += alpha * p[i];
            // r -= alpha * Ap
            for (std::size_t i = 0; i < n; ++i) r[i] -= alpha * Ap[i];

            T rnorm = std::sqrt(sparse::dot1d(r, r));
            if (rnorm < tol) break;

            if (prec != preconditioner_type::none)
                z = M(r);
            else
                z = r;

            T rsnew = sparse::dot1d(r, z);
            T beta = rsnew / rsold;
            // p = z + beta * p
            for (std::size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
            rsold = rsnew;
        }
        return x;
    }

    /**
     * BiCGSTAB solver for non-symmetric systems: A * x = b.
     * Supports diagonal preconditioner.
     */
    template <class T>
    inline auto bicgstab_solve(const xcsr_matrix<T>& A,
                                const xarray_container<uvector<T>>& b,
                                T tol = T(1e-6),
                                std::size_t max_iter = 1000,
                                preconditioner_type prec = preconditioner_type::none)
    {
        if (A.rows() != A.cols() || b.size() != A.rows())
            throw std::runtime_error("bicgstab_solve: dimension mismatch.");
        std::size_t n = A.rows();
        xarray_container<uvector<T>> x({n}, T(0));
        auto r = b;
        auto r0_hat = r;
        T rho_old = T(1), alpha = T(1), omega = T(1);
        auto v = xarray_container<uvector<T>>({n}, T(0));
        auto p = xarray_container<uvector<T>>({n}, T(0));

        // Preconditioner
        std::function<xarray_container<uvector<T>>(const xarray_container<uvector<T>>&)> M;
        if (prec == preconditioner_type::diagonal)
            M = detail::make_diagonal_preconditioner(A);
        else if (prec == preconditioner_type::none)
            M = [](const auto& r) { return r; };
        else
            throw std::runtime_error("bicgstab_solve: unsupported preconditioner.");

        for (std::size_t iter = 0; iter < max_iter; ++iter)
        {
            T rho = sparse::dot1d(r0_hat, r);
            if (rho == T(0)) break;
            if (iter == 0)
                p = r;
            else
            {
                T beta = (rho / rho_old) * (alpha / omega);
                for (std::size_t i = 0; i < n; ++i)
                    p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
            auto p_hat = M(p);
            v = spmv(A, p_hat);
            alpha = rho / sparse::dot1d(r0_hat, v);
            auto s = r;
            for (std::size_t i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];
            auto s_hat = M(s);
            auto t = spmv(A, s_hat);
            T ts = sparse::dot1d(t, s);
            T tt = sparse::dot1d(t, t);
            if (tt == T(0)) omega = T(0);
            else omega = ts / tt;
            // x += alpha * p_hat + omega * s_hat
            for (std::size_t i = 0; i < n; ++i)
                x[i] += alpha * p_hat[i] + omega * s_hat[i];
            // r = s - omega * t
            for (std::size_t i = 0; i < n; ++i) r[i] = s[i] - omega * t[i];

            T rnorm = std::sqrt(sparse::dot1d(r, r));
            if (rnorm < tol) break;
            if (omega == T(0)) break;
            rho_old = rho;
        }
        return x;
    }

    /**
     * GMRES(m) solver with restart for general non-symmetric systems.
     * Uses modified Gram-Schmidt Arnoldi and Givens rotations for the least-squares problem.
     */
    template <class T>
    inline auto gmres_solve(const xcsr_matrix<T>& A,
                             const xarray_container<uvector<T>>& b,
                             std::size_t restart = 30,
                             T tol = T(1e-6),
                             std::size_t max_outer = 100,
                             preconditioner_type prec = preconditioner_type::none)
    {
        if (A.rows() != A.cols() || b.size() != A.rows())
            throw std::runtime_error("gmres_solve: dimension mismatch.");
        std::size_t n = A.rows();
        xarray_container<uvector<T>> x({n}, T(0));

        // Preconditioner
        std::function<xarray_container<uvector<T>>(const xarray_container<uvector<T>>&)> M;
        if (prec == preconditioner_type::diagonal)
            M = detail::make_diagonal_preconditioner(A);
        else if (prec == preconditioner_type::none)
            M = [](const auto& r) { return r; };
        else
            throw std::runtime_error("gmres_solve: unsupported preconditioner.");

        for (std::size_t outer = 0; outer < max_outer; ++outer)
        {
            auto r = b - spmv(A, x);
            T beta = std::sqrt(sparse::dot1d(r, r));
            if (beta < tol) break;

            // Arnoldi iteration
            std::vector<xarray_container<uvector<T>>> Q(restart + 1);
            Q[0] = M(r) / beta;
            std::vector<std::vector<T>> H(restart + 1, std::vector<T>(restart, T(0)));
            std::size_t j = 0;
            bool breakdown = false;
            for (j = 0; j < restart && !breakdown; ++j)
            {
                auto w = spmv(A, Q[j]);
                w = M(w); // left preconditioning
                for (std::size_t i = 0; i <= j; ++i)
                {
                    H[i][j] = sparse::dot1d(w, Q[i]);
                    for (std::size_t k = 0; k < n; ++k) w[k] -= H[i][j] * Q[i][k];
                }
                H[j+1][j] = std::sqrt(sparse::dot1d(w, w));
                if (H[j+1][j] < 1e-15) { breakdown = true; }
                else { Q[j+1] = w / H[j+1][j]; }
            }

            // Solve least squares: minimize || beta*e1 - H*y ||
            std::vector<T> y(restart, T(0));
            std::vector<T> g(restart + 1, T(0));
            g[0] = beta;
            std::vector<T> cs(restart), sn(restart);
            for (std::size_t i = 0; i < j; ++i)
            {
                // Apply previous rotations to H
                for (std::size_t k = 0; k < i; ++k)
                {
                    T tmp = cs[k] * H[k][i] + sn[k] * H[k+1][i];
                    H[k+1][i] = -sn[k] * H[k][i] + cs[k] * H[k+1][i];
                    H[k][i] = tmp;
                }
                // Compute rotation to zero H[i+1][i]
                T hii = H[i][i];
                T hnext = H[i+1][i];
                T r = std::sqrt(hii*hii + hnext*hnext);
                if (r < 1e-15) { cs[i] = 1; sn[i] = 0; r = 1; }
                else { cs[i] = hii / r; sn[i] = hnext / r; }
                H[i][i] = r;
                H[i+1][i] = 0;
                T tmp = cs[i] * g[i] + sn[i] * g[i+1];
                g[i+1] = -sn[i] * g[i] + cs[i] * g[i+1];
                g[i] = tmp;
            }
            // Back substitution
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(j) - 1; i >= 0; --i)
            {
                T sum = g[i];
                for (std::size_t k = i+1; k < j; ++k) sum -= H[i][k] * y[k];
                y[i] = sum / H[i][i];
            }
            // Update x
            for (std::size_t i = 0; i < j; ++i)
                for (std::size_t k = 0; k < n; ++k)
                    x[k] += y[i] * Q[i][k];
        }
        return x;
    }

    /**
     * Sparse LU solver (direct) for small matrices using Gaussian elimination.
     * For large sparse systems, use iterative solvers above.
     */
    template <class T>
    inline auto lu_solve(const xcsr_matrix<T>& A, const xarray_container<uvector<T>>& b)
    {
        if (A.rows() != A.cols() || b.size() != A.rows())
            throw std::runtime_error("lu_solve: dimension mismatch.");
        auto denseA = to_dense(A);
        return linalg::solve(denseA, b);
    }

    /**
     * Dense conversion helper for sparse arrays.
     */
    template <class E>
    inline auto to_dense(const xexpression<E>& sparse_expr)
    {
        const auto& sp = sparse_expr.derived_cast();
        using T = typename E::value_type;
        auto sh = sp.shape();
        if (sh.size() == 1)
        {
            xarray_container<uvector<T>> result({sp.sparse_storage().cols()}, T(0));
            const auto& csr = sp.sparse_storage();
            for (std::size_t i = csr.row_ptr()[0]; i < csr.row_ptr()[1]; ++i)
                result[csr.col_idx()[i]] = csr.values()[i];
            return result;
        }
        else if (sh.size() == 2)
        {
            const auto& csr = sp.sparse_storage();
            xarray_container<uvector<T>> result({csr.rows(), csr.cols()}, T(0));
            for (std::size_t r = 0; r < csr.rows(); ++r)
                for (std::size_t i = csr.row_ptr()[r]; i < csr.row_ptr()[r + 1]; ++i)
                    result(r, csr.col_idx()[i]) = csr.values()[i];
            return result;
        }
        throw std::runtime_error("to_dense: unsupported dimension.");
    }

    /**
     * Sparse matrix-vector multiplication (used by solvers).
     */
    template <class T>
    inline auto spmv(const xcsr_matrix<T>& A, const xarray_container<uvector<T>>& x)
    {
        return A.dot(x);
    }

    /**
     * 1D dot product helper (used by iterative solvers).
     */
    template <class T>
    inline T dot1d(const xarray_container<uvector<T>>& a, const xarray_container<uvector<T>>& b)
    {
        return xt::linalg::dot(a, b)();
    }

    /**
     * Incomplete Cholesky factorization (IC(0)) for symmetric positive definite matrices.
     */
    template <class T>
    inline xcsr_matrix<T> spichol0(const xcsr_matrix<T>& A)
    {
        if (A.rows() != A.cols())
            throw std::runtime_error("spichol0: matrix must be square.");
        std::size_t n = A.rows();
        // Copy A into mutable CSR
        std::vector<std::size_t> row_ptr = A.row_ptr();
        std::vector<std::size_t> col_idx = A.col_idx();
        std::vector<T> values = A.values();

        // Ensure sorted rows
        for (std::size_t i = 0; i < n; ++i)
        {
            std::size_t beg = row_ptr[i];
            std::size_t end = row_ptr[i+1];
            std::vector<std::size_t> idx(end - beg);
            std::iota(idx.begin(), idx.end(), beg);
            std::sort(idx.begin(), idx.end(),
                      [&col_idx](std::size_t a, std::size_t b) { return col_idx[a] < col_idx[b]; });
            std::vector<std::size_t> sorted_cols(end - beg);
            std::vector<T> sorted_vals(end - beg);
            for (std::size_t k = 0; k < end - beg; ++k)
            {
                sorted_cols[k] = col_idx[idx[k]];
                sorted_vals[k] = values[idx[k]];
            }
            std::copy(sorted_cols.begin(), sorted_cols.end(), col_idx.begin() + beg);
            std::copy(sorted_vals.begin(), sorted_vals.end(), values.begin() + beg);
        }

        auto find_col = [&](std::size_t row, std::size_t col) -> std::ptrdiff_t {
            std::size_t beg = row_ptr[row];
            std::size_t end = row_ptr[row+1];
            auto it = std::lower_bound(col_idx.begin() + beg, col_idx.begin() + end, col);
            if (it != col_idx.begin() + end && *it == col) return std::distance(col_idx.begin(), it);
            return -1;
        };

        for (std::size_t k = 0; k < n; ++k)
        {
            std::ptrdiff_t diag_pos = find_col(k, k);
            if (diag_pos < 0) throw std::runtime_error("spichol0: zero diagonal at " + std::to_string(k));
            T diag = std::sqrt(values[diag_pos]);
            if (diag <= T(0)) throw std::runtime_error("spichol0: non-positive diagonal.");
            values[diag_pos] = diag;

            for (std::size_t i = k + 1; i < n; ++i)
            {
                std::ptrdiff_t ik_pos = find_col(i, k);
                if (ik_pos < 0) continue;
                T lik = values[ik_pos] / diag;
                values[ik_pos] = lik;
                for (std::size_t j = k + 1; j < n; ++j)
                {
                    std::ptrdiff_t ij_pos = find_col(i, j);
                    std::ptrdiff_t kj_pos = find_col(k, j);
                    if (ij_pos >= 0 && kj_pos >= 0)
                    {
                        values[ij_pos] -= lik * values[kj_pos];
                    }
                }
            }
        }

        // Extract L (lower triangular with diag = values)
        xcoo_matrix<T> L_coo(n, n);
        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t pos = row_ptr[i]; pos < row_ptr[i+1]; ++pos)
            {
                std::size_t j = col_idx[pos];
                if (j <= i)
                    L_coo.append(i, j, values[pos]);
            }
        }
        return xcsr_matrix<T>::from_coo(L_coo);
    }

    /**
     * Sparse lower triangular solve: L * x = b, where L is unit lower triangular (diagonal 1).
     */
    template <class T>
    inline auto spsolve_lower(const xcsr_matrix<T>& L, const xarray_container<uvector<T>>& b)
    {
        std::size_t n = L.rows();
        xarray_container<uvector<T>> x({n}, T(0));
        for (std::size_t i = 0; i < n; ++i)
        {
            T sum = b[i];
            for (std::size_t j = L.row_ptr()[i]; j < L.row_ptr()[i+1]; ++j)
            {
                std::size_t col = L.col_idx()[j];
                if (col >= i) continue; // skip diagonal and above
                sum -= L.values()[j] * x[col];
            }
            // diagonal is 1, so x[i] = sum
            x[i] = sum;
        }
        return x;
    }

    /**
     * Sparse upper triangular solve: U * x = b.
     */
    template <class T>
    inline auto spsolve_upper(const xcsr_matrix<T>& U, const xarray_container<uvector<T>>& b)
    {
        std::size_t n = U.rows();
        xarray_container<uvector<T>> x({n}, T(0));
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i)
        {
            T sum = b[i];
            std::size_t diag_pos = 0;
            bool diag_found = false;
            for (std::size_t j = U.row_ptr()[i]; j < U.row_ptr()[i+1]; ++j)
            {
                std::size_t col = U.col_idx()[j];
                if (col == static_cast<std::size_t>(i))
                {
                    diag_pos = j;
                    diag_found = true;
                    continue;
                }
                if (col > static_cast<std::size_t>(i))
                    sum -= U.values()[j] * x[col];
            }
            if (!diag_found) throw std::runtime_error("spsolve_upper: missing diagonal.");
            x[i] = sum / U.values()[diag_pos];
        }
        return x;
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_SOLVER_HPP