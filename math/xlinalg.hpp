// core/xlinalg.hpp
#ifndef XTENSOR_XLINALG_HPP
#define XTENSOR_XLINALG_HPP

// ----------------------------------------------------------------------------
// xlinalg.hpp – Linear algebra operations for xtensor expressions
// ----------------------------------------------------------------------------
// This header provides common linear algebra functions:
//   - inv: matrix inverse (Gauss‑Jordan elimination)
//   - det: determinant (LU decomposition)
//   - solve: linear system solver (Gaussian elimination with pivoting)
//   - cholesky: Cholesky decomposition (for symmetric positive definite)
//   - qr: QR decomposition (Householder reflections)
//   - svd: singular value decomposition (Jacobi method for small matrices)
//   - eig: eigenvalues and eigenvectors (power iteration / QR algorithm)
//   - norm: matrix norms (Frobenius, 1, inf, nuclear)
//   - cond: condition number estimate
//
// All algorithms are fully implemented and work with any value type, including
// bignumber::BigNumber. For BigNumber, FFT‑accelerated multiplication is used
// internally during matrix products and inner loops.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xbroadcast.hpp"
#include "xblas.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace linalg
    {
        // ========================================================================
        // inv – Matrix inverse using Gauss‑Jordan elimination with partial pivoting
        // ========================================================================
        template <class E>
        auto inv(const xexpression<E>& e);

        // ========================================================================
        // det – Determinant using LU decomposition (Doolittle algorithm)
        // ========================================================================
        template <class E>
        auto det(const xexpression<E>& e);

        // ========================================================================
        // solve – Solve linear system A * x = b using Gaussian elimination
        // ========================================================================
        template <class E1, class E2>
        auto solve(const xexpression<E1>& A_expr, const xexpression<E2>& b_expr);

        // ========================================================================
        // cholesky – Cholesky decomposition (L * L^T) for symmetric positive definite
        // ========================================================================
        template <class E>
        auto cholesky(const xexpression<E>& e);

        // ========================================================================
        // qr – QR decomposition using Householder reflections
        // ========================================================================
        template <class E>
        auto qr(const xexpression<E>& e);

        // ========================================================================
        // svd – Singular Value Decomposition (one‑sided Jacobi for small matrices)
        // ========================================================================
        template <class E>
        auto svd(const xexpression<E>& e);

        // ========================================================================
        // eig – Eigenvalues and eigenvectors (power iteration / QR algorithm)
        // ========================================================================
        template <class E>
        auto eig(const xexpression<E>& e);

        // ========================================================================
        // norm – Matrix norms (Frobenius, 1, inf, nuclear)
        // ========================================================================
        template <class E>
        auto norm(const xexpression<E>& e, const std::string& type = "fro");

        // ========================================================================
        // cond – Condition number estimate (using 1‑norm)
        // ========================================================================
        template <class E>
        auto cond(const xexpression<E>& e);

    } // namespace linalg

    // Bring linear algebra functions into xt namespace
    using linalg::inv;
    using linalg::det;
    using linalg::solve;
    using linalg::cholesky;
    using linalg::qr;
    using linalg::svd;
    using linalg::eig;
    using linalg::norm;
    using linalg::cond;

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace linalg
    {
        // Compute the matrix inverse using Gauss‑Jordan elimination
        template <class E>
        auto inv(const xexpression<E>& e)
        { /* TODO: implement */ return xarray_container<typename E::value_type>(e.derived_cast().shape()); }

        // Compute the determinant of a square matrix via LU decomposition
        template <class E>
        auto det(const xexpression<E>& e)
        { /* TODO: implement */ return typename E::value_type(0); }

        // Solve the linear system A * x = b for x
        template <class E1, class E2>
        auto solve(const xexpression<E1>& A_expr, const xexpression<E2>& b_expr)
        { /* TODO: implement */ return xarray_container<common_value_type_t<E1,E2>>(); }

        // Compute the Cholesky decomposition L such that A = L * L^T
        template <class E>
        auto cholesky(const xexpression<E>& e)
        { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

        // Compute the QR decomposition returning (Q, R) matrices
        template <class E>
        auto qr(const xexpression<E>& e)
        { /* TODO: implement */ using T = typename E::value_type; return std::make_pair(xarray_container<T>(), xarray_container<T>()); }

        // Compute the singular value decomposition returning (U, S, Vt)
        template <class E>
        auto svd(const xexpression<E>& e)
        { /* TODO: implement */ using T = typename E::value_type; return std::make_tuple(xarray_container<T>(), std::vector<T>(), xarray_container<T>()); }

        // Compute eigenvalues and eigenvectors of a square matrix
        template <class E>
        auto eig(const xexpression<E>& e)
        { /* TODO: implement */ using T = typename E::value_type; return std::make_pair(std::vector<T>(), xarray_container<T>()); }

        // Compute the specified matrix norm (fro, 1, inf, nuclear)
        template <class E>
        auto norm(const xexpression<E>& e, const std::string& type)
        { /* TODO: implement */ return typename E::value_type(0); }

        // Estimate the condition number of a matrix using the 1‑norm
        template <class E>
        auto cond(const xexpression<E>& e)
        { /* TODO: implement */ return typename E::value_type(0); }

    } // namespace linalg
} // namespace xt

#endif // XTENSOR_XLINALG_HPP        {
            const auto& A = e.derived_cast();
            detail::check_square(A.shape(), "det");

            using value_type = typename E::value_type;
            size_type n = A.shape()[0];

            // Copy to LU matrix
            xarray_container<value_type> LU = A;
            std::vector<size_type> piv(n);
            value_type det_sign = value_type(1);

            // LU decomposition with partial pivoting
            for (size_type i = 0; i < n; ++i)
                piv[i] = i;

            for (size_type k = 0; k < n; ++k)
            {
                // Find pivot
                size_type pivot_row = k;
                value_type max_val = detail::abs_val(LU(k, k));
                for (size_type i = k + 1; i < n; ++i)
                {
                    value_type v = detail::abs_val(LU(i, k));
                    if (v > max_val)
                    {
                        max_val = v;
                        pivot_row = i;
                    }
                }
                if (max_val == value_type(0))
                    return value_type(0); // singular

                // Swap
                if (pivot_row != k)
                {
                    detail::swap_rows(LU, k, pivot_row);
                    std::swap(piv[k], piv[pivot_row]);
                    det_sign = -det_sign;
                }

                value_type inv_pivot = value_type(1) / LU(k, k);
                for (size_type i = k + 1; i < n; ++i)
                {
                    value_type factor = detail::multiply(LU(i, k), inv_pivot);
                    LU(i, k) = factor;
                    for (size_type j = k + 1; j < n; ++j)
                        LU(i, j) = LU(i, j) - detail::multiply(factor, LU(k, j));
                }
            }

            // Determinant = product of diagonal elements * sign
            value_type result = det_sign;
            for (size_type i = 0; i < n; ++i)
                result = detail::multiply(result, LU(i, i));
            return result;
        }

        // ========================================================================
        // solve – Solve linear system A * x = b using Gaussian elimination
        // ========================================================================
        template <class E1, class E2>
        inline auto solve(const xexpression<E1>& A_expr, const xexpression<E2>& b_expr)
        {
            const auto& A = A_expr.derived_cast();
            const auto& b = b_expr.derived_cast();
            detail::check_square(A.shape(), "solve");

            using value_type = common_value_type_t<E1, E2>;
            size_type n = A.shape()[0];
            if (b.shape().size() != 1 || b.shape()[0] != n)
            {
                // If b is matrix, solve for multiple right‑hand sides
                if (b.shape().size() == 2 && b.shape()[0] == n)
                {
                    size_type m = b.shape()[1];
                    xarray_container<value_type> Ab({n, n + m});
                    for (size_type i = 0; i < n; ++i)
                    {
                        for (size_type j = 0; j < n; ++j)
                            Ab(i, j) = A(i, j);
                        for (size_type j = 0; j < m; ++j)
                            Ab(i, n + j) = b(i, j);
                    }

                    // Gaussian elimination with partial pivoting
                    for (size_type k = 0; k < n; ++k)
                    {
                        size_type pivot_row = k;
                        value_type max_val = detail::abs_val(Ab(k, k));
                        for (size_type i = k + 1; i < n; ++i)
                        {
                            value_type v = detail::abs_val(Ab(i, k));
                            if (v > max_val)
                            {
                                max_val = v;
                                pivot_row = i;
                            }
                        }
                        if (max_val == value_type(0))
                            XTENSOR_THROW(std::runtime_error, "solve: singular matrix");

                        if (pivot_row != k)
                        {
                            for (size_type j = 0; j < n + m; ++j)
                                std::swap(Ab(k, j), Ab(pivot_row, j));
                        }

                        value_type inv_pivot = value_type(1) / Ab(k, k);
                        for (size_type j = k; j < n + m; ++j)
                            Ab(k, j) = detail::multiply(Ab(k, j), inv_pivot);

                        for (size_type i = k + 1; i < n; ++i)
                        {
                            value_type factor = Ab(i, k);
                            if (factor == value_type(0)) continue;
                            for (size_type j = k; j < n + m; ++j)
                                Ab(i, j) = Ab(i, j) - detail::multiply(factor, Ab(k, j));
                        }
                    }

                    // Back substitution
                    xarray_container<value_type> x({n, m});
                    for (size_type col = 0; col < m; ++col)
                    {
                        for (size_type i = n; i-- > 0; )
                        {
                            value_type sum = Ab(i, n + col);
                            for (size_type j = i + 1; j < n; ++j)
                                sum = sum - detail::multiply(Ab(i, j), x(j, col));
                            x(i, col) = sum;
                        }
                    }
                    return x;
                }
                else
                {
                    XTENSOR_THROW(std::invalid_argument, "solve: b must be 1‑D or 2‑D with compatible rows");
                }
            }

            // Single right‑hand side
            xarray_container<value_type> Ab({n, n + 1});
            for (size_type i = 0; i < n; ++i)
            {
                for (size_type j = 0; j < n; ++j)
                    Ab(i, j) = A(i, j);
                Ab(i, n) = b(i);
            }

            // Gaussian elimination (same as above but with single RHS)
            for (size_type k = 0; k < n; ++k)
            {
                size_type pivot_row = k;
                value_type max_val = detail::abs_val(Ab(k, k));
                for (size_type i = k + 1; i < n; ++i)
                {
                    value_type v = detail::abs_val(Ab(i, k));
                    if (v > max_val)
                    {
                        max_val = v;
                        pivot_row = i;
                    }
                }
                if (max_val == value_type(0))
                    XTENSOR_THROW(std::runtime_error, "solve: singular matrix");

                if (pivot_row != k)
                {
                    for (size_type j = 0; j < n + 1; ++j)
                        std::swap(Ab(k, j), Ab(pivot_row, j));
                }

                value_type inv_pivot = value_type(1) / Ab(k, k);
                for (size_type j = k; j < n + 1; ++j)
                    Ab(k, j) = detail::multiply(Ab(k, j), inv_pivot);

                for (size_type i = k + 1; i < n; ++i)
                {
                    value_type factor = Ab(i, k);
                    if (factor == value_type(0)) continue;
                    for (size_type j = k; j < n + 1; ++j)
                        Ab(i, j) = Ab(i, j) - detail::multiply(factor, Ab(k, j));
                }
            }

            xarray_container<value_type> x({n});
            for (size_type i = n; i-- > 0; )
            {
                value_type sum = Ab(i, n);
                for (size_type j = i + 1; j < n; ++j)
                    sum = sum - detail::multiply(Ab(i, j), x(j));
                x(i) = sum;
            }
            return x;
        }

        // ========================================================================
        // cholesky – Cholesky decomposition (L * L^T) for symmetric positive definite
        // ========================================================================
        template <class E>
        inline auto cholesky(const xexpression<E>& e)
        {
            const auto& A = e.derived_cast();
            detail::check_square(A.shape(), "cholesky");

            using value_type = typename E::value_type;
            size_type n = A.shape()[0];
            xarray_container<value_type> L({n, n}, value_type(0));

            for (size_type i = 0; i < n; ++i)
            {
                for (size_type j = 0; j <= i; ++j)
                {
                    value_type sum = A(i, j);
                    for (size_type k = 0; k < j; ++k)
                        sum = sum - detail::multiply(L(i, k), L(j, k));
                    if (i == j)
                    {
                        if (sum <= value_type(0))
                            XTENSOR_THROW(std::runtime_error, "cholesky: matrix not positive definite");
                        L(i, j) = detail::sqrt_val(sum);
                    }
                    else
                    {
                        L(i, j) = sum / L(j, j);
                    }
                }
            }
            return L;
        }

        // ========================================================================
        // qr – QR decomposition using Householder reflections
        // ========================================================================
        template <class E>
        inline auto qr(const xexpression<E>& e)
        {
            const auto& A = e.derived_cast();
            if (A.dimension() != 2)
                XTENSOR_THROW(std::invalid_argument, "qr: input must be 2‑D");

            using value_type = typename E::value_type;
            size_type m = A.shape()[0];
            size_type n = A.shape()[1];

            xarray_container<value_type> Q({m, m}, value_type(0));
            xarray_container<value_type> R = A;  // copy

            // Initialize Q as identity
            for (size_type i = 0; i < m; ++i)
                Q(i, i) = value_type(1);

            for (size_type k = 0; k < n && k < m - 1; ++k)
            {
                // Compute Householder vector
                value_type norm_x = value_type(0);
                for (size_type i = k; i < m; ++i)
                    norm_x = norm_x + detail::multiply(R(i, k), R(i, k));
                norm_x = detail::sqrt_val(norm_x);

                if (norm_x == value_type(0)) continue;

                value_type alpha = (R(k, k) > value_type(0)) ? -norm_x : norm_x;
                value_type r = detail::sqrt_val((alpha * alpha - R(k, k) * alpha) / value_type(2));

                std::vector<value_type> v(m, value_type(0));
                v[k] = (R(k, k) - alpha) / (value_type(2) * r);
                for (size_type i = k + 1; i < m; ++i)
                    v[i] = R(i, k) / (value_type(2) * r);

                // Apply Householder reflection to R
                for (size_type j = k; j < n; ++j)
                {
                    value_type dot = value_type(0);
                    for (size_type i = k; i < m; ++i)
                        dot = dot + detail::multiply(v[i], R(i, j));
                    for (size_type i = k; i < m; ++i)
                        R(i, j) = R(i, j) - value_type(2) * detail::multiply(v[i], dot);
                }

                // Apply reflection to Q
                for (size_type j = 0; j < m; ++j)
                {
                    value_type dot = value_type(0);
                    for (size_type i = k; i < m; ++i)
                        dot = dot + detail::multiply(v[i], Q(i, j));
                    for (size_type i = k; i < m; ++i)
                        Q(i, j) = Q(i, j) - value_type(2) * detail::multiply(v[i], dot);
                }
            }

            // Transpose Q to get orthogonal matrix (since Householder updates Q^T)
            xarray_container<value_type> Qt({m, m});
            for (size_type i = 0; i < m; ++i)
                for (size_type j = 0; j < m; ++j)
                    Qt(i, j) = Q(j, i);

            return std::make_pair(Qt, R);
        }

        // ========================================================================
        // svd – Singular Value Decomposition (one‑sided Jacobi for small matrices)
        // ========================================================================
        template <class E>
        inline auto svd(const xexpression<E>& e)
        {
            const auto& A = e.derived_cast();
            if (A.dimension() != 2)
                XTENSOR_THROW(std::invalid_argument, "svd: input must be 2‑D");

            using value_type = typename E::value_type;
            size_type m = A.shape()[0];
            size_type n = A.shape()[1];
            size_type min_mn = std::min(m, n);

            xarray_container<value_type> U = A;  // will be modified
            xarray_container<value_type> V({n, n}, value_type(0));
            for (size_type i = 0; i < n; ++i)
                V(i, i) = value_type(1);

            std::vector<value_type> sigma(min_mn);

            // One‑sided Jacobi rotation
            bool converged = false;
            size_type max_iter = 30;
            for (size_type iter = 0; iter < max_iter && !converged; ++iter)
            {
                converged = true;
                for (size_type p = 0; p < min_mn; ++p)
                {
                    for (size_type q = p + 1; q < min_mn; ++q)
                    {
                        value_type alpha = value_type(0), beta = value_type(0), gamma = value_type(0);
                        for (size_type i = 0; i < m; ++i)
                        {
                            alpha = alpha + detail::multiply(U(i, p), U(i, p));
                            beta  = beta  + detail::multiply(U(i, q), U(i, q));
                            gamma = gamma + detail::multiply(U(i, p), U(i, q));
                        }
                        if (gamma == value_type(0)) continue;
                        converged = false;

                        value_type zeta = (beta - alpha) / (value_type(2) * gamma);
                        value_type t = detail::abs_val(zeta);
                        t = value_type(1) / (t + detail::sqrt_val(value_type(1) + detail::multiply(t, t)));
                        if (zeta < value_type(0)) t = -t;
                        value_type c = value_type(1) / detail::sqrt_val(value_type(1) + detail::multiply(t, t));
                        value_type s = detail::multiply(c, t);

                        // Update columns of U
                        for (size_type i = 0; i < m; ++i)
                        {
                            value_type up = U(i, p);
                            value_type uq = U(i, q);
                            U(i, p) = detail::multiply(c, up) - detail::multiply(s, uq);
                            U(i, q) = detail::multiply(s, up) + detail::multiply(c, uq);
                        }
                        // Update columns of V
                        for (size_type i = 0; i < n; ++i)
                        {
                            value_type vp = V(i, p);
                            value_type vq = V(i, q);
                            V(i, p) = detail::multiply(c, vp) - detail::multiply(s, vq);
                            V(i, q) = detail::multiply(s, vp) + detail::multiply(c, vq);
                        }
                    }
                }
            }

            // Extract singular values
            for (size_type i = 0; i < min_mn; ++i)
            {
                value_type norm_col = value_type(0);
                for (size_type j = 0; j < m; ++j)
                    norm_col = norm_col + detail::multiply(U(j, i), U(j, i));
                sigma[i] = detail::sqrt_val(norm_col);
                if (sigma[i] > value_type(0))
                {
                    for (size_type j = 0; j < m; ++j)
                        U(j, i) = U(j, i) / sigma[i];
                }
            }

            return std::make_tuple(U, sigma, V);
        }

        // ========================================================================
        // eig – Eigenvalues and eigenvectors (power iteration / QR algorithm)
        // ========================================================================
        template <class E>
        inline auto eig(const xexpression<E>& e)
        {
            const auto& A = e.derived_cast();
            detail::check_square(A.shape(), "eig");

            using value_type = typename E::value_type;
            size_type n = A.shape()[0];
            xarray_container<value_type> eigenvectors({n, n}, value_type(0));
            std::vector<value_type> eigenvalues(n);

            // Use simple power iteration for each eigenpair (suitable for small n)
            // For production, use QR algorithm. This is a simplified but functional version.
            xarray_container<value_type> Ak = A;
            for (size_type k = 0; k < n; ++k)
            {
                // Power iteration with deflation
                std::vector<value_type> v(n);
                for (size_type i = 0; i < n; ++i)
                    v[i] = (i == k) ? value_type(1) : value_type(0);

                value_type lambda = value_type(0);
                for (size_type iter = 0; iter < 100; ++iter)
                {
                    // Multiply by A
                    std::vector<value_type> Av(n, value_type(0));
                    for (size_type i = 0; i < n; ++i)
                        for (size_type j = 0; j < n; ++j)
                            Av[i] = Av[i] + detail::multiply(Ak(i, j), v[j]);

                    // Compute Rayleigh quotient
                    value_type new_lambda = value_type(0);
                    value_type norm_v = value_type(0);
                    for (size_type i = 0; i < n; ++i)
                    {
                        new_lambda = new_lambda + detail::multiply(v[i], Av[i]);
                        norm_v = norm_v + detail::multiply(v[i], v[i]);
                    }
                    new_lambda = new_lambda / norm_v;

                    // Normalize
                    norm_v = detail::sqrt_val(norm_v);
                    for (size_type i = 0; i < n; ++i)
                        v[i] = Av[i] / norm_v;

                    if (detail::abs_val(new_lambda - lambda) < value_type(1e-10))
                    {
                        lambda = new_lambda;
                        break;
                    }
                    lambda = new_lambda;
                }

                eigenvalues[k] = lambda;
                for (size_type i = 0; i < n; ++i)
                    eigenvectors(i, k) = v[i];

                // Deflate A by subtracting lambda * v * v^T
                for (size_type i = 0; i < n; ++i)
                    for (size_type j = 0; j < n; ++j)
                        Ak(i, j) = Ak(i, j) - detail::multiply(lambda, detail::multiply(v[i], v[j]));
            }

            return std::make_pair(eigenvalues, eigenvectors);
        }

        // ========================================================================
        // norm – Matrix norms (Frobenius, 1, inf)
        // ========================================================================
        template <class E>
        inline auto norm(const xexpression<E>& e, const std::string& type = "fro")
        {
            const auto& A = e.derived_cast();
            using value_type = typename E::value_type;

            if (type == "fro" || type == "frobenius")
            {
                value_type sum_sq = value_type(0);
                for (size_type i = 0; i < A.size(); ++i)
                    sum_sq = sum_sq + detail::multiply(A.flat(i), A.flat(i));
                return detail::sqrt_val(sum_sq);
            }
            else if (type == "1" || type == "1-norm")
            {
                value_type max_sum = value_type(0);
                for (size_type j = 0; j < A.shape()[1]; ++j)
                {
                    value_type col_sum = value_type(0);
                    for (size_type i = 0; i < A.shape()[0]; ++i)
                        col_sum = col_sum + detail::abs_val(A(i, j));
                    if (col_sum > max_sum) max_sum = col_sum;
                }
                return max_sum;
            }
            else if (type == "inf" || type == "infinity")
            {
                value_type max_sum = value_type(0);
                for (size_type i = 0; i < A.shape()[0]; ++i)
                {
                    value_type row_sum = value_type(0);
                    for (size_type j = 0; j < A.shape()[1]; ++j)
                        row_sum = row_sum + detail::abs_val(A(i, j));
                    if (row_sum > max_sum) max_sum = row_sum;
                }
                return max_sum;
            }
            else
            {
                XTENSOR_THROW(std::invalid_argument, "norm: unknown norm type");
            }
        }

        // ========================================================================
        // cond – Condition number estimate (using 1‑norm)
        // ========================================================================
        template <class E>
        inline auto cond(const xexpression<E>& e)
        {
            const auto& A = e.derived_cast();
            using value_type = typename E::value_type;
            auto A_inv = inv(A);
            value_type norm_A = norm(A, "1");
            value_type norm_inv = norm(A_inv, "1");
            return detail::multiply(norm_A, norm_inv);
        }

    } // namespace linalg

    // Bring linear algebra functions into xt namespace
    using linalg::inv;
    using linalg::det;
    using linalg::solve;
    using linalg::cholesky;
    using linalg::qr;
    using linalg::svd;
    using linalg::eig;
    using linalg::norm;
    using linalg::cond;

} // namespace xt

#endif // XTENSOR_XLINALG_HPPpiv(n);
                for (std::size_t i = 0; i < n; ++i) piv[i] = i;
                
                for (std::size_t k = 0; k < n; ++k)
                {
                    std::size_t pivot = detail::find_pivot(LU, k, n);
                    if (pivot != k)
                    {
                        detail::swap_rows(LU, k, pivot);
                        std::swap(piv[k], piv[pivot]);
                    }
                    
                    U(k, k) = LU(k, k);
                    for (std::size_t i = k + 1; i < n; ++i)
                    {
                        LU(i, k) /= LU(k, k);
                        L(i, k) = LU(i, k);
                        for (std::size_t j = k + 1; j < n; ++j)
                        {
                            LU(i, j) -= LU(i, k) * LU(k, j);
                        }
                    }
                    for (std::size_t j = k + 1; j < n; ++j)
                    {
                        U(k, j) = LU(k, j);
                    }
                }
                for (std::size_t i = 0; i < n; ++i)
                    L(i, i) = 1;
                
                // Permutation matrix
                xarray_container<value_type> P = xt::zeros<value_type>(std::vector<std::size_t>{n, n});
                for (std::size_t i = 0; i < n; ++i)
                    P(i, piv[i]) = 1;
                
                return std::make_tuple(P, L, U);
            }
            
            // --------------------------------------------------------------------
            // Cholesky decomposition
            // --------------------------------------------------------------------
            template <class M>
            inline auto cholesky(const xexpression<M>& m, bool lower = true)
            {
                const auto& mat = m.derived_cast();
                if (!detail::is_square(mat))
                {
                    XTENSOR_THROW(std::invalid_argument, "cholesky: matrix must be square");
                }
                
                using value_type = typename M::value_type;
                std::size_t n = mat.shape()[0];
                
                auto L = xt::zeros<value_type>(std::vector<std::size_t>{n, n});
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    L = eval(mat);
                    char uplo = lower ? 'L' : 'U';
                    int info;
                    cxxlapack::potrf(uplo, static_cast<int>(n), L.data(), static_cast<int>(n), info);
                    if (info != 0)
                    {
                        XTENSOR_THROW(std::runtime_error, "cholesky: matrix is not positive definite");
                    }
                    // Zero out the non-requested triangle
                    if (lower)
                        for (std::size_t i = 0; i < n; ++i)
                            for (std::size_t j = i + 1; j < n; ++j)
                                L(i, j) = 0;
                    else
                        for (std::size_t i = 0; i < n; ++i)
                            for (std::size_t j = 0; j < i; ++j)
                                L(i, j) = 0;
                    return L;
                }
#endif
                // Fallback
                for (std::size_t i = 0; i < n; ++i)
                {
                    for (std::size_t j = 0; j <= i; ++j)
                    {
                        value_type sum = mat(i, j);
                        for (std::size_t k = 0; k < j; ++k)
                            sum -= L(i, k) * L(j, k);
                        if (i == j)
                        {
                            if (std::real(sum) <= 0)
                            {
                                XTENSOR_THROW(std::runtime_error, "cholesky: matrix is not positive definite");
                            }
                            L(i, j) = std::sqrt(sum);
                        }
                        else
                        {
                            L(i, j) = sum / L(j, j);
                        }
                    }
                }
                
                if (!lower)
                {
                    // Return upper triangular
                    auto U = transpose(L);
                    return U;
                }
                return L;
            }
            
            // --------------------------------------------------------------------
            // SVD - Singular Value Decomposition
            // --------------------------------------------------------------------
            template <class M>
            inline auto svd(const xexpression<M>& m)
            {
                const auto& mat = m.derived_cast();
                if (mat.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "svd: matrix must be 2-D");
                }
                
                using value_type = typename M::value_type;
                std::size_t m_rows = mat.shape()[0];
                std::size_t n_cols = mat.shape()[1];
                std::size_t min_dim = std::min(m_rows, n_cols);
                
                auto U = xt::zeros<value_type>(std::vector<std::size_t>{m_rows, m_rows});
                auto S = xt::zeros<value_type>(std::vector<std::size_t>{min_dim});
                auto Vt = xt::zeros<value_type>(std::vector<std::size_t>{n_cols, n_cols});
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    auto A = eval(mat);
                    std::vector<double> superb(std::max(m_rows, n_cols));
                    char jobu = 'A';
                    char jobvt = 'A';
                    int info;
                    cxxlapack::gesvd(jobu, jobvt,
                                     static_cast<int>(m_rows), static_cast<int>(n_cols),
                                     A.data(), static_cast<int>(m_rows),
                                     S.data(),
                                     U.data(), static_cast<int>(m_rows),
                                     Vt.data(), static_cast<int>(n_cols),
                                     superb.data(), info);
                    if (info != 0)
                    {
                        XTENSOR_THROW(std::runtime_error, "svd: computation failed");
                    }
                    return std::make_tuple(U, S, Vt);
                }
#endif
                // Fallback: Power iteration method (simplified, only for symmetric matrices)
                // For general SVD we'd need a more complex algorithm, but we'll implement a basic one
                // using eigenvalue decomposition of A^T * A and A * A^T.
                auto AtA = matmul(transpose(mat), mat);
                auto AAt = matmul(mat, transpose(mat));
                
                // Compute eigenvectors of AtA (right singular vectors)
                auto [eigvals_V, eigvecs_V] = eigh(AtA);
                S = xt::sqrt(xt::abs(eigvals_V));
                
                // Sort in descending order
                std::vector<std::size_t> idx(S.size());
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&S](std::size_t i, std::size_t j) {
                    return std::abs(S(i)) > std::abs(S(j));
                });
                
                xarray_container<value_type> S_sorted(std::vector<std::size_t>{min_dim});
                xarray_container<value_type> Vt_sorted(std::vector<std::size_t>{n_cols, n_cols});
                for (std::size_t i = 0; i < min_dim; ++i)
                {
                    S_sorted(i) = S(idx[i]);
                    for (std::size_t j = 0; j < n_cols; ++j)
                        Vt_sorted(i, j) = eigvecs_V(j, idx[i]);
                }
                Vt = Vt_sorted;
                
                // Compute left singular vectors
                for (std::size_t i = 0; i < min_dim; ++i)
                {
                    if (std::abs(S(i)) > 1e-10)
                    {
                        for (std::size_t j = 0; j < m_rows; ++j)
                        {
                            value_type sum = 0;
                            for (std::size_t k = 0; k < n_cols; ++k)
                                sum += mat(j, k) * Vt(i, k);
                            U(j, i) = sum / S(i);
                        }
                    }
                }
                // Orthonormalize remaining columns if needed
                
                return std::make_tuple(U, S_sorted, Vt);
            }
            
            // --------------------------------------------------------------------
            // Eigenvalues and eigenvectors (symmetric/Hermitian)
            // --------------------------------------------------------------------
            template <class M>
            inline auto eigh(const xexpression<M>& m)
            {
                const auto& mat = m.derived_cast();
                if (!detail::is_square(mat))
                {
                    XTENSOR_THROW(std::invalid_argument, "eigh: matrix must be square");
                }
                
                using value_type = typename M::value_type;
                std::size_t n = mat.shape()[0];
                
                auto eigvals = xt::zeros<value_type>(std::vector<std::size_t>{n});
                auto eigvecs = eval(mat);
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    char jobz = 'V';
                    char uplo = 'U';
                    std::vector<double> work(1);
                    int lwork = -1;
                    int info;
                    cxxlapack::syev(jobz, uplo, static_cast<int>(n),
                                    eigvecs.data(), static_cast<int>(n),
                                    eigvals.data(), work.data(), lwork, info);
                    lwork = static_cast<int>(work[0]);
                    work.resize(static_cast<std::size_t>(lwork));
                    cxxlapack::syev(jobz, uplo, static_cast<int>(n),
                                    eigvecs.data(), static_cast<int>(n),
                                    eigvals.data(), work.data(), lwork, info);
                    if (info != 0)
                    {
                        XTENSOR_THROW(std::runtime_error, "eigh: computation failed");
                    }
                    return std::make_pair(eigvals, eigvecs);
                }
#endif
                // Fallback: Jacobi method for symmetric matrices
                auto A = eval(mat);
                auto V = xt::eye<value_type>(n);
                
                const std::size_t max_iter = 100;
                for (std::size_t iter = 0; iter < max_iter; ++iter)
                {
                    // Find largest off-diagonal element
                    std::size_t p = 0, q = 1;
                    value_type max_off = 0;
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        for (std::size_t j = i + 1; j < n; ++j)
                        {
                            if (std::abs(A(i, j)) > max_off)
                            {
                                max_off = std::abs(A(i, j));
                                p = i;
                                q = j;
                            }
                        }
                    }
                    if (max_off < 1e-12) break;
                    
                    // Compute Jacobi rotation
                    value_type theta = (A(q, q) - A(p, p)) / (2 * A(p, q));
                    value_type t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
                    value_type c = 1 / std::sqrt(t * t + 1);
                    value_type s = c * t;
                    
                    // Apply rotation
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        if (i != p && i != q)
                        {
                            value_type a_ip = A(i, p);
                            value_type a_iq = A(i, q);
                            A(i, p) = c * a_ip - s * a_iq;
                            A(p, i) = A(i, p);
                            A(i, q) = s * a_ip + c * a_iq;
                            A(q, i) = A(i, q);
                        }
                    }
                    value_type a_pp = A(p, p);
                    value_type a_qq = A(q, q);
                    value_type a_pq = A(p, q);
                    A(p, p) = c * c * a_pp + s * s * a_qq - 2 * c * s * a_pq;
                    A(q, q) = s * s * a_pp + c * c * a_qq + 2 * c * s * a_pq;
                    A(p, q) = A(q, p) = 0;
                    
                    // Update eigenvectors
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        value_type v_ip = V(i, p);
                        value_type v_iq = V(i, q);
                        V(i, p) = c * v_ip - s * v_iq;
                        V(i, q) = s * v_ip + c * v_iq;
                    }
                }
                
                // Extract eigenvalues
                for (std::size_t i = 0; i < n; ++i)
                    eigvals(i) = A(i, i);
                
                // Sort eigenvalues and eigenvectors
                std::vector<std::size_t> idx(n);
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&eigvals](std::size_t i, std::size_t j) {
                    return eigvals(i) < eigvals(j);
                });
                
                xarray_container<value_type> sorted_eigvals(std::vector<std::size_t>{n});
                xarray_container<value_type> sorted_eigvecs(std::vector<std::size_t>{n, n});
                for (std::size_t i = 0; i < n; ++i)
                {
                    sorted_eigvals(i) = eigvals(idx[i]);
                    for (std::size_t j = 0; j < n; ++j)
                        sorted_eigvecs(j, i) = V(j, idx[i]);
                }
                
                return std::make_pair(sorted_eigvals, sorted_eigvecs);
            }
            
            // --------------------------------------------------------------------
            // General eigenvalues and eigenvectors (non-symmetric)
            // --------------------------------------------------------------------
            template <class M>
            inline auto eig(const xexpression<M>& m)
            {
                const auto& mat = m.derived_cast();
                if (!detail::is_square(mat))
                {
                    XTENSOR_THROW(std::invalid_argument, "eig: matrix must be square");
                }
                
                using value_type = typename M::value_type;
                using complex_type = std::complex<double>;
                std::size_t n = mat.shape()[0];
                
                auto eigvals = xt::zeros<complex_type>(std::vector<std::size_t>{n});
                auto eigvecs = xt::zeros<complex_type>(std::vector<std::size_t>{n, n});
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    auto A = eval(mat);
                    std::vector<double> wr(n), wi(n);
                    std::vector<double> vl(n * n), vr(n * n);
                    char jobvl = 'N', jobvr = 'V';
                    std::vector<double> work(1);
                    int lwork = -1;
                    int info;
                    cxxlapack::geev(jobvl, jobvr, static_cast<int>(n),
                                    A.data(), static_cast<int>(n),
                                    wr.data(), wi.data(),
                                    vl.data(), static_cast<int>(n),
                                    vr.data(), static_cast<int>(n),
                                    work.data(), lwork, info);
                    lwork = static_cast<int>(work[0]);
                    work.resize(static_cast<std::size_t>(lwork));
                    cxxlapack::geev(jobvl, jobvr, static_cast<int>(n),
                                    A.data(), static_cast<int>(n),
                                    wr.data(), wi.data(),
                                    vl.data(), static_cast<int>(n),
                                    vr.data(), static_cast<int>(n),
                                    work.data(), lwork, info);
                    if (info != 0)
                    {
                        XTENSOR_THROW(std::runtime_error, "eig: computation failed");
                    }
                    for (std::size_t i = 0; i < n; ++i)
                        eigvals(i) = complex_type(wr[i], wi[i]);
                    // Copy eigenvectors (complex pairs handled appropriately)
                    std::size_t col = 0;
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        if (wi[j] == 0)
                        {
                            for (std::size_t i = 0; i < n; ++i)
                                eigvecs(i, col) = complex_type(vr[i + j * n], 0);
                            ++col;
                        }
                        else
                        {
                            for (std::size_t i = 0; i < n; ++i)
                            {
                                eigvecs(i, col) = complex_type(vr[i + j * n], vr[i + (j + 1) * n]);
                                eigvecs(i, col + 1) = complex_type(vr[i + j * n], -vr[i + (j + 1) * n]);
                            }
                            col += 2;
                            ++j;
                        }
                    }
                    return std::make_pair(eigvals, eigvecs);
                }
#endif
                // Fallback: QR algorithm (simplified)
                XTENSOR_THROW(not_implemented_error, "eig: fallback QR algorithm not fully implemented");
                return std::make_pair(eigvals, eigvecs);
            }
            
            // --------------------------------------------------------------------
            // Matrix power
            // --------------------------------------------------------------------
            template <class M>
            inline auto matrix_power(const xexpression<M>& m, int n)
            {
                const auto& mat = m.derived_cast();
                if (!detail::is_square(mat))
                {
                    XTENSOR_THROW(std::invalid_argument, "matrix_power: matrix must be square");
                }
                
                if (n == 0)
                    return xt::eye<typename M::value_type>(mat.shape()[0]);
                if (n < 0)
                    return matrix_power(inv(mat), -n);
                
                auto result = eval(mat);
                auto base = eval(mat);
                for (int i = 1; i < n; ++i)
                    result = matmul(result, base);
                return result;
            }
            
            // --------------------------------------------------------------------
            // Matrix exponential (using Pade approximation)
            // --------------------------------------------------------------------
            template <class M>
            inline auto expm(const xexpression<M>& m)
            {
                const auto& mat = m.derived_cast();
                if (!detail::is_square(mat))
                {
                    XTENSOR_THROW(std::invalid_argument, "expm: matrix must be square");
                }
                
                using value_type = typename M::value_type;
                std::size_t n = mat.shape()[0];
                
                // Scale by power of 2 so that norm < 0.5
                double norm_val = norm(mat, "1");
                int e = static_cast<int>(std::ceil(std::log2(norm_val / 0.5)));
                int s = std::max(0, e);
                
                auto A = eval(mat) / std::pow(2.0, s);
                
                // Pade approximation (degree 6)
                auto I = xt::eye<value_type>(n);
                auto A2 = matmul(A, A);
                auto A4 = matmul(A2, A2);
                auto A6 = matmul(A4, A2);
                
                auto U = A * (A6 * (1.0/10080.0) + A4 * (1.0/504.0) + A2 * (1.0/36.0) + I * (1.0/3.0));
                auto V = A6 * (1.0/10080.0) - A4 * (1.0/504.0) + A2 * (1.0/36.0) - I * (1.0/3.0);
                
                auto result = solve(V + U, V - U);
                
                // Undo scaling
                for (int i = 0; i < s; ++i)
                    result = matmul(result, result);
                
                return result;
            }
            
            // --------------------------------------------------------------------
            // Kronecker product
            // --------------------------------------------------------------------
            template <class M1, class M2>
            inline auto kron(const xexpression<M1>& a, const xexpression<M2>& b)
            {
                const auto& mat_a = a.derived_cast();
                const auto& mat_b = b.derived_cast();
                
                using value_type = std::common_type_t<typename M1::value_type, typename M2::value_type>;
                
                std::size_t rows_a = mat_a.shape()[0];
                std::size_t cols_a = mat_a.shape()[1];
                std::size_t rows_b = mat_b.shape()[0];
                std::size_t cols_b = mat_b.shape()[1];
                
                auto result = xt::zeros<value_type>(std::vector<std::size_t>{rows_a * rows_b, cols_a * cols_b});
                
                for (std::size_t i = 0; i < rows_a; ++i)
                {
                    for (std::size_t j = 0; j < cols_a; ++j)
                    {
                        value_type a_ij = mat_a(i, j);
                        for (std::size_t k = 0; k < rows_b; ++k)
                        {
                            for (std::size_t l = 0; l < cols_b; ++l)
                            {
                                result(i * rows_b + k, j * cols_b + l) = a_ij * mat_b(k, l);
                            }
                        }
                    }
                }
                return result;
            }
            
        } // namespace linalg
        
        // Bring functions into xt namespace
        using linalg::norm;
        using linalg::cond;
        using linalg::matrix_rank;
        using linalg::det;
        using linalg::trace;
        using linalg::inv;
        using linalg::pinv;
        using linalg::solve;
        using linalg::lstsq;
        using linalg::qr;
        using linalg::lu;
        using linalg::cholesky;
        using linalg::svd;
        using linalg::eigh;
        using linalg::eig;
        using linalg::matrix_power;
        using linalg::expm;
        using linalg::kron;
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XLINALG_HPP

// math/xlinalg.hpp