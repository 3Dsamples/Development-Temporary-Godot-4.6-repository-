// math/xlinalg.hpp

#ifndef XTENSOR_XLINALG_HPP
#define XTENSOR_XLINALG_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "xblas.hpp"

#include <complex>
#include <type_traits>
#include <utility>
#include <vector>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <tuple>

#if XTENSOR_HAS_BLAS
    #include <cxxlapack.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace linalg
        {
            // --------------------------------------------------------------------
            // Utility functions
            // --------------------------------------------------------------------
            namespace detail
            {
                // Check if matrix is square
                template <class M>
                inline bool is_square(const M& m)
                {
                    return m.dimension() == 2 && m.shape()[0] == m.shape()[1];
                }
                
                // Check if matrix is symmetric (within tolerance)
                template <class M>
                inline bool is_symmetric(const M& m, double tol = 1e-10)
                {
                    if (!is_square(m)) return false;
                    std::size_t n = m.shape()[0];
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        for (std::size_t j = i + 1; j < n; ++j)
                        {
                            if (std::abs(m(i, j) - m(j, i)) > tol)
                                return false;
                        }
                    }
                    return true;
                }
                
                // Check if matrix is positive definite (Cholesky attempt)
                template <class M>
                inline bool is_positive_definite(const M& m)
                {
                    if (!is_symmetric(m)) return false;
                    std::size_t n = m.shape()[0];
                    // Try Cholesky decomposition
                    auto L = eval(m);
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        for (std::size_t j = 0; j <= i; ++j)
                        {
                            double sum = static_cast<double>(L(i, j));
                            for (std::size_t k = 0; k < j; ++k)
                                sum -= static_cast<double>(L(i, k)) * static_cast<double>(L(j, k));
                            if (i == j)
                            {
                                if (sum <= 0) return false;
                                L(i, j) = static_cast<typename M::value_type>(std::sqrt(sum));
                            }
                            else
                            {
                                L(i, j) = static_cast<typename M::value_type>(sum / static_cast<double>(L(j, j)));
                            }
                        }
                    }
                    return true;
                }
                
                // Pivot for LU decomposition
                template <class T>
                inline std::size_t find_pivot(const xarray_container<T>& A, std::size_t k, std::size_t n)
                {
                    std::size_t pivot = k;
                    T max_val = std::abs(A(k, k));
                    for (std::size_t i = k + 1; i < n; ++i)
                    {
                        T abs_val = std::abs(A(i, k));
                        if (abs_val > max_val)
                        {
                            max_val = abs_val;
                            pivot = i;
                        }
                    }
                    return pivot;
                }
                
                // Swap rows of a matrix
                template <class T>
                inline void swap_rows(xarray_container<T>& A, std::size_t r1, std::size_t r2)
                {
                    std::size_t n = A.shape()[1];
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        std::swap(A(r1, j), A(r2, j));
                    }
                }
                
                // Swap elements of a vector
                template <class V>
                inline void swap_elements(V& v, std::size_t i, std::size_t j)
                {
                    std::swap(v(i), v(j));
                }
            }
            
            // --------------------------------------------------------------------
            // Matrix norms
            // --------------------------------------------------------------------
            template <class M>
            inline auto norm(const xexpression<M>& m, const std::string& ord = "fro")
            {
                const auto& mat = m.derived_cast();
                using value_type = typename M::value_type;
                using real_type = typename std::conditional_t<std::is_arithmetic_v<value_type>,
                                                              value_type,
                                                              typename value_type::value_type>;
                
                if (mat.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "norm: matrix must be 2-D");
                }
                
                if (ord == "fro" || ord == "frobenius")
                {
                    real_type sum = 0;
                    for (std::size_t i = 0; i < mat.size(); ++i)
                    {
                        if constexpr (std::is_arithmetic_v<value_type>)
                            sum += mat.flat(i) * mat.flat(i);
                        else
                            sum += std::norm(mat.flat(i));
                    }
                    return std::sqrt(sum);
                }
                else if (ord == "nuc" || ord == "nuclear")
                {
                    // Sum of singular values
                    auto s = svd(mat);
                    const auto& S = std::get<1>(s);
                    real_type sum = 0;
                    for (std::size_t i = 0; i < S.size(); ++i)
                        sum += std::abs(S(i));
                    return sum;
                }
                else if (ord == "inf" || std::isdigit(ord[0]))
                {
                    int p = (ord == "inf") ? -1 : std::stoi(ord);
                    if (p == 1)
                    {
                        // Max absolute column sum
                        real_type max_sum = 0;
                        for (std::size_t j = 0; j < mat.shape()[1]; ++j)
                        {
                            real_type col_sum = 0;
                            for (std::size_t i = 0; i < mat.shape()[0]; ++i)
                                col_sum += std::abs(mat(i, j));
                            max_sum = std::max(max_sum, col_sum);
                        }
                        return max_sum;
                    }
                    else if (p == -1 || p == std::numeric_limits<int>::max())
                    {
                        // Max absolute row sum
                        real_type max_sum = 0;
                        for (std::size_t i = 0; i < mat.shape()[0]; ++i)
                        {
                            real_type row_sum = 0;
                            for (std::size_t j = 0; j < mat.shape()[1]; ++j)
                                row_sum += std::abs(mat(i, j));
                            max_sum = std::max(max_sum, row_sum);
                        }
                        return max_sum;
                    }
                    else if (p == 2)
                    {
                        // Largest singular value
                        auto s = svd(mat);
                        const auto& S = std::get<1>(s);
                        return S(0);
                    }
                }
                
                XTENSOR_THROW(std::invalid_argument, "norm: unsupported order");
                return real_type(0);
            }
            
            // --------------------------------------------------------------------
            // Matrix condition number
            // --------------------------------------------------------------------
            template <class M>
            inline auto cond(const xexpression<M>& m, const std::string& p = "2")
            {
                const auto& mat = m.derived_cast();
                if (p == "2")
                {
                    auto s = svd(mat);
                    const auto& S = std::get<1>(s);
                    if (S(S.size() - 1) == 0)
                        return std::numeric_limits<double>::infinity();
                    return static_cast<double>(S(0)) / static_cast<double>(S(S.size() - 1));
                }
                else
                {
                    auto inv = inv(mat);
                    return norm(mat, p) * norm(inv, p);
                }
            }
            
            // --------------------------------------------------------------------
            // Matrix rank
            // --------------------------------------------------------------------
            template <class M>
            inline std::size_t matrix_rank(const xexpression<M>& m, double tol = 1e-10)
            {
                const auto& mat = m.derived_cast();
                auto s = svd(mat);
                const auto& S = std::get<1>(s);
                double max_s = static_cast<double>(S(0));
                if (max_s == 0) return 0;
                std::size_t rank = 0;
                for (std::size_t i = 0; i < S.size(); ++i)
                {
                    if (static_cast<double>(std::abs(S(i))) > tol * max_s)
                        ++rank;
                }
                return rank;
            }
            
            // --------------------------------------------------------------------
            // Matrix determinant
            // --------------------------------------------------------------------
            template <class M>
            inline auto det(const xexpression<M>& m)
            {
                const auto& mat = m.derived_cast();
                if (!detail::is_square(mat))
                {
                    XTENSOR_THROW(std::invalid_argument, "det: matrix must be square");
                }
                
                using value_type = typename M::value_type;
                std::size_t n = mat.shape()[0];
                
                // Use LU decomposition
                auto LU = eval(mat);
                std::vector<std::size_t> piv(n);
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    std::vector<int> ipiv(n);
                    int info;
                    cxxlapack::getrf(static_cast<int>(n), static_cast<int>(n),
                                     LU.data(), static_cast<int>(n), ipiv.data(), info);
                    if (info != 0) return value_type(0);
                    
                    value_type det_val = 1;
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        det_val *= LU(i, i);
                        if (ipiv[i] != static_cast<int>(i) + 1)
                            det_val = -det_val;
                    }
                    return det_val;
                }
#endif
                // Fallback LU
                for (std::size_t i = 0; i < n; ++i)
                    piv[i] = i;
                
                std::size_t sign = 1;
                for (std::size_t k = 0; k < n; ++k)
                {
                    std::size_t pivot = detail::find_pivot(LU, k, n);
                    if (pivot != k)
                    {
                        detail::swap_rows(LU, k, pivot);
                        std::swap(piv[k], piv[pivot]);
                        sign = -sign;
                    }
                    if (LU(k, k) == 0) return value_type(0);
                    
                    for (std::size_t i = k + 1; i < n; ++i)
                    {
                        LU(i, k) /= LU(k, k);
                        for (std::size_t j = k + 1; j < n; ++j)
                        {
                            LU(i, j) -= LU(i, k) * LU(k, j);
                        }
                    }
                }
                
                value_type det_val = sign;
                for (std::size_t i = 0; i < n; ++i)
                    det_val *= LU(i, i);
                return det_val;
            }
            
            // --------------------------------------------------------------------
            // Matrix trace
            // --------------------------------------------------------------------
            template <class M>
            inline auto trace(const xexpression<M>& m)
            {
                const auto& mat = m.derived_cast();
                if (mat.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "trace: matrix must be 2-D");
                }
                std::size_t n = std::min(mat.shape()[0], mat.shape()[1]);
                using value_type = typename M::value_type;
                value_type sum = 0;
                for (std::size_t i = 0; i < n; ++i)
                    sum += mat(i, i);
                return sum;
            }
            
            // --------------------------------------------------------------------
            // Matrix inverse
            // --------------------------------------------------------------------
            template <class M>
            inline auto inv(const xexpression<M>& m)
            {
                const auto& mat = m.derived_cast();
                if (!detail::is_square(mat))
                {
                    XTENSOR_THROW(std::invalid_argument, "inv: matrix must be square");
                }
                
                using value_type = typename M::value_type;
                std::size_t n = mat.shape()[0];
                auto result = eval(mat);
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    std::vector<int> ipiv(n);
                    int info;
                    cxxlapack::getrf(static_cast<int>(n), static_cast<int>(n),
                                     result.data(), static_cast<int>(n), ipiv.data(), info);
                    if (info != 0)
                    {
                        XTENSOR_THROW(std::runtime_error, "inv: matrix is singular");
                    }
                    cxxlapack::getri(static_cast<int>(n), result.data(), static_cast<int>(n),
                                     ipiv.data(), info);
                    if (info != 0)
                    {
                        XTENSOR_THROW(std::runtime_error, "inv: inversion failed");
                    }
                    return result;
                }
#endif
                // Fallback: Gauss-Jordan elimination
                xarray_container<value_type> augmented(n, std::vector<std::size_t>{n, 2 * n});
                for (std::size_t i = 0; i < n; ++i)
                {
                    for (std::size_t j = 0; j < n; ++j)
                        augmented(i, j) = mat(i, j);
                    for (std::size_t j = 0; j < n; ++j)
                        augmented(i, j + n) = (i == j) ? 1 : 0;
                }
                
                for (std::size_t i = 0; i < n; ++i)
                {
                    std::size_t pivot = i;
                    for (std::size_t k = i + 1; k < n; ++k)
                    {
                        if (std::abs(augmented(k, i)) > std::abs(augmented(pivot, i)))
                            pivot = k;
                    }
                    if (pivot != i)
                    {
                        for (std::size_t j = 0; j < 2 * n; ++j)
                            std::swap(augmented(i, j), augmented(pivot, j));
                    }
                    
                    value_type piv_val = augmented(i, i);
                    if (piv_val == 0)
                    {
                        XTENSOR_THROW(std::runtime_error, "inv: matrix is singular");
                    }
                    for (std::size_t j = 0; j < 2 * n; ++j)
                        augmented(i, j) /= piv_val;
                    
                    for (std::size_t k = 0; k < n; ++k)
                    {
                        if (k != i)
                        {
                            value_type factor = augmented(k, i);
                            for (std::size_t j = 0; j < 2 * n; ++j)
                                augmented(k, j) -= factor * augmented(i, j);
                        }
                    }
                }
                
                xarray_container<value_type> inv_mat(std::vector<std::size_t>{n, n});
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        inv_mat(i, j) = augmented(i, j + n);
                return inv_mat;
            }
            
            // --------------------------------------------------------------------
            // Matrix pseudo-inverse (Moore-Penrose)
            // --------------------------------------------------------------------
            template <class M>
            inline auto pinv(const xexpression<M>& m, double rcond = 1e-15)
            {
                const auto& mat = m.derived_cast();
                auto [U, s, Vt] = svd(mat);
                
                using value_type = typename M::value_type;
                std::size_t m_rows = mat.shape()[0];
                std::size_t n_cols = mat.shape()[1];
                std::size_t min_dim = std::min(m_rows, n_cols);
                
                // Threshold for singular values
                double cutoff = rcond * static_cast<double>(std::abs(s(0)));
                
                // Compute pseudo-inverse of singular values
                xarray_container<value_type> s_inv(std::vector<std::size_t>{min_dim}, value_type(0));
                for (std::size_t i = 0; i < min_dim; ++i)
                {
                    if (std::abs(s(i)) > cutoff)
                        s_inv(i) = value_type(1) / s(i);
                }
                
                // Compute V * S^+ * U^T
                auto V = transpose(Vt);
                xarray_container<value_type> result(std::vector<std::size_t>{n_cols, m_rows}, value_type(0));
                
                for (std::size_t i = 0; i < n_cols; ++i)
                {
                    for (std::size_t j = 0; j < m_rows; ++j)
                    {
                        value_type sum = 0;
                        for (std::size_t k = 0; k < min_dim; ++k)
                        {
                            sum += V(i, k) * s_inv(k) * U(j, k);
                        }
                        result(i, j) = sum;
                    }
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Solve linear system A * x = b
            // --------------------------------------------------------------------
            template <class M, class V>
            inline auto solve(const xexpression<M>& a, const xexpression<V>& b)
            {
                const auto& mat_a = a.derived_cast();
                const auto& vec_b = b.derived_cast();
                
                if (!detail::is_square(mat_a))
                {
                    XTENSOR_THROW(std::invalid_argument, "solve: A must be square");
                }
                if (vec_b.dimension() == 1)
                {
                    if (mat_a.shape()[0] != vec_b.size())
                    {
                        XTENSOR_THROW(std::invalid_argument, "solve: dimension mismatch");
                    }
                }
                else if (vec_b.dimension() == 2)
                {
                    if (mat_a.shape()[0] != vec_b.shape()[0])
                    {
                        XTENSOR_THROW(std::invalid_argument, "solve: dimension mismatch");
                    }
                }
                else
                {
                    XTENSOR_THROW(std::invalid_argument, "solve: b must be 1-D or 2-D");
                }
                
                using value_type = typename M::value_type;
                std::size_t n = mat_a.shape()[0];
                auto result = eval(vec_b);
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, double>)
                {
                    auto A = eval(mat_a);
                    std::vector<int> ipiv(n);
                    int info;
                    cxxlapack::gesv(static_cast<int>(n), static_cast<int>(vec_b.dimension() == 1 ? 1 : vec_b.shape()[1]),
                                    A.data(), static_cast<int>(n), ipiv.data(),
                                    result.data(), static_cast<int>(n), info);
                    if (info != 0)
                    {
                        XTENSOR_THROW(std::runtime_error, "solve: matrix is singular");
                    }
                    return result;
                }
#endif
                // Fallback: LU decomposition
                auto LU = eval(mat_a);
                std::vector<std::size_t> piv(n);
                for (std::size_t i = 0; i < n; ++i) piv[i] = i;
                
                for (std::size_t k = 0; k < n; ++k)
                {
                    std::size_t pivot = detail::find_pivot(LU, k, n);
                    if (pivot != k)
                    {
                        detail::swap_rows(LU, k, pivot);
                        std::swap(piv[k], piv[pivot]);
                    }
                    for (std::size_t i = k + 1; i < n; ++i)
                    {
                        LU(i, k) /= LU(k, k);
                        for (std::size_t j = k + 1; j < n; ++j)
                            LU(i, j) -= LU(i, k) * LU(k, j);
                    }
                }
                
                // Apply permutation to b
                auto b_perm = eval(vec_b);
                if (vec_b.dimension() == 1)
                {
                    for (std::size_t i = 0; i < n; ++i)
                        b_perm(i) = vec_b(piv[i]);
                    
                    // Forward substitution
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        for (std::size_t j = 0; j < i; ++j)
                            b_perm(i) -= LU(i, j) * b_perm(j);
                    }
                    // Backward substitution
                    for (std::size_t i = n; i-- > 0; )
                    {
                        for (std::size_t j = i + 1; j < n; ++j)
                            b_perm(i) -= LU(i, j) * b_perm(j);
                        b_perm(i) /= LU(i, i);
                    }
                }
                else
                {
                    std::size_t nrhs = vec_b.shape()[1];
                    for (std::size_t j = 0; j < nrhs; ++j)
                    {
                        for (std::size_t i = 0; i < n; ++i)
                            b_perm(i, j) = vec_b(piv[i], j);
                    }
                    // Forward substitution
                    for (std::size_t j = 0; j < nrhs; ++j)
                        for (std::size_t i = 0; i < n; ++i)
                            for (std::size_t k = 0; k < i; ++k)
                                b_perm(i, j) -= LU(i, k) * b_perm(k, j);
                    // Backward substitution
                    for (std::size_t j = 0; j < nrhs; ++j)
                        for (std::size_t i = n; i-- > 0; )
                        {
                            for (std::size_t k = i + 1; k < n; ++k)
                                b_perm(i, j) -= LU(i, k) * b_perm(k, j);
                            b_perm(i, j) /= LU(i, i);
                        }
                }
                return b_perm;
            }
            
            // --------------------------------------------------------------------
            // Least squares solution: min ||A*x - b||
            // --------------------------------------------------------------------
            template <class M, class V>
            inline auto lstsq(const xexpression<M>& a, const xexpression<V>& b)
            {
                const auto& mat_a = a.derived_cast();
                const auto& vec_b = b.derived_cast();
                
                if (mat_a.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "lstsq: A must be 2-D");
                }
                
                // Use pseudo-inverse: x = pinv(A) * b
                auto A_pinv = pinv(mat_a);
                
                if (vec_b.dimension() == 1)
                {
                    return matvec(A_pinv, vec_b);
                }
                else
                {
                    return matmul(A_pinv, vec_b);
                }
            }
            
            // --------------------------------------------------------------------
            // QR decomposition
            // --------------------------------------------------------------------
            template <class M>
            inline auto qr(const xexpression<M>& m)
            {
                const auto& mat = m.derived_cast();
                if (mat.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "qr: matrix must be 2-D");
                }
                
                using value_type = typename M::value_type;
                std::size_t m_rows = mat.shape()[0];
                std::size_t n_cols = mat.shape()[1];
                std::size_t min_dim = std::min(m_rows, n_cols);
                
                auto Q = eval(mat);
                auto R = xt::zeros<value_type>(std::vector<std::size_t>{min_dim, n_cols});
                
                // Gram-Schmidt process
                std::vector<xarray_container<value_type>> q_columns;
                
                for (std::size_t k = 0; k < min_dim; ++k)
                {
                    // Extract column k
                    xarray_container<value_type> a_k(std::vector<std::size_t>{m_rows});
                    for (std::size_t i = 0; i < m_rows; ++i)
                        a_k(i) = mat(i, k);
                    
                    // Subtract projections onto previous q vectors
                    for (std::size_t j = 0; j < k; ++j)
                    {
                        value_type r_jk = 0;
                        for (std::size_t i = 0; i < m_rows; ++i)
                            r_jk += q_columns[j](i) * a_k(i);
                        R(j, k) = r_jk;
                        for (std::size_t i = 0; i < m_rows; ++i)
                            a_k(i) -= r_jk * q_columns[j](i);
                    }
                    
                    // Normalize
                    value_type norm = 0;
                    for (std::size_t i = 0; i < m_rows; ++i)
                        norm += a_k(i) * a_k(i);
                    norm = std::sqrt(norm);
                    R(k, k) = norm;
                    if (norm > 0)
                    {
                        for (std::size_t i = 0; i < m_rows; ++i)
                            a_k(i) /= norm;
                    }
                    q_columns.push_back(a_k);
                    
                    for (std::size_t i = 0; i < m_rows; ++i)
                        Q(i, k) = a_k(i);
                }
                
                // Compute remaining R entries
                for (std::size_t k = min_dim; k < n_cols; ++k)
                {
                    for (std::size_t j = 0; j < min_dim; ++j)
                    {
                        value_type sum = 0;
                        for (std::size_t i = 0; i < m_rows; ++i)
                            sum += q_columns[j](i) * mat(i, k);
                        R(j, k) = sum;
                    }
                }
                
                return std::make_pair(Q, R);
            }
            
            // --------------------------------------------------------------------
            // LU decomposition
            // --------------------------------------------------------------------
            template <class M>
            inline auto lu(const xexpression<M>& m)
            {
                const auto& mat = m.derived_cast();
                if (!detail::is_square(mat))
                {
                    XTENSOR_THROW(std::invalid_argument, "lu: matrix must be square");
                }
                
                using value_type = typename M::value_type;
                std::size_t n = mat.shape()[0];
                
                auto LU = eval(mat);
                auto L = xt::zeros<value_type>(std::vector<std::size_t>{n, n});
                auto U = xt::zeros<value_type>(std::vector<std::size_t>{n, n});
                std::vector<std::size_t> piv(n);
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