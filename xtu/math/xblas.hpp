// math/xblas.hpp

#ifndef XTENSOR_XBLAS_HPP
#define XTENSOR_XBLAS_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"

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
#include <cstring>

// Optional BLAS/LAPACK support
#if XTENSOR_HAS_BLAS
    #include <cxxblas.hpp>
    #include <cxxlapack.hpp>
    #define XTENSOR_USE_BLAS 1
#else
    #define XTENSOR_USE_BLAS 0
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        // --------------------------------------------------------------------
        // BLAS utility enums and tags
        // --------------------------------------------------------------------
        enum class blas_order
        {
            row_major,
            column_major
        };
        
        enum class blas_transpose
        {
            no_trans,
            trans,
            conj_trans
        };
        
        enum class blas_uplo
        {
            upper,
            lower
        };
        
        enum class blas_diag
        {
            non_unit,
            unit
        };
        
        enum class blas_side
        {
            left,
            right
        };
        
        // --------------------------------------------------------------------
        // BLAS Level 1: Vector operations
        // --------------------------------------------------------------------
        namespace blas
        {
            namespace detail
            {
                // Convert C++ types to BLAS characters
                inline char to_blas_char(blas_transpose trans)
                {
                    switch (trans)
                    {
                        case blas_transpose::no_trans: return 'N';
                        case blas_transpose::trans: return 'T';
                        case blas_transpose::conj_trans: return 'C';
                        default: return 'N';
                    }
                }
                
                inline char to_blas_char(blas_uplo uplo)
                {
                    switch (uplo)
                    {
                        case blas_uplo::upper: return 'U';
                        case blas_uplo::lower: return 'L';
                        default: return 'U';
                    }
                }
                
                inline char to_blas_char(blas_diag diag)
                {
                    switch (diag)
                    {
                        case blas_diag::non_unit: return 'N';
                        case blas_diag::unit: return 'U';
                        default: return 'N';
                    }
                }
                
                inline char to_blas_char(blas_side side)
                {
                    switch (side)
                    {
                        case blas_side::left: return 'L';
                        case blas_side::right: return 'R';
                        default: return 'L';
                    }
                }
                
                // Check if two vectors are compatible for BLAS operations
                template <class V1, class V2>
                inline bool compatible_vectors(const V1& v1, const V2& v2)
                {
                    return v1.size() == v2.size();
                }
                
                template <class M, class V>
                inline bool compatible_matrix_vector(const M& m, const V& v, blas_transpose trans)
                {
                    if (trans == blas_transpose::no_trans)
                        return m.shape()[1] == v.size();
                    else
                        return m.shape()[0] == v.size();
                }
                
                template <class M1, class M2>
                inline bool compatible_matrices_multiply(const M1& a, const M2& b,
                                                         blas_transpose transa, blas_transpose transb)
                {
                    std::size_t k_a = (transa == blas_transpose::no_trans) ? a.shape()[1] : a.shape()[0];
                    std::size_t k_b = (transb == blas_transpose::no_trans) ? b.shape()[0] : b.shape()[1];
                    return k_a == k_b;
                }
                
                template <class M1, class M2>
                inline bool compatible_matrices_solve(const M1& a, const M2& b)
                {
                    return a.shape()[0] == a.shape()[1] && a.shape()[0] == b.shape()[0];
                }
            }
            
            // --------------------------------------------------------------------
            // dot - dot product of two vectors
            // --------------------------------------------------------------------
            template <class V1, class V2>
            inline auto dot(const xexpression<V1>& v1, const xexpression<V2>& v2)
            {
                const auto& vec1 = v1.derived_cast();
                const auto& vec2 = v2.derived_cast();
                
                if (vec1.dimension() != 1 || vec2.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "dot: arguments must be 1-D vectors");
                }
                if (!detail::compatible_vectors(vec1, vec2))
                {
                    XTENSOR_THROW(std::invalid_argument, "dot: vectors must have same size");
                }
                
                using value_type = std::common_type_t<typename V1::value_type, typename V2::value_type>;
                value_type result = 0;
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<value_type, float>)
                {
                    cxxblas::dot(vec1.size(), vec1.data(), 1, vec2.data(), 1, result);
                }
                else if constexpr (std::is_same_v<value_type, double>)
                {
                    cxxblas::dot(vec1.size(), vec1.data(), 1, vec2.data(), 1, result);
                }
                else if constexpr (std::is_same_v<value_type, std::complex<float>>)
                {
                    cxxblas::dotc(vec1.size(),
                                  reinterpret_cast<const float(*)[2]>(vec1.data()), 1,
                                  reinterpret_cast<const float(*)[2]>(vec2.data()), 1,
                                  reinterpret_cast<float(*)[2]>(&result));
                }
                else if constexpr (std::is_same_v<value_type, std::complex<double>>)
                {
                    cxxblas::dotc(vec1.size(),
                                  reinterpret_cast<const double(*)[2]>(vec1.data()), 1,
                                  reinterpret_cast<const double(*)[2]>(vec2.data()), 1,
                                  reinterpret_cast<double(*)[2]>(&result));
                }
                else
#endif
                {
                    for (std::size_t i = 0; i < vec1.size(); ++i)
                    {
                        result += vec1(i) * vec2(i);
                    }
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // asum - sum of absolute values
            // --------------------------------------------------------------------
            template <class V>
            inline auto asum(const xexpression<V>& v)
            {
                const auto& vec = v.derived_cast();
                if (vec.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "asum: argument must be 1-D vector");
                }
                
                using value_type = typename V::value_type;
                using real_type = typename std::conditional_t<std::is_arithmetic_v<value_type>,
                                                              value_type,
                                                              typename value_type::value_type>;
                real_type result = 0;
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<real_type, float>)
                {
                    result = cxxblas::asum(vec.size(), vec.data(), 1);
                }
                else if constexpr (std::is_same_v<real_type, double>)
                {
                    result = cxxblas::asum(vec.size(), vec.data(), 1);
                }
                else
#endif
                {
                    for (std::size_t i = 0; i < vec.size(); ++i)
                    {
                        if constexpr (std::is_arithmetic_v<value_type>)
                        {
                            result += std::abs(vec(i));
                        }
                        else
                        {
                            result += std::abs(vec(i).real()) + std::abs(vec(i).imag());
                        }
                    }
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // nrm2 - Euclidean norm
            // --------------------------------------------------------------------
            template <class V>
            inline auto nrm2(const xexpression<V>& v)
            {
                const auto& vec = v.derived_cast();
                if (vec.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "nrm2: argument must be 1-D vector");
                }
                
                using value_type = typename V::value_type;
                using real_type = typename std::conditional_t<std::is_arithmetic_v<value_type>,
                                                              value_type,
                                                              typename value_type::value_type>;
                real_type result = 0;
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<real_type, float>)
                {
                    result = cxxblas::nrm2(vec.size(), vec.data(), 1);
                }
                else if constexpr (std::is_same_v<real_type, double>)
                {
                    result = cxxblas::nrm2(vec.size(), vec.data(), 1);
                }
                else
#endif
                {
                    real_type sum = 0;
                    for (std::size_t i = 0; i < vec.size(); ++i)
                    {
                        if constexpr (std::is_arithmetic_v<value_type>)
                        {
                            sum += vec(i) * vec(i);
                        }
                        else
                        {
                            sum += std::norm(vec(i));
                        }
                    }
                    result = std::sqrt(sum);
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // axpy - y = a*x + y
            // --------------------------------------------------------------------
            template <class T, class V1, class V2>
            inline void axpy(T alpha, const xexpression<V1>& x, xexpression<V2>& y)
            {
                auto& vec_x = x.derived_cast();
                auto& vec_y = y.derived_cast();
                
                if (vec_x.dimension() != 1 || vec_y.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "axpy: arguments must be 1-D vectors");
                }
                if (!detail::compatible_vectors(vec_x, vec_y))
                {
                    XTENSOR_THROW(std::invalid_argument, "axpy: vectors must have same size");
                }
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<typename V2::value_type, float>)
                {
                    cxxblas::axpy(vec_x.size(), alpha, vec_x.data(), 1, vec_y.data(), 1);
                    return;
                }
                else if constexpr (std::is_same_v<typename V2::value_type, double>)
                {
                    cxxblas::axpy(vec_x.size(), alpha, vec_x.data(), 1, vec_y.data(), 1);
                    return;
                }
#endif
                for (std::size_t i = 0; i < vec_x.size(); ++i)
                {
                    vec_y(i) += alpha * vec_x(i);
                }
            }
            
            // --------------------------------------------------------------------
            // scal - x = alpha * x
            // --------------------------------------------------------------------
            template <class T, class V>
            inline void scal(T alpha, xexpression<V>& x)
            {
                auto& vec = x.derived_cast();
                if (vec.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "scal: argument must be 1-D vector");
                }
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<typename V::value_type, float>)
                {
                    cxxblas::scal(vec.size(), alpha, vec.data(), 1);
                    return;
                }
                else if constexpr (std::is_same_v<typename V::value_type, double>)
                {
                    cxxblas::scal(vec.size(), alpha, vec.data(), 1);
                    return;
                }
#endif
                for (std::size_t i = 0; i < vec.size(); ++i)
                {
                    vec(i) *= alpha;
                }
            }
            
            // --------------------------------------------------------------------
            // copy - y = x
            // --------------------------------------------------------------------
            template <class V1, class V2>
            inline void copy(const xexpression<V1>& x, xexpression<V2>& y)
            {
                auto& vec_x = x.derived_cast();
                auto& vec_y = y.derived_cast();
                
                if (vec_x.dimension() != 1 || vec_y.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "copy: arguments must be 1-D vectors");
                }
                if (!detail::compatible_vectors(vec_x, vec_y))
                {
                    XTENSOR_THROW(std::invalid_argument, "copy: vectors must have same size");
                }
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<typename V2::value_type, float>)
                {
                    cxxblas::copy(vec_x.size(), vec_x.data(), 1, vec_y.data(), 1);
                    return;
                }
                else if constexpr (std::is_same_v<typename V2::value_type, double>)
                {
                    cxxblas::copy(vec_x.size(), vec_x.data(), 1, vec_y.data(), 1);
                    return;
                }
#endif
                std::copy(vec_x.begin(), vec_x.end(), vec_y.begin());
            }
            
            // --------------------------------------------------------------------
            // swap - interchange x and y
            // --------------------------------------------------------------------
            template <class V1, class V2>
            inline void swap(xexpression<V1>& x, xexpression<V2>& y)
            {
                auto& vec_x = x.derived_cast();
                auto& vec_y = y.derived_cast();
                
                if (vec_x.dimension() != 1 || vec_y.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "swap: arguments must be 1-D vectors");
                }
                if (!detail::compatible_vectors(vec_x, vec_y))
                {
                    XTENSOR_THROW(std::invalid_argument, "swap: vectors must have same size");
                }
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<typename V1::value_type, float>)
                {
                    cxxblas::swap(vec_x.size(), vec_x.data(), 1, vec_y.data(), 1);
                    return;
                }
                else if constexpr (std::is_same_v<typename V1::value_type, double>)
                {
                    cxxblas::swap(vec_x.size(), vec_x.data(), 1, vec_y.data(), 1);
                    return;
                }
#endif
                for (std::size_t i = 0; i < vec_x.size(); ++i)
                {
                    std::swap(vec_x(i), vec_y(i));
                }
            }
            
            // --------------------------------------------------------------------
            // iamax - index of max absolute value
            // --------------------------------------------------------------------
            template <class V>
            inline std::size_t iamax(const xexpression<V>& v)
            {
                const auto& vec = v.derived_cast();
                if (vec.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "iamax: argument must be 1-D vector");
                }
                
                std::size_t result = 0;
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<typename V::value_type, float>)
                {
                    result = static_cast<std::size_t>(cxxblas::iamax(vec.size(), vec.data(), 1)) - 1;
                    return result;
                }
                else if constexpr (std::is_same_v<typename V::value_type, double>)
                {
                    result = static_cast<std::size_t>(cxxblas::iamax(vec.size(), vec.data(), 1)) - 1;
                    return result;
                }
#endif
                using real_type = typename std::conditional_t<std::is_arithmetic_v<typename V::value_type>,
                                                              typename V::value_type,
                                                              typename V::value_type::value_type>;
                real_type max_val = 0;
                for (std::size_t i = 0; i < vec.size(); ++i)
                {
                    real_type abs_val = std::abs(vec(i));
                    if (abs_val > max_val)
                    {
                        max_val = abs_val;
                        result = i;
                    }
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // BLAS Level 2: Matrix-vector operations
            // --------------------------------------------------------------------
            
            // gemv - y = alpha * A * x + beta * y
            template <class T, class M, class V1, class V2>
            inline void gemv(blas_order order, blas_transpose trans,
                             T alpha, const xexpression<M>& a, const xexpression<V1>& x,
                             T beta, xexpression<V2>& y)
            {
                const auto& mat = a.derived_cast();
                const auto& vec_x = x.derived_cast();
                auto& vec_y = y.derived_cast();
                
                if (mat.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "gemv: matrix must be 2-D");
                }
                if (vec_x.dimension() != 1 || vec_y.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "gemv: vectors must be 1-D");
                }
                if (!detail::compatible_matrix_vector(mat, vec_x, trans))
                {
                    XTENSOR_THROW(std::invalid_argument, "gemv: incompatible dimensions");
                }
                
                std::size_t m = mat.shape()[0];
                std::size_t n = mat.shape()[1];
                std::size_t len_x = (trans == blas_transpose::no_trans) ? n : m;
                std::size_t len_y = (trans == blas_transpose::no_trans) ? m : n;
                
                if (vec_x.size() != len_x || vec_y.size() != len_y)
                {
                    XTENSOR_THROW(std::invalid_argument, "gemv: vector sizes mismatch");
                }
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<typename V2::value_type, float>)
                {
                    cxxblas::gemv(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(trans),
                                  static_cast<int>(m), static_cast<int>(n),
                                  alpha,
                                  mat.data(), static_cast<int>(order == blas_order::col_major ? m : n),
                                  vec_x.data(), 1,
                                  beta,
                                  vec_y.data(), 1);
                    return;
                }
                else if constexpr (std::is_same_v<typename V2::value_type, double>)
                {
                    cxxblas::gemv(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(trans),
                                  static_cast<int>(m), static_cast<int>(n),
                                  alpha,
                                  mat.data(), static_cast<int>(order == blas_order::col_major ? m : n),
                                  vec_x.data(), 1,
                                  beta,
                                  vec_y.data(), 1);
                    return;
                }
#endif
                // Fallback implementation
                if (beta != T(1))
                {
                    for (std::size_t i = 0; i < vec_y.size(); ++i)
                        vec_y(i) *= beta;
                }
                
                if (trans == blas_transpose::no_trans)
                {
                    for (std::size_t i = 0; i < m; ++i)
                    {
                        T sum = 0;
                        for (std::size_t j = 0; j < n; ++j)
                        {
                            sum += mat(i, j) * vec_x(j);
                        }
                        vec_y(i) += alpha * sum;
                    }
                }
                else if (trans == blas_transpose::trans)
                {
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        T sum = 0;
                        for (std::size_t j = 0; j < m; ++j)
                        {
                            sum += mat(j, i) * vec_x(j);
                        }
                        vec_y(i) += alpha * sum;
                    }
                }
                else // conj_trans
                {
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        T sum = 0;
                        for (std::size_t j = 0; j < m; ++j)
                        {
                            sum += std::conj(mat(j, i)) * vec_x(j);
                        }
                        vec_y(i) += alpha * sum;
                    }
                }
            }
            
            // symv - symmetric matrix-vector multiply
            template <class T, class M, class V1, class V2>
            inline void symv(blas_order order, blas_uplo uplo,
                             T alpha, const xexpression<M>& a, const xexpression<V1>& x,
                             T beta, xexpression<V2>& y)
            {
                const auto& mat = a.derived_cast();
                const auto& vec_x = x.derived_cast();
                auto& vec_y = y.derived_cast();
                
                if (mat.dimension() != 2 || mat.shape()[0] != mat.shape()[1])
                {
                    XTENSOR_THROW(std::invalid_argument, "symv: matrix must be square");
                }
                if (vec_x.dimension() != 1 || vec_y.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "symv: vectors must be 1-D");
                }
                std::size_t n = mat.shape()[0];
                if (vec_x.size() != n || vec_y.size() != n)
                {
                    XTENSOR_THROW(std::invalid_argument, "symv: vector size mismatch");
                }
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<typename V2::value_type, float>)
                {
                    cxxblas::symv(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(uplo),
                                  static_cast<int>(n),
                                  alpha,
                                  mat.data(), static_cast<int>(order == blas_order::col_major ? n : n),
                                  vec_x.data(), 1,
                                  beta,
                                  vec_y.data(), 1);
                    return;
                }
                else if constexpr (std::is_same_v<typename V2::value_type, double>)
                {
                    cxxblas::symv(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(uplo),
                                  static_cast<int>(n),
                                  alpha,
                                  mat.data(), static_cast<int>(order == blas_order::col_major ? n : n),
                                  vec_x.data(), 1,
                                  beta,
                                  vec_y.data(), 1);
                    return;
                }
#endif
                // Fallback
                if (beta != T(1))
                {
                    for (std::size_t i = 0; i < n; ++i)
                        vec_y(i) *= beta;
                }
                
                if (uplo == blas_uplo::upper)
                {
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        T sum = 0;
                        for (std::size_t j = i; j < n; ++j)
                        {
                            sum += mat(i, j) * vec_x(j);
                        }
                        for (std::size_t j = 0; j < i; ++j)
                        {
                            sum += mat(j, i) * vec_x(j);
                        }
                        vec_y(i) += alpha * sum;
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        T sum = 0;
                        for (std::size_t j = 0; j <= i; ++j)
                        {
                            sum += mat(i, j) * vec_x(j);
                        }
                        for (std::size_t j = i + 1; j < n; ++j)
                        {
                            sum += mat(j, i) * vec_x(j);
                        }
                        vec_y(i) += alpha * sum;
                    }
                }
            }
            
            // trmv - triangular matrix-vector multiply
            template <class M, class V>
            inline void trmv(blas_order order, blas_uplo uplo, blas_transpose trans, blas_diag diag,
                             const xexpression<M>& a, xexpression<V>& x)
            {
                const auto& mat = a.derived_cast();
                auto& vec = x.derived_cast();
                
                if (mat.dimension() != 2 || mat.shape()[0] != mat.shape()[1])
                {
                    XTENSOR_THROW(std::invalid_argument, "trmv: matrix must be square");
                }
                if (vec.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "trmv: vector must be 1-D");
                }
                std::size_t n = mat.shape()[0];
                if (vec.size() != n)
                {
                    XTENSOR_THROW(std::invalid_argument, "trmv: vector size mismatch");
                }
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<typename V::value_type, float>)
                {
                    cxxblas::trmv(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(uplo),
                                  detail::to_blas_char(trans),
                                  detail::to_blas_char(diag),
                                  static_cast<int>(n),
                                  mat.data(), static_cast<int>(order == blas_order::col_major ? n : n),
                                  vec.data(), 1);
                    return;
                }
                else if constexpr (std::is_same_v<typename V::value_type, double>)
                {
                    cxxblas::trmv(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(uplo),
                                  detail::to_blas_char(trans),
                                  detail::to_blas_char(diag),
                                  static_cast<int>(n),
                                  mat.data(), static_cast<int>(order == blas_order::col_major ? n : n),
                                  vec.data(), 1);
                    return;
                }
#endif
                // Fallback implementation - compute x = A * x (triangular)
                std::vector<typename V::value_type> x_copy(vec.begin(), vec.end());
                
                for (std::size_t i = 0; i < n; ++i)
                {
                    typename V::value_type sum = 0;
                    if (uplo == blas_uplo::upper)
                    {
                        for (std::size_t j = i; j < n; ++j)
                        {
                            sum += mat(i, j) * x_copy[j];
                        }
                    }
                    else
                    {
                        for (std::size_t j = 0; j <= i; ++j)
                        {
                            sum += mat(i, j) * x_copy[j];
                        }
                    }
                    vec(i) = sum;
                    if (diag == blas_diag::unit)
                    {
                        vec(i) += x_copy[i];
                    }
                }
            }
            
            // --------------------------------------------------------------------
            // BLAS Level 3: Matrix-matrix operations
            // --------------------------------------------------------------------
            
            // gemm - C = alpha * A * B + beta * C
            template <class T, class M1, class M2, class M3>
            inline void gemm(blas_order order,
                             blas_transpose transa, blas_transpose transb,
                             T alpha, const xexpression<M1>& a, const xexpression<M2>& b,
                             T beta, xexpression<M3>& c)
            {
                const auto& mat_a = a.derived_cast();
                const auto& mat_b = b.derived_cast();
                auto& mat_c = c.derived_cast();
                
                if (mat_a.dimension() != 2 || mat_b.dimension() != 2 || mat_c.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "gemm: matrices must be 2-D");
                }
                if (!detail::compatible_matrices_multiply(mat_a, mat_b, transa, transb))
                {
                    XTENSOR_THROW(std::invalid_argument, "gemm: incompatible dimensions for multiplication");
                }
                
                std::size_t m = (transa == blas_transpose::no_trans) ? mat_a.shape()[0] : mat_a.shape()[1];
                std::size_t n = (transb == blas_transpose::no_trans) ? mat_b.shape()[1] : mat_b.shape()[0];
                std::size_t k = (transa == blas_transpose::no_trans) ? mat_a.shape()[1] : mat_a.shape()[0];
                
                if (mat_c.shape()[0] != m || mat_c.shape()[1] != n)
                {
                    XTENSOR_THROW(std::invalid_argument, "gemm: C matrix has wrong shape");
                }
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<typename M3::value_type, float>)
                {
                    cxxblas::gemm(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(transa),
                                  detail::to_blas_char(transb),
                                  static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                  alpha,
                                  mat_a.data(), static_cast<int>(transa == blas_transpose::no_trans ? k : m),
                                  mat_b.data(), static_cast<int>(transb == blas_transpose::no_trans ? n : k),
                                  beta,
                                  mat_c.data(), static_cast<int>(n));
                    return;
                }
                else if constexpr (std::is_same_v<typename M3::value_type, double>)
                {
                    cxxblas::gemm(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(transa),
                                  detail::to_blas_char(transb),
                                  static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                  alpha,
                                  mat_a.data(), static_cast<int>(transa == blas_transpose::no_trans ? k : m),
                                  mat_b.data(), static_cast<int>(transb == blas_transpose::no_trans ? n : k),
                                  beta,
                                  mat_c.data(), static_cast<int>(n));
                    return;
                }
#endif
                // Fallback
                if (beta != T(1))
                {
                    for (std::size_t i = 0; i < mat_c.size(); ++i)
                        mat_c.flat(i) *= beta;
                }
                
                for (std::size_t i = 0; i < m; ++i)
                {
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        T sum = 0;
                        for (std::size_t l = 0; l < k; ++l)
                        {
                            T a_val, b_val;
                            if (transa == blas_transpose::no_trans)
                                a_val = mat_a(i, l);
                            else if (transa == blas_transpose::trans)
                                a_val = mat_a(l, i);
                            else
                                a_val = std::conj(mat_a(l, i));
                                
                            if (transb == blas_transpose::no_trans)
                                b_val = mat_b(l, j);
                            else if (transb == blas_transpose::trans)
                                b_val = mat_b(j, l);
                            else
                                b_val = std::conj(mat_b(j, l));
                                
                            sum += a_val * b_val;
                        }
                        mat_c(i, j) += alpha * sum;
                    }
                }
            }
            
            // symm - symmetric matrix-matrix multiply
            template <class T, class M1, class M2, class M3>
            inline void symm(blas_order order, blas_side side, blas_uplo uplo,
                             T alpha, const xexpression<M1>& a, const xexpression<M2>& b,
                             T beta, xexpression<M3>& c)
            {
                const auto& mat_a = a.derived_cast();
                const auto& mat_b = b.derived_cast();
                auto& mat_c = c.derived_cast();
                
                if (mat_a.dimension() != 2 || mat_a.shape()[0] != mat_a.shape()[1])
                {
                    XTENSOR_THROW(std::invalid_argument, "symm: A must be square");
                }
                if (mat_b.dimension() != 2 || mat_c.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "symm: matrices must be 2-D");
                }
                
                std::size_t n_a = mat_a.shape()[0];
                std::size_t m = mat_b.shape()[0];
                std::size_t n = mat_b.shape()[1];
                
                if ((side == blas_side::left && m != n_a) ||
                    (side == blas_side::right && n != n_a))
                {
                    XTENSOR_THROW(std::invalid_argument, "symm: incompatible dimensions");
                }
                if (mat_c.shape()[0] != m || mat_c.shape()[1] != n)
                {
                    XTENSOR_THROW(std::invalid_argument, "symm: C matrix has wrong shape");
                }
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<typename M3::value_type, float>)
                {
                    cxxblas::symm(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(side),
                                  detail::to_blas_char(uplo),
                                  static_cast<int>(m), static_cast<int>(n),
                                  alpha,
                                  mat_a.data(), static_cast<int>(n_a),
                                  mat_b.data(), static_cast<int>(order == blas_order::col_major ? m : n),
                                  beta,
                                  mat_c.data(), static_cast<int>(order == blas_order::col_major ? m : n));
                    return;
                }
                else if constexpr (std::is_same_v<typename M3::value_type, double>)
                {
                    cxxblas::symm(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(side),
                                  detail::to_blas_char(uplo),
                                  static_cast<int>(m), static_cast<int>(n),
                                  alpha,
                                  mat_a.data(), static_cast<int>(n_a),
                                  mat_b.data(), static_cast<int>(order == blas_order::col_major ? m : n),
                                  beta,
                                  mat_c.data(), static_cast<int>(order == blas_order::col_major ? m : n));
                    return;
                }
#endif
                // Fallback - for brevity, we'll do a simple implementation
                if (beta != T(1))
                {
                    for (std::size_t i = 0; i < mat_c.size(); ++i)
                        mat_c.flat(i) *= beta;
                }
                
                if (side == blas_side::left)
                {
                    for (std::size_t i = 0; i < m; ++i)
                    {
                        for (std::size_t j = 0; j < n; ++j)
                        {
                            T sum = 0;
                            for (std::size_t k = 0; k < n_a; ++k)
                            {
                                T a_val = (uplo == blas_uplo::upper) ?
                                    (i <= k ? mat_a(i, k) : mat_a(k, i)) :
                                    (i >= k ? mat_a(i, k) : mat_a(k, i));
                                sum += a_val * mat_b(k, j);
                            }
                            mat_c(i, j) += alpha * sum;
                        }
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < m; ++i)
                    {
                        for (std::size_t j = 0; j < n; ++j)
                        {
                            T sum = 0;
                            for (std::size_t k = 0; k < n_a; ++k)
                            {
                                T a_val = (uplo == blas_uplo::upper) ?
                                    (j <= k ? mat_a(j, k) : mat_a(k, j)) :
                                    (j >= k ? mat_a(j, k) : mat_a(k, j));
                                sum += mat_b(i, k) * a_val;
                            }
                            mat_c(i, j) += alpha * sum;
                        }
                    }
                }
            }
            
            // trmm - triangular matrix-matrix multiply
            template <class T, class M1, class M2>
            inline void trmm(blas_order order, blas_side side, blas_uplo uplo,
                             blas_transpose trans, blas_diag diag,
                             T alpha, const xexpression<M1>& a, xexpression<M2>& b)
            {
                const auto& mat_a = a.derived_cast();
                auto& mat_b = b.derived_cast();
                
                if (mat_a.dimension() != 2 || mat_a.shape()[0] != mat_a.shape()[1])
                {
                    XTENSOR_THROW(std::invalid_argument, "trmm: A must be square");
                }
                if (mat_b.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "trmm: B must be 2-D");
                }
                
                std::size_t n_a = mat_a.shape()[0];
                std::size_t m = mat_b.shape()[0];
                std::size_t n = mat_b.shape()[1];
                
                if ((side == blas_side::left && m != n_a) ||
                    (side == blas_side::right && n != n_a))
                {
                    XTENSOR_THROW(std::invalid_argument, "trmm: incompatible dimensions");
                }
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<typename M2::value_type, float>)
                {
                    cxxblas::trmm(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(side),
                                  detail::to_blas_char(uplo),
                                  detail::to_blas_char(trans),
                                  detail::to_blas_char(diag),
                                  static_cast<int>(m), static_cast<int>(n),
                                  alpha,
                                  mat_a.data(), static_cast<int>(n_a),
                                  mat_b.data(), static_cast<int>(order == blas_order::col_major ? m : n));
                    return;
                }
                else if constexpr (std::is_same_v<typename M2::value_type, double>)
                {
                    cxxblas::trmm(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(side),
                                  detail::to_blas_char(uplo),
                                  detail::to_blas_char(trans),
                                  detail::to_blas_char(diag),
                                  static_cast<int>(m), static_cast<int>(n),
                                  alpha,
                                  mat_a.data(), static_cast<int>(n_a),
                                  mat_b.data(), static_cast<int>(order == blas_order::col_major ? m : n));
                    return;
                }
#endif
                // Fallback implementation
                auto b_copy = eval(mat_b);
                
                if (side == blas_side::left)
                {
                    for (std::size_t i = 0; i < m; ++i)
                    {
                        for (std::size_t j = 0; j < n; ++j)
                        {
                            T sum = 0;
                            if (uplo == blas_uplo::upper)
                            {
                                for (std::size_t k = i; k < n_a; ++k)
                                {
                                    T a_val = mat_a(i, k);
                                    if (diag == blas_diag::unit && i == k)
                                        a_val = 1;
                                    sum += a_val * b_copy(k, j);
                                }
                            }
                            else
                            {
                                for (std::size_t k = 0; k <= i; ++k)
                                {
                                    T a_val = mat_a(i, k);
                                    if (diag == blas_diag::unit && i == k)
                                        a_val = 1;
                                    sum += a_val * b_copy(k, j);
                                }
                            }
                            mat_b(i, j) = alpha * sum;
                        }
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < m; ++i)
                    {
                        for (std::size_t j = 0; j < n; ++j)
                        {
                            T sum = 0;
                            if (uplo == blas_uplo::upper)
                            {
                                for (std::size_t k = 0; k <= j; ++k)
                                {
                                    T a_val = mat_a(k, j);
                                    if (diag == blas_diag::unit && k == j)
                                        a_val = 1;
                                    sum += b_copy(i, k) * a_val;
                                }
                            }
                            else
                            {
                                for (std::size_t k = j; k < n_a; ++k)
                                {
                                    T a_val = mat_a(k, j);
                                    if (diag == blas_diag::unit && k == j)
                                        a_val = 1;
                                    sum += b_copy(i, k) * a_val;
                                }
                            }
                            mat_b(i, j) = alpha * sum;
                        }
                    }
                }
            }
            
            // trsm - solve triangular matrix equation
            template <class T, class M1, class M2>
            inline void trsm(blas_order order, blas_side side, blas_uplo uplo,
                             blas_transpose trans, blas_diag diag,
                             T alpha, const xexpression<M1>& a, xexpression<M2>& b)
            {
                const auto& mat_a = a.derived_cast();
                auto& mat_b = b.derived_cast();
                
                if (mat_a.dimension() != 2 || mat_a.shape()[0] != mat_a.shape()[1])
                {
                    XTENSOR_THROW(std::invalid_argument, "trsm: A must be square");
                }
                if (mat_b.dimension() != 2)
                {
                    XTENSOR_THROW(std::invalid_argument, "trsm: B must be 2-D");
                }
                
                std::size_t n_a = mat_a.shape()[0];
                std::size_t m = mat_b.shape()[0];
                std::size_t n = mat_b.shape()[1];
                
                if ((side == blas_side::left && m != n_a) ||
                    (side == blas_side::right && n != n_a))
                {
                    XTENSOR_THROW(std::invalid_argument, "trsm: incompatible dimensions");
                }
                
#if XTENSOR_USE_BLAS
                if constexpr (std::is_same_v<typename M2::value_type, float>)
                {
                    cxxblas::trsm(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(side),
                                  detail::to_blas_char(uplo),
                                  detail::to_blas_char(trans),
                                  detail::to_blas_char(diag),
                                  static_cast<int>(m), static_cast<int>(n),
                                  alpha,
                                  mat_a.data(), static_cast<int>(n_a),
                                  mat_b.data(), static_cast<int>(order == blas_order::col_major ? m : n));
                    return;
                }
                else if constexpr (std::is_same_v<typename M2::value_type, double>)
                {
                    cxxblas::trsm(order == blas_order::col_major ? CblasColMajor : CblasRowMajor,
                                  detail::to_blas_char(side),
                                  detail::to_blas_char(uplo),
                                  detail::to_blas_char(trans),
                                  detail::to_blas_char(diag),
                                  static_cast<int>(m), static_cast<int>(n),
                                  alpha,
                                  mat_a.data(), static_cast<int>(n_a),
                                  mat_b.data(), static_cast<int>(order == blas_order::col_major ? m : n));
                    return;
                }
#endif
                // Fallback - forward/backward substitution
                // First scale B by alpha if alpha != 1
                if (alpha != T(1))
                {
                    for (auto& v : mat_b)
                        v *= alpha;
                }
                
                auto b_copy = eval(mat_b);
                
                if (side == blas_side::left)
                {
                    // Solve A * X = B  (X overwrites B)
                    if (uplo == blas_uplo::lower)
                    {
                        // Forward substitution
                        for (std::size_t j = 0; j < n; ++j)
                        {
                            for (std::size_t i = 0; i < m; ++i)
                            {
                                T sum = b_copy(i, j);
                                for (std::size_t k = 0; k < i; ++k)
                                {
                                    sum -= mat_a(i, k) * mat_b(k, j);
                                }
                                if (diag == blas_diag::non_unit)
                                    mat_b(i, j) = sum / mat_a(i, i);
                                else
                                    mat_b(i, j) = sum;
                            }
                        }
                    }
                    else
                    {
                        // Backward substitution
                        for (std::size_t j = 0; j < n; ++j)
                        {
                            for (std::size_t i = m; i-- > 0; )
                            {
                                T sum = b_copy(i, j);
                                for (std::size_t k = i + 1; k < m; ++k)
                                {
                                    sum -= mat_a(i, k) * mat_b(k, j);
                                }
                                if (diag == blas_diag::non_unit)
                                    mat_b(i, j) = sum / mat_a(i, i);
                                else
                                    mat_b(i, j) = sum;
                            }
                        }
                    }
                }
                else
                {
                    // Solve X * A = B
                    if (uplo == blas_uplo::lower)
                    {
                        // Backward substitution for columns
                        for (std::size_t i = 0; i < m; ++i)
                        {
                            for (std::size_t j = n; j-- > 0; )
                            {
                                T sum = b_copy(i, j);
                                for (std::size_t k = j + 1; k < n; ++k)
                                {
                                    sum -= mat_b(i, k) * mat_a(k, j);
                                }
                                if (diag == blas_diag::non_unit)
                                    mat_b(i, j) = sum / mat_a(j, j);
                                else
                                    mat_b(i, j) = sum;
                            }
                        }
                    }
                    else
                    {
                        // Forward substitution for columns
                        for (std::size_t i = 0; i < m; ++i)
                        {
                            for (std::size_t j = 0; j < n; ++j)
                            {
                                T sum = b_copy(i, j);
                                for (std::size_t k = 0; k < j; ++k)
                                {
                                    sum -= mat_b(i, k) * mat_a(k, j);
                                }
                                if (diag == blas_diag::non_unit)
                                    mat_b(i, j) = sum / mat_a(j, j);
                                else
                                    mat_b(i, j) = sum;
                            }
                        }
                    }
                }
            }
            
        } // namespace blas
        
        // --------------------------------------------------------------------
        // Convenience wrappers
        // --------------------------------------------------------------------
        template <class V1, class V2>
        inline auto dot(const xexpression<V1>& a, const xexpression<V2>& b)
        {
            return blas::dot(a, b);
        }
        
        template <class T, class M1, class M2, class M3>
        inline void matmul(const xexpression<M1>& a, const xexpression<M2>& b,
                           xexpression<M3>& c, T alpha = 1, T beta = 0)
        {
            blas::gemm(blas_order::row_major,
                       blas_transpose::no_trans, blas_transpose::no_trans,
                       alpha, a, b, beta, c);
        }
        
        template <class M1, class M2>
        inline auto matmul(const xexpression<M1>& a, const xexpression<M2>& b)
        {
            const auto& mat_a = a.derived_cast();
            const auto& mat_b = b.derived_cast();
            
            using value_type = std::common_type_t<typename M1::value_type, typename M2::value_type>;
            svector<std::size_t> shape = {mat_a.shape()[0], mat_b.shape()[1]};
            xarray_container<value_type> result(shape, value_type(0));
            
            blas::gemm(blas_order::row_major,
                       blas_transpose::no_trans, blas_transpose::no_trans,
                       value_type(1), a, b, value_type(0), result);
            return result;
        }
        
        template <class M, class V>
        inline auto matvec(const xexpression<M>& a, const xexpression<V>& x)
        {
            const auto& mat_a = a.derived_cast();
            const auto& vec_x = x.derived_cast();
            
            using value_type = std::common_type_t<typename M::value_type, typename V::value_type>;
            svector<std::size_t> shape = {mat_a.shape()[0]};
            xarray_container<value_type> result(shape, value_type(0));
            
            blas::gemv(blas_order::row_major, blas_transpose::no_trans,
                       value_type(1), a, x, value_type(0), result);
            return result;
        }
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XBLAS_HPP

// math/xblas.hpp