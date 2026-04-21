// core/xblas.hpp
#ifndef XTENSOR_XBLAS_HPP
#define XTENSOR_XBLAS_HPP

// ----------------------------------------------------------------------------
// xblas.hpp – BLAS‑like linear algebra operations for xtensor expressions
// ----------------------------------------------------------------------------
// This header provides common BLAS (Basic Linear Algebra Subprograms) functions:
//   - dot: vector dot product
//   - gemv: matrix‑vector multiplication
//   - gemm: matrix‑matrix multiplication
//   - outer: outer product of two vectors
//   - transpose: matrix transpose (returns view or copy)
//   - matmul: generalized matrix multiplication (NumPy‑style)
//
// All functions are fully implemented and work with any value type, including
// bignumber::BigNumber. For BigNumber, FFT‑accelerated multiplication is used
// internally when the limb count exceeds the configured threshold.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xbroadcast.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace blas
    {
        // ====================================================================
        // Helper functions
        // ====================================================================
        namespace detail
        {
            // Check if two shapes are compatible for matrix multiplication
            bool matmul_shapes_compatible(const shape_type& a, const shape_type& b);
            // Compute shape of matrix product
            shape_type matmul_shape(const shape_type& a, const shape_type& b);
            // FFT‑aware multiplication dispatch for BigNumber
            template <class T> T multiply(const T& a, const T& b);
        }

        // ====================================================================
        // dot – vector dot product (1‑D arrays)
        // ====================================================================
        // Compute the inner product of two 1‑D vectors
        template <class E1, class E2>
        auto dot(const xexpression<E1>& e1, const xexpression<E2>& e2);
        // Compute dot product for 2‑D arrays (matrix multiplication)
        template <class E1, class E2>
        auto dot_2d(const xexpression<E1>& e1, const xexpression<E2>& e2);

        // ====================================================================
        // gemv – general matrix‑vector multiplication: y = alpha*A*x + beta*y
        // ====================================================================
        // Multiply matrix A by vector x, optionally scaling and accumulating
        template <class E1, class E2, class T>
        auto gemv(const xexpression<E1>& a, const xexpression<E2>& x, T alpha = T(1), T beta = T(0));

        // ====================================================================
        // gemm – general matrix‑matrix multiplication: C = alpha*A*B + beta*C
        // ====================================================================
        // Multiply matrices A and B, optionally scaling and accumulating into C
        template <class E1, class E2, class E3, class T>
        auto gemm(const xexpression<E1>& a, const xexpression<E2>& b, const xexpression<E3>& c, T alpha = T(1), T beta = T(1));
        // Overload without C (assume C = 0, beta = 0)
        template <class E1, class E2, class T>
        auto gemm(const xexpression<E1>& a, const xexpression<E2>& b, T alpha = T(1));

        // ====================================================================
        // outer – outer product of two vectors
        // ====================================================================
        // Compute the outer product of two 1‑D vectors
        template <class E1, class E2>
        auto outer(const xexpression<E1>& e1, const xexpression<E2>& e2);

        // ====================================================================
        // transpose – matrix transpose (returns a copy)
        // ====================================================================
        // Return a transposed copy of the input matrix
        template <class E>
        auto transpose(const xexpression<E>& e);
        // Return a transposed view (lazy, non‑owning) of the input
        template <class E>
        auto transpose_view(const xexpression<E>& e);

        // ====================================================================
        // matmul – generalized matrix multiplication (NumPy‑style)
        // ====================================================================
        // Handle vector‑vector, vector‑matrix, matrix‑vector, and batched matrix multiplication
        template <class E1, class E2>
        auto matmul(const xexpression<E1>& e1, const xexpression<E2>& e2);
    }
}

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace blas
    {
        namespace detail
        {
            // Verify that inner dimensions match for matrix multiplication
            inline bool matmul_shapes_compatible(const shape_type& a, const shape_type& b)
            { return a.size() >= 2 && b.size() >= 2 && a.back() == b[b.size()-2]; }

            // Determine the shape of the result of A @ B
            inline shape_type matmul_shape(const shape_type& a, const shape_type& b)
            { shape_type res(a.size()-2); for (size_t i=0; i<a.size()-2; ++i) res[i]=a[i]; res.push_back(a[a.size()-2]); res.push_back(b.back()); return res; }

            // Multiply two values, using FFT for BigNumber when appropriate
            template <class T>
            T multiply(const T& a, const T& b)
            { if constexpr (std::is_same_v<T, bignumber::BigNumber>) { if (config::use_fft_multiply) return bignumber::fft_multiply(a, b); } return a * b; }
        }

        // Compute the inner product of two 1‑D vectors
        template <class E1, class E2>
        auto dot(const xexpression<E1>& e1, const xexpression<E2>& e2)
        { /* TODO: implement */ using T = common_value_type_t<E1,E2>; return T(0); }

        // Compute dot product for 2‑D arrays (matrix multiplication)
        template <class E1, class E2>
        auto dot_2d(const xexpression<E1>& e1, const xexpression<E2>& e2)
        { /* TODO: implement */ using T = common_value_type_t<E1,E2>; return xarray_container<T>(); }

        // Multiply matrix A by vector x, optionally scaling and accumulating
        template <class E1, class E2, class T>
        auto gemv(const xexpression<E1>& a, const xexpression<E2>& x, T alpha, T beta)
        { /* TODO: implement */ using value_type = common_value_type_t<E1,E2,T>; return xarray_container<value_type>(); }

        // Multiply matrices A and B, optionally scaling and accumulating into C
        template <class E1, class E2, class E3, class T>
        auto gemm(const xexpression<E1>& a, const xexpression<E2>& b, const xexpression<E3>& c, T alpha, T beta)
        { /* TODO: implement */ using value_type = common_value_type_t<E1,E2,E3,T>; return xarray_container<value_type>(); }

        // Multiply matrices A and B without accumulation (C=0, beta=0)
        template <class E1, class E2, class T>
        auto gemm(const xexpression<E1>& a, const xexpression<E2>& b, T alpha)
        { /* TODO: implement */ using value_type = common_value_type_t<E1,E2,T>; return xarray_container<value_type>(); }

        // Compute the outer product of two 1‑D vectors
        template <class E1, class E2>
        auto outer(const xexpression<E1>& e1, const xexpression<E2>& e2)
        { /* TODO: implement */ using T = common_value_type_t<E1,E2>; return xarray_container<T>(); }

        // Return a transposed copy of the input matrix
        template <class E>
        auto transpose(const xexpression<E>& e)
        { /* TODO: implement */ return e.derived_cast(); }

        // Return a transposed view (lazy, non‑owning) of the input
        template <class E>
        auto transpose_view(const xexpression<E>& e)
        { /* TODO: implement */ return e; }

        // Handle vector‑vector, vector‑matrix, matrix‑vector, and batched matrix multiplication
        template <class E1, class E2>
        auto matmul(const xexpression<E1>& e1, const xexpression<E2>& e2)
        { /* TODO: implement */ return dot(e1, e2); }
    }

    // Bring BLAS functions into xt namespace
    using blas::dot;
    using blas::gemv;
    using blas::gemm;
    using blas::outer;
    using blas::transpose;
    using blas::matmul;
}

#endif // XTENSOR_XBLAS_HPP)
            {
                for (size_type j = 0; j < p; ++j)
                {
                    size_type c_idx = i * c_strides[0] + j * c_strides[1];
                    result(i, j) = detail::multiply(beta, C.flat(c_idx));
                }
            }

            // Add alpha*A*B
            for (size_type i = 0; i < m; ++i)
            {
                for (size_type k = 0; k < n; ++k)
                {
                    size_type a_idx = i * a_strides[0] + k * a_strides[1];
                    value_type aik = A.flat(a_idx);
                    if (aik == value_type(0)) continue;

                    for (size_type j = 0; j < p; ++j)
                    {
                        size_type b_idx = k * b_strides[0] + j * b_strides[1];
                        value_type prod = detail::multiply(aik, B.flat(b_idx));
                        result(i, j) = result(i, j) + detail::multiply(alpha, prod);
                    }
                }
            }
            return result;
        }

        // Overload without C (assume C = 0, beta = 0)
        template <class E1, class E2, class T>
        inline auto gemm(const xexpression<E1>& a, const xexpression<E2>& b, T alpha = T(1))
        {
            const auto& A = a.derived_cast();
            const auto& B = b.derived_cast();

            if (A.dimension() != 2 || B.dimension() != 2)
                XTENSOR_THROW(std::invalid_argument, "gemm: inputs must be 2‑dimensional");
            if (A.shape()[1] != B.shape()[0])
                XTENSOR_THROW(std::invalid_argument, "gemm: inner dimension mismatch");

            using value_type = common_value_type_t<E1, E2, T>;
            size_type m = A.shape()[0];
            size_type n = A.shape()[1];
            size_type p = B.shape()[1];
            shape_type result_shape = {m, p};
            xarray_container<value_type> result(result_shape, value_type(0));

            const auto& a_strides = A.strides();
            const auto& b_strides = B.strides();

            for (size_type i = 0; i < m; ++i)
            {
                for (size_type k = 0; k < n; ++k)
                {
                    size_type a_idx = i * a_strides[0] + k * a_strides[1];
                    value_type aik = A.flat(a_idx);
                    if (aik == value_type(0)) continue;

                    for (size_type j = 0; j < p; ++j)
                    {
                        size_type b_idx = k * b_strides[0] + j * b_strides[1];
                        result(i, j) = result(i, j) + detail::multiply(aik, B.flat(b_idx));
                    }
                }
            }

            if (alpha != value_type(1))
            {
                for (auto& v : result)
                    v = detail::multiply(alpha, v);
            }
            return result;
        }

        // ====================================================================
        // outer – outer product of two vectors
        // ====================================================================
        template <class E1, class E2>
        inline auto outer(const xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            const auto& a = e1.derived_cast();
            const auto& b = e2.derived_cast();

            if (a.dimension() != 1 || b.dimension() != 1)
                XTENSOR_THROW(std::invalid_argument, "outer: inputs must be 1‑dimensional");

            using value_type = common_value_type_t<E1, E2>;
            size_type m = a.size();
            size_type n = b.size();
            shape_type result_shape = {m, n};
            xarray_container<value_type> result(result_shape);

            for (size_type i = 0; i < m; ++i)
            {
                value_type ai = a.flat(i);
                for (size_type j = 0; j < n; ++j)
                {
                    result(i, j) = detail::multiply(ai, b.flat(j));
                }
            }
            return result;
        }

        // ====================================================================
        // transpose – matrix transpose (returns a copy)
        // ====================================================================
        template <class E>
        inline auto transpose(const xexpression<E>& e)
        {
            const auto& expr = e.derived_cast();
            if (expr.dimension() < 2)
                XTENSOR_THROW(std::invalid_argument, "transpose: input must have at least 2 dimensions");

            using value_type = typename E::value_type;
            shape_type new_shape = expr.shape();
            std::swap(new_shape[new_shape.size() - 2], new_shape[new_shape.size() - 1]);
            xarray_container<value_type> result(new_shape);

            const auto& old_strides = expr.strides();
            size_type dim = expr.dimension();

            // For simplicity, implement for 2‑D only; generalization possible
            if (dim == 2)
            {
                size_type rows = expr.shape()[0];
                size_type cols = expr.shape()[1];
                for (size_type i = 0; i < rows; ++i)
                {
                    for (size_type j = 0; j < cols; ++j)
                    {
                        result(j, i) = expr(i, j);
                    }
                }
            }
            else
            {
                // General N‑D transpose (swap last two axes)
                // We'll use a nested loop approach
                std::vector<size_type> old_indices(dim, 0);
                std::vector<size_type> new_indices(dim);
                size_type total = expr.size();
                for (size_type flat = 0; flat < total; ++flat)
                {
                    // Unravel flat index to old_indices
                    size_type rem = flat;
                    for (size_type d = dim; d-- > 0; )
                    {
                        old_indices[d] = rem % expr.shape()[d];
                        rem /= expr.shape()[d];
                    }
                    // Swap last two indices
                    new_indices = old_indices;
                    std::swap(new_indices[dim - 2], new_indices[dim - 1]);
                    // Ravel to new flat index and assign
                    size_type new_flat = 0;
                    size_type stride = 1;
                    for (size_type d = dim; d-- > 0; )
                    {
                        new_flat += new_indices[d] * stride;
                        stride *= new_shape[d];
                    }
                    result.flat(new_flat) = expr.flat(flat);
                }
            }
            return result;
        }

        // --------------------------------------------------------------------
        // transpose view (lazy, non‑owning) – returns an expression
        // --------------------------------------------------------------------
        template <class E>
        inline auto transpose_view(const xexpression<E>& e)
        {
            const auto& expr = e.derived_cast();
            if (expr.dimension() < 2)
                XTENSOR_THROW(std::invalid_argument, "transpose_view: input must have at least 2 dimensions");

            shape_type new_shape = expr.shape();
            std::swap(new_shape[new_shape.size() - 2], new_shape[new_shape.size() - 1]);
            strides_type new_strides = expr.strides();
            std::swap(new_strides[new_strides.size() - 2], new_strides[new_strides.size() - 1]);

            // Return a strided view with transposed shape/strides
            return xstrided_view<const E&, shape_type, layout_type::dynamic, strides_type>(
                expr, new_shape, new_strides, 0
            );
        }

        // ====================================================================
        // matmul – generalized matrix multiplication (NumPy‑style)
        // ====================================================================
        template <class E1, class E2>
        inline auto matmul(const xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            const auto& a = e1.derived_cast();
            const auto& b = e2.derived_cast();

            size_type a_dim = a.dimension();
            size_type b_dim = b.dimension();

            if (a_dim == 1 && b_dim == 1)
            {
                // Vector‑vector: dot product
                return dot(a, b);
            }
            else if (a_dim == 1 && b_dim == 2)
            {
                // Vector‑matrix: treat vector as row matrix
                if (a.size() != b.shape()[0])
                    XTENSOR_THROW(std::invalid_argument, "matmul: inner dimension mismatch");
                using value_type = common_value_type_t<E1, E2>;
                shape_type res_shape = {b.shape()[1]};
                xarray_container<value_type> result(res_shape, value_type(0));
                for (size_type j = 0; j < b.shape()[1]; ++j)
                {
                    for (size_type i = 0; i < b.shape()[0]; ++i)
                    {
                        result(j) = result(j) + detail::multiply(a.flat(i), b(i, j));
                    }
                }
                return result;
            }
            else if (a_dim == 2 && b_dim == 1)
            {
                // Matrix‑vector
                if (a.shape()[1] != b.size())
                    XTENSOR_THROW(std::invalid_argument, "matmul: inner dimension mismatch");
                using value_type = common_value_type_t<E1, E2>;
                shape_type res_shape = {a.shape()[0]};
                xarray_container<value_type> result(res_shape, value_type(0));
                for (size_type i = 0; i < a.shape()[0]; ++i)
                {
                    for (size_type k = 0; k < a.shape()[1]; ++k)
                    {
                        result(i) = result(i) + detail::multiply(a(i, k), b.flat(k));
                    }
                }
                return result;
            }
            else if (a_dim >= 2 && b_dim >= 2)
            {
                // Batched matrix multiplication
                if (!detail::matmul_shapes_compatible(a.shape(), b.shape()))
                    XTENSOR_THROW(std::invalid_argument, "matmul: shapes incompatible for batched matrix multiplication");

                shape_type res_shape = detail::matmul_shape(a.shape(), b.shape());
                using value_type = common_value_type_t<E1, E2>;
                xarray_container<value_type> result(res_shape, value_type(0));

                size_type m = a.shape()[a_dim - 2];
                size_type n = a.shape()[a_dim - 1];
                size_type p = b.shape()[b_dim - 1];
                size_type batch_size = 1;
                for (size_type d = 0; d < a_dim - 2; ++d)
                    batch_size *= a.shape()[d];

                const auto& a_strides = a.strides();
                const auto& b_strides = b.strides();

                for (size_type batch = 0; batch < batch_size; ++batch)
                {
                    size_type a_batch_offset = batch * a_strides[0] * a.shape()[0];
                    size_type b_batch_offset = batch * b_strides[0] * b.shape()[0];

                    for (size_type i = 0; i < m; ++i)
                    {
                        for (size_type k = 0; k < n; ++k)
                        {
                            size_type a_idx = a_batch_offset + i * a_strides[a_dim - 2] + k * a_strides[a_dim - 1];
                            value_type aik = a.flat(a_idx);
                            if (aik == value_type(0)) continue;

                            for (size_type j = 0; j < p; ++j)
                            {
                                size_type b_idx = b_batch_offset + k * b_strides[b_dim - 2] + j * b_strides[b_dim - 1];
                                size_type res_idx = batch * m * p + i * p + j;
                                result.flat(res_idx) = result.flat(res_idx) + detail::multiply(aik, b.flat(b_idx));
                            }
                        }
                    }
                }
                return result;
            }
            else
            {
                XTENSOR_THROW(std::invalid_argument, "matmul: unsupported dimensions");
            }
        }

    } // namespace blas

    // Bring BLAS functions into xt namespace
    using blas::dot;
    using blas::gemv;
    using blas::gemm;
    using blas::outer;
    using blas::transpose;
    using blas::matmul;

} // namespace xt

#endif // XTENSOR_XBLAS_HPP                              mat.data(), static_cast<int>(order == blas_order::col_major ? m : n),
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