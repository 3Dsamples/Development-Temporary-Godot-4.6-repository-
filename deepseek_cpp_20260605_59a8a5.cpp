//File 0209 : sparse/xsparse_operation.hpp
//Sparse element-wise arithmetic operations: addition, subtraction, scaling, and mixed sparse-dense operations with SIMD traversal and expression templates.
#ifndef XTENSOR_XSPARSE_OPERATION_HPP
#define XTENSOR_XSPARSE_OPERATION_HPP

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
#include "../core/xexpression.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xarray.hpp"
#include "../core/xeval.hpp"
#include "../core/xsparse.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xcsr.hpp"
#include "../sparse/xcsc.hpp"
#include "../sparse/xsparse_array.hpp"
#include "../sparse/xsparse_tensor.hpp"

namespace xt {
namespace sparse {

    namespace detail
    {
        // Functor for element-wise addition (scalar)
        struct sparse_plus
        {
            template <class T>
            T operator()(T a, T b) const { return a + b; }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> a, xsimd::batch<T, default_simd_arch> b) const { return a + b; }
        };

        struct sparse_minus
        {
            template <class T>
            T operator()(T a, T b) const { return a - b; }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> a, xsimd::batch<T, default_simd_arch> b) const { return a - b; }
        };

        struct sparse_multiplies
        {
            template <class T>
            T operator()(T a, T b) const { return a * b; }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> a, xsimd::batch<T, default_simd_arch> b) const { return a * b; }
        };

        struct sparse_divides
        {
            template <class T>
            T operator()(T a, T b) const { return a / b; }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> a, xsimd::batch<T, default_simd_arch> b) const { return a / b; }
        };

        struct sparse_negate
        {
            template <class T>
            T operator()(T a) const { return -a; }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> a) const { return -a; }
        };
    }

    /**
     * Element-wise sparse + sparse addition for CSR matrices.
     */
    template <class T>
    inline auto operator+(const xcsr_matrix<T>& A, const xcsr_matrix<T>& B)
    {
        return xcsr_matrix<T>::add(A, B);
    }

    /**
     * Element-wise sparse - sparse subtraction for CSR matrices.
     */
    template <class T>
    inline auto operator-(const xcsr_matrix<T>& A, const xcsr_matrix<T>& B)
    {
        auto negB = B;
        negB *= T(-1);
        return xcsr_matrix<T>::add(A, negB);
    }

    /**
     * Element-wise sparse * scalar multiplication.
     */
    template <class T>
    inline auto operator*(const xcsr_matrix<T>& A, T scalar)
    {
        auto result = A;
        result *= scalar;
        return result;
    }

    template <class T>
    inline auto operator*(T scalar, const xcsr_matrix<T>& A)
    {
        return A * scalar;
    }

    /**
     * Element-wise sparse / scalar division.
     */
    template <class T>
    inline auto operator/(const xcsr_matrix<T>& A, T scalar)
    {
        auto result = A;
        result *= T(1) / scalar;
        return result;
    }

    /**
     * Element-wise unary negation.
     */
    template <class T>
    inline auto operator-(const xcsr_matrix<T>& A)
    {
        return A * T(-1);
    }

    // CSC operations (similar patterns)
    template <class T>
    inline auto operator+(const xcsc_matrix<T>& A, const xcsc_matrix<T>& B)
    {
        return xcsc_matrix<T>::add(A, B);
    }

    template <class T>
    inline auto operator-(const xcsc_matrix<T>& A, const xcsc_matrix<T>& B)
    {
        auto negB = B;
        negB *= T(-1);
        return xcsc_matrix<T>::add(A, negB);
    }

    template <class T>
    inline auto operator*(const xcsc_matrix<T>& A, T scalar)
    {
        auto result = A;
        result *= scalar;
        return result;
    }

    template <class T>
    inline auto operator*(T scalar, const xcsc_matrix<T>& A)
    {
        return A * scalar;
    }

    template <class T>
    inline auto operator/(const xcsc_matrix<T>& A, T scalar)
    {
        auto result = A;
        result *= T(1) / scalar;
        return result;
    }

    template <class T>
    inline auto operator-(const xcsc_matrix<T>& A)
    {
        return A * T(-1);
    }

    // Mixed sparse-dense operations (return dense)
    template <class SpExpr, class E,
              std::enable_if_t<is_expression_v<E>, int> = 0>
    inline auto operator+(const xexpression<SpExpr>& sparse_expr, const xexpression<E>& dense_expr)
    {
        const auto& sp = sparse_expr.derived_cast();
        const auto& dense = dense_expr.derived_cast();
        // Convert sparse to dense and add
        auto dense_sp = to_dense(sp);
        return dense_sp + dense;
    }

    template <class E, class SpExpr,
              std::enable_if_t<is_expression_v<E>, int> = 0>
    inline auto operator+(const xexpression<E>& dense_expr, const xexpression<SpExpr>& sparse_expr)
    {
        return sparse_expr + dense_expr;
    }

    template <class SpExpr, class E,
              std::enable_if_t<is_expression_v<E>, int> = 0>
    inline auto operator-(const xexpression<SpExpr>& sparse_expr, const xexpression<E>& dense_expr)
    {
        const auto& sp = sparse_expr.derived_cast();
        const auto& dense = dense_expr.derived_cast();
        auto dense_sp = to_dense(sp);
        return dense_sp - dense;
    }

    template <class E, class SpExpr,
              std::enable_if_t<is_expression_v<E>, int> = 0>
    inline auto operator-(const xexpression<E>& dense_expr, const xexpression<SpExpr>& sparse_expr)
    {
        const auto& dense = dense_expr.derived_cast();
        const auto& sp = sparse_expr.derived_cast();
        auto dense_sp = to_dense(sp);
        return dense - dense_sp;
    }

    // Sparse-sparse mixed format operations: CSR + COO, etc.
    template <class T>
    inline auto operator+(const xcsr_matrix<T>& csr, const xcoo_matrix<T>& coo)
    {
        auto csr_from_coo = xcsr_matrix<T>::from_coo(coo);
        return csr + csr_from_coo;
    }

    template <class T>
    inline auto operator+(const xcoo_matrix<T>& coo, const xcsr_matrix<T>& csr)
    {
        return csr + coo;
    }

    /**
     * Dense scaling of a sparse array (sparse * dense scalar broadcast).
     * This creates a lazy expression that scales each non-zero element.
     */
    template <class SpExpr, class T,
              std::enable_if_t<is_expression_v<SpExpr>, int> = 0>
    inline auto operator*(const xexpression<SpExpr>& sparse_expr, T scalar)
    {
        const auto& sp = sparse_expr.derived_cast();
        // For sparse arrays, scaling maintains sparsity pattern; we can create a scaled copy.
        // For generality, we'll convert to dense and scale, or implement lazy scaled view.
        // Here we produce a dense result for simplicity, but this could be a lazy scaled view.
        return to_dense(sp) * scalar;
    }

    template <class SpExpr, class T,
              std::enable_if_t<is_expression_v<SpExpr>, int> = 0>
    inline auto operator*(T scalar, const xexpression<SpExpr>& sparse_expr)
    {
        return sparse_expr * scalar;
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_OPERATION_HPP