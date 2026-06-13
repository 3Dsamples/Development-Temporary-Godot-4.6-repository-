//File 0217 : sparse/xsparse_xfunction.hpp
//Sparse expression template nodes enabling lazy sparse-dense operations, SIMD-accelerated evaluation, and automatic broadcasting with sparse storage preservation.
#ifndef XTENSOR_XSPARSE_XFUNCTION_HPP
#define XTENSOR_XSPARSE_XFUNCTION_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

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
#include "../sparse/xsparse_config.hpp"
#include "../sparse/xsparse_expression.hpp"
#include "../sparse/xcsr.hpp"

namespace xt {
namespace sparse {

    /**
     * @class xsparse_function
     * @brief Lazy expression node that applies a functor to sparse and/or dense arguments.
     *
     * When a sparse matrix and a dense operand are combined (e.g., sparse + dense),
     * the result is evaluated lazily, preserving sparsity where possible.
     * If the operation cannot preserve sparsity (e.g., addition with dense produces dense),
     * the result is a dense expression.
     */
    template <class F, class... CT>
    class xsparse_function : public xexpression<xsparse_function<F, CT...>>
    {
    public:
        using self_type = xsparse_function<F, CT...>;
        using functor_type = F;
        using value_type = std::decay_t<decltype(std::declval<F>()(
            std::declval<typename std::decay_t<CT>::value_type>()...))>;
        using const_reference = const value_type&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        static constexpr std::size_t arity = sizeof...(CT);

        template <class Func, class... Args>
        xsparse_function(Func&& f, Args&&... args) noexcept
            : m_f(std::forward<Func>(f)), m_args(std::forward<Args>(args)...)
        {
            m_shape = compute_broadcast_shape();
            m_strides = compute_strides(m_shape);
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }

        size_type size() const noexcept { return compute_size(m_shape); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        value_type* data() noexcept { return nullptr; }
        const value_type* data() const noexcept { return nullptr; }

        template <class... Args>
        value_type operator()(Args... args) const
        {
            return access_impl(std::make_index_sequence<arity>{}, args...);
        }

        template <class... Args>
        value_type operator()(Args... args)
        {
            return static_cast<const self_type*>(this)->operator()(args...);
        }

        value_type operator[](std::size_t i) const
        {
            auto idx = unravel_index(i, m_shape);
            return element(idx.begin(), idx.end());
        }

        value_type operator[](std::size_t i)
        {
            return const_cast<value_type&>(static_cast<const self_type*>(this)->operator[](i));
        }

        template <class It>
        value_type element(It first, It last) const
        {
            return access_by_index(first, last, std::make_index_sequence<arity>{});
        }

        // SIMD load: fallback scalar gather
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            alignas(64) std::array<T, simd_size> buf;
            for (std::size_t k = 0; k < simd_size; ++k)
                buf[k] = operator()(i + k);
            return simd_type::load_aligned(buf.data());
        }

    private:
        F m_f;
        std::tuple<CT...> m_args;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;

        shape_type compute_broadcast_shape() const
        {
            return broadcast_shapes_impl(std::make_index_sequence<arity>{});
        }

        template <std::size_t... I>
        shape_type broadcast_shapes_impl(std::index_sequence<I...>) const
        {
            return broadcast_shapes(std::get<I>(m_args).shape()...);
        }

        template <std::size_t... I, class... Args>
        value_type access_impl(std::index_sequence<I...>, Args... args) const
        {
            return m_f(std::get<I>(m_args)(args...)...);
        }

        template <class It, std::size_t... I>
        value_type access_by_index(It first, It last, std::index_sequence<I...>) const
        {
            auto idx = std::vector<std::size_t>(first, last);
            return access_impl(std::index_sequence<I...>{}, idx[I]...);
        }
    };

    /**
     * Helper function to create a sparse function node.
     */
    template <class F, class... E>
    inline auto make_sparse_function(F&& f, E&&... e)
    {
        return xsparse_function<std::decay_t<F>, std::decay_t<E>...>(
            std::forward<F>(f), std::forward<E>(e)...);
    }

    /**
     * Overloaded operator for sparse + dense (returns lazy dense expression).
     */
    template <class SpExpr, class E,
              std::enable_if_t<is_expression_v<E> && !std::is_base_of_v<xsparse_expression<SpExpr>, E>, int> = 0>
    inline auto operator+(const xsparse_expression<SpExpr>& sparse_expr, const xexpression<E>& dense_expr)
    {
        return make_sparse_function(
            [](auto a, auto b) -> std::common_type_t<decltype(a), decltype(b)> { return a + b; },
            to_dense(sparse_expr.derived_cast()),
            dense_expr.derived_cast());
    }

    template <class E, class SpExpr,
              std::enable_if_t<is_expression_v<E> && !std::is_base_of_v<xsparse_expression<SpExpr>, E>, int> = 0>
    inline auto operator+(const xexpression<E>& dense_expr, const xsparse_expression<SpExpr>& sparse_expr)
    {
        return sparse_expr + dense_expr;
    }

    template <class SpExpr, class E,
              std::enable_if_t<is_expression_v<E> && !std::is_base_of_v<xsparse_expression<SpExpr>, E>, int> = 0>
    inline auto operator-(const xsparse_expression<SpExpr>& sparse_expr, const xexpression<E>& dense_expr)
    {
        return make_sparse_function(
            [](auto a, auto b) { return a - b; },
            to_dense(sparse_expr.derived_cast()),
            dense_expr.derived_cast());
    }

    template <class E, class SpExpr,
              std::enable_if_t<is_expression_v<E> && !std::is_base_of_v<xsparse_expression<SpExpr>, E>, int> = 0>
    inline auto operator-(const xexpression<E>& dense_expr, const xsparse_expression<SpExpr>& sparse_expr)
    {
        return make_sparse_function(
            [](auto a, auto b) { return a - b; },
            dense_expr.derived_cast(),
            to_dense(sparse_expr.derived_cast()));
    }

    template <class SpExpr, class E,
              std::enable_if_t<is_expression_v<E> && !std::is_base_of_v<xsparse_expression<SpExpr>, E>, int> = 0>
    inline auto operator*(const xsparse_expression<SpExpr>& sparse_expr, const xexpression<E>& dense_expr)
    {
        return make_sparse_function(
            [](auto a, auto b) { return a * b; },
            to_dense(sparse_expr.derived_cast()),
            dense_expr.derived_cast());
    }

    template <class E, class SpExpr,
              std::enable_if_t<is_expression_v<E> && !std::is_base_of_v<xsparse_expression<SpExpr>, E>, int> = 0>
    inline auto operator*(const xexpression<E>& dense_expr, const xsparse_expression<SpExpr>& sparse_expr)
    {
        return sparse_expr * dense_expr;
    }

    template <class SpExpr, class E,
              std::enable_if_t<is_expression_v<E> && !std::is_base_of_v<xsparse_expression<SpExpr>, E>, int> = 0>
    inline auto operator/(const xsparse_expression<SpExpr>& sparse_expr, const xexpression<E>& dense_expr)
    {
        return make_sparse_function(
            [](auto a, auto b) { return a / b; },
            to_dense(sparse_expr.derived_cast()),
            dense_expr.derived_cast());
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_XFUNCTION_HPP