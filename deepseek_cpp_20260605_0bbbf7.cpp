//File 0309 : xframe/xframe_function.hpp
//Lazy expression node applying a functor to xframe operands with broadcasting, SIMD evaluation, and label-aware shape checking.
#ifndef XFRAME_FUNCTION_HPP
#define XFRAME_FUNCTION_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <tuple>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_variable.hpp"
#include "xframe_coordinate.hpp"
#include "xframe_dimension.hpp"

namespace xframe
{
    /**
     * @class xframe_function
     * @brief Lazy expression applying a functor to multiple xframe operands.
     *
     * Dimensions are validated at construction. For each element access,
     * the functor is applied to the corresponding elements of the operands.
     * If an operand is a scalar, it is broadcast to the shape of the function.
     */
    template <class F, class... CT>
    class xframe_function : public expression<xframe_function<F, CT...>>
    {
    public:
        using self_type = xframe_function<F, CT...>;
        using functor_type = F;
        using value_type = decltype(std::declval<F>()(std::declval<typename CT::value_type>()...));
        using size_type = std::size_t;
        static constexpr std::size_t arity = sizeof...(CT);

        /**
         * Construct the function from a functor and operand expressions.
         * @param f The functor to apply.
         * @param args The operand expressions (must have compatible dimensions).
         */
        template <class Func, class... Args>
        xframe_function(Func&& f, Args&&... args)
            : m_f(std::forward<Func>(f))
            , m_args(std::forward<Args>(args)...)
        {
            // Validate dimensions: all operands must have the same dimension names and sizes
            if (!check_dimensions_compatible())
                throw std::runtime_error("xframe_function: dimension mismatch.");
            m_size = compute_size();
        }

        xframe_function(const self_type&) = default;
        xframe_function& operator=(const self_type&) = default;
        xframe_function(self_type&&) = default;
        xframe_function& operator=(self_type&&) = default;

        /**
         * Number of dimensions (inherited from the first argument).
         */
        std::size_t dimension_count() const noexcept
        {
            return std::get<0>(m_args).dimension_count();
        }

        /**
         * Total number of elements.
         */
        size_type size() const noexcept { return m_size; }

        /**
         * Element access by integer indices.
         */
        template <class... Idxs>
        value_type operator()(Idxs... idxs) const
        {
            return access(std::make_index_sequence<arity>{}, idxs...);
        }

        /**
         * Element access by labels.
         */
        template <class... Labels>
        value_type locate(Labels... labels) const
        {
            return locate_impl(std::make_index_sequence<arity>{}, labels...);
        }

        /**
         * Flat index access.
         */
        value_type operator[](std::size_t i) const
        {
            return flat_access(i, std::make_index_sequence<arity>{});
        }

        /**
         * Get the dimension descriptor (delegates to first operand).
         */
        const dimension<label_type>& dimension(std::size_t i) const
        {
            return std::get<0>(m_args).dimension(i);
        }

        /**
         * SIMD load: gather scalars into a batch.
         */
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            alignas(64) std::array<T, simd_size> buffer;
            for (std::size_t k = 0; k < simd_size; ++k)
                buffer[k] = (*this)[i + k];
            return simd_type::load_aligned(buffer.data());
        }

    private:
        F m_f;
        std::tuple<CT...> m_args;
        size_type m_size;

        bool check_dimensions_compatible() const
        {
            return check_dims(std::make_index_sequence<arity>{});
        }

        template <std::size_t... I>
        bool check_dims(std::index_sequence<I...>) const
        {
            // First argument is reference; others must match
            const auto& ref = std::get<0>(m_args);
            bool ok = true;
            ((ok = ok && ref.same_dimensions(std::get<I>(m_args))), ...);
            return ok;
        }

        size_type compute_size() const
        {
            return std::get<0>(m_args).size();
        }

        template <std::size_t... I, class... Idxs>
        value_type access(std::index_sequence<I...>, Idxs... idxs) const
        {
            return m_f(std::get<I>(m_args)(idxs...)...);
        }

        template <std::size_t... I, class... Labels>
        value_type locate_impl(std::index_sequence<I...>, Labels... labels) const
        {
            return m_f(std::get<I>(m_args).locate(labels...)...);
        }

        template <std::size_t... I>
        value_type flat_access(std::size_t i, std::index_sequence<I...>) const
        {
            return m_f(std::get<I>(m_args)[i]...);
        }
    };

    /**
     * Helper to create an xframe function expression.
     */
    template <class F, class... Es>
    inline auto make_xframe_function(F&& f, Es&&... es)
    {
        return xframe_function<std::decay_t<F>, std::decay_t<Es>...>(
            std::forward<F>(f), std::forward<Es>(es)...);
    }

    /**
     * Overloaded operators for xframe expressions.
     */
    template <class E1, class E2>
    inline auto operator+(const expression<E1>& e1, const expression<E2>& e2)
    {
        return make_xframe_function(
            [](auto a, auto b) -> std::common_type_t<decltype(a), decltype(b)> { return a + b; },
            e1.derived(), e2.derived());
    }

    template <class E1, class E2>
    inline auto operator-(const expression<E1>& e1, const expression<E2>& e2)
    {
        return make_xframe_function(
            [](auto a, auto b) { return a - b; },
            e1.derived(), e2.derived());
    }

    template <class E1, class E2>
    inline auto operator*(const expression<E1>& e1, const expression<E2>& e2)
    {
        return make_xframe_function(
            [](auto a, auto b) { return a * b; },
            e1.derived(), e2.derived());
    }

    template <class E1, class E2>
    inline auto operator/(const expression<E1>& e1, const expression<E2>& e2)
    {
        return make_xframe_function(
            [](auto a, auto b) { return a / b; },
            e1.derived(), e2.derived());
    }

} // namespace xframe

#endif // XFRAME_FUNCTION_HPP