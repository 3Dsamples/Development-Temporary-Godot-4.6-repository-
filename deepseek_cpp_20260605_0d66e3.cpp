//File 0334 : xframe/xaxis_function.hpp
//Lazy expression node applying a functor to axis operands: supports broadcasting, label alignment, SIMD evaluation, and element-wise math for axis objects.
#ifndef XFRAME_XAXIS_FUNCTION_HPP
#define XFRAME_XAXIS_FUNCTION_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <tuple>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_coordinate.hpp"
#include "xframe_dimension.hpp"
#include "xaxis.hpp"
#include "xaxis_base.hpp"

namespace xframe {
namespace axis {

    /**
     * @class xaxis_function
     * @brief Lazy expression that applies a functor to axis operands.
     *
     * Each operand can be an axis, an axis expression leaf, or a scalar.
     * The function broadcasts along the common axis, aligning labels when
     * possible. Element access evaluates the functor at the given coordinate.
     * SIMD loading gathers scalar results into a batch.
     */
    template <class F, class... CT>
    class xaxis_function : public expression<xaxis_function<F, CT...>>
    {
    public:
        using self_type = xaxis_function<F, CT...>;
        using functor_type = F;
        using value_type = std::decay_t<decltype(std::declval<F>()(
            std::declval<typename CT::value_type>()...))>;
        using size_type = std::size_t;
        using label_type = std::string;
        using coordinate_type = coordinate<label_type>;
        static constexpr std::size_t arity = sizeof...(CT);

        template <class Func, class... Args>
        xaxis_function(Func&& f, Args&&... args)
            : m_f(std::forward<Func>(f))
            , m_args(std::forward<Args>(args)...)
        {
            // Determine the common axis: all operands must share the same axis name,
            // or be scalars (size 1). The resulting axis is the largest of the operands.
            m_axis = compute_common_axis(std::make_index_sequence<arity>{});
            m_size = m_axis.size();
        }

        xaxis_function(const self_type&) = default;
        xaxis_function& operator=(const self_type&) = default;
        xaxis_function(self_type&&) = default;
        xaxis_function& operator=(self_type&&) = default;

        std::size_t dimension_count() const noexcept { return 1; }
        size_type size() const noexcept { return m_size; }

        const dimension<label_type>& dimension(size_type i) const
        {
            if (i != 0) throw std::out_of_range("xaxis_function: dimension index out of range.");
            return m_axis.dimension();
        }

        const xaxis<label_type>& axis() const noexcept { return m_axis; }

        /**
         * Element access: evaluate functor at the given index.
         */
        template <class... Args>
        value_type operator()(size_type idx) const
        {
            return access_at(idx, std::make_index_sequence<arity>{});
        }

        value_type operator[](size_type idx) const { return (*this)(idx); }

        template <class... Labels>
        value_type locate(const label_type& label) const
        {
            size_type idx = m_axis.index_of(label);
            if (idx >= m_axis.size())
                throw std::out_of_range("xaxis_function: label not found.");
            return (*this)(idx);
        }

        /**
         * SIMD load: gather scalar results into a batch.
         */
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            alignas(64) std::array<T, simd_size> buf;
            for (std::size_t k = 0; k < simd_size; ++k)
                buf[k] = (*this)[i + k];
            return simd_type::load_aligned(buf.data());
        }

        const F& functor() const noexcept { return m_f; }

    private:
        F m_f;
        std::tuple<CT...> m_args;
        xaxis<label_type> m_axis;
        size_type m_size;

        /**
         * Compute the common axis by aligning the first operand with all others.
         * If a scalar is encountered, its size is 1 and it broadcasts.
         * All operands must have compatible axis names.
         */
        template <std::size_t... I>
        xaxis<label_type> compute_common_axis(std::index_sequence<I...>)
        {
            // Get the axis of the first operand
            const auto& first_axis = get_axis(std::get<0>(m_args));
            xaxis<label_type> result = first_axis;
            // Align with remaining operands
            auto align = [&](const auto& op) {
                const auto& ax = get_axis(op);
                if (ax.name() == result.name())
                {
                    if (ax.size() > result.size())
                        result = ax; // take larger
                }
                else if (ax.size() != 1 && result.size() != 1)
                {
                    throw std::runtime_error("xaxis_function: incompatible axis names.");
                }
            };
            (align(std::get<I>(m_args)), ...);
            return result;
        }

        /**
         * Extract the axis from an operand.
         * For xaxis, return it directly; for xaxis_expression_leaf, return its axis.
         */
        template <class T>
        static const xaxis<label_type>& get_axis(const T& op)
        {
            if constexpr (std::is_same_v<std::decay_t<T>, xaxis<label_type>>)
            {
                return op;
            }
            else if constexpr (std::is_same_v<std::decay_t<T>, xaxis_expression_leaf<xaxis<label_type>, double>>)
            {
                return op.axis();
            }
            else if constexpr (std::is_same_v<std::decay_t<T>, xaxis_scalar<double, label_type>>)
            {
                // axis_scalar has its own dimension; we'll create a temporary axis from it.
                // For simplicity, throw: not yet supported.
                throw std::runtime_error("get_axis: xaxis_scalar not directly convertible to xaxis.");
            }
            else
            {
                static_assert(std::is_same_v<T, void>, "Unsupported operand type in xaxis_function.");
            }
        }

        template <std::size_t... I>
        value_type access_at(size_type idx, std::index_sequence<I...>) const
        {
            return m_f(get_element(std::get<I>(m_args), idx)...);
        }

        template <class T>
        static value_type get_element(const T& op, size_type idx)
        {
            if constexpr (std::is_arithmetic_v<T>)
                return static_cast<value_type>(op); // scalar broadcast
            else if constexpr (std::is_same_v<std::decay_t<T>, xaxis<label_type>>)
                return static_cast<value_type>(std::stod(op[idx])); // convert label to number
            else if constexpr (std::is_same_v<std::decay_t<T>, xaxis_expression_leaf<xaxis<label_type>, double>>)
                return op[idx]; // returns double directly
            else
                return static_cast<value_type>(op[idx]);
        }
    };

    /**
     * Overloaded operators for axes.
     */
    template <class L>
    inline auto operator+(const xaxis<L>& a, const xaxis<L>& b)
    {
        return xaxis_function<std::plus<double>, xaxis<L>, xaxis<L>>(
            std::plus<double>{}, a, b);
    }

    template <class L>
    inline auto operator-(const xaxis<L>& a, const xaxis<L>& b)
    {
        return xaxis_function<std::minus<double>, xaxis<L>, xaxis<L>>(
            std::minus<double>{}, a, b);
    }

    template <class L>
    inline auto operator*(const xaxis<L>& a, double scalar)
    {
        return xaxis_function<std::multiplies<double>, xaxis<L>, double>(
            std::multiplies<double>{}, a, scalar);
    }

    template <class L>
    inline auto operator*(double scalar, const xaxis<L>& a)
    {
        return a * scalar;
    }

    template <class L>
    inline auto operator/(const xaxis<L>& a, double scalar)
    {
        return xaxis_function<std::divides<double>, xaxis<L>, double>(
            std::divides<double>{}, a, scalar);
    }

} // namespace axis
} // namespace xframe

#endif // XFRAME_XAXIS_FUNCTION_HPP