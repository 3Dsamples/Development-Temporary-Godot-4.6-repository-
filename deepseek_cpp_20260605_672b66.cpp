//File 0350 : xframe/xvariable_function.hpp
//Lazy expression node applying a functor to variable operands: supports element‑wise arithmetic, broadcasting, SIMD evaluation, and full expression integration.
#ifndef XFRAME_XVARIABLE_FUNCTION_HPP
#define XFRAME_XVARIABLE_FUNCTION_HPP

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
#include "xvariable.hpp"
#include "xvariable_base.hpp"
#include "xvariable_assign.hpp"

namespace xframe
{
    /**
     * @class xvariable_function
     * @brief Lazy expression that applies a binary or unary functor to variable operands.
     *
     * The result has the size of the first operand (broadcasting is supported:
     * if an operand has size 1, it is broadcast to the common size).
     * Element access evaluates the functor on the fly, and SIMD loads gather
     * scalar results into a batch for performance.
     */
    template <class F, class... CT>
    class xvariable_function : public expression<xvariable_function<F, CT...>>
    {
    public:
        using self_type = xvariable_function<F, CT...>;
        using functor_type = F;
        using value_type = std::decay_t<decltype(std::declval<F>()(
            std::declval<typename std::decay_t<CT>::value_type>()...))>;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator = pointer;
        using const_iterator = const_pointer;
        using label_type = std::string;
        static constexpr std::size_t arity = sizeof...(CT);

        /**
         * Construct the function expression.
         * @param f The functor to apply.
         * @param args The operand expressions (variables or scalars).
         */
        template <class Func, class... Args>
        xvariable_function(Func&& f, Args&&... args)
            : m_f(std::forward<Func>(f))
            , m_args(std::forward<Args>(args)...)
        {
            m_size = compute_common_size(std::make_index_sequence<arity>{});
            if (m_size == 0)
                throw std::runtime_error("xvariable_function: operands have incompatible sizes.");
            // Compute the common label by concatenating operand names.
            m_label = compute_common_label(std::make_index_sequence<arity>{});
        }

        xvariable_function(const self_type&) = default;
        xvariable_function& operator=(const self_type&) = default;
        xvariable_function(self_type&&) = default;
        xvariable_function& operator=(self_type&&) = default;

        /**
         * Size and label.
         */
        size_type size() const noexcept { return m_size; }
        const label_type& name() const noexcept { return m_label; }

        /**
         * Element access by index.
         */
        value_type operator[](size_type i) const
        {
            return access_at(i, std::make_index_sequence<arity>{});
        }

        /**
         * Data pointer: the expression does not own contiguous storage.
         */
        pointer data() noexcept { return nullptr; }
        const_pointer data() const noexcept { return nullptr; }

        /**
         * Iterators: generate on‑the‑fly via linear access.
         */
        iterator begin() noexcept { return nullptr; }
        iterator end() noexcept { return nullptr; }
        const_iterator begin() const noexcept { return nullptr; }
        const_iterator end() const noexcept { return nullptr; }

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

        /**
         * Fill the expression into a destination variable (materialization).
         */
        template <class Var>
        void assign_to(Var& dst) const
        {
            if (dst.size() != m_size)
                dst.resize(m_size);
            using T = typename Var::value_type;
            T* dst_data = dst.data();
            if constexpr (simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = m_size / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    alignas(64) std::array<T, simd_size> buf;
                    for (std::size_t k = 0; k < simd_size; ++k)
                        buf[k] = static_cast<T>((*this)[i * simd_size + k]);
                    simd_type v = simd_type::load_aligned(buf.data());
                    v.store_unaligned(dst_data + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < m_size; ++i)
                    dst_data[i] = static_cast<T>((*this)[i]);
            }
            else
            {
                for (std::size_t i = 0; i < m_size; ++i)
                    dst_data[i] = static_cast<T>((*this)[i]);
            }
        }

        const F& functor() const noexcept { return m_f; }
        const std::tuple<CT...>& operands() const noexcept { return m_args; }

    private:
        F m_f;
        std::tuple<CT...> m_args;
        size_type m_size = 0;
        label_type m_label;

        /**
         * Compute the common size: all operands must have the same size,
         * or be size 1 (scalar broadcast). The result takes the maximum.
         */
        template <std::size_t... I>
        size_type compute_common_size(std::index_sequence<I...>) const
        {
            size_type sz = 0;
            auto check = [&](const auto& op) {
                size_type s = get_size(op);
                if (sz == 0) sz = s;
                else if (s != 1 && sz != 1 && s != sz)
                    throw std::runtime_error("xvariable_function: size mismatch.");
                else if (s > sz) sz = s;
            };
            (check(std::get<I>(m_args)), ...);
            return sz;
        }

        /**
         * Extract size from an operand.
         * For variables, returns var.size(); for scalars, returns 1.
         */
        template <class T>
        static size_type get_size(const T& op)
        {
            if constexpr (std::is_arithmetic_v<T>)
                return 1;
            else if constexpr (has_size_member_v<T>)
                return op.size();
            else
                return 1; // fallback for unknown types
        }

        // Helper trait to detect if a type has a size() member
        template <class T, class = void>
        struct has_size_member : std::false_type {};
        template <class T>
        struct has_size_member<T, std::void_t<decltype(std::declval<const T&>().size())>>
            : std::true_type {};
        template <class T>
        static inline constexpr bool has_size_member_v = has_size_member<T>::value;

        /**
         * Compute a label by concatenating operand names.
         */
        template <std::size_t... I>
        label_type compute_common_label(std::index_sequence<I...>) const
        {
            label_type result;
            auto append = [&](const auto& op) {
                if constexpr (has_name_member_v<std::decay_t<decltype(op)>>)
                {
                    if (!result.empty()) result += "_";
                    result += op.name();
                }
            };
            (append(std::get<I>(m_args)), ...);
            return result;
        }

        // Helper trait to detect if a type has a name() member
        template <class T, class = void>
        struct has_name_member : std::false_type {};
        template <class T>
        struct has_name_member<T, std::void_t<decltype(std::declval<const T&>().name())>>
            : std::true_type {};
        template <class T>
        static inline constexpr bool has_name_member_v = has_name_member<T>::value;

        /**
         * Access element at index i by applying the functor to each operand.
         * Handles scalar broadcast: if an operand has size 1, its value at index 0 is used.
         */
        template <std::size_t... I>
        value_type access_at(size_type i, std::index_sequence<I...>) const
        {
            return m_f(get_value(std::get<I>(m_args), i)...);
        }

        /**
         * Get the value of an operand at index i.
         * If the operand is arithmetic (scalar), returns it directly.
         * If the operand has size 1, returns its element 0.
         * Otherwise returns op[i].
         */
        template <class T>
        static auto get_value(const T& op, size_type i) -> decltype(op[0])
        {
            if constexpr (std::is_arithmetic_v<T>)
                return static_cast<value_type>(op);
            else if constexpr (has_size_member_v<T>)
            {
                if (op.size() == 1) return op[0];
                return op[i];
            }
            else
                return op[i];
        }
    };

    /**
     * Helper to create an xvariable_function.
     */
    template <class F, class... Es>
    inline auto make_variable_function(F&& f, Es&&... es)
    {
        return xvariable_function<std::decay_t<F>, std::decay_t<Es>...>(
            std::forward<F>(f), std::forward<Es>(es)...);
    }

    /**
     * Overloaded arithmetic operators for variables.
     */
    template <class T, class L>
    inline auto operator+(const variable<T, L>& a, const variable<T, L>& b)
    {
        return make_variable_function(std::plus<T>{}, a, b);
    }

    template <class T, class L>
    inline auto operator-(const variable<T, L>& a, const variable<T, L>& b)
    {
        return make_variable_function(std::minus<T>{}, a, b);
    }

    template <class T, class L>
    inline auto operator*(const variable<T, L>& a, const variable<T, L>& b)
    {
        return make_variable_function(std::multiplies<T>{}, a, b);
    }

    template <class T, class L>
    inline auto operator/(const variable<T, L>& a, const variable<T, L>& b)
    {
        return make_variable_function(std::divides<T>{}, a, b);
    }

    template <class T, class L>
    inline auto operator+(const variable<T, L>& a, T scalar)
    {
        return make_variable_function(
            [s = scalar](T x) { return x + s; }, a);
    }

    template <class T, class L>
    inline auto operator+(T scalar, const variable<T, L>& a)
    {
        return a + scalar;
    }

    template <class T, class L>
    inline auto operator-(const variable<T, L>& a, T scalar)
    {
        return make_variable_function(
            [s = scalar](T x) { return x - s; }, a);
    }

    template <class T, class L>
    inline auto operator-(T scalar, const variable<T, L>& a)
    {
        return make_variable_function(
            [s = scalar](T x) { return s - x; }, a);
    }

    template <class T, class L>
    inline auto operator*(const variable<T, L>& a, T scalar)
    {
        return make_variable_function(
            [s = scalar](T x) { return x * s; }, a);
    }

    template <class T, class L>
    inline auto operator*(T scalar, const variable<T, L>& a)
    {
        return a * scalar;
    }

    template <class T, class L>
    inline auto operator/(const variable<T, L>& a, T scalar)
    {
        return make_variable_function(
            [s = scalar](T x) { return x / s; }, a);
    }

    template <class T, class L>
    inline auto operator-(const variable<T, L>& a)
    {
        return make_variable_function(std::negate<T>{}, a);
    }

    // Comparison operators
    template <class T, class L>
    inline auto operator==(const variable<T, L>& a, const variable<T, L>& b)
    {
        return make_variable_function(std::equal_to<T>{}, a, b);
    }

    template <class T, class L>
    inline auto operator!=(const variable<T, L>& a, const variable<T, L>& b)
    {
        return make_variable_function(std::not_equal_to<T>{}, a, b);
    }

    template <class T, class L>
    inline auto operator<(const variable<T, L>& a, const variable<T, L>& b)
    {
        return make_variable_function(std::less<T>{}, a, b);
    }

    template <class T, class L>
    inline auto operator<=(const variable<T, L>& a, const variable<T, L>& b)
    {
        return make_variable_function(std::less_equal<T>{}, a, b);
    }

    template <class T, class L>
    inline auto operator>(const variable<T, L>& a, const variable<T, L>& b)
    {
        return make_variable_function(std::greater<T>{}, a, b);
    }

    template <class T, class L>
    inline auto operator>=(const variable<T, L>& a, const variable<T, L>& b)
    {
        return make_variable_function(std::greater_equal<T>{}, a, b);
    }

} // namespace xframe

#endif // XFRAME_XVARIABLE_FUNCTION_HPP