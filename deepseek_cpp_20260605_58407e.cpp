//File 0359 : xframe/xvariable_scalar.hpp
//Scalar variable expression: wraps a single value as a lazy variable of size 1 that broadcasts in element‑wise operations, with SIMD‑aware access and full expression integration.
#ifndef XFRAME_XVARIABLE_SCALAR_HPP
#define XFRAME_XVARIABLE_SCALAR_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xvariable.hpp"

namespace xframe
{
    /**
     * @class xvariable_scalar
     * @brief A lazy scalar expression that behaves like a variable of size 1.
     *
     * When used in arithmetic with variables, it broadcasts to the size
     * of the other operand. It stores a single value and provides element
     * access returning that same value for any index. SIMD loads broadcast
     * the scalar into a batch.
     */
    template <class T = double, class L = label_type>
    class xvariable_scalar : public expression<xvariable_scalar<T, L>>
    {
    public:
        using self_type = xvariable_scalar<T, L>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using label_type = L;

        /**
         * Construct from a scalar value.
         */
        explicit xvariable_scalar(T value = T{}) noexcept : m_value(value) {}

        xvariable_scalar(const self_type&) = default;
        xvariable_scalar& operator=(const self_type&) = default;
        xvariable_scalar(self_type&&) = default;
        xvariable_scalar& operator=(self_type&&) = default;

        /**
         * A scalar variable has size 1 (can be broadcast to any size).
         */
        size_type size() const noexcept { return 1; }
        bool empty() const noexcept { return false; }

        /**
         * Element access returns the stored scalar.
         */
        const_reference operator[](size_type) const noexcept { return m_value; }
        reference operator[](size_type) noexcept { return m_value; }

        /**
         * Data pointer to the scalar.
         */
        pointer data() noexcept { return &m_value; }
        const_pointer data() const noexcept { return &m_value; }

        /**
         * Label is empty for scalar.
         */
        const label_type& name() const noexcept { return m_empty_label; }

        /**
         * SIMD load: broadcast the scalar value into a batch.
         */
        template <class Align, class U = T>
        auto load_simd(std::size_t) const
        {
            using simd_type = xsimd::batch<U, default_simd_arch>;
            return simd_type(static_cast<U>(m_value));
        }

        /**
         * Compound assignment operators (modify the scalar value).
         */
        template <class U>
        self_type& operator+=(U val) { m_value += static_cast<T>(val); return *this; }
        template <class U>
        self_type& operator-=(U val) { m_value -= static_cast<T>(val); return *this; }
        template <class U>
        self_type& operator*=(U val) { m_value *= static_cast<T>(val); return *this; }
        template <class U>
        self_type& operator/=(U val) { m_value /= static_cast<T>(val); return *this; }

    private:
        T m_value;
        label_type m_empty_label;
    };

    /**
     * Helper to create a scalar variable expression.
     */
    template <class T = double>
    inline auto make_scalar_variable(T value)
    {
        return xvariable_scalar<T>(value);
    }

} // namespace xframe

#endif // XFRAME_XVARIABLE_SCALAR_HPP