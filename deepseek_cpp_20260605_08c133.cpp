//File 0060 : core/xoptional.hpp
//Optional value support for xtensor arrays: xoptional type combining value and missing flag, with SIMD-accelerated operations, lazy propagation of missing values, and full expression integration.
#ifndef XTENSOR_XOPTIONAL_HPP
#define XTENSOR_XOPTIONAL_HPP

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <limits>
#include <cmath>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xarray.hpp"
#include "xscalar.hpp"

namespace xt
{
    /*********************************************
     * xoptional – value with missing flag
     *********************************************/
    template <class T>
    class xoptional
    {
    public:
        using value_type = T;

        constexpr xoptional() noexcept
            : m_value(T{}), m_has_value(false) {}

        constexpr xoptional(const T& value) noexcept
            : m_value(value), m_has_value(true) {}

        constexpr xoptional(const T& value, bool has_val) noexcept
            : m_value(value), m_has_value(has_val) {}

        constexpr xoptional(bool has_val, const T& value) noexcept
            : m_value(value), m_has_value(has_val) {}

        constexpr xoptional(const xoptional&) = default;
        constexpr xoptional& operator=(const xoptional&) = default;
        constexpr xoptional(xoptional&&) = default;
        constexpr xoptional& operator=(xoptional&&) = default;

        constexpr bool has_value() const noexcept { return m_has_value; }
        constexpr operator bool() const noexcept { return m_has_value; }
        constexpr const T& value() const noexcept { return m_value; }
        constexpr T& value() noexcept { return m_value; }
        constexpr const T& operator*() const noexcept { return m_value; }
        constexpr T& operator*() noexcept { return m_value; }
        constexpr const T* operator->() const noexcept { return &m_value; }
        constexpr T* operator->() noexcept { return &m_value; }

        template <class U>
        constexpr T value_or(U&& default_value) const noexcept
        {
            return m_has_value ? m_value : static_cast<T>(std::forward<U>(default_value));
        }

        friend constexpr bool operator==(const xoptional& lhs, const xoptional& rhs) noexcept
        {
            if (lhs.m_has_value != rhs.m_has_value) return false;
            if (!lhs.m_has_value) return true;
            return lhs.m_value == rhs.m_value;
        }

        friend constexpr bool operator!=(const xoptional& lhs, const xoptional& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        friend constexpr bool operator<(const xoptional& lhs, const xoptional& rhs) noexcept
        {
            if (!lhs.m_has_value || !rhs.m_has_value) return false;
            return lhs.m_value < rhs.m_value;
        }

        friend constexpr bool operator<=(const xoptional& lhs, const xoptional& rhs) noexcept
        {
            if (!lhs.m_has_value || !rhs.m_has_value) return false;
            return lhs.m_value <= rhs.m_value;
        }

        friend constexpr bool operator>(const xoptional& lhs, const xoptional& rhs) noexcept
        {
            if (!lhs.m_has_value || !rhs.m_has_value) return false;
            return lhs.m_value > rhs.m_value;
        }

        friend constexpr bool operator>=(const xoptional& lhs, const xoptional& rhs) noexcept
        {
            if (!lhs.m_has_value || !rhs.m_has_value) return false;
            return lhs.m_value >= rhs.m_value;
        }

        // Arithmetic operations: propagate missing value
        friend constexpr xoptional operator+(const xoptional& a, const xoptional& b) noexcept
        {
            if (!a.m_has_value || !b.m_has_value) return xoptional(false, T{});
            return xoptional(a.m_value + b.m_value);
        }

        friend constexpr xoptional operator-(const xoptional& a, const xoptional& b) noexcept
        {
            if (!a.m_has_value || !b.m_has_value) return xoptional(false, T{});
            return xoptional(a.m_value - b.m_value);
        }

        friend constexpr xoptional operator*(const xoptional& a, const xoptional& b) noexcept
        {
            if (!a.m_has_value || !b.m_has_value) return xoptional(false, T{});
            return xoptional(a.m_value * b.m_value);
        }

        friend constexpr xoptional operator/(const xoptional& a, const xoptional& b) noexcept
        {
            if (!a.m_has_value || !b.m_has_value) return xoptional(false, T{});
            return xoptional(a.m_value / b.m_value);
        }

        friend constexpr xoptional operator-(const xoptional& a) noexcept
        {
            if (!a.m_has_value) return xoptional(false, T{});
            return xoptional(-a.m_value);
        }

        xoptional& operator+=(const xoptional& other) noexcept
        {
            if (!m_has_value || !other.m_has_value) { m_has_value = false; return *this; }
            m_value += other.m_value;
            return *this;
        }

        xoptional& operator-=(const xoptional& other) noexcept
        {
            if (!m_has_value || !other.m_has_value) { m_has_value = false; return *this; }
            m_value -= other.m_value;
            return *this;
        }

        xoptional& operator*=(const xoptional& other) noexcept
        {
            if (!m_has_value || !other.m_has_value) { m_has_value = false; return *this; }
            m_value *= other.m_value;
            return *this;
        }

        xoptional& operator/=(const xoptional& other) noexcept
        {
            if (!m_has_value || !other.m_has_value) { m_has_value = false; return *this; }
            m_value /= other.m_value;
            return *this;
        }

    private:
        T m_value;
        bool m_has_value;
    };

    /**
     * @class xoptional_assembly
     * @brief A container that stores values and missing flags as separate arrays,
     *        enabling SIMD-accelerated operations on optional data.
     */
    template <class T>
    class xoptional_assembly
    {
    public:
        using value_type = xoptional<T>;
        using size_type = std::size_t;

        xoptional_assembly() = default;

        xoptional_assembly(const xarray_container<uvector<T>>& values,
                           const xarray_container<uvector<bool>>& flags)
            : m_values(values), m_flags(flags)
        {
            if (values.shape() != flags.shape())
                throw std::runtime_error("xoptional_assembly: shape mismatch between values and flags.");
        }

        xoptional_assembly(xarray_container<uvector<T>>&& values,
                           xarray_container<uvector<bool>>&& flags)
            : m_values(std::move(values)), m_flags(std::move(flags))
        {
            if (m_values.shape() != m_flags.shape())
                throw std::runtime_error("xoptional_assembly: shape mismatch.");
        }

        const xarray_container<uvector<T>>& values() const noexcept { return m_values; }
        xarray_container<uvector<T>>& values() noexcept { return m_values; }
        const xarray_container<uvector<bool>>& flags() const noexcept { return m_flags; }
        xarray_container<uvector<bool>>& flags() noexcept { return m_flags; }

        value_type element(const std::vector<size_type>& idx) const
        {
            return value_type(m_values.element(idx.begin(), idx.end()),
                              m_flags.element(idx.begin(), idx.end()));
        }

        void set_element(const std::vector<size_type>& idx, const value_type& val)
        {
            m_values.element(idx.begin(), idx.end()) = val.value();
            m_flags.element(idx.begin(), idx.end()) = val.has_value();
        }

        size_type size() const noexcept { return m_values.size(); }
        auto shape() const noexcept { return m_values.shape(); }

        /**
         * Count of non-missing elements.
         */
        size_type count() const noexcept
        {
            size_type cnt = 0;
            for (size_type i = 0; i < m_flags.size(); ++i)
                if (m_flags[i]) ++cnt;
            return cnt;
        }

        /**
         * Sum of non-missing values (0 for missing).
         */
        T sum() const noexcept
        {
            T s = T(0);
            if constexpr (is_simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                const T* val_ptr = m_values.data();
                const bool* flg_ptr = m_flags.data();
                std::size_t n = m_values.size();
                std::size_t vec_count = n / simd_size;
                simd_type vsum(0);
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    // Load flags as integer mask
                    alignas(64) std::array<T, simd_size> mask_vals;
                    for (std::size_t k = 0; k < simd_size; ++k)
                        mask_vals[k] = flg_ptr[i * simd_size + k] ? T(1) : T(0);
                    simd_type mask = simd_type::load_aligned(mask_vals.data());
                    simd_type vals = simd_type::load_unaligned(val_ptr + i * simd_size);
                    vsum = vsum + vals * mask;
                }
                T tmp[simd_size];
                vsum.store_unaligned(tmp);
                for (std::size_t k = 0; k < simd_size; ++k) s += tmp[k];
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    if (flg_ptr[i]) s += val_ptr[i];
            }
            else
            {
                for (size_type i = 0; i < m_values.size(); ++i)
                    if (m_flags[i]) s += m_values[i];
            }
            return s;
        }

        /**
         * Mean of non-missing values.
         */
        T mean() const noexcept
        {
            size_type cnt = count();
            if (cnt == 0) return std::numeric_limits<T>::quiet_NaN();
            return sum() / static_cast<T>(cnt);
        }

    private:
        xarray_container<uvector<T>> m_values;
        xarray_container<uvector<bool>> m_flags;
    };

    /**
     * Helper: split an xoptional array into values and flags.
     */
    template <class T>
    inline auto split_optional(const xarray_container<uvector<xoptional<T>>>& opt_array)
    {
        auto sh = opt_array.shape();
        xarray_container<uvector<T>> values(sh);
        xarray_container<uvector<bool>> flags(sh);
        for (std::size_t i = 0; i < opt_array.size(); ++i)
        {
            values[i] = opt_array[i].value();
            flags[i] = opt_array[i].has_value();
        }
        return xoptional_assembly<T>(std::move(values), std::move(flags));
    }

    /**
     * Helper: merge values and flags into an xoptional array.
     */
    template <class T>
    inline auto merge_optional(const xoptional_assembly<T>& assembly)
    {
        auto sh = assembly.shape();
        xarray_container<uvector<xoptional<T>>> result(sh);
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            result[i] = xoptional<T>(assembly.values()[i], assembly.flags()[i]);
        }
        return result;
    }

    // Specialization of xcontainer_inner_types for xoptional_assembly
    template <class T>
    struct xcontainer_inner_types<xoptional_assembly<T>>
    {
        using value_type = xoptional<T>;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<uvector<value_type>, DEFAULT_LAYOUT, shape_type>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

} // namespace xt

#endif // XTENSOR_XOPTIONAL_HPP