//File 0349 : xframe/xvariable_base.hpp
//CRTP base class for all variable types: provides common interface for size, data access, resizing, filling, SIMD-accelerated arithmetic, and expression integration.
#ifndef XFRAME_XVARIABLE_BASE_HPP
#define XFRAME_XVARIABLE_BASE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"

namespace xframe
{
    /**
     * @class xvariable_base
     * @brief CRTP base class for all variable types.
     *
     * Provides the common interface for 1D data containers: size, data(),
     * operator[], resizing, filling, iterators, and arithmetic operations.
     * Derived classes must implement the actual storage and element access.
     * The base handles SIMD-accelerated arithmetic via the simd_enabled_v trait.
     */
    template <class D, class T = double, class L = label_type>
    class xvariable_base : public expression<D>
    {
    public:
        using derived_type = D;
        using self_type = xvariable_base<D, T, L>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator = pointer;
        using const_iterator = const_pointer;
        using label_type = L;

        /**
         * Construct with size and optional name.
         */
        explicit xvariable_base(size_type size, const label_type& name = label_type{})
            : m_name(name), m_size(size)
        {
        }

        xvariable_base(const self_type&) = default;
        xvariable_base& operator=(const self_type&) = default;
        xvariable_base(self_type&&) = default;
        xvariable_base& operator=(self_type&&) = default;

        virtual ~xvariable_base() = default;

        /**
         * Size and name.
         */
        size_type size() const noexcept { return m_size; }
        bool empty() const noexcept { return m_size == 0; }
        const label_type& name() const noexcept { return m_name; }
        void set_name(const label_type& n) { m_name = n; }

        /**
         * Data access (must be overridden by derived).
         */
        virtual pointer data() noexcept = 0;
        virtual const_pointer data() const noexcept = 0;

        /**
         * Element access.
         */
        reference operator[](size_type i) { return data()[i]; }
        const_reference operator[](size_type i) const { return data()[i]; }

        reference at(size_type i)
        {
            if (i >= m_size) throw std::out_of_range("xvariable_base::at");
            return data()[i];
        }
        const_reference at(size_type i) const
        {
            if (i >= m_size) throw std::out_of_range("xvariable_base::at");
            return data()[i];
        }

        reference front() { return data()[0]; }
        const_reference front() const { return data()[0]; }
        reference back() { return data()[m_size - 1]; }
        const_reference back() const { return data()[m_size - 1]; }

        /**
         * Iterators.
         */
        iterator begin() noexcept { return data(); }
        iterator end() noexcept { return data() + m_size; }
        const_iterator begin() const noexcept { return data(); }
        const_iterator end() const noexcept { return data() + m_size; }
        const_iterator cbegin() const noexcept { return data(); }
        const_iterator cend() const noexcept { return data() + m_size; }

        /**
         * Resize (must be overridden by derived for actual reallocation).
         */
        virtual void resize(size_type new_size) = 0;

        /**
         * Reserve capacity (optional, default no‑op).
         */
        virtual void reserve(size_type cap) { /* no‑op */ }

        /**
         * Fill with a constant value.
         */
        virtual void fill(const_reference val)
        {
            std::fill(data(), data() + m_size, val);
        }

        /**
         * SIMD load.
         */
        template <class Align, class U = T>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<U, default_simd_arch>;
            return simd_type::load_unaligned(data() + i);
        }

        /**
         * Arithmetic assignment operators (in‑place).
         * These work with any derived type that provides proper data().
         */
        template <class Other>
        derived_type& operator+=(const Other& rhs)
        {
            if (m_size != rhs.size())
                throw std::runtime_error("xvariable_base::operator+=: size mismatch.");
            T* d = data();
            const T* r = rhs.data();
            if constexpr (simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = m_size / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type va = simd_type::load_unaligned(d + i * simd_size);
                    simd_type vb = simd_type::load_unaligned(r + i * simd_size);
                    (va + vb).store_unaligned(d + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < m_size; ++i)
                    d[i] += r[i];
            }
            else
            {
                for (std::size_t i = 0; i < m_size; ++i)
                    d[i] += r[i];
            }
            return static_cast<derived_type&>(*this);
        }

        template <class Other>
        derived_type& operator-=(const Other& rhs)
        {
            if (m_size != rhs.size())
                throw std::runtime_error("xvariable_base::operator-=: size mismatch.");
            T* d = data();
            const T* r = rhs.data();
            if constexpr (simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = m_size / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type va = simd_type::load_unaligned(d + i * simd_size);
                    simd_type vb = simd_type::load_unaligned(r + i * simd_size);
                    (va - vb).store_unaligned(d + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < m_size; ++i)
                    d[i] -= r[i];
            }
            else
            {
                for (std::size_t i = 0; i < m_size; ++i)
                    d[i] -= r[i];
            }
            return static_cast<derived_type&>(*this);
        }

        derived_type& operator*=(const_reference scalar)
        {
            T* d = data();
            if constexpr (simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = m_size / simd_size;
                simd_type vs(scalar);
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type v = simd_type::load_unaligned(d + i * simd_size);
                    (v * vs).store_unaligned(d + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < m_size; ++i)
                    d[i] *= scalar;
            }
            else
            {
                for (std::size_t i = 0; i < m_size; ++i)
                    d[i] *= scalar;
            }
            return static_cast<derived_type&>(*this);
        }

        derived_type& operator/=(const_reference scalar)
        {
            T* d = data();
            if constexpr (simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = m_size / simd_size;
                simd_type vs(scalar);
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type v = simd_type::load_unaligned(d + i * simd_size);
                    (v / vs).store_unaligned(d + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < m_size; ++i)
                    d[i] /= scalar;
            }
            else
            {
                for (std::size_t i = 0; i < m_size; ++i)
                    d[i] /= scalar;
            }
            return static_cast<derived_type&>(*this);
        }

    protected:
        label_type m_name;
        size_type m_size;
    };

} // namespace xframe

#endif // XFRAME_XVARIABLE_BASE_HPP