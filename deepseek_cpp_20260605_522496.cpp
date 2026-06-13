//File 0347 : xframe/xvariable.hpp
//Original xframe variable: a named, typed 1D array with optional coordinate labels, SIMD-aligned storage, and expression integration.
#ifndef XFRAME_XVARIABLE_HPP
#define XFRAME_XVARIABLE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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
     * @class variable
     * @brief A named 1D array of values with optional coordinate labels.
     *
     * The fundamental data carrier in xframe. Stores a contiguous buffer
     * of type T with SIMD-friendly alignment. Supports element access,
     * iteration, resizing, and arithmetic with other variables.
     */
    template <class T = double, class L = label_type>
    class variable : public expression<variable<T, L>>
    {
    public:
        using self_type = variable<T, L>;
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
         * Default constructor – empty variable.
         */
        variable() noexcept
            : m_size(0), m_capacity(0), m_data(nullptr) {}

        /**
         * Construct with size and optional name.
         */
        explicit variable(size_type size, const label_type& name = label_type{})
            : m_name(name), m_size(size), m_capacity(align_up(size, 64 / sizeof(T)))
        {
            if (m_capacity > 0)
            {
                m_data = allocate(m_capacity);
                std::fill(m_data, m_data + m_size, value_type{});
            }
        }

        /**
         * Construct with size, name, and initializer list of values.
         */
        variable(size_type size, const label_type& name,
                 std::initializer_list<T> values)
            : m_name(name), m_size(size), m_capacity(align_up(size, 64 / sizeof(T)))
        {
            m_data = allocate(m_capacity);
            std::size_t n = std::min(size, values.size());
            std::copy(values.begin(), values.begin() + n, m_data);
            if (n < size) std::fill(m_data + n, m_data + size, T{});
        }

        /**
         * Construct from a vector.
         */
        explicit variable(const std::vector<T>& values,
                          const label_type& name = label_type{})
            : m_name(name), m_size(values.size()),
              m_capacity(align_up(values.size(), 64 / sizeof(T)))
        {
            m_data = allocate(m_capacity);
            std::copy(values.begin(), values.end(), m_data);
        }

        /**
         * Copy constructor.
         */
        variable(const self_type& rhs)
            : m_name(rhs.m_name), m_size(rhs.m_size),
              m_capacity(rhs.m_capacity)
        {
            if (m_capacity > 0)
            {
                m_data = allocate(m_capacity);
                std::copy(rhs.m_data, rhs.m_data + m_size, m_data);
            }
        }

        /**
         * Move constructor.
         */
        variable(self_type&& rhs) noexcept
            : m_name(std::move(rhs.m_name)), m_size(rhs.m_size),
              m_capacity(rhs.m_capacity), m_data(rhs.m_data)
        {
            rhs.m_data = nullptr;
            rhs.m_size = 0;
            rhs.m_capacity = 0;
        }

        /**
         * Copy assignment.
         */
        self_type& operator=(const self_type& rhs)
        {
            if (this != &rhs)
            {
                deallocate();
                m_name = rhs.m_name;
                m_size = rhs.m_size;
                m_capacity = rhs.m_capacity;
                if (m_capacity > 0)
                {
                    m_data = allocate(m_capacity);
                    std::copy(rhs.m_data, rhs.m_data + m_size, m_data);
                }
            }
            return *this;
        }

        /**
         * Move assignment.
         */
        self_type& operator=(self_type&& rhs) noexcept
        {
            if (this != &rhs)
            {
                deallocate();
                m_name = std::move(rhs.m_name);
                m_size = rhs.m_size;
                m_capacity = rhs.m_capacity;
                m_data = rhs.m_data;
                rhs.m_data = nullptr;
                rhs.m_size = 0;
                rhs.m_capacity = 0;
            }
            return *this;
        }

        ~variable() { deallocate(); }

        /**
         * Size and name.
         */
        size_type size() const noexcept { return m_size; }
        bool empty() const noexcept { return m_size == 0; }
        const label_type& name() const noexcept { return m_name; }
        void set_name(const label_type& n) { m_name = n; }

        /**
         * Data access.
         */
        pointer data() noexcept { return m_data; }
        const_pointer data() const noexcept { return m_data; }

        reference operator[](size_type i) { return m_data[i]; }
        const_reference operator[](size_type i) const { return m_data[i]; }

        reference at(size_type i)
        {
            if (i >= m_size) throw std::out_of_range("variable::at");
            return m_data[i];
        }
        const_reference at(size_type i) const
        {
            if (i >= m_size) throw std::out_of_range("variable::at");
            return m_data[i];
        }

        reference front() { return m_data[0]; }
        const_reference front() const { return m_data[0]; }
        reference back() { return m_data[m_size - 1]; }
        const_reference back() const { return m_data[m_size - 1]; }

        /**
         * Iterators.
         */
        iterator begin() noexcept { return m_data; }
        iterator end() noexcept { return m_data + m_size; }
        const_iterator begin() const noexcept { return m_data; }
        const_iterator end() const noexcept { return m_data + m_size; }
        const_iterator cbegin() const noexcept { return m_data; }
        const_iterator cend() const noexcept { return m_data + m_size; }

        /**
         * Resize.
         */
        void resize(size_type new_size)
        {
            if (new_size <= m_capacity)
            {
                m_size = new_size;
                return;
            }
            size_type new_cap = align_up(new_size, 64 / sizeof(T));
            pointer new_data = allocate(new_cap);
            std::copy(m_data, m_data + m_size, new_data);
            std::fill(new_data + m_size, new_data + new_size, value_type{});
            deallocate();
            m_data = new_data;
            m_size = new_size;
            m_capacity = new_cap;
        }

        /**
         * Fill with constant value.
         */
        void fill(const_reference val)
        {
            std::fill(m_data, m_data + m_size, val);
        }

        /**
         * SIMD load.
         */
        template <class Align, class U = T>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<U, default_simd_arch>;
            return simd_type::load_unaligned(m_data + i);
        }

        /**
         * Element‑wise addition (in‑place).
         */
        self_type& operator+=(const self_type& rhs)
        {
            if (m_size != rhs.m_size)
                throw std::runtime_error("variable::operator+=: size mismatch.");
            if constexpr (simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = m_size / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type a = simd_type::load_unaligned(m_data + i * simd_size);
                    simd_type b = simd_type::load_unaligned(rhs.m_data + i * simd_size);
                    (a + b).store_unaligned(m_data + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < m_size; ++i)
                    m_data[i] += rhs.m_data[i];
            }
            else
            {
                for (std::size_t i = 0; i < m_size; ++i)
                    m_data[i] += rhs.m_data[i];
            }
            return *this;
        }

        self_type& operator-=(const self_type& rhs)
        {
            if (m_size != rhs.m_size)
                throw std::runtime_error("variable::operator-=: size mismatch.");
            if constexpr (simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = m_size / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type a = simd_type::load_unaligned(m_data + i * simd_size);
                    simd_type b = simd_type::load_unaligned(rhs.m_data + i * simd_size);
                    (a - b).store_unaligned(m_data + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < m_size; ++i)
                    m_data[i] -= rhs.m_data[i];
            }
            return *this;
        }

        self_type& operator*=(const_reference scalar)
        {
            if constexpr (simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = m_size / simd_size;
                simd_type vs(scalar);
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type v = simd_type::load_unaligned(m_data + i * simd_size);
                    (v * vs).store_unaligned(m_data + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < m_size; ++i)
                    m_data[i] *= scalar;
            }
            else
            {
                for (std::size_t i = 0; i < m_size; ++i)
                    m_data[i] *= scalar;
            }
            return *this;
        }

        self_type& operator/=(const_reference scalar)
        {
            if constexpr (simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = m_size / simd_size;
                simd_type vs(scalar);
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type v = simd_type::load_unaligned(m_data + i * simd_size);
                    (v / vs).store_unaligned(m_data + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < m_size; ++i)
                    m_data[i] /= scalar;
            }
            return *this;
        }

    private:
        label_type m_name;
        size_type m_size;
        size_type m_capacity;
        pointer m_data = nullptr;

        static pointer allocate(size_type n)
        {
            void* ptr = nullptr;
            #if defined(_MSC_VER)
                ptr = _aligned_malloc(n * sizeof(T), simd_alignment);
            #else
                if (::posix_memalign(&ptr, simd_alignment, n * sizeof(T)) != 0)
                    ptr = nullptr;
            #endif
            if (!ptr) throw std::bad_alloc();
            return static_cast<pointer>(ptr);
        }

        static void deallocate_aligned(pointer p) noexcept
        {
            #if defined(_MSC_VER)
                _aligned_free(p);
            #else
                std::free(p);
            #endif
        }

        void deallocate() noexcept
        {
            if (m_data)
            {
                deallocate_aligned(m_data);
                m_data = nullptr;
            }
        }

        static size_type align_up(size_type n, size_type align) noexcept
        {
            return (n + align - 1) & ~(align - 1);
        }
    };

    // Free functions for variable arithmetic
    template <class T, class L>
    inline auto operator+(const variable<T, L>& a, const variable<T, L>& b)
    {
        variable<T, L> result(a.size(), a.name() + L("+") + b.name());
        const T* ad = a.data();
        const T* bd = b.data();
        T* rd = result.data();
        std::size_t n = a.size();
        if constexpr (simd_enabled_v<T>)
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            std::size_t vec_count = n / simd_size;
            for (std::size_t i = 0; i < vec_count; ++i)
            {
                simd_type va = simd_type::load_unaligned(ad + i * simd_size);
                simd_type vb = simd_type::load_unaligned(bd + i * simd_size);
                (va + vb).store_unaligned(rd + i * simd_size);
            }
            for (std::size_t i = vec_count * simd_size; i < n; ++i)
                rd[i] = ad[i] + bd[i];
        }
        else
        {
            for (std::size_t i = 0; i < n; ++i) rd[i] = ad[i] + bd[i];
        }
        return result;
    }

    template <class T, class L>
    inline auto operator-(const variable<T, L>& a, const variable<T, L>& b)
    {
        variable<T, L> result(a.size(), a.name() + L("-") + b.name());
        const T* ad = a.data();
        const T* bd = b.data();
        T* rd = result.data();
        std::size_t n = a.size();
        if constexpr (simd_enabled_v<T>)
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            std::size_t vec_count = n / simd_size;
            for (std::size_t i = 0; i < vec_count; ++i)
            {
                simd_type va = simd_type::load_unaligned(ad + i * simd_size);
                simd_type vb = simd_type::load_unaligned(bd + i * simd_size);
                (va - vb).store_unaligned(rd + i * simd_size);
            }
            for (std::size_t i = vec_count * simd_size; i < n; ++i)
                rd[i] = ad[i] - bd[i];
        }
        else
        {
            for (std::size_t i = 0; i < n; ++i) rd[i] = ad[i] - bd[i];
        }
        return result;
    }

    template <class T, class L>
    inline auto operator*(const variable<T, L>& a, const_reference<T> scalar)
    {
        variable<T, L> result(a.size(), a.name());
        const T* ad = a.data();
        T* rd = result.data();
        std::size_t n = a.size();
        if constexpr (simd_enabled_v<T>)
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            std::size_t vec_count = n / simd_size;
            simd_type vs(scalar);
            for (std::size_t i = 0; i < vec_count; ++i)
            {
                simd_type v = simd_type::load_unaligned(ad + i * simd_size);
                (v * vs).store_unaligned(rd + i * simd_size);
            }
            for (std::size_t i = vec_count * simd_size; i < n; ++i)
                rd[i] = ad[i] * scalar;
        }
        else
        {
            for (std::size_t i = 0; i < n; ++i) rd[i] = ad[i] * scalar;
        }
        return result;
    }

} // namespace xframe

#endif // XFRAME_XVARIABLE_HPP