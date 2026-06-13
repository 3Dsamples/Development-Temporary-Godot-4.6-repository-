//File 0105 : numdot/array.h
//Multidimensional array container with dynamic or fixed rank, SIMD-aligned storage, expression integration, slices, views, and broadcasting.
#ifndef NUMDOT_ARRAY_H
#define NUMDOT_ARRAY_H

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

#include "config.h"
#include "forward.h"
#include "types.h"
#include "shape.h"

namespace numdot
{
    template <class T, std::size_t N = dynamic_rank>
    class array : public expression<array<T, N>>
    {
    public:
        using self_type = array<T, N>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::conditional_t<N == dynamic_rank, std::vector<size_type>, std::array<size_type, N>>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using iterator = pointer;
        using const_iterator = const_pointer;
        static constexpr layout layout = default_layout;
        static constexpr std::size_t rank_value = N;

        // Default constructor
        array() noexcept : m_storage() {}

        // Construct with shape
        explicit array(const shape_type& shape)
            : m_storage(compute_size(shape))
            , m_shape(shape)
        {
            compute_strides();
        }

        // Construct with shape and initial value
        array(const shape_type& shape, const_reference val)
            : m_storage(compute_size(shape), val)
            , m_shape(shape)
        {
            compute_strides();
        }

        // Construct from initializer list (flat)
        array(std::initializer_list<T> init)
            : m_storage(init)
            , m_shape({init.size()})
        {
            compute_strides();
        }

        // Copy constructor
        array(const self_type&) = default;
        array& operator=(const self_type&) = default;

        // Move constructor
        array(self_type&&) noexcept = default;
        array& operator=(self_type&&) noexcept = default;

        // Expression assignment
        template <class E, disable_if_expression_t<E, int> = 0>
        array& operator=(const expression<E>& e)
        {
            const auto& src = e.derived();
            if (this == &src) return *this;
            m_shape = src.shape();
            m_storage.resize(compute_size(m_shape));
            compute_strides();
            // Copy elements (could use SIMD)
            auto* dst = m_storage.data();
            const auto* src_data = src.data();
            if (src_data)
            {
                std::copy(src_data, src_data + size(), dst);
            }
            else
            {
                for (size_type i = 0; i < size(); ++i)
                    dst[i] = src[i];
            }
            return *this;
        }

        // Assignment from a value (fill)
        array& operator=(const_reference val)
        {
            std::fill(m_storage.begin(), m_storage.end(), val);
            return *this;
        }

        // Size
        size_type size() const noexcept { return m_storage.size(); }
        size_type dimension() const noexcept { return m_shape.size(); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        void set_shape(const shape_type& s) { m_shape = s; compute_strides(); }
        void set_strides(const strides_type& st) { m_strides = st; m_backstrides = compute_backstrides(st, m_shape); }

        // Element access
        template <class... Args>
        reference operator()(Args... args)
        {
            return element_impl(std::index_sequence_for<Args...>{}, args...);
        }

        template <class... Args>
        const_reference operator()(Args... args) const
        {
            return element_impl(std::index_sequence_for<Args...>{}, args...);
        }

        reference operator[](size_type i) { return m_storage[i]; }
        const_reference operator[](size_type i) const { return m_storage[i]; }

        reference at(size_type i) { if (i >= size()) throw std::out_of_range("array::at"); return m_storage[i]; }
        const_reference at(size_type i) const { if (i >= size()) throw std::out_of_range("array::at"); return m_storage[i]; }

        pointer data() noexcept { return m_storage.data(); }
        const_pointer data() const noexcept { return m_storage.data(); }

        // Iterators
        iterator begin() noexcept { return m_storage.data(); }
        iterator end() noexcept { return m_storage.data() + size(); }
        const_iterator begin() const noexcept { return m_storage.data(); }
        const_iterator end() const noexcept { return m_storage.data() + size(); }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

        // Reshape (only if same total size)
        void reshape(const shape_type& new_shape)
        {
            if (compute_size(new_shape) != size())
                throw std::runtime_error("Reshape cannot change total size.");
            m_shape = new_shape;
            compute_strides();
        }

        // Resize (reallocate if needed)
        void resize(const shape_type& new_shape)
        {
            m_shape = new_shape;
            std::size_t new_size = compute_size(new_shape);
            m_storage.resize(new_size);
            compute_strides();
        }

        // SIMD load
        template <class Align, class U = T>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<U, default_simd_arch>;
            return simd_type::load_unaligned(data() + i);
        }

        // Get storage reference
        uvector<T>& storage() noexcept { return m_storage; }
        const uvector<T>& storage() const noexcept { return m_storage; }

    private:
        uvector<T> m_storage;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;

        void compute_strides()
        {
            m_strides = numdot::compute_strides(m_shape);
            m_backstrides = compute_backstrides(m_strides, m_shape);
        }

        template <std::size_t... I, class... Args>
        reference element_impl(std::index_sequence<I...>, Args... args)
        {
            size_type idx = ravel_index(shape_type{static_cast<size_type>(args)...}, m_strides);
            return m_storage[idx];
        }

        template <std::size_t... I, class... Args>
        const_reference element_impl(std::index_sequence<I...>, Args... args) const
        {
            size_type idx = ravel_index(shape_type{static_cast<size_type>(args)...}, m_strides);
            return m_storage[idx];
        }
    };

    // Explicit deduction guide for array from initializer_list
    template <class T>
    array(std::initializer_list<T>) -> array<T, dynamic_rank>;

    // array_adaptor: wraps external memory with optional ownership
    template <class T>
    class array_adaptor : public expression<array_adaptor<T>>
    {
    public:
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using shape_type = std::vector<size_type>;
        using strides_type = shape_type;
        static constexpr layout layout = default_layout;

        array_adaptor() noexcept : m_data(nullptr), m_size(0) {}

        array_adaptor(pointer data, const shape_type& shape, bool own = false)
            : m_data(data), m_shape(shape), m_size(compute_size(shape)), m_owns(own ? std::make_unique<bool>(true) : nullptr)
        {
            compute_strides();
        }

        array_adaptor(const array_adaptor&) = delete;
        array_adaptor& operator=(const array_adaptor&) = delete;
        array_adaptor(array_adaptor&& other) noexcept
            : m_data(other.m_data), m_shape(std::move(other.m_shape)), m_size(other.m_size), m_owns(std::move(other.m_owns))
        {
            other.m_data = nullptr; other.m_size = 0;
        }
        array_adaptor& operator=(array_adaptor&& other) noexcept
        {
            if (this != &other)
            {
                if (m_owns && m_data) delete[] m_data;
                m_data = other.m_data; m_shape = std::move(other.m_shape); m_size = other.m_size; m_owns = std::move(other.m_owns);
                other.m_data = nullptr; other.m_size = 0;
            }
            return *this;
        }
        ~array_adaptor() { if (m_owns && m_data) delete[] m_data; }

        size_type size() const noexcept { return m_size; }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }

        pointer data() noexcept { return m_data; }
        const_pointer data() const noexcept { return m_data; }

        template <class... Args>
        reference operator()(Args... args)
        {
            return m_data[ravel_index(shape_type{static_cast<size_type>(args)...}, m_strides)];
        }
        template <class... Args>
        const_reference operator()(Args... args) const
        {
            return m_data[ravel_index(shape_type{static_cast<size_type>(args)...}, m_strides)];
        }

        reference operator[](size_type i) { return m_data[i]; }
        const_reference operator[](size_type i) const { return m_data[i]; }

    private:
        pointer m_data;
        shape_type m_shape;
        strides_type m_strides;
        size_type m_size;
        std::unique_ptr<bool> m_owns;

        void compute_strides()
        {
            m_strides = numdot::compute_strides(m_shape);
        }
    };

    // Convenience alias for fixed rank
    template <class T, std::size_t N>
    using fixed_array = array<T, N>;

} // namespace numdot

#endif // NUMDOT_ARRAY_H