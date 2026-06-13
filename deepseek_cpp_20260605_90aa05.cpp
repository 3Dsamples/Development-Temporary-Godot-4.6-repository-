//File 0039 : core/xscalar.hpp
//Scalar constant expression with broadcast promotion, SIMD loading, and full integration into expression system.
#ifndef XTENSOR_XSCALAR_HPP
#define XTENSOR_XSCALAR_HPP

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xstrides.hpp"

namespace xt
{
    /*******************************
     * xscalar – constant scalar value
     *******************************/
    template <class T>
    class xscalar : public xexpression<xscalar<T>>
    {
    public:
        using self_type = xscalar<T>;
        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::array<size_type, 0>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;

        xscalar() noexcept = default;
        explicit xscalar(const T& value) noexcept : m_value(value) {}
        explicit xscalar(T&& value) noexcept : m_value(std::move(value)) {}

        xscalar(const self_type&) = default;
        xscalar& operator=(const self_type&) = default;
        xscalar(self_type&&) = default;
        xscalar& operator=(self_type&&) = default;

        size_type size() const noexcept { return 1; }
        shape_type shape() const noexcept { return shape_type{}; }

        const_reference operator()() const noexcept { return m_value; }
        reference operator()() noexcept { return m_value; }

        const_reference operator[](size_type) const noexcept { return m_value; }
        reference operator[](size_type) noexcept { return m_value; }

        template <class... Args>
        const_reference operator()(Args...) const noexcept { return m_value; }

        template <class... Args>
        reference operator()(Args...) noexcept { return m_value; }

        template <class It>
        const_reference element(It, It) const noexcept { return m_value; }

        template <class It>
        reference element(It, It) noexcept { return m_value; }

        pointer data() noexcept { return &m_value; }
        const_pointer data() const noexcept { return &m_value; }

        using iterator = pointer;
        using const_iterator = const_pointer;
        iterator begin() noexcept { return data(); }
        iterator end() noexcept { return data() + 1; }
        const_iterator begin() const noexcept { return data(); }
        const_iterator end() const noexcept { return data() + 1; }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

        template <class S>
        bool broadcast_shape(S& s, bool = false) const
        {
            s = m_shape;
            return true;
        }

        template <class S>
        bool has_linear_assign(const S&) const noexcept { return true; }

    private:
        T m_value;
    };

    /*******************************
     * xcontainer_inner_types for xscalar
     *******************************/
    template <class T>
    struct xcontainer_inner_types<xscalar<T>>
    {
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::array<size_type, 0>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<uvector<value_type>, DEFAULT_LAYOUT, std::vector<size_type>>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

} // namespace xt

#endif // XTENSOR_XSCALAR_HPP