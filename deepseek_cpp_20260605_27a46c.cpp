//File 0018 : core/xexpression.hpp
//Base class for all expressions via CRTP; provides shape, size, operator(), and element access interface.
#ifndef XTENSOR_XEXPRESSION_HPP
#define XTENSOR_XEXPRESSION_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    /**
     * @class xexpression
     * @brief CRTP base class for all tensor expressions.
     *
     * Defines the common interface that all expressions must provide, relying on
     * the derived class to implement shape(), element(), etc.
     */
    template <class D>
    class xexpression
    {
    public:
        using derived_type = D;
        using inner_types = xcontainer_inner_types<D>;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename inner_types::pointer;
        using const_pointer = typename inner_types::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename inner_types::difference_type;
        using shape_type = typename inner_types::shape_type;
        using strides_type = typename inner_types::strides_type;
        using backstrides_type = typename inner_types::backstrides_type;

        // CRTP derived cast
        derived_type& derived_cast() noexcept
        {
            return *static_cast<derived_type*>(this);
        }
        const derived_type& derived_cast() const noexcept
        {
            return *static_cast<const derived_type*>(this);
        }

        // Size and shape
        size_type size() const noexcept
        {
            return derived_cast().size();
        }
        size_type dimension() const noexcept
        {
            return derived_cast().shape().size();
        }
        shape_type shape() const noexcept
        {
            return derived_cast().shape();
        }
        layout_type layout() const noexcept
        {
            return inner_types::layout;
        }

        // Access operators
        template <class... Args>
        reference operator()(Args... args)
        {
            return derived_cast()(args...);
        }
        template <class... Args>
        const_reference operator()(Args... args) const
        {
            return derived_cast()(args...);
        }

        reference operator[](size_type i)
        {
            return derived_cast()[i];
        }
        const_reference operator[](size_type i) const
        {
            return derived_cast()[i];
        }

        template <class... Args>
        reference at(Args... args)
        {
            return derived_cast().at(args...);
        }
        template <class... Args>
        const_reference at(Args... args) const
        {
            return derived_cast().at(args...);
        }

        template <class... Args>
        reference periodic(Args... args)
        {
            return derived_cast().periodic(args...);
        }
        template <class... Args>
        const_reference periodic(Args... args) const
        {
            return derived_cast().periodic(args...);
        }

        // Element access via iterator pair
        template <class It>
        reference element(It first, It last)
        {
            return derived_cast().element(first, last);
        }
        template <class It>
        const_reference element(It first, It last) const
        {
            return derived_cast().element(first, last);
        }

        // Data access (if contiguous, else may throw)
        pointer data() noexcept
        {
            return derived_cast().data();
        }
        const_pointer data() const noexcept
        {
            return derived_cast().data();
        }

        // Iterator interface (if applicable)
        using iterator = pointer;
        using const_iterator = const_pointer;
        iterator begin() noexcept { return data(); }
        iterator end() noexcept { return data() + size(); }
        const_iterator begin() const noexcept { return data(); }
        const_iterator end() const noexcept { return data() + size(); }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

        // Expression should also support broadcasting, etc.
        template <class S>
        bool broadcast_shape(S& s, bool reuse_cache = false) const
        {
            return derived_cast().broadcast_shape(s, reuse_cache);
        }

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept
        {
            return derived_cast().has_linear_assign(strides);
        }

    protected:
        xexpression() = default;
        ~xexpression() = default;
        xexpression(const xexpression&) = default;
        xexpression& operator=(const xexpression&) = default;
        xexpression(xexpression&&) = default;
        xexpression& operator=(xexpression&&) = default;
    };

} // namespace xt

#endif // XTENSOR_XEXPRESSION_HPP