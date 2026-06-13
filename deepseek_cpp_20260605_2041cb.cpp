//File 0044 : core/xaccessible.hpp
//Accessible CRTP mixin providing bounds-checked, periodic, and unchecked element access with full in_bounds and index validation.
#ifndef XTENSOR_XACCESSIBLE_HPP
#define XTENSOR_XACCESSIBLE_HPP

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    /**
     * @class xaccessible
     * @brief CRTP mixin that provides full element access interface.
     *
     * Provides operator(), operator[], at(), periodic(), unchecked(), back(),
     * front(), and in_bounds() for any derived expression that implements
     * element() and shape().
     */
    template <class D>
    class xaccessible
    {
    public:
        using derived_type = D;
        using inner_types = xcontainer_inner_types<D>;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using size_type = typename inner_types::size_type;
        using shape_type = typename inner_types::shape_type;

        /**
         * Multi-dimensional element access via variadic indices.
         */
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

        /**
         * Linear flat index access.
         */
        reference operator[](size_type i)
        {
            return derived_cast()[i];
        }

        const_reference operator[](size_type i) const
        {
            return derived_cast()[i];
        }

        /**
         * Bounds-checked access via at().
         */
        template <class... Args>
        reference at(Args... args)
        {
            check_bounds(args...);
            return derived_cast()(args...);
        }

        template <class... Args>
        const_reference at(Args... args) const
        {
            check_bounds(args...);
            return derived_cast()(args...);
        }

        /**
         * Element access with periodic boundary conditions.
         */
        template <class... Args>
        reference periodic(Args... args)
        {
            adjust_periodic(args...);
            return derived_cast()(args...);
        }

        template <class... Args>
        const_reference periodic(Args... args) const
        {
            adjust_periodic(args...);
            return derived_cast()(args...);
        }

        /**
         * Element access without bounds checking (unchecked).
         */
        template <class... Args>
        reference unchecked(Args... args)
        {
            return derived_cast()(args...);
        }

        template <class... Args>
        const_reference unchecked(Args... args) const
        {
            return derived_cast()(args...);
        }

        /**
         * Check whether all indices are within bounds.
         */
        template <class... Args>
        bool in_bounds(Args... args) const
        {
            auto idx = std::array<size_type, sizeof...(Args)>{static_cast<size_type>(args)...};
            const auto& sh = derived_cast().shape();
            if (idx.size() != sh.size())
            {
                return false;
            }
            for (std::size_t i = 0; i < sh.size(); ++i)
            {
                if (idx[i] >= sh[i])
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * Access the first element (linear index 0).
         */
        reference front()
        {
            return derived_cast().data()[0];
        }

        const_reference front() const
        {
            return derived_cast().data()[0];
        }

        /**
         * Access the last element (linear index size()-1).
         */
        reference back()
        {
            return derived_cast().data()[derived_cast().size() - 1];
        }

        const_reference back() const
        {
            return derived_cast().data()[derived_cast().size() - 1];
        }

        /**
         * Return a flat pointer to the first element (linear access).
         */
        reference flat(size_type i)
        {
            return derived_cast().data()[i];
        }

        const_reference flat(size_type i) const
        {
            return derived_cast().data()[i];
        }

    protected:
        xaccessible() = default;
        ~xaccessible() = default;
        xaccessible(const xaccessible&) = default;
        xaccessible& operator=(const xaccessible&) = default;
        xaccessible(xaccessible&&) = default;
        xaccessible& operator=(xaccessible&&) = default;

        derived_type& derived_cast() noexcept
        {
            return *static_cast<derived_type*>(this);
        }

        const derived_type& derived_cast() const noexcept
        {
            return *static_cast<const derived_type*>(this);
        }

    private:
        /**
         * Throws std::out_of_range if any index exceeds the corresponding dimension.
         */
        template <class... Args>
        void check_bounds(Args... args) const
        {
            if (!in_bounds(args...))
            {
                throw std::out_of_range("Index out of bounds in xaccessible::at()");
            }
        }

        /**
         * Applys periodic wrapping: negative indices wrap around, indices >= size wrap around.
         */
        template <class... Args>
        void adjust_periodic(Args&... args) const
        {
            auto idx = std::array<size_type, sizeof...(Args)>{static_cast<size_type>(args)...};
            const auto& sh = derived_cast().shape();
            for (std::size_t i = 0; i < sizeof...(Args); ++i)
            {
                if (sh[i] > 0)
                {
                    idx[i] = ((idx[i] % sh[i]) + sh[i]) % sh[i];
                }
            }
            std::size_t pos = 0;
            ((args = idx[pos++]), ...);
        }
    };

} // namespace xt

#endif // XTENSOR_XACCESSIBLE_HPP