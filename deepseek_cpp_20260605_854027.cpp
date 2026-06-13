//File 0004 : core/xsemantic.hpp
//Semantic base classes enabling temporary type resolution and assignment for containers and views.
#ifndef XTENSOR_XSEMANTIC_HPP
#define XTENSOR_XSEMANTIC_HPP

#include <type_traits>
#include <utility>

#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    /**
     * @class xcontainer_semantic
     * @brief Provides assignment semantics for container types.
     *
     * Enables assignment from any xtensor expression by evaluating it into a temporary
     * of the container's own type and then move-assigning it. This ensures lazy evaluation
     * and preserves value semantics.
     */
    template <class D>
    class xcontainer_semantic : public xexpression<D>
    {
    public:

        using derived_type = D;
        using temporary_type = typename xcontainer_inner_types<D>::temporary_type;

        /**
         * Assign the result of an expression to the container.
         */
        template <class E>
        disable_xexpression<E, derived_type&> operator=(const E& e)
        {
            temporary_type tmp(e);
            derived_cast() = std::move(tmp);
            return derived_cast();
        }

        /**
         * Copy assignment from another container of the same type.
         */
        derived_type& operator=(const derived_type& rhs)
        {
            if (this != &rhs)
            {
                derived_cast().data().resize(rhs.size());
                std::copy(rhs.data().begin(), rhs.data().end(), derived_cast().data().begin());
                derived_cast().set_shape(rhs.shape());
            }
            return derived_cast();
        }

        /**
         * Move assignment from another container of the same type.
         */
        derived_type& operator=(derived_type&& rhs) noexcept
        {
            if (this != &rhs)
            {
                derived_cast().data() = std::move(rhs.data());
                derived_cast().set_shape(rhs.shape());
            }
            return derived_cast();
        }

        /**
         * Compound assignment operators: delegate to expression evaluation.
         */
        template <class E>
        disable_xexpression<E, derived_type&> operator+=(const E& e)
        {
            return derived_cast() = derived_cast() + e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator-=(const E& e)
        {
            return derived_cast() = derived_cast() - e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator*=(const E& e)
        {
            return derived_cast() = derived_cast() * e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator/=(const E& e)
        {
            return derived_cast() = derived_cast() / e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator%=(const E& e)
        {
            return derived_cast() = derived_cast() % e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator&=(const E& e)
        {
            return derived_cast() = derived_cast() & e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator|=(const E& e)
        {
            return derived_cast() = derived_cast() | e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator^=(const E& e)
        {
            return derived_cast() = derived_cast() ^ e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator<<=(const E& e)
        {
            return derived_cast() = derived_cast() << e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator>>=(const E& e)
        {
            return derived_cast() = derived_cast() >> e;
        }

    protected:

        xcontainer_semantic() = default;
        ~xcontainer_semantic() = default;

        xcontainer_semantic(const xcontainer_semantic&) = default;
        xcontainer_semantic& operator=(const xcontainer_semantic&) = default;

        xcontainer_semantic(xcontainer_semantic&&) = default;
        xcontainer_semantic& operator=(xcontainer_semantic&&) = default;

        derived_type& derived_cast() noexcept
        {
            return *static_cast<derived_type*>(this);
        }

        const derived_type& derived_cast() const noexcept
        {
            return *static_cast<const derived_type*>(this);
        }
    };

    /**
     * @class xview_semantic
     * @brief Provides assignment semantics for view types.
     *
     * Views do not own data; assignment forwards to the underlying container or evaluates
     * into a temporary and copies elementwise.
     */
    template <class D>
    class xview_semantic : public xexpression<D>
    {
    public:

        using derived_type = D;
        using temporary_type = typename xcontainer_inner_types<D>::temporary_type;

        /**
         * Assign an expression to the view, performing element-wise copy.
         */
        template <class E>
        disable_xexpression<E, derived_type&> operator=(const E& e)
        {
            temporary_type tmp(e);
            derived_cast().assign_temporary(tmp);
            return derived_cast();
        }

        /**
         * Copy assignment from another view of the same type (if it points to the same container).
         */
        derived_type& operator=(const derived_type& rhs)
        {
            if (this != &rhs)
            {
                derived_cast().assign_temporary(rhs);
            }
            return derived_cast();
        }

        /**
         * Move assignment from another view of the same type.
         */
        derived_type& operator=(derived_type&& rhs) noexcept
        {
            if (this != &rhs)
            {
                derived_cast().assign_temporary(std::move(rhs));
            }
            return derived_cast();
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator+=(const E& e)
        {
            return derived_cast() = derived_cast() + e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator-=(const E& e)
        {
            return derived_cast() = derived_cast() - e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator*=(const E& e)
        {
            return derived_cast() = derived_cast() * e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator/=(const E& e)
        {
            return derived_cast() = derived_cast() / e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator%=(const E& e)
        {
            return derived_cast() = derived_cast() % e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator&=(const E& e)
        {
            return derived_cast() = derived_cast() & e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator|=(const E& e)
        {
            return derived_cast() = derived_cast() | e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator^=(const E& e)
        {
            return derived_cast() = derived_cast() ^ e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator<<=(const E& e)
        {
            return derived_cast() = derived_cast() << e;
        }

        template <class E>
        disable_xexpression<E, derived_type&> operator>>=(const E& e)
        {
            return derived_cast() = derived_cast() >> e;
        }

    protected:

        xview_semantic() = default;
        ~xview_semantic() = default;

        xview_semantic(const xview_semantic&) = default;
        xview_semantic& operator=(const xview_semantic&) = default;

        xview_semantic(xview_semantic&&) = default;
        xview_semantic& operator=(xview_semantic&&) = default;

        derived_type& derived_cast() noexcept
        {
            return *static_cast<derived_type*>(this);
        }

        const derived_type& derived_cast() const noexcept
        {
            return *static_cast<const derived_type*>(this);
        }
    };

}  // namespace xt

#endif  // XTENSOR_XSEMANTIC_HPP