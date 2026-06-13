//File 0004 (UPDATED) : core/xsemantic.hpp
//Full semantic base class hierarchy: xsemantic_base for shared semantics, is_sharable, xsharable_expression, scalar_computed_assign, and all bitwise compound operators.
#ifndef XTENSOR_XSEMANTIC_HPP
#define XTENSOR_XSEMANTIC_HPP

#include <type_traits>
#include <utility>

#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    /****************************
     * xsemantic_base – common assign logic for containers and views
     ****************************/
    template <class D>
    class xsemantic_base : public xexpression<D>
    {
    public:
        using derived_type = D;
        using temporary_type = typename xcontainer_inner_types<D>::temporary_type;

        template <class E>
        disable_xexpression<E, derived_type&> operator=(const E& e);

        template <class E>
        disable_xexpression<E, derived_type&> assign(const E& e);

        template <class E>
        disable_xexpression<E, derived_type&> plus_assign(const E& e);

        template <class E>
        disable_xexpression<E, derived_type&> minus_assign(const E& e);

        template <class E>
        disable_xexpression<E, derived_type&> multiplies_assign(const E& e);

        template <class E>
        disable_xexpression<E, derived_type&> divides_assign(const E& e);

        template <class E>
        disable_xexpression<E, derived_type&> modulus_assign(const E& e);

        template <class E>
        disable_xexpression<E, derived_type&> bitwise_and_assign(const E& e);

        template <class E>
        disable_xexpression<E, derived_type&> bitwise_or_assign(const E& e);

        template <class E>
        disable_xexpression<E, derived_type&> bitwise_xor_assign(const E& e);

        template <class E>
        disable_xexpression<E, derived_type&> left_shift_assign(const E& e);

        template <class E>
        disable_xexpression<E, derived_type&> right_shift_assign(const E& e);

    protected:
        xsemantic_base() = default;
        ~xsemantic_base() = default;
        xsemantic_base(const xsemantic_base&) = default;
        xsemantic_base& operator=(const xsemantic_base&) = default;
        xsemantic_base(xsemantic_base&&) = default;
        xsemantic_base& operator=(xsemantic_base&&) = default;

        derived_type& derived_cast() noexcept { return *static_cast<derived_type*>(this); }
        const derived_type& derived_cast() const noexcept { return *static_cast<const derived_type*>(this); }
    };

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        temporary_type tmp(e);
        return derived_cast() = std::move(tmp);
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::assign(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return operator=(e);
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::plus_assign(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return derived_cast() = derived_cast() + e;
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::minus_assign(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return derived_cast() = derived_cast() - e;
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::multiplies_assign(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return derived_cast() = derived_cast() * e;
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::divides_assign(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return derived_cast() = derived_cast() / e;
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::modulus_assign(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return derived_cast() = derived_cast() % e;
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::bitwise_and_assign(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return derived_cast() = derived_cast() & e;
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::bitwise_or_assign(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return derived_cast() = derived_cast() | e;
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::bitwise_xor_assign(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return derived_cast() = derived_cast() ^ e;
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::left_shift_assign(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return derived_cast() = derived_cast() << e;
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::right_shift_assign(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return derived_cast() = derived_cast() >> e;
    }

    /****************************
     * scalar_computed_assign – for compound assignment of scalars
     ****************************/
    template <class D, class T>
    class scalar_computed_assign
    {
    public:
        using derived_type = D;
        derived_type& operator+=(const T& scalar);
        derived_type& operator-=(const T& scalar);
        derived_type& operator*=(const T& scalar);
        derived_type& operator/=(const T& scalar);
        derived_type& operator%=(const T& scalar);
        derived_type& operator&=(const T& scalar);
        derived_type& operator|=(const T& scalar);
        derived_type& operator^=(const T& scalar);
        derived_type& operator<<=(const T& scalar);
        derived_type& operator>>=(const T& scalar);

    protected:
        scalar_computed_assign() = default;
        ~scalar_computed_assign() = default;
        scalar_computed_assign(const scalar_computed_assign&) = default;
        scalar_computed_assign& operator=(const scalar_computed_assign&) = default;
        scalar_computed_assign(scalar_computed_assign&&) = default;
        scalar_computed_assign& operator=(scalar_computed_assign&&) = default;

        derived_type& derived_cast() noexcept { return *static_cast<derived_type*>(this); }
        const derived_type& derived_cast() const noexcept { return *static_cast<const derived_type*>(this); }
    };

    template <class D, class T>
    inline auto scalar_computed_assign<D, T>::operator+=(const T& scalar) -> derived_type&
    {
        return derived_cast() = derived_cast() + scalar;
    }
    template <class D, class T>
    inline auto scalar_computed_assign<D, T>::operator-=(const T& scalar) -> derived_type&
    {
        return derived_cast() = derived_cast() - scalar;
    }
    template <class D, class T>
    inline auto scalar_computed_assign<D, T>::operator*=(const T& scalar) -> derived_type&
    {
        return derived_cast() = derived_cast() * scalar;
    }
    template <class D, class T>
    inline auto scalar_computed_assign<D, T>::operator/=(const T& scalar) -> derived_type&
    {
        return derived_cast() = derived_cast() / scalar;
    }
    template <class D, class T>
    inline auto scalar_computed_assign<D, T>::operator%=(const T& scalar) -> derived_type&
    {
        return derived_cast() = derived_cast() % scalar;
    }
    template <class D, class T>
    inline auto scalar_computed_assign<D, T>::operator&=(const T& scalar) -> derived_type&
    {
        return derived_cast() = derived_cast() & scalar;
    }
    template <class D, class T>
    inline auto scalar_computed_assign<D, T>::operator|=(const T& scalar) -> derived_type&
    {
        return derived_cast() = derived_cast() | scalar;
    }
    template <class D, class T>
    inline auto scalar_computed_assign<D, T>::operator^=(const T& scalar) -> derived_type&
    {
        return derived_cast() = derived_cast() ^ scalar;
    }
    template <class D, class T>
    inline auto scalar_computed_assign<D, T>::operator<<=(const T& scalar) -> derived_type&
    {
        return derived_cast() = derived_cast() << scalar;
    }
    template <class D, class T>
    inline auto scalar_computed_assign<D, T>::operator>>=(const T& scalar) -> derived_type&
    {
        return derived_cast() = derived_cast() >> scalar;
    }

    /****************************
     * is_sharable traits and xsharable_expression
     ****************************/
    template <class E>
    struct is_sharable : std::true_type {};

    template <class T>
    struct is_sharable<xscalar<T>> : std::false_type {};

    template <class F, class... CT>
    struct is_sharable<xfunction<F, CT...>> : std::false_type {};

    template <class D>
    class xsharable_expression : public xsemantic_base<D>
    {
    public:
        using derived_type = D;
        using temporary_type = typename xcontainer_inner_types<D>::temporary_type;

        template <class E>
        disable_xexpression<E, derived_type&> shared_assign(const E& e);

    protected:
        xsharable_expression() = default;
        ~xsharable_expression() = default;
        xsharable_expression(const xsharable_expression&) = default;
        xsharable_expression& operator=(const xsharable_expression&) = default;
        xsharable_expression(xsharable_expression&&) = default;
        xsharable_expression& operator=(xsharable_expression&&) = default;
    };

    template <class D>
    template <class E>
    inline auto xsharable_expression<D>::shared_assign(const E& e) -> disable_xexpression<E, derived_type&>
    {
        if constexpr (is_sharable<E>::value)
        {
            derived_cast().resize(e.shape());
            // share data? For simplicity, copy elements
            std::copy(e.begin(), e.end(), derived_cast().begin());
            return derived_cast();
        }
        else
        {
            return this->operator=(e);
        }
    }

    /****************************
     * xcontainer_semantic and xview_semantic (inherit from xsemantic_base)
     ****************************/
    template <class D>
    class xcontainer_semantic : public xsharable_expression<D>,
                                public scalar_computed_assign<D, typename D::value_type>
    {
    public:
        using derived_type = D;
        using temporary_type = typename xcontainer_inner_types<D>::temporary_type;

        template <class E>
        disable_xexpression<E, derived_type&> operator=(const E& e);

        derived_type& operator=(const derived_type& rhs);
        derived_type& operator=(derived_type&& rhs) noexcept;

        template <class E>
        disable_xexpression<E, derived_type&> operator+=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator-=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator*=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator/=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator%=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator&=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator|=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator^=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator<<=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator>>=(const E& e);

    protected:
        xcontainer_semantic() = default;
        ~xcontainer_semantic() = default;
        xcontainer_semantic(const xcontainer_semantic&) = default;
        xcontainer_semantic& operator=(const xcontainer_semantic&) = default;
        xcontainer_semantic(xcontainer_semantic&&) = default;
        xcontainer_semantic& operator=(xcontainer_semantic&&) = default;

        derived_type& derived_cast() noexcept { return *static_cast<derived_type*>(this); }
        const derived_type& derived_cast() const noexcept { return *static_cast<const derived_type*>(this); }
    };

    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::operator=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        temporary_type tmp(e);
        return derived_cast() = std::move(tmp);
    }

    template <class D>
    inline auto xcontainer_semantic<D>::operator=(const derived_type& rhs) -> derived_type&
    {
        if (this != &rhs)
        {
            derived_cast().data().resize(rhs.size());
            std::copy(rhs.data().begin(), rhs.data().end(), derived_cast().data().begin());
            derived_cast().set_shape(rhs.shape());
        }
        return derived_cast();
    }

    template <class D>
    inline auto xcontainer_semantic<D>::operator=(derived_type&& rhs) noexcept -> derived_type&
    {
        if (this != &rhs)
        {
            derived_cast().data() = std::move(rhs.data());
            derived_cast().set_shape(rhs.shape());
        }
        return derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::operator+=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() + e; }
    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::operator-=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() - e; }
    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::operator*=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() * e; }
    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::operator/=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() / e; }
    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::operator%=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() % e; }
    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::operator&=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() & e; }
    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::operator|=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() | e; }
    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::operator^=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() ^ e; }
    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::operator<<=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() << e; }
    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::operator>>=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() >> e; }

    template <class D>
    class xview_semantic : public xsharable_expression<D>,
                           public scalar_computed_assign<D, typename D::value_type>
    {
    public:
        using derived_type = D;
        using temporary_type = typename xcontainer_inner_types<D>::temporary_type;

        template <class E>
        disable_xexpression<E, derived_type&> operator=(const E& e);

        derived_type& operator=(const derived_type& rhs);
        derived_type& operator=(derived_type&& rhs) noexcept;

        template <class E>
        disable_xexpression<E, derived_type&> operator+=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator-=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator*=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator/=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator%=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator&=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator|=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator^=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator<<=(const E& e);
        template <class E>
        disable_xexpression<E, derived_type&> operator>>=(const E& e);

    protected:
        xview_semantic() = default;
        ~xview_semantic() = default;
        xview_semantic(const xview_semantic&) = default;
        xview_semantic& operator=(const xview_semantic&) = default;
        xview_semantic(xview_semantic&&) = default;
        xview_semantic& operator=(xview_semantic&&) = default;

        derived_type& derived_cast() noexcept { return *static_cast<derived_type*>(this); }
        const derived_type& derived_cast() const noexcept { return *static_cast<const derived_type*>(this); }
    };

    template <class D>
    template <class E>
    inline auto xview_semantic<D>::operator=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        temporary_type tmp(e);
        derived_cast().assign_temporary(tmp);
        return derived_cast();
    }

    template <class D>
    inline auto xview_semantic<D>::operator=(const derived_type& rhs) -> derived_type&
    {
        if (this != &rhs) { derived_cast().assign_temporary(rhs); }
        return derived_cast();
    }

    template <class D>
    inline auto xview_semantic<D>::operator=(derived_type&& rhs) noexcept -> derived_type&
    {
        if (this != &rhs) { derived_cast().assign_temporary(std::move(rhs)); }
        return derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xview_semantic<D>::operator+=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() + e; }
    template <class D>
    template <class E>
    inline auto xview_semantic<D>::operator-=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() - e; }
    template <class D>
    template <class E>
    inline auto xview_semantic<D>::operator*=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() * e; }
    template <class D>
    template <class E>
    inline auto xview_semantic<D>::operator/=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() / e; }
    template <class D>
    template <class E>
    inline auto xview_semantic<D>::operator%=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() % e; }
    template <class D>
    template <class E>
    inline auto xview_semantic<D>::operator&=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() & e; }
    template <class D>
    template <class E>
    inline auto xview_semantic<D>::operator|=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() | e; }
    template <class D>
    template <class E>
    inline auto xview_semantic<D>::operator^=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() ^ e; }
    template <class D>
    template <class E>
    inline auto xview_semantic<D>::operator<<=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() << e; }
    template <class D>
    template <class E>
    inline auto xview_semantic<D>::operator>>=(const E& e) -> disable_xexpression<E, derived_type&>
    { return derived_cast() = derived_cast() >> e; }

} // namespace xt

#endif // XTENSOR_XSEMANTIC_HPP