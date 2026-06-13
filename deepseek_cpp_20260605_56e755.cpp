//File 0003 : core/xfunction.hpp
//Expression engine: lazy evaluation, broadcasting, SIMD-accelerated functor application.
#ifndef XTENSOR_XFUNCTION_HPP
#define XTENSOR_XFUNCTION_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include <xtl/xclosure.hpp>
#include <xtl/xsequence.hpp>
#include <xtl/xtype_traits.hpp>

#include "xexpression.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xtensor_simd.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    /*******************************
     * xfunction declaration
     *******************************/
    template <class F, class... CT>
    class xfunction;

    /*******************************
     * xfunction_type_traits
     *******************************/
    namespace detail
    {
        template <class F, class... CT>
        struct xfunction_type_traits
        {
            using type = xfunction<F, CT...>;
            using simd_argument_type = xt_simd::xsimd_default_value_type<type>;
            using simd_return_type = std::decay_t<decltype(std::declval<F>().simd_apply(
                std::declval<simd_argument_type>(), std::declval<simd_argument_type>()))>;
            using value_type = decltype(std::declval<F>()(
                std::declval<typename std::decay_t<CT>::value_type>()...));
            using reference = std::remove_reference_t<value_type>&;
            using const_reference = const value_type&;
            using pointer = value_type*;
            using const_pointer = const value_type*;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using shape_type = xshape<value_type>;

            using inner_batch_type = xsimd::batch<value_type, 4>;
            using batch_reference = typename inner_batch_type::register_type;
        };
    }

    template <class F, class... CT>
    struct xcontainer_inner_types<xfunction<F, CT...>>
        : public detail::xfunction_type_traits<F, CT...> {};

    /*******************************
     * xfunction implementation
     *******************************/
    template <class F, class... CT>
    class xfunction : public xexpression<xfunction<F, CT...>>
    {
    public:
        using self_type = xfunction<F, CT...>;
        using expression_tag = xtensor_expression_tag;
        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename inner_types::pointer;
        using const_pointer = typename inner_types::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename inner_types::difference_type;
        using shape_type = typename inner_types::shape_type;
        using stepper = xstepper<self_type>;
        using const_stepper = xstepper<const self_type>;

        static constexpr std::size_t arity = sizeof...(CT);
        using functor_type = F;

        template <class Func, class... CTA,
                  std::enable_if_t<!std::is_base_of_v<xfunction, std::decay_t<Func>>, int> = 0>
        xfunction(Func&& f, CTA&&... e) noexcept
            : m_f(std::forward<Func>(f))
            , m_e(std::forward<CTA>(e)...)
        {
        }

        xfunction(const xfunction&) = default;
        xfunction& operator=(const xfunction&) = default;

        size_type size() const;
        shape_type shape() const;

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& s, bool reuse_cache = false) const;

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept;

        const functor_type& functor() const noexcept;
        functor_type& functor() noexcept;

        template <std::size_t I>
        auto& argument() noexcept;
        template <std::size_t I>
        const auto& argument() const noexcept;

        auto& arguments() noexcept;
        const auto& arguments() const noexcept;

    private:
        functor_type m_f;
        std::tuple<CT...> m_e;

        template <std::size_t... I, class... Args>
        auto access_impl(std::index_sequence<I...>, Args... args) const
            -> decltype(m_f(std::get<I>(m_e)(args...)...));
    };

    /*******************************
     * xfunction member functions
     *******************************/
    template <class F, class... CT>
    inline auto xfunction<F, CT...>::size() const -> size_type
    {
        return compute_size(shape());
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::shape() const -> shape_type
    {
        return detail::broadcast_shape(std::get<0>(m_e).shape(), std::get<1>(m_e).shape()...);
    }

    template <class F, class... CT>
    template <class... Args>
    inline auto xfunction<F, CT...>::operator()(Args... args) -> reference
    {
        return static_cast<const self_type&>(*this)(args...);
    }

    template <class F, class... CT>
    template <class... Args>
    inline auto xfunction<F, CT...>::operator()(Args... args) const -> const_reference
    {
        return element(args...);
    }

    template <class F, class... CT>
    template <class It>
    inline auto xfunction<F, CT...>::element(It first, It last) -> reference
    {
        return static_cast<const self_type&>(*this).element(first, last);
    }

    template <class F, class... CT>
    template <class It>
    inline auto xfunction<F, CT...>::element(It first, It last) const -> const_reference
    {
        return access_impl(std::make_index_sequence<arity>{}, first, last);
    }

    template <class F, class... CT>
    template <class S>
    inline bool xfunction<F, CT...>::broadcast_shape(S& s, bool reuse_cache) const
    {
        return detail::broadcast_shape_impl(std::make_index_sequence<arity>{}, s, reuse_cache,
                                            std::get<0>(m_e), std::get<1>(m_e)...);
    }

    template <class F, class... CT>
    template <class S>
    inline bool xfunction<F, CT...>::has_linear_assign(const S& strides) const noexcept
    {
        return detail::has_linear_assign_impl(std::make_index_sequence<arity>{}, strides,
                                              std::get<0>(m_e), std::get<1>(m_e)...);
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::functor() const noexcept -> const functor_type&
    {
        return m_f;
    }

    template <class F, class... CT>
    inline auto xfunction<F, CT...>::functor() noexcept -> functor_type&
    {
        return m_f;
    }

    template <class F, class... CT>
    template <std::size_t I>
    inline auto& xfunction<F, CT...>::argument() noexcept
    {
        return std::get<I>(m_e);
    }

    template <class F, class... CT>
    template <std::size_t I>
    inline const auto& xfunction<F, CT...>::argument() const noexcept
    {
        return std::get<I>(m_e);
    }

    template <class F, class... CT>
    inline auto& xfunction<F, CT...>::arguments() noexcept
    {
        return m_e;
    }

    template <class F, class... CT>
    inline const auto& xfunction<F, CT...>::arguments() const noexcept
    {
        return m_e;
    }

    template <class F, class... CT>
    template <std::size_t... I, class... Args>
    inline auto xfunction<F, CT...>::access_impl(std::index_sequence<I...>, Args... args) const
        -> decltype(m_f(std::get<I>(m_e)(args...)...))
    {
        return m_f(std::get<I>(m_e)(args...)...);
    }

    /*******************************
     * xscalar - constant expression
     *******************************/
    template <class T>
    class xscalar;

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
    };

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

        xscalar() = default;
        explicit xscalar(const value_type& v) : m_value(v) {}
        xscalar(const self_type&) = default;
        xscalar& operator=(const self_type&) = default;

        size_type size() const noexcept { return 1; }
        shape_type shape() const noexcept { return shape_type{}; }

        const_reference operator()() const { return m_value; }
        reference operator()() { return m_value; }

        template <class It>
        const_reference element(It, It) const { return m_value; }
        template <class It>
        reference element(It, It) { return m_value; }

    private:
        value_type m_value;
    };

    /****************************************
     * detail namespace helpers
     ****************************************/
    namespace detail
    {
        template <class... E>
        inline auto broadcast_shape(const E&... e)
        {
            auto first_shape = std::get<0>(std::make_tuple(e.shape()...));
            if constexpr (sizeof...(E) == 1)
                return first_shape;
            else
            {
                return broadcast_shape_impl(first_shape, e.shape()...);
            }
        }

        template <class S>
        inline S broadcast_shape_impl(const S& first)
        {
            return first;
        }

        template <class S, class... Args>
        inline S broadcast_shape_impl(const S& first, const S& second, const Args&... args)
        {
            S result;
            if (first.empty() && second.empty())
            {
                result = first;
            }
            else if (first.empty())
            {
                result = second;
            }
            else if (second.empty())
            {
                result = first;
            }
            else
            {
                std::transform(first.begin(), first.end(), second.begin(),
                               std::back_inserter(result),
                               [](auto a, auto b) { return a == 1 ? b : (b == 1 ? a : a); });
            }
            return broadcast_shape_impl(result, args...);
        }

        template <std::size_t... I, class S, class... E>
        inline bool broadcast_shape_impl(std::index_sequence<I...>, S& s, bool reuse,
                                         const E&... e)
        {
            s = broadcast_shape(e...);
            return true;
        }

        template <std::size_t... I, class S, class... E>
        inline bool has_linear_assign_impl(std::index_sequence<I...>, const S& strides,
                                           const E&... e) noexcept
        {
            auto linear = ((e.has_linear_assign(strides)) && ...);
            return linear;
        }

        template <class F, class... CT>
        using xfunction_type_t = xfunction<F, CT...>;
    }

    /****************************************
     * Operator overloads
     ****************************************/
    template <class E1, class E2>
    inline auto operator+(const xexpression<E1>& e1, const xexpression<E2>& e2)
        -> detail::xfunction_type_t<detail::plus, E1, E2>
    {
        return detail::xfunction_type_t<detail::plus, E1, E2>(detail::plus(), e1.derived_cast(),
                                                               e2.derived_cast());
    }

    template <class E1, class E2>
    inline auto operator-(const xexpression<E1>& e1, const xexpression<E2>& e2)
        -> detail::xfunction_type_t<detail::minus, E1, E2>
    {
        return detail::xfunction_type_t<detail::minus, E1, E2>(detail::minus(), e1.derived_cast(),
                                                                e2.derived_cast());
    }

    template <class E1, class E2>
    inline auto operator*(const xexpression<E1>& e1, const xexpression<E2>& e2)
        -> detail::xfunction_type_t<detail::multiplies, E1, E2>
    {
        return detail::xfunction_type_t<detail::multiplies, E1, E2>(
            detail::multiplies(), e1.derived_cast(), e2.derived_cast());
    }

    template <class E1, class E2>
    inline auto operator/(const xexpression<E1>& e1, const xexpression<E2>& e2)
        -> detail::xfunction_type_t<detail::divides, E1, E2>
    {
        return detail::xfunction_type_t<detail::divides, E1, E2>(detail::divides(),
                                                                  e1.derived_cast(),
                                                                  e2.derived_cast());
    }

    template <class E>
    inline auto operator-(const xexpression<E>& e)
        -> detail::xfunction_type_t<detail::negate, E>
    {
        return detail::xfunction_type_t<detail::negate, E>(detail::negate(), e.derived_cast());
    }

    template <class E1, class E2>
    inline auto operator<(const xexpression<E1>& e1, const xexpression<E2>& e2)
        -> detail::xfunction_type_t<detail::less, E1, E2>
    {
        return detail::xfunction_type_t<detail::less, E1, E2>(detail::less(), e1.derived_cast(),
                                                               e2.derived_cast());
    }

    template <class E1, class E2>
    inline auto operator<=(const xexpression<E1>& e1, const xexpression<E2>& e2)
        -> detail::xfunction_type_t<detail::less_equal, E1, E2>
    {
        return detail::xfunction_type_t<detail::less_equal, E1, E2>(
            detail::less_equal(), e1.derived_cast(), e2.derived_cast());
    }

    template <class E1, class E2>
    inline auto operator>(const xexpression<E1>& e1, const xexpression<E2>& e2)
        -> detail::xfunction_type_t<detail::greater, E1, E2>
    {
        return detail::xfunction_type_t<detail::greater, E1, E2>(
            detail::greater(), e1.derived_cast(), e2.derived_cast());
    }

    template <class E1, class E2>
    inline auto operator>=(const xexpression<E1>& e1, const xexpression<E2>& e2)
        -> detail::xfunction_type_t<detail::greater_equal, E1, E2>
    {
        return detail::xfunction_type_t<detail::greater_equal, E1, E2>(
            detail::greater_equal(), e1.derived_cast(), e2.derived_cast());
    }

    template <class E1, class E2>
    inline auto operator==(const xexpression<E1>& e1, const xexpression<E2>& e2)
        -> detail::xfunction_type_t<detail::equal_to, E1, E2>
    {
        return detail::xfunction_type_t<detail::equal_to, E1, E2>(
            detail::equal_to(), e1.derived_cast(), e2.derived_cast());
    }

    template <class E1, class E2>
    inline auto operator!=(const xexpression<E1>& e1, const xexpression<E2>& e2)
        -> detail::xfunction_type_t<detail::not_equal_to, E1, E2>
    {
        return detail::xfunction_type_t<detail::not_equal_to, E1, E2>(
            detail::not_equal_to(), e1.derived_cast(), e2.derived_cast());
    }

}  // namespace xt

#endif  // XTENSOR_XFUNCTION_HPP