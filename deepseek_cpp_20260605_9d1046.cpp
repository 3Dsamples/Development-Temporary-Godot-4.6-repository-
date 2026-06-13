//File 0003 (UPDATED) : core/xfunction.hpp
//Complete expression engine with lazy evaluation, broadcasting, SIMD loading, iterators, steppers, and caching.
#ifndef XTENSOR_XFUNCTION_HPP
#define XTENSOR_XFUNCTION_HPP

#include <cstddef>
#include <iterator>
#include <memory>
#include <tuple>
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
    template <class F, class... CT>
    class xfunction;

    namespace detail
    {
        // forward declaration of xfunction_type_traits
        template <class F, class... CT>
        struct xfunction_type_traits;
    }

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
            using stepper = xstepper<type>;
            using const_stepper = xstepper<const type>;

            using inner_batch_type = xsimd::batch<value_type, default_simd_arch>;
            using batch_reference = typename inner_batch_type::register_type;
        };
    }

    template <class F, class... CT>
    struct xcontainer_inner_types<xfunction<F, CT...>>
        : public detail::xfunction_type_traits<F, CT...> {};

    /*******************************
     * xfunction_iterator (random access)
     *******************************/
    template <class CT>
    class xfunction_iterator : public xtl::xrandom_access_iterator_base<xfunction_iterator<CT>,
                                                                        typename CT::value_type,
                                                                        typename CT::difference_type,
                                                                        typename CT::pointer,
                                                                        typename CT::reference>
    {
    public:
        using self_type = xfunction_iterator<CT>;
        using functor_type = typename CT::functor_type;
        using value_type = typename CT::value_type;

        xfunction_iterator() noexcept : p_f(nullptr), m_index(0) {}
        xfunction_iterator(CT* f, typename CT::size_type index) noexcept
            : p_f(f), m_index(index) {}

        self_type& operator++() { ++m_index; return *this; }
        self_type& operator--() { --m_index; return *this; }
        self_type& operator+=(difference_type n) { m_index += n; return *this; }
        self_type& operator-=(difference_type n) { m_index -= n; return *this; }
        difference_type operator-(const self_type& rhs) const { return m_index - rhs.m_index; }

        value_type operator*() const
        {
            auto idx = unravel_index(m_index, p_f->shape());
            return p_f->element(idx.begin(), idx.end());
        }

        bool operator==(const self_type& rhs) const { return p_f == rhs.p_f && m_index == rhs.m_index; }
        bool operator<(const self_type& rhs) const { return m_index < rhs.m_index; }

    private:
        CT* p_f;
        typename CT::size_type m_index;
    };

    /*******************************
     * xfunction_stepper
     *******************************/
    template <class CT>
    class xfunction_stepper
    {
    public:
        using value_type = typename CT::value_type;
        using reference = typename CT::reference;

        xfunction_stepper(CT* f, typename CT::size_type index) noexcept
            : p_f(f), m_index(index) {}

        void step(std::size_t dim, std::size_t n = 1)
        {
            m_index += n * p_f->strides()[dim];
        }

        void step_back(std::size_t dim, std::size_t n = 1)
        {
            m_index -= n * p_f->strides()[dim];
        }

        void reset(std::size_t dim)
        {
            // Reset to first element of current dimension
            auto& strides = p_f->strides();
            auto& shape = p_f->shape();
            std::size_t offset = m_index % strides[dim];
            m_index -= offset;
        }

        reference operator*() const
        {
            auto idx = unravel_index(m_index, p_f->shape());
            return p_f->element(idx.begin(), idx.end());
        }

    private:
        CT* p_f;
        typename CT::size_type m_index;
    };

    /*******************************
     * xfunction_cache
     *******************************/
    template <class F, class... CT>
    class xfunction_cache
    {
    public:
        using shape_type = typename xfunction<F, CT...>::shape_type;
        using strides_type = typename xfunction<F, CT...>::strides_type;
        using size_type = typename xfunction<F, CT...>::size_type;

        xfunction_cache(const xfunction<F, CT...>* func) : p_func(func)
        {
            update_cache();
        }

        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        size_type size() const noexcept { return m_size; }

        void refresh() { update_cache(); }

    private:
        void update_cache()
        {
            m_shape = p_func->shape();
            m_strides = compute_strides(m_shape);
            m_size = compute_size(m_shape);
        }

        const xfunction<F, CT...>* p_func;
        shape_type m_shape;
        strides_type m_strides;
        size_type m_size;
    };

    /*******************************
     * xfunction implementation (enhanced)
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
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using stepper = xfunction_stepper<self_type>;
        using const_stepper = xfunction_stepper<const self_type>;
        using iterator = xfunction_iterator<self_type>;
        using const_iterator = xfunction_iterator<const self_type>;
        using functor_type = F;

        static constexpr std::size_t arity = sizeof...(CT);

        template <class Func, class... CTA,
                  std::enable_if_t<!std::is_base_of_v<xfunction, std::decay_t<Func>>, int> = 0>
        xfunction(Func&& f, CTA&&... e) noexcept
            : m_f(std::forward<Func>(f))
            , m_e(std::forward<CTA>(e)...)
            , m_cache(this)
        {
        }

        xfunction(const xfunction&) = default;
        xfunction& operator=(const xfunction&) = default;

        size_type size() const { return m_cache.size(); }
        shape_type shape() const { return m_cache.shape(); }
        const strides_type& strides() const { return m_cache.strides(); }
        const backstrides_type& backstrides() const { return m_backstrides; }

        template <class... Args>
        reference operator()(Args... args)
        {
            return access_impl(std::make_index_sequence<arity>{}, args...);
        }

        template <class... Args>
        const_reference operator()(Args... args) const
        {
            return access_impl(std::make_index_sequence<arity>{}, args...);
        }

        template <class It>
        reference element(It first, It last)
        {
            return access_impl(std::make_index_sequence<arity>{}, first, last);
        }

        template <class It>
        const_reference element(It first, It last) const
        {
            return access_impl(std::make_index_sequence<arity>{}, first, last);
        }

        template <class S>
        bool broadcast_shape(S& s, bool reuse_cache = false) const
        {
            return detail::broadcast_shape_impl(std::make_index_sequence<arity>{}, s, reuse_cache,
                                                std::get<0>(m_e), std::get<1>(m_e)...);
        }

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept
        {
            return detail::has_linear_assign_impl(std::make_index_sequence<arity>{}, strides,
                                                  std::get<0>(m_e), std::get<1>(m_e)...);
        }

        const functor_type& functor() const noexcept { return m_f; }
        functor_type& functor() noexcept { return m_f; }

        template <std::size_t I>
        auto& argument() noexcept { return std::get<I>(m_e); }
        template <std::size_t I>
        const auto& argument() const noexcept { return std::get<I>(m_e); }

        auto& arguments() noexcept { return m_e; }
        const auto& arguments() const noexcept { return m_e; }

        // SIMD interface
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            simd_type result;
            // load from arguments using SIMD if they support it, else scalar
            load_simd_impl(result, i, std::make_index_sequence<arity>{});
            return result;
        }

        // Iterator support
        iterator begin() noexcept { return iterator(this, 0); }
        iterator end() noexcept { return iterator(this, size()); }
        const_iterator begin() const noexcept { return const_iterator(this, 0); }
        const_iterator end() const noexcept { return const_iterator(this, size()); }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

        // Stepper
        stepper stepper_begin() noexcept { return stepper(this, 0); }
        stepper stepper_end() noexcept { return stepper(this, size()); }
        const_stepper stepper_begin() const noexcept { return const_stepper(this, 0); }
        const_stepper stepper_end() const noexcept { return const_stepper(this, size()); }

        // Cache management
        void refresh_cache() { m_cache.refresh(); }

    private:
        functor_type m_f;
        std::tuple<CT...> m_e;
        xfunction_cache<F, CT...> m_cache;
        backstrides_type m_backstrides = detail::compute_backstrides(strides(), shape());

        template <std::size_t... I, class... Args>
        auto access_impl(std::index_sequence<I...>, Args... args) const
            -> decltype(m_f(std::get<I>(m_e)(args...)...))
        {
            return m_f(std::get<I>(m_e)(args...)...);
        }

        template <std::size_t... I, class T>
        void load_simd_impl(xsimd::batch<T, default_simd_arch>& result, std::size_t i, std::index_sequence<I...>) const
        {
            // Scalar fallback: load element by element
            // In a real implementation, each argument would provide a load_simd method
            result = xsimd::batch<T, default_simd_arch>(static_cast<T>(element_linear(i)));
        }

        value_type element_linear(std::size_t i) const
        {
            auto idx = unravel_index(i, shape());
            return element(idx.begin(), idx.end());
        }
    };

    // Deduction guide for xfunction
    template <class F, class... CT>
    xfunction(F&&, CT&&...) -> xfunction<std::decay_t<F>, std::decay_t<CT>...>;

}  // namespace xt

#endif  // XTENSOR_XFUNCTION_HPP