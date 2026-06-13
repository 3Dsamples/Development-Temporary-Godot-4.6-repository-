//File 0056 : views/xfunctor_view.hpp
//Functor view: applies a unary functor element-wise to a base expression lazily, with SIMD-aware evaluation, iterators, and steppers.
#ifndef XTENSOR_XFUNCTOR_VIEW_HPP
#define XTENSOR_XFUNCTOR_VIEW_HPP

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xexpression.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xiterator.hpp"
#include "../core/xaccessible.hpp"

namespace xt
{
    /*********************************************
     * xfunctor_view: lazy unary function application
     *********************************************/
    template <class F, class CT>
    class xfunctor_view;

    template <class F, class CT>
    struct xcontainer_inner_types<xfunctor_view<F, CT>>
    {
        using storage_type = typename std::decay_t<CT>::storage_type;
        using value_type = std::decay_t<decltype(std::declval<F>()(std::declval<typename std::decay_t<CT>::value_type>()))>;
        using reference = std::remove_reference_t<value_type>&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = typename std::decay_t<CT>::shape_type;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<uvector<value_type>, DEFAULT_LAYOUT, shape_type>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

    template <class F, class CT>
    class xfunctor_view : public xexpression<xfunctor_view<F, CT>>,
                          public xaccessible<xfunctor_view<F, CT>>
    {
    public:
        using self_type = xfunctor_view<F, CT>;
        using base_type = xexpression<self_type>;
        using inner_types = xcontainer_inner_types<self_type>;
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
        using expression_type = std::decay_t<CT>;
        using functor_type = F;

        using stepper = xstepper<self_type>;
        using const_stepper = xstepper<const self_type>;
        using iterator = xiterator<self_type>;
        using const_iterator = xconst_iterator<self_type>;

        /**
         * Construct a functor view: f(e).
         */
        template <class Func, class E>
        xfunctor_view(Func&& f, E&& e) noexcept
            : m_f(std::forward<Func>(f)), m_e(std::forward<E>(e))
        {
        }

        xfunctor_view(const self_type&) = default;
        xfunctor_view& operator=(const self_type&) = default;
        xfunctor_view(self_type&&) = default;
        xfunctor_view& operator=(self_type&&) = default;

        size_type size() const noexcept { return m_e.size(); }
        const shape_type& shape() const noexcept { return m_e.shape(); }
        const strides_type& strides() const noexcept { return m_e.strides(); }
        const backstrides_type& backstrides() const noexcept { return m_e.backstrides(); }

        void set_shape(const shape_type& s) { /* immutable */ }
        void set_strides(const strides_type& st) { /* immutable */ }

        template <class... Args>
        const_reference operator()(Args... args) const
        {
            return access_impl(std::make_index_sequence<sizeof...(Args)>{}, args...);
        }

        template <class... Args>
        reference operator()(Args... args)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this)(args...));
        }

        reference operator[](size_type i) { return operator()(i); }
        const_reference operator[](size_type i) const { return operator()(i); }

        template <class It>
        const_reference element(It first, It last) const
        {
            auto idx = std::vector<size_type>(first, last);
            return m_f(m_e.element(idx.begin(), idx.end()));
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
        }

        pointer data() noexcept { return nullptr; }
        const_pointer data() const noexcept { return nullptr; }

        // Iterator support
        iterator begin() noexcept { return iterator(this, 0); }
        iterator end() noexcept { return iterator(this, size()); }
        const_iterator begin() const noexcept { return const_iterator(this, 0); }
        const_iterator end() const noexcept { return const_iterator(this, size()); }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

        // Stepper support
        stepper stepper_begin() noexcept { return stepper(this, 0); }
        stepper stepper_end() noexcept { return stepper(this, size()); }
        const_stepper stepper_begin() const noexcept { return const_stepper(this, 0); }
        const_stepper stepper_end() const noexcept { return const_stepper(this, size()); }

        // SIMD loading
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            simd_type result;
            constexpr std::size_t simd_size = simd_type::size;
            alignas(64) std::array<T, simd_size> buffer;
            for (std::size_t k = 0; k < simd_size; ++k)
                buffer[k] = operator()(i + k);
            return simd_type::load_aligned(buffer.data());
        }

        const expression_type& expression() const noexcept { return m_e; }
        const functor_type& functor() const noexcept { return m_f; }

    private:
        F m_f;
        CT m_e;

        template <std::size_t... I, class... Args>
        const_reference access_impl(std::index_sequence<I...>, Args... args) const
        {
            return m_f(m_e(args...));
        }
    };

    /**
     * Free function to create a functor view.
     */
    template <class F, class E>
    inline auto functor_view(F&& f, E&& e)
    {
        return xfunctor_view<std::decay_t<F>, std::decay_t<E>>(
            std::forward<F>(f), std::forward<E>(e));
    }

} // namespace xt

#endif // XTENSOR_XFUNCTOR_VIEW_HPP