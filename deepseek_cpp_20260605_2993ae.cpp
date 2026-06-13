//File 0018 (UPDATED) : core/xexpression.hpp
//Full expression base with xaccessible, xconst_iterable, in_bounds, unchecked, back, front, periodic, and SIMD loading.
#ifndef XTENSOR_XEXPRESSION_HPP
#define XTENSOR_XEXPRESSION_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"

namespace xt
{
    /*******************************
     * xaccessible mixin
     *******************************/
    template <class D>
    class xaccessible
    {
    public:
        using derived_type = D;
        using inner_types = xcontainer_inner_types<D>;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using size_type = typename inner_types::size_type;

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

        reference operator[](size_type i) { return derived_cast()[i]; }
        const_reference operator[](size_type i) const { return derived_cast()[i]; }

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

        template <class... Args>
        bool in_bounds(Args... args) const
        {
            auto idx = std::array<size_type, sizeof...(Args)>{static_cast<size_type>(args)...};
            auto& sh = derived_cast().shape();
            if (idx.size() != sh.size()) return false;
            for (std::size_t i = 0; i < sh.size(); ++i)
                if (idx[i] >= sh[i]) return false;
            return true;
        }

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

        reference back() { return derived_cast().data()[derived_cast().size() - 1]; }
        const_reference back() const { return derived_cast().data()[derived_cast().size() - 1]; }
        reference front() { return derived_cast().data()[0]; }
        const_reference front() const { return derived_cast().data()[0]; }

    private:
        derived_type& derived_cast() noexcept { return *static_cast<derived_type*>(this); }
        const derived_type& derived_cast() const noexcept { return *static_cast<const derived_type*>(this); }

        template <class... Args>
        void check_bounds(Args... args) const
        {
            if (!in_bounds(args...))
                throw std::out_of_range("Index out of bounds.");
        }
        template <class... Args>
        void adjust_periodic(Args&... args) const
        {
            auto idx = std::array<size_type, sizeof...(Args)>{static_cast<size_type>(args)...};
            auto& sh = derived_cast().shape();
            for (std::size_t i = 0; i < sh.size(); ++i)
                if (sh[i] > 0) idx[i] = ((idx[i] % sh[i]) + sh[i]) % sh[i];
            std::size_t pos = 0;
            ((args = idx[pos++]), ...);
        }
    };

    /*******************************
     * xconst_iterable mixin
     *******************************/
    template <class D>
    class xconst_iterable
    {
    public:
        using derived_type = D;
        using const_iterator = const typename D::value_type*;

        const_iterator begin() const noexcept { return derived_cast().data(); }
        const_iterator end() const noexcept { return derived_cast().data() + derived_cast().size(); }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

    private:
        const derived_type& derived_cast() const noexcept { return *static_cast<const derived_type*>(this); }
    };

    /*******************************
     * xexpression CRTP base
     *******************************/
    template <class D>
    class xexpression : public xaccessible<D>,
                        public xconst_iterable<D>
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

        derived_type& derived_cast() noexcept { return *static_cast<derived_type*>(this); }
        const derived_type& derived_cast() const noexcept { return *static_cast<const derived_type*>(this); }

        size_type size() const noexcept { return derived_cast().size(); }
        size_type dimension() const noexcept { return derived_cast().shape().size(); }
        shape_type shape() const noexcept { return derived_cast().shape(); }
        layout_type layout() const noexcept { return inner_types::layout; }

        pointer data() noexcept { return derived_cast().data(); }
        const_pointer data() const noexcept { return derived_cast().data(); }

        using xaccessible<D>::operator();
        using xaccessible<D>::operator[];
        using xaccessible<D>::at;
        using xaccessible<D>::periodic;
        using xaccessible<D>::in_bounds;
        using xaccessible<D>::unchecked;
        using xaccessible<D>::back;
        using xaccessible<D>::front;

        using xconst_iterable<D>::begin;
        using xconst_iterable<D>::end;
        using xconst_iterable<D>::cbegin;
        using xconst_iterable<D>::cend;

        template <class It>
        reference element(It first, It last) { return derived_cast().element(first, last); }
        template <class It>
        const_reference element(It first, It last) const { return derived_cast().element(first, last); }

        template <class S>
        bool broadcast_shape(S& s, bool reuse_cache = false) const { return derived_cast().broadcast_shape(s, reuse_cache); }

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept { return derived_cast().has_linear_assign(strides); }

        // SIMD interface
        template <class Align = aligned_mode::unaligned, class T = value_type>
        auto load_simd(std::size_t i) const { return derived_cast().template load_simd<Align, T>(i); }

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