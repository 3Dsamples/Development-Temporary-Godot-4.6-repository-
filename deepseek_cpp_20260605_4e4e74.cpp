//File 0037 : core/xgenerator.hpp
//Lazy generator expression with on‑access computation, SIMD stepper, random‑access iterator, and broadcasting support.
#ifndef XTENSOR_XGENERATOR_HPP
#define XTENSOR_XGENERATOR_HPP

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xmath.hpp"
#include "xstrides.hpp"

namespace xt
{
    template <class F, class S>
    class xgenerator;

    template <class F, class S>
    struct xcontainer_inner_types<xgenerator<F, S>>
    {
        using value_type = std::decay_t<decltype(std::declval<F>()())>;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = S;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<uvector<value_type>, DEFAULT_LAYOUT, shape_type>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

    /**
     * @class xgenerator
     * @brief Lazy expression that evaluates a functor on element access.
     *
     * The functor must be callable with either no arguments (for scalar access),
     * a single size_type argument (for linear offset), or an iterator pair (for
     * multi-dimensional index). The shape is stored by value.
     */
    template <class F, class S>
    class xgenerator : public xexpression<xgenerator<F, S>>
    {
    public:
        using self_type = xgenerator<F, S>;
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

        using stepper = xstepper<self_type>;
        using const_stepper = xstepper<const self_type>;
        using iterator = xfunction_iterator<self_type>;
        using const_iterator = xfunction_iterator<const self_type>;

        xgenerator(F&& f, const S& shape)
            : m_f(std::forward<F>(f)), m_shape(shape)
        {
            m_strides = compute_strides(m_shape);
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }

        xgenerator(const self_type&) = default;
        xgenerator& operator=(const self_type&) = default;
        xgenerator(self_type&&) = default;
        xgenerator& operator=(self_type&&) = default;

        size_type size() const noexcept { return compute_size(m_shape); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        void set_shape(const shape_type& s) { m_shape = s; compute_strides(); }
        void set_strides(const strides_type& st) { m_strides = st; m_backstrides = detail::compute_backstrides(st, m_shape); }

        template <class... Args>
        const_reference operator()(Args... args) const
        {
            if constexpr (sizeof...(Args) == 0)
                return m_eval();
            else if constexpr (sizeof...(Args) == 1)
                return m_eval(static_cast<size_type>(args...));
            else
                return element(std::begin(std::array<size_type, sizeof...(Args)>{static_cast<size_type>(args)...}),
                               std::end(std::array<size_type, sizeof...(Args)>{static_cast<size_type>(args)...}));
        }

        template <class... Args>
        reference operator()(Args... args)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).operator()(args...));
        }

        reference operator[](size_type i) { return operator()(i); }
        const_reference operator[](size_type i) const { return operator()(i); }

        template <class It>
        const_reference element(It first, It last) const
        {
            return m_f(first, last);
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
        }

        pointer data() noexcept { return nullptr; }
        const_pointer data() const noexcept { return nullptr; }

        // Iterators (linear)
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

        // SIMD loading (for contiguous? – not applicable; scalar fallback)
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            simd_type result;
            alignas(64) std::array<T, simd_type::size> buf;
            for (std::size_t k = 0; k < simd_type::size; ++k)
                buf[k] = operator()(i + k);
            return simd_type::load_aligned(buf.data());
        }

    private:
        F m_f;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;

        void compute_strides()
        {
            m_strides = xt::compute_strides(m_shape);
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }

        // Scalar evaluation helpers
        template <class T = value_type>
        T m_eval() const { return m_f(); }

        template <class T = value_type>
        T m_eval(size_type i) const
        {
            auto idx = unravel_index(i, m_shape);
            return m_f(idx.begin(), idx.end());
        }
    };

    // Deduction guide
    template <class F, class S>
    xgenerator(F&&, S&&) -> xgenerator<std::decay_t<F>, std::decay_t<S>>;

    /*******************************
     * detail::make_xgenerator
     *******************************/
    namespace detail
    {
        template <class F, class S>
        inline auto make_xgenerator(F&& f, S&& shape)
        {
            return xgenerator<std::decay_t<F>, std::decay_t<S>>(std::forward<F>(f), std::forward<S>(shape));
        }
    }

    /*******************************
     * xstepper for xgenerator
     *******************************/
    template <class F, class S>
    class xstepper<xgenerator<F, S>>
    {
    public:
        using view_type = xgenerator<F, S>;
        using value_type = typename view_type::value_type;
        using reference = value_type&;
        using size_type = typename view_type::size_type;

        xstepper(view_type* v, size_type offset) : p_view(v), m_offset(offset) {}
        void step(size_type dim, size_type n = 1) { m_offset += n * p_view->strides()[dim]; }
        void step_back(size_type dim, size_type n = 1) { m_offset -= n * p_view->strides()[dim]; }
        void reset(size_type dim) { m_offset = m_offset % p_view->strides()[dim]; }
        reference operator*() const { return (*p_view)[m_offset]; }

    private:
        view_type* p_view;
        size_type m_offset;
    };

    template <class F, class S>
    class xstepper<const xgenerator<F, S>>
    {
    public:
        using view_type = const xgenerator<F, S>;
        using value_type = typename view_type::value_type;
        using const_reference = const value_type&;
        using size_type = typename view_type::size_type;

        xstepper(view_type* v, size_type offset) : p_view(v), m_offset(offset) {}
        void step(size_type dim, size_type n = 1) { m_offset += n * p_view->strides()[dim]; }
        void step_back(size_type dim, size_type n = 1) { m_offset -= n * p_view->strides()[dim]; }
        void reset(size_type dim) { m_offset = m_offset % p_view->strides()[dim]; }
        const_reference operator*() const { return (*p_view)[m_offset]; }

    private:
        view_type* p_view;
        size_type m_offset;
    };

    /*******************************
     * xfunction_iterator for xgenerator
     *******************************/
    template <class F, class S>
    class xfunction_iterator<xgenerator<F, S>>
        : public xtl::xrandom_access_iterator_base<xfunction_iterator<xgenerator<F, S>>,
                                                    typename xgenerator<F, S>::value_type,
                                                    typename xgenerator<F, S>::difference_type,
                                                    typename xgenerator<F, S>::pointer,
                                                    typename xgenerator<F, S>::reference>
    {
    public:
        using self_type = xfunction_iterator<xgenerator<F, S>>;
        using value_type = typename xgenerator<F, S>::value_type;
        using size_type = typename xgenerator<F, S>::size_type;

        xfunction_iterator() noexcept : p_f(nullptr), m_index(0) {}
        xfunction_iterator(const xgenerator<F, S>* f, size_type index) noexcept
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
        const xgenerator<F, S>* p_f;
        size_type m_index;
    };

} // namespace xt

#endif // XTENSOR_XGENERATOR_HPP