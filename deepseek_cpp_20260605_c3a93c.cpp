//File 0059 : views/xmasked_view.hpp
//Masked view providing filtered access to a base expression using a boolean mask, with lazy evaluation, SIMD-aware iteration, and NaN-safe element access.
#ifndef XTENSOR_XMASKED_VIEW_HPP
#define XTENSOR_XMASKED_VIEW_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xexpression.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xaccessible.hpp"
#include "../core/xiterable.hpp"
#include "../core/xexception.hpp"
#include "../core/xshape.hpp"

namespace xt
{
    /*********************************************
     * xmasked_view – lazy boolean mask filtering
     *********************************************/
    template <class CT, class M>
    class xmasked_view;

    template <class CT, class M>
    struct xcontainer_inner_types<xmasked_view<CT, M>>
    {
        using base_expression_type = std::decay_t<CT>;
        using mask_expression_type = std::decay_t<M>;
        using value_type = typename base_expression_type::value_type;
        using reference = typename base_expression_type::reference;
        using const_reference = typename base_expression_type::const_reference;
        using pointer = typename base_expression_type::pointer;
        using const_pointer = typename base_expression_type::const_pointer;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = typename base_expression_type::shape_type;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<uvector<value_type>, DEFAULT_LAYOUT, shape_type>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

    template <class CT, class M>
    class xmasked_view : public xexpression<xmasked_view<CT, M>>,
                          public xaccessible<xmasked_view<CT, M>>
    {
    public:
        using self_type = xmasked_view<CT, M>;
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
        using mask_type = std::decay_t<M>;

        using stepper = xstepper<self_type>;
        using const_stepper = xstepper<const self_type>;
        using iterator = xiterator<self_type>;
        using const_iterator = xconst_iterator<self_type>;

        /**
         * Construct a masked view: base filtered by mask.
         * @param base The base expression.
         * @param mask Boolean expression of same shape as base.
         */
        template <class E, class Mask>
        xmasked_view(E&& base, Mask&& mask)
            : m_base(std::forward<E>(base)), m_mask(std::forward<Mask>(mask))
        {
            auto base_shape = m_base.shape();
            auto mask_shape = m_mask.shape();
            if (!same_shape(base_shape, mask_shape))
                throw xshape_error::incompatible_shapes(base_shape, mask_shape);

            m_shape = base_shape;
            m_strides = m_base.strides();
            m_backstrides = m_base.backstrides();

            // Count number of true elements for size
            m_true_count = compute_true_count();
        }

        xmasked_view(const self_type&) = default;
        xmasked_view& operator=(const self_type&) = default;
        xmasked_view(self_type&&) = default;
        xmasked_view& operator=(self_type&&) = default;

        size_type size() const noexcept { return m_true_count; }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        void set_shape(const shape_type& s) { m_shape = s; }
        void set_strides(const strides_type& st) { m_strides = st; m_backstrides = detail::compute_backstrides(st, m_shape); }

        /**
         * Element access – returns base element if mask is true, else NaN/0.
         */
        template <class... Args>
        const_reference operator()(Args... args) const
        {
            std::array<size_type, sizeof...(Args)> idx{static_cast<size_type>(args)...};
            return element(idx.begin(), idx.end());
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
            if (m_mask.element(idx.begin(), idx.end()))
            {
                return m_base.element(idx.begin(), idx.end());
            }
            // Return cached NaN/zero for masked elements
            m_cached_masked_value = value_type(0);
            if constexpr (std::is_floating_point_v<value_type>)
                m_cached_masked_value = std::numeric_limits<value_type>::quiet_NaN();
            return m_cached_masked_value;
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
        }

        pointer data() noexcept { return nullptr; }
        const_pointer data() const noexcept { return nullptr; }

        /**
         * Returns true for each position that is both in bounds and unmasked.
         */
        bool is_unmasked(const std::vector<size_type>& idx) const
        {
            return m_mask.element(idx.begin(), idx.end());
        }

        /**
         * Collect only unmasked elements into a 1D array.
         */
        auto unmasked_values() const
        {
            using result_type = xarray_container<uvector<value_type>, DEFAULT_LAYOUT, std::vector<size_type>>;
            result_type result({m_true_count});
            std::size_t pos = 0;
            for (std::size_t i = 0; i < m_base.size(); ++i)
            {
                auto idx = unravel_index(i, m_shape);
                if (is_unmasked(idx))
                    result[pos++] = m_base.element(idx.begin(), idx.end());
            }
            return result;
        }

        /**
         * Fill masked elements with a specified value.
         */
        auto filled(value_type fill_value) const
        {
            auto result = eval(m_base);
            for (std::size_t i = 0; i < m_base.size(); ++i)
            {
                auto idx = unravel_index(i, m_shape);
                if (!is_unmasked(idx))
                    result.element(idx.begin(), idx.end()) = fill_value;
            }
            return result;
        }

        // Iterators – iterate over all elements (masked + unmasked)
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

        // SIMD load (with mask awareness)
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

        const expression_type& base() const noexcept { return m_base; }
        const mask_type& mask() const noexcept { return m_mask; }
        size_type true_count() const noexcept { return m_true_count; }

    private:
        CT m_base;
        M m_mask;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
        size_type m_true_count;
        mutable value_type m_cached_masked_value = value_type(0);

        size_type compute_true_count() const
        {
            size_type count = 0;
            for (std::size_t i = 0; i < m_mask.size(); ++i)
            {
                auto idx = unravel_index(i, m_mask.shape());
                if (m_mask.element(idx.begin(), idx.end()))
                    ++count;
            }
            return count;
        }
    };

    /**
     * Free function to create a masked view.
     */
    template <class E, class M>
    inline auto masked_view(E&& base, M&& mask)
    {
        return xmasked_view<std::decay_t<E>, std::decay_t<M>>(
            std::forward<E>(base), std::forward<M>(mask));
    }

    /**
     * Convenience: where(condition, x, y) – selects from x where condition is true, else y.
     */
    template <class C, class E1, class E2>
    inline auto where(const C& condition, const E1& x, const E2& y)
    {
        // Lazily evaluate: condition ? x : y
        using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
        struct where_functor
        {
            template <class CondVal, class XVal, class YVal>
            auto operator()(CondVal c, XVal a, YVal b) const
            {
                return c ? static_cast<value_type>(a) : static_cast<value_type>(b);
            }
        };
        return detail::make_xfunction(where_functor{}, condition.derived_cast(),
                                      x.derived_cast(), y.derived_cast());
    }

} // namespace xt

#endif // XTENSOR_XMASKED_VIEW_HPP