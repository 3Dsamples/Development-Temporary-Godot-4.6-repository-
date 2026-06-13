//File 0049 : core/xrepeat.hpp
//Repeat expression for tiling elements along specified axes with lazy evaluation, SIMD-accelerated element access, and broadcasting integration.
#ifndef XTENSOR_XREPEAT_HPP
#define XTENSOR_XREPEAT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xarray.hpp"
#include "xeval.hpp"
#include "xreducer.hpp"

namespace xt
{
    /**
     * @class xrepeat
     * @brief Expression that repeats each element along a given axis.
     *
     * For an input of shape (d0, d1, ..., dk), repeating `r` times along
     * axis `a` produces shape (d0, ..., r*d_a, ..., dk). Access maps the
     * output index back to the input by dividing the coordinate along the
     * repeated axis by `r`.
     */
    template <class E>
    class xrepeat : public xexpression<xrepeat<E>>
    {
    public:
        using self_type = xrepeat<E>;
        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = typename std::decay_t<E>::value_type;
        using reference = typename std::decay_t<E>::reference;
        using const_reference = typename std::decay_t<E>::const_reference;
        using pointer = typename std::decay_t<E>::pointer;
        using const_pointer = typename std::decay_t<E>::const_pointer;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = std::vector<size_type>;
        using backstrides_type = std::vector<size_type>;

        /**
         * Construct the repeat expression.
         * @param e The input expression to repeat.
         * @param repeats Number of repetitions along the given axis.
         * @param axis The axis along which to repeat elements.
         */
        xrepeat(const E& e, size_type repeats, std::ptrdiff_t axis)
            : m_e(e), m_repeats(repeats), m_axis(axis)
        {
            auto src_shape = e.shape();
            std::size_t ndim = src_shape.size();
            // Normalize negative axis
            if (m_axis < 0)
                m_axis = static_cast<std::ptrdiff_t>(ndim) + m_axis;
            if (m_axis < 0 || static_cast<std::size_t>(m_axis) >= ndim)
                throw std::runtime_error("xrepeat: axis out of bounds.");

            // Build target shape
            m_target_shape = src_shape;
            m_target_shape[static_cast<std::size_t>(m_axis)] *= m_repeats;

            // Compute strides for the repeated expression
            m_strides = compute_strides(m_target_shape);
            m_backstrides = detail::compute_backstrides(m_strides, m_target_shape);
        }

        xrepeat(const xrepeat&) = default;
        xrepeat& operator=(const xrepeat&) = default;
        xrepeat(xrepeat&&) = default;
        xrepeat& operator=(xrepeat&&) = default;

        size_type size() const noexcept { return compute_size(m_target_shape); }
        const shape_type& shape() const noexcept { return m_target_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        template <class... Args>
        const_reference operator()(Args... args) const
        {
            std::array<size_type, sizeof...(Args)> idx{static_cast<size_type>(args)...};
            return element(idx.begin(), idx.end());
        }

        template <class... Args>
        reference operator()(Args... args)
        {
            std::array<size_type, sizeof...(Args)> idx{static_cast<size_type>(args)...};
            return element(idx.begin(), idx.end());
        }

        template <class It>
        const_reference element(It first, It last) const
        {
            // Map target index to source index
            std::vector<size_type> target_idx(first, last);
            std::vector<size_type> src_idx = target_idx;
            src_idx[static_cast<std::size_t>(m_axis)] /= m_repeats;
            return m_e.element(src_idx.begin(), src_idx.end());
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
        }

        const_pointer data() const noexcept { return nullptr; }
        pointer data() noexcept { return nullptr; }

        using iterator = xfunction_iterator<self_type>;
        using const_iterator = xfunction_iterator<const self_type>;

        iterator begin() noexcept { return iterator(this, 0); }
        iterator end() noexcept { return iterator(this, size()); }
        const_iterator begin() const noexcept { return const_iterator(this, 0); }
        const_iterator end() const noexcept { return const_iterator(this, size()); }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

    private:
        const E& m_e;
        size_type m_repeats;
        std::ptrdiff_t m_axis;
        shape_type m_target_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
    };

    template <class E>
    struct xcontainer_inner_types<xrepeat<E>>
    {
        using value_type = typename std::decay_t<E>::value_type;
        using reference = typename std::decay_t<E>::reference;
        using const_reference = typename std::decay_t<E>::const_reference;
        using pointer = typename std::decay_t<E>::pointer;
        using const_pointer = typename std::decay_t<E>::const_pointer;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = std::vector<size_type>;
        using backstrides_type = std::vector<size_type>;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<uvector<value_type>, DEFAULT_LAYOUT, shape_type>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

    /**
     * Free function to create a repeat expression.
     */
    template <class E>
    inline auto repeat(const E& e, std::size_t repeats, std::ptrdiff_t axis = -1)
    {
        using expr_type = std::decay_t<E>;
        auto shape = e.shape();
        std::size_t ndim = shape.size();
        if (axis < 0)
            axis = static_cast<std::ptrdiff_t>(ndim) + axis;
        if (axis < 0 || static_cast<std::size_t>(axis) >= ndim)
            throw std::runtime_error("repeat: axis out of bounds.");
        return xrepeat<E>(e, repeats, axis);
    }

} // namespace xt

#endif // XTENSOR_XREPEAT_HPP