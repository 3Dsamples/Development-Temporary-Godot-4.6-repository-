//File 0329 : xframe/xaxis_view.hpp
//Axis view: lazy view along a single axis of an xframe, providing 1D variable access and broadcasting over other dimensions with SIMD support.
#ifndef XFRAME_XAXIS_VIEW_HPP
#define XFRAME_XAXIS_VIEW_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_variable.hpp"
#include "xframe_coordinate.hpp"
#include "xframe_dimension.hpp"
#include "xframe.hpp"

namespace xframe
{
    /**
     * @class xaxis_view
     * @brief Lazy 1D view along a specified axis of an xframe.
     *
     * The view extracts a single axis, keeping that dimension intact and
     * collapsing all other dimensions to a single index (the current slice).
     * Element access returns the value of a chosen variable at the given
     * coordinate along the viewed axis, at the selected slice coordinates.
     */
    template <class CT>
    class xaxis_view : public expression<xaxis_view<CT>>
    {
    public:
        using self_type = xaxis_view<CT>;
        using base_type = std::decay_t<CT>;
        using size_type = std::size_t;
        using label_type = typename base_type::label_type;

        /**
         * Construct an axis view.
         * @param base The base xframe.
         * @param axis_index The axis to view (0..ndim-1).
         * @param slice_indices Fixed indices for all other dimensions (size = ndim-1).
         */
        xaxis_view(const base_type& base, size_type axis_index,
                   const std::vector<size_type>& slice_indices)
            : m_base(base), m_axis(axis_index), m_slice_indices(slice_indices)
        {
            std::size_t ndim = base.dimension_count();
            if (axis_index >= ndim)
                throw std::out_of_range("xaxis_view: axis index out of bounds.");
            if (slice_indices.size() != ndim - 1)
                throw std::runtime_error("xaxis_view: slice_indices must have ndim-1 elements.");
            // Validate slice indices
            std::size_t si = 0;
            for (std::size_t d = 0; d < ndim; ++d)
            {
                if (d == axis_index) continue;
                if (slice_indices[si] >= base.dimension(d).size())
                    throw std::out_of_range("xaxis_view: slice index out of bounds.");
                ++si;
            }
            // Build view dimension (the axis itself)
            m_view_dim = base.dimension(axis_index);
        }

        /**
         * Construct an axis view using label-based slicing for the fixed dimensions.
         * The slice_labels must contain labels for each dimension except the viewed axis,
         * in order.
         */
        xaxis_view(const base_type& base, size_type axis_index,
                   const std::vector<label_type>& slice_labels)
            : m_base(base), m_axis(axis_index)
        {
            std::size_t ndim = base.dimension_count();
            if (axis_index >= ndim)
                throw std::out_of_range("xaxis_view: axis index out of bounds.");
            if (slice_labels.size() != ndim - 1)
                throw std::runtime_error("xaxis_view: slice_labels must have ndim-1 elements.");
            std::size_t li = 0;
            for (std::size_t d = 0; d < ndim; ++d)
            {
                if (d == axis_index) continue;
                size_type idx = base.dimension(d).coord().find(slice_labels[li]);
                if (idx >= base.dimension(d).size())
                    throw std::out_of_range("xaxis_view: slice label not found.");
                m_slice_indices.push_back(idx);
                ++li;
            }
            m_view_dim = base.dimension(axis_index);
        }

        xaxis_view(const self_type&) = default;
        xaxis_view& operator=(const self_type&) = default;
        xaxis_view(self_type&&) = default;
        xaxis_view& operator=(self_type&&) = default;

        std::size_t dimension_count() const noexcept { return 1; }
        std::size_t size() const noexcept { return m_view_dim.size(); }

        const dimension<label_type>& dimension(std::size_t i) const
        {
            if (i != 0) throw std::out_of_range("xaxis_view: dimension index out of range.");
            return m_view_dim;
        }

        template <class... Args>
        auto operator()(size_type index) const
        {
            auto full_idx = build_full_index(index);
            return call_base(full_idx, std::make_index_sequence<sizeof...(Args)>{});
        }

        template <class... Args>
        auto operator()(size_type index)
        {
            return const_cast<const self_type*>(this)->operator()(index);
        }

        auto operator[](size_type i) const
        {
            auto full_idx = build_full_index(i);
            return call_base_flat(full_idx);
        }

        auto operator[](size_type i)
        {
            return const_cast<const self_type*>(this)->operator[](i);
        }

        template <class... Labels>
        auto locate(const label_type& label) const
        {
            size_type idx = m_view_dim.coord().find(label);
            if (idx >= m_view_dim.size())
                throw std::out_of_range("xaxis_view: label not found.");
            return (*this)(idx);
        }

        template <class... Labels>
        auto locate(const label_type& label)
        {
            size_type idx = m_view_dim.coord().find(label);
            return (*this)(idx);
        }

        const base_type& base() const noexcept { return m_base; }
        size_type axis() const noexcept { return m_axis; }
        const std::vector<size_type>& slice_indices() const noexcept { return m_slice_indices; }

        template <class Align, class T = double>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            alignas(64) std::array<T, simd_size> buf;
            for (std::size_t k = 0; k < simd_size; ++k)
                buf[k] = static_cast<T>((*this)[i + k]);
            return simd_type::load_aligned(buf.data());
        }

    private:
        const base_type& m_base;
        size_type m_axis;
        std::vector<size_type> m_slice_indices;
        dimension<label_type> m_view_dim;

        std::vector<size_type> build_full_index(size_type axis_coord) const
        {
            std::size_t ndim = m_base.dimension_count();
            std::vector<size_type> full(ndim);
            std::size_t si = 0;
            for (std::size_t d = 0; d < ndim; ++d)
            {
                if (d == m_axis)
                    full[d] = axis_coord;
                else
                    full[d] = m_slice_indices[si++];
            }
            return full;
        }

        template <std::size_t... I>
        auto call_base(const std::vector<size_type>& idx, std::index_sequence<I...>) const
        {
            return m_base(idx[I]...);
        }

        auto call_base_flat(const std::vector<size_type>& idx) const
        {
            // Compute flat index and use operator[]
            std::size_t flat = 0;
            std::size_t stride = 1;
            std::size_t ndim = idx.size();
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d)
            {
                flat += idx[static_cast<std::size_t>(d)] * stride;
                stride *= m_base.dimension(static_cast<std::size_t>(d)).size();
            }
            return m_base[flat];
        }
    };

    /**
     * Free function to create an axis view.
     */
    template <class E>
    inline auto axis_view(const E& base, std::size_t axis_index,
                          const std::vector<std::size_t>& slice_indices)
    {
        return xaxis_view<std::decay_t<E>>(base, axis_index, slice_indices);
    }

    template <class E>
    inline auto axis_view(const E& base, std::size_t axis_index,
                          const std::vector<label_type>& slice_labels)
    {
        return xaxis_view<std::decay_t<E>>(base, axis_index, slice_labels);
    }

} // namespace xframe

#endif // XFRAME_XAXIS_VIEW_HPP