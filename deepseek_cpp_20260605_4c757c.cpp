//File 0323 : xframe/xaxis_index_slice.hpp
//Axis‑aligned integer‑index slice view: selects a subset of coordinates along one dimension, preserving other dimensions, with lazy SIMD element access.
#ifndef XFRAME_XAXIS_INDEX_SLICE_HPP
#define XFRAME_XAXIS_INDEX_SLICE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_coordinate.hpp"
#include "xframe_dimension.hpp"
#include "xframe.hpp"

namespace xframe
{
    /**
     * @class xaxis_index_slice
     * @brief Lazy view that restricts one dimension of an xframe to a given set of indices.
     *
     * For example, if the original xframe has dimensions (time=10, sensor=5),
     * `axis_index_slice(frame, 0, {0,2,4})` produces a view with time dimension
     * reduced to coordinates at indices 0, 2, and 4, while the sensor dimension
     * remains unchanged.
     */
    template <class CT>
    class xaxis_index_slice : public expression<xaxis_index_slice<CT>>
    {
    public:
        using self_type = xaxis_index_slice<CT>;
        using base_type = std::decay_t<CT>;
        using size_type = std::size_t;
        using label_type = typename base_type::label_type;

        /**
         * Construct the slice view using integer indices.
         * @param base The base xframe.
         * @param axis_index The dimension to slice.
         * @param indices The list of integer indices to keep.
         */
        xaxis_index_slice(const base_type& base,
                          size_type axis_index,
                          std::initializer_list<size_type> indices)
            : m_base(base), m_axis(axis_index), m_selected_indices(indices)
        {
            validate();
            build_view_dimensions();
        }

        /**
         * Construct the slice view using label values.
         * Labels are looked up in the coordinate of the specified axis.
         */
        xaxis_index_slice(const base_type& base,
                          size_type axis_index,
                          std::initializer_list<label_type> labels)
            : m_base(base), m_axis(axis_index)
        {
            const auto& coord = base.dimension(axis_index).coord();
            for (const auto& lbl : labels)
            {
                size_type idx = coord.find(lbl);
                if (idx >= coord.size())
                    throw std::out_of_range("xaxis_index_slice: label not found in dimension coordinate.");
                m_selected_indices.push_back(idx);
            }
            validate();
            build_view_dimensions();
        }

        /**
         * Construct the slice view using a vector of indices.
         */
        xaxis_index_slice(const base_type& base,
                          size_type axis_index,
                          const std::vector<size_type>& indices)
            : m_base(base), m_axis(axis_index), m_selected_indices(indices)
        {
            validate();
            build_view_dimensions();
        }

        xaxis_index_slice(const self_type&) = default;
        xaxis_index_slice& operator=(const self_type&) = default;
        xaxis_index_slice(self_type&&) = default;
        xaxis_index_slice& operator=(self_type&&) = default;

        std::size_t dimension_count() const noexcept { return m_view_dimensions.size(); }

        std::size_t size() const noexcept
        {
            std::size_t s = 1;
            for (const auto& d : m_view_dimensions) s *= d.size();
            return s;
        }

        const dimension<label_type>& dimension(std::size_t i) const { return m_view_dimensions[i]; }

        /**
         * Element access by integer coordinates.
         */
        template <class... Args>
        auto operator()(Args... args) const
        {
            std::array<size_type, sizeof...(Args)> idx{static_cast<size_type>(args)...};
            return element(idx.begin(), idx.end());
        }

        template <class... Args>
        auto operator()(Args... args)
        {
            return const_cast<const self_type*>(this)->operator()(args...);
        }

        /**
         * Element access by labels.
         */
        template <class... Args>
        auto locate(Args... labels) const
        {
            std::array<size_type, sizeof...(Args)> idx;
            map_labels_to_indices(idx, labels...);
            return element(idx.begin(), idx.end());
        }

        auto operator[](size_type i) const
        {
            auto idx = unravel_flat_index(i);
            return element(idx.begin(), idx.end());
        }

        auto operator[](size_type i)
        {
            return const_cast<const self_type*>(this)->operator[](i);
        }

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

        const base_type& base() const noexcept { return m_base; }
        const std::vector<size_type>& selected_indices() const noexcept { return m_selected_indices; }

    private:
        const base_type& m_base;
        size_type m_axis;
        std::vector<size_type> m_selected_indices;
        std::vector<dimension<label_type>> m_view_dimensions;

        void validate()
        {
            std::size_t ndim = m_base.dimension_count();
            if (m_axis >= ndim)
                throw std::out_of_range("xaxis_index_slice: axis index out of bounds.");
            for (auto idx : m_selected_indices)
                if (idx >= m_base.dimension(m_axis).size())
                    throw std::out_of_range("xaxis_index_slice: selected index out of bounds.");
        }

        void build_view_dimensions()
        {
            std::size_t ndim = m_base.dimension_count();
            m_view_dimensions.clear();
            for (std::size_t d = 0; d < ndim; ++d)
            {
                if (d == m_axis)
                {
                    // Build coordinate from selected indices
                    coordinate<label_type> new_coord;
                    for (auto idx : m_selected_indices)
                        new_coord.push_back(m_base.dimension(d).coord()[idx]);
                    m_view_dimensions.emplace_back(m_base.dimension(d).name(), std::move(new_coord),
                                                   m_base.dimension(d).unit(), m_base.dimension(d).description());
                }
                else
                {
                    m_view_dimensions.push_back(m_base.dimension(d));
                }
            }
        }

        template <class It>
        auto element(It first, It last) const
        {
            std::size_t ndim = m_view_dimensions.size();
            std::vector<size_type> base_idx(ndim);
            for (std::size_t d = 0; d < ndim; ++d)
            {
                if (d == m_axis)
                    base_idx[d] = m_selected_indices[*first++];
                else
                    base_idx[d] = *first++;
            }
            return call_base(base_idx, std::make_index_sequence<sizeof...(first)>{});
        }

        template <std::size_t... I>
        auto call_base(const std::vector<size_type>& idx, std::index_sequence<I...>) const
        {
            return m_base(idx[I]...);
        }

        std::vector<size_type> unravel_flat_index(size_type flat) const
        {
            std::size_t ndim = m_view_dimensions.size();
            std::vector<size_type> idx(ndim);
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(ndim) - 1; i >= 0; --i)
            {
                idx[static_cast<std::size_t>(i)] = flat % m_view_dimensions[static_cast<std::size_t>(i)].size();
                flat /= m_view_dimensions[static_cast<std::size_t>(i)].size();
            }
            return idx;
        }

        template <class... Labels>
        void map_labels_to_indices(std::array<size_type, sizeof...(Labels)>& idx, Labels... labels) const
        {
            std::size_t pos = 0;
            ((idx[pos++] = m_view_dimensions[pos].coord().find(labels)), ...);
        }
    };

    /**
     * Free function to create an axis index slice view.
     */
    template <class E>
    inline auto axis_index_slice(const E& base, std::size_t axis,
                                 std::initializer_list<std::size_t> indices)
    {
        return xaxis_index_slice<std::decay_t<E>>(base, axis, indices);
    }

    template <class E>
    inline auto axis_label_slice(const E& base, std::size_t axis,
                                 std::initializer_list<label_type> labels)
    {
        return xaxis_index_slice<std::decay_t<E>>(base, axis, labels);
    }

} // namespace xframe

#endif // XFRAME_XAXIS_INDEX_SLICE_HPP