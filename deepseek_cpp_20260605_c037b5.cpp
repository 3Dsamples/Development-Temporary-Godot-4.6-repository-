//File 0324 : xframe/xaxis_label_slice.hpp
//Lazy view that restricts one dimension of an xframe to a given set of coordinate labels, with SIMD element access and label‑based lookup.
#ifndef XFRAME_XAXIS_LABEL_SLICE_HPP
#define XFRAME_XAXIS_LABEL_SLICE_HPP

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
     * @class xaxis_label_slice
     * @brief Lazy view that selects a subset of coordinates along one dimension
     *        using explicit label values.
     *
     * Unlike xaxis_index_slice, this class is specialized for label‑based selection,
     * accepting a container of labels. The resulting view retains the same dimension
     * names and unit metadata but with a reduced coordinate array.
     */
    template <class CT>
    class xaxis_label_slice : public expression<xaxis_label_slice<CT>>
    {
    public:
        using self_type = xaxis_label_slice<CT>;
        using base_type = std::decay_t<CT>;
        using size_type = std::size_t;
        using label_type = typename base_type::label_type;

        /**
         * Construct from the base xframe, an axis index, and a list of labels.
         * The labels must be present in the coordinate of the specified axis.
         */
        xaxis_label_slice(const base_type& base,
                          size_type axis_index,
                          std::initializer_list<label_type> labels)
            : m_base(base), m_axis(axis_index)
        {
            const auto& coord = base.dimension(axis_index).coord();
            for (const auto& lbl : labels)
            {
                size_type idx = coord.find(lbl);
                if (idx >= coord.size())
                    throw std::out_of_range("xaxis_label_slice: label '" + lbl + "' not found in dimension coordinate.");
                m_selected_indices.push_back(idx);
            }
            validate();
            build_view_dimensions();
        }

        /**
         * Construct using a vector of labels.
         */
        xaxis_label_slice(const base_type& base,
                          size_type axis_index,
                          const std::vector<label_type>& labels)
            : m_base(base), m_axis(axis_index)
        {
            const auto& coord = base.dimension(axis_index).coord();
            for (const auto& lbl : labels)
            {
                size_type idx = coord.find(lbl);
                if (idx >= coord.size())
                    throw std::out_of_range("xaxis_label_slice: label '" + lbl + "' not found.");
                m_selected_indices.push_back(idx);
            }
            validate();
            build_view_dimensions();
        }

        xaxis_label_slice(const self_type&) = default;
        xaxis_label_slice& operator=(const self_type&) = default;
        xaxis_label_slice(self_type&&) = default;
        xaxis_label_slice& operator=(self_type&&) = default;

        std::size_t dimension_count() const noexcept { return m_view_dimensions.size(); }

        std::size_t size() const noexcept
        {
            std::size_t s = 1;
            for (const auto& d : m_view_dimensions) s *= d.size();
            return s;
        }

        const dimension<label_type>& dimension(std::size_t i) const { return m_view_dimensions[i]; }

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
            if (m_axis >= m_base.dimension_count())
                throw std::out_of_range("xaxis_label_slice: axis index out of bounds.");
        }

        void build_view_dimensions()
        {
            std::size_t ndim = m_base.dimension_count();
            m_view_dimensions.clear();
            for (std::size_t d = 0; d < ndim; ++d)
            {
                if (d == m_axis)
                {
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
     * Free function to create a label‑based axis slice.
     */
    template <class E>
    inline auto axis_label_slice(const E& base, std::size_t axis,
                                 std::initializer_list<label_type> labels)
    {
        return xaxis_label_slice<std::decay_t<E>>(base, axis, labels);
    }

    template <class E>
    inline auto axis_label_slice(const E& base, std::size_t axis,
                                 const std::vector<label_type>& labels)
    {
        return xaxis_label_slice<std::decay_t<E>>(base, axis, labels);
    }

} // namespace xframe

#endif // XFRAME_XAXIS_LABEL_SLICE_HPP