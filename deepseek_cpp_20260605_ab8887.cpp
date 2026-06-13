//File 0311 : xframe/xframe_offset_view.hpp
//Offset view shifting the origin of an xframe by integer or label coordinates, with SIMD-accelerated element access and full expression integration.
#ifndef XFRAME_OFFSET_VIEW_HPP
#define XFRAME_OFFSET_VIEW_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_dimension.hpp"
#include "xframe_variable.hpp"
#include "xframe.hpp"

namespace xframe
{
    /**
     * @class xframe_offset_view
     * @brief View that shifts the origin of an xframe by given offsets.
     *
     * The resulting view has the same dimension names but reduced sizes.
     * Offsets can be specified as integer indices or label values.
     * Element access translates the view coordinates back to the original
     * by adding the offsets.
     */
    template <class CT>
    class xframe_offset_view : public expression<xframe_offset_view<CT>>
    {
    public:
        using self_type = xframe_offset_view<CT>;
        using base_type = std::decay_t<CT>;
        using size_type = std::size_t;
        using label_type = typename base_type::label_type;

        /**
         * Construct an offset view using integer offsets.
         * @param base The base xframe.
         * @param offsets Vector of integer offsets, one per dimension.
         */
        xframe_offset_view(const base_type& base,
                          const std::vector<std::ptrdiff_t>& offsets)
            : m_base(base), m_offsets(offsets.size())
        {
            std::size_t ndim = base.dimension_count();
            if (offsets.size() != ndim)
                throw std::runtime_error("xframe_offset_view: offset count must match dimension count.");
            for (std::size_t d = 0; d < ndim; ++d)
            {
                std::ptrdiff_t off = offsets[d];
                if (off < 0) off += static_cast<std::ptrdiff_t>(base.dimension(d).size());
                if (off < 0 || static_cast<size_type>(off) >= base.dimension(d).size())
                    throw std::out_of_range("xframe_offset_view: offset out of bounds.");
                m_offsets[d] = static_cast<size_type>(off);
                m_view_sizes.push_back(base.dimension(d).size() - m_offsets[d]);
                m_view_dimensions.push_back(dimension<label_type>(base.dimension(d).name(), m_view_sizes.back()));
            }
        }

        /**
         * Construct an offset view using label offsets.
         * @param base The base xframe.
         * @param label_offsets Vector of labels; each is the first coordinate in the view.
         */
        xframe_offset_view(const base_type& base,
                          const std::vector<label_type>& label_offsets)
            : m_base(base), m_offsets(label_offsets.size())
        {
            std::size_t ndim = base.dimension_count();
            if (label_offsets.size() != ndim)
                throw std::runtime_error("xframe_offset_view: label offset count mismatch.");
            for (std::size_t d = 0; d < ndim; ++d)
            {
                size_type idx = base.dimension(d).index_of(label_offsets[d]);
                if (idx >= base.dimension(d).size())
                    throw std::out_of_range("xframe_offset_view: label not found in dimension.");
                m_offsets[d] = idx;
                m_view_sizes.push_back(base.dimension(d).size() - idx);
                m_view_dimensions.push_back(dimension<label_type>(base.dimension(d).name(), m_view_sizes.back()));
            }
        }

        xframe_offset_view(const self_type&) = default;
        xframe_offset_view& operator=(const self_type&) = default;
        xframe_offset_view(self_type&&) = default;
        xframe_offset_view& operator=(self_type&&) = default;

        std::size_t dimension_count() const noexcept { return m_view_dimensions.size(); }
        std::size_t size() const noexcept
        {
            std::size_t s = 1;
            for (auto sz : m_view_sizes) s *= sz;
            return s;
        }

        const dimension<label_type>& dimension(std::size_t i) const { return m_view_dimensions[i]; }

        template <class... Args>
        auto operator()(Args... args) const
        {
            std::array<std::size_t, sizeof...(Args)> idx{static_cast<std::size_t>(args)...};
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
            std::array<std::size_t, sizeof...(Args)> idx;
            map_labels_to_indices(idx, labels...);
            return element(idx.begin(), idx.end());
        }

        auto operator[](std::size_t i) const
        {
            auto idx = unravel_index(i, m_view_sizes);
            return element(idx.begin(), idx.end());
        }

        auto operator[](std::size_t i)
        {
            return const_cast<const self_type*>(this)->operator[](i);
        }

        const base_type& base() const noexcept { return m_base; }
        const std::vector<size_type>& offsets() const noexcept { return m_offsets; }

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
        std::vector<size_type> m_offsets;
        std::vector<size_type> m_view_sizes;
        std::vector<dimension<label_type>> m_view_dimensions;

        template <class It>
        auto element(It first, It last) const
        {
            std::vector<std::size_t> base_idx;
            for (size_type d = 0; d < m_view_dimensions.size(); ++d)
                base_idx.push_back(m_offsets[d] + *first++);
            // Call base with variadic indices
            return call_base_with_indices(base_idx, std::make_index_sequence<sizeof...(first)>{});
        }

        template <std::size_t... I>
        auto call_base_with_indices(const std::vector<std::size_t>& idx, std::index_sequence<I...>) const
        {
            return m_base(idx[I]...);
        }

        template <class... Labels>
        void map_labels_to_indices(std::array<std::size_t, sizeof...(Labels)>& idx, Labels... labels) const
        {
            std::size_t pos = 0;
            ((idx[pos++] = m_view_dimensions[pos].index_of(labels)), ...);
        }
    };

    /**
     * Helper to create an offset view by integer offsets.
     */
    template <class E>
    inline auto offset_view(const E& base, const std::vector<std::ptrdiff_t>& offsets)
    {
        return xframe_offset_view<std::decay_t<E>>(base, offsets);
    }

    /**
     * Helper to create an offset view by label offsets.
     */
    template <class E>
    inline auto offset_view_by_label(const E& base, const std::vector<label_type>& labels)
    {
        return xframe_offset_view<std::decay_t<E>>(base, labels);
    }

} // namespace xframe

#endif // XFRAME_OFFSET_VIEW_HPP