//File 0344 : xframe/xexpand_dims_view.hpp
//Expand dimensions view: inserts a new axis of size 1 into an existing xframe, enabling broadcasting for simulations with SIMD‑accelerated access.
#ifndef XFRAME_XEXPAND_DIMS_VIEW_HPP
#define XFRAME_XEXPAND_DIMS_VIEW_HPP

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
#include "xframe.hpp"

namespace xframe
{
    /**
     * @class xexpand_dims_view
     * @brief Lazy view that adds a new dimension of size 1 at a specified axis.
     *
     * The new dimension has a single coordinate label (default "new_axis")
     * and the underlying data is unchanged. All element accesses treat the
     * new dimension as having only index 0, effectively broadcasting the
     * original data along that dimension. This is useful for aligning arrays
     * with different ranks in arithmetic operations.
     */
    template <class CT>
    class xexpand_dims_view : public expression<xexpand_dims_view<CT>>
    {
    public:
        using self_type = xexpand_dims_view<CT>;
        using base_type = std::decay_t<CT>;
        using size_type = std::size_t;
        using label_type = typename base_type::label_type;

        /**
         * Construct the expanded view.
         * @param base The base xframe (must outlive this view).
         * @param axis_index Position where the new axis is inserted (0..ndim).
         * @param new_axis_name Name of the new axis.
         */
        xexpand_dims_view(const base_type& base,
                          size_type axis_index,
                          const label_type& new_axis_name = label_type("new_axis"))
            : m_base(base), m_axis(axis_index)
        {
            std::size_t ndim = base.dimension_count();
            if (axis_index > ndim)
                throw std::out_of_range("xexpand_dims_view: axis index out of bounds.");
            // Build view dimensions: insert a new size‑1 dimension at axis_index.
            for (std::size_t d = 0; d < ndim; ++d)
            {
                if (d == axis_index)
                {
                    coordinate<label_type> single;
                    single.push_back(new_axis_name);
                    m_view_dimensions.emplace_back(new_axis_name, std::move(single));
                }
                m_view_dimensions.push_back(base.dimension(d));
            }
            // If axis_index == ndim, append at end.
            if (axis_index == ndim)
            {
                coordinate<label_type> single;
                single.push_back(new_axis_name);
                m_view_dimensions.emplace_back(new_axis_name, std::move(single));
            }
        }

        xexpand_dims_view(const self_type&) = default;
        xexpand_dims_view& operator=(const self_type&) = default;
        xexpand_dims_view(self_type&&) = default;
        xexpand_dims_view& operator=(self_type&&) = default;

        std::size_t dimension_count() const noexcept { return m_view_dimensions.size(); }

        std::size_t size() const noexcept { return m_base.size(); }

        const dimension<label_type>& dimension(std::size_t i) const
        {
            if (i >= m_view_dimensions.size())
                throw std::out_of_range("xexpand_dims_view: dimension index out of range.");
            return m_view_dimensions[i];
        }

        /**
         * Element access: the new dimension always has index 0.
         * The indices passed must include the new axis coordinate (which is ignored).
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

        auto operator[](size_type flat) const
        {
            auto idx = unravel_flat_index(flat);
            return element(idx.begin(), idx.end());
        }

        auto operator[](size_type flat)
        {
            return const_cast<const self_type*>(this)->operator[](flat);
        }

        template <class... Args>
        auto locate(Args... labels) const
        {
            std::array<size_type, sizeof...(Args)> idx;
            map_labels_to_indices(idx, labels...);
            return element(idx.begin(), idx.end());
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
        size_type expanded_axis() const noexcept { return m_axis; }

    private:
        const base_type& m_base;
        size_type m_axis;
        std::vector<dimension<label_type>> m_view_dimensions;

        template <class It>
        auto element(It first, It last) const
        {
            // Remove the new axis coordinate (which is always 0)
            std::vector<size_type> base_idx;
            std::size_t ndim_view = m_view_dimensions.size();
            for (std::size_t d = 0; d < ndim_view; ++d)
            {
                if (d == m_axis)
                {
                    // Skip the new axis index
                    ++first;
                    continue;
                }
                base_idx.push_back(*first++);
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
            // Same flat index because the new dim size is 1.
            return detail::unravel_index(flat, m_base);
        }

        template <class... Labels>
        void map_labels_to_indices(std::array<size_type, sizeof...(Labels)>& idx, Labels... labels) const
        {
            std::size_t pos = 0;
            for (std::size_t d = 0; d < m_view_dimensions.size(); ++d)
            {
                if (d == m_axis) { idx[pos++] = 0; continue; }
                idx[pos++] = m_view_dimensions[d].index_of(labels);
            }
        }
    };

    /**
     * Free function to create an expanded‑dimension view.
     */
    template <class E>
    inline auto expand_dims(const E& base, std::size_t axis,
                            const label_type& name = label_type("new_axis"))
    {
        return xexpand_dims_view<std::decay_t<E>>(base, axis, name);
    }

} // namespace xframe

#endif // XFRAME_XEXPAND_DIMS_VIEW_HPP