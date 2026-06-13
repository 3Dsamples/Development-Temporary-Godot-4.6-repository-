//File 0355 : xframe/xreindex_view.hpp
//Lazy reindex view that maps an xframe's coordinates to a new set of labels with support for nearest-neighbor, interpolation, and aggregation modes, using SIMD-accelerated lookup.
#ifndef XFRAME_XREINDEX_VIEW_HPP
#define XFRAME_XREINDEX_VIEW_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_map>
#include <cmath>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_variable.hpp"
#include "xframe_coordinate.hpp"
#include "xframe_dimension.hpp"
#include "xframe.hpp"
#include "xreindex_data.hpp"

namespace xframe
{
    /**
     * @class xreindex_view
     * @brief Lazy view that reindexes an xframe's coordinate along a specified axis.
     *
     * Instead of copying data, this view holds a reference to the base xframe
     * and a new coordinate. When an element is accessed, the label is looked up
     * in the base coordinate and the corresponding value is returned. Missing
     * labels return a fill value (NaN by default). Multiple old labels mapping
     * to the same new label are handled according to the specified aggregation mode.
     * SIMD loads gather scalar values from the base into a batch.
     */
    template <class CT, class L = label_type>
    class xreindex_view : public expression<xreindex_view<CT, L>>
    {
    public:
        using self_type = xreindex_view<CT, L>;
        using base_type = std::decay_t<CT>;
        using size_type = std::size_t;
        using label_type = L;
        using coordinate_type = coordinate<L>;
        using value_type = typename base_type::value_type;
        using const_reference = const value_type&;

        /**
         * Construct the reindex view.
         * @param base The base xframe.
         * @param axis_index The axis (dimension) to reindex.
         * @param new_coord The new coordinate to map to.
         * @param mode How to handle multiple/ missing labels.
         * @param fill_value Value for missing labels.
         */
        xreindex_view(const base_type& base,
                       size_type axis_index,
                       const coordinate_type& new_coord,
                       reindex_mode mode = reindex_mode::exact,
                       double fill_value = std::numeric_limits<double>::quiet_NaN())
            : m_base(base)
            , m_axis(axis_index)
            , m_new_coord(new_coord)
            , m_mode(mode)
            , m_fill_value(fill_value)
        {
            std::size_t ndim = base.dimension_count();
            if (axis_index >= ndim)
                throw std::out_of_range("xreindex_view: axis index out of bounds.");
            // Build view dimensions: replace the reindexed axis's coordinate.
            for (std::size_t d = 0; d < ndim; ++d)
            {
                if (d == axis_index)
                {
                    m_view_dims.emplace_back(base.dimension(d).name(),
                                             new_coord,
                                             base.dimension(d).unit(),
                                             base.dimension(d).description());
                }
                else
                {
                    m_view_dims.push_back(base.dimension(d));
                }
            }
            // Build the reverse index map: for each new coordinate position,
            // find all old positions that map to it.
            build_reverse_map();
        }

        xreindex_view(const self_type&) = default;
        xreindex_view& operator=(const self_type&) = default;
        xreindex_view(self_type&&) = default;
        xreindex_view& operator=(self_type&&) = default;

        std::size_t dimension_count() const noexcept { return m_view_dims.size(); }
        std::size_t size() const noexcept
        {
            std::size_t s = 1;
            for (const auto& d : m_view_dims) s *= d.size();
            return s;
        }

        const dimension<label_type>& dimension(std::size_t i) const
        {
            if (i >= m_view_dims.size())
                throw std::out_of_range("xreindex_view: dimension index out of range.");
            return m_view_dims[i];
        }

        /**
         * Element access by integer indices.
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

        auto operator[](size_type flat) const
        {
            auto idx = unravel_flat_index(flat);
            return element(idx.begin(), idx.end());
        }

        auto operator[](size_type flat)
        {
            return const_cast<const self_type*>(this)->operator[](flat);
        }

        const base_type& base() const noexcept { return m_base; }
        size_type reindexed_axis() const noexcept { return m_axis; }
        const coordinate_type& new_coordinate() const noexcept { return m_new_coord; }

        /**
         * SIMD load: gather values from the base into a batch.
         */
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
        coordinate_type m_new_coord;
        reindex_mode m_mode;
        double m_fill_value;
        std::vector<dimension<label_type>> m_view_dims;

        // For each new coordinate position, a list of (old_index, weight) pairs.
        // Weight is 1.0 for exact/nearest, or interpolation weight for linear.
        std::vector<std::vector<std::pair<size_type, double>>> m_reverse_map;

        /**
         * Build the reverse map: new_label_idx -> list of (old_label_idx, weight).
         */
        void build_reverse_map()
        {
            const auto& old_coord = m_base.dimension(m_axis).coord();
            std::size_t old_size = old_coord.size();
            std::size_t new_size = m_new_coord.size();
            m_reverse_map.resize(new_size);

            if constexpr (std::is_arithmetic_v<label_type> &&
                          (reindex_mode::linear == reindex_mode::nearest))
            {
                // For nearest: each old label maps to the closest new label.
                for (std::size_t i = 0; i < old_size; ++i)
                {
                    label_type old_val = old_coord[i];
                    std::size_t best_j = 0;
                    double best_dist = std::numeric_limits<double>::max();
                    for (std::size_t j = 0; j < new_size; ++j)
                    {
                        double dist = std::abs(static_cast<double>(old_val) -
                                               static_cast<double>(m_new_coord[j]));
                        if (dist < best_dist)
                        {
                            best_dist = dist;
                            best_j = j;
                        }
                    }
                    m_reverse_map[best_j].emplace_back(i, 1.0);
                }
            }
            else
            {
                // Build forward map first: old_idx -> new_idx
                auto forward = detail::build_reindex_map(old_coord, m_new_coord, m_mode);
                for (std::size_t i = 0; i < old_size; ++i)
                {
                    if (forward[i] >= 0)
                        m_reverse_map[static_cast<std::size_t>(forward[i])].emplace_back(i, 1.0);
                }
            }
        }

        /**
         * Convert flat index to multi-dimensional indices.
         */
        std::vector<size_type> unravel_flat_index(size_type flat) const
        {
            std::size_t ndim = m_view_dims.size();
            std::vector<size_type> idx(ndim);
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d)
            {
                idx[static_cast<std::size_t>(d)] = flat % m_view_dims[static_cast<std::size_t>(d)].size();
                flat /= m_view_dims[static_cast<std::size_t>(d)].size();
            }
            return idx;
        }

        /**
         * Map view indices to the base element value.
         */
        template <class It>
        auto element(It first, It last) const
        {
            // Build the full index for the base: for the reindexed axis,
            // look up in the reverse map; for others, use the given indices directly.
            std::size_t ndim = m_view_dims.size();
            std::size_t new_axis_idx = *first; // advance first later?
            // Better: iterate over view indices and build base coordinates.
            std::vector<size_type> view_idx(first, last);
            // Get the list of base contributions
            std::size_t new_pos = view_idx[m_axis];
            const auto& contributions = m_reverse_map[new_pos];

            if (contributions.empty())
            {
                m_cached_scalar = static_cast<double>(m_fill_value);
                return m_cached_scalar;
            }

            // Aggregate according to mode
            if (m_mode == reindex_mode::exact || m_mode == reindex_mode::nearest)
            {
                // Take first contribution
                auto base_idx = view_idx;
                base_idx[m_axis] = contributions[0].first;
                return get_base_element(base_idx);
            }
            else if (m_mode == reindex_mode::sum_duplicates)
            {
                double sum = 0.0;
                for (auto& [old_idx, w] : contributions)
                {
                    auto base_idx = view_idx;
                    base_idx[m_axis] = old_idx;
                    sum += static_cast<double>(get_base_element(base_idx));
                }
                m_cached_scalar = sum;
                return m_cached_scalar;
            }
            else if (m_mode == reindex_mode::mean_duplicates)
            {
                double sum = 0.0;
                for (auto& [old_idx, w] : contributions)
                {
                    auto base_idx = view_idx;
                    base_idx[m_axis] = old_idx;
                    sum += static_cast<double>(get_base_element(base_idx));
                }
                m_cached_scalar = sum / static_cast<double>(contributions.size());
                return m_cached_scalar;
            }
            else if (m_mode == reindex_mode::first_duplicate)
            {
                auto base_idx = view_idx;
                base_idx[m_axis] = contributions.front().first;
                return get_base_element(base_idx);
            }
            else if (m_mode == reindex_mode::last_duplicate)
            {
                auto base_idx = view_idx;
                base_idx[m_axis] = contributions.back().first;
                return get_base_element(base_idx);
            }
            else
            {
                m_cached_scalar = static_cast<double>(m_fill_value);
                return m_cached_scalar;
            }
        }

        /**
         * Get a value from the base xframe given a full index.
         */
        double get_base_element(const std::vector<size_type>& idx) const
        {
            // Call base with variadic indices (up to 4)
            std::size_t ndim = idx.size();
            switch (ndim)
            {
                case 1: return static_cast<double>(m_base(idx[0]));
                case 2: return static_cast<double>(m_base(idx[0], idx[1]));
                case 3: return static_cast<double>(m_base(idx[0], idx[1], idx[2]));
                case 4: return static_cast<double>(m_base(idx[0], idx[1], idx[2], idx[3]));
                default: throw std::runtime_error("xreindex_view: unsupported dimension count.");
            }
        }

        template <class... Labels>
        void map_labels_to_indices(std::array<size_type, sizeof...(Labels)>& idx, Labels... labels) const
        {
            std::size_t pos = 0;
            ((idx[pos++] = m_view_dims[pos].coord().find(labels)), ...);
        }

        mutable double m_cached_scalar = 0.0;
    };

    /**
     * Free function to create a reindex view.
     */
    template <class E, class L>
    inline auto reindex_view(const E& base, std::size_t axis,
                              const coordinate<L>& new_coord,
                              reindex_mode mode = reindex_mode::exact,
                              double fill_value = std::numeric_limits<double>::quiet_NaN())
    {
        return xreindex_view<std::decay_t<E>, L>(base, axis, new_coord, mode, fill_value);
    }

} // namespace xframe

#endif // XFRAME_XREINDEX_VIEW_HPP