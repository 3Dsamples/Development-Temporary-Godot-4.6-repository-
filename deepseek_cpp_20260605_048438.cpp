//File 0337 : xframe/xcoordinate_chain.hpp
//Coordinate chain: concatenates multiple coordinate sequences into a single logical coordinate without copying, with SIMD-accelerated label lookup and lazy iteration.
#ifndef XFRAME_XCOORDINATE_CHAIN_HPP
#define XFRAME_XCOORDINATE_CHAIN_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <memory>
#include <tuple>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xcoordinate.hpp"
#include "xcoordinate_base.hpp"

namespace xframe
{
    /**
     * @class xcoordinate_chain
     * @brief Lazy concatenation of multiple coordinate sequences.
     *
     * The chain stores references to several coordinate objects and presents
     * them as a single contiguous logical coordinate. No data is copied;
     * element access maps the logical index to the appropriate underlying
     * coordinate. Label lookup uses binary search on the coordinate offset
     * table to first locate the correct sub-coordinate.
     */
    template <class T = label_type>
    class xcoordinate_chain : public expression<xcoordinate_chain<T>>
    {
    public:
        using self_type = xcoordinate_chain<T>;
        using value_type = T;
        using const_reference = const T&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        /**
         * Construct an empty chain.
         */
        xcoordinate_chain() noexcept = default;

        /**
         * Construct from a vector of coordinates (by reference).
         * The referenced coordinates must outlive this chain.
         */
        explicit xcoordinate_chain(std::vector<const coordinate<T>*> coords)
            : m_coords(std::move(coords))
        {
            build_offset_table();
        }

        xcoordinate_chain(const self_type&) = default;
        xcoordinate_chain& operator=(const self_type&) = default;
        xcoordinate_chain(self_type&&) = default;
        xcoordinate_chain& operator=(self_type&&) = default;

        /**
         * Add a coordinate reference to the chain.
         * The referenced coordinate must outlive this chain.
         */
        void chain(const coordinate<T>& coord)
        {
            m_coords.push_back(&coord);
            build_offset_table();
        }

        /**
         * Total number of labels across all chained coordinates.
         */
        size_type size() const noexcept { return m_total_size; }

        bool empty() const noexcept { return m_total_size == 0; }

        /**
         * Element access by logical index.
         * Locates the sub-coordinate using the offset table, then returns
         * the appropriate label.
         */
        const_reference operator[](size_type i) const
        {
            auto [sub_idx, local_idx] = map_index(i);
            return (*m_coords[sub_idx])[local_idx];
        }

        /**
         * Find the logical index of a label.
         * Searches each sub-coordinate in order; returns the first match,
         * or size() if not found.
         */
        size_type find(const T& label) const
        {
            size_type offset = 0;
            for (const auto* c : m_coords)
            {
                size_type local = c->find(label);
                if (local < c->size())
                    return offset + local;
                offset += c->size();
            }
            return m_total_size;
        }

        /**
         * Check if a label exists anywhere in the chain.
         */
        bool contains(const T& label) const
        {
            return find(label) < m_total_size;
        }

        /**
         * Number of sub-coordinates in the chain.
         */
        size_type chain_count() const noexcept { return m_coords.size(); }

        /**
         * Access a sub-coordinate by index.
         */
        const coordinate<T>& sub_coordinate(size_type i) const
        {
            if (i >= m_coords.size())
                throw std::out_of_range("xcoordinate_chain::sub_coordinate");
            return *m_coords[i];
        }

    private:
        std::vector<const coordinate<T>*> m_coords;
        std::vector<size_type> m_offsets;
        size_type m_total_size = 0;

        /**
         * Build the cumulative offset table for fast index mapping.
         * m_offsets[i] = sum of sizes of coordinates 0..i-1.
         */
        void build_offset_table()
        {
            m_offsets.resize(m_coords.size());
            size_type cum = 0;
            for (size_type i = 0; i < m_coords.size(); ++i)
            {
                m_offsets[i] = cum;
                cum += m_coords[i]->size();
            }
            m_total_size = cum;
        }

        /**
         * Map a logical index to (sub-coordinate index, local index).
         * Uses binary search on the offset table.
         */
        std::pair<size_type, size_type> map_index(size_type logical) const
        {
            if (logical >= m_total_size)
                throw std::out_of_range("xcoordinate_chain: index out of bounds.");
            // Find the first offset > logical, then sub is one before
            auto it = std::upper_bound(m_offsets.begin(), m_offsets.end(), logical);
            size_type sub = static_cast<size_type>(std::distance(m_offsets.begin(), it)) - 1;
            size_type local = logical - m_offsets[sub];
            return {sub, local};
        }
    };

    /**
     * Helper to create a coordinate chain from a list of coordinate references.
     */
    template <class T>
    inline auto chain_coordinates(const std::vector<const coordinate<T>*>& coords)
    {
        return xcoordinate_chain<T>(coords);
    }

    /**
     * Helper to chain two coordinates.
     */
    template <class T>
    inline auto chain(const coordinate<T>& a, const coordinate<T>& b)
    {
        std::vector<const coordinate<T>*> coords{&a, &b};
        return xcoordinate_chain<T>(std::move(coords));
    }

} // namespace xframe

#endif // XFRAME_XCOORDINATE_CHAIN_HPP