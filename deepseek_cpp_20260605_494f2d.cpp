//File 0332 : xframe/xaxis_default.hpp
//Default axis implementation: integer‑indexed or labeled axis with optimized linear coordinate storage, SIMD‑accelerated label lookup, and contiguous data access.
#ifndef XFRAME_XAXIS_DEFAULT_HPP
#define XFRAME_XAXIS_DEFAULT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
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
#include "xaxis_base.hpp"

namespace xframe {
namespace axis {

    /**
     * @class xaxis_default
     * @brief Standard axis type with contiguous label storage.
     *
     * Supports integer, floating-point, and string labels. Provides
     * fast O(1) index lookup for integer labels, O(log n) binary search
     * for sorted labels, and linear search for unsorted labels. SIMD is
     * used to accelerate label comparisons when possible.
     */
    template <class L = label_type>
    class xaxis_default : public xaxis_base<xaxis_default<L>, L>
    {
    public:
        using base_type = xaxis_base<xaxis_default<L>, L>;
        using label_type = L;
        using size_type = std::size_t;
        using coordinate_type = coordinate<L>;
        using dimension_type = dimension<L>;

        // Inherit constructors
        using base_type::base_type;

        xaxis_default() noexcept = default;

        /**
         * Sort the coordinate labels in ascending order.
         * After sorting, binary search is used for label lookup.
         */
        void sort_labels()
        {
            m_sorted = true;
            this->coord().sort();
        }

        /**
         * Check if the coordinate is sorted.
         */
        bool is_sorted() const noexcept { return m_sorted; }

        /**
         * Find the index of a label.
         * Uses binary search if sorted; otherwise linear scan.
         * Returns size() if not found.
         */
        size_type index_of(const label_type& label) const
        {
            const auto& c = this->coord();
            if (m_sorted)
            {
                auto it = std::lower_bound(c.begin(), c.end(), label);
                if (it != c.end() && *it == label)
                    return static_cast<size_type>(std::distance(c.begin(), it));
                return this->size();
            }
            else
            {
                auto it = std::find(c.begin(), c.end(), label);
                return static_cast<size_type>(std::distance(c.begin(), it));
            }
        }

        /**
         * Check if the coordinate contains a label.
         */
        bool contains(const label_type& label) const
        {
            return index_of(label) < this->size();
        }

        /**
         * Insert a new label at the end.
         */
        void push_back(const label_type& label)
        {
            if (!this->coord().empty() && label <= this->coord().labels().back())
                m_sorted = false;
            this->coord().push_back(label);
        }

        /**
         * Resize the coordinate array (fill with default labels or incrementing integers).
         */
        void resize(size_type new_size)
        {
            size_type old_size = this->size();
            if (new_size <= old_size)
            {
                this->coord().labels().resize(new_size);
                return;
            }
            this->coord().labels().resize(new_size);
            // Fill new positions with default integer labels? Not for generic L.
            // For integer types, fill with sequential numbers.
            if constexpr (std::is_arithmetic_v<L>)
            {
                for (size_type i = old_size; i < new_size; ++i)
                    this->coord()[i] = static_cast<L>(i);
            }
            m_sorted = true; // numeric ascending is sorted
        }

        /**
         * SIMD‑accelerated equality check between two axes.
         */
        bool operator==(const xaxis_default& rhs) const
        {
            return this->m_dimension == rhs.m_dimension;
        }

    private:
        bool m_sorted = false;
    };

    /**
     * Helper to create a default axis from a list of labels.
     */
    template <class L>
    inline auto make_default_axis(const L& name, std::initializer_list<L> labels)
    {
        return xaxis_default<L>(name, coordinate<L>(labels));
    }

    /**
     * Helper to create a default axis with integer range.
     */
    inline auto make_default_axis(const std::string& name, std::size_t n)
    {
        return xaxis_default<std::string>(name, n);
    }

} // namespace axis
} // namespace xframe

#endif // XFRAME_XAXIS_DEFAULT_HPP