//File 0351 : xframe/xvariable_masked_view.hpp
//Masked view for variables: filters elements using a boolean mask, provides lazy element access, SIMD‑accelerated gathering of unmasked values, and memory‑efficient representation.
#ifndef XFRAME_XVARIABLE_MASKED_VIEW_HPP
#define XFRAME_XVARIABLE_MASKED_VIEW_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xvariable.hpp"

namespace xframe
{
    /**
     * @class xvariable_masked_view
     * @brief Lazy view that filters a variable's elements using a boolean mask.
     *
     * Only elements whose mask value is true are visible. The view presents a
     * contiguous logical sequence of those elements.  Element access maps the
     * logical index to the underlying physical index via a pre‑computed offset
     * table.  SIMD loads gather the unmasked values into a batch.
     */
    template <class CT, class M>
    class xvariable_masked_view : public expression<xvariable_masked_view<CT, M>>
    {
    public:
        using self_type = xvariable_masked_view<CT, M>;
        using base_type = std::decay_t<CT>;
        using mask_type = std::decay_t<M>;
        using value_type = typename base_type::value_type;
        using const_reference = const value_type&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        /**
         * Construct the masked view.
         * @param base The base variable.
         * @param mask A boolean variable of the same size as base.
         */
        xvariable_masked_view(const base_type& base, const mask_type& mask)
            : m_base(base), m_mask(mask)
        {
            if (base.size() != mask.size())
                throw std::runtime_error("xvariable_masked_view: size mismatch between base and mask.");
            build_index_map();
        }

        xvariable_masked_view(const self_type&) = default;
        xvariable_masked_view& operator=(const self_type&) = default;
        xvariable_masked_view(self_type&&) = default;
        xvariable_masked_view& operator=(self_type&&) = default;

        /**
         * Number of unmasked (visible) elements.
         */
        size_type size() const noexcept { return m_visible_size; }

        /**
         * Element access by logical index.
         */
        const_reference operator[](size_type logical_idx) const
        {
            if (logical_idx >= m_visible_size)
                throw std::out_of_range("xvariable_masked_view: index out of bounds.");
            return m_base[m_index_map[logical_idx]];
        }

        /**
         * Direct access to the underlying base and mask.
         */
        const base_type& base() const noexcept { return m_base; }
        const mask_type& mask() const noexcept { return m_mask; }

        /**
         * Check if a physical index is unmasked.
         */
        bool is_unmasked(size_type phys_idx) const
        {
            return m_mask[phys_idx];
        }

        /**
         * Fill unmasked elements of the base with a value.
         */
        void fill_unmasked(value_type val)
        {
            for (size_type i = 0; i < m_visible_size; ++i)
                const_cast<value_type&>(m_base[m_index_map[i]]) = val;
        }

        /**
         * Collect all unmasked values into a new variable.
         */
        auto unmasked_values() const
        {
            variable<value_type> result(m_visible_size, m_base.name() + "_unmasked");
            for (size_type i = 0; i < m_visible_size; ++i)
                result[i] = m_base[m_index_map[i]];
            return result;
        }

        /**
         * Count of masked (hidden) elements.
         */
        size_type masked_count() const noexcept { return m_base.size() - m_visible_size; }

        /**
         * SIMD load: gather unmasked values.
         */
        template <class Align, class T = value_type>
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
        const mask_type& m_mask;
        std::vector<size_type> m_index_map;
        size_type m_visible_size = 0;

        /**
         * Build a mapping from logical index to physical index for all unmasked entries.
         */
        void build_index_map()
        {
            m_index_map.clear();
            m_index_map.reserve(m_base.size());
            for (size_type i = 0; i < m_base.size(); ++i)
                if (m_mask[i])
                    m_index_map.push_back(i);
            m_visible_size = m_index_map.size();
        }
    };

    /**
     * Free function to create a masked view of a variable.
     */
    template <class E, class M>
    inline auto masked_view(const variable<E>& base, const M& mask)
    {
        return xvariable_masked_view<variable<E>, M>(base, mask);
    }

} // namespace xframe

#endif // XFRAME_XVARIABLE_MASKED_VIEW_HPP