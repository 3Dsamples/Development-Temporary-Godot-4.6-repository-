//File 0361 : xframe/xvariable_view.hpp
//Variable view: a lazy non‑owning view into a sub‑range of an existing variable, with SIMD‑accelerated access, step semantics, and expression integration.
#ifndef XFRAME_XVARIABLE_VIEW_HPP
#define XFRAME_XVARIABLE_VIEW_HPP

#include <algorithm>
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
#include "xvariable.hpp"

namespace xframe
{
    /**
     * @class xvariable_view
     * @brief Lazy view representing a sub‑range of a variable.
     *
     * The view stores a reference to the base variable and a range
     * (start, stop, step). Element access maps the logical index back
     * to the base variable using the range parameters. No data is copied.
     * SIMD loads are supported for contiguous sub‑ranges (step=1).
     */
    template <class T = double, class L = label_type>
    class xvariable_view : public expression<xvariable_view<T, L>>
    {
    public:
        using self_type = xvariable_view<T, L>;
        using base_type = variable<T, L>;
        using value_type = T;
        using const_reference = const T&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using label_type = L;

        /**
         * Construct a view of the entire variable.
         */
        explicit xvariable_view(const base_type& base) noexcept
            : m_base(&base), m_start(0), m_stop(base.size()), m_step(1), m_size(base.size())
        {
        }

        /**
         * Construct a view of a sub‑range [start, stop) with step.
         */
        xvariable_view(const base_type& base,
                       size_type start,
                       size_type stop,
                       size_type step = 1)
            : m_base(&base), m_start(start), m_stop(stop), m_step(step)
        {
            if (m_step == 0)
                throw std::runtime_error("xvariable_view: step must not be zero.");
            if (m_start > m_stop || m_stop > base.size())
                throw std::out_of_range("xvariable_view: start/stop out of bounds.");
            m_size = (m_stop - m_start + m_step - 1) / m_step;
        }

        xvariable_view(const self_type&) = default;
        xvariable_view& operator=(const self_type&) = default;
        xvariable_view(self_type&&) = default;
        xvariable_view& operator=(self_type&&) = default;

        /**
         * Number of elements in this view.
         */
        size_type size() const noexcept { return m_size; }
        bool empty() const noexcept { return m_size == 0; }

        /**
         * Name of the view (inherits from base).
         */
        const label_type& name() const noexcept { return m_base->name(); }

        /**
         * Element access: maps logical index i to base index m_start + i * m_step.
         */
        const_reference operator[](size_type i) const
        {
            if (i >= m_size)
                throw std::out_of_range("xvariable_view: index out of bounds.");
            return (*m_base)[m_start + i * m_step];
        }

        /**
         * SIMD load: if contiguous (step=1), load directly from base memory.
         */
        template <class Align, class U = T>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<U, default_simd_arch>;
            if (m_step == 1)
            {
                return simd_type::load_unaligned(m_base->data() + m_start + i);
            }
            constexpr std::size_t simd_size = simd_type::size;
            alignas(64) std::array<U, simd_size> buf;
            for (std::size_t k = 0; k < simd_size; ++k)
                buf[k] = static_cast<U>((*this)[i + k]);
            return simd_type::load_aligned(buf.data());
        }

        /**
         * Fill the view elements with a value (modifies the underlying variable).
         */
        void fill(const_reference val)
        {
            for (size_type i = 0; i < m_size; ++i)
                const_cast<T&>((*m_base)[m_start + i * m_step]) = val;
        }

        /**
         * Convert to a concrete variable (materialize).
         */
        base_type materialize() const
        {
            base_type result(m_size, m_base->name());
            if (m_step == 1)
            {
                std::copy(m_base->data() + m_start,
                          m_base->data() + m_stop,
                          result.data());
            }
            else
            {
                for (size_type i = 0; i < m_size; ++i)
                    result[i] = (*m_base)[m_start + i * m_step];
            }
            return result;
        }

        const base_type& base() const noexcept { return *m_base; }
        size_type start() const noexcept { return m_start; }
        size_type stop() const noexcept { return m_stop; }
        size_type step() const noexcept { return m_step; }

    private:
        const base_type* m_base;
        size_type m_start;
        size_type m_stop;
        size_type m_step;
        size_type m_size;
    };

    /**
     * Helper to create a variable view from a range.
     */
    template <class T, class L>
    inline auto view(const variable<T, L>& var,
                     std::size_t start, std::size_t stop, std::size_t step = 1)
    {
        return xvariable_view<T, L>(var, start, stop, step);
    }

    /**
     * Helper to create a view of the entire variable.
     */
    template <class T, class L>
    inline auto view(const variable<T, L>& var)
    {
        return xvariable_view<T, L>(var);
    }

} // namespace xframe

#endif // XFRAME_XVARIABLE_VIEW_HPP