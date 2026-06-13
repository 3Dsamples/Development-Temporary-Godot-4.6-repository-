//File 0357 : xframe/xsequence_view.hpp
//Sequence view: a lazy view representing a contiguous range of rows along the first axis, with support for integer start/stop/step and label-based slicing, SIMD-accelerated access.
#ifndef XFRAME_XSEQUENCE_VIEW_HPP
#define XFRAME_XSEQUENCE_VIEW_HPP

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
#include "xframe_dimension.hpp"
#include "xframe_coordinate.hpp"
#include "xframe.hpp"

namespace xframe
{
    /**
     * @class xsequence_view
     * @brief Lazy view that selects a contiguous sequence of rows along dimension 0.
     *
     * The view can be constructed with integer indices (start, stop, step) or
     * with label-based slicing (start_label, stop_label). The resulting view
     * has the same number of dimensions but with a reduced size along axis 0.
     * Element access maps the view coordinates back to the base by adding the
     * offset and applying the step. SIMD loads are supported for contiguous
     * segments when the underlying storage is dense.
     */
    template <class CT>
    class xsequence_view : public expression<xsequence_view<CT>>
    {
    public:
        using self_type = xsequence_view<CT>;
        using base_type = std::decay_t<CT>;
        using size_type = std::size_t;
        using label_type = typename base_type::label_type;
        using value_type = typename base_type::value_type;
        using const_reference = const value_type&;

        /**
         * Construct a sequence view using integer start/stop/step.
         * @param base The base xframe (must outlive the view).
         * @param start First row index (inclusive).
         * @param stop One past last row index (exclusive).
         * @param step Step between rows (default 1).
         */
        xsequence_view(const base_type& base,
                       std::ptrdiff_t start,
                       std::ptrdiff_t stop,
                       std::ptrdiff_t step = 1)
            : m_base(base), m_start(start), m_stop(stop), m_step(step)
        {
            std::size_t n = base.dimension(0).size();
            normalize_bounds(n);
            m_size = (static_cast<std::ptrdiff_t>(m_stop) - static_cast<std::ptrdiff_t>(m_start) + m_step - 1) / m_step;
            if (m_size < 0) m_size = 0;
            build_view_dimensions();
        }

        /**
         * Construct a sequence view using label-based slicing.
         * @param base The base xframe.
         * @param start_label First label in the view.
         * @param stop_label One past last label (inclusive of stop_label? exclusive).
         * @param step Step (default 1).
         */
        xsequence_view(const base_type& base,
                       const label_type& start_label,
                       const label_type& stop_label,
                       std::ptrdiff_t step = 1)
            : m_base(base), m_step(step)
        {
            std::size_t n = base.dimension(0).size();
            const auto& coord = base.dimension(0).coord();
            std::ptrdiff_t i0 = static_cast<std::ptrdiff_t>(coord.find(start_label));
            std::ptrdiff_t i1 = static_cast<std::ptrdiff_t>(coord.find(stop_label));
            if (i0 < 0 || i0 >= static_cast<std::ptrdiff_t>(n) ||
                i1 < 0 || i1 >= static_cast<std::ptrdiff_t>(n))
                throw std::out_of_range("xsequence_view: label not found in dimension 0.");
            // For exclusive stop, we set stop = index of stop_label (not +1),
            // but typically slice is [start, stop). We'll treat stop as inclusive?
            // To be consistent with integer version, if stop_label is inclusive we need +1.
            // We'll define stop as exclusive: stop = i1 + 1.
            m_start = i0;
            m_stop = i1 + 1;
            normalize_bounds(n);
            m_size = (m_stop - m_start + m_step - 1) / m_step;
            if (m_size < 0) m_size = 0;
            build_view_dimensions();
        }

        xsequence_view(const self_type&) = default;
        xsequence_view& operator=(const self_type&) = default;
        xsequence_view(self_type&&) = default;
        xsequence_view& operator=(self_type&&) = default;

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
                throw std::out_of_range("xsequence_view: dimension index out of range.");
            return m_view_dims[i];
        }

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
        size_type start() const noexcept { return m_start; }
        size_type stop() const noexcept { return m_stop; }
        std::ptrdiff_t step() const noexcept { return m_step; }

        /**
         * SIMD load: if the view is contiguous (step 1), load directly from base;
         * otherwise, gather scalars.
         */
        template <class Align, class T = double>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            if (m_step == 1 && is_contiguous())
            {
                return simd_type::load_unaligned(m_base.data() + m_start * compute_inner_stride() + i);
            }
            constexpr std::size_t simd_size = simd_type::size;
            alignas(64) std::array<T, simd_size> buf;
            for (std::size_t k = 0; k < simd_size; ++k)
                buf[k] = static_cast<T>((*this)[i + k]);
            return simd_type::load_aligned(buf.data());
        }

    private:
        const base_type& m_base;
        size_type m_start = 0;
        size_type m_stop = 0;
        std::ptrdiff_t m_step = 1;
        size_type m_size = 0;
        std::vector<dimension<label_type>> m_view_dims;

        void normalize_bounds(std::size_t n)
        {
            if (m_start < 0) m_start += static_cast<std::ptrdiff_t>(n);
            if (m_stop < 0) m_stop += static_cast<std::ptrdiff_t>(n);
            m_start = std::max<std::ptrdiff_t>(0, m_start);
            m_stop = std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(n), m_stop);
            if (m_start >= m_stop)
            {
                m_start = 0;
                m_stop = 0;
            }
        }

        void build_view_dimensions()
        {
            std::size_t ndim = m_base.dimension_count();
            for (std::size_t d = 0; d < ndim; ++d)
            {
                if (d == 0)
                {
                    coordinate<label_type> new_coord;
                    for (size_type i = 0; i < static_cast<size_type>(m_size); ++i)
                    {
                        size_type base_idx = m_start + i * static_cast<size_type>(m_step);
                        if (base_idx < m_base.dimension(0).size())
                            new_coord.push_back(m_base.dimension(0).coord()[base_idx]);
                    }
                    m_view_dims.emplace_back(m_base.dimension(0).name(),
                                             std::move(new_coord),
                                             m_base.dimension(0).unit(),
                                             m_base.dimension(0).description());
                }
                else
                {
                    m_view_dims.push_back(m_base.dimension(d));
                }
            }
        }

        bool is_contiguous() const noexcept
        {
            // Contiguous if step is 1 and the base is row-major.
            if (m_step != 1) return false;
            return true; // xframe data is always row-major
        }

        std::size_t compute_inner_stride() const
        {
            std::size_t stride = 1;
            for (std::size_t d = 1; d < m_base.dimension_count(); ++d)
                stride *= m_base.dimension(d).size();
            return stride;
        }

        template <class It>
        auto element(It first, It last) const
        {
            std::vector<size_type> view_idx(first, last);
            std::vector<size_type> base_idx = view_idx;
            // Adjust the first coordinate (axis 0)
            if (base_idx[0] < static_cast<size_type>(m_size))
            {
                base_idx[0] = m_start + base_idx[0] * static_cast<size_type>(m_step);
            }
            return call_base(base_idx);
        }

        auto call_base(const std::vector<size_type>& idx) const
        {
            std::size_t ndim = idx.size();
            switch (ndim)
            {
                case 1: return static_cast<double>(m_base(idx[0]));
                case 2: return static_cast<double>(m_base(idx[0], idx[1]));
                case 3: return static_cast<double>(m_base(idx[0], idx[1], idx[2]));
                case 4: return static_cast<double>(m_base(idx[0], idx[1], idx[2], idx[3]));
                default: throw std::runtime_error("xsequence_view: unsupported dimension count.");
            }
        }

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

        template <class... Labels>
        void map_labels_to_indices(std::array<size_type, sizeof...(Labels)>& idx, Labels... labels) const
        {
            std::size_t pos = 0;
            ((idx[pos++] = m_view_dims[pos].coord().find(labels)), ...);
        }
    };

    /**
     * Free function to create a sequence view.
     */
    template <class E>
    inline auto sequence_view(const E& base,
                              std::ptrdiff_t start,
                              std::ptrdiff_t stop,
                              std::ptrdiff_t step = 1)
    {
        return xsequence_view<std::decay_t<E>>(base, start, stop, step);
    }

    template <class E>
    inline auto sequence_view(const E& base,
                              const label_type& start_label,
                              const label_type& stop_label,
                              std::ptrdiff_t step = 1)
    {
        return xsequence_view<std::decay_t<E>>(base, start_label, stop_label, step);
    }

} // namespace xframe

#endif // XFRAME_XSEQUENCE_VIEW_HPP