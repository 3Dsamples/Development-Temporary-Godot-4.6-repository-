//File 0070 : views/xoffset_view.hpp
//Offset view providing a shifted window into an expression with lazy evaluation, SIMD-accelerated access, and stride-preserving semantics.
#ifndef XTENSOR_XOFFSET_VIEW_HPP
#define XTENSOR_XOFFSET_VIEW_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xexpression.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xaccessible.hpp"
#include "../core/xiterable.hpp"
#include "../core/xexception.hpp"
#include "../core/xshape.hpp"

namespace xt
{
    /**
     * @class xoffset_view
     * @brief View that shifts the origin of an expression by a given offset.
     *
     * The resulting view has the same shape as the base expression minus the offset
     * (or a specified target shape). All element accesses are translated by
     * subtracting the offset before forwarding to the base.
     */
    template <class CT>
    class xoffset_view : public xexpression<xoffset_view<CT>>,
                          public xaccessible<xoffset_view<CT>>
    {
    public:
        using self_type = xoffset_view<CT>;
        using base_type = xexpression<self_type>;
        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = typename std::decay_t<CT>::value_type;
        using reference = typename std::decay_t<CT>::reference;
        using const_reference = typename std::decay_t<CT>::const_reference;
        using pointer = typename std::decay_t<CT>::pointer;
        using const_pointer = typename std::decay_t<CT>::const_pointer;
        using size_type = typename std::decay_t<CT>::size_type;
        using difference_type = typename std::decay_t<CT>::difference_type;
        using shape_type = std::vector<size_type>;
        using strides_type = std::vector<size_type>;
        using backstrides_type = std::vector<size_type>;
        using expression_type = std::decay_t<CT>;

        using stepper = xstepper<self_type>;
        using const_stepper = xstepper<const self_type>;
        using iterator = xiterator<self_type>;
        using const_iterator = xconst_iterator<self_type>;

        /**
         * Construct an offset view with a given offset into the base expression.
         * @param e The base expression.
         * @param offset The multi-dimensional offset (positive values shift into the array).
         */
        template <class E>
        xoffset_view(E&& e, const std::vector<std::ptrdiff_t>& offset)
            : m_e(std::forward<E>(e))
            , m_offset(offset.size())
        {
            auto base_shape = m_e.shape();
            std::size_t ndim = base_shape.size();
            if (offset.size() != ndim)
                throw std::runtime_error("xoffset_view: offset dimension must match base rank.");

            // Normalize negative offsets (from end)
            for (std::size_t d = 0; d < ndim; ++d)
            {
                std::ptrdiff_t off = offset[d];
                if (off < 0) off += static_cast<std::ptrdiff_t>(base_shape[d]);
                if (off < 0 || static_cast<std::size_t>(off) > base_shape[d])
                    throw std::out_of_range("xoffset_view: offset out of bounds.");
                m_offset[d] = static_cast<std::size_t>(off);
            }

            // Compute linear offset from multi-dimensional offset
            m_linear_offset = 0;
            auto base_strides = m_e.strides();
            for (std::size_t d = 0; d < ndim; ++d)
            {
                m_linear_offset += m_offset[d] * base_strides[d];
            }

            // Compute the new shape: base_shape - offset
            m_shape = base_shape;
            for (std::size_t d = 0; d < ndim; ++d)
            {
                m_shape[d] = base_shape[d] - m_offset[d];
            }

            // Strides are inherited from the base expression
            m_strides = base_strides;
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }

        /**
         * Construct an offset view with explicit target shape.
         * The view extends from offset to offset + target_shape.
         */
        template <class E>
        xoffset_view(E&& e, const std::vector<std::ptrdiff_t>& offset, const shape_type& target_shape)
            : m_e(std::forward<E>(e))
            , m_offset(offset.size())
        {
            auto base_shape = m_e.shape();
            std::size_t ndim = base_shape.size();
            if (offset.size() != ndim || target_shape.size() != ndim)
                throw std::runtime_error("xoffset_view: dimension mismatch.");

            for (std::size_t d = 0; d < ndim; ++d)
            {
                std::ptrdiff_t off = offset[d];
                if (off < 0) off += static_cast<std::ptrdiff_t>(base_shape[d]);
                if (off < 0 || static_cast<std::size_t>(off) + target_shape[d] > base_shape[d])
                    throw std::out_of_range("xoffset_view: offset + shape exceeds base bounds.");
                m_offset[d] = static_cast<std::size_t>(off);
            }

            auto base_strides = m_e.strides();
            m_linear_offset = 0;
            for (std::size_t d = 0; d < ndim; ++d)
                m_linear_offset += m_offset[d] * base_strides[d];

            m_shape = target_shape;
            m_strides = base_strides;
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }

        xoffset_view(const self_type&) = default;
        xoffset_view& operator=(const self_type&) = default;
        xoffset_view(self_type&&) = default;
        xoffset_view& operator=(self_type&&) = default;

        size_type size() const noexcept { return compute_size(m_shape); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        void set_shape(const shape_type& s) { m_shape = s; }
        void set_strides(const strides_type& st) { m_strides = st; m_backstrides = detail::compute_backstrides(st, m_shape); }

        /**
         * Multi-dimensional access: adds the offset to the given indices.
         */
        template <class... Args>
        const_reference operator()(Args... args) const
        {
            std::array<size_type, sizeof...(Args)> idx{static_cast<size_type>(args)...};
            return element(idx.begin(), idx.end());
        }

        template <class... Args>
        reference operator()(Args... args)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this)(args...));
        }

        reference operator[](size_type i) { return operator()(i); }
        const_reference operator[](size_type i) const { return operator()(i); }

        template <class It>
        const_reference element(It first, It last) const
        {
            auto idx = std::vector<size_type>(first, last);
            // Shift indices by the offset
            for (std::size_t d = 0; d < idx.size(); ++d)
                idx[d] += m_offset[d];
            return m_e.element(idx.begin(), idx.end());
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
        }

        pointer data() noexcept { return m_e.data() + m_linear_offset; }
        const_pointer data() const noexcept { return m_e.data() + m_linear_offset; }

        // Iterators
        iterator begin() noexcept { return iterator(this, 0); }
        iterator end() noexcept { return iterator(this, size()); }
        const_iterator begin() const noexcept { return const_iterator(this, 0); }
        const_iterator end() const noexcept { return const_iterator(this, size()); }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

        // Stepper
        stepper stepper_begin() noexcept { return stepper(this, m_linear_offset); }
        stepper stepper_end() noexcept { return stepper(this, m_linear_offset + size()); }
        const_stepper stepper_begin() const noexcept { return const_stepper(this, m_linear_offset); }
        const_stepper stepper_end() const noexcept { return const_stepper(this, m_linear_offset + size()); }

        // SIMD load: if the view is contiguous, load directly from base + offset
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            // Check if the innermost dimension is contiguous after offset
            bool contiguous = true;
            std::size_t expected = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(m_shape.size()) - 1; d >= 0; --d)
            {
                if (m_strides[static_cast<std::size_t>(d)] != expected)
                {
                    contiguous = false;
                    break;
                }
                expected *= m_shape[static_cast<std::size_t>(d)];
            }
            if (contiguous)
            {
                return simd_type::load_unaligned(m_e.data() + m_linear_offset + i);
            }
            // Fallback scalar gather
            alignas(64) std::array<T, simd_type::size> buffer;
            for (std::size_t k = 0; k < simd_type::size; ++k)
                buffer[k] = operator()(i + k);
            return simd_type::load_aligned(buffer.data());
        }

        const expression_type& expression() const noexcept { return m_e; }
        const std::vector<size_type>& offset() const noexcept { return m_offset; }
        size_type linear_offset() const noexcept { return m_linear_offset; }

    private:
        CT m_e;
        std::vector<size_type> m_offset;
        size_type m_linear_offset;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
    };

    template <class CT>
    struct xcontainer_inner_types<xoffset_view<CT>>
    {
        using value_type = typename std::decay_t<CT>::value_type;
        using reference = typename std::decay_t<CT>::reference;
        using const_reference = typename std::decay_t<CT>::const_reference;
        using pointer = typename std::decay_t<CT>::pointer;
        using const_pointer = typename std::decay_t<CT>::const_pointer;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = std::vector<size_type>;
        using backstrides_type = std::vector<size_type>;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<uvector<value_type>, DEFAULT_LAYOUT, shape_type>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

    /**
     * Free function to create an offset view.
     */
    template <class E>
    inline auto offset_view(E&& e, const std::vector<std::ptrdiff_t>& offset)
    {
        return xoffset_view<std::decay_t<E>>(std::forward<E>(e), offset);
    }

    template <class E>
    inline auto offset_view(E&& e, const std::vector<std::ptrdiff_t>& offset,
                            const std::vector<std::size_t>& shape)
    {
        return xoffset_view<std::decay_t<E>>(std::forward<E>(e), offset, shape);
    }

} // namespace xt

#endif // XTENSOR_XOFFSET_VIEW_HPP