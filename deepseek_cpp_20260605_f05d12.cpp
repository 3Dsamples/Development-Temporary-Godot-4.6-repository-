//File 0112 : numdot/view.h
//Sliced and strided views: lazy evaluation with SIMD-accelerated element access, broadcasting integration, and iterator support.
#ifndef NUMDOT_VIEW_H
#define NUMDOT_VIEW_H

#include <type_traits>
#include <utility>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <functional>

#include "config.h"
#include "forward.h"
#include "types.h"
#include "shape.h"
#include "elementwise.h"
#include "broadcast.h"
#include "slicing.h"

namespace numdot
{
    /**
     * @class strided_view
     * @brief Non-owning view into an expression with custom shape, strides, and offset.
     *
     * Allows slicing, broadcasting, and arbitrary strided access to the
     * underlying data without copying. Supports SIMD loading when contiguous.
     */
    template <class CT, class S = std::vector<std::size_t>>
    class strided_view : public expression<strided_view<CT, S>>
    {
    public:
        using self_type = strided_view<CT, S>;
        using value_type = typename std::decay_t<CT>::value_type;
        using reference = typename std::decay_t<CT>::reference;
        using const_reference = typename std::decay_t<CT>::const_reference;
        using pointer = typename std::decay_t<CT>::pointer;
        using const_pointer = typename std::decay_t<CT>::const_pointer;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = S;
        using strides_type = S;
        using backstrides_type = S;
        using iterator = pointer;
        using const_iterator = const_pointer;
        static constexpr layout layout = default_layout;

        /**
         * Construct a strided view.
         * @param expr The base expression.
         * @param shape The desired view shape.
         * @param strides The strides for each dimension.
         * @param offset Linear offset into the base expression.
         */
        template <class E>
        strided_view(E&& expr, const shape_type& shape, const strides_type& strides,
                    size_type offset = 0)
            : m_expr(std::forward<E>(expr))
            , m_shape(shape)
            , m_strides(strides)
            , m_offset(offset)
        {
            if (shape.size() != strides.size())
                throw std::runtime_error("strided_view: shape and strides must have same rank.");
            m_backstrides = compute_backstrides(m_strides, m_shape);
        }

        /**
         * Construct a strided view using slices from the base expression.
         */
        template <class E, class... Slices>
        strided_view(E&& expr, Slices&&... slices)
            : m_expr(std::forward<E>(expr))
        {
            auto old_shape = m_expr.shape();
            auto old_strides = m_expr.strides();
            auto slice_tuple = std::make_tuple(std::forward<Slices>(slices)...);
            std::tie(m_shape, m_strides) = compute_sliced_view(slice_tuple, old_shape, old_strides);
            m_offset = 0;
            m_backstrides = compute_backstrides(m_strides, m_shape);
        }

        strided_view(const self_type&) = default;
        strided_view& operator=(const self_type&) = default;
        strided_view(self_type&&) = default;
        strided_view& operator=(self_type&&) = default;

        size_type size() const noexcept { return compute_size(m_shape); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        void set_shape(const shape_type& s) { m_shape = s; }
        void set_strides(const strides_type& st) { m_strides = st; m_backstrides = compute_backstrides(st, m_shape); }

        pointer data() noexcept { return m_expr.data() + m_offset; }
        const_pointer data() const noexcept { return m_expr.data() + m_offset; }

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

        const_reference operator[](size_type i) const
        {
            return data()[compute_linear_offset(i)];
        }

        reference operator[](size_type i)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this)[i]);
        }

        template <class It>
        const_reference element(It first, It last) const
        {
            auto idx = shape_type(first, last);
            return m_expr.data()[m_offset + ravel_index(idx, m_strides)];
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
        }

        // Iterators
        iterator begin() noexcept { return data(); }
        iterator end() noexcept { return data() + size(); }
        const_iterator begin() const noexcept { return data(); }
        const_iterator end() const noexcept { return data() + size(); }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

        // SIMD load for contiguous views
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            if (is_contiguous())
                return simd_type::load_unaligned(data() + i);
            constexpr std::size_t simd_size = simd_type::size;
            alignas(64) std::array<T, simd_size> buf;
            for (std::size_t k = 0; k < simd_size; ++k)
                buf[k] = operator()(i + k);
            return simd_type::load_aligned(buf.data());
        }

        bool is_contiguous() const noexcept
        {
            std::size_t expected = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(m_shape.size()) - 1; d >= 0; --d)
            {
                if (m_strides[static_cast<std::size_t>(d)] != expected)
                    return false;
                expected *= m_shape[static_cast<std::size_t>(d)];
            }
            return true;
        }

        const std::decay_t<CT>& base() const noexcept { return m_expr; }
        size_type offset() const noexcept { return m_offset; }

    private:
        CT m_expr;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
        size_type m_offset;

        size_type compute_linear_offset(size_type i) const
        {
            auto idx = unravel_index(i, m_shape);
            return ravel_index(idx, m_strides);
        }
    };

    /**
     * Free function to create a strided view.
     */
    template <class E, class S>
    inline auto make_strided_view(E&& e, const S& shape, const S& strides, std::size_t offset = 0)
    {
        return strided_view<std::decay_t<E>, S>(std::forward<E>(e), shape, strides, offset);
    }

    /**
     * Free function to slice an expression using variadic slice descriptors.
     */
    template <class E, class... Slices>
    inline auto slice(E&& e, Slices&&... slices)
    {
        return strided_view<std::decay_t<E>, std::vector<std::size_t>>(
            std::forward<E>(e), std::forward<Slices>(slices)...);
    }

    /**
     * @class offset_view
     * @brief View that offsets into the base expression by a given linear offset,
     *        preserving shape and strides.
     */
    template <class CT>
    class offset_view : public expression<offset_view<CT>>
    {
    public:
        using self_type = offset_view<CT>;
        using value_type = typename std::decay_t<CT>::value_type;
        using reference = typename std::decay_t<CT>::reference;
        using const_reference = typename std::decay_t<CT>::const_reference;
        using pointer = typename std::decay_t<CT>::pointer;
        using const_pointer = typename std::decay_t<CT>::const_pointer;
        using size_type = std::size_t;
        using shape_type = typename std::decay_t<CT>::shape_type;
        using strides_type = shape_type;
        static constexpr layout layout = default_layout;

        template <class E>
        offset_view(E&& expr, size_type offset)
            : m_expr(std::forward<E>(expr)), m_offset(offset) {}

        size_type size() const noexcept { return m_expr.size() - m_offset; }
        const shape_type& shape() const noexcept { return m_expr.shape(); }
        const strides_type& strides() const noexcept { return m_expr.strides(); }

        pointer data() noexcept { return m_expr.data() + m_offset; }
        const_pointer data() const noexcept { return m_expr.data() + m_offset; }

        template <class... Args>
        const_reference operator()(Args... args) const
        {
            return m_expr.data()[m_offset + ravel_index(shape_type{static_cast<size_type>(args)...}, m_expr.strides())];
        }

        template <class... Args>
        reference operator()(Args... args)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this)(args...));
        }

        const_reference operator[](size_type i) const { return data()[i]; }
        reference operator[](size_type i) { return data()[i]; }

    private:
        CT m_expr;
        size_type m_offset;
    };

    /**
     * Free function to create an offset view.
     */
    template <class E>
    inline auto offset(E&& e, std::size_t off)
    {
        return offset_view<std::decay_t<E>>(std::forward<E>(e), off);
    }

} // namespace numdot

#endif // NUMDOT_VIEW_H