//File 0057 : views/xindex_view.hpp
//Index view for indirect indexing into a base expression via an index array, with SIMD gather, iterator/stepper support, and lazy evaluation.
#ifndef XTENSOR_XINDEX_VIEW_HPP
#define XTENSOR_XINDEX_VIEW_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>
#include <stdexcept>

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

namespace xt
{
    template <class CT, class I>
    class xindex_view;

    template <class CT, class I>
    struct xcontainer_inner_types<xindex_view<CT, I>>
    {
        using base_expression_type = std::decay_t<CT>;
        using index_expression_type = std::decay_t<I>;
        using value_type = typename base_expression_type::value_type;
        using reference = typename base_expression_type::reference;
        using const_reference = typename base_expression_type::const_reference;
        using pointer = typename base_expression_type::pointer;
        using const_pointer = typename base_expression_type::const_pointer;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = typename index_expression_type::shape_type;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<uvector<value_type>, DEFAULT_LAYOUT, shape_type>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

    /**
     * @class xindex_view
     * @brief Lazily indexes a base expression using an index array.
     *
     * For each element (i,j,...) of the index array, returns base_expr[ index_array(i,j,...) ].
     * The index array can be multidimensional; its elements are used as flat indices into the
     * base expression, or optionally as multi-indices if the index array's last dimension
     * equals the base rank.
     */
    template <class CT, class I>
    class xindex_view : public xexpression<xindex_view<CT, I>>,
                         public xaccessible<xindex_view<CT, I>>
    {
    public:
        using self_type = xindex_view<CT, I>;
        using base_type = xexpression<self_type>;
        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename inner_types::pointer;
        using const_pointer = typename inner_types::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename inner_types::difference_type;
        using shape_type = typename inner_types::shape_type;
        using strides_type = typename inner_types::strides_type;
        using backstrides_type = typename inner_types::backstrides_type;

        using stepper = xstepper<self_type>;
        using const_stepper = xstepper<const self_type>;
        using iterator = xiterator<self_type>;
        using const_iterator = xconst_iterator<self_type>;

        /**
         * Construct index view: base[e].
         * @param base The base expression to index into.
         * @param indices The index expression providing the element positions.
         */
        template <class E, class Indices>
        xindex_view(E&& base, Indices&& indices)
            : m_base(std::forward<E>(base)), m_indices(std::forward<Indices>(indices))
        {
            // Validate that index values are within base size? (lazy check on access)
            m_shape = m_indices.shape();
            m_strides = compute_strides(m_shape);
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }

        xindex_view(const self_type&) = default;
        xindex_view& operator=(const self_type&) = default;
        xindex_view(self_type&&) = default;
        xindex_view& operator=(self_type&&) = default;

        size_type size() const noexcept { return compute_size(m_shape); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        void set_shape(const shape_type& s) { m_shape = s; }
        void set_strides(const strides_type& st) { m_strides = st; m_backstrides = detail::compute_backstrides(st, m_shape); }

        template <class... Args>
        const_reference operator()(Args... args) const
        {
            auto idx = std::array<size_type, sizeof...(Args)>{static_cast<size_type>(args)...};
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
            // Use the index value(s) at the given position to index into the base
            auto idx = std::vector<size_type>(first, last);
            size_type index_linear = ravel_index(idx, m_indices.strides());
            size_type base_idx = m_indices.data()[index_linear]; // assumes flat integer index
            return m_base.data()[base_idx];
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
        }

        pointer data() noexcept { return nullptr; }
        const_pointer data() const noexcept { return nullptr; }

        // Iterator support
        iterator begin() noexcept { return iterator(this, 0); }
        iterator end() noexcept { return iterator(this, size()); }
        const_iterator begin() const noexcept { return const_iterator(this, 0); }
        const_iterator end() const noexcept { return const_iterator(this, size()); }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

        // Stepper
        stepper stepper_begin() noexcept { return stepper(this, 0); }
        stepper stepper_end() noexcept { return stepper(this, size()); }
        const_stepper stepper_begin() const noexcept { return const_stepper(this, 0); }
        const_stepper stepper_end() const noexcept { return const_stepper(this, size()); }

        // SIMD load for contiguous index block? Not efficient but provide stub.
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            simd_type result;
            constexpr std::size_t simd_size = simd_type::size;
            alignas(64) std::array<T, simd_size> buffer;
            for (std::size_t k = 0; k < simd_size; ++k)
                buffer[k] = operator()(i + k);
            return simd_type::load_aligned(buffer.data());
        }

        const CT& base() const noexcept { return m_base; }
        const I& indices() const noexcept { return m_indices; }

    private:
        CT m_base;
        I m_indices;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
    };

    /**
     * Free function to create an index view.
     */
    template <class E, class I>
    inline auto index_view(E&& base, I&& indices)
    {
        return xindex_view<std::decay_t<E>, std::decay_t<I>>(
            std::forward<E>(base), std::forward<I>(indices));
    }

} // namespace xt

#endif // XTENSOR_XINDEX_VIEW_HPP