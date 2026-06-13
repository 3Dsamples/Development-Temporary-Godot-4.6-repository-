//File 0205 : sparse/xsparse_view.hpp
//Sparse strided view with lazy element access, SIMD-accelerated lookup, and full expression integration for sparse arrays.
#ifndef XTENSOR_XSPARSE_VIEW_HPP
#define XTENSOR_XSPARSE_VIEW_HPP

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
#include "../core/xarray.hpp"
#include "../core/xeval.hpp"
#include "../core/xsparse.hpp"
#include "../sparse/xsparse_array.hpp"
#include "../sparse/xsparse_tensor.hpp"

namespace xt {
namespace sparse {

    /**
     * @class xsparse_strided_view
     * @brief Lazy strided view over a sparse expression.
     *
     * Allows slicing and broadcasting of sparse arrays without copying the underlying
     * sparse storage. Element lookup passes through to the base sparse expression
     * after mapping indices according to the view's strides.
     */
    template <class CT>
    class xsparse_strided_view : public xexpression<xsparse_strided_view<CT>>
    {
    public:
        using self_type = xsparse_strided_view<CT>;
        using base_type = std::decay_t<CT>;
        using value_type = typename base_type::value_type;
        using const_reference = typename base_type::const_reference;
        using reference = typename base_type::reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = std::vector<size_type>;
        using backstrides_type = std::vector<size_type>;

        template <class E>
        xsparse_strided_view(E&& base, const shape_type& shape, const strides_type& strides,
                             size_type offset = 0)
            : m_base(std::forward<E>(base))
            , m_shape(shape)
            , m_strides(strides)
            , m_offset(offset)
        {
            if (shape.size() != strides.size())
                throw std::runtime_error("xsparse_strided_view: shape and strides must match.");
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }

        template <class E, class... Slices>
        xsparse_strided_view(E&& base, Slices&&... slices)
            : m_base(std::forward<E>(base))
        {
            auto old_shape = m_base.shape();
            auto old_strides = compute_strides(old_shape);
            auto slice_tuple = std::make_tuple(std::forward<Slices>(slices)...);
            std::tie(m_shape, m_strides) = compute_sliced_view(slice_tuple, old_shape, old_strides);
            m_offset = 0;
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }

        size_type size() const noexcept { return compute_size(m_shape); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

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
            auto view_idx = std::vector<size_type>(first, last);
            // Map view index to base index using strides
            size_type base_offset = m_offset;
            for (std::size_t d = 0; d < view_idx.size(); ++d)
                base_offset += view_idx[d] * m_strides[d];
            auto base_shape = m_base.shape();
            auto base_idx = unravel_index(base_offset, base_shape);
            return m_base.element(base_idx.begin(), base_idx.end());
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
        }

        pointer data() noexcept { return nullptr; }
        const_pointer data() const noexcept { return nullptr; }

        const base_type& base() const noexcept { return m_base; }
        size_type offset() const noexcept { return m_offset; }

    private:
        CT m_base;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
        size_type m_offset;
    };

    /**
     * Create a strided view into a sparse array.
     */
    template <class E>
    inline auto strided_sparse_view(E&& base, const std::vector<std::size_t>& shape,
                                    const std::vector<std::size_t>& strides, std::size_t offset = 0)
    {
        return xsparse_strided_view<std::decay_t<E>>(std::forward<E>(base), shape, strides, offset);
    }

    /**
     * Slice a sparse array using variadic slice descriptors.
     */
    template <class E, class... Slices>
    inline auto sparse_slice(E&& base, Slices&&... slices)
    {
        return xsparse_strided_view<std::decay_t<E>>(std::forward<E>(base),
                                                     std::forward<Slices>(slices)...);
    }

    /**
     * Diagonal view of a sparse 2D matrix: extracts a strided 1D view along the diagonal.
     */
    template <class E>
    inline auto sparse_diagonal(E&& base, std::ptrdiff_t offset = 0)
    {
        auto sh = base.shape();
        if (sh.size() != 2)
            throw std::runtime_error("sparse_diagonal: requires 2D array.");
        std::size_t nrows = sh[0], ncols = sh[1];
        std::size_t diag_len = (offset >= 0) ? std::min(nrows, ncols - static_cast<std::size_t>(offset))
                                             : std::min(nrows - static_cast<std::size_t>(-offset), ncols);
        if (diag_len == 0)
            throw std::runtime_error("sparse_diagonal: empty diagonal.");
        // The diagonal is a strided view: shape = {diag_len}, strides = {row_stride + col_stride}
        auto base_strides = compute_strides(sh);
        std::size_t row_stride = base_strides[0];
        std::size_t col_stride = base_strides[1];
        std::size_t start = (offset >= 0) ? static_cast<std::size_t>(offset) * col_stride
                                          : static_cast<std::size_t>(-offset) * row_stride;
        return xsparse_strided_view<std::decay_t<E>>(std::forward<E>(base),
                                                     std::vector<std::size_t>{diag_len},
                                                     std::vector<std::size_t>{row_stride + col_stride},
                                                     start);
    }

    /**
     * Transpose view of a sparse 2D matrix (swap axes).
     */
    template <class E>
    inline auto sparse_transpose_view(E&& base)
    {
        auto sh = base.shape();
        if (sh.size() != 2)
            throw std::runtime_error("sparse_transpose_view: requires 2D array.");
        auto base_strides = compute_strides(sh);
        std::vector<std::size_t> new_shape{sh[1], sh[0]};
        std::vector<std::size_t> new_strides{base_strides[1], base_strides[0]};
        return xsparse_strided_view<std::decay_t<E>>(std::forward<E>(base), new_shape, new_strides, 0);
    }

    /**
     * Row or column slice of a sparse matrix.
     */
    template <class E>
    inline auto sparse_row(E&& base, std::size_t row_index)
    {
        auto sh = base.shape();
        if (sh.size() != 2) throw std::runtime_error("sparse_row: requires 2D array.");
        if (row_index >= sh[0]) throw std::out_of_range("sparse_row: row index out of bounds.");
        auto base_strides = compute_strides(sh);
        std::size_t offset = row_index * base_strides[0];
        return xsparse_strided_view<std::decay_t<E>>(std::forward<E>(base),
                                                     std::vector<std::size_t>{sh[1]},
                                                     std::vector<std::size_t>{base_strides[1]},
                                                     offset);
    }

    template <class E>
    inline auto sparse_col(E&& base, std::size_t col_index)
    {
        auto sh = base.shape();
        if (sh.size() != 2) throw std::runtime_error("sparse_col: requires 2D array.");
        if (col_index >= sh[1]) throw std::out_of_range("sparse_col: col index out of bounds.");
        auto base_strides = compute_strides(sh);
        std::size_t offset = col_index * base_strides[1];
        return xsparse_strided_view<std::decay_t<E>>(std::forward<E>(base),
                                                     std::vector<std::size_t>{sh[0]},
                                                     std::vector<std::size_t>{base_strides[0]},
                                                     offset);
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_VIEW_HPP