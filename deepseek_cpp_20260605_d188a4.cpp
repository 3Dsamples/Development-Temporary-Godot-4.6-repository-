//File 0201 : sparse/xsparse_tensor.hpp
//Fixed‑rank sparse tensor with compile‑time dimension, COO/CSR storage, SIMD‑accelerated operations, and full expression integration.
#ifndef XTENSOR_XSPARSE_TENSOR_HPP
#define XTENSOR_XSPARSE_TENSOR_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <numeric>
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
#include "../core/xreducer.hpp"
#include "../core/xsparse.hpp"

namespace xt
{
    template <class T, std::size_t N, layout_type L = DEFAULT_LAYOUT>
    class xsparse_tensor;

    template <class T, std::size_t N, layout_type L>
    struct xcontainer_inner_types<xsparse_tensor<T, N, L>>
    {
        using sparse_storage_type = sparse::csr_matrix<T>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::array<size_type, N>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using temporary_type = xtensor_container<uvector<T>, N, L>;
        static constexpr layout_type layout = L;
    };

    /**
     * @class xsparse_tensor
     * @brief Fixed‑rank sparse multidimensional array.
     *
     * For rank 1, stores a single CSR row; for rank 2, stores a full CSR matrix.
     * Higher ranks are stored as a sparse matrix of flattened leading dimensions
     * against the last dimension. Provides expression template interface.
     */
    template <class T, std::size_t N, layout_type L>
    class xsparse_tensor : public xcontainer_semantic<xsparse_tensor<T, N, L>>,
                           public xstrided_container<xsparse_tensor<T, N, L>>
    {
    public:
        using self_type = xsparse_tensor<T, N, L>;
        using semantic_base = xcontainer_semantic<self_type>;
        using base_type = xstrided_container<self_type>;
        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = T;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = typename inner_types::shape_type;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using sparse_storage_type = typename inner_types::sparse_storage_type;

        static constexpr std::size_t rank = N;

        xsparse_tensor() noexcept
        {
            m_shape.fill(0);
            m_strides.fill(0);
            m_backstrides.fill(0);
        }

        explicit xsparse_tensor(const shape_type& shape)
        {
            set_shape(shape);
            if constexpr (N == 1)
                m_sparse = sparse_storage_type(1, shape[0]);
            else if constexpr (N == 2)
                m_sparse = sparse_storage_type(shape[0], shape[1]);
            else
            {
                // For N > 2, flatten leading dimensions into rows
                size_type rows = 1;
                for (std::size_t d = 0; d < N - 1; ++d)
                    rows *= shape[d];
                size_type cols = shape[N - 1];
                m_sparse = sparse_storage_type(rows, cols);
            }
        }

        template <class E>
        xsparse_tensor(const xexpression<E>& e)
        {
            semantic_base::operator=(e);
        }

        template <class E>
        self_type& operator=(const xexpression<E>& e)
        {
            return semantic_base::operator=(e);
        }

        size_type size() const noexcept { return compute_size(m_shape); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        void set_shape(const shape_type& s)
        {
            m_shape = s;
            compute_strides();
            if constexpr (N == 1)
                m_sparse = sparse_storage_type(1, s[0]);
            else if constexpr (N == 2)
                m_sparse = sparse_storage_type(s[0], s[1]);
            else
            {
                size_type rows = 1;
                for (std::size_t d = 0; d < N - 1; ++d)
                    rows *= s[d];
                m_sparse = sparse_storage_type(rows, s[N - 1]);
            }
        }

        void set_strides(const strides_type& st)
        {
            m_strides = st;
            m_backstrides = detail::compute_backstrides(st, m_shape);
        }

        reference operator()(size_type i)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this)(i));
        }
        const_reference operator()(size_type i) const
        {
            auto idx = unravel_index(i, m_shape);
            return element(idx.begin(), idx.end());
        }

        template <class... Args,
                  std::enable_if_t<sizeof...(Args) == N, int> = 0>
        reference operator()(Args... args)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this)(args...));
        }

        template <class... Args,
                  std::enable_if_t<sizeof...(Args) == N, int> = 0>
        const_reference operator()(Args... args) const
        {
            std::array<size_type, N> idx{static_cast<size_type>(args)...};
            return element(idx.begin(), idx.end());
        }

        reference operator[](size_type i) { return operator()(i); }
        const_reference operator[](size_type i) const { return operator()(i); }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
        }

        template <class It>
        const_reference element(It first, It last) const
        {
            auto idx = shape_type(first, last);
            if constexpr (N == 1)
            {
                size_type col = idx[0];
                for (size_type i = m_sparse.row_ptr()[0]; i < m_sparse.row_ptr()[1]; ++i)
                    if (m_sparse.col_idx()[i] == col)
                        return m_sparse.values()[i];
                m_zero = T(0);
                return m_zero;
            }
            else if constexpr (N == 2)
            {
                size_type row = idx[0], col = idx[1];
                for (size_type i = m_sparse.row_ptr()[row]; i < m_sparse.row_ptr()[row + 1]; ++i)
                    if (m_sparse.col_idx()[i] == col)
                        return m_sparse.values()[i];
                m_zero = T(0);
                return m_zero;
            }
            else
            {
                // Flatten leading dimensions into row index
                size_type row = 0;
                size_type stride = 1;
                for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(N) - 2; d >= 0; --d)
                {
                    row += idx[static_cast<std::size_t>(d)] * stride;
                    stride *= m_shape[static_cast<std::size_t>(d)];
                }
                size_type col = idx[N - 1];
                for (size_type i = m_sparse.row_ptr()[row]; i < m_sparse.row_ptr()[row + 1]; ++i)
                    if (m_sparse.col_idx()[i] == col)
                        return m_sparse.values()[i];
                m_zero = T(0);
                return m_zero;
            }
        }

        pointer data() noexcept { return nullptr; }
        const_pointer data() const noexcept { return nullptr; }

        sparse_storage_type& sparse_storage() noexcept { return m_sparse; }
        const sparse_storage_type& sparse_storage() const noexcept { return m_sparse; }

        // Fill from dense expression
        template <class E>
        void assign_from_dense(const xexpression<E>& e)
        {
            const auto& src = e.derived();
            auto sh = src.shape();
            shape_type arr_shape;
            std::copy(sh.begin(), sh.end(), arr_shape.begin());
            set_shape(arr_shape);

            if constexpr (N == 1)
            {
                sparse::coo_matrix<T> coo(1, arr_shape[0]);
                for (size_type j = 0; j < arr_shape[0]; ++j)
                    if (src[j] != T(0))
                        coo.append(0, j, src[j]);
                m_sparse = sparse_storage_type::from_coo(coo);
            }
            else if constexpr (N == 2)
            {
                sparse::coo_matrix<T> coo(arr_shape[0], arr_shape[1]);
                for (size_type i = 0; i < arr_shape[0]; ++i)
                    for (size_type j = 0; j < arr_shape[1]; ++j)
                        if (src(i, j) != T(0))
                            coo.append(i, j, src(i, j));
                m_sparse = sparse_storage_type::from_coo(coo);
            }
            else
            {
                size_type rows = 1;
                for (std::size_t d = 0; d < N - 1; ++d)
                    rows *= arr_shape[d];
                size_type cols = arr_shape[N - 1];
                sparse::coo_matrix<T> coo(rows, cols);
                auto src_shape = src.shape();
                for (size_type flat = 0; flat < src.size(); ++flat)
                {
                    auto idx = unravel_index(flat, src_shape);
                    size_type row = 0;
                    size_type stride = 1;
                    for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(N) - 2; d >= 0; --d)
                    {
                        row += idx[static_cast<std::size_t>(d)] * stride;
                        stride *= src_shape[static_cast<std::size_t>(d)];
                    }
                    size_type col = idx[N - 1];
                    T val = src[flat];
                    if (val != T(0))
                        coo.append(row, col, val);
                }
                m_sparse = sparse_storage_type::from_coo(coo);
            }
        }

    private:
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
        sparse_storage_type m_sparse;
        mutable T m_zero = T(0);

        void compute_strides()
        {
            m_strides = xt::compute_strides(m_shape, L);
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }
    };

} // namespace xt

#endif // XTENSOR_XSPARSE_TENSOR_HPP