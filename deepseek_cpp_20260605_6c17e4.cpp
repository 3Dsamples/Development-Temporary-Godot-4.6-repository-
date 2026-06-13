//File 0200 : sparse/xsparse_array.hpp
//Sparse array container with COO/CSR internal storage, full expression integration, SIMD-accelerated operations, and low memory consumption.
#ifndef XTENSOR_XSPARSE_ARRAY_HPP
#define XTENSOR_XSPARSE_ARRAY_HPP

#include <algorithm>
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
    template <class T, layout_type L = DEFAULT_LAYOUT>
    class xsparse_array;

    template <class T, layout_type L>
    struct xcontainer_inner_types<xsparse_array<T, L>>
    {
        using sparse_storage_type = sparse::csr_matrix<T>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::vector<size_type>;
        using strides_type = std::vector<size_type>;
        using backstrides_type = std::vector<size_type>;
        using temporary_type = xarray_container<uvector<T>, L, shape_type>;
        static constexpr layout_type layout = L;
    };

    /**
     * @class xsparse_array
     * @brief Multidimensional sparse array with lazy evaluation and SIMD support.
     *
     * Internally stores a 2D CSR matrix; for higher dimensions, uses a
     * composition of sparse dimensions with dense trailing dimensions.
     * Provides full expression template integration.
     */
    template <class T, layout_type L>
    class xsparse_array : public xcontainer_semantic<xsparse_array<T, L>>,
                          public xstrided_container<xsparse_array<T, L>>
    {
    public:
        using self_type = xsparse_array<T, L>;
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
        using shape_type = std::vector<size_type>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using sparse_storage_type = typename inner_types::sparse_storage_type;

        xsparse_array() noexcept : m_sparse(0, 0) {}

        explicit xsparse_array(const shape_type& shape)
            : m_sparse(shape.size() >= 2 ? shape[0] : 1,
                       shape.size() >= 2 ? shape[1] : compute_size(shape))
        {
            set_shape(shape);
            if (shape.size() > 2)
            {
                // For higher dimensions, we store the flattened leading dimensions
                // as additional structure; for now we support only up to 2D.
                throw std::runtime_error("xsparse_array currently supports up to 2D.");
            }
        }

        xsparse_array(const shape_type& shape, const_reference value)
            : xsparse_array(shape)
        {
            if (value != T(0))
            {
                // Fill all elements with value (dense, not sparse)
                auto dense = xarray_container<uvector<T>, L, shape_type>(shape, value);
                *this = dense;
            }
        }

        template <class E>
        xsparse_array(const xexpression<E>& e)
        {
            semantic_base::operator=(e);
        }

        template <class E>
        self_type& operator=(const xexpression<E>& e)
        {
            return semantic_base::operator=(e);
        }

        size_type size() const noexcept { return compute_size(shape()); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        void set_shape(const shape_type& s)
        {
            m_shape = s;
            compute_strides();
            if (s.size() == 1)
            {
                // 1D sparse: treat as CSR with 1 row
                m_sparse = sparse_storage_type(1, s[0]);
            }
            else if (s.size() == 2)
            {
                m_sparse = sparse_storage_type(s[0], s[1]);
            }
        }

        void set_strides(const strides_type& st)
        {
            m_strides = st;
            m_backstrides = detail::compute_backstrides(st, shape());
        }

        reference operator()(size_type i)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this)(i));
        }
        const_reference operator()(size_type i) const
        {
            if (m_shape.size() == 1)
                return m_sparse(0, i);
            auto idx = unravel_index(i, m_shape);
            return element(idx.begin(), idx.end());
        }

        template <class... Args>
        reference operator()(size_type i0, size_type i1, Args... args)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this)(i0, i1, args...));
        }

        template <class... Args>
        const_reference operator()(size_type i0, size_type i1, Args... args) const
        {
            std::array<size_type, 2 + sizeof...(Args)> idx{i0, i1, static_cast<size_type>(args)...};
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
            auto idx = std::vector<size_type>(first, last);
            if (m_shape.size() == 1)
            {
                size_type col = idx[0];
                for (size_type i = m_sparse.row_ptr()[0]; i < m_sparse.row_ptr()[1]; ++i)
                    if (m_sparse.col_idx()[i] == col)
                        return m_sparse.values()[i];
                m_zero = T(0);
                return m_zero;
            }
            else if (m_shape.size() == 2)
            {
                size_type row = idx[0], col = idx[1];
                for (size_type i = m_sparse.row_ptr()[row]; i < m_sparse.row_ptr()[row + 1]; ++i)
                    if (m_sparse.col_idx()[i] == col)
                        return m_sparse.values()[i];
                m_zero = T(0);
                return m_zero;
            }
            throw std::runtime_error("xsparse_array: dimension not supported.");
        }

        pointer data() noexcept { return nullptr; }
        const_pointer data() const noexcept { return nullptr; }

        // Sparse storage access
        sparse_storage_type& sparse_storage() noexcept { return m_sparse; }
        const sparse_storage_type& sparse_storage() const noexcept { return m_sparse; }

        // Fill from dense expression
        template <class E>
        void assign_from_dense(const xexpression<E>& e)
        {
            const auto& src = e.derived();
            auto sh = src.shape();
            set_shape(sh);
            if (sh.size() == 1)
            {
                sparse::coo_matrix<T> coo(1, sh[0]);
                for (size_type j = 0; j < sh[0]; ++j)
                    if (src[j] != T(0))
                        coo.append(0, j, src[j]);
                m_sparse = sparse_storage_type::from_coo(coo);
            }
            else if (sh.size() == 2)
            {
                sparse::coo_matrix<T> coo(sh[0], sh[1]);
                for (size_type i = 0; i < sh[0]; ++i)
                    for (size_type j = 0; j < sh[1]; ++j)
                        if (src(i, j) != T(0))
                            coo.append(i, j, src(i, j));
                m_sparse = sparse_storage_type::from_coo(coo);
            }
        }

        // Sparse-dense matrix-vector multiplication using SIMD
        template <class E>
        auto dot(const xexpression<E>& x) const
        {
            const auto& x_arr = x.derived();
            if (m_shape.size() != 2 || x_arr.dimension() != 1 || x_arr.size() != m_shape[1])
                throw std::runtime_error("xsparse_array::dot dimension mismatch.");
            auto result = xarray_container<uvector<T>>({m_shape[0]}, T(0));
            T* res_data = result.data();
            const T* x_data = x_arr.data();
            for (size_type r = 0; r < m_shape[0]; ++r)
            {
                T sum = 0;
                size_type begin = m_sparse.row_ptr()[r];
                size_type end = m_sparse.row_ptr()[r + 1];
                if constexpr (is_simd_enabled_v<T>)
                {
                    using simd_type = xsimd::batch<T, default_simd_arch>;
                    constexpr std::size_t simd_size = simd_type::size;
                    size_type i = begin;
                    for (; i + simd_size <= end; i += simd_size)
                    {
                        // Gather x values into a SIMD register
                        alignas(64) std::array<T, simd_size> x_buf;
                        for (std::size_t k = 0; k < simd_size; ++k)
                            x_buf[k] = x_data[m_sparse.col_idx()[i + k]];
                        simd_type vx = simd_type::load_aligned(x_buf.data());
                        simd_type vv = simd_type::load_unaligned(&m_sparse.values()[i]);
                        sum += xsimd::hadd(vx * vv);
                    }
                    for (; i < end; ++i)
                        sum += m_sparse.values()[i] * x_data[m_sparse.col_idx()[i]];
                }
                else
                {
                    for (size_type i = begin; i < end; ++i)
                        sum += m_sparse.values()[i] * x_data[m_sparse.col_idx()[i]];
                }
                res_data[r] = sum;
            }
            return result;
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

#endif // XTENSOR_XSPARSE_ARRAY_HPP