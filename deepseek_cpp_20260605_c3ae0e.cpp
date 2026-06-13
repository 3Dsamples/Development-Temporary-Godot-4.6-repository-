//File 0203 : sparse/xsparse_assign.hpp
//Sparse assignment engine: dense-to-sparse conversion with SIMD, sparse-to-dense, and sparse-sparse element-wise operations.
#ifndef XTENSOR_XSPARSE_ASSIGN_HPP
#define XTENSOR_XSPARSE_ASSIGN_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

    namespace detail
    {
        /**
         * Convert a dense 1D expression to COO entries with SIMD filtering of zeros.
         */
        template <class T>
        inline void dense1d_to_coo(const T* data, std::size_t n, coo_matrix<T>& coo)
        {
            constexpr std::size_t block_size = 64;
            for (std::size_t i = 0; i < n; i += block_size)
            {
                std::size_t end = std::min(i + block_size, n);
                if constexpr (is_simd_enabled_v<T>)
                {
                    using simd_type = xsimd::batch<T, default_simd_arch>;
                    constexpr std::size_t simd_size = simd_type::size;
                    for (std::size_t j = i; j + simd_size <= end; j += simd_size)
                    {
                        simd_type vals = simd_type::load_unaligned(data + j);
                        // compare with zero
                        auto mask = vals != simd_type(0);
                        alignas(64) std::array<T, simd_size> buf;
                        vals.store_aligned(buf.data());
                        alignas(64) std::array<bool, simd_size> mbuf;
                        mask.store_aligned(mbuf.data());
                        for (std::size_t k = 0; k < simd_size; ++k)
                            if (mbuf[k])
                                coo.append(0, j + k, buf[k]);
                    }
                }
                for (std::size_t j = i; j < end; ++j)
                    if (data[j] != T(0))
                        coo.append(0, j, data[j]);
            }
        }

        /**
         * Convert a dense 2D expression to COO entries with SIMD filtering of zeros.
         */
        template <class T>
        inline void dense2d_to_coo(const T* data, std::size_t rows, std::size_t cols,
                                   coo_matrix<T>& coo)
        {
            for (std::size_t i = 0; i < rows; ++i)
            {
                const T* row = data + i * cols;
                if constexpr (is_simd_enabled_v<T>)
                {
                    using simd_type = xsimd::batch<T, default_simd_arch>;
                    constexpr std::size_t simd_size = simd_type::size;
                    for (std::size_t j = 0; j + simd_size <= cols; j += simd_size)
                    {
                        simd_type vals = simd_type::load_unaligned(row + j);
                        auto mask = vals != simd_type(0);
                        alignas(64) std::array<T, simd_size> buf;
                        vals.store_aligned(buf.data());
                        alignas(64) std::array<bool, simd_size> mbuf;
                        mask.store_aligned(mbuf.data());
                        for (std::size_t k = 0; k < simd_size; ++k)
                            if (mbuf[k])
                                coo.append(i, j + k, buf[k]);
                    }
                }
                else
                {
                    for (std::size_t j = 0; j < cols; ++j)
                        if (row[j] != T(0))
                            coo.append(i, j, row[j]);
                }
            }
        }

        /**
         * Convert COO to dense 1D/2D array.
         */
        template <class T>
        inline void coo_to_dense(const coo_matrix<T>& coo,
                                 xarray_container<uvector<T>>& dense)
        {
            dense.fill(T(0));
            for (std::size_t i = 0; i < coo.nnz(); ++i)
                dense(coo.row_indices()[i], coo.col_indices()[i]) = coo.values()[i];
        }
    }

    /**
     * Assign a dense expression to a sparse array.
     */
    template <class E, class SpExpr>
    inline void assign_dense_to_sparse(const xexpression<E>& dense,
                                        SpExpr& sparse_expr)
    {
        const auto& src = dense.derived();
        auto& dst = sparse_expr.derived();
        auto sh = src.shape();
        dst.set_shape(sh);
        if (sh.size() == 1)
        {
            coo_matrix<typename E::value_type> coo(1, sh[0]);
            detail::dense1d_to_coo(src.data(), sh[0], coo);
            dst.sparse_storage() = csr_matrix<typename E::value_type>::from_coo(coo);
        }
        else if (sh.size() == 2)
        {
            coo_matrix<typename E::value_type> coo(sh[0], sh[1]);
            detail::dense2d_to_coo(src.data(), sh[0], sh[1], coo);
            dst.sparse_storage() = csr_matrix<typename E::value_type>::from_coo(coo);
        }
        else
        {
            throw std::runtime_error("assign_dense_to_sparse: dimension not supported.");
        }
    }

    /**
     * Assign a sparse expression to a dense array.
     */
    template <class SpExpr, class E>
    inline void assign_sparse_to_dense(const SpExpr& sparse_expr,
                                        xexpression<E>& dense)
    {
        const auto& src = sparse_expr.derived();
        auto& dst = dense.derived();
        auto sh = src.shape();
        if (sh.size() == 1)
        {
            auto sp = src.sparse_storage();
            dst.resize({sp.cols()});
            std::fill(dst.data(), dst.data() + dst.size(), 0);
            for (std::size_t i = sp.row_ptr()[0]; i < sp.row_ptr()[1]; ++i)
                dst[sp.col_idx()[i]] = sp.values()[i];
        }
        else if (sh.size() == 2)
        {
            auto sp = src.sparse_storage();
            dst.resize({sp.rows(), sp.cols()});
            std::fill(dst.data(), dst.data() + dst.size(), 0);
            for (std::size_t r = 0; r < sp.rows(); ++r)
                for (std::size_t i = sp.row_ptr()[r]; i < sp.row_ptr()[r + 1]; ++i)
                    dst(r, sp.col_idx()[i]) = sp.values()[i];
        }
        else
        {
            throw std::runtime_error("assign_sparse_to_dense: dimension not supported.");
        }
    }

    /**
     * Element-wise sparse + sparse addition: C = A + B.
     */
    template <class T>
    inline auto spadd(const csr_matrix<T>& A, const csr_matrix<T>& B)
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            throw std::runtime_error("spadd: shape mismatch.");
        return add(A, B);
    }

    /**
     * Element-wise sparse - sparse subtraction: C = A - B.
     */
    template <class T>
    inline auto spsubtract(const csr_matrix<T>& A, const csr_matrix<T>& B)
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            throw std::runtime_error("spsubtract: shape mismatch.");
        // Copy B, negate values, then add
        auto negB = B;
        for (auto& v : negB.values()) v = -v;
        return add(A, negB);
    }

    /**
     * Element-wise sparse * scalar multiplication.
     */
    template <class T>
    inline auto spscale(const csr_matrix<T>& A, T alpha)
    {
        auto result = A;
        for (auto& v : result.values()) v *= alpha;
        return result;
    }

    /**
     * Sparse matrix transpose.
     */
    template <class T>
    inline auto sptranspose(const csr_matrix<T>& A)
    {
        return A.transpose();
    }

    /**
     * Convert a sparse array to a dense one, returning a new xarray.
     */
    template <class E>
    inline auto to_dense(const xexpression<E>& sparse_expr)
    {
        const auto& sp = sparse_expr.derived();
        using T = typename E::value_type;
        auto sh = sp.shape();
        if (sh.size() == 1)
        {
            xarray_container<uvector<T>> result({sp.sparse_storage().cols()}, T(0));
            const auto& csr = sp.sparse_storage();
            for (std::size_t i = csr.row_ptr()[0]; i < csr.row_ptr()[1]; ++i)
                result[csr.col_idx()[i]] = csr.values()[i];
            return result;
        }
        else if (sh.size() == 2)
        {
            xarray_container<uvector<T>> result({csr.rows(), csr.cols()}, T(0));
            const auto& csr = sp.sparse_storage();
            for (std::size_t r = 0; r < csr.rows(); ++r)
                for (std::size_t i = csr.row_ptr()[r]; i < csr.row_ptr()[r + 1]; ++i)
                    result(r, csr.col_idx()[i]) = csr.values()[i];
            return result;
        }
        throw std::runtime_error("to_dense: dimension not supported.");
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_ASSIGN_HPP