//File 0202 : sparse/xsparse_reducer.hpp
//Sparse reduction operations: sum, prod, amax, amin with SIMD aggregation, parallel chunking, and sparse-aware traversal.
#ifndef XTENSOR_XSPARSE_REDUCER_HPP
#define XTENSOR_XSPARSE_REDUCER_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <thread>
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
#include "../sparse/xsparse_array.hpp"
#include "../sparse/xsparse_tensor.hpp"

namespace xt {
namespace sparse {

    namespace detail
    {
        /**
         * Sequential reduction over CSR values and indices.
         */
        template <class T, class BinaryOp, class Init>
        inline T reduce_csr_values(const csr_matrix<T>& mat, BinaryOp op, Init init_val)
        {
            T result = init_val;
            const T* vals = mat.values().data();
            std::size_t nnz = mat.nnz();
            if constexpr (is_simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = nnz / simd_size;
                simd_type vacc(init_val);
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type vdata = simd_type::load_unaligned(vals + i * simd_size);
                    vacc = op.simd_apply(vacc, vdata);
                }
                T tmp[simd_size];
                vacc.store_unaligned(tmp);
                for (std::size_t k = 0; k < simd_size; ++k) result = op(result, tmp[k]);
                for (std::size_t i = vec_count * simd_size; i < nnz; ++i)
                    result = op(result, vals[i]);
            }
            else
            {
                for (std::size_t i = 0; i < nnz; ++i)
                    result = op(result, vals[i]);
            }
            return result;
        }

        /**
         * Parallel reduction over CSR values using divide-and-conquer.
         */
        template <class T, class BinaryOp, class Init>
        inline T reduce_csr_values_parallel(const csr_matrix<T>& mat, BinaryOp op, Init init_val)
        {
            std::size_t nnz = mat.nnz();
            const T* vals = mat.values().data();
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 2;
            if (nnz < num_threads * 1024)
                return reduce_csr_values(mat, op, init_val);

            std::size_t chunk = (nnz + num_threads - 1) / num_threads;
            std::vector<T> partials(num_threads, init_val);
            std::vector<std::thread> threads;
            for (unsigned int t = 0; t < num_threads; ++t)
            {
                threads.emplace_back([&, t]() {
                    std::size_t start = t * chunk;
                    std::size_t end = std::min(start + chunk, nnz);
                    T local = init_val;
                    for (std::size_t i = start; i < end; ++i)
                        local = op(local, vals[i]);
                    partials[t] = local;
                });
            }
            for (auto& th : threads) th.join();
            T result = init_val;
            for (auto& p : partials) result = op(result, p);
            return result;
        }
    }

    /**
     * Sum of all non-zero elements in a sparse expression.
     */
    template <class E>
    inline auto spsum(const xexpression<E>& e)
    {
        const auto& expr = e.derived();
        using T = typename E::value_type;
        const auto& sp = expr.sparse_storage();
        return detail::reduce_csr_values_parallel(sp, detail::plus{}, T(0));
    }

    /**
     * Product of all non-zero elements.
     */
    template <class E>
    inline auto spprod(const xexpression<E>& e)
    {
        const auto& expr = e.derived();
        using T = typename E::value_type;
        const auto& sp = expr.sparse_storage();
        return detail::reduce_csr_values_parallel(sp, detail::multiplies{}, T(1));
    }

    /**
     * Maximum of non-zero elements, with explicit zero considered if sparse count < total elements.
     */
    template <class E>
    inline auto spmax(const xexpression<E>& e)
    {
        const auto& expr = e.derived();
        using T = typename E::value_type;
        const auto& sp = expr.sparse_storage();
        T max_val = detail::reduce_csr_values_parallel(sp, detail::maximum{},
                                                       std::numeric_limits<T>::lowest());
        // If there are implicit zeros, the maximum could be zero if all explicit are negative.
        if (sp.nnz() < expr.size() && T(0) > max_val)
            max_val = T(0);
        return max_val;
    }

    /**
     * Minimum of non-zero elements, with explicit zero considered if sparse count < total elements.
     */
    template <class E>
    inline auto spmin(const xexpression<E>& e)
    {
        const auto& expr = e.derived();
        using T = typename E::value_type;
        const auto& sp = expr.sparse_storage();
        T min_val = detail::reduce_csr_values_parallel(sp, detail::minimum{},
                                                       std::numeric_limits<T>::max());
        if (sp.nnz() < expr.size() && T(0) < min_val)
            min_val = T(0);
        return min_val;
    }

    /**
     * Sparse-dense vector dot product using SIMD: sum_i A_row_i * x_col_i for all non-zero.
     */
    template <class E1, class E2>
    inline auto spdot(const xexpression<E1>& sparse_mat, const xexpression<E2>& dense_vec)
    {
        const auto& sp = sparse_mat.derived();
        const auto& dv = dense_vec.derived();
        using T = std::common_type_t<typename E1::value_type, typename E2::value_type>;
        const auto& csr = sp.sparse_storage();
        if (dv.dimension() != 1 || dv.size() != csr.cols())
            throw std::runtime_error("spdot: dimension mismatch.");
        xarray_container<uvector<T>> result({csr.rows()}, T(0));
        T* res = result.data();
        const T* x = dv.data();
        for (std::size_t r = 0; r < csr.rows(); ++r)
        {
            T sum = 0;
            std::size_t beg = csr.row_ptr()[r];
            std::size_t end = csr.row_ptr()[r + 1];
            const std::size_t* cols = csr.col_idx().data();
            const T* vals = csr.values().data();
            if constexpr (is_simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t i = beg;
                for (; i + simd_size <= end; i += simd_size)
                {
                    alignas(64) std::array<T, simd_size> x_buf;
                    alignas(64) std::array<T, simd_size> v_buf;
                    for (std::size_t k = 0; k < simd_size; ++k)
                    {
                        x_buf[k] = x[cols[i + k]];
                        v_buf[k] = vals[i + k];
                    }
                    simd_type vx = simd_type::load_aligned(x_buf.data());
                    simd_type vv = simd_type::load_aligned(v_buf.data());
                    simd_type prod = vx * vv;
                    sum += xsimd::hadd(prod);
                }
                for (; i < end; ++i)
                    sum += vals[i] * x[cols[i]];
            }
            else
            {
                for (std::size_t i = beg; i < end; ++i)
                    sum += vals[i] * x[cols[i]];
            }
            res[r] = sum;
        }
        return result;
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_REDUCER_HPP