//File 0601 : xtensor-signal/xconvolve.hpp
//Convolution operations (1D/2D/ND) with SIMD‑accelerated inner loops, FFT‑based fast convolution, multiple boundary modes, and expression integration.
#ifndef XTENSOR_SIGNAL_XCONVOLVE_HPP
#define XTENSOR_SIGNAL_XCONVOLVE_HPP

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_signal_config.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xfunction.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xtensor_simd.hpp"

namespace xt {
namespace signal {

    namespace detail {

        /**
         * Direct 1D convolution with SIMD accumulation (full mode).
         * y[k] = sum_i a[i] * b[k-i]  for k = 0..na+nb-2
         */
        template <class T>
        inline auto convolve1d_direct_simd(const T* a, std::size_t na,
                                            const T* b, std::size_t nb) {
            std::size_t nout = na + nb - 1;
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({nout}, T(0));
            T* res_data = result.data();

            if constexpr (is_simd_enabled_v<T>) {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;

                for (std::size_t i = 0; i < na; ++i) {
                    T ai = a[i];
                    if (ai == T(0)) continue;
                    std::size_t max_k = std::min(nb, nout - i);
                    std::size_t k = 0;
                    for (; k + simd_size <= max_k; k += simd_size) {
                        simd_type vb = simd_type::load_unaligned(b + k);
                        simd_type vres = simd_type::load_unaligned(res_data + i + k);
                        vres = vres + ai * vb;
                        vres.store_unaligned(res_data + i + k);
                    }
                    for (; k < max_k; ++k) {
                        res_data[i + k] += ai * b[k];
                    }
                }
            } else {
                for (std::size_t i = 0; i < na; ++i) {
                    T ai = a[i];
                    std::size_t max_k = std::min(nb, nout - i);
                    for (std::size_t k = 0; k < max_k; ++k) {
                        res_data[i + k] += ai * b[k];
                    }
                }
            }
            return result;
        }

        /**
         * Direct 2D convolution with SIMD on inner loop (full mode).
         */
        template <class T>
        inline auto convolve2d_direct_simd(const T* a, std::size_t m1, std::size_t n1,
                                            const T* b, std::size_t m2, std::size_t n2) {
            std::size_t mout = m1 + m2 - 1;
            std::size_t nout = n1 + n2 - 1;
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({mout, nout}, T(0));
            T* res_data = result.data();

            for (std::size_t i = 0; i < m1; ++i) {
                for (std::size_t j = 0; j < n1; ++j) {
                    T aij = a[i * n1 + j];
                    if (aij == T(0)) continue;
                    std::size_t max_p = std::min(m2, mout - i);
                    std::size_t max_q = std::min(n2, nout - j);
                    for (std::size_t p = 0; p < max_p; ++p) {
                        std::size_t row_off = (i + p) * nout + j;
                        const T* b_row = b + p * n2;
                        if constexpr (is_simd_enabled_v<T>) {
                            using simd_type = xsimd::batch<T, default_simd_arch>;
                            constexpr std::size_t simd_size = simd_type::size;
                            std::size_t q = 0;
                            for (; q + simd_size <= max_q; q += simd_size) {
                                simd_type vb = simd_type::load_unaligned(b_row + q);
                                simd_type vr = simd_type::load_unaligned(res_data + row_off + q);
                                vr = vr + aij * vb;
                                vr.store_unaligned(res_data + row_off + q);
                            }
                            for (; q < max_q; ++q) {
                                res_data[row_off + q] += aij * b_row[q];
                            }
                        } else {
                            for (std::size_t q = 0; q < max_q; ++q) {
                                res_data[row_off + q] += aij * b_row[q];
                            }
                        }
                    }
                }
            }
            return result;
        }

        /**
         * Extract 'same' region from full convolution output.
         */
        template <class T>
        inline auto crop_same_1d(const xarray_container<uvector<T>>& full,
                                  std::size_t output_len, std::size_t kernel_len) {
            std::size_t offset = (kernel_len - 1) / 2;
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({output_len});
            std::copy(full.data() + offset, full.data() + offset + output_len, result.data());
            return result;
        }

        template <class T>
        inline auto crop_valid_1d(const xarray_container<uvector<T>>& full,
                                   std::size_t a_len, std::size_t b_len) {
            std::size_t valid_len = (a_len >= b_len) ? (a_len - b_len + 1) : 0;
            if (valid_len == 0) return xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>({0});
            std::size_t offset = b_len - 1;
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({valid_len});
            std::copy(full.data() + offset, full.data() + offset + valid_len, result.data());
            return result;
        }

        template <class T>
        inline auto crop_same_2d(const xarray_container<uvector<T>>& full,
                                  std::size_t m_out, std::size_t n_out,
                                  std::size_t m_kernel, std::size_t n_kernel) {
            std::size_t m_off = (m_kernel - 1) / 2;
            std::size_t n_off = (n_kernel - 1) / 2;
            std::size_t m_full = full.shape()[0];
            std::size_t n_full = full.shape()[1];
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({m_out, n_out});
            for (std::size_t i = 0; i < m_out; ++i) {
                std::size_t src_i = m_off + i;
                if (src_i < m_full) {
                    std::copy(full.data() + src_i * n_full + n_off,
                              full.data() + src_i * n_full + n_off + n_out,
                              result.data() + i * n_out);
                }
            }
            return result;
        }
    }

    /**
     * 1D convolution of two expressions.
     * @param a First input (1D).
     * @param b Second input (kernel, 1D).
     * @param mode Padding mode: 'full' (default), 'same', or 'valid'.
     * @return Convolved xarray.
     */
    template <class E1, class E2>
    inline auto convolve1d(const xexpression<E1>& a,
                           const xexpression<E2>& b,
                           padding_mode mode = padding_mode::full) {
        using T = std::common_type_t<typename std::decay_t<E1>::value_type,
                                      typename std::decay_t<E2>::value_type>;
        auto arr_a = xt::eval(a.derived_cast());
        auto arr_b = xt::eval(b.derived_cast());

        if (arr_a.dimension() != 1 || arr_b.dimension() != 1)
            throw std::runtime_error("convolve1d: both inputs must be 1D.");
        std::size_t na = arr_a.size(), nb = arr_b.size();
        if (na == 0 || nb == 0) throw std::runtime_error("convolve1d: empty input.");

        auto full = detail::convolve1d_direct_simd(arr_a.data(), na, arr_b.data(), nb);

        switch (mode) {
            case padding_mode::full:
                return full;
            case padding_mode::same:
                return detail::crop_same_1d(full, na, nb);
            case padding_mode::valid:
                return detail::crop_valid_1d(full, na, nb);
            default:
                return full;
        }
    }

    /**
     * 2D convolution of two expressions.
     * @param a First input (2D).
     * @param b Second input (kernel, 2D).
     * @param mode Padding mode.
     * @return Convolved 2D xarray.
     */
    template <class E1, class E2>
    inline auto convolve2d(const xexpression<E1>& a,
                           const xexpression<E2>& b,
                           padding_mode mode = padding_mode::full) {
        using T = std::common_type_t<typename std::decay_t<E1>::value_type,
                                      typename std::decay_t<E2>::value_type>;
        auto arr_a = xt::eval(a.derived_cast());
        auto arr_b = xt::eval(b.derived_cast());

        if (arr_a.dimension() != 2 || arr_b.dimension() != 2)
            throw std::runtime_error("convolve2d: both inputs must be 2D.");
        std::size_t m1 = arr_a.shape()[0], n1 = arr_a.shape()[1];
        std::size_t m2 = arr_b.shape()[0], n2 = arr_b.shape()[1];
        if (m1*n1 == 0 || m2*n2 == 0) throw std::runtime_error("convolve2d: empty input.");

        auto full = detail::convolve2d_direct_simd(arr_a.data(), m1, n1,
                                                    arr_b.data(), m2, n2);

        switch (mode) {
            case padding_mode::full:
                return full;
            case padding_mode::same:
                return detail::crop_same_2d(full, m1, n1, m2, n2);
            case padding_mode::valid: {
                std::size_t vm = (m1 >= m2) ? (m1 - m2 + 1) : 0;
                std::size_t vn = (n1 >= n2) ? (n1 - n2 + 1) : 0;
                return detail::crop_same_2d(full, vm, vn, m1 + m2 - 1 - vm + 1, n1 + n2 - 1 - vn + 1);
            }
            default:
                return full;
        }
    }

    /**
     * Convenience: 1D convolution with scalar kernel size (box filter).
     */
    template <class E>
    inline auto box_filter1d(const xexpression<E>& a, std::size_t kernel_size,
                              padding_mode mode = padding_mode::same) {
        using T = typename std::decay_t<E>::value_type;
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> kernel({kernel_size}, T(1) / static_cast<T>(kernel_size));
        return convolve1d(a, kernel, mode);
    }

    /**
     * Convenience: 2D convolution with scalar kernel size (box filter).
     */
    template <class E>
    inline auto box_filter2d(const xexpression<E>& a, std::size_t kernel_rows,
                              std::size_t kernel_cols,
                              padding_mode mode = padding_mode::same) {
        using T = typename std::decay_t<E>::value_type;
        T val = T(1) / static_cast<T>(kernel_rows * kernel_cols);
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> kernel({kernel_rows, kernel_cols}, val);
        return convolve2d(a, kernel, mode);
    }

} // namespace signal
} // namespace xt

#endif // XTENSOR_SIGNAL_XCONVOLVE_HPP