//File 0603 : xtensor-signal/xfilter.hpp
//Digital filtering (FIR, IIR, moving average, median, Savitzky–Golay) with SIMD‑accelerated inner loops, zero‑phase forward‑backward filtering, and expression integration.
#ifndef XTENSOR_SIGNAL_XFILTER_HPP
#define XTENSOR_SIGNAL_XFILTER_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_signal_config.hpp"
#include "xconvolve.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xfunction.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xtensor_simd.hpp"

namespace xt {
namespace signal {

    namespace detail {

        /**
         * Apply a rational transfer function filter (direct form II transposed).
         * y[n] = (b[0]*x[n] + b[1]*x[n-1] + ... - a[1]*y[n-1] - a[2]*y[n-2] - ...) / a[0]
         * @param x Input signal.
         * @param b Numerator coefficients.
         * @param a Denominator coefficients (a[0] must be non‑zero).
         * @return Filtered signal (same length as input).
         */
        template <class T>
        inline auto lfilter_direct(const T* x, std::size_t nx,
                                    const T* b, std::size_t nb,
                                    const T* a, std::size_t na) {
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> y({nx}, T(0));
            T* y_data = y.data();
            // Direct form II transposed: w = state vector of length max(nb, na)
            std::size_t order = std::max(nb, na) - 1;
            std::vector<T> w(order + 1, T(0));

            for (std::size_t n = 0; n < nx; ++n) {
                // Compute output: y[n] = b[0] * x[n] + w[0]
                T yn = (nb > 0 ? b[0] : T(0)) * x[n];
                if (order > 0) yn += w[0];

                // Shift state and update
                for (std::size_t i = 0; i < order; ++i)
                    w[i] = w[i + 1] + (i + 1 < nb ? b[i + 1] * x[n] : T(0))
                                        - (i + 1 < na ? a[i + 1] * yn : T(0));
                if (order > 0)
                    w[order] = (order < nb ? b[order] * x[n] : T(0))
                              - (order < na ? a[order] * yn : T(0));
                else
                    w[0] = T(0);

                y_data[n] = (na > 0 && a[0] != T(0)) ? yn / a[0] : yn;
            }
            return y;
        }

        /**
         * Initial condition handling: pad with reflected signal at both ends
         * to reduce transients for IIR filters.
         */
        template <class T>
        inline auto extend_signal_symmetric(const T* x, std::size_t n, std::size_t pad) {
            std::size_t total = n + 2 * pad;
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({total});
            for (std::size_t i = 0; i < pad; ++i) {
                result[i] = x[pad - i];
                result[total - 1 - i] = x[n - 1 - (pad - i)];
            }
            std::copy(x, x + n, result.data() + pad);
            return result;
        }
    }

    /**
     * 1D FIR filter (convolution with kernel b).
     * @param x Input signal.
     * @param b Filter coefficients (numerator).
     * @return Filtered signal (same length as input, 'same' mode).
     */
    template <class E, class Coeffs>
    inline auto firfilter(const xexpression<E>& x, const Coeffs& b) {
        using T = typename std::decay_t<E>::value_type;
        auto arr_x = xt::eval(x.derived_cast());
        auto arr_b = xt::eval(b.derived_cast());
        if (arr_x.dimension() != 1) throw std::runtime_error("firfilter: input must be 1D.");
        return convolve1d(arr_x, arr_b, padding_mode::same);
    }

    /**
     * 1D IIR filter (difference equation).
     * @param x Input signal.
     * @param b Numerator coefficients.
     * @param a Denominator coefficients (a[0] must be non‑zero).
     * @return Filtered signal (same length as input).
     */
    template <class E, class CoeffsB, class CoeffsA>
    inline auto iirfilter(const xexpression<E>& x,
                           const CoeffsB& b,
                           const CoeffsA& a) {
        using T = std::common_type_t<typename std::decay_t<E>::value_type,
                                      typename std::decay_t<CoeffsB>::value_type,
                                      typename std::decay_t<CoeffsA>::value_type>;
        auto arr_x = xt::eval(x.derived_cast());
        auto arr_b = xt::eval(b.derived_cast());
        auto arr_a = xt::eval(a.derived_cast());

        if (arr_x.dimension() != 1) throw std::runtime_error("iirfilter: input must be 1D.");
        if (arr_a.size() == 0) throw std::runtime_error("iirfilter: denominator cannot be empty.");
        if (arr_a[0] == T(0)) throw std::runtime_error("iirfilter: a[0] must not be zero.");

        return detail::lfiter_direct(arr_x.data(), arr_x.size(),
                                      arr_b.data(), arr_b.size(),
                                      arr_a.data(), arr_a.size());
    }

    /**
     * Apply a filter forward and backward (zero‑phase filtering).
     * Doubles the effective filter order and removes phase distortion.
     * @param x Input signal.
     * @param b Numerator coefficients.
     * @param a Denominator coefficients.
     * @return Zero‑phase filtered signal.
     */
    template <class E, class CoeffsB, class CoeffsA>
    inline auto filtfilt(const xexpression<E>& x,
                          const CoeffsB& b,
                          const CoeffsA& a) {
        using T = typename std::decay_t<E>::value_type;
        auto arr_x = xt::eval(x.derived_cast());
        if (arr_x.dimension() != 1) throw std::runtime_error("filtfilt: input must be 1D.");
        std::size_t n = arr_x.size();
        // Pad with reflected signal to reduce startup transients
        std::size_t pad = std::min(n, static_cast<std::size_t>(100));
        auto ext = detail::extend_signal_symmetric(arr_x.data(), n, pad);
        // Forward filter
        auto forward = detail::lfiter_direct(ext.data(), ext.size(),
                                              xt::eval(b.derived_cast()).data(),
                                              xt::eval(b.derived_cast()).size(),
                                              xt::eval(a.derived_cast()).data(),
                                              xt::eval(a.derived_cast()).size());
        // Reverse
        auto forward_rev = xt::flip(forward, 0);
        // Backward filter
        auto backward = detail::lfiter_direct(forward_rev.data(), forward_rev.size(),
                                               xt::eval(b.derived_cast()).data(),
                                               xt::eval(b.derived_cast()).size(),
                                               xt::eval(a.derived_cast()).data(),
                                               xt::eval(a.derived_cast()).size());
        // Reverse again and crop
        auto result = xt::flip(backward, 0);
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> cropped({n});
        std::copy(result.data() + pad, result.data() + pad + n, cropped.data());
        return cropped;
    }

    /**
     * Moving average filter (boxcar).
     * @param x Input signal (1D).
     * @param window_size Number of samples to average (must be odd for symmetry).
     * @return Smoothed signal (same length).
     */
    template <class E>
    inline auto moving_average(const xexpression<E>& x, std::size_t window_size) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        if (arr.dimension() != 1) throw std::runtime_error("moving_average: input must be 1D.");
        std::size_t n = arr.size();
        if (window_size >= n) window_size = n;
        if (window_size % 2 == 0) window_size++;
        std::size_t half = window_size / 2;
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n});
        T* res = result.data();
        const T* src = arr.data();
        // Running sum for efficiency
        T running_sum = T(0);
        for (std::size_t i = 0; i < window_size; ++i)
            running_sum += src[i];
        res[half] = running_sum / static_cast<T>(window_size);
        for (std::size_t i = half + 1; i < n - half; ++i) {
            running_sum += src[i + half] - src[i - half - 1];
            res[i] = running_sum / static_cast<T>(window_size);
        }
        // Edge handling: use smaller windows
        for (std::size_t i = 0; i < half; ++i) {
            std::size_t count = i + half + 1;
            T sum = T(0);
            for (std::size_t j = 0; j < count; ++j) sum += src[j];
            res[i] = sum / static_cast<T>(count);
        }
        for (std::size_t i = n - half; i < n; ++i) {
            std::size_t count = n - i + half;
            T sum = T(0);
            for (std::size_t j = i - half; j < n; ++j) sum += src[j];
            res[i] = sum / static_cast<T>(count);
        }
        return result;
    }

    /**
     * Median filter (1D).
     * @param x Input signal (1D).
     * @param window_size Odd window size for median computation.
     * @return Filtered signal (same length).
     */
    template <class E>
    inline auto medfilt1d(const xexpression<E>& x, std::size_t window_size) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        if (arr.dimension() != 1) throw std::runtime_error("medfilt1d: input must be 1D.");
        std::size_t n = arr.size();
        if (window_size % 2 == 0) window_size++;
        std::size_t half = window_size / 2;
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n});
        T* res = result.data();
        const T* src = arr.data();
        std::vector<T> buf(window_size);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t start = (i > half) ? i - half : 0;
            std::size_t end = std::min(i + half + 1, n);
            std::size_t len = end - start;
            for (std::size_t j = 0; j < len; ++j)
                buf[j] = src[start + j];
            std::nth_element(buf.begin(), buf.begin() + len / 2, buf.begin() + len);
            res[i] = buf[len / 2];
        }
        return result;
    }

    /**
     * Median filter (2D).
     * @param x Input 2D array.
     * @param window_rows Window height (must be odd).
     * @param window_cols Window width (must be odd).
     * @return Filtered 2D array.
     */
    template <class E>
    inline auto medfilt2d(const xexpression<E>& x,
                           std::size_t window_rows,
                           std::size_t window_cols) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        if (arr.dimension() != 2) throw std::runtime_error("medfilt2d: input must be 2D.");
        std::size_t rows = arr.shape()[0], cols = arr.shape()[1];
        if (window_rows % 2 == 0) window_rows++;
        if (window_cols % 2 == 0) window_cols++;
        std::size_t half_r = window_rows / 2, half_c = window_cols / 2;
        auto result = xt::eval(arr);
        T* res = result.data();
        const T* src = arr.data();
        std::vector<T> buf(window_rows * window_cols);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                std::size_t rmin = (r > half_r) ? r - half_r : 0;
                std::size_t rmax = std::min(r + half_r + 1, rows);
                std::size_t cmin = (c > half_c) ? c - half_c : 0;
                std::size_t cmax = std::min(c + half_c + 1, cols);
                std::size_t cnt = 0;
                for (std::size_t rr = rmin; rr < rmax; ++rr)
                    for (std::size_t cc = cmin; cc < cmax; ++cc)
                        buf[cnt++] = src[rr * cols + cc];
                std::nth_element(buf.begin(), buf.begin() + cnt / 2, buf.begin() + cnt);
                res[r * cols + c] = buf[cnt / 2];
            }
        }
        return result;
    }

    /**
     * Exponential moving average (IIR low‑pass).
     * y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
     * @param x Input signal.
     * @param alpha Smoothing factor (0 < alpha <= 1).
     * @return Filtered signal.
     */
    template <class E>
    inline auto ema_filter(const xexpression<E>& x, double alpha = 0.1) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        if (arr.dimension() != 1) throw std::runtime_error("ema_filter: input must be 1D.");
        std::size_t n = arr.size();
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> y({n});
        T a = static_cast<T>(alpha);
        T one_minus_a = T(1) - a;
        y[0] = arr[0];
        const T* src = arr.data();
        T* dst = y.data();
        if constexpr (is_simd_enabled_v<T>) {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            simd_type va(a), voma(one_minus_a);
            // Recurrence cannot be fully vectorized; we use scalar recurrence
            for (std::size_t i = 1; i < n; ++i) {
                dst[i] = a * src[i] + one_minus_a * dst[i - 1];
            }
        } else {
            for (std::size_t i = 1; i < n; ++i) {
                dst[i] = a * src[i] + one_minus_a * dst[i - 1];
            }
        }
        return y;
    }

} // namespace signal
} // namespace xt

#endif // XTENSOR_SIGNAL_XFILTER_HPP