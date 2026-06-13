//File 0602 : xtensor-signal/xcorrelate.hpp
//Correlation operations (1D/2D) with SIMD‑accelerated inner loops, FFT‑based cross‑correlation, normalized and phase correlation, and expression integration.
#ifndef XTENSOR_SIGNAL_XCORRELATE_HPP
#define XTENSOR_SIGNAL_XCORRELATE_HPP

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
#include "xconvolve.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xfunction.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xtensor_simd.hpp"
#include "xtensor/xfft.hpp"

namespace xt {
namespace signal {

    namespace detail {

        /**
         * Flip a 1D kernel (reverse order). This is used because correlation
         * is equivalent to convolution with reversed kernel.
         */
        template <class T>
        inline auto reverse_array(const T* data, std::size_t n) {
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n});
            for (std::size_t i = 0; i < n; ++i)
                result[i] = data[n - 1 - i];
            return result;
        }

        /**
         * Flip a 2D kernel along both axes.
         */
        template <class T>
        inline auto reverse_2d(const xarray_container<uvector<T>>& kernel) {
            auto sh = kernel.shape();
            auto rev = xt::flip(xt::flip(kernel, 0), 1);
            return rev;
        }

        /**
         * Compute normalized cross-correlation (NCC) between two 1D signals.
         * NCC[k] = sum_i (a[i]-mean_a) * (b[i-k]-mean_b) / (std_a * std_b * n)
         */
        template <class T>
        inline auto normalized_cross_correlation(const T* a, std::size_t na,
                                                  const T* b, std::size_t nb) {
            T mean_a = T(0), mean_b = T(0);
            for (std::size_t i = 0; i < na; ++i) mean_a += a[i];
            for (std::size_t i = 0; i < nb; ++i) mean_b += b[i];
            mean_a /= static_cast<T>(na);
            mean_b /= static_cast<T>(nb);

            T std_a = T(0), std_b = T(0);
            for (std::size_t i = 0; i < na; ++i) {
                T diff = a[i] - mean_a;
                std_a += diff * diff;
            }
            for (std::size_t i = 0; i < nb; ++i) {
                T diff = b[i] - mean_b;
                std_b += diff * diff;
            }
            std_a = std::sqrt(std_a / static_cast<T>(na));
            std_b = std::sqrt(std_b / static_cast<T>(nb));

            T denom = std_a * std_b * static_cast<T>(na);
            if (denom == T(0)) denom = T(1);

            // Correlation via convolution with reversed b
            auto b_rev = reverse_array(b, nb);
            auto conv = convolve1d_direct_simd(a, na, b_rev.data(), nb);
            for (std::size_t k = 0; k < conv.size(); ++k)
                conv[k] = (conv[k] - mean_a * mean_b * static_cast<T>(na)) / denom;

            return conv;
        }

        /**
         * Compute phase correlation for image registration (2D).
         * Returns the correlation surface (same shape as a).
         */
        template <class T>
        inline auto phase_correlate2d(const xarray_container<uvector<T>>& a,
                                       const xarray_container<uvector<T>>& b) {
            using complex_type = std::complex<T>;
            auto sh = a.shape();
            // Ensure same size by padding the smaller
            std::size_t m = std::max(a.shape()[0], b.shape()[0]);
            std::size_t n = std::max(a.shape()[1], b.shape()[1]);
            // Pad to m x n
            auto a_pad = xt::pad(a, {{0, m - sh[0]}, {0, n - sh[1]}}, pad::mode::constant, T(0));
            auto b_pad = xt::pad(b, {{0, m - sh[0]}, {0, n - sh[1]}}, pad::mode::constant, T(0));

            // FFT of both
            auto A = xt::fft::fftn(a_pad);
            auto B = xt::fft::fftn(b_pad);
            // Normalized cross‑power spectrum: R = A * conj(B) / |A * conj(B)|
            for (std::size_t i = 0; i < A.size(); ++i) {
                complex_type num = A[i] * std::conj(B[i]);
                T mag = std::abs(num);
                if (mag > T(1e-15))
                    A[i] = num / mag;
                else
                    A[i] = complex_type(T(0), T(0));
            }
            // Inverse FFT to get correlation surface
            auto corr = xt::fft::ifftn(A);
            return xt::real(corr);
        }
    }

    /**
     * 1D cross‑correlation of two expressions.
     * Equivalent to convolution with reversed kernel.
     * @param a First input (1D).
     * @param b Second input (1D).
     * @param normalized If true, returns normalized cross‑correlation (NCC).
     * @return Cross‑correlation result (full length).
     */
    template <class E1, class E2>
    inline auto correlate1d(const xexpression<E1>& a,
                            const xexpression<E2>& b,
                            bool normalized = false) {
        using T = std::common_type_t<typename std::decay_t<E1>::value_type,
                                      typename std::decay_t<E2>::value_type>;
        auto arr_a = xt::eval(a.derived_cast());
        auto arr_b = xt::eval(b.derived_cast());

        if (arr_a.dimension() != 1 || arr_b.dimension() != 1)
            throw std::runtime_error("correlate1d: both inputs must be 1D.");
        std::size_t na = arr_a.size(), nb = arr_b.size();

        if (normalized) {
            return detail::normalized_cross_correlation(arr_a.data(), na, arr_b.data(), nb);
        } else {
            auto b_rev = detail::reverse_array(arr_b.data(), nb);
            return detail::convolve1d_direct_simd(arr_a.data(), na, b_rev.data(), nb);
        }
    }

    /**
     * 2D cross‑correlation of two expressions.
     * Equivalent to convolution with reversed kernel.
     * @param a First input (2D).
     * @param b Second input (template, 2D).
     * @param normalized If true, returns normalized cross‑correlation.
     * @return Cross‑correlation result (full size).
     */
    template <class E1, class E2>
    inline auto correlate2d(const xexpression<E1>& a,
                            const xexpression<E2>& b,
                            bool normalized = false) {
        using T = std::common_type_t<typename std::decay_t<E1>::value_type,
                                      typename std::decay_t<E2>::value_type>;
        auto arr_a = xt::eval(a.derived_cast());
        auto arr_b = xt::eval(b.derived_cast());

        if (arr_a.dimension() != 2 || arr_b.dimension() != 2)
            throw std::runtime_error("correlate2d: both inputs must be 2D.");
        std::size_t m1 = arr_a.shape()[0], n1 = arr_a.shape()[1];
        std::size_t m2 = arr_b.shape()[0], n2 = arr_b.shape()[1];

        if (normalized) {
            // Normalized correlation is phase correlation
            return detail::phase_correlate2d(arr_a, arr_b);
        } else {
            auto rev_b = detail::reverse_2d(arr_b);
            return detail::convolve2d_direct_simd(arr_a.data(), m1, n1,
                                                   rev_b.data(), m2, n2);
        }
    }

    /**
     * Auto‑correlation of a 1D signal (correlation with itself).
     */
    template <class E>
    inline auto autocorrelate1d(const xexpression<E>& a, bool normalized = false) {
        return correlate1d(a, a, normalized);
    }

    /**
     * Phase correlation for 2D image registration.
     * Returns the correlation surface (same shape as first input after padding).
     */
    template <class E1, class E2>
    inline auto phase_correlate(const xexpression<E1>& a, const xexpression<E2>& b) {
        using T = typename std::decay_t<E1>::value_type;
        auto arr_a = xt::eval(a.derived_cast());
        auto arr_b = xt::eval(b.derived_cast());
        return detail::phase_correlate2d(arr_a, arr_b);
    }

} // namespace signal
} // namespace xt

#endif // XTENSOR_SIGNAL_XCORRELATE_HPP