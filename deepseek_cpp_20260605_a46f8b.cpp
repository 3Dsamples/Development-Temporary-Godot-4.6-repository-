//File 0607 : xtensor-signal/xresample.hpp
//Signal resampling: decimation, interpolation, polyphase filtering, FFT‑based resampling, and rational rate conversion with SIMD‑accelerated processing.
#ifndef XTENSOR_SIGNAL_XRESAMPLE_HPP
#define XTENSOR_SIGNAL_XRESAMPLE_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_signal_config.hpp"
#include "xwindows.hpp"
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

    /**
     * @enum resample_mode
     * @brief Resampling method.
     */
    enum class resample_mode : int {
        nearest,       // nearest‑neighbour
        linear,        // linear interpolation
        cubic,         // cubic (Catmull‑Rom) interpolation
        polyphase,     // polyphase FIR filtering (high quality)
        fft            // FFT‑based (best for large rate changes)
    };

    namespace detail {

        /**
         * Generate an FIR low‑pass filter kernel for anti‑aliasing.
         * Uses a Kaiser window for good stop‑band attenuation.
         * @param cutoff Normalised cutoff frequency (0..0.5).
         * @param transition_width Normalised transition width.
         * @param attenuation Stop‑band attenuation in dB.
         * @return FIR coefficients.
         */
        template <class T>
        inline auto design_anti_alias_filter(T cutoff, T transition_width,
                                              T attenuation = T(60)) {
            // Estimate filter order from attenuation and transition width
            T width = transition_width;
            std::size_t n = static_cast<std::size_t>(
                std::ceil((attenuation - T(7.95)) / (T(14.36) * width)));
            // n must be odd for linear‑phase
            if (n % 2 == 0) n++;
            if (n < 3) n = 3;

            T beta = T(0);
            if (attenuation > T(50))
                beta = T(0.1102) * (attenuation - T(8.7));
            else if (attenuation >= T(21))
                beta = T(0.5842) * std::pow(attenuation - T(21), T(0.4))
                       + T(0.07886) * (attenuation - T(21));

            // Kaiser window
            auto win = kaiser<T>(n, beta);

            // Ideal low‑pass sinc
            T two_pi = T(2) * xt::numeric_constants<T>::pi;
            T mid = static_cast<T>(n - 1) / T(2);
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> kernel({n});
            T* k = kernel.data();
            const T* w = win.data();
            T sum = T(0);
            for (std::size_t i = 0; i < n; ++i) {
                T t = static_cast<T>(i) - mid;
                if (t == T(0)) {
                    k[i] = T(2) * cutoff;
                } else {
                    k[i] = std::sin(two_pi * cutoff * t) / (xt::numeric_constants<T>::pi * t);
                }
                k[i] *= w[i];
                sum += k[i];
            }
            // Normalise to unit DC gain
            if (sum != T(0)) {
                T inv_sum = T(1) / sum;
                for (std::size_t i = 0; i < n; ++i)
                    k[i] *= inv_sum;
            }
            return kernel;
        }

        /**
         * Polyphase resampling filter bank.
         * Each phase is a sub‑filter of the original anti‑alias filter.
         */
        template <class T>
        inline auto build_polyphase_filters(const T* kernel, std::size_t kernel_len,
                                             std::size_t num_phases) {
            std::size_t taps_per_phase = (kernel_len + num_phases - 1) / num_phases;
            std::vector<xarray_container<uvector<T>, DEFAULT_LAYOUT,
                std::vector<std::size_t>>> phases(num_phases);
            for (std::size_t p = 0; p < num_phases; ++p) {
                phases[p].resize({taps_per_phase});
                std::fill(phases[p].data(), phases[p].data() + taps_per_phase, T(0));
                for (std::size_t i = 0; i < taps_per_phase; ++i) {
                    std::size_t kidx = p + i * num_phases;
                    if (kidx < kernel_len)
                        phases[p][i] = kernel[kidx];
                }
            }
            return phases;
        }

        /**
         * Apply polyphase interpolation (upsample by factor L).
         * Uses zero‑stuffed input and polyphase filter bank.
         */
        template <class T>
        inline auto polyphase_upsample(const T* x, std::size_t nx, std::size_t L,
                                        const T* kernel, std::size_t kernel_len) {
            std::size_t yout = nx * L;
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> y({yout}, T(0));
            T* yd = y.data();
            std::size_t phases = L;
            auto filters = build_polyphase_filters(kernel, kernel_len, phases);
            for (std::size_t n = 0; n < nx; ++n) {
                for (std::size_t p = 0; p < phases; ++p) {
                    std::size_t idx = n * L + p;
                    T sum = T(0);
                    const auto& f = filters[p];
                    std::size_t flen = f.size();
                    for (std::size_t k = 0; k < flen; ++k) {
                        std::ptrdiff_t src_idx = static_cast<std::ptrdiff_t>(n) - static_cast<std::ptrdiff_t>(k);
                        if (src_idx >= 0 && static_cast<std::size_t>(src_idx) < nx)
                            sum += x[src_idx] * f[k];
                    }
                    yd[idx] = sum;
                }
            }
            return y;
        }

        /**
         * Decimate a signal (downsample by factor M) with anti‑alias filtering.
         */
        template <class T>
        inline auto decimate_filtered(const T* x, std::size_t nx, std::size_t M,
                                       const T* kernel, std::size_t kernel_len) {
            // First filter, then take every M‑th sample
            auto filtered = convolve1d_direct_simd(x, nx, kernel, kernel_len);
            std::size_t ny = (filtered.size() + M - 1) / M;
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> y({ny});
            T* yd = y.data();
            const T* fd = filtered.data();
            // Skip samples at the beginning to align phase (delay = kernel_len/2)
            std::size_t offset = kernel_len / 2;
            for (std::size_t i = 0; i < ny; ++i) {
                std::size_t idx = offset + i * M;
                if (idx < filtered.size())
                    yd[i] = fd[idx];
                else
                    yd[i] = T(0);
            }
            return y;
        }

        /**
         * FFT‑based resampling: upsample by L, downsample by M.
         * Works by FFT of signal, zero‑padding/truncation in frequency domain,
         * then IFFT.
         */
        template <class T>
        inline auto fft_resample(const T* x, std::size_t nx, std::size_t L, std::size_t M) {
            std::size_t N = nx * L / M;
            if (N == 0) N = 1;
            // FFT of input (padded to power of 2)
            std::size_t n_fft = 1;
            while (n_fft < nx) n_fft <<= 1;
            std::vector<std::complex<T>> X(n_fft, std::complex<T>(0, 0));
            for (std::size_t i = 0; i < nx; ++i)
                X[i] = std::complex<T>(x[i], T(0));
            xt::fft::detail::fft_radix2(X.data(), n_fft, false);

            // New spectrum size
            std::size_t n_fft_out = 1;
            while (n_fft_out < N) n_fft_out <<= 1;
            std::vector<std::complex<T>> Y(n_fft_out, std::complex<T>(0, 0));

            // Copy / interpolate frequency bins
            std::size_t min_bins = std::min(n_fft / 2 + 1, n_fft_out / 2 + 1);
            for (std::size_t k = 0; k < min_bins; ++k) {
                Y[k] = X[k];
                if (k > 0 && n_fft_out - k < n_fft_out)
                    Y[n_fft_out - k] = std::conj(Y[k]);
            }

            // IFFT
            xt::fft::detail::fft_radix2(Y.data(), n_fft_out, true);

            // Crop to desired length
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({N});
            T factor = static_cast<T>(L) / static_cast<T>(M);
            for (std::size_t i = 0; i < N; ++i)
                result[i] = std::real(Y[i]) * factor;
            return result;
        }

        /**
         * Nearest‑neighbour resampling.
         */
        template <class T>
        inline auto resample_nearest(const T* x, std::size_t nx, std::size_t ny) {
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> y({ny});
            for (std::size_t i = 0; i < ny; ++i) {
                double src = (static_cast<double>(i) + 0.5) * nx / ny;
                std::size_t idx = static_cast<std::size_t>(std::floor(src));
                if (idx >= nx) idx = nx - 1;
                y[i] = x[idx];
            }
            return y;
        }

        /**
         * Linear interpolation resampling.
         */
        template <class T>
        inline auto resample_linear(const T* x, std::size_t nx, std::size_t ny) {
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> y({ny});
            for (std::size_t i = 0; i < ny; ++i) {
                double src = (static_cast<double>(i) + 0.5) * nx / ny - 0.5;
                if (src < 0) src = 0;
                std::size_t lo = static_cast<std::size_t>(std::floor(src));
                std::size_t hi = std::min(lo + 1, nx - 1);
                if (lo >= nx) lo = nx - 1;
                double frac = src - static_cast<double>(lo);
                y[i] = static_cast<T>((T(1) - frac) * x[lo] + frac * x[hi]);
            }
            return y;
        }

        /**
         * Cubic (Catmull‑Rom) resampling.
         */
        template <class T>
        inline T cubic_kernel(T s) {
            s = std::abs(s);
            if (s <= T(1))
                return (T(1.5) * s - T(2.5)) * s * s + T(1);
            else if (s <= T(2))
                return ((-T(0.5) * s + T(2.5)) * s - T(4.0)) * s + T(2.0);
            else
                return T(0);
        }

        template <class T>
        inline auto resample_cubic(const T* x, std::size_t nx, std::size_t ny) {
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> y({ny});
            for (std::size_t i = 0; i < ny; ++i) {
                double src = (static_cast<double>(i) + 0.5) * nx / ny - 0.5;
                if (src < 0) src = 0;
                std::size_t center = static_cast<std::size_t>(std::floor(src));
                if (center >= nx) center = nx - 1;
                T sum = T(0);
                for (std::ptrdiff_t k = -1; k <= 2; ++k) {
                    std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(center) + k;
                    if (idx < 0) idx = 0;
                    if (static_cast<std::size_t>(idx) >= nx) idx = static_cast<std::ptrdiff_t>(nx) - 1;
                    T s = static_cast<T>(src - static_cast<double>(idx));
                    sum += x[idx] * cubic_kernel(s);
                }
                y[i] = sum;
            }
            return y;
        }
    }

    /**
     * Resample a 1D signal to a new length.
     * @param x Input signal (1D).
     * @param new_length Desired output length.
     * @param mode Resampling method.
     * @return Resampled signal (1D).
     */
    template <class E>
    inline auto resample(const xexpression<E>& x,
                         std::size_t new_length,
                         resample_mode mode = resample_mode::polyphase) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        if (arr.dimension() != 1)
            throw std::runtime_error("resample: input must be 1D.");
        std::size_t nx = arr.size();
        if (nx == 0 || new_length == 0)
            throw std::runtime_error("resample: zero length.");

        if (mode == resample_mode::nearest) {
            return detail::resample_nearest(arr.data(), nx, new_length);
        } else if (mode == resample_mode::linear) {
            return detail::resample_linear(arr.data(), nx, new_length);
        } else if (mode == resample_mode::cubic) {
            return detail::resample_cubic(arr.data(), nx, new_length);
        } else if (mode == resample_mode::polyphase) {
            // Determine up/down sampling factors
            std::size_t gcd = std::gcd(nx, new_length);
            std::size_t L = new_length / gcd;
            std::size_t M = nx / gcd;
            T cutoff = std::min(T(0.5) / static_cast<T>(L),
                                T(0.5) / static_cast<T>(M));
            T trans_width = cutoff * T(0.1);
            auto kernel = detail::design_anti_alias_filter(cutoff, trans_width);
            // Upsample then decimate or decimate then upsample
            if (L > 1 && M == 1) {
                return detail::polyphase_upsample(arr.data(), nx, L,
                                                   kernel.data(), kernel.size());
            } else if (M > 1 && L == 1) {
                return detail::decimate_filtered(arr.data(), nx, M,
                                                  kernel.data(), kernel.size());
            } else {
                // Rational rate change: upsample by L, then decimate by M
                auto upsampled = detail::polyphase_upsample(arr.data(), nx, L,
                                                             kernel.data(), kernel.size());
                return detail::decimate_filtered(upsampled.data(), upsampled.size(), M,
                                                  kernel.data(), kernel.size());
            }
        } else if (mode == resample_mode::fft) {
            std::size_t gcd = std::gcd(nx, new_length);
            std::size_t L = new_length / gcd;
            std::size_t M = nx / gcd;
            return detail::fft_resample(arr.data(), nx, L, M);
        }
        throw std::runtime_error("Unknown resample mode.");
    }

    /**
     * Resample using a fixed rational ratio L/M.
     * @param x Input signal.
     * @param L Upsampling factor.
     * @param M Downsampling factor.
     * @param mode Resampling method.
     * @return Resampled signal.
     */
    template <class E>
    inline auto resample_rational(const xexpression<E>& x,
                                   std::size_t L, std::size_t M,
                                   resample_mode mode = resample_mode::polyphase) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        std::size_t nx = arr.size();
        std::size_t new_length = nx * L / M;
        if (new_length == 0) new_length = 1;
        return resample(x, new_length, mode);
    }

    /**
     * Decimate (downsample) a signal by an integer factor.
     * @param x Input signal.
     * @param factor Downsampling factor.
     * @param apply_filter If true, apply anti‑alias filter before decimation.
     * @return Decimated signal.
     */
    template <class E>
    inline auto decimate(const xexpression<E>& x,
                         std::size_t factor,
                         bool apply_filter = true) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        if (arr.dimension() != 1)
            throw std::runtime_error("decimate: input must be 1D.");
        std::size_t nx = arr.size();
        std::size_t ny = (nx + factor - 1) / factor;

        if (apply_filter) {
            T cutoff = T(0.45) / static_cast<T>(factor);
            T trans = cutoff * T(0.1);
            auto kernel = detail::design_anti_alias_filter(cutoff, trans);
            return detail::decimate_filtered(arr.data(), nx, factor,
                                              kernel.data(), kernel.size());
        } else {
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> y({ny});
            for (std::size_t i = 0; i < ny; ++i)
                y[i] = arr[std::min(i * factor, nx - 1)];
            return y;
        }
    }

    /**
     * Upsample (interpolate) a signal by an integer factor.
     * @param x Input signal.
     * @param factor Upsampling factor.
     * @param apply_filter If true, apply interpolation filter.
     * @return Upsampled signal.
     */
    template <class E>
    inline auto upsample(const xexpression<E>& x,
                         std::size_t factor,
                         bool apply_filter = true) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        if (arr.dimension() != 1)
            throw std::runtime_error("upsample: input must be 1D.");
        std::size_t nx = arr.size();
        std::size_t ny = nx * factor;

        if (apply_filter) {
            T cutoff = T(0.45) / static_cast<T>(factor);
            T trans = cutoff * T(0.1);
            auto kernel = detail::design_anti_alias_filter(cutoff, trans);
            return detail::polyphase_upsample(arr.data(), nx, factor,
                                               kernel.data(), kernel.size());
        } else {
            // Zero‑stuffing
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> y({ny}, T(0));
            for (std::size_t i = 0; i < nx; ++i)
                y[i * factor] = arr[i];
            return y;
        }
    }

} // namespace signal
} // namespace xt

#endif // XTENSOR_SIGNAL_XRESAMPLE_HPP