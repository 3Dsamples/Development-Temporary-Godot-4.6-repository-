//File 0608 : xtensor-signal/xsignal_common.hpp
//Common utilities for xtensor‑signal: type traits, frequency axis generation, decibel conversion, and helper functions shared across signal processing modules.
#ifndef XTENSOR_SIGNAL_COMMON_HPP
#define XTENSOR_SIGNAL_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_signal_config.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xtensor_simd.hpp"

namespace xt {
namespace signal {

    /**
     * @struct signal_traits
     * @brief Helper to extract value type from an expression.
     */
    template <class T, class = void>
    struct signal_traits { using value_type = T; };
    template <class T>
    struct signal_traits<T, std::void_t<typename T::value_type>> {
        using value_type = typename T::value_type;
    };
    template <class T>
    using signal_value_type_t = typename signal_traits<T>::value_type;

    /**
     * Generate linearly spaced frequency axis for a given FFT size and sample rate.
     * @param n Number of points (e.g., n_fft).
     * @param fs Sample rate (default 1.0).
     * @param half_spectrum If true, return only positive frequencies (n/2+1).
     * @return 1D array of frequencies.
     */
    template <class T = double>
    inline auto frequency_axis(std::size_t n, T fs = T(1), bool half_spectrum = true) {
        std::size_t n_out = half_spectrum ? (n / 2 + 1) : n;
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> f({n_out});
        T factor = fs / static_cast<T>(n);
        for (std::size_t i = 0; i < n_out; ++i)
            f[i] = static_cast<T>(i) * factor;
        return f;
    }

    /**
     * Generate time axis for a signal.
     * @param n Number of samples.
     * @param dt Sample spacing (default 1.0).
     * @param t0 Start time (default 0.0).
     * @return 1D array of time values.
     */
    template <class T = double>
    inline auto time_axis(std::size_t n, T dt = T(1), T t0 = T(0)) {
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> t({n});
        for (std::size_t i = 0; i < n; ++i)
            t[i] = t0 + static_cast<T>(i) * dt;
        return t;
    }

    /**
     * Convert magnitude to decibels.
     * @param mag Magnitude (positive).
     * @param floor Minimum value to avoid log(0).
     * @return Value in dB (10*log10(mag)).
     */
    template <class T>
    inline T magnitude_to_db(T mag, T floor = T(1e-15)) {
        return T(10) * std::log10(std::max(mag, floor));
    }

    /**
     * Convert decibels back to magnitude.
     * @param db Value in dB.
     * @return Linear magnitude.
     */
    template <class T>
    inline T db_to_magnitude(T db) {
        return std::pow(T(10), db / T(10));
    }

    /**
     * Apply decibel conversion element‑wise to an array.
     * @param x Input array (positive magnitudes).
     * @param floor Minimum value for log.
     * @return Array in dB.
     */
    template <class E>
    inline auto to_db(const xexpression<E>& x, double floor = 1e-15) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        auto result = xt::eval(arr);
        T* d = result.data();
        std::size_t n = result.size();
        T fl = static_cast<T>(floor);
        for (std::size_t i = 0; i < n; ++i)
            d[i] = magnitude_to_db(arr[i], fl);
        return result;
    }

    /**
     * Convert an array from decibels to linear magnitude.
     * @param x Array in dB.
     * @return Linear magnitude array.
     */
    template <class E>
    inline auto from_db(const xexpression<E>& x) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        auto result = xt::eval(arr);
        T* d = result.data();
        std::size_t n = result.size();
        for (std::size_t i = 0; i < n; ++i)
            d[i] = db_to_magnitude(arr[i]);
        return result;
    }

    /**
     * Compute the RMS (root‑mean‑square) of a signal.
     * @param x Input signal.
     * @return RMS value.
     */
    template <class E>
    inline auto rms(const xexpression<E>& x) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        T sq_sum = xt::sum(arr * arr)();
        return std::sqrt(sq_sum / static_cast<T>(arr.size()));
    }

    /**
     * Normalise a signal by its peak amplitude.
     * @param x Input signal.
     * @return Normalised signal.
     */
    template <class E>
    inline auto normalise_peak(const xexpression<E>& x) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        T peak = T(0);
        for (std::size_t i = 0; i < arr.size(); ++i) {
            T val = std::abs(arr[i]);
            if (val > peak) peak = val;
        }
        if (peak > T(0)) return arr / peak;
        return arr;
    }

    /**
     * Compute the envelope of a signal using Hilbert transform (via FFT).
     * @param x Input signal.
     * @return Envelope (same length).
     */
    template <class E>
    inline auto envelope(const xexpression<E>& x) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        if (arr.dimension() != 1) throw std::runtime_error("envelope: input must be 1D.");
        std::size_t n = arr.size();
        // Compute analytic signal via FFT
        std::vector<std::complex<T>> spec(n);
        for (std::size_t i = 0; i < n; ++i)
            spec[i] = std::complex<T>(arr[i], T(0));
        xt::fft::detail::fft_radix2(spec.data(), n, false);
        // Zero negative frequencies and double positive
        for (std::size_t i = 1; i < (n + 1) / 2; ++i)
            spec[i] *= T(2);
        for (std::size_t i = (n + 1) / 2 + 1; i < n; ++i)
            spec[i] = std::complex<T>(0, 0);
        xt::fft::detail::fft_radix2(spec.data(), n, true);
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n});
        for (std::size_t i = 0; i < n; ++i)
            result[i] = std::abs(spec[i]);
        return result;
    }

    /**
     * Compute the autocorrelation of a signal via FFT.
     * @param x Input signal.
     * @return Autocorrelation (same length).
     */
    template <class E>
    inline auto autocorrelate_fft(const xexpression<E>& x) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(x.derived_cast());
        if (arr.dimension() != 1) throw std::runtime_error("autocorrelate_fft: input must be 1D.");
        std::size_t n = arr.size();
        std::size_t n_fft = 1;
        while (n_fft < 2 * n) n_fft <<= 1;
        std::vector<std::complex<T>> X(n_fft, std::complex<T>(0, 0));
        for (std::size_t i = 0; i < n; ++i)
            X[i] = std::complex<T>(arr[i], T(0));
        xt::fft::detail::fft_radix2(X.data(), n_fft, false);
        for (std::size_t i = 0; i < n_fft; ++i)
            X[i] = X[i] * std::conj(X[i]);
        xt::fft::detail::fft_radix2(X.data(), n_fft, true);
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({n});
        for (std::size_t i = 0; i < n; ++i)
            result[i] = std::real(X[i]) / static_cast<T>(n);
        return result;
    }

} // namespace signal
} // namespace xt

#endif // XTENSOR_SIGNAL_COMMON_HPP