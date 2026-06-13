//File 0604 : xtensor-signal/xwindows.hpp
//Window functions (Hamming, Hanning, Blackman, Bartlett, Kaiser, Flattop, rectangular) with SIMD‑accelerated generation and spectral analysis integration.
#ifndef XTENSOR_SIGNAL_XWINDOWS_HPP
#define XTENSOR_SIGNAL_XWINDOWS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_signal_config.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor_simd.hpp"

namespace xt {
namespace signal {

    namespace detail {

        /**
         * Generate a Hamming window of length n.
         * w[i] = 0.54 - 0.46 * cos(2π i / (n-1))
         */
        template <class T>
        inline auto make_hamming(std::size_t n) {
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> w({n});
            T* d = w.data();
            if (n == 0) return w;
            if (n == 1) { d[0] = T(1); return w; }
            T denom = T(1) / static_cast<T>(n - 1);
            T two_pi = T(2) * xt::numeric_constants<T>::pi;
            if constexpr (is_simd_enabled_v<T>) {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                simd_type v0_54(0.54), v0_46(0.46), v2pi(two_pi), vdenom(denom);
                for (std::size_t i = 0; i < vec_count; ++i) {
                    alignas(64) std::array<T, simd_size> idx;
                    for (std::size_t k = 0; k < simd_size; ++k)
                        idx[k] = static_cast<T>(i * simd_size + k);
                    simd_type vi = simd_type::load_aligned(idx.data());
                    simd_type vcos = xsimd::cos(v2pi * vi * vdenom);
                    simd_type vw = v0_54 - v0_46 * vcos;
                    vw.store_unaligned(d + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    d[i] = T(0.54) - T(0.46) * std::cos(two_pi * static_cast<T>(i) * denom);
            } else {
                for (std::size_t i = 0; i < n; ++i)
                    d[i] = T(0.54) - T(0.46) * std::cos(two_pi * static_cast<T>(i) * denom);
            }
            return w;
        }

        /**
         * Generate a Hanning (Hann) window.
         * w[i] = 0.5 * (1 - cos(2π i / (n-1)))
         */
        template <class T>
        inline auto make_hanning(std::size_t n) {
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> w({n});
            T* d = w.data();
            if (n <= 1) { if (n==1) d[0] = T(1); return w; }
            T denom = T(1) / static_cast<T>(n - 1);
            T two_pi = T(2) * xt::numeric_constants<T>::pi;
            if constexpr (is_simd_enabled_v<T>) {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                simd_type v0_5(0.5), v2pi(two_pi), vdenom(denom);
                for (std::size_t i = 0; i < vec_count; ++i) {
                    alignas(64) std::array<T, simd_size> idx;
                    for (std::size_t k = 0; k < simd_size; ++k)
                        idx[k] = static_cast<T>(i * simd_size + k);
                    simd_type vi = simd_type::load_aligned(idx.data());
                    simd_type vcos = xsimd::cos(v2pi * vi * vdenom);
                    simd_type vw = v0_5 * (simd_type(1) - vcos);
                    vw.store_unaligned(d + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    d[i] = T(0.5) * (T(1) - std::cos(two_pi * static_cast<T>(i) * denom));
            } else {
                for (std::size_t i = 0; i < n; ++i)
                    d[i] = T(0.5) * (T(1) - std::cos(two_pi * static_cast<T>(i) * denom));
            }
            return w;
        }

        /**
         * Generate a Blackman window.
         * w[i] = 0.42 - 0.5 cos(2πi/(n-1)) + 0.08 cos(4πi/(n-1))
         */
        template <class T>
        inline auto make_blackman(std::size_t n) {
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> w({n});
            T* d = w.data();
            if (n <= 1) { if (n==1) d[0]=T(1); return w; }
            T denom = T(1) / static_cast<T>(n - 1);
            T two_pi = T(2) * xt::numeric_constants<T>::pi;
            if constexpr (is_simd_enabled_v<T>) {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                simd_type v0_42(0.42), v0_5(0.5), v0_08(0.08), v2pi(two_pi), vdenom(denom);
                for (std::size_t i = 0; i < vec_count; ++i) {
                    alignas(64) std::array<T, simd_size> idx;
                    for (std::size_t k = 0; k < simd_size; ++k)
                        idx[k] = static_cast<T>(i * simd_size + k);
                    simd_type vi = simd_type::load_aligned(idx.data());
                    simd_type cos1 = xsimd::cos(v2pi * vi * vdenom);
                    simd_type cos2 = xsimd::cos(v2pi * vi * vdenom * 2);
                    simd_type vw = v0_42 - v0_5 * cos1 + v0_08 * cos2;
                    vw.store_unaligned(d + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i) {
                    T c1 = std::cos(two_pi * i * denom);
                    T c2 = std::cos(2 * two_pi * i * denom);
                    d[i] = T(0.42) - T(0.5) * c1 + T(0.08) * c2;
                }
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    T c1 = std::cos(two_pi * i * denom);
                    T c2 = std::cos(2 * two_pi * i * denom);
                    d[i] = T(0.42) - T(0.5) * c1 + T(0.08) * c2;
                }
            }
            return w;
        }

        /**
         * Generate a Bartlett (triangular) window.
         * w[i] = 1 - |(i - (n-1)/2) / ((n-1)/2)|
         */
        template <class T>
        inline auto make_bartlett(std::size_t n) {
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> w({n});
            T* d = w.data();
            if (n == 0) return w;
            T mid = static_cast<T>(n - 1) / T(2);
            T scale = T(2) / static_cast<T>(n - 1);
            for (std::size_t i = 0; i < n; ++i) {
                T val = T(1) - std::abs(static_cast<T>(i) - mid) * scale;
                d[i] = std::max(T(0), val);
            }
            return w;
        }

        /**
         * Generate a Kaiser window with shape parameter beta.
         * w[i] = I0(beta * sqrt(1 - (2i/(n-1) - 1)^2)) / I0(beta)
         */
        template <class T>
        inline auto make_kaiser(std::size_t n, T beta) {
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> w({n});
            T* d = w.data();
            if (n <= 1) { if (n==1) d[0]=T(1); return w; }
            // Bessel I0 via series approximation
            auto bessel_i0 = [](T x) {
                T sum = T(1), term = T(1);
                for (int k = 1; k <= 20; ++k) {
                    term *= (x * x) / (T(4) * k * k);
                    sum += term;
                }
                return sum;
            };
            T denom = bessel_i0(beta);
            T alpha = (static_cast<T>(n) - T(1)) / T(2);
            for (std::size_t i = 0; i < n; ++i) {
                T val = (T(2) * i / (n - 1)) - T(1);
                T arg = beta * std::sqrt(T(1) - val * val);
                d[i] = bessel_i0(arg) / denom;
            }
            return w;
        }

        /**
         * Generate a flat‑top window.
         */
        template <class T>
        inline auto make_flattop(std::size_t n) {
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> w({n});
            T* d = w.data();
            if (n <= 1) { if (n==1) d[0]=T(1); return w; }
            // Coefficients for flattop window
            constexpr T a0 = T(0.21557895), a1 = T(0.41663158), a2 = T(0.277263158),
                        a3 = T(0.083578947), a4 = T(0.006947368);
            T two_pi = T(2) * xt::numeric_constants<T>::pi;
            T denom = T(1) / static_cast<T>(n - 1);
            for (std::size_t i = 0; i < n; ++i) {
                T x = two_pi * i * denom;
                d[i] = a0 - a1 * std::cos(x) + a2 * std::cos(T(2)*x)
                       - a3 * std::cos(T(3)*x) + a4 * std::cos(T(4)*x);
            }
            return w;
        }

        /**
         * Generate a rectangular window.
         */
        template <class T>
        inline auto make_rectangular(std::size_t n) {
            return xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>({n}, T(1));
        }
    }

    /**
     * Generate a window function of a given type and size.
     * @param type The window type.
     * @param n Window length.
     * @param param Additional parameter (used by Kaiser as beta).
     * @return 1D xarray containing the window.
     */
    template <class T = double>
    inline auto window(window_type type, std::size_t n, T param = T(3.0)) {
        switch (type) {
            case window_type::hamming:     return detail::make_hamming<T>(n);
            case window_type::hanning:     return detail::make_hanning<T>(n);
            case window_type::blackman:    return detail::make_blackman<T>(n);
            case window_type::bartlett:    return detail::make_bartlett<T>(n);
            case window_type::kaiser:      return detail::make_kaiser<T>(n, param);
            case window_type::flattop:     return detail::make_flattop<T>(n);
            case window_type::rectangular: return detail::make_rectangular<T>(n);
            default: throw std::runtime_error("Unknown window type.");
        }
    }

    // Convenience functions for common window types
    template <class T = double> inline auto hamming(std::size_t n)    { return detail::make_hamming<T>(n); }
    template <class T = double> inline auto hanning(std::size_t n)    { return detail::make_hanning<T>(n); }
    template <class T = double> inline auto blackman(std::size_t n)   { return detail::make_blackman<T>(n); }
    template <class T = double> inline auto bartlett(std::size_t n)   { return detail::make_bartlett<T>(n); }
    template <class T = double> inline auto kaiser(std::size_t n, T beta = T(3.0)) { return detail::make_kaiser<T>(n, beta); }
    template <class T = double> inline auto flattop(std::size_t n)    { return detail::make_flattop<T>(n); }
    template <class T = double> inline auto rectangular(std::size_t n){ return detail::make_rectangular<T>(n); }

} // namespace signal
} // namespace xt

#endif // XTENSOR_SIGNAL_XWINDOWS_HPP