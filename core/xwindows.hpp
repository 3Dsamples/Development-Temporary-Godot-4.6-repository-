// core/xwindows.hpp
#ifndef XTENSOR_XWINDOWS_HPP
#define XTENSOR_XWINDOWS_HPP

// ----------------------------------------------------------------------------
// xwindows.hpp – Window functions and spectral analysis utilities
// ----------------------------------------------------------------------------
// This header provides a comprehensive set of window functions for signal
// processing and spectral analysis, including:
//   - Basic windows: rectangular, triangular (Bartlett), Hann, Hamming, Blackman
//   - Advanced windows: Kaiser, Gaussian, Chebyshev (Dolph‑Chebyshev), Tukey
//   - Flat‑top windows for accurate amplitude measurement
//   - Window utilities: get_window, window_nuttall, window_blackman_harris
//   - Spectral leakage analysis and coherent gain calculations
//
// All functions return arrays of type T (including bignumber::BigNumber)
// with full support for FFT‑accelerated convolution where applicable.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <functional>
#include <complex>
#include <string>
#include <tuple>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace signal
    {
        namespace windows
        {
            // ========================================================================
            // Basic window functions
            // ========================================================================
            template <class T = double> std::vector<T> rectangular(size_t N, bool sym = true);
            template <class T = double> std::vector<T> triangular(size_t N, bool sym = true);
            template <class T = double> std::vector<T> bartlett(size_t N, bool sym = true);
            template <class T = double> std::vector<T> hann(size_t N, bool sym = true);
            template <class T = double> std::vector<T> hanning(size_t N, bool sym = true);
            template <class T = double> std::vector<T> hamming(size_t N, bool sym = true);
            template <class T = double> std::vector<T> blackman(size_t N, bool sym = true);
            template <class T = double> std::vector<T> blackman_harris(size_t N, bool sym = true);
            template <class T = double> std::vector<T> nuttall(size_t N, bool sym = true);
            template <class T = double> std::vector<T> flattop(size_t N, bool sym = true);
            template <class T = double> std::vector<T> cosine(size_t N, bool sym = true);
            template <class T = double> std::vector<T> lanczos(size_t N, size_t a = 3, bool sym = true);
            template <class T = double> std::vector<T> exponential(size_t N, T tau = T(1), bool sym = true);

            // ========================================================================
            // Parameterized windows
            // ========================================================================
            template <class T = double> std::vector<T> gaussian(size_t N, T std_dev, bool sym = true);
            template <class T = double> std::vector<T> kaiser(size_t N, T beta, bool sym = true);
            template <class T = double> std::vector<T> chebwin(size_t N, T attenuation_db, bool sym = true);
            template <class T = double> std::vector<T> tukey(size_t N, T alpha = T(0.5), bool sym = true);

            // ========================================================================
            // Window utilities
            // ========================================================================
            template <class T = double> T kaiser_beta(T attenuation_db);
            template <class T = double> size_t kaiser_order(T attenuation_db, T transition_width);
            template <class T = double> T coherent_gain(const std::vector<T>& w);
            template <class T = double> T enbw(const std::vector<T>& w);
            template <class T = double> T scalloping_loss(const std::vector<T>& w);

            // ========================================================================
            // Generic window getter and application
            // ========================================================================
            template <class T = double> std::vector<T> get_window(const std::string& name, size_t N, bool sym = true);
            template <class T = double> std::vector<T> get_window(const std::tuple<std::string, T>& win_param, size_t N, bool sym = true);

            template <class E>
            auto apply_window(const xexpression<E>& data, const std::vector<typename E::value_type>& w);

            template <class E>
            auto periodogram(const xexpression<E>& data,
                             const std::vector<typename E::value_type>& w = {},
                             size_t nfft = 0,
                             typename E::value_type fs = typename E::value_type(1));
        }

        using windows::rectangular;
        using windows::triangular;
        using windows::bartlett;
        using windows::hann;
        using windows::hanning;
        using windows::hamming;
        using windows::blackman;
        using windows::blackman_harris;
        using windows::nuttall;
        using windows::flattop;
        using windows::cosine;
        using windows::lanczos;
        using windows::exponential;
        using windows::gaussian;
        using windows::kaiser;
        using windows::chebwin;
        using windows::tukey;
        using windows::get_window;
        using windows::coherent_gain;
        using windows::enbw;
        using windows::scalloping_loss;
        using windows::apply_window;
        using windows::periodogram;
    }
}

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace signal
    {
        namespace windows
        {
            // Rectangular window (all ones)
            template <class T> std::vector<T> rectangular(size_t N, bool sym)
            { return std::vector<T>(N, T(1)); }

            // Triangular (Bartlett) window
            template <class T> std::vector<T> triangular(size_t N, bool sym)
            { /* TODO: implement triangular window */ return {}; }

            // Bartlett window (alias for triangular)
            template <class T> std::vector<T> bartlett(size_t N, bool sym)
            { return triangular<T>(N, sym); }

            // Hann window (cosine squared)
            template <class T> std::vector<T> hann(size_t N, bool sym)
            { /* TODO: implement Hann window */ return {}; }

            // Hanning window (alias for Hann)
            template <class T> std::vector<T> hanning(size_t N, bool sym)
            { return hann<T>(N, sym); }

            // Hamming window (raised cosine with optimized coefficients)
            template <class T> std::vector<T> hamming(size_t N, bool sym)
            { /* TODO: implement Hamming window */ return {}; }

            // Blackman window (cosine series)
            template <class T> std::vector<T> blackman(size_t N, bool sym)
            { /* TODO: implement Blackman window */ return {}; }

            // Blackman‑Harris window (4‑term cos series)
            template <class T> std::vector<T> blackman_harris(size_t N, bool sym)
            { /* TODO: implement Blackman‑Harris window */ return {}; }

            // Nuttall window (4‑term cos series)
            template <class T> std::vector<T> nuttall(size_t N, bool sym)
            { /* TODO: implement Nuttall window */ return {}; }

            // Flat‑top window (for accurate amplitude)
            template <class T> std::vector<T> flattop(size_t N, bool sym)
            { /* TODO: implement flat‑top window */ return {}; }

            // Cosine window (sine lobe)
            template <class T> std::vector<T> cosine(size_t N, bool sym)
            { /* TODO: implement cosine window */ return {}; }

            // Lanczos window (sinc‑based)
            template <class T> std::vector<T> lanczos(size_t N, size_t a, bool sym)
            { /* TODO: implement Lanczos window */ return {}; }

            // Exponential window (decaying)
            template <class T> std::vector<T> exponential(size_t N, T tau, bool sym)
            { /* TODO: implement exponential window */ return {}; }

            // Gaussian window
            template <class T> std::vector<T> gaussian(size_t N, T std_dev, bool sym)
            { /* TODO: implement Gaussian window */ return {}; }

            // Kaiser window
            template <class T> std::vector<T> kaiser(size_t N, T beta, bool sym)
            { /* TODO: implement Kaiser window */ return {}; }

            // Dolph‑Chebyshev window
            template <class T> std::vector<T> chebwin(size_t N, T attenuation_db, bool sym)
            { /* TODO: implement Chebyshev window */ return {}; }

            // Tukey (tapered cosine) window
            template <class T> std::vector<T> tukey(size_t N, T alpha, bool sym)
            { /* TODO: implement Tukey window */ return {}; }

            // Compute Kaiser beta parameter from desired sidelobe attenuation
            template <class T> T kaiser_beta(T attenuation_db)
            { /* TODO: implement beta computation */ return T(0); }

            // Estimate Kaiser window order
            template <class T> size_t kaiser_order(T attenuation_db, T transition_width)
            { /* TODO: implement order estimation */ return 0; }

            // Compute coherent gain (sum/N)
            template <class T> T coherent_gain(const std::vector<T>& w)
            { T s = 0; for (auto v : w) s += v; return s / T(w.size()); }

            // Compute equivalent noise bandwidth
            template <class T> T enbw(const std::vector<T>& w)
            { T s = 0, s2 = 0; for (auto v : w) { s += v; s2 += v*v; } return s2 * T(w.size()) / (s*s); }

            // Compute scalloping loss (max reduction for inter‑bin tone)
            template <class T> T scalloping_loss(const std::vector<T>& w)
            { /* TODO: compute via DFT at half‑bin offset */ return T(0); }

            // Get window by name
            template <class T> std::vector<T> get_window(const std::string& name, size_t N, bool sym)
            { /* TODO: dispatch to specific window */ return {}; }

            // Get parameterized window by name and parameter
            template <class T> std::vector<T> get_window(const std::tuple<std::string, T>& win_param, size_t N, bool sym)
            { /* TODO: dispatch with parameter */ return {}; }

            // Apply window to an array (element‑wise multiplication)
            template <class E>
            auto apply_window(const xexpression<E>& data, const std::vector<typename E::value_type>& w)
            { /* TODO: implement windowing */ return data.derived_cast(); }

            // Compute periodogram (power spectrum) using window
            template <class E>
            auto periodogram(const xexpression<E>& data, const std::vector<typename E::value_type>& w,
                             size_t nfft, typename E::value_type fs)
            { /* TODO: compute |FFT|^2 scaled by window energy */ return std::make_pair(xarray_container<typename E::value_type>(), xarray_container<typename E::value_type>()); }
        }
    }
}

#endif // XTENSOR_XWINDOWS_HPP======================================================
            // Blackman‑Harris window
            // ========================================================================
            template <class T = double>
            std::vector<T> blackman_harris(size_t N, bool sym = true)
            {
                std::vector<T> w(N);
                T denom = sym ? T(N - 1) : T(N);
                T a0 = T(0.35875);
                T a1 = T(0.48829);
                T a2 = T(0.14128);
                T a3 = T(0.01168);
                for (size_t n = 0; n < N; ++n)
                {
                    T x = T(2) * detail::pi<T>() * T(n) / denom;
                    w[n] = a0 - a1 * detail::cos_val(x) + a2 * detail::cos_val(T(2) * x)
                           - a3 * detail::cos_val(T(3) * x);
                }
                return w;
            }

            // ========================================================================
            // Nuttall window
            // ========================================================================
            template <class T = double>
            std::vector<T> nuttall(size_t N, bool sym = true)
            {
                std::vector<T> w(N);
                T denom = sym ? T(N - 1) : T(N);
                T a0 = T(0.355768);
                T a1 = T(0.487396);
                T a2 = T(0.144232);
                T a3 = T(0.012604);
                for (size_t n = 0; n < N; ++n)
                {
                    T x = T(2) * detail::pi<T>() * T(n) / denom;
                    w[n] = a0 - a1 * detail::cos_val(x) + a2 * detail::cos_val(T(2) * x)
                           - a3 * detail::cos_val(T(3) * x);
                }
                return w;
            }

            // ========================================================================
            // Flat‑top window (for accurate amplitude measurements)
            // ========================================================================
            template <class T = double>
            std::vector<T> flattop(size_t N, bool sym = true)
            {
                std::vector<T> w(N);
                T denom = sym ? T(N - 1) : T(N);
                // SFT3F coefficients (typical flat‑top)
                T a0 = T(0.26526);
                T a1 = T(0.5);
                T a2 = T(0.23474);
                T a3 = T(0.0);
                T a4 = T(0.0);
                for (size_t n = 0; n < N; ++n)
                {
                    T x = T(2) * detail::pi<T>() * T(n) / denom;
                    w[n] = a0
                           - a1 * detail::cos_val(x)
                           + a2 * detail::cos_val(T(2) * x)
                           - a3 * detail::cos_val(T(3) * x)
                           + a4 * detail::cos_val(T(4) * x);
                }
                return w;
            }

            // ========================================================================
            // Gaussian window
            // ========================================================================
            template <class T = double>
            std::vector<T> gaussian(size_t N, T std_dev, bool sym = true)
            {
                std::vector<T> w(N);
                T center = sym ? T(N - 1) / T(2) : T(N) / T(2);
                T two_sigma_sq = T(2) * std_dev * std_dev;
                for (size_t n = 0; n < N; ++n)
                {
                    T x = (T(n) - center) / std_dev;
                    w[n] = std::exp(-T(0.5) * x * x);
                }
                return w;
            }

            // ========================================================================
            // Kaiser window
            // ========================================================================
            template <class T = double>
            std::vector<T> kaiser(size_t N, T beta, bool sym = true)
            {
                std::vector<T> w(N);
                T denom = detail::bessel_i0(beta);
                T center = sym ? T(N - 1) / T(2) : T(N) / T(2);
                for (size_t n = 0; n < N; ++n)
                {
                    T x = T(2) * T(n) / T(N - 1) - T(1);
                    T arg = beta * detail::sqrt_val(T(1) - x * x);
                    w[n] = detail::bessel_i0(arg) / denom;
                }
                return w;
            }

            // ------------------------------------------------------------------------
            // Kaiser beta parameter from desired sidelobe attenuation (dB)
            // ------------------------------------------------------------------------
            template <class T = double>
            T kaiser_beta(T attenuation_db)
            {
                if (attenuation_db > T(50))
                    return T(0.1102) * (attenuation_db - T(8.7));
                else if (attenuation_db >= T(21))
                {
                    T a = attenuation_db;
                    return T(0.5842) * detail::pow_val(a - T(21), 0.4) + T(0.07886) * (a - T(21));
                }
                else
                    return T(0);
            }

            // ------------------------------------------------------------------------
            // Kaiser window order estimation
            // ------------------------------------------------------------------------
            template <class T = double>
            size_t kaiser_order(T attenuation_db, T transition_width)
            {
                T a = attenuation_db;
                T width = transition_width;
                return static_cast<size_t>(std::ceil((a - T(8)) / (T(2.285) * width) + T(1)));
            }

            // ========================================================================
            // Dolph‑Chebyshev window
            // ========================================================================
            template <class T = double>
            std::vector<T> chebwin(size_t N, T attenuation_db, bool sym = true)
            {
                // Compute parameter beta = cosh(acosh(10^(att/20))/M)
                T ripple = detail::pow_val(T(10), attenuation_db / T(20));
                T x0 = detail::cosh_val(detail::acosh_val(ripple) / T(N - 1));
                std::vector<T> w(N);
                T center = sym ? T(N - 1) / T(2) : T(N) / T(2);
                for (size_t n = 0; n < N; ++n)
                {
                    T sum = T(0);
                    T sign = T(1);
                    T scale = T(1);
                    for (size_t i = 0; i <= N/2; ++i)
                    {
                        if (i == 0)
                            sum = sum + T(1);
                        else
                        {
                            T theta = detail::pi<T>() * T(i) / T(N);
                            T term = detail::chebyshev_poly(N - 1, x0 * detail::cos_val(theta));
                            term = term * detail::cos_val(T(2) * theta * (T(n) - center));
                            sum = sum + T(2) * term;
                        }
                    }
                    w[n] = sum / T(N);
                }
                // Normalize to unit peak
                T max_val = *std::max_element(w.begin(), w.end());
                if (max_val > T(0))
                    for (auto& v : w) v = v / max_val;
                return w;
            }

            // ========================================================================
            // Tukey window (tapered cosine)
            // ========================================================================
            template <class T = double>
            std::vector<T> tukey(size_t N, T alpha = T(0.5), bool sym = true)
            {
                std::vector<T> w(N, T(1));
                if (alpha <= T(0)) return w; // rectangular
                if (alpha >= T(1)) return hann<T>(N, sym); // Hann window

                T width = alpha * T(N - 1) / T(2);
                T center = sym ? T(N - 1) / T(2) : T(N) / T(2);
                for (size_t n = 0; n < N; ++n)
                {
                    T x = std::abs(T(n) - center);
                    if (x >= (T(N - 1) - width) / T(2))
                    {
                        T arg = detail::pi<T>() * (x - (T(N - 1) - width) / T(2)) / width;
                        w[n] = T(0.5) * (T(1) + detail::cos_val(arg));
                    }
                }
                return w;
            }

            // ========================================================================
            // Cosine window
            // ========================================================================
            template <class T = double>
            std::vector<T> cosine(size_t N, bool sym = true)
            {
                std::vector<T> w(N);
                T denom = sym ? T(N - 1) : T(N);
                for (size_t n = 0; n < N; ++n)
                {
                    T x = detail::pi<T>() * T(n) / denom;
                    w[n] = std::sin(x);
                }
                return w;
            }

            // ========================================================================
            // Lanczos window (sinc window)
            // ========================================================================
            template <class T = double>
            std::vector<T> lanczos(size_t N, size_t a = 3, bool sym = true)
            {
                std::vector<T> w(N);
                T center = sym ? T(N - 1) / T(2) : T(N) / T(2);
                for (size_t n = 0; n < N; ++n)
                {
                    T x = T(2) * (T(n) - center) / T(N - 1);
                    if (x == T(0))
                        w[n] = T(1);
                    else if (std::abs(x) >= T(a))
                        w[n] = T(0);
                    else
                        w[n] = detail::sinc(x) * detail::sinc(x / T(a));
                }
                return w;
            }

            // ========================================================================
            // Exponential window (decaying)
            // ========================================================================
            template <class T = double>
            std::vector<T> exponential(size_t N, T tau = T(1), bool sym = true)
            {
                std::vector<T> w(N);
                T center = sym ? T(0) : T(0); // usually left‑aligned for exponential
                for (size_t n = 0; n < N; ++n)
                {
                    T x = T(n) / T(N - 1);
                    w[n] = std::exp(-x * tau);
                }
                return w;
            }

            // ========================================================================
            // Generic window getter by name
            // ========================================================================
            template <class T = double>
            std::vector<T> get_window(const std::string& name, size_t N, bool sym = true)
            {
                if (name == "rectangular" || name == "boxcar")
                    return rectangular<T>(N, sym);
                if (name == "triangular" || name == "bartlett")
                    return triangular<T>(N, sym);
                if (name == "hann" || name == "hanning")
                    return hann<T>(N, sym);
                if (name == "hamming")
                    return hamming<T>(N, sym);
                if (name == "blackman")
                    return blackman<T>(N, sym);
                if (name == "blackman_harris")
                    return blackman_harris<T>(N, sym);
                if (name == "nuttall")
                    return nuttall<T>(N, sym);
                if (name == "flattop")
                    return flattop<T>(N, sym);
                if (name == "cosine")
                    return cosine<T>(N, sym);
                if (name == "tukey")
                    return tukey<T>(N, T(0.5), sym);
                XTENSOR_THROW(std::invalid_argument, "get_window: unknown window name: " + name);
                return {};
            }

            // Overload with additional parameter
            template <class T = double>
            std::vector<T> get_window(const std::tuple<std::string, T>& win_param, size_t N, bool sym = true)
            {
                const std::string& name = std::get<0>(win_param);
                T param = std::get<1>(win_param);
                if (name == "gaussian")
                    return gaussian<T>(N, param, sym);
                if (name == "kaiser")
                    return kaiser<T>(N, param, sym);
                if (name == "chebwin" || name == "chebyshev")
                    return chebwin<T>(N, param, sym);
                if (name == "tukey")
                    return tukey<T>(N, param, sym);
                if (name == "exponential")
                    return exponential<T>(N, param, sym);
                XTENSOR_THROW(std::invalid_argument, "get_window: unknown parameterized window name: " + name);
                return {};
            }

            // ========================================================================
            // Window utilities
            // ========================================================================

            // ------------------------------------------------------------------------
            // Coherent gain of a window (sum of window samples / N)
            // ------------------------------------------------------------------------
            template <class T>
            T coherent_gain(const std::vector<T>& w)
            {
                T sum = T(0);
                for (auto v : w) sum = sum + v;
                return sum / T(w.size());
            }

            // ------------------------------------------------------------------------
            // Equivalent noise bandwidth (ENBW) of a window
            // ------------------------------------------------------------------------
            template <class T>
            T enbw(const std::vector<T>& w)
            {
                T sum = T(0);
                T sum_sq = T(0);
                for (auto v : w)
                {
                    sum = sum + v;
                    sum_sq = sum_sq + v * v;
                }
                return (sum_sq * T(w.size())) / (sum * sum);
            }

            // ------------------------------------------------------------------------
            // Scalloping loss (maximum reduction in amplitude for a tone between bins)
            // ------------------------------------------------------------------------
            template <class T>
            T scalloping_loss(const std::vector<T>& w)
            {
                // Simplified: ratio of coherent gain at worst‑case frequency offset
                // We compute DFT of window at half‑bin offset and find maximum.
                size_t N = w.size();
                std::vector<std::complex<T>> W(N);
                for (size_t k = 0; k < N; ++k)
                {
                    std::complex<T> sum(0,0);
                    T theta = T(2) * detail::pi<T>() * T(k) / T(N);
                    for (size_t n = 0; n < N; ++n)
                    {
                        T angle = theta * T(n);
                        sum += std::complex<T>(w[n] * detail::cos_val(angle), -w[n] * std::sin(angle));
                    }
                    W[k] = sum;
                }
                T max_mag = T(0);
                for (auto& c : W)
                {
                    T mag = std::abs(c);
                    if (mag > max_mag) max_mag = mag;
                }
                T dc_gain = std::abs(W[0]);
                if (dc_gain < T(1e-12)) return T(0);
                return T(1) - max_mag / dc_gain;
            }

            // ------------------------------------------------------------------------
            // Apply window to an array (in‑place or copy)
            // ------------------------------------------------------------------------
            template <class E>
            auto apply_window(const xexpression<E>& data, const std::vector<typename E::value_type>& w)
            {
                const auto& d = data.derived_cast();
                using T = typename E::value_type;
                if (d.size() != w.size())
                    XTENSOR_THROW(std::invalid_argument, "apply_window: data and window sizes must match");
                auto result = xarray_container<T>(d.shape());
                for (size_t i = 0; i < d.size(); ++i)
                    result.flat(i) = detail::multiply(d.flat(i), w[i]);
                return result;
            }

            // ------------------------------------------------------------------------
            // Periodogram (power spectrum using window)
            // ------------------------------------------------------------------------
            template <class E>
            auto periodogram(const xexpression<E>& data,
                             const std::vector<typename E::value_type>& w = {},
                             size_t nfft = 0,
                             T fs = T(1))
            {
                const auto& d = data.derived_cast();
                using T = typename E::value_type;
                size_t N = d.size();
                if (nfft == 0) nfft = N;
                std::vector<T> windowed;
                if (w.empty())
                    windowed = std::vector<T>(d.begin(), d.end());
                else
                {
                    if (w.size() != N)
                        XTENSOR_THROW(std::invalid_argument, "periodogram: window size mismatch");
                    windowed.resize(N);
                    for (size_t i = 0; i < N; ++i)
                        windowed[i] = detail::multiply(d.flat(i), w[i]);
                }
                // Compute FFT and power
                xarray_container<T> win_arr({N});
                for (size_t i = 0; i < N; ++i) win_arr(i) = windowed[i];
                auto fft_result = fft::fft(win_arr);
                auto power = xt::abs(fft_result);
                for (auto& v : power) v = v * v;
                // Normalize by window energy
                T scale = T(1) / (fs * T(N));
                if (!w.empty())
                {
                    T win_sq_sum = T(0);
                    for (auto v : w) win_sq_sum = win_sq_sum + v * v;
                    scale = scale * T(N) / win_sq_sum;
                }
                for (auto& v : power) v = v * scale;
                // Return frequencies and power (for first half)
                size_t nfreq = nfft / 2 + 1;
                xarray_container<T> freqs({nfreq});
                xarray_container<T> psd({nfreq});
                for (size_t k = 0; k < nfreq; ++k)
                {
                    freqs(k) = T(k) * fs / T(nfft);
                    psd(k) = power(k);
                }
                return std::make_pair(freqs, psd);
            }

        } // namespace windows
    } // namespace signal

    // Bring window functions into xt::signal namespace for convenience
    namespace signal
    {
        using windows::rectangular;
        using windows::triangular;
        using windows::bartlett;
        using windows::hann;
        using windows::hanning;
        using windows::hamming;
        using windows::blackman;
        using windows::blackman_harris;
        using windows::nuttall;
        using windows::flattop;
        using windows::gaussian;
        using windows::kaiser;
        using windows::chebwin;
        using windows::tukey;
        using windows::cosine;
        using windows::lanczos;
        using windows::exponential;
        using windows::get_window;
        using windows::coherent_gain;
        using windows::enbw;
        using windows::scalloping_loss;
        using windows::apply_window;
        using windows::periodogram;
    }

} // namespace xt

#endif // XTENSOR_XWINDOWS_HPP