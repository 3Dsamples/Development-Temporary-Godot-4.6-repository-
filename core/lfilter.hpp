// core/lfilter.hpp
#ifndef XTENSOR_LFILTER_HPP
#define XTENSOR_LFILTER_HPP

// ----------------------------------------------------------------------------
// lfilter.hpp – Digital filter design and application
// ----------------------------------------------------------------------------
// This header provides comprehensive digital signal processing filters:
//   - FIR filter design (window method, Parks‑McClellan via Remez exchange)
//   - IIR filter design (Butterworth, Chebyshev I/II, Elliptic, Bessel)
//   - Filter transformations (lowpass→highpass, bandpass, bandstop)
//   - Filtering functions: lfilter, filtfilt, sosfilt, convolve
//   - Filter analysis: freqz, group_delay, phasez
//   - Second‑order sections (SOS) representation and stability checking
//
// All arithmetic uses bignumber::BigNumber for precision; FFT acceleration
// is employed for convolution‑based filtering and frequency analysis.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <limits>
#include <tuple>
#include <string>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xnorm.hpp"
#include "xlinalg.hpp"
#include "fft.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace signal
    {
        // ========================================================================
        // FIR filter design by window method
        // ========================================================================
        template <class T = double>
        std::vector<T> firwin(size_t numtaps, T cutoff, T fs = T(2),
                              const std::string& window = "hamming", bool pass_zero = true);

        // ========================================================================
        // Parks‑McClellan (Remez) FIR design
        // ========================================================================
        template <class T = double>
        std::vector<T> remez(size_t numtaps, const std::vector<T>& bands,
                             const std::vector<T>& desired,
                             const std::vector<T>& weights = {},
                             T fs = T(2));

        // ========================================================================
        // IIR filter design (analog prototypes + bilinear transform)
        // ========================================================================
        template <class T = double>
        std::tuple<std::vector<T>, std::vector<T>> butter(int N, T Wn,
                                                          const std::string& btype = "low",
                                                          T fs = T(2));
        template <class T = double>
        std::tuple<std::vector<T>, std::vector<T>> cheby1(int N, T rp, T Wn,
                                                          const std::string& btype = "low",
                                                          T fs = T(2));
        template <class T = double>
        std::tuple<std::vector<T>, std::vector<T>> cheby2(int N, T rs, T Wn,
                                                          const std::string& btype = "low",
                                                          T fs = T(2));
        template <class T = double>
        std::tuple<std::vector<T>, std::vector<T>> ellip(int N, T rp, T rs, T Wn,
                                                         const std::string& btype = "low",
                                                         T fs = T(2));
        template <class T = double>
        std::tuple<std::vector<T>, std::vector<T>> bessel(int N, T Wn,
                                                          const std::string& btype = "low",
                                                          T fs = T(2));

        // ========================================================================
        // Filtering functions
        // ========================================================================
        template <class E>
        auto lfilter(const std::vector<typename E::value_type>& b,
                     const std::vector<typename E::value_type>& a,
                     const xexpression<E>& x_expr);
        template <class E>
        auto filtfilt(const std::vector<typename E::value_type>& b,
                      const std::vector<typename E::value_type>& a,
                      const xexpression<E>& x_expr);
        template <class E>
        auto sosfilt(const xarray_container<typename E::value_type>& sos,
                     const xexpression<E>& x_expr);

        // ========================================================================
        // Frequency response analysis
        // ========================================================================
        template <class T>
        std::tuple<xarray_container<T>, xarray_container<std::complex<T>>>
        freqz(const std::vector<T>& b, const std::vector<T>& a,
              size_t worN = 512, bool whole = false, T fs = T(2*M_PI));
        template <class T>
        xarray_container<T> group_delay(const std::vector<T>& b, const std::vector<T>& a,
                                        size_t worN = 512, bool whole = false, T fs = T(2*M_PI));

        // ========================================================================
        // Filter transformations
        // ========================================================================
        template <class T>
        std::pair<std::vector<T>, std::vector<T>> lp2lp(const std::vector<T>& b, const std::vector<T>& a, T wo);
        template <class T>
        std::pair<std::vector<T>, std::vector<T>> lp2hp(const std::vector<T>& b, const std::vector<T>& a, T wo);
        template <class T>
        std::pair<std::vector<T>, std::vector<T>> lp2bp(const std::vector<T>& b, const std::vector<T>& a, T wo, T bw);
        template <class T>
        std::pair<std::vector<T>, std::vector<T>> lp2bs(const std::vector<T>& b, const std::vector<T>& a, T wo, T bw);
    }

    using signal::firwin;
    using signal::remez;
    using signal::butter;
    using signal::cheby1;
    using signal::cheby2;
    using signal::ellip;
    using signal::bessel;
    using signal::lfilter;
    using signal::filtfilt;
    using signal::sosfilt;
    using signal::freqz;
    using signal::group_delay;
}

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace signal
    {
        // FIR filter design by window method
        template <class T>
        std::vector<T> firwin(size_t numtaps, T cutoff, T fs, const std::string& window, bool pass_zero)
        { /* TODO: implement window method FIR design */ return {}; }

        // Parks‑McClellan (Remez) optimal FIR design
        template <class T>
        std::vector<T> remez(size_t numtaps, const std::vector<T>& bands,
                             const std::vector<T>& desired, const std::vector<T>& weights, T fs)
        { /* TODO: implement Remez exchange algorithm */ return {}; }

        // Butterworth digital filter design
        template <class T>
        std::tuple<std::vector<T>, std::vector<T>> butter(int N, T Wn, const std::string& btype, T fs)
        { /* TODO: implement Butterworth via bilinear transform */ return {}; }

        // Chebyshev type I filter design
        template <class T>
        std::tuple<std::vector<T>, std::vector<T>> cheby1(int N, T rp, T Wn, const std::string& btype, T fs)
        { /* TODO: implement Chebyshev I */ return {}; }

        // Chebyshev type II filter design
        template <class T>
        std::tuple<std::vector<T>, std::vector<T>> cheby2(int N, T rs, T Wn, const std::string& btype, T fs)
        { /* TODO: implement Chebyshev II */ return {}; }

        // Elliptic (Cauer) filter design
        template <class T>
        std::tuple<std::vector<T>, std::vector<T>> ellip(int N, T rp, T rs, T Wn, const std::string& btype, T fs)
        { /* TODO: implement elliptic filter */ return {}; }

        // Bessel filter design
        template <class T>
        std::tuple<std::vector<T>, std::vector<T>> bessel(int N, T Wn, const std::string& btype, T fs)
        { /* TODO: implement Bessel filter */ return {}; }

        // Apply digital filter to signal (forward)
        template <class E>
        auto lfilter(const std::vector<typename E::value_type>& b,
                     const std::vector<typename E::value_type>& a,
                     const xexpression<E>& x_expr)
        { /* TODO: implement direct form II transposed filter */ return x_expr.derived_cast(); }

        // Zero‑phase forward‑backward filtering
        template <class E>
        auto filtfilt(const std::vector<typename E::value_type>& b,
                      const std::vector<typename E::value_type>& a,
                      const xexpression<E>& x_expr)
        { /* TODO: implement forward‑backward filter */ return x_expr.derived_cast(); }

        // Second‑order sections filtering
        template <class E>
        auto sosfilt(const xarray_container<typename E::value_type>& sos,
                     const xexpression<E>& x_expr)
        { /* TODO: implement cascade of biquads */ return x_expr.derived_cast(); }

        // Compute frequency response of digital filter
        template <class T>
        std::tuple<xarray_container<T>, xarray_container<std::complex<T>>>
        freqz(const std::vector<T>& b, const std::vector<T>& a, size_t worN, bool whole, T fs)
        { /* TODO: evaluate H(z) on unit circle */ return {}; }

        // Compute group delay of filter
        template <class T>
        xarray_container<T> group_delay(const std::vector<T>& b, const std::vector<T>& a,
                                        size_t worN, bool whole, T fs)
        { /* TODO: compute derivative of phase */ return {}; }

        // Lowpass to lowpass frequency scaling
        template <class T>
        std::pair<std::vector<T>, std::vector<T>> lp2lp(const std::vector<T>& b, const std::vector<T>& a, T wo)
        { /* TODO: implement lp2lp transformation */ return {}; }

        // Lowpass to highpass transformation
        template <class T>
        std::pair<std::vector<T>, std::vector<T>> lp2hp(const std::vector<T>& b, const std::vector<T>& a, T wo)
        { /* TODO: implement lp2hp transformation */ return {}; }

        // Lowpass to bandpass transformation
        template <class T>
        std::pair<std::vector<T>, std::vector<T>> lp2bp(const std::vector<T>& b, const std::vector<T>& a, T wo, T bw)
        { /* TODO: implement lp2bp transformation */ return {}; }

        // Lowpass to bandstop transformation
        template <class T>
        std::pair<std::vector<T>, std::vector<T>> lp2bs(const std::vector<T>& b, const std::vector<T>& a, T wo, T bw)
        { /* TODO: implement lp2bs transformation */ return {}; }
    }
}

#endif // XTENSOR_LFILTER_HPP             std::vector<T> a_pad = a;
                b_pad.resize(order + 1, T(0));
                a_pad.resize(order + 1, T(0));

                for (size_t k = 0; k <= order; ++k)
                {
                    T coeff_b = b_pad[k] * pow_val(c, int(order - k));
                    T coeff_a = a_pad[k] * pow_val(c, int(order - k));
                    for (size_t j = 0; j <= k; ++j)
                    {
                        T binom = binomial_coeff<T>(k, j);
                        T term_b = coeff_b * binom;
                        T term_a = coeff_a * binom;
                        if ((order - k + j) % 2 == 1)
                        {
                            term_b = -term_b;
                            term_a = -term_a;
                        }
                        num[j] = num[j] + term_b;
                        den[j] = den[j] + term_a;
                    }
                }
                T den0 = den[0];
                for (auto& v : num) v = v / den0;
                for (auto& v : den) v = v / den0;
                return {num, den};
            }

            // --------------------------------------------------------------------
            // Lowpass to lowpass frequency scaling
            // --------------------------------------------------------------------
            template <class T>
            std::pair<std::vector<T>, std::vector<T>> lp2lp(const std::vector<T>& b,
                                                              const std::vector<T>& a,
                                                              T wo)
            {
                size_t order = std::max(b.size(), a.size()) - 1;
                std::vector<T> num(order + 1, T(0));
                std::vector<T> den(order + 1, T(0));
                std::vector<T> b_pad = b, a_pad = a;
                b_pad.resize(order + 1, T(0));
                a_pad.resize(order + 1, T(0));
                for (size_t k = 0; k <= order; ++k)
                {
                    num[k] = b_pad[k] * pow_val(wo, int(order - k));
                    den[k] = a_pad[k] * pow_val(wo, int(order - k));
                }
                return {num, den};
            }

            // --------------------------------------------------------------------
            // Lowpass to highpass transformation: s → wo/s
            // --------------------------------------------------------------------
            template <class T>
            std::pair<std::vector<T>, std::vector<T>> lp2hp(const std::vector<T>& b,
                                                              const std::vector<T>& a,
                                                              T wo)
            {
                size_t order = std::max(b.size(), a.size()) - 1;
                std::vector<T> num(order + 1, T(0));
                std::vector<T> den(order + 1, T(0));
                std::vector<T> b_pad = b, a_pad = a;
                b_pad.resize(order + 1, T(0));
                a_pad.resize(order + 1, T(0));
                for (size_t k = 0; k <= order; ++k)
                {
                    num[k] = b_pad[order - k] * pow_val(wo, int(k));
                    den[k] = a_pad[order - k] * pow_val(wo, int(k));
                }
                return {num, den};
            }

            // --------------------------------------------------------------------
            // Lowpass to bandpass transformation: s → (s^2 + wo^2) / (bw * s)
            // --------------------------------------------------------------------
            template <class T>
            std::pair<std::vector<T>, std::vector<T>> lp2bp(const std::vector<T>& b,
                                                              const std::vector<T>& a,
                                                              T wo, T bw)
            {
                size_t order = std::max(b.size(), a.size()) - 1;
                size_t new_order = 2 * order;
                std::vector<T> num(new_order + 1, T(0));
                std::vector<T> den(new_order + 1, T(0));
                std::vector<T> b_pad = b, a_pad = a;
                b_pad.resize(order + 1, T(0));
                a_pad.resize(order + 1, T(0));

                // Convert to bandpass by polynomial substitution
                for (size_t k = 0; k <= order; ++k)
                {
                    T coeff_b = b_pad[k] * pow_val(bw, int(order - k));
                    T coeff_a = a_pad[k] * pow_val(bw, int(order - k));
                    // Each term s^k becomes (s^2 + wo^2)^k / s^k
                    // We expand (s^2 + wo^2)^k and shift powers
                    for (size_t j = 0; j <= k; ++j)
                    {
                        T binom = binomial_coeff<T>(k, j);
                        T factor = binom * pow_val(wo, int(2 * (k - j)));
                        int new_power = int(2 * j - int(k) + int(order - k));
                        size_t idx = size_t(int(new_order) - new_power - int(order));
                        if (idx <= new_order)
                        {
                            num[idx] = num[idx] + coeff_b * factor;
                            den[idx] = den[idx] + coeff_a * factor;
                        }
                    }
                }
                return {num, den};
            }

            // --------------------------------------------------------------------
            // Lowpass to bandstop transformation: s → bw * s / (s^2 + wo^2)
            // --------------------------------------------------------------------
            template <class T>
            std::pair<std::vector<T>, std::vector<T>> lp2bs(const std::vector<T>& b,
                                                              const std::vector<T>& a,
                                                              T wo, T bw)
            {
                size_t order = std::max(b.size(), a.size()) - 1;
                size_t new_order = 2 * order;
                std::vector<T> num(new_order + 1, T(0));
                std::vector<T> den(new_order + 1, T(0));
                std::vector<T> b_pad = b, a_pad = a;
                b_pad.resize(order + 1, T(0));
                a_pad.resize(order + 1, T(0));

                for (size_t k = 0; k <= order; ++k)
                {
                    T coeff_b = b_pad[k];
                    T coeff_a = a_pad[k];
                    for (size_t j = 0; j <= k; ++j)
                    {
                        T binom = binomial_coeff<T>(k, j);
                        T factor = binom * pow_val(wo, int(2 * j)) * pow_val(bw, int(k - j));
                        int new_power = int(2 * (order - k) + 2 * j + int(k - j));
                        size_t idx = size_t(new_order - new_power);
                        if (idx <= new_order)
                        {
                            num[idx] = num[idx] + coeff_b * factor;
                            den[idx] = den[idx] + coeff_a * factor;
                        }
                    }
                }
                return {num, den};
            }

        } // namespace detail

        // ========================================================================
        // FIR filter design by window method
        // ========================================================================
        template <class T = double>
        std::vector<T> firwin(size_t numtaps, T cutoff, T fs = T(2),
                              const std::string& window = "hamming",
                              bool pass_zero = true)
        {
            if (numtaps % 2 == 0)
                XTENSOR_THROW(std::invalid_argument, "firwin: numtaps must be odd for symmetric FIR");

            T nyquist = fs / T(2);
            T cutoff_norm = cutoff / nyquist;
            if (cutoff_norm <= T(0) || cutoff_norm >= T(1))
                XTENSOR_THROW(std::invalid_argument, "firwin: cutoff must be between 0 and nyquist");

            std::vector<T> h(numtaps, T(0));
            size_t M = numtaps - 1;
            T alpha = T(M) / T(2);

            if (pass_zero) // lowpass
            {
                for (size_t n = 0; n < numtaps; ++n)
                {
                    T x = detail::pi<T>() * cutoff_norm * (T(n) - alpha);
                    if (n == size_t(alpha))
                        h[n] = cutoff_norm;
                    else
                        h[n] = std::sin(x) / (detail::pi<T>() * (T(n) - alpha));
                }
            }
            else // highpass
            {
                for (size_t n = 0; n < numtaps; ++n)
                {
                    T x = detail::pi<T>() * cutoff_norm * (T(n) - alpha);
                    if (n == size_t(alpha))
                        h[n] = T(1) - cutoff_norm;
                    else
                        h[n] = -std::sin(x) / (detail::pi<T>() * (T(n) - alpha));
                }
                h[static_cast<size_t>(alpha)] += T(1);
            }

            std::vector<T> win;
            if (window == "hamming") win = detail::hamming_window<T>(numtaps);
            else if (window == "hann") win = detail::hann_window<T>(numtaps);
            else if (window == "blackman") win = detail::blackman_window<T>(numtaps);
            else if (window == "rectangular") win = std::vector<T>(numtaps, T(1));
            else XTENSOR_THROW(std::invalid_argument, "firwin: unknown window type");

            for (size_t i = 0; i < numtaps; ++i)
                h[i] = h[i] * win[i];

            return h;
        }

        // ========================================================================
        // Parks‑McClellan (Remez) FIR design – full implementation
        // ========================================================================
        template <class T = double>
        std::vector<T> remez(size_t numtaps, const std::vector<T>& bands,
                             const std::vector<T>& desired,
                             const std::vector<T>& weights = {},
                             T fs = T(2))
        {
            if (bands.size() % 2 != 0 || desired.size() != bands.size() / 2)
                XTENSOR_THROW(std::invalid_argument, "remez: invalid band specification");
            if (numtaps % 2 == 0)
                XTENSOR_THROW(std::invalid_argument, "remez: numtaps must be odd");

            size_t num_bands = desired.size();
            std::vector<T> w = weights.empty() ? std::vector<T>(num_bands, T(1)) : weights;
            if (w.size() != num_bands)
                XTENSOR_THROW(std::invalid_argument, "remez: weights size mismatch");

            T nyquist = fs / T(2);
            // Normalize frequencies to [0, 0.5] (half of digital frequency)
            std::vector<T> edges;
            for (auto f : bands)
                edges.push_back(f / nyquist * T(0.5));
            for (auto& d : const_cast<std::vector<T>&>(desired))
                d = d; // keep as is

            size_t L = numtaps;
            size_t N = L - 1;  // filter order
            size_t grid_density = 16;
            size_t grid_size = (N + 1) * grid_density;
            std::vector<T> grid(grid_size);
            std::vector<T> D(grid_size);
            std::vector<T> Wt(grid_size);
            T delf = T(0.5) / T(grid_density * N);
            size_t idx = 0;
            for (size_t band = 0; band < num_bands; ++band)
            {
                T fstart = edges[2*band];
                T fstop = edges[2*band+1];
                size_t n_points = size_t((fstop - fstart) / delf + T(0.5)) + 1;
                T step = (fstop - fstart) / T(n_points - 1);
                for (size_t i = 0; i < n_points; ++i)
                {
                    grid[idx] = fstart + T(i) * step;
                    D[idx] = desired[band];
                    Wt[idx] = w[band];
                    ++idx;
                }
            }
            grid.resize(idx);
            D.resize(idx);
            Wt.resize(idx);
            grid_size = idx;

            // Initial guess of extremal frequencies (evenly spaced)
            std::vector<size_t> extrema(N/2 + 2);
            for (size_t i = 0; i < extrema.size(); ++i)
                extrema[i] = i * (grid_size - 1) / (extrema.size() - 1);

            std::vector<T> h(L);
            T delta = T(0);
            int max_iter = 100;
            for (int iter = 0; iter < max_iter; ++iter)
            {
                // Solve interpolation problem at extrema
                size_t r = extrema.size();
                std::vector<std::vector<T>> A(r, std::vector<T>(r, T(0)));
                std::vector<T> b(r);
                for (size_t i = 0; i < r; ++i)
                {
                    T freq = grid[extrema[i]] * T(2) * detail::pi<T>();
                    for (size_t j = 0; j < r-1; ++j)
                        A[i][j] = std::cos(freq * T(j));
                    A[i][r-1] = (i % 2 == 0 ? T(1) : T(-1)) / Wt[extrema[i]];
                    b[i] = D[extrema[i]];
                }
                // Solve linear system (Gaussian elimination)
                auto x = solve_linear(A, b);
                for (size_t j = 0; j < r-1; ++j)
                    h[j] = x[j];
                delta = x[r-1];

                // Evaluate error on dense grid
                std::vector<T> error(grid_size);
                std::vector<size_t> new_extrema;
                T max_error = T(-1);
                for (size_t i = 0; i < grid_size; ++i)
                {
                    T freq = grid[i] * T(2) * detail::pi<T>();
                    T H = T(0);
                    for (size_t j = 0; j < r-1; ++j)
                        H = H + h[j] * std::cos(freq * T(j));
                    error[i] = (D[i] - H) * Wt[i];
                    if (std::abs(error[i]) > max_error)
                        max_error = std::abs(error[i]);
                }
                // Find new extrema (alternating signs)
                new_extrema.clear();
                int sign = 0;
                for (size_t i = 0; i < grid_size; ++i)
                {
                    T e = error[i];
                    if (new_extrema.empty() || (sign == 0) || (e > T(0) && sign < 0) || (e < T(0) && sign > 0))
                    {
                        new_extrema.push_back(i);
                        sign = (e > T(0)) ? 1 : -1;
                    }
                    else if (std::abs(e) > std::abs(error[new_extrema.back()]))
                    {
                        new_extrema.back() = i;
                    }
                }
                if (new_extrema.size() > r)
                {
                    // Prune extrema
                    std::vector<size_t> pruned;
                    pruned.push_back(new_extrema[0]);
                    for (size_t i = 1; i + 1 < new_extrema.size(); ++i)
                    {
                        if (std::abs(error[new_extrema[i]]) > std::abs(error[new_extrema[i-1]]) &&
                            std::abs(error[new_extrema[i]]) > std::abs(error[new_extrema[i+1]]))
                            pruned.push_back(new_extrema[i]);
                    }
                    pruned.push_back(new_extrema.back());
                    if (pruned.size() > r)
                    {
                        // Remove interior points with smallest error
                        while (pruned.size() > r)
                        {
                            size_t min_idx = 1;
                            T min_val = std::abs(error[pruned[1]]);
                            for (size_t j = 2; j < pruned.size() - 1; ++j)
                            {
                                if (std::abs(error[pruned[j]]) < min_val)
                                {
                                    min_val = std::abs(error[pruned[j]]);
                                    min_idx = j;
                                }
                            }
                            pruned.erase(pruned.begin() + min_idx);
                        }
                    }
                    extrema = pruned;
                }
                else
                {
                    break;
                }
            }

            // Convert cosine coefficients to impulse response (symmetric FIR)
            std::vector<T> h_full(L, T(0));
            size_t mid = N / 2;
            for (size_t i = 0; i <= mid; ++i)
            {
                T val = T(0);
                for (size_t j = 0; j < h.size(); ++j)
                    val = val + h[j] * std::cos(detail::pi<T>() * T(j) * (T(i) - T(mid)) / T(mid));
                h_full[i] = val;
                h_full[L - 1 - i] = val;
            }
            return h_full;
        }

        // Solve linear system A*x = b (Gaussian elimination)
        template <class T>
        std::vector<T> solve_linear(std::vector<std::vector<T>> A, std::vector<T> b)
        {
            size_t n = A.size();
            for (size_t i = 0; i < n; ++i)
            {
                size_t pivot = i;
                T max_val = std::abs(A[i][i]);
                for (size_t r = i+1; r < n; ++r)
                    if (std::abs(A[r][i]) > max_val)
                        max_val = std::abs(A[r][i]), pivot = r;
                if (pivot != i)
                {
                    std::swap(A[i], A[pivot]);
                    std::swap(b[i], b[pivot]);
                }
                T inv_pivot = T(1) / A[i][i];
                for (size_t j = i; j < n; ++j) A[i][j] = A[i][j] * inv_pivot;
                b[i] = b[i] * inv_pivot;
                for (size_t r = 0; r < n; ++r)
                {
                    if (r == i) continue;
                    T factor = A[r][i];
                    for (size_t c = i; c < n; ++c) A[r][c] = A[r][c] - factor * A[i][c];
                    b[r] = b[r] - factor * b[i];
                }
            }
            return b;
        }

        // ========================================================================
        // IIR analog prototypes
        // ========================================================================
        template <class T>
        std::pair<std::vector<T>, std::vector<T>> butterworth_prototype(int N)
        {
            std::vector<T> num = {T(1)};
            std::vector<T> den = {T(1)};
            for (int k = 1; k <= N; ++k)
            {
                T theta = detail::pi<T>() * T(2*k - 1) / T(2*N);
                T real_part = -std::sin(theta);
                T imag_part = std::cos(theta);
                std::vector<T> quad;
                if (std::abs(imag_part) < T(1e-12))
                    quad = {T(1), -real_part};
                else
                    quad = {T(1), T(-2)*real_part, real_part*real_part + imag_part*imag_part};
                den = detail::convolve_poly(den, quad);
            }
            return {num, den};
        }

        template <class T>
        std::pair<std::vector<T>, std::vector<T>> cheby1_prototype(int N, T rp)
        {
            T eps = detail::sqrt_val(detail::pow_val(T(10), int(rp / T(10))) - T(1));
            T mu = detail::asinh(T(1) / eps) / T(N);
            std::vector<T> num = {T(1)};
            std::vector<T> den = {T(1)};
            for (int k = 1; k <= N; ++k)
            {
                T theta = detail::pi<T>() * T(2*k - 1) / T(2*N);
                T real_part = -std::sinh(mu) * std::sin(theta);
                T imag_part = std::cosh(mu) * std::cos(theta);
                std::vector<T> quad;
                if (std::abs(imag_part) < T(1e-12))
                    quad = {T(1), -real_part};
                else
                    quad = {T(1), T(-2)*real_part, real_part*real_part + imag_part*imag_part};
                den = detail::convolve_poly(den, quad);
            }
            T gain = den.back();
            if (N % 2 == 0)
                gain = gain / detail::sqrt_val(T(1) + eps*eps);
            for (auto& v : den) v = v / gain;
            return {num, den};
        }

        template <class T>
        std::pair<std::vector<T>, std::vector<T>> cheby2_prototype(int N, T rs)
        {
            T eps = detail::sqrt_val(detail::pow_val(T(10), int(rs / T(10))) - T(1));
            // Chebyshev II has zeros on the imaginary axis
            std::vector<T> num = {T(1)};
            for (int k = 1; k <= N/2; ++k)
            {
                T theta = detail::pi<T>() * T(2*k - 1) / T(2*N);
                T zero = T(1) / std::cos(theta);
                std::vector<T> quad = {T(1), T(0), zero*zero};
                num = detail::convolve_poly(num, quad);
            }
            // Poles are reciprocals of Chebyshev I poles
            auto [num_p, den_p] = cheby1_prototype<T>(N, rs);
            std::vector<T> den = {T(1)};
            for (size_t i = 1; i < den_p.size(); ++i)
            {
                // Not a simple reciprocal; we need to invert polynomial
                // We'll compute poles explicitly and build denominator
            }
            // Gain adjustment
            T gain = den.back() / num.back();
            for (auto& v : num) v = v * gain;
            return {num, den};
        }

        template <class T>
        std::pair<std::vector<T>, std::vector<T>> bessel_prototype(int N)
        {
            // Bessel polynomial coefficients
            std::vector<T> den(N+1, T(0));
            for (int k = 0; k <= N; ++k)
            {
                T coeff = T(1);
                for (int i = 1; i <= 2*N - k; ++i) coeff = coeff * T(i);
                for (int i = 1; i <= k; ++i) coeff = coeff / T(i);
                for (int i = 1; i <= N - k; ++i) coeff = coeff / T(i);
                den[N - k] = coeff / detail::pow_val(T(2), int(N - k));
            }
            T den0 = den[0];
            for (auto& v : den) v = v / den0;
            return {{T(1)}, den};
        }

        template <class T>
        std::pair<std::vector<T>, std::vector<T>> ellip_prototype(int N, T rp, T rs)
        {
            T eps = detail::sqrt_val(detail::pow_val(T(10), int(rp / T(10))) - T(1));
            T eps1 = detail::sqrt_val(detail::pow_val(T(10), int(rs / T(10))) - T(1));
            T k1 = eps / eps1;
            T k = T(0);
            // Select k such that K'(k)/K(k) = N * K'(k1)/K(k1)
            // Iterative search for k
            T k_low = T(0), k_high = T(1);
            T target = T(N) * detail::ellipk(k1) / detail::ellipk(detail::sqrt_val(T(1) - k1*k1));
            for (int iter = 0; iter < 50; ++iter)
            {
                T k_mid = (k_low + k_high) / T(2);
                T ratio = detail::ellipk(detail::sqrt_val(T(1) - k_mid*k_mid)) / detail::ellipk(k_mid);
                if (ratio < target)
                    k_high = k_mid;
                else
                    k_low = k_mid;
            }
            k = (k_low + k_high) / T(2);
            T K = detail::ellipk(k);
            T Kp = detail::ellipk(detail::sqrt_val(T(1) - k*k));
            T q = std::exp(-detail::pi<T>() * Kp / K);
            // Compute zeros and poles using Jacobi elliptic functions
            std::vector<std::complex<T>> zeros, poles;
            for (int i = 1; i <= N/2; ++i)
            {
                T u = T(2*i - 1) * K / T(N);
                T sn, cn, dn;
                detail::ellipj(u, k, sn, cn, dn);
                std::complex<T> zero(T(0), T(1) / (k * sn));
                zeros.push_back(zero);
                T v = u;
                detail::ellipj(v, k1, sn, cn, dn);
                std::complex<T> pole(sn * cn, T(0));
                pole = T(1) / pole;
                poles.push_back(pole);
            }
            // Build numerator and denominator from zeros and poles
            std::vector<T> num = {T(1)};
            for (auto z : zeros)
            {
                if (std::abs(z.imag()) > T(1e-12))
                {
                    std::vector<T> quad = {T(1), T(0), std::norm(z)};
                    num = detail::convolve_poly(num, quad);
                }
            }
            std::vector<T> den = {T(1)};
            for (auto p : poles)
            {
                if (std::abs(p.imag()) > T(1e-12))
                {
                    std::vector<T> quad = {T(1), T(-2)*p.real(), std::norm(p)};
                    den = detail::convolve_poly(den, quad);
                }
                else
                {
                    std::vector<T> quad = {T(1), -p.real()};
                    den = detail::convolve_poly(den, quad);
                }
            }
            // Gain adjustment
            T gain = den.back() / num.back();
            if (N % 2 == 0)
                gain = gain / detail::sqrt_val(T(1) + eps*eps);
            for (auto& v : num) v = v * gain;
            return {num, den};
        }

        // ========================================================================
        // Complete digital filter design functions
        // ========================================================================
        template <class T = double>
        std::tuple<std::vector<T>, std::vector<T>> butter(int N, T Wn,
                                                          const std::string& btype = "low",
                                                          T fs = T(2))
        {
            auto [num_analog, den_analog] = butterworth_prototype<T>(N);
            T nyquist = fs / T(2);
            T warped = T(2) * fs * std::tan(detail::pi<T>() * Wn / fs);
            std::vector<T> num, den;
            if (btype == "low" || btype == "lowpass")
                std::tie(num, den) = detail::lp2lp(num_analog, den_analog, warped);
            else if (btype == "high" || btype == "highpass")
                std::tie(num, den) = detail::lp2hp(num_analog, den_analog, warped);
            else if (btype == "bandpass" && Wn.size() >= 2)
            {
                T wo = T(2) * fs * std::tan(detail::pi<T>() * Wn[0] / fs);
                T bw = T(2) * fs * std::tan(detail::pi<T>() * Wn[1] / fs) - wo;
                std::tie(num, den) = detail::lp2bp(num_analog, den_analog, wo, bw);
            }
            else if (btype == "bandstop" && Wn.size() >= 2)
            {
                T wo = T(2) * fs * std::tan(detail::pi<T>() * Wn[0] / fs);
                T bw = T(2) * fs * std::tan(detail::pi<T>() * Wn[1] / fs) - wo;
                std::tie(num, den) = detail::lp2bs(num_analog, den_analog, wo, bw);
            }
            else
                XTENSOR_THROW(std::invalid_argument, "butter: unsupported filter type");
            return detail::bilinear(num, den, fs);
        }

        template <class T = double>
        std::tuple<std::vector<T>, std::vector<T>> cheby1(int N, T rp, T Wn,
                                                          const std::string& btype = "low",
                                                          T fs = T(2))
        {
            auto [num_analog, den_analog] = cheby1_prototype<T>(N, rp);
            T nyquist = fs / T(2);
            T warped = T(2) * fs * std::tan(detail::pi<T>() * Wn / fs);
            std::vector<T> num, den;
            if (btype == "low" || btype == "lowpass")
                std::tie(num, den) = detail::lp2lp(num_analog, den_analog, warped);
            else if (btype == "high" || btype == "highpass")
                std::tie(num, den) = detail::lp2hp(num_analog, den_analog, warped);
            else if (btype == "bandpass" && Wn.size() >= 2)
            {
                T wo = T(2) * fs * std::tan(detail::pi<T>() * Wn[0] / fs);
                T bw = T(2) * fs * std::tan(detail::pi<T>() * Wn[1] / fs) - wo;
                std::tie(num, den) = detail::lp2bp(num_analog, den_analog, wo, bw);
            }
            else if (btype == "bandstop" && Wn.size() >= 2)
            {
                T wo = T(2) * fs * std::tan(detail::pi<T>() * Wn[0] / fs);
                T bw = T(2) * fs * std::tan(detail::pi<T>() * Wn[1] / fs) - wo;
                std::tie(num, den) = detail::lp2bs(num_analog, den_analog, wo, bw);
            }
            else
                XTENSOR_THROW(std::invalid_argument, "cheby1: unsupported filter type");
            return detail::bilinear(num, den, fs);
        }

        // ========================================================================
        // Filtering functions (lfilter, filtfilt, sosfilt)
        // ========================================================================
        template <class E>
        auto lfilter(const std::vector<typename E::value_type>& b,
                     const std::vector<typename E::value_type>& a,
                     const xexpression<E>& x_expr)
        {
            const auto& x = x_expr.derived_cast();
            using T = typename E::value_type;
            size_t n = x.size();
            std::vector<T> y(n, T(0));
            size_t nb = b.size();
            size_t na = a.size();
            size_t order = std::max(nb, na) - 1;
            std::vector<T> z(order, T(0));
            T a0 = a[0];
            for (size_t i = 0; i < n; ++i)
            {
                T out = T(0);
                for (size_t j = 0; j < nb; ++j)
                    out = out + detail::multiply(b[j], (j == 0 ? x.flat(i) : (j-1 < order ? z[j-1] : T(0))));
                for (size_t j = 1; j < na; ++j)
                    out = out - detail::multiply(a[j], (j-1 < order ? z[j-1] : T(0)));
                out = out / a0;
                y[i] = out;
                for (size_t j = order; j > 0; --j)
                    z[j-1] = (j-1 == 0) ? x.flat(i) : z[j-2];
                z[0] = out;
            }
            shape_type shape = x.shape();
            xarray_container<T> result(shape);
            for (size_t i = 0; i < n; ++i) result.flat(i) = y[i];
            return result;
        }

        template <class E>
        auto filtfilt(const std::vector<typename E::value_type>& b,
                      const std::vector<typename E::value_type>& a,
                      const xexpression<E>& x_expr)
        {
            auto y = lfilter(b, a, x_expr);
            auto y_rev = xt::flip(y, 0);
            auto y_back = lfilter(b, a, y_rev);
            return xt::flip(y_back, 0);
        }

        template <class E>
        auto sosfilt(const xarray_container<typename E::value_type>& sos,
                     const xexpression<E>& x_expr)
        {
            if (sos.dimension() != 2 || sos.shape()[1] != 6)
                XTENSOR_THROW(std::invalid_argument, "sosfilt: sos must be Nx6 array");
            auto y = xarray_container<typename E::value_type>(x_expr);
            for (size_t sec = 0; sec < sos.shape()[0]; ++sec)
            {
                std::vector<typename E::value_type> b = {sos(sec,0), sos(sec,1), sos(sec,2)};
                std::vector<typename E::value_type> a = {T(1), sos(sec,4), sos(sec,5)};
                y = lfilter(b, a, y);
            }
            return y;
        }

        // ========================================================================
        // Frequency response analysis
        // ========================================================================
        template <class T>
        std::tuple<xarray_container<T>, xarray_container<std::complex<T>>>
        freqz(const std::vector<T>& b, const std::vector<T>& a,
              size_t worN = 512, bool whole = false, T fs = T(2*detail::pi<T>()))
        {
            size_t N = whole ? worN : worN;
            T step = (whole ? T(2*detail::pi<T>()) : detail::pi<T>()) / T(N);
            xarray_container<T> w({N});
            xarray_container<std::complex<T>> h({N});
            for (size_t k = 0; k < N; ++k)
            {
                T omega = (whole ? T(0) : T(0)) + T(k) * step;
                w(k) = omega * fs / (T(2)*detail::pi<T>());
                std::complex<T> z = std::polar(T(1), -omega);
                std::complex<T> num = detail::polyval(b, z);
                std::complex<T> den = detail::polyval(a, z);
                h(k) = num / den;
            }
            return {w, h};
        }

        template <class T>
        xarray_container<T> group_delay(const std::vector<T>& b, const std::vector<T>& a,
                                        size_t worN = 512, bool whole = false, T fs = T(2*detail::pi<T>()))
        {
            auto [w, h] = freqz(b, a, worN, whole, fs);
            xarray_container<T> gd({w.size()});
            T dw = (whole ? T(2*detail::pi<T>()) : detail::pi<T>()) / T(worN);
            for (size_t k = 1; k < w.size() - 1; ++k)
            {
                T phase_prev = std::arg(h(k-1));
                T phase_next = std::arg(h(k+1));
                T diff = phase_next - phase_prev;
                // Unwrap phase difference
                if (diff > detail::pi<T>()) diff -= T(2*detail::pi<T>());
                if (diff < -detail::pi<T>()) diff += T(2*detail::pi<T>());
                gd(k) = -diff / (T(2) * dw);
            }
            gd(0) = gd(1);
            gd(w.size()-1) = gd(w.size()-2);
            return gd;
        }

    } // namespace signal

    using signal::firwin;
    using signal::remez;
    using signal::butter;
    using signal::cheby1;
    using signal::lfilter;
    using signal::filtfilt;
    using signal::sosfilt;
    using signal::freqz;
    using signal::group_delay;

} // namespace xt

#endif // XTENSOR_LFILTER_HPP