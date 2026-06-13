//File 0606 : xtensor-signal/xstft.hpp
//Short‑Time Fourier Transform (STFT) and inverse STFT with SIMD‑accelerated FFT, overlapping windows, and perfect reconstruction support for time‑frequency analysis.
#ifndef XTENSOR_SIGNAL_XSTFT_HPP
#define XTENSOR_SIGNAL_XSTFT_HPP

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
     * @enum stft_padding_mode
     * @brief How to pad the signal at boundaries for STFT.
     */
    enum class stft_padding_mode : int {
        zero,         // zero-pad
        reflect,      // reflect at boundaries
        constant      // pad with a constant value
    };

    namespace detail {

        /**
         * Pad a 1D signal symmetrically at both ends.
         */
        template <class T>
        inline auto pad_reflect(const T* x, std::size_t n, std::size_t pad_left,
                                 std::size_t pad_right) {
            std::size_t total = n + pad_left + pad_right;
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({total});
            T* d = result.data();
            for (std::size_t i = 0; i < pad_left; ++i)
                d[i] = x[pad_left - i];
            std::copy(x, x + n, d + pad_left);
            for (std::size_t i = 0; i < pad_right; ++i)
                d[pad_left + n + i] = x[n - 2 - i];
            return result;
        }

        /**
         * Pad a 1D signal with zeros.
         */
        template <class T>
        inline auto pad_zero(const T* x, std::size_t n, std::size_t pad_left,
                              std::size_t pad_right) {
            std::size_t total = n + pad_left + pad_right;
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({total}, T(0));
            T* d = result.data();
            std::copy(x, x + n, d + pad_left);
            return result;
        }

        /**
         * Pad a 1D signal with a constant value.
         */
        template <class T>
        inline auto pad_constant(const T* x, std::size_t n, std::size_t pad_left,
                                  std::size_t pad_right, T constant = T(0)) {
            std::size_t total = n + pad_left + pad_right;
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({total}, constant);
            T* d = result.data();
            std::copy(x, x + n, d + pad_left);
            return result;
        }

        /**
         * Compute the forward STFT for a 1D real signal.
         * @param x Padded input signal (length total_len).
         * @param window Window function (length win_len).
         * @param n_fft Number of FFT points.
         * @param hop_length Hop between frames.
         * @return Complex 2D array (n_frames × (n_fft/2+1)).
         */
        template <class T>
        inline auto stft_forward(const T* x, std::size_t total_len,
                                  const T* window, std::size_t win_len,
                                  std::size_t n_fft, std::size_t hop_length) {
            std::size_t n_frames = (total_len >= win_len)
                ? (total_len - win_len) / hop_length + 1
                : 0;
            xarray_container<uvector<std::complex<T>>, DEFAULT_LAYOUT,
                std::vector<std::size_t>> stft_matrix({n_frames, n_fft / 2 + 1});

            std::vector<std::complex<T>> fft_buf(n_fft, std::complex<T>(0, 0));
            for (std::size_t f = 0; f < n_frames; ++f) {
                std::size_t offset = f * hop_length;
                // Windowing
                for (std::size_t i = 0; i < win_len; ++i) {
                    fft_buf[i] = std::complex<T>(x[offset + i] * window[i], T(0));
                }
                std::fill(fft_buf.begin() + win_len, fft_buf.end(),
                          std::complex<T>(0, 0));
                // FFT
                xt::fft::detail::fft_radix2(fft_buf.data(), n_fft, false);
                // Store half-spectrum
                std::copy(fft_buf.begin(), fft_buf.begin() + n_fft / 2 + 1,
                          stft_matrix.data() + f * (n_fft / 2 + 1));
            }
            return stft_matrix;
        }

        /**
         * Compute the inverse STFT using the overlap‑add method.
         * @param stft_matrix Complex STFT (n_frames × (n_fft/2+1)).
         * @param window Window function (length win_len).
         * @param hop_length Hop size.
         * @param original_length Desired length of the output signal.
         * @return Real 1D reconstructed signal.
         */
        template <class T>
        inline auto stft_inverse(
            const xarray_container<uvector<std::complex<T>>>& stft_matrix,
            const T* window, std::size_t win_len,
            std::size_t hop_length, std::size_t original_length) {
            auto sh = stft_matrix.shape();
            std::size_t n_frames = sh[0];
            std::size_t n_fft = (sh[1] - 1) * 2;

            // Reconstructed signal (overlap‑add buffer)
            std::size_t recon_len = (n_frames - 1) * hop_length + n_fft;
            std::vector<T> recon(recon_len, T(0));
            std::vector<T> win_squared_sum(recon_len, T(0));

            std::vector<std::complex<T>> fft_buf(n_fft);
            for (std::size_t f = 0; f < n_frames; ++f) {
                // Copy half-spectrum and enforce conjugate symmetry
                for (std::size_t k = 0; k < n_fft / 2 + 1; ++k)
                    fft_buf[k] = stft_matrix(f, k);
                for (std::size_t k = n_fft / 2 + 1; k < n_fft; ++k)
                    fft_buf[k] = std::conj(stft_matrix(f, n_fft - k));

                // Inverse FFT
                xt::fft::detail::fft_radix2(fft_buf.data(), n_fft, true);

                // Overlap‑add
                std::size_t offset = f * hop_length;
                for (std::size_t i = 0; i < n_fft; ++i) {
                    T sample = std::real(fft_buf[i]);
                    // Apply window again for reconstruction
                    T w = (i < win_len) ? window[i] : T(0);
                    recon[offset + i] += sample * w;
                    if (i < win_len)
                        win_squared_sum[offset + i] += w * w;
                }
            }

            // Normalize by the sum of squared windows
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({original_length});
            for (std::size_t i = 0; i < original_length; ++i) {
                if (win_squared_sum[i] > T(1e-15))
                    result[i] = recon[i] / win_squared_sum[i];
                else
                    result[i] = recon[i];
            }
            return result;
        }
    }

    /**
     * Compute the Short‑Time Fourier Transform (STFT) of a 1D signal.
     * @param x Input signal (1D).
     * @param n_fft Number of FFT points.
     * @param hop_length Hop between frames.
     * @param win_len Window length (if 0, defaults to n_fft).
     * @param win_type Type of window.
     * @param padding Boundary padding mode.
     * @return Complex 2D STFT matrix (n_frames × (n_fft/2+1)).
     */
    template <class E>
    inline auto stft(const xexpression<E>& x,
                     std::size_t n_fft = 2048,
                     std::size_t hop_length = 512,
                     std::size_t win_len = 0,
                     window_type win_type = window_type::hanning,
                     stft_padding_mode padding = stft_padding_mode::reflect) {
        using T = typename std::decay_t<E>::value_type;
        auto signal = xt::eval(x.derived_cast());
        if (signal.dimension() != 1)
            throw std::runtime_error("stft: input must be 1D.");
        std::size_t sig_len = signal.size();

        if (win_len == 0) win_len = n_fft;
        if (win_len > n_fft) win_len = n_fft;

        // Generate window
        auto win = window<T>(win_type, win_len);

        // Pad the signal to ensure full coverage
        std::size_t pad_left = win_len / 2;
        std::size_t pad_right = win_len / 2;
        if (padding == stft_padding_mode::reflect) {
            auto padded = detail::pad_reflect(signal.data(), sig_len, pad_left, pad_right);
            return detail::stft_forward(padded.data(), padded.size(),
                                         win.data(), win_len, n_fft, hop_length);
        } else if (padding == stft_padding_mode::zero) {
            auto padded = detail::pad_zero(signal.data(), sig_len, pad_left, pad_right);
            return detail::stft_forward(padded.data(), padded.size(),
                                         win.data(), win_len, n_fft, hop_length);
        } else { // constant
            auto padded = detail::pad_constant(signal.data(), sig_len, pad_left, pad_right, T(0));
            return detail::stft_forward(padded.data(), padded.size(),
                                         win.data(), win_len, n_fft, hop_length);
        }
    }

    /**
     * Compute the inverse STFT (reconstruct the signal from its STFT).
     * Uses the overlap‑add method with the same window.
     * @param stft_matrix Complex STFT (n_frames × (n_fft/2+1)).
     * @param original_length Desired length of the output signal.
     * @param hop_length Hop size used in the forward STFT.
     * @param win_type Window type used in forward STFT.
     * @return Reconstructed 1D signal.
     */
    template <class E>
    inline auto istft(const xexpression<E>& stft_matrix,
                      std::size_t original_length,
                      std::size_t hop_length = 512,
                      window_type win_type = window_type::hanning) {
        using T = typename std::decay_t<E>::value_type::value_type;
        auto stft_mat = xt::eval(stft_matrix.derived_cast());
        if (stft_mat.dimension() != 2)
            throw std::runtime_error("istft: input must be 2D STFT matrix.");
        std::size_t n_fft_half = stft_mat.shape()[1];
        std::size_t n_fft = (n_fft_half - 1) * 2;
        std::size_t win_len = n_fft; // default window length = n_fft

        auto win = window<T>(win_type, win_len);
        return detail::stft_inverse(stft_mat, win.data(), win_len,
                                     hop_length, original_length);
    }

    /**
     * Compute the magnitude spectrogram from the STFT.
     * @param stft_matrix Complex STFT.
     * @return Real 2D magnitude array.
     */
    template <class E>
    inline auto magnitude_spectrogram(const xexpression<E>& stft_matrix) {
        auto stft_mat = xt::eval(stft_matrix.derived_cast());
        return xt::abs(stft_mat);
    }

    /**
     * Compute the phase spectrogram (angle) from the STFT.
     * @param stft_matrix Complex STFT.
     * @return Real 2D phase array (radians).
     */
    template <class E>
    inline auto phase_spectrogram(const xexpression<E>& stft_matrix) {
        auto stft_mat = xt::eval(stft_matrix.derived_cast());
        return xt::arg(stft_mat);
    }

    /**
     * Compute the energy in each STFT frame.
     * @param stft_matrix Complex STFT.
     * @return 1D energy per frame.
     */
    template <class E>
    inline auto frame_energy(const xexpression<E>& stft_matrix) {
        using T = typename std::decay_t<E>::value_type;
        auto stft_mat = xt::eval(stft_matrix.derived_cast());
        auto sh = stft_mat.shape();
        std::size_t n_frames = sh[0];
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> energy({n_frames});
        const auto* src = stft_mat.data();
        for (std::size_t f = 0; f < n_frames; ++f) {
            T sum = 0;
            for (std::size_t k = 0; k < sh[1]; ++k) {
                T mag = std::abs(src[f * sh[1] + k]);
                sum += mag * mag;
            }
            energy[f] = sum;
        }
        return energy;
    }

} // namespace signal
} // namespace xt

#endif // XTENSOR_SIGNAL_XSTFT_HPP