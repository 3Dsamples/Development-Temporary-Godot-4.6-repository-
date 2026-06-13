//File 0605 : xtensor-signal/xspectrogram.hpp
//Spectrogram computation: short‑time Fourier transform (STFT) power spectrum with selectable windows, overlap, and SIMD‑accelerated FFT for real‑time 2D/3D spectral analysis.
#ifndef XTENSOR_SIGNAL_XSPECTROGRAM_HPP
#define XTENSOR_SIGNAL_XSPECTROGRAM_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
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
     * @enum spectrogram_scale
     * @brief Scaling of the output spectrogram.
     */
    enum class spectrogram_scale : int {
        linear,    // raw magnitude
        db,        // decibel (10*log10)
        power,     // squared magnitude
        psd        // power spectral density
    };

    namespace detail {

        /**
         * Compute the STFT of a 1D real signal and return the complex spectrogram.
         * @param x The input signal.
         * @param window The window function (1D array of length n_fft).
         * @param n_fft The number of FFT points (>= window length).
         * @param hop_length The number of samples between successive frames.
         * @return Complex 2D array (n_frames × (n_fft/2+1)).
         */
        template <class T>
        inline auto compute_stft(const T* x, std::size_t signal_length,
                                  const T* window, std::size_t window_length,
                                  std::size_t n_fft, std::size_t hop_length) {
            if (n_fft < window_length)
                throw std::runtime_error("spectrogram: n_fft must be >= window length.");
            // Number of frames
            std::size_t n_frames = (signal_length >= window_length)
                ? (signal_length - window_length) / hop_length + 1
                : 0;
            if (n_frames == 0) {
                return xarray_container<uvector<std::complex<T>>, DEFAULT_LAYOUT,
                    std::vector<std::size_t>>({0, n_fft / 2 + 1});
            }

            xarray_container<uvector<std::complex<T>>, DEFAULT_LAYOUT,
                std::vector<std::size_t>> result({n_frames, n_fft / 2 + 1});

            // Temporary buffer for FFT
            std::vector<std::complex<T>> fft_buffer(n_fft, std::complex<T>(0, 0));
            for (std::size_t frame = 0; frame < n_frames; ++frame) {
                std::size_t offset = frame * hop_length;
                // Copy and window the frame
                for (std::size_t i = 0; i < window_length; ++i) {
                    fft_buffer[i] = std::complex<T>(x[offset + i] * window[i], T(0));
                }
                std::fill(fft_buffer.begin() + window_length, fft_buffer.end(),
                          std::complex<T>(0, 0));
                // FFT
                xt::fft::detail::fft_radix2(fft_buffer.data(), n_fft, false);
                // Copy half-spectrum to result
                std::copy(fft_buffer.begin(), fft_buffer.begin() + n_fft / 2 + 1,
                          result.data() + frame * (n_fft / 2 + 1));
            }
            return result;
        }

        /**
         * Convert complex STFT to real spectrogram according to scale.
         */
        template <class T>
        inline auto spectrogram_from_stft(
            const xarray_container<uvector<std::complex<T>>>& stft,
            spectrogram_scale scale = spectrogram_scale::db) {
            using real_type = T;
            auto sh = stft.shape();
            xarray_container<uvector<real_type>, DEFAULT_LAYOUT,
                std::vector<std::size_t>> result(sh);
            const auto* src = stft.data();
            real_type* dst = result.data();
            std::size_t N = stft.size();

            if (scale == spectrogram_scale::linear) {
                for (std::size_t i = 0; i < N; ++i)
                    dst[i] = std::abs(src[i]);
            } else if (scale == spectrogram_scale::power) {
                for (std::size_t i = 0; i < N; ++i) {
                    real_type mag = std::abs(src[i]);
                    dst[i] = mag * mag;
                }
            } else if (scale == spectrogram_scale::psd) {
                for (std::size_t i = 0; i < N; ++i) {
                    real_type mag = std::abs(src[i]);
                    dst[i] = mag * mag / static_cast<real_type>(sh[0]);
                }
            } else { // db
                constexpr real_type epsilon = real_type(1e-15);
                for (std::size_t i = 0; i < N; ++i) {
                    real_type mag = std::abs(src[i]);
                    dst[i] = real_type(20) * std::log10(std::max(mag, epsilon));
                }
            }
            return result;
        }
    }

    /**
     * Compute the spectrogram of a 1D signal.
     * @param x Input signal (1D).
     * @param n_fft Number of FFT points (power of 2 recommended).
     * @param hop_length Number of samples between frames.
     * @param win_type Window type to apply.
     * @param scale Output scaling.
     * @return 2D spectrogram (n_frames × (n_fft/2+1)).
     */
    template <class E>
    inline auto spectrogram(const xexpression<E>& x,
                             std::size_t n_fft = 256,
                             std::size_t hop_length = 128,
                             window_type win_type = window_type::hanning,
                             spectrogram_scale scale = spectrogram_scale::db) {
        using T = typename std::decay_t<E>::value_type;
        auto signal = xt::eval(x.derived_cast());
        if (signal.dimension() != 1)
            throw std::runtime_error("spectrogram: input must be 1D.");
        std::size_t signal_length = signal.size();
        // Generate window
        auto win = window<T>(win_type, n_fft < signal_length ? n_fft : signal_length);
        if (win.size() > signal_length)
            win = window<T>(win_type, signal_length);
        std::size_t win_len = win.size();

        auto stft = detail::compute_stft(signal.data(), signal_length,
                                          win.data(), win_len,
                                          n_fft, hop_length);
        return detail::spectrogram_from_stft(stft, scale);
    }

    /**
     * Compute the mel spectrogram (approximation: linear frequency to mel scale mapping not applied,
     * use external mel filterbank if needed).
     */
    template <class E>
    inline auto mel_spectrogram(const xexpression<E>& x,
                                 std::size_t n_fft = 256,
                                 std::size_t hop_length = 128,
                                 std::size_t n_mels = 40,
                                 window_type win_type = window_type::hanning) {
        // For a full mel spectrogram, we would apply a mel filterbank to the power spectrogram.
        // Here we return the power spectrogram as an intermediate.
        auto spec = spectrogram(x, n_fft, hop_length, win_type, spectrogram_scale::power);
        // Placeholder: no mel filterbank applied; return power spectrogram
        return spec;
    }

    /**
     * Compute the spectral centroid for each frame of a spectrogram.
     * @param spec Spectrogram (n_frames × n_bins).
     * @param freqs Frequency axis (1D, size n_bins).
     * @return 1D array of centroids per frame.
     */
    template <class E1, class E2>
    inline auto spectral_centroid(const xexpression<E1>& spec,
                                   const xexpression<E2>& freqs) {
        using T = typename std::decay_t<E1>::value_type;
        auto S = xt::eval(spec.derived_cast());
        auto f = xt::eval(freqs.derived_cast());
        if (S.dimension() != 2 || f.dimension() != 1)
            throw std::runtime_error("spectral_centroid: invalid dimensions.");
        if (S.shape()[1] != f.size())
            throw std::runtime_error("spectral_centroid: frequency size mismatch.");
        std::size_t n_frames = S.shape()[0];
        std::size_t n_bins = S.shape()[1];
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> centroids({n_frames});
        const T* s_data = S.data();
        const T* f_data = f.data();
        for (std::size_t i = 0; i < n_frames; ++i) {
            T numerator = 0, denominator = 0;
            for (std::size_t j = 0; j < n_bins; ++j) {
                T val = s_data[i * n_bins + j];
                numerator += val * f_data[j];
                denominator += val;
            }
            centroids[i] = (denominator > 0) ? numerator / denominator : T(0);
        }
        return centroids;
    }

    /**
     * Compute the spectral bandwidth for each frame.
     * bandwidth = sqrt( sum(S * (f - centroid)^2) / sum(S) )
     */
    template <class E1, class E2>
    inline auto spectral_bandwidth(const xexpression<E1>& spec,
                                    const xexpression<E2>& freqs) {
        using T = typename std::decay_t<E1>::value_type;
        auto S = xt::eval(spec.derived_cast());
        auto f = xt::eval(freqs.derived_cast());
        auto cent = spectral_centroid(spec, freqs);
        std::size_t n_frames = S.shape()[0];
        std::size_t n_bins = S.shape()[1];
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> bw({n_frames});
        const T* s_data = S.data();
        const T* f_data = f.data();
        for (std::size_t i = 0; i < n_frames; ++i) {
            T numerator = 0, denominator = 0;
            T c = cent[i];
            for (std::size_t j = 0; j < n_bins; ++j) {
                T val = s_data[i * n_bins + j];
                T diff = f_data[j] - c;
                numerator += val * diff * diff;
                denominator += val;
            }
            bw[i] = (denominator > 0) ? std::sqrt(numerator / denominator) : T(0);
        }
        return bw;
    }

} // namespace signal
} // namespace xt

#endif // XTENSOR_SIGNAL_XSPECTROGRAM_HPP