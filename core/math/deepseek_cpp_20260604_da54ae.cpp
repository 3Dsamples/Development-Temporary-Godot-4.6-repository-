// system name : onetbb-warp
// File 0019 : core/math/signal.h
// Description : Signal processing: FFT, window functions, convolution, correlation for simulation.

#ifndef __TBB_WARP_CORE_MATH_SIGNAL_H
#define __TBB_WARP_CORE_MATH_SIGNAL_H

#include "core/math/constants.h"
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <cstdint>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Window functions (N‑point)
// ============================================================

inline std::vector<float> hanning_window(std::size_t n) {
    std::vector<float> w(n);
    float denom = static_cast<float>(n - 1);
    if (n == 1) { w[0] = 1.0f; return w; }
    for (std::size_t i = 0; i < n; ++i)
        w[i] = 0.5f * (1.0f - std::cos(TAU_F * i / denom));
    return w;
}

inline std::vector<float> hamming_window(std::size_t n) {
    std::vector<float> w(n);
    float denom = static_cast<float>(n - 1);
    if (n == 1) { w[0] = 1.0f; return w; }
    for (std::size_t i = 0; i < n; ++i)
        w[i] = 0.54f - 0.46f * std::cos(TAU_F * i / denom);
    return w;
}

inline std::vector<float> blackman_window(std::size_t n) {
    std::vector<float> w(n);
    float denom = static_cast<float>(n - 1);
    if (n == 1) { w[0] = 1.0f; return w; }
    for (std::size_t i = 0; i < n; ++i)
        w[i] = 0.42f - 0.5f * std::cos(TAU_F * i / denom) + 0.08f * std::cos(2.0f * TAU_F * i / denom);
    return w;
}

inline std::vector<float> blackman_harris_window(std::size_t n) {
    std::vector<float> w(n);
    float denom = static_cast<float>(n - 1);
    if (n == 1) { w[0] = 1.0f; return w; }
    for (std::size_t i = 0; i < n; ++i) {
        float a0 = 0.35875f, a1 = 0.48829f, a2 = 0.14128f, a3 = 0.01168f;
        w[i] = a0 - a1 * std::cos(TAU_F * i / denom) + a2 * std::cos(2.0f * TAU_F * i / denom)
               - a3 * std::cos(3.0f * TAU_F * i / denom);
    }
    return w;
}

inline std::vector<float> kaiser_window(std::size_t n, float beta = 6.0f) {
    std::vector<float> w(n);
    float denom = static_cast<float>(n - 1) * 0.5f;
    float i0_beta = std::cyl_bessel_i(0, beta); // C++17 math
    for (std::size_t i = 0; i < n; ++i) {
        float x = 2.0f * i / denom - 1.0f;
        float arg = beta * std::sqrt(1.0f - x*x);
        w[i] = std::cyl_bessel_i(0, arg) / i0_beta;
    }
    return w;
}

// ============================================================
// Forward FFT (Cooley‑Tukey radix‑2, in‑place)
// ============================================================

inline void fft(std::vector<std::complex<float>>& data) {
    std::size_t n = data.size();
    if (n <= 1) return;
    // Bit‑reversal permutation
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }
    // Radix‑2 stages
    for (std::size_t len = 2; len <= n; len <<= 1) {
        float angle = -TAU_F / len;
        std::complex<float> wlen(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (std::size_t j = 0; j < len/2; ++j) {
                std::complex<float> u = data[i + j];
                std::complex<float> v = data[i + j + len/2] * w;
                data[i + j] = u + v;
                data[i + j + len/2] = u - v;
                w *= wlen;
            }
        }
    }
}

inline std::vector<std::complex<float>> fft(const std::vector<float>& real) {
    std::vector<std::complex<float>> data(real.size());
    for (std::size_t i = 0; i < real.size(); ++i) data[i] = std::complex<float>(real[i], 0.0f);
    fft(data);
    return data;
}

// ============================================================
// Inverse FFT (in‑place)
// ============================================================

inline void ifft(std::vector<std::complex<float>>& data) {
    std::size_t n = data.size();
    for (auto& c : data) c = std::conj(c);
    fft(data);
    for (auto& c : data) c = std::conj(c) / static_cast<float>(n);
}

inline std::vector<std::complex<float>> ifft(const std::vector<std::complex<float>>& freq) {
    std::vector<std::complex<float>> data = freq;
    ifft(data);
    return data;
}

// ============================================================
// Real FFT of real data (returns complex spectrum)
// ============================================================

inline std::vector<std::complex<float>> real_fft(const std::vector<float>& real) {
    return fft(real);
}

// ============================================================
// Convolution (linear) via FFT
// ============================================================

inline std::vector<float> fft_convolution(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t n = 1;
    while (n < a.size() + b.size() - 1) n <<= 1;
    std::vector<std::complex<float>> A(n, {0.0f,0.0f}), B(n, {0.0f,0.0f});
    for (std::size_t i = 0; i < a.size(); ++i) A[i] = a[i];
    for (std::size_t i = 0; i < b.size(); ++i) B[i] = b[i];
    fft(A); fft(B);
    for (std::size_t i = 0; i < n; ++i) A[i] *= B[i];
    ifft(A);
    std::vector<float> result(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = A[i].real();
    return result;
}

// ============================================================
// Cross‑correlation via FFT
// ============================================================

inline std::vector<float> fft_correlation(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t n = 1;
    while (n < a.size() + b.size() - 1) n <<= 1;
    std::vector<std::complex<float>> A(n, {0.0f,0.0f}), B(n, {0.0f,0.0f});
    for (std::size_t i = 0; i < a.size(); ++i) A[i] = a[i];
    for (std::size_t i = 0; i < b.size(); ++i) B[i] = b[i];
    fft(A); fft(B);
    for (std::size_t i = 0; i < n; ++i) A[i] *= std::conj(B[i]);
    ifft(A);
    std::vector<float> result(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = A[i].real();
    return result;
}

// ============================================================
// Direct convolution (naive, for small sequences)
// ============================================================

inline std::vector<float> direct_convolution(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t len = a.size() + b.size() - 1;
    std::vector<float> result(len, 0.0f);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j)
            result[i+j] += a[i] * b[j];
    return result;
}

// ============================================================
// Direct cross‑correlation
// ============================================================

inline std::vector<float> direct_correlation(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t len = a.size() + b.size() - 1;
    std::vector<float> result(len, 0.0f);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j)
            result[i + b.size() - 1 - j] += a[i] * b[j]; // lag positive
    return result;
}

// ============================================================
// Autocorrelation (using FFT)
// ============================================================

inline std::vector<float> autocorrelation(const std::vector<float>& a) {
    std::size_t n = 1;
    while (n < 2 * a.size() - 1) n <<= 1;
    std::vector<std::complex<float>> A(n, {0.0f,0.0f});
    for (std::size_t i = 0; i < a.size(); ++i) A[i] = a[i];
    fft(A);
    for (std::size_t i = 0; i < n; ++i) A[i] *= std::conj(A[i]);
    ifft(A);
    std::vector<float> result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) result[i] = A[i].real();
    return result;
}

// ============================================================
// Spectrogram helper (STFT)
// ============================================================

inline std::vector<std::vector<float>> spectrogram(const std::vector<float>& signal,
                                                   std::size_t window_size,
                                                   std::size_t hop_size,
                                                   const std::vector<float>& window_func) {
    std::size_t num_frames = (signal.size() - window_size) / hop_size + 1;
    std::vector<std::vector<float>> spec(num_frames, std::vector<float>(window_size/2+1, 0.0f));
    std::vector<float> buffer(window_size);
    for (std::size_t f = 0; f < num_frames; ++f) {
        for (std::size_t i = 0; i < window_size; ++i)
            buffer[i] = signal[f * hop_size + i] * window_func[i];
        auto freq = real_fft(buffer);
        for (std::size_t k = 0; k < window_size/2+1; ++k)
            spec[f][k] = std::abs(freq[k]);
    }
    return spec;
}

// ============================================================
// Power spectrum from FFT
// ============================================================

inline std::vector<float> power_spectrum(const std::vector<std::complex<float>>& fft_data) {
    std::size_t n = fft_data.size() / 2 + 1;
    std::vector<float> psd(n);
    for (std::size_t i = 0; i < n; ++i)
        psd[i] = std::norm(fft_data[i]);
    return psd;
}

// ============================================================
// Goertzel algorithm for single‑frequency detection
// ============================================================

inline float goertzel(const std::vector<float>& signal, float target_freq, float sample_rate) {
    std::size_t n = signal.size();
    float omega = TAU_F * target_freq / sample_rate;
    float coeff = 2.0f * std::cos(omega);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        s0 = signal[i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    float real = s1 - s2 * std::cos(omega);
    float imag = s2 * std::sin(omega);
    return std::sqrt(real*real + imag*imag);
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_SIGNAL_H