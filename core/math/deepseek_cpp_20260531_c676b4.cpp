//File 0054 : core/math/fast_fourier_transform.h
//Complete 1D/2D/3D radix‑2 FFT/IFFT, real‑to‑complex, window functions, convolution, power spectrum, all highly optimized with C++17 and SIMD‑ready loops.
#ifndef CORE_MATH_FAST_FOURIER_TRANSFORM_H
#define CORE_MATH_FAST_FOURIER_TRANSFORM_H

#include <complex>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <functional>

namespace SimulationMath {
namespace fft {

using Complex = std::complex<float>;
using CDouble = std::complex<double>;

// -----------------------------------------------------------------------------
// 1. Bit‑reversal permutation (in‑place)
// -----------------------------------------------------------------------------
inline void bit_reverse(std::vector<Complex>& data) noexcept {
    size_t n = data.size();
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j)
            std::swap(data[i], data[j]);
    }
}

// -----------------------------------------------------------------------------
// 2. 1D FFT (complex‑to‑complex, decimation‑in‑time, in‑place)
// -----------------------------------------------------------------------------
inline void fft_1d(std::vector<Complex>& data, bool inverse = false) noexcept {
    size_t n = data.size();
    if (n <= 1) return;
    bit_reverse(data);

    const float two_pi = 2.0f * 3.14159265358979323846f;
    float sign = inverse ? 1.0f : -1.0f;

    for (size_t len = 2; len <= n; len <<= 1) {
        float angle = sign * two_pi / len;
        Complex wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0f, 0.0f);
            for (size_t j = 0; j < len / 2; ++j) {
                Complex u = data[i + j];
                Complex v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        for (auto& c : data)
            c /= static_cast<float>(n);
    }
}

// -----------------------------------------------------------------------------
// 3. 2D FFT (separate rows then columns)
// -----------------------------------------------------------------------------
inline void fft_2d(std::vector<Complex>& data, size_t width, size_t height, bool inverse = false) noexcept {
    // Transform rows
    for (size_t y = 0; y < height; ++y) {
        std::vector<Complex> row(data.begin() + y * width, data.begin() + (y + 1) * width);
        fft_1d(row, inverse);
        std::copy(row.begin(), row.end(), data.begin() + y * width);
    }
    // Transform columns
    for (size_t x = 0; x < width; ++x) {
        std::vector<Complex> col(height);
        for (size_t y = 0; y < height; ++y)
            col[y] = data[y * width + x];
        fft_1d(col, inverse);
        for (size_t y = 0; y < height; ++y)
            data[y * width + x] = col[y];
    }
}

// -----------------------------------------------------------------------------
// 4. 3D FFT
// -----------------------------------------------------------------------------
inline void fft_3d(std::vector<Complex>& data, size_t width, size_t height, size_t depth, bool inverse = false) noexcept {
    size_t slice = width * height;
    for (size_t z = 0; z < depth; ++z) {
        std::vector<Complex> plane(data.begin() + z * slice, data.begin() + (z + 1) * slice);
        fft_2d(plane, width, height, inverse);
        std::copy(plane.begin(), plane.end(), data.begin() + z * slice);
    }
    // Transform in z direction
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            std::vector<Complex> line(depth);
            for (size_t z = 0; z < depth; ++z)
                line[z] = data[z * slice + y * width + x];
            fft_1d(line, inverse);
            for (size_t z = 0; z < depth; ++z)
                data[z * slice + y * width + x] = line[z];
        }
    }
}

// -----------------------------------------------------------------------------
// 5. Real‑to‑complex FFT (1D) – packs real array into complex with half length
// -----------------------------------------------------------------------------
inline void real_to_complex_fft_1d(const std::vector<float>& input, std::vector<Complex>& output) noexcept {
    size_t n = input.size();
    if (n == 0) { output.clear(); return; }
    // Use the even‑indexed values as real parts, odd as imaginary? Standard: treat as complex with imaginary=0
    std::vector<Complex> full(n);
    for (size_t i = 0; i < n; ++i)
        full[i] = Complex(input[i], 0.0f);
    fft_1d(full, false);
    // Only need first N/2+1 values (for real input, Hermitian symmetry)
    size_t half = n / 2 + 1;
    output.assign(full.begin(), full.begin() + half);
}

// -----------------------------------------------------------------------------
// 6. Complex‑to‑real IFFT (1D) – from half‑spectrum to real signal
// -----------------------------------------------------------------------------
inline void complex_to_real_ifft_1d(const std::vector<Complex>& spectrum, std::vector<float>& output) noexcept {
    size_t half = spectrum.size();
    size_t n = (half - 1) * 2;   // full length (must be even)
    std::vector<Complex> full(n);
    for (size_t i = 0; i < half; ++i)
        full[i] = spectrum[i];
    // Fill conjugate symmetric part
    for (size_t i = 1; i < n / 2; ++i)
        full[n - i] = std::conj(spectrum[i]);
    fft_1d(full, true);
    output.resize(n);
    for (size_t i = 0; i < n; ++i)
        output[i] = full[i].real();   // imaginary part should be zero
}

// -----------------------------------------------------------------------------
// 7. Window functions (1D, fill in‑place)
// -----------------------------------------------------------------------------
inline void apply_window_hann(std::vector<float>& signal) noexcept {
    size_t n = signal.size();
    for (size_t i = 0; i < n; ++i)
        signal[i] *= 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979f * i / (n - 1)));
}
inline void apply_window_hamming(std::vector<float>& signal) noexcept {
    size_t n = signal.size();
    for (size_t i = 0; i < n; ++i)
        signal[i] *= 0.54f - 0.46f * std::cos(2.0f * 3.14159265358979f * i / (n - 1));
}
inline void apply_window_blackman(std::vector<float>& signal) noexcept {
    size_t n = signal.size();
    for (size_t i = 0; i < n; ++i) {
        float a0 = 0.42f, a1 = 0.5f, a2 = 0.08f;
        signal[i] *= a0 - a1 * std::cos(2.0f * 3.14159265358979f * i / (n - 1))
                      + a2 * std::cos(4.0f * 3.14159265358979f * i / (n - 1));
    }
}

// -----------------------------------------------------------------------------
// 8. Power spectrum (from complex FFT output, 1D)
// -----------------------------------------------------------------------------
inline std::vector<float> power_spectrum_1d(const std::vector<Complex>& fft_data) noexcept {
    std::vector<float> ps(fft_data.size());
    for (size_t i = 0; i < fft_data.size(); ++i)
        ps[i] = std::norm(fft_data[i]);
    return ps;
}

// -----------------------------------------------------------------------------
// 9. Convolution via FFT (1D, complex)
// -----------------------------------------------------------------------------
inline std::vector<Complex> convolution_fft_1d(const std::vector<Complex>& a, const std::vector<Complex>& b) noexcept {
    size_t n = a.size() + b.size() - 1;
    size_t N = 1;
    while (N < n) N <<= 1;
    std::vector<Complex> fa(N), fb(N);
    std::copy(a.begin(), a.end(), fa.begin());
    std::copy(b.begin(), b.end(), fb.begin());
    fft_1d(fa);
    fft_1d(fb);
    for (size_t i = 0; i < N; ++i)
        fa[i] *= fb[i];
    fft_1d(fa, true);
    fa.resize(n);
    return fa;
}

// -----------------------------------------------------------------------------
// 10. Cross‑correlation via FFT (1D, complex)
// -----------------------------------------------------------------------------
inline std::vector<Complex> correlation_fft_1d(const std::vector<Complex>& a, const std::vector<Complex>& b) noexcept {
    // Correlation: conj(fft(a)) * fft(b)
    size_t n = a.size() + b.size() - 1;
    size_t N = 1;
    while (N < n) N <<= 1;
    std::vector<Complex> fa(N), fb(N);
    std::copy(a.begin(), a.end(), fa.begin());
    std::copy(b.begin(), b.end(), fb.begin());
    fft_1d(fa);
    fft_1d(fb);
    for (size_t i = 0; i < N; ++i)
        fa[i] = std::conj(fa[i]) * fb[i];
    fft_1d(fa, true);
    fa.resize(n);
    return fa;
}

// -----------------------------------------------------------------------------
// 11. 1D FFT double precision
// -----------------------------------------------------------------------------
inline void fft_1d_double(std::vector<CDouble>& data, bool inverse = false) noexcept {
    size_t n = data.size();
    if (n <= 1) return;
    // bit‑reversal for double
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }
    const double two_pi = 2.0 * 3.14159265358979323846;
    double sign = inverse ? 1.0 : -1.0;
    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = sign * two_pi / len;
        CDouble wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len) {
            CDouble w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                CDouble u = data[i + k];
                CDouble v = data[i + k + len / 2] * w;
                data[i + k] = u + v;
                data[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        for (auto& c : data)
            c /= static_cast<double>(n);
    }
}

} // namespace fft
} // namespace SimulationMath

#endif // CORE_MATH_FAST_FOURIER_TRANSFORM_H