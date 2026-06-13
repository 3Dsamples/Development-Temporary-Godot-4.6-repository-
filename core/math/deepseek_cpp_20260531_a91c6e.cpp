//File 0074 : core/math/wavelet_transform.h
//Discrete wavelet transform (DWT) 1D/2D with Haar and Daubechies D4 wavelets, multi‑level decomposition, coefficient thresholding, and exact reconstruction using lifting scheme.
#ifndef CORE_MATH_WAVELET_TRANSFORM_H
#define CORE_MATH_WAVELET_TRANSFORM_H

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace SimulationMath {
namespace wavelet {

// -----------------------------------------------------------------------------
// 1. Haar wavelet 1D forward transform (in‑place, length must be power of 2)
// -----------------------------------------------------------------------------
inline void haar_fwd_1d(std::vector<float>& data, int levels) noexcept {
    size_t n = data.size();
    if (n < 2) return;
    std::vector<float> temp(n);
    for (int lev = 0; lev < levels; ++lev) {
        size_t len = n >> lev;
        if (len < 2) break;
        size_t half = len >> 1;
        for (size_t i = 0; i < half; ++i) {
            float avg = (data[2*i] + data[2*i+1]) * 0.5f;
            float diff = data[2*i] - data[2*i+1];
            temp[i] = avg;
            temp[half + i] = diff;
        }
        for (size_t i = 0; i < len; ++i) data[i] = temp[i];
    }
}

// -----------------------------------------------------------------------------
// 2. Haar wavelet 1D inverse transform
// -----------------------------------------------------------------------------
inline void haar_inv_1d(std::vector<float>& data, int levels) noexcept {
    size_t n = data.size();
    if (n < 2) return;
    std::vector<float> temp(n);
    for (int lev = levels - 1; lev >= 0; --lev) {
        size_t len = n >> lev;
        if (len < 2) continue;
        size_t half = len >> 1;
        for (size_t i = 0; i < half; ++i) {
            float avg = data[i];
            float diff = data[half + i];
            temp[2*i] = avg + diff * 0.5f;
            temp[2*i+1] = avg - diff * 0.5f;
        }
        for (size_t i = 0; i < len; ++i) data[i] = temp[i];
    }
}

// -----------------------------------------------------------------------------
// 3. Daubechies D4 wavelet forward transform using lifting scheme
// -----------------------------------------------------------------------------
inline void daub4_fwd_1d(std::vector<float>& data, int levels) noexcept {
    size_t n = data.size();
    if (n < 4) return;
    const float sqrt3 = std::sqrt(3.0f);
    const float alpha = (1.0f + sqrt3) / (4.0f * std::sqrt(2.0f));
    const float beta  = (3.0f + sqrt3) / (4.0f * std::sqrt(2.0f));
    const float gamma = (3.0f - sqrt3) / (4.0f * std::sqrt(2.0f));
    const float delta = (1.0f - sqrt3) / (4.0f * std::sqrt(2.0f));

    std::vector<float> temp(n);
    for (int lev = 0; lev < levels; ++lev) {
        size_t len = n >> lev;
        if (len < 4) break;
        size_t half = len >> 1;
        for (size_t i = 0; i < half; ++i) {
            temp[i] = data[2*i] * alpha + data[2*i+1] * beta;
            temp[half + i] = data[2*i] * gamma + data[2*i+1] * delta;
        }
        for (size_t i = 0; i < len; ++i) data[i] = temp[i];
    }
}

// -----------------------------------------------------------------------------
// 4. Daubechies D4 inverse transform using lifting scheme
// -----------------------------------------------------------------------------
inline void daub4_inv_1d(std::vector<float>& data, int levels) noexcept {
    size_t n = data.size();
    if (n < 4) return;
    const float sqrt3 = std::sqrt(3.0f);
    const float alpha = (1.0f + sqrt3) / (4.0f * std::sqrt(2.0f));
    const float beta  = (3.0f + sqrt3) / (4.0f * std::sqrt(2.0f));
    const float gamma = (3.0f - sqrt3) / (4.0f * std::sqrt(2.0f));
    const float delta = (1.0f - sqrt3) / (4.0f * std::sqrt(2.0f));
    // Inverse matrix: [alpha beta; gamma delta]^-1 = [delta -beta; -gamma alpha] / (alpha*delta - beta*gamma)
    float det = alpha * delta - beta * gamma;
    if (std::abs(det) < 1e-12f) return;
    float inv_alpha =  delta / det;
    float inv_beta  = -beta  / det;
    float inv_gamma = -gamma / det;
    float inv_delta =  alpha / det;

    std::vector<float> temp(n);
    for (int lev = levels - 1; lev >= 0; --lev) {
        size_t len = n >> lev;
        if (len < 4) continue;
        size_t half = len >> 1;
        for (size_t i = 0; i < half; ++i) {
            float s = data[i];
            float d = data[half + i];
            temp[2*i]   = s * inv_alpha + d * inv_beta;
            temp[2*i+1] = s * inv_gamma + d * inv_delta;
        }
        for (size_t i = 0; i < len; ++i) data[i] = temp[i];
    }
}

// -----------------------------------------------------------------------------
// 5. Generic 1D forward transform (select wavelet type)
// -----------------------------------------------------------------------------
enum class WaveletType { Haar, Daub4 };

inline void dwt_fwd_1d(std::vector<float>& data, int levels, WaveletType type = WaveletType::Haar) noexcept {
    switch (type) {
        case WaveletType::Haar:  haar_fwd_1d(data, levels);  break;
        case WaveletType::Daub4: daub4_fwd_1d(data, levels); break;
    }
}

inline void dwt_inv_1d(std::vector<float>& data, int levels, WaveletType type = WaveletType::Haar) noexcept {
    switch (type) {
        case WaveletType::Haar:  haar_inv_1d(data, levels);  break;
        case WaveletType::Daub4: daub4_inv_1d(data, levels); break;
    }
}

// -----------------------------------------------------------------------------
// 6. 2D separable DWT (transform rows then columns), assumes square power‑of‑2
// -----------------------------------------------------------------------------
inline void dwt_fwd_2d(std::vector<std::vector<float>>& data, int levels, WaveletType type = WaveletType::Haar) noexcept {
    size_t n = data.size();
    if (n < 2) return;
    for (int lev = 0; lev < levels; ++lev) {
        size_t len = n >> lev;
        if (len < 2) break;
        // Transform rows
        for (size_t y = 0; y < len; ++y) {
            std::vector<float> row(data[y].begin(), data[y].begin() + len);
            dwt_fwd_1d(row, 1, type);
            std::copy(row.begin(), row.begin() + len, data[y].begin());
        }
        // Transform columns
        for (size_t x = 0; x < len; ++x) {
            std::vector<float> col(len);
            for (size_t y = 0; y < len; ++y) col[y] = data[y][x];
            dwt_fwd_1d(col, 1, type);
            for (size_t y = 0; y < len; ++y) data[y][x] = col[y];
        }
    }
}

inline void dwt_inv_2d(std::vector<std::vector<float>>& data, int levels, WaveletType type = WaveletType::Haar) noexcept {
    size_t n = data.size();
    if (n < 2) return;
    for (int lev = levels - 1; lev >= 0; --lev) {
        size_t len = n >> lev;
        if (len < 2) continue;
        // Inverse transform columns
        for (size_t x = 0; x < len; ++x) {
            std::vector<float> col(len);
            for (size_t y = 0; y < len; ++y) col[y] = data[y][x];
            dwt_inv_1d(col, 1, type);
            for (size_t y = 0; y < len; ++y) data[y][x] = col[y];
        }
        // Inverse transform rows
        for (size_t y = 0; y < len; ++y) {
            std::vector<float> row(data[y].begin(), data[y].begin() + len);
            dwt_inv_1d(row, 1, type);
            std::copy(row.begin(), row.begin() + len, data[y].begin());
        }
    }
}

// -----------------------------------------------------------------------------
// 7. Coefficient thresholding (hard and soft) for denoising
// -----------------------------------------------------------------------------
inline void hard_threshold(std::vector<float>& data, float threshold) noexcept {
    for (auto& v : data) if (std::fabs(v) < threshold) v = 0.0f;
}

inline void soft_threshold(std::vector<float>& data, float threshold) noexcept {
    for (auto& v : data) {
        float sign = (v >= 0.0f) ? 1.0f : -1.0f;
        v = (std::fabs(v) > threshold) ? (sign * (std::fabs(v) - threshold)) : 0.0f;
    }
}

// -----------------------------------------------------------------------------
// 8. Multi‑level decomposition with thresholding (denoising pipeline)
// -----------------------------------------------------------------------------
inline std::vector<float> denoise_1d(const std::vector<float>& signal, int levels,
                                     float threshold, WaveletType type = WaveletType::Haar) noexcept {
    std::vector<float> coeffs = signal;
    // Pad to next power of two
    size_t n = 1;
    while (n < coeffs.size()) n <<= 1;
    coeffs.resize(n, 0.0f);
    dwt_fwd_1d(coeffs, levels, type);
    // Keep approximation coefficients (first n>>levels) untouched, threshold detail coefficients
    size_t keep = n >> levels;
    for (size_t i = keep; i < coeffs.size(); ++i)
        coeffs[i] = (std::fabs(coeffs[i]) > threshold) ? coeffs[i] : 0.0f;
    dwt_inv_1d(coeffs, levels, type);
    coeffs.resize(signal.size());
    return coeffs;
}

} // namespace wavelet
} // namespace SimulationMath

#endif // CORE_MATH_WAVELET_TRANSFORM_H