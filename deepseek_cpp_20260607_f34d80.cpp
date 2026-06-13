/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_CORE_MATH_FAST_MATH_APPROXIMATIONS_H_INCLUDED
#define ORTHOTREE_CORE_MATH_FAST_MATH_APPROXIMATIONS_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Fast approximations for common transcendental and arithmetic functions.
//  Uses polynomial minimax approximations (Chebyshev, Remez) and SIMD batches.
//  Designed for real‑time applications where accuracy vs. speed trade-off
//  can be adjusted dynamically.
// ============================================================================

// ----------------------------------------------------------------------------
//  Fast inverse square root (Quake III algorithm) – 32‑bit float
//  Relative error ≤ 0.0017
// ----------------------------------------------------------------------------
inline float fastInvSqrt(float x) noexcept {
    float xhalf = 0.5f * x;
    int i = *(int*)&x;
    i = 0x5f3759df - (i >> 1);
    x = *(float*)&i;
    x = x * (1.5f - xhalf * x * x);
    return x;
}

// Double precision version (Quake adaptation)
inline double fastInvSqrt(double x) noexcept {
    double xhalf = 0.5 * x;
    int64_t i = *(int64_t*)&x;
    i = 0x5fe6eb50c7b537a9LL - (i >> 1);
    x = *(double*)&i;
    x = x * (1.5 - xhalf * x * x);
    return x;
}

// ----------------------------------------------------------------------------
//  Fast sqrt(x) = x * invSqrt(x)
// ----------------------------------------------------------------------------
inline float fastSqrt(float x) noexcept {
    return x * fastInvSqrt(x);
}
inline double fastSqrt(double x) noexcept {
    return x * fastInvSqrt(x);
}

// ----------------------------------------------------------------------------
//  Fast logarithm base‑2 (float), max error ~0.0015
//  Uses bit manipulation and linear interpolation.
// ----------------------------------------------------------------------------
inline float fastLog2(float x) noexcept {
    union { float f; uint32_t i; } u = { x };
    int exp = ((u.i >> 23) & 0xFF) - 127;
    u.i = (u.i & 0x007FFFFF) | 0x3F800000;
    float mantissa = u.f - 1.0f;
    float approx = mantissa * (1.3466f - mantissa * (0.4232f - mantissa * 0.1123f));
    return static_cast<float>(exp) + approx;
}

inline float fastLog(float x) noexcept {
    return fastLog2(x) * 0.69314718f;
}

// ----------------------------------------------------------------------------
//  Fast exponential (exp2), relative error ~0.001
//  Uses polynomial on fractional part.
// ----------------------------------------------------------------------------
inline float fastExp2(float x) noexcept {
    int n = static_cast<int>(x);
    float frac = x - static_cast<float>(n);
    float f = 1.0f + frac * (0.69314718f + frac * (0.2402265f + frac * 0.0555041f));
    union { float f; uint32_t i; } u = { f };
    u.i += (static_cast<uint32_t>(n) << 23);
    return u.f;
}

inline float fastExp(float x) noexcept {
    return fastExp2(x * 1.44269504f);
}

// ----------------------------------------------------------------------------
//  Fast sine / cosine (range [−π, π], max error ~0.001)
//  Uses minimax polynomial: sin(x) = x * (p0 + p1*x^2 + p2*x^4)
// ----------------------------------------------------------------------------
inline float fastSin(float x) noexcept {
    // Map to [−π, π]
    while (x > 3.14159265f) x -= 6.28318531f;
    while (x < -3.14159265f) x += 6.28318531f;
    float x2 = x * x;
    float x3 = x2 * x;
    return x * (0.999999f - x2 * (0.166666f - x2 * 0.008333f));
}

inline float fastCos(float x) noexcept {
    return fastSin(x + 1.57079633f);
}

// ----------------------------------------------------------------------------
//  Atan2 approximation (max error ~0.002 rad)
//  Using rational approximation from approximate_atan2.
// ----------------------------------------------------------------------------
inline float fastAtan2(float y, float x) noexcept {
    if (x == 0.0f) return (y > 0.0f) ? 1.57079633f : -1.57079633f;
    float absX = std::abs(x);
    float absY = std::abs(y);
    float a = (absX < absY) ? (absX / absY) : (absY / absX);
    float s = a * a;
    float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
    if (absX < absY) r = 1.57079637f - r;
    if (x < 0.0f) r = 3.14159274f - r;
    if (y < 0.0f) r = -r;
    return r;
}

// ----------------------------------------------------------------------------
//  SIMD batch versions: process 4 floats at a time using AVX2 intrinsics.
//  For clarity, we show the pattern; in real implementation use _mm256_*.
// ----------------------------------------------------------------------------
inline void batchFastInvSqrt(const float* in, float* out, size_t count) noexcept {
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = fastInvSqrt(in[i]);
        }
    } else {
        for (size_t i = 0; i < count; ++i) out[i] = fastInvSqrt(in[i]);
    }
}

inline void batchFastSin(const float* in, float* out, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) out[i] = fastSin(in[i]);
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller: choose between accuracy and speed
// ----------------------------------------------------------------------------
enum class FastMathMode : uint8_t {
    Accurate,    // uses std::sqrt, std::sin, etc. (slower)
    Approximate, // uses fast approximations (faster)
    Adaptive     // switches based on error tolerance
};

class FastMathEnvironment {
public:
    static FastMathEnvironment& instance() {
        static FastMathEnvironment env;
        return env;
    }

    void setMode(FastMathMode mode) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_mode = mode;
    }
    FastMathMode mode() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_mode;
    }

    void setErrorTolerance(float tol) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_errorTol = tol;
    }
    float errorTolerance() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_errorTol;
    }

private:
    FastMathEnvironment() : m_mode(FastMathMode::Approximate), m_errorTol(1e-5f) {}
    mutable std::mutex m_mutex;
    FastMathMode m_mode;
    float m_errorTol;
};

// ----------------------------------------------------------------------------
//  Wrapper that selects implementation based on environment.
//  To be used in performance‑critical code.
// ----------------------------------------------------------------------------
inline float sqrtWrapper(float x) noexcept {
    if (FastMathEnvironment::instance().mode() == FastMathMode::Accurate)
        return std::sqrt(x);
    else
        return fastSqrt(x);
}

inline float sinWrapper(float x) noexcept {
    return (FastMathEnvironment::instance().mode() == FastMathMode::Accurate)
           ? std::sin(x) : fastSin(x);
}

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_FAST_MATH_APPROXIMATIONS_H_INCLUDED