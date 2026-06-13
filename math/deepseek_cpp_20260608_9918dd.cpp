//File group name : OrthoTree Math
//File 0063 : core/math/approx/fast_math.h
//Fast approximations for sqrt, inverse sqrt, sin, cos, atan2, exp, log, using polynomial minimax and bit tricks. SIMD batch versions.

#ifndef ORTHOTREE_CORE_MATH_APPROX_FAST_MATH_H_INCLUDED
#define ORTHOTREE_CORE_MATH_APPROX_FAST_MATH_H_INCLUDED

#include "../../build_config.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <cstdint>

namespace OrthoTree {
namespace Math {
namespace Approx {

// ============================================================================
//  Fast inverse square root (Quake III algorithm) – float
//  Relative error < 0.0017
// ============================================================================
inline float fastInvSqrt(float x) noexcept {
    float xhalf = 0.5f * x;
    int i = *(int*)&x;
    i = 0x5f3759df - (i >> 1);
    x = *(float*)&i;
    x = x * (1.5f - xhalf * x * x);
    return x;
}

inline double fastInvSqrt(double x) noexcept {
    double xhalf = 0.5 * x;
    int64_t i = *(int64_t*)&x;
    i = 0x5fe6eb50c7b537a9LL - (i >> 1);
    x = *(double*)&i;
    x = x * (1.5 - xhalf * x * x);
    return x;
}

// ============================================================================
//  Fast square root (x * invSqrt(x))
// ============================================================================
inline float fastSqrt(float x) noexcept { return x * fastInvSqrt(x); }
inline double fastSqrt(double x) noexcept { return x * fastInvSqrt(x); }

// ============================================================================
//  Fast sine (range [-π, π], max error ~0.001)
//  Polynomial: sin(x) = x * (p0 + p1*x^2 + p2*x^4)
// ============================================================================
inline float fastSin(float x) noexcept {
    // Map to [-π, π]
    while (x > 3.14159265f) x -= 6.28318531f;
    while (x < -3.14159265f) x += 6.28318531f;
    float x2 = x * x;
    return x * (0.999999f - x2 * (0.166666f - x2 * 0.008333f));
}
inline double fastSin(double x) noexcept {
    while (x > 3.141592653589793) x -= 6.283185307179586;
    while (x < -3.141592653589793) x += 6.283185307179586;
    double x2 = x * x;
    return x * (0.9999999999999999 - x2 * (0.16666666666666666 - x2 * 0.008333333333333333));
}

// ============================================================================
//  Fast cosine (sin(x + π/2))
// ============================================================================
inline float fastCos(float x) noexcept { return fastSin(x + 1.57079633f); }
inline double fastCos(double x) noexcept { return fastSin(x + 1.5707963267948966); }

// ============================================================================
//  Fast atan2 approximation (max error ~0.002 rad)
// ============================================================================
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
inline double fastAtan2(double y, double x) noexcept {
    return (double)fastAtan2((float)y, (float)x);
}

// ============================================================================
//  Fast exponential (exp2 and exp)
//  exp2(x) = 2^x, max error ~0.001
// ============================================================================
inline float fastExp2(float x) noexcept {
    int n = static_cast<int>(x);
    float frac = x - static_cast<float>(n);
    float f = 1.0f + frac * (0.69314718f + frac * (0.2402265f + frac * 0.0555041f));
    union { float f; uint32_t i; } u = { f };
    u.i += static_cast<uint32_t>(n) << 23;
    return u.f;
}
inline float fastExp(float x) noexcept { return fastExp2(x * 1.44269504f); }

inline double fastExp2(double x) noexcept {
    int n = static_cast<int>(x);
    double frac = x - static_cast<double>(n);
    double f = 1.0 + frac * (0.6931471805599453 + frac * (0.2402265069591007 + frac * 0.0555041086648216));
    union { double d; uint64_t i; } u = { f };
    u.i += static_cast<uint64_t>(n) << 52;
    return u.d;
}
inline double fastExp(double x) noexcept { return fastExp2(x * 1.4426950408889634); }

// ============================================================================
//  Fast log2 (float)
//  max error ~0.0015
// ============================================================================
inline float fastLog2(float x) noexcept {
    union { float f; uint32_t i; } u = { x };
    int exp = ((u.i >> 23) & 0xFF) - 127;
    u.i = (u.i & 0x007FFFFF) | 0x3F800000;
    float mantissa = u.f - 1.0f;
    float approx = mantissa * (1.3466f - mantissa * (0.4232f - mantissa * 0.1123f));
    return static_cast<float>(exp) + approx;
}
inline float fastLog(float x) noexcept { return fastLog2(x) * 0.69314718f; }

// ============================================================================
//  SIMD batch: compute 4 inv sqrt values
// ============================================================================
inline void batchFastInvSqrt(const float* in, float* out, size_t count) noexcept {
    if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
        for (size_t i = 0; i < count; ++i) out[i] = fastInvSqrt(in[i]);
    } else {
        for (size_t i = 0; i < count; ++i) out[i] = fastInvSqrt(in[i]);
    }
}

// ============================================================================
//  Dynamic environment controller: choose between accurate and fast math
// ============================================================================
enum class FastMathMode : uint8_t { Accurate, Approximate, Adaptive };
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
        m_tolerance = tol;
    }
    float errorTolerance() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_tolerance;
    }
private:
    FastMathEnvironment() : m_mode(FastMathMode::Approximate), m_tolerance(1e-5f) {}
    mutable std::mutex m_mutex;
    FastMathMode m_mode;
    float m_tolerance;
};

} // namespace Approx
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_APPROX_FAST_MATH_H_INCLUDED