//File 0036 : core/math/math_functions.h
//Fast scalar/SIMD transcendental approximations (sin, cos, tan, atan2, exp, log, pow), Horner polynomial evaluation, and utility functions (smooth clamp, remap, cubic root).
#ifndef CORE_MATH_MATH_FUNCTIONS_H
#define CORE_MATH_MATH_FUNCTIONS_H

#include "vector_math.h"
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

namespace SimulationMath {
namespace math_func {

// -----------------------------------------------------------------------------
// 1. Horner evaluation (polynomial P(x) = a[0] + a[1]*x + ... + a[N-1]*x^(N-1))
// -----------------------------------------------------------------------------
inline float horner(float x, const float* coeffs, int degree) noexcept {
    float result = coeffs[degree];
    for (int i = degree - 1; i >= 0; --i)
        result = result * x + coeffs[i];
    return result;
}

// -----------------------------------------------------------------------------
// 2. Fast sine / cosine (max error ~ 1e-4, polynomial approximation)
// -----------------------------------------------------------------------------
inline void fast_sincos(float x, float& sin_out, float& cos_out) noexcept {
    // Reduce to [-PI, PI]
    const float inv_pi = 1.0f / 3.14159265358979323846f;
    float q = x * inv_pi;
    int k = static_cast<int>(q);
    float val = q - k;
    // Quadrant adjustment
    if (k & 1) { val = 1.0f - val; }
    // Sine polynomial on [0,1] (odd)
    float val2 = val * val;
    float s = val * (1.57079632679f + val2 * (-0.64596409751f + val2 * 0.07969262625f));
    float c = 1.57079632679f + val * (-1.57079632679f + val2 * (0.64596409751f - val * 0.07969262625f));
    if (k & 1) { std::swap(s, c); }
    // Adjust sign based on quadrant (k&2)
    if ((k & 2) != 0) {
        sin_out = -s;
        cos_out = -c;
    } else {
        sin_out = s;
        cos_out = c;
    }
}

inline float fast_sin(float x) noexcept {
    float s, c;
    fast_sincos(x, s, c);
    return s;
}

inline float fast_cos(float x) noexcept {
    float s, c;
    fast_sincos(x, s, c);
    return c;
}

// -----------------------------------------------------------------------------
// 3. Fast tan (uses sin/cos)
// -----------------------------------------------------------------------------
inline float fast_tan(float x) noexcept {
    float s, c;
    fast_sincos(x, s, c);
    if (std::abs(c) < 1e-6f) return s / 1e-6f;
    return s / c;
}

// -----------------------------------------------------------------------------
// 4. Fast atan2 (approximation with error ~ 0.005 rad)
// -----------------------------------------------------------------------------
inline float fast_atan2(float y, float x) noexcept {
    float ay = std::abs(y);
    float ax = std::abs(x);
    float a = (ay < ax) ? (ay / ax) : (ax / ay);
    float s = a * a;
    float r = a * (0.99997726f + s * (-0.33262347f + s * (0.19354346f + s * (-0.11643287f + s * (0.05265332f - s * 0.0117212f)))));
    if (ay > ax) r = 1.57079632679f - r;
    if (x < 0.0f) r = 3.14159265358979f - r;
    return (y < 0.0f) ? -r : r;
}

// -----------------------------------------------------------------------------
// 5. Fast exponential (base e) – range reduction to [-0.5,0.5]
// -----------------------------------------------------------------------------
inline float fast_exp(float x) noexcept {
    // e^x = 2^(x / ln2)
    const float inv_ln2 = 1.4426950408889634f;
    float k = std::floor(x * inv_ln2);
    float f = x - k * 0.6931471805599453f; // ln2
    // polynomial for 2^f, f in [0,1]
    float f2 = f * f;
    float val = 1.0f + f * (0.9999999995f + f2 * (0.4999999206f + f2 * (0.1666653019f + f2 * (0.0416573475f + f2 * (0.0083013598f + f2 * (0.0013298820f + f2 * 0.0001413161f))))));
    // Scale by 2^k
    int ik = static_cast<int>(k);
    if (ik >= 0)
        return val * (1u << ik);
    else
        return val / (1u << (-ik));
}

// -----------------------------------------------------------------------------
// 6. Fast natural logarithm – range reduction to [0.5,2.0]
// -----------------------------------------------------------------------------
inline float fast_log(float x) noexcept {
    // log(x) = log(m * 2^e) = log(m) + e*log(2)
    int e;
    float m = std::frexp(x, &e); // m in [0.5,1)
    float y = (m - 1.0f) / (m + 1.0f); // y in [ -1/3, 0 ]
    float y2 = y * y;
    float s = y * (2.0f + y2 * (0.6666666667f + y2 * (0.4f + y2 * (0.2857142857f + y2 * 0.2222222222f))));
    return s + e * 0.6931471805599453f;
}

// -----------------------------------------------------------------------------
// 7. Fast power: x^y = exp(y * log(x))
// -----------------------------------------------------------------------------
inline float fast_pow(float x, float y) noexcept {
    if (x <= 0.0f) return 0.0f;
    return fast_exp(y * fast_log(x));
}

// -----------------------------------------------------------------------------
// 8. Fast reciprocal square root (already available via SIMD, scalar version)
// -----------------------------------------------------------------------------
inline float fast_rsqrt(float x) noexcept {
    long i;
    float x2 = x * 0.5f;
    float y = x;
    i = *(long*)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float*)&i;
    y = y * (1.5f - (x2 * y * y));
    return y;
}

// -----------------------------------------------------------------------------
// 9. Cubic root (fast approximation, then Newton refinement)
// -----------------------------------------------------------------------------
inline float fast_cbrt(float x) noexcept {
    int e;
    float m = std::frexp(std::abs(x), &e);
    // initial guess: m^(1/3) ≈ a + b*m
    float guess = 0.0f;
    if (m > 0.0f) {
        // Approximation: m^(1/3) in [0.5,1) -> linear interpolation
        guess = 0.7937f * m + 0.2063f;
    }
    // Newton iteration: y = y - (y^3 - m) / (3*y^2) = (2*y + m/(y*y))/3
    float y = guess;
    for (int i = 0; i < 3; ++i) {
        float y2 = y * y;
        y = (2.0f * y + m / y2) / 3.0f;
    }
    // scale by 2^(e/3)
    float scale = std::pow(2.0f, e / 3.0f); // could also use fast exp2
    float result = y * scale;
    return (x < 0.0f) ? -result : result;
}

// -----------------------------------------------------------------------------
// 10. Smooth clamp (Hermite interpolation between limits)
// -----------------------------------------------------------------------------
inline float smooth_clamp(float x, float low, float high) noexcept {
    if (x <= low) return low;
    if (x >= high) return high;
    float t = (x - low) / (high - low);
    return low + (high - low) * t * t * (3.0f - 2.0f * t);
}

// -----------------------------------------------------------------------------
// 11. Remap value from one range to another
// -----------------------------------------------------------------------------
inline float remap(float value, float in_min, float in_max, float out_min, float out_max) noexcept {
    float t = (value - in_min) / (in_max - in_min);
    return out_min + t * (out_max - out_min);
}

} // namespace math_func
} // namespace SimulationMath

#endif // CORE_MATH_MATH_FUNCTIONS_H