//File 0049 : core/math/easing.h
//Standard easing functions (Penner easing) for animations: linear, quad, cubic, quart, quint, sine, expo, circ, elastic, back, bounce, with in/out/inOut variants, supporting float/double and SIMD-ready.
#ifndef CORE_MATH_EASING_H
#define CORE_MATH_EASING_H

#include <cmath>
#include <algorithm>

namespace SimulationMath {
namespace easing {

// -----------------------------------------------------------------------------
// 1. Linear (no easing)
// -----------------------------------------------------------------------------
inline float linear(float t) noexcept { return t; }

// -----------------------------------------------------------------------------
// 2. Quadratic
// -----------------------------------------------------------------------------
inline float quad_in(float t) noexcept { return t * t; }
inline float quad_out(float t) noexcept { return t * (2.0f - t); }
inline float quad_in_out(float t) noexcept {
    if (t < 0.5f) return 2.0f * t * t;
    t = 2.0f * t - 1.0f;
    return -0.5f * (t * (t - 2.0f) - 1.0f);
}

// -----------------------------------------------------------------------------
// 3. Cubic
// -----------------------------------------------------------------------------
inline float cubic_in(float t) noexcept { return t * t * t; }
inline float cubic_out(float t) noexcept { t -= 1.0f; return t * t * t + 1.0f; }
inline float cubic_in_out(float t) noexcept {
    if (t < 0.5f) return 4.0f * t * t * t;
    t = 2.0f * t - 2.0f;
    return 0.5f * (t * t * t + 2.0f);
}

// -----------------------------------------------------------------------------
// 4. Quartic
// -----------------------------------------------------------------------------
inline float quart_in(float t) noexcept { return t * t * t * t; }
inline float quart_out(float t) noexcept { t -= 1.0f; return 1.0f - t * t * t * t; }
inline float quart_in_out(float t) noexcept {
    if (t < 0.5f) return 8.0f * t * t * t * t;
    t = 2.0f * t - 2.0f;
    return 0.5f * (8.0f - t * t * t * t);
}

// -----------------------------------------------------------------------------
// 5. Quintic
// -----------------------------------------------------------------------------
inline float quint_in(float t) noexcept { return t * t * t * t * t; }
inline float quint_out(float t) noexcept { t -= 1.0f; return t * t * t * t * t + 1.0f; }
inline float quint_in_out(float t) noexcept {
    if (t < 0.5f) return 16.0f * t * t * t * t * t;
    t = 2.0f * t - 2.0f;
    return 0.5f * (t * t * t * t * t + 2.0f);
}

// -----------------------------------------------------------------------------
// 6. Sine
// -----------------------------------------------------------------------------
inline float sine_in(float t) noexcept { return 1.0f - std::cos(t * 1.57079632679f); }
inline float sine_out(float t) noexcept { return std::sin(t * 1.57079632679f); }
inline float sine_in_out(float t) noexcept { return 0.5f * (1.0f - std::cos(3.14159265359f * t)); }

// -----------------------------------------------------------------------------
// 7. Exponential
// -----------------------------------------------------------------------------
inline float expo_in(float t) noexcept { return (t <= 0.0f) ? 0.0f : std::pow(2.0f, 10.0f * (t - 1.0f)); }
inline float expo_out(float t) noexcept { return (t >= 1.0f) ? 1.0f : 1.0f - std::pow(2.0f, -10.0f * t); }
inline float expo_in_out(float t) noexcept {
    if (t <= 0.0f) return 0.0f;
    if (t >= 1.0f) return 1.0f;
    if (t < 0.5f) return 0.5f * std::pow(2.0f, 20.0f * t - 10.0f);
    return 0.5f * (2.0f - std::pow(2.0f, -20.0f * t + 10.0f));
}

// -----------------------------------------------------------------------------
// 8. Circular
// -----------------------------------------------------------------------------
inline float circ_in(float t) noexcept { return 1.0f - std::sqrt(1.0f - t * t); }
inline float circ_out(float t) noexcept { t -= 1.0f; return std::sqrt(1.0f - t * t); }
inline float circ_in_out(float t) noexcept {
    if (t < 0.5f) return 0.5f * (1.0f - std::sqrt(1.0f - 4.0f * t * t));
    t = 2.0f * t - 2.0f;
    return 0.5f * (std::sqrt(1.0f - t * t) + 1.0f);
}

// -----------------------------------------------------------------------------
// 9. Elastic
// -----------------------------------------------------------------------------
inline float elastic_in(float t) noexcept {
    if (t <= 0.0f) return 0.0f;
    if (t >= 1.0f) return 1.0f;
    return std::pow(2.0f, 10.0f * (t - 1.0f)) * std::sin((t - 1.0f) * 2.0f * 3.14159265359f * 2.5f);
}
inline float elastic_out(float t) noexcept {
    if (t <= 0.0f) return 0.0f;
    if (t >= 1.0f) return 1.0f;
    return 1.0f - std::pow(2.0f, -10.0f * t) * std::cos(t * 2.0f * 3.14159265359f * 2.5f);
}
inline float elastic_in_out(float t) noexcept {
    if (t <= 0.0f) return 0.0f;
    if (t >= 1.0f) return 1.0f;
    if (t < 0.5f) {
        t = 2.0f * t - 1.0f;
        return 0.5f * std::pow(2.0f, 10.0f * t) * std::sin((t - 0.1125f) * 2.0f * 3.14159265359f / 0.45f);
    }
    t = 2.0f * t - 1.0f;
    return 0.5f * (1.0f - std::pow(2.0f, -10.0f * t) * std::sin((t - 0.1125f) * 2.0f * 3.14159265359f / 0.45f)) + 0.5f;
}

// -----------------------------------------------------------------------------
// 10. Back (overshoot)
// -----------------------------------------------------------------------------
inline float back_in(float t) noexcept { return t * t * (2.70158f * t - 1.70158f); }
inline float back_out(float t) noexcept { t -= 1.0f; return t * t * (2.70158f * t + 1.70158f) + 1.0f; }
inline float back_in_out(float t) noexcept {
    const float s = 1.70158f * 1.525f;
    if (t < 0.5f) return 0.5f * (4.0f * t * t * ((s + 1.0f) * 2.0f * t - s));
    t = 2.0f * t - 2.0f;
    return 0.5f * (t * t * ((s + 1.0f) * t + s) + 2.0f);
}

// -----------------------------------------------------------------------------
// 11. Bounce
// -----------------------------------------------------------------------------
inline float bounce_out(float t) noexcept {
    if (t < 1.0f / 2.75f) return 7.5625f * t * t;
    if (t < 2.0f / 2.75f) { t -= 1.5f / 2.75f; return 7.5625f * t * t + 0.75f; }
    if (t < 2.5f / 2.75f) { t -= 2.25f / 2.75f; return 7.5625f * t * t + 0.9375f; }
    t -= 2.625f / 2.75f;
    return 7.5625f * t * t + 0.984375f;
}
inline float bounce_in(float t) noexcept { return 1.0f - bounce_out(1.0f - t); }
inline float bounce_in_out(float t) noexcept {
    if (t < 0.5f) return 0.5f * bounce_in(2.0f * t);
    return 0.5f * bounce_out(2.0f * t - 1.0f) + 0.5f;
}

} // namespace easing
} // namespace SimulationMath

#endif // CORE_MATH_EASING_H