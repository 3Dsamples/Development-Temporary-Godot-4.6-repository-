//File 0033 : core/math/color_math.h
//Color space conversions (sRGB↔linear, HSV, HSL, YUV, CIE XYZ, CIELAB), blending modes, gradient generation, and perceptual‑color wrapper via Oklab/OKLCH from perceptual_color.h.
#ifndef CORE_MATH_COLOR_MATH_H
#define CORE_MATH_COLOR_MATH_H

#include "vector_math.h"
#include "perceptual_color.h"
#include <algorithm>
#include <cmath>

namespace SimulationMath {
namespace color_math {

// -----------------------------------------------------------------------------
// 1. sRGB ↔ linear conversions (scalar and SIMD)
// -----------------------------------------------------------------------------
inline float linear_to_srgb(float c) noexcept {
    if (c <= 0.0031308f) return 12.92f * c;
    return 1.055f * std::pow(c, 1.0f/2.4f) - 0.055f;
}
inline float srgb_to_linear(float c) noexcept {
    if (c <= 0.04045f) return c / 12.92f;
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

inline DirectX::XMVECTOR linear_to_srgb(DirectX::FXMVECTOR v) noexcept {
    float c[4];
    vector_math::store4(c, v);
    for (int i=0; i<3; ++i) c[i] = linear_to_srgb(c[i]);
    return vector_math::load4(c);
}
inline DirectX::XMVECTOR srgb_to_linear(DirectX::FXMVECTOR v) noexcept {
    float c[4];
    vector_math::store4(c, v);
    for (int i=0; i<3; ++i) c[i] = srgb_to_linear(c[i]);
    return vector_math::load4(c);
}

// -----------------------------------------------------------------------------
// 2. RGB ↔ HSV (H in [0,1], S,V in [0,1])
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR rgb_to_hsv(DirectX::FXMVECTOR rgb) noexcept {
    float r = vector_math::get_x(rgb), g = vector_math::get_y(rgb), b = vector_math::get_z(rgb);
    float max_val = std::max({r,g,b});
    float min_val = std::min({r,g,b});
    float delta = max_val - min_val;
    float h = 0.0f, s = 0.0f, v = max_val;
    if (delta > 1e-6f) {
        s = delta / max_val;
        if (max_val == r)
            h = (g - b) / delta + (g < b ? 6.0f : 0.0f);
        else if (max_val == g)
            h = (b - r) / delta + 2.0f;
        else
            h = (r - g) / delta + 4.0f;
        h /= 6.0f;
    }
    return DirectX::XMVectorSet(h, s, v, 1.0f);
}

inline DirectX::XMVECTOR hsv_to_rgb(DirectX::FXMVECTOR hsv) noexcept {
    float h = vector_math::get_x(hsv), s = vector_math::get_y(hsv), v = vector_math::get_z(hsv);
    float r=0,g=0,b=0;
    if (s <= 1e-6f) { r=g=b=v; }
    else {
        h *= 6.0f;
        int sector = (int)h;
        float frac = h - sector;
        float p = v * (1.0f - s);
        float q = v * (1.0f - s * frac);
        float t = v * (1.0f - s * (1.0f - frac));
        switch (sector % 6) {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            case 5: r = v; g = p; b = q; break;
        }
    }
    return DirectX::XMVectorSet(r, g, b, 1.0f);
}

// -----------------------------------------------------------------------------
// 3. RGB ↔ HSL
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR rgb_to_hsl(DirectX::FXMVECTOR rgb) noexcept {
    float r = vector_math::get_x(rgb), g = vector_math::get_y(rgb), b = vector_math::get_z(rgb);
    float max_val = std::max({r,g,b}), min_val = std::min({r,g,b});
    float L = (max_val + min_val) * 0.5f;
    float delta = max_val - min_val;
    float h = 0.0f, s = 0.0f;
    if (delta > 1e-6f) {
        s = (L <= 0.5f) ? (delta / (max_val + min_val)) : (delta / (2.0f - max_val - min_val));
        if (max_val == r) h = (g - b) / delta + (g < b ? 6.0f : 0.0f);
        else if (max_val == g) h = (b - r) / delta + 2.0f;
        else h = (r - g) / delta + 4.0f;
        h /= 6.0f;
    }
    return DirectX::XMVectorSet(h, s, L, 1.0f);
}

inline float hue_to_rgb(float p, float q, float t) noexcept {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f/6.0f) return p + (q - p) * 6.0f * t;
    if (t < 0.5f) return q;
    if (t < 2.0f/3.0f) return p + (q - p) * (2.0f/3.0f - t) * 6.0f;
    return p;
}
inline DirectX::XMVECTOR hsl_to_rgb(DirectX::FXMVECTOR hsl) noexcept {
    float h = vector_math::get_x(hsl), s = vector_math::get_y(hsl), L = vector_math::get_z(hsl);
    float r,g,b;
    if (s <= 1e-6f) { r=g=b=L; }
    else {
        float q = (L < 0.5f) ? (L * (1.0f + s)) : (L + s - L * s);
        float p = 2.0f * L - q;
        r = hue_to_rgb(p, q, h + 1.0f/3.0f);
        g = hue_to_rgb(p, q, h);
        b = hue_to_rgb(p, q, h - 1.0f/3.0f);
    }
    return DirectX::XMVectorSet(r, g, b, 1.0f);
}

// -----------------------------------------------------------------------------
// 4. RGB ↔ YUV (BT.709)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR rgb_to_yuv(DirectX::FXMVECTOR rgb) noexcept {
    float r = vector_math::get_x(rgb), g = vector_math::get_y(rgb), b = vector_math::get_z(rgb);
    float y  =  0.2126f * r + 0.7152f * g + 0.0722f * b;
    float u  = -0.09991f* r - 0.33609f* g + 0.436f   * b;
    float v  =  0.615f   * r - 0.55861f* g - 0.05639f * b;
    return DirectX::XMVectorSet(y, u, v, 1.0f);
}
inline DirectX::XMVECTOR yuv_to_rgb(DirectX::FXMVECTOR yuv) noexcept {
    float y = vector_math::get_x(yuv), u = vector_math::get_y(yuv), v = vector_math::get_z(yuv);
    float r = y + 1.28033f * v;
    float g = y - 0.21482f * u - 0.38059f * v;
    float b = y + 2.12798f * u;
    return DirectX::XMVectorSet(r, g, b, 1.0f);
}

// -----------------------------------------------------------------------------
// 5. RGB ↔ CIE XYZ (sRGB primaries, D65)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR rgb_to_xyz(DirectX::FXMVECTOR rgb) noexcept {
    float r = vector_math::get_x(rgb), g = vector_math::get_y(rgb), b = vector_math::get_z(rgb);
    float x = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
    float y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
    float z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;
    return DirectX::XMVectorSet(x, y, z, 1.0f);
}
inline DirectX::XMVECTOR xyz_to_rgb(DirectX::FXMVECTOR xyz) noexcept {
    float x = vector_math::get_x(xyz), y = vector_math::get_y(xyz), z = vector_math::get_z(xyz);
    float r =  3.2404542f * x - 1.5371385f * y - 0.4985314f * z;
    float g = -0.9692660f * x + 1.8760108f * y + 0.0415560f * z;
    float b =  0.0556434f * x - 0.2040259f * y + 1.0572252f * z;
    return DirectX::XMVectorSet(r, g, b, 1.0f);
}

// -----------------------------------------------------------------------------
// 6. CIE XYZ ↔ CIE LAB (D65 white point)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR xyz_to_lab(DirectX::FXMVECTOR xyz) noexcept {
    float x = vector_math::get_x(xyz) / 0.95047f;
    float y = vector_math::get_y(xyz) / 1.0f;
    float z = vector_math::get_z(xyz) / 1.08883f;
    auto f = [](float t){ return (t > 0.008856f) ? std::cbrt(t) : (7.787f * t + 16.0f/116.0f); };
    float fx = f(x), fy = f(y), fz = f(z);
    float L = 116.0f * fy - 16.0f;
    float a = 500.0f * (fx - fy);
    float b = 200.0f * (fy - fz);
    return DirectX::XMVectorSet(L, a, b, 1.0f);
}
inline DirectX::XMVECTOR lab_to_xyz(DirectX::FXMVECTOR lab) noexcept {
    float L = vector_math::get_x(lab), a = vector_math::get_y(lab), b = vector_math::get_z(lab);
    float fy = (L + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - b / 200.0f;
    auto inv = [](float t){ return (t > 0.206897f) ? (t*t*t) : ((t - 16.0f/116.0f) / 7.787f); };
    float x = inv(fx) * 0.95047f;
    float y = inv(fy) * 1.0f;
    float z = inv(fz) * 1.08883f;
    return DirectX::XMVectorSet(x, y, z, 1.0f);
}

// -----------------------------------------------------------------------------
// 7. Blending modes (assumes inputs normalized [0,1])
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR blend_add(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept {
    return DirectX::XMVectorAdd(a, b);
}
inline DirectX::XMVECTOR blend_multiply(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept {
    return DirectX::XMVectorMultiply(a, b);
}
inline DirectX::XMVECTOR blend_overlay(DirectX::FXMVECTOR base, DirectX::FXMVECTOR blend) noexcept {
    // Use per-component if logic
    float bR = vector_math::get_x(base), bG = vector_math::get_y(base), bB = vector_math::get_z(base);
    float oR = vector_math::get_x(blend), oG = vector_math::get_y(blend), oB = vector_math::get_z(blend);
    auto overlay_ch = [](float b, float o) { return (b < 0.5f) ? (2.0f*b*o) : (1.0f - 2.0f*(1.0f-b)*(1.0f-o)); };
    return DirectX::XMVectorSet(overlay_ch(bR,oR), overlay_ch(bG,oG), overlay_ch(bB,oB), 1.0f);
}
inline DirectX::XMVECTOR blend_screen(DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) noexcept {
    return DirectX::XMVectorSubtract(
        DirectX::XMVectorReplicate(1.0f),
        DirectX::XMVectorMultiply(
            DirectX::XMVectorSubtract(DirectX::XMVectorReplicate(1.0f), a),
            DirectX::XMVectorSubtract(DirectX::XMVectorReplicate(1.0f), b)));
}

// -----------------------------------------------------------------------------
// 8. Linear gradient along axis
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR gradient_lerp(DirectX::FXMVECTOR color_start, DirectX::FXMVECTOR color_end, float t) noexcept {
    return vector_math::lerp(color_start, color_end, std::max(0.0f, std::min(t,1.0f)));
}

// -----------------------------------------------------------------------------
// 9. Wrappers for Oklab/OKLCH operations (from perceptual_color.h) using Vector4
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR apply_perfect_tint(DirectX::FXMVECTOR base_linear, DirectX::FXMVECTOR tint_linear, float factor) noexcept {
    Vector4 base(vector_math::get_x(base_linear), vector_math::get_y(base_linear), vector_math::get_z(base_linear), 1.0f);
    Vector4 tint(vector_math::get_x(tint_linear), vector_math::get_y(tint_linear), vector_math::get_z(tint_linear), 1.0f);
    Vector4 result = PerceptualColorEngine::ApplyPerfectTint(base, tint, factor);
    return DirectX::XMVectorSet(result.r, result.g, result.b, 1.0f);
}

inline DirectX::XMVECTOR apply_chromatization(DirectX::FXMVECTOR color, float scale) noexcept {
    Vector4 col(vector_math::get_x(color), vector_math::get_y(color), vector_math::get_z(color), 1.0f);
    Vector4 result = PerceptualColorEngine::ApplyChromatization(col, scale);
    return DirectX::XMVectorSet(result.r, result.g, result.b, 1.0f);
}

} // namespace color_math
} // namespace SimulationMath

#endif // CORE_MATH_COLOR_MATH_H