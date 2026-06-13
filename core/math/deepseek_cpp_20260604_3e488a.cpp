// system name : onetbb-warp
// File 0026 : core/math/color.h
// Description : Color spaces (RGB, HSV, HSL, XYZ, Lab, LCH, YUV, YCbCr), conversions, and blending.

#ifndef __TBB_WARP_CORE_MATH_COLOR_H
#define __TBB_WARP_CORE_MATH_COLOR_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include <cmath>
#include <type_traits>
#include <algorithm>

namespace tbb {
namespace core {
namespace math {
namespace color {

// ============================================================
// RGB <-> HSV
// ============================================================

inline vector3<float> rgb_to_hsv(const vector3<float>& rgb) {
    float r = rgb.x, g = rgb.y, b = rgb.z;
    float max_val = max({r, g, b});
    float min_val = min({r, g, b});
    float delta = max_val - min_val;
    float h = 0.0f, s = 0.0f, v = max_val;
    if (delta > 1e-6f) {
        s = delta / max_val;
        if (max_val == r)      h = (g - b) / delta + (g < b ? 6.0f : 0.0f);
        else if (max_val == g) h = (b - r) / delta + 2.0f;
        else                   h = (r - g) / delta + 4.0f;
        h /= 6.0f;
    }
    return {h, s, v};
}

inline vector3<float> hsv_to_rgb(const vector3<float>& hsv) {
    float h = hsv.x, s = hsv.y, v = hsv.z;
    if (s <= 0.0f) return {v, v, v};
    h = (h - std::floor(h)) * 6.0f;
    int i = static_cast<int>(h);
    float f = h - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    switch (i) {
        case 0: return {v, t, p};
        case 1: return {q, v, p};
        case 2: return {p, v, t};
        case 3: return {p, q, v};
        case 4: return {t, p, v};
        default: return {v, p, q};
    }
}

// ============================================================
// RGB <-> HSL
// ============================================================

inline vector3<float> rgb_to_hsl(const vector3<float>& rgb) {
    float r = rgb.x, g = rgb.y, b = rgb.z;
    float max_val = max({r, g, b});
    float min_val = min({r, g, b});
    float delta = max_val - min_val;
    float l = (max_val + min_val) * 0.5f;
    float h = 0.0f, s = 0.0f;
    if (delta > 1e-6f) {
        s = (l > 0.5f) ? delta / (2.0f - max_val - min_val) : delta / (max_val + min_val);
        if (max_val == r)      h = (g - b) / delta + (g < b ? 6.0f : 0.0f);
        else if (max_val == g) h = (b - r) / delta + 2.0f;
        else                   h = (r - g) / delta + 4.0f;
        h /= 6.0f;
    }
    return {h, s, l};
}

inline float hue_to_rgb(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f/6.0f) return p + (q - p) * 6.0f * t;
    if (t < 0.5f)      return q;
    if (t < 2.0f/3.0f) return p + (q - p) * (2.0f/3.0f - t) * 6.0f;
    return p;
}

inline vector3<float> hsl_to_rgb(const vector3<float>& hsl) {
    float h = hsl.x, s = hsl.y, l = hsl.z;
    if (s <= 0.0f) return {l, l, l};
    float q = (l < 0.5f) ? l * (1.0f + s) : l + s - l * s;
    float p = 2.0f * l - q;
    return {hue_to_rgb(p, q, h + 1.0f/3.0f),
            hue_to_rgb(p, q, h),
            hue_to_rgb(p, q, h - 1.0f/3.0f)};
}

// ============================================================
// RGB <-> XYZ (sRGB linear, D65 illuminant)
// ============================================================

inline vector3<float> linear_rgb_to_xyz(const vector3<float>& rgb) {
    float x = rgb.x*0.4124564f + rgb.y*0.3575761f + rgb.z*0.1804375f;
    float y = rgb.x*0.2126729f + rgb.y*0.7151522f + rgb.z*0.0721750f;
    float z = rgb.x*0.0193339f + rgb.y*0.1191920f + rgb.z*0.9503041f;
    return {x, y, z};
}

inline vector3<float> xyz_to_linear_rgb(const vector3<float>& xyz) {
    float r = xyz.x* 3.2404542f + xyz.y*-1.5371385f + xyz.z*-0.4985314f;
    float g = xyz.x*-0.9692660f + xyz.y* 1.8760108f + xyz.z* 0.0415560f;
    float b = xyz.x* 0.0556434f + xyz.y*-0.2040259f + xyz.z* 1.0572252f;
    return {r, g, b};
}

// ============================================================
// RGB <-> Lab (CIE L*a*b*, D65)
// ============================================================

inline float lab_f(float t) {
    const float delta = 6.0f/29.0f;
    if (t > delta*delta*delta) return std::cbrt(t);
    return t / (3.0f * delta*delta) + 4.0f/29.0f;
}

inline float lab_f_inv(float t) {
    const float delta = 6.0f/29.0f;
    if (t > delta) return t*t*t;
    return 3.0f * delta*delta * (t - 4.0f/29.0f);
}

inline vector3<float> xyz_to_lab(const vector3<float>& xyz, const vector3<float>& white = {0.95047f,1.0f,1.08883f}) {
    float fy = lab_f(xyz.y / white.y);
    float L = 116.0f * fy - 16.0f;
    float a = 500.0f * (lab_f(xyz.x / white.x) - fy);
    float b = 200.0f * (fy - lab_f(xyz.z / white.z));
    return {L, a, b};
}

inline vector3<float> lab_to_xyz(const vector3<float>& lab, const vector3<float>& white = {0.95047f,1.0f,1.08883f}) {
    float fy = (lab.x + 16.0f) / 116.0f;
    float fx = lab.y / 500.0f + fy;
    float fz = fy - lab.z / 200.0f;
    return {white.x * lab_f_inv(fx), white.y * lab_f_inv(fy), white.z * lab_f_inv(fz)};
}

inline vector3<float> rgb_to_lab(const vector3<float>& rgb) {
    return xyz_to_lab(linear_rgb_to_xyz(rgb));
}

inline vector3<float> lab_to_rgb(const vector3<float>& lab) {
    return xyz_to_linear_rgb(lab_to_xyz(lab));
}

// ============================================================
// RGB <-> LCH (Lightness, Chroma, Hue)
// ============================================================

inline vector3<float> lab_to_lch(const vector3<float>& lab) {
    float L = lab.x;
    float C = std::sqrt(lab.y*lab.y + lab.z*lab.z);
    float H = std::atan2(lab.z, lab.y);
    if (H < 0.0f) H += TAU_F;
    return {L, C, H};
}

inline vector3<float> lch_to_lab(const vector3<float>& lch) {
    return {lch.x, lch.y * std::cos(lch.z), lch.y * std::sin(lch.z)};
}

// ============================================================
// RGB <-> YUV (BT.601)
// ============================================================

inline vector3<float> rgb_to_yuv(const vector3<float>& rgb) {
    float y =  0.299f*rgb.x + 0.587f*rgb.y + 0.114f*rgb.z;
    float u = -0.14713f*rgb.x - 0.28886f*rgb.y + 0.436f*rgb.z;
    float v =  0.615f*rgb.x - 0.51499f*rgb.y - 0.10001f*rgb.z;
    return {y, u, v};
}

inline vector3<float> yuv_to_rgb(const vector3<float>& yuv) {
    float r = yuv.x + 1.13983f*yuv.z;
    float g = yuv.x - 0.39465f*yuv.y - 0.58060f*yuv.z;
    float b = yuv.x + 2.03211f*yuv.y;
    return {r, g, b};
}

// ============================================================
// RGB <-> YCbCr (BT.709)
// ============================================================

inline vector3<float> rgb_to_ycbcr(const vector3<float>& rgb) {
    float y  =  0.2126f*rgb.x + 0.7152f*rgb.y + 0.0722f*rgb.z;
    float cb = (rgb.z - y) / 1.8556f;
    float cr = (rgb.x - y) / 1.5748f;
    return {y, cb, cr};
}

inline vector3<float> ycbcr_to_rgb(const vector3<float>& ycbcr) {
    float r = ycbcr.x + 1.5748f*ycbcr.z;
    float g = ycbcr.x - 0.187324f*ycbcr.y - 0.468124f*ycbcr.z;
    float b = ycbcr.x + 1.8556f*ycbcr.y;
    return {r, g, b};
}

// ============================================================
// sRGB transfer functions (gamma)
// ============================================================

inline float srgb_to_linear(float c) {
    if (c <= 0.04045f) return c / 12.92f;
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

inline float linear_to_srgb(float c) {
    if (c <= 0.0031308f) return c * 12.92f;
    return 1.055f * std::pow(c, 1.0f/2.4f) - 0.055f;
}

inline vector3<float> srgb_to_linear_rgb(const vector3<float>& srgb) {
    return {srgb_to_linear(srgb.x), srgb_to_linear(srgb.y), srgb_to_linear(srgb.z)};
}

inline vector3<float> linear_rgb_to_srgb(const vector3<float>& linear) {
    return {linear_to_srgb(linear.x), linear_to_srgb(linear.y), linear_to_srgb(linear.z)};
}

// ============================================================
// Color temperature to RGB (approximation)
// ============================================================

inline vector3<float> color_temperature_to_rgb(float kelvin) {
    float temp = kelvin / 100.0f;
    float r, g, b;
    if (temp <= 66.0f) {
        r = 1.0f;
        g = 0.3900815787690196f * std::log(temp) - 0.631841443788627f;
        b = 0.0f;
        if (temp <= 19.0f) b = 0.0f;
        else b = 0.543206789110196f * std::log(temp-10.0f) - 1.19625408914f;
    } else {
        r = 1.292936186062745f * std::pow(temp-60.0f, -0.1332047592f);
        g = 1.129890860895294f * std::pow(temp-60.0f, -0.0755148492f);
        b = 1.0f;
    }
    return {clamp(r, 0.0f, 1.0f), clamp(g, 0.0f, 1.0f), clamp(b, 0.0f, 1.0f)};
}

// ============================================================
// Luminance (perceptual)
// ============================================================

inline float luminance_rgb(const vector3<float>& rgb) {
    return 0.2126f*rgb.x + 0.7152f*rgb.y + 0.0722f*rgb.z;
}

// ============================================================
// Contrast ratio
// ============================================================

inline float contrast_ratio(const vector3<float>& a, const vector3<float>& b) {
    float la = luminance_rgb(a) + 0.05f;
    float lb = luminance_rgb(b) + 0.05f;
    return (la > lb) ? la / lb : lb / la;
}

// ============================================================
// Color blending modes
// ============================================================

inline vector3<float> blend_alpha(const vector3<float>& src, const vector3<float>& dst, float alpha) {
    return src * alpha + dst * (1.0f - alpha);
}

inline vector3<float> blend_multiply(const vector3<float>& a, const vector3<float>& b) {
    return a * b;
}

inline vector3<float> blend_screen(const vector3<float>& a, const vector3<float>& b) {
    return {1.0f - (1.0f - a.x)*(1.0f - b.x),
            1.0f - (1.0f - a.y)*(1.0f - b.y),
            1.0f - (1.0f - a.z)*(1.0f - b.z)};
}

inline vector3<float> blend_overlay(const vector3<float>& a, const vector3<float>& b) {
    auto overlay_channel = [](float x, float y) {
        return (x < 0.5f) ? 2.0f*x*y : 1.0f - 2.0f*(1.0f-x)*(1.0f-y);
    };
    return {overlay_channel(a.x, b.x), overlay_channel(a.y, b.y), overlay_channel(a.z, b.z)};
}

inline vector3<float> blend_soft_light(const vector3<float>& a, const vector3<float>& b) {
    auto soft = [](float x, float y) {
        if (y <= 0.5f) return x - (1.0f - 2.0f*y) * x * (1.0f - x);
        return x + (2.0f*y - 1.0f) * (std::sqrt(x) - x);
    };
    return {soft(a.x, b.x), soft(a.y, b.y), soft(a.z, b.z)};
}

// ============================================================
// Color harmony (complementary, analogous, triadic, tetradic)
// ============================================================

inline vector3<float> complementary_hue(float hue) {
    return {std::fmod(hue + 0.5f, 1.0f), 1.0f, 1.0f};
}

inline std::array<vector3<float>, 3> analogous_hues(float hue) {
    return {{hsv_to_rgb({std::fmod(hue - 1.0f/12.0f, 1.0f), 1.0f, 1.0f}),
             hsv_to_rgb({hue, 1.0f, 1.0f}),
             hsv_to_rgb({std::fmod(hue + 1.0f/12.0f, 1.0f), 1.0f, 1.0f})}};
}

inline std::array<vector3<float>, 3> triadic_hues(float hue) {
    return {{hsv_to_rgb({hue, 1.0f, 1.0f}),
             hsv_to_rgb({std::fmod(hue + 1.0f/3.0f, 1.0f), 1.0f, 1.0f}),
             hsv_to_rgb({std::fmod(hue + 2.0f/3.0f, 1.0f), 1.0f, 1.0f})}};
}

inline std::array<vector3<float>, 4> tetradic_hues(float hue) {
    return {{hsv_to_rgb({hue, 1.0f, 1.0f}),
             hsv_to_rgb({std::fmod(hue + 1.0f/3.0f, 1.0f), 1.0f, 1.0f}),
             hsv_to_rgb({std::fmod(hue + 0.5f, 1.0f), 1.0f, 1.0f}),
             hsv_to_rgb({std::fmod(hue + 5.0f/6.0f, 1.0f), 1.0f, 1.0f})}};
}

} // namespace color
} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_COLOR_H