// File 0027 : core/math/color.h
// RGBA color with perceptual Oklab/OKLCH conversions, perfect tint, chromatization, and blending.

#pragma once

#include "constants.h"
#include "vec4.h"
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace wp {

// ── Perceptual color spaces ────────────────────────────────────────
template <typename T>
struct Oklab {
    T L, a, b;
    constexpr Oklab(T L_=T(0), T a_=T(0), T b_=T(0)) noexcept : L(L_), a(a_), b(b_) {}
};

template <typename T>
struct OKLCH {
    T L, C, H;   // H in radians, normalized to (-pi, pi] or [0, 2pi), we store raw
    constexpr OKLCH(T L_=T(0), T C_=T(0), T H_=T(0)) noexcept : L(L_), C(C_), H(H_) {}
};

// ── Color class ──────────────────────────────────────────────────────
template <typename T>
struct color {
    T r, g, b, a;

    constexpr color() noexcept : r(T(0)), g(T(0)), b(T(0)), a(T(1)) {}
    constexpr color(T r_, T g_, T b_, T a_ = T(1)) noexcept : r(r_), g(g_), b(b_), a(a_) {}
    constexpr explicit color(const vec4<T>& v) noexcept : r(v.x), g(v.y), b(v.z), a(v.w) {}
    constexpr operator vec4<T>() const noexcept { return vec4<T>(r, g, b, a); }
    template <typename U> constexpr explicit color(const color<U>& o) noexcept : r(static_cast<T>(o.r)), g(static_cast<T>(o.g)), b(static_cast<T>(o.b)), a(static_cast<T>(o.a)) {}

    constexpr T& operator[](int i) noexcept { return (&r)[i]; }
    constexpr T  operator[](int i) const noexcept { return (&r)[i]; }

    constexpr color operator+(const color& o) const noexcept { return color(r+o.r, g+o.g, b+o.b, a+o.a); }
    constexpr color operator-(const color& o) const noexcept { return color(r-o.r, g-o.g, b-o.b, a-o.a); }
    constexpr color operator*(T s) const noexcept { return color(r*s, g*s, b*s, a*s); }
    constexpr color operator/(T s) const noexcept { return color(r/s, g/s, b/s, a/s); }
    constexpr color operator*(const color& o) const noexcept { return color(r*o.r, g*o.g, b*o.b, a*o.a); }
    constexpr color& operator+=(const color& o) noexcept { r+=o.r; g+=o.g; b+=o.b; a+=o.a; return *this; }
    constexpr color& operator-=(const color& o) noexcept { r-=o.r; g-=o.g; b-=o.b; a-=o.a; return *this; }
    constexpr color& operator*=(T s) noexcept { r*=s; g*=s; b*=s; a*=s; return *this; }
    constexpr color& operator/=(T s) noexcept { r/=s; g/=s; b/=s; a/=s; return *this; }
    constexpr bool operator==(const color& o) const noexcept { return r==o.r && g==o.g && b==o.b && a==o.a; }
    constexpr bool operator!=(const color& o) const noexcept { return !(*this == o); }

    constexpr color& saturate() noexcept { r=clamp(r,T(0),T(1)); g=clamp(g,T(0),T(1)); b=clamp(b,T(0),T(1)); a=clamp(a,T(0),T(1)); return *this; }
    constexpr color saturated() const noexcept { color c=*this; c.saturate(); return c; }

    constexpr T luminance() const noexcept { return T(0.2126)*r + T(0.7152)*g + T(0.0722)*b; }
    constexpr color grayscale() const noexcept { T l=luminance(); return color(l,l,l,a); }
    constexpr color alpha_blend(const color& bg) const noexcept {
        color dst;
        dst.r = r*a + bg.r*(T(1)-a);
        dst.g = g*a + bg.g*(T(1)-a);
        dst.b = b*a + bg.b*(T(1)-a);
        dst.a = a + bg.a*(T(1)-a);
        return dst;
    }

    // sRGB / Linear
    static T srgb_to_linear(T c) noexcept {
        if (c <= T(0.04045)) return c / T(12.92);
        return std::pow((c + T(0.055)) / T(1.055), T(2.4));
    }
    static T linear_to_srgb(T c) noexcept {
        if (c <= T(0.0031308)) return T(12.92) * c;
        return T(1.055) * std::pow(c, T(1)/T(2.4)) - T(0.055);
    }
    color srgb_to_linear() const noexcept { return color(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a); }
    color linear_to_srgb() const noexcept { return color(linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b), a); }

    // ── Perceptual conversions ──────────────────────────────────────

    /** Linear RGB → Oklab */
    static Oklab<T> to_oklab(const color& c) noexcept {
        // LMS matrix
        T l = T(0.4122214708)*c.r + T(0.5363325363)*c.g + T(0.0514459929)*c.b;
        T m = T(0.2119034982)*c.r + T(0.6806995451)*c.g + T(0.1073969566)*c.b;
        T s = T(0.0883024619)*c.r + T(0.2817188376)*c.g + T(0.6299787005)*c.b;
        // cube root
        T l1 = std::cbrt(l);
        T m1 = std::cbrt(m);
        T s1 = std::cbrt(s);
        Oklab<T> lab;
        lab.L = T(0.2104542553)*l1 + T(0.7936177850)*m1 - T(0.0040720468)*s1;
        lab.a = T(1.9779984951)*l1 - T(2.4285922050)*m1 + T(0.4505937099)*s1;
        lab.b = T(0.0259040371)*l1 + T(0.7827717662)*m1 - T(0.8086758033)*s1;
        return lab;
    }

    /** Oklab → Linear RGB, alpha given, gamut clamped */
    static color from_oklab(const Oklab<T>& lab, T alpha = T(1)) noexcept {
        T l1 = lab.L + T(0.3963377774)*lab.a + T(0.2158037573)*lab.b;
        T m1 = lab.L - T(0.1055613458)*lab.a - T(0.0638541728)*lab.b;
        T s1 = lab.L - T(0.0894841775)*lab.a - T(1.2914855480)*lab.b;
        T l = l1 * l1 * l1;
        T m = m1 * m1 * m1;
        T s = s1 * s1 * s1;
        color res;
        res.r = T(+4.0767416621)*l - T(3.3077115913)*m + T(0.2309699292)*s;
        res.g = T(-1.2684380046)*l + T(2.6097574011)*m - T(0.3413193965)*s;
        res.b = T(-0.0041960863)*l - T(0.7034186147)*m + T(1.7076147010)*s;
        res.a = alpha;
        // Gamut clamp
        res.r = clamp(res.r, T(0), T(1));
        res.g = clamp(res.g, T(0), T(1));
        res.b = clamp(res.b, T(0), T(1));
        return res;
    }

    /** Linear RGB → OKLCH (perceptual polar) */
    static OKLCH<T> to_oklch(const color& c) noexcept {
        Oklab<T> lab = to_oklab(c);
        OKLCH<T> lch;
        lch.L = lab.L;
        lch.C = std::sqrt(lab.a*lab.a + lab.b*lab.b);
        lch.H = std::atan2(lab.b, lab.a);
        return lch;
    }

    /** OKLCH → Linear RGB */
    static color from_oklch(const OKLCH<T>& lch, T alpha = T(1)) noexcept {
        Oklab<T> lab;
        lab.L = lch.L;
        lab.a = lch.C * std::cos(lch.H);
        lab.b = lch.C * std::sin(lch.H);
        return from_oklab(lab, alpha);
    }

    /** Perfect tint: replaces base texture hue with tint hue, preserves lightness, blends via factor */
    static color perfect_tint(const color& base, const color& tint, T factor) noexcept {
        OKLCH<T> base_lch = to_oklch(base);
        OKLCH<T> tint_lch = to_oklch(tint);
        OKLCH<T> target_lch;
        target_lch.L = base_lch.L;                                    // preserve texture lightness
        target_lch.H = tint_lch.H;                                    // adopt tint hue
        T tint_saturation = tint_lch.C / (tint_lch.L + T(0.00001));    // saturation ratio
        target_lch.C = base_lch.C * clamp(tint_saturation, T(0), T(1)); // scale chroma by tint saturation
        color tinted = from_oklch(target_lch, base.a);
        // alpha‑aware lerp
        color output;
        output.r = base.r + factor * (tinted.r - base.r);
        output.g = base.g + factor * (tinted.g - base.g);
        output.b = base.b + factor * (tinted.b - base.b);
        output.a = base.a;
        return output;
    }

    /** Chromatization: scale chroma (vibrancy) without shifting hue or lightness */
    static color chromatize(const color& base, T scale) noexcept {
        OKLCH<T> lch = to_oklch(base);
        lch.C *= scale;
        return from_oklch(lch, base.a);
    }

    // ── HSV (Hue in [0,1], S,V in [0,1]) ────────────────────────────
    static color hsv_to_rgb(T h, T s, T v, T a = T(1)) noexcept {
        h = std::fmod(h, T(1)); if (h < T(0)) h += T(1);
        T c = v * s;
        T x = c * (T(1) - std::abs(std::fmod(h * T(6), T(2)) - T(1)));
        T m = v - c;
        color res;
        if      (h < T(1)/T(6)) { res.r=c; res.g=x; res.b=T(0); }
        else if (h < T(2)/T(6)) { res.r=x; res.g=c; res.b=T(0); }
        else if (h < T(3)/T(6)) { res.r=T(0); res.g=c; res.b=x; }
        else if (h < T(4)/T(6)) { res.r=T(0); res.g=x; res.b=c; }
        else if (h < T(5)/T(6)) { res.r=x; res.g=T(0); res.b=c; }
        else                   { res.r=c; res.g=T(0); res.b=x; }
        res.r += m; res.g += m; res.b += m;
        res.a = a;
        return res;
    }
    void rgb_to_hsv(T& h, T& s, T& v) const noexcept {
        T cmax = std::max({r,g,b});
        T cmin = std::min({r,g,b});
        T delta = cmax - cmin;
        v = cmax;
        s = (cmax > T(0)) ? (delta / cmax) : T(0);
        if (delta < epsilon<T>) h = T(0);
        else if (cmax == r) h = std::fmod((g - b) / delta, T(6));
        else if (cmax == g) h = (b - r) / delta + T(2);
        else                h = (r - g) / delta + T(4);
        h /= T(6);
        if (h < T(0)) h += T(1);
    }

    // ── HSL ──────────────────────────────────────────────────────────
    static color hsl_to_rgb(T h, T s, T l, T a = T(1)) noexcept {
        h = std::fmod(h, T(1)); if (h < T(0)) h += T(1);
        T c = (T(1) - std::abs(T(2)*l - T(1))) * s;
        T x = c * (T(1) - std::abs(std::fmod(h * T(6), T(2)) - T(1)));
        T m = l - c / T(2);
        color res;
        if      (h < T(1)/T(6)) { res.r=c; res.g=x; res.b=T(0); }
        else if (h < T(2)/T(6)) { res.r=x; res.g=c; res.b=T(0); }
        else if (h < T(3)/T(6)) { res.r=T(0); res.g=c; res.b=x; }
        else if (h < T(4)/T(6)) { res.r=T(0); res.g=x; res.b=c; }
        else if (h < T(5)/T(6)) { res.r=x; res.g=T(0); res.b=c; }
        else                   { res.r=c; res.g=T(0); res.b=x; }
        res.r += m; res.g += m; res.b += m;
        res.a = a;
        return res;
    }
    void rgb_to_hsl(T& h, T& s, T& l) const noexcept {
        T cmax = std::max({r,g,b});
        T cmin = std::min({r,g,b});
        T delta = cmax - cmin;
        l = (cmax + cmin) * T(0.5);
        if (delta < epsilon<T>) { s = T(0); h = T(0); }
        else {
            s = (l <= T(0.5)) ? (delta / (cmax + cmin)) : (delta / (T(2) - cmax - cmin));
            if (cmax == r) h = (g - b) / delta + (g < b ? T(6) : T(0));
            else if (cmax == g) h = (b - r) / delta + T(2);
            else h = (r - g) / delta + T(4);
            h /= T(6);
        }
    }

    // ── Predefined constants ──────────────────────────────────────────
    static constexpr color white()   noexcept { return color(T(1),T(1),T(1),T(1)); }
    static constexpr color black()   noexcept { return color(T(0),T(0),T(0),T(1)); }
    static constexpr color red()     noexcept { return color(T(1),T(0),T(0),T(1)); }
    static constexpr color green()   noexcept { return color(T(0),T(1),T(0),T(1)); }
    static constexpr color blue()    noexcept { return color(T(0),T(0),T(1),T(1)); }
    static constexpr color yellow()  noexcept { return color(T(1),T(1),T(0),T(1)); }
    static constexpr color cyan()    noexcept { return color(T(0),T(1),T(1),T(1)); }
    static constexpr color magenta() noexcept { return color(T(1),T(0),T(1),T(1)); }
    static constexpr color gray(T v=T(0.5)) noexcept { return color(v,v,v,T(1)); }
};

template <typename T> constexpr color<T> operator*(T s, const color<T>& c) noexcept { return c*s; }

using colorf = color<float>;
using colord = color<double>;

} // namespace wp