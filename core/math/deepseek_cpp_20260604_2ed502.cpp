// system name : onetbb-warp
// File 0029 : core/math/complex.h
// Description : Complex number class with full arithmetic, polar, and transcendental functions.

#ifndef __TBB_WARP_CORE_MATH_COMPLEX_H
#define __TBB_WARP_CORE_MATH_COMPLEX_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include <cmath>
#include <type_traits>
#include <complex>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Complex number class template
// ============================================================

template<typename T>
struct complex {
    using value_type = T;

    T re, im;

    // ---- Constructors ----
    constexpr complex() noexcept : re(T(0)), im(T(0)) {}
    constexpr complex(T r, T i = T(0)) noexcept : re(r), im(i) {}
    template<typename U>
    constexpr explicit complex(const complex<U>& other) noexcept
        : re(static_cast<T>(other.re)), im(static_cast<T>(other.im)) {}
    constexpr complex(const std::complex<T>& std_complex) noexcept : re(std_complex.real()), im(std_complex.imag()) {}
    constexpr complex(const vector2<T>& v) noexcept : re(v.x), im(v.y) {}

    // ---- Conversion to std::complex ----
    constexpr operator std::complex<T>() const noexcept { return std::complex<T>(re, im); }
    constexpr operator vector2<T>() const noexcept { return vector2<T>(re, im); }

    // ---- Compound assignment ----
    constexpr complex& operator+=(const complex& c) noexcept { re+=c.re; im+=c.im; return *this; }
    constexpr complex& operator-=(const complex& c) noexcept { re-=c.re; im-=c.im; return *this; }
    constexpr complex& operator*=(const complex& c) noexcept {
        T new_re = re*c.re - im*c.im;
        T new_im = re*c.im + im*c.re;
        re = new_re; im = new_im;
        return *this;
    }
    constexpr complex& operator*=(T s) noexcept { re*=s; im*=s; return *this; }
    constexpr complex& operator/=(const complex& c) noexcept {
        T denom = c.re*c.re + c.im*c.im;
        T new_re = (re*c.re + im*c.im) / denom;
        T new_im = (im*c.re - re*c.im) / denom;
        re = new_re; im = new_im;
        return *this;
    }
    constexpr complex& operator/=(T s) noexcept { re/=s; im/=s; return *this; }

    // ---- Unary ----
    constexpr complex operator+() const noexcept { return *this; }
    constexpr complex operator-() const noexcept { return complex(-re, -im); }

    // ---- Conversion ----
    explicit constexpr operator bool() const noexcept { return re!=T(0) || im!=T(0); }
};

// ============================================================
// Binary operators
// ============================================================

template<typename T> constexpr complex<T> operator+(const complex<T>& a, const complex<T>& b) noexcept { return {a.re+b.re, a.im+b.im}; }
template<typename T> constexpr complex<T> operator-(const complex<T>& a, const complex<T>& b) noexcept { return {a.re-b.re, a.im-b.im}; }
template<typename T> constexpr complex<T> operator*(const complex<T>& a, const complex<T>& b) noexcept {
    return {a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
}
template<typename T> constexpr complex<T> operator*(T s, const complex<T>& c) noexcept { return {s*c.re, s*c.im}; }
template<typename T> constexpr complex<T> operator*(const complex<T>& c, T s) noexcept { return {c.re*s, c.im*s}; }
template<typename T> constexpr complex<T> operator/(const complex<T>& a, const complex<T>& b) noexcept {
    T denom = b.re*b.re + b.im*b.im;
    return {(a.re*b.re + a.im*b.im)/denom, (a.im*b.re - a.re*b.im)/denom};
}
template<typename T> constexpr complex<T> operator/(const complex<T>& c, T s) noexcept { return {c.re/s, c.im/s}; }
template<typename T> constexpr bool operator==(const complex<T>& a, const complex<T>& b) noexcept { return a.re==b.re && a.im==b.im; }
template<typename T> constexpr bool operator!=(const complex<T>& a, const complex<T>& b) noexcept { return !(a==b); }

// ============================================================
// Basic properties
// ============================================================

template<typename T> constexpr T real(const complex<T>& c) noexcept { return c.re; }
template<typename T> constexpr T imag(const complex<T>& c) noexcept { return c.im; }
template<typename T> constexpr T norm_sq(const complex<T>& c) noexcept { return c.re*c.re + c.im*c.im; }
template<typename T> T norm(const complex<T>& c) noexcept { return std::sqrt(norm_sq(c)); }
template<typename T> constexpr T arg(const complex<T>& c) noexcept { return std::atan2(c.im, c.re); }
template<typename T> constexpr complex<T> conjugate(const complex<T>& c) noexcept { return {c.re, -c.im}; }
template<typename T> complex<T> inverse(const complex<T>& c) noexcept {
    T denom = norm_sq(c);
    return {c.re/denom, -c.im/denom};
}

// ============================================================
// Polar form
// ============================================================

template<typename T>
constexpr complex<T> from_polar(T radius, T angle) noexcept {
    return {radius * std::cos(angle), radius * std::sin(angle)};
}

template<typename T>
void to_polar(const complex<T>& c, T& radius, T& angle) noexcept {
    radius = norm(c);
    angle = arg(c);
}

// ============================================================
// Exponential and trigonometric functions
// ============================================================

template<typename T>
complex<T> exp(const complex<T>& c) noexcept {
    T e = std::exp(c.re);
    return {e * std::cos(c.im), e * std::sin(c.im)};
}

template<typename T>
complex<T> log(const complex<T>& c) noexcept {
    T r = norm(c);
    T theta = arg(c);
    return {std::log(r), theta};
}

template<typename T>
complex<T> pow(const complex<T>& base, const complex<T>& exponent) noexcept {
    return exp(log(base) * exponent);
}

template<typename T>
complex<T> pow(const complex<T>& base, T exponent) noexcept {
    T r = std::pow(norm(base), exponent);
    T theta = arg(base) * exponent;
    return {r * std::cos(theta), r * std::sin(theta)};
}

template<typename T>
complex<T> sqrt(const complex<T>& c) noexcept {
    T r = std::sqrt(norm(c));
    T theta = arg(c) * T(0.5);
    return {r * std::cos(theta), r * std::sin(theta)};
}

// Trigonometric
template<typename T>
complex<T> sin(const complex<T>& c) noexcept {
    return {std::sin(c.re)*std::cosh(c.im), std::cos(c.re)*std::sinh(c.im)};
}

template<typename T>
complex<T> cos(const complex<T>& c) noexcept {
    return {std::cos(c.re)*std::cosh(c.im), -std::sin(c.re)*std::sinh(c.im)};
}

template<typename T>
complex<T> tan(const complex<T>& c) noexcept {
    return sin(c) / cos(c);
}

template<typename T>
complex<T> sinh(const complex<T>& c) noexcept {
    return {std::sinh(c.re)*std::cos(c.im), std::cosh(c.re)*std::sin(c.im)};
}

template<typename T>
complex<T> cosh(const complex<T>& c) noexcept {
    return {std::cosh(c.re)*std::cos(c.im), std::sinh(c.re)*std::sin(c.im)};
}

template<typename T>
complex<T> tanh(const complex<T>& c) noexcept {
    return sinh(c) / cosh(c);
}

// Inverse trigonometric
template<typename T>
complex<T> asin(const complex<T>& c) noexcept {
    complex<T> i(0,1);
    return -i * log(i * c + sqrt(complex<T>(1) - c * c));
}

template<typename T>
complex<T> acos(const complex<T>& c) noexcept {
    complex<T> i(0,1);
    return -i * log(c + i * sqrt(complex<T>(1) - c * c));
}

template<typename T>
complex<T> atan(const complex<T>& c) noexcept {
    complex<T> i(0,1);
    return i * T(0.5) * log((i + c) / (i - c));
}

// ============================================================
// Type aliases
// ============================================================

using complexf = complex<float>;
using complexd = complex<double>;

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_COMPLEX_H