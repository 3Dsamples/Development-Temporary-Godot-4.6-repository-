// File 0036 : core/math/autodiff.h
// Forward-mode automatic differentiation with dual numbers for scalar and small vector/matrix operations.

#pragma once

#include "constants.h"
#include "vec2.h"
#include "vec3.h"
#include "vec4.h"
#include <cmath>
#include <type_traits>

namespace wp {

// ── Dual number: value + derivative ─────────────────────────────────
template <typename T>
struct dual {
    T val;
    T deriv;

    constexpr dual() noexcept : val(T(0)), deriv(T(0)) {}
    constexpr dual(T v, T d = T(0)) noexcept : val(v), deriv(d) {}
    template <typename U> constexpr explicit dual(const dual<U>& o) noexcept : val(static_cast<T>(o.val)), deriv(static_cast<T>(o.deriv)) {}

    // Arithmetic
    constexpr dual operator+(const dual& o) const noexcept { return dual(val + o.val, deriv + o.deriv); }
    constexpr dual operator-(const dual& o) const noexcept { return dual(val - o.val, deriv - o.deriv); }
    constexpr dual operator*(const dual& o) const noexcept { return dual(val * o.val, val * o.deriv + deriv * o.val); }
    constexpr dual operator/(const dual& o) const noexcept {
        T inv = T(1) / o.val;
        return dual(val * inv, (deriv - val * inv * o.deriv) * inv);
    }
    constexpr dual operator-() const noexcept { return dual(-val, -deriv); }

    constexpr dual& operator+=(const dual& o) noexcept { *this = *this + o; return *this; }
    constexpr dual& operator-=(const dual& o) noexcept { *this = *this - o; return *this; }
    constexpr dual& operator*=(const dual& o) noexcept { *this = *this * o; return *this; }
    constexpr dual& operator/=(const dual& o) noexcept { *this = *this / o; return *this; }

    bool operator==(const dual& o) const noexcept { return val == o.val && deriv == o.deriv; }
    bool operator!=(const dual& o) const noexcept { return !(*this == o); }
};

// Scalar operands
template <typename T> constexpr dual<T> operator+(T a, const dual<T>& b) noexcept { return dual<T>(a + b.val, b.deriv); }
template <typename T> constexpr dual<T> operator-(T a, const dual<T>& b) noexcept { return dual<T>(a - b.val, -b.deriv); }
template <typename T> constexpr dual<T> operator*(T a, const dual<T>& b) noexcept { return dual<T>(a * b.val, a * b.deriv); }
template <typename T> constexpr dual<T> operator/(T a, const dual<T>& b) noexcept {
    T inv = T(1) / b.val;
    return dual<T>(a * inv, -a * inv * b.deriv * inv);
}
template <typename T> constexpr dual<T> operator+(const dual<T>& a, T b) noexcept { return dual<T>(a.val + b, a.deriv); }
template <typename T> constexpr dual<T> operator-(const dual<T>& a, T b) noexcept { return dual<T>(a.val - b, a.deriv); }
template <typename T> constexpr dual<T> operator*(const dual<T>& a, T b) noexcept { return dual<T>(a.val * b, a.deriv * b); }
template <typename T> constexpr dual<T> operator/(const dual<T>& a, T b) noexcept { return dual<T>(a.val / b, a.deriv / b); }

// ── Elementary functions ───────────────────────────────────────────
template <typename T>
dual<T> sin(const dual<T>& a) { return dual<T>(std::sin(a.val), std::cos(a.val) * a.deriv); }
template <typename T>
dual<T> cos(const dual<T>& a) { return dual<T>(std::cos(a.val), -std::sin(a.val) * a.deriv); }
template <typename T>
dual<T> tan(const dual<T>& a) { T s = std::sin(a.val), c = std::cos(a.val); return dual<T>(s/c, a.deriv / (c*c)); }
template <typename T>
dual<T> exp(const dual<T>& a) { return dual<T>(std::exp(a.val), std::exp(a.val) * a.deriv); }
template <typename T>
dual<T> log(const dual<T>& a) { return dual<T>(std::log(a.val), a.deriv / a.val); }
template <typename T>
dual<T> sqrt(const dual<T>& a) { T r = std::sqrt(a.val); return dual<T>(r, T(0.5) * a.deriv / r); }
template <typename T>
dual<T> cbrt(const dual<T>& a) { T r = std::cbrt(a.val); return dual<T>(r, a.deriv / (T(3) * r * r)); }
template <typename T>
dual<T> abs(const dual<T>& a) { return dual<T>(std::abs(a.val), (a.val >= T(0) ? T(1) : T(-1)) * a.deriv); }
template <typename T>
dual<T> pow(const dual<T>& a, const dual<T>& b) {
    T p = std::pow(a.val, b.val);
    return dual<T>(p, p * (b.deriv * std::log(a.val) + b.val * a.deriv / a.val));
}
template <typename T>
dual<T> pow(T a, const dual<T>& b) { return pow(dual<T>(a, T(0)), b); }
template <typename T>
dual<T> pow(const dual<T>& a, T b) { return pow(a, dual<T>(b, T(0))); }
template <typename T>
dual<T> atan2(const dual<T>& y, const dual<T>& x) {
    T denom = x.val*x.val + y.val*y.val;
    return dual<T>(std::atan2(y.val, x.val), (x.val * y.deriv - y.val * x.deriv) / denom);
}
template <typename T>
dual<T> hypot(const dual<T>& x, const dual<T>& y) { return sqrt(x*x + y*y); }

// ── Dual vector (generic length) ────────────────────────────────────
template <int L, typename T>
struct dual_vec {
    vec_t<L, dual<T>> components;

    constexpr dual_vec() noexcept : components(T(0)) {}
    constexpr dual_vec(const vec_t<L, T>& val, const vec_t<L, T>& deriv) noexcept {
        for (int i = 0; i < L; ++i) components[i] = dual<T>(val[i], deriv[i]);
    }
    explicit constexpr dual_vec(const vec_t<L, T>& val) noexcept : components(dual<T>(val[0],T(0)), dual<T>(val[1],T(0)), dual<T>(val[2],T(0))) {}

    vec_t<L, T> val() const noexcept { vec_t<L,T> r; for(int i=0;i<L;++i) r[i]=components[i].val; return r; }
    vec_t<L, T> deriv() const noexcept { vec_t<L,T> r; for(int i=0;i<L;++i) r[i]=components[i].deriv; return r; }

    dual_vec operator+(const dual_vec& o) const noexcept { dual_vec r; for(int i=0;i<L;++i) r.components[i]=components[i]+o.components[i]; return r; }
    dual_vec operator-(const dual_vec& o) const noexcept { dual_vec r; for(int i=0;i<L;++i) r.components[i]=components[i]-o.components[i]; return r; }
    dual_vec operator*(T s) const noexcept { dual_vec r; for(int i=0;i<L;++i) r.components[i]=components[i]*s; return r; }
    friend dual_vec operator*(T s, const dual_vec& v) noexcept { dual_vec r; for(int i=0;i<L;++i) r.components[i]=s*v.components[i]; return r; }
};

template <int L, typename T> dual_vec<L,T> dot(const dual_vec<L,T>& a, const dual_vec<L,T>& b) noexcept {
    dual<T> sum(T(0));
    for(int i=0;i<L;++i) sum = sum + a.components[i]*b.components[i];
    return dual_vec<L,T>(vec_t<L,T>(sum.val), vec_t<L,T>(sum.deriv));
}
template <typename T>
dual_vec<3,T> cross(const dual_vec<3,T>& a, const dual_vec<3,T>& b) noexcept {
    return dual_vec<3,T>(cross(a.val(), b.val()), cross(a.deriv(), b.val()) + cross(a.val(), b.deriv()));
}
template <int L, typename T>
dual<T> length(const dual_vec<L,T>& v) noexcept { return sqrt(dot(v,v).components[0]); }
template <int L, typename T>
dual_vec<L,T> normalize(const dual_vec<L,T>& v) noexcept { auto l = length(v); return v * (T(1) / l.val); }

// Convenience aliases
using dualf = dual<float>;
using duald = dual<double>;
using dual_vec3f = dual_vec<3, float>;
using dual_vec3d = dual_vec<3, double>;

// ── Gradient evaluator for scalar function f(x) via dual ───────────
template <typename T, typename Func>
T finite_difference_gradient(Func f, const vec_t<3,T>& x, T h = T(1e-4)) {
    // use central difference for each component; not dual but often used.
    // Not implemented; focus on dual.
    return T(0);
}

// ── Simple Jacobian via dual numbers: compute Jacobian of vector function F: R^n -> R^m ──
// This function evaluates F on inputs provided as dual numbers; the derivative part of output
// corresponds to directional derivative. For full Jacobian, call with each basis vector as seed.

// Helper: make a dual vector with seed derivative in k-th component
template <int N, typename T>
dual_vec<N,T> make_seed(const vec_t<N,T>& x, int k) {
    vec_t<N,T> d(T(0));
    d[k] = T(1);
    return dual_vec<N,T>(x, d);
}

} // namespace wp