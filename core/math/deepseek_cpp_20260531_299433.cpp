// SPDX-FileCopyrightText: Copyright (c) 2025 – C++17 Math/Physics Library
// SPDX-License-Identifier: MIT
#pragma once

#include "mat.hpp"

namespace wp {

template <typename Type>
struct alignas(4 * sizeof(Type)) quat_t {
    Type x, y, z, w;   // imaginary (x,y,z), real (w)

    constexpr quat_t() noexcept : x(0), y(0), z(0), w(1) {}
    constexpr quat_t(Type x_, Type y_, Type z_, Type w_) noexcept : x(x_), y(y_), z(z_), w(w_) {}
    explicit constexpr quat_t(const vec_t<3, Type>& v, Type w_ = Type(0)) noexcept : x(v[0]), y(v[1]), z(v[2]), w(w_) {}

    template <typename Other> explicit constexpr quat_t(const quat_t<Other>& q) noexcept
        : x(static_cast<Type>(q.x)), y(static_cast<Type>(q.y)), z(static_cast<Type>(q.z)), w(static_cast<Type>(q.w)) {}

    constexpr Type  operator[](int i) const noexcept {
        switch (i) { case 0: return x; case 1: return y; case 2: return z; case 3: return w; }
        return x;
    }
    constexpr Type& operator[](int i) noexcept {
        switch (i) { case 0: return x; case 1: return y; case 2: return z; case 3: return w; }
        return x;
    }

    constexpr vec_t<3, Type> imag() const noexcept { return {x, y, z}; }
};

using quat   = quat_t<float>;
using quath  = quat_t<half>;
using quatf  = quat_t<float>;
using quatd  = quat_t<double>;

// ── Arithmetic ──
template <typename T> constexpr quat_t<T> operator+(const quat_t<T>& a, const quat_t<T>& b) noexcept {
    return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w};
}
template <typename T> constexpr quat_t<T> operator-(const quat_t<T>& a, const quat_t<T>& b) noexcept {
    return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w};
}
template <typename T> constexpr quat_t<T> operator*(const quat_t<T>& a, T s) noexcept {
    return {a.x*s, a.y*s, a.z*s, a.w*s};
}
template <typename T> constexpr quat_t<T> operator*(T s, const quat_t<T>& a) noexcept { return a * s; }
template <typename T> constexpr quat_t<T> operator/(const quat_t<T>& a, T s) noexcept { return a * safe_rcp(s); }

// ── Quaternion multiplication (Hamilton product) ──
template <typename T> constexpr quat_t<T> mul(const quat_t<T>& a, const quat_t<T>& b) noexcept {
    return {
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    };
}

// ── Conjugate, norm, normalize ──
template <typename T> constexpr quat_t<T> conjugate(const quat_t<T>& q) noexcept { return {-q.x, -q.y, -q.z, q.w}; }

template <typename T> constexpr T norm_sq(const quat_t<T>& q) noexcept { return q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w; }

template <typename T> T norm(const quat_t<T>& q) noexcept { return std::sqrt(norm_sq(q)); }

template <typename T> quat_t<T> normalize(const quat_t<T>& q) noexcept {
    T n = norm(q);
    return (n > Constants<T>::epsilon) ? q / n : quat_t<T>{};
}

// ── Inverse ──
template <typename T> quat_t<T> inverse(const quat_t<T>& q) noexcept {
    return conjugate(q) / norm_sq(q);
}

// ── Rotate a 3D vector ──
template <typename T> vec_t<3,T> rotate(const quat_t<T>& q, const vec_t<3,T>& v) noexcept {
    quat_t<T> p(v, T(0));
    quat_t<T> r = mul(mul(q, p), conjugate(q));
    return {r.x, r.y, r.z};
}

// ── Spherical linear interpolation (slerp) ──
template <typename T> quat_t<T> slerp(const quat_t<T>& qa, const quat_t<T>& qb, T t) noexcept {
    T cos_theta = qa.x*qb.x + qa.y*qb.y + qa.z*qb.z + qa.w*qb.w;
    quat_t<T> qb2 = qb;
    if (cos_theta < T(0)) { cos_theta = -cos_theta; qb2 = -qb2; }
    T k0, k1;
    if (cos_theta > T(1) - Constants<T>::epsilon) {
        k0 = T(1) - t; k1 = t;
    } else {
        T sin_theta = std::sqrt(T(1) - cos_theta*cos_theta);
        T theta = std::atan2(sin_theta, cos_theta);
        T inv_sin = T(1) / sin_theta;
        k0 = std::sin((T(1)-t)*theta) * inv_sin;
        k1 = std::sin(t*theta) * inv_sin;
    }
    return qa * k0 + qb2 * k1;
}

// ── Conversion to/from rotation matrix ──
template <typename T> mat_t<3,3,T> to_matrix(const quat_t<T>& q) noexcept {
    T xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z;
    T xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z;
    T wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;
    mat_t<3,3,T> r;
    r(0,0)=T(1)-T(2)*(yy+zz); r(0,1)=T(2)*(xy-wz);     r(0,2)=T(2)*(xz+wy);
    r(1,0)=T(2)*(xy+wz);     r(1,1)=T(1)-T(2)*(xx+zz); r(1,2)=T(2)*(yz-wx);
    r(2,0)=T(2)*(xz-wy);     r(2,1)=T(2)*(yz+wx);     r(2,2)=T(1)-T(2)*(xx+yy);
    return r;
}

template <typename T> quat_t<T> from_matrix(const mat_t<3,3,T>& m) noexcept {
    T tr = m(0,0)+m(1,1)+m(2,2);
    if (tr > T(0)) {
        T s = std::sqrt(tr+T(1)) * T(2);
        return {(m(1,2)-m(2,1))/s, (m(2,0)-m(0,2))/s, (m(0,1)-m(1,0))/s, s/T(4)};
    } else if (m(0,0)>m(1,1) && m(0,0)>m(2,2)) {
        T s = std::sqrt(T(1)+m(0,0)-m(1,1)-m(2,2)) * T(2);
        return {s/T(4), (m(0,1)+m(1,0))/s, (m(2,0)+m(0,2))/s, (m(1,2)-m(2,1))/s};
    } else if (m(1,1)>m(2,2)) {
        T s = std::sqrt(T(1)+m(1,1)-m(0,0)-m(2,2)) * T(2);
        return {(m(0,1)+m(1,0))/s, s/T(4), (m(1,2)+m(2,1))/s, (m(2,0)-m(0,2))/s};
    } else {
        T s = std::sqrt(T(1)+m(2,2)-m(0,0)-m(1,1)) * T(2);
        return {(m(2,0)+m(0,2))/s, (m(1,2)+m(2,1))/s, s/T(4), (m(0,1)-m(1,0))/s};
    }
}

} // namespace wp