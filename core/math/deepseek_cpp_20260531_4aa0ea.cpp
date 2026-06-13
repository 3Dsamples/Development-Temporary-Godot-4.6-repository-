// SPDX-FileCopyrightText: Copyright (c) 2025 – C++17 Math/Physics Library
// SPDX-License-Identifier: MIT
#pragma once

#include "builtin.hpp"
#include <cassert>
#include <initializer_list>
#include <algorithm>
#include <numeric>

namespace wp {

// ── Generic fixed‑size vector (Length >= 1) ──
template <int Length, typename Type>
struct alignas(sizeof(Type) * Length) vec_t {
    static_assert(Length >= 1, "Vector length must be >= 1");

    Type c[Length];

    // ── Constructors ──
    constexpr vec_t() noexcept : c{} {}

    constexpr explicit vec_t(Type s) noexcept {
        for (int i = 0; i < Length; ++i) c[i] = s;
    }

    template <typename OtherType>
    constexpr explicit vec_t(const vec_t<Length, OtherType>& other) noexcept {
        for (int i = 0; i < Length; ++i) c[i] = static_cast<Type>(other[i]);
    }

    constexpr vec_t(std::initializer_list<Type> il) noexcept {
        int i = 0;
        for (auto v : il) { if (i < Length) c[i++] = v; }
        for (; i < Length; ++i) c[i] = Type(0);
    }

    // Convenience constructors for common lengths
    constexpr vec_t(Type x, Type y) noexcept : c{x, y} { static_assert(Length == 2); }
    constexpr vec_t(Type x, Type y, Type z) noexcept : c{x, y, z} { static_assert(Length == 3); }
    constexpr vec_t(Type x, Type y, Type z, Type w) noexcept : c{x, y, z, w} { static_assert(Length == 4); }

    // ── Element access ──
    constexpr Type  operator[](int i) const noexcept { assert(i >= 0 && i < Length); return c[i]; }
    constexpr Type& operator[](int i)       noexcept { assert(i >= 0 && i < Length); return c[i]; }

    // ── Iterators ──
    constexpr const Type* begin() const noexcept { return c; }
    constexpr const Type* end()   const noexcept { return c + Length; }
    constexpr Type*       begin()       noexcept { return c; }
    constexpr Type*       end()         noexcept { return c + Length; }
};

// ── Common type aliases (matching Warp naming) ──
using vec2b  = vec_t<2, int8>;    using vec3b  = vec_t<3, int8>;    using vec4b  = vec_t<4, int8>;
using vec2ub = vec_t<2, uint8>;   using vec3ub = vec_t<3, uint8>;   using vec4ub = vec_t<4, uint8>;
using vec2s  = vec_t<2, int16>;   using vec3s  = vec_t<3, int16>;   using vec4s  = vec_t<4, int16>;
using vec2us = vec_t<2, uint16>;  using vec3us = vec_t<3, uint16>;  using vec4us = vec_t<4, uint16>;
using vec2i  = vec_t<2, int32>;   using vec3i  = vec_t<3, int32>;   using vec4i  = vec_t<4, int32>;
using vec2ui = vec_t<2, uint32>;  using vec3ui = vec_t<3, uint32>;  using vec4ui = vec_t<4, uint32>;
using vec2l  = vec_t<2, int64>;   using vec3l  = vec_t<3, int64>;   using vec4l  = vec_t<4, int64>;
using vec2ul = vec_t<2, uint64>;  using vec3ul = vec_t<3, uint64>;  using vec4ul = vec_t<4, uint64>;
using vec2h  = vec_t<2, half>;    using vec3h  = vec_t<3, half>;    using vec4h  = vec_t<4, half>;
using vec2   = vec_t<2, float>;   using vec3   = vec_t<3, float>;   using vec4   = vec_t<4, float>;
using vec2d  = vec_t<2, double>;  using vec3d  = vec_t<3, double>;  using vec4d  = vec_t<4, double>;

// ── Arithmetic operators ──
#define WP_VEC_UNARY_OP(OP) \
    template <int L, typename T> constexpr vec_t<L,T> operator OP (const vec_t<L,T>& a) noexcept { \
        vec_t<L,T> r; for (int i=0;i<L;++i) r[i] = OP a[i]; return r; \
    }
WP_VEC_UNARY_OP(-)
WP_VEC_UNARY_OP(+)
#undef WP_VEC_UNARY_OP

#define WP_VEC_BIN_OP(OP) \
    template <int L, typename T> constexpr vec_t<L,T> operator OP (const vec_t<L,T>& a, const vec_t<L,T>& b) noexcept { \
        vec_t<L,T> r; for (int i=0;i<L;++i) r[i] = a[i] OP b[i]; return r; \
    } \
    template <int L, typename T> constexpr vec_t<L,T> operator OP (const vec_t<L,T>& a, T s) noexcept { \
        vec_t<L,T> r; for (int i=0;i<L;++i) r[i] = a[i] OP s; return r; \
    } \
    template <int L, typename T> constexpr vec_t<L,T> operator OP (T s, const vec_t<L,T>& a) noexcept { \
        vec_t<L,T> r; for (int i=0;i<L;++i) r[i] = s OP a[i]; return r; \
    }
WP_VEC_BIN_OP(+)
WP_VEC_BIN_OP(-)
WP_VEC_BIN_OP(*)
WP_VEC_BIN_OP(/ )
#undef WP_VEC_BIN_OP

#define WP_VEC_COMPOUND(OP) \
    template <int L, typename T> constexpr vec_t<L,T>& operator OP##= (vec_t<L,T>& a, const vec_t<L,T>& b) noexcept { \
        for (int i=0;i<L;++i) a[i] OP##= b[i]; return a; \
    } \
    template <int L, typename T> constexpr vec_t<L,T>& operator OP##= (vec_t<L,T>& a, T s) noexcept { \
        for (int i=0;i<L;++i) a[i] OP##= s; return a; \
    }
WP_VEC_COMPOUND(+)
WP_VEC_COMPOUND(-)
WP_VEC_COMPOUND(*)
WP_VEC_COMPOUND(/ )
#undef WP_VEC_COMPOUND

template <int L, typename T> constexpr bool operator==(const vec_t<L,T>& a, const vec_t<L,T>& b) noexcept {
    for (int i=0;i<L;++i) if (a[i] != b[i]) return false;
    return true;
}
template <int L, typename T> constexpr bool operator!=(const vec_t<L,T>& a, const vec_t<L,T>& b) noexcept {
    return !(a == b);
}

// ── Geometric operations ──
template <int L, typename T> constexpr T dot(const vec_t<L,T>& a, const vec_t<L,T>& b) noexcept {
    T s = 0; for (int i=0;i<L;++i) s += a[i] * b[i]; return s;
}

template <typename T> constexpr vec_t<3,T> cross(const vec_t<3,T>& a, const vec_t<3,T>& b) noexcept {
    return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
}

template <int L, typename T> constexpr T length_sq(const vec_t<L,T>& v) noexcept { return dot(v,v); }

template <int L, typename T> T length(const vec_t<L,T>& v) noexcept { return std::sqrt(length_sq(v)); }

template <int L, typename T> vec_t<L,T> normalize(const vec_t<L,T>& v) noexcept {
    T l = length(v);
    if (l > Constants<T>::epsilon) return v / l;
    return vec_t<L,T>(T(0));
}

template <int L, typename T> constexpr T distance_sq(const vec_t<L,T>& a, const vec_t<L,T>& b) noexcept {
    return length_sq(a - b);
}
template <int L, typename T> T distance(const vec_t<L,T>& a, const vec_t<L,T>& b) noexcept {
    return length(a - b);
}

template <int L, typename T> constexpr vec_t<L,T> min(const vec_t<L,T>& a, const vec_t<L,T>& b) noexcept {
    vec_t<L,T> r; for (int i=0;i<L;++i) r[i] = a[i] < b[i] ? a[i] : b[i]; return r;
}
template <int L, typename T> constexpr vec_t<L,T> max(const vec_t<L,T>& a, const vec_t<L,T>& b) noexcept {
    vec_t<L,T> r; for (int i=0;i<L;++i) r[i] = a[i] > b[i] ? a[i] : b[i]; return r;
}

template <int L, typename T> constexpr vec_t<L,T> abs(const vec_t<L,T>& v) noexcept {
    vec_t<L,T> r; for (int i=0;i<L;++i) r[i] = std::abs(v[i]); return r;
}

// ── Outer product (vec3 → mat33) ──
template <typename T> mat_t<3,3,T> outer(const vec_t<3,T>& a, const vec_t<3,T>& b) noexcept;

} // namespace wp