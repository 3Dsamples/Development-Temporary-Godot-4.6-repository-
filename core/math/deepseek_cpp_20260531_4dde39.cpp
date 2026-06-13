// SPDX-FileCopyrightText: Copyright (c) 2025 – C++17 Math/Physics Library
// SPDX-License-Identifier: MIT
#pragma once

#include "vec.hpp"
#include <cassert>

namespace wp {

template <int Rows, int Cols, typename Type>
struct alignas(sizeof(Type) * Rows * Cols) mat_t {
    static_assert(Rows >= 1 && Cols >= 1, "Matrix dimensions must be >= 1");
    Type data[Rows][Cols];   // column‑major storage for cache efficiency on GPU/CPU

    // ── Constructors ──
    constexpr mat_t() noexcept : data{} {}

    constexpr explicit mat_t(Type s) noexcept {
        for (int i = 0; i < Rows; ++i)
            for (int j = 0; j < Cols; ++j)
                data[i][j] = (i == j) ? s : Type(0);  // diagonal by default
    }

    // Matrix from column vectors
    template <int... Is>
    constexpr mat_t(const vec_t<Rows, Type>&... cols) noexcept : data{} {
        static_assert(sizeof...(Is) == Cols, "Column count mismatch");
        const vec_t<Rows, Type>* arr[] = {&cols...};
        for (int j = 0; j < Cols; ++j)
            for (int i = 0; i < Rows; ++i)
                data[i][j] = (*arr[j])[i];
    }

    // ── Element access ──
    constexpr Type  operator()(int row, int col) const noexcept { assert(row>=0&&row<Rows&&col>=0&&col<Cols); return data[row][col]; }
    constexpr Type& operator()(int row, int col)       noexcept { assert(row>=0&&row<Rows&&col>=0&&col<Cols); return data[row][col]; }

    // Column access
    constexpr vec_t<Rows, Type> col(int j) const noexcept {
        vec_t<Rows, Type> r;
        for (int i = 0; i < Rows; ++i) r[i] = data[i][j];
        return r;
    }
};

// ── Common matrix aliases ──
using mat22  = mat_t<2,2,float>;   using mat22d = mat_t<2,2,double>;
using mat33  = mat_t<3,3,float>;   using mat33d = mat_t<3,3,double>;
using mat44  = mat_t<4,4,float>;   using mat44d = mat_t<4,4,double>;
using mat22h = mat_t<2,2,half>;    using mat33h = mat_t<3,3,half>;
using mat_nn = mat_t<4,4,float>;   // generic placeholder

// ── Matrix arithmetic ──
template <int R, int C, typename T>
constexpr mat_t<R,C,T> operator+(const mat_t<R,C,T>& a, const mat_t<R,C,T>& b) noexcept {
    mat_t<R,C,T> r;
    for (int i=0;i<R;++i) for (int j=0;j<C;++j) r.data[i][j] = a.data[i][j] + b.data[i][j];
    return r;
}

template <int R, int C, typename T>
constexpr mat_t<R,C,T> operator-(const mat_t<R,C,T>& a, const mat_t<R,C,T>& b) noexcept {
    mat_t<R,C,T> r;
    for (int i=0;i<R;++i) for (int j=0;j<C;++j) r.data[i][j] = a.data[i][j] - b.data[i][j];
    return r;
}

template <int R, int C, typename T>
constexpr mat_t<R,C,T> operator*(const mat_t<R,C,T>& a, T s) noexcept {
    mat_t<R,C,T> r;
    for (int i=0;i<R;++i) for (int j=0;j<C;++j) r.data[i][j] = a.data[i][j] * s;
    return r;
}

template <int R, int C, typename T>
constexpr mat_t<R,C,T> operator*(T s, const mat_t<R,C,T>& a) noexcept { return a * s; }

// ── Matrix–vector multiplication ──
template <int R, int C, typename T>
constexpr vec_t<R,T> mul(const mat_t<R,C,T>& m, const vec_t<C,T>& v) noexcept {
    vec_t<R,T> r(T(0));
    for (int i=0;i<R;++i) for (int j=0;j<C;++j) r[i] += m.data[i][j] * v[j];
    return r;
}

// ── Matrix–matrix multiplication ──
template <int R, int K, int C, typename T>
constexpr mat_t<R,C,T> mul(const mat_t<R,K,T>& a, const mat_t<K,C,T>& b) noexcept {
    mat_t<R,C,T> r(T(0));
    for (int i=0;i<R;++i)
        for (int j=0;j<C;++j)
            for (int k=0;k<K;++k)
                r.data[i][j] += a.data[i][k] * b.data[k][j];
    return r;
}

// ── Transpose ──
template <int R, int C, typename T>
constexpr mat_t<C,R,T> transpose(const mat_t<R,C,T>& m) noexcept {
    mat_t<C,R,T> r;
    for (int i=0;i<R;++i) for (int j=0;j<C;++j) r.data[j][i] = m.data[i][j];
    return r;
}

// ── Identity ──
template <int N, typename T> constexpr mat_t<N,N,T> identity() noexcept {
    mat_t<N,N,T> r(T(0));
    for (int i=0;i<N;++i) r.data[i][i] = T(1);
    return r;
}

// ── Trace, determinant (specialized for 2×2, 3×3, 4×4) ──
template <typename T> constexpr T det(const mat_t<2,2,T>& m) noexcept {
    return m(0,0)*m(1,1) - m(0,1)*m(1,0);
}

template <typename T> constexpr T det(const mat_t<3,3,T>& m) noexcept {
    return m(0,0)*(m(1,1)*m(2,2)-m(1,2)*m(2,1))
         - m(0,1)*(m(1,0)*m(2,2)-m(1,2)*m(2,0))
         + m(0,2)*(m(1,0)*m(2,1)-m(1,1)*m(2,0));
}

template <typename T> constexpr T trace(const mat_t<3,3,T>& m) noexcept {
    return m(0,0) + m(1,1) + m(2,2);
}

// ── Inverse (3×3, analytic) ──
template <typename T> mat_t<3,3,T> inverse(const mat_t<3,3,T>& m) noexcept {
    T d = det(m);
    if (std::abs(d) < Constants<T>::epsilon) return identity<3,T>();
    T inv_d = T(1) / d;
    mat_t<3,3,T> r;
    r(0,0) = (m(1,1)*m(2,2)-m(1,2)*m(2,1)) * inv_d;
    r(0,1) = (m(0,2)*m(2,1)-m(0,1)*m(2,2)) * inv_d;
    r(0,2) = (m(0,1)*m(1,2)-m(0,2)*m(1,1)) * inv_d;
    r(1,0) = (m(1,2)*m(2,0)-m(1,0)*m(2,2)) * inv_d;
    r(1,1) = (m(0,0)*m(2,2)-m(0,2)*m(2,0)) * inv_d;
    r(1,2) = (m(0,2)*m(1,0)-m(0,0)*m(1,2)) * inv_d;
    r(2,0) = (m(1,0)*m(2,1)-m(1,1)*m(2,0)) * inv_d;
    r(2,1) = (m(0,1)*m(2,0)-m(0,0)*m(2,1)) * inv_d;
    r(2,2) = (m(0,0)*m(1,1)-m(0,1)*m(1,0)) * inv_d;
    return r;
}

// ── Outer product (definition) ──
template <typename T>
mat_t<3,3,T> outer(const vec_t<3,T>& a, const vec_t<3,T>& b) noexcept {
    mat_t<3,3,T> r;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) r.data[i][j] = a[i] * b[j];
    return r;
}

} // namespace wp