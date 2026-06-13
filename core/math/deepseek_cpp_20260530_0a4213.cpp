// File 0037 : core/math/matrix_decomp.h
// LU decomposition with partial pivoting for 3×3 and 4×4 matrices, linear system solving, and determinant.

#pragma once

#include "mat3.h"
#include "mat4.h"
#include "vec3.h"
#include "vec4.h"
#include "constants.h"
#include <algorithm>
#include <cmath>

namespace wp {

// ── 3×3 LU decomposition ───────────────────────────────────────────
// Returns permutation matrix P, unit lower triangular L, upper triangular U.
// A = P * L * U   (if P is the permutation, it's applied as A[P[i], :] = L*U? Usually we store permutation vector.)
// We store permutation as an array of row indices perm[3].
template <typename T>
void lu_decompose(const mat3<T>& A, mat3<T>& L, mat3<T>& U, int perm[3]) {
    // Make a copy of A for factorization
    L = identity3<T>();
    U = A;
    for (int i = 0; i < 3; ++i) perm[i] = i;

    for (int j = 0; j < 3; ++j) {
        // Pivot
        int max_row = j;
        T max_val = std::abs(U(j, j));
        for (int i = j + 1; i < 3; ++i) {
            T val = std::abs(U(i, j));
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }
        if (max_val < epsilon<T>) continue; // singular
        if (max_row != j) {
            // Swap rows in U and L (only columns below diagonal? We'll swap rows in L for columns < j, and swap whole rows in U)
            for (int k = 0; k < 3; ++k) {
                std::swap(U(j, k), U(max_row, k));
            }
            // Swap rows in L up to j-1 (since L has 1 on diagonal)
            for (int k = 0; k < j; ++k) {
                std::swap(L(j, k), L(max_row, k));
            }
            std::swap(perm[j], perm[max_row]);
        }
        // Eliminate
        for (int i = j + 1; i < 3; ++i) {
            T factor = U(i, j) / U(j, j);
            L(i, j) = factor;
            for (int k = j; k < 3; ++k) {
                U(i, k) -= factor * U(j, k);
            }
        }
    }
}

// ── Solve Ax = b using LU decomposition (3×3) ───────────────────────
template <typename T>
vec3<T> solve_linear_3x3(const mat3<T>& A, const vec3<T>& b) {
    mat3<T> L, U;
    int perm[3];
    lu_decompose(A, L, U, perm);

    // Apply permutation to b: Pb = b[perm]
    vec3<T> Pb( b[perm[0]], b[perm[1]], b[perm[2]] );

    // Forward substitution: L y = Pb
    vec3<T> y;
    y[0] = Pb[0];
    y[1] = Pb[1] - L(1,0) * y[0];
    y[2] = Pb[2] - L(2,0) * y[0] - L(2,1) * y[1];

    // Back substitution: U x = y
    vec3<T> x;
    x[2] = y[2] / U(2,2);
    x[1] = (y[1] - U(1,2) * x[2]) / U(1,1);
    x[0] = (y[0] - U(0,1) * x[1] - U(0,2) * x[2]) / U(0,0);
    return x;
}

// ── 4×4 LU decomposition ───────────────────────────────────────────
template <typename T>
void lu_decompose(const mat4<T>& A, mat4<T>& L, mat4<T>& U, int perm[4]) {
    L = identity4<T>();
    U = A;
    for (int i = 0; i < 4; ++i) perm[i] = i;

    for (int j = 0; j < 4; ++j) {
        int max_row = j;
        T max_val = std::abs(U(j, j));
        for (int i = j + 1; i < 4; ++i) {
            T val = std::abs(U(i, j));
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }
        if (max_val < epsilon<T>) continue;
        if (max_row != j) {
            for (int k = 0; k < 4; ++k) std::swap(U(j, k), U(max_row, k));
            for (int k = 0; k < j; ++k) std::swap(L(j, k), L(max_row, k));
            std::swap(perm[j], perm[max_row]);
        }
        for (int i = j + 1; i < 4; ++i) {
            T factor = U(i, j) / U(j, j);
            L(i, j) = factor;
            for (int k = j; k < 4; ++k) U(i, k) -= factor * U(j, k);
        }
    }
}

// ── Solve Ax = b using LU decomposition (4×4) ───────────────────────
template <typename T>
vec4<T> solve_linear_4x4(const mat4<T>& A, const vec4<T>& b) {
    mat4<T> L, U;
    int perm[4];
    lu_decompose(A, L, U, perm);

    vec4<T> Pb;
    for (int i = 0; i < 4; ++i) Pb[i] = b[perm[i]];

    vec4<T> y;
    y[0] = Pb[0];
    y[1] = Pb[1] - L(1,0) * y[0];
    y[2] = Pb[2] - L(2,0) * y[0] - L(2,1) * y[1];
    y[3] = Pb[3] - L(3,0) * y[0] - L(3,1) * y[1] - L(3,2) * y[2];

    vec4<T> x;
    x[3] = y[3] / U(3,3);
    x[2] = (y[2] - U(2,3) * x[3]) / U(2,2);
    x[1] = (y[1] - U(1,2) * x[2] - U(1,3) * x[3]) / U(1,1);
    x[0] = (y[0] - U(0,1) * x[1] - U(0,2) * x[2] - U(0,3) * x[3]) / U(0,0);
    return x;
}

// ── Determinant via LU (product of U diagonal, sign from permutation) ──
template <typename T>
T determinant_lu(const mat3<T>& A) {
    mat3<T> L, U;
    int perm[3];
    lu_decompose(A, L, U, perm);
    // compute permutation sign
    int sign = 1;
    int p[3] = {perm[0], perm[1], perm[2]};
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (p[i] > p[j]) sign = -sign;
    return sign * U(0,0) * U(1,1) * U(2,2);
}

template <typename T>
T determinant_lu(const mat4<T>& A) {
    mat4<T> L, U;
    int perm[4];
    lu_decompose(A, L, U, perm);
    int sign = 1;
    int p[4] = {perm[0], perm[1], perm[2], perm[3]};
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            if (p[i] > p[j]) sign = -sign;
    return sign * U(0,0) * U(1,1) * U(2,2) * U(3,3);
}

} // namespace wp