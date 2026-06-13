// File 0038 : core/math/cholesky.h
// Cholesky decomposition (LLᵀ) for 3×3 and 4×4 symmetric positive‑definite matrices, and linear system solving.

#pragma once

#include "mat3.h"
#include "mat4.h"
#include "vec3.h"
#include "vec4.h"
#include "constants.h"
#include <cmath>

namespace wp {

// ── Cholesky decomposition of a 3×3 SPD matrix A = L·Lᵀ ─────────────
// Stores the lower‑triangular factor L (including diagonal).
// Returns true if A is numerically positive definite.
template <typename T>
bool cholesky_decompose(const mat3<T>& A, mat3<T>& L) {
    L = A;  // work in place

    // Row i
    for (int i = 0; i < 3; ++i) {
        // Compute off‑diagonal elements of column i (below diagonal)
        for (int j = i; j < 3; ++j) {   // actually we need only j > i later, but we'll compute L(j,i) from row j
            T sum = L(j, i);
            for (int k = 0; k < i; ++k) {
                sum -= L(j, k) * L(i, k);
            }
            if (j == i) {
                if (sum <= T(0)) return false;      // not positive definite
                L(i, i) = std::sqrt(sum);
            } else {
                L(j, i) = sum / L(i, i);
                L(i, j) = T(0);                     // zero upper triangle for clarity
            }
        }
    }

    // Ensure the upper triangle is exactly zero
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            L(i, j) = T(0);
    return true;
}

// ── Solve A·x = b using the Cholesky factor L (A = L·Lᵀ) ────────────
template <typename T>
vec3<T> cholesky_solve(const mat3<T>& L, const vec3<T>& b) {
    // Forward substitution: L·y = b
    vec3<T> y;
    y[0] = b[0] / L(0, 0);
    y[1] = (b[1] - L(1, 0) * y[0]) / L(1, 1);
    y[2] = (b[2] - L(2, 0) * y[0] - L(2, 1) * y[1]) / L(2, 2);

    // Back substitution: Lᵀ·x = y
    vec3<T> x;
    x[2] = y[2] / L(2, 2);
    x[1] = (y[1] - L(2, 1) * x[2]) / L(1, 1);
    x[0] = (y[0] - L(2, 0) * x[2] - L(1, 0) * x[1]) / L(0, 0);
    return x;
}

// ── 4×4 Cholesky decomposition ──────────────────────────────────────
template <typename T>
bool cholesky_decompose(const mat4<T>& A, mat4<T>& L) {
    L = A;

    for (int i = 0; i < 4; ++i) {
        for (int j = i; j < 4; ++j) {
            T sum = L(j, i);
            for (int k = 0; k < i; ++k) {
                sum -= L(j, k) * L(i, k);
            }
            if (j == i) {
                if (sum <= T(0)) return false;
                L(i, i) = std::sqrt(sum);
            } else {
                L(j, i) = sum / L(i, i);
                L(i, j) = T(0);
            }
        }
    }

    // Clear upper triangle
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            L(i, j) = T(0);
    return true;
}

// ── Solve 4×4 system using Cholesky factor ──────────────────────────
template <typename T>
vec4<T> cholesky_solve(const mat4<T>& L, const vec4<T>& b) {
    // Forward: L·y = b
    vec4<T> y;
    y[0] = b[0] / L(0, 0);
    y[1] = (b[1] - L(1, 0) * y[0]) / L(1, 1);
    y[2] = (b[2] - L(2, 0) * y[0] - L(2, 1) * y[1]) / L(2, 2);
    y[3] = (b[3] - L(3, 0) * y[0] - L(3, 1) * y[1] - L(3, 2) * y[2]) / L(3, 3);

    // Backward: Lᵀ·x = y
    vec4<T> x;
    x[3] = y[3] / L(3, 3);
    x[2] = (y[2] - L(3, 2) * x[3]) / L(2, 2);
    x[1] = (y[1] - L(3, 1) * x[3] - L(2, 1) * x[2]) / L(1, 1);
    x[0] = (y[0] - L(3, 0) * x[3] - L(2, 0) * x[2] - L(1, 0) * x[1]) / L(0, 0);
    return x;
}

} // namespace wp