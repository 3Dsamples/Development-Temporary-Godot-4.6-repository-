// File 0038 : core/math/cholesky.h
// Cholesky decomposition (LLT) for 3×3 and 4×4 symmetric positive‑definite matrices, solving linear systems.

#pragma once

#include "mat3.h"
#include "mat4.h"
#include "vec3.h"
#include "vec4.h"
#include "constants.h"
#include <cmath>

namespace wp {

// ── Cholesky decomposition of a 3×3 SPD matrix A → L where A = L * Lᵗ ──
// Returns true if A was successfully decomposed, false if not SPD.
template <typename T>
bool cholesky_decompose(const mat3<T>& A, mat3<T>& L) {
    // Copy A to L for in‑place factorization
    L = A;
    for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
            T sum = L(i, j);
            for (int k = 0; k < i; ++k)
                sum -= L(i, k) * L(j, k);
            if (i == j) {
                if (sum <= T(0)) return false; // not positive definite
                L(i, i) = std::sqrt(sum);
                // Scale column i by 1 / L(i,i) to avoid later divisions (classic variant)
                // Instead, we store the diagonal factor and later use L(i,i) directly.
                // For Cholesky-Banachiewicz, we only fill lower triangle.
                T inv = T(1) / L(i, i);
                for (int r = j + 1; r < 3; ++r) {
                    T s = L(r, i);
                    for (int k = 0; k < i; ++k) s -= L(r, k) * L(i, k);
                    L(r, i) = s * inv;
                }
                // Actually we should compute column i entries below diagonal directly.
                // The above approach is mixed. Let's implement the standard Cholesky‑Banachiewicz:
            } else {
                // j > i, we are computing L(j,i) later? Standard method:
                // We need to fill only lower triangle L(j,i) for j>=i.
                // We'll rewrite using nested loops correctly.
            }
        }
    }
    // To avoid confusion, rewrite properly:
    L = A;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j) {
            T sum = L(i, j);
            for (int k = 0; k < j; ++k)
                sum -= L(i, k) * L(j, k);
            L(i, j) = sum / L(j, j);
        }
        T sum = L(i, i);
        for (int k = 0; k < i; ++k)
            sum -= L(i, k) * L(i, k);
        if (sum <= T(0)) return false;
        L(i, i) = std::sqrt(sum);
    }
    // Zero out upper triangle for clarity
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            L(i, j) = T(0);
    return true;
}

// ── Solve 3×3 linear system A*x = b using Cholesky factors (L*Lᵗ) ──
// L must be the lower triangular factor from Cholesky.
template <typename T>
vec3<T> cholesky_solve(const mat3<T>& L, const vec3<T>& b) {
    // Forward substitution: L * y = b
    vec3<T> y;
    y[0] = b[0] / L(0,0);
    y[1] = (b[1] - L(1,0) * y[0]) / L(1,1);
    y[2] = (b[2] - L(2,0) * y[0] - L(2,1) * y[1]) / L(2,2);
    // Back substitution: Lᵗ * x = y
    vec3<T> x;
    x[2] = y[2] / L(2,2);
    x[1] = (y[1] - L(2,1) * x[2]) / L(1,1);
    x[0] = (y[0] - L(2,0) * x[2] - L(1,0) * x[1]) / L(0,0);
    return x;
}

// ── Cholesky decomposition of a 4×4 SPD matrix ──────────────────────
template <typename T>
bool cholesky_decompose(const mat4<T>& A, mat4<T>& L) {
    L = A;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < i; ++j) {
            T sum = L(i, j);
            for (int k = 0; k < j; ++k)
                sum -= L(i, k) * L(j, k);
            L(i, j) = sum / L(j, j);
        }
        T sum = L(i, i);
        for (int k = 0; k < i; ++k)
            sum -= L(i, k) * L(i, k);
        if (sum <= T(0)) return false;
        L(i, i) = std::sqrt(sum);
    }
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            L(i, j) = T(0);
    return true;
}

// ── Solve 4×4 linear system using Cholesky factors ──────────────────
template <typename T>
vec4<T> cholesky_solve(const mat4<T>& L, const vec4<T>& b) {
    vec4<T> y;
    y[0] = b[0] / L(0,0);
    y[1] = (b[1] - L(1,0) * y[0]) / L(1,1);
    y[2] = (b[2] - L(2,0) * y[0] - L(2,1) * y[1]) / L(2,2);
    y[3] = (b[3] - L(3,0) * y[0] - L(3,1) * y[1] - L(3,2) * y[2]) / L(3,3);

    vec4<T> x;
    x[3] = y[3] / L(3,3);
    x[2] = (y[2] - L(3,2) * x[3]) / L(2,2);
    x[1] = (y[1] - L(3,1) * x[3] - L(2,1) * x[2]) / L(1,1);
    x[0] = (y[0] - L(3,0) * x[3] - L(2,0) * x[2] - L(1,0) * x[1]) / L(0,0);
    return x;
}

} // namespace wp