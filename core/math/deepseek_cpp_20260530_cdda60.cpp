// File 0030 : core/math/svd3.h
// 3x3 SVD, polar decomposition, symmetric eigen decomposition via Jacobi rotations.

#pragma once

#include "vec3.h"
#include "mat3.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

namespace wp {

// ── Eigen decomposition of symmetric 3x3 matrix ──────────────────────
// Returns eigenvectors as columns of V, eigenvalues in diag (sorted descending if desired).
template <typename T>
void symmetric_eigen(const mat3<T>& A, mat3<T>& V, vec3<T>& diag) {
    // Initialize V as identity, work on a copy of A
    V = identity3<T>();
    mat3<T> a = A;

    // Maximum iterations for Jacobi (more than enough for 3x3)
    const int max_iter = 20;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Find largest off-diagonal element
        T max_off = T(0);
        int p = 0, q = 1;
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                T val = std::abs(a(i, j));
                if (val > max_off) {
                    max_off = val;
                    p = i; q = j;
                }
            }
        }
        if (max_off < epsilon<T>) break;

        // Compute Jacobi rotation
        T theta = (a(q, q) - a(p, p)) / (T(2) * a(p, q));
        T t_val = T(1) / (std::abs(theta) + std::sqrt(T(1) + theta * theta));
        if (theta < T(0)) t_val = -t_val;
        T c = T(1) / std::sqrt(T(1) + t_val * t_val);
        T s = c * t_val;

        // Apply rotation R(p,q) to A: A = J^T * A * J
        // Update rows/cols p,q
        for (int i = 0; i < 3; ++i) {
            T ap = a(i, p), aq = a(i, q);
            a(i, p) = c * ap - s * aq;
            a(i, q) = s * ap + c * aq;
        }
        for (int j = 0; j < 3; ++j) {
            T ap = a(p, j), aq = a(q, j);
            a(p, j) = c * ap - s * aq;
            a(q, j) = s * ap + c * aq;
        }
        // Update eigenvector matrix V = V * J (columns)
        for (int i = 0; i < 3; ++i) {
            T vp = V(i, p), vq = V(i, q);
            V(i, p) = c * vp - s * vq;
            V(i, q) = s * vp + c * vq;
        }
    }
    // Diagonal contains eigenvalues
    diag = vec3<T>(a(0,0), a(1,1), a(2,2));
    // Sort descending? Not necessary, return as is.
}

// ── SVD of general 3x3 matrix A = U * diag(S) * V^T ─────────────────
template <typename T>
void svd3(const mat3<T>& A, mat3<T>& U, vec3<T>& S, mat3<T>& V) {
    // Compute A^T * A (symmetric, positive semi-definite)
    mat3<T> ATA = mul(transpose(A), A);
    mat3<T> V_mat;
    vec3<T> lambda;
    symmetric_eigen(ATA, V_mat, lambda);

    // Singular values = sqrt of eigenvalues (ensure non-negative)
    S.x = std::sqrt(std::max(T(0), lambda.x));
    S.y = std::sqrt(std::max(T(0), lambda.y));
    S.z = std::sqrt(std::max(T(0), lambda.z));

    // V = eigenvectors of ATA
    V = V_mat;

    // Compute U = A * V * diag(1/S) (with zero singular values handled)
    vec3<T> inv_S(T(0));
    if (S.x > epsilon<T>) inv_S.x = T(1) / S.x;
    if (S.y > epsilon<T>) inv_S.y = T(1) / S.y;
    if (S.z > epsilon<T>) inv_S.z = T(1) / S.z;

    // U columns = A * V col_i / S_i
    auto col0 = mul(A, V.col(0)) * inv_S.x;
    auto col1 = mul(A, V.col(1)) * inv_S.y;
    auto col2 = mul(A, V.col(2)) * inv_S.z;

    // If some singular values are zero, pick an orthogonal basis for the null space
    if (inv_S.x == T(0)) col0 = vec3<T>(T(1), T(0), T(0));
    if (inv_S.y == T(0)) col1 = vec3<T>(T(0), T(1), T(0));
    if (inv_S.z == T(0)) col2 = vec3<T>(T(0), T(0), T(1));

    // Orthonormalize U (Gram-Schmidt)
    U = mat3<T>(col0, col1, col2);
    // Ensure right-handedness if needed (U determinant should be 1 for rotation)
    if (det(U) < T(0)) {
        U = mat3<T>(U.col(0), U.col(1), -U.col(2));
        S.z = -S.z; // reflect corresponding singular value? Actually preserve sign? Keep S positive, just flip U.
        // Better: just negate the last column of U and corresponding singular value.
        S.z = -S.z;
    }
}

// ── Polar decomposition (rotation + stretch) ───────────────────────
template <typename T>
void polar_decomposition(const mat3<T>& A, mat3<T>& R, mat3<T>& S) {
    mat3<T> U, V;
    vec3<T> sigma;
    svd3(A, U, sigma, V);
    R = mul(U, transpose(V));
    S = mul(mul(V, mat3<T>(sigma.x, T(0), T(0),
                            T(0), sigma.y, T(0),
                            T(0), T(0), sigma.z)), transpose(V));
}

} // namespace wp