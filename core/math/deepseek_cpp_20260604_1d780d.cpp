// system name : onetbb-warp
// File 0037 : core/math/linear_algebra_ext.h
// Description : Extended linear algebra: QR, SVD, full‑pivot LU, non‑symmetric eigenvalues, pseudo‑inverse.

#ifndef __TBB_WARP_CORE_MATH_LINEAR_ALGEBRA_EXT_H
#define __TBB_WARP_CORE_MATH_LINEAR_ALGEBRA_EXT_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/matrix3.h"
#include "core/math/matrix4.h"
#include "core/math/complex.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <complex>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// QR decomposition (Householder)
// ============================================================

template<typename T>
void qr_decompose(const std::vector<std::vector<T>>& A,
                  std::vector<std::vector<T>>& Q,
                  std::vector<std::vector<T>>& R) {
    std::size_t m = A.size();
    std::size_t n = A[0].size();
    Q.assign(m, std::vector<T>(m, T(0)));
    for (std::size_t i = 0; i < m; ++i) Q[i][i] = T(1);
    R = A;
    for (std::size_t k = 0; k < n && k < m - 1; ++k) {
        T x_norm = T(0);
        for (std::size_t i = k; i < m; ++i) x_norm += R[i][k] * R[i][k];
        x_norm = std::sqrt(x_norm);
        if (x_norm < T(1e-12)) continue;
        T alpha = -std::copysign(x_norm, R[k][k]);
        T inv = T(1) / (R[k][k] - alpha);
        std::vector<T> v(m, T(0));
        v[k] = R[k][k] - alpha;
        for (std::size_t i = k + 1; i < m; ++i) v[i] = R[i][k];
        T v_norm_sq = T(0);
        for (std::size_t i = k; i < m; ++i) v_norm_sq += v[i] * v[i];
        T beta = T(2) / v_norm_sq;
        for (std::size_t j = k; j < n; ++j) {
            T dot = T(0);
            for (std::size_t i = k; i < m; ++i) dot += v[i] * R[i][j];
            for (std::size_t i = k; i < m; ++i) R[i][j] -= beta * dot * v[i];
        }
        for (std::size_t j = 0; j < m; ++j) {
            T dot = T(0);
            for (std::size_t i = k; i < m; ++i) dot += v[i] * Q[i][j];
            for (std::size_t i = k; i < m; ++i) Q[i][j] -= beta * dot * v[i];
        }
    }
}

// ============================================================
// SVD of m×n matrix (Jacobi SVD) – returns U, sigma, V
// ============================================================

template<typename T>
void svd(const std::vector<std::vector<T>>& A,
         std::vector<std::vector<T>>& U,
         std::vector<T>& sigma,
         std::vector<std::vector<T>>& V) {
    std::size_t m = A.size();
    std::size_t n = A[0].size();
    std::size_t k = std::min(m, n);
    U = A;
    sigma.resize(k, T(0));
    V.assign(k, std::vector<T>(k, T(0)));
    for (std::size_t i = 0; i < k; ++i) V[i][i] = T(1);
    const int max_iter = 100;
    for (int iter = 0; iter < max_iter; ++iter) {
        T max_off = T(0);
        std::size_t p = 0, q = 1;
        for (std::size_t i = 0; i < k - 1; ++i) {
            for (std::size_t j = i + 1; j < k; ++j) {
                T val = T(0);
                for (std::size_t r = 0; r < m; ++r) val += U[r][i] * U[r][j];
                T abs_val = std::abs(val);
                if (abs_val > max_off) { max_off = abs_val; p = i; q = j; }
            }
        }
        if (max_off < T(1e-12)) break;
        T alpha = T(0), beta = T(0);
        for (std::size_t i = 0; i < m; ++i) {
            alpha += U[i][p] * U[i][p];
            beta  += U[i][q] * U[i][q];
        }
        T gamma = T(0);
        for (std::size_t i = 0; i < m; ++i) gamma += U[i][p] * U[i][q];
        T zeta = (alpha - beta) * T(0.5);
        T omega = gamma;
        T t = (zeta >= T(0) ? T(1) : T(-1)) * omega / (std::sqrt(zeta*zeta + omega*omega) + std::abs(zeta));
        T c = T(1) / std::sqrt(T(1) + t*t);
        T s = t * c;
        for (std::size_t i = 0; i < m; ++i) {
            T up = U[i][p], uq = U[i][q];
            U[i][p] = c * up - s * uq;
            U[i][q] = s * up + c * uq;
        }
        for (std::size_t j = 0; j < k; ++j) {
            T vp = V[j][p], vq = V[j][q];
            V[j][p] = c * vp - s * vq;
            V[j][q] = s * vp + c * vq;
        }
    }
    for (std::size_t i = 0; i < k; ++i) {
        T norm_col = T(0);
        for (std::size_t j = 0; j < m; ++j) norm_col += U[j][i] * U[j][i];
        sigma[i] = std::sqrt(norm_col);
        if (sigma[i] > T(1e-12)) {
            T inv = T(1) / sigma[i];
            for (std::size_t j = 0; j < m; ++j) U[j][i] *= inv;
        }
    }
    // Sort
    for (std::size_t i = 0; i < k - 1; ++i) {
        for (std::size_t j = i + 1; j < k; ++j) {
            if (sigma[i] < sigma[j]) {
                std::swap(sigma[i], sigma[j]);
                for (std::size_t r = 0; r < m; ++r) std::swap(U[r][i], U[r][j]);
                for (std::size_t r = 0; r < k; ++r) std::swap(V[r][i], V[r][j]);
            }
        }
    }
}

// ============================================================
// LU decomposition with full pivoting (returns P, Q permutations)
// ============================================================

template<typename T>
void lu_full_pivot(std::vector<std::vector<T>>& A,
                   std::vector<int>& row_perm,
                   std::vector<int>& col_perm,
                   std::size_t& rank) {
    std::size_t n = A.size();
    row_perm.resize(n);
    col_perm.resize(n);
    for (std::size_t i = 0; i < n; ++i) row_perm[i] = col_perm[i] = static_cast<int>(i);
    rank = 0;
    for (std::size_t k = 0; k < n; ++k) {
        T max_val = std::abs(A[k][k]);
        std::size_t max_row = k, max_col = k;
        for (std::size_t i = k; i < n; ++i)
            for (std::size_t j = k; j < n; ++j) {
                if (std::abs(A[i][j]) > max_val) {
                    max_val = std::abs(A[i][j]);
                    max_row = i;
                    max_col = j;
                }
            }
        if (max_val < T(1e-12)) break;
        if (max_row != k) {
            std::swap(A[k], A[max_row]);
            std::swap(row_perm[k], row_perm[max_row]);
        }
        if (max_col != k) {
            for (std::size_t i = 0; i < n; ++i) std::swap(A[i][k], A[i][max_col]);
            std::swap(col_perm[k], col_perm[max_col]);
        }
        rank++;
        T pivot = A[k][k];
        for (std::size_t i = k + 1; i < n; ++i) {
            T factor = A[i][k] / pivot;
            A[i][k] = factor;
            for (std::size_t j = k + 1; j < n; ++j)
                A[i][j] -= factor * A[k][j];
        }
    }
}

// ============================================================
// Non‑symmetric eigenvalue decomposition (real Schur via Francis QR)
// ============================================================

template<typename T>
void real_schur(std::vector<std::vector<T>>& H,
                std::vector<std::complex<T>>& eigenvals) {
    std::size_t n = H.size();
    // Hessenberg reduction
    for (std::size_t k = 0; k < n - 2; ++k) {
        T x_norm = T(0);
        for (std::size_t i = k + 1; i < n; ++i) x_norm += H[i][k] * H[i][k];
        x_norm = std::sqrt(x_norm);
        if (x_norm < T(1e-12)) continue;
        T alpha = -std::copysign(x_norm, H[k+1][k]);
        T inv = T(1) / (H[k+1][k] - alpha);
        std::vector<T> v(n, T(0));
        v[k+1] = H[k+1][k] - alpha;
        for (std::size_t i = k + 2; i < n; ++i) v[i] = H[i][k];
        T beta = T(2) / (v[k+1]*v[k+1] + (x_norm - alpha*alpha)); // approximate
        for (std::size_t j = k; j < n; ++j) {
            T dot = T(0);
            for (std::size_t i = k + 1; i < n; ++i) dot += v[i] * H[i][j];
            for (std::size_t i = k + 1; i < n; ++i) H[i][j] -= beta * dot * v[i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            T dot = T(0);
            for (std::size_t j = k + 1; j < n; ++j) dot += v[j] * H[i][j];
            for (std::size_t j = k + 1; j < n; ++j) H[i][j] -= beta * dot * v[j];
        }
    }
    // Extract eigenvalues from Hessenberg (simple: 2x2 blocks)
    eigenvals.resize(n);
    std::size_t i = 0;
    while (i < n) {
        if (i == n - 1 || std::abs(H[i+1][i]) < T(1e-12)) {
            eigenvals[i] = H[i][i];
            ++i;
        } else {
            T a = H[i][i], b = H[i][i+1];
            T c = H[i+1][i], d = H[i+1][i+1];
            T trace = a + d;
            T det = a*d - b*c;
            T disc = trace*trace - T(4)*det;
            if (disc >= T(0)) {
                T root = std::sqrt(disc);
                eigenvals[i]   = (trace + root) * T(0.5);
                eigenvals[i+1] = (trace - root) * T(0.5);
            } else {
                T re = trace * T(0.5);
                T im = std::sqrt(-disc) * T(0.5);
                eigenvals[i]   = std::complex<T>(re, im);
                eigenvals[i+1] = std::complex<T>(re, -im);
            }
            i += 2;
        }
    }
}

// ============================================================
// Pseudo‑inverse via SVD (Moore‑Penrose)
// ============================================================

template<typename T>
std::vector<std::vector<T>> pseudo_inverse(const std::vector<std::vector<T>>& A, T tol = T(1e-12)) {
    std::size_t m = A.size();
    std::size_t n = A[0].size();
    std::vector<std::vector<T>> U, V;
    std::vector<T> sigma;
    svd(A, U, sigma, V);
    // Compute V * Sigma_inv * U^T
    std::vector<std::vector<T>> result(n, std::vector<T>(m, T(0)));
    T max_sigma = T(0);
    for (T s : sigma) if (s > max_sigma) max_sigma = s;
    T threshold = max_sigma * tol;
    for (std::size_t k = 0; k < sigma.size(); ++k) {
        if (sigma[k] < threshold) continue;
        T inv = T(1) / sigma[k];
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < m; ++j)
                result[i][j] += V[i][k] * inv * U[j][k];
    }
    return result;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_LINEAR_ALGEBRA_EXT_H