// system name : onetbb-warp
// File 0017 : core/math/linear_system.h
// Description : Direct and iterative solvers for large sparse linear systems.

#ifndef __TBB_WARP_CORE_MATH_LINEAR_SYSTEM_H
#define __TBB_WARP_CORE_MATH_LINEAR_SYSTEM_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/matrix3.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <limits>
#include <functional>
#include <stdexcept>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Forward substitution (lower triangular)
// ============================================================

template<typename T>
std::vector<T> forward_substitution(const std::vector<std::vector<T>>& L,
                                    const std::vector<T>& b) {
    std::size_t n = b.size();
    std::vector<T> x(n, T(0));
    for (std::size_t i = 0; i < n; ++i) {
        T sum = b[i];
        for (std::size_t j = 0; j < i; ++j) sum -= L[i][j] * x[j];
        if (std::abs(L[i][i]) < T(1e-12))
            throw std::runtime_error("Singular lower triangular matrix");
        x[i] = sum / L[i][i];
    }
    return x;
}

// ============================================================
// Backward substitution (upper triangular)
// ============================================================

template<typename T>
std::vector<T> backward_substitution(const std::vector<std::vector<T>>& U,
                                     const std::vector<T>& b) {
    std::size_t n = b.size();
    std::vector<T> x(n, T(0));
    for (std::size_t i = n; i-- > 0;) {
        T sum = b[i];
        for (std::size_t j = i + 1; j < n; ++j) sum -= U[i][j] * x[j];
        if (std::abs(U[i][i]) < T(1e-12))
            throw std::runtime_error("Singular upper triangular matrix");
        x[i] = sum / U[i][i];
    }
    return x;
}

// ============================================================
// LU decomposition with partial pivoting (Doolittle)
// ============================================================

template<typename T>
void lu_decompose(std::vector<std::vector<T>>& A, std::vector<std::size_t>& pivot,
                  std::size_t& pivot_sign) {
    std::size_t n = A.size();
    pivot.resize(n);
    std::iota(pivot.begin(), pivot.end(), 0);
    pivot_sign = 1;
    std::vector<T> scales(n);
    for (std::size_t i = 0; i < n; ++i) {
        T max_val = T(0);
        for (std::size_t j = 0; j < n; ++j)
            max_val = std::max(max_val, std::abs(A[i][j]));
        if (max_val < T(1e-12))
            throw std::runtime_error("Singular matrix in LU decomposition");
        scales[i] = T(1) / max_val;
    }
    for (std::size_t k = 0; k < n - 1; ++k) {
        std::size_t max_row = k;
        T max_ratio = std::abs(A[pivot[k]][k]) * scales[pivot[k]];
        for (std::size_t i = k + 1; i < n; ++i) {
            T ratio = std::abs(A[pivot[i]][k]) * scales[pivot[i]];
            if (ratio > max_ratio) { max_ratio = ratio; max_row = i; }
        }
        if (max_row != k) {
            std::swap(pivot[k], pivot[max_row]);
            pivot_sign = -pivot_sign;
        }
        T pivot_val = A[pivot[k]][k];
        if (std::abs(pivot_val) < T(1e-12))
            throw std::runtime_error("Zero pivot in LU decomposition");
        for (std::size_t i = k + 1; i < n; ++i) {
            T factor = A[pivot[i]][k] / pivot_val;
            A[pivot[i]][k] = factor;
            for (std::size_t j = k + 1; j < n; ++j)
                A[pivot[i]][j] -= factor * A[pivot[k]][j];
        }
    }
}

// ============================================================
// Solve A x = b using pre‑computed LU decomposition
// ============================================================

template<typename T>
std::vector<T> lu_solve(const std::vector<std::vector<T>>& LU,
                        const std::vector<std::size_t>& pivot,
                        const std::vector<T>& b) {
    std::size_t n = b.size();
    std::vector<T> x(n);
    // Forward substitution (apply permutation and solve L y = b)
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = b[pivot[i]];
        for (std::size_t j = 0; j < i; ++j)
            x[i] -= LU[pivot[i]][j] * x[j];
    }
    // Backward substitution (solve U x = y)
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = i + 1; j < n; ++j)
            x[i] -= LU[pivot[i]][j] * x[j];
        x[i] /= LU[pivot[i]][i];
    }
    return x;
}

// ============================================================
// Cholesky decomposition (LLT) for symmetric positive definite matrices
// ============================================================

template<typename T>
std::vector<std::vector<T>> cholesky_decompose(const std::vector<std::vector<T>>& A) {
    std::size_t n = A.size();
    std::vector<std::vector<T>> L(n, std::vector<T>(n, T(0)));
    for (std::size_t i = 0; i < n; ++i) {
        T sum = A[i][i];
        for (std::size_t j = 0; j < i; ++j)
            sum -= L[i][j] * L[i][j];
        if (sum <= T(0))
            throw std::runtime_error("Matrix is not positive definite");
        L[i][i] = std::sqrt(sum);
        for (std::size_t j = i + 1; j < n; ++j) {
            sum = A[j][i];
            for (std::size_t k = 0; k < i; ++k)
                sum -= L[j][k] * L[i][k];
            L[j][i] = sum / L[i][i];
        }
    }
    return L;
}

// ============================================================
// Solve A x = b using Cholesky factor L
// ============================================================

template<typename T>
std::vector<T> cholesky_solve(const std::vector<std::vector<T>>& L,
                               const std::vector<T>& b) {
    std::size_t n = b.size();
    std::vector<T> y = forward_substitution(L, b);
    // Transpose L to get U
    std::vector<std::vector<T>> U(n, std::vector<T>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            U[j][i] = L[i][j];
    return backward_substitution(U, y);
}

// ============================================================
// Conjugate Gradient for symmetric positive definite A x = b
// ============================================================

template<typename T>
std::vector<T> conjugate_gradient(const std::vector<std::vector<T>>& A,
                                  const std::vector<T>& b,
                                  std::size_t max_iter = 1000,
                                  T tol = T(1e-6)) {
    std::size_t n = b.size();
    std::vector<T> x(n, T(0));
    std::vector<T> r = b; // initial residual
    std::vector<T> p = r;
    T rsold = T(0);
    for (std::size_t i = 0; i < n; ++i) rsold += r[i] * r[i];
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        // Ap = A * p
        std::vector<T> Ap(n, T(0));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                Ap[i] += A[i][j] * p[j];
        T pAp = T(0);
        for (std::size_t i = 0; i < n; ++i) pAp += p[i] * Ap[i];
        if (std::abs(pAp) < T(1e-12)) break;
        T alpha = rsold / pAp;
        for (std::size_t i = 0; i < n; ++i) x[i] += alpha * p[i];
        for (std::size_t i = 0; i < n; ++i) r[i] -= alpha * Ap[i];
        T rsnew = T(0);
        for (std::size_t i = 0; i < n; ++i) rsnew += r[i] * r[i];
        if (std::sqrt(rsnew) < tol) break;
        T beta = rsnew / rsold;
        for (std::size_t i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }
    return x;
}

// ============================================================
// Jacobi iterative solver
// ============================================================

template<typename T>
std::vector<T> jacobi_iteration(const std::vector<std::vector<T>>& A,
                                const std::vector<T>& b,
                                std::size_t max_iter = 1000,
                                T tol = T(1e-6)) {
    std::size_t n = b.size();
    std::vector<T> x(n, T(0));
    std::vector<T> x_new(n, T(0));
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            T sigma = T(0);
            for (std::size_t j = 0; j < n; ++j)
                if (j != i) sigma += A[i][j] * x[j];
            if (std::abs(A[i][i]) < T(1e-12))
                throw std::runtime_error("Zero diagonal element in Jacobi");
            x_new[i] = (b[i] - sigma) / A[i][i];
        }
        T error = T(0);
        for (std::size_t i = 0; i < n; ++i) {
            T diff = x_new[i] - x[i];
            error += diff * diff;
        }
        x = x_new;
        if (std::sqrt(error) < tol) break;
    }
    return x;
}

// ============================================================
// Gauss‑Seidel iterative solver
// ============================================================

template<typename T>
std::vector<T> gauss_seidel(const std::vector<std::vector<T>>& A,
                            const std::vector<T>& b,
                            std::size_t max_iter = 1000,
                            T tol = T(1e-6)) {
    std::size_t n = b.size();
    std::vector<T> x(n, T(0));
    std::vector<T> x_old(n, T(0));
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        x_old = x;
        for (std::size_t i = 0; i < n; ++i) {
            T sigma = T(0);
            for (std::size_t j = 0; j < n; ++j)
                if (j != i) sigma += A[i][j] * x[j];
            if (std::abs(A[i][i]) < T(1e-12))
                throw std::runtime_error("Zero diagonal element in Gauss‑Seidel");
            x[i] = (b[i] - sigma) / A[i][i];
        }
        T error = T(0);
        for (std::size_t i = 0; i < n; ++i) {
            T diff = x[i] - x_old[i];
            error += diff * diff;
        }
        if (std::sqrt(error) < tol) break;
    }
    return x;
}

// ============================================================
// Successive Over‑Relaxation (SOR)
// ============================================================

template<typename T>
std::vector<T> sor(const std::vector<std::vector<T>>& A,
                   const std::vector<T>& b,
                   T omega = T(1.2),
                   std::size_t max_iter = 1000,
                   T tol = T(1e-6)) {
    std::size_t n = b.size();
    std::vector<T> x(n, T(0));
    std::vector<T> x_old(n, T(0));
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        x_old = x;
        for (std::size_t i = 0; i < n; ++i) {
            T sigma = T(0);
            for (std::size_t j = 0; j < n; ++j)
                if (j != i) sigma += A[i][j] * x[j];
            if (std::abs(A[i][i]) < T(1e-12))
                throw std::runtime_error("Zero diagonal element in SOR");
            T update = (b[i] - sigma) / A[i][i];
            x[i] = x_old[i] + omega * (update - x_old[i]);
        }
        T error = T(0);
        for (std::size_t i = 0; i < n; ++i) {
            T diff = x[i] - x_old[i];
            error += diff * diff;
        }
        if (std::sqrt(error) < tol) break;
    }
    return x;
}

// ============================================================
// Incomplete Cholesky preconditioner (0‑fill)
// ============================================================

template<typename T>
std::vector<std::vector<T>> incomplete_cholesky(const std::vector<std::vector<T>>& A) {
    std::size_t n = A.size();
    std::vector<std::vector<T>> L(n, std::vector<T>(n, T(0)));
    for (std::size_t i = 0; i < n; ++i) {
        T sum = A[i][i];
        for (std::size_t k = 0; k < i; ++k)
            sum -= L[i][k] * L[i][k];
        if (sum <= T(0)) sum = T(1);
        L[i][i] = std::sqrt(sum);
        for (std::size_t j = i + 1; j < n; ++j) {
            if (std::abs(A[j][i]) < T(1e-12)) continue;
            sum = A[j][i];
            for (std::size_t k = 0; k < i; ++k)
                sum -= L[j][k] * L[i][k];
            L[j][i] = sum / L[i][i];
        }
    }
    return L;
}

// ============================================================
// Preconditioned Conjugate Gradient (with IC(0) preconditioner)
// ============================================================

template<typename T>
std::vector<T> pcg(const std::vector<std::vector<T>>& A,
                   const std::vector<T>& b,
                   std::size_t max_iter = 1000,
                   T tol = T(1e-6)) {
    std::size_t n = b.size();
    std::vector<std::vector<T>> L = incomplete_cholesky(A);
    std::vector<T> x(n, T(0));
    std::vector<T> r = b;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            r[i] -= A[i][j] * x[j];
    // solve L y = r, then L^T z = y to get preconditioned residual
    std::vector<T> z(n, T(0));
    // forward
    for (std::size_t i = 0; i < n; ++i) {
        T sum = r[i];
        for (std::size_t j = 0; j < i; ++j) sum -= L[i][j] * z[j];
        z[i] = sum / L[i][i];
    }
    std::vector<T> p = z;
    T rsold = T(0);
    for (std::size_t i = 0; i < n; ++i) rsold += r[i] * z[i];
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        std::vector<T> Ap(n, T(0));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                Ap[i] += A[i][j] * p[j];
        T pAp = T(0);
        for (std::size_t i = 0; i < n; ++i) pAp += p[i] * Ap[i];
        if (std::abs(pAp) < T(1e-12)) break;
        T alpha = rsold / pAp;
        for (std::size_t i = 0; i < n; ++i) x[i] += alpha * p[i];
        for (std::size_t i = 0; i < n; ++i) r[i] -= alpha * Ap[i];
        // preconditioner solve
        std::vector<T> z_new(n, T(0));
        for (std::size_t i = 0; i < n; ++i) {
            T sum = r[i];
            for (std::size_t j = 0; j < i; ++j) sum -= L[i][j] * z_new[j];
            z_new[i] = sum / L[i][i];
        }
        T rsnew = T(0);
        for (std::size_t i = 0; i < n; ++i) rsnew += r[i] * z_new[i];
        if (std::sqrt(rsnew) < tol) break;
        T beta = rsnew / rsold;
        for (std::size_t i = 0; i < n; ++i) p[i] = z_new[i] + beta * p[i];
        rsold = rsnew;
        z = z_new;
    }
    return x;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_LINEAR_SYSTEM_H