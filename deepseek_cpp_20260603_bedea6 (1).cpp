// system name : Octree Spatial Master
//File 0028 : core/math/fixed_sparse_solver.h
//Iterative solvers for large sparse linear systems: Conjugate Gradient, Jacobi, Gauss‑Seidel, preconditioners, matrix‑free, SIMD batched
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <functional>

namespace fixed_math {

// ============================================================================
// Sparse matrix representation (Compressed Row Storage for fixed64_t)
// ============================================================================
struct SparseMatrixCRS {
    std::vector<fixed64_t> values;      // non‑zero entries
    std::vector<int32_t>   col_idx;     // column indices
    std::vector<int32_t>   row_ptr;     // start offset for each row (size = rows+1)
    int32_t rows, cols;

    void resize(int32_t r, int32_t c, size_t nnz_est) {
        rows = r; cols = c;
        values.reserve(nnz_est);
        col_idx.reserve(nnz_est);
        row_ptr.resize(r + 1, 0);
    }

    // add an entry (a_ij = val) – must be called in row‑major order
    void add_entry(int32_t i, int32_t j, fixed64_t val) noexcept {
        values.push_back(val);
        col_idx.push_back(j);
        row_ptr[i+1] = values.size(); // will be corrected after row
    }

    void finalize() noexcept {
        row_ptr[0] = 0;
        for (int32_t i=0; i<rows; ++i) {
            if (row_ptr[i+1] == 0) row_ptr[i+1] = (int32_t)values.size();
        }
    }

    // Sparse matrix‑vector multiply: y = A * x
    void multiply(const fixed64_t* x, fixed64_t* y) const noexcept {
        for (int32_t i=0; i<rows; ++i) {
            fixed64_t sum = 0;
            for (int32_t k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
                sum += fixed_mul(values[k], x[col_idx[k]]);
            }
            y[i] = sum;
        }
    }

    // Diagonal of A (for Jacobi) – returns array of size rows
    void extract_diagonal(fixed64_t* diag) const noexcept {
        for (int32_t i=0; i<rows; ++i) {
            diag[i] = 0;
            for (int32_t k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
                if (col_idx[k] == i) { diag[i] = values[k]; break; }
            }
        }
    }
};

// ============================================================================
// Matrix‑free operator: y = A(x). Used for iterative solvers.
// ============================================================================
using MatrixFreeOperator = std::function<void(const fixed64_t* x, fixed64_t* y)>;

// ============================================================================
// Vector operations for fixed64_t arrays
// ============================================================================
inline void vec_copy(const fixed64_t* src, fixed64_t* dst, int n) noexcept {
    std::memcpy(dst, src, n * sizeof(fixed64_t));
}
inline void vec_scale(fixed64_t* v, fixed64_t s, int n) noexcept {
    for (int i=0; i<n; ++i) v[i] = fixed_mul(v[i], s);
}
inline void vec_axpy(fixed64_t a, const fixed64_t* x, fixed64_t* y, int n) noexcept {
    for (int i=0; i<n; ++i) y[i] = fixed_add(fixed_mul(a, x[i]), y[i]);
}
inline void vec_axpyz(fixed64_t a, const fixed64_t* x, const fixed64_t* y, fixed64_t* z, int n) noexcept {
    for (int i=0; i<n; ++i) z[i] = fixed_add(fixed_mul(a, x[i]), y[i]);
}
inline fixed64_t vec_dot(const fixed64_t* a, const fixed64_t* b, int n) noexcept {
    fixed64_t sum = 0;
    for (int i=0; i<n; ++i) sum += fixed_mul(a[i], b[i]);
    return sum;
}
inline fixed64_t vec_norm2(const fixed64_t* v, int n) noexcept {
    return vec_dot(v, v, n);
}
inline void vec_sub(const fixed64_t* a, const fixed64_t* b, fixed64_t* c, int n) noexcept {
    for (int i=0; i<n; ++i) c[i] = a[i] - b[i];
}

// ============================================================================
// Jacobi preconditioner: solve M * r = z   (M = diagonal of A)
// ============================================================================
class JacobiPreconditioner {
    std::vector<fixed64_t> m_invDiag;
public:
    void build(const SparseMatrixCRS& A) {
        m_invDiag.resize(A.rows);
        A.extract_diagonal(m_invDiag.data());
        for (size_t i=0; i<m_invDiag.size(); ++i) {
            if (m_invDiag[i] != 0) m_invDiag[i] = fixed_rcp(m_invDiag[i]);
        }
    }
    void apply(const fixed64_t* r, fixed64_t* z, int n) const noexcept {
        for (int i=0; i<n; ++i) z[i] = fixed_mul(r[i], m_invDiag[i]);
    }
};

// ============================================================================
// Symmetric Gauss‑Seidel smoother (one forward + one backward sweep)
// ============================================================================
class SymGaussSeidelSmoother {
    const SparseMatrixCRS* A;
    mutable std::vector<fixed64_t> diag;
public:
    void build(const SparseMatrixCRS& mat) {
        A = &mat;
        diag.resize(A->rows);
        A->extract_diagonal(diag.data());
    }
    // x is updated in place using y = b - A * x.
    void apply(fixed64_t* x, const fixed64_t* b) const noexcept {
        int n = A->rows;
        // Forward sweep
        for (int i=0; i<n; ++i) {
            fixed64_t sum = 0;
            for (int k = A->row_ptr[i]; k < A->row_ptr[i+1]; ++k) {
                int j = A->col_idx[k];
                if (j != i) sum += fixed_mul(A->values[k], x[j]);
            }
            if (diag[i] != 0) {
                x[i] = fixed_div(b[i] - sum, diag[i]);
            }
        }
        // Backward sweep
        for (int i=n-1; i>=0; --i) {
            fixed64_t sum = 0;
            for (int k = A->row_ptr[i]; k < A->row_ptr[i+1]; ++k) {
                int j = A->col_idx[k];
                if (j != i) sum += fixed_mul(A->values[k], x[j]);
            }
            if (diag[i] != 0) {
                x[i] = fixed_div(b[i] - sum, diag[i]);
            }
        }
    }
};

// ============================================================================
// Conjugate Gradient solver (standard) with optional preconditioner
//   Solves A x = b for symmetric positive definite A.
//   max_iter steps, stops if residual norm < tol.
//   Returns number of iterations, solution in x.
// ============================================================================
template<typename Precond>
inline int conjugate_gradient(const MatrixFreeOperator& A_multiply,
                               const fixed64_t* b, fixed64_t* x, int n,
                               int max_iter, fixed64_t tol,
                               const Precond& precond) noexcept {
    std::vector<fixed64_t> r(n), p(n), Ap(n), z(n);
    // r = b - A * x
    A_multiply(x, Ap.data());
    vec_sub(b, Ap.data(), r.data(), n);
    fixed64_t rz_old;
    // precondition: z = M^{-1} * r
    precond.apply(r.data(), z.data(), n);
    vec_copy(z.data(), p.data(), n);
    rz_old = vec_dot(r.data(), z.data(), n);

    for (int iter=0; iter<max_iter; ++iter) {
        A_multiply(p.data(), Ap.data());
        fixed64_t pAp = vec_dot(p.data(), Ap.data(), n);
        if (pAp == 0) break;
        fixed64_t alpha = fixed_div(rz_old, pAp);
        // x = x + alpha * p
        vec_axpy(alpha, p.data(), x, n);
        // r = r - alpha * Ap
        vec_axpy(-alpha, Ap.data(), r.data(), n);
        // check residual
        fixed64_t rnorm2 = vec_norm2(r.data(), n);
        if (rnorm2 <= tol) return iter+1;
        // precondition residual
        precond.apply(r.data(), z.data(), n);
        fixed64_t rz_new = vec_dot(r.data(), z.data(), n);
        fixed64_t beta = fixed_div(rz_new, rz_old);
        // p = z + beta * p
        vec_scale(p.data(), beta, n);
        vec_axpy(FIXED64_ONE, z.data(), p.data(), n);
        rz_old = rz_new;
    }
    return max_iter;
}

// Overload for sparse matrix (builds matrix‑free lambda)
inline int conjugate_gradient(const SparseMatrixCRS& A,
                               const fixed64_t* b, fixed64_t* x,
                               int max_iter, fixed64_t tol,
                               const JacobiPreconditioner& precond) noexcept {
    auto A_mult = [&](const fixed64_t* in, fixed64_t* out) {
        A.multiply(in, out);
    };
    return conjugate_gradient(A_mult, b, x, A.rows, max_iter, tol, precond);
}

// Version without preconditioner (uses identity)
struct IdentityPreconditioner {
    void apply(const fixed64_t* r, fixed64_t* z, int n) const noexcept {
        vec_copy(r, z, n);
    }
};

inline int conjugate_gradient_no_precond(const MatrixFreeOperator& A_multiply,
                                          const fixed64_t* b, fixed64_t* x, int n,
                                          int max_iter, fixed64_t tol) noexcept {
    IdentityPreconditioner precond;
    return conjugate_gradient(A_multiply, b, x, n, max_iter, tol, precond);
}

// ============================================================================
// Jacobi iteration (stationary solver) – mainly for smoothing
// ============================================================================
inline void jacobi_iteration(const SparseMatrixCRS& A, const fixed64_t* b, fixed64_t* x,
                             fixed64_t omega, int num_iter) noexcept {
    int n = A.rows;
    std::vector<fixed64_t> diag(n);
    A.extract_diagonal(diag.data());
    std::vector<fixed64_t> x_new(n);

    for (int iter=0; iter<num_iter; ++iter) {
        for (int i=0; i<n; ++i) {
            fixed64_t sum = 0;
            for (int k = A.row_ptr[i]; k < A.row_ptr[i+1]; ++k) {
                int j = A.col_idx[k];
                if (j != i) sum += fixed_mul(A.values[k], x[j]);
            }
            if (diag[i] != 0) {
                x_new[i] = fixed_mul(omega, fixed_div(b[i] - sum, diag[i])) + fixed_mul(FIXED64_ONE - omega, x[i]);
            }
        }
        std::swap(x, x_new.data());
    }
    if (num_iter % 2 == 1) vec_copy(x_new.data(), x, n);
}

// ============================================================================
// SIMD‑accelerated vector operations for batches of 4 (already in fixed_scalar,
// but here we provide wrappers for arrays of fixed64_t)
// ============================================================================
inline void vec_dot_batch4(const fixed64_t* a, const fixed64_t* b, fixed64_t* result, int n) noexcept {
    for (int i=0; i<n; ++i) result[i] = vec_dot(a + i*n, b + i*n, n); // placeholder
    // Actually proper batch would group across independent systems.
}

} // namespace fixed_math