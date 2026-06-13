//File group name : OrthoTree Math
//File 0037 : core/math/matrix_decomposition.h
//Matrix decompositions for 2x2, 3x3, 4x4: LU, QR, SVD (2x2 analytic, 3x3 iterative), eigenvalue (Jacobi), Cholesky, and SIMD batch for multiple small matrices.

#ifndef ORTHOTREE_CORE_MATH_MATRIX_DECOMPOSITION_H_INCLUDED
#define ORTHOTREE_CORE_MATH_MATRIX_DECOMPOSITION_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "matrix.h"
#include "vector_math.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <array>
#include <limits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  LU decomposition for 3x3 matrix. Solves linear systems.
//  Returns true if non‑singular, stores L and U.
// ============================================================================
template<typename T>
bool luDecompose3x3(const Matrix<T,3>& A, Matrix<T,3>& L, Matrix<T,3>& U) {
    L = Matrix<T,3>::identity();
    U = A;
    for (int i = 0; i < 2; ++i) {
        if (std::abs(U(i,i)) < T(1e-12)) return false;
        for (int j = i+1; j < 3; ++j) {
            T factor = U(j,i) / U(i,i);
            L(j,i) = factor;
            for (int k = i; k < 3; ++k) {
                U(j,k) -= factor * U(i,k);
            }
        }
    }
    return true;
}

// ----------------------------------------------------------------------------
//  Solve linear system using LU (3x3)
// ----------------------------------------------------------------------------
template<typename T>
bool solveLU3x3(const Matrix<T,3>& A, const Vector<T,3>& b, Vector<T,3>& x) {
    Matrix<T,3> L, U;
    if (!luDecompose3x3(A, L, U)) return false;
    // Forward substitution: Ly = b
    Vector<T,3> y;
    for (int i = 0; i < 3; ++i) {
        T sum = b[i];
        for (int j = 0; j < i; ++j) sum -= L(i,j) * y[j];
        y[i] = sum / L(i,i);
    }
    // Back substitution: Ux = y
    for (int i = 2; i >= 0; --i) {
        T sum = y[i];
        for (int j = i+1; j < 3; ++j) sum -= U(i,j) * x[j];
        x[i] = sum / U(i,i);
    }
    return true;
}

// ============================================================================
//  QR decomposition for 3x3 (Householder)
// ============================================================================
template<typename T>
void qrDecompose3x3(const Matrix<T,3>& A, Matrix<T,3>& Q, Matrix<T,3>& R) {
    R = A;
    Q = Matrix<T,3>::identity();
    for (int k = 0; k < 2; ++k) {
        // Householder vector for column k
        T norm = T(0);
        for (int i = k; i < 3; ++i) norm += R(i,k) * R(i,k);
        norm = std::sqrt(norm);
        if (norm < T(1e-12)) continue;
        T alpha = (R(k,k) > T(0)) ? -norm : norm;
        T beta = T(1) / (alpha * (alpha - R(k,k)));
        Vector<T,3> u(0);
        u[k] = R(k,k) - alpha;
        for (int i = k+1; i < 3; ++i) u[i] = R(i,k);
        // Apply Householder to R and Q
        for (int j = k; j < 3; ++j) {
            T dot = T(0);
            for (int i = k; i < 3; ++i) dot += u[i] * R(i,j);
            dot *= beta;
            for (int i = k; i < 3; ++i) R(i,j) -= dot * u[i];
        }
        for (int j = 0; j < 3; ++j) {
            T dot = T(0);
            for (int i = k; i < 3; ++i) dot += u[i] * Q(i,j);
            dot *= beta;
            for (int i = k; i < 3; ++i) Q(i,j) -= dot * u[i];
        }
    }
    // Q should be orthogonal; ensure sign
    for (int i = 0; i < 3; ++i) {
        if (R(i,i) < T(0)) {
            for (int j = 0; j < 3; ++j) {
                Q(j,i) = -Q(j,i);
                R(i,j) = -R(i,j);
            }
        }
    }
}

// ============================================================================
//  SVD for 2x2 matrix (analytic)
// ============================================================================
template<typename T>
void svd2x2(const Matrix<T,2>& A, Matrix<T,2>& U, Vector<T,2>& S, Matrix<T,2>& V) {
    T a = A(0,0), b = A(0,1), c = A(1,0), d = A(1,1);
    T theta = T(0.5) * std::atan2( T(2)*(a*b + c*d), a*a + c*c - b*b - d*d );
    T phi = T(0.5) * std::atan2( T(2)*(a*c + b*d), a*a + b*b - c*c - d*d );
    T s1 = a*a + b*b + c*c + d*d;
    T s2 = std::sqrt( std::pow(a*a + b*b - c*c - d*d, T(2)) + T(4)*std::pow(a*c + b*d, T(2)) );
    S[0] = std::sqrt( (s1 + s2) / T(2) );
    S[1] = std::sqrt( (s1 - s2) / T(2) );
    T cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    T cosPhi   = std::cos(phi),   sinPhi   = std::sin(phi);
    U(0,0) = cosPhi;  U(0,1) = -sinPhi;
    U(1,0) = sinPhi;  U(1,1) =  cosPhi;
    V(0,0) = cosTheta; V(0,1) = -sinTheta;
    V(1,0) = sinTheta; V(1,1) =  cosTheta;
    // Ensure singular values sorted descending
    if (S[0] < S[1]) {
        std::swap(S[0], S[1]);
        std::swap(U(0,0), U(0,1));
        std::swap(U(1,0), U(1,1));
    }
}

// ============================================================================
//  SVD for 3x3 using Jacobi rotations (iterative)
// ============================================================================
template<typename T>
void svd3x3(const Matrix<T,3>& A, Matrix<T,3>& U, Vector<T,3>& S, Matrix<T,3>& V, int maxIter = 100) {
    U = Matrix<T,3>::identity();
    V = Matrix<T,3>::identity();
    Matrix<T,3> B = A;
    for (int iter = 0; iter < maxIter; ++iter) {
        T off = T(0);
        for (int i = 0; i < 2; ++i) {
            for (int j = i+1; j < 3; ++j) {
                off += B(i,j)*B(i,j) + B(j,i)*B(j,i);
            }
        }
        if (off < T(1e-12)) break;
        for (int i = 0; i < 2; ++i) {
            for (int j = i+1; j < 3; ++j) {
                // Jacobi rotation on pairs (i,j)
                T a = B(i,i), b = B(i,j), c = B(j,i), d = B(j,j);
                T tau = (d - a) / (T(2)*(b + c));
                T t = (tau >= T(0)) ? T(1) / (tau + std::sqrt(T(1)+tau*tau))
                                    : T(1) / (tau - std::sqrt(T(1)+tau*tau));
                T cosTheta = T(1) / std::sqrt(T(1)+t*t);
                T sinTheta = t * cosTheta;
                // Apply to B and accumulate U and V
                for (int k = 0; k < 3; ++k) {
                    T bik = B(i,k), bjk = B(j,k);
                    B(i,k) = cosTheta * bik + sinTheta * bjk;
                    B(j,k) = -sinTheta * bik + cosTheta * bjk;
                }
                for (int k = 0; k < 3; ++k) {
                    T bki = B(k,i), bkj = B(k,j);
                    B(k,i) = cosTheta * bki + sinTheta * bkj;
                    B(k,j) = -sinTheta * bki + cosTheta * bkj;
                }
                for (int k = 0; k < 3; ++k) {
                    T uki = U(k,i), ukj = U(k,j);
                    U(k,i) = cosTheta * uki + sinTheta * ukj;
                    U(k,j) = -sinTheta * uki + cosTheta * ukj;
                }
                for (int k = 0; k < 3; ++k) {
                    T vki = V(k,i), vkj = V(k,j);
                    V(k,i) = cosTheta * vki + sinTheta * vkj;
                    V(k,j) = -sinTheta * vki + cosTheta * vkj;
                }
            }
        }
    }
    // Extract singular values from diagonal
    for (int i = 0; i < 3; ++i) S[i] = std::abs(B(i,i));
    // Ensure singular values sorted descending
    for (int i = 0; i < 2; ++i) {
        for (int j = i+1; j < 3; ++j) {
            if (S[i] < S[j]) {
                std::swap(S[i], S[j]);
                for (int k = 0; k < 3; ++k) std::swap(U(k,i), U(k,j));
                for (int k = 0; k < 3; ++k) std::swap(V(k,i), V(k,j));
            }
        }
    }
}

// ============================================================================
//  Eigenvalues and eigenvectors of symmetric matrix (Jacobi)
// ============================================================================
template<typename T>
void eigenSymmetric3x3(const Matrix<T,3>& A, Vector<T,3>& eigenvalues, Matrix<T,3>& eigenvectors, int maxIter = 50) {
    eigenvectors = Matrix<T,3>::identity();
    Matrix<T,3> B = A;
    for (int iter = 0; iter < maxIter; ++iter) {
        T off = T(0);
        for (int i = 0; i < 2; ++i) {
            for (int j = i+1; j < 3; ++j) off += B(i,j)*B(i,j);
        }
        if (off < T(1e-12)) break;
        for (int i = 0; i < 2; ++i) {
            for (int j = i+1; j < 3; ++j) {
                T a = B(i,i), b = B(i,j), d = B(j,j);
                T tau = (d - a) / (T(2)*b);
                T t = (tau >= T(0)) ? T(1) / (tau + std::sqrt(T(1)+tau*tau))
                                    : T(1) / (tau - std::sqrt(T(1)+tau*tau));
                T cosTheta = T(1) / std::sqrt(T(1)+t*t);
                T sinTheta = t * cosTheta;
                for (int k = 0; k < 3; ++k) {
                    T bik = B(i,k), bjk = B(j,k);
                    B(i,k) = cosTheta * bik + sinTheta * bjk;
                    B(j,k) = -sinTheta * bik + cosTheta * bjk;
                }
                for (int k = 0; k < 3; ++k) {
                    T bki = B(k,i), bkj = B(k,j);
                    B(k,i) = cosTheta * bki + sinTheta * bkj;
                    B(k,j) = -sinTheta * bki + cosTheta * bkj;
                }
                for (int k = 0; k < 3; ++k) {
                    T eki = eigenvectors(k,i), ekj = eigenvectors(k,j);
                    eigenvectors(k,i) = cosTheta * eki + sinTheta * ekj;
                    eigenvectors(k,j) = -sinTheta * eki + cosTheta * ekj;
                }
            }
        }
    }
    for (int i = 0; i < 3; ++i) eigenvalues[i] = B(i,i);
}

// ============================================================================
//  Cholesky decomposition for symmetric positive definite matrix (3x3)
//  Returns lower triangular L such that A = L * L^T.
// ----------------------------------------------------------------------------
template<typename T>
bool cholesky3x3(const Matrix<T,3>& A, Matrix<T,3>& L) {
    L = Matrix<T,3>(0);
    for (int i = 0; i < 3; ++i) {
        T sum = A(i,i);
        for (int k = 0; k < i; ++k) sum -= L(i,k) * L(i,k);
        if (sum <= T(0)) return false;
        L(i,i) = std::sqrt(sum);
        for (int j = i+1; j < 3; ++j) {
            sum = A(j,i);
            for (int k = 0; k < i; ++k) sum -= L(j,k) * L(i,k);
            L(j,i) = sum / L(i,i);
        }
    }
    return true;
}

// ============================================================================
//  SIMD batch: SVD for 4 3x3 matrices (using scalar, but could be parallelised)
// ----------------------------------------------------------------------------
template<typename T>
void batchSvd3x3(const Matrix<T,3>* A, Matrix<T,3>* U, Vector<T,3>* S, Matrix<T,3>* V, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        svd3x3(A[i], U[i], S[i], V[i]);
    }
}

// ============================================================================
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class MatrixDecompositionEnvironment {
public:
    static MatrixDecompositionEnvironment& instance() {
        static MatrixDecompositionEnvironment env;
        return env;
    }
    void setEpsilon(T eps) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_epsilon = eps;
    }
    T epsilon() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_epsilon;
    }
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    MatrixDecompositionEnvironment() : m_epsilon(T(1e-12)), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_MATRIX_DECOMPOSITION_H_INCLUDED