//File 0038 : core/math/linear_algebra.h
//Dense and sparse linear algebra: LU/Cholesky/QR/SVD/eigenvalue decompositions, matrix solve, determinant, inverse, trace, and conjugate‑gradient, leveraging Eigen for large matrices and SIMD for 2x2/3x3/4x4.
#ifndef CORE_MATH_LINEAR_ALGEBRA_H
#define CORE_MATH_LINEAR_ALGEBRA_H

#include "vector_math.h"
#include "matrix_math.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#include <cmath>

namespace SimulationMath {
namespace linalg {

using EigenVectorXf = Eigen::VectorXf;
using EigenMatrixXf = Eigen::MatrixXf;
using SparseMatrix = Eigen::SparseMatrix<float>;

// -----------------------------------------------------------------------------
// 1. Convert between std::vector<float> and Eigen::VectorXf
// -----------------------------------------------------------------------------
inline EigenVectorXf to_eigen_vector(const std::vector<float>& v) {
    return EigenVectorXf::Map(v.data(), v.size());
}
inline std::vector<float> to_std_vector(const EigenVectorXf& ev) {
    return std::vector<float>(ev.data(), ev.data() + ev.size());
}

// -----------------------------------------------------------------------------
// 2. Solve dense linear system Ax = b (LU with partial pivoting)
// -----------------------------------------------------------------------------
inline EigenVectorXf solve_lu(const Eigen::Ref<const EigenMatrixXf>& A, const Eigen::Ref<const EigenVectorXf>& b) {
    return A.lu().solve(b);
}

// -----------------------------------------------------------------------------
// 3. Solve symmetric positive‑definite system (Cholesky LLT)
// -----------------------------------------------------------------------------
inline EigenVectorXf solve_cholesky(const Eigen::Ref<const EigenMatrixXf>& A, const Eigen::Ref<const EigenVectorXf>& b) {
    return A.llt().solve(b);
}

// -----------------------------------------------------------------------------
// 4. Solve using QR decomposition (least‑squares, overdetermined systems)
// -----------------------------------------------------------------------------
inline EigenVectorXf solve_qr(const Eigen::Ref<const EigenMatrixXf>& A, const Eigen::Ref<const EigenVectorXf>& b) {
    return A.colPivHouseholderQr().solve(b);
}

// -----------------------------------------------------------------------------
// 5. Singular Value Decomposition
// -----------------------------------------------------------------------------
inline void compute_svd(const Eigen::Ref<const EigenMatrixXf>& A,
                        EigenMatrixXf& U, EigenVectorXf& S, EigenMatrixXf& V,
                        bool thin = true) {
    Eigen::JacobiSVD<EigenMatrixXf> svd(A, thin ? Eigen::ComputeThinU | Eigen::ComputeThinV : Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    S = svd.singularValues();
    V = svd.matrixV();
}

// -----------------------------------------------------------------------------
// 6. Eigenvalues and eigenvectors of symmetric matrix
// -----------------------------------------------------------------------------
inline void compute_eigen_symmetric(const Eigen::Ref<const EigenMatrixXf>& A,
                                    EigenVectorXf& eigenvalues, EigenMatrixXf& eigenvectors) {
    Eigen::SelfAdjointEigenSolver<EigenMatrixXf> eigensolver(A);
    eigenvalues = eigensolver.eigenvalues();
    eigenvectors = eigensolver.eigenvectors();
}

// -----------------------------------------------------------------------------
// 7. Compute determinant of a dense matrix
// -----------------------------------------------------------------------------
inline float determinant(const Eigen::Ref<const EigenMatrixXf>& A) {
    return A.determinant();
}

// -----------------------------------------------------------------------------
// 8. Compute matrix inverse
// -----------------------------------------------------------------------------
inline EigenMatrixXf inverse(const Eigen::Ref<const EigenMatrixXf>& A) {
    return A.inverse();
}

// -----------------------------------------------------------------------------
// 9. Trace
// -----------------------------------------------------------------------------
inline float trace(const Eigen::Ref<const EigenMatrixXf>& A) {
    return A.trace();
}

// -----------------------------------------------------------------------------
// 10. Conjugate Gradient for sparse or matrix‑free (already in numerical, but sparse‑aware version)
// -----------------------------------------------------------------------------
inline EigenVectorXf solve_sparse_cg(const SparseMatrix& A, const Eigen::Ref<const EigenVectorXf>& b,
                                     int max_iter = 200, float tolerance = 1e-6f) {
    Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper> cg;
    cg.compute(A);
    cg.setMaxIterations(max_iter);
    cg.setTolerance(tolerance);
    return cg.solve(b);
}

// -----------------------------------------------------------------------------
// 11. Sparse LU
// -----------------------------------------------------------------------------
inline EigenVectorXf solve_sparse_lu(SparseMatrix& A, const Eigen::Ref<const EigenVectorXf>& b) {
    Eigen::SparseLU<SparseMatrix> solver;
    solver.compute(A);
    return solver.solve(b);
}

// -----------------------------------------------------------------------------
// 12. Sparse Cholesky (requires SPD)
// -----------------------------------------------------------------------------
inline EigenVectorXf solve_sparse_cholesky(SparseMatrix& A, const Eigen::Ref<const EigenVectorXf>& b) {
    Eigen::SimplicialLDLT<SparseMatrix> solver;
    solver.compute(A);
    return solver.solve(b);
}

// -----------------------------------------------------------------------------
// 13. Build sparse matrix from triplets (convenience)
// -----------------------------------------------------------------------------
inline SparseMatrix sparse_from_triplets(int rows, int cols,
                                          const std::vector<Eigen::Triplet<float>>& triplets) {
    SparseMatrix mat(rows, cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
}

// -----------------------------------------------------------------------------
// 14. SIMD‑accelerated 2x2/3x3/4x4 determinant (using DirectXMath)
// -----------------------------------------------------------------------------
inline float det_2x2(float a00, float a01, float a10, float a11) noexcept {
    return a00*a11 - a01*a10;
}
inline float det_3x3(const DirectX::XMMATRIX& m) noexcept {
    DirectX::XMVECTOR det;
    DirectX::XMMatrixDeterminant(det, m);
    return DirectX::XMVectorGetX(det);
}
inline float det_4x4(const DirectX::XMMATRIX& m) noexcept {
    DirectX::XMVECTOR det;
    DirectX::XMMatrixDeterminant(det, m);
    return DirectX::XMVectorGetX(det);
}

// -----------------------------------------------------------------------------
// 15. Inverse of 2x2/3x3/4x4 via DirectXMath
// -----------------------------------------------------------------------------
inline DirectX::XMMATRIX inverse_4x4(const DirectX::XMMATRIX& m) noexcept {
    return DirectX::XMMatrixInverse(nullptr, m);
}
// For 2x2 and 3x3 we can convert to XMMATRIX with identity rows and use built‑in.

} // namespace linalg
} // namespace SimulationMath

#endif // CORE_MATH_LINEAR_ALGEBRA_H