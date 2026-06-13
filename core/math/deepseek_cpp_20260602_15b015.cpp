//File 0105 : core/math/mesh_spectral_analysis.h
//Spectral analysis of triangle meshes: small‑eigenvalue decomposition of cotangent Laplacian (generalized problem Lx = λMx), and simple shape features using the spectrum.
#ifndef CORE_MATH_MESH_SPECTRAL_ANALYSIS_H
#define CORE_MATH_MESH_SPECTRAL_ANALYSIS_H

#include "mesh_data.h"                   // HalfEdgeMesh
#include "mesh_laplacian_operators.h"    // cotangent_laplacian (L and M)
#include "linear_algebra.h"              // Eigen dense
#include "math_constants.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace SimulationMath {
namespace mesh_spectral {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. Compute the first k eigenpairs (eigenvalues and eigenvectors) of the generalized eigenvalue problem
//    L * x = λ * M * x, where L is cotangent Laplacian and M is lumped mass matrix.
//    This function converts the sparse matrices to dense and uses Eigen's GeneralizedSelfAdjointEigenSolver,
//    therefore it is intended for meshes with up to a few thousand vertices.
// -----------------------------------------------------------------------------
inline void compute_laplacian_spectrum(const HalfEdgeMesh& mesh,
                                       int k,
                                       std::vector<float>& eigenvalues,
                                       std::vector<std::vector<float>>& eigenvectors) noexcept {
    eigenvalues.clear();
    eigenvectors.clear();
    size_t nv = mesh.vertex_count();
    if (nv == 0 || k <= 0 || k > (int)nv) return;

    // Assemble L and M
    Eigen::SparseMatrix<float> L;
    Eigen::DiagonalMatrix<float, Eigen::Dynamic> M;
    laplacian::cotangent_laplacian(mesh, L, M);   // from mesh_laplacian_operators.h

    // Convert to dense matrices (since Eigen's generalized eigensolver is dense)
    Eigen::MatrixXf L_dense = Eigen::MatrixXf(L);
    Eigen::MatrixXf M_dense = M.toDenseMatrix();

    // Solve generalized symmetric eigenvalue problem L x = λ M x
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXf> solver(L_dense, M_dense);
    if (solver.info() != Eigen::Success) return;

    // Eigenvalues are sorted in increasing order
    Eigen::VectorXf evals = solver.eigenvalues();
    Eigen::MatrixXf evecs = solver.eigenvectors();

    int actual_k = std::min(k, (int)evals.size());
    eigenvalues.resize(actual_k);
    eigenvectors.resize(actual_k, std::vector<float>(nv, 0.0f));
    for (int i = 0; i < actual_k; ++i) {
        eigenvalues[i] = evals(i);
        for (size_t j = 0; j < nv; ++j) {
            eigenvectors[i][j] = evecs(j, i);
        }
    }
}

// -----------------------------------------------------------------------------
// 2. Compute the heat kernel signature (HKS) at vertices using spectral approximation:
//    hks_i(t) = Σ_{l=0}^{k-1} exp(-λ_l * t) * φ_l(i)^2
//    t is the time parameter.
// -----------------------------------------------------------------------------
inline std::vector<float> heat_kernel_signature(const std::vector<float>& eigenvalues,
                                                const std::vector<std::vector<float>>& eigenvectors,
                                                float t) noexcept {
    size_t nv = eigenvectors.empty() ? 0 : eigenvectors[0].size();
    std::vector<float> hks(nv, 0.0f);
    int k = std::min((int)eigenvalues.size(), (int)eigenvectors.size());
    for (int l = 0; l < k; ++l) {
        float coeff = std::exp(-eigenvalues[l] * t);
        for (size_t i = 0; i < nv; ++i) {
            float val = eigenvectors[l][i];
            hks[i] += coeff * val * val;
        }
    }
    return hks;
}

// -----------------------------------------------------------------------------
// 3. Approximate the commute‑time distance between two vertices using the spectral formula:
//    d_ct(i,j)^2 = Σ_{l=1}^{k-1} (1/λ_l) * (φ_l(i) - φ_l(j))^2
// -----------------------------------------------------------------------------
inline float commute_time_distance(const std::vector<float>& eigenvalues,
                                   const std::vector<std::vector<float>>& eigenvectors,
                                   uint32_t vi, uint32_t vj) noexcept {
    float sum = 0.0f;
    int k = std::min((int)eigenvalues.size(), (int)eigenvectors.size());
    // Start from l=1 to skip the trivial zero eigenvalue (λ0 ≈ 0)
    for (int l = 1; l < k; ++l) {
        float inv_lambda = (eigenvalues[l] > 1e-12f) ? 1.0f / eigenvalues[l] : 0.0f;
        float diff = eigenvectors[l][vi] - eigenvectors[l][vj];
        sum += inv_lambda * diff * diff;
    }
    return std::sqrt(sum);
}

} // namespace mesh_spectral
} // namespace SimulationMath

#endif // CORE_MATH_MESH_SPECTRAL_ANALYSIS_H