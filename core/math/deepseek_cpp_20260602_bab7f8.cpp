//File 0098 : core/math/mesh_deformation.h
//As‑rigid‑as‑possible (ARAP) surface deformation using symmetric energy, local rotation fitting via SVD, global Poisson solve with handles as soft constraints, and full mathematical derivation.
#ifndef CORE_MATH_MESH_DEFORMATION_H
#define CORE_MATH_MESH_DEFORMATION_H

#include "mesh_data.h"                   // HalfEdgeMesh
#include "vector_math.h"
#include "linear_algebra.h"              // Eigen types, SVD
#include "mesh_laplacian_operators.h"    // cotangent_laplacian (provides L and M)
#include "math_constants.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

namespace SimulationMath {
namespace mesh_deformation {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. ARAP deformation context: stores original positions, cotangent Laplacian,
//    and one‑ring adjacency with weights for rotation fitting.
// -----------------------------------------------------------------------------
class ARAPDeformer {
public:
    ARAPDeformer(const HalfEdgeMesh& mesh) {
        const auto& verts = mesh.vertices();
        nv_ = verts.size();
        original_positions_.resize(nv_);
        for (size_t i = 0; i < nv_; ++i) original_positions_[i] = verts[i].position;

        // Precompute cotangent Laplacian L and lumped mass M (unused here but may be useful)
        laplacian::cotangent_laplacian(mesh, L_, M_);

        // Build one‑ring neighbour list with cotangent weights, identical to the off‑diagonal entries of L.
        build_one_ring_weights(mesh);

        current_positions_ = original_positions_;
    }

    // -----------------------------------------------------------------------
    // Set handle vertices and their target positions.
    // -----------------------------------------------------------------------
    void set_handles(const std::vector<uint32_t>& handle_vertices,
                     const std::vector<DirectX::XMVECTOR>& target_positions) noexcept {
        handle_vertices_ = handle_vertices;
        handle_targets_  = target_positions;
    }

    // -----------------------------------------------------------------------
    // Perform ARAP deformation (alternating local rotation estimation and global solve).
    // -----------------------------------------------------------------------
    void deform(int iterations = 3) noexcept {
        if (handle_vertices_.empty() || nv_ == 0) return;

        // Per‑vertex rotation matrices (initialised to identity)
        std::vector<Eigen::Matrix3f> rotations(nv_, Eigen::Matrix3f::Identity());

        for (int iter = 0; iter < iterations; ++iter) {
            estimate_rotations(rotations);      // Step 1: local rotation fitting
            solve_positions(rotations);         // Step 2: global Poisson system
        }
    }

    // -----------------------------------------------------------------------
    // Retrieve deformed vertex positions.
    // -----------------------------------------------------------------------
    std::vector<DirectX::XMVECTOR> get_deformed_positions() const noexcept {
        return current_positions_;
    }

private:
    size_t nv_;
    std::vector<DirectX::XMVECTOR> original_positions_;
    std::vector<DirectX::XMVECTOR> current_positions_;
    Eigen::SparseMatrix<float> L_;                          // cotangent Laplacian
    Eigen::DiagonalMatrix<float, Eigen::Dynamic> M_;        // lumped mass (unused)
    std::vector<uint32_t> handle_vertices_;
    std::vector<DirectX::XMVECTOR> handle_targets_;

    // One‑ring adjacency with cotangent weights (used for rotation estimation and RHS assembly)
    struct NeighborInfo {
        uint32_t idx;
        float weight;            // cotan weight = 0.5 * (cot α + cot β)
    };
    std::vector<std::vector<NeighborInfo>> one_ring_;

    // -----------------------------------------------------------------------
    // Build one‑ring neighbour list with symmetric cotangent weights.
    // These weights are identical to the off‑diagonal entries of L.
    // -----------------------------------------------------------------------
    void build_one_ring_weights(const HalfEdgeMesh& mesh) {
        one_ring_.resize(nv_);
        const auto& hedges = mesh.half_edges();
        const auto& verts  = mesh.vertices();

        for (size_t i = 0; i < hedges.size(); ++i) {
            const HalfEdge& he = hedges[i];
            if (he.twin_edge == 0xFFFFFFFFu) continue;
            const HalfEdge& twin = hedges[he.twin_edge];
            uint32_t vi = he.vertex_index;
            uint32_t vj = hedges[he.next_edge].vertex_index;
            uint32_t vopp_a = hedges[he.next_edge].next_edge;
            vopp_a = hedges[vopp_a].vertex_index;
            uint32_t vopp_b = hedges[twin.next_edge].next_edge;
            vopp_b = hedges[vopp_b].vertex_index;

            DirectX::XMVECTOR pi = verts[vi].position;
            DirectX::XMVECTOR pj = verts[vj].position;
            DirectX::XMVECTOR pa = verts[vopp_a].position;
            DirectX::XMVECTOR pb = verts[vopp_b].position;

            auto cot = [](DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) -> float {
                float dot = vector_math::dot3_scalar(a, b);
                DirectX::XMVECTOR cross = vector_math::cross3(a, b);
                float len_cross = vector_math::length3_scalar(cross);
                if (len_cross < 1e-12f) return 0.0f;
                return dot / len_cross;
            };
            float cot_a = cot(DirectX::XMVectorSubtract(pi, pa), DirectX::XMVectorSubtract(pj, pa));
            float cot_b = cot(DirectX::XMVectorSubtract(pi, pb), DirectX::XMVectorSubtract(pj, pb));
            float w = 0.5f * (cot_a + cot_b);

            one_ring_[vi].push_back({vj, w});
            one_ring_[vj].push_back({vi, w});
        }
    }

    // -----------------------------------------------------------------------
    // Estimate optimal rotation matrix for each vertex by minimising
    //   ∑_{j∈N(i)} w_ij ‖ (p_j' - p_i') - R_i (p_j - p_i) ‖²
    // Solution: R_i = V·U^T from SVD of covariance matrix S = ∑_j w_ij (e_orig_j) (e_cur_j)^T
    // -----------------------------------------------------------------------
    void estimate_rotations(std::vector<Eigen::Matrix3f>& rotations) noexcept {
        for (size_t i = 0; i < nv_; ++i) {
            Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
            DirectX::XMVECTOR pi_orig = original_positions_[i];
            DirectX::XMVECTOR pi_cur  = current_positions_[i];
            for (const auto& nb : one_ring_[i]) {
                uint32_t j = nb.idx;
                float w = nb.weight;
                DirectX::XMVECTOR pj_orig = original_positions_[j];
                DirectX::XMVECTOR pj_cur  = current_positions_[j];
                // Original edge vector: p_j - p_i
                Eigen::Vector3f e_orig(
                    vector_math::get_x(pj_orig) - vector_math::get_x(pi_orig),
                    vector_math::get_y(pj_orig) - vector_math::get_y(pi_orig),
                    vector_math::get_z(pj_orig) - vector_math::get_z(pi_orig));
                // Current edge vector: p_j' - p_i'
                Eigen::Vector3f e_cur(
                    vector_math::get_x(pj_cur) - vector_math::get_x(pi_cur),
                    vector_math::get_y(pj_cur) - vector_math::get_y(pi_cur),
                    vector_math::get_z(pj_cur) - vector_math::get_z(pi_cur));
                S += w * (e_orig * e_cur.transpose());
            }
            // SVD of S to get rotation
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
            if (R.determinant() < 0.0f) {
                // Ensure proper rotation (no reflection)
                Eigen::Matrix3f U = svd.matrixU();
                U.col(2) *= -1.0f;
                R = svd.matrixV() * U.transpose();
            }
            rotations[i] = R;
        }
    }

    // -----------------------------------------------------------------------
    // Global step: solve the symmetric ARAP linear system.
    //
    // Energy:
    //   E(p') = ∑_i ∑_{j∈N(i)} w_ij ‖ (p_i' - p_j') - (R_i + R_j)/2 (p_i - p_j) ‖²
    //
    // Setting ∇_{p_i'} E = 0 yields:
    //   ∑_{j∈N(i)} w_ij (p_i' - p_j') = ∑_{j∈N(i)} w_ij (R_i + R_j)/2 (p_i - p_j)
    //
    // The left‑hand side is exactly (L p')_i where L is the cotangent Laplacian.
    // The right‑hand side is denoted b_i.
    //
    // We enforce handle vertices as soft constraints: add a large weight λ = 1e8
    // to the diagonal and add λ * target to the RHS.
    // -----------------------------------------------------------------------
    void solve_positions(const std::vector<Eigen::Matrix3f>& rotations) noexcept {
        const float handle_weight = 1e8f;

        // Build right‑hand side B (n × 3)
        Eigen::MatrixXf B(nv_, 3);
        B.setZero();
        for (size_t i = 0; i < nv_; ++i) {
            Eigen::Vector3f bi = Eigen::Vector3f::Zero();
            DirectX::XMVECTOR pi_orig = original_positions_[i];
            const Eigen::Matrix3f& Ri = rotations[i];
            for (const auto& nb : one_ring_[i]) {
                uint32_t j = nb.idx;
                float w = nb.weight;
                DirectX::XMVECTOR pj_orig = original_positions_[j];
                const Eigen::Matrix3f& Rj = rotations[j];
                // Edge vector in reference: p_i - p_j
                float ex = vector_math::get_x(pi_orig) - vector_math::get_x(pj_orig);
                float ey = vector_math::get_y(pi_orig) - vector_math::get_y(pj_orig);
                float ez = vector_math::get_z(pi_orig) - vector_math::get_z(pj_orig);
                Eigen::Vector3f e_ref(ex, ey, ez);
                // Average rotation applied to the reference edge
                Eigen::Matrix3f Ravg = 0.5f * (Ri + Rj);
                Eigen::Vector3f rotated = Ravg * e_ref;   // (R_i + R_j)/2 * (p_i - p_j)
                bi += w * rotated;
            }
            B(i,0) = bi.x();
            B(i,1) = bi.y();
            B(i,2) = bi.z();
        }

        // Build the system matrix A = L + diag(handle_weight for handles)
        // Start with L as a sparse matrix.
        Eigen::SparseMatrix<float> A = L_;
        std::vector<Eigen::Triplet<float>> extra_triplets;
        // Add handle constraints to A and adjust B
        for (size_t k = 0; k < handle_vertices_.size(); ++k) {
            uint32_t v = handle_vertices_[k];
            if (v >= nv_) continue;
            extra_triplets.emplace_back(v, v, handle_weight);
            DirectX::XMVECTOR target = handle_targets_[k];
            B(v,0) += handle_weight * vector_math::get_x(target);
            B(v,1) += handle_weight * vector_math::get_y(target);
            B(v,2) += handle_weight * vector_math::get_z(target);
        }
        // Merge extra triplets into A (we create a new matrix to keep things simple)
        std::vector<Eigen::Triplet<float>> all_triplets;
        for (int k = 0; k < L_.outerSize(); ++k) {
            for (Eigen::SparseMatrix<float>::InnerIterator it(L_, k); it; ++it) {
                all_triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }
        for (const auto& t : extra_triplets) all_triplets.push_back(t);
        A.setFromTriplets(all_triplets.begin(), all_triplets.end());

        // Solve A X = B for each coordinate
        Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success) return;

        Eigen::MatrixXf X(nv_, 3);
        for (int c = 0; c < 3; ++c) {
            Eigen::VectorXf b = B.col(c);
            Eigen::VectorXf x = solver.solve(b);
            X.col(c) = x;
        }

        // Update current positions
        for (size_t i = 0; i < nv_; ++i) {
            current_positions_[i] = DirectX::XMVectorSet(X(i,0), X(i,1), X(i,2), 0.0f);
        }
    }
};

} // namespace mesh_deformation
} // namespace SimulationMath

#endif // CORE_MATH_MESH_DEFORMATION_H