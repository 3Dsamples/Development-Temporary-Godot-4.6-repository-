//9/40
//File 0089 : core/math/mesh_laplacian_operators.h
//Discrete Laplacian and mass matrices on triangle meshes: uniform, cotangent, mean‑value weights, Voronoi mass, and a generic Poisson solver for scalar fields.
#ifndef CORE_MATH_MESH_LAPLACIAN_OPERATORS_H
#define CORE_MATH_MESH_LAPLACIAN_OPERATORS_H

#include "mesh_data.h"               // HalfEdgeMesh
#include "vector_math.h"
#include "linear_algebra.h"          // Eigen sparse matrices
#include "math_constants.h"
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

namespace SimulationMath {
namespace laplacian {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. Uniform Laplacian (graph Laplacian) – Lij = 1 if edge, diagonal = -degree
// -----------------------------------------------------------------------------
inline Eigen::SparseMatrix<float> uniform_laplacian(const HalfEdgeMesh& mesh) noexcept {
    size_t nv = mesh.vertex_count();
    Eigen::SparseMatrix<float> L(nv, nv);
    std::vector<Eigen::Triplet<float>> triplets;
    std::vector<float> diag(nv, 0.0f);
    const auto& hedges = mesh.half_edges();
    for (size_t i = 0; i < hedges.size(); ++i) {
        const HalfEdge& he = hedges[i];
        if (he.twin_edge == 0xFFFFFFFFu) continue;
        uint32_t vi = he.vertex_index;
        uint32_t vj = hedges[he.next_edge].vertex_index;
        triplets.emplace_back(vi, vj, 1.0f);
        triplets.emplace_back(vj, vi, 1.0f);
        diag[vi] -= 1.0f;
        diag[vj] -= 1.0f;
    }
    for (size_t i = 0; i < nv; ++i) triplets.emplace_back(i, i, diag[i]);
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

// -----------------------------------------------------------------------------
// 2. Cotangent Laplacian (standard, symmetric) + lumped mass matrix (area/3)
// -----------------------------------------------------------------------------
inline void cotangent_laplacian(const HalfEdgeMesh& mesh,
                                Eigen::SparseMatrix<float>& L,
                                Eigen::DiagonalMatrix<float, Eigen::Dynamic>& M) noexcept {
    size_t nv = mesh.vertex_count();
    L.resize(nv, nv);
    M = Eigen::DiagonalMatrix<float, Eigen::Dynamic>(nv);
    M.setZero();
    std::vector<Eigen::Triplet<float>> triplets;
    std::vector<float> diag(nv, 0.0f);
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
        if (vi < nv && vj < nv) {
            triplets.emplace_back(vi, vj, w);
            triplets.emplace_back(vj, vi, w);
            diag[vi] -= w;
            diag[vj] -= w;
        }
    }
    for (size_t i = 0; i < nv; ++i) triplets.emplace_back(i, i, diag[i]);
    L.setFromTriplets(triplets.begin(), triplets.end());

    // Mass (lumped)
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        const MeshFace& face = mesh.faces()[f];
        uint32_t he0 = face.first_edge;
        if (he0 == 0xFFFFFFFFu) continue;
        uint32_t v0 = hedges[he0].vertex_index;
        uint32_t v1 = hedges[hedges[he0].next_edge].vertex_index;
        uint32_t v2 = hedges[hedges[he0].next_edge].next_edge;
        v2 = hedges[v2].vertex_index;
        DirectX::XMVECTOR p0 = verts[v0].position;
        DirectX::XMVECTOR p1 = verts[v1].position;
        DirectX::XMVECTOR p2 = verts[v2].position;
        DirectX::XMVECTOR cross = vector_math::cross3(DirectX::XMVectorSubtract(p1, p0),
                                                      DirectX::XMVectorSubtract(p2, p0));
        float area = 0.5f * vector_math::length3_scalar(cross);
        float contrib = area / 3.0f;
        M.diagonal()(v0) += contrib;
        M.diagonal()(v1) += contrib;
        M.diagonal()(v2) += contrib;
    }
}

// -----------------------------------------------------------------------------
// 3. Mean‑value Laplacian (better for irregular meshes) – weights derived from angle
// -----------------------------------------------------------------------------
inline Eigen::SparseMatrix<float> mean_value_laplacian(const HalfEdgeMesh& mesh) noexcept {
    size_t nv = mesh.vertex_count();
    Eigen::SparseMatrix<float> L(nv, nv);
    std::vector<Eigen::Triplet<float>> triplets;
    std::vector<float> diag(nv, 0.0f);
    const auto& hedges = mesh.half_edges();
    const auto& verts  = mesh.vertices();

    for (size_t i = 0; i < hedges.size(); ++i) {
        const HalfEdge& he = hedges[i];
        if (he.twin_edge == 0xFFFFFFFFu) continue;
        uint32_t vi = he.vertex_index;
        uint32_t vj = hedges[he.next_edge].vertex_index;

        // For edge (vi,vj), compute the two angles opposite to this edge in its two adjacent triangles.
        uint32_t twin = he.twin_edge;
        uint32_t vopp_a = hedges[hedges[he].next_edge].next_edge;
        vopp_a = hedges[vopp_a].vertex_index;
        uint32_t vopp_b = hedges[hedges[twin].next_edge].next_edge;
        vopp_b = hedges[vopp_b].vertex_index;

        DirectX::XMVECTOR pi = verts[vi].position;
        DirectX::XMVECTOR pj = verts[vj].position;
        DirectX::XMVECTOR pa = verts[vopp_a].position;
        DirectX::XMVECTOR pb = verts[vopp_b].position;

        auto angle_at = [&](DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, DirectX::FXMVECTOR c) {
            DirectX::XMVECTOR u = DirectX::XMVectorSubtract(b, a);
            DirectX::XMVECTOR v = DirectX::XMVectorSubtract(c, a);
            float dot = vector_math::dot3_scalar(u, v);
            float lenu = vector_math::length3_scalar(u);
            float lenv = vector_math::length3_scalar(v);
            if (lenu < 1e-12f || lenv < 1e-12f) return 0.0f;
            return std::acos(std::clamp(dot / (lenu * lenv), -1.0f, 1.0f));
        };
        float angle_a = angle_at(vopp_a, vi, vj);  // angle at opposite vertex in face A
        float angle_b = angle_at(vopp_b, vi, vj);

        // Mean value weight = (tan(angle/2)) / distance, but common version uses tan(angle/2).
        // Here we use the standard mean‑value weights for Laplacian: w_ij = (tan(alpha/2) + tan(beta/2)) / ||vi - vj||
        float dist = vector_math::length3_scalar(DirectX::XMVectorSubtract(pj, pi));
        if (dist < 1e-12f) continue;
        float w = (std::tan(angle_a * 0.5f) + std::tan(angle_b * 0.5f)) / dist;
        if (vi < nv && vj < nv) {
            triplets.emplace_back(vi, vj, w);
            triplets.emplace_back(vj, vi, w);
            diag[vi] -= w;
            diag[vj] -= w;
        }
    }
    for (size_t i = 0; i < nv; ++i) triplets.emplace_back(i, i, diag[i]);
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

// -----------------------------------------------------------------------------
// 4. Voronoi area mass matrix (dual cell area)
// -----------------------------------------------------------------------------
inline Eigen::DiagonalMatrix<float, Eigen::Dynamic> voronoi_mass(const HalfEdgeMesh& mesh) noexcept {
    size_t nv = mesh.vertex_count();
    Eigen::DiagonalMatrix<float, Eigen::Dynamic> M(nv);
    M.setZero();
    const auto& hedges = mesh.half_edges();
    const auto& verts  = mesh.vertices();
    // For each vertex, accumulate area using cotangent formula for Voronoi region.
    for (size_t i = 0; i < hedges.size(); ++i) {
        const HalfEdge& he = hedges[i];
        if (he.twin_edge == 0xFFFFFFFFu) continue;
        uint32_t vi = he.vertex_index;
        uint32_t vj = hedges[he.next_edge].vertex_index;
        uint32_t vopp_a = hedges[he.next_edge].next_edge;
        vopp_a = hedges[vopp_a].vertex_index;
        uint32_t vopp_b = hedges[hedges[he.twin_edge].next_edge].next_edge;
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
        // The Voronoi area contribution to vertex vi from this edge: (cot_a + cot_b) * ||vi-vj||^2 / 8
        float len2 = vector_math::length_sq3_scalar(DirectX::XMVectorSubtract(pj, pi));
        float contrib = (cot_a + cot_b) * len2 / 8.0f;
        M.diagonal()(vi) += contrib;
        // symmetric contribution to vj? Actually the area at vi is sum over incident edges. So we add only to vi.
        // But the edge also contributes to vj? No, Voronoi area at vi only. This loop processes each half‑edge vi, so it's fine.
    }
    // For boundary vertices, Voronoi area may be underestimated; we ignore correction.
    return M;
}

// -----------------------------------------------------------------------------
// 5. Poisson solver: solve L * X = b, with optional constraint fixing a vertex.
//    If pinned vertex index given, its value is set to pinned_value.
// -----------------------------------------------------------------------------
inline Eigen::VectorXf solve_poisson(const Eigen::SparseMatrix<float>& L,
                                     const Eigen::VectorXf& b,
                                     uint32_t pinned_vertex = 0xFFFFFFFFu,
                                     float pinned_value = 0.0f) noexcept {
    size_t n = L.rows();
    if (pinned_vertex < n) {
        // Modify L and b to enforce Dirichlet condition.
        Eigen::SparseMatrix<float> Lmod = L;
        Eigen::VectorXf bmod = b;
        // Zero out row/col of pinned vertex, set diagonal = 1, bmod[pinned] = pinned_value
        // We'll rebuild with triplets.
        std::vector<Eigen::Triplet<float>> triplets;
        for (int k = 0; k < L.outerSize(); ++k) {
            for (Eigen::SparseMatrix<float>::InnerIterator it(L, k); it; ++it) {
                if (it.row() != pinned_vertex && it.col() != pinned_vertex)
                    triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }
        triplets.emplace_back(pinned_vertex, pinned_vertex, 1.0f);
        Lmod.setFromTriplets(triplets.begin(), triplets.end());
        bmod[pinned_vertex] = pinned_value;

        Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
        solver.compute(Lmod);
        if (solver.info() != Eigen::Success) return b;
        return solver.solve(bmod);
    } else {
        // No constraint; use CG with positive semi-definite L (might need regularization)
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> cg;
        cg.compute(L);
        cg.setTolerance(1e-6f);
        cg.setMaxIterations(500);
        return cg.solve(b);
    }
}

} // namespace laplacian
} // namespace SimulationMath

#endif // CORE_MATH_MESH_LAPLACIAN_OPERATORS_H