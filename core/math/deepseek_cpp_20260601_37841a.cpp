//7/40
//File 0087 : core/math/mesh_parameterization.h
//Discrete harmonic (Tutte) mesh parameterization for disk‑topology surfaces: boundary mapped to unit circle via chord‑length, interior solved via cotangent Laplacian, and UV stored as vertex colors.
#ifndef CORE_MATH_MESH_PARAMETERIZATION_H
#define CORE_MATH_MESH_PARAMETERIZATION_H

#include "mesh_data.h"               // HalfEdgeMesh
#include "vector_math.h"
#include "linear_algebra.h"          // Eigen types, sparse solvers
#include "math_constants.h"
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>

namespace SimulationMath {
namespace parameterization {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. Collect boundary vertices in order (assumes single closed boundary)
// -----------------------------------------------------------------------------
inline std::vector<uint32_t> find_boundary_cycle(const HalfEdgeMesh& mesh) noexcept {
    const auto& hedges = mesh.half_edges();
    std::vector<uint32_t> cycle;
    // locate a boundary half‑edge
    uint32_t start_he = 0xFFFFFFFFu;
    for (size_t i = 0; i < hedges.size(); ++i) {
        if (hedges[i].twin_edge == 0xFFFFFFFFu) {
            start_he = (uint32_t)i;
            break;
        }
    }
    if (start_he == 0xFFFFFFFFu) return cycle; // no boundary

    // walk along the boundary face
    uint32_t he = start_he;
    do {
        cycle.push_back(hedges[he].vertex_index);
        he = hedges[he].next_edge;
    } while (he != start_he);
    return cycle;
}

// -----------------------------------------------------------------------------
// 2. Tutte embedding: boundary → circle, interior → harmonic (cotan Laplacian)
//    Returns per‑vertex UV coordinates packed into the mesh vertex color attribute.
// -----------------------------------------------------------------------------
inline void tutte_embedding(HalfEdgeMesh& mesh) noexcept {
    size_t nv = mesh.vertex_count();
    if (nv == 0) return;

    std::vector<uint32_t> boundary = find_boundary_cycle(mesh);
    if (boundary.empty()) return;

    const auto& verts = mesh.vertices();
    const auto& hedges = mesh.half_edges();
    size_t bsize = boundary.size();

    // Step 1: chord‑length parameterization of the boundary → unit circle
    std::vector<float> boundary_dist(bsize + 1, 0.0f);
    for (size_t i = 0; i < bsize; ++i) {
        uint32_t v0 = boundary[i];
        uint32_t v1 = boundary[(i + 1) % bsize];
        DirectX::XMVECTOR p0 = verts[v0].position;
        DirectX::XMVECTOR p1 = verts[v1].position;
        float len = vector_math::length3_scalar(DirectX::XMVectorSubtract(p1, p0));
        boundary_dist[i + 1] = boundary_dist[i] + len;
    }
    float total_len = boundary_dist.back();
    if (total_len < 1e-12f) total_len = 1.0f;

    // UV storage for solving
    std::vector<std::pair<float, float>> uv(nv, {0.0f, 0.0f});
    for (size_t i = 0; i < bsize; ++i) {
        float t = boundary_dist[i] / total_len;
        float angle = t * 2.0f * constants::PIf;
        uv[boundary[i]] = {std::cos(angle), std::sin(angle)};
    }

    // Mark interior vertices
    std::vector<bool> is_interior(nv, true);
    for (uint32_t v : boundary) is_interior[v] = false;
    int interior_count = (int)std::count(is_interior.begin(), is_interior.end(), true);
    if (interior_count == 0) {
        // only boundary, write UV to vertex colors and return
        for (size_t i = 0; i < nv; ++i) {
            auto& col = const_cast<DirectX::XMVECTOR&>(verts[i].color);
            col = DirectX::XMVectorSet(uv[i].first, uv[i].second, 0.0f, 1.0f);
        }
        return;
    }

    // Step 2: assemble cotangent Laplacian L
    Eigen::SparseMatrix<float> L(nv, nv);
    std::vector<Eigen::Triplet<float>> triplets;
    std::vector<float> diag(nv, 0.0f);
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
        DirectX::XMVECTOR popp_a = verts[vopp_a].position;
        DirectX::XMVECTOR popp_b = verts[vopp_b].position;

        auto cot = [](DirectX::FXMVECTOR a, DirectX::FXMVECTOR b) -> float {
            float dot = vector_math::dot3_scalar(a, b);
            DirectX::XMVECTOR cross = vector_math::cross3(a, b);
            float len_cross = vector_math::length3_scalar(cross);
            if (len_cross < 1e-12f) return 0.0f;
            return dot / len_cross;
        };
        float cot_a = cot(DirectX::XMVectorSubtract(pi, popp_a),
                          DirectX::XMVectorSubtract(pj, popp_a));
        float cot_b = cot(DirectX::XMVectorSubtract(pi, popp_b),
                          DirectX::XMVectorSubtract(pj, popp_b));
        float w = 0.5f * (cot_a + cot_b);

        if (vi < nv && vj < nv) {
            triplets.emplace_back(vi, vj, w);
            triplets.emplace_back(vj, vi, w);
            diag[vi] -= w;
            diag[vj] -= w;
        }
    }
    for (size_t i = 0; i < nv; ++i)
        triplets.emplace_back(i, i, diag[i]);
    L.setFromTriplets(triplets.begin(), triplets.end());

    // Step 3: build interior subsystem L_int * u_int = -L_bound * u_bound
    std::vector<int> interior_idx(nv, -1);
    int idx = 0;
    for (size_t v = 0; v < nv; ++v) {
        if (is_interior[v]) interior_idx[v] = idx++;
    }

    Eigen::SparseMatrix<float> Lint(interior_count, interior_count);
    Eigen::MatrixXf B(interior_count, 2);
    B.setZero();
    std::vector<Eigen::Triplet<float>> lint_triplets;

    for (int k = 0; k < L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<float>::InnerIterator it(L, k); it; ++it) {
            size_t row = it.row();
            size_t col = it.col();
            float val = it.value();
            if (is_interior[row] && is_interior[col]) {
                lint_triplets.emplace_back(interior_idx[row], interior_idx[col], val);
            } else if (is_interior[row] && !is_interior[col]) {
                // boundary contribution to right‑hand side
                float u_val = uv[col].first;
                float v_val = uv[col].second;
                B(interior_idx[row], 0) -= val * u_val;
                B(interior_idx[row], 1) -= val * v_val;
            }
        }
    }
    Lint.setFromTriplets(lint_triplets.begin(), lint_triplets.end());

    // Step 4: solve for interior UV coordinates
    Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
    solver.compute(Lint);
    if (solver.info() != Eigen::Success) return;

    for (int c = 0; c < 2; ++c) {
        Eigen::VectorXf b = B.col(c);
        Eigen::VectorXf x = solver.solve(b);
        for (size_t v = 0; v < nv; ++v) {
            if (is_interior[v]) {
                if (c == 0) uv[v].first  = x[interior_idx[v]];
                else        uv[v].second = x[interior_idx[v]];
            }
        }
    }

    // Step 5: store UV coordinates in vertex color channels (r=u, g=v, b=0, a=1)
    for (size_t i = 0; i < nv; ++i) {
        auto& col = const_cast<DirectX::XMVECTOR&>(verts[i].color);
        col = DirectX::XMVectorSet(uv[i].first, uv[i].second, 0.0f, 1.0f);
    }
    mesh.compute_vertex_normals(); // keep normals unchanged
}

} // namespace parameterization
} // namespace SimulationMath

#endif // CORE_MATH_MESH_PARAMETERIZATION_H