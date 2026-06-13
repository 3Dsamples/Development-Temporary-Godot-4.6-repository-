//6/40
//File 0086 : core/math/mesh_simplification.h
//Quadric error metric edge‑collapse simplification for triangle meshes: vertex quadrics, optimal contraction position, greedy heap‑based collapse with full topology update.
#ifndef CORE_MATH_MESH_SIMPLIFICATION_H
#define CORE_MATH_MESH_SIMPLIFICATION_H

#include "mesh_data.h"               // HalfEdgeMesh (only for input/output)
#include "vector_math.h"
#include "linear_algebra.h"          // Eigen types
#include "math_constants.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <unordered_set>
#include <queue>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <algorithm>
#include <set>

namespace SimulationMath {
namespace simplification {

// -----------------------------------------------------------------------------
// 1. Quadric – symmetric 4×4 matrix representing squared distance to planes
// -----------------------------------------------------------------------------
struct Quadric {
    Eigen::Matrix4f Q;
    Quadric() noexcept : Q(Eigen::Matrix4f::Zero()) {}
    explicit Quadric(const Eigen::Matrix4f& m) noexcept : Q(m) {}
    void add(const Quadric& other) noexcept { Q += other.Q; }
    float evaluate(const DirectX::XMVECTOR& p) const noexcept {
        Eigen::Vector4f hp(vector_math::get_x(p), vector_math::get_y(p),
                           vector_math::get_z(p), 1.0f);
        return hp.dot(Q * hp);
    }
};

// -----------------------------------------------------------------------------
// 2. Build quadric for a triangle given its unit normal and distance from origin
// -----------------------------------------------------------------------------
inline Quadric plane_quadric(const DirectX::XMVECTOR& normal, float d) noexcept {
    float nx = vector_math::get_x(normal), ny = vector_math::get_y(normal), nz = vector_math::get_z(normal);
    Quadric q;
    q.Q << nx*nx, nx*ny, nx*nz, nx*d,
           nx*ny, ny*ny, ny*nz, ny*d,
           nx*nz, ny*nz, nz*nz, nz*d,
           nx*d,  ny*d,  nz*d,  d*d;
    return q;
}

// -----------------------------------------------------------------------------
// 3. Optimal contraction target that minimizes Q0+Q1, solving 3×3 system
// -----------------------------------------------------------------------------
inline bool optimal_contraction(const Quadric& q0, const Quadric& q1,
                                const DirectX::XMVECTOR& p0, const DirectX::XMVECTOR& p1,
                                DirectX::XMVECTOR& out_opt, float& out_error) noexcept {
    Quadric q = q0;
    q.add(q1);
    Eigen::Matrix4f A = q.Q;
    Eigen::Matrix3f A33 = A.topLeftCorner<3,3>();
    Eigen::Vector3f b   = -A.block<3,1>(0,3);   // because gradient: A33 * x + A34 = 0
    Eigen::ColPivHouseholderQR<Eigen::Matrix3f> solver(A33);
    if (solver.rank() < 3) {
        out_opt = DirectX::XMVectorScale(DirectX::XMVectorAdd(p0, p1), 0.5f);
        out_error = q.evaluate(out_opt);
        return true;
    }
    Eigen::Vector3f x = solver.solve(b);
    out_opt = DirectX::XMVectorSet(x(0), x(1), x(2), 0.0f);
    out_error = q.evaluate(out_opt);
    if (!std::isfinite(out_error)) {
        out_opt = DirectX::XMVectorScale(DirectX::XMVectorAdd(p0, p1), 0.5f);
        out_error = q.evaluate(out_opt);
    }
    return true;
}

// -----------------------------------------------------------------------------
// 4. Edge collapse candidate (for min‑heap)
// -----------------------------------------------------------------------------
struct CollapseEdge {
    uint32_t v0, v1;
    DirectX::XMVECTOR target;
    float cost;
    bool operator<(const CollapseEdge& o) const noexcept { return cost > o.cost; }
};

// -----------------------------------------------------------------------------
// 5. Mesh simplifier – works on own copy of vertices and faces, produces HalfEdgeMesh
// -----------------------------------------------------------------------------
class MeshSimplifier {
public:
    MeshSimplifier(const HalfEdgeMesh& mesh, size_t target_face_count = 0)
        : target_faces_(target_face_count) {
        // extract vertices and triangular faces from the half‑edge structure
        const auto& verts = mesh.vertices();
        const auto& hedges = mesh.half_edges();
        const auto& faces = mesh.faces();
        size_t nv = verts.size();
        vertices_.resize(nv);
        for (size_t i = 0; i < nv; ++i)
            vertices_[i] = verts[i].position;
        // store only triangles (ignore polygons with >3 vertices for simplicity)
        for (const auto& face : faces) {
            uint32_t he0 = face.first_edge;
            if (he0 == 0xFFFFFFFFu) continue;
            std::vector<uint32_t> v;
            uint32_t he = he0;
            do {
                v.push_back(hedges[he].vertex_index);
                he = hedges[he].next_edge;
            } while (he != he0);
            if (v.size() == 3)
                faces_.push_back(v);
        }
        removed_.assign(nv, false);
        compute_quadrics();
        build_heap();
    }

    // -----------------------------------------------------------------------
    // Run simplification; returns the resulting mesh
    // -----------------------------------------------------------------------
    HalfEdgeMesh simplify() {
        size_t current_faces = faces_.size();
        while (!heap_.empty() && (target_faces_ == 0 || current_faces > target_faces_)) {
            CollapseEdge top = heap_.top();
            heap_.pop();

            if (removed_[top.v0] || removed_[top.v1]) continue;
            // Optional: check that the edge still exists in the current face list
            if (!edge_exists(top.v0, top.v1)) continue;

            perform_collapse(top.v0, top.v1, top.target);
            current_faces = faces_.size();
        }
        return build_result();
    }

private:
    std::vector<DirectX::XMVECTOR> vertices_;
    std::vector<std::vector<uint32_t>> faces_;   // each entry: triangle indices
    std::vector<bool> removed_;
    size_t target_faces_;
    std::vector<Quadric> quadrics_;
    std::priority_queue<CollapseEdge> heap_;

    // -----------------------------------------------------------------------
    // Compute initial quadrics from all faces
    // -----------------------------------------------------------------------
    void compute_quadrics() {
        quadrics_.assign(vertices_.size(), Quadric());
        for (const auto& f : faces_) {
            DirectX::XMVECTOR p0 = vertices_[f[0]];
            DirectX::XMVECTOR p1 = vertices_[f[1]];
            DirectX::XMVECTOR p2 = vertices_[f[2]];
            DirectX::XMVECTOR normal = vector_math::normalize3(vector_math::cross3(
                DirectX::XMVectorSubtract(p1, p0), DirectX::XMVectorSubtract(p2, p0)));
            float d = -vector_math::dot3_scalar(normal, p0);
            Quadric qf = plane_quadric(normal, d);
            for (uint32_t idx : f) quadrics_[idx].add(qf);
        }
    }

    // -----------------------------------------------------------------------
    // Build heap of all edges (each unique edge appears once)
    // -----------------------------------------------------------------------
    void build_heap() {
        std::set<uint64_t> inserted;
        for (const auto& f : faces_) {
            for (int i = 0; i < 3; ++i) {
                uint32_t v0 = f[i], v1 = f[(i+1)%3];
                if (v0 > v1) std::swap(v0, v1);
                uint64_t key = (uint64_t(v0) << 32) | v1;
                if (inserted.count(key)) continue;
                inserted.insert(key);
                DirectX::XMVECTOR target;
                float cost;
                if (optimal_contraction(quadrics_[v0], quadrics_[v1],
                                        vertices_[v0], vertices_[v1], target, cost)) {
                    heap_.push({v0, v1, target, cost});
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Check if an edge (v0,v1) still appears in any face
    // -----------------------------------------------------------------------
    bool edge_exists(uint32_t v0, uint32_t v1) const noexcept {
        for (const auto& f : faces_) {
            bool h0 = false, h1 = false;
            for (uint32_t v : f) {
                if (v == v0) h0 = true;
                if (v == v1) h1 = true;
            }
            if (h0 && h1) return true;
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // Collapse edge: v1 removed, v0 moved to target; faces updated accordingly
    // -----------------------------------------------------------------------
    void perform_collapse(uint32_t v0, uint32_t v1, DirectX::FXMVECTOR target) {
        // move v0
        vertices_[v0] = target;
        // process faces: remove those that contain both v0 and v1; otherwise replace v1 with v0
        std::vector<std::vector<uint32_t>> new_faces;
        for (const auto& f : faces_) {
            bool has_v0 = false, has_v1 = false;
            for (uint32_t v : f) {
                if (v == v0) has_v0 = true;
                if (v == v1) has_v1 = true;
            }
            if (has_v1) {
                if (has_v0) {
                    // degenerate triangle – discard
                    continue;
                } else {
                    // replace v1 by v0
                    auto tmp = f;
                    for (uint32_t& v : tmp) if (v == v1) v = v0;
                    // check for duplicate vertices in the triangle
                    if (tmp[0] != tmp[1] && tmp[1] != tmp[2] && tmp[2] != tmp[0])
                        new_faces.push_back(tmp);
                }
            } else {
                new_faces.push_back(f);
            }
        }
        faces_.swap(new_faces);
        removed_[v1] = true;

        // recompute quadrics for v0 and all adjacent vertices (neighbors via remaining faces)
        recompute_quadrics_affected(v0);
        // rebuild heap (could be done incrementally, but rebuilding is simpler and correct)
        rebuild_heap();
    }

    // -----------------------------------------------------------------------
    // Recompute quadrics for v0 and all vertices sharing a face with it
    // -----------------------------------------------------------------------
    void recompute_quadrics_affected(uint32_t v0) {
        // collect all vertices that need recomputation
        std::set<uint32_t> affected;
        for (const auto& f : faces_) {
            bool has_v0 = false;
            for (uint32_t v : f) if (v == v0) { has_v0 = true; break; }
            if (has_v0) {
                for (uint32_t v : f) affected.insert(v);
            }
        }
        // recompute quadrics for affected vertices from scratch (sum over incident faces)
        for (uint32_t v : affected) {
            quadrics_[v] = Quadric();
        }
        for (const auto& f : faces_) {
            DirectX::XMVECTOR p0 = vertices_[f[0]];
            DirectX::XMVECTOR p1 = vertices_[f[1]];
            DirectX::XMVECTOR p2 = vertices_[f[2]];
            DirectX::XMVECTOR normal = vector_math::normalize3(vector_math::cross3(
                DirectX::XMVectorSubtract(p1, p0), DirectX::XMVectorSubtract(p2, p0)));
            float d = -vector_math::dot3_scalar(normal, p0);
            Quadric qf = plane_quadric(normal, d);
            for (uint32_t idx : f) {
                if (affected.count(idx)) quadrics_[idx].add(qf);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Rebuild the heap from scratch (O(E log E))
    // -----------------------------------------------------------------------
    void rebuild_heap() {
        heap_ = {};
        std::set<uint64_t> inserted;
        for (const auto& f : faces_) {
            for (int i = 0; i < 3; ++i) {
                uint32_t v0 = f[i], v1 = f[(i+1)%3];
                if (v0 > v1) std::swap(v0, v1);
                uint64_t key = (uint64_t(v0) << 32) | v1;
                if (inserted.count(key)) continue;
                inserted.insert(key);
                DirectX::XMVECTOR target;
                float cost;
                if (optimal_contraction(quadrics_[v0], quadrics_[v1],
                                        vertices_[v0], vertices_[v1], target, cost)) {
                    heap_.push({v0, v1, target, cost});
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Build a HalfEdgeMesh from the remaining vertices and faces
    // -----------------------------------------------------------------------
    HalfEdgeMesh build_result() {
        // compact vertices: map old index -> new index (only non‑removed)
        std::vector<uint32_t> old_to_new(vertices_.size(), 0xFFFFFFFFu);
        std::vector<DirectX::XMVECTOR> new_verts;
        for (size_t i = 0; i < vertices_.size(); ++i) {
            if (!removed_[i]) {
                old_to_new[i] = static_cast<uint32_t>(new_verts.size());
                new_verts.push_back(vertices_[i]);
            }
        }
        // collect faces using new indices
        std::vector<std::vector<uint32_t>> new_faces;
        for (const auto& f : faces_) {
            std::vector<uint32_t> nf;
            for (uint32_t v : f) {
                if (old_to_new[v] != 0xFFFFFFFFu)
                    nf.push_back(old_to_new[v]);
            }
            if (nf.size() == 3)
                new_faces.push_back(nf);
        }
        // construct HalfEdgeMesh
        HalfEdgeMesh result;
        for (const auto& p : new_verts) result.add_vertex(p);
        for (const auto& f : new_faces) result.add_face(f);
        result.link_twins();
        result.compute_face_normals();
        result.compute_vertex_normals();
        return result;
    }
};

} // namespace simplification
} // namespace SimulationMath

#endif // CORE_MATH_MESH_SIMPLIFICATION_H