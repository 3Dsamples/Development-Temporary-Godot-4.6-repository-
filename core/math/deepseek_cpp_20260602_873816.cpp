//File 0100 : core/math/mesh_registration.h
//Iterative Closest Point (ICP) rigid registration: point‑to‑point with spatial‑hash NN, point‑to‑plane with linearised Gauss‑Newton, and mesh‑to‑mesh via distance query; full mathematical derivations.
#ifndef CORE_MATH_MESH_REGISTRATION_H
#define CORE_MATH_MESH_REGISTRATION_H

#include "vector_math.h"
#include "geometry_primitives.h"         // AABB, Triangle, closest_point_triangle
#include "rigid_transform_fit.h"        // find_rigid_transform (Kabsch)
#include "mesh_distance.h"              // MeshDistanceQuery, closest_point_on_mesh
#include "math_constants.h"
#include "linear_algebra.h"             // Eigen types, SVD, Cholesky
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <tuple>

namespace SimulationMath {
namespace mesh_registration {

// -----------------------------------------------------------------------------
// 1. ICP result structure
// -----------------------------------------------------------------------------
struct ICPResult {
    DirectX::XMMATRIX rotation;
    DirectX::XMVECTOR translation;
    float rms_error;
    bool converged;
};

// -----------------------------------------------------------------------------
// 2. Simple spatial hash grid for fast nearest‑neighbour search (used in ICP)
// -----------------------------------------------------------------------------
class PointCloudHashGrid {
public:
    PointCloudHashGrid(const std::vector<DirectX::XMVECTOR>& points, float cell_size)
        : cell_size_(cell_size), inv_cell_size_(1.0f / cell_size) {
        build(points);
    }

    // Find the index of the closest point to `query` (brute‑force within the cell and neighbouring cells)
    size_t find_closest(DirectX::FXMVECTOR query) const noexcept {
        int cx = static_cast<int>(std::floor(vector_math::get_x(query) * inv_cell_size_));
        int cy = static_cast<int>(std::floor(vector_math::get_y(query) * inv_cell_size_));
        int cz = static_cast<int>(std::floor(vector_math::get_z(query) * inv_cell_size_));

        float best_sq = std::numeric_limits<float>::max();
        size_t best_idx = 0;

        // Search in the 3x3x3 block of cells
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = cx + dx;
                    int ny = cy + dy;
                    int nz = cz + dz;
                    uint64_t key = encode_key(nx, ny, nz);
                    auto it = grid_.find(key);
                    if (it == grid_.end()) continue;
                    for (size_t idx : it->second) {
                        float dist_sq = vector_math::length_sq3_scalar(
                            DirectX::XMVectorSubtract(points_[idx], query));
                        if (dist_sq < best_sq) {
                            best_sq = dist_sq;
                            best_idx = idx;
                        }
                    }
                }
            }
        }
        return best_idx;
    }

    // Get the point at a given index
    DirectX::XMVECTOR point(size_t idx) const noexcept { return points_[idx]; }

private:
    float cell_size_;
    float inv_cell_size_;
    std::vector<DirectX::XMVECTOR> points_;
    std::unordered_map<uint64_t, std::vector<size_t>> grid_;

    void build(const std::vector<DirectX::XMVECTOR>& points) {
        points_ = points;
        for (size_t i = 0; i < points_.size(); ++i) {
            int cx = static_cast<int>(std::floor(vector_math::get_x(points_[i]) * inv_cell_size_));
            int cy = static_cast<int>(std::floor(vector_math::get_y(points_[i]) * inv_cell_size_));
            int cz = static_cast<int>(std::floor(vector_math::get_z(points_[i]) * inv_cell_size_));
            uint64_t key = encode_key(cx, cy, cz);
            grid_[key].push_back(i);
        }
    }

    static uint64_t encode_key(int x, int y, int z) noexcept {
        uint64_t ux = static_cast<uint64_t>(x) & 0x1FFFFF;
        uint64_t uy = static_cast<uint64_t>(y) & 0x1FFFFF;
        uint64_t uz = static_cast<uint64_t>(z) & 0x1FFFFF;
        return ux | (uy << 21) | (uz << 42);
    }
};

// -----------------------------------------------------------------------------
// 3. Point‑to‑point ICP between two arbitrary point clouds (automatic correspondence via spatial hash)
//    Returns the rigid transformation aligning source to target.
// -----------------------------------------------------------------------------
inline ICPResult icp_point_to_point(
    const std::vector<DirectX::XMVECTOR>& source,
    const std::vector<DirectX::XMVECTOR>& target,
    int max_iterations = 50,
    float tolerance = 1e-6f) noexcept {

    ICPResult res;
    res.rotation = DirectX::XMMatrixIdentity();
    res.translation = DirectX::XMVectorZero();
    res.converged = false;

    if (source.empty() || target.empty()) return res;

    size_t n = source.size();
    // Build spatial hash on target
    float avg_spacing = 0.0f;
    for (size_t i = 0; i < target.size(); ++i) {
        if (i > 0) {
            avg_spacing += std::sqrt(vector_math::length_sq3_scalar(
                DirectX::XMVectorSubtract(target[i], target[0]))) / target.size();
        }
    }
    float cell_size = std::max(0.1f, avg_spacing * 2.0f);
    PointCloudHashGrid target_grid(target, cell_size);

    // Current transformed source (starts as original)
    std::vector<Eigen::Vector3f> src_cur(n);
    for (size_t i = 0; i < n; ++i) {
        src_cur[i] = Eigen::Vector3f(vector_math::get_x(source[i]),
                                     vector_math::get_y(source[i]),
                                     vector_math::get_z(source[i]));
    }

    // Accumulated transformation
    Eigen::Matrix3f R_acc = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t_acc = Eigen::Vector3f::Zero();

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Find correspondences (closest points in target for each transformed source)
        std::vector<Eigen::Vector3f> tgt_match(n);
        for (size_t i = 0; i < n; ++i) {
            DirectX::XMVECTOR p = DirectX::XMVectorSet(src_cur[i].x(), src_cur[i].y(), src_cur[i].z(), 0.0f);
            size_t idx = target_grid.find_closest(p);
            DirectX::XMVECTOR closest = target_grid.point(idx);
            tgt_match[i] = Eigen::Vector3f(vector_math::get_x(closest),
                                          vector_math::get_y(closest),
                                          vector_math::get_z(closest));
        }

        // Compute optimal rigid transformation from current source to matched target
        rigid_fit::RigidTransformResult fit = rigid_fit::find_rigid_transform(src_cur, tgt_match);
        if (!fit.valid) break;

        Eigen::Matrix3f R_inc = matrix_math::to_eigen_matrix(fit.rotation).block<3,3>(0,0);
        Eigen::Vector3f t_inc(vector_math::get_x(fit.translation),
                              vector_math::get_y(fit.translation),
                              vector_math::get_z(fit.translation));

        // Update accumulated transform
        R_acc = R_inc * R_acc;
        t_acc = R_inc * t_acc + t_inc;

        // Update current source positions
        for (size_t i = 0; i < n; ++i) {
            src_cur[i] = R_inc * src_cur[i] + t_inc;
        }

        // Convergence check
        float angle = std::acos(std::min((R_inc.trace()-1.0f)*0.5f, 1.0f));
        if (angle < tolerance && t_inc.norm() < tolerance) {
            res.converged = true;
            break;
        }
    }

    // Build final transformation
    auto eigen3x3_to_xmmatrix = [](const Eigen::Matrix3f& m) -> DirectX::XMMATRIX {
        DirectX::XMMATRIX out;
        out.r[0] = DirectX::XMVectorSet(m(0,0), m(1,0), m(2,0), 0.0f);
        out.r[1] = DirectX::XMVectorSet(m(0,1), m(1,1), m(2,1), 0.0f);
        out.r[2] = DirectX::XMVectorSet(m(0,2), m(1,2), m(2,2), 0.0f);
        out.r[3] = DirectX::XMVectorSet(0,0,0,1);
        return out;
    };
    res.rotation = eigen3x3_to_xmmatrix(R_acc);
    res.translation = DirectX::XMVectorSet(t_acc.x(), t_acc.y(), t_acc.z(), 0.0f);

    // Final RMS error
    float rms = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        DirectX::XMVECTOR p = DirectX::XMVectorSet(src_cur[i].x(), src_cur[i].y(), src_cur[i].z(), 0.0f);
        size_t idx = target_grid.find_closest(p);
        DirectX::XMVECTOR closest = target_grid.point(idx);
        Eigen::Vector3f diff = src_cur[i] - Eigen::Vector3f(vector_math::get_x(closest),
                                                            vector_math::get_y(closest),
                                                            vector_math::get_z(closest));
        rms += diff.squaredNorm();
    }
    res.rms_error = std::sqrt(rms / n);
    return res;
}

// -----------------------------------------------------------------------------
// 4. Point‑to‑plane ICP using linearised rotation (Gauss‑Newton)
//    Each correspondence provides source point, target point, and target unit normal.
//    Minimizes ∑ ( (R·p_src + t - p_tgt)·n_tgt )².
//    Jacobian derived analytically: ∂(R·p)/∂ω = -R·skew(p)  (R ≈ I+skew(ω))
//    Residual: r = n·(R·p + t - p_tgt).  Update: ΔR = exp(skew(ω)), Δt.
//    Builds 6×6 normal equations and solves with Cholesky.
// -----------------------------------------------------------------------------
struct ICPPlaneCorrespondence {
    DirectX::XMVECTOR source;
    DirectX::XMVECTOR target;
    DirectX::XMVECTOR target_normal;   // unit length
};

inline ICPResult icp_point_to_plane(
    const std::vector<ICPPlaneCorrespondence>& correspondences,
    int max_iterations = 50,
    float tolerance = 1e-6f) noexcept {

    ICPResult res;
    res.rotation = DirectX::XMMatrixIdentity();
    res.translation = DirectX::XMVectorZero();
    res.converged = false;

    size_t n = correspondences.size();
    if (n == 0) return res;

    // Initial transformation (identity)
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t = Eigen::Vector3f::Zero();

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Build Jacobian and residual for linearised system
        Eigen::Matrix<float, 6, 6> JTJ = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> JTr = Eigen::Matrix<float, 6, 1>::Zero();

        for (size_t i = 0; i < n; ++i) {
            DirectX::XMVECTOR src = correspondences[i].source;
            DirectX::XMVECTOR tgt = correspondences[i].target;
            DirectX::XMVECTOR nrm = correspondences[i].target_normal;

            Eigen::Vector3f ps(vector_math::get_x(src), vector_math::get_y(src), vector_math::get_z(src));
            Eigen::Vector3f pt(vector_math::get_x(tgt), vector_math::get_y(tgt), vector_math::get_z(tgt));
            Eigen::Vector3f nn(vector_math::get_x(nrm), vector_math::get_y(nrm), vector_math::get_z(nrm));

            Eigen::Vector3f ps_trans = R * ps + t;
            float residual = nn.dot(ps_trans - pt);

            // Jacobian of (R·p + t) w.r.t ω at ω=0 (ΔR ≈ I + skew(ω))
            // ∂(R·p)/∂ω = -skew(p)  →  change in transformed point = skew(ω)·ps_trans
            // Residual change: Δr = n·(skew(ω)·ps_trans + Δt) = ω·(ps_trans × n) + n·Δt
            Eigen::Vector3f j_rot = ps_trans.cross(nn);  // ps_trans × n
            Eigen::Vector3f j_trans = nn;

            // Accumulate J^T J and J^T r
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    JTJ(row, col) += j_rot(row) * j_rot(col);
                }
                for (int col = 0; col < 3; ++col) {
                    JTJ(row, 3+col) += j_rot(row) * j_trans(col);
                    JTJ(3+col, row) += j_trans(col) * j_rot(row);
                }
                for (int col = 0; col < 3; ++col) {
                    JTJ(3+row, 3+col) += j_trans(row) * j_trans(col);
                }
                JTr(row)   += j_rot(row) * residual;
                JTr(3+row) += j_trans(row) * residual;
            }
        }

        // Solve for update [ω, Δt]
        Eigen::LDLT<Eigen::Matrix<float, 6, 6>> solver(JTJ);
        if (solver.info() != Eigen::Success) break;
        Eigen::Matrix<float, 6, 1> dx = solver.solve(-JTr);

        Eigen::Vector3f omega(dx(0), dx(1), dx(2));
        Eigen::Vector3f delta_t(dx(3), dx(4), dx(5));

        // Update R = ΔR * R, ΔR = exp(skew(omega))
        float angle = omega.norm();
        Eigen::Matrix3f delta_R = Eigen::Matrix3f::Identity();
        if (angle > 1e-12f) {
            Eigen::Vector3f axis = omega / angle;
            delta_R = Eigen::AngleAxisf(angle, axis).toRotationMatrix();
        }
        R = delta_R * R;
        t = delta_R * t + delta_t;

        // Convergence
        if (angle < tolerance && delta_t.norm() < tolerance) {
            res.converged = true;
            break;
        }
    }

    // Compute final RMS error
    float rms = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        Eigen::Vector3f ps(vector_math::get_x(correspondences[i].source),
                           vector_math::get_y(correspondences[i].source),
                           vector_math::get_z(correspondences[i].source));
        Eigen::Vector3f pt(vector_math::get_x(correspondences[i].target),
                           vector_math::get_y(correspondences[i].target),
                           vector_math::get_z(correspondences[i].target));
        Eigen::Vector3f nn(vector_math::get_x(correspondences[i].target_normal),
                           vector_math::get_y(correspondences[i].target_normal),
                           vector_math::get_z(correspondences[i].target_normal));
        Eigen::Vector3f ps_trans = R * ps + t;
        float r = nn.dot(ps_trans - pt);
        rms += r * r;
    }
    res.rms_error = std::sqrt(rms / n);

    auto eigen3x3_to_xmmatrix = [](const Eigen::Matrix3f& m) -> DirectX::XMMATRIX {
        DirectX::XMMATRIX out;
        out.r[0] = DirectX::XMVectorSet(m(0,0), m(1,0), m(2,0), 0.0f);
        out.r[1] = DirectX::XMVectorSet(m(0,1), m(1,1), m(2,1), 0.0f);
        out.r[2] = DirectX::XMVectorSet(m(0,2), m(1,2), m(2,2), 0.0f);
        out.r[3] = DirectX::XMVectorSet(0,0,0,1);
        return out;
    };
    res.rotation = eigen3x3_to_xmmatrix(R);
    res.translation = DirectX::XMVectorSet(t.x(), t.y(), t.z(), 0.0f);
    return res;
}

// -----------------------------------------------------------------------------
// 5. ICP between two meshes using closest‑point correspondences (point‑to‑point)
//    Uses MeshDistanceQuery for efficient closest point search on the target mesh.
//    Source vertices are transformed iteratively.
// -----------------------------------------------------------------------------
inline ICPResult icp_mesh_to_mesh_point_to_point(
    const HalfEdgeMesh& source_mesh,
    const HalfEdgeMesh& target_mesh,
    int max_iterations = 50,
    float tolerance = 1e-6f,
    size_t max_sample_points = 5000) noexcept {

    ICPResult res;
    res.rotation = DirectX::XMMatrixIdentity();
    res.translation = DirectX::XMVectorZero();
    res.converged = false;

    const auto& src_verts = source_mesh.vertices();
    size_t nv = src_verts.size();
    if (nv == 0) return res;

    std::vector<DirectX::XMVECTOR> src_samples;
    if (nv > max_sample_points) {
        size_t step = nv / max_sample_points;
        for (size_t i = 0; i < nv; i += step)
            src_samples.push_back(src_verts[i].position);
    } else {
        src_samples.resize(nv);
        for (size_t i = 0; i < nv; ++i) src_samples[i] = src_verts[i].position;
    }

    MeshDistanceQuery dist_query(target_mesh, 1.0f);

    Eigen::Matrix3f R_acc = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t_acc = Eigen::Vector3f::Zero();

    for (int iter = 0; iter < max_iterations; ++iter) {
        std::vector<Eigen::Vector3f> src_cur_eigen(src_samples.size());
        std::vector<Eigen::Vector3f> tgt_eigen(src_samples.size());

        for (size_t i = 0; i < src_samples.size(); ++i) {
            Eigen::Vector3f ps(vector_math::get_x(src_samples[i]),
                               vector_math::get_y(src_samples[i]),
                               vector_math::get_z(src_samples[i]));
            Eigen::Vector3f ps_trans = R_acc * ps + t_acc;
            DirectX::XMVECTOR ps_trans_dx = DirectX::XMVectorSet(ps_trans.x(), ps_trans.y(), ps_trans.z(), 0.0f);
            DirectX::XMVECTOR closest;
            dist_query.closest_point(ps_trans_dx, target_mesh, closest);
            tgt_eigen[i] = Eigen::Vector3f(vector_math::get_x(closest),
                                          vector_math::get_y(closest),
                                          vector_math::get_z(closest));
            src_cur_eigen[i] = ps_trans;
        }

        rigid_fit::RigidTransformResult fit = rigid_fit::find_rigid_transform(src_cur_eigen, tgt_eigen);
        if (!fit.valid) break;
        Eigen::Matrix3f R_inc = matrix_math::to_eigen_matrix(fit.rotation).block<3,3>(0,0);
        Eigen::Vector3f t_inc(vector_math::get_x(fit.translation),
                              vector_math::get_y(fit.translation),
                              vector_math::get_z(fit.translation));

        R_acc = R_inc * R_acc;
        t_acc = R_inc * t_acc + t_inc;

        float angle = std::acos(std::min((R_inc.trace()-1.0f)*0.5f, 1.0f));
        if (angle < tolerance && t_inc.norm() < tolerance) {
            res.converged = true;
            break;
        }
    }

    auto eigen3x3_to_xmmatrix = [](const Eigen::Matrix3f& m) -> DirectX::XMMATRIX {
        DirectX::XMMATRIX out;
        out.r[0] = DirectX::XMVectorSet(m(0,0), m(1,0), m(2,0), 0.0f);
        out.r[1] = DirectX::XMVectorSet(m(0,1), m(1,1), m(2,1), 0.0f);
        out.r[2] = DirectX::XMVectorSet(m(0,2), m(1,2), m(2,2), 0.0f);
        out.r[3] = DirectX::XMVectorSet(0,0,0,1);
        return out;
    };
    res.rotation = eigen3x3_to_xmmatrix(R_acc);
    res.translation = DirectX::XMVectorSet(t_acc.x(), t_acc.y(), t_acc.z(), 0.0f);

    float rms = 0.0f;
    for (size_t i = 0; i < src_samples.size(); ++i) {
        DirectX::XMVECTOR closest;
        Eigen::Vector3f ps(vector_math::get_x(src_samples[i]),
                           vector_math::get_y(src_samples[i]),
                           vector_math::get_z(src_samples[i]));
        Eigen::Vector3f ps_trans = R_acc * ps + t_acc;
        DirectX::XMVECTOR ps_trans_dx = DirectX::XMVectorSet(ps_trans.x(), ps_trans.y(), ps_trans.z(), 0.0f);
        dist_query.closest_point(ps_trans_dx, target_mesh, closest);
        Eigen::Vector3f diff = ps_trans - Eigen::Vector3f(vector_math::get_x(closest),
                                                          vector_math::get_y(closest),
                                                          vector_math::get_z(closest));
        rms += diff.squaredNorm();
    }
    res.rms_error = std::sqrt(rms / src_samples.size());
    return res;
}

} // namespace mesh_registration
} // namespace SimulationMath

#endif // CORE_MATH_MESH_REGISTRATION_H