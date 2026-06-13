//File 0070 : core/math/rigid_transform_fit.h
//Optimal rigid transformation (Kabsch‑Umeyama) aligning two 3D point sets: rotation, translation, uniform scaling, root‑mean‑square error, and Sim3 alignment.
#ifndef CORE_MATH_RIGID_TRANSFORM_FIT_H
#define CORE_MATH_RIGID_TRANSFORM_FIT_H

#include "vector_math.h"
#include "matrix_math.h"
#include "linear_algebra.h"    // Eigen and solve
#include <vector>
#include <cmath>
#include <cstdint>

namespace SimulationMath {
namespace rigid_fit {

using EigenMatrix3f = Eigen::Matrix3f;
using EigenVector3f = Eigen::Vector3f;

// -----------------------------------------------------------------------------
// 1. Compute centroid of a point set
// -----------------------------------------------------------------------------
inline EigenVector3f centroid(const std::vector<EigenVector3f>& points) noexcept {
    EigenVector3f sum = EigenVector3f::Zero();
    for (const auto& p : points) sum += p;
    if (points.empty()) return sum;
    return sum / static_cast<float>(points.size());
}

// Same but using DirectX vectors
inline DirectX::XMVECTOR centroid_dx(const std::vector<DirectX::XMVECTOR>& points) noexcept {
    DirectX::XMVECTOR sum = DirectX::XMVectorZero();
    for (const auto& p : points) sum = DirectX::XMVectorAdd(sum, p);
    if (points.empty()) return sum;
    return DirectX::XMVectorScale(sum, 1.0f / points.size());
}

// -----------------------------------------------------------------------------
// 2. Compute optimal rotation between two sets (no scaling, no reflection)
// -----------------------------------------------------------------------------
struct RigidTransformResult {
    DirectX::XMMATRIX rotation;      // 3x3 rotation matrix (orthonormal)
    DirectX::XMVECTOR translation;   // post‑rotation translation
    float rms_error;                 // root‑mean‑square error after alignment
    bool valid;                      // false if computation failed
};

inline RigidTransformResult find_rigid_transform(
    const std::vector<EigenVector3f>& source,
    const std::vector<EigenVector3f>& target) noexcept {

    RigidTransformResult res;
    res.valid = false;
    res.rotation = DirectX::XMMatrixIdentity();
    res.translation = DirectX::XMVectorZero();
    res.rms_error = 0.0f;

    if (source.empty() || source.size() != target.size()) return res;

    size_t n = source.size();
    EigenVector3f src_centroid = centroid(source);
    EigenVector3f dst_centroid = centroid(target);

    // Center the points
    std::vector<EigenVector3f> src_centered(n), dst_centered(n);
    for (size_t i = 0; i < n; ++i) {
        src_centered[i] = source[i] - src_centroid;
        dst_centered[i] = target[i] - dst_centroid;
    }

    // Build covariance matrix H = sum(src_i * dst_i^T)
    EigenMatrix3f H = EigenMatrix3f::Zero();
    for (size_t i = 0; i < n; ++i) {
        H += src_centered[i] * dst_centered[i].transpose();
    }

    // SVD of H
    Eigen::JacobiSVD<EigenMatrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    EigenMatrix3f U = svd.matrixU();
    EigenMatrix3f V = svd.matrixV();

    // Rotation R = V * U^T, but ensure proper rotation (det = +1)
    EigenMatrix3f R = V * U.transpose();
    if (R.determinant() < 0.0f) {
        // Flip sign of the last column of V (or row of U)
        V.col(2) *= -1.0f;
        R = V * U.transpose();
    }

    // Convert to XMMATRIX
    DirectX::XMMATRIX rotMat = matrix_math::to_directx_matrix(R); // we need a helper; use our matrix_math conversion
    // We'll instead build manually from Eigen
    DirectX::XMMATRIX dxRot;
    dxRot.r[0] = DirectX::XMVectorSet(R(0,0), R(1,0), R(2,0), 0.0f);
    dxRot.r[1] = DirectX::XMVectorSet(R(0,1), R(1,1), R(2,1), 0.0f);
    dxRot.r[2] = DirectX::XMVectorSet(R(0,2), R(1,2), R(2,2), 0.0f);
    dxRot.r[3] = DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);
    res.rotation = dxRot;

    // Translation t = dst_centroid - R * src_centroid
    EigenVector3f t = dst_centroid - R * src_centroid;
    res.translation = DirectX::XMVectorSet(t.x(), t.y(), t.z(), 0.0f);

    // Compute RMS error
    float error_sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        EigenVector3f aligned = R * source[i] + t;
        float dist_sq = (aligned - target[i]).squaredNorm();
        error_sum += dist_sq;
    }
    res.rms_error = std::sqrt(error_sum / n);
    res.valid = true;
    return res;
}

// -----------------------------------------------------------------------------
// 3. Rigid transform with uniform scaling (Umeyama algorithm)
// -----------------------------------------------------------------------------
struct SimTransformResult {
    DirectX::XMMATRIX rotation;
    DirectX::XMVECTOR translation;
    float scale;
    float rms_error;
    bool valid;
};

inline SimTransformResult find_sim_transform(
    const std::vector<EigenVector3f>& source,
    const std::vector<EigenVector3f>& target) noexcept {

    SimTransformResult res;
    res.valid = false;
    res.rotation = DirectX::XMMatrixIdentity();
    res.translation = DirectX::XMVectorZero();
    res.scale = 1.0f;
    res.rms_error = 0.0f;

    if (source.empty() || source.size() != target.size()) return res;

    size_t n = source.size();
    EigenVector3f src_centroid = centroid(source);
    EigenVector3f dst_centroid = centroid(target);

    // Center points
    std::vector<EigenVector3f> src_centered(n), dst_centered(n);
    float src_var = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        src_centered[i] = source[i] - src_centroid;
        dst_centered[i] = target[i] - dst_centroid;
        src_var += src_centered[i].squaredNorm();
    }
    src_var /= n;

    // Covariance H
    EigenMatrix3f H = EigenMatrix3f::Zero();
    for (size_t i = 0; i < n; ++i)
        H += src_centered[i] * dst_centered[i].transpose();
    H /= n;

    // SVD of H
    Eigen::JacobiSVD<EigenMatrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    EigenMatrix3f U = svd.matrixU();
    EigenMatrix3f V = svd.matrixV();
    EigenVector3f S = svd.singularValues();

    // Rotation
    EigenMatrix3f R = V * U.transpose();
    if (R.determinant() < 0.0f) {
        V.col(2) *= -1.0f;
        R = V * U.transpose();
    }

    // Uniform scale: 1/σ_x² * sum(σ_i * D_i), where D = diag(1,1,det(U*V^T))? Umeyama formula:
    // s = (1/σ_x²) * trace(S * D) with D = diag(1,1,det(V*U^T))
    float detVUT = (V * U.transpose()).determinant(); // should be 1 after correction
    if (detVUT < 0.0f) detVUT = -1.0f; // just in case
    // Actually the correct formula from Umeyama: s = (1/σ_x²) * sum(σ_i * D_i)
    // Where D_1=D_2=1, D_3 = det(V*U^T). We have already ensured determinant = 1 by flipping V, so D_3 = 1.
    // So s = (σ1 + σ2 + σ3) / src_var
    float trace_S = S(0) + S(1) + S(2);
    float scale = (src_var > 1e-12f) ? trace_S / src_var : 1.0f;
    res.scale = scale;

    // Translation t = dst_centroid - s * R * src_centroid
    EigenVector3f t = dst_centroid - scale * R * src_centroid;

    // Rotation matrix to XMMATRIX
    DirectX::XMMATRIX dxRot;
    dxRot.r[0] = DirectX::XMVectorSet(R(0,0), R(1,0), R(2,0), 0.0f);
    dxRot.r[1] = DirectX::XMVectorSet(R(0,1), R(1,1), R(2,1), 0.0f);
    dxRot.r[2] = DirectX::XMVectorSet(R(0,2), R(1,2), R(2,2), 0.0f);
    dxRot.r[3] = DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);
    res.rotation = dxRot;
    res.translation = DirectX::XMVectorSet(t.x(), t.y(), t.z(), 0.0f);

    // RMS error
    float err_sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        EigenVector3f aligned = scale * R * source[i] + t;
        err_sum += (aligned - target[i]).squaredNorm();
    }
    res.rms_error = std::sqrt(err_sum / n);
    res.valid = true;
    return res;
}

} // namespace rigid_fit
} // namespace SimulationMath

#endif // CORE_MATH_RIGID_TRANSFORM_FIT_H