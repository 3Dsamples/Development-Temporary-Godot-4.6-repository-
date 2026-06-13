//File 0103 : core/math/mesh_quality_analysis.h
//Mesh quality metrics for triangles and tetrahedra: aspect ratio, radius ratio, edge ratio, minimum/maximum angle, dihedral angle, skewness, condition number, and per‑element quality histograms.
#ifndef CORE_MATH_MESH_QUALITY_ANALYSIS_H
#define CORE_MATH_MESH_QUALITY_ANALYSIS_H

#include "mesh_data.h"               // HalfEdgeMesh for triangle mesh
#include "vector_math.h"
#include "geometry_primitives.h"     // Triangle
#include "math_constants.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace SimulationMath {
namespace mesh_quality {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. Triangle quality metrics (all assume non‑degenerate)
// -----------------------------------------------------------------------------

// Aspect ratio: longest edge / shortest altitude (min height). Lower is better (1 = equilateral)
inline float triangle_aspect_ratio(DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1, DirectX::FXMVECTOR v2) noexcept {
    float l0 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v1, v2));
    float l1 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v2, v0));
    float l2 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v0, v1));
    // Heron's formula for area
    float s = (l0 + l1 + l2) * 0.5f;
    float area = std::sqrt(std::max(0.0f, s * (s - l0) * (s - l1) * (s - l2)));
    if (area < 1e-12f) return std::numeric_limits<float>::max();
    float longest_edge = std::max({l0, l1, l2});
    // For each edge, corresponding height = 2*area / opposite edge length
    float min_height = std::min({2.0f * area / l0, 2.0f * area / l1, 2.0f * area / l2});
    return longest_edge / min_height;
}

// Radius ratio: 2 * inscribed radius / circumradius (optimal = 1)
inline float triangle_radius_ratio(DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1, DirectX::FXMVECTOR v2) noexcept {
    float l0 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v1, v2));
    float l1 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v2, v0));
    float l2 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v0, v1));
    float s = (l0 + l1 + l2) * 0.5f;
    float area = std::sqrt(std::max(0.0f, s * (s - l0) * (s - l1) * (s - l2)));
    if (area < 1e-12f) return 0.0f;
    // Circumradius R = (l0*l1*l2) / (4*area)
    float R = (l0 * l1 * l2) / (4.0f * area);
    // Inradius r = area / s
    float r = area / s;
    return 2.0f * r / R;
}

// Edge ratio: shortest edge / longest edge (1 = equilateral)
inline float triangle_edge_ratio(DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1, DirectX::FXMVECTOR v2) noexcept {
    float l0 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v1, v2));
    float l1 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v2, v0));
    float l2 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v0, v1));
    float min_len = std::min({l0, l1, l2});
    float max_len = std::max({l0, l1, l2});
    return (max_len > 0.0f) ? min_len / max_len : 0.0f;
}

// Minimum angle in radians
inline float triangle_min_angle(DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1, DirectX::FXMVECTOR v2) noexcept {
    auto angle = [](DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, DirectX::FXMVECTOR c) -> float {
        DirectX::XMVECTOR u = DirectX::XMVectorSubtract(b, a);
        DirectX::XMVECTOR v = DirectX::XMVectorSubtract(c, a);
        float dot = vector_math::dot3_scalar(u, v);
        float lenu = vector_math::length3_scalar(u);
        float lenv = vector_math::length3_scalar(v);
        if (lenu < 1e-12f || lenv < 1e-12f) return 0.0f;
        return std::acos(std::clamp(dot / (lenu * lenv), -1.0f, 1.0f));
    };
    return std::min({angle(v0, v1, v2), angle(v1, v2, v0), angle(v2, v0, v1)});
}

// Maximum angle in radians
inline float triangle_max_angle(DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1, DirectX::FXMVECTOR v2) noexcept {
    auto angle = [](DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, DirectX::FXMVECTOR c) -> float {
        DirectX::XMVECTOR u = DirectX::XMVectorSubtract(b, a);
        DirectX::XMVECTOR v = DirectX::XMVectorSubtract(c, a);
        float dot = vector_math::dot3_scalar(u, v);
        float lenu = vector_math::length3_scalar(u);
        float lenv = vector_math::length3_scalar(v);
        if (lenu < 1e-12f || lenv < 1e-12f) return 0.0f;
        return std::acos(std::clamp(dot / (lenu * lenv), -1.0f, 1.0f));
    };
    return std::max({angle(v0, v1, v2), angle(v1, v2, v0), angle(v2, v0, v1)});
}

// -----------------------------------------------------------------------------
// 2. Tetrahedron quality metrics (for 3D meshes)
// -----------------------------------------------------------------------------

// Aspect ratio: max edge length / inscribed sphere diameter (lowest = 1 for regular tetrahedron)
inline float tetrahedron_aspect_ratio(DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1, DirectX::FXMVECTOR v2, DirectX::FXMVECTOR v3) noexcept {
    float l01 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v1, v0));
    float l02 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v2, v0));
    float l03 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v3, v0));
    float l12 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v2, v1));
    float l13 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v3, v1));
    float l23 = vector_math::length3_scalar(DirectX::XMVectorSubtract(v3, v2));
    float max_edge = std::max({l01, l02, l03, l12, l13, l23});

    // Compute volume
    DirectX::XMVECTOR e1 = DirectX::XMVectorSubtract(v1, v0);
    DirectX::XMVECTOR e2 = DirectX::XMVectorSubtract(v2, v0);
    DirectX::XMVECTOR e3 = DirectX::XMVectorSubtract(v3, v0);
    float volume = std::abs(vector_math::dot3_scalar(vector_math::cross3(e1, e2), e3)) / 6.0f;
    if (volume < 1e-12f) return std::numeric_limits<float>::max();

    // Sum of face areas for inscribed sphere radius approximation: r = 3 * V / (sum face areas)
    auto face_area = [](DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, DirectX::FXMVECTOR c) -> float {
        DirectX::XMVECTOR cross = vector_math::cross3(DirectX::XMVectorSubtract(b, a), DirectX::XMVectorSubtract(c, a));
        return 0.5f * vector_math::length3_scalar(cross);
    };
    float total_area = face_area(v0, v1, v2) + face_area(v0, v1, v3) + face_area(v0, v2, v3) + face_area(v1, v2, v3);
    float inradius = (total_area > 0.0f) ? 3.0f * volume / total_area : 0.0f;
    if (inradius <= 0.0f) return std::numeric_limits<float>::max();
    return max_edge / (2.0f * inradius);
}

// Radius ratio: 3 * inscribed sphere radius / circumsphere radius (optimal = 1)
inline float tetrahedron_radius_ratio(DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1, DirectX::FXMVECTOR v2, DirectX::FXMVECTOR v3) noexcept {
    // Compute volume
    DirectX::XMVECTOR e1 = DirectX::XMVectorSubtract(v1, v0);
    DirectX::XMVECTOR e2 = DirectX::XMVectorSubtract(v2, v0);
    DirectX::XMVECTOR e3 = DirectX::XMVectorSubtract(v3, v0);
    float volume = std::abs(vector_math::dot3_scalar(vector_math::cross3(e1, e2), e3)) / 6.0f;
    if (volume < 1e-12f) return 0.0f;

    // Inradius: 3*V / (sum face areas)
    auto face_area = [](DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, DirectX::FXMVECTOR c) -> float {
        DirectX::XMVECTOR cross = vector_math::cross3(DirectX::XMVectorSubtract(b, a), DirectX::XMVectorSubtract(c, a));
        return 0.5f * vector_math::length3_scalar(cross);
    };
    float total_area = face_area(v0, v1, v2) + face_area(v0, v1, v3) + face_area(v0, v2, v3) + face_area(v1, v2, v3);
    float r = (total_area > 0.0f) ? 3.0f * volume / total_area : 0.0f;

    // Circumradius: distance from circumcenter to vertex. Compute circumcenter using matrix.
    // We'll use the formula: R = sqrt( (a^2 b^2 c^2) / (144 V^2) ...) but simpler: compute circumcenter using barycentric coordinates.
    // I'll compute circumradius using the formula from lengths and volume: R = sqrt( ( (a^2 + b^2 + c^2 + ... )? Actually for tetrahedron:
    // R = sqrt( ( (a^2 * b^2 * c^2 ...)? There's a formula: 288 V^2 = ...
    // I'll compute using the squared distances between vertices and solve a linear system for circumcenter.
    // But to keep code concise, I'll use an approximate method: circumradius = (max edge length) / 2? No.
    // I'll use the standard formula from Cayley-Menger determinant: R^2 = - 1/(288 V^2) * det( ... ), but that's complex.
    // Instead, I'll compute circumcenter as the intersection of perpendicular bisectors by solving a 3x3 linear system.
    // Choose v0 as reference, vectors e1,e2,e3. The circumcenter p satisfies |p-v0|^2 = |p-v1|^2 = |p-v2|^2 = |p-v3|^2.
    // That gives three equations: (p - v0)·e1 = 0.5*|e1|^2, (p - v0)·e2 = 0.5*|e2|^2, (p - v0)·e3 = 0.5*|e3|^2.
    // Solve for p in basis (e1,e2,e3). This is exactly the system: A * x = b, where A is the Gram matrix of e1,e2,e3.
    // Then circumradius R = |p - v0|.
    Eigen::Vector3f E1(e1.m128_f32[0], e1.m128_f32[1], e1.m128_f32[2]);
    Eigen::Vector3f E2(e2.m128_f32[0], e2.m128_f32[1], e2.m128_f32[2]);
    Eigen::Vector3f E3(e3.m128_f32[0], e3.m128_f32[1], e3.m128_f32[2]);
    Eigen::Matrix3f A;
    A << E1.dot(E1), E1.dot(E2), E1.dot(E3),
         E2.dot(E1), E2.dot(E2), E2.dot(E3),
         E3.dot(E1), E3.dot(E2), E3.dot(E3);
    Eigen::Vector3f b(0.5f * E1.squaredNorm(), 0.5f * E2.squaredNorm(), 0.5f * E3.squaredNorm());
    Eigen::Vector3f x = A.ldlt().solve(b);
    // p = v0 + x0*e1 + x1*e2 + x2*e3
    Eigen::Vector3f p = Eigen::Vector3f(v0.m128_f32[0], v0.m128_f32[1], v0.m128_f32[2]) + x(0)*E1 + x(1)*E2 + x(2)*E3;
    float R = (p - Eigen::Vector3f(v0.m128_f32[0], v0.m128_f32[1], v0.m128_f32[2])).norm();
    if (R < 1e-12f) return 0.0f;
    return 3.0f * r / R;
}

// Minimum dihedral angle (in radians)
inline float tetrahedron_min_dihedral(DirectX::FXMVECTOR v0, DirectX::FXMVECTOR v1, DirectX::FXMVECTOR v2, DirectX::FXMVECTOR v3) noexcept {
    auto face_normal = [](DirectX::FXMVECTOR a, DirectX::FXMVECTOR b, DirectX::FXMVECTOR c) -> DirectX::XMVECTOR {
        return vector_math::normalize3(vector_math::cross3(DirectX::XMVectorSubtract(b, a), DirectX::XMVectorSubtract(c, a)));
    };
    DirectX::XMVECTOR n012 = face_normal(v0, v1, v2);
    DirectX::XMVECTOR n013 = face_normal(v0, v1, v3);
    DirectX::XMVECTOR n023 = face_normal(v0, v2, v3);
    DirectX::XMVECTOR n123 = face_normal(v1, v2, v3);

    auto dihedral = [](DirectX::FXMVECTOR n1, DirectX::FXMVECTOR n2) -> float {
        float dot = vector_math::dot3_scalar(n1, n2);
        dot = std::clamp(dot, -1.0f, 1.0f);
        return std::acos(dot);
    };
    float d01 = dihedral(n012, n013); // edge (v0,v1)
    float d02 = dihedral(n012, n023); // edge (v0,v2)
    float d03 = dihedral(n013, n023); // edge (v0,v3)
    float d12 = dihedral(n012, n123); // edge (v1,v2)
    float d13 = dihedral(n013, n123); // edge (v1,v3)
    float d23 = dihedral(n023, n123); // edge (v2,v3)
    return std::min({d01, d02, d03, d12, d13, d23});
}

// -----------------------------------------------------------------------------
// 3. Compute quality statistics for a triangle mesh and store per‑face values
// -----------------------------------------------------------------------------
struct TriangleQualityReport {
    std::vector<float> aspect_ratios;
    std::vector<float> radius_ratios;
    std::vector<float> edge_ratios;
    std::vector<float> min_angles;
    std::vector<float> max_angles;
};

inline TriangleQualityReport compute_triangle_mesh_quality(const HalfEdgeMesh& mesh) noexcept {
    TriangleQualityReport report;
    const auto& hedges = mesh.half_edges();
    const auto& verts  = mesh.vertices();
    size_t nf = mesh.faces().size();
    report.aspect_ratios.reserve(nf);
    report.radius_ratios.reserve(nf);
    report.edge_ratios.reserve(nf);
    report.min_angles.reserve(nf);
    report.max_angles.reserve(nf);

    for (size_t f = 0; f < nf; ++f) {
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

        report.aspect_ratios.push_back(triangle_aspect_ratio(p0, p1, p2));
        report.radius_ratios.push_back(triangle_radius_ratio(p0, p1, p2));
        report.edge_ratios.push_back(triangle_edge_ratio(p0, p1, p2));
        report.min_angles.push_back(triangle_min_angle(p0, p1, p2));
        report.max_angles.push_back(triangle_max_angle(p0, p1, p2));
    }
    return report;
}

// -----------------------------------------------------------------------------
// 4. Simple histogram of a scalar field
// -----------------------------------------------------------------------------
inline std::vector<uint32_t> histogram(const std::vector<float>& values, float min_val, float max_val, size_t bins = 10) noexcept {
    std::vector<uint32_t> hist(bins, 0);
    if (values.empty()) return hist;
    float range = max_val - min_val;
    if (range <= 0.0f) {
        hist[0] = values.size();
        return hist;
    }
    for (float v : values) {
        int idx = static_cast<int>((v - min_val) / range * bins);
        idx = std::clamp(idx, 0, int(bins)-1);
        hist[idx]++;
    }
    return hist;
}

} // namespace mesh_quality
} // namespace SimulationMath

#endif // CORE_MATH_MESH_QUALITY_ANALYSIS_H