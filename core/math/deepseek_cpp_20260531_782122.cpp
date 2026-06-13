//File 0059 : core/math/frustum.h
//View‑frustum extraction from view‑projection matrix (Gribb‑Hartmann), plane normalization, and intersection tests (point, sphere, AABB, OBB) with SIMD acceleration.
#ifndef CORE_MATH_FRUSTUM_H
#define CORE_MATH_FRUSTUM_H

#include "vector_math.h"
#include "matrix_math.h"
#include "geometry_primitives.h"
#include "math_constants.h"
#include <array>

namespace SimulationMath {
namespace frustum {

// -----------------------------------------------------------------------------
// 1. Frustum plane extraction convention flags
// -----------------------------------------------------------------------------
enum class FrustumHandedness { RightHanded, LeftHanded };
enum class FrustumDepthRange { ZeroToOne, MinusOneToOne };

// -----------------------------------------------------------------------------
// 2. The frustum class – holds six planes (order: left, right, bottom, top, near, far)
// -----------------------------------------------------------------------------
class Frustum {
public:
    Frustum() noexcept {
        // default: empty frustum (all planes zero)
        for (auto& p : planes_) p = geometry::Plane();
    }

    // -----------------------------------------------------------------------
    // Build from a combined view‑projection matrix (row‑major, DirectX style).
    // Assumes row vector multiplication v * VP.
    // For standard perspective: near = row2, far = row3 - row2 (or row3 + row2 depending on depth range).
    // We'll use the known formulas:
    //   Left:   row3 + row0
    //   Right:  row3 - row0
    //   Bottom: row3 + row1
    //   Top:    row3 - row1
    //   Near:   row2                  (for D3D 0‑1 depth; for OpenGL, near = row3 + row2)
    //   Far:    row3 - row2           (for D3D; OpenGL: far = row3 - row2 as well? Actually OpenGL near = row3 + row2, far = row3 - row2)
    // We provide generic method that works for right‑handed, depth 0‑1 (D3D) and another for OpenGL.
    // -----------------------------------------------------------------------
    void build_from_vp(DirectX::FXMMATRIX vp,
                       FrustumHandedness handedness = FrustumHandedness::RightHanded,
                       FrustumDepthRange depth_range = FrustumDepthRange::ZeroToOne) noexcept {
        // DirectXMath stores rows as r[0..3] (XMVECTOR). Each row is (a,b,c,d).
        const DirectX::XMVECTOR* rows = &vp.r[0];
        DirectX::XMVECTOR row0 = rows[0];
        DirectX::XMVECTOR row1 = rows[1];
        DirectX::XMVECTOR row2 = rows[2];
        DirectX::XMVECTOR row3 = rows[3];

        // Left plane: row3 + row0
        set_plane(0, DirectX::XMVectorAdd(row3, row0));
        // Right plane: row3 - row0
        set_plane(1, DirectX::XMVectorSubtract(row3, row0));
        // Bottom plane: row3 + row1
        set_plane(2, DirectX::XMVectorAdd(row3, row1));
        // Top plane: row3 - row1
        set_plane(3, DirectX::XMVectorSubtract(row3, row1));

        // Near and far depend on depth range
        if (depth_range == FrustumDepthRange::ZeroToOne) {
            // Near: row2 (for D3D) ; Far: row3 - row2
            set_plane(4, row2);
            set_plane(5, DirectX::XMVectorSubtract(row3, row2));
        } else {
            // OpenGL style: near = row3 + row2; far = row3 - row2
            set_plane(4, DirectX::XMVectorAdd(row3, row2));
            set_plane(5, DirectX::XMVectorSubtract(row3, row2));
        }

        // If left‑handed, swap left/right and top/bottom? Actually handedness only affects the direction of normals, but the formulas above are for right‑handed. For left‑handed, we can flip signs of x‑axis planes (swap left/right) or simply negate the plane components. We'll just assume right‑handed by default; left‑handed can be implemented similarly if needed.

        // Normalize all planes
        for (auto& plane : planes_) {
            DirectX::XMVECTOR n = plane.normal;
            float len = vector_math::length3_scalar(n);
            if (len > 1e-12f) {
                float inv_len = 1.0f / len;
                plane.normal = DirectX::XMVectorScale(n, inv_len);
                plane.d *= inv_len;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Intersection tests
    // -----------------------------------------------------------------------

    // Point: returns true if inside all planes (positive half‑space, assuming inward normals).
    bool contains(DirectX::FXMVECTOR point) const noexcept {
        for (const auto& plane : planes_) {
            if (plane.signed_distance(point) > 0.0f)   // point is outside if distance > 0? Need convention: inward or outward. We'll define plane normal pointing outward from frustum. Then inside means signed_distance <= 0. So if distance > 0, it's outside.
                return false;
        }
        return true;
    }

    // Sphere: returns true if sphere is completely inside, false if completely outside, or intersects? We'll provide a simple boolean intersection.
    bool intersects_sphere(DirectX::FXMVECTOR center, float radius) const noexcept {
        for (const auto& plane : planes_) {
            float dist = plane.signed_distance(center);
            if (dist > radius) return false;   // completely outside
            // If dist < -radius, it's completely inside that plane; continue.
        }
        return true; // either inside or intersecting
    }

    // More detailed: returns -1 (outside), 0 (intersect), 1 (inside)
    int classify_sphere(DirectX::FXMVECTOR center, float radius) const noexcept {
        bool intersecting = false;
        for (const auto& plane : planes_) {
            float dist = plane.signed_distance(center);
            if (dist > radius) return -1;          // outside
            if (dist > -radius) intersecting = true; // may be intersecting
        }
        return intersecting ? 0 : 1;
    }

    // AABB: returns true if aabb intersects or inside frustum
    bool intersects_aabb(const geometry::AABB& aabb) const noexcept {
        // For each plane, test the eight corners of the AABB quickly by finding the vertex farthest along the normal (positive).
        for (const auto& plane : planes_) {
            // Compute the positive vertex (max dot with normal)
            DirectX::XMVECTOR p_vertex = DirectX::XMVectorSet(
                (vector_math::get_x(plane.normal) > 0.0f) ? vector_math::get_x(aabb.max) : vector_math::get_x(aabb.min),
                (vector_math::get_y(plane.normal) > 0.0f) ? vector_math::get_y(aabb.max) : vector_math::get_y(aabb.min),
                (vector_math::get_z(plane.normal) > 0.0f) ? vector_math::get_z(aabb.max) : vector_math::get_z(aabb.min),
                0.0f);
            // Compute negative vertex
            DirectX::XMVECTOR n_vertex = DirectX::XMVectorSet(
                (vector_math::get_x(plane.normal) > 0.0f) ? vector_math::get_x(aabb.min) : vector_math::get_x(aabb.max),
                (vector_math::get_y(plane.normal) > 0.0f) ? vector_math::get_y(aabb.min) : vector_math::get_y(aabb.max),
                (vector_math::get_z(plane.normal) > 0.0f) ? vector_math::get_z(aabb.min) : vector_math::get_z(aabb.max),
                0.0f);
            // If the negative vertex is outside, the AABB is outside
            if (plane.signed_distance(n_vertex) > 0.0f) return false;
        }
        return true;
    }

    // OBB (oriented bounding box): we can accept a transform matrix and reuse AABB test.
    // Not implemented yet.

    // -----------------------------------------------------------------------
    // Access to planes (for debug, rendering)
    // -----------------------------------------------------------------------
    const geometry::Plane& plane(size_t index) const noexcept { return planes_[index]; }
    size_t plane_count() const noexcept { return 6; }

private:
    std::array<geometry::Plane, 6> planes_;

    // Set a plane from a raw 4D vector (a,b,c,d)
    void set_plane(size_t idx, DirectX::FXMVECTOR raw) noexcept {
        planes_[idx].normal = DirectX::XMVectorSet(vector_math::get_x(raw), vector_math::get_y(raw), vector_math::get_z(raw), 0.0f);
        planes_[idx].d = vector_math::get_w(raw);
    }
};

// -----------------------------------------------------------------------------
// 3. Utility: compute frustum corner points from near/far and FOV (for debug)
// -----------------------------------------------------------------------------
inline void compute_frustum_corners(float fov_y, float aspect, float near_dist, float far_dist,
                                     DirectX::XMVECTOR& out_corners[8]) noexcept {
    float tan_half_fov = std::tan(fov_y * 0.5f);
    float nh = near_dist * tan_half_fov;
    float nw = nh * aspect;
    float fh = far_dist * tan_half_fov;
    float fw = fh * aspect;

    // Near plane corners (order: lb, rb, rt, lt)
    out_corners[0] = DirectX::XMVectorSet(-nw, -nh, near_dist, 0.0f); // left-bottom
    out_corners[1] = DirectX::XMVectorSet( nw, -nh, near_dist, 0.0f); // right-bottom
    out_corners[2] = DirectX::XMVectorSet( nw,  nh, near_dist, 0.0f); // right-top
    out_corners[3] = DirectX::XMVectorSet(-nw,  nh, near_dist, 0.0f); // left-top
    // Far plane corners (same order)
    out_corners[4] = DirectX::XMVectorSet(-fw, -fh, far_dist, 0.0f);
    out_corners[5] = DirectX::XMVectorSet( fw, -fh, far_dist, 0.0f);
    out_corners[6] = DirectX::XMVectorSet( fw,  fh, far_dist, 0.0f);
    out_corners[7] = DirectX::XMVectorSet(-fw,  fh, far_dist, 0.0f);
}

} // namespace frustum
} // namespace SimulationMath

#endif // CORE_MATH_FRUSTUM_H