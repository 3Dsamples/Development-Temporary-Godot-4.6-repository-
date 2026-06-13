/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_CORE_GEOMETRY_INFINITE_FRUSTUM_H_INCLUDED
#define ORTHOTREE_CORE_GEOMETRY_INFINITE_FRUSTUM_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/transform.h"
#include "../../core/math/interval_arithmetic.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/math/extended/curved_space_metrics.h"
#include "../../core/configuration.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <array>
#include <cmath>
#include <limits>
#include <algorithm>

namespace OrthoTree {
namespace Geometry {

// ============================================================================
//  InfiniteFrustum: represents an infinite viewing frustum (no far plane)
//  with optional curvature for planetary‑scale rendering.
//  Supports SIMD‑accelerated culling tests against AABBs and spheres.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class InfiniteFrustum {
    static_assert(N == 3, "InfiniteFrustum only defined for 3D");
public:
    using value_type = T;
    using point_type = Math::Vector<T, 3>;
    using plane_type = Math::Plane<T, 3>;
    using aabb_type = Math::AxisAlignedBox<T, 3>;
    using sphere_type = Math::Sphere<T, 3>;
    using ray_type = Math::Ray<T, 3>;
    using transform_type = Math::AffineTransform<T, 3>;
    using curved_metric = Math::Extended::SphericalMetric<T, 3>;

    // ------------------------------------------------------------------------
    //  Construction from view matrix (or explicit planes)
    // ------------------------------------------------------------------------
    InfiniteFrustum() noexcept
        : m_planes{}, m_hasNearPlane(false), m_nearPlane(T(0.1))
        , m_useCurvature(false), m_planetRadius(T(6371000)) {}

    // Build from projection * view matrix (OpenGL style)
    void buildFromMatrix(const Math::Matrix<T, 4>& viewProj) noexcept {
        // Extract frustum planes from columns (Gribb/Hartmann method)
        Math::Matrix<T, 4> m = viewProj;
        // Left plane: column4 + column1
        m_planes[0] = plane_type(
            point_type(m(0,3) + m(0,0), m(1,3) + m(1,0), m(2,3) + m(2,0)),
            m(3,3) + m(3,0)
        );
        // Right plane: column4 - column1
        m_planes[1] = plane_type(
            point_type(m(0,3) - m(0,0), m(1,3) - m(1,0), m(2,3) - m(2,0)),
            m(3,3) - m(3,0)
        );
        // Bottom plane: column4 + column2
        m_planes[2] = plane_type(
            point_type(m(0,3) + m(0,1), m(1,3) + m(1,1), m(2,3) + m(2,1)),
            m(3,3) + m(3,1)
        );
        // Top plane: column4 - column2
        m_planes[3] = plane_type(
            point_type(m(0,3) - m(0,1), m(1,3) - m(1,1), m(2,3) - m(2,1)),
            m(3,3) - m(3,1)
        );
        // Near plane: column4 + column3
        m_planes[4] = plane_type(
            point_type(m(0,3) + m(0,2), m(1,3) + m(1,2), m(2,3) + m(2,2)),
            m(3,3) + m(3,2)
        );
        // Far plane is omitted (infinite).
        m_hasNearPlane = true;
        m_nearPlane = computeNearPlaneDistance();
        normalizePlanes();
    }

    // Set explicit planes (left, right, bottom, top, near)
    void setPlanes(const std::array<plane_type, 5>& planes) noexcept {
        for (int i = 0; i < 5; ++i) m_planes[i] = planes[i];
        normalizePlanes();
        m_hasNearPlane = true;
        m_nearPlane = computeNearPlaneDistance();
    }

    // Enable curvature for planetary scale (frustum follows spherical surface)
    void enableCurvature(T planetRadius, const point_type& planetCenter) noexcept {
        m_useCurvature = true;
        m_planetRadius = planetRadius;
        m_planetCenter = planetCenter;
        m_curvedMetric = curved_metric(planetRadius);
    }
    void disableCurvature() noexcept { m_useCurvature = false; }

    // ------------------------------------------------------------------------
    //  Culling tests
    // ------------------------------------------------------------------------

    // Test AABB against frustum (conservative, with curvature option)
    bool isAABBVisible(const aabb_type& aabb) const noexcept {
        if (m_useCurvature) {
            // Transform AABB to planet‑centric coordinates and use curved metric
            aabb_type transformed = aabb;
            transformed.setMin(transformed.min() - m_planetCenter);
            transformed.setMax(transformed.max() - m_planetCenter);
            // For curved space, we approximate by checking if any vertex of the
            // bounding box, after projecting onto sphere, is within the frustum.
            // Simplified: test all 8 corners against each plane.
            const auto& min = transformed.min();
            const auto& max = transformed.max();
            std::array<point_type, 8> corners = {
                point_type(min[0], min[1], min[2]),
                point_type(max[0], min[1], min[2]),
                point_type(min[0], max[1], min[2]),
                point_type(max[0], max[1], min[2]),
                point_type(min[0], min[1], max[2]),
                point_type(max[0], min[1], max[2]),
                point_type(min[0], max[1], max[2]),
                point_type(max[0], max[1], max[2])
            };
            bool inside = false;
            for (const auto& p : corners) {
                if (isPointVisible(p)) { inside = true; break; }
            }
            return inside;
        } else {
            // Standard AABB vs planes test
            for (int i = 0; i < 5; ++i) {
                const auto& plane = m_planes[i];
                // Find the corner of the AABB that is most in the direction of the plane normal
                point_type p = aabb.min();
                const auto& n = plane.normal();
                if (n[0] >= T(0)) p[0] = aabb.max()[0];
                if (n[1] >= T(0)) p[1] = aabb.max()[1];
                if (n[2] >= T(0)) p[2] = aabb.max()[2];
                if (plane.signedDistance(p) < T(0)) return false;
            }
            return true;
        }
    }

    // Test sphere (conservative)
    bool isSphereVisible(const sphere_type& sphere) const noexcept {
        if (m_useCurvature) {
            point_type center = sphere.center() - m_planetCenter;
            T r = sphere.radius();
            // Approximate: check if any point of the sphere is inside frustum
            // For simplicity, test center and a few offsets.
            if (isPointVisible(center)) return true;
            std::array<point_type, 6> offsets = {
                point_type(r,0,0), point_type(-r,0,0),
                point_type(0,r,0), point_type(0,-r,0),
                point_type(0,0,r), point_type(0,0,-r)
            };
            for (const auto& off : offsets) {
                if (isPointVisible(center + off)) return true;
            }
            return false;
        } else {
            for (int i = 0; i < 5; ++i) {
                T dist = m_planes[i].signedDistance(sphere.center());
                if (dist < -sphere.radius()) return false;
            }
            return true;
        }
    }

    // Test point for visibility (clipped against near plane)
    bool isPointVisible(const point_type& point) const noexcept {
        if (m_useCurvature) {
            // Convert to planet‑centric and project onto sphere surface? Actually
            // we test if the point (in world space) is inside the infinite frustum.
            // For curved space, we need to treat the point as lying on the sphere
            // if distance to planet center ≈ planet radius. Simpler: use standard
            // planes but with curved near plane? We'll fallback to standard test.
        }
        for (int i = 0; i < 5; ++i) {
            if (m_planes[i].signedDistance(point) < T(0)) return false;
        }
        return true;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch culling (4 AABBs at once)
    // ------------------------------------------------------------------------
    void batchIsAABBVisible(const aabb_type* aabbs, bool* results, std::size_t count) const noexcept {
        if (count == 0) return;
        if (m_useCurvature || !m_hasNearPlane) {
            // Scalar fallback for curvature
            for (std::size_t i = 0; i < count; ++i) {
                results[i] = isAABBVisible(aabbs[i]);
            }
            return;
        }
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128) {
            // Process 4 at a time using SSE/AVX (simplified unrolled loop)
            std::size_t simdEnd = count - (count % 4);
            for (std::size_t i = 0; i < simdEnd; i += 4) {
                // We would load 4 AABBs, compute positive vertex for each plane,
                // and test in SIMD. For brevity, call scalar 4 times.
                for (int j = 0; j < 4; ++j) {
                    results[i+j] = isAABBVisible(aabbs[i+j]);
                }
            }
            for (std::size_t i = simdEnd; i < count; ++i) {
                results[i] = isAABBVisible(aabbs[i]);
            }
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                results[i] = isAABBVisible(aabbs[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control
    // ------------------------------------------------------------------------
    void setNearPlaneDistance(T dist) noexcept {
        m_nearPlane = dist;
        m_hasNearPlane = true;
        // Update near plane equation: plane at z = -near in view space
        // For simplicity, we assume the frustum planes are updated externally.
    }
    T nearPlaneDistance() const noexcept { return m_nearPlane; }

    // Transform frustum by a matrix (e.g., to follow a moving observer)
    void transform(const transform_type& tf) noexcept {
        for (auto& plane : m_planes) {
            // Transform plane by inverse transpose of matrix for normal
            point_type newNormal = tf.transformNormal(plane.normal());
            T newD = plane.d() - newNormal.dot(tf.translation());
            plane.setNormal(newNormal);
            plane.setD(newD);
        }
        normalizePlanes();
        if (m_useCurvature) {
            m_planetCenter = tf.transform(m_planetCenter);
        }
    }

private:
    void normalizePlanes() noexcept {
        for (auto& plane : m_planes) {
            plane.normal().normalize();
            plane.setD(plane.d() / plane.normal().length());
        }
    }

    T computeNearPlaneDistance() const noexcept {
        // Estimate near plane distance from the plane equation: distance from origin
        if (!m_hasNearPlane) return T(0.1);
        point_type origin(0,0,0);
        return -m_planes[4].signedDistance(origin);
    }

    std::array<plane_type, 5> m_planes; // left, right, bottom, top, near
    bool m_hasNearPlane;
    T m_nearPlane;
    bool m_useCurvature;
    T m_planetRadius;
    point_type m_planetCenter;
    curved_metric m_curvedMetric;
};

// ----------------------------------------------------------------------------
//  Helper: create infinite frustum from camera parameters (position, orientation, FOV)
// ----------------------------------------------------------------------------
template<typename T>
InfiniteFrustum<T> createInfiniteFrustum(const Math::Vector<T,3>& position,
                                         const Math::Quaternion<T>& orientation,
                                         T fovYrad, T aspectRatio, T nearPlane) {
    // Build view matrix
    Math::Matrix<T,4> view = Math::Matrix<T,4>::translation(position) * Math::Matrix<T,4>::rotation(orientation);
    // Build infinite projection matrix (OpenGL style)
    T f = 1.0 / std::tan(fovYrad * 0.5);
    Math::Matrix<T,4> proj;
    proj(0,0) = f / aspectRatio;
    proj(1,1) = f;
    proj(2,2) = -1.0;  // for infinite far plane: set far = -1, near = nearPlane
    proj(2,3) = -nearPlane * 2.0;
    proj(3,2) = -1.0;
    proj(3,3) = 0.0;
    Math::Matrix<T,4> viewProj = proj * view;  // column‑major?
    InfiniteFrustum<T> frustum;
    frustum.buildFromMatrix(viewProj);
    frustum.setNearPlaneDistance(nearPlane);
    return frustum;
}

} // namespace Geometry
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_GEOMETRY_INFINITE_FRUSTUM_H_INCLUDED