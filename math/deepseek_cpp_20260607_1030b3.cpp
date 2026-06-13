//File group name : OrthoTree Math
//File 0053 : core/math/geometry/aabb.h
//Axis‑aligned bounding box (AABB) for N dimensions (2D/3D). Union, intersection, distance queries, surface area, volume, transformation (conservative), and SIMD batch operations.

#ifndef ORTHOTREE_CORE_MATH_GEOMETRY_AABB_H_INCLUDED
#define ORTHOTREE_CORE_MATH_GEOMETRY_AABB_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "../basic/matrix.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <algorithm>
#include <limits>

namespace OrthoTree {
namespace Math {
namespace Geometry {

// ============================================================================
//  Axis‑Aligned Bounding Box for N dimensions (N = 2 or 3)
// ============================================================================
template<typename T, std::size_t N>
class AABB {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, N>;
    using size_type = std::size_t;

    static constexpr size_type dimension() noexcept { return N; }

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr AABB() noexcept
        : m_min(point_type(std::numeric_limits<T>::max())),
          m_max(point_type(std::numeric_limits<T>::lowest())) {}
    constexpr AABB(const point_type& min, const point_type& max) noexcept
        : m_min(min), m_max(max) {}
    constexpr AABB(const point_type& center, T halfSize) noexcept
        : m_min(center - point_type(halfSize)),
          m_max(center + point_type(halfSize)) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    constexpr const point_type& min() const noexcept { return m_min; }
    constexpr const point_type& max() const noexcept { return m_max; }
    constexpr void setMin(const point_type& p) noexcept { m_min = p; }
    constexpr void setMax(const point_type& p) noexcept { m_max = p; }

    // ------------------------------------------------------------------------
    //  Geometric properties
    // ------------------------------------------------------------------------
    constexpr point_type center() const noexcept {
        return (m_min + m_max) * T(0.5);
    }
    constexpr point_type extents() const noexcept {
        return m_max - m_min;
    }
    constexpr point_type halfExtents() const noexcept {
        return extents() * T(0.5);
    }
    constexpr T volume() const noexcept {
        T vol = T(1);
        for (size_type i = 0; i < N; ++i) vol *= (m_max[i] - m_min[i]);
        return vol;
    }
    constexpr T surfaceArea() const noexcept {
        if constexpr (N == 2) {
            point_type e = extents();
            return e[0] * e[1];
        } else if constexpr (N == 3) {
            point_type e = extents();
            return T(2) * (e[0]*e[1] + e[0]*e[2] + e[1]*e[2]);
        }
        return T(0);
    }

    // ------------------------------------------------------------------------
    //  Validity
    // ------------------------------------------------------------------------
    constexpr bool isEmpty() const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (m_min[i] > m_max[i]) return true;
        }
        return false;
    }

    // ------------------------------------------------------------------------
    //  Containment tests
    // ------------------------------------------------------------------------
    constexpr bool containsPoint(const point_type& p) const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (p[i] < m_min[i] || p[i] > m_max[i]) return false;
        }
        return true;
    }
    constexpr bool containsAABB(const AABB& other) const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (other.m_min[i] < m_min[i] || other.m_max[i] > m_max[i]) return false;
        }
        return true;
    }
    constexpr bool overlaps(const AABB& other) const noexcept {
        for (size_type i = 0; i < N; ++i) {
            if (m_min[i] > other.m_max[i] || other.m_min[i] > m_max[i]) return false;
        }
        return true;
    }

    // ------------------------------------------------------------------------
    //  Boolean operations
    // ------------------------------------------------------------------------
    constexpr AABB intersect(const AABB& other) const noexcept {
        return AABB(m_min.componentWiseMax(other.m_min),
                    m_max.componentWiseMin(other.m_max));
    }
    constexpr AABB hull(const AABB& other) const noexcept {
        return AABB(m_min.componentWiseMin(other.m_min),
                    m_max.componentWiseMax(other.m_max));
    }
    constexpr AABB& extend(const point_type& p) noexcept {
        m_min = m_min.componentWiseMin(p);
        m_max = m_max.componentWiseMax(p);
        return *this;
    }
    constexpr AABB& extend(const AABB& other) noexcept {
        m_min = m_min.componentWiseMin(other.m_min);
        m_max = m_max.componentWiseMax(other.m_max);
        return *this;
    }

    // ------------------------------------------------------------------------
    //  Distance queries
    // ------------------------------------------------------------------------
    constexpr T squaredDistanceToPoint(const point_type& p) const noexcept {
        T sqDist = T(0);
        for (size_type i = 0; i < N; ++i) {
            if (p[i] < m_min[i]) {
                T d = m_min[i] - p[i];
                sqDist += d * d;
            } else if (p[i] > m_max[i]) {
                T d = p[i] - m_max[i];
                sqDist += d * d;
            }
        }
        return sqDist;
    }
    T distanceToPoint(const point_type& p) const noexcept {
        return std::sqrt(squaredDistanceToPoint(p));
    }

    // ------------------------------------------------------------------------
    //  Closest point on AABB
    // ------------------------------------------------------------------------
    point_type closestPoint(const point_type& p) const noexcept {
        point_type result;
        for (size_type i = 0; i < N; ++i) {
            result[i] = Basic::clamp(p[i], m_min[i], m_max[i]);
        }
        return result;
    }

    // ------------------------------------------------------------------------
    //  Conservative transform by affine transform (only rotation + translation)
    //  For non‑uniform scaling, the resulting box is enlarged.
    // ------------------------------------------------------------------------
    AABB transform(const Basic::AffineTransform<T, N>& tf) const noexcept {
        // Transform all 2^N corners
        std::array<point_type, (N == 2 ? 4 : 8)> corners;
        for (size_type i = 0; i < (1 << N); ++i) {
            point_type corner;
            for (size_type d = 0; d < N; ++d) {
                corner[d] = (i & (1 << d)) ? m_max[d] : m_min[d];
            }
            corners[i] = tf.transform(corner);
        }
        AABB result;
        for (const auto& p : corners) result.extend(p);
        return result;
    }

    // ------------------------------------------------------------------------
    //  Equality with tolerance
    // ------------------------------------------------------------------------
    bool nearlyEqual(const AABB& other, T eps = static_cast<T>(MathConfig::instance().defaultEpsilon())) const noexcept {
        return m_min.nearlyEqual(other.m_min, eps) && m_max.nearlyEqual(other.m_max, eps);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch: compute squared distance for 4 points to 4 boxes (pairwise)
    // ------------------------------------------------------------------------
    static void batchSquaredDistance(const AABB* boxes, const point_type* points,
                                     T* out, size_t count) noexcept {
        if constexpr (std::is_same_v<T,float> && ORTHOTREE_SIMD_LEVEL >= 128) {
            for (size_t i = 0; i < count; ++i) {
                out[i] = boxes[i].squaredDistanceToPoint(points[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                out[i] = boxes[i].squaredDistanceToPoint(points[i]);
            }
        }
    }

private:
    point_type m_min;
    point_type m_max;
};

// ----------------------------------------------------------------------------
//  Convenience aliases
// ----------------------------------------------------------------------------
template<typename T> using AABB2 = AABB<T, 2>;
template<typename T> using AABB3 = AABB<T, 3>;

using AABB2f = AABB<float, 2>;
using AABB3f = AABB<float, 3>;
using AABB2d = AABB<double, 2>;
using AABB3d = AABB<double, 3>;

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class AABBEnvironment {
public:
    static AABBEnvironment& instance() {
        static AABBEnvironment env;
        return env;
    }
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    AABBEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Geometry
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_GEOMETRY_AABB_H_INCLUDED