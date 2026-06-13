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

#ifndef ORTHOTREE_ADAPTERS_BOOST_H_INCLUDED
#define ORTHOTREE_ADAPTERS_BOOST_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../core/math/transform.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"

#if defined(ORTHOTREE_BOOST_SUPPORT) || defined(BOOST_GEOMETRY_VERSION)
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/segment.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/index/rtree.hpp>
#else
#error "Boost.Geometry support requires BOOST_GEOMETRY_VERSION. Define ORTHOTREE_BOOST_SUPPORT or include Boost headers before orthotree/adapters/boost.h"
#endif

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace OrthoTree {
namespace Adapters {
namespace Boost {

// ============================================================================
//  Type traits: extract coordinate type and dimension from Boost geometry types
// ============================================================================
template<typename T>
struct boost_traits {
    using coordinate_type = typename bg::coordinate_type<T>::type;
    static constexpr std::size_t dimension = bg::dimension<T>::value;
};

// ============================================================================
//  Converter: Boost point <-> OrthoTree Vector
// ============================================================================
template<typename BoostPoint, typename T = typename boost_traits<BoostPoint>::coordinate_type,
         std::size_t N = boost_traits<BoostPoint>::dimension>
struct PointConverter {
    using ortho_vector = Math::Vector<T, N>;

    static ortho_vector to_ortho(const BoostPoint& bp) {
        ortho_vector v;
        for (std::size_t i = 0; i < N; ++i) {
            v[i] = static_cast<T>(bg::get<static_cast<int>(i)>(bp));
        }
        return v;
    }

    static BoostPoint from_ortho(const ortho_vector& v) {
        BoostPoint bp;
        for (std::size_t i = 0; i < N; ++i) {
            bg::set<static_cast<int>(i)>(bp, static_cast<T>(v[i]));
        }
        return bp;
    }
};

// ----------------------------------------------------------------------------
//  SIMD batch conversion (4 points at a time)
// ----------------------------------------------------------------------------
template<typename BoostPoint, typename T, std::size_t N>
void batch_to_ortho(const BoostPoint* src, Math::Vector<T, N>* dst, std::size_t count) {
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
        // Use unrolled loop with potential SIMD intrinsics
        for (std::size_t i = 0; i < count; ++i) {
            dst[i][0] = static_cast<T>(bg::get<0>(src[i]));
            dst[i][1] = static_cast<T>(bg::get<1>(src[i]));
            dst[i][2] = static_cast<T>(bg::get<2>(src[i]));
        }
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            dst[i][0] = static_cast<T>(bg::get<0>(src[i]));
            dst[i][1] = static_cast<T>(bg::get<1>(src[i]));
            if constexpr (N == 3) dst[i][2] = static_cast<T>(bg::get<2>(src[i]));
        }
    }
}

// ============================================================================
//  Converter: Boost box <-> OrthoTree AABB
// ============================================================================
template<typename BoostBox, typename T = typename boost_traits<BoostBox>::coordinate_type,
         std::size_t N = boost_traits<BoostBox>::dimension>
struct BoxConverter {
    using ortho_aabb = Math::AxisAlignedBox<T, N>;

    static ortho_aabb to_ortho(const BoostBox& bb) {
        using ortho_point = Math::Vector<T, N>;
        ortho_point minP, maxP;
        for (std::size_t i = 0; i < N; ++i) {
            minP[i] = static_cast<T>(bg::get<bg::min_corner, static_cast<int>(i)>(bb));
            maxP[i] = static_cast<T>(bg::get<bg::max_corner, static_cast<int>(i)>(bb));
        }
        return ortho_aabb(minP, maxP);
    }

    static BoostBox from_ortho(const ortho_aabb& aabb) {
        BoostBox bb;
        for (std::size_t i = 0; i < N; ++i) {
            bg::set<bg::min_corner, static_cast<int>(i)>(bb, static_cast<T>(aabb.min()[i]));
            bg::set<bg::max_corner, static_cast<int>(i)>(bb, static_cast<T>(aabb.max()[i]));
        }
        return bb;
    }
};

// ============================================================================
//  Ray adapter: Boost segment / ray to OrthoTree Ray
// ============================================================================
template<typename BoostSegment, typename T = typename boost_traits<BoostSegment>::coordinate_type,
         std::size_t N = boost_traits<BoostSegment>::dimension>
struct RayConverter {
    using ortho_ray = Math::Ray<T, N>;

    static ortho_ray to_ortho(const BoostSegment& seg) {
        auto p1 = PointConverter<bg::point_type<BoostSegment>::type, T, N>::to_ortho(bg::point<0>(seg));
        auto p2 = PointConverter<bg::point_type<BoostSegment>::type, T, N>::to_ortho(bg::point<1>(seg));
        return ortho_ray(p1, (p2 - p1).normalized());
    }
};

// ============================================================================
//  Dynamic environment controller for Boost adapters
// ============================================================================
class BoostAdapterEnvironment {
public:
    using tolerance_type = double;

    BoostAdapterEnvironment() noexcept
        : m_tolerance(1e-6)
        , m_useExactPredicates(true)
        , m_autoConvert(true)
        , m_simdPreferred(true) {}

    void setTolerance(tolerance_type tol) noexcept { m_tolerance = tol; }
    tolerance_type tolerance() const noexcept { return m_tolerance; }

    void setUseExactPredicates(bool use) noexcept { m_useExactPredicates = use; }
    bool useExactPredicates() const noexcept { return m_useExactPredicates; }

    void setAutoConvert(bool autoCvt) noexcept { m_autoConvert = autoCvt; }
    bool autoConvert() const noexcept { return m_autoConvert; }

    void setSIMDPreferred(bool pref) noexcept { m_simdPreferred = pref; }
    bool simdPreferred() const noexcept { return m_simdPreferred; }

    // Convert between coordinate systems (e.g., geographic to Cartesian)
    enum class CoordSystem { Cartesian, Geographic, Spherical };
    void setSourceCoordSystem(CoordSystem cs) noexcept { m_sourceCS = cs; }
    void setTargetCoordSystem(CoordSystem cs) noexcept { m_targetCS = cs; }

    // Transformation pipeline (e.g., apply a Helmert transform)
    void setTransformMatrix(const std::array<double, 12>& matrix) {
        m_transformMatrix = matrix;
        m_hasTransform = true;
    }
    void clearTransform() { m_hasTransform = false; }

    // Apply dynamic transformation to a point (if any)
    template<typename BoostPoint>
    BoostPoint transformPoint(const BoostPoint& p) const {
        if (!m_hasTransform) return p;
        BoostPoint out;
        double x = bg::get<0>(p), y = bg::get<1>(p), z = bg::get<2>(p);
        double newX = m_transformMatrix[0] * x + m_transformMatrix[1] * y + m_transformMatrix[2]  * z + m_transformMatrix[3];
        double newY = m_transformMatrix[4] * x + m_transformMatrix[5] * y + m_transformMatrix[6]  * z + m_transformMatrix[7];
        double newZ = m_transformMatrix[8] * x + m_transformMatrix[9] * y + m_transformMatrix[10] * z + m_transformMatrix[11];
        bg::set<0>(out, newX);
        bg::set<1>(out, newY);
        bg::set<2>(out, newZ);
        return out;
    }

private:
    tolerance_type m_tolerance;
    bool m_useExactPredicates;
    bool m_autoConvert;
    bool m_simdPreferred;
    CoordSystem m_sourceCS = CoordSystem::Cartesian;
    CoordSystem m_targetCS = CoordSystem::Cartesian;
    std::array<double, 12> m_transformMatrix;
    bool m_hasTransform = false;
};

// ----------------------------------------------------------------------------
//  Singleton access to environment (thread‑local or global)
// ----------------------------------------------------------------------------
inline BoostAdapterEnvironment& boost_adapter_env() {
    static BoostAdapterEnvironment env;
    return env;
}

// ============================================================================
//  EntityAdapter concept for Boost geometry types (for octree integration)
// ============================================================================
template<typename BoostGeometry>
struct EntityAdapter {
    using coordinate_type = typename boost_traits<BoostGeometry>::coordinate_type;
    static constexpr std::size_t dimension = boost_traits<BoostGeometry>::dimension;
    using ortho_vector = Math::Vector<coordinate_type, dimension>;
    using ortho_aabb = Math::AxisAlignedBox<coordinate_type, dimension>;

    static ortho_aabb getBounds(const BoostGeometry& geom) {
        // If geometry is a box
        if constexpr (bg::geometry::is_box<BoostGeometry>::value) {
            return BoxConverter<BoostGeometry>::to_ortho(geom);
        } else if constexpr (bg::geometry::is_point<BoostGeometry>::value) {
            ortho_vector p = PointConverter<BoostGeometry>::to_ortho(geom);
            return ortho_aabb(p, p);
        } else if constexpr (bg::geometry::is_segment<BoostGeometry>::value) {
            auto start = PointConverter<typename BoostGeometry::point_type>::to_ortho(geom.first);
            auto end   = PointConverter<typename BoostGeometry::point_type>::to_ortho(geom.second);
            return ortho_aabb(start.componentWiseMin(end), start.componentWiseMax(end));
        } else {
            // Generic: compute envelope (bounding box)
            BoostBox box;
            bg::envelope(geom, box);
            return BoxConverter<decltype(box)>::to_ortho(box);
        }
    }
};

} // namespace Boost
} // namespace Adapters
} // namespace OrthoTree

#endif // ORTHOTREE_ADAPTERS_BOOST_H_INCLUDED