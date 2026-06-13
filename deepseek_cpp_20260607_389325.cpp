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

#ifndef ORTHOTREE_ADAPTERS_CGAL_H_INCLUDED
#define ORTHOTREE_ADAPTERS_CGAL_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../core/math/transform.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"

#if defined(ORTHOTREE_CGAL_SUPPORT) || defined(CGAL_VERSION)
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Point_2.h>
#include <CGAL/Point_3.h>
#include <CGAL/Iso_rectangle_2.h>
#include <CGAL/Iso_cuboid_3.h>
#include <CGAL/Ray_2.h>
#include <CGAL/Ray_3.h>
#include <CGAL/Segment_2.h>
#include <CGAL/Segment_3.h>
#else
#error "CGAL support requires CGAL headers. Define ORTHOTREE_CGAL_SUPPORT or include CGAL before orthotree/adapters/cgal.h"
#endif

namespace OrthoTree {
namespace Adapters {
namespace CGAL {

// ============================================================================
//  Kernel type selection (double by default)
// ============================================================================
using DefaultKernel = CGAL::Simple_cartesian<double>;

// ============================================================================
//  Type traits: extract dimension and coordinate type
// ============================================================================
template<typename T>
struct cgal_traits {
    using Kernel = typename T::R;
    using FT = typename Kernel::FT;
    static constexpr int dimension = T::dimension;
};

// ============================================================================
//  Point converter: CGAL::Point_2/3 <-> OrthoTree Vector
// ============================================================================
template<typename CGALPoint, typename T = typename cgal_traits<CGALPoint>::FT,
         int N = cgal_traits<CGALPoint>::dimension>
struct PointConverter {
    using ortho_vector = Math::Vector<T, N>;

    static ortho_vector to_ortho(const CGALPoint& p) {
        ortho_vector v;
        for (int i = 0; i < N; ++i) {
            v[i] = static_cast<T>(p[i]);
        }
        return v;
    }

    static CGALPoint from_ortho(const ortho_vector& v) {
        if constexpr (N == 2) {
            return CGALPoint(static_cast<T>(v[0]), static_cast<T>(v[1]));
        } else {
            return CGALPoint(static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2]));
        }
    }
};

// ----------------------------------------------------------------------------
//  SIMD batch conversion of CGAL points to OrthoTree vectors
// ----------------------------------------------------------------------------
template<typename CGALPoint, typename T, int N>
void batch_to_ortho(const CGALPoint* src, Math::Vector<T, N>* dst, std::size_t count) {
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
        // Unrolled loop with potential SIMD (pseudo)
        for (std::size_t i = 0; i < count; ++i) {
            dst[i][0] = static_cast<T>(src[i][0]);
            dst[i][1] = static_cast<T>(src[i][1]);
            dst[i][2] = static_cast<T>(src[i][2]);
        }
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            for (int d = 0; d < N; ++d) {
                dst[i][d] = static_cast<T>(src[i][d]);
            }
        }
    }
}

// ============================================================================
//  Bounding box converter: CGAL::Iso_rectangle_2/Iso_cuboid_3 <-> AABB
// ============================================================================
template<typename CGALBox, typename T = typename cgal_traits<CGALBox>::FT,
         int N = cgal_traits<CGALBox>::dimension>
struct BoxConverter {
    using ortho_aabb = Math::AxisAlignedBox<T, N>;

    static ortho_aabb to_ortho(const CGALBox& box) {
        using ortho_point = Math::Vector<T, N>;
        ortho_point minP, maxP;
        for (int i = 0; i < N; ++i) {
            minP[i] = static_cast<T>(box.min(i));
            maxP[i] = static_cast<T>(box.max(i));
        }
        return ortho_aabb(minP, maxP);
    }

    static CGALBox from_ortho(const ortho_aabb& aabb) {
        if constexpr (N == 2) {
            return CGALBox(
                CGAL::Point_2<DefaultKernel>(aabb.min()[0], aabb.min()[1]),
                CGAL::Point_2<DefaultKernel>(aabb.max()[0], aabb.max()[1])
            );
        } else {
            return CGALBox(
                CGAL::Point_3<DefaultKernel>(aabb.min()[0], aabb.min()[1], aabb.min()[2]),
                CGAL::Point_3<DefaultKernel>(aabb.max()[0], aabb.max()[1], aabb.max()[2])
            );
        }
    }
};

// ============================================================================
//  Ray/Segment converter: CGAL::Ray_2/3 or Segment -> OrthoTree Ray
// ============================================================================
template<typename CGALRay, typename T = typename cgal_traits<CGALRay>::FT,
         int N = cgal_traits<CGALRay>::dimension>
struct RayConverter {
    using ortho_ray = Math::Ray<T, N>;

    static ortho_ray to_ortho(const CGALRay& ray) {
        auto origin = PointConverter<typename CGALRay::Point_2, T, N>::to_ortho(ray.source());
        auto second = PointConverter<typename CGALRay::Point_2, T, N>::to_ortho(ray.second_point());
        auto dir = (second - origin).normalized();
        return ortho_ray(origin, dir);
    }
};

// ============================================================================
//  Dynamic environment controller for CGAL adapters (precision, kernel)
// ============================================================================
class CGALAdapterEnvironment {
public:
    using FT = double;

    CGALAdapterEnvironment() noexcept
        : m_useExactPredicates(false)
        , m_useSIMD(true)
        , m_autoConvert(true) {}

    void setUseExactPredicates(bool use) noexcept { m_useExactPredicates = use; }
    bool useExactPredicates() const noexcept { return m_useExactPredicates; }

    void setUseSIMD(bool use) noexcept { m_useSIMD = use; }
    bool useSIMD() const noexcept { return m_useSIMD; }

    void setAutoConvert(bool autoCvt) noexcept { m_autoConvert = autoCvt; }
    bool autoConvert() const noexcept { return m_autoConvert; }

    // Kernel switching (if CGAL supports runtime kernel selection? Not really, but we can store a flag)
    enum class KernelType { Cartesian, ExactPredicates, ExactConstructions };
    void setKernelType(KernelType kt) noexcept { m_kernel = kt; }
    KernelType kernelType() const noexcept { return m_kernel; }

    // Transformation: apply a similarity transform to all points (e.g., scaling, rotation)
    void setTransformMatrix(const std::array<FT, 12>& mat) {
        m_transform = mat;
        m_hasTransform = true;
    }
    void clearTransform() { m_hasTransform = false; }

    template<typename CGALPoint>
    CGALPoint transformPoint(const CGALPoint& p) const {
        if (!m_hasTransform) return p;
        FT x = static_cast<FT>(p[0]), y = static_cast<FT>(p[1]), z = static_cast<FT>(p[2]);
        FT nx = m_transform[0] * x + m_transform[1] * y + m_transform[2] * z + m_transform[3];
        FT ny = m_transform[4] * x + m_transform[5] * y + m_transform[6] * z + m_transform[7];
        FT nz = m_transform[8] * x + m_transform[9] * y + m_transform[10] * z + m_transform[11];
        if constexpr (CGALPoint::dimension == 2) {
            return CGALPoint(nx, ny);
        } else {
            return CGALPoint(nx, ny, nz);
        }
    }

private:
    bool m_useExactPredicates;
    bool m_useSIMD;
    bool m_autoConvert;
    KernelType m_kernel = KernelType::Cartesian;
    std::array<FT, 12> m_transform;
    bool m_hasTransform = false;
};

// ----------------------------------------------------------------------------
//  Singleton access (thread‑local or global)
// ----------------------------------------------------------------------------
inline CGALAdapterEnvironment& cgal_adapter_env() {
    static CGALAdapterEnvironment env;
    return env;
}

// ============================================================================
//  Entity adapter for CGAL geometry types (for octree integration)
// ============================================================================
template<typename CGALGeometry>
struct EntityAdapter {
    using FT = typename cgal_traits<CGALGeometry>::FT;
    static constexpr int dimension = cgal_traits<CGALGeometry>::dimension;
    using ortho_aabb = Math::AxisAlignedBox<FT, dimension>;
    using ortho_point = Math::Vector<FT, dimension>;

    static ortho_aabb getBounds(const CGALGeometry& geom) {
        if constexpr (std::is_same_v<CGALGeometry, CGAL::Point_2<DefaultKernel>> ||
                      std::is_same_v<CGALGeometry, CGAL::Point_3<DefaultKernel>>) {
            ortho_point p = PointConverter<CGALGeometry>::to_ortho(geom);
            return ortho_aabb(p, p);
        } else if constexpr (std::is_same_v<CGALGeometry, CGAL::Iso_rectangle_2<DefaultKernel>> ||
                              std::is_same_v<CGALGeometry, CGAL::Iso_cuboid_3<DefaultKernel>>) {
            return BoxConverter<CGALGeometry>::to_ortho(geom);
        } else {
            // For other types, compute bounding box using CGAL::bbox_2/3
            auto bbox = geom.bbox();
            return BoxConverter<decltype(bbox)>::to_ortho(bbox);
        }
    }
};

} // namespace CGAL
} // namespace Adapters
} // namespace OrthoTree

#endif // ORTHOTREE_ADAPTERS_CGAL_H_INCLUDED