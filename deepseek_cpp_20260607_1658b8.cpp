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

#ifndef ORTHOTREE_ADAPTERS_XYZ_H_INCLUDED
#define ORTHOTREE_ADAPTERS_XYZ_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"

#include <type_traits>
#include <cstddef>

namespace OrthoTree {
namespace Adapters {
namespace XYZ {

// ============================================================================
//  Trait to detect if a type has x, y, (z) members (trivial struct)
//  This is similar to the general adapters but specialised for 2D/3D
// ============================================================================
template<typename T, typename = void>
struct is_xyz_point : std::false_type {};

template<typename T>
struct is_xyz_point<T, std::void_t<decltype(T::x), decltype(T::y), decltype(T::z)>>
    : std::true_type {};

template<typename T>
struct is_xy_point : std::false_type {};

template<typename T>
struct is_xy_point<T, std::void_t<decltype(T::x), decltype(T::y)>>
    : std::integral_constant<bool, !is_xyz_point<T>::value> {};

// ============================================================================
//  Converters for trivial struct {T x,y,z} or {T x,y}
// ============================================================================
template<typename StructType>
struct StructPointConverter {
    using scalar_type = std::conditional_t<
        std::is_arithmetic_v<decltype(StructType::x)>,
        decltype(StructType::x),
        float
    >;
    static constexpr int dimension = (is_xyz_point<StructType>::value) ? 3 : 2;
    using ortho_vector = Math::Vector<scalar_type, dimension>;

    static ortho_vector to_ortho(const StructType& p) {
        if constexpr (dimension == 3) {
            return ortho_vector(p.x, p.y, p.z);
        } else {
            return ortho_vector(p.x, p.y);
        }
    }

    static StructType from_ortho(const ortho_vector& v) {
        StructType result;
        result.x = static_cast<scalar_type>(v[0]);
        result.y = static_cast<scalar_type>(v[1]);
        if constexpr (dimension == 3) result.z = static_cast<scalar_type>(v[2]);
        return result;
    }
};

// ----------------------------------------------------------------------------
//  SIMD batch conversion (4 points at a time)
// ----------------------------------------------------------------------------
template<typename StructType>
void batch_to_ortho(const StructType* src, Math::Vector<typename StructPointConverter<StructType>::scalar_type,
                                                       StructPointConverter<StructType>::dimension>* dst,
                    std::size_t count) {
    using T = typename StructPointConverter<StructType>::scalar_type;
    constexpr int N = StructPointConverter<StructType>::dimension;
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
        for (std::size_t i = 0; i < count; ++i) {
            dst[i][0] = static_cast<T>(src[i].x);
            dst[i][1] = static_cast<T>(src[i].y);
            dst[i][2] = static_cast<T>(src[i].z);
        }
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            dst[i][0] = static_cast<T>(src[i].x);
            dst[i][1] = static_cast<T>(src[i].y);
            if constexpr (N == 3) dst[i][2] = static_cast<T>(src[i].z);
        }
    }
}

// ============================================================================
//  Bounding box adapter for structs representing AABB (min and max points)
//  The user can provide a struct with two points: .min and .max, or a pair.
//  We'll support std::pair<StructPoint, StructPoint> as fallback.
// ============================================================================
template<typename StructBox>
struct BoxConverter {
    using point_type = typename StructBox::point_type; // user must provide
    using scalar_type = typename StructPointConverter<point_type>::scalar_type;
    static constexpr int dimension = StructPointConverter<point_type>::dimension;
    using ortho_aabb = Math::AxisAlignedBox<scalar_type, dimension>;

    static ortho_aabb to_ortho(const StructBox& box) {
        auto minP = StructPointConverter<point_type>::to_ortho(box.min);
        auto maxP = StructPointConverter<point_type>::to_ortho(box.max);
        return ortho_aabb(minP, maxP);
    }

    static StructBox from_ortho(const ortho_aabb& aabb) {
        StructBox result;
        result.min = StructPointConverter<point_type>::from_ortho(aabb.min());
        result.max = StructPointConverter<point_type>::from_ortho(aabb.max());
        return result;
    }
};

// ----------------------------------------------------------------------------
//  Generic pair converter (for std::pair<StructPoint, StructPoint>)
// ----------------------------------------------------------------------------
template<typename PointType>
struct PairBoxConverter {
    using scalar_type = typename StructPointConverter<PointType>::scalar_type;
    static constexpr int dimension = StructPointConverter<PointType>::dimension;
    using ortho_aabb = Math::AxisAlignedBox<scalar_type, dimension>;

    static ortho_aabb to_ortho(const std::pair<PointType, PointType>& pair) {
        auto minP = StructPointConverter<PointType>::to_ortho(pair.first);
        auto maxP = StructPointConverter<PointType>::to_ortho(pair.second);
        return ortho_aabb(minP, maxP);
    }

    static std::pair<PointType, PointType> from_ortho(const ortho_aabb& aabb) {
        auto minP = StructPointConverter<PointType>::from_ortho(aabb.min());
        auto maxP = StructPointConverter<PointType>::from_ortho(aabb.max());
        return {minP, maxP};
    }
};

// ============================================================================
//  Dynamic environment controller for XYZ adapters
// ============================================================================
class XYZAdapterEnvironment {
public:
    using Scalar = double;

    XYZAdapterEnvironment() noexcept
        : m_useSIMD(true)
        , m_autoConvert(true)
        , m_useDoublePrecision(true) {}

    void setUseSIMD(bool use) noexcept { m_useSIMD = use; }
    bool useSIMD() const noexcept { return m_useSIMD; }

    void setAutoConvert(bool autoCvt) noexcept { m_autoConvert = autoCvt; }
    bool autoConvert() const noexcept { return m_autoConvert; }

    void setUseDoublePrecision(bool use) noexcept { m_useDoublePrecision = use; }
    bool useDoublePrecision() const noexcept { return m_useDoublePrecision; }

    // Global transform: apply scaling, offset, rotation to all points
    void setTransform(const Math::AffineTransform<double, 3>& tf) { m_transform = tf; m_hasTransform = true; }
    void clearTransform() { m_hasTransform = false; }

    template<typename StructType>
    StructType transformPoint(const StructType& p) const {
        if (!m_hasTransform) return p;
        auto orthoP = StructPointConverter<StructType>::to_ortho(p);
        auto transformed = m_transform.transform(orthoP);
        return StructPointConverter<StructType>::from_ortho(transformed);
    }

private:
    bool m_useSIMD;
    bool m_autoConvert;
    bool m_useDoublePrecision;
    Math::AffineTransform<double, 3> m_transform;
    bool m_hasTransform = false;
};

// ----------------------------------------------------------------------------
//  Singleton access
// ----------------------------------------------------------------------------
inline XYZAdapterEnvironment& xyz_adapter_env() {
    static XYZAdapterEnvironment env;
    return env;
}

// ============================================================================
//  Entity adapter for xyz types (for octree integration)
// ============================================================================
template<typename T>
struct EntityAdapter {
    using point_type = T;
    using scalar_type = typename StructPointConverter<point_type>::scalar_type;
    static constexpr int dimension = StructPointConverter<point_type>::dimension;
    using ortho_aabb = Math::AxisAlignedBox<scalar_type, dimension>;

    static ortho_aabb getBounds(const T& geom) {
        // If T is a point (has x,y, optional z)
        if constexpr (is_xyz_point<T>::value || is_xy_point<T>::value) {
            auto p = StructPointConverter<T>::to_ortho(geom);
            return ortho_aabb(p, p);
        } else {
            // Assume it's a box with .min and .max members
            auto minP = StructPointConverter<decltype(geom.min)>::to_ortho(geom.min);
            auto maxP = StructPointConverter<decltype(geom.max)>::to_ortho(geom.max);
            return ortho_aabb(minP, maxP);
        }
    }
};

} // namespace XYZ
} // namespace Adapters
} // namespace OrthoTree

#endif // ORTHOTREE_ADAPTERS_XYZ_H_INCLUDED