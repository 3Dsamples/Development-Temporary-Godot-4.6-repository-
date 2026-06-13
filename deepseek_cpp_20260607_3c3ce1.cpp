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

#ifndef ORTHOTREE_ADAPTERS_UNREAL_H_INCLUDED
#define ORTHOTREE_ADAPTERS_UNREAL_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../core/math/transform.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"

#if defined(ORTHOTREE_UNREAL_SUPPORT) || (defined(UE_BUILD_DEBUG) || defined(UE_BUILD_DEVELOPMENT) || defined(UE_BUILD_SHIPPING))
#include "Math/Vector.h"
#include "Math/Box.h"
#include "Math/Transform.h"
#include "Math/Matrix.h"
#else
#error "Unreal Engine support requires Unreal headers or define ORTHOTREE_UNREAL_SUPPORT. Include Unreal Math headers before orthotree/adapters/unreal.h"
#endif

namespace OrthoTree {
namespace Adapters {
namespace Unreal {

// ============================================================================
//  Type traits: extract scalar type and dimension from Unreal FVector
// ============================================================================
template<typename T>
struct unreal_traits;

// FVector (float 3D)
template<>
struct unreal_traits<FVector> {
    using Scalar = float;
    static constexpr int dimension = 3;
    static constexpr bool is_double = false;
};

// FVector3d (double 3D) – available in UE5+
template<>
struct unreal_traits<FVector3d> {
    using Scalar = double;
    static constexpr int dimension = 3;
    static constexpr bool is_double = true;
};

// FVector2D (float 2D)
template<>
struct unreal_traits<FVector2D> {
    using Scalar = float;
    static constexpr int dimension = 2;
    static constexpr bool is_double = false;
};

// ============================================================================
//  Point converter: FVector, FVector3d, FVector2D <-> OrthoTree Vector
// ============================================================================
template<typename UnrealVec>
struct PointConverter {
    using Scalar = typename unreal_traits<UnrealVec>::Scalar;
    static constexpr int N = unreal_traits<UnrealVec>::dimension;
    using ortho_vector = Math::Vector<Scalar, N>;

    static ortho_vector to_ortho(const UnrealVec& v) {
        ortho_vector result;
        for (int i = 0; i < N; ++i) {
            result[i] = static_cast<Scalar>(v[i]);
        }
        return result;
    }

    static UnrealVec from_ortho(const ortho_vector& v) {
        if constexpr (N == 2) {
            return UnrealVec(static_cast<Scalar>(v[0]), static_cast<Scalar>(v[1]));
        } else {
            return UnrealVec(static_cast<Scalar>(v[0]), static_cast<Scalar>(v[1]), static_cast<Scalar>(v[2]));
        }
    }
};

// ----------------------------------------------------------------------------
//  SIMD batch conversion (4 vectors at a time)
// ----------------------------------------------------------------------------
template<typename UnrealVec, typename Scalar, int N>
void batch_to_ortho(const UnrealVec* src, Math::Vector<Scalar, N>* dst, std::size_t count) {
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
        for (std::size_t i = 0; i < count; ++i) {
            dst[i][0] = static_cast<Scalar>(src[i].X);
            dst[i][1] = static_cast<Scalar>(src[i].Y);
            dst[i][2] = static_cast<Scalar>(src[i].Z);
        }
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            for (int d = 0; d < N; ++d) {
                dst[i][d] = static_cast<Scalar>(src[i][d]);
            }
        }
    }
}

// ============================================================================
//  Bounding box converter: FBox (or FBox2D) <-> OrthoTree AABB
// ============================================================================
template<typename UnrealBox, typename Scalar = float, int N = 3>
struct BoxConverter {
    using ortho_aabb = Math::AxisAlignedBox<Scalar, N>;

    static ortho_aabb to_ortho(const UnrealBox& box) {
        using ortho_point = Math::Vector<Scalar, N>;
        ortho_point minP, maxP;
        if constexpr (N == 2) {
            minP[0] = static_cast<Scalar>(box.Min.X);
            minP[1] = static_cast<Scalar>(box.Min.Y);
            maxP[0] = static_cast<Scalar>(box.Max.X);
            maxP[1] = static_cast<Scalar>(box.Max.Y);
        } else {
            minP[0] = static_cast<Scalar>(box.Min.X);
            minP[1] = static_cast<Scalar>(box.Min.Y);
            minP[2] = static_cast<Scalar>(box.Min.Z);
            maxP[0] = static_cast<Scalar>(box.Max.X);
            maxP[1] = static_cast<Scalar>(box.Max.Y);
            maxP[2] = static_cast<Scalar>(box.Max.Z);
        }
        return ortho_aabb(minP, maxP);
    }

    static UnrealBox from_ortho(const ortho_aabb& aabb) {
        UnrealBox result;
        if constexpr (N == 2) {
            result.Min = Unreal::UnrealVec(aabb.min()[0], aabb.min()[1]);
            result.Max = Unreal::UnrealVec(aabb.max()[0], aabb.max()[1]);
        } else {
            result.Min = Unreal::UnrealVec(aabb.min()[0], aabb.min()[1], aabb.min()[2]);
            result.Max = Unreal::UnrealVec(aabb.max()[0], aabb.max()[1], aabb.max()[2]);
        }
        return result;
    }
};

// ============================================================================
//  Transform converter: FTransform (Unreal) -> OrthoTree AffineTransform
//  Uses: location, rotation (quaternion), scale
// ============================================================================
template<typename UnrealTransform, typename Scalar = float, int N = 3>
struct TransformConverter {
    using ortho_transform = Math::AffineTransform<Scalar, N>;

    static ortho_transform to_ortho(const UnrealTransform& tf) {
        // Extract translation
        Math::Vector<Scalar, N> trans(
            static_cast<Scalar>(tf.GetLocation().X),
            static_cast<Scalar>(tf.GetLocation().Y),
            static_cast<Scalar>(tf.GetLocation().Z)
        );
        // Extract rotation as quaternion
        FQuat rot = tf.GetRotation();
        Math::Quaternion<Scalar> q(
            static_cast<Scalar>(rot.W),
            static_cast<Scalar>(rot.X),
            static_cast<Scalar>(rot.Y),
            static_cast<Scalar>(rot.Z)
        );
        // Extract scale
        FVector scaleVec = tf.GetScale3D();
        Math::Vector<Scalar, N> scale(
            static_cast<Scalar>(scaleVec.X),
            static_cast<Scalar>(scaleVec.Y),
            static_cast<Scalar>(scaleVec.Z)
        );
        // Build matrix: rotation * scale (assuming linear part = R * diag(scale))
        Math::Matrix<Scalar, N> mat = Math::Matrix<Scalar, N>::rotation(q);
        // Apply scaling: for each axis, multiply column by scale factor
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                mat(i, j) *= scale[i];
            }
        }
        return ortho_transform(mat, trans);
    }

    static UnrealTransform from_ortho(const ortho_transform& ot) {
        // Decompose matrix into rotation and scale (simplified: assume no shear)
        Math::Matrix<Scalar, N> mat = ot.matrix();
        // Compute scale as length of each column
        UnrealVec scaleVec;
        for (int i = 0; i < N; ++i) {
            Scalar len = Scalar(0);
            for (int j = 0; j < N; ++j) len += mat(j, i) * mat(j, i);
            len = std::sqrt(len);
            scaleVec[i] = static_cast<Scalar>(len);
            for (int j = 0; j < N; ++j) mat(j, i) /= len;
        }
        // Now mat is orthogonal (rotation)
        Math::Quaternion<Scalar> q = Math::Quaternion<Scalar>::fromMatrix(mat);
        // Unreal quaternion is (X,Y,Z,W) where W is real
        FQuat rot(static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()), static_cast<float>(q.w()));
        FVector trans(static_cast<float>(ot.translation()[0]),
                      static_cast<float>(ot.translation()[1]),
                      static_cast<float>(ot.translation()[2]));
        FVector scale(static_cast<float>(scaleVec[0]),
                      static_cast<float>(scaleVec[1]),
                      static_cast<float>(scaleVec[2]));
        return UnrealTransform(rot, trans, scale);
    }
};

// ============================================================================
//  Ray converter: Unreal's FRay (if exists) or using FVector pair
//  In Unreal, a ray can be represented as FVector origin + direction.
//  We'll use a simple struct if FRay not available.
// ============================================================================
struct UnrealRay {
    FVector Origin;
    FVector Direction;
};

template<typename UnrealRayType = UnrealRay>
struct RayConverter {
    using Scalar = float;
    static constexpr int N = 3;
    using ortho_ray = Math::Ray<Scalar, N>;

    static ortho_ray to_ortho(const UnrealRayType& ray) {
        auto origin = PointConverter<FVector>::to_ortho(ray.Origin);
        auto dir = PointConverter<FVector>::to_ortho(ray.Direction);
        return ortho_ray(origin, dir.normalized());
    }

    static UnrealRayType from_ortho(const ortho_ray& ray) {
        UnrealRayType result;
        result.Origin = PointConverter<FVector>::from_ortho(ray.origin());
        result.Direction = PointConverter<FVector>::from_ortho(ray.direction());
        return result;
    }
};

// ============================================================================
//  Dynamic environment controller for Unreal adapters
// ============================================================================
class UnrealAdapterEnvironment {
public:
    using Scalar = float;   // Unreal primarily uses float

    UnrealAdapterEnvironment() noexcept
        : m_useSIMD(true)
        , m_autoConvert(true)
        , m_useHighPrecision(false)
        , m_hasTransform(false) {}

    void setUseSIMD(bool use) noexcept { m_useSIMD = use; }
    bool useSIMD() const noexcept { return m_useSIMD; }

    void setAutoConvert(bool autoCvt) noexcept { m_autoConvert = autoCvt; }
    bool autoConvert() const noexcept { return m_autoConvert; }

    void setUseHighPrecision(bool high) noexcept { m_useHighPrecision = high; }
    bool useHighPrecision() const noexcept { return m_useHighPrecision; }

    // Apply a world transform (FTransform) to all points
    void setTransform(const FTransform& tf) { m_transform = tf; m_hasTransform = true; }
    void clearTransform() { m_hasTransform = false; }

    template<typename UnrealPoint>
    UnrealPoint transformPoint(const UnrealPoint& p) const {
        if (!m_hasTransform) return p;
        return m_transform.TransformPosition(p);
    }

private:
    bool m_useSIMD;
    bool m_autoConvert;
    bool m_useHighPrecision;
    FTransform m_transform;
    bool m_hasTransform;
};

// ----------------------------------------------------------------------------
//  Singleton access (thread‑local or global)
// ----------------------------------------------------------------------------
inline UnrealAdapterEnvironment& unreal_adapter_env() {
    static UnrealAdapterEnvironment env;
    return env;
}

// ============================================================================
//  Entity adapter for Unreal types (for octree integration)
// ============================================================================
template<typename UnrealGeometry>
struct EntityAdapter {
    using Scalar = float;
    static constexpr int dimension = 3;
    using ortho_aabb = Math::AxisAlignedBox<Scalar, dimension>;
    using ortho_point = Math::Vector<Scalar, dimension>;

    static ortho_aabb getBounds(const UnrealGeometry& geom) {
        if constexpr (std::is_same_v<UnrealGeometry, FVector> ||
                      std::is_same_v<UnrealGeometry, FVector3d>) {
            ortho_point p = PointConverter<UnrealGeometry>::to_ortho(geom);
            return ortho_aabb(p, p);
        } else if constexpr (std::is_same_v<UnrealGeometry, FBox>) {
            return BoxConverter<FBox>::to_ortho(geom);
        } else if constexpr (std::is_same_v<UnrealGeometry, FBox2D>) {
            // For 2D box, we need to treat as 2D AABB
            using ortho_aabb2d = Math::AxisAlignedBox<Scalar, 2>;
            auto minP = Math::Vector<Scalar,2>(
                static_cast<Scalar>(geom.Min.X),
                static_cast<Scalar>(geom.Min.Y));
            auto maxP = Math::Vector<Scalar,2>(
                static_cast<Scalar>(geom.Max.X),
                static_cast<Scalar>(geom.Max.Y));
            // But we need 3D AABB? For consistency, extend to 3D with Z=0
            ortho_aabb result;
            result.setMin(ortho_point(minP[0], minP[1], 0));
            result.setMax(ortho_point(maxP[0], maxP[1], 0));
            return result;
        } else {
            // For custom types, assume they have GetBounds() method
            FBox bounds = geom.GetBounds();
            return BoxConverter<FBox>::to_ortho(bounds);
        }
    }
};

} // namespace Unreal
} // namespace Adapters
} // namespace OrthoTree

#endif // ORTHOTREE_ADAPTERS_UNREAL_H_INCLUDED