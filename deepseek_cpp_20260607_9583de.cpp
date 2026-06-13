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

#ifndef ORTHOTREE_ADAPTERS_EIGEN_H_INCLUDED
#define ORTHOTREE_ADAPTERS_EIGEN_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../core/math/transform.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"

#if defined(ORTHOTREE_EIGEN_SUPPORT) || defined(EIGEN_WORLD_VERSION)
#include <Eigen/Core>
#include <Eigen/Geometry>
#else
#error "Eigen support requires Eigen headers. Define ORTHOTREE_EIGEN_SUPPORT or include Eigen before orthotree/adapters/eigen.h"
#endif

namespace OrthoTree {
namespace Adapters {
namespace Eigen {

// ============================================================================
//  Scalar type extraction (Eigen::Matrix scalar)
// ============================================================================
template<typename Derived>
struct eigen_traits {
    using Scalar = typename Derived::Scalar;
    static constexpr int RowsAtCompileTime = Derived::RowsAtCompileTime;
    static constexpr int ColsAtCompileTime = Derived::ColsAtCompileTime;
    static constexpr int Dimension = (RowsAtCompileTime == 1 && ColsAtCompileTime == 2) ? 2 :
                                     (RowsAtCompileTime == 1 && ColsAtCompileTime == 3) ? 3 :
                                     (RowsAtCompileTime == 2 && ColsAtCompileTime == 1) ? 2 :
                                     (RowsAtCompileTime == 3 && ColsAtCompileTime == 1) ? 3 : 0;
};

// ============================================================================
//  Point converter: Eigen::Vector2d/Vector3f etc. <-> OrthoTree Vector
// ============================================================================
template<typename EigenVector, typename T = typename eigen_traits<EigenVector>::Scalar,
         int N = eigen_traits<EigenVector>::Dimension>
struct PointConverter {
    static_assert(N == 2 || N == 3, "Only 2D or 3D vectors supported");
    using ortho_vector = Math::Vector<T, N>;

    static ortho_vector to_ortho(const EigenVector& ev) {
        ortho_vector v;
        for (int i = 0; i < N; ++i) {
            v[i] = static_cast<T>(ev[i]);
        }
        return v;
    }

    static EigenVector from_ortho(const ortho_vector& v) {
        if constexpr (N == 2) {
            return EigenVector(static_cast<T>(v[0]), static_cast<T>(v[1]));
        } else {
            return EigenVector(static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2]));
        }
    }
};

// ----------------------------------------------------------------------------
//  SIMD batch conversion (4 vectors at a time using Eigen's packet ops if possible)
// ----------------------------------------------------------------------------
template<typename EigenVector, typename T, int N>
void batch_to_ortho(const EigenVector* src, Math::Vector<T, N>* dst, std::size_t count) {
    constexpr int PacketSize = ORTHOTREE_SIMD_LEVEL >= 128 ? 4 : 1;
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3 && PacketSize == 4) {
        // We can use Eigen's internal packet operations, but for simplicity,
        // we use a loop with unrolling. In a real implementation, we would
        // use Eigen::Matrix<T,3,4> to load 4 vectors at once.
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
//  Bounding box converter: Eigen::AlignedBox2/3 -> OrthoTree AABB
// ============================================================================
template<typename EigenAlignedBox, typename T = typename EigenAlignedBox::Scalar,
         int N = (EigenAlignedBox::Dim == 2) ? 2 : 3>
struct BoxConverter {
    using ortho_aabb = Math::AxisAlignedBox<T, N>;

    static ortho_aabb to_ortho(const EigenAlignedBox& box) {
        using ortho_point = Math::Vector<T, N>;
        ortho_point minP, maxP;
        for (int i = 0; i < N; ++i) {
            minP[i] = static_cast<T>(box.min()[i]);
            maxP[i] = static_cast<T>(box.max()[i]);
        }
        return ortho_aabb(minP, maxP);
    }

    static EigenAlignedBox from_ortho(const ortho_aabb& aabb) {
        if constexpr (N == 2) {
            return EigenAlignedBox(
                Eigen::Vector2<T>(aabb.min()[0], aabb.min()[1]),
                Eigen::Vector2<T>(aabb.max()[0], aabb.max()[1])
            );
        } else {
            return EigenAlignedBox(
                Eigen::Vector3<T>(aabb.min()[0], aabb.min()[1], aabb.min()[2]),
                Eigen::Vector3<T>(aabb.max()[0], aabb.max()[1], aabb.max()[2])
            );
        }
    }
};

// ============================================================================
//  Transform converter: Eigen::Transform (affine) -> OrthoTree AffineTransform
// ============================================================================
template<typename EigenTransform, typename T = typename EigenTransform::Scalar,
         int N = EigenTransform::Dim>
struct TransformConverter {
    using ortho_transform = Math::AffineTransform<T, N>;

    static ortho_transform to_ortho(const EigenTransform& et) {
        // Extract matrix (linear part) and translation
        Math::Matrix<T, N> mat;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                mat(i, j) = static_cast<T>(et.matrix()(i, j));
            }
        }
        Math::Vector<T, N> trans;
        for (int i = 0; i < N; ++i) {
            trans[i] = static_cast<T>(et.translation()[i]);
        }
        return ortho_transform(mat, trans);
    }

    static EigenTransform from_ortho(const ortho_transform& ot) {
        Eigen::Matrix<T, N, N> mat;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                mat(i, j) = ot.matrix()(i, j);
            }
        }
        Eigen::Matrix<T, N, 1> trans;
        for (int i = 0; i < N; ++i) {
            trans[i] = ot.translation()[i];
        }
        return EigenTransform(mat, trans);
    }
};

// ============================================================================
//  Ray converter: Eigen::ParametrizedLine -> OrthoTree Ray
// ============================================================================
template<typename EigenLine, typename T = typename EigenLine::Scalar,
         int N = EigenLine::DimAtCompileTime>
struct RayConverter {
    using ortho_ray = Math::Ray<T, N>;

    static ortho_ray to_ortho(const EigenLine& line) {
        auto origin = PointConverter<Eigen::Matrix<T, N, 1>, T, N>::to_ortho(line.origin());
        auto dir = PointConverter<Eigen::Matrix<T, N, 1>, T, N>::to_ortho(line.direction());
        return ortho_ray(origin, dir);
    }

    static EigenLine from_ortho(const ortho_ray& ray) {
        auto origin = PointConverter<Eigen::Matrix<T, N, 1>, T, N>::from_ortho(ray.origin());
        auto dir = PointConverter<Eigen::Matrix<T, N, 1>, T, N>::from_ortho(ray.direction());
        return EigenLine(origin, dir);
    }
};

// ============================================================================
//  Dynamic environment controller for Eigen adapters
// ============================================================================
class EigenAdapterEnvironment {
public:
    using Scalar = double;

    EigenAdapterEnvironment() noexcept
        : m_useSIMD(true)
        , m_autoConvert(true)
        , m_useAlignedVectors(true)
        , m_preferDouble(true) {}

    void setUseSIMD(bool use) noexcept { m_useSIMD = use; }
    bool useSIMD() const noexcept { return m_useSIMD; }

    void setAutoConvert(bool autoCvt) noexcept { m_autoConvert = autoCvt; }
    bool autoConvert() const noexcept { return m_autoConvert; }

    void setUseAlignedVectors(bool aligned) noexcept { m_useAlignedVectors = aligned; }
    bool useAlignedVectors() const noexcept { return m_useAlignedVectors; }

    void setPreferDouble(bool prefer) noexcept { m_preferDouble = prefer; }
    bool preferDouble() const noexcept { return m_preferDouble; }

    // Apply a linear transformation to all points (e.g., scaling, rotation)
    void setTransformationMatrix(const ::Eigen::Matrix<Scalar, 3, 3>& mat) {
        m_transform = mat;
        m_hasTransform = true;
    }
    void clearTransformation() { m_hasTransform = false; }

    template<typename EigenVector>
    EigenVector transformPoint(const EigenVector& p) const {
        if (!m_hasTransform) return p;
        if constexpr (EigenVector::RowsAtCompileTime == 2) {
            ::Eigen::Vector3<Scalar> p3(p[0], p[1], 0);
            auto res3 = m_transform * p3;
            return EigenVector(res3[0], res3[1]);
        } else {
            return (m_transform * p).eval();
        }
    }

private:
    bool m_useSIMD;
    bool m_autoConvert;
    bool m_useAlignedVectors;
    bool m_preferDouble;
    ::Eigen::Matrix<Scalar, 3, 3> m_transform;
    bool m_hasTransform = false;
};

// ----------------------------------------------------------------------------
//  Singleton access (thread‑local or global)
// ----------------------------------------------------------------------------
inline EigenAdapterEnvironment& eigen_adapter_env() {
    static EigenAdapterEnvironment env;
    return env;
}

// ============================================================================
//  Entity adapter for Eigen types (for octree integration)
// ============================================================================
template<typename EigenGeometry>
struct EntityAdapter {
    using Scalar = typename eigen_traits<EigenGeometry>::Scalar;
    static constexpr int dimension = eigen_traits<EigenGeometry>::Dimension;
    using ortho_aabb = Math::AxisAlignedBox<Scalar, dimension>;
    using ortho_point = Math::Vector<Scalar, dimension>;

    static ortho_aabb getBounds(const EigenGeometry& geom) {
        if constexpr (std::is_same_v<EigenGeometry, ::Eigen::Vector2<Scalar>> ||
                      std::is_same_v<EigenGeometry, ::Eigen::Vector3<Scalar>>) {
            ortho_point p = PointConverter<EigenGeometry>::to_ortho(geom);
            return ortho_aabb(p, p);
        } else if constexpr (std::is_same_v<EigenGeometry, ::Eigen::AlignedBox<Scalar, 2>> ||
                              std::is_same_v<EigenGeometry, ::Eigen::AlignedBox<Scalar, 3>>) {
            return BoxConverter<EigenGeometry>::to_ortho(geom);
        } else {
            // For other types, compute bounding box using Eigen's internal method if available
            // As fallback, treat as point at origin
            return ortho_aabb(ortho_point(0), ortho_point(0));
        }
    }
};

} // namespace Eigen
} // namespace Adapters
} // namespace OrthoTree

#endif // ORTHOTREE_ADAPTERS_EIGEN_H_INCLUDED