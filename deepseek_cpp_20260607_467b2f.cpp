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

#ifndef ORTHOTREE_ADAPTERS_GLM_H_INCLUDED
#define ORTHOTREE_ADAPTERS_GLM_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../core/math/transform.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"

#if defined(ORTHOTREE_GLM_SUPPORT) || defined(GLM_VERSION)
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#else
#error "GLM support requires GLM headers. Define ORTHOTREE_GLM_SUPPORT or include GLM before orthotree/adapters/glm.h"
#endif

namespace OrthoTree {
namespace Adapters {
namespace GLM {

// ============================================================================
//  Type traits: extract scalar type and dimension from GLM vectors
// ============================================================================
template<typename T>
struct glm_traits;

template<glm::length_t L, typename T, glm::qualifier Q>
struct glm_traits<glm::vec<L, T, Q>> {
    using Scalar = T;
    static constexpr int dimension = L;
    static constexpr glm::qualifier qualifier = Q;
};

// ============================================================================
//  Point converter: glm::vec2/3/4 <-> OrthoTree Vector
// ============================================================================
template<typename GlmVec, typename T = typename glm_traits<GlmVec>::Scalar,
         int N = glm_traits<GlmVec>::dimension>
struct PointConverter {
    static_assert(N == 2 || N == 3, "Only 2D or 3D vectors supported");
    using ortho_vector = Math::Vector<T, N>;

    static ortho_vector to_ortho(const GlmVec& v) {
        ortho_vector result;
        for (int i = 0; i < N; ++i) result[i] = static_cast<T>(v[i]);
        return result;
    }

    static GlmVec from_ortho(const ortho_vector& v) {
        if constexpr (N == 2) return GlmVec(static_cast<T>(v[0]), static_cast<T>(v[1]));
        else return GlmVec(static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2]));
    }
};

// ----------------------------------------------------------------------------
//  SIMD batch conversion (4 vectors at a time)
// ----------------------------------------------------------------------------
template<typename GlmVec, typename T, int N>
void batch_to_ortho(const GlmVec* src, Math::Vector<T, N>* dst, std::size_t count) {
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
        for (std::size_t i = 0; i < count; ++i) {
            dst[i][0] = static_cast<T>(src[i].x);
            dst[i][1] = static_cast<T>(src[i].y);
            dst[i][2] = static_cast<T>(src[i].z);
        }
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            for (int d = 0; d < N; ++d) dst[i][d] = static_cast<T>(src[i][d]);
        }
    }
}

// ============================================================================
//  Bounding box converter: GLM doesn't have a standard AABB type;
//  we assume the user provides a pair of points (min, max) or a custom struct.
//  For simplicity, we handle std::pair<glm::vec, glm::vec>.
// ============================================================================
template<typename GlmVec>
struct BoxConverter {
    using ortho_aabb = Math::AxisAlignedBox<typename glm_traits<GlmVec>::Scalar,
                                            glm_traits<GlmVec>::dimension>;

    static ortho_aabb to_ortho(const std::pair<GlmVec, GlmVec>& pair) {
        auto minVec = PointConverter<GlmVec>::to_ortho(pair.first);
        auto maxVec = PointConverter<GlmVec>::to_ortho(pair.second);
        return ortho_aabb(minVec, maxVec);
    }

    static std::pair<GlmVec, GlmVec> from_ortho(const ortho_aabb& aabb) {
        auto minVec = PointConverter<GlmVec>::from_ortho(aabb.min());
        auto maxVec = PointConverter<GlmVec>::from_ortho(aabb.max());
        return {minVec, maxVec};
    }
};

// ============================================================================
//  Ray converter: glm::ray? GLM doesn't have a built‑in ray, but we can
//  use std::pair<origin, direction> or a custom struct.
//  For demonstration, we use std::pair<glm::vec, glm::vec>.
// ============================================================================
template<typename GlmVec>
struct RayConverter {
    using ortho_ray = Math::Ray<typename glm_traits<GlmVec>::Scalar,
                                glm_traits<GlmVec>::dimension>;

    static ortho_ray to_ortho(const std::pair<GlmVec, GlmVec>& rayPair) {
        auto origin = PointConverter<GlmVec>::to_ortho(rayPair.first);
        auto dir = PointConverter<GlmVec>::to_ortho(rayPair.second);
        return ortho_ray(origin, dir.normalized());
    }

    static std::pair<GlmVec, GlmVec> from_ortho(const ortho_ray& ray) {
        auto origin = PointConverter<GlmVec>::from_ortho(ray.origin());
        auto dir = PointConverter<GlmVec>::from_ortho(ray.direction());
        return {origin, dir};
    }
};

// ============================================================================
//  Transform converter: glm::mat4 -> OrthoTree AffineTransform (3D only)
// ============================================================================
template<typename T, glm::qualifier Q = glm::defaultp>
struct TransformConverter {
    using ortho_transform = Math::AffineTransform<T, 3>;

    static ortho_transform to_ortho(const glm::mat<4, 4, T, Q>& mat) {
        Math::Matrix<T, 3> rot;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                rot(i, j) = static_cast<T>(mat[i][j]);
        Math::Vector<T, 3> trans(mat[3][0], mat[3][1], mat[3][2]);
        return ortho_transform(rot, trans);
    }

    static glm::mat<4, 4, T, Q> from_ortho(const ortho_transform& ot) {
        glm::mat<4, 4, T, Q> mat(1.0f);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                mat[i][j] = ot.matrix()(i, j);
        mat[3][0] = ot.translation()[0];
        mat[3][1] = ot.translation()[1];
        mat[3][2] = ot.translation()[2];
        return mat;
    }
};

// ============================================================================
//  Dynamic environment controller for GLM adapters
// ============================================================================
class GLMAdapterEnvironment {
public:
    using Scalar = float;   // GLM default is float, but can be double

    GLMAdapterEnvironment() noexcept
        : m_useSIMD(true)
        , m_autoConvert(true)
        , m_useHighPrecision(false) {}

    void setUseSIMD(bool use) noexcept { m_useSIMD = use; }
    bool useSIMD() const noexcept { return m_useSIMD; }

    void setAutoConvert(bool autoCvt) noexcept { m_autoConvert = autoCvt; }
    bool autoConvert() const noexcept { return m_autoConvert; }

    void setUseHighPrecision(bool high) noexcept { m_useHighPrecision = high; }
    bool useHighPrecision() const noexcept { return m_useHighPrecision; }

    // Apply a 4x4 transformation matrix to all points (e.g., world transform)
    void setTransformMatrix(const glm::mat4& mat) { m_transform = mat; m_hasTransform = true; }
    void clearTransform() { m_hasTransform = false; }

    template<typename GlmVec>
    GlmVec transformPoint(const GlmVec& p) const {
        if (!m_hasTransform) return p;
        glm::vec4 p4(p.x, p.y, (GlmVec::length() == 3) ? p.z : 0.0f, 1.0f);
        auto res4 = m_transform * p4;
        if constexpr (GlmVec::length() == 2) {
            return GlmVec(res4.x, res4.y);
        } else {
            return GlmVec(res4.x, res4.y, res4.z);
        }
    }

private:
    bool m_useSIMD;
    bool m_autoConvert;
    bool m_useHighPrecision;
    glm::mat4 m_transform;
    bool m_hasTransform = false;
};

// ----------------------------------------------------------------------------
//  Singleton access
// ----------------------------------------------------------------------------
inline GLMAdapterEnvironment& glm_adapter_env() {
    static GLMAdapterEnvironment env;
    return env;
}

// ============================================================================
//  Entity adapter for GLM types (for octree integration)
// ============================================================================
template<typename GlmGeometry>
struct EntityAdapter {
    using Scalar = typename glm_traits<GlmGeometry>::Scalar;
    static constexpr int dimension = glm_traits<GlmGeometry>::dimension;
    using ortho_aabb = Math::AxisAlignedBox<Scalar, dimension>;
    using ortho_point = Math::Vector<Scalar, dimension>;

    static ortho_aabb getBounds(const GlmGeometry& geom) {
        // If it's a vector (point), treat as zero‑sized AABB
        if constexpr (std::is_same_v<GlmGeometry, glm::vec<dimension, Scalar>>) {
            ortho_point p = PointConverter<GlmGeometry>::to_ortho(geom);
            return ortho_aabb(p, p);
        } else {
            // For other types (pair of points), assume it's a box
            if constexpr (dimension == 2) {
                auto pair = static_cast<std::pair<glm::vec<2, Scalar>, glm::vec<2, Scalar>>>(geom);
                return BoxConverter<glm::vec<2, Scalar>>::to_ortho(pair);
            } else {
                auto pair = static_cast<std::pair<glm::vec<3, Scalar>, glm::vec<3, Scalar>>>(geom);
                return BoxConverter<glm::vec<3, Scalar>>::to_ortho(pair);
            }
        }
    }
};

} // namespace GLM
} // namespace Adapters
} // namespace OrthoTree

#endif // ORTHOTREE_ADAPTERS_GLM_H_INCLUDED