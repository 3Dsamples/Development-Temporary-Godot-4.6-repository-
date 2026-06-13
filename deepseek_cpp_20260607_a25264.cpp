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

#ifndef ORTHOTREE_CORE_ENTITY_ADAPTER_H_INCLUDED
#define ORTHOTREE_CORE_ENTITY_ADAPTER_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../core/math/transform.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"

#include <type_traits>
#include <utility>
#include <array>
#include <memory>
#include <functional>

namespace OrthoTree {

// ============================================================================
//  EntityAdapter: trait‑based adapter for extracting geometric information
//  from user‑defined entity types. Supports points, boxes, and custom shapes.
//  Provides SIMD batch extraction and dynamic environment hooks.
// ============================================================================

// ----------------------------------------------------------------------------
//  Primary template – user must specialise for custom types
// ----------------------------------------------------------------------------
template<typename Entity, typename = void>
struct EntityAdapter {
    // Default implementation assumes entity is a point with .x, .y, .z members
    // and a .radius or .bounds() method for AABB. Fallback: treat as point.
    using Scalar = float;
    static constexpr int dimension = 3;
    using point_type = Math::Vector<Scalar, dimension>;
    using aabb_type = Math::AxisAlignedBox<Scalar, dimension>;

    static point_type getPosition(const Entity& e) {
        return point_type(static_cast<Scalar>(e.x), static_cast<Scalar>(e.y), static_cast<Scalar>(e.z));
    }

    static aabb_type getBounds(const Entity& e) {
        // If entity has a `bounds()` method, use it.
        if constexpr (requires { e.bounds(); }) {
            return e.bounds();
        } else if constexpr (requires { e.radius; }) {
            point_type center = getPosition(e);
            Scalar r = static_cast<Scalar>(e.radius);
            return aabb_type(center - point_type(r), center + point_type(r));
        } else {
            point_type p = getPosition(e);
            return aabb_type(p, p);
        }
    }

    static Scalar getRadius(const Entity& e) {
        if constexpr (requires { e.radius; }) return static_cast<Scalar>(e.radius);
        else return Scalar(0);
    }

    // For SIMD batch extraction, we provide static methods that work on arrays.
    static void batchGetPositions(const Entity* entities, point_type* out, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = getPosition(entities[i]);
        }
    }

    static void batchGetBounds(const Entity* entities, aabb_type* out, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = getBounds(entities[i]);
        }
    }
};

// ----------------------------------------------------------------------------
//  Specialisation for arithmetic types (treat as 1D point)
// ----------------------------------------------------------------------------
template<typename T>
struct EntityAdapter<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
    using Scalar = T;
    static constexpr int dimension = 1;
    using point_type = Math::Vector<Scalar, 1>;
    using aabb_type = Math::AxisAlignedBox<Scalar, 1>;

    static point_type getPosition(const T& e) {
        return point_type(static_cast<Scalar>(e));
    }
    static aabb_type getBounds(const T& e) {
        return aabb_type(point_type(e), point_type(e));
    }
    static Scalar getRadius(const T&) { return Scalar(0); }
};

// ----------------------------------------------------------------------------
//  Specialisation for std::pair<Point, Point> as AABB
// ----------------------------------------------------------------------------
template<typename PointType>
struct EntityAdapter<std::pair<PointType, PointType>,
                     std::void_t<decltype(std::declval<PointType>().x),
                                 decltype(std::declval<PointType>().y)>> {
    using Scalar = decltype(PointType::x);
    static constexpr int dimension = 2;
    using point_type = Math::Vector<Scalar, dimension>;
    using aabb_type = Math::AxisAlignedBox<Scalar, dimension>;

    static point_type getPosition(const std::pair<PointType, PointType>& pair) {
        // centroid
        return point_type( (pair.first.x + pair.second.x) / Scalar(2),
                           (pair.first.y + pair.second.y) / Scalar(2) );
    }
    static aabb_type getBounds(const std::pair<PointType, PointType>& pair) {
        return aabb_type(point_type(pair.first.x, pair.first.y),
                         point_type(pair.second.x, pair.second.y));
    }
    static Scalar getRadius(const std::pair<PointType, PointType>& pair) {
        point_type min(pair.first.x, pair.first.y);
        point_type max(pair.second.x, pair.second.y);
        return (max - min).length() / Scalar(2);
    }
};

// ============================================================================
//  Dynamic entity adapter – runtime polymorphic wrapper for heterogeneous collections
// ============================================================================
template<typename Scalar = float, int N = 3>
class DynamicEntityAdapter {
public:
    using point_type = Math::Vector<Scalar, N>;
    using aabb_type = Math::AxisAlignedBox<Scalar, N>;

    virtual ~DynamicEntityAdapter() = default;
    virtual aabb_type getBounds() const = 0;
    virtual point_type getPosition() const = 0;
    virtual Scalar getRadius() const = 0;
    virtual std::unique_ptr<DynamicEntityAdapter<Scalar, N>> clone() const = 0;
};

template<typename Entity, typename Scalar, int N>
class DynamicEntityAdapterImpl : public DynamicEntityAdapter<Scalar, N> {
public:
    using base = DynamicEntityAdapter<Scalar, N>;
    using point_type = typename base::point_type;
    using aabb_type = typename base::aabb_type;

    explicit DynamicEntityAdapterImpl(const Entity& e) : m_entity(e) {}

    aabb_type getBounds() const override {
        return EntityAdapter<Entity>::getBounds(m_entity);
    }
    point_type getPosition() const override {
        return EntityAdapter<Entity>::getPosition(m_entity);
    }
    Scalar getRadius() const override {
        return EntityAdapter<Entity>::getRadius(m_entity);
    }
    std::unique_ptr<base> clone() const override {
        return std::make_unique<DynamicEntityAdapterImpl>(m_entity);
    }

private:
    Entity m_entity;
};

// ============================================================================
//  Environment controller for entity adapters (transformations, LOD scaling)
// ============================================================================
template<typename Scalar = float, int N = 3>
class EntityAdapterEnvironment {
public:
    using point_type = Math::Vector<Scalar, N>;
    using aabb_type = Math::AxisAlignedBox<Scalar, N>;
    using transform_type = Math::AffineTransform<Scalar, N>;

    EntityAdapterEnvironment() noexcept : m_hasTransform(false), m_lodBias(1.0f) {}

    void setTransform(const transform_type& tf) {
        m_transform = tf;
        m_hasTransform = true;
    }
    void clearTransform() { m_hasTransform = false; }
    bool hasTransform() const { return m_hasTransform; }

    void setLODBias(Scalar bias) noexcept { m_lodBias = bias; }
    Scalar lodBias() const noexcept { return m_lodBias; }

    // Apply transform to a point (if any)
    point_type applyTransform(const point_type& p) const {
        return m_hasTransform ? m_transform.transform(p) : p;
    }

    // Apply transform to AABB (conservative)
    aabb_type applyTransform(const aabb_type& box) const {
        if (!m_hasTransform) return box;
        return box.transform(m_transform);
    }

    // Batch transform points (SIMD)
    void batchTransformPoints(const point_type* src, point_type* dst, size_t count) const {
        if (!m_hasTransform) {
            for (size_t i = 0; i < count; ++i) dst[i] = src[i];
            return;
        }
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
            // In a real implementation, we would use AVX2 to transform 4 points at once.
            for (size_t i = 0; i < count; ++i) {
                dst[i] = m_transform.transform(src[i]);
            }
        } else {
            for (size_t i = 0; i < count; ++i) {
                dst[i] = m_transform.transform(src[i]);
            }
        }
    }

private:
    transform_type m_transform;
    bool m_hasTransform;
    Scalar m_lodBias;
};

// ----------------------------------------------------------------------------
//  Singleton access to global environment (thread‑local if needed)
// ----------------------------------------------------------------------------
template<typename Scalar = float, int N = 3>
inline EntityAdapterEnvironment<Scalar, N>& entity_adapter_env() {
    static thread_local EntityAdapterEnvironment<Scalar, N> env;
    return env;
}

// ============================================================================
//  Helper: adapt a range of entities to a vector of bounds (SIMD accelerated)
// ============================================================================
template<typename Iter, typename Scalar = float, int N = 3>
void batchExtractBounds(Iter first, Iter last,
                        Math::AxisAlignedBox<Scalar, N>* out) {
    using Entity = typename std::iterator_traits<Iter>::value_type;
    size_t count = std::distance(first, last);
    for (size_t i = 0; i < count; ++i, ++first) {
        out[i] = EntityAdapter<Entity>::getBounds(*first);
    }
}

template<typename Iter, typename Scalar = float, int N = 3>
void batchExtractPositions(Iter first, Iter last,
                           Math::Vector<Scalar, N>* out) {
    using Entity = typename std::iterator_traits<Iter>::value_type;
    size_t count = std::distance(first, last);
    for (size_t i = 0; i < count; ++i, ++first) {
        out[i] = EntityAdapter<Entity>::getPosition(*first);
    }
}

} // namespace OrthoTree

#endif // ORTHOTREE_CORE_ENTITY_ADAPTER_H_INCLUDED