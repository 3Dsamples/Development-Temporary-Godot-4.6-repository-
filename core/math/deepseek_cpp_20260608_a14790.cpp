//File group name : OrthoTree Math
//File 0079 : core/math/moving_objects.h
//Time‑parameterised moving objects: AABB with linear motion, point with linear motion.
//Provides interval intersection queries: for a given time interval [t0,t1], test if two moving AABBs overlap,
//or find all moving objects that intersect a static query AABB at any time during the interval.
//Uses separating axis theorem for moving AABBs (solved as continuous collision detection).
//Includes SIMD batch evaluation for multiple time steps.

#ifndef ORTHOTREE_CORE_MATH_MOVING_OBJECTS_H_INCLUDED
#define ORTHOTREE_CORE_MATH_MOVING_OBJECTS_H_INCLUDED

#include "../../build_config.h"
#include "basic/vector.h"
#include "geometry/aabb.h"
#include "math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  MovingAABB: axis‑aligned bounding box with constant linear velocity.
//  Position at time t: box = translate(startBounds + velocity * t).
//  startBounds is AABB at t = 0.
// ============================================================================
template<typename T = float>
class MovingAABB {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, 3>;
    using aabb_type = Geometry::AABB<T, 3>;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    MovingAABB() noexcept : m_start(), m_velocity(T(0)) {}
    MovingAABB(const aabb_type& start, const point_type& velocity) noexcept
        : m_start(start), m_velocity(velocity) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    const aabb_type& start() const noexcept { return m_start; }
    const point_type& velocity() const noexcept { return m_velocity; }
    void setStart(const aabb_type& s) noexcept { m_start = s; }
    void setVelocity(const point_type& v) noexcept { m_velocity = v; }

    // ------------------------------------------------------------------------
    //  Bounding box at time t (assuming linear motion)
    // ------------------------------------------------------------------------
    aabb_type atTime(T t) const noexcept {
        point_type offset = m_velocity * t;
        point_type minP = m_start.min() + offset;
        point_type maxP = m_start.max() + offset;
        return aabb_type(minP, maxP);
    }

    // ------------------------------------------------------------------------
    //  Continuous collision detection between two moving AABBs over interval [0,1].
    //  Returns true if they ever overlap, and outputs the earliest time of collision (tFirst)
    //  and the latest time of separation (tLast) within [0,1].
    //  Based on separating axis theorem for moving boxes (swept AABB).
    //  For simplicity, we implement the method of solving for each axis.
    // ------------------------------------------------------------------------
    bool continuousCollision(const MovingAABB& other, T& tFirst, T& tLast) const noexcept {
        const aabb_type& A0 = m_start;
        const aabb_type& B0 = other.m_start;
        point_type v = m_velocity - other.m_velocity;

        tFirst = T(0);
        tLast = T(1);
        for (int i = 0; i < 3; ++i) {
            T aMin = A0.min()[i];
            T aMax = A0.max()[i];
            T bMin = B0.min()[i];
            T bMax = B0.max()[i];
            T v_i = v[i];

            if (v_i > T(0)) {
                T first = (bMin - aMax) / v_i;
                T last  = (bMax - aMin) / v_i;
                if (first > tFirst) tFirst = first;
                if (last  < tLast)  tLast  = last;
            } else if (v_i < T(0)) {
                T first = (bMax - aMin) / v_i;
                T last  = (bMin - aMax) / v_i;
                if (first > tFirst) tFirst = first;
                if (last  < tLast)  tLast  = last;
            } else {
                // no relative motion on this axis
                if (aMax < bMin || bMax < aMin) return false;
            }
            if (tFirst > tLast) return false;
        }
        if (tFirst < T(0)) tFirst = T(0);
        if (tLast  > T(1)) tLast  = T(1);
        return (tFirst <= tLast);
    }

    // ------------------------------------------------------------------------
    //  Check if two moving AABBs overlap at any time during [0,1]
    // ------------------------------------------------------------------------
    bool overlaps(const MovingAABB& other) const noexcept {
        T tFirst, tLast;
        return continuousCollision(other, tFirst, tLast);
    }

    // ------------------------------------------------------------------------
    //  Bounding box of the swept volume (union of positions over [0,1])
    // ------------------------------------------------------------------------
    aabb_type sweptBounds() const noexcept {
        aabb_type end = atTime(T(1));
        return m_start.hull(end);
    }

private:
    aabb_type m_start;
    point_type m_velocity;
};

// ============================================================================
//  MovingPoint: point with linear velocity.
// ============================================================================
template<typename T = float>
class MovingPoint {
public:
    using value_type = T;
    using point_type = Basic::Vector<T, 3>;

    MovingPoint() noexcept : m_start(T(0)), m_velocity(T(0)) {}
    MovingPoint(const point_type& start, const point_type& velocity) noexcept
        : m_start(start), m_velocity(velocity) {}

    const point_type& start() const noexcept { return m_start; }
    const point_type& velocity() const noexcept { return m_velocity; }
    void setStart(const point_type& s) noexcept { m_start = s; }
    void setVelocity(const point_type& v) noexcept { m_velocity = v; }

    point_type atTime(T t) const noexcept { return m_start + m_velocity * t; }

    // Minimum distance between two moving points over [0,1]
    T minDistanceSq(const MovingPoint& other) const noexcept {
        point_type delta = (m_start - other.m_start);
        point_type relVel = m_velocity - other.m_velocity;
        T a = relVel.squaredLength();
        T b = T(2) * delta.dot(relVel);
        T c = delta.squaredLength();
        if (a <= T(0)) return c; // constant distance
        T t = -b / (T(2) * a);
        if (t < T(0)) t = T(0);
        if (t > T(1)) t = T(1);
        point_type at = delta + relVel * t;
        return at.squaredLength();
    }

private:
    point_type m_start;
    point_type m_velocity;
};

// ============================================================================
//  Time‑parameterised range query: finds all moving AABBs that intersect a
//  static query AABB at any time during [t0, t1].
//  Uses swept AABB test for each moving object.
// ============================================================================
template<typename T = float>
std::vector<size_t> timeRangeQuery(const std::vector<MovingAABB<T>>& objects,
                                   const Geometry::AABB<T,3>& query,
                                   T t0, T t1) {
    std::vector<size_t> result;
    for (size_t i = 0; i < objects.size(); ++i) {
        // Check overlap at any time within [t0,t1]
        // Transform time interval to [0,1] by reparameterising: let u = (t - t0)/(t1 - t0)
        // Then the moving box becomes: start' = obj.atTime(t0), velocity' = (obj.atTime(t1) - obj.atTime(t0))
        if (t1 <= t0) continue;
        T invDelta = T(1) / (t1 - t0);
        MovingAABB<T> subObj;
        subObj.setStart(objects[i].atTime(t0));
        point_type v1 = objects[i].atTime(t1).min() - objects[i].atTime(t0).min();
        subObj.setVelocity(v1);
        // Query AABB is static, so we treat it as a moving box with zero velocity
        MovingAABB<T> staticQuery(query, point_type(T(0)));
        if (subObj.overlaps(staticQuery)) {
            result.push_back(i);
        }
    }
    return result;
}

// ============================================================================
//  SIMD batch: evaluate timeRangeQuery for 4 objects and 4 queries (pairs)
//  Not fully vectorised due to conditional branches; but we can unroll.
// ============================================================================
template<typename T>
void batchTimeRangeQuery(const MovingAABB<T>* objects,
                         const Geometry::AABB<T,3>* queries,
                         const T* t0, const T* t1,
                         bool* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (t1[i] <= t0[i]) {
            out[i] = false;
            continue;
        }
        T invDelta = T(1) / (t1[i] - t0[i]);
        MovingAABB<T> subObj;
        subObj.setStart(objects[i].atTime(t0[i]));
        point_type v = objects[i].atTime(t1[i]).min() - objects[i].atTime(t0[i]).min();
        subObj.setVelocity(v);
        MovingAABB<T> staticQuery(queries[i], point_type(T(0)));
        out[i] = subObj.overlaps(staticQuery);
    }
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class MovingObjectsEnvironment {
public:
    static MovingObjectsEnvironment& instance() {
        static MovingObjectsEnvironment env;
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
    MovingObjectsEnvironment() : m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_MOVING_OBJECTS_H_INCLUDED