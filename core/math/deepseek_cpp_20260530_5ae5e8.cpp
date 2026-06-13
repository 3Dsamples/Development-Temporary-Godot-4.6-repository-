// File 0015 : core/math/frustum.h
// View frustum defined by six planes, extracted from matrix, with intersection tests against AABB, sphere, capsule, and point.

#pragma once

#include "plane.h"
#include "aabb.h"
#include "sphere.h"
#include "capsule.h"
#include "ray.h"
#include "mat4.h"
#include <array>

namespace wp {

template <typename T>
class frustum {
public:
    enum PlaneIndex { LEFT = 0, RIGHT, BOTTOM, TOP, NEAR, FAR, COUNT = 6 };

    constexpr frustum() noexcept {
        for (int i = 0; i < 6; ++i)
            m_planes[i] = plane<T>();
    }

    // Build from a modelview*projection matrix (combined view-projection)
    // Assuming column-major storage: extracted planes.
    explicit frustum(const mat4<T>& m) noexcept {
        extract(m);
    }

    const plane<T>& plane_left()   const noexcept { return m_planes[LEFT]; }
    const plane<T>& plane_right()  const noexcept { return m_planes[RIGHT]; }
    const plane<T>& plane_bottom() const noexcept { return m_planes[BOTTOM]; }
    const plane<T>& plane_top()    const noexcept { return m_planes[TOP]; }
    const plane<T>& plane_near()   const noexcept { return m_planes[NEAR]; }
    const plane<T>& plane_far()    const noexcept { return m_planes[FAR]; }

    const plane<T>& operator[](int i) const noexcept { return m_planes[i]; }

    // Intersection tests (fully inside, intersecting, or outside)
    // Returns true if object is at least partially inside (i.e., visible).
    bool contains_point(const vec3<T>& p) const noexcept {
        for (int i = 0; i < 6; ++i)
            if (m_planes[i].distance(p) < T(0)) return false;
        return true;
    }

    bool intersects_aabb(const aabb<T>& box) const noexcept {
        // AABB vs all planes
        for (int i = 0; i < 6; ++i) {
            vec3<T> p_vertex = box.min;
            if (m_planes[i].normal.x >= T(0)) p_vertex.x = box.max.x;
            if (m_planes[i].normal.y >= T(0)) p_vertex.y = box.max.y;
            if (m_planes[i].normal.z >= T(0)) p_vertex.z = box.max.z;
            if (m_planes[i].distance(p_vertex) < T(0))
                return false; // completely outside this plane
        }
        return true;
    }

    bool intersects_sphere(const sphere<T>& s) const noexcept {
        for (int i = 0; i < 6; ++i)
            if (m_planes[i].distance(s.center) < -s.radius)
                return false;
        return true;
    }

    bool intersects_capsule(const capsule<T>& cap) const noexcept {
        // use bounding sphere test for now (exact test could check segment)
        sphere<T> bs = cap.bounding_sphere();
        return intersects_sphere(bs);
    }

private:
    plane<T> m_planes[6];

    void extract(const mat4<T>& m) {
        // Left plane: row4 + row1
        m_planes[LEFT]   = plane<T>(
            vec3<T>(m.m30 + m.m00, m.m31 + m.m01, m.m32 + m.m02),
            m.m33 + m.m03
        ).normalized();
        // Right plane: row4 - row1
        m_planes[RIGHT]  = plane<T>(
            vec3<T>(m.m30 - m.m00, m.m31 - m.m01, m.m32 - m.m02),
            m.m33 - m.m03
        ).normalized();
        // Bottom plane: row4 + row2
        m_planes[BOTTOM] = plane<T>(
            vec3<T>(m.m30 + m.m10, m.m31 + m.m11, m.m32 + m.m12),
            m.m33 + m.m13
        ).normalized();
        // Top plane: row4 - row2
        m_planes[TOP]    = plane<T>(
            vec3<T>(m.m30 - m.m10, m.m31 - m.m11, m.m32 - m.m12),
            m.m33 - m.m13
        ).normalized();
        // Near plane: row4 + row3 (or row3? depends on matrix handedness; typical OpenGL uses row3 for near)
        m_planes[NEAR]   = plane<T>(
            vec3<T>(m.m20, m.m21, m.m22),
            m.m23
        ).normalized();
        // Far plane: row4 - row3
        m_planes[FAR]    = plane<T>(
            vec3<T>(m.m30 - m.m20, m.m31 - m.m21, m.m32 - m.m22),
            m.m33 - m.m23
        ).normalized();
    }
};

// Convenience alias
using frustumf = frustum<float>;
using frustumd = frustum<double>;

} // namespace wp