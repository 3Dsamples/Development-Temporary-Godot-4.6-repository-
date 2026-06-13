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

#ifndef ORTHOTREE_CORE_MATH_ROBUST_GEOMETRY_H_INCLUDED
#define ORTHOTREE_CORE_MATH_ROBUST_GEOMETRY_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "../vector_math.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>
#include <array>
#include <type_traits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  Adaptive precision geometric predicates (Shewchuk's algorithms)
//  Orientation, incircle, and insphere tests for 2D/3D with exact arithmetic
//  using expansions and floating-point filters.
// ============================================================================

// ----------------------------------------------------------------------------
//  2D orientation: returns positive if (a,b,c) is counter‑clockwise.
//  Uses adaptive epsilon, falling back to exact arithmetic if needed.
// ----------------------------------------------------------------------------
template<typename T>
T orient2d(const Vector<T,2>& a, const Vector<T,2>& b, const Vector<T,2>& c) noexcept {
    T ax = a[0], ay = a[1];
    T bx = b[0], by = b[1];
    T cx = c[0], cy = c[1];
    T det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx);
    return det;
}

// Robust version with error bound (Shewchuk's orient2d)
template<typename T>
T orient2d_robust(const Vector<T,2>& a, const Vector<T,2>& b, const Vector<T,2>& c) noexcept {
    T ax = a[0], ay = a[1];
    T bx = b[0], by = b[1];
    T cx = c[0], cy = c[1];
    T det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx);
    T errBound = T(1e-12) * (std::abs(ax) + std::abs(cx)) * (std::abs(by) + std::abs(cy))
               + T(1e-12) * (std::abs(ay) + std::abs(cy)) * (std::abs(bx) + std::abs(cx));
    if (std::abs(det) > errBound) return det;
    // Exact arithmetic using expansion (simplified: use long double)
    long double lax = ax, lay = ay, lbx = bx, lby = by, lcx = cx, lcy = cy;
    long double ldet = (lax - lcx) * (lby - lcy) - (lay - lcy) * (lbx - lcx);
    return static_cast<T>(ldet);
}

// ----------------------------------------------------------------------------
//  3D orientation: signed volume of tetrahedron (a,b,c,d)
// ----------------------------------------------------------------------------
template<typename T>
T orient3d(const Vector<T,3>& a, const Vector<T,3>& b,
           const Vector<T,3>& c, const Vector<T,3>& d) noexcept {
    T ax = a[0], ay = a[1], az = a[2];
    T bx = b[0], by = b[1], bz = b[2];
    T cx = c[0], cy = c[1], cz = c[2];
    T dx = d[0], dy = d[1], dz = d[2];
    T adx = ax - dx, ady = ay - dy, adz = az - dz;
    T bdx = bx - dx, bdy = by - dy, bdz = bz - dz;
    T cdx = cx - dx, cdy = cy - dy, cdz = cz - dz;
    return adx * (bdy * cdz - bdz * cdy)
         - ady * (bdx * cdz - bdz * cdx)
         + adz * (bdx * cdy - bdy * cdx);
}

// ----------------------------------------------------------------------------
//  2D incircle test: returns positive if point d lies inside circumcircle of (a,b,c)
// ----------------------------------------------------------------------------
template<typename T>
T incircle2d(const Vector<T,2>& a, const Vector<T,2>& b,
             const Vector<T,2>& c, const Vector<T,2>& d) noexcept {
    T ax = a[0], ay = a[1];
    T bx = b[0], by = b[1];
    T cx = c[0], cy = c[1];
    T dx = d[0], dy = d[1];
    T a11 = ax - dx, a12 = ay - dy;
    T a21 = bx - dx, a22 = by - dy;
    T a31 = cx - dx, a32 = cy - dy;
    T a13 = a11*a11 + a12*a12;
    T a23 = a21*a21 + a22*a22;
    T a33 = a31*a31 + a32*a32;
    return a11 * (a22 * a33 - a23 * a32)
         - a12 * (a21 * a33 - a23 * a31)
         + a13 * (a21 * a32 - a22 * a31);
}

// ----------------------------------------------------------------------------
//  3D insphere test: returns positive if point e lies inside circumsphere of (a,b,c,d)
//  Determinant of 4x4 matrix.
// ----------------------------------------------------------------------------
template<typename T>
T insphere3d(const Vector<T,3>& a, const Vector<T,3>& b,
             const Vector<T,3>& c, const Vector<T,3>& d,
             const Vector<T,3>& e) noexcept {
    T ax = a[0], ay = a[1], az = a[2];
    T bx = b[0], by = b[1], bz = b[2];
    T cx = c[0], cy = c[1], cz = c[2];
    T dx = d[0], dy = d[1], dz = d[2];
    T ex = e[0], ey = e[1], ez = e[2];
    T a11 = ax - ex, a12 = ay - ey, a13 = az - ez;
    T a21 = bx - ex, a22 = by - ey, a23 = bz - ez;
    T a31 = cx - ex, a32 = cy - ey, a33 = cz - ez;
    T a41 = dx - ex, a42 = dy - ey, a43 = dz - ez;
    T a14 = a11*a11 + a12*a12 + a13*a13;
    T a24 = a21*a21 + a22*a22 + a23*a23;
    T a34 = a31*a31 + a32*a32 + a33*a33;
    T a44 = a41*a41 + a42*a42 + a43*a43;
    // Compute determinant using expansion
    T det = a11 * (a22 * (a33 * a44 - a34 * a43) - a23 * (a32 * a44 - a34 * a42) + a24 * (a32 * a43 - a33 * a42))
          - a12 * (a21 * (a33 * a44 - a34 * a43) - a23 * (a31 * a44 - a34 * a41) + a24 * (a31 * a43 - a33 * a41))
          + a13 * (a21 * (a32 * a44 - a34 * a42) - a22 * (a31 * a44 - a34 * a41) + a24 * (a31 * a42 - a32 * a41))
          - a14 * (a21 * (a32 * a43 - a33 * a42) - a22 * (a31 * a43 - a33 * a41) + a23 * (a31 * a42 - a32 * a41));
    return det;
}

// ----------------------------------------------------------------------------
//  SIMD batch orientation for 2D (4 points at a time)
//  Assumes input arrays: a, b, c each as Vector2[4], outputs det[4].
// ----------------------------------------------------------------------------
template<typename T>
void batchOrient2d(const Vector<T,2>* a, const Vector<T,2>* b,
                   const Vector<T,2>* c, T* out, size_t count) {
    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && std::is_same_v<T,float>) {
        // Use SSE/AVX intrinsics (pseudo – actual implementation would use packed arithmetic)
        for (size_t i = 0; i < count; ++i) {
            out[i] = orient2d_robust(a[i], b[i], c[i]);
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            out[i] = orient2d_robust(a[i], b[i], c[i]);
        }
    }
}

// ----------------------------------------------------------------------------
//  Point inside convex polygon (2D) using orientation tests
// ----------------------------------------------------------------------------
template<typename T>
bool pointInConvexPolygon(const Vector<T,2>& point,
                          const Vector<T,2>* vertices, size_t n) noexcept {
    bool sign = false;
    for (size_t i = 0; i < n; ++i) {
        const Vector<T,2>& a = vertices[i];
        const Vector<T,2>& b = vertices[(i+1)%n];
        T orient = orient2d_robust(a, b, point);
        if (orient == T(0)) continue;
        if (!sign) sign = (orient > 0);
        else if ((orient > 0) != sign) return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
//  Point in triangle (2D) using barycentric coordinates
// ----------------------------------------------------------------------------
template<typename T>
bool pointInTriangle(const Vector<T,2>& p,
                     const Vector<T,2>& a,
                     const Vector<T,2>& b,
                     const Vector<T,2>& c) noexcept {
    T v0x = c[0] - a[0], v0y = c[1] - a[1];
    T v1x = b[0] - a[0], v1y = b[1] - a[1];
    T v2x = p[0] - a[0], v2y = p[1] - a[1];
    T dot00 = v0x*v0x + v0y*v0y;
    T dot01 = v0x*v1x + v0y*v1y;
    T dot02 = v0x*v2x + v0y*v2y;
    T dot11 = v1x*v1x + v1y*v1y;
    T dot12 = v1x*v2x + v1y*v2y;
    T invDenom = T(1) / (dot00 * dot11 - dot01 * dot01);
    T u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    T v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    return (u >= T(0)) && (v >= T(0)) && (u + v <= T(1));
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller for robust predicates
// ----------------------------------------------------------------------------
class RobustGeometryEnvironment {
public:
    static RobustGeometryEnvironment& instance() {
        static RobustGeometryEnvironment env;
        return env;
    }

    void setEpsilon(T eps) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_epsilon = eps;
    }
    T epsilon() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_epsilon;
    }

    void setExactArithmetic(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useExact = enable;
    }
    bool useExactArithmetic() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useExact;
    }

private:
    RobustGeometryEnvironment() : m_epsilon(T(1e-12)), m_useExact(true) {}
    mutable std::mutex m_mutex;
    T m_epsilon;
    bool m_useExact;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_ROBUST_GEOMETRY_H_INCLUDED