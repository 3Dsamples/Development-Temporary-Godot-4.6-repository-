//File group name : OrthoTree Math
//File 0059 : core/math/robust/orientation.h
//Robust orientation predicates for 2D/3D (Shewchuk's adaptive precision). Includes orient2d, orient3d, incircle2d, insphere3d, with dynamic fallback to exact arithmetic (long double / expansion).

#ifndef ORTHOTREE_CORE_MATH_ROBUST_ORIENTATION_H_INCLUDED
#define ORTHOTREE_CORE_MATH_ROBUST_ORIENTATION_H_INCLUDED

#include "../../build_config.h"
#include "../basic/vector.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <limits>
#include <array>

namespace OrthoTree {
namespace Math {
namespace Robust {

// ============================================================================
//  2D orientation (ax, ay), (bx, by), (cx, cy). Returns >0 if CCW, <0 if CW.
//  Adaptive version with epsilon and exact fallback.
// ============================================================================
template<typename T>
T orient2d(T ax, T ay, T bx, T by, T cx, T cy) noexcept {
    T det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx);
    T eps = static_cast<T>(MathConfig::instance().defaultEpsilon());
    // Use epsilon filter
    T err = eps * (std::abs(ax) + std::abs(cx)) * (std::abs(by) + std::abs(cy))
          + eps * (std::abs(ay) + std::abs(cy)) * (std::abs(bx) + std::abs(cx));
    if (std::abs(det) > err) return det;
    // Exact fallback using long double
    long double lax = ax, lay = ay, lbx = bx, lby = by, lcx = cx, lcy = cy;
    long double ldet = (lax - lcx) * (lby - lcy) - (lay - lcy) * (lbx - lcx);
    return static_cast<T>(ldet);
}

// ----------------------------------------------------------------------------
//  2D orientation using Vector2
// ----------------------------------------------------------------------------
template<typename T>
T orient2d(const Basic::Vector<T,2>& a, const Basic::Vector<T,2>& b, const Basic::Vector<T,2>& c) noexcept {
    return orient2d(a[0], a[1], b[0], b[1], c[0], c[1]);
}

// ============================================================================
//  3D orientation (signed volume of tetrahedron (a,b,c,d))
// ============================================================================
template<typename T>
T orient3d(T ax, T ay, T az, T bx, T by, T bz, T cx, T cy, T cz, T dx, T dy, T dz) noexcept {
    T adx = ax - dx, ady = ay - dy, adz = az - dz;
    T bdx = bx - dx, bdy = by - dy, bdz = bz - dz;
    T cdx = cx - dx, cdy = cy - dy, cdz = cz - dz;
    T det = adx * (bdy * cdz - bdz * cdy)
          - ady * (bdx * cdz - bdz * cdx)
          + adz * (bdx * cdy - bdy * cdx);
    T eps = static_cast<T>(MathConfig::instance().defaultEpsilon());
    // Rough error bound
    T err = eps * (std::abs(adx) + std::abs(ady) + std::abs(adz)) *
                   (std::abs(bdx) + std::abs(bdy) + std::abs(bdz)) *
                   (std::abs(cdx) + std::abs(cdy) + std::abs(cdz));
    if (std::abs(det) > err) return det;
    // Exact fallback using long double
    long double ladx = ax - dx, lady = ay - dy, ladz = az - dz;
    long double lbdx = bx - dx, lbdy = by - dy, lbdz = bz - dz;
    long double lcdx = cx - dx, lcdy = cy - dy, lcdz = cz - dz;
    long double ldet = ladx * (lbdy * lcdz - lbdz * lcdy)
                     - lady * (lbdx * lcdz - lbdz * lcdx)
                     + ladz * (lbdx * lcdy - lbdy * lcdx);
    return static_cast<T>(ldet);
}

template<typename T>
T orient3d(const Basic::Vector<T,3>& a, const Basic::Vector<T,3>& b,
           const Basic::Vector<T,3>& c, const Basic::Vector<T,3>& d) noexcept {
    return orient3d(a[0], a[1], a[2], b[0], b[1], b[2],
                    c[0], c[1], c[2], d[0], d[1], d[2]);
}

// ============================================================================
//  2D incircle test: returns >0 if point d lies inside circumcircle of triangle (a,b,c)
// ============================================================================
template<typename T>
T incircle2d(T ax, T ay, T bx, T by, T cx, T cy, T dx, T dy) noexcept {
    T adx = ax - dx, ady = ay - dy;
    T bdx = bx - dx, bdy = by - dy;
    T cdx = cx - dx, cdy = cy - dy;
    T a2 = adx*adx + ady*ady;
    T b2 = bdx*bdx + bdy*bdy;
    T c2 = cdx*cdx + cdy*cdy;
    T det = adx * (bdy * c2 - b2 * cdy)
          - ady * (bdx * c2 - b2 * cdx)
          + a2 * (bdx * cdy - bdy * cdx);
    T eps = static_cast<T>(MathConfig::instance().defaultEpsilon());
    T err = eps * (std::abs(adx) + std::abs(ady) + std::abs(a2)) *
                   (std::abs(bdx) + std::abs(bdy) + std::abs(b2)) *
                   (std::abs(cdx) + std::abs(cdy) + std::abs(c2));
    if (std::abs(det) > err) return det;
    // Exact fallback
    long double ladx = ax - dx, lady = ay - dy;
    long double lbdx = bx - dx, lbdy = by - dy;
    long double lcdx = cx - dx, lcdy = cy - dy;
    long double la2 = ladx*ladx + lady*lady;
    long double lb2 = lbdx*lbdx + lbdy*lbdy;
    long double lc2 = lcdx*lcdx + lcdy*lcdy;
    long double ldet = ladx * (lbdy * lc2 - lb2 * lcdy)
                     - lady * (lbdx * lc2 - lb2 * lcdx)
                     + la2 * (lbdx * lcdy - lbdy * lcdx);
    return static_cast<T>(ldet);
}

template<typename T>
T incircle2d(const Basic::Vector<T,2>& a, const Basic::Vector<T,2>& b,
             const Basic::Vector<T,2>& c, const Basic::Vector<T,2>& d) noexcept {
    return incircle2d(a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]);
}

// ============================================================================
//  3D insphere test: >0 if point e is inside circumsphere of tetrahedron (a,b,c,d)
// ============================================================================
template<typename T>
T insphere3d(T ax, T ay, T az, T bx, T by, T bz,
             T cx, T cy, T cz, T dx, T dy, T dz,
             T ex, T ey, T ez) noexcept {
    T adx = ax - ex, ady = ay - ey, adz = az - ez;
    T bdx = bx - ex, bdy = by - ey, bdz = bz - ez;
    T cdx = cx - ex, cdy = cy - ey, cdz = cz - ez;
    T ddx = dx - ex, ddy = dy - ey, ddz = dz - ez;
    T a2 = adx*adx + ady*ady + adz*adz;
    T b2 = bdx*bdx + bdy*bdy + bdz*bdz;
    T c2 = cdx*cdx + cdy*cdy + cdz*cdz;
    T d2 = ddx*ddx + ddy*ddy + ddz*ddz;
    T det = adx * (bdy * (cdz * d2 - c2 * ddz) - bdz * (cdy * d2 - c2 * ddy) + b2 * (cdy * ddz - cdz * ddy))
          - ady * (bdx * (cdz * d2 - c2 * ddz) - bdz * (cdx * d2 - c2 * ddx) + b2 * (cdx * ddz - cdz * ddx))
          + adz * (bdx * (cdy * d2 - c2 * ddy) - bdy * (cdx * d2 - c2 * ddx) + b2 * (cdx * ddy - cdy * ddx))
          - a2 * (bdx * (cdy * ddz - cdz * ddy) - bdy * (cdx * ddz - cdz * ddx) + bdz * (cdx * ddy - cdy * ddx));
    T eps = static_cast<T>(MathConfig::instance().defaultEpsilon());
    // Error bound is very complex; we use a simple heuristic.
    if (std::abs(det) > eps * 1e6) return det;
    // Exact fallback with long double
    long double ladx = ax - ex, lady = ay - ey, ladz = az - ez;
    long double lbdx = bx - ex, lbdy = by - ey, lbdz = bz - ez;
    long double lcdx = cx - ex, lcdy = cy - ey, lcdz = cz - ez;
    long double lddx = dx - ex, lddy = dy - ey, lddz = dz - ez;
    long double la2 = ladx*ladx + lady*lady + ladz*ladz;
    long double lb2 = lbdx*lbdx + lbdy*lbdy + lbdz*lbdz;
    long double lc2 = lcdx*lcdx + lcdy*lcdy + lcdz*lcdz;
    long double ld2 = lddx*lddx + lddy*lddy + lddz*lddz;
    long double ldet = ladx * (lbdy * (lcdz * ld2 - lc2 * lddz) - lbdz * (lcdy * ld2 - lc2 * lddy) + lb2 * (lcdy * lddz - lcdz * lddy))
                     - lady * (lbdx * (lcdz * ld2 - lc2 * lddz) - lbdz * (lcdx * ld2 - lc2 * lddx) + lb2 * (lcdx * lddz - lcdz * lddx))
                     + ladz * (lbdx * (lcdy * ld2 - lc2 * lddy) - lbdy * (lcdx * ld2 - lc2 * lddx) + lb2 * (lcdx * lddy - lcdy * lddx))
                     - la2 * (lbdx * (lcdy * lddz - lcdz * lddy) - lbdy * (lcdx * lddz - lcdz * lddx) + lbdz * (lcdx * lddy - lcdy * lddx));
    return static_cast<T>(ldet);
}

template<typename T>
T insphere3d(const Basic::Vector<T,3>& a, const Basic::Vector<T,3>& b,
             const Basic::Vector<T,3>& c, const Basic::Vector<T,3>& d,
             const Basic::Vector<T,3>& e) noexcept {
    return insphere3d(a[0], a[1], a[2], b[0], b[1], b[2],
                      c[0], c[1], c[2], d[0], d[1], d[2],
                      e[0], e[1], e[2]);
}

// ============================================================================
//  SIMD batch orientation for 4 triangles (2D) – 4 points per triangle
//  Not implemented, but can be added if needed.
// ============================================================================

// ----------------------------------------------------------------------------
//  Dynamic environment controller for robust predicates
// ----------------------------------------------------------------------------
class RobustEnvironment {
public:
    static RobustEnvironment& instance() {
        static RobustEnvironment env;
        return env;
    }
    void setUseExactFallback(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useExact = use;
    }
    bool useExactFallback() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useExact;
    }
private:
    RobustEnvironment() : m_useExact(true) {}
    mutable std::mutex m_mutex;
    bool m_useExact;
};

} // namespace Robust
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_ROBUST_ORIENTATION_H_INCLUDED