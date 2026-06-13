//File group name : OrthoTree Math
//File 0060 : core/math/numerical/root_finding.h
//Root finding algorithms (bisection, Newton‑Raphson, secant, Brent) for scalar functions. Supports double/float, tolerances, max iterations, and dynamic environment controls.

#ifndef ORTHOTREE_CORE_MATH_NUMERICAL_ROOT_FINDING_H_INCLUDED
#define ORTHOTREE_CORE_MATH_NUMERICAL_ROOT_FINDING_H_INCLUDED

#include "../../build_config.h"
#include "../math_config.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cmath>
#include <functional>
#include <limits>
#include <algorithm>

namespace OrthoTree {
namespace Math {
namespace Numerical {

// ============================================================================
//  Bisection method: find root of f(x) in interval [a,b] where f(a)*f(b) < 0.
// ============================================================================
template<typename T>
T bisection(const std::function<T(T)>& f, T a, T b,
            T tol = static_cast<T>(MathConfig::instance().defaultEpsilon()),
            int maxIter = static_cast<int>(MathConfig::instance().maxIterations())) {
    T fa = f(a), fb = f(b);
    if (fa * fb >= T(0)) return a; // no sign change
    for (int iter = 0; iter < maxIter; ++iter) {
        T c = (a + b) * T(0.5);
        T fc = f(c);
        if (std::abs(fc) < tol || (b - a) * T(0.5) < tol) return c;
        if (fa * fc < T(0)) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }
    return (a + b) * T(0.5);
}

// ============================================================================
//  Newton‑Raphson method (requires derivative fprime)
// ============================================================================
template<typename T>
T newton(const std::function<T(T)>& f, const std::function<T(T)>& fprime,
         T x0, T tol = static_cast<T>(MathConfig::instance().defaultEpsilon()),
         int maxIter = static_cast<int>(MathConfig::instance().maxIterations())) {
    T x = x0;
    for (int iter = 0; iter < maxIter; ++iter) {
        T fx = f(x);
        if (std::abs(fx) < tol) return x;
        T fp = fprime(x);
        if (std::abs(fp) < T(1e-12)) break;
        T dx = fx / fp;
        x -= dx;
        if (std::abs(dx) < tol * std::abs(x) + tol) break;
    }
    return x;
}

// ============================================================================
//  Secant method (no derivative required)
// ============================================================================
template<typename T>
T secant(const std::function<T(T)>& f, T x0, T x1,
         T tol = static_cast<T>(MathConfig::instance().defaultEpsilon()),
         int maxIter = static_cast<int>(MathConfig::instance().maxIterations())) {
    T f0 = f(x0), f1 = f(x1);
    for (int iter = 0; iter < maxIter; ++iter) {
        if (std::abs(f1) < tol) return x1;
        T x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        T f2 = f(x2);
        x0 = x1; f0 = f1;
        x1 = x2; f1 = f2;
        if (std::abs(x1 - x0) < tol * std::abs(x1) + tol) break;
    }
    return x1;
}

// ============================================================================
//  Brent's method (robust, combines bisection, secant, inverse quadratic)
//  Based on Brent's algorithm from "Algorithms for Minimization without Derivatives".
// ============================================================================
template<typename T>
T brent(const std::function<T(T)>& f, T a, T b,
        T tol = static_cast<T>(MathConfig::instance().defaultEpsilon()),
        int maxIter = static_cast<int>(MathConfig::instance().maxIterations())) {
    T fa = f(a), fb = f(b);
    if (fa * fb >= T(0)) return a;
    T c = a, fc = fa;
    T d = T(0), e = T(0);
    for (int iter = 0; iter < maxIter; ++iter) {
        if ((fb > T(0) && fc > T(0)) || (fb < T(0) && fc < T(0))) {
            c = a; fc = fa;
            d = e = b - a;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        T tol1 = tol * std::abs(b) + T(0.5) * tol;
        T xm = (c - b) * T(0.5);
        if (std::abs(xm) <= tol1 || fb == T(0)) return b;
        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            T s;
            if (a == c) {
                // secant
                s = -fb * (b - a) / (fb - fa);
            } else {
                // inverse quadratic interpolation
                T q = fa / fc, r = fb / fc;
                T p = s = r * ( (b - a) * q * (q - r) - (b - c) * (r - T(1)) );
                q = (q - T(1)) * (r - T(1)) * (s - T(1));
                s = p / q;
            }
            if (std::abs(s) < T(0.5) * std::abs(e) && s > T(0) && s < std::abs(xm)) {
                e = d;
                d = s;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b; fa = fb;
        if (std::abs(d) > tol1) b += d;
        else b += (xm > T(0) ? tol1 : -tol1);
        fb = f(b);
    }
    return b;
}

// ============================================================================
//  Multi‑root finder for polynomials (real roots) – wrapper for cubic/quartic
//  Could use companion matrix, but for now we rely on earlier solvers.
//  Placeholder for completeness.
// ============================================================================
template<typename T>
int polynomialRoots(const T* coeffs, int degree, T* roots) {
    if (degree == 2) {
        T a = coeffs[2], b = coeffs[1], c = coeffs[0];
        return solveQuadratic(a, b, c, roots[0], roots[1]);
    } else if (degree == 3) {
        T a = coeffs[3], b = coeffs[2], c = coeffs[1], d = coeffs[0];
        return solveCubic(a, b, c, d, roots);
    } else if (degree == 4) {
        T a = coeffs[4], b = coeffs[3], c = coeffs[2], d = coeffs[1], e = coeffs[0];
        return solveQuartic(a, b, c, d, e, roots);
    }
    return 0;
}

// ============================================================================
//  SIMD batch: bisection on 4 functions (not typical, but we can vectorize loops)
//  For simplicity, we provide a batch wrapper that processes 4 intervals.
// ============================================================================
template<typename T>
void batchBisection(const std::function<T(T)>& f, const T* a, const T* b,
                    T* out, size_t count, T tol = MathConfig::instance().defaultEpsilon()) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = bisection(f, a[i], b[i], tol);
    }
}

// ----------------------------------------------------------------------------
//  Dynamic environment controller
// ----------------------------------------------------------------------------
class RootFindingEnvironment {
public:
    static RootFindingEnvironment& instance() {
        static RootFindingEnvironment env;
        return env;
    }
    void setDefaultTolerance(T tol) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_defaultTol = tol;
    }
    T defaultTolerance() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_defaultTol;
    }
    void setMaxIterations(int maxIter) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_maxIter = maxIter;
    }
    int maxIterations() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_maxIter;
    }
private:
    RootFindingEnvironment() : m_defaultTol(T(1e-8)), m_maxIter(100) {}
    mutable std::mutex m_mutex;
    T m_defaultTol;
    int m_maxIter;
};

} // namespace Numerical
} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_NUMERICAL_ROOT_FINDING_H_INCLUDED