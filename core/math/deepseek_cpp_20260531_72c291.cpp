//File 0037 : core/math/numerical_methods.h
//Efficient numerical algorithms: root finding (Newton, bisection, secant), optimization (golden‑section, gradient descent, CG for linear systems), and integration (trapezoidal, Simpson) – all template‑based for scalar/SIMD.
#ifndef CORE_MATH_NUMERICAL_METHODS_H
#define CORE_MATH_NUMERICAL_METHODS_H

#include "vector_math.h"
#include <functional>
#include <cmath>
#include <limits>
#include <cstdint>

namespace SimulationMath {
namespace numerical {

// -----------------------------------------------------------------------------
// 1. Newton's method for root finding (f: float->float, derivative provided)
// -----------------------------------------------------------------------------
template <typename Func, typename Deriv>
float newton_raphson(float x0, const Func& f, const Deriv& df, float tol = 1e-6f, int max_iter = 50) noexcept {
    float x = x0;
    for (int i = 0; i < max_iter; ++i) {
        float fx = f(x);
        if (std::abs(fx) < tol) break;
        float dx = df(x);
        if (std::abs(dx) < 1e-12f) break;
        x -= fx / dx;
    }
    return x;
}

// -----------------------------------------------------------------------------
// 2. Bisection method (requires interval [a,b] with f(a)*f(b) < 0)
// -----------------------------------------------------------------------------
template <typename Func>
float bisection(const Func& f, float a, float b, float tol = 1e-6f, int max_iter = 50) noexcept {
    float fa = f(a);
    float fb = f(b);
    if (fa * fb >= 0.0f) return a; // no sign change
    float m;
    for (int i = 0; i < max_iter; ++i) {
        m = (a + b) * 0.5f;
        float fm = f(m);
        if (std::abs(fm) < tol) break;
        if (fa * fm < 0.0f) { b = m; fb = fm; }
        else { a = m; fa = fm; }
    }
    return m;
}

// -----------------------------------------------------------------------------
// 3. Secant method (derivative‑free)
// -----------------------------------------------------------------------------
template <typename Func>
float secant(const Func& f, float x0, float x1, float tol = 1e-6f, int max_iter = 50) noexcept {
    float f0 = f(x0);
    float f1 = f(x1);
    float x2;
    for (int i = 0; i < max_iter; ++i) {
        if (std::abs(f1 - f0) < 1e-12f) break;
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        if (std::abs(f(x2)) < tol) return x2;
        x0 = x1; f0 = f1;
        x1 = x2; f1 = f(x1);
    }
    return x2;
}

// -----------------------------------------------------------------------------
// 4. Golden‑section search for minimum of unimodal function on [a,b]
// -----------------------------------------------------------------------------
template <typename Func>
float golden_section_min(const Func& f, float a, float b, float tol = 1e-6f) noexcept {
    const float inv_phi = 0.6180339887498949f; // (sqrt(5)-1)/2
    float x1 = b - inv_phi * (b - a);
    float x2 = a + inv_phi * (b - a);
    float f1 = f(x1), f2 = f(x2);
    while (b - a > tol) {
        if (f1 < f2) {
            b = x2; x2 = x1; f2 = f1;
            x1 = b - inv_phi * (b - a);
            f1 = f(x1);
        } else {
            a = x1; x1 = x2; f1 = f2;
            x2 = a + inv_phi * (b - a);
            f2 = f(x2);
        }
    }
    return (a + b) * 0.5f;
}

// -----------------------------------------------------------------------------
// 5. Gradient descent for multivariate function (vector of size N)
// -----------------------------------------------------------------------------
template <typename Vector, typename Func, typename Grad>
Vector gradient_descent(const Func& f, const Grad& grad, const Vector& x0,
                        float learning_rate = 0.01f, int max_iter = 1000, float tol = 1e-6f) noexcept {
    Vector x = x0;
    for (int i = 0; i < max_iter; ++i) {
        Vector g = grad(x);
        float norm_g = g.norm();
        if (norm_g < tol) break;
        x = x - learning_rate * g;
    }
    return x;
}

// -----------------------------------------------------------------------------
// 6. Conjugate Gradient solver for Ax = b (A is symmetric positive definite, matrix‑free)
// -----------------------------------------------------------------------------
template <typename Vector, typename MatrixOp>
Vector conjugate_gradient(const MatrixOp& A, const Vector& b, const Vector& x0,
                          int max_iter = 200, float tol = 1e-6f) noexcept {
    Vector x = x0;
    Vector r = b - A(x);
    Vector p = r;
    float rs_old = r.dot(r);
    for (int i = 0; i < max_iter; ++i) {
        Vector Ap = A(p);
        float alpha = rs_old / std::max(p.dot(Ap), 1e-12f);
        x = x + alpha * p;
        r = r - alpha * Ap;
        float rs_new = r.dot(r);
        if (std::sqrt(rs_new) < tol) break;
        float beta = rs_new / rs_old;
        p = r + beta * p;
        rs_old = rs_new;
    }
    return x;
}

// -----------------------------------------------------------------------------
// 7. Trapezoidal integration (uniform steps)
// -----------------------------------------------------------------------------
template <typename Func>
float trapezoidal(const Func& f, float a, float b, int n = 1000) noexcept {
    float h = (b - a) / n;
    float sum = 0.5f * (f(a) + f(b));
    for (int i = 1; i < n; ++i)
        sum += f(a + i * h);
    return sum * h;
}

// -----------------------------------------------------------------------------
// 8. Simpson's 1/3 integration (n must be even)
// -----------------------------------------------------------------------------
template <typename Func>
float simpson(const Func& f, float a, float b, int n = 1000) noexcept {
    if (n % 2 == 1) ++n; // ensure even
    float h = (b - a) / n;
    float sum = f(a) + f(b);
    for (int i = 1; i < n; i += 2)
        sum += 4.0f * f(a + i * h);
    for (int i = 2; i < n - 1; i += 2)
        sum += 2.0f * f(a + i * h);
    return sum * h / 3.0f;
}

} // namespace numerical
} // namespace SimulationMath

#endif // CORE_MATH_NUMERICAL_METHODS_H