// File 0022 : core/math/numeric.h
// Numerical methods: root finding (bisection, Newton, secant), ODE integration (RK4), quadrature (Simpson), least squares, power iteration.

#pragma once

#include "constants.h"
#include "vec3.h"
#include "mat3.h"
#include <cmath>
#include <functional>
#include <vector>

namespace wp {

// ── Root finding ────────────────────────────────────────────────────
template <typename T, typename Func>
T bisection(Func f, T a, T b, T tol = MathConst<T>::epsilon, int max_iter = 100) {
    T fa = f(a);
    T fb = f(b);
    if (fa * fb > T(0)) return std::numeric_limits<T>::quiet_NaN(); // no root guarantee
    for (int i = 0; i < max_iter; ++i) {
        T c = (a + b) * T(0.5);
        T fc = f(c);
        if (std::abs(fc) <= tol || (b - a) * T(0.5) <= tol) return c;
        if (fa * fc < T(0)) { b = c; fb = fc; }
        else { a = c; fa = fc; }
    }
    return (a + b) * T(0.5);
}

template <typename T, typename Func, typename Deriv>
T newton_raphson(Func f, Deriv df, T x0, T tol = MathConst<T>::epsilon, int max_iter = 100) {
    T x = x0;
    for (int i = 0; i < max_iter; ++i) {
        T fx = f(x);
        if (std::abs(fx) <= tol) return x;
        T dfx = df(x);
        if (std::abs(dfx) < tol) break;
        T delta = fx / dfx;
        x -= delta;
        if (std::abs(delta) <= tol) return x;
    }
    return x;
}

template <typename T, typename Func>
T secant(Func f, T x0, T x1, T tol = MathConst<T>::epsilon, int max_iter = 100) {
    T fx0 = f(x0);
    T fx1 = f(x1);
    for (int i = 0; i < max_iter; ++i) {
        if (std::abs(fx1) <= tol) return x1;
        T denom = fx1 - fx0;
        if (std::abs(denom) < tol) break;
        T x2 = x1 - fx1 * (x1 - x0) / denom;
        x0 = x1; fx0 = fx1;
        x1 = x2; fx1 = f(x2);
        if (std::abs(x1 - x0) <= tol) return x1;
    }
    return x1;
}

// ── Ordinary Differential Equations ─────────────────────────────────
template <typename T, typename State, typename Func>
State rk4(Func f, const State& y, T t, T h) {
    State k1 = f(t, y);
    State k2 = f(t + h * T(0.5), y + k1 * (h * T(0.5)));
    State k3 = f(t + h * T(0.5), y + k2 * (h * T(0.5)));
    State k4 = f(t + h, y + k3 * h);
    return y + (k1 + k2 * T(2) + k3 * T(2) + k4) * (h / T(6));
}

// Integrate ODE over interval [t0, t_end] with fixed step size
template <typename T, typename State, typename Func>
State integrate_rk4(Func f, State y0, T t0, T t_end, T step) {
    State y = y0;
    T t = t0;
    while (t < t_end) {
        T h = std::min(step, t_end - t);
        y = rk4(f, y, t, h);
        t += h;
    }
    return y;
}

// ── Quadrature (Simpson's rule) ─────────────────────────────────────
template <typename T, typename Func>
T simpson_1d(Func f, T a, T b, int n = 100) {
    if (n % 2 == 1) n++; // must be even
    T h = (b - a) / T(n);
    T sum = f(a) + f(b);
    for (int i = 1; i < n; ++i) {
        T x = a + h * T(i);
        sum += (i % 2 == 0) ? f(x) * T(2) : f(x) * T(4);
    }
    return sum * h / T(3);
}

// ── Linear Least Squares ─────────────────────────────────────────────
// Solve overdetermined system A*x = b using normal equations (A^T A x = A^T b)
// A is MxN (M >= N), b length M, x length N; return true if success
template <typename T>
bool linear_least_squares(const std::vector<std::vector<T>>& A,
                          const std::vector<T>& b,
                          std::vector<T>& x) {
    size_t M = A.size();
    if (M == 0) return false;
    size_t N = A[0].size();
    if (M < N) return false;
    // Form normal equations: C = A^T A, d = A^T b
    std::vector<std::vector<T>> C(N, std::vector<T>(N, T(0)));
    std::vector<T> d(N, T(0));
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T aj = A[i][j];
            for (size_t k = 0; k < N; ++k) C[j][k] += aj * A[i][k];
            d[j] += aj * b[i];
        }
    }
    // Solve C * x = d using Gaussian elimination with partial pivoting
    for (size_t k = 0; k < N; ++k) {
        // pivot
        size_t max_row = k;
        T max_val = std::abs(C[k][k]);
        for (size_t i = k+1; i < N; ++i) {
            if (std::abs(C[i][k]) > max_val) {
                max_val = std::abs(C[i][k]);
                max_row = i;
            }
        }
        if (max_val < MathConst<T>::epsilon) return false;
        if (max_row != k) {
            std::swap(C[k], C[max_row]);
            std::swap(d[k], d[max_row]);
        }
        // eliminate
        T pivot = C[k][k];
        for (size_t j = k; j < N; ++j) C[k][j] /= pivot;
        d[k] /= pivot;
        for (size_t i = k+1; i < N; ++i) {
            T factor = C[i][k];
            if (std::abs(factor) > MathConst<T>::epsilon) {
                for (size_t j = k; j < N; ++j) C[i][j] -= factor * C[k][j];
                d[i] -= factor * d[k];
            }
        }
    }
    // back substitution
    x.assign(N, T(0));
    for (int i = static_cast<int>(N)-1; i >= 0; --i) {
        x[i] = d[i];
        for (size_t j = i+1; j < N; ++j) x[i] -= C[i][j] * x[j];
    }
    return true;
}

// ── Power iteration for dominant eigenvalue of 3x3 symmetric matrix ──
template <typename T>
T dominant_eigenvalue(const mat3<T>& A, vec3<T> initial = vec3<T>(T(1)), T tol = MathConst<T>::epsilon, int max_iter = 100) {
    vec3<T> v = initial;
    T lambda = T(0);
    for (int i = 0; i < max_iter; ++i) {
        vec3<T> Av = mul(A, v);
        T lambda_new = length(Av);
        if (lambda_new < tol) break;
        v = Av / lambda_new;
        if (std::abs(lambda_new - lambda) < tol) return lambda_new;
        lambda = lambda_new;
    }
    return lambda;
}

} // namespace wp