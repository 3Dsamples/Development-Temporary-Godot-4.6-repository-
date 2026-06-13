// system name : onetbb-warp
// File 0016 : core/math/optimization.h
// Description : Numerical optimization: root finding, gradient descent, Newton, BFGS, Nelder‑Mead.

#ifndef __TBB_WARP_CORE_MATH_OPTIMIZATION_H
#define __TBB_WARP_CORE_MATH_OPTIMIZATION_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include <cmath>
#include <functional>
#include <limits>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <tuple>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Golden‑section search for 1D minimum
// ============================================================

template<typename Func>
float golden_section_search(Func f, float a, float b, float tol = 1e-6f, int max_iter = 100) {
    const float inv_phi = 0.6180339887498949f; // (sqrt(5)-1)/2
    float x1 = b - inv_phi * (b - a);
    float x2 = a + inv_phi * (b - a);
    float f1 = f(x1), f2 = f(x2);
    for (int i = 0; i < max_iter && (b - a) > tol; ++i) {
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

// ============================================================
// Brent's method for 1D minimum (robust)
// ============================================================

template<typename Func>
float brent_minimize(Func f, float a, float b, float tol = 1e-6f, int max_iter = 100) {
    const float golden = 0.3819660112501051f; // 1 - inv_phi
    float x = a + golden * (b - a);
    float w = x, v = x;
    float fx = f(x), fw = fx, fv = fx;
    float d = 0.0f, e = 0.0f;
    float xm, tol1, tol2, r, p, q, u, fu;
    for (int i = 0; i < max_iter; ++i) {
        xm = (a + b) * 0.5f;
        tol1 = tol * std::abs(x) + 1e-10f;
        tol2 = 2.0f * tol1;
        if (std::abs(x - xm) <= tol2 - (b - a) * 0.5f) return x;
        if (std::abs(e) > tol1) {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0f * (q - r);
            if (q > 0.0f) p = -p; else q = -q;
            r = e; e = d;
            if (std::abs(p) < std::abs(0.5f * q * r) && p > q*(a-x) && p < q*(b-x)) {
                d = p / q; u = x + d;
                if (u - a < tol2 || b - u < tol2) d = (xm - x) < 0 ? -tol1 : tol1;
            } else { e = (x >= xm) ? a - x : b - x; d = golden * e; }
        } else { e = (x >= xm) ? a - x : b - x; d = golden * e; }
        u = x + ((std::abs(d) > tol1) ? d : ((d>0) ? tol1 : -tol1));
        fu = f(u);
        if (fu <= fx) {
            if (u >= x) a = x; else b = x;
            v = w; fv = fw; w = x; fw = fx; x = u; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) { v = w; fv = fw; w = u; fw = fu; }
            else if (fu <= fv || v == x || v == w) { v = u; fv = fu; }
        }
    }
    return x;
}

// ============================================================
// Gradient descent with backtracking line search (1D input)
// ============================================================

template<typename Func, typename GradFunc>
float gradient_descent(Func f, GradFunc grad, float x0, float step = 0.1f, float tol = 1e-6f,
                       int max_iter = 1000, float alpha = 0.5f, float beta = 0.5f) {
    float x = x0;
    float g = grad(x);
    for (int i = 0; i < max_iter && std::abs(g) > tol; ++i) {
        float t = step;
        float fx = f(x);
        float x_new = x - t * g;
        while (f(x_new) > fx - alpha * t * g * g && t > 1e-12f) {
            t *= beta;
            x_new = x - t * g;
        }
        x = x_new;
        g = grad(x);
    }
    return x;
}

// ============================================================
// Gradient descent for vector functions (fixed step)
// ============================================================

template<typename Func, typename GradFunc, std::size_t N>
std::array<float, N> gradient_descent_vec(Func f, GradFunc grad,
                                          const std::array<float, N>& x0,
                                          float step = 0.1f, float tol = 1e-6f, int max_iter = 1000) {
    std::array<float, N> x = x0;
    auto g = grad(x);
    float norm_g = 0.0f;
    for (int i = 0; i < max_iter; ++i) {
        norm_g = 0.0f;
        for (std::size_t j = 0; j < N; ++j) norm_g += g[j] * g[j];
        norm_g = std::sqrt(norm_g);
        if (norm_g < tol) break;
        for (std::size_t j = 0; j < N; ++j) x[j] -= step * g[j];
        g = grad(x);
    }
    return x;
}

// ============================================================
// Conjugate gradient (Fletcher‑Reeves) for quadratic forms
// ============================================================

template<typename Func, typename GradFunc, std::size_t N>
std::array<float, N> conjugate_gradient(Func f, GradFunc grad,
                                        const std::array<float, N>& x0,
                                        float tol = 1e-6f, int max_iter = 200) {
    std::array<float, N> x = x0;
    auto g = grad(x);
    std::array<float, N> d = g;
    for (std::size_t j = 0; j < N; ++j) d[j] = -d[j];
    float g_norm_sq = 0.0f;
    for (std::size_t j = 0; j < N; ++j) g_norm_sq += g[j] * g[j];
    for (int i = 0; i < max_iter && std::sqrt(g_norm_sq) > tol; ++i) {
        // line search (exact for quadratic – we approximate)
        float t = 0.1f;
        float fx = f(x);
        std::array<float, N> x_new;
        float best = fx;
        float best_t = 0.0f;
        for (int ls = 0; ls < 20; ++ls) {
            for (std::size_t j = 0; j < N; ++j) x_new[j] = x[j] + t * d[j];
            float val = f(x_new);
            if (val < best) { best = val; best_t = t; }
            t *= 0.5f;
        }
        for (std::size_t j = 0; j < N; ++j) x[j] += best_t * d[j];
        auto g_new = grad(x);
        float g_new_norm_sq = 0.0f;
        for (std::size_t j = 0; j < N; ++j) g_new_norm_sq += g_new[j] * g_new[j];
        float beta_fr = g_new_norm_sq / (g_norm_sq + 1e-12f);
        for (std::size_t j = 0; j < N; ++j) d[j] = -g_new[j] + beta_fr * d[j];
        g = g_new;
        g_norm_sq = g_new_norm_sq;
    }
    return x;
}

// ============================================================
// Newton's method for root finding (1D)
// ============================================================

template<typename Func, typename DerivFunc>
float newton_raphson(Func f, DerivFunc df, float x0, float tol = 1e-6f, int max_iter = 100) {
    float x = x0;
    for (int i = 0; i < max_iter; ++i) {
        float fx = f(x);
        float dfx = df(x);
        if (std::abs(dfx) < 1e-12f) break;
        float x_new = x - fx / dfx;
        if (std::abs(x_new - x) < tol) return x_new;
        x = x_new;
    }
    return x;
}

// ============================================================
// Secant method for root finding (1D)
// ============================================================

template<typename Func>
float secant_method(Func f, float x0, float x1, float tol = 1e-6f, int max_iter = 100) {
    float f0 = f(x0), f1 = f(x1);
    for (int i = 0; i < max_iter; ++i) {
        if (std::abs(f1 - f0) < 1e-12f) break;
        float x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        if (std::abs(x2 - x1) < tol) return x2;
        x0 = x1; f0 = f1;
        x1 = x2; f1 = f(x2);
    }
    return x1;
}

// ============================================================
// BFGS (Broyden‑Fletcher‑Goldfarb‑Shanno) for N dimensions
// ============================================================

template<typename Func, typename GradFunc>
std::vector<float> bfgs(Func f, GradFunc grad, const std::vector<float>& x0,
                        float tol = 1e-6f, int max_iter = 500) {
    std::size_t n = x0.size();
    std::vector<float> x = x0;
    std::vector<float> g(n);
    // compute initial gradient
    {
        auto g0 = grad(x);
        for (std::size_t i = 0; i < n; ++i) g[i] = g0[i];
    }
    // initial inverse Hessian approximation = I
    std::vector<float> H(n * n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) H[i * n + i] = 1.0f;

    auto dot_vec = [&](const std::vector<float>& a, const std::vector<float>& b) {
        float s = 0.0f;
        for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
        return s;
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        float g_norm = std::sqrt(dot_vec(g, g));
        if (g_norm < tol) break;

        // search direction p = -H * g
        std::vector<float> p(n, 0.0f);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                p[i] -= H[i * n + j] * g[j];

        // line search (backtracking)
        float alpha = 1.0f;
        float fx = f(x);
        float c1 = 1e-4f;
        std::vector<float> x_new(n);
        while (true) {
            for (std::size_t i = 0; i < n; ++i) x_new[i] = x[i] + alpha * p[i];
            float f_new = f(x_new);
            float expected = fx + c1 * alpha * dot_vec(g, p);
            if (f_new <= expected || alpha < 1e-12f) break;
            alpha *= 0.5f;
        }
        for (std::size_t i = 0; i < n; ++i) x_new[i] = x[i] + alpha * p[i];

        std::vector<float> g_new(n);
        {
            auto g0_new = grad(x_new);
            for (std::size_t i = 0; i < n; ++i) g_new[i] = g0_new[i];
        }

        std::vector<float> s(n), y(n);
        for (std::size_t i = 0; i < n; ++i) { s[i] = x_new[i] - x[i]; y[i] = g_new[i] - g[i]; }
        float rho = 1.0f / (dot_vec(y, s) + 1e-12f);

        // BFGS update: H = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
        std::vector<float> Hy(n, 0.0f);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                Hy[i] += H[i * n + j] * y[j];

        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                H[i * n + j] += rho * s[i] * s[j];
        float rho_sy = rho * dot_vec(s, y);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                H[i * n + j] -= rho * (Hy[i] * s[j] + s[i] * Hy[j]) - rho_sy * Hy[i] * s[j];

        x = x_new;
        g = g_new;
    }
    return x;
}

// ============================================================
// Nelder‑Mead simplex (derivative‑free)
// ============================================================

template<typename Func>
std::vector<float> nelder_mead(Func f, std::vector<float> x0,
                               float alpha = 1.0f, float gamma = 2.0f,
                               float rho = 0.5f, float sigma = 0.5f,
                               float tol = 1e-6f, int max_iter = 1000) {
    std::size_t n = x0.size();
    std::vector<std::vector<float>> simplex(n + 1, x0);
    for (std::size_t i = 0; i < n; ++i) simplex[i][i] += (simplex[i][i] == 0.0f) ? 0.1f : simplex[i][i] * 1.1f;
    std::vector<float> fvals(n + 1);
    for (std::size_t i = 0; i <= n; ++i) fvals[i] = f(simplex[i]);

    auto centroid = [&](const std::vector<std::vector<float>>& pts, std::size_t exclude) {
        std::vector<float> c(n, 0.0f);
        for (std::size_t i = 0; i <= n; ++i) {
            if (i == exclude) continue;
            for (std::size_t j = 0; j < n; ++j) c[j] += pts[i][j];
        }
        for (std::size_t j = 0; j < n; ++j) c[j] /= static_cast<float>(n);
        return c;
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        // sort by f
        std::vector<std::size_t> idx(n+1);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return fvals[a] < fvals[b]; });
        if (std::sqrt(fvals[idx[n]] - fvals[idx[0]]) < tol) break;

        auto xc = centroid(simplex, idx[n]);
        auto xr = xc;
        for (std::size_t j = 0; j < n; ++j) xr[j] += alpha * (xc[j] - simplex[idx[n]][j]);
        float fr = f(xr);

        if (fvals[idx[0]] <= fr && fr < fvals[idx[n-1]]) {
            simplex[idx[n]] = xr; fvals[idx[n]] = fr;
        } else if (fr < fvals[idx[0]]) {
            auto xe = xc;
            for (std::size_t j = 0; j < n; ++j) xe[j] += gamma * (xr[j] - xc[j]);
            float fe = f(xe);
            if (fe < fr) { simplex[idx[n]] = xe; fvals[idx[n]] = fe; }
            else { simplex[idx[n]] = xr; fvals[idx[n]] = fr; }
        } else {
            if (fr < fvals[idx[n]]) {
                simplex[idx[n]] = xr; fvals[idx[n]] = fr;
            }
            auto xc2 = centroid(simplex, idx[n]);
            auto xco = xc2;
            for (std::size_t j = 0; j < n; ++j) xco[j] += rho * (simplex[idx[n]][j] - xc2[j]);
            float fco = f(xco);
            if (fco < fvals[idx[n]]) {
                simplex[idx[n]] = xco; fvals[idx[n]] = fco;
            } else {
                for (std::size_t i = 0; i <= n; ++i) {
                    if (i == idx[0]) continue;
                    for (std::size_t j = 0; j < n; ++j)
                        simplex[i][j] = simplex[idx[0]][j] + sigma * (simplex[i][j] - simplex[idx[0]][j]);
                    fvals[i] = f(simplex[i]);
                }
            }
        }
    }
    // return best
    auto best = std::min_element(fvals.begin(), fvals.end()) - fvals.begin();
    return simplex[best];
}

// ============================================================
// Bisection root finding
// ============================================================

template<typename Func>
float bisection(Func f, float a, float b, float tol = 1e-6f, int max_iter = 100) {
    float fa = f(a), fb = f(b);
    if (fa * fb > 0) return std::numeric_limits<float>::quiet_NaN();
    float c;
    for (int i = 0; i < max_iter; ++i) {
        c = (a + b) * 0.5f;
        float fc = f(c);
        if (std::abs(fc) < tol) return c;
        if (fa * fc < 0) { b = c; fb = fc; }
        else { a = c; fa = fc; }
    }
    return c;
}

// ============================================================
// False position (regula falsi)
// ============================================================

template<typename Func>
float false_position(Func f, float a, float b, float tol = 1e-6f, int max_iter = 100) {
    float fa = f(a), fb = f(b);
    if (fa * fb > 0) return std::numeric_limits<float>::quiet_NaN();
    float c = a;
    for (int i = 0; i < max_iter; ++i) {
        c = (a * fb - b * fa) / (fb - fa);
        float fc = f(c);
        if (std::abs(fc) < tol) return c;
        if (fa * fc < 0) { b = c; fb = fc; }
        else { a = c; fa = fc; }
    }
    return c;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_OPTIMIZATION_H