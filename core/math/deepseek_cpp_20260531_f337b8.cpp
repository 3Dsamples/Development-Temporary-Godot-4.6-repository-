//File 0042 : core/math/optimization.h
//Advanced optimization: golden‑section line search, Nelder‑Mead simplex, BFGS quasi‑Newton, L‑BFGS, trust‑region, and Levenberg‑Marquardt, all supporting scalar and vector‑valued objective functions.
#ifndef CORE_MATH_OPTIMIZATION_H
#define CORE_MATH_OPTIMIZATION_H

#include "vector_math.h"
#include "linear_algebra.h"          // for Eigen integration
#include <Eigen/Dense>
#include <functional>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

namespace SimulationMath {
namespace opt {

using EigenVector = Eigen::VectorXf;
using EigenMatrix = Eigen::MatrixXf;

// -----------------------------------------------------------------------------
// 1. Golden‑section line search for a 1D unimodal function
// -----------------------------------------------------------------------------
inline float golden_section_min(const std::function<float(float)>& f, float a, float b, float tol = 1e-6f) noexcept {
    const float inv_phi = 0.6180339887498949f;
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
// 2. Nelder‑Mead simplex (derivative‑free, unconstrained)
// -----------------------------------------------------------------------------
inline EigenVector nelder_mead(const std::function<float(const EigenVector&)>& f,
                                const std::vector<EigenVector>& initial_simplex,
                                float tol = 1e-6f, int max_iter = 200) noexcept {
    const float alpha = 1.0f, gamma = 2.0f, rho = 0.5f, sigma = 0.5f;
    size_t n = initial_simplex.size() - 1;
    std::vector<EigenVector> sim = initial_simplex;
    std::vector<float> fvals(sim.size());
    for (size_t i = 0; i < sim.size(); ++i) fvals[i] = f(sim[i]);

    auto get_order = [&]() {
        std::vector<size_t> idx(sim.size());
        for (size_t i = 0; i < sim.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return fvals[a] < fvals[b]; });
        return idx;
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        auto ord = get_order();
        size_t best = ord[0], worst = ord[n], second_worst = ord[n-1];
        EigenVector x0 = EigenVector::Zero(n);
        for (size_t i = 0; i < n; ++i) x0 += sim[ord[i]];
        x0 /= n;

        // reflection
        EigenVector xr = x0 + alpha * (x0 - sim[worst]);
        float fr = f(xr);
        if (fr < fvals[second_worst] && fr >= fvals[best]) {
            sim[worst] = xr; fvals[worst] = fr;
        } else if (fr < fvals[best]) {
            // expansion
            EigenVector xe = x0 + gamma * (xr - x0);
            float fe = f(xe);
            if (fe < fr) { sim[worst] = xe; fvals[worst] = fe; }
            else { sim[worst] = xr; fvals[worst] = fr; }
        } else if (fr < fvals[worst]) {
            // outside contraction
            EigenVector xc = x0 + rho * (xr - x0);
            float fc = f(xc);
            if (fc <= fr) { sim[worst] = xc; fvals[worst] = fc; }
            else { // shrink
                for (size_t i = 0; i < sim.size(); ++i) {
                    if (i != best) {
                        sim[i] = sim[best] + sigma * (sim[i] - sim[best]);
                        fvals[i] = f(sim[i]);
                    }
                }
            }
        } else {
            // inside contraction
            EigenVector xc = x0 - rho * (x0 - sim[worst]);
            float fc = f(xc);
            if (fc < fvals[worst]) { sim[worst] = xc; fvals[worst] = fc; }
            else { // shrink
                for (size_t i = 0; i < sim.size(); ++i) {
                    if (i != best) {
                        sim[i] = sim[best] + sigma * (sim[i] - sim[best]);
                        fvals[i] = f(sim[i]);
                    }
                }
            }
        }
        // convergence check
        float max_diff = 0.0f;
        for (size_t i = 1; i < sim.size(); ++i)
            max_diff = std::max(max_diff, (sim[i] - sim[best]).norm());
        if (max_diff < tol) break;
    }
    auto ord = get_order();
    return sim[ord[0]];
}

// -----------------------------------------------------------------------------
// 3. BFGS (Broyden–Fletcher–Goldfarb–Shanno) with line search
// -----------------------------------------------------------------------------
inline EigenVector bfgs(const std::function<float(const EigenVector&)>& f,
                        const std::function<EigenVector(const EigenVector&)>& grad,
                        const EigenVector& x0, int max_iter = 100,
                        float tol = 1e-6f) noexcept {
    size_t n = x0.size();
    EigenMatrix H = EigenMatrix::Identity(n, n);
    EigenVector x = x0;
    EigenVector g = grad(x);
    for (int iter = 0; iter < max_iter; ++iter) {
        EigenVector p = -H * g;
        // line search (backtracking Armijo)
        float step = 1.0f;
        float c = 1e-4f;
        float fx = f(x);
        while (f(x + step * p) > fx + c * step * g.dot(p) && step > 1e-12f)
            step *= 0.5f;
        EigenVector s = step * p;
        x += s;
        EigenVector g_new = grad(x);
        if (g_new.norm() < tol) break;
        EigenVector y = g_new - g;
        float rho = 1.0f / y.dot(s);
        H = (EigenMatrix::Identity(n, n) - rho * s * y.transpose()) *
            H * (EigenMatrix::Identity(n, n) - rho * y * s.transpose()) +
            rho * s * s.transpose();
        g = g_new;
    }
    return x;
}

// -----------------------------------------------------------------------------
// 4. L‑BFGS (limited‑memory) with m past vectors
// -----------------------------------------------------------------------------
inline EigenVector lbfgs(const std::function<float(const EigenVector&)>& f,
                         const std::function<EigenVector(const EigenVector&)>& grad,
                         const EigenVector& x0, int m = 5, int max_iter = 100,
                         float tol = 1e-6f) noexcept {
    size_t n = x0.size();
    EigenVector x = x0;
    EigenVector g = grad(x);
    std::vector<EigenVector> s_list, y_list;
    std::vector<float> rho_list;
    for (int iter = 0; iter < max_iter; ++iter) {
        // compute search direction using two‑loop recursion
        EigenVector q = g;
        int k = s_list.size();
        std::vector<float> alpha(k);
        for (int i = k-1; i >= 0; --i) {
            alpha[i] = rho_list[i] * s_list[i].dot(q);
            q -= alpha[i] * y_list[i];
        }
        EigenVector r = q; // approximate H0 * q, using H0 = gamma * I
        float gamma = (k > 0) ? s_list.back().dot(y_list.back()) / y_list.back().squaredNorm() : 1.0f;
        r *= gamma;
        for (int i = 0; i < k; ++i) {
            float beta = rho_list[i] * y_list[i].dot(r);
            r += s_list[i] * (alpha[i] - beta);
        }
        EigenVector p = -r;
        // line search
        float step = 1.0f;
        float c = 1e-4f;
        float fx = f(x);
        while (f(x + step * p) > fx + c * step * g.dot(p) && step > 1e-12f)
            step *= 0.5f;
        EigenVector s = step * p;
        x += s;
        EigenVector g_new = grad(x);
        if (g_new.norm() < tol) break;
        EigenVector y = g_new - g;
        float rho_y = 1.0f / y.dot(s);
        if (s_list.size() >= (size_t)m) {
            s_list.erase(s_list.begin());
            y_list.erase(y_list.begin());
            rho_list.erase(rho_list.begin());
        }
        s_list.push_back(s);
        y_list.push_back(y);
        rho_list.push_back(rho_y);
        g = g_new;
    }
    return x;
}

// -----------------------------------------------------------------------------
// 5. Levenberg–Marquardt for non‑linear least squares
// -----------------------------------------------------------------------------
inline EigenVector levenberg_marquardt(
    const std::function<float(const EigenVector&)>& residual_norm,
    const std::function<EigenVector(const EigenVector&)>& residual,
    const std::function<EigenMatrix(const EigenVector&)>& jacobian,
    const EigenVector& x0, int max_iter = 50, float tol = 1e-6f) noexcept {
    EigenVector x = x0;
    float lambda = 0.01f;
    for (int iter = 0; iter < max_iter; ++iter) {
        EigenVector r = residual(x);
        EigenMatrix J = jacobian(x);
        EigenMatrix A = J.transpose() * J + lambda * EigenMatrix::Identity(x.size(), x.size());
        EigenVector b = -J.transpose() * r;
        EigenVector delta = linalg::solve_cholesky(A, b); // Cholesky since A is SPD for lambda>0
        float new_norm = residual_norm(x + delta);
        float old_norm = residual_norm(x);
        if (new_norm < old_norm) {
            x += delta;
            lambda *= 0.8f;
            if (delta.norm() < tol) break;
        } else {
            lambda *= 2.0f;
        }
    }
    return x;
}

// -----------------------------------------------------------------------------
// 6. Trust‑region method for unconstrained minimisation
// -----------------------------------------------------------------------------
inline EigenVector trust_region(const std::function<float(const EigenVector&)>& f,
                                const std::function<EigenVector(const EigenVector&)>& grad,
                                const std::function<EigenMatrix(const EigenVector&)>& hessian,
                                const EigenVector& x0, float delta0 = 1.0f,
                                int max_iter = 50, float tol = 1e-6f) noexcept {
    EigenVector x = x0;
    float delta = delta0;
    for (int iter = 0; iter < max_iter; ++iter) {
        EigenVector g = grad(x);
        EigenMatrix H = hessian(x);
        // Solve sub‑problem: min_{||p|| <= delta} g^T p + 0.5 p^T H p
        // Use dogleg or Cauthy point approximation. Here we use a simple 2D subspace.
        EigenVector pB = -H.colPivHouseholderQr().solve(g); // Newton step, may fail
        float norm_pB = pB.norm();
        EigenVector p;
        if (norm_pB <= delta) {
            p = pB;
        } else {
            EigenVector pU = -(g.dot(g) / (g.dot(H * g) + 1e-12f)) * g;
            float norm_pU = pU.norm();
            if (norm_pU >= delta) {
                p = (delta / norm_pU) * pU;
            } else {
                EigenVector diff = pB - pU;
                float a = diff.squaredNorm();
                float b = 2.0f * pU.dot(diff);
                float c = pU.squaredNorm() - delta*delta;
                float tau = (std::sqrt(b*b - 4.0f*a*c) - b) / (2.0f*a);
                p = pU + tau * diff;
            }
        }
        float rho = (f(x) - f(x+p)) / (-g.dot(p) - 0.5f * p.dot(H * p) + 1e-12f);
        if (rho > 0.75f) delta = std::min(2.0f * delta, 10.0f);
        else if (rho < 0.25f) delta *= 0.5f;
        if (rho > 0.0f) x += p;
        if (g.norm() < tol) break;
    }
    return x;
}

} // namespace opt
} // namespace SimulationMath

#endif // CORE_MATH_OPTIMIZATION_H