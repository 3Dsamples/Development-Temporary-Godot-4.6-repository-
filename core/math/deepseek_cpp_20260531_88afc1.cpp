//File 0041 : core/math/curve.h
//Parametric curve types: polyline, Bezier, Catmull‑Rom, B‑spline, NURBS (non‑uniform rational B‑spline), with evaluation, derivative, arc‑length parameterisation, and closest point queries.
#ifndef CORE_MATH_CURVE_H
#define CORE_MATH_CURVE_H

#include "vector_math.h"
#include "interpolation.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace SimulationMath {
namespace curve {

// -----------------------------------------------------------------------------
// 1. Basic parametric curve abstract interface (C++17 template‑free)
// -----------------------------------------------------------------------------
class ParametricCurve {
public:
    virtual ~ParametricCurve() = default;
    virtual DirectX::XMVECTOR evaluate(float t) const noexcept = 0;
    virtual DirectX::XMVECTOR derivative(float t) const noexcept { // finite difference fallback
        const float eps = 1e-4f;
        DirectX::XMVECTOR p0 = evaluate(t - eps);
        DirectX::XMVECTOR p1 = evaluate(t + eps);
        return DirectX::XMVectorScale(DirectX::XMVectorSubtract(p1, p0), 1.0f/(2.0f*eps));
    }
    virtual float arc_length(float t0, float t1, int segments = 64) const noexcept {
        float len = 0.0f;
        float dt = (t1 - t0) / segments;
        DirectX::XMVECTOR prev = evaluate(t0);
        for (int i = 1; i <= segments; ++i) {
            float t = t0 + i * dt;
            DirectX::XMVECTOR curr = evaluate(t);
            len += vector_math::length3_scalar(DirectX::XMVectorSubtract(curr, prev));
            prev = curr;
        }
        return len;
    }
};

// -----------------------------------------------------------------------------
// 2. Polyline curve (linear interpolation between vertices)
// -----------------------------------------------------------------------------
class PolylineCurve : public ParametricCurve {
    std::vector<DirectX::XMVECTOR> pts_;
public:
    explicit PolylineCurve(const std::vector<DirectX::XMVECTOR>& points) : pts_(points) {}
    DirectX::XMVECTOR evaluate(float t) const noexcept override {
        if (pts_.empty()) return DirectX::XMVectorZero();
        if (pts_.size() == 1) return pts_[0];
        size_t n = pts_.size() - 1;
        float s = std::max(0.0f, std::min(t, 1.0f)) * n;
        size_t idx = std::min(static_cast<size_t>(s), n-1);
        float frac = s - idx;
        return interpolation::lerp(pts_[idx], pts_[idx+1], frac);
    }
};

// -----------------------------------------------------------------------------
// 3. Bezier curve (arbitrary degree via de Casteljau)
// -----------------------------------------------------------------------------
class BezierCurve : public ParametricCurve {
    std::vector<DirectX::XMVECTOR> control_;
public:
    explicit BezierCurve(const std::vector<DirectX::XMVECTOR>& ctrl) : control_(ctrl) {}
    DirectX::XMVECTOR evaluate(float t) const noexcept override {
        return interpolation::bezier_evaluate(control_, t);
    }
    DirectX::XMVECTOR derivative(float t) const noexcept override {
        if (control_.size() < 2) return DirectX::XMVectorZero();
        size_t n = control_.size() - 1;
        std::vector<DirectX::XMVECTOR> dpoints(n);
        for (size_t i = 0; i < n; ++i)
            dpoints[i] = DirectX::XMVectorScale(DirectX::XMVectorSubtract(control_[i+1], control_[i]), static_cast<float>(n));
        return interpolation::bezier_evaluate(dpoints, t);
    }
};

// -----------------------------------------------------------------------------
// 4. Catmull‑Rom curve (interpolating control points)
// -----------------------------------------------------------------------------
class CatmullRomCurve : public ParametricCurve {
    std::vector<DirectX::XMVECTOR> pts_;
    float alpha_;
public:
    CatmullRomCurve(const std::vector<DirectX::XMVECTOR>& points, float alpha = 0.5f) : pts_(points), alpha_(alpha) {}
    DirectX::XMVECTOR evaluate(float t) const noexcept override {
        if (pts_.size() < 2) return pts_.empty() ? DirectX::XMVectorZero() : pts_[0];
        size_t n = pts_.size() - 1;
        float s = std::max(0.0f, std::min(t, 1.0f)) * n;
        size_t idx = std::min(static_cast<size_t>(s), n-1);
        float frac = s - idx;
        DirectX::XMVECTOR p0 = (idx == 0) ? pts_[0] : pts_[idx-1];
        DirectX::XMVECTOR p1 = pts_[idx];
        DirectX::XMVECTOR p2 = pts_[idx+1];
        DirectX::XMVECTOR p3 = (idx+2 < pts_.size()) ? pts_[idx+2] : pts_[idx+1];
        return interpolation::catmull_rom(p0, p1, p2, p3, frac, alpha_);
    }
};

// -----------------------------------------------------------------------------
// 5. Uniform cubic B‑Spline
// -----------------------------------------------------------------------------
class UniformBSpline : public ParametricCurve {
    std::vector<DirectX::XMVECTOR> control_;
public:
    explicit UniformBSpline(const std::vector<DirectX::XMVECTOR>& ctrl) : control_(ctrl) {}
    DirectX::XMVECTOR evaluate(float t) const noexcept override {
        if (control_.size() < 4) return control_.empty() ? DirectX::XMVectorZero() : control_[0];
        size_t num_segments = control_.size() - 3;
        float s = std::max(0.0f, std::min(t, 1.0f)) * num_segments;
        size_t idx = std::min(static_cast<size_t>(s), num_segments-1);
        float frac = s - idx;
        DirectX::XMVECTOR p0 = control_[idx];
        DirectX::XMVECTOR p1 = control_[idx+1];
        DirectX::XMVECTOR p2 = control_[idx+2];
        DirectX::XMVECTOR p3 = control_[idx+3];
        return interpolation::cubic_bspline(p0, p1, p2, p3, frac);
    }
};

// -----------------------------------------------------------------------------
// 6. NURBS curve (non‑uniform rational B‑spline)
// -----------------------------------------------------------------------------
class NURBSCurve : public ParametricCurve {
    std::vector<DirectX::XMVECTOR> control_; // homogeneous (x,y,z,w) where w is weight
    std::vector<float> knots_;
    int degree_;
public:
    NURBSCurve(const std::vector<DirectX::XMVECTOR>& weighted_ctrl,
               const std::vector<float>& knots, int degree)
        : control_(weighted_ctrl), knots_(knots), degree_(degree) {}

    DirectX::XMVECTOR evaluate(float t) const noexcept override {
        // Find span
        int n = static_cast<int>(control_.size()) - 1;
        if (n < degree_) return DirectX::XMVectorZero();
        float clamped_t = std::max(knots_[degree_], std::min(t, knots_[n+1]));
        int span = degree_;
        while (span < n && knots_[span+1] < clamped_t) ++span;
        // Compute basis functions
        float N[32]; // support up to degree 31
        float Nd[32];
        for (int i = 0; i <= degree_; ++i) {
            N[i] = (clamped_t >= knots_[span - degree_ + i] && clamped_t <= knots_[span - degree_ + i + 1]) ? 1.0f : 0.0f;
        }
        for (int d = 1; d <= degree_; ++d) {
            for (int i = 0; i <= degree_ - d; ++i) {
                int ki = span - degree_ + i;
                float denom1 = knots_[ki + d] - knots_[ki];
                float denom2 = knots_[ki + d + 1] - knots_[ki + 1];
                float left = (denom1 > 1e-12f) ? (clamped_t - knots_[ki]) / denom1 * N[i] : 0.0f;
                float right = (denom2 > 1e-12f) ? (knots_[ki + d + 1] - clamped_t) / denom2 * N[i+1] : 0.0f;
                Nd[i] = left + right;
            }
            for (int i = 0; i <= degree_ - d; ++i) N[i] = Nd[i];
        }
        // Compute point as weighted sum
        DirectX::XMVECTOR result = DirectX::XMVectorZero();
        for (int i = 0; i <= degree_; ++i) {
            int idx = span - degree_ + i;
            DirectX::XMVECTOR weighted = DirectX::XMVectorScale(control_[idx], N[i]);
            result = DirectX::XMVectorAdd(result, weighted);
        }
        // Divide by w component
        float w = vector_math::get_w(result);
        if (std::abs(w) > 1e-12f) {
            result = DirectX::XMVectorScale(result, 1.0f / w);
            result = DirectX::XMVectorSetW(result, 1.0f);
        }
        return result;
    }
};

} // namespace curve
} // namespace SimulationMath

#endif // CORE_MATH_CURVE_H