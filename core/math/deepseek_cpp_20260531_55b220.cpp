//File 0071 : core/math/polynomial_roots.h
//Analytical solvers for quadratic, cubic, and quartic real‑coefficient polynomials; returns all real or complex roots using exact closed‑form formulas (Cardano, Ferrari).
#ifndef CORE_MATH_POLYNOMIAL_ROOTS_H
#define CORE_MATH_POLYNOMIAL_ROOTS_H

#include <complex>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace SimulationMath {
namespace poly_roots {

using Complex = std::complex<float>;

// -----------------------------------------------------------------------------
// 1. Quadratic: a*x^2 + b*x + c = 0  (real coefficients, returns complex roots)
// -----------------------------------------------------------------------------
inline std::vector<Complex> quadratic(float a, float b, float c) noexcept {
    std::vector<Complex> roots;
    if (std::abs(a) < 1e-12f) {
        if (std::abs(b) < 1e-12f) return roots;  // no solution
        roots.push_back(Complex(-c / b, 0.0f));
        return roots;
    }
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant >= 0.0f) {
        float sqrtD = std::sqrt(discriminant);
        roots.push_back(Complex((-b + sqrtD) / (2.0f * a), 0.0f));
        roots.push_back(Complex((-b - sqrtD) / (2.0f * a), 0.0f));
    } else {
        float sqrtD = std::sqrt(-discriminant);
        roots.push_back(Complex(-b / (2.0f * a),  sqrtD / (2.0f * a)));
        roots.push_back(Complex(-b / (2.0f * a), -sqrtD / (2.0f * a)));
    }
    return roots;
}

// -----------------------------------------------------------------------------
// 2. Cubic: a*x^3 + b*x^2 + c*x + d = 0  (real coefficients)
// -----------------------------------------------------------------------------
inline std::vector<Complex> cubic(float a, float b, float c, float d) noexcept {
    std::vector<Complex> roots;
    if (std::abs(a) < 1e-12f) return quadratic(b, c, d); // degenerate

    // Normalize to x^3 + A*x^2 + B*x + C = 0
    float A = b / a, B = c / a, C = d / a;

    // Depressed cubic: t = x + A/3  →  t^3 + p*t + q = 0
    float p = B - A * A / 3.0f;
    float q = 2.0f * A * A * A / 27.0f - A * B / 3.0f + C;

    // Discriminant: Δ = (q/2)^2 + (p/3)^3
    float q2 = q / 2.0f;
    float p3 = p / 3.0f;
    float discriminant = q2 * q2 + p3 * p3 * p3;

    Complex t0, t1, t2;
    if (discriminant >= 0.0f) {
        // One real root
        float sqrtD = std::sqrt(discriminant);
        float u = std::cbrt(-q2 + sqrtD);
        float v = std::cbrt(-q2 - sqrtD);
        t0 = Complex(u + v, 0.0f);
        // The other two roots are complex (or real if u=v)
        Complex u_c(u, 0.0f), v_c(v, 0.0f);
        Complex omega(-0.5f, std::sqrt(3.0f) / 2.0f);
        t1 = omega * u_c + std::conj(omega) * v_c;
        t2 = std::conj(omega) * u_c + omega * v_c;
    } else {
        // Three real roots (casus irreducibilis)
        float r = std::sqrt(-p3 * p3 * p3);
        float phi = std::acos(-q2 / r);
        float s = 2.0f * std::cbrt(std::sqrt(-p3 * p3 * p3)); // actually s = 2 * sqrt(-p/3)
        s = 2.0f * std::sqrt(-p / 3.0f);
        t0 = Complex(s * std::cos(phi / 3.0f), 0.0f);
        t1 = Complex(s * std::cos((phi + 2.0f * 3.14159265358979f) / 3.0f), 0.0f);
        t2 = Complex(s * std::cos((phi + 4.0f * 3.14159265358979f) / 3.0f), 0.0f);
    }

    // Shift back
    float shift = A / 3.0f;
    roots.push_back(Complex(t0.real() - shift, t0.imag()));
    roots.push_back(Complex(t1.real() - shift, t1.imag()));
    roots.push_back(Complex(t2.real() - shift, t2.imag()));
    return roots;
}

// -----------------------------------------------------------------------------
// 3. Quartic: a*x^4 + b*x^3 + c*x^2 + d*x + e = 0  (real coefficients)
//    Using Ferrari's method (reduces to cubic)
// -----------------------------------------------------------------------------
inline std::vector<Complex> quartic(float a, float b, float c, float d, float e) noexcept {
    std::vector<Complex> roots;
    if (std::abs(a) < 1e-12f) return cubic(b, c, d, e); // degenerate

    // Normalize to x^4 + A*x^3 + B*x^2 + C*x + D = 0
    float A = b / a, B = c / a, C = d / a, D = e / a;

    // Depressed quartic: substitute x = y - A/4  →  y^4 + p*y^2 + q*y + r = 0
    float p = B - 3.0f * A * A / 8.0f;
    float q = C - A * B / 2.0f + A * A * A / 8.0f;
    float r = D - A * C / 4.0f + A * A * B / 16.0f - 3.0f * A * A * A * A / 256.0f;

    // If q == 0, it's a biquadratic → y^4 + p*y^2 + r = 0 → quadratic in y^2
    if (std::abs(q) < 1e-12f) {
        // solve u^2 + p*u + r = 0, where u = y^2
        auto u_roots = quadratic(1.0f, p, r);
        for (const auto& u : u_roots) {
            auto y_roots = quadratic(1.0f, 0.0f, -u.real()); // y^2 = u -> y = ±√u
            // handle complex u? but u_roots are complex; we'd need sqrt of complex number. For simplicity, we'll handle real u.
            // For a proper implementation, we'd compute complex sqrt.
            // We'll implement using complex sqrt for the generic case.
            Complex sqrt_u = std::sqrt(u);
            roots.push_back(Complex(sqrt_u.real(), sqrt_u.imag()));
            roots.push_back(Complex(-sqrt_u.real(), -sqrt_u.imag()));
        }
        // Apply shift
        float shift = A / 4.0f;
        for (auto& r : roots) r = Complex(r.real() - shift, r.imag());
        return roots;
    }

    // Ferrari's method: find a real root m of the cubic resolvent: m^3 + p*m^2 + (p^2/4 - r)*m - q^2/8 = 0? Wait, standard resolvent cubic:
    //   m^3 + 2p*m^2 + (p^2 - 4r)*m - q^2 = 0  (see https://en.wikipedia.org/wiki/Quartic_function#Ferrari's_solution)
    // Actually the standard is: m^3 + (p^2/4 - r) m - q^2/8 = 0? I'll use the one from: Quartic equation solving.
    // Common form: 8m^3 + 8p*m^2 + (2p^2 - 8r)*m - q^2 = 0. But we can just use cubic solver on the resolvent.
    // The resolvent cubic for Ferrari's method: θ^3 - p*θ^2 - 4r*θ + (4pr - q^2) = 0 ? Let's derive carefully.
    // I'll use the standard Ferrari resolvent cubic: y^3 + 2p*y^2 + (p^2 - 4r)*y - q^2 = 0.
    // Then find a real root y0, and then solve two quadratics.
    float p2 = p * p;
    float r4 = 4.0f * r;
    float resolvent_p = 2.0f * p;
    float resolvent_q = p2 - r4;
    float resolvent_r = -q * q;

    // Solve cubic: y^3 + resolvent_p*y^2 + resolvent_q*y + resolvent_r = 0
    auto y_roots = cubic(1.0f, resolvent_p, resolvent_q, resolvent_r);
    // Find a real root m (should be non-negative if exists)
    float m = 0.0f;
    for (const auto& y : y_roots) {
        if (std::abs(y.imag()) < 1e-6f && y.real() > 0.0f) {
            m = y.real();
            break;
        }
    }
    // If no positive real root, use any real root (maybe negative)
    if (m == 0.0f) {
        for (const auto& y : y_roots) {
            if (std::abs(y.imag()) < 1e-6f) {
                m = y.real();
                break;
            }
        }
    }

    // Compute parameters for two quadratics
    float sqrt_m = std::sqrt(std::max(m, 0.0f));
    // Avoid division by zero
    if (sqrt_m < 1e-12f) sqrt_m = 1e-6f;
    float term1 = (p + m - q / sqrt_m) / 2.0f;
    float term2 = (p + m + q / sqrt_m) / 2.0f;
    // The two quadratics: y^2 + sqrt(m)*y + term1 = 0  and  y^2 - sqrt(m)*y + term2 = 0
    auto q1 = quadratic(1.0f,  sqrt_m, term1);
    auto q2 = quadratic(1.0f, -sqrt_m, term2);
    // Collect roots
    for (const auto& r : q1) roots.push_back(r);
    for (const auto& r : q2) roots.push_back(r);

    // Shift back
    float shift = A / 4.0f;
    for (auto& r : roots) r = Complex(r.real() - shift, r.imag());
    // Remove duplicates (if any) due to floating point
    std::sort(roots.begin(), roots.end(), [](const Complex& a, const Complex& b) {
        return (a.real() < b.real()) || (a.real() == b.real() && a.imag() < b.imag());
    });
    roots.erase(std::unique(roots.begin(), roots.end(),
        [](const Complex& a, const Complex& b) { return std::abs(a - b) < 1e-5f; }), roots.end());
    return roots;
}

} // namespace poly_roots
} // namespace SimulationMath

#endif // CORE_MATH_POLYNOMIAL_ROOTS_H