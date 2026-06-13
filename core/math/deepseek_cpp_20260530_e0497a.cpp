// File 0034 : core/math/polynomial.h
// Polynomial evaluation, differentiation, and analytic roots (quadratic, cubic, quartic) plus Durand-Kerner.

#pragma once

#include "constants.h"
#include <complex>
#include <vector>
#include <cmath>
#include <limits>

namespace wp {

// ── Evaluate polynomial P(x) = a[0] + a[1]*x + ... + a[n]*x^n ──────
template <typename T>
T polyval(const std::vector<T>& coeffs, T x) {
    if (coeffs.empty()) return T(0);
    T result = coeffs.back();
    for (int i = static_cast<int>(coeffs.size()) - 2; i >= 0; --i)
        result = result * x + coeffs[i];
    return result;
}

// Evaluate polynomial and its derivative simultaneously (Horner)
template <typename T>
std::pair<T, T> polyval_d(const std::vector<T>& coeffs, T x) {
    if (coeffs.empty()) return {T(0), T(0)};
    T val = coeffs.back();
    T d   = T(0);
    for (int i = static_cast<int>(coeffs.size()) - 2; i >= 0; --i) {
        d   = d * x + val;
        val = val * x + coeffs[i];
    }
    return {val, d};
}

// ── Quadratic equation: a*x² + b*x + c = 0 ─────────────────────────
// Returns number of real roots, fills r0, r1 in ascending order.
template <typename T>
int solve_quadratic(T a, T b, T c, T& r0, T& r1) {
    if (std::abs(a) < epsilon<T>) {
        // degenerate: linear
        if (std::abs(b) < epsilon<T>) return 0;
        r0 = -c / b;
        r1 = r0;
        return 1;
    }
    T disc = b*b - T(4)*a*c;
    if (disc < T(0)) return 0;
    if (disc < epsilon<T>) {
        r0 = -b / (T(2)*a);
        r1 = r0;
        return 1;
    }
    T sqrt_disc = std::sqrt(disc);
    T inv2a = T(1) / (T(2)*a);
    r0 = (-b - sqrt_disc) * inv2a;
    r1 = (-b + sqrt_disc) * inv2a;
    if (r0 > r1) std::swap(r0, r1);
    return 2;
}

// ── Cubic equation: a*x³ + b*x² + c*x + d = 0 ───────────────────────
// Returns number of real roots, fills out array of up to 3.
template <typename T>
int solve_cubic(T a, T b, T c, T d, T out[3]) {
    if (std::abs(a) < epsilon<T>) {
        // degenerate to quadratic
        return solve_quadratic(b, c, d, out[0], out[1]);
    }
    // Normalize to x³ + px + q = 0 via substitution x = t - b/(3a)
    T inv_a = T(1) / a;
    T bn = b * inv_a;
    T cn = c * inv_a;
    T dn = d * inv_a;
    T p = cn - bn * bn / T(3);
    T q = T(2)*bn*bn*bn / T(27) - bn*cn / T(3) + dn;
    // Discriminant: D = (q/2)² + (p/3)³
    T q_half = q * T(0.5);
    T p_third = p / T(3);
    T D = q_half * q_half + p_third * p_third * p_third;
    if (D > T(0)) {
        // One real root
        T sqrt_D = std::sqrt(D);
        T u = std::cbrt(-q_half + sqrt_D);
        T v = std::cbrt(-q_half - sqrt_D);
        out[0] = u + v - bn / T(3);
        return 1;
    } else if (D < T(0)) {
        // Three real roots
        T r = std::sqrt(-p_third * p_third * p_third); // = sqrt(-p³/27)
        T phi = std::acos(-q_half / r);
        T two_root_p = T(2) * std::sqrt(-p_third);
        for (int k = 0; k < 3; ++k) {
            out[k] = two_root_p * std::cos((phi + T(2)*pi<T>*k) / T(3)) - bn / T(3);
        }
        // Sort
        if (out[0] > out[1]) std::swap(out[0], out[1]);
        if (out[1] > out[2]) { std::swap(out[1], out[2]); if (out[0] > out[1]) std::swap(out[0], out[1]); }
        return 3;
    } else {
        // D == 0: one real and one double real
        T u = std::cbrt(-q_half);
        out[0] = T(2) * u - bn / T(3);
        out[1] = -u - bn / T(3);
        if (out[0] > out[1]) std::swap(out[0], out[1]);
        return 2;
    }
}

// ── Quartic equation: a*x⁴ + b*x³ + c*x² + d*x + e = 0 ──────────────
// Returns number of real roots (up to 4).
template <typename T>
int solve_quartic(T a, T b, T c, T d, T e, T out[4]) {
    if (std::abs(a) < epsilon<T>) {
        // degenerate to cubic
        return solve_cubic(b, c, d, e, out);
    }
    // Normalize to x⁴ + ax³ + bx² + cx + d = 0
    T inv_a = T(1) / a;
    b *= inv_a;
    c *= inv_a;
    d *= inv_a;
    e *= inv_a;

    // Depressed quartic: let x = y - b/4
    T b4 = b / T(4);
    T p = c - T(6)*b4*b4;
    T q = d - T(2)*c*b4 + T(8)*b4*b4*b4;
    T r = e - d*b4 + c*b4*b4 - T(3)*b4*b4*b4*b4;
    // Now solve y⁴ + p y² + q y + r = 0

    // Ferrari's method: find a real root of the resolvent cubic
    // z³ + (p/2) z² + (p²/16 - r/4) z - q²/64 = 0
    T a3 = T(1);
    T b3 = p / T(2);
    T c3 = (p * p) / T(16) - r / T(4);
    T d3 = - (q * q) / T(64);
    T cubic_roots[3];
    int num_cubic = solve_cubic(a3, b3, c3, d3, cubic_roots);
    if (num_cubic < 1) return 0;
    // Choose the largest real root of the cubic to avoid numerical issues
    T z = cubic_roots[num_cubic - 1];

    T u = T(2) * z - p;
    T v;
    if (u < T(0)) u = T(0);
    u = std::sqrt(u);
    // Now we have two quadratic equations: y² ± u y + (z ∓ q/(2u)) = 0
    // Case 1: plus sign
    T a1 = T(1), b1 = u;
    T c1 = z - q / (T(2) * u + epsilon<T>);  // avoid division by zero
    // Case 2: minus sign
    T a2 = T(1), b2 = -u;
    T c2 = z + q / (T(2) * u + epsilon<T>);

    T temp[4];
    int count = 0;
    count += solve_quadratic(a1, b1, c1, temp[count], temp[count+1]);
    count += solve_quadratic(a2, b2, c2, temp[count], temp[count+1]);

    // Shift back x = y - b/4
    for (int i = 0; i < count; ++i)
        out[i] = temp[i] - b4;

    // Sort results
    std::sort(out, out + count);
    return count;
}

// ── General polynomial root finding using Durand-Kerner method ──────
// Approximate all roots (complex) of polynomial with real coefficients.
// coeffs[0] + coeffs[1]*x + ... + coeffs[n]*x^n = 0, n >= 1.
// Returns vector of complex roots. Uses initial guess on unit circle.
template <typename T>
std::vector<std::complex<T>> poly_roots_durand_kerner(const std::vector<T>& coeffs,
                                                      int max_iter = 200, T tol = epsilon<T>) {
    int n = static_cast<int>(coeffs.size()) - 1;
    if (n < 1) return {};
    // Normalize leading coefficient to 1
    std::vector<T> c = coeffs;
    T lead = c.back();
    for (auto& v : c) v /= lead;
    std::vector<std::complex<T>> roots(n);
    // Initialize roots as complex points on circle with radius slightly > 1
    T radius = T(1.5);
    for (int i = 0; i < n; ++i) {
        T angle = T(2) * pi<T> * i / T(n) + T(0.5);
        roots[i] = std::complex<T>(radius * std::cos(angle), radius * std::sin(angle));
    }
    for (int iter = 0; iter < max_iter; ++iter) {
        T max_delta = T(0);
        for (int k = 0; k < n; ++k) {
            std::complex<T> num = polyval(c, roots[k]); // evaluate polynomial at root
            std::complex<T> denom(1, 0);
            for (int j = 0; j < n; ++j) {
                if (j != k)
                    denom *= (roots[k] - roots[j]);
            }
            std::complex<T> correction = num / denom;
            roots[k] -= correction;
            max_delta = std::max(max_delta, std::abs(correction));
        }
        if (max_delta < tol)
            break;
    }
    return roots;
}

// Overload for complex evaluation via real polyval
template <typename T>
std::complex<T> polyval(const std::vector<T>& coeffs, const std::complex<T>& x) {
    if (coeffs.empty()) return T(0);
    std::complex<T> result(coeffs.back());
    for (int i = static_cast<int>(coeffs.size()) - 2; i >= 0; --i)
        result = result * x + coeffs[i];
    return result;
}

} // namespace wp