// File 0049 : core/math/bspline_interp.h
// Uniform B-spline interpolation for scalar data: basis functions, de Boor evaluation, and curve fitting to data points.

#pragma once

#include "constants.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace wp {

// ── 1D uniform B-spline basis functions ─────────────────────────────
// Compute the value of the i-th B-spline of degree `degree` at parameter `t` (t in [0,1) for a uniform knot vector).
// The knot vector is implicitly evenly spaced with multiplicity = degree+1 at the ends (clamped).
template <typename T>
T bspline_basis(int i, int degree, T t, const std::vector<T>& knots) {
    int n = static_cast<int>(knots.size()) - 1;
    if (degree == 0) {
        return (t >= knots[i] && t < knots[i+1]) ? T(1) : T(0);
    }
    T left = T(0), right = T(0);
    T denom1 = knots[i+degree] - knots[i];
    T denom2 = knots[i+degree+1] - knots[i+1];
    if (std::abs(denom1) > epsilon<T>)
        left = (t - knots[i]) / denom1 * bspline_basis(i, degree-1, t, knots);
    if (std::abs(denom2) > epsilon<T>)
        right = (knots[i+degree+1] - t) / denom2 * bspline_basis(i+1, degree-1, t, knots);
    return left + right;
}

// ── Evaluate B-spline curve at parameter t (clamped) ───────────────
template <typename T>
T bspline_eval(const std::vector<T>& control_points,
               const std::vector<T>& knots, int degree, T t) {
    // Ensure t is within valid range [knots[degree], knots[knots.size()-1-degree])
    t = clamp(t, knots[degree], knots[knots.size()-1-degree] - epsilon<T>);
    // Use de Boor algorithm
    int n_cp = static_cast<int>(control_points.size());
    // Find knot span index s such that t is in [knots[s], knots[s+1])
    int s = degree;
    for (int i = degree; i < static_cast<int>(knots.size())-1-degree; ++i) {
        if (t >= knots[i] && t < knots[i+1]) {
            s = i;
            break;
        }
    }
    std::vector<T> d(degree+1);
    for (int j = 0; j <= degree; ++j)
        d[j] = control_points[s - degree + j];

    for (int r = 1; r <= degree; ++r) {
        for (int j = degree; j >= r; --j) {
            T alpha = (t - knots[s - degree + j]) / (knots[j + s - r + 1] - knots[s - degree + j]);
            d[j] = (T(1) - alpha) * d[j-1] + alpha * d[j];
        }
    }
    return d[degree];
}

// ── Build clamped uniform knot vector for given number of control points and degree ──
template <typename T>
std::vector<T> make_clamped_knots(int num_control_points, int degree) {
    int n = num_control_points - 1;
    int m = n + degree + 1;
    std::vector<T> knots(m + 1);
    for (int i = 0; i <= m; ++i) {
        if (i <= degree) knots[i] = T(0);
        else if (i >= m - degree) knots[i] = T(1);
        else knots[i] = T(i - degree) / T(n - degree + 1);
    }
    return knots;
}

// ── Fit B-spline curve through given data points via global interpolation ──
// This function assumes the number of data points matches the number of control points,
// using parameter values by chord length and solving for control points.
template <typename T>
bool bspline_interpolate(const std::vector<T>& x_data, const std::vector<T>& y_data,
                         int degree, std::vector<T>& control_points,
                         std::vector<T>& knots) {
    int n = static_cast<int>(x_data.size());
    if (n < 2 || degree < 1 || degree >= n) return false;

    // Build knot vector
    knots = make_clamped_knots<T>(n, degree); // control points count = n

    // Compute parameter values t_i for each data point using chord length
    std::vector<T> t_vec(n);
    t_vec[0] = T(0);
    for (int i = 1; i < n; ++i)
        t_vec[i] = t_vec[i-1] + std::abs(x_data[i] - x_data[i-1]);
    T total_len = t_vec.back();
    if (total_len < epsilon<T>) total_len = T(1);
    for (int i = 0; i < n; ++i)
        t_vec[i] = knots[degree] + (knots[knots.size()-1-degree] - knots[degree]) * (t_vec[i] / total_len);

    // Build interpolation matrix: N(i,j) = basis_j(t_i) for j=0..n-1
    // Solve linear system A * P = Y, where A is n x n, P = control points, Y = y_data.
    // We'll use Gaussian elimination with partial pivoting.

    std::vector<std::vector<T>> A(n, std::vector<T>(n, T(0)));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = bspline_basis(j, degree, t_vec[i], knots);
        }
    }

    // Gaussian elimination
    for (int k = 0; k < n; ++k) {
        // Partial pivot
        int max_row = k;
        T max_val = std::abs(A[k][k]);
        for (int i = k+1; i < n; ++i) {
            if (std::abs(A[i][k]) > max_val) {
                max_val = std::abs(A[i][k]);
                max_row = i;
            }
        }
        if (max_val < epsilon<T>) return false;
        if (max_row != k) {
            std::swap(A[k], A[max_row]);
            std::swap(y_data[k], y_data[max_row]); // we need a copy of y_data; pass by value?
        }
        // Normalize row
        T inv_pivot = T(1) / A[k][k];
        for (int j = k; j < n; ++j) A[k][j] *= inv_pivot;
        y_data[k] *= inv_pivot;

        // Eliminate other rows
        for (int i = 0; i < n; ++i) {
            if (i == k) continue;
            T factor = A[i][k];
            if (std::abs(factor) < epsilon<T>) continue;
            for (int j = k; j < n; ++j) A[i][j] -= factor * A[k][j];
            y_data[i] -= factor * y_data[k];
        }
    }

    control_points = y_data; // now contains solution
    return true;
}

} // namespace wp