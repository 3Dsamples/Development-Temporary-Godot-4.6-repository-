// File 0035 : core/math/statistics.h
// Descriptive statistics: mean, variance, covariance, PCA, linear regression, and distribution sampling.

#pragma once

#include "vec3.h"
#include "vec2.h"
#include "mat3.h"
#include "svd3.h"
#include "constants.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>

namespace wp {

// ── Scalar statistics ──────────────────────────────────────────────
template <typename T>
T mean(const std::vector<T>& data) {
    if (data.empty()) return T(0);
    return std::accumulate(data.begin(), data.end(), T(0)) / T(data.size());
}

template <typename T>
T variance(const std::vector<T>& data, bool unbiased = true) {
    if (data.size() <= 1) return T(0);
    T mu = mean(data);
    T sum_sq = T(0);
    for (const auto& x : data) sum_sq += (x - mu) * (x - mu);
    return sum_sq / T(unbiased ? data.size() - 1 : data.size());
}

template <typename T>
T stdev(const std::vector<T>& data, bool unbiased = true) {
    return std::sqrt(variance(data, unbiased));
}

template <typename T>
T median(std::vector<T> data) {
    if (data.empty()) return T(0);
    std::sort(data.begin(), data.end());
    size_t n = data.size();
    if (n % 2 == 0)
        return (data[n/2 - 1] + data[n/2]) * T(0.5);
    return data[n/2];
}

template <typename T>
T percentile(const std::vector<T>& data, T p) {
    if (data.empty()) return T(0);
    std::vector<T> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>(p * T(sorted.size() - 1));
    return sorted[std::min(idx, sorted.size() - 1)];
}

// ── Vector statistics ───────────────────────────────────────────────
template <typename T, int Dim>
vec_t<Dim, T> mean(const std::vector<vec_t<Dim, T>>& points) {
    vec_t<Dim, T> sum(T(0));
    for (const auto& p : points) sum = sum + p;
    return sum / T(points.size());
}

template <typename T>
mat3<T> covariance_matrix(const std::vector<vec3<T>>& points) {
    if (points.size() < 2) return mat3<T>(T(0));
    vec3<T> mu = mean(points);
    mat3<T> cov(T(0));
    for (const auto& p : points) {
        vec3<T> d = p - mu;
        cov(0,0) += d.x * d.x;
        cov(0,1) += d.x * d.y;
        cov(0,2) += d.x * d.z;
        cov(1,1) += d.y * d.y;
        cov(1,2) += d.y * d.z;
        cov(2,2) += d.z * d.z;
    }
    // Fill symmetric
    cov(1,0) = cov(0,1);
    cov(2,0) = cov(0,2);
    cov(2,1) = cov(1,2);
    // Unbiased estimate
    T inv_n1 = T(1) / T(points.size() - 1);
    cov = cov * inv_n1;
    return cov;
}

// ── Principal Component Analysis (returns eigenvalues and eigenvectors) ──
template <typename T>
struct PCAResult {
    vec3<T> eigenvalues;
    mat3<T> eigenvectors;   // columns are principal components sorted by decreasing eigenvalue
};

template <typename T>
PCAResult<T> pca(const std::vector<vec3<T>>& points) {
    mat3<T> cov = covariance_matrix(points);
    mat3<T> V;
    vec3<T> lambda;
    symmetric_eigen(cov, V, lambda);
    // Sort descending (lambda, V columns)
    std::array<int, 3> idx = {0, 1, 2};
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return lambda[a] > lambda[b]; });
    PCAResult<T> res;
    res.eigenvalues = vec3<T>(lambda[idx[0]], lambda[idx[1]], lambda[idx[2]]);
    res.eigenvectors = mat3<T>(V.col(idx[0]), V.col(idx[1]), V.col(idx[2]));
    return res;
}

// ── Linear regression (simple y = a*x + b) ──────────────────────────
template <typename T>
std::pair<T, T> linear_regression(const std::vector<T>& x, const std::vector<T>& y) {
    if (x.size() != y.size() || x.empty()) return {T(0), T(0)};
    T sum_x = T(0), sum_y = T(0), sum_xx = T(0), sum_xy = T(0);
    size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xx += x[i] * x[i];
        sum_xy += x[i] * y[i];
    }
    T denom = T(n) * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < epsilon<T>) return {T(0), mean(y)};
    T a = (T(n) * sum_xy - sum_x * sum_y) / denom;
    T b = (sum_y - a * sum_x) / T(n);
    return {a, b};
}

// ── Exponential Moving Average ──────────────────────────────────────
template <typename T>
class ExponentialMovingAverage {
public:
    ExponentialMovingAverage(T alpha = T(0.1)) : m_alpha(alpha), m_ema(T(0)), m_initialized(false) {}
    void update(T value) {
        if (!m_initialized) { m_ema = value; m_initialized = true; }
        else m_ema = m_alpha * value + (T(1) - m_alpha) * m_ema;
    }
    T value() const { return m_ema; }
    void reset() { m_initialized = false; }
private:
    T m_alpha;
    T m_ema;
    bool m_initialized;
};

// ── Weighted mean and covariance (for particles) ────────────────────
template <typename T>
vec3<T> weighted_mean(const std::vector<vec3<T>>& points, const std::vector<T>& weights) {
    if (points.empty()) return vec3<T>(T(0));
    vec3<T> sum(T(0));
    T total_w = T(0);
    for (size_t i = 0; i < points.size(); ++i) {
        T w = weights[i];
        sum = sum + points[i] * w;
        total_w += w;
    }
    return sum / total_w;
}

template <typename T>
mat3<T> weighted_covariance(const std::vector<vec3<T>>& points, const std::vector<T>& weights) {
    if (points.size() < 2) return mat3<T>(T(0));
    T total_w = std::accumulate(weights.begin(), weights.end(), T(0));
    vec3<T> mu = weighted_mean(points, weights);
    mat3<T> cov(T(0));
    for (size_t i = 0; i < points.size(); ++i) {
        vec3<T> d = points[i] - mu;
        T w = weights[i];
        cov(0,0) += w * d.x * d.x;
        cov(0,1) += w * d.x * d.y;
        cov(0,2) += w * d.x * d.z;
        cov(1,1) += w * d.y * d.y;
        cov(1,2) += w * d.y * d.z;
        cov(2,2) += w * d.z * d.z;
    }
    cov(1,0) = cov(0,1);
    cov(2,0) = cov(0,2);
    cov(2,1) = cov(1,2);
    T inv_total = T(1) / total_w;
    return cov * inv_total;
}

} // namespace wp