// include/xtu/stats/xstats.hpp
// xtensor-unified - Statistical functions: correlation, covariance, histogram, percentiles, moments
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_STATS_XSTATS_HPP
#define XTU_STATS_XSTATS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/containers/xarray.hpp"
#include "xtu/containers/xtensor.hpp"
#include "xtu/math/xreducer.hpp"
#include "xtu/math/xmissing.hpp"
#include "xtu/math/xsorting.hpp"
#include "xtu/manipulation/xmanipulation.hpp"

XTU_NAMESPACE_BEGIN
namespace stats {

// #############################################################################
// Basic statistics (mean, var, std, etc.) extended with axis support
// #############################################################################

/// Compute mean along specified axes (returns scalar if axes empty)
template <class E>
auto mean(const xexpression<E>& e, const std::vector<size_t>& axes = {}) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    if (axes.empty()) {
        value_type sum = 0;
        for (size_t i = 0; i < expr.size(); ++i) sum += expr.flat(i);
        return sum / static_cast<value_type>(expr.size());
    }
    // Use reducer for multi-axis
    return math::mean(e, axes);
}

/// Compute variance along axes
template <class E>
auto var(const xexpression<E>& e, const std::vector<size_t>& axes = {}, size_t ddof = 0) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    if (axes.empty()) {
        value_type m = mean(e);
        value_type sum_sq = 0;
        for (size_t i = 0; i < expr.size(); ++i) {
            value_type diff = expr.flat(i) - m;
            sum_sq += diff * diff;
        }
        size_t n = expr.size();
        if (n <= ddof) return std::numeric_limits<value_type>::quiet_NaN();
        return sum_sq / static_cast<value_type>(n - ddof);
    }
    // For multi-axis, we need a custom reducer (not implemented here)
    XTU_THROW(std::runtime_error, "var with axes not yet fully implemented; use math::var reducer");
}

/// Compute standard deviation
template <class E>
auto stddev(const xexpression<E>& e, const std::vector<size_t>& axes = {}, size_t ddof = 0) {
    auto v = var(e, axes, ddof);
    if constexpr (std::is_arithmetic_v<decltype(v)>) {
        return std::sqrt(v);
    } else {
        return sqrt(v);
    }
}

/// Compute skewness (third standardized moment)
template <class E>
auto skew(const xexpression<E>& e) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    value_type m = mean(e);
    value_type m2 = 0, m3 = 0;
    size_t n = expr.size();
    for (size_t i = 0; i < n; ++i) {
        value_type diff = expr.flat(i) - m;
        value_type diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
    }
    if (n <= 2) return std::numeric_limits<value_type>::quiet_NaN();
    value_type variance = m2 / static_cast<value_type>(n);
    if (variance == 0) return value_type(0);
    value_type skew_val = (m3 / static_cast<value_type>(n)) / (variance * std::sqrt(variance));
    // Bias correction for small samples
    value_type factor = std::sqrt(static_cast<value_type>(n * (n - 1))) / static_cast<value_type>(n - 2);
    return skew_val * factor;
}

/// Compute kurtosis (fourth standardized moment, Fisher definition)
template <class E>
auto kurtosis(const xexpression<E>& e) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    value_type m = mean(e);
    value_type m2 = 0, m4 = 0;
    size_t n = expr.size();
    for (size_t i = 0; i < n; ++i) {
        value_type diff = expr.flat(i) - m;
        value_type diff2 = diff * diff;
        m2 += diff2;
        m4 += diff2 * diff2;
    }
    if (n <= 3) return std::numeric_limits<value_type>::quiet_NaN();
    value_type variance = m2 / static_cast<value_type>(n);
    if (variance == 0) return value_type(0);
    value_type kurt = (m4 / static_cast<value_type>(n)) / (variance * variance) - value_type(3);
    // Bias correction
    value_type factor = static_cast<value_type>(n - 1) / static_cast<value_type>((n - 2) * (n - 3));
    return (static_cast<value_type>(n + 1) * kurt + value_type(6)) * factor;
}

// #############################################################################
// Covariance and correlation
// #############################################################################

/// Covariance between two 1D arrays
template <class E1, class E2>
auto cov(const xexpression<E1>& x, const xexpression<E2>& y, size_t ddof = 1) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Covariance requires 1D arrays");
    XTU_ASSERT_MSG(a.size() == b.size(), "Arrays must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    size_t n = a.size();
    if (n <= ddof) return std::numeric_limits<value_type>::quiet_NaN();
    value_type mean_x = mean(x);
    value_type mean_y = mean(y);
    value_type sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += (static_cast<value_type>(a[i]) - mean_x) * (static_cast<value_type>(b[i]) - mean_y);
    }
    return sum / static_cast<value_type>(n - ddof);
}

/// Covariance matrix for 2D data (rows = observations, cols = variables)
template <class E>
auto cov_matrix(const xexpression<E>& data, size_t ddof = 1) {
    const auto& mat = data.derived_cast();
    XTU_ASSERT_MSG(mat.dimension() == 2, "Input must be 2D (observations x variables)");
    using value_type = typename E::value_type;
    size_t n_obs = mat.shape()[0];
    size_t n_vars = mat.shape()[1];
    if (n_obs <= ddof) XTU_THROW(std::runtime_error, "Insufficient observations for ddof");
    
    // Compute mean of each variable
    xarray_container<value_type> means({n_vars});
    for (size_t j = 0; j < n_vars; ++j) {
        value_type sum = 0;
        for (size_t i = 0; i < n_obs; ++i) sum += mat(i, j);
        means[j] = sum / static_cast<value_type>(n_obs);
    }
    // Compute covariance matrix
    xarray_container<value_type> result({n_vars, n_vars});
    for (size_t i = 0; i < n_vars; ++i) {
        for (size_t j = 0; j < n_vars; ++j) {
            value_type sum = 0;
            for (size_t k = 0; k < n_obs; ++k) {
                sum += (mat(k, i) - means[i]) * (mat(k, j) - means[j]);
            }
            result(i, j) = sum / static_cast<value_type>(n_obs - ddof);
        }
    }
    return result;
}

/// Pearson correlation coefficient between two 1D arrays
template <class E1, class E2>
auto corr(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Correlation requires 1D arrays");
    XTU_ASSERT_MSG(a.size() == b.size(), "Arrays must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    value_type cov_xy = cov(x, y, 0); // population covariance
    value_type std_x = stddev(x);
    value_type std_y = stddev(y);
    if (std_x == 0 || std_y == 0) return std::numeric_limits<value_type>::quiet_NaN();
    return cov_xy / (std_x * std_y);
}

/// Correlation matrix for 2D data
template <class E>
auto corr_matrix(const xexpression<E>& data) {
    auto cov_mat = cov_matrix(data);
    size_t n = cov_mat.shape()[0];
    // Extract standard deviations
    xarray_container<typename decltype(cov_mat)::value_type> stds({n});
    for (size_t i = 0; i < n; ++i) stds[i] = std::sqrt(cov_mat(i, i));
    // Compute correlation matrix
    auto result = cov_mat;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (stds[i] > 0 && stds[j] > 0) {
                result(i, j) /= (stds[i] * stds[j]);
            } else {
                result(i, j) = std::numeric_limits<typename decltype(cov_mat)::value_type>::quiet_NaN();
            }
        }
    }
    return result;
}

// #############################################################################
// Histogram
// #############################################################################

/// Compute histogram counts and bin edges
template <class E>
auto histogram(const xexpression<E>& data, size_t bins = 10, 
               std::pair<double, double> range = {0, 0}) {
    const auto& arr = data.derived_cast();
    XTU_ASSERT_MSG(arr.dimension() == 1, "Histogram requires 1D data");
    using value_type = typename E::value_type;
    
    // Determine range
    double min_val, max_val;
    if (range.first == 0 && range.second == 0) {
        min_val = static_cast<double>(*std::min_element(arr.begin(), arr.end()));
        max_val = static_cast<double>(*std::max_element(arr.begin(), arr.end()));
    } else {
        min_val = range.first;
        max_val = range.second;
    }
    if (min_val >= max_val) {
        XTU_THROW(std::invalid_argument, "Invalid histogram range");
    }
    
    // Compute bin edges
    xarray_container<double> edges({bins + 1});
    double bin_width = (max_val - min_val) / static_cast<double>(bins);
    for (size_t i = 0; i <= bins; ++i) {
        edges[i] = min_val + static_cast<double>(i) * bin_width;
    }
    
    // Count occurrences
    xarray_container<size_t> counts({bins});
    std::fill(counts.begin(), counts.end(), 0);
    for (const auto& v : arr) {
        double val = static_cast<double>(v);
        if (val >= min_val && val < max_val) {
            size_t bin = static_cast<size_t>((val - min_val) / bin_width);
            if (bin >= bins) bin = bins - 1;
            ++counts[bin];
        } else if (val == max_val) {
            ++counts[bins - 1];
        }
    }
    return std::make_pair(std::move(counts), std::move(edges));
}

/// Compute 2D histogram (heatmap)
template <class E1, class E2>
auto histogram2d(const xexpression<E1>& x, const xexpression<E2>& y,
                 size_t bins_x = 10, size_t bins_y = 10,
                 std::pair<double, double> range_x = {0, 0},
                 std::pair<double, double> range_y = {0, 0}) {
    const auto& arr_x = x.derived_cast();
    const auto& arr_y = y.derived_cast();
    XTU_ASSERT_MSG(arr_x.dimension() == 1 && arr_y.dimension() == 1, "Histogram2d requires 1D data");
    XTU_ASSERT_MSG(arr_x.size() == arr_y.size(), "Arrays must have same size");
    
    double min_x, max_x, min_y, max_y;
    if (range_x.first == 0 && range_x.second == 0) {
        min_x = static_cast<double>(*std::min_element(arr_x.begin(), arr_x.end()));
        max_x = static_cast<double>(*std::max_element(arr_x.begin(), arr_x.end()));
    } else { min_x = range_x.first; max_x = range_x.second; }
    if (range_y.first == 0 && range_y.second == 0) {
        min_y = static_cast<double>(*std::min_element(arr_y.begin(), arr_y.end()));
        max_y = static_cast<double>(*std::max_element(arr_y.begin(), arr_y.end()));
    } else { min_y = range_y.first; max_y = range_y.second; }
    
    double bin_width_x = (max_x - min_x) / static_cast<double>(bins_x);
    double bin_width_y = (max_y - min_y) / static_cast<double>(bins_y);
    
    xarray_container<size_t> counts({bins_y, bins_x});
    std::fill(counts.begin(), counts.end(), 0);
    
    for (size_t i = 0; i < arr_x.size(); ++i) {
        double vx = static_cast<double>(arr_x[i]);
        double vy = static_cast<double>(arr_y[i]);
        if (vx >= min_x && vx < max_x && vy >= min_y && vy < max_y) {
            size_t bx = static_cast<size_t>((vx - min_x) / bin_width_x);
            size_t by = static_cast<size_t>((vy - min_y) / bin_width_y);
            if (bx >= bins_x) bx = bins_x - 1;
            if (by >= bins_y) by = bins_y - 1;
            ++counts(by, bx);
        } else if (vx == max_x && vy == max_y) {
            ++counts(bins_y - 1, bins_x - 1);
        }
    }
    
    xarray_container<double> edges_x({bins_x + 1});
    xarray_container<double> edges_y({bins_y + 1});
    for (size_t i = 0; i <= bins_x; ++i) edges_x[i] = min_x + static_cast<double>(i) * bin_width_x;
    for (size_t i = 0; i <= bins_y; ++i) edges_y[i] = min_y + static_cast<double>(i) * bin_width_y;
    
    return std::make_tuple(std::move(counts), std::move(edges_x), std::move(edges_y));
}

// #############################################################################
// Percentiles and quantiles
// #############################################################################

/// Compute q-th percentile (0 <= q <= 100)
template <class E>
auto percentile(const xexpression<E>& data, double q, const std::string& method = "linear") {
    const auto& arr = data.derived_cast();
    XTU_ASSERT_MSG(arr.dimension() == 1, "Percentile requires 1D data");
    XTU_ASSERT_MSG(q >= 0.0 && q <= 100.0, "Percentile must be between 0 and 100");
    using value_type = typename E::value_type;
    
    // Copy and sort
    std::vector<value_type> sorted(arr.begin(), arr.end());
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n == 0) return std::numeric_limits<value_type>::quiet_NaN();
    if (n == 1) return sorted[0];
    
    double pos = (n - 1) * q / 100.0;
    size_t idx_low = static_cast<size_t>(std::floor(pos));
    size_t idx_high = static_cast<size_t>(std::ceil(pos));
    
    if (method == "linear") {
        if (idx_low == idx_high) return sorted[idx_low];
        double frac = pos - idx_low;
        return static_cast<value_type>(sorted[idx_low] * (1.0 - frac) + sorted[idx_high] * frac);
    } else if (method == "lower") {
        return sorted[idx_low];
    } else if (method == "higher") {
        return sorted[idx_high];
    } else if (method == "nearest") {
        return (pos - idx_low < 0.5) ? sorted[idx_low] : sorted[idx_high];
    } else if (method == "midpoint") {
        return static_cast<value_type>((sorted[idx_low] + sorted[idx_high]) / 2.0);
    } else {
        XTU_THROW(std::invalid_argument, "Unknown percentile method");
    }
}

/// Median (50th percentile)
template <class E>
auto median(const xexpression<E>& data) {
    return percentile(data, 50.0);
}

/// Quantiles (multiple percentiles at once)
template <class E>
auto quantile(const xexpression<E>& data, const std::vector<double>& q, const std::string& method = "linear") {
    const auto& arr = data.derived_cast();
    xarray_container<typename E::value_type> result({q.size()});
    for (size_t i = 0; i < q.size(); ++i) {
        result[i] = percentile(data, q[i], method);
    }
    return result;
}

/// Interquartile range (IQR)
template <class E>
auto iqr(const xexpression<E>& data) {
    auto q75 = percentile(data, 75.0);
    auto q25 = percentile(data, 25.0);
    return q75 - q25;
}

// #############################################################################
// Moments and descriptive statistics
// #############################################################################

/// Compute central moments of given order
template <class E>
auto moment(const xexpression<E>& data, size_t order, bool central = true) {
    const auto& arr = data.derived_cast();
    using value_type = typename E::value_type;
    size_t n = arr.size();
    if (n == 0) return std::numeric_limits<value_type>::quiet_NaN();
    
    value_type center = central ? mean(data) : value_type(0);
    value_type sum = 0;
    for (size_t i = 0; i < n; ++i) {
        value_type diff = arr.flat(i) - center;
        sum += std::pow(diff, static_cast<int>(order));
    }
    return sum / static_cast<value_type>(n);
}

/// Geometric mean
template <class E>
auto gmean(const xexpression<E>& data) {
    const auto& arr = data.derived_cast();
    using value_type = typename E::value_type;
    value_type prod = 1;
    size_t count = 0;
    for (const auto& v : arr) {
        if (v <= 0) XTU_THROW(std::domain_error, "Geometric mean requires positive values");
        prod *= v;
        ++count;
    }
    return std::pow(prod, 1.0 / static_cast<double>(count));
}

/// Harmonic mean
template <class E>
auto hmean(const xexpression<E>& data) {
    const auto& arr = data.derived_cast();
    using value_type = typename E::value_type;
    value_type sum_inv = 0;
    size_t count = 0;
    for (const auto& v : arr) {
        if (v == 0) XTU_THROW(std::domain_error, "Harmonic mean requires non-zero values");
        sum_inv += value_type(1) / v;
        ++count;
    }
    return static_cast<value_type>(count) / sum_inv;
}

/// Trimmed mean (remove proportion of extremes)
template <class E>
auto trim_mean(const xexpression<E>& data, double proportion) {
    const auto& arr = data.derived_cast();
    XTU_ASSERT_MSG(proportion >= 0.0 && proportion < 0.5, "Proportion must be in [0, 0.5)");
    using value_type = typename E::value_type;
    std::vector<value_type> sorted(arr.begin(), arr.end());
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    size_t trim = static_cast<size_t>(n * proportion);
    if (trim * 2 >= n) return std::numeric_limits<value_type>::quiet_NaN();
    value_type sum = 0;
    for (size_t i = trim; i < n - trim; ++i) sum += sorted[i];
    return sum / static_cast<value_type>(n - 2 * trim);
}

// #############################################################################
// Statistical tests
// #############################################################################

/// T-test (one-sample)
template <class E>
auto ttest_1samp(const xexpression<E>& data, double popmean) {
    const auto& arr = data.derived_cast();
    using value_type = typename E::value_type;
    size_t n = arr.size();
    if (n < 2) return std::numeric_limits<value_type>::quiet_NaN();
    value_type m = mean(data);
    value_type s = stddev(data, {}, 1); // sample stddev
    value_type se = s / std::sqrt(static_cast<value_type>(n));
    return (m - popmean) / se;
}

/// T-test (two independent samples)
template <class E1, class E2>
auto ttest_ind(const xexpression<E1>& a, const xexpression<E2>& b, bool equal_var = true) {
    const auto& arr_a = a.derived_cast();
    const auto& arr_b = b.derived_cast();
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    size_t n1 = arr_a.size(), n2 = arr_b.size();
    if (n1 < 2 || n2 < 2) return std::numeric_limits<value_type>::quiet_NaN();
    value_type m1 = mean(a), m2 = mean(b);
    value_type v1 = var(a, {}, 1), v2 = var(b, {}, 1);
    value_type t;
    if (equal_var) {
        value_type pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / static_cast<value_type>(n1 + n2 - 2);
        t = (m1 - m2) / std::sqrt(pooled_var * (1.0/static_cast<value_type>(n1) + 1.0/static_cast<value_type>(n2)));
    } else {
        t = (m1 - m2) / std::sqrt(v1/static_cast<value_type>(n1) + v2/static_cast<value_type>(n2));
    }
    return t;
}

/// Z-score normalization (standardization)
template <class E>
auto zscore(const xexpression<E>& data) {
    const auto& arr = data.derived_cast();
    using value_type = typename E::value_type;
    value_type m = mean(data);
    value_type s = stddev(data);
    if (s == 0) XTU_THROW(std::runtime_error, "Zero standard deviation in zscore");
    xarray_container<value_type> result(arr.shape());
    for (size_t i = 0; i < arr.size(); ++i) {
        result.flat(i) = (arr.flat(i) - m) / s;
    }
    return result;
}

/// Min-max scaling to [feature_range[0], feature_range[1]]
template <class E>
auto minmax_scale(const xexpression<E>& data, double min_val = 0.0, double max_val = 1.0) {
    const auto& arr = data.derived_cast();
    using value_type = typename E::value_type;
    value_type data_min = *std::min_element(arr.begin(), arr.end());
    value_type data_max = *std::max_element(arr.begin(), arr.end());
    value_type range = data_max - data_min;
    if (range == 0) {
        xarray_container<value_type> result(arr.shape());
        std::fill(result.begin(), result.end(), static_cast<value_type>((min_val + max_val) / 2.0));
        return result;
    }
    xarray_container<value_type> result(arr.shape());
    value_type scale = static_cast<value_type>(max_val - min_val) / range;
    for (size_t i = 0; i < arr.size(); ++i) {
        result.flat(i) = static_cast<value_type>(min_val) + (arr.flat(i) - data_min) * scale;
    }
    return result;
}

// #############################################################################
// Binning and digitization
// #############################################################################

/// Assign data to bins, returning bin indices (0-based)
template <class E>
auto digitize(const xexpression<E>& data, const std::vector<double>& bins) {
    const auto& arr = data.derived_cast();
    XTU_ASSERT_MSG(std::is_sorted(bins.begin(), bins.end()), "Bins must be monotonically increasing");
    xarray_container<size_t> result(arr.shape());
    for (size_t i = 0; i < arr.size(); ++i) {
        double val = static_cast<double>(arr.flat(i));
        auto it = std::upper_bound(bins.begin(), bins.end(), val);
        size_t idx = static_cast<size_t>(std::distance(bins.begin(), it));
        if (idx == 0) idx = 0;
        else if (idx > bins.size()) idx = bins.size();
        else idx = idx - 1;
        result.flat(i) = idx;
    }
    return result;
}

/// Bincount: count occurrences of non-negative integers
template <class E>
auto bincount(const xexpression<E>& data, size_t minlength = 0) {
    const auto& arr = data.derived_cast();
    using value_type = typename E::value_type;
    if (arr.size() == 0) return xarray_container<size_t>();
    value_type max_val = *std::max_element(arr.begin(), arr.end());
    XTU_ASSERT_MSG(max_val >= 0, "bincount requires non-negative integers");
    size_t size = std::max(static_cast<size_t>(max_val) + 1, minlength);
    xarray_container<size_t> result({size});
    std::fill(result.begin(), result.end(), 0);
    for (const auto& v : arr) {
        size_t idx = static_cast<size_t>(v);
        if (idx < size) ++result[idx];
    }
    return result;
}

} // namespace stats

// Bring into main namespace for convenience
using stats::mean;
using stats::var;
using stats::stddev;
using stats::skew;
using stats::kurtosis;
using stats::cov;
using stats::cov_matrix;
using stats::corr;
using stats::corr_matrix;
using stats::histogram;
using stats::histogram2d;
using stats::percentile;
using stats::median;
using stats::quantile;
using stats::iqr;
using stats::moment;
using stats::gmean;
using stats::hmean;
using stats::trim_mean;
using stats::ttest_1samp;
using stats::ttest_ind;
using stats::zscore;
using stats::minmax_scale;
using stats::digitize;
using stats::bincount;

XTU_NAMESPACE_END

#endif // XTU_STATS_XSTATS_HPP