// system name : onetbb-warp
// File 0015 : core/math/statistics.h
// Description : Descriptive statistics, regression, distributions, and streaming moments.

#ifndef __TBB_WARP_CORE_MATH_STATISTICS_H
#define __TBB_WARP_CORE_MATH_STATISTICS_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <iterator>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Mean
// ============================================================

template<typename Iterator>
auto arithmetic_mean(Iterator first, Iterator last) {
    using T = typename std::iterator_traits<Iterator>::value_type;
    T sum = T(0);
    std::size_t count = 0;
    for (auto it = first; it != last; ++it, ++count) sum += *it;
    return (count > 0) ? sum / static_cast<T>(count) : T(0);
}

template<typename Container>
auto mean(const Container& data) {
    return arithmetic_mean(std::begin(data), std::end(data));
}

// ============================================================
// Weighted mean
// ============================================================

template<typename ValIter, typename WgtIter>
auto weighted_mean(ValIter vfirst, ValIter vlast, WgtIter wfirst) {
    using T = typename std::iterator_traits<ValIter>::value_type;
    T sum = T(0), weight_sum = T(0);
    while (vfirst != vlast) {
        T w = static_cast<T>(*wfirst);
        sum += (*vfirst) * w;
        weight_sum += w;
        ++vfirst; ++wfirst;
    }
    return (weight_sum > T(0)) ? sum / weight_sum : T(0);
}

// ============================================================
// Median (requires sorting – copies data)
// ============================================================

template<typename Container>
auto median(Container data) {
    using T = typename Container::value_type;
    if (data.empty()) return T(0);
    std::nth_element(data.begin(), data.begin() + data.size()/2, data.end());
    T med = data[data.size()/2];
    if (data.size() % 2 == 0) {
        T med2 = *std::max_element(data.begin(), data.begin() + data.size()/2);
        return (med + med2) / T(2);
    }
    return med;
}

// ============================================================
// Variance (population and sample) – two‑pass for stability
// ============================================================

template<typename Iterator>
auto variance_population(Iterator first, Iterator last) {
    using T = typename std::iterator_traits<Iterator>::value_type;
    T m = arithmetic_mean(first, last);
    T sum_sq = T(0);
    std::size_t n = 0;
    for (auto it = first; it != last; ++it, ++n) {
        T diff = *it - m;
        sum_sq += diff * diff;
    }
    return (n > 0) ? sum_sq / static_cast<T>(n) : T(0);
}

template<typename Iterator>
auto variance_sample(Iterator first, Iterator last) {
    using T = typename std::iterator_traits<Iterator>::value_type;
    T m = arithmetic_mean(first, last);
    T sum_sq = T(0);
    std::size_t n = 0;
    for (auto it = first; it != last; ++it, ++n) {
        T diff = *it - m;
        sum_sq += diff * diff;
    }
    return (n > 1) ? sum_sq / static_cast<T>(n - 1) : T(0);
}

template<typename Container> auto var_pop(const Container& d) { return variance_population(std::begin(d),std::end(d)); }
template<typename Container> auto var_samp(const Container& d) { return variance_sample(std::begin(d),std::end(d)); }
template<typename Container> auto stddev_pop(const Container& d) { return std::sqrt(var_pop(d)); }
template<typename Container> auto stddev_samp(const Container& d) { return std::sqrt(var_samp(d)); }

// ============================================================
// Welford's online algorithm (mean, variance, skewness, kurtosis)
// ============================================================

template<typename T>
struct welford_accumulator {
    std::size_t count = 0;
    T mean = T(0);
    T M2 = T(0);
    T M3 = T(0);
    T M4 = T(0);

    void push(T value) noexcept {
        ++count;
        T delta = value - mean;
        T delta_n = delta / static_cast<T>(count);
        T term1 = delta * delta_n * static_cast<T>(count - 1);
        mean += delta_n;
        T delta2 = value - mean;
        T term2 = delta * delta2;
        M4 += term1 * term2 * (static_cast<T>(count) * static_cast<T>(count) - static_cast<T>(3) * static_cast<T>(count) + static_cast<T>(3)) +
              static_cast<T>(6) * M2 * delta_n * delta_n - static_cast<T>(4) * M3 * delta_n;
        M3 += delta_n * (term1 - static_cast<T>(3) * M2);
        M2 += term1;
    }

    T population_variance() const noexcept { return (count > 0) ? M2 / static_cast<T>(count) : T(0); }
    T sample_variance() const noexcept { return (count > 1) ? M2 / static_cast<T>(count - 1) : T(0); }
    T population_stddev() const noexcept { return std::sqrt(population_variance()); }
    T sample_stddev() const noexcept { return std::sqrt(sample_variance()); }
    T skewness() const noexcept {
        if (count < 2 || M2 < 1e-12) return T(0);
        return std::sqrt(static_cast<T>(count)) * M3 / (M2 * std::sqrt(M2));
    }
    T kurtosis() const noexcept {
        if (count < 3 || M2 < 1e-12) return T(0);
        return static_cast<T>(count) * M4 / (M2 * M2) - T(3);
    }
};

// ============================================================
// Covariance and Pearson correlation
// ============================================================

template<typename Iterator1, typename Iterator2>
auto covariance(Iterator1 xfirst, Iterator1 xlast, Iterator2 yfirst) {
    using T = typename std::iterator_traits<Iterator1>::value_type;
    T mx = arithmetic_mean(xfirst, xlast);
    std::size_t n = 0;
    auto it = xfirst;
    auto iy = yfirst;
    T sum = T(0);
    while (it != xlast) {
        sum += (*it - mx) * (static_cast<T>(*iy) - mx); // assumes y also has same type? use mean of y separately
        ++it; ++iy; ++n;
    }
    return (n > 1) ? sum / static_cast<T>(n - 1) : T(0);
}

template<typename Container1, typename Container2>
auto covariance(const Container1& x, const Container2& y) {
    return covariance(std::begin(x), std::end(x), std::begin(y));
}

template<typename Iterator1, typename Iterator2>
auto pearson_correlation(Iterator1 xfirst, Iterator1 xlast, Iterator2 yfirst) {
    using T = typename std::iterator_traits<Iterator1>::value_type;
    T mx = arithmetic_mean(xfirst, xlast);
    // Compute mean of y separately
    auto iy = yfirst;
    T my = T(0);
    std::size_t n = 0;
    for (auto it = xfirst; it != xlast; ++it, ++iy, ++n) my += *iy;
    if (n == 0) return T(0);
    my /= static_cast<T>(n);
    T cov = T(0), varx = T(0), vary = T(0);
    iy = yfirst;
    for (auto it = xfirst; it != xlast; ++it, ++iy) {
        T dx = *it - mx, dy = static_cast<T>(*iy) - my;
        cov += dx * dy;
        varx += dx * dx;
        vary += dy * dy;
    }
    if (varx < 1e-12 || vary < 1e-12) return T(0);
    return cov / (std::sqrt(varx) * std::sqrt(vary));
}

template<typename Container1, typename Container2>
auto pearson_correlation(const Container1& x, const Container2& y) {
    return pearson_correlation(std::begin(x), std::end(x), std::begin(y));
}

// ============================================================
// Spearman rank correlation
// ============================================================

template<typename Container1, typename Container2>
auto spearman_rank_correlation(const Container1& x, const Container2& y) {
    using T = typename Container1::value_type;
    std::size_t n = std::min(x.size(), y.size());
    if (n < 2) return T(0);
    // Compute ranks for x
    std::vector<std::pair<T,std::size_t>> x_ranked(n), y_ranked(n);
    for (std::size_t i=0; i<n; ++i) { x_ranked[i]={x[i],i}; y_ranked[i]={y[i],i}; }
    std::sort(x_ranked.begin(), x_ranked.end());
    std::sort(y_ranked.begin(), y_ranked.end());
    std::vector<T> rank_x(n), rank_y(n);
    for (std::size_t i=0; i<n; ++i) rank_x[x_ranked[i].second] = static_cast<T>(i+1);
    for (std::size_t i=0; i<n; ++i) rank_y[y_ranked[i].second] = static_cast<T>(i+1);
    T sum_d2 = T(0);
    for (std::size_t i=0; i<n; ++i) { T d = rank_x[i]-rank_y[i]; sum_d2 += d*d; }
    return T(1) - (T(6)*sum_d2) / (static_cast<T>(n*(n*n-1)));
}

// ============================================================
// Simple linear regression y = a + b*x
// ============================================================

template<typename Container1, typename Container2>
auto linear_regression(const Container1& x, const Container2& y) {
    using T = typename Container1::value_type;
    std::size_t n = std::min(x.size(), y.size());
    if (n < 2) return std::make_tuple(T(0), T(0), T(0)); // intercept, slope, r^2
    T sum_x=0, sum_y=0, sum_xx=0, sum_xy=0, sum_yy=0;
    for (std::size_t i=0; i<n; ++i) {
        T xi = x[i], yi = static_cast<T>(y[i]);
        sum_x += xi; sum_y += yi;
        sum_xx += xi*xi; sum_xy += xi*yi; sum_yy += yi*yi;
    }
    T denom = n*sum_xx - sum_x*sum_x;
    if (std::abs(denom) < T(1e-12)) return std::make_tuple(T(0), T(0), T(0));
    T slope = (n*sum_xy - sum_x*sum_y) / denom;
    T intercept = (sum_y - slope*sum_x) / n;
    T r_denom = std::sqrt((n*sum_xx - sum_x*sum_x) * (n*sum_yy - sum_y*sum_y));
    T r = (r_denom > T(1e-12)) ? (n*sum_xy - sum_x*sum_y) / r_denom : T(0);
    return std::make_tuple(intercept, slope, r*r);
}

// ============================================================
// Histogram
// ============================================================

template<typename Iterator>
auto histogram(Iterator first, Iterator last, std::size_t bins,
               typename std::iterator_traits<Iterator>::value_type min_val = std::numeric_limits<
                   typename std::iterator_traits<Iterator>::value_type>::max(),
               typename std::iterator_traits<Iterator>::value_type max_val = std::numeric_limits<
                   typename std::iterator_traits<Iterator>::value_type>::lowest())
{
    using T = typename std::iterator_traits<Iterator>::value_type;
    if (min_val > max_val) {
        if (first == last) return std::make_pair(std::vector<std::size_t>(bins,0), std::vector<T>(bins+1,T(0)));
        auto [min_it, max_it] = std::minmax_element(first, last);
        min_val = *min_it; max_val = *max_it;
    }
    T range = max_val - min_val;
    if (range <= T(0)) range = T(1);
    std::vector<std::size_t> counts(bins, 0);
    std::vector<T> edges(bins + 1);
    for (std::size_t i = 0; i <= bins; ++i) edges[i] = min_val + (range * static_cast<T>(i) / static_cast<T>(bins));
    for (auto it = first; it != last; ++it) {
        T val = *it;
        if (val < min_val || val > max_val) continue;
        std::size_t idx = static_cast<std::size_t>((val - min_val) / range * bins);
        if (idx >= bins) idx = bins - 1;
        ++counts[idx];
    }
    return std::make_pair(counts, edges);
}

// ============================================================
// Moving averages (SMA, EMA)
// ============================================================

template<typename Container>
auto simple_moving_average(const Container& data, std::size_t window) {
    using T = typename Container::value_type;
    std::vector<T> result;
    if (data.size() < window) return result;
    result.reserve(data.size() - window + 1);
    T sum = std::accumulate(data.begin(), data.begin()+window, T(0));
    result.push_back(sum / static_cast<T>(window));
    for (std::size_t i = window; i < data.size(); ++i) {
        sum += data[i] - data[i - window];
        result.push_back(sum / static_cast<T>(window));
    }
    return result;
}

template<typename Container>
auto exponential_moving_average(const Container& data, float alpha) {
    using T = typename Container::value_type;
    std::vector<T> result(data.size());
    if (data.empty()) return result;
    result[0] = data[0];
    for (std::size_t i = 1; i < data.size(); ++i)
        result[i] = alpha * data[i] + (1.0f - alpha) * result[i-1];
    return result;
}

// ============================================================
// Cumulative sum
// ============================================================

template<typename Container>
auto cumulative_sum(const Container& data) {
    using T = typename Container::value_type;
    std::vector<T> result(data.size());
    T running = T(0);
    for (std::size_t i = 0; i < data.size(); ++i) { running += data[i]; result[i] = running; }
    return result;
}

// ============================================================
// Normal (Gaussian) PDF / CDF / quantile
// ============================================================

inline float normal_pdf(float x, float mean = 0.0f, float sigma = 1.0f) {
    float z = (x - mean) / sigma;
    return std::exp(-0.5f * z * z) / (sigma * std::sqrt(TAU_F));
}

inline float normal_cdf(float x, float mean = 0.0f, float sigma = 1.0f) {
    float z = (x - mean) / (sigma * SQRT2_F);
    return 0.5f * (1.0f + std::erf(z));
}

inline float normal_quantile(float p, float mean = 0.0f, float sigma = 1.0f) {
    if (p <= 0.0f) return -std::numeric_limits<float>::infinity();
    if (p >= 1.0f) return std::numeric_limits<float>::infinity();
    // Abramowitz and Stegun approximation
    float a[] = {2.50662823884f, -18.61500062529f, 41.39119773534f, -25.44106049637f};
    float b[] = {-8.47351093090f, 23.08336743743f, -21.06224101826f, 3.13082909833f};
    float c[] = {0.3374754822726147f, 0.9761690190917186f, 0.1607979714918209f, 0.0276438810333863f,
                 0.0038405729373609f, 0.0003951896511919f, 0.0000321767881768f, 0.0000002888167364f,
                 0.0000003960315187f};
    float pp = (p < 0.5f) ? p : 1.0f - p;
    float t = std::sqrt(-2.0f * std::log(pp));
    float num = c[0] + t*(c[1] + t*(c[2] + t*(c[3] + t*(c[4] + t*(c[5] + t*(c[6] + t*(c[7] + t*c[8])))))));
    float den = 1.0f + t*(a[0] + t*(a[1] + t*(a[2] + t*a[3])));
    float x = t - num/den;
    if (p < 0.5f) x = -x;
    return mean + sigma * x;
}

// ============================================================
// Exponential PDF / CDF
// ============================================================

inline float exponential_pdf(float x, float lambda = 1.0f) {
    if (x < 0.0f) return 0.0f;
    return lambda * std::exp(-lambda * x);
}

inline float exponential_cdf(float x, float lambda = 1.0f) {
    if (x <= 0.0f) return 0.0f;
    return 1.0f - std::exp(-lambda * x);
}

// ============================================================
// Uniform PDF / CDF
// ============================================================

inline float uniform_pdf(float x, float a = 0.0f, float b = 1.0f) {
    if (x < a || x > b) return 0.0f;
    return 1.0f / (b - a);
}

inline float uniform_cdf(float x, float a = 0.0f, float b = 1.0f) {
    if (x < a) return 0.0f;
    if (x > b) return 1.0f;
    return (x - a) / (b - a);
}

// ============================================================
// Robust statistics: Median Absolute Deviation (MAD)
// ============================================================

template<typename Container>
auto median_absolute_deviation(const Container& data, float scale = 1.4826f) {
    using T = typename Container::value_type;
    if (data.empty()) return T(0);
    T med = median(data);
    std::vector<T> abs_dev(data.size());
    for (std::size_t i=0; i<data.size(); ++i) abs_dev[i] = std::abs(data[i] - med);
    return scale * median(abs_dev);
}

// ============================================================
// Bootstrap confidence interval (percentile method)
// ============================================================

template<typename Container, typename StatisticFunc>
auto bootstrap_ci(const Container& data, StatisticFunc stat, int n_bootstrap = 2000, float alpha = 0.05f) {
    using T = typename Container::value_type;
    std::vector<T> boot_stats(n_bootstrap);
    std::mt19937 rng(42); // fixed seed for reproducibility
    std::uniform_int_distribution<std::size_t> dist(0, data.size()-1);
    for (int b=0; b<n_bootstrap; ++b) {
        std::vector<T> sample(data.size());
        for (std::size_t i=0; i<data.size(); ++i) sample[i] = data[dist(rng)];
        boot_stats[b] = stat(sample);
    }
    std::sort(boot_stats.begin(), boot_stats.end());
    int lower_idx = static_cast<int>(alpha/2.0f * n_bootstrap);
    int upper_idx = static_cast<int>((1.0f - alpha/2.0f) * n_bootstrap);
    return std::make_pair(boot_stats[lower_idx], boot_stats[upper_idx]);
}

// ============================================================
// Quantile
// ============================================================

template<typename Container>
auto quantile(const Container& data, float q) {
    using T = typename Container::value_type;
    if (data.empty()) return T(0);
    std::vector<T> sorted(data.begin(), data.end());
    std::sort(sorted.begin(), sorted.end());
    float pos = q * (sorted.size() - 1);
    std::size_t idx = static_cast<std::size_t>(pos);
    float frac = pos - idx;
    if (idx + 1 >= sorted.size()) return sorted.back();
    return sorted[idx] + frac * (sorted[idx+1] - sorted[idx]);
}

// ============================================================
// Interquartile Range (IQR)
// ============================================================

template<typename Container>
auto iqr(const Container& data) {
    using T = typename Container::value_type;
    return quantile(data, 0.75f) - quantile(data, 0.25f);
}

// ============================================================
// Z‑score normalization
// ============================================================

template<typename Container>
auto z_score_normalize(const Container& data) {
    using T = typename Container::value_type;
    T m = mean(data);
    T s = stddev_pop(data);
    std::vector<T> result(data.size());
    for (std::size_t i=0; i<data.size(); ++i) result[i] = (data[i] - m) / (s + T(1e-12));
    return result;
}

// ============================================================
// Min‑max normalization
// ============================================================

template<typename Container>
auto min_max_normalize(const Container& data, T target_min = T(0), T target_max = T(1)) {
    using T = typename Container::value_type;
    auto [min_it, max_it] = std::minmax_element(std::begin(data), std::end(data));
    T min_val = *min_it, max_val = *max_it;
    T range = max_val - min_val;
    if (range < T(1e-12)) range = T(1);
    std::vector<T> result(data.size());
    for (std::size_t i=0; i<data.size(); ++i)
        result[i] = target_min + (data[i] - min_val) / range * (target_max - target_min);
    return result;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_STATISTICS_H