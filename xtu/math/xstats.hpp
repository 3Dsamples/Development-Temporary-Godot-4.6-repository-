// math/xstats.hpp

#ifndef XTENSOR_XSTATS_HPP
#define XTENSOR_XSTATS_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xreducer.hpp"
#include "../core/xaccumulator.hpp"
#include "../core/xview.hpp"
#include "xsorting.hpp"
#include "xmissing.hpp"

#include <cmath>
#include <complex>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <vector>
#include <functional>
#include <stdexcept>
#include <random>
#include <limits>
#include <tuple>
#include <map>
#include <unordered_map>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace stats
        {
            // --------------------------------------------------------------------
            // Helper function objects for reductions with multiple outputs
            // --------------------------------------------------------------------
            
            template <class T>
            struct moment_fun
            {
                std::size_t order;
                bool central;
                
                moment_fun(std::size_t ord, bool cent = true) : order(ord), central(cent) {}
                
                T operator()(T* first, T* last) const
                {
                    std::size_t n = static_cast<std::size_t>(std::distance(first, last));
                    if (n == 0) return T(0);
                    
                    if (central)
                    {
                        T mean = std::accumulate(first, last, T(0)) / static_cast<T>(n);
                        T sum = 0;
                        for (auto it = first; it != last; ++it)
                        {
                            T diff = *it - mean;
                            sum += std::pow(diff, static_cast<T>(order));
                        }
                        return sum / static_cast<T>(n);
                    }
                    else
                    {
                        T sum = 0;
                        for (auto it = first; it != last; ++it)
                        {
                            sum += std::pow(*it, static_cast<T>(order));
                        }
                        return sum / static_cast<T>(n);
                    }
                }
            };
            
            template <class T>
            struct skewness_fun
            {
                T operator()(T* first, T* last) const
                {
                    std::size_t n = static_cast<std::size_t>(std::distance(first, last));
                    if (n < 3) return T(0);
                    
                    T mean = std::accumulate(first, last, T(0)) / static_cast<T>(n);
                    T m2 = 0, m3 = 0;
                    for (auto it = first; it != last; ++it)
                    {
                        T diff = *it - mean;
                        m2 += diff * diff;
                        m3 += diff * diff * diff;
                    }
                    m2 /= static_cast<T>(n);
                    m3 /= static_cast<T>(n);
                    if (m2 == 0) return T(0);
                    return m3 / std::pow(m2, T(1.5));
                }
            };
            
            template <class T>
            struct kurtosis_fun
            {
                T operator()(T* first, T* last) const
                {
                    std::size_t n = static_cast<std::size_t>(std::distance(first, last));
                    if (n < 4) return T(0);
                    
                    T mean = std::accumulate(first, last, T(0)) / static_cast<T>(n);
                    T m2 = 0, m4 = 0;
                    for (auto it = first; it != last; ++it)
                    {
                        T diff = *it - mean;
                        T diff2 = diff * diff;
                        m2 += diff2;
                        m4 += diff2 * diff2;
                    }
                    m2 /= static_cast<T>(n);
                    m4 /= static_cast<T>(n);
                    if (m2 == 0) return T(0);
                    return m4 / (m2 * m2) - T(3);
                }
            };
            
            template <class T>
            struct mode_fun
            {
                T operator()(T* first, T* last) const
                {
                    if (first == last) return T(0);
                    std::map<T, std::size_t> freq;
                    for (auto it = first; it != last; ++it)
                        freq[*it]++;
                    T mode_val = first[0];
                    std::size_t max_count = 0;
                    for (const auto& p : freq)
                    {
                        if (p.second > max_count)
                        {
                            max_count = p.second;
                            mode_val = p.first;
                        }
                    }
                    return mode_val;
                }
            };
            
            // --------------------------------------------------------------------
            // Descriptive statistics
            // --------------------------------------------------------------------
            
            // Mean
            template <class E>
            inline auto mean(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                value_type sum = std::accumulate(expr.begin(), expr.end(), value_type(0));
                return sum / static_cast<value_type>(expr.size());
            }
            
            template <class E>
            inline auto mean(const xexpression<E>& e, std::size_t axis)
            {
                return xt::mean(e, {axis}, false);
            }
            
            template <class E>
            inline auto mean(const xexpression<E>& e, const std::vector<std::size_t>& axes, bool keepdims = false)
            {
                return xt::mean(e, axes, keepdims);
            }
            
            // Weighted mean
            template <class E, class W>
            inline auto weighted_mean(const xexpression<E>& e, const xexpression<W>& weights)
            {
                const auto& expr = e.derived_cast();
                const auto& w = weights.derived_cast();
                if (expr.size() != w.size())
                    XTENSOR_THROW(std::invalid_argument, "weighted_mean: size mismatch");
                using value_type = typename E::value_type;
                value_type sum = 0, sum_w = 0;
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    sum += expr.flat(i) * w.flat(i);
                    sum_w += w.flat(i);
                }
                return sum / sum_w;
            }
            
            // Harmonic mean
            template <class E>
            inline auto harmonic_mean(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                value_type sum_inv = 0;
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    if (expr.flat(i) == 0) return value_type(0);
                    sum_inv += value_type(1) / expr.flat(i);
                }
                return static_cast<value_type>(expr.size()) / sum_inv;
            }
            
            // Geometric mean
            template <class E>
            inline auto geometric_mean(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                value_type prod = 1;
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    if (expr.flat(i) <= 0) return value_type(0);
                    prod *= expr.flat(i);
                }
                return std::pow(prod, value_type(1) / static_cast<value_type>(expr.size()));
            }
            
            // Median
            template <class E>
            inline auto median(const xexpression<E>& e)
            {
                auto flat = eval(e);
                std::sort(flat.begin(), flat.end());
                std::size_t n = flat.size();
                if (n % 2 == 1)
                    return flat(n / 2);
                else
                    return (flat(n / 2 - 1) + flat(n / 2)) / typename E::value_type(2);
            }
            
            template <class E>
            inline auto median(const xexpression<E>& e, std::size_t axis)
            {
                return xt::median(e, axis);
            }
            
            // Mode
            template <class E>
            inline auto mode(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                std::unordered_map<value_type, std::size_t> freq;
                for (auto it = expr.begin(); it != expr.end(); ++it)
                    freq[*it]++;
                value_type mode_val = expr.flat(0);
                std::size_t max_count = 0;
                for (const auto& p : freq)
                {
                    if (p.second > max_count)
                    {
                        max_count = p.second;
                        mode_val = p.first;
                    }
                }
                return mode_val;
            }
            
            // Variance (population, use ddof=1 for sample)
            template <class E>
            inline auto var(const xexpression<E>& e, std::size_t ddof = 0)
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                value_type m = mean(expr);
                value_type sum_sq = 0;
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    value_type diff = expr.flat(i) - m;
                    sum_sq += diff * diff;
                }
                return sum_sq / static_cast<value_type>(expr.size() - ddof);
            }
            
            template <class E>
            inline auto var(const xexpression<E>& e, std::size_t axis, std::size_t ddof = 0)
            {
                return xt::variance(e, {axis}, ddof != 0);
            }
            
            // Standard deviation
            template <class E>
            inline auto stddev(const xexpression<E>& e, std::size_t ddof = 0)
            {
                return std::sqrt(var(e, ddof));
            }
            
            template <class E>
            inline auto stddev(const xexpression<E>& e, std::size_t axis, std::size_t ddof = 0)
            {
                return xt::stddev(e, {axis}, ddof != 0);
            }
            
            // Standard error of the mean
            template <class E>
            inline auto sem(const xexpression<E>& e, std::size_t ddof = 1)
            {
                const auto& expr = e.derived_cast();
                return stddev(expr, ddof) / std::sqrt(static_cast<double>(expr.size()));
            }
            
            // Skewness
            template <class E>
            inline auto skew(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                return reduce_all(expr, skewness_fun<value_type>{})(0);
            }
            
            // Kurtosis (excess)
            template <class E>
            inline auto kurtosis(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                return reduce_all(expr, kurtosis_fun<value_type>{})(0);
            }
            
            // Moment (raw or central)
            template <class E>
            inline auto moment(const xexpression<E>& e, std::size_t order, bool central = true)
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                return reduce_all(expr, moment_fun<value_type>(order, central))(0);
            }
            
            // --------------------------------------------------------------------
            // Quantiles and percentiles
            // --------------------------------------------------------------------
            
            template <class E>
            inline auto quantile(const xexpression<E>& e, double q, std::size_t axis = 0)
            {
                return xt::quantile(e, q, axis);
            }
            
            template <class E>
            inline auto percentile(const xexpression<E>& e, double p, std::size_t axis = 0)
            {
                return xt::percentile(e, p, axis);
            }
            
            template <class E>
            inline auto quantiles(const xexpression<E>& e, const std::vector<double>& q, std::size_t axis = 0)
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                std::size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), expr.dimension());
                std::size_t axis_len = expr.shape()[ax];
                std::size_t num_slices = expr.size() / axis_len;
                
                std::vector<std::size_t> result_shape = expr.shape();
                result_shape[ax] = q.size();
                xarray_container<value_type> result(result_shape);
                
                for (std::size_t slice = 0; slice < num_slices; ++slice)
                {
                    std::vector<value_type> buffer(axis_len);
                    std::vector<std::size_t> coords(expr.dimension(), 0);
                    std::size_t temp = slice;
                    for (std::size_t d = 0; d < expr.dimension(); ++d)
                    {
                        if (d == ax) continue;
                        std::size_t stride_after = 1;
                        for (std::size_t k = d + 1; k < expr.dimension(); ++k)
                            if (k != ax) stride_after *= expr.shape()[k];
                        coords[d] = temp / stride_after;
                        temp %= stride_after;
                    }
                    for (std::size_t i = 0; i < axis_len; ++i)
                    {
                        coords[ax] = i;
                        buffer[i] = expr.element(coords);
                    }
                    std::sort(buffer.begin(), buffer.end());
                    for (std::size_t qi = 0; qi < q.size(); ++qi)
                    {
                        double pos = q[qi] * static_cast<double>(axis_len - 1);
                        std::size_t idx_low = static_cast<std::size_t>(std::floor(pos));
                        std::size_t idx_high = static_cast<std::size_t>(std::ceil(pos));
                        double frac = pos - std::floor(pos);
                        value_type q_val;
                        if (idx_low == idx_high)
                            q_val = buffer[idx_low];
                        else
                            q_val = buffer[idx_low] * (1.0 - frac) + buffer[idx_high] * frac;
                        coords[ax] = qi;
                        result.element(coords) = q_val;
                    }
                }
                return result;
            }
            
            // Interquartile range
            template <class E>
            inline auto iqr(const xexpression<E>& e, std::size_t axis = 0)
            {
                auto q75 = quantile(e, 0.75, axis);
                auto q25 = quantile(e, 0.25, axis);
                return q75 - q25;
            }
            
            // --------------------------------------------------------------------
            // Summary statistics
            // --------------------------------------------------------------------
            
            template <class E>
            struct summary_stats
            {
                using value_type = typename E::value_type;
                value_type min, max, mean, median, stddev;
                std::size_t count;
            };
            
            template <class E>
            inline auto summary(const xexpression<E>& e)
            {
                const auto& expr = e.derived_cast();
                summary_stats<E> s;
                s.count = expr.size();
                if (s.count == 0) return s;
                auto minmax = std::minmax_element(expr.begin(), expr.end());
                s.min = *minmax.first;
                s.max = *minmax.second;
                s.mean = stats::mean(expr);
                s.median = stats::median(expr);
                s.stddev = stats::stddev(expr);
                return s;
            }
            
            // Describe (similar to pandas)
            template <class E>
            inline auto describe(const xexpression<E>& e, const std::vector<double>& percentiles = {0.25, 0.5, 0.75})
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                std::vector<std::pair<std::string, value_type>> result;
                result.push_back({"count", static_cast<value_type>(expr.size())});
                auto minmax = std::minmax_element(expr.begin(), expr.end());
                result.push_back({"min", *minmax.first});
                result.push_back({"max", *minmax.second});
                result.push_back({"mean", stats::mean(expr)});
                result.push_back({"std", stats::stddev(expr)});
                for (double p : percentiles)
                {
                    result.push_back({std::to_string(static_cast<int>(p * 100)) + "%", quantile(expr, p)(0)});
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Covariance and correlation
            // --------------------------------------------------------------------
            
            template <class E1, class E2>
            inline auto cov(const xexpression<E1>& e1, const xexpression<E2>& e2, std::size_t ddof = 1)
            {
                const auto& expr1 = e1.derived_cast();
                const auto& expr2 = e2.derived_cast();
                if (expr1.size() != expr2.size())
                    XTENSOR_THROW(std::invalid_argument, "cov: size mismatch");
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                value_type m1 = mean(expr1);
                value_type m2 = mean(expr2);
                value_type sum = 0;
                for (std::size_t i = 0; i < expr1.size(); ++i)
                    sum += (expr1.flat(i) - m1) * (expr2.flat(i) - m2);
                return sum / static_cast<value_type>(expr1.size() - ddof);
            }
            
            template <class E>
            inline auto cov_matrix(const xexpression<E>& e, std::size_t ddof = 1)
            {
                const auto& expr = e.derived_cast();
                if (expr.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "cov_matrix: requires 2D matrix (observations x variables)");
                std::size_t n_obs = expr.shape()[0];
                std::size_t n_vars = expr.shape()[1];
                using value_type = typename E::value_type;
                xarray_container<value_type> result(std::vector<std::size_t>{n_vars, n_vars});
                // Compute means of each variable
                std::vector<value_type> means(n_vars);
                for (std::size_t j = 0; j < n_vars; ++j)
                {
                    value_type sum = 0;
                    for (std::size_t i = 0; i < n_obs; ++i)
                        sum += expr(i, j);
                    means[j] = sum / static_cast<value_type>(n_obs);
                }
                for (std::size_t j1 = 0; j1 < n_vars; ++j1)
                {
                    for (std::size_t j2 = j1; j2 < n_vars; ++j2)
                    {
                        value_type sum = 0;
                        for (std::size_t i = 0; i < n_obs; ++i)
                            sum += (expr(i, j1) - means[j1]) * (expr(i, j2) - means[j2]);
                        result(j1, j2) = result(j2, j1) = sum / static_cast<value_type>(n_obs - ddof);
                    }
                }
                return result;
            }
            
            template <class E1, class E2>
            inline auto corrcoef(const xexpression<E1>& e1, const xexpression<E2>& e2)
            {
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                auto c = cov(e1, e2);
                auto s1 = stddev(e1);
                auto s2 = stddev(e2);
                if (s1 == 0 || s2 == 0) return value_type(0);
                return c / (s1 * s2);
            }
            
            template <class E>
            inline auto corr_matrix(const xexpression<E>& e)
            {
                auto cov_mat = cov_matrix(e);
                std::size_t n = cov_mat.shape()[0];
                auto result = cov_mat;
                for (std::size_t i = 0; i < n; ++i)
                {
                    for (std::size_t j = 0; j < n; ++j)
                    {
                        auto denom = std::sqrt(cov_mat(i, i) * cov_mat(j, j));
                        result(i, j) = denom > 0 ? cov_mat(i, j) / denom : 0;
                    }
                }
                return result;
            }
            
            // Spearman rank correlation
            template <class E1, class E2>
            inline auto spearmanr(const xexpression<E1>& e1, const xexpression<E2>& e2)
            {
                const auto& expr1 = e1.derived_cast();
                const auto& expr2 = e2.derived_cast();
                if (expr1.size() != expr2.size())
                    XTENSOR_THROW(std::invalid_argument, "spearmanr: size mismatch");
                auto rank1 = argsort(argsort(expr1));
                auto rank2 = argsort(argsort(expr2));
                return corrcoef(rank1, rank2);
            }
            
            // --------------------------------------------------------------------
            // Histogram and binning
            // --------------------------------------------------------------------
            
            template <class E>
            inline auto histogram(const xexpression<E>& e, std::size_t bins = 10,
                                  const std::pair<double, double>& range = {0, 0})
            {
                const auto& expr = e.derived_cast();
                auto flat = eval(expr);
                double min_val = range.first == range.second ? *std::min_element(flat.begin(), flat.end()) : range.first;
                double max_val = range.first == range.second ? *std::max_element(flat.begin(), flat.end()) : range.second;
                double bin_width = (max_val - min_val) / bins;
                
                std::vector<std::size_t> counts(bins, 0);
                for (auto val : flat)
                {
                    if (val < min_val || val > max_val) continue;
                    std::size_t idx = static_cast<std::size_t>((val - min_val) / bin_width);
                    if (idx == bins) idx = bins - 1;
                    counts[idx]++;
                }
                return counts;
            }
            
            template <class E>
            inline auto histogram2d(const xexpression<E>& x, const xexpression<E>& y, std::size_t bins = 10)
            {
                const auto& xexpr = x.derived_cast();
                const auto& yexpr = y.derived_cast();
                if (xexpr.size() != yexpr.size())
                    XTENSOR_THROW(std::invalid_argument, "histogram2d: size mismatch");
                double xmin = *std::min_element(xexpr.begin(), xexpr.end());
                double xmax = *std::max_element(xexpr.begin(), xexpr.end());
                double ymin = *std::min_element(yexpr.begin(), yexpr.end());
                double ymax = *std::max_element(yexpr.begin(), yexpr.end());
                double xbin = (xmax - xmin) / bins;
                double ybin = (ymax - ymin) / bins;
                xarray_container<std::size_t> result(std::vector<std::size_t>{bins, bins}, 0);
                for (std::size_t i = 0; i < xexpr.size(); ++i)
                {
                    std::size_t ix = static_cast<std::size_t>((xexpr.flat(i) - xmin) / xbin);
                    std::size_t iy = static_cast<std::size_t>((yexpr.flat(i) - ymin) / ybin);
                    if (ix < bins && iy < bins)
                        result(ix, iy)++;
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Bivariate statistics
            // --------------------------------------------------------------------
            
            template <class E1, class E2>
            inline auto linear_regression(const xexpression<E1>& x, const xexpression<E2>& y)
            {
                const auto& xexpr = x.derived_cast();
                const auto& yexpr = y.derived_cast();
                if (xexpr.size() != yexpr.size())
                    XTENSOR_THROW(std::invalid_argument, "linear_regression: size mismatch");
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                value_type xm = mean(xexpr);
                value_type ym = mean(yexpr);
                value_type num = 0, den = 0;
                for (std::size_t i = 0; i < xexpr.size(); ++i)
                {
                    num += (xexpr.flat(i) - xm) * (yexpr.flat(i) - ym);
                    den += (xexpr.flat(i) - xm) * (xexpr.flat(i) - xm);
                }
                value_type slope = num / den;
                value_type intercept = ym - slope * xm;
                return std::make_pair(slope, intercept);
            }
            
            // --------------------------------------------------------------------
            // Bootstrap and resampling
            // --------------------------------------------------------------------
            
            template <class E, class Func>
            inline auto bootstrap(const xexpression<E>& e, Func&& statistic, std::size_t n_resamples = 1000,
                                  std::size_t seed = 0)
            {
                const auto& expr = e.derived_cast();
                std::mt19937 gen(seed ? seed : std::random_device{}());
                std::uniform_int_distribution<std::size_t> dist(0, expr.size() - 1);
                using value_type = typename std::invoke_result_t<Func, decltype(expr)>;
                std::vector<value_type> results(n_resamples);
                auto sample = eval(expr);
                for (std::size_t b = 0; b < n_resamples; ++b)
                {
                    for (std::size_t i = 0; i < expr.size(); ++i)
                        sample.flat(i) = expr.flat(dist(gen));
                    results[b] = statistic(sample);
                }
                return results;
            }
            
            template <class E>
            inline auto bootstrap_ci(const xexpression<E>& e, double alpha = 0.05, std::size_t n_resamples = 1000)
            {
                auto boot_means = bootstrap(e, [](const auto& x){ return mean(x); }, n_resamples);
                std::sort(boot_means.begin(), boot_means.end());
                std::size_t lower_idx = static_cast<std::size_t>(n_resamples * alpha / 2);
                std::size_t upper_idx = static_cast<std::size_t>(n_resamples * (1 - alpha / 2));
                return std::make_pair(boot_means[lower_idx], boot_means[upper_idx]);
            }
            
            // --------------------------------------------------------------------
            // Outlier detection
            // --------------------------------------------------------------------
            
            template <class E>
            inline auto zscore(const xexpression<E>& e, std::size_t ddof = 0)
            {
                const auto& expr = e.derived_cast();
                auto m = mean(expr);
                auto s = stddev(expr, ddof);
                return (expr - m) / s;
            }
            
            template <class E>
            inline auto outlier_mask_iqr(const xexpression<E>& e, double factor = 1.5)
            {
                const auto& expr = e.derived_cast();
                auto q1 = quantile(expr, 0.25)(0);
                auto q3 = quantile(expr, 0.75)(0);
                auto iqr_val = q3 - q1;
                auto lower = q1 - factor * iqr_val;
                auto upper = q3 + factor * iqr_val;
                xarray_container<bool> mask(expr.shape());
                for (std::size_t i = 0; i < expr.size(); ++i)
                    mask.flat(i) = (expr.flat(i) < lower || expr.flat(i) > upper);
                return mask;
            }
            
            // --------------------------------------------------------------------
            // Entropy and information theory
            // --------------------------------------------------------------------
            
            template <class E>
            inline auto entropy(const xexpression<E>& e, double base = std::exp(1.0))
            {
                const auto& expr = e.derived_cast();
                using value_type = typename E::value_type;
                // Normalize to probability distribution
                auto sum = std::accumulate(expr.begin(), expr.end(), value_type(0));
                if (sum == 0) return 0.0;
                double ent = 0.0;
                for (auto val : expr)
                {
                    if (val > 0)
                    {
                        double p = static_cast<double>(val) / static_cast<double>(sum);
                        ent -= p * std::log(p);
                    }
                }
                if (base != std::exp(1.0))
                    ent /= std::log(base);
                return ent;
            }
            
            template <class E1, class E2>
            inline auto mutual_information(const xexpression<E1>& x, const xexpression<E2>& y, std::size_t bins = 10)
            {
                auto hist2d = histogram2d(x, y, bins);
                double total = static_cast<double>(std::accumulate(hist2d.begin(), hist2d.end(), std::size_t(0)));
                if (total == 0) return 0.0;
                double mi = 0.0;
                std::vector<double> px(bins, 0.0), py(bins, 0.0);
                for (std::size_t i = 0; i < bins; ++i)
                    for (std::size_t j = 0; j < bins; ++j)
                        px[i] += static_cast<double>(hist2d(i, j)) / total;
                for (std::size_t j = 0; j < bins; ++j)
                    for (std::size_t i = 0; i < bins; ++i)
                        py[j] += static_cast<double>(hist2d(i, j)) / total;
                for (std::size_t i = 0; i < bins; ++i)
                {
                    for (std::size_t j = 0; j < bins; ++j)
                    {
                        double pxy = static_cast<double>(hist2d(i, j)) / total;
                        if (pxy > 0)
                            mi += pxy * std::log(pxy / (px[i] * py[j]));
                    }
                }
                return mi;
            }
            
        } // namespace stats
        
        // Bring into xt namespace
        using stats::mean;
        using stats::weighted_mean;
        using stats::harmonic_mean;
        using stats::geometric_mean;
        using stats::median;
        using stats::mode;
        using stats::var;
        using stats::stddev;
        using stats::sem;
        using stats::skew;
        using stats::kurtosis;
        using stats::moment;
        using stats::quantile;
        using stats::percentile;
        using stats::quantiles;
        using stats::iqr;
        using stats::summary;
        using stats::describe;
        using stats::cov;
        using stats::cov_matrix;
        using stats::corrcoef;
        using stats::corr_matrix;
        using stats::spearmanr;
        using stats::histogram;
        using stats::histogram2d;
        using stats::linear_regression;
        using stats::bootstrap;
        using stats::bootstrap_ci;
        using stats::zscore;
        using stats::outlier_mask_iqr;
        using stats::entropy;
        using stats::mutual_information;
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XSTATS_HPP

// math/xstats.hpp