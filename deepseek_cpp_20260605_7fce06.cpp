//File 0028 : core/xstatistics.hpp
//Statistical functions: mean, variance, skewness, kurtosis, covariance, correlation, histogram, percentiles with SIMD reductions.
#ifndef XTENSOR_XSTATISTICS_HPP
#define XTENSOR_XSTATISTICS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xarray.hpp"
#include "xeval.hpp"
#include "xreducer.hpp"
#include "xsort.hpp"
#include "xmanipulation.hpp"

namespace xt {
namespace statistics {

    /*********************************************
     * Mean
     *********************************************/
    /**
     * Compute the arithmetic mean over all elements.
     */
    template <class E>
    inline auto mean(const E& e) {
        auto s = xt::sum(e)();
        return s / static_cast<double>(e.size());
    }

    /**
     * Compute the mean along a given axis.
     */
    template <class E>
    inline auto mean(const E& e, std::size_t axis) {
        auto s = xt::sum(e, axis);
        double count = static_cast<double>(e.shape()[axis]);
        return s / count;
    }

    /**
     * Compute the mean over multiple axes.
     */
    template <class E, class X>
    inline auto mean(const E& e, X&& axes) {
        auto s = xt::sum(e, std::forward<X>(axes));
        double total = 1.0;
        auto sh = e.shape();
        for (auto ax : axes) total *= sh[ax];
        return s / total;
    }

    /*********************************************
     * Variance (population: ddof=0, sample: ddof=1)
     *********************************************/
    /**
     * Compute variance over all elements, with given degrees of freedom (ddof).
     */
    template <class E>
    inline auto variance(const E& e, int ddof = 0) {
        double m = mean(e);
        auto diff = xt::eval(e) - m;
        auto sq = diff * diff;
        double s = xt::sum(sq)();
        double N = static_cast<double>(e.size()) - ddof;
        return s / N;
    }

    /**
     * Variance along an axis.
     */
    template <class E>
    inline auto variance(const E& e, std::size_t axis, int ddof = 0) {
        auto m = mean(e, axis);
        auto diff = e - m;
        auto sq = diff * diff;
        auto s = xt::sum(sq, axis);
        double N = static_cast<double>(e.shape()[axis]) - ddof;
        return s / N;
    }

    /*********************************************
     * Standard deviation
     *********************************************/
    /**
     * Standard deviation over all elements.
     */
    template <class E>
    inline auto stddev(const E& e, int ddof = 0) {
        return xt::sqrt(variance(e, ddof));
    }

    /**
     * Standard deviation along an axis.
     */
    template <class E>
    inline auto stddev(const E& e, std::size_t axis, int ddof = 0) {
        return xt::sqrt(variance(e, axis, ddof));
    }

    /*********************************************
     * Skewness (Fisher-Pearson coefficient)
     *********************************************/
    /**
     * Compute skewness over all elements.
     */
    template <class E>
    inline auto skew(const E& e) {
        double m = mean(e);
        double s = stddev(e, 1); // sample std
        double N = static_cast<double>(e.size());
        auto diff = xt::eval(e) - m;
        auto cubed = diff * diff * diff;
        double sum_cub = xt::sum(cubed)();
        return (N / ((N-1)*(N-2))) * sum_cub / (s*s*s);
    }

    /*********************************************
     * Kurtosis (excess kurtosis, Fisher)
     *********************************************/
    /**
     * Compute excess kurtosis over all elements.
     */
    template <class E>
    inline auto kurtosis(const E& e) {
        double m = mean(e);
        double N = static_cast<double>(e.size());
        auto diff = xt::eval(e) - m;
        auto s2 = xt::sum(diff * diff)() / N; // biased variance
        auto s4 = xt::sum(diff * diff * diff * diff)() / N;
        return (s4 / (s2 * s2)) - 3.0;
    }

    /*********************************************
     * Covariance matrix (2D input: rows=variables, cols=observations)
     *********************************************/
    /**
     * Compute covariance matrix of a set of variables (rows=variables, columns=observations).
     */
    template <class E>
    inline auto cov(const E& e) {
        auto sh = e.shape();
        if (sh.size() != 2)
            throw std::runtime_error("cov requires 2D array (variables x observations).");
        std::size_t n_vars = sh[0];
        std::size_t n_obs = sh[1];
        auto m = xt::mean(e, 1); // mean per variable
        auto centered = e - xt::view(m, xt::all(), xt::newaxis()); // broadcast?
        // Use matrix multiplication: (centered * centered.T) / (n_obs - 1)
        auto cov = xt::linalg::matmul(centered, xt::transpose(centered));
        return cov / static_cast<double>(n_obs - 1);
    }

    /*********************************************
     * Pearson correlation coefficient matrix
     *********************************************/
    /**
     * Compute Pearson correlation matrix.
     */
    template <class E>
    inline auto corrcoef(const E& e) {
        auto cov_mat = cov(e);
        auto std_vec = xt::sqrt(xt::diagonal(cov_mat));
        // Normalize: cov(i,j) / (std[i]*std[j])
        auto corr = xt::eval(cov_mat);
        auto n = corr.shape()[0];
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                corr(i,j) = cov_mat(i,j) / (std_vec[i] * std_vec[j]);
        return corr;
    }

    // Utility: extract diagonal of a 2D matrix as a 1D array
    template <class E>
    inline auto diagonal(const E& mat) {
        auto sh = mat.shape();
        if (sh.size() != 2) throw std::runtime_error("diagonal requires 2D matrix.");
        std::size_t n = std::min(sh[0], sh[1]);
        xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> diag({n});
        for (std::size_t i = 0; i < n; ++i) diag[i] = mat(i,i);
        return diag;
    }

    /*********************************************
     * Percentile / Quantile
     *********************************************/
    /**
     * Compute the q-th percentile of a 1D array using linear interpolation.
     */
    template <class E>
    inline auto percentile(const E& e, double q) {
        if (q < 0.0 || q > 100.0) throw std::runtime_error("Percentile must be in [0,100].");
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("percentile requires 1D array.");
        std::size_t n = arr.size();
        if (n == 0) return static_cast<typename std::decay_t<E>::value_type>(0);
        auto sorted = xt::sort(arr);
        double index = q / 100.0 * (n - 1);
        std::size_t lo = static_cast<std::size_t>(std::floor(index));
        std::size_t hi = std::min(lo+1, n-1);
        double frac = index - lo;
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }

    /**
     * Compute the median (50th percentile).
     */
    template <class E>
    inline auto median(const E& e) {
        return percentile(e, 50.0);
    }

    /**
     * Compute multiple percentiles. Returns array of length equal to q's size.
     */
    template <class E, class Q>
    inline auto percentile(const E& e, const Q& q_values) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("percentile requires 1D array.");
        auto sorted = xt::sort(arr);
        std::size_t n = arr.size();
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({q_values.size()});
        for (std::size_t i = 0; i < q_values.size(); ++i) {
            double q = q_values[i];
            double index = q / 100.0 * (n - 1);
            std::size_t lo = static_cast<std::size_t>(std::floor(index));
            std::size_t hi = std::min(lo+1, n-1);
            double frac = index - lo;
            result[i] = sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
        }
        return result;
    }

    /*********************************************
     * Histogram (counts in bins)
     *********************************************/
    /**
     * Compute histogram with given bin edges. Returns counts and bin edges.
     */
    template <class E, class BinEdges>
    inline auto histogram(const E& e, const BinEdges& edges) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(e);
        auto bins_arr = xt::eval(edges);
        if (bins_arr.dimension() != 1) throw std::runtime_error("Bin edges must be 1D.");
        std::size_t nbins = bins_arr.size() - 1;
        xarray_container<uvector<std::size_t>, DEFAULT_LAYOUT, std::vector<std::size_t>> counts({nbins}, 0);
        for (std::size_t i = 0; i < arr.size(); ++i) {
            T val = arr[i];
            if (val < bins_arr[0] || val >= bins_arr[nbins]) continue;
            auto it = std::upper_bound(bins_arr.data(), bins_arr.data() + bins_arr.size(), val);
            std::size_t idx = static_cast<std::size_t>(it - bins_arr.data() - 1);
            if (idx < nbins) counts[idx]++;
        }
        return std::make_pair(counts, xt::eval(edges));
    }

    /*********************************************
     * Moments of given order
     *********************************************/
    /**
     * Compute the n-th central moment.
     */
    template <class E>
    inline auto moment(const E& e, std::size_t order) {
        double m = mean(e);
        auto diff = xt::eval(e) - m;
        auto powered = xt::pow(diff, static_cast<double>(order));
        return xt::sum(powered)() / static_cast<double>(e.size());
    }

    /*********************************************
     * Correlation coefficient between two 1D arrays
     *********************************************/
    /**
     * Compute Pearson correlation between two 1D arrays.
     */
    template <class E1, class E2>
    inline auto corr(const E1& x, const E2& y) {
        auto a = xt::eval(x), b = xt::eval(y);
        if (a.dimension()!=1 || b.dimension()!=1 || a.size()!=b.size())
            throw std::runtime_error("corr requires two 1D arrays of same length.");
        double n = static_cast<double>(a.size());
        double mx = mean(a), my = mean(b);
        auto dx = a - mx, dy = b - my;
        double cov = xt::sum(dx * dy)() / (n - 1);
        double sx = stddev(a, 1), sy = stddev(b, 1);
        return cov / (sx * sy);
    }

} // namespace statistics
} // namespace xt

#endif // XTENSOR_XSTATISTICS_HPP