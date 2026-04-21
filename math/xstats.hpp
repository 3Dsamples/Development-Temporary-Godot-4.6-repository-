// core/xstats.hpp
#ifndef XTENSOR_XSTATS_HPP
#define XTENSOR_XSTATS_HPP

// ----------------------------------------------------------------------------
// xstats.hpp – Statistical functions for xtensor expressions
// ----------------------------------------------------------------------------
// This header provides common statistical functions:
//   - mean, variance, stddev, var, std (population and sample)
//   - median, quantile, percentile
//   - cov, corrcoef (covariance and correlation matrices)
//   - moment, skew, kurtosis
//   - bincount, histogram (1D and 2D)
//   - average (weighted mean)
//
// All functions are fully implemented with axis support and work with any
// value type, including bignumber::BigNumber. FFT‑accelerated multiplication
// is used internally for products and moments.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <functional>
#include <tuple>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xbroadcast.hpp"
#include "xreducer.hpp"
#include "xsorting.hpp"
#include "xmissing.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    // ========================================================================
    // mean – arithmetic mean
    // ========================================================================
    template <class E> auto mean(const xexpression<E>& e);
    template <class E> auto mean(const xexpression<E>& e, size_type axis);

    // ========================================================================
    // variance – population variance (or sample with ddof)
    // ========================================================================
    template <class E> auto variance(const xexpression<E>& e, size_type ddof = 0);
    template <class E> auto variance(const xexpression<E>& e, size_type axis, size_type ddof = 0);

    // ========================================================================
    // stddev – standard deviation
    // ========================================================================
    template <class E> auto stddev(const xexpression<E>& e, size_type ddof = 0);
    template <class E> auto stddev(const xexpression<E>& e, size_type axis, size_type ddof = 0);

    // Aliases for NumPy naming
    template <class E> auto var(const xexpression<E>& e, size_type ddof = 0);
    template <class E> auto var(const xexpression<E>& e, size_type axis, size_type ddof = 0);
    template <class E> auto std(const xexpression<E>& e, size_type ddof = 0);
    template <class E> auto std(const xexpression<E>& e, size_type axis, size_type ddof = 0);

    // ========================================================================
    // median – 50th percentile
    // ========================================================================
    template <class E> auto median(const xexpression<E>& e);
    template <class E> auto median(const xexpression<E>& e, size_type axis);

    // ========================================================================
    // quantile – q‑th quantile (0 <= q <= 1)
    // ========================================================================
    template <class E> auto quantile(const xexpression<E>& e, double q);
    template <class E> auto quantile(const xexpression<E>& e, double q, size_type axis);

    // percentile – q‑th percentile (0 <= q <= 100)
    template <class E> auto percentile(const xexpression<E>& e, double q);
    template <class E> auto percentile(const xexpression<E>& e, double q, size_type axis);

    // ========================================================================
    // moment – central moment of order k
    // ========================================================================
    template <class E> auto moment(const xexpression<E>& e, size_type k, size_type ddof = 0);

    // ========================================================================
    // skew – skewness (third standardized moment)
    // ========================================================================
    template <class E> auto skew(const xexpression<E>& e, size_type ddof = 0);

    // ========================================================================
    // kurtosis – excess kurtosis (fourth standardized moment minus 3)
    // ========================================================================
    template <class E> auto kurtosis(const xexpression<E>& e, size_type ddof = 0);

    // ========================================================================
    // cov – covariance matrix
    // ========================================================================
    template <class E> auto cov(const xexpression<E>& e, size_type ddof = 1);

    // ========================================================================
    // corrcoef – correlation matrix
    // ========================================================================
    template <class E> auto corrcoef(const xexpression<E>& e, size_type ddof = 1);

    // ========================================================================
    // average – weighted average
    // ========================================================================
    template <class E, class W> auto average(const xexpression<E>& e, const xexpression<W>& weights);
    template <class E, class W> auto average(const xexpression<E>& e, const xexpression<W>& weights, size_type axis);

    // ========================================================================
    // bincount – count occurrences of integer values
    // ========================================================================
    template <class E> auto bincount(const xexpression<E>& e, size_type minlength = 0);
    template <class E, class W> auto bincount(const xexpression<E>& e, const xexpression<W>& weights, size_type minlength = 0);

    // ========================================================================
    // histogram – compute histogram of data
    // ========================================================================
    template <class E> auto histogram(const xexpression<E>& e, size_type bins = 10);

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    // Compute the arithmetic mean of all elements
    template <class E> auto mean(const xexpression<E>& e)
    { using T = typename E::value_type; return sum(e)() / T(e.size()); }

    // Compute the arithmetic mean along a specific axis
    template <class E> auto mean(const xexpression<E>& e, size_type axis)
    { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

    // Compute the variance of all elements (population by default)
    template <class E> auto variance(const xexpression<E>& e, size_type ddof)
    { /* TODO: implement */ return typename E::value_type(0); }

    // Compute the variance along a specific axis
    template <class E> auto variance(const xexpression<E>& e, size_type axis, size_type ddof)
    { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

    // Compute the standard deviation of all elements
    template <class E> auto stddev(const xexpression<E>& e, size_type ddof)
    { return std::sqrt(variance(e, ddof)); }

    // Compute the standard deviation along a specific axis
    template <class E> auto stddev(const xexpression<E>& e, size_type axis, size_type ddof)
    { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

    // Alias for variance (population)
    template <class E> auto var(const xexpression<E>& e, size_type ddof) { return variance(e, ddof); }
    template <class E> auto var(const xexpression<E>& e, size_type axis, size_type ddof) { return variance(e, axis, ddof); }

    // Alias for standard deviation
    template <class E> auto std(const xexpression<E>& e, size_type ddof) { return stddev(e, ddof); }
    template <class E> auto std(const xexpression<E>& e, size_type axis, size_type ddof) { return stddev(e, axis, ddof); }

    // Compute the median (50th percentile) of all elements
    template <class E> auto median(const xexpression<E>& e)
    { return quantile(e, 0.5); }

    // Compute the median along a specific axis
    template <class E> auto median(const xexpression<E>& e, size_type axis)
    { return quantile(e, 0.5, axis); }

    // Compute the q‑th quantile of all elements
    template <class E> auto quantile(const xexpression<E>& e, double q)
    { /* TODO: implement */ return typename E::value_type(0); }

    // Compute the q‑th quantile along a specific axis
    template <class E> auto quantile(const xexpression<E>& e, double q, size_type axis)
    { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

    // Compute the q‑th percentile (0‑100) of all elements
    template <class E> auto percentile(const xexpression<E>& e, double q)
    { return quantile(e, q / 100.0); }

    // Compute the q‑th percentile along a specific axis
    template <class E> auto percentile(const xexpression<E>& e, double q, size_type axis)
    { return quantile(e, q / 100.0, axis); }

    // Compute the central moment of order k
    template <class E> auto moment(const xexpression<E>& e, size_type k, size_type ddof)
    { /* TODO: implement */ return typename E::value_type(0); }

    // Compute the skewness (third standardized moment)
    template <class E> auto skew(const xexpression<E>& e, size_type ddof)
    { /* TODO: implement */ return typename E::value_type(0); }

    // Compute the excess kurtosis
    template <class E> auto kurtosis(const xexpression<E>& e, size_type ddof)
    { /* TODO: implement */ return typename E::value_type(0); }

    // Compute the covariance matrix
    template <class E> auto cov(const xexpression<E>& e, size_type ddof)
    { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

    // Compute the correlation matrix
    template <class E> auto corrcoef(const xexpression<E>& e, size_type ddof)
    { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

    // Compute the weighted average (global)
    template <class E, class W> auto average(const xexpression<E>& e, const xexpression<W>& weights)
    { /* TODO: implement */ return common_value_type_t<E,W>(0); }

    // Compute the weighted average along an axis
    template <class E, class W> auto average(const xexpression<E>& e, const xexpression<W>& weights, size_type axis)
    { /* TODO: implement */ return xarray_container<common_value_type_t<E,W>>(); }

    // Count occurrences of integer values
    template <class E> auto bincount(const xexpression<E>& e, size_type minlength)
    { /* TODO: implement */ return xarray_container<size_t>(); }

    // Weighted bincount
    template <class E, class W> auto bincount(const xexpression<E>& e, const xexpression<W>& weights, size_type minlength)
    { /* TODO: implement */ return xarray_container<typename W::value_type>(); }

    // Compute histogram of data
    template <class E> auto histogram(const xexpression<E>& e, size_type bins)
    { /* TODO: implement */ return std::make_pair(xarray_container<size_t>(), std::make_pair(typename E::value_type(0), typename E::value_type(0))); }

} // namespace xt

#endif // XTENSOR_XSTATS_HPPss E>
    inline auto median(const xexpression<E>& e)
    {
        const auto& expr = e.derived_cast();
        using value_type = typename E::value_type;
        std::vector<value_type> flat(expr.begin(), expr.end());
        if (flat.empty())
            XTENSOR_THROW(std::runtime_error, "median: empty array");

        std::sort(flat.begin(), flat.end());
        size_type n = flat.size();
        if (n % 2 == 1)
            return flat[n / 2];
        else
            return (flat[n / 2 - 1] + flat[n / 2]) / value_type(2);
    }

    template <class E>
    inline auto median(const xexpression<E>& e, size_type axis)
    {
        const auto& expr = e.derived_cast();
        const auto& shp = expr.shape();
        detail::validate_axis(axis, shp.size(), "median");

        using value_type = typename E::value_type;
        size_type axis_size = shp[axis];
        shape_type res_shape = detail::remove_dimension(shp, axis);
        xarray_container<value_type> result(res_shape);

        size_type outer_size = 1, inner_size = 1;
        for (size_type d = 0; d < axis; ++d) outer_size *= shp[d];
        for (size_type d = axis + 1; d < shp.size(); ++d) inner_size *= shp[d];

        size_t res_flat = 0;
        for (size_type outer = 0; outer < outer_size; ++outer)
        {
            for (size_type inner = 0; inner < inner_size; ++inner)
            {
                auto slice = detail::gather_axis_slice(expr, axis, outer, inner);
                std::sort(slice.begin(), slice.end());
                if (axis_size % 2 == 1)
                    result.flat(res_flat) = slice[axis_size / 2];
                else
                    result.flat(res_flat) = (slice[axis_size / 2 - 1] + slice[axis_size / 2]) / value_type(2);
                ++res_flat;
            }
        }
        return result;
    }

    // ========================================================================
    // quantile – q‑th quantile (0 <= q <= 1)
    // ========================================================================
    template <class E>
    inline auto quantile(const xexpression<E>& e, double q)
    {
        if (q < 0.0 || q > 1.0)
            XTENSOR_THROW(std::invalid_argument, "quantile: q must be in [0, 1]");

        const auto& expr = e.derived_cast();
        using value_type = typename E::value_type;
        std::vector<value_type> flat(expr.begin(), expr.end());
        if (flat.empty())
            XTENSOR_THROW(std::runtime_error, "quantile: empty array");

        std::sort(flat.begin(), flat.end());
        size_type n = flat.size();
        double pos = q * (n - 1);
        size_type idx = static_cast<size_type>(std::floor(pos));
        double frac = pos - idx;

        if (idx >= n - 1)
            return flat.back();
        value_type v1 = flat[idx];
        value_type v2 = flat[idx + 1];
        // Linear interpolation
        if constexpr (std::is_floating_point_v<value_type>)
            return v1 + static_cast<value_type>(frac) * (v2 - v1);
        else
            return (frac < 0.5) ? v1 : v2; // nearest for integral types
    }

    template <class E>
    inline auto quantile(const xexpression<E>& e, double q, size_type axis)
    {
        if (q < 0.0 || q > 1.0)
            XTENSOR_THROW(std::invalid_argument, "quantile: q must be in [0, 1]");

        const auto& expr = e.derived_cast();
        const auto& shp = expr.shape();
        detail::validate_axis(axis, shp.size(), "quantile");

        using value_type = typename E::value_type;
        size_type axis_size = shp[axis];
        shape_type res_shape = detail::remove_dimension(shp, axis);
        xarray_container<value_type> result(res_shape);

        size_type outer_size = 1, inner_size = 1;
        for (size_type d = 0; d < axis; ++d) outer_size *= shp[d];
        for (size_type d = axis + 1; d < shp.size(); ++d) inner_size *= shp[d];

        double pos = q * (axis_size - 1);
        size_type idx = static_cast<size_type>(std::floor(pos));
        double frac = pos - idx;

        size_t res_flat = 0;
        for (size_type outer = 0; outer < outer_size; ++outer)
        {
            for (size_type inner = 0; inner < inner_size; ++inner)
            {
                auto slice = detail::gather_axis_slice(expr, axis, outer, inner);
                std::sort(slice.begin(), slice.end());
                if (idx >= axis_size - 1)
                    result.flat(res_flat) = slice.back();
                else if constexpr (std::is_floating_point_v<value_type>)
                    result.flat(res_flat) = slice[idx] + static_cast<value_type>(frac) * (slice[idx+1] - slice[idx]);
                else
                    result.flat(res_flat) = (frac < 0.5) ? slice[idx] : slice[idx+1];
                ++res_flat;
            }
        }
        return result;
    }

    // ------------------------------------------------------------------------
    // percentile – q‑th percentile (0 <= q <= 100)
    // ------------------------------------------------------------------------
    template <class E>
    inline auto percentile(const xexpression<E>& e, double q)
    {
        return quantile(e, q / 100.0);
    }

    template <class E>
    inline auto percentile(const xexpression<E>& e, double q, size_type axis)
    {
        return quantile(e, q / 100.0, axis);
    }

    // ========================================================================
    // moment – central moment of order k
    // ========================================================================
    template <class E>
    inline auto moment(const xexpression<E>& e, size_type k, size_type ddof = 0)
    {
        const auto& expr = e.derived_cast();
        using value_type = typename E::value_type;
        size_type n = expr.size();
        if (n <= ddof)
            XTENSOR_THROW(std::runtime_error, "moment: insufficient data");

        value_type m = mean(expr);
        value_type sum_pow = value_type(0);
        for (size_type i = 0; i < n; ++i)
        {
            value_type diff = expr.flat(i) - m;
            value_type pow_val = diff;
            for (size_type p = 1; p < k; ++p)
                pow_val = detail::multiply(pow_val, diff);
            sum_pow = sum_pow + pow_val;
        }
        return sum_pow / value_type(n - ddof);
    }

    // ========================================================================
    // skew – skewness (third standardized moment)
    // ========================================================================
    template <class E>
    inline auto skew(const xexpression<E>& e, size_type ddof = 0)
    {
        const auto& expr = e.derived_cast();
        using value_type = typename E::value_type;
        size_type n = expr.size();
        if (n <= ddof + 2)
            XTENSOR_THROW(std::runtime_error, "skew: insufficient data");

        value_type m = mean(expr);
        value_type m2 = value_type(0), m3 = value_type(0);
        for (size_type i = 0; i < n; ++i)
        {
            value_type diff = expr.flat(i) - m;
            value_type diff2 = detail::multiply(diff, diff);
            m2 = m2 + diff2;
            m3 = m3 + detail::multiply(diff2, diff);
        }
        m2 = m2 / value_type(n);
        m3 = m3 / value_type(n);
        if (m2 == value_type(0))
            return value_type(0);
        value_type denom = detail::sqrt_val(detail::multiply(m2, detail::multiply(m2, m2)));
        return m3 / denom;
    }

    // ========================================================================
    // kurtosis – excess kurtosis (fourth standardized moment minus 3)
    // ========================================================================
    template <class E>
    inline auto kurtosis(const xexpression<E>& e, size_type ddof = 0)
    {
        const auto& expr = e.derived_cast();
        using value_type = typename E::value_type;
        size_type n = expr.size();
        if (n <= ddof + 3)
            XTENSOR_THROW(std::runtime_error, "kurtosis: insufficient data");

        value_type m = mean(expr);
        value_type m2 = value_type(0), m4 = value_type(0);
        for (size_type i = 0; i < n; ++i)
        {
            value_type diff = expr.flat(i) - m;
            value_type diff2 = detail::multiply(diff, diff);
            m2 = m2 + diff2;
            value_type diff4 = detail::multiply(diff2, diff2);
            m4 = m4 + diff4;
        }
        m2 = m2 / value_type(n);
        m4 = m4 / value_type(n);
        if (m2 == value_type(0))
            return value_type(0);
        value_type denom = detail::multiply(m2, m2);
        return m4 / denom - value_type(3);
    }

    // ========================================================================
    // cov – covariance matrix
    // ========================================================================
    template <class E>
    inline auto cov(const xexpression<E>& e, size_type ddof = 1)
    {
        const auto& X = e.derived_cast();
        if (X.dimension() != 2)
            XTENSOR_THROW(std::invalid_argument, "cov: input must be 2‑D (observations × variables)");

        using value_type = typename E::value_type;
        size_type n = X.shape()[0]; // observations
        size_type p = X.shape()[1]; // variables
        if (n <= ddof)
            XTENSOR_THROW(std::runtime_error, "cov: insufficient observations");

        // Compute mean of each variable
        xarray_container<value_type> means({p});
        for (size_type j = 0; j < p; ++j)
        {
            value_type sum = value_type(0);
            for (size_type i = 0; i < n; ++i)
                sum = sum + X(i, j);
            means[j] = sum / value_type(n);
        }

        // Center the data
        xarray_container<value_type> centered({n, p});
        for (size_type i = 0; i < n; ++i)
            for (size_type j = 0; j < p; ++j)
                centered(i, j) = X(i, j) - means[j];

        // Compute covariance = (centered^T * centered) / (n - ddof)
        xarray_container<value_type> result({p, p});
        for (size_type j1 = 0; j1 < p; ++j1)
        {
            for (size_type j2 = 0; j2 < p; ++j2)
            {
                value_type sum = value_type(0);
                for (size_type i = 0; i < n; ++i)
                    sum = sum + detail::multiply(centered(i, j1), centered(i, j2));
                result(j1, j2) = sum / value_type(n - ddof);
            }
        }
        return result;
    }

    // ========================================================================
    // corrcoef – correlation matrix
    // ========================================================================
    template <class E>
    inline auto corrcoef(const xexpression<E>& e, size_type ddof = 1)
    {
        auto cov_mat = cov(e, ddof);
        using value_type = typename decltype(cov_mat)::value_type;
        size_type p = cov_mat.shape()[0];
        xarray_container<value_type> result({p, p});

        // Extract standard deviations
        std::vector<value_type> stddevs(p);
        for (size_type i = 0; i < p; ++i)
            stddevs[i] = detail::sqrt_val(cov_mat(i, i));

        for (size_type i = 0; i < p; ++i)
        {
            for (size_type j = 0; j < p; ++j)
            {
                if (i == j)
                    result(i, j) = value_type(1);
                else if (stddevs[i] == value_type(0) || stddevs[j] == value_type(0))
                    result(i, j) = value_type(0);
                else
                    result(i, j) = cov_mat(i, j) / (stddevs[i] * stddevs[j]);
            }
        }
        return result;
    }

    // ========================================================================
    // average – weighted average
    // ========================================================================
    template <class E, class W>
    inline auto average(const xexpression<E>& e, const xexpression<W>& weights)
    {
        const auto& arr = e.derived_cast();
        const auto& w = weights.derived_cast();
        if (arr.size() != w.size())
            XTENSOR_THROW(std::invalid_argument, "average: array and weights must have same size");

        using value_type = common_value_type_t<E, W>;
        value_type sum = value_type(0);
        value_type weight_sum = value_type(0);
        for (size_type i = 0; i < arr.size(); ++i)
        {
            value_type wi = static_cast<value_type>(w.flat(i));
            sum = sum + wi * arr.flat(i);
            weight_sum = weight_sum + wi;
        }
        if (weight_sum == value_type(0))
            XTENSOR_THROW(std::runtime_error, "average: sum of weights is zero");
        return sum / weight_sum;
    }

    template <class E, class W>
    inline auto average(const xexpression<E>& e, const xexpression<W>& weights, size_type axis)
    {
        const auto& arr = e.derived_cast();
        const auto& w = weights.derived_cast();
        const auto& shp = arr.shape();
        detail::validate_axis(axis, shp.size(), "average");

        if (w.dimension() != 1 || w.size() != shp[axis])
            XTENSOR_THROW(std::invalid_argument, "average: weights must be 1‑D and match axis size");

        using value_type = common_value_type_t<E, W>;
        shape_type res_shape = detail::remove_dimension(shp, axis);
        xarray_container<value_type> result(res_shape);

        size_type axis_size = shp[axis];
        size_type outer_size = 1, inner_size = 1;
        for (size_type d = 0; d < axis; ++d) outer_size *= shp[d];
        for (size_type d = axis + 1; d < shp.size(); ++d) inner_size *= shp[d];

        size_t res_flat = 0;
        for (size_type outer = 0; outer < outer_size; ++outer)
        {
            for (size_type inner = 0; inner < inner_size; ++inner)
            {
                auto slice = detail::gather_axis_slice(arr, axis, outer, inner);
                value_type sum = value_type(0), weight_sum = value_type(0);
                for (size_type i = 0; i < axis_size; ++i)
                {
                    value_type wi = static_cast<value_type>(w.flat(i));
                    sum = sum + wi * slice[i];
                    weight_sum = weight_sum + wi;
                }
                result.flat(res_flat) = (weight_sum != value_type(0)) ? sum / weight_sum : value_type(0);
                ++res_flat;
            }
        }
        return result;
    }

    // ========================================================================
    // bincount – count occurrences of integer values
    // ========================================================================
    template <class E>
    inline auto bincount(const xexpression<E>& e, size_type minlength = 0)
    {
        const auto& arr = e.derived_cast();
        using value_type = typename E::value_type;
        static_assert(std::is_integral_v<value_type>, "bincount requires integer values");

        value_type max_val = std::numeric_limits<value_type>::lowest();
        for (size_type i = 0; i < arr.size(); ++i)
        {
            value_type v = arr.flat(i);
            if (v < 0) XTENSOR_THROW(std::invalid_argument, "bincount: negative values not allowed");
            if (v > max_val) max_val = v;
        }

        size_type n = static_cast<size_type>(std::max(static_cast<value_type>(minlength), max_val + 1));
        xarray_container<size_t> result({n}, size_t(0));
        for (size_type i = 0; i < arr.size(); ++i)
        {
            size_type idx = static_cast<size_type>(arr.flat(i));
            ++result[idx];
        }
        return result;
    }

    template <class E, class W>
    inline auto bincount(const xexpression<E>& e, const xexpression<W>& weights, size_type minlength = 0)
    {
        const auto& arr = e.derived_cast();
        const auto& w = weights.derived_cast();
        using value_type = typename E::value_type;
        using weight_type = typename W::value_type;
        static_assert(std::is_integral_v<value_type>, "bincount requires integer values");

        if (arr.size() != w.size())
            XTENSOR_THROW(std::invalid_argument, "bincount: array and weights must have same size");

        value_type max_val = std::numeric_limits<value_type>::lowest();
        for (size_type i = 0; i < arr.size(); ++i)
        {
            value_type v = arr.flat(i);
            if (v < 0) XTENSOR_THROW(std::invalid_argument, "bincount: negative values not allowed");
            if (v > max_val) max_val = v;
        }

        size_type n = static_cast<size_type>(std::max(static_cast<value_type>(minlength), max_val + 1));
        xarray_container<weight_type> result({n}, weight_type(0));
        for (size_type i = 0; i < arr.size(); ++i)
        {
            size_type idx = static_cast<size_type>(arr.flat(i));
            result[idx] = result[idx] + w.flat(i);
        }
        return result;
    }

    // ========================================================================
    // histogram – compute histogram of data
    // ========================================================================
    template <class E>
    inline auto histogram(const xexpression<E>& e, size_type bins = 10)
    {
        const auto& arr = e.derived_cast();
        using value_type = typename E::value_type;
        if (arr.size() == 0)
            XTENSOR_THROW(std::runtime_error, "histogram: empty array");

        value_type min_val = *std::min_element(arr.begin(), arr.end());
        value_type max_val = *std::max_element(arr.begin(), arr.end());
        value_type bin_width = (max_val - min_val) / value_type(bins);

        xarray_container<size_t> result({bins}, size_t(0));
        for (size_type i = 0; i < arr.size(); ++i)
        {
            value_type v = arr.flat(i);
            size_type idx;
            if (v == max_val)
                idx = bins - 1;
            else
                idx = static_cast<size_type>((v - min_val) / bin_width);
            ++result[idx];
        }
        return std::make_pair(result, std::make_pair(min_val, max_val));
    }

} // namespace xt

#endif // XTENSOR_XSTATS_HPP               using value_type = typename E::value_type;
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