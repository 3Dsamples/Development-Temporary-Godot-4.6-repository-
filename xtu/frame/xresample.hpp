// include/xtu/frame/xresample.hpp
// xtensor-unified - Time series resampling and rolling window operations
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_FRAME_XRESAMPLE_HPP
#define XTU_FRAME_XRESAMPLE_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/containers/xarray.hpp"
#include "xtu/frame/xvariable.hpp"
#include "xtu/frame/xcoordinate_system.hpp"
#include "xtu/frame/xgroupby.hpp"
#include "xtu/math/xreducer.hpp"

XTU_NAMESPACE_BEGIN
namespace frame {

// #############################################################################
// Time frequency rules
// #############################################################################
struct frequency {
    std::string name;
    int64_t nanoseconds;
    
    static frequency from_string(const std::string& freq) {
        if (freq.empty()) return {"", 0};
        char unit = freq.back();
        int64_t mult = 1;
        std::string num_part = freq.substr(0, freq.size() - 1);
        if (!num_part.empty() && std::isdigit(num_part[0])) {
            mult = std::stoll(num_part);
        } else if (num_part.empty()) {
            mult = 1;
        }
        int64_t ns = 0;
        switch (unit) {
            case 'N': ns = 1; break;                    // nanosecond
            case 'U': ns = 1000; break;                 // microsecond
            case 'L': ns = 1000000; break;              // millisecond
            case 'S': ns = 1000000000; break;           // second
            case 'T': ns = 60LL * 1000000000LL; break;  // minute
            case 'H': ns = 3600LL * 1000000000LL; break; // hour
            case 'D': ns = 86400LL * 1000000000LL; break; // day
            case 'W': ns = 7LL * 86400LL * 1000000000LL; break; // week
            case 'M': ns = 30LL * 86400LL * 1000000000LL; break; // month (approx)
            case 'Y': ns = 365LL * 86400LL * 1000000000LL; break; // year (approx)
            default: XTU_THROW(std::invalid_argument, "Unknown frequency unit: " + std::string(1, unit));
        }
        return {freq, mult * ns};
    }
};

// #############################################################################
// Resampler class (returned by dataframe.resample())
// #############################################################################
template <class C = xcoordinate_system>
class Resampler {
public:
    using coordinate_system_type = C;
    using size_type = xtu::size_type;
    using dataframe_type = xdataframe<C>;

private:
    const dataframe_type* m_df;
    std::string m_time_column;
    frequency m_freq;
    std::string m_closed;      // 'left' or 'right'
    std::string m_label;       // 'left' or 'right'
    std::string m_offset;      // optional offset string
    
    std::vector<int64_t> m_time_values;  // nanoseconds from epoch
    std::vector<int64_t> m_bin_edges;
    std::vector<size_type> m_bin_assignments;  // row -> bin index

    // Parse time column to nanoseconds
    int64_t parse_time(size_type row) const {
        auto it = m_df->m_variables.find(m_time_column);
        if (it == m_df->m_variables.end()) {
            XTU_THROW(std::runtime_error, "Time column not found: " + m_time_column);
        }
        // Assume double representing Unix timestamp in seconds
        auto* holder_double = dynamic_cast<const typename dataframe_type::template variable_holder<double>*>(it->second.get());
        if (holder_double) {
            return static_cast<int64_t>(holder_double->var.data()[row] * 1000000000.0);
        }
        // Assume int64 representing nanoseconds
        auto* holder_int64 = dynamic_cast<const typename dataframe_type::template variable_holder<int64_t>*>(it->second.get());
        if (holder_int64) {
            return holder_int64->var.data()[row];
        }
        XTU_THROW(std::runtime_error, "Time column must be numeric (double or int64)");
    }

    void build_bins() {
        size_type n = m_df->nrows();
        m_time_values.resize(n);
        for (size_type i = 0; i < n; ++i) {
            m_time_values[i] = parse_time(i);
        }
        if (n == 0) return;
        int64_t t_min = *std::min_element(m_time_values.begin(), m_time_values.end());
        int64_t t_max = *std::max_element(m_time_values.begin(), m_time_values.end());
        // Align to frequency
        int64_t bin_start = (t_min / m_freq.nanoseconds) * m_freq.nanoseconds;
        if (m_closed == "right") {
            bin_start += m_freq.nanoseconds;
        }
        // Build bin edges
        for (int64_t edge = bin_start; edge <= t_max + m_freq.nanoseconds; edge += m_freq.nanoseconds) {
            m_bin_edges.push_back(edge);
        }
        // Assign each row to a bin
        m_bin_assignments.resize(n);
        for (size_type i = 0; i < n; ++i) {
            int64_t t = m_time_values[i];
            // Find bin index
            size_t bin_idx = 0;
            if (m_closed == "left") {
                for (; bin_idx < m_bin_edges.size() - 1; ++bin_idx) {
                    if (t >= m_bin_edges[bin_idx] && t < m_bin_edges[bin_idx + 1]) break;
                }
            } else { // right
                for (; bin_idx < m_bin_edges.size() - 1; ++bin_idx) {
                    if (t > m_bin_edges[bin_idx] && t <= m_bin_edges[bin_idx + 1]) break;
                }
            }
            if (bin_idx >= m_bin_edges.size() - 1) bin_idx = m_bin_edges.size() - 2;
            m_bin_assignments[i] = bin_idx;
        }
    }

public:
    Resampler(const dataframe_type& df, const std::string& time_col, const std::string& freq_rule,
              const std::string& closed = "left", const std::string& label = "left")
        : m_df(&df), m_time_column(time_col), m_freq(frequency::from_string(freq_rule)),
          m_closed(closed), m_label(label) {
        build_bins();
    }

    // #########################################################################
    // Aggregation methods
    // #########################################################################
    
    /// Apply aggregation function to each bin
    template <class AggFunc>
    dataframe_type agg(const std::string& column, AggFunc agg_func) const {
        size_t n_bins = m_bin_edges.size() - 1;
        if (n_bins == 0) {
            return dataframe_type();
        }
        // Group indices by bin
        std::vector<std::vector<size_type>> bin_groups(n_bins);
        for (size_type i = 0; i < m_bin_assignments.size(); ++i) {
            bin_groups[m_bin_assignments[i]].push_back(i);
        }
        // Build result dataframe
        coordinate_system_type coords;
        coords = coordinate_system_type({"index"});
        xaxis<size_t, size_type> idx_axis;
        for (size_t i = 0; i < n_bins; ++i) idx_axis.push_back(i);
        coords.set_axis(0, idx_axis);
        dataframe_type result(coords);
        
        // Add time index column (bin label)
        xarray_container<double> time_index({n_bins});
        for (size_t i = 0; i < n_bins; ++i) {
            int64_t label_time = (m_label == "left") ? m_bin_edges[i] : m_bin_edges[i+1];
            time_index[i] = static_cast<double>(label_time) / 1000000000.0;
        }
        result.add_column(m_time_column, time_index);
        
        // Aggregate specified column
        std::vector<double> agg_values(n_bins);
        for (size_t b = 0; b < n_bins; ++b) {
            const auto& indices = bin_groups[b];
            std::vector<double> vals;
            for (size_type idx : indices) {
                double val = get_numeric_value(column, idx);
                if (!std::isnan(val)) vals.push_back(val);
            }
            agg_values[b] = agg_func(vals);
        }
        xarray_container<double> agg_data({n_bins});
        for (size_t i = 0; i < n_bins; ++i) agg_data[i] = agg_values[i];
        result.add_column(column, agg_data);
        return result;
    }

    /// Multiple aggregations on multiple columns
    dataframe_type agg(const std::map<std::string, std::string>& agg_dict) const {
        size_t n_bins = m_bin_edges.size() - 1;
        if (n_bins == 0) return dataframe_type();
        // Build bin groups
        std::vector<std::vector<size_type>> bin_groups(n_bins);
        for (size_type i = 0; i < m_bin_assignments.size(); ++i) {
            bin_groups[m_bin_assignments[i]].push_back(i);
        }
        // Build result dataframe
        coordinate_system_type coords;
        coords = coordinate_system_type({"index"});
        xaxis<size_t, size_type> idx_axis;
        for (size_t i = 0; i < n_bins; ++i) idx_axis.push_back(i);
        coords.set_axis(0, idx_axis);
        dataframe_type result(coords);
        
        // Time index
        xarray_container<double> time_index({n_bins});
        for (size_t i = 0; i < n_bins; ++i) {
            int64_t label_time = (m_label == "left") ? m_bin_edges[i] : m_bin_edges[i+1];
            time_index[i] = static_cast<double>(label_time) / 1000000000.0;
        }
        result.add_column(m_time_column, time_index);
        
        for (const auto& kv : agg_dict) {
            const std::string& col = kv.first;
            const std::string& agg_name = kv.second;
            std::vector<double> agg_values(n_bins);
            for (size_t b = 0; b < n_bins; ++b) {
                const auto& indices = bin_groups[b];
                std::vector<double> vals;
                for (size_type idx : indices) {
                    double val = get_numeric_value(col, idx);
                    if (!std::isnan(val)) vals.push_back(val);
                }
                if (agg_name == "sum") {
                    agg_values[b] = std::accumulate(vals.begin(), vals.end(), 0.0);
                } else if (agg_name == "mean") {
                    agg_values[b] = vals.empty() ? NAN : std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<double>(vals.size());
                } else if (agg_name == "min") {
                    agg_values[b] = vals.empty() ? NAN : *std::min_element(vals.begin(), vals.end());
                } else if (agg_name == "max") {
                    agg_values[b] = vals.empty() ? NAN : *std::max_element(vals.begin(), vals.end());
                } else if (agg_name == "count") {
                    agg_values[b] = static_cast<double>(vals.size());
                } else if (agg_name == "first") {
                    agg_values[b] = vals.empty() ? NAN : vals.front();
                } else if (agg_name == "last") {
                    agg_values[b] = vals.empty() ? NAN : vals.back();
                } else {
                    XTU_THROW(std::invalid_argument, "Unsupported aggregation: " + agg_name);
                }
            }
            xarray_container<double> agg_data({n_bins});
            for (size_t i = 0; i < n_bins; ++i) agg_data[i] = agg_values[i];
            result.add_column(col, agg_data);
        }
        return result;
    }

    // Convenience methods
    dataframe_type sum() const { return agg({"__all__"}, "sum"); }
    dataframe_type mean() const { return agg({"__all__"}, "mean"); }
    dataframe_type min() const { return agg({"__all__"}, "min"); }
    dataframe_type max() const { return agg({"__all__"}, "max"); }
    dataframe_type count() const { return agg({"__all__"}, "count"); }
    dataframe_type first() const { return agg({"__all__"}, "first"); }
    dataframe_type last() const { return agg({"__all__"}, "last"); }
    dataframe_type ohlc() const {
        // Open, High, Low, Close for each bin
        // Requires specific column, not implemented here
        XTU_THROW(std::runtime_error, "ohlc not yet implemented");
    }

private:
    double get_numeric_value(const std::string& col, size_type row) const {
        auto it = m_df->m_variables.find(col);
        if (it == m_df->m_variables.end()) return NAN;
        auto* holder_double = dynamic_cast<const typename dataframe_type::template variable_holder<double>*>(it->second.get());
        if (holder_double) return holder_double->var.data()[row];
        auto* holder_int = dynamic_cast<const typename dataframe_type::template variable_holder<int>*>(it->second.get());
        if (holder_int) return static_cast<double>(holder_int->var.data()[row]);
        return NAN;
    }

    dataframe_type agg(const std::vector<std::string>& cols, const std::string& agg_name) const {
        std::map<std::string, std::string> agg_dict;
        for (const auto& col : cols) agg_dict[col] = agg_name;
        return agg(agg_dict);
    }
};

// #############################################################################
// Rolling window class
// #############################################################################
template <class C = xcoordinate_system>
class Rolling {
public:
    using dataframe_type = xdataframe<C>;
    using size_type = xtu::size_type;

private:
    const dataframe_type* m_df;
    size_type m_window;
    size_type m_min_periods;
    bool m_center;
    std::string m_win_type;  // window type (unused, reserved)

public:
    Rolling(const dataframe_type& df, size_type window, size_type min_periods = 0, bool center = false)
        : m_df(&df), m_window(window), m_min_periods(min_periods == 0 ? window : min_periods), m_center(center) {}

    // #########################################################################
    // Aggregation methods
    // #########################################################################
    dataframe_type sum() const { return apply_agg([](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0);
    }); }
    
    dataframe_type mean() const { return apply_agg([](const std::vector<double>& v) {
        if (v.empty()) return NAN;
        return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    }); }
    
    dataframe_type min() const { return apply_agg([](const std::vector<double>& v) {
        if (v.empty()) return NAN;
        return *std::min_element(v.begin(), v.end());
    }); }
    
    dataframe_type max() const { return apply_agg([](const std::vector<double>& v) {
        if (v.empty()) return NAN;
        return *std::max_element(v.begin(), v.end());
    }); }
    
    dataframe_type std() const { return apply_agg([](const std::vector<double>& v) {
        if (v.size() < 2) return NAN;
        double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double sum_sq = 0;
        for (double x : v) { double d = x - m; sum_sq += d * d; }
        return std::sqrt(sum_sq / (v.size() - 1));
    }); }
    
    dataframe_type var() const { return apply_agg([](const std::vector<double>& v) {
        if (v.size() < 2) return NAN;
        double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double sum_sq = 0;
        for (double x : v) { double d = x - m; sum_sq += d * d; }
        return sum_sq / (v.size() - 1);
    }); }
    
    dataframe_type count() const { return apply_agg([](const std::vector<double>& v) {
        return static_cast<double>(v.size());
    }); }

    // Apply custom aggregation
    template <class Func>
    dataframe_type apply(Func&& func) const {
        return apply_agg(std::forward<Func>(func));
    }

private:
    template <class AggFunc>
    dataframe_type apply_agg(AggFunc agg_func) const {
        size_type n = m_df->nrows();
        // Build result dataframe with same index as original
        coordinate_system_type coords = m_df->coords();
        dataframe_type result(coords);
        
        for (const auto& col_name : m_df->column_names()) {
            auto it = m_df->m_variables.find(col_name);
            if (it == m_df->m_variables.end()) continue;
            auto* holder_double = dynamic_cast<const typename dataframe_type::template variable_holder<double>*>(it->second.get());
            if (!holder_double) continue;
            const auto& src_data = holder_double->var.data();
            xarray_container<double> res_data({n});
            for (size_type i = 0; i < n; ++i) {
                size_type start, end;
                if (m_center) {
                    int half = static_cast<int>(m_window) / 2;
                    start = (i >= static_cast<size_type>(half)) ? i - half : 0;
                    end = std::min(i + half + 1, n);
                } else {
                    start = (i >= m_window) ? i - m_window + 1 : 0;
                    end = i + 1;
                }
                size_type count = end - start;
                if (count < m_min_periods) {
                    res_data[i] = NAN;
                } else {
                    std::vector<double> window_vals;
                    window_vals.reserve(count);
                    for (size_type j = start; j < end; ++j) {
                        double val = src_data[j];
                        if (!std::isnan(val)) window_vals.push_back(val);
                    }
                    res_data[i] = agg_func(window_vals);
                }
            }
            result.add_column(col_name, res_data);
        }
        return result;
    }
};

// #############################################################################
// Expanding window class
// #############################################################################
template <class C = xcoordinate_system>
class Expanding {
public:
    using dataframe_type = xdataframe<C>;
    using size_type = xtu::size_type;

private:
    const dataframe_type* m_df;
    size_type m_min_periods;

public:
    explicit Expanding(const dataframe_type& df, size_type min_periods = 1)
        : m_df(&df), m_min_periods(min_periods) {}

    dataframe_type sum() const { return apply_agg([](double acc, double val) { return acc + val; }, 0.0); }
    dataframe_type mean() const {
        return apply_agg_with_count([](double sum, size_t count) { return count > 0 ? sum / static_cast<double>(count) : NAN; });
    }
    dataframe_type min() const { return apply_agg([](double acc, double val) { return std::min(acc, val); }, std::numeric_limits<double>::max()); }
    dataframe_type max() const { return apply_agg([](double acc, double val) { return std::max(acc, val); }, -std::numeric_limits<double>::max()); }
    dataframe_type count() const { return apply_agg_with_count([](double, size_t count) { return static_cast<double>(count); }); }
    dataframe_type std() const {
        XTU_THROW(std::runtime_error, "Expanding std requires online algorithm, not yet implemented");
    }

private:
    template <class AggFunc>
    dataframe_type apply_agg(AggFunc agg_func, double init) const {
        size_type n = m_df->nrows();
        coordinate_system_type coords = m_df->coords();
        dataframe_type result(coords);
        for (const auto& col_name : m_df->column_names()) {
            auto it = m_df->m_variables.find(col_name);
            if (it == m_df->m_variables.end()) continue;
            auto* holder_double = dynamic_cast<const typename dataframe_type::template variable_holder<double>*>(it->second.get());
            if (!holder_double) continue;
            const auto& src_data = holder_double->var.data();
            xarray_container<double> res_data({n});
            double accum = init;
            for (size_type i = 0; i < n; ++i) {
                if (!std::isnan(src_data[i])) {
                    accum = agg_func(accum, src_data[i]);
                }
                res_data[i] = (i + 1 >= m_min_periods) ? accum : NAN;
            }
            result.add_column(col_name, res_data);
        }
        return result;
    }

    template <class AggFunc>
    dataframe_type apply_agg_with_count(AggFunc agg_func) const {
        size_type n = m_df->nrows();
        coordinate_system_type coords = m_df->coords();
        dataframe_type result(coords);
        for (const auto& col_name : m_df->column_names()) {
            auto it = m_df->m_variables.find(col_name);
            if (it == m_df->m_variables.end()) continue;
            auto* holder_double = dynamic_cast<const typename dataframe_type::template variable_holder<double>*>(it->second.get());
            if (!holder_double) continue;
            const auto& src_data = holder_double->var.data();
            xarray_container<double> res_data({n});
            double sum = 0.0;
            size_t count = 0;
            for (size_type i = 0; i < n; ++i) {
                if (!std::isnan(src_data[i])) {
                    sum += src_data[i];
                    ++count;
                }
                res_data[i] = (i + 1 >= m_min_periods) ? agg_func(sum, count) : NAN;
            }
            result.add_column(col_name, res_data);
        }
        return result;
    }
};

// #############################################################################
// Shift / lag operations
// #############################################################################
template <class C>
xdataframe<C> shift(const xdataframe<C>& df, int periods) {
    size_type n = df.nrows();
    coordinate_system_type coords = df.coords();
    xdataframe<C> result(coords);
    for (const auto& col_name : df.column_names()) {
        auto it = df.m_variables.find(col_name);
        if (it == df.m_variables.end()) continue;
        auto* holder_double = dynamic_cast<const typename xdataframe<C>::template variable_holder<double>*>(it->second.get());
        if (!holder_double) continue;
        const auto& src_data = holder_double->var.data();
        xarray_container<double> res_data({n});
        for (size_type i = 0; i < n; ++i) {
            int src_idx = static_cast<int>(i) - periods;
            if (src_idx >= 0 && src_idx < static_cast<int>(n)) {
                res_data[i] = src_data[static_cast<size_type>(src_idx)];
            } else {
                res_data[i] = NAN;
            }
        }
        result.add_column(col_name, res_data);
    }
    return result;
}

// #############################################################################
// Difference operation
// #############################################################################
template <class C>
xdataframe<C> diff(const xdataframe<C>& df, int periods = 1) {
    size_type n = df.nrows();
    coordinate_system_type coords = df.coords();
    xdataframe<C> result(coords);
    for (const auto& col_name : df.column_names()) {
        auto it = df.m_variables.find(col_name);
        if (it == df.m_variables.end()) continue;
        auto* holder_double = dynamic_cast<const typename xdataframe<C>::template variable_holder<double>*>(it->second.get());
        if (!holder_double) continue;
        const auto& src_data = holder_double->var.data();
        xarray_container<double> res_data({n});
        for (size_type i = 0; i < n; ++i) {
            int prev_idx = static_cast<int>(i) - periods;
            if (prev_idx >= 0) {
                res_data[i] = src_data[i] - src_data[static_cast<size_type>(prev_idx)];
            } else {
                res_data[i] = NAN;
            }
        }
        result.add_column(col_name, res_data);
    }
    return result;
}

// #############################################################################
// Percentage change
// #############################################################################
template <class C>
xdataframe<C> pct_change(const xdataframe<C>& df, int periods = 1) {
    size_type n = df.nrows();
    coordinate_system_type coords = df.coords();
    xdataframe<C> result(coords);
    for (const auto& col_name : df.column_names()) {
        auto it = df.m_variables.find(col_name);
        if (it == df.m_variables.end()) continue;
        auto* holder_double = dynamic_cast<const typename xdataframe<C>::template variable_holder<double>*>(it->second.get());
        if (!holder_double) continue;
        const auto& src_data = holder_double->var.data();
        xarray_container<double> res_data({n});
        for (size_type i = 0; i < n; ++i) {
            int prev_idx = static_cast<int>(i) - periods;
            if (prev_idx >= 0 && src_data[static_cast<size_type>(prev_idx)] != 0) {
                res_data[i] = (src_data[i] - src_data[static_cast<size_type>(prev_idx)]) / src_data[static_cast<size_type>(prev_idx)];
            } else {
                res_data[i] = NAN;
            }
        }
        result.add_column(col_name, res_data);
    }
    return result;
}

} // namespace frame

// Bring into main namespace for convenience
using frame::Resampler;
using frame::Rolling;
using frame::Expanding;
using frame::shift;
using frame::diff;
using frame::pct_change;

XTU_NAMESPACE_END

#endif // XTU_FRAME_XRESAMPLE_HPP