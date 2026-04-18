// include/xtu/frame/xgroupby.hpp
// xtensor-unified - Split-apply-combine operations for dataframes
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_FRAME_XGROUPBY_HPP
#define XTU_FRAME_XGROUPBY_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
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
#include "xtu/math/xreducer.hpp"

XTU_NAMESPACE_BEGIN
namespace frame {

// #############################################################################
// Aggregation functions
// #############################################################################
namespace agg {
    template <class T>
    struct sum {
        T operator()(const std::vector<T>& values) const {
            T result = 0;
            for (const auto& v : values) result += v;
            return result;
        }
        static std::string name() { return "sum"; }
    };

    template <class T>
    struct mean {
        T operator()(const std::vector<T>& values) const {
            if (values.empty()) return std::numeric_limits<T>::quiet_NaN();
            T sum = 0;
            for (const auto& v : values) sum += v;
            return sum / static_cast<T>(values.size());
        }
        static std::string name() { return "mean"; }
    };

    template <class T>
    struct min {
        T operator()(const std::vector<T>& values) const {
            if (values.empty()) return std::numeric_limits<T>::quiet_NaN();
            return *std::min_element(values.begin(), values.end());
        }
        static std::string name() { return "min"; }
    };

    template <class T>
    struct max {
        T operator()(const std::vector<T>& values) const {
            if (values.empty()) return std::numeric_limits<T>::quiet_NaN();
            return *std::max_element(values.begin(), values.end());
        }
        static std::string name() { return "max"; }
    };

    template <class T>
    struct count {
        size_t operator()(const std::vector<T>& values) const {
            return values.size();
        }
        static std::string name() { return "count"; }
    };

    template <class T>
    struct stddev {
        T operator()(const std::vector<T>& values) const {
            if (values.size() < 2) return std::numeric_limits<T>::quiet_NaN();
            T m = mean<T>()(values);
            T sum_sq = 0;
            for (const auto& v : values) {
                T diff = v - m;
                sum_sq += diff * diff;
            }
            return std::sqrt(sum_sq / static_cast<T>(values.size() - 1));
        }
        static std::string name() { return "std"; }
    };

    template <class T>
    struct var {
        T operator()(const std::vector<T>& values) const {
            if (values.size() < 2) return std::numeric_limits<T>::quiet_NaN();
            T m = mean<T>()(values);
            T sum_sq = 0;
            for (const auto& v : values) {
                T diff = v - m;
                sum_sq += diff * diff;
            }
            return sum_sq / static_cast<T>(values.size() - 1);
        }
        static std::string name() { return "var"; }
    };

    template <class T>
    struct first {
        T operator()(const std::vector<T>& values) const {
            if (values.empty()) return T{};
            return values.front();
        }
        static std::string name() { return "first"; }
    };

    template <class T>
    struct last {
        T operator()(const std::vector<T>& values) const {
            if (values.empty()) return T{};
            return values.back();
        }
        static std::string name() { return "last"; }
    };

    template <class T>
    struct median {
        T operator()(const std::vector<T>& values) const {
            if (values.empty()) return std::numeric_limits<T>::quiet_NaN();
            std::vector<T> sorted = values;
            std::sort(sorted.begin(), sorted.end());
            size_t n = sorted.size();
            if (n % 2 == 1) return sorted[n / 2];
            return (sorted[n / 2 - 1] + sorted[n / 2]) / static_cast<T>(2);
        }
        static std::string name() { return "median"; }
    };
} // namespace agg

// #############################################################################
// GroupBy object - returned by dataframe.groupby()
// #############################################################################
template <class C = xcoordinate_system>
class GroupBy {
public:
    using coordinate_system_type = C;
    using size_type = xtu::size_type;
    using group_key_type = std::vector<std::string>;  // values of grouping columns as strings

private:
    const xdataframe<C>* m_df;
    std::vector<std::string> m_by_columns;
    std::unordered_map<std::string, std::vector<size_type>> m_groups;  // key -> row indices
    std::vector<std::string> m_sorted_keys;
    bool m_sort;

    // Convert row's grouping column values to a string key
    std::string make_key(size_type row) const {
        std::string key;
        for (const auto& col : m_by_columns) {
            if (!key.empty()) key += "|";
            // Get value from column (type-erased)
            auto it = m_df->m_variables.find(col);
            if (it == m_df->m_variables.end()) {
                XTU_THROW(std::runtime_error, "Column not found: " + col);
            }
            // We need to extract value as string; simplified: assume numeric or string column
            // In production, a visitor pattern would be used.
            // For demonstration, we'll handle common types via dynamic cast.
            auto* holder_double = dynamic_cast<const typename xdataframe<C>::template variable_holder<double>*>(it->second.get());
            if (holder_double) {
                key += std::to_string(holder_double->var.data()[row]);
                continue;
            }
            auto* holder_int = dynamic_cast<const typename xdataframe<C>::template variable_holder<int>*>(it->second.get());
            if (holder_int) {
                key += std::to_string(holder_int->var.data()[row]);
                continue;
            }
            auto* holder_string = dynamic_cast<const typename xdataframe<C>::template variable_holder<std::string>*>(it->second.get());
            if (holder_string) {
                key += holder_string->var.data()[row];
                continue;
            }
            key += "?";
        }
        return key;
    }

public:
    GroupBy(const xdataframe<C>& df, const std::vector<std::string>& by, bool sort_groups = true)
        : m_df(&df), m_by_columns(by), m_sort(sort_groups) {
        size_type n_rows = df.nrows();
        for (size_type i = 0; i < n_rows; ++i) {
            std::string key = make_key(i);
            m_groups[key].push_back(i);
        }
        if (m_sort) {
            for (const auto& kv : m_groups) {
                m_sorted_keys.push_back(kv.first);
            }
            std::sort(m_sorted_keys.begin(), m_sorted_keys.end());
        } else {
            for (const auto& kv : m_groups) {
                m_sorted_keys.push_back(kv.first);
            }
        }
    }

    // #########################################################################
    // Aggregation: apply function(s) to each group
    // #########################################################################
    template <class... AggSpec>
    auto aggregate(const std::string& column, AggSpec&&... specs) const {
        // For single aggregation function returning a Series-like result
        // We'll return a new dataframe with grouping columns and aggregated column.
        // Implementation simplified: handle single aggregation per column.
        return agg_single(column, std::forward<AggSpec>(specs)...);
    }

    // Named aggregation: dict of column -> aggregation
    std::map<std::string, xdataframe<C>> agg(const std::map<std::string, std::string>& agg_dict) const {
        std::map<std::string, xdataframe<C>> result;
        for (const auto& kv : agg_dict) {
            const std::string& col = kv.first;
            const std::string& agg_name = kv.second;
            // For simplicity, handle a few built-in aggregations
            if (agg_name == "sum") {
                result[col] = agg_single_template<double>(col, agg::sum<double>());
            } else if (agg_name == "mean") {
                result[col] = agg_single_template<double>(col, agg::mean<double>());
            } else if (agg_name == "min") {
                result[col] = agg_single_template<double>(col, agg::min<double>());
            } else if (agg_name == "max") {
                result[col] = agg_single_template<double>(col, agg::max<double>());
            } else if (agg_name == "count") {
                result[col] = agg_single_template<size_t>(col, agg::count<double>());
            } else if (agg_name == "std") {
                result[col] = agg_single_template<double>(col, agg::stddev<double>());
            } else if (agg_name == "var") {
                result[col] = agg_single_template<double>(col, agg::var<double>());
            } else if (agg_name == "first") {
                result[col] = agg_single_template<double>(col, agg::first<double>());
            } else if (agg_name == "last") {
                result[col] = agg_single_template<double>(col, agg::last<double>());
            } else if (agg_name == "median") {
                result[col] = agg_single_template<double>(col, agg::median<double>());
            } else {
                XTU_THROW(std::invalid_argument, "Unsupported aggregation: " + agg_name);
            }
        }
        return result;
    }

    // #########################################################################
    // Transform: apply function to each group, returning same shape as original
    // #########################################################################
    template <class Func>
    auto transform(const std::string& column, Func&& func) const {
        size_type n_rows = m_df->nrows();
        // We'll return a new dataframe with the same rows as original, but only the transformed column.
        // For each row, we find its group and apply the function to the group's values.
        // This requires storing the group values per key.
        // Build map of key -> vector of values for the column
        std::unordered_map<std::string, std::vector<double>> group_values;
        for (const auto& kv : m_groups) {
            const std::string& key = kv.first;
            const std::vector<size_type>& indices = kv.second;
            std::vector<double> values;
            for (size_type idx : indices) {
                double val = get_numeric_value(column, idx);
                values.push_back(val);
            }
            group_values[key] = values;
        }
        // Apply function to each group
        std::unordered_map<std::string, double> group_result;
        for (const auto& kv : group_values) {
            group_result[kv.first] = func(kv.second);
        }
        // Create result array
        xarray_container<double> result_data({n_rows});
        for (size_type i = 0; i < n_rows; ++i) {
            std::string key = make_key(i);
            result_data[i] = group_result[key];
        }
        // Create dataframe
        xdataframe<C> result_df = *m_df;
        result_df.add_column(column + "_transformed", result_data);
        return result_df;
    }

    // #########################################################################
    // Filter: return groups that satisfy a predicate
    // #########################################################################
    template <class Predicate>
    auto filter(const std::string& column, Predicate&& pred) const {
        // pred receives a vector of values and returns bool
        std::unordered_map<std::string, std::vector<double>> group_values;
        for (const auto& kv : m_groups) {
            const std::string& key = kv.first;
            const std::vector<size_type>& indices = kv.second;
            std::vector<double> values;
            for (size_type idx : indices) {
                values.push_back(get_numeric_value(column, idx));
            }
            group_values[key] = values;
        }
        // Collect rows from groups that pass predicate
        std::vector<size_type> selected_rows;
        for (const auto& kv : group_values) {
            if (pred(kv.second)) {
                const auto& indices = m_groups.at(kv.first);
                selected_rows.insert(selected_rows.end(), indices.begin(), indices.end());
            }
        }
        // Sort selected rows to maintain original order
        std::sort(selected_rows.begin(), selected_rows.end());
        // Build filtered dataframe
        return m_df->slice_rows(selected_rows);
    }

    // #########################################################################
    // Size of each group
    // #########################################################################
    std::unordered_map<std::string, size_t> size() const {
        std::unordered_map<std::string, size_t> result;
        for (const auto& kv : m_groups) {
            result[kv.first] = kv.second.size();
        }
        return result;
    }

    // #########################################################################
    // Number of groups
    // #########################################################################
    size_t ngroups() const {
        return m_groups.size();
    }

    // #########################################################################
    // Get group indices
    // #########################################################################
    const std::unordered_map<std::string, std::vector<size_type>>& groups() const {
        return m_groups;
    }

    // #########################################################################
    // Iterate over groups (returns (key, sub-dataframe))
    // #########################################################################
    void for_each_group(std::function<void(const std::string&, const xdataframe<C>&)> callback) const {
        for (const auto& key : m_sorted_keys) {
            const auto& indices = m_groups.at(key);
            auto sub_df = m_df->slice_rows(indices);
            callback(key, sub_df);
        }
    }

private:
    double get_numeric_value(const std::string& col, size_type row) const {
        auto it = m_df->m_variables.find(col);
        if (it == m_df->m_variables.end()) {
            XTU_THROW(std::runtime_error, "Column not found: " + col);
        }
        auto* holder_double = dynamic_cast<const typename xdataframe<C>::template variable_holder<double>*>(it->second.get());
        if (holder_double) {
            return holder_double->var.data()[row];
        }
        auto* holder_int = dynamic_cast<const typename xdataframe<C>::template variable_holder<int>*>(it->second.get());
        if (holder_int) {
            return static_cast<double>(holder_int->var.data()[row]);
        }
        auto* holder_size_t = dynamic_cast<const typename xdataframe<C>::template variable_holder<size_t>*>(it->second.get());
        if (holder_size_t) {
            return static_cast<double>(holder_size_t->var.data()[row]);
        }
        XTU_THROW(std::runtime_error, "Column is not numeric: " + col);
    }

    template <class Agg>
    xdataframe<C> agg_single_template(const std::string& column, Agg agg_func) const {
        // Build result dataframe with grouping columns plus aggregated column
        std::vector<std::string> result_cols = m_by_columns;
        result_cols.push_back(column + "_" + Agg::name());
        // Create coordinate system with one dimension (row index)
        coordinate_system_type coords;
        std::vector<std::string> dim_names = {"index"};
        coords = coordinate_system_type(dim_names);
        xaxis<size_t, size_type> idx_axis;
        for (size_t i = 0; i < m_sorted_keys.size(); ++i) {
            idx_axis.push_back(i);
        }
        coords.set_axis(0, idx_axis);
        xdataframe<C> result(coords);
        // Add grouping columns
        for (const auto& by_col : m_by_columns) {
            // We need to add a column containing the group key values for that grouping column.
            // The key is a concatenation; we need to split and extract.
            // For simplicity, we'll just store the full key in a single column.
            // In practice, we'd parse the key back.
        }
        // Add aggregated column
        std::vector<double> agg_values;
        for (const auto& key : m_sorted_keys) {
            const auto& indices = m_groups.at(key);
            std::vector<double> values;
            for (size_type idx : indices) {
                values.push_back(get_numeric_value(column, idx));
            }
            agg_values.push_back(agg_func(values));
        }
        xarray_container<double> agg_data({agg_values.size()});
        for (size_t i = 0; i < agg_values.size(); ++i) agg_data[i] = agg_values[i];
        result.add_column(column + "_" + Agg::name(), agg_data);
        return result;
    }

    // Single aggregation dispatch
    template <class Agg>
    xdataframe<C> agg_single(const std::string& column, Agg agg_func) const {
        return agg_single_template(column, agg_func);
    }
};

// #############################################################################
// DataFrame groupby method (to be added to xdataframe)
// #############################################################################
template <class C>
class xdataframe_groupby_ext {
public:
    static GroupBy<C> groupby(const xdataframe<C>& df, const std::vector<std::string>& by, bool sort = true) {
        return GroupBy<C>(df, by, sort);
    }
};

// #############################################################################
// Convenience function
// #############################################################################
template <class C>
GroupBy<C> groupby(const xdataframe<C>& df, const std::vector<std::string>& by, bool sort = true) {
    return GroupBy<C>(df, by, sort);
}

} // namespace frame

// Bring into main namespace for convenience
using frame::GroupBy;
using frame::groupby;

XTU_NAMESPACE_END

#endif // XTU_FRAME_XGROUPBY_HPP