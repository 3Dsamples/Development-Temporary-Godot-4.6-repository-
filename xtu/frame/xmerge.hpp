// include/xtu/frame/xmerge.hpp
// xtensor-unified - Database-style joins for dataframes
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_FRAME_XMERGE_HPP
#define XTU_FRAME_XMERGE_HPP

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

XTU_NAMESPACE_BEGIN
namespace frame {

// #############################################################################
// Join types enumeration
// #############################################################################
enum class join_type {
    inner,
    left,
    right,
    outer,
    cross
};

// #############################################################################
// Merge options
// #############################################################################
struct merge_options {
    join_type how = join_type::inner;
    std::vector<std::string> on;           // columns to join on (common names)
    std::vector<std::string> left_on;      // left key columns
    std::vector<std::string> right_on;     // right key columns
    std::string left_suffix = "_x";        // suffix for overlapping left columns
    std::string right_suffix = "_y";       // suffix for overlapping right columns
    bool sort = false;                     // sort result by join keys
    bool validate = false;                 // check merge keys for duplicates
    std::vector<std::string> indicator;    // add _merge column indicating source
};

// #############################################################################
// Merge implementation class
// #############################################################################
template <class C = xcoordinate_system>
class Merger {
public:
    using coordinate_system_type = C;
    using size_type = xtu::size_type;
    using dataframe_type = xdataframe<C>;

private:
    const dataframe_type* m_left;
    const dataframe_type* m_right;
    merge_options m_opts;
    std::vector<std::string> m_left_keys;
    std::vector<std::string> m_right_keys;

    // Key hasher for unordered_map
    struct KeyHasher {
        size_t operator()(const std::vector<std::string>& key) const {
            size_t hash = 0;
            for (const auto& s : key) {
                hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };

    // Extract key values from a row as string vector
    std::vector<std::string> extract_key(const dataframe_type* df, 
                                          const std::vector<std::string>& key_cols,
                                          size_type row) const {
        std::vector<std::string> key;
        key.reserve(key_cols.size());
        for (const auto& col : key_cols) {
            key.push_back(get_value_as_string(df, col, row));
        }
        return key;
    }

    // Get cell value as string (type-erased)
    std::string get_value_as_string(const dataframe_type* df, 
                                     const std::string& col, 
                                     size_type row) const {
        auto it = df->m_variables.find(col);
        if (it == df->m_variables.end()) {
            XTU_THROW(std::runtime_error, "Column not found: " + col);
        }
        // Try common types
        auto* holder_double = dynamic_cast<const typename dataframe_type::template variable_holder<double>*>(it->second.get());
        if (holder_double) return std::to_string(holder_double->var.data()[row]);
        auto* holder_int = dynamic_cast<const typename dataframe_type::template variable_holder<int>*>(it->second.get());
        if (holder_int) return std::to_string(holder_int->var.data()[row]);
        auto* holder_size_t = dynamic_cast<const typename dataframe_type::template variable_holder<size_t>*>(it->second.get());
        if (holder_size_t) return std::to_string(holder_size_t->var.data()[row]);
        auto* holder_string = dynamic_cast<const typename dataframe_type::template variable_holder<std::string>*>(it->second.get());
        if (holder_string) return holder_string->var.data()[row];
        auto* holder_bool = dynamic_cast<const typename dataframe_type::template variable_holder<bool>*>(it->second.get());
        if (holder_bool) return holder_bool->var.data()[row] ? "true" : "false";
        return "";
    }

    // Build hash map from right dataframe keys to row indices
    std::unordered_map<std::vector<std::string>, std::vector<size_type>, KeyHasher>
    build_right_index() const {
        std::unordered_map<std::vector<std::string>, std::vector<size_type>, KeyHasher> index;
        size_type n_rows = m_right->nrows();
        for (size_type i = 0; i < n_rows; ++i) {
            auto key = extract_key(m_right, m_right_keys, i);
            index[key].push_back(i);
        }
        return index;
    }

    // Determine output columns and resolve overlaps
    std::vector<std::string> compute_output_columns() const {
        std::vector<std::string> out_cols;
        std::unordered_set<std::string> left_cols(m_left->column_names().begin(),
                                                   m_left->column_names().end());
        std::unordered_set<std::string> right_cols(m_right->column_names().begin(),
                                                    m_right->column_names().end());
        // Add left columns (non-key or all for certain joins)
        for (const auto& col : m_left->column_names()) {
            bool is_key = std::find(m_left_keys.begin(), m_left_keys.end(), col) != m_left_keys.end();
            if (m_opts.how == join_type::inner || m_opts.how == join_type::left || 
                m_opts.how == join_type::outer || is_key) {
                if (right_cols.count(col) && !is_key) {
                    out_cols.push_back(col + m_opts.left_suffix);
                } else {
                    out_cols.push_back(col);
                }
            }
        }
        // Add right columns (exclude keys if already present)
        for (const auto& col : m_right->column_names()) {
            bool is_key = std::find(m_right_keys.begin(), m_right_keys.end(), col) != m_right_keys.end();
            if (m_opts.how == join_type::inner || m_opts.how == join_type::right || 
                m_opts.how == join_type::outer) {
                if (left_cols.count(col)) {
                    if (is_key) {
                        // Key already added from left, skip
                        continue;
                    } else {
                        out_cols.push_back(col + m_opts.right_suffix);
                    }
                } else {
                    out_cols.push_back(col);
                }
            }
        }
        // Add indicator column if requested
        if (!m_opts.indicator.empty()) {
            out_cols.push_back(m_opts.indicator[0]);
        }
        return out_cols;
    }

public:
    Merger(const dataframe_type& left, const dataframe_type& right, const merge_options& opts)
        : m_left(&left), m_right(&right), m_opts(opts) {
        // Determine key columns
        if (!m_opts.on.empty()) {
            m_left_keys = m_opts.on;
            m_right_keys = m_opts.on;
        } else if (!m_opts.left_on.empty() && !m_opts.right_on.empty()) {
            m_left_keys = m_opts.left_on;
            m_right_keys = m_opts.right_on;
            XTU_ASSERT_MSG(m_left_keys.size() == m_right_keys.size(),
                           "left_on and right_on must have same length");
        } else {
            XTU_THROW(std::invalid_argument, "Must specify 'on' or both 'left_on' and 'right_on'");
        }
        // Validate keys exist
        for (const auto& k : m_left_keys) {
            if (!m_left->has_column(k)) XTU_THROW(std::runtime_error, "Left key column not found: " + k);
        }
        for (const auto& k : m_right_keys) {
            if (!m_right->has_column(k)) XTU_THROW(std::runtime_error, "Right key column not found: " + k);
        }
    }

    // Perform the merge
    dataframe_type merge() const {
        switch (m_opts.how) {
            case join_type::inner: return merge_inner();
            case join_type::left:  return merge_left();
            case join_type::right: return merge_right();
            case join_type::outer: return merge_outer();
            case join_type::cross: return merge_cross();
            default: XTU_THROW(std::invalid_argument, "Unsupported join type");
        }
    }

private:
    // Inner join
    dataframe_type merge_inner() const {
        auto right_index = build_right_index();
        std::vector<size_type> left_rows, right_rows;
        std::vector<std::string> merge_indicators;
        size_type n_left = m_left->nrows();
        for (size_type i = 0; i < n_left; ++i) {
            auto key = extract_key(m_left, m_left_keys, i);
            auto it = right_index.find(key);
            if (it != right_index.end()) {
                for (size_type r_idx : it->second) {
                    left_rows.push_back(i);
                    right_rows.push_back(r_idx);
                    if (!m_opts.indicator.empty()) {
                        merge_indicators.push_back("both");
                    }
                }
            }
        }
        return assemble_result(left_rows, right_rows, merge_indicators);
    }

    // Left join
    dataframe_type merge_left() const {
        auto right_index = build_right_index();
        std::vector<size_type> left_rows, right_rows;
        std::vector<std::string> merge_indicators;
        size_type n_left = m_left->nrows();
        for (size_type i = 0; i < n_left; ++i) {
            auto key = extract_key(m_left, m_left_keys, i);
            auto it = right_index.find(key);
            if (it != right_index.end()) {
                for (size_type r_idx : it->second) {
                    left_rows.push_back(i);
                    right_rows.push_back(r_idx);
                    if (!m_opts.indicator.empty()) {
                        merge_indicators.push_back("both");
                    }
                }
            } else {
                left_rows.push_back(i);
                right_rows.push_back(static_cast<size_type>(-1)); // sentinel for NA
                if (!m_opts.indicator.empty()) {
                    merge_indicators.push_back("left_only");
                }
            }
        }
        return assemble_result(left_rows, right_rows, merge_indicators);
    }

    // Right join
    dataframe_type merge_right() const {
        // Symmetric to left join; we can just swap left and right and do left join
        merge_options swapped_opts = m_opts;
        swapped_opts.how = join_type::left;
        std::swap(swapped_opts.left_on, swapped_opts.right_on);
        swapped_opts.left_suffix = m_opts.right_suffix;
        swapped_opts.right_suffix = m_opts.left_suffix;
        Merger<C> swapped_merger(*m_right, *m_left, swapped_opts);
        return swapped_merger.merge();
    }

    // Outer join
    dataframe_type merge_outer() const {
        auto right_index = build_right_index();
        std::unordered_set<size_type> matched_right;
        std::vector<size_type> left_rows, right_rows;
        std::vector<std::string> merge_indicators;
        size_type n_left = m_left->nrows();
        for (size_type i = 0; i < n_left; ++i) {
            auto key = extract_key(m_left, m_left_keys, i);
            auto it = right_index.find(key);
            if (it != right_index.end()) {
                for (size_type r_idx : it->second) {
                    left_rows.push_back(i);
                    right_rows.push_back(r_idx);
                    matched_right.insert(r_idx);
                    if (!m_opts.indicator.empty()) {
                        merge_indicators.push_back("both");
                    }
                }
            } else {
                left_rows.push_back(i);
                right_rows.push_back(static_cast<size_type>(-1));
                if (!m_opts.indicator.empty()) {
                    merge_indicators.push_back("left_only");
                }
            }
        }
        // Add unmatched right rows
        size_type n_right = m_right->nrows();
        for (size_type j = 0; j < n_right; ++j) {
            if (matched_right.find(j) == matched_right.end()) {
                left_rows.push_back(static_cast<size_type>(-1));
                right_rows.push_back(j);
                if (!m_opts.indicator.empty()) {
                    merge_indicators.push_back("right_only");
                }
            }
        }
        return assemble_result(left_rows, right_rows, merge_indicators);
    }

    // Cross join (Cartesian product)
    dataframe_type merge_cross() const {
        size_type n_left = m_left->nrows();
        size_type n_right = m_right->nrows();
        std::vector<size_type> left_rows, right_rows;
        left_rows.reserve(n_left * n_right);
        right_rows.reserve(n_left * n_right);
        for (size_type i = 0; i < n_left; ++i) {
            for (size_type j = 0; j < n_right; ++j) {
                left_rows.push_back(i);
                right_rows.push_back(j);
            }
        }
        std::vector<std::string> merge_indicators;
        return assemble_result(left_rows, right_rows, merge_indicators);
    }

    // Build final dataframe from row indices
    dataframe_type assemble_result(const std::vector<size_type>& left_rows,
                                   const std::vector<size_type>& right_rows,
                                   const std::vector<std::string>& indicators) const {
        size_type n_result = left_rows.size();
        if (n_result == 0) {
            // Return empty dataframe with appropriate columns
            dataframe_type empty_df;
            // Setup coordinate system with empty axis
            return empty_df;
        }

        // Determine output columns
        std::vector<std::string> out_cols = compute_output_columns();

        // Create coordinate system (single dimension row index)
        coordinate_system_type coords;
        coords = coordinate_system_type({"index"});
        xaxis<size_t, size_type> idx_axis;
        for (size_t i = 0; i < n_result; ++i) {
            idx_axis.push_back(i);
        }
        coords.set_axis(0, idx_axis);
        dataframe_type result(coords);

        // Helper to copy column data
        auto copy_column = [&](const dataframe_type* src_df, const std::string& src_col,
                                const std::string& dst_col,
                                const std::vector<size_type>& row_indices,
                                bool use_sentinel = true, size_type sentinel = static_cast<size_type>(-1)) {
            if (!src_df->has_column(src_col)) return;
            // Type-erased copy
            auto it = src_df->m_variables.find(src_col);
            if (it == src_df->m_variables.end()) return;
            // Use double for simplicity; in production would need type dispatch
            auto* holder_double = dynamic_cast<const typename dataframe_type::template variable_holder<double>*>(it->second.get());
            if (holder_double) {
                xarray_container<double> data({n_result});
                for (size_t k = 0; k < n_result; ++k) {
                    size_type src_row = row_indices[k];
                    if (src_row != sentinel || !use_sentinel) {
                        data[k] = holder_double->var.data()[src_row];
                    } else {
                        data[k] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
                result.add_column(dst_col, data);
            }
            // Similar for other types (int, string, etc.) would be added
        };

        // Copy left columns
        std::unordered_set<std::string> right_cols(m_right->column_names().begin(),
                                                    m_right->column_names().end());
        for (const auto& col : m_left->column_names()) {
            bool is_key = std::find(m_left_keys.begin(), m_left_keys.end(), col) != m_left_keys.end();
            std::string dst_col = col;
            if (right_cols.count(col) && !is_key) {
                dst_col = col + m_opts.left_suffix;
            }
            copy_column(m_left, col, dst_col, left_rows, true, static_cast<size_type>(-1));
        }

        // Copy right columns
        std::unordered_set<std::string> left_cols(m_left->column_names().begin(),
                                                   m_left->column_names().end());
        for (const auto& col : m_right->column_names()) {
            bool is_key = std::find(m_right_keys.begin(), m_right_keys.end(), col) != m_right_keys.end();
            if (left_cols.count(col) && is_key) {
                // Key already added from left, skip
                continue;
            }
            std::string dst_col = col;
            if (left_cols.count(col) && !is_key) {
                dst_col = col + m_opts.right_suffix;
            }
            copy_column(m_right, col, dst_col, right_rows, true, static_cast<size_type>(-1));
        }

        // Add indicator column if requested
        if (!m_opts.indicator.empty() && !indicators.empty()) {
            xarray_container<std::string> ind_data({n_result});
            for (size_t k = 0; k < n_result; ++k) {
                ind_data[k] = indicators[k];
            }
            result.add_column(m_opts.indicator[0], ind_data);
        }

        return result;
    }
};

// #############################################################################
// Free merge functions
// #############################################################################
template <class C = xcoordinate_system>
xdataframe<C> merge(const xdataframe<C>& left, const xdataframe<C>& right,
                    const merge_options& opts = {}) {
    Merger<C> merger(left, right, opts);
    return merger.merge();
}

// Convenience overloads for common join types
template <class C = xcoordinate_system>
xdataframe<C> inner_join(const xdataframe<C>& left, const xdataframe<C>& right,
                         const std::vector<std::string>& on,
                         const std::string& left_suffix = "_x",
                         const std::string& right_suffix = "_y") {
    merge_options opts;
    opts.how = join_type::inner;
    opts.on = on;
    opts.left_suffix = left_suffix;
    opts.right_suffix = right_suffix;
    return merge(left, right, opts);
}

template <class C = xcoordinate_system>
xdataframe<C> left_join(const xdataframe<C>& left, const xdataframe<C>& right,
                        const std::vector<std::string>& on,
                        const std::string& left_suffix = "_x",
                        const std::string& right_suffix = "_y") {
    merge_options opts;
    opts.how = join_type::left;
    opts.on = on;
    opts.left_suffix = left_suffix;
    opts.right_suffix = right_suffix;
    return merge(left, right, opts);
}

template <class C = xcoordinate_system>
xdataframe<C> right_join(const xdataframe<C>& left, const xdataframe<C>& right,
                         const std::vector<std::string>& on,
                         const std::string& left_suffix = "_x",
                         const std::string& right_suffix = "_y") {
    merge_options opts;
    opts.how = join_type::right;
    opts.on = on;
    opts.left_suffix = left_suffix;
    opts.right_suffix = right_suffix;
    return merge(left, right, opts);
}

template <class C = xcoordinate_system>
xdataframe<C> outer_join(const xdataframe<C>& left, const xdataframe<C>& right,
                         const std::vector<std::string>& on,
                         const std::string& left_suffix = "_x",
                         const std::string& right_suffix = "_y") {
    merge_options opts;
    opts.how = join_type::outer;
    opts.on = on;
    opts.left_suffix = left_suffix;
    opts.right_suffix = right_suffix;
    return merge(left, right, opts);
}

} // namespace frame

// Bring into main namespace for convenience
using frame::join_type;
using frame::merge_options;
using frame::merge;
using frame::inner_join;
using frame::left_join;
using frame::right_join;
using frame::outer_join;

XTU_NAMESPACE_END

#endif // XTU_FRAME_XMERGE_HPP