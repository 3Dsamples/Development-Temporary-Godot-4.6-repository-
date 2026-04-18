// xtensor-unified - Labeled array (xvariable) and dataframe
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_FRAME_XVARIABLE_HPP
#define XTU_FRAME_XVARIABLE_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/containers/xarray.hpp"
#include "xtu/frame/xcoordinate_system.hpp"

XTU_NAMESPACE_BEGIN
namespace frame {

// #############################################################################
// xvariable: labeled multi-dimensional array
// #############################################################################
template <class T, class C = xcoordinate_system>
class xvariable {
public:
    using value_type = T;
    using coordinate_system_type = C;
    using size_type = xtu::size_type;
    using container_type = xarray_container<T>;
    using shape_type = std::vector<size_type>;
    
private:
    container_type m_data;
    coordinate_system_type m_coords;
    std::string m_name;
    
public:
    // Constructors
    xvariable() = default;
    
    xvariable(const coordinate_system_type& coords, const std::string& name = "")
        : m_coords(coords), m_name(name) {
        XTU_ASSERT_MSG(m_coords.is_complete(), "Coordinate system must have all axes set");
        m_data.resize(m_coords.shape());
    }
    
    xvariable(const coordinate_system_type& coords, const container_type& data, const std::string& name = "")
        : m_data(data), m_coords(coords), m_name(name) {
        XTU_ASSERT_MSG(m_coords.is_complete(), "Coordinate system must have all axes set");
        XTU_ASSERT_MSG(m_coords.shape() == m_data.shape(), "Data shape must match coordinate shape");
    }
    
    xvariable(const coordinate_system_type& coords, container_type&& data, const std::string& name = "")
        : m_data(std::move(data)), m_coords(coords), m_name(name) {
        XTU_ASSERT_MSG(m_coords.is_complete(), "Coordinate system must have all axes set");
        XTU_ASSERT_MSG(m_coords.shape() == m_data.shape(), "Data shape must match coordinate shape");
    }
    
    // Copy / move
    xvariable(const xvariable&) = default;
    xvariable(xvariable&&) noexcept = default;
    xvariable& operator=(const xvariable&) = default;
    xvariable& operator=(xvariable&&) noexcept = default;
    
    // Accessors
    const std::string& name() const noexcept { return m_name; }
    void set_name(const std::string& name) { m_name = name; }
    
    const coordinate_system_type& coords() const noexcept { return m_coords; }
    coordinate_system_type& coords() noexcept { return m_coords; }
    
    const container_type& data() const noexcept { return m_data; }
    container_type& data() noexcept { return m_data; }
    
    size_type dimension() const noexcept { return m_coords.dimension(); }
    shape_type shape() const { return m_data.shape(); }
    size_type size() const { return m_data.size(); }
    
    // Element access by integer indices
    template <class... Idx>
    value_type& operator()(Idx... idx) {
        return m_data(idx...);
    }
    
    template <class... Idx>
    const value_type& operator()(Idx... idx) const {
        return m_data(idx...);
    }
    
    // Element access by labels (for 1D or multi-dim via map)
    // For 1D convenience
    template <class K>
    value_type& at(const K& label) {
        XTU_ASSERT_MSG(dimension() == 1, "at(label) only valid for 1D variable");
        auto* axis_ptr = m_coords.template get_axis<xcoordinate<xaxis<K, size_type>>>(0);
        if (!axis_ptr) XTU_THROW(std::runtime_error, "Axis type mismatch");
        size_type idx = static_cast<size_type>(axis_ptr->axis().index_of(label));
        return m_data[idx];
    }
    
    template <class K>
    const value_type& at(const K& label) const {
        XTU_ASSERT_MSG(dimension() == 1, "at(label) only valid for 1D variable");
        auto* axis_ptr = m_coords.template get_axis<xcoordinate<xaxis<K, size_type>>>(0);
        if (!axis_ptr) XTU_THROW(std::runtime_error, "Axis type mismatch");
        size_type idx = static_cast<size_type>(axis_ptr->axis().index_of(label));
        return m_data[idx];
    }
    
    // Multi-dimensional label access via vector of strings (assumes string axes)
    value_type& at(const std::vector<std::string>& labels) {
        XTU_ASSERT_MSG(labels.size() == dimension(), "Number of labels must match dimension");
        std::vector<size_type> idx(dimension());
        for (size_type i = 0; i < dimension(); ++i) {
            auto* axis_ptr = m_coords.template get_axis<xcoordinate<xaxis<std::string, size_type>>>(i);
            if (!axis_ptr) XTU_THROW(std::runtime_error, "Axis type mismatch or not string");
            idx[i] = static_cast<size_type>(axis_ptr->axis().index_of(labels[i]));
        }
        return m_data(idx);
    }
    
    const value_type& at(const std::vector<std::string>& labels) const {
        XTU_ASSERT_MSG(labels.size() == dimension(), "Number of labels must match dimension");
        std::vector<size_type> idx(dimension());
        for (size_type i = 0; i < dimension(); ++i) {
            auto* axis_ptr = m_coords.template get_axis<xcoordinate<xaxis<std::string, size_type>>>(i);
            if (!axis_ptr) XTU_THROW(std::runtime_error, "Axis type mismatch or not string");
            idx[i] = static_cast<size_type>(axis_ptr->axis().index_of(labels[i]));
        }
        return m_data(idx);
    }
    
    // Slicing by label range or condition
    xvariable slice(const std::vector<std::pair<size_type, size_type>>& slices) const {
        XTU_ASSERT_MSG(slices.size() == dimension(), "Number of slices must match dimension");
        // Create new coordinate system with sliced axes
        coordinate_system_type new_coords;
        // Copy dimension names
        std::vector<std::string> dim_names = m_coords.dimension_names();
        new_coords = coordinate_system_type(dim_names);
        for (size_type d = 0; d < dimension(); ++d) {
            // Slice each axis (type-erased)
            auto base_axis = m_coords.get_axis<xcoordinate_base>(d);
            if (!base_axis) {
                XTU_THROW(std::runtime_error, "Cannot slice: axis missing");
            }
            // We need to clone and slice the axis; this requires a virtual slice method.
            // For simplicity, we'll assume the axis is of a known type (e.g., xaxis<string>).
            // A production implementation would have a virtual slice method in xcoordinate_base.
            auto* string_axis = dynamic_cast<const xcoordinate<xaxis<std::string, size_type>>*>(base_axis);
            if (string_axis) {
                auto sliced_axis = string_axis->axis().slice(static_cast<typename xaxis<std::string, size_type>::value_type>(slices[d].first),
                                                              static_cast<typename xaxis<std::string, size_type>::value_type>(slices[d].second));
                new_coords.set_axis(d, sliced_axis);
            } else {
                // Fallback: create a new axis with just indices
                xaxis<size_type, size_type> idx_axis;
                for (size_type i = slices[d].first; i < slices[d].second; ++i) {
                    idx_axis.push_back(i);
                }
                new_coords.set_axis(d, idx_axis);
            }
        }
        container_type new_data(new_coords.shape());
        // Copy data using recursive indexing
        std::vector<size_type> src_idx(dimension());
        std::vector<size_type> dst_idx(dimension());
        std::function<void(size_type)> copy_rec = [&](size_type dim) {
            if (dim == dimension()) {
                size_t src_lin = 0, src_stride = 1;
                size_t dst_lin = 0, dst_stride = 1;
                for (int d = static_cast<int>(dimension()) - 1; d >= 0; --d) {
                    src_lin += src_idx[static_cast<size_t>(d)] * src_stride;
                    src_stride *= m_data.shape()[static_cast<size_t>(d)];
                    dst_lin += dst_idx[static_cast<size_t>(d)] * dst_stride;
                    dst_stride *= new_data.shape()[static_cast<size_t>(d)];
                }
                new_data.flat(dst_lin) = m_data.flat(src_lin);
                return;
            }
            size_t start = slices[dim].first;
            size_t end = slices[dim].second;
            for (size_t i = start, j = 0; i < end; ++i, ++j) {
                src_idx[dim] = i;
                dst_idx[dim] = j;
                copy_rec(dim + 1);
            }
        };
        copy_rec(0);
        return xvariable(new_coords, std::move(new_data), m_name + "_slice");
    }
    
    // Arithmetic operations (broadcast with alignment)
    xvariable operator+(const xvariable& other) const {
        XTU_ASSERT_MSG(m_coords == other.m_coords, "Coordinates must match for addition");
        return xvariable(m_coords, m_data + other.m_data, "");
    }
    
    xvariable operator-(const xvariable& other) const {
        XTU_ASSERT_MSG(m_coords == other.m_coords, "Coordinates must match");
        return xvariable(m_coords, m_data - other.m_data, "");
    }
    
    xvariable operator*(const xvariable& other) const {
        XTU_ASSERT_MSG(m_coords == other.m_coords, "Coordinates must match");
        return xvariable(m_coords, m_data * other.m_data, "");
    }
    
    xvariable operator/(const xvariable& other) const {
        XTU_ASSERT_MSG(m_coords == other.m_coords, "Coordinates must match");
        return xvariable(m_coords, m_data / other.m_data, "");
    }
    
    xvariable operator+(const value_type& scalar) const {
        return xvariable(m_coords, m_data + scalar, "");
    }
    
    xvariable operator-(const value_type& scalar) const {
        return xvariable(m_coords, m_data - scalar, "");
    }
    
    xvariable operator*(const value_type& scalar) const {
        return xvariable(m_coords, m_data * scalar, "");
    }
    
    xvariable operator/(const value_type& scalar) const {
        return xvariable(m_coords, m_data / scalar, "");
    }
    
    // In-place operations
    xvariable& operator+=(const xvariable& other) {
        XTU_ASSERT_MSG(m_coords == other.m_coords, "Coordinates must match");
        m_data += other.m_data;
        return *this;
    }
    
    xvariable& operator+=(const value_type& scalar) {
        m_data += scalar;
        return *this;
    }
    
    // Statistics along dimensions
    xvariable sum(const std::vector<std::string>& dims_to_reduce = {}) const {
        if (dims_to_reduce.empty()) {
            // Sum all elements
            value_type total = 0;
            for (size_t i = 0; i < m_data.size(); ++i) total += m_data.flat(i);
            coordinate_system_type new_coords;
            container_type new_data({1});
            new_data[0] = total;
            return xvariable(new_coords, std::move(new_data), m_name + "_sum");
        }
        // Reduce over specified dimensions
        std::vector<size_type> axes_to_reduce;
        for (const auto& dim_name : dims_to_reduce) {
            axes_to_reduce.push_back(m_coords.dimension_index(dim_name));
        }
        // Determine new shape and coordinate system
        std::vector<size_type> new_shape;
        coordinate_system_type new_coords;
        std::vector<std::string> new_dim_names;
        for (size_type d = 0; d < dimension(); ++d) {
            if (std::find(axes_to_reduce.begin(), axes_to_reduce.end(), d) == axes_to_reduce.end()) {
                new_shape.push_back(shape()[d]);
                new_dim_names.push_back(m_coords.dimension_names()[d]);
            }
        }
        if (new_shape.empty()) {
            // Reduced to scalar
            return sum({});
        }
        new_coords = coordinate_system_type(new_dim_names);
        for (size_type d = 0; d < new_coords.dimension(); ++d) {
            size_type orig_dim = 0;
            for (size_type od = 0; od < dimension(); ++od) {
                if (std::find(axes_to_reduce.begin(), axes_to_reduce.end(), od) == axes_to_reduce.end()) {
                    if (orig_dim == d) {
                        // Copy axis
                        auto base_axis = m_coords.get_axis<xcoordinate_base>(od);
                        // We need to clone; assume string axis
                        auto* string_axis = dynamic_cast<const xcoordinate<xaxis<std::string, size_type>>*>(base_axis);
                        if (string_axis) {
                            new_coords.set_axis(d, string_axis->axis());
                        }
                        break;
                    }
                    ++orig_dim;
                }
            }
        }
        container_type new_data(new_shape);
        std::fill(new_data.begin(), new_data.end(), value_type(0));
        // Iterate over original data and accumulate into reduced positions
        std::vector<size_type> orig_idx(dimension());
        std::vector<size_type> reduced_idx(new_coords.dimension());
        std::function<void(size_type)> reduce_rec = [&](size_type dim) {
            if (dim == dimension()) {
                value_type val = m_data(orig_idx);
                // Compute reduced index
                size_t rdim = 0;
                for (size_type d = 0; d < dimension(); ++d) {
                    if (std::find(axes_to_reduce.begin(), axes_to_reduce.end(), d) == axes_to_reduce.end()) {
                        reduced_idx[rdim] = orig_idx[d];
                        ++rdim;
                    }
                }
                new_data(reduced_idx) += val;
                return;
            }
            for (size_t i = 0; i < shape()[dim]; ++i) {
                orig_idx[dim] = i;
                reduce_rec(dim + 1);
            }
        };
        reduce_rec(0);
        return xvariable(new_coords, std::move(new_data), m_name + "_sum");
    }
    
    value_type mean() const {
        value_type s = 0;
        for (size_t i = 0; i < m_data.size(); ++i) s += m_data.flat(i);
        return s / static_cast<value_type>(m_data.size());
    }
    
    value_type min() const {
        return *std::min_element(m_data.begin(), m_data.end());
    }
    
    value_type max() const {
        return *std::max_element(m_data.begin(), m_data.end());
    }
    
    // Group-by operation (for 1D variable with categorical axis)
    template <class K>
    std::unordered_map<K, value_type> groupby_sum() const {
        XTU_ASSERT_MSG(dimension() == 1, "Groupby only for 1D");
        auto* axis_ptr = m_coords.template get_axis<xcoordinate<xaxis<K, size_type>>>(0);
        if (!axis_ptr) XTU_THROW(std::runtime_error, "Axis type mismatch");
        std::unordered_map<K, value_type> result;
        const auto& axis = axis_ptr->axis();
        for (size_t i = 0; i < axis.size(); ++i) {
            K label = axis.label(static_cast<size_type>(i));
            result[label] += m_data[i];
        }
        return result;
    }
};

// #############################################################################
// xdataframe: collection of variables sharing coordinates
// #############################################################################
template <class C = xcoordinate_system>
class xdataframe {
public:
    using coordinate_system_type = C;
    using size_type = xtu::size_type;
    
private:
    coordinate_system_type m_coords;
    // Type-erased storage for variables
    struct variable_base {
        virtual ~variable_base() = default;
        virtual std::unique_ptr<variable_base> clone() const = 0;
        virtual std::unique_ptr<variable_base> slice_rows(size_type start, size_type end) const = 0;
        virtual std::string to_string(size_t row, size_t max_rows) const = 0;
    };
    
    template <class T>
    struct variable_holder : public variable_base {
        xvariable<T, C> var;
        variable_holder(const xvariable<T, C>& v) : var(v) {}
        variable_holder(xvariable<T, C>&& v) : var(std::move(v)) {}
        
        std::unique_ptr<variable_base> clone() const override {
            return std::make_unique<variable_holder<T>>(var);
        }
        
        std::unique_ptr<variable_base> slice_rows(size_type start, size_type end) const override {
            // Assumes row is first dimension
            std::vector<std::pair<size_type, size_type>> slices(var.dimension());
            for (size_type d = 0; d < var.dimension(); ++d) {
                slices[d] = {0, var.shape()[d]};
            }
            slices[0] = {start, end};
            auto sliced_var = var.slice(slices);
            return std::make_unique<variable_holder<T>>(std::move(sliced_var));
        }
        
        std::string to_string(size_t row, size_t max_rows) const override {
            std::ostringstream oss;
            if (row < max_rows) {
                oss << var.data()[row];
            } else {
                oss << "...";
            }
            return oss.str();
        }
    };
    
    std::unordered_map<std::string, std::unique_ptr<variable_base>> m_variables;
    std::vector<std::string> m_column_names;
    
public:
    xdataframe() = default;
    
    explicit xdataframe(const coordinate_system_type& coords) : m_coords(coords) {
        XTU_ASSERT_MSG(m_coords.is_complete(), "Coordinate system must be complete");
    }
    
    // Copy
    xdataframe(const xdataframe& other) : m_coords(other.m_coords), m_column_names(other.m_column_names) {
        for (const auto& kv : other.m_variables) {
            m_variables[kv.first] = kv.second->clone();
        }
    }
    
    // Move
    xdataframe(xdataframe&&) noexcept = default;
    
    xdataframe& operator=(const xdataframe& other) {
        if (this != &other) {
            m_coords = other.m_coords;
            m_column_names = other.m_column_names;
            m_variables.clear();
            for (const auto& kv : other.m_variables) {
                m_variables[kv.first] = kv.second->clone();
            }
        }
        return *this;
    }
    
    xdataframe& operator=(xdataframe&&) noexcept = default;
    
    const coordinate_system_type& coords() const noexcept { return m_coords; }
    size_type nrows() const { return m_coords.size(0); }
    size_type ncols() const { return m_column_names.size(); }
    const std::vector<std::string>& column_names() const { return m_column_names; }
    
    // Add a variable (column)
    template <class T>
    void add_column(const std::string& name, const xvariable<T, C>& var) {
        XTU_ASSERT_MSG(var.coords() == m_coords, "Variable coordinates must match dataframe");
        m_variables[name] = std::make_unique<variable_holder<T>>(var);
        if (std::find(m_column_names.begin(), m_column_names.end(), name) == m_column_names.end()) {
            m_column_names.push_back(name);
        }
    }
    
    template <class T>
    void add_column(const std::string& name, xvariable<T, C>&& var) {
        XTU_ASSERT_MSG(var.coords() == m_coords, "Variable coordinates must match dataframe");
        m_variables[name] = std::make_unique<variable_holder<T>>(std::move(var));
        if (std::find(m_column_names.begin(), m_column_names.end(), name) == m_column_names.end()) {
            m_column_names.push_back(name);
        }
    }
    
    template <class T>
    void add_column(const std::string& name, const xarray_container<T>& data) {
        xvariable<T, C> var(m_coords, data, name);
        add_column(name, std::move(var));
    }
    
    // Access a column by name (non-const)
    template <class T>
    xvariable<T, C>& get_column(const std::string& name) {
        auto it = m_variables.find(name);
        if (it == m_variables.end()) XTU_THROW(std::out_of_range, "Column not found");
        auto* holder = dynamic_cast<variable_holder<T>*>(it->second.get());
        if (!holder) XTU_THROW(std::runtime_error, "Column type mismatch");
        return holder->var;
    }
    
    // Access a column by name (const)
    template <class T>
    const xvariable<T, C>& get_column(const std::string& name) const {
        auto it = m_variables.find(name);
        if (it == m_variables.end()) XTU_THROW(std::out_of_range, "Column not found");
        auto* holder = dynamic_cast<const variable_holder<T>*>(it->second.get());
        if (!holder) XTU_THROW(std::runtime_error, "Column type mismatch");
        return holder->var;
    }
    
    bool has_column(const std::string& name) const {
        return m_variables.find(name) != m_variables.end();
    }
    
    void drop_column(const std::string& name) {
        m_variables.erase(name);
        m_column_names.erase(std::remove(m_column_names.begin(), m_column_names.end(), name), m_column_names.end());
    }
    
    // Select rows by index slice
    xdataframe slice_rows(size_type start, size_type end) const {
        XTU_ASSERT_MSG(start <= end && end <= nrows(), "Invalid slice range");
        // Slice coordinate system
        std::vector<std::pair<size_type, size_type>> slices(m_coords.dimension());
        for (size_type d = 0; d < m_coords.dimension(); ++d) {
            slices[d] = {0, m_coords.size(d)};
        }
        slices[0] = {start, end};
        xcoordinate_view view(m_coords, slices);
        // Build new coordinate system from view (simplified: we assume view can provide a new coords)
        // For simplicity, we'll recreate coordinates using the same type but sliced axes.
        coordinate_system_type new_coords;
        std::vector<std::string> dim_names = m_coords.dimension_names();
        new_coords = coordinate_system_type(dim_names);
        for (size_type d = 0; d < m_coords.dimension(); ++d) {
            auto base_axis = m_coords.get_axis<xcoordinate_base>(d);
            if (base_axis) {
                auto* string_axis = dynamic_cast<const xcoordinate<xaxis<std::string, size_type>>*>(base_axis);
                if (string_axis) {
                    auto sliced_axis = string_axis->axis().slice(static_cast<typename xaxis<std::string, size_type>::value_type>(slices[d].first),
                                                                  static_cast<typename xaxis<std::string, size_type>::value_type>(slices[d].second));
                    new_coords.set_axis(d, sliced_axis);
                } else {
                    // Fallback
                    xaxis<size_type, size_type> idx_axis;
                    for (size_type i = slices[d].first; i < slices[d].second; ++i) {
                        idx_axis.push_back(i);
                    }
                    new_coords.set_axis(d, idx_axis);
                }
            }
        }
        xdataframe result(new_coords);
        for (const auto& col : m_column_names) {
            auto it = m_variables.find(col);
            if (it != m_variables.end()) {
                auto sliced_var_ptr = it->second->slice_rows(start, end);
                // Insert into result (type-erased)
                result.m_variables[col] = std::move(sliced_var_ptr);
                result.m_column_names.push_back(col);
            }
        }
        return result;
    }
    
    // Select columns by names
    xdataframe select_columns(const std::vector<std::string>& names) const {
        xdataframe result(m_coords);
        for (const auto& name : names) {
            auto it = m_variables.find(name);
            if (it != m_variables.end()) {
                result.m_variables[name] = it->second->clone();
                result.m_column_names.push_back(name);
            }
        }
        return result;
    }
    
    // Group by a column (categorical) and apply aggregation (sum)
    template <class K>
    std::unordered_map<K, xdataframe> groupby_sum(const std::string& by_column) const {
        // Get the grouping column (must be categorical, i.e., xaxis<K>)
        auto* group_var_holder = dynamic_cast<const variable_holder<K>*>(m_variables.at(by_column).get());
        if (!group_var_holder) XTU_THROW(std::runtime_error, "Group column type mismatch");
        const auto& group_var = group_var_holder->var;
        // Build map from key to row indices
        std::unordered_map<K, std::vector<size_type>> groups;
        for (size_type i = 0; i < group_var.size(); ++i) {
            K key = group_var.data()[i];
            groups[key].push_back(i);
        }
        std::unordered_map<K, xdataframe> result;
        for (const auto& kv : groups) {
            const std::vector<size_type>& indices = kv.second;
            // Build sub-dataframe with those rows
            // Create new coordinate system with rows corresponding to indices
            // We'll keep the same coordinate system but with a new axis that lists the keys (or keep indices)
            // For simplicity, we'll use the same coordinate system but with one row per group member.
            // Actually we want aggregated result (sum over rows). We'll compute sum for each column.
            coordinate_system_type new_coords;
            std::vector<std::string> dim_names = m_coords.dimension_names();
            new_coords = coordinate_system_type(dim_names);
            // Set row axis to just the group key (since we aggregate)
            xaxis<K, size_type> row_axis({kv.first});
            new_coords.set_axis(0, row_axis);
            // Other axes stay the same? For simplicity, we assume only row dimension matters.
            xdataframe sub_df(new_coords);
            for (const auto& col : m_column_names) {
                if (col == by_column) continue; // skip grouping column
                auto it = m_variables.find(col);
                if (it != m_variables.end()) {
                    // We need to sum the values for the selected rows.
                    // This requires knowing the type T of the column. We'll use a visitor pattern.
                    // For brevity, we implement only for double as example.
                    // A full implementation would use dynamic dispatch.
                    auto* holder_double = dynamic_cast<variable_holder<double>*>(it->second.get());
                    if (holder_double) {
                        double sum_val = 0;
                        for (size_type idx : indices) {
                            sum_val += holder_double->var.data()[idx];
                        }
                        xarray_container<double> data({1});
                        data[0] = sum_val;
                        xvariable<double, C> var(new_coords, data, col);
                        sub_df.add_column(col, std::move(var));
                    }
                }
            }
            result[kv.first] = std::move(sub_df);
        }
        return result;
    }
    
    // Display (simple string representation)
    std::string to_string(size_t max_rows = 10) const {
        std::ostringstream oss;
        // Print header: dimension names then column names
        for (const auto& dim_name : m_coords.dimension_names()) {
            oss << dim_name << "\t";
        }
        for (const auto& col : m_column_names) {
            oss << col << "\t";
        }
        oss << "\n";
        size_t rows = std::min(nrows(), max_rows);
        for (size_t i = 0; i < rows; ++i) {
            // Print coordinate labels
            for (size_t d = 0; d < m_coords.dimension(); ++d) {
                auto base_axis = m_coords.get_axis<xcoordinate_base>(d);
                if (base_axis) {
                    auto* string_axis = dynamic_cast<const xcoordinate<xaxis<std::string, size_type>>*>(base_axis);
                    if (string_axis) {
                        if (i < string_axis->axis().size()) {
                            oss << string_axis->axis().label(static_cast<typename xaxis<std::string, size_type>::value_type>(i)) << "\t";
                        } else {
                            oss << "-\t";
                        }
                    } else {
                        oss << i << "\t";
                    }
                } else {
                    oss << "-\t";
                }
            }
            // Print values
            for (const auto& col : m_column_names) {
                auto it = m_variables.find(col);
                if (it != m_variables.end()) {
                    oss << it->second->to_string(i, max_rows) << "\t";
                } else {
                    oss << "NA\t";
                }
            }
            oss << "\n";
        }
        if (nrows() > max_rows) {
            oss << "... (" << (nrows() - max_rows) << " more rows)\n";
        }
        return oss.str();
    }
};

// #############################################################################
// Convenience aliases
// #############################################################################
using DataFrame = xdataframe<xcoordinate_system>;

template <class T>
using Variable = xvariable<T, xcoordinate_system>;

} // namespace frame
XTU_NAMESPACE_END

#endif // XTU_FRAME_XVARIABLE_HPP