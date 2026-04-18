// xtensor-unified - Coordinate system for labeled arrays and dataframes
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_FRAME_XCOORDINATE_SYSTEM_HPP
#define XTU_FRAME_XCOORDINATE_SYSTEM_HPP

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

XTU_NAMESPACE_BEGIN
namespace frame {

// #############################################################################
// Forward declarations
// #############################################################################
class xcoordinate_base;
template <class D> class xcoordinate;
template <class K, class V> class xaxis;
class xdimension;
class xcoordinate_system;
class xcoordinate_view;

// #############################################################################
// Dimension label and metadata
// #############################################################################
class xdimension {
public:
    using size_type = xtu::size_type;
    
private:
    std::string m_name;
    size_type m_size;
    
public:
    xdimension() : m_name(""), m_size(0) {}
    xdimension(const std::string& name, size_type size) : m_name(name), m_size(size) {}
    xdimension(const char* name, size_type size) : m_name(name), m_size(size) {}
    
    const std::string& name() const noexcept { return m_name; }
    size_type size() const noexcept { return m_size; }
    
    void set_name(const std::string& name) { m_name = name; }
    void set_size(size_type size) { m_size = size; }
    
    bool operator==(const xdimension& other) const {
        return m_name == other.m_name && m_size == other.m_size;
    }
    bool operator!=(const xdimension& other) const { return !(*this == other); }
};

// #############################################################################
// Axis: mapping from label to integer index
// #############################################################################
template <class K, class V = xtu::index_type>
class xaxis {
public:
    using key_type = K;
    using value_type = V;
    using size_type = xtu::size_type;
    using map_type = std::unordered_map<key_type, value_type>;
    using vector_type = std::vector<key_type>;
    
private:
    vector_type m_labels;      // labels in order
    map_type m_label_to_index; // fast lookup
    std::string m_name;
    
public:
    // Constructors
    xaxis() = default;
    
    explicit xaxis(const std::string& name) : m_name(name) {}
    
    xaxis(const std::vector<key_type>& labels, const std::string& name = "")
        : m_labels(labels), m_name(name) {
        rebuild_index();
    }
    
    xaxis(std::initializer_list<key_type> labels, const std::string& name = "")
        : m_labels(labels), m_name(name) {
        rebuild_index();
    }
    
    template <class Iterator>
    xaxis(Iterator first, Iterator last, const std::string& name = "")
        : m_labels(first, last), m_name(name) {
        rebuild_index();
    }
    
    // Copy / move
    xaxis(const xaxis&) = default;
    xaxis(xaxis&&) noexcept = default;
    xaxis& operator=(const xaxis&) = default;
    xaxis& operator=(xaxis&&) noexcept = default;
    
    // Rebuild index map
    void rebuild_index() {
        m_label_to_index.clear();
        for (value_type i = 0; i < static_cast<value_type>(m_labels.size()); ++i) {
            m_label_to_index[m_labels[static_cast<size_type>(i)]] = i;
        }
    }
    
    // Capacity
    size_type size() const noexcept { return m_labels.size(); }
    bool empty() const noexcept { return m_labels.empty(); }
    const std::string& name() const noexcept { return m_name; }
    void set_name(const std::string& name) { m_name = name; }
    
    // Label access
    const vector_type& labels() const noexcept { return m_labels; }
    vector_type& labels() noexcept { return m_labels; }
    
    const key_type& label(value_type idx) const {
        XTU_ASSERT_MSG(idx >= 0 && static_cast<size_type>(idx) < m_labels.size(), "Index out of range");
        return m_labels[static_cast<size_type>(idx)];
    }
    
    // Index lookup
    value_type index_of(const key_type& label) const {
        auto it = m_label_to_index.find(label);
        if (it == m_label_to_index.end()) {
            XTU_THROW(std::out_of_range, "Label not found in axis");
        }
        return it->second;
    }
    
    bool contains(const key_type& label) const {
        return m_label_to_index.find(label) != m_label_to_index.end();
    }
    
    // Add label at end
    void push_back(const key_type& label) {
        m_label_to_index[label] = static_cast<value_type>(m_labels.size());
        m_labels.push_back(label);
    }
    
    // Remove label
    void erase(const key_type& label) {
        auto it = m_label_to_index.find(label);
        if (it != m_label_to_index.end()) {
            m_label_to_index.erase(it);
            // Remove from vector and update indices
            m_labels.erase(std::remove(m_labels.begin(), m_labels.end(), label), m_labels.end());
            rebuild_index();
        }
    }
    
    // Subset axis with given labels (preserving order)
    xaxis subset(const std::vector<key_type>& labels_subset) const {
        xaxis result(m_name);
        for (const auto& lbl : labels_subset) {
            if (contains(lbl)) {
                result.push_back(lbl);
            }
        }
        return result;
    }
    
    // Subset by index range
    xaxis slice(value_type start, value_type end) const {
        XTU_ASSERT_MSG(start >= 0 && end <= static_cast<value_type>(size()) && start <= end, "Invalid slice range");
        xaxis result(m_name);
        for (value_type i = start; i < end; ++i) {
            result.push_back(m_labels[static_cast<size_type>(i)]);
        }
        return result;
    }
    
    // Equality
    bool operator==(const xaxis& other) const {
        return m_name == other.m_name && m_labels == other.m_labels;
    }
    bool operator!=(const xaxis& other) const { return !(*this == other); }
};

// #############################################################################
// Coordinate system: collection of axes (dimensions)
// #############################################################################
class xcoordinate_system {
public:
    using size_type = xtu::size_type;
    using axis_ptr = std::shared_ptr<xcoordinate_base>;
    
private:
    std::vector<axis_ptr> m_axes;
    std::vector<std::string> m_dimension_names;
    
public:
    // Constructors
    xcoordinate_system() = default;
    
    explicit xcoordinate_system(const std::vector<std::string>& dim_names) 
        : m_dimension_names(dim_names), m_axes(dim_names.size()) {}
    
    xcoordinate_system(std::initializer_list<std::string> dim_names)
        : m_dimension_names(dim_names), m_axes(dim_names.size()) {}
    
    // Copy / move
    xcoordinate_system(const xcoordinate_system&) = default;
    xcoordinate_system(xcoordinate_system&&) noexcept = default;
    xcoordinate_system& operator=(const xcoordinate_system&) = default;
    xcoordinate_system& operator=(xcoordinate_system&&) noexcept = default;
    
    // Dimension count
    size_type dimension() const noexcept { return m_axes.size(); }
    const std::vector<std::string>& dimension_names() const noexcept { return m_dimension_names; }
    
    // Set axis for a dimension
    template <class K, class V>
    void set_axis(size_type dim, const xaxis<K, V>& axis) {
        XTU_ASSERT_MSG(dim < dimension(), "Dimension index out of range");
        m_axes[dim] = std::make_shared<xcoordinate<xaxis<K, V>>>(axis);
    }
    
    // Get axis by index (returns typed axis or base)
    template <class AxisType = xcoordinate_base>
    const AxisType* get_axis(size_type dim) const {
        XTU_ASSERT_MSG(dim < dimension(), "Dimension index out of range");
        if (auto* ptr = dynamic_cast<const AxisType*>(m_axes[dim].get())) {
            return ptr;
        }
        return nullptr;
    }
    
    template <class AxisType = xcoordinate_base>
    AxisType* get_axis(size_type dim) {
        XTU_ASSERT_MSG(dim < dimension(), "Dimension index out of range");
        if (auto* ptr = dynamic_cast<AxisType*>(m_axes[dim].get())) {
            return ptr;
        }
        return nullptr;
    }
    
    // Get dimension index by name
    size_type dimension_index(const std::string& name) const {
        auto it = std::find(m_dimension_names.begin(), m_dimension_names.end(), name);
        if (it == m_dimension_names.end()) {
            XTU_THROW(std::out_of_range, "Dimension name not found");
        }
        return static_cast<size_type>(std::distance(m_dimension_names.begin(), it));
    }
    
    // Shape (sizes of each dimension)
    std::vector<size_type> shape() const {
        std::vector<size_type> shp(dimension());
        for (size_type i = 0; i < dimension(); ++i) {
            if (m_axes[i]) {
                shp[i] = m_axes[i]->size();
            } else {
                shp[i] = 0;
            }
        }
        return shp;
    }
    
    // Size of specific dimension
    size_type size(size_type dim) const {
        XTU_ASSERT_MSG(dim < dimension() && m_axes[dim], "Axis not set or dimension invalid");
        return m_axes[dim]->size();
    }
    
    // Check if all axes are set
    bool is_complete() const {
        for (const auto& ax : m_axes) if (!ax) return false;
        return true;
    }
    
    // Align two coordinate systems (return mapping of indices)
    static std::pair<std::vector<size_type>, std::vector<size_type>> 
    align(const xcoordinate_system& left, const xcoordinate_system& right, 
          const std::vector<std::string>& dims) {
        std::vector<size_type> left_idx, right_idx;
        for (const auto& dim : dims) {
            left_idx.push_back(left.dimension_index(dim));
            right_idx.push_back(right.dimension_index(dim));
        }
        return {left_idx, right_idx};
    }
    
    // Equality
    bool operator==(const xcoordinate_system& other) const {
        if (dimension() != other.dimension()) return false;
        for (size_type i = 0; i < dimension(); ++i) {
            if (!m_axes[i] || !other.m_axes[i]) {
                if (m_axes[i] != other.m_axes[i]) return false;
                continue;
            }
            if (!m_axes[i]->equals(*other.m_axes[i])) return false;
        }
        return m_dimension_names == other.m_dimension_names;
    }
    
    bool operator!=(const xcoordinate_system& other) const { return !(*this == other); }
};

// #############################################################################
// Base class for type-erased axis
// #############################################################################
class xcoordinate_base {
public:
    virtual ~xcoordinate_base() = default;
    virtual size_t size() const = 0;
    virtual bool equals(const xcoordinate_base& other) const = 0;
    virtual std::unique_ptr<xcoordinate_base> clone() const = 0;
    virtual std::string to_string() const = 0;
};

// #############################################################################
// Type-erased wrapper for xaxis
// #############################################################################
template <class AxisType>
class xcoordinate : public xcoordinate_base {
private:
    AxisType m_axis;
    
public:
    using axis_type = AxisType;
    using size_type = xtu::size_type;
    
    xcoordinate() = default;
    explicit xcoordinate(const AxisType& axis) : m_axis(axis) {}
    explicit xcoordinate(AxisType&& axis) : m_axis(std::move(axis)) {}
    
    const AxisType& axis() const { return m_axis; }
    AxisType& axis() { return m_axis; }
    
    size_t size() const override { return m_axis.size(); }
    
    bool equals(const xcoordinate_base& other) const override {
        auto* other_typed = dynamic_cast<const xcoordinate<AxisType>*>(&other);
        if (!other_typed) return false;
        return m_axis == other_typed->m_axis;
    }
    
    std::unique_ptr<xcoordinate_base> clone() const override {
        return std::make_unique<xcoordinate<AxisType>>(m_axis);
    }
    
    std::string to_string() const override {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < m_axis.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << m_axis.label(static_cast<typename AxisType::value_type>(i));
        }
        oss << "]";
        return oss.str();
    }
};

// #############################################################################
// Coordinate view: slice of a coordinate system
// #############################################################################
class xcoordinate_view {
private:
    const xcoordinate_system* m_system;
    std::vector<std::pair<size_t, size_t>> m_slices; // (start, end) per dimension
    
public:
    xcoordinate_view() : m_system(nullptr) {}
    
    xcoordinate_view(const xcoordinate_system& sys,
                     const std::vector<std::pair<size_t, size_t>>& slices)
        : m_system(&sys), m_slices(slices) {
        XTU_ASSERT_MSG(m_slices.size() == sys.dimension(), "Slice vector must match dimension");
    }
    
    // Full view (no slicing)
    explicit xcoordinate_view(const xcoordinate_system& sys)
        : m_system(&sys) {
        m_slices.resize(sys.dimension());
        for (size_t i = 0; i < sys.dimension(); ++i) {
            m_slices[i] = {0, sys.size(i)};
        }
    }
    
    size_t dimension() const { return m_system ? m_system->dimension() : 0; }
    
    std::pair<size_t, size_t> slice(size_t dim) const {
        XTU_ASSERT_MSG(dim < dimension(), "Dimension out of range");
        return m_slices[dim];
    }
    
    // Convert global index to view index (if in view)
    bool contains(size_t dim, size_t global_idx) const {
        auto sl = slice(dim);
        return global_idx >= sl.first && global_idx < sl.second;
    }
    
    // Map view index to global index
    size_t to_global(size_t dim, size_t view_idx) const {
        auto sl = slice(dim);
        return sl.first + view_idx;
    }
    
    // Get coordinate system shape of the view
    std::vector<size_t> shape() const {
        std::vector<size_t> shp(dimension());
        for (size_t i = 0; i < dimension(); ++i) {
            shp[i] = m_slices[i].second - m_slices[i].first;
        }
        return shp;
    }
};

} // namespace frame
XTU_NAMESPACE_END

#endif // XTU_FRAME_XCOORDINATE_SYSTEM_HPP