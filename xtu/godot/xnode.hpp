// include/xtu/godot/xnode.hpp
// xtensor-unified - Scene node system for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XNODE_HPP
#define XTU_GODOT_XNODE_HPP

// godot/xnode.hpp

#ifndef XTENSOR_XNODE_HPP
#define XTENSOR_XNODE_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xintersection.hpp"
#include "xvariant.hpp"
#include "xclassdb.hpp"
#include "xresource.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/node.hpp>
    #include <godot_cpp/classes/node3d.hpp>
    #include <godot_cpp/classes/node2d.hpp>
    #include <godot_cpp/classes/scene_tree.hpp>
    #include <godot_cpp/classes/window.hpp>
    #include <godot_cpp/classes/viewport.hpp>
    #include <godot_cpp/classes/camera3d.hpp>
    #include <godot_cpp/classes/mesh_instance3d.hpp>
    #include <godot_cpp/classes/mesh.hpp>
    #include <godot_cpp/classes/array_mesh.hpp>
    #include <godot_cpp/classes/immediate_mesh.hpp>
    #include <godot_cpp/classes/standard_material3d.hpp>
    #include <godot_cpp/classes/transform3d.hpp>
    #include <godot_cpp/classes/basis.hpp>
    #include <godot_cpp/classes/quaternion.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/transform3d.hpp>
    #include <godot_cpp/variant/vector3.hpp>
    #include <godot_cpp/variant/color.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // XTensorNode - Base node for tensor operations in Godot scenes
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XTensorNode : public godot::Node
            {
                GDCLASS(XTensorNode, godot::Node)

            private:
                godot::Ref<XTensorResource> m_tensor_resource;
                godot::String m_tensor_path;
                bool m_auto_load = false;
                bool m_dirty = true;
                std::vector<std::string> m_operation_stack;
                godot::Dictionary m_node_metadata;

            protected:
                static void _bind_methods()
                {
                    // Core tensor operations
                    godot::ClassDB::bind_method(godot::D_METHOD("set_tensor_resource", "resource"), &XTensorNode::set_tensor_resource);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_tensor_resource"), &XTensorNode::get_tensor_resource);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_tensor_path", "path"), &XTensorNode::set_tensor_path);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_tensor_path"), &XTensorNode::get_tensor_path);
                    
                    // Data access
                    godot::ClassDB::bind_method(godot::D_METHOD("get_data"), &XTensorNode::get_data);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_data", "data"), &XTensorNode::set_data);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_shape"), &XTensorNode::get_shape);
                    godot::ClassDB::bind_method(godot::D_METHOD("reshape", "shape"), &XTensorNode::reshape);
                    godot::ClassDB::bind_method(godot::D_METHOD("size"), &XTensorNode::size);
                    godot::ClassDB::bind_method(godot::D_METHOD("dimension"), &XTensorNode::dimension);
                    
                    // Math operations
                    godot::ClassDB::bind_method(godot::D_METHOD("add", "other"), &XTensorNode::add);
                    godot::ClassDB::bind_method(godot::D_METHOD("subtract", "other"), &XTensorNode::subtract);
                    godot::ClassDB::bind_method(godot::D_METHOD("multiply", "other"), &XTensorNode::multiply);
                    godot::ClassDB::bind_method(godot::D_METHOD("divide", "other"), &XTensorNode::divide);
                    godot::ClassDB::bind_method(godot::D_METHOD("matmul", "other"), &XTensorNode::matmul);
                    godot::ClassDB::bind_method(godot::D_METHOD("transpose"), &XTensorNode::transpose);
                    godot::ClassDB::bind_method(godot::D_METHOD("sum", "axis"), &XTensorNode::sum, godot::DEFVAL(-1));
                    godot::ClassDB::bind_method(godot::D_METHOD("mean", "axis"), &XTensorNode::mean, godot::DEFVAL(-1));
                    godot::ClassDB::bind_method(godot::D_METHOD("max", "axis"), &XTensorNode::max, godot::DEFVAL(-1));
                    godot::ClassDB::bind_method(godot::D_METHOD("min", "axis"), &XTensorNode::min, godot::DEFVAL(-1));
                    godot::ClassDB::bind_method(godot::D_METHOD("norm", "ord"), &XTensorNode::norm, godot::DEFVAL("l2"));
                    
                    // Indexing and slicing
                    godot::ClassDB::bind_method(godot::D_METHOD("slice", "start_indices", "end_indices"), &XTensorNode::slice);
                    godot::ClassDB::bind_method(godot::D_METHOD("index", "indices"), &XTensorNode::index);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_element", "indices"), &XTensorNode::get_element);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_element", "indices", "value"), &XTensorNode::set_element);
                    
                    // Linear algebra
                    godot::ClassDB::bind_method(godot::D_METHOD("solve", "b"), &XTensorNode::solve);
                    godot::ClassDB::bind_method(godot::D_METHOD("inverse"), &XTensorNode::inverse);
                    godot::ClassDB::bind_method(godot::D_METHOD("determinant"), &XTensorNode::determinant);
                    godot::ClassDB::bind_method(godot::D_METHOD("eigenvalues"), &XTensorNode::eigenvalues);
                    godot::ClassDB::bind_method(godot::D_METHOD("svd"), &XTensorNode::svd);
                    
                    // Statistics
                    godot::ClassDB::bind_method(godot::D_METHOD("stddev", "axis"), &XTensorNode::stddev, godot::DEFVAL(-1));
                    godot::ClassDB::bind_method(godot::D_METHOD("variance", "axis"), &XTensorNode::variance, godot::DEFVAL(-1));
                    godot::ClassDB::bind_method(godot::D_METHOD("quantile", "q", "axis"), &XTensorNode::quantile, godot::DEFVAL(0));
                    godot::ClassDB::bind_method(godot::D_METHOD("median", "axis"), &XTensorNode::median, godot::DEFVAL(-1));
                    
                    // I/O
                    godot::ClassDB::bind_method(godot::D_METHOD("load_tensor", "path"), &XTensorNode::load_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("save_tensor", "path"), &XTensorNode::save_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("to_json"), &XTensorNode::to_json);
                    godot::ClassDB::bind_method(godot::D_METHOD("from_json", "json"), &XTensorNode::from_json);
                    
                    // Conversion
                    godot::ClassDB::bind_method(godot::D_METHOD("to_vector3_array"), &XTensorNode::to_vector3_array);
                    godot::ClassDB::bind_method(godot::D_METHOD("to_vector2_array"), &XTensorNode::to_vector2_array);
                    godot::ClassDB::bind_method(godot::D_METHOD("to_color_array"), &XTensorNode::to_color_array);
                    godot::ClassDB::bind_method(godot::D_METHOD("to_transform_array"), &XTensorNode::to_transform_array);
                    godot::ClassDB::bind_method(godot::D_METHOD("from_vector3_array", "array"), &XTensorNode::from_vector3_array);
                    godot::ClassDB::bind_method(godot::D_METHOD("from_vector2_array", "array"), &XTensorNode::from_vector2_array);
                    
                    // Utility
                    godot::ClassDB::bind_method(godot::D_METHOD("copy"), &XTensorNode::copy);
                    godot::ClassDB::bind_method(godot::D_METHOD("is_valid"), &XTensorNode::is_valid);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear"), &XTensorNode::clear);
                    
                    // Static factory
                    godot::ClassDB::bind_static_method("XTensorNode", godot::D_METHOD("create_zeros", "shape"), &XTensorNode::create_zeros);
                    godot::ClassDB::bind_static_method("XTensorNode", godot::D_METHOD("create_ones", "shape"), &XTensorNode::create_ones);
                    godot::ClassDB::bind_static_method("XTensorNode", godot::D_METHOD("create_random", "shape"), &XTensorNode::create_random);
                    godot::ClassDB::bind_static_method("XTensorNode", godot::D_METHOD("create_identity", "n"), &XTensorNode::create_identity);
                    godot::ClassDB::bind_static_method("XTensorNode", godot::D_METHOD("create_linspace", "start", "stop", "num"), &XTensorNode::create_linspace);
                    
                    // Properties
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "tensor_resource", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_tensor_resource", "get_tensor_resource");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "tensor_path", godot::PROPERTY_HINT_FILE, "*.npy,*.npz,*.json"), "set_tensor_path", "get_tensor_path");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "auto_load"), "set_auto_load", "get_auto_load");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::DICTIONARY, "node_metadata"), "set_node_metadata", "get_node_metadata");
                    
                    // Signals
                    ADD_SIGNAL(godot::MethodInfo("tensor_loaded"));
                    ADD_SIGNAL(godot::MethodInfo("tensor_changed"));
                    ADD_SIGNAL(godot::MethodInfo("operation_performed", godot::PropertyInfo(godot::Variant::STRING, "op_name")));
                }

            public:
                XTensorNode()
                {
                    if (!m_tensor_resource.is_valid())
                    {
                        m_tensor_resource.instantiate();
                    }
                }

                void _ready() override
                {
                    if (m_auto_load && !m_tensor_path.is_empty())
                    {
                        load_tensor(m_tensor_path);
                    }
                }

                // Tensor resource access
                void set_tensor_resource(const godot::Ref<XTensorResource>& resource)
                {
                    if (resource.is_valid())
                    {
                        m_tensor_resource = resource;
                        m_dirty = false;
                        emit_signal("tensor_changed");
                    }
                }

                godot::Ref<XTensorResource> get_tensor_resource() const
                {
                    return m_tensor_resource;
                }

                void set_tensor_path(const godot::String& path)
                {
                    m_tensor_path = path;
                }

                godot::String get_tensor_path() const
                {
                    return m_tensor_path;
                }

                void set_auto_load(bool enable)
                {
                    m_auto_load = enable;
                }

                bool get_auto_load() const
                {
                    return m_auto_load;
                }

                // Data access
                godot::Variant get_data() const
                {
                    if (m_tensor_resource.is_valid())
                        return m_tensor_resource->get_data();
                    return godot::Variant();
                }

                void set_data(const godot::Variant& data)
                {
                    ensure_resource();
                    m_tensor_resource->set_data(data);
                    m_dirty = false;
                    emit_signal("tensor_changed");
                }

                godot::PackedInt64Array get_shape() const
                {
                    if (m_tensor_resource.is_valid())
                        return m_tensor_resource->get_shape();
                    return godot::PackedInt64Array();
                }

                void reshape(const godot::PackedInt64Array& shape)
                {
                    ensure_resource();
                    m_tensor_resource->reshape(shape);
                    emit_signal("operation_performed", "reshape");
                    emit_signal("tensor_changed");
                }

                int64_t size() const
                {
                    if (m_tensor_resource.is_valid())
                        return m_tensor_resource->get_size();
                    return 0;
                }

                int64_t dimension() const
                {
                    if (m_tensor_resource.is_valid())
                        return m_tensor_resource->get_dimension();
                    return 0;
                }

                // Math operations
                godot::Ref<XTensorNode> add(const godot::Ref<XTensorNode>& other) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    auto b = other->m_tensor_resource->m_data.to_double_array();
                    result->set_data(XVariant::from_xarray(a + b).variant());
                    return result;
                }

                godot::Ref<XTensorNode> subtract(const godot::Ref<XTensorNode>& other) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    auto b = other->m_tensor_resource->m_data.to_double_array();
                    result->set_data(XVariant::from_xarray(a - b).variant());
                    return result;
                }

                godot::Ref<XTensorNode> multiply(const godot::Ref<XTensorNode>& other) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    auto b = other->m_tensor_resource->m_data.to_double_array();
                    result->set_data(XVariant::from_xarray(a * b).variant());
                    return result;
                }

                godot::Ref<XTensorNode> divide(const godot::Ref<XTensorNode>& other) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    auto b = other->m_tensor_resource->m_data.to_double_array();
                    result->set_data(XVariant::from_xarray(a / b).variant());
                    return result;
                }

                godot::Ref<XTensorNode> matmul(const godot::Ref<XTensorNode>& other) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    auto b = other->m_tensor_resource->m_data.to_double_array();
                    if (a.dimension() != 2 || b.dimension() != 2)
                    {
                        godot::UtilityFunctions::printerr("matmul requires 2D matrices");
                        return result;
                    }
                    result->set_data(XVariant::from_xarray(xt::linalg::dot(a, b)).variant());
                    return result;
                }

                godot::Ref<XTensorNode> transpose() const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    result->set_data(XVariant::from_xarray(xt::transpose(a)).variant());
                    return result;
                }

                godot::Ref<XTensorNode> sum(int64_t axis) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    xarray_container<double> res;
                    if (axis < 0)
                        res = xt::sum(a)();
                    else
                        res = xt::sum(a, {static_cast<size_t>(axis)});
                    result->set_data(XVariant::from_xarray(res).variant());
                    return result;
                }

                godot::Ref<XTensorNode> mean(int64_t axis) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    xarray_container<double> res;
                    if (axis < 0)
                        res = xt::mean(a)();
                    else
                        res = xt::mean(a, {static_cast<size_t>(axis)});
                    result->set_data(XVariant::from_xarray(res).variant());
                    return result;
                }

                godot::Ref<XTensorNode> max(int64_t axis) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    xarray_container<double> res;
                    if (axis < 0)
                        res = xt::amax(a)();
                    else
                        res = xt::amax(a, {static_cast<size_t>(axis)});
                    result->set_data(XVariant::from_xarray(res).variant());
                    return result;
                }

                godot::Ref<XTensorNode> min(int64_t axis) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    xarray_container<double> res;
                    if (axis < 0)
                        res = xt::amin(a)();
                    else
                        res = xt::amin(a, {static_cast<size_t>(axis)});
                    result->set_data(XVariant::from_xarray(res).variant());
                    return result;
                }

                double norm(const godot::String& ord) const
                {
                    auto a = m_tensor_resource->m_data.to_double_array();
                    std::string ord_str = ord.utf8().get_data();
                    return xt::norm_dispatch(a, ord_str);
                }

                // Indexing and slicing
                godot::Ref<XTensorNode> slice(const godot::PackedInt64Array& start, const godot::PackedInt64Array& end) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    size_t ndim = a.dimension();
                    std::vector<xt::xrange<size_t>> ranges;
                    for (int i = 0; i < start.size() && i < end.size() && static_cast<size_t>(i) < ndim; ++i)
                    {
                        ranges.emplace_back(static_cast<size_t>(start[i]), static_cast<size_t>(end[i]));
                    }
                    for (size_t i = ranges.size(); i < ndim; ++i)
                        ranges.emplace_back(0, a.shape()[i]);
                    
                    auto sliced = xt::view(a, ranges[0]);
                    // For multiple dimensions, we need to chain views; simplified for now
                    result->set_data(XVariant::from_xarray(sliced).variant());
                    return result;
                }

                godot::Variant get_element(const godot::PackedInt64Array& indices) const
                {
                    auto a = m_tensor_resource->m_data.to_double_array();
                    std::vector<size_t> idx;
                    for (int i = 0; i < indices.size(); ++i)
                        idx.push_back(static_cast<size_t>(indices[i]));
                    if (idx.size() == a.dimension())
                        return godot::Variant(a.element(idx));
                    else if (idx.size() == 0 && a.dimension() == 0)
                        return godot::Variant(a());
                    return godot::Variant();
                }

                void set_element(const godot::PackedInt64Array& indices, double value)
                {
                    ensure_resource();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    std::vector<size_t> idx;
                    for (int i = 0; i < indices.size(); ++i)
                        idx.push_back(static_cast<size_t>(indices[i]));
                    if (idx.size() == a.dimension())
                    {
                        a.element(idx) = value;
                        m_tensor_resource->set_data(XVariant::from_xarray(a).variant());
                        emit_signal("tensor_changed");
                    }
                }

                godot::Ref<XTensorNode> index(const godot::PackedInt64Array& indices) const
                {
                    // Advanced indexing: return sub-tensor
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    // Simplified: return element as scalar if indices match dimension
                    if (indices.size() == a.dimension())
                    {
                        std::vector<size_t> idx;
                        for (int i = 0; i < indices.size(); ++i)
                            idx.push_back(static_cast<size_t>(indices[i]));
                        xarray_container<double> scalar({});
                        scalar() = a.element(idx);
                        result->set_data(XVariant::from_xarray(scalar).variant());
                    }
                    return result;
                }

                // Linear algebra
                godot::Ref<XTensorNode> solve(const godot::Ref<XTensorNode>& b) const
                {
                    auto result = create_result_node();
                    auto A = m_tensor_resource->m_data.to_double_array();
                    auto B = b->m_tensor_resource->m_data.to_double_array();
                    try
                    {
                        auto x = xt::linalg::solve(A, B);
                        result->set_data(XVariant::from_xarray(x).variant());
                    }
                    catch (...)
                    {
                        godot::UtilityFunctions::printerr("solve: matrix is singular");
                    }
                    return result;
                }

                godot::Ref<XTensorNode> inverse() const
                {
                    auto result = create_result_node();
                    auto A = m_tensor_resource->m_data.to_double_array();
                    try
                    {
                        auto inv = xt::linalg::inv(A);
                        result->set_data(XVariant::from_xarray(inv).variant());
                    }
                    catch (...)
                    {
                        godot::UtilityFunctions::printerr("inverse: matrix is singular");
                    }
                    return result;
                }

                double determinant() const
                {
                    auto A = m_tensor_resource->m_data.to_double_array();
                    return xt::linalg::det(A);
                }

                godot::PackedFloat64Array eigenvalues() const
                {
                    godot::PackedFloat64Array result;
                    auto A = m_tensor_resource->m_data.to_double_array();
                    if (A.dimension() == 2 && A.shape()[0] == A.shape()[1])
                    {
                        auto [eigvals, eigvecs] = xt::linalg::eigh(A);
                        for (size_t i = 0; i < eigvals.size(); ++i)
                            result.append(eigvals(i));
                    }
                    return result;
                }

                godot::Array svd() const
                {
                    godot::Array result;
                    auto A = m_tensor_resource->m_data.to_double_array();
                    if (A.dimension() == 2)
                    {
                        auto [U, S, Vt] = xt::linalg::svd(A);
                        godot::Ref<XTensorNode> u_node;
                        u_node.instantiate();
                        u_node->set_data(XVariant::from_xarray(U).variant());
                        godot::Ref<XTensorNode> s_node;
                        s_node.instantiate();
                        s_node->set_data(XVariant::from_xarray(S).variant());
                        godot::Ref<XTensorNode> v_node;
                        v_node.instantiate();
                        v_node->set_data(XVariant::from_xarray(Vt).variant());
                        result.append(u_node);
                        result.append(s_node);
                        result.append(v_node);
                    }
                    return result;
                }

                // Statistics
                godot::Ref<XTensorNode> stddev(int64_t axis) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    xarray_container<double> res;
                    if (axis < 0)
                        res = xt::stddev(a)();
                    else
                        res = xt::stddev(a, {static_cast<size_t>(axis)});
                    result->set_data(XVariant::from_xarray(res).variant());
                    return result;
                }

                godot::Ref<XTensorNode> variance(int64_t axis) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    xarray_container<double> res;
                    if (axis < 0)
                        res = xt::variance(a)();
                    else
                        res = xt::variance(a, {static_cast<size_t>(axis)});
                    result->set_data(XVariant::from_xarray(res).variant());
                    return result;
                }

                godot::Ref<XTensorNode> quantile(double q, int64_t axis) const
                {
                    auto result = create_result_node();
                    auto a = m_tensor_resource->m_data.to_double_array();
                    auto res = xt::quantile(a, q, static_cast<size_t>(axis));
                    result->set_data(XVariant::from_xarray(res).variant());
                    return result;
                }

                godot::Ref<XTensorNode> median(int64_t axis) const
                {
                    return quantile(0.5, axis);
                }

                // I/O
                bool load_tensor(const godot::String& path)
                {
                    ensure_resource();
                    bool success = m_tensor_resource->load_from_file(path);
                    if (success)
                    {
                        m_tensor_path = path;
                        emit_signal("tensor_loaded");
                        emit_signal("tensor_changed");
                    }
                    return success;
                }

                bool save_tensor(const godot::String& path) const
                {
                    if (m_tensor_resource.is_valid())
                        return m_tensor_resource->save_to_file(path);
                    return false;
                }

                godot::String to_json() const
                {
                    if (m_tensor_resource.is_valid())
                        return m_tensor_resource->to_json();
                    return "{}";
                }

                void from_json(const godot::String& json)
                {
                    ensure_resource();
                    m_tensor_resource->from_json(json);
                    emit_signal("tensor_changed");
                }

                // Conversions
                godot::PackedVector3Array to_vector3_array() const
                {
                    godot::PackedVector3Array arr;
                    auto a = m_tensor_resource->m_data.to_double_array();
                    if (a.dimension() == 2 && a.shape()[1] == 3)
                    {
                        for (size_t i = 0; i < a.shape()[0]; ++i)
                        {
                            arr.append(godot::Vector3(
                                static_cast<float>(a(i, 0)),
                                static_cast<float>(a(i, 1)),
                                static_cast<float>(a(i, 2))
                            ));
                        }
                    }
                    return arr;
                }

                godot::PackedVector2Array to_vector2_array() const
                {
                    godot::PackedVector2Array arr;
                    auto a = m_tensor_resource->m_data.to_double_array();
                    if (a.dimension() == 2 && a.shape()[1] == 2)
                    {
                        for (size_t i = 0; i < a.shape()[0]; ++i)
                        {
                            arr.append(godot::Vector2(
                                static_cast<float>(a(i, 0)),
                                static_cast<float>(a(i, 1))
                            ));
                        }
                    }
                    return arr;
                }

                godot::PackedColorArray to_color_array() const
                {
                    godot::PackedColorArray arr;
                    auto a = m_tensor_resource->m_data.to_double_array();
                    if (a.dimension() == 2 && (a.shape()[1] == 3 || a.shape()[1] == 4))
                    {
                        for (size_t i = 0; i < a.shape()[0]; ++i)
                        {
                            float r = static_cast<float>(std::clamp(a(i, 0), 0.0, 1.0));
                            float g = static_cast<float>(std::clamp(a(i, 1), 0.0, 1.0));
                            float b = static_cast<float>(std::clamp(a(i, 2), 0.0, 1.0));
                            float alpha = (a.shape()[1] == 4) ? static_cast<float>(std::clamp(a(i, 3), 0.0, 1.0)) : 1.0f;
                            arr.append(godot::Color(r, g, b, alpha));
                        }
                    }
                    return arr;
                }

                godot::Array to_transform_array() const
                {
                    godot::Array arr;
                    auto a = m_tensor_resource->m_data.to_double_array();
                    if (a.dimension() == 3 && a.shape()[1] == 4 && a.shape()[2] == 4)
                    {
                        for (size_t i = 0; i < a.shape()[0]; ++i)
                        {
                            godot::Transform3D t;
                            godot::Basis b;
                            b.set_row(0, godot::Vector3(a(i, 0, 0), a(i, 0, 1), a(i, 0, 2)));
                            b.set_row(1, godot::Vector3(a(i, 1, 0), a(i, 1, 1), a(i, 1, 2)));
                            b.set_row(2, godot::Vector3(a(i, 2, 0), a(i, 2, 1), a(i, 2, 2)));
                            t.set_basis(b);
                            t.set_origin(godot::Vector3(a(i, 0, 3), a(i, 1, 3), a(i, 2, 3)));
                            arr.append(t);
                        }
                    }
                    return arr;
                }

                void from_vector3_array(const godot::PackedVector3Array& array)
                {
                    xarray_container<double> arr({static_cast<size_t>(array.size()), 3});
                    for (int i = 0; i < array.size(); ++i)
                    {
                        godot::Vector3 v = array[i];
                        arr(i, 0) = v.x;
                        arr(i, 1) = v.y;
                        arr(i, 2) = v.z;
                    }
                    set_data(XVariant::from_xarray(arr).variant());
                }

                void from_vector2_array(const godot::PackedVector2Array& array)
                {
                    xarray_container<double> arr({static_cast<size_t>(array.size()), 2});
                    for (int i = 0; i < array.size(); ++i)
                    {
                        godot::Vector2 v = array[i];
                        arr(i, 0) = v.x;
                        arr(i, 1) = v.y;
                    }
                    set_data(XVariant::from_xarray(arr).variant());
                }

                // Utility
                godot::Ref<XTensorNode> copy() const
                {
                    auto result = create_result_node();
                    result->set_data(m_tensor_resource->get_data());
                    return result;
                }

                bool is_valid() const
                {
                    return m_tensor_resource.is_valid() && m_tensor_resource->get_size() > 0;
                }

                void clear()
                {
                    m_tensor_resource.unref();
                    m_tensor_resource.instantiate();
                    m_dirty = true;
                    emit_signal("tensor_changed");
                }

                // Static factory
                static godot::Ref<XTensorNode> create_zeros(const godot::PackedInt64Array& shape)
                {
                    godot::Ref<XTensorNode> node;
                    node.instantiate();
                    std::vector<size_t> sh;
                    for (int i = 0; i < shape.size(); ++i)
                        sh.push_back(static_cast<size_t>(shape[i]));
                    node->set_data(XVariant::from_xarray(xt::zeros<double>(sh)).variant());
                    return node;
                }

                static godot::Ref<XTensorNode> create_ones(const godot::PackedInt64Array& shape)
                {
                    godot::Ref<XTensorNode> node;
                    node.instantiate();
                    std::vector<size_t> sh;
                    for (int i = 0; i < shape.size(); ++i)
                        sh.push_back(static_cast<size_t>(shape[i]));
                    node->set_data(XVariant::from_xarray(xt::ones<double>(sh)).variant());
                    return node;
                }

                static godot::Ref<XTensorNode> create_random(const godot::PackedInt64Array& shape)
                {
                    godot::Ref<XTensorNode> node;
                    node.instantiate();
                    std::vector<size_t> sh;
                    for (int i = 0; i < shape.size(); ++i)
                        sh.push_back(static_cast<size_t>(shape[i]));
                    node->set_data(XVariant::from_xarray(xt::random<double>(sh)).variant());
                    return node;
                }

                static godot::Ref<XTensorNode> create_identity(int64_t n)
                {
                    godot::Ref<XTensorNode> node;
                    node.instantiate();
                    node->set_data(XVariant::from_xarray(xt::eye<double>(static_cast<size_t>(n))).variant());
                    return node;
                }

                static godot::Ref<XTensorNode> create_linspace(double start, double stop, int64_t num)
                {
                    godot::Ref<XTensorNode> node;
                    node.instantiate();
                    node->set_data(XVariant::from_xarray(xt::linspace<double>(start, stop, static_cast<size_t>(num))).variant());
                    return node;
                }

                // Metadata
                void set_node_metadata(const godot::Dictionary& meta) { m_node_metadata = meta; }
                godot::Dictionary get_node_metadata() const { return m_node_metadata; }

            private:
                void ensure_resource()
                {
                    if (!m_tensor_resource.is_valid())
                        m_tensor_resource.instantiate();
                }

                godot::Ref<XTensorNode> create_result_node() const
                {
                    godot::Ref<XTensorNode> node;
                    node.instantiate();
                    return node;
                }
            };

            // --------------------------------------------------------------------
            // XTensorMeshNode - Node for visualizing tensors as meshes
            // --------------------------------------------------------------------
            class XTensorMeshNode : public godot::MeshInstance3D
            {
                GDCLASS(XTensorMeshNode, godot::MeshInstance3D)

            private:
                godot::Ref<XTensorNode> m_tensor_node;
                godot::String m_tensor_path;
                bool m_auto_update = true;
                godot::Color m_vertex_color = godot::Color(1, 1, 1, 1);
                bool m_use_vertex_colors = true;
                float m_point_size = 0.05f;
                float m_line_width = 0.02f;
                godot::AABB m_bounds;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_tensor_node", "node"), &XTensorMeshNode::set_tensor_node);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_tensor_node"), &XTensorMeshNode::get_tensor_node);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_tensor_path", "path"), &XTensorMeshNode::set_tensor_path);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_tensor_path"), &XTensorMeshNode::get_tensor_path);
                    godot::ClassDB::bind_method(godot::D_METHOD("update_mesh"), &XTensorMeshNode::update_mesh);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_auto_update", "enabled"), &XTensorMeshNode::set_auto_update);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_auto_update"), &XTensorMeshNode::get_auto_update);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_vertex_color", "color"), &XTensorMeshNode::set_vertex_color);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_vertex_color"), &XTensorMeshNode::get_vertex_color);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_point_size", "size"), &XTensorMeshNode::set_point_size);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_point_size"), &XTensorMeshNode::get_point_size);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_bounds"), &XTensorMeshNode::get_bounds);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "tensor_node", godot::PROPERTY_HINT_NODE_TYPE, "XTensorNode"), "set_tensor_node", "get_tensor_node");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "tensor_path", godot::PROPERTY_HINT_FILE, "*.npy,*.npz,*.json"), "set_tensor_path", "get_tensor_path");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "auto_update"), "set_auto_update", "get_auto_update");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::COLOR, "vertex_color"), "set_vertex_color", "get_vertex_color");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "point_size", godot::PROPERTY_HINT_RANGE, "0.001,1.0,0.001"), "set_point_size", "get_point_size");
                }

            public:
                void _ready() override
                {
                    if (m_tensor_path.is_empty() && !m_tensor_node.is_valid())
                    {
                        // Try to find child XTensorNode
                        godot::TypedArray<godot::Node> children = get_children();
                        for (int i = 0; i < children.size(); ++i)
                        {
                            godot::Node* child = godot::Object::cast_to<godot::Node>(children[i]);
                            if (child && child->is_class("XTensorNode"))
                            {
                                m_tensor_node = godot::Ref<XTensorNode>(godot::Object::cast_to<XTensorNode>(child));
                                break;
                            }
                        }
                    }
                    
                    if (!m_tensor_path.is_empty())
                    {
                        godot::Ref<XTensorNode> node;
                        node.instantiate();
                        if (node->load_tensor(m_tensor_path))
                            m_tensor_node = node;
                    }
                    
                    update_mesh();
                }

                void set_tensor_node(const godot::Ref<XTensorNode>& node)
                {
                    if (m_tensor_node.is_valid())
                    {
                        m_tensor_node->disconnect("tensor_changed", callable_mp(this, &XTensorMeshNode::_on_tensor_changed));
                    }
                    m_tensor_node = node;
                    if (m_tensor_node.is_valid())
                    {
                        m_tensor_node->connect("tensor_changed", callable_mp(this, &XTensorMeshNode::_on_tensor_changed));
                        if (m_auto_update)
                            update_mesh();
                    }
                }

                godot::Ref<XTensorNode> get_tensor_node() const
                {
                    return m_tensor_node;
                }

                void set_tensor_path(const godot::String& path)
                {
                    m_tensor_path = path;
                }

                godot::String get_tensor_path() const
                {
                    return m_tensor_path;
                }

                void set_auto_update(bool enabled)
                {
                    m_auto_update = enabled;
                }

                bool get_auto_update() const
                {
                    return m_auto_update;
                }

                void set_vertex_color(const godot::Color& color)
                {
                    m_vertex_color = color;
                    if (m_auto_update) update_mesh();
                }

                godot::Color get_vertex_color() const
                {
                    return m_vertex_color;
                }

                void set_point_size(float size)
                {
                    m_point_size = size;
                    if (m_auto_update) update_mesh();
                }

                float get_point_size() const
                {
                    return m_point_size;
                }

                godot::AABB get_bounds() const
                {
                    return m_bounds;
                }

                void update_mesh()
                {
                    if (!m_tensor_node.is_valid() || !m_tensor_node->is_valid())
                    {
                        set_mesh(godot::Ref<godot::ArrayMesh>());
                        return;
                    }
                    
                    auto data = m_tensor_node->get_tensor_resource()->m_data.to_double_array();
                    size_t dim = data.dimension();
                    
                    godot::Ref<godot::ArrayMesh> mesh;
                    mesh.instantiate();
                    
                    godot::Array arrays;
                    arrays.resize(godot::Mesh::ARRAY_MAX);
                    
                    if (dim == 2 && data.shape()[1] == 3)
                    {
                        // Point cloud
                        build_point_cloud(data, arrays);
                    }
                    else if (dim == 3 && data.shape()[1] == 3 && data.shape()[2] == 3)
                    {
                        // Mesh triangles (faces x 3 vertices x 3 coords)
                        build_triangle_mesh(data, arrays);
                    }
                    else if (dim == 2 && data.shape()[1] == 2)
                    {
                        // 2D points as line strip
                        build_line_strip(data, arrays);
                    }
                    else
                    {
                        // Fallback: treat as flattened point list
                        build_generic_points(data, arrays);
                    }
                    
                    if (arrays[godot::Mesh::ARRAY_VERTEX].get_type() != godot::Variant::NIL)
                    {
                        mesh->add_surface_from_arrays(godot::Mesh::PRIMITIVE_TRIANGLES, arrays);
                        set_mesh(mesh);
                    }
                    
                    // Compute bounds
                    godot::PackedVector3Array vertices = arrays[godot::Mesh::ARRAY_VERTEX];
                    if (vertices.size() > 0)
                    {
                        godot::Vector3 min_pt = vertices[0];
                        godot::Vector3 max_pt = vertices[0];
                        for (int i = 1; i < vertices.size(); ++i)
                        {
                            min_pt = min_pt.min(vertices[i]);
                            max_pt = max_pt.max(vertices[i]);
                        }
                        m_bounds = godot::AABB(min_pt, max_pt - min_pt);
                    }
                }

            private:
                void build_point_cloud(const xarray_container<double>& data, godot::Array& arrays)
                {
                    size_t n = data.shape()[0];
                    godot::PackedVector3Array vertices;
                    godot::PackedColorArray colors;
                    
                    vertices.resize(static_cast<int>(n));
                    colors.resize(static_cast<int>(n));
                    
                    for (size_t i = 0; i < n; ++i)
                    {
                        vertices.set(static_cast<int>(i), godot::Vector3(
                            static_cast<float>(data(i, 0)),
                            static_cast<float>(data(i, 1)),
                            static_cast<float>(data(i, 2))
                        ));
                        colors.set(static_cast<int>(i), m_vertex_color);
                    }
                    
                    arrays[godot::Mesh::ARRAY_VERTEX] = vertices;
                    arrays[godot::Mesh::ARRAY_COLOR] = colors;
                }

                void build_triangle_mesh(const xarray_container<double>& data, godot::Array& arrays)
                {
                    size_t n_faces = data.shape()[0];
                    godot::PackedVector3Array vertices;
                    godot::PackedVector3Array normals;
                    godot::PackedColorArray colors;
                    
                    vertices.resize(static_cast<int>(n_faces * 3));
                    normals.resize(static_cast<int>(n_faces * 3));
                    colors.resize(static_cast<int>(n_faces * 3));
                    
                    for (size_t f = 0; f < n_faces; ++f)
                    {
                        godot::Vector3 v0(data(f, 0, 0), data(f, 0, 1), data(f, 0, 2));
                        godot::Vector3 v1(data(f, 1, 0), data(f, 1, 1), data(f, 1, 2));
                        godot::Vector3 v2(data(f, 2, 0), data(f, 2, 1), data(f, 2, 2));
                        
                        godot::Vector3 normal = (v1 - v0).cross(v2 - v0).normalized();
                        
                        size_t base = f * 3;
                        vertices.set(static_cast<int>(base), v0);
                        vertices.set(static_cast<int>(base + 1), v1);
                        vertices.set(static_cast<int>(base + 2), v2);
                        normals.set(static_cast<int>(base), normal);
                        normals.set(static_cast<int>(base + 1), normal);
                        normals.set(static_cast<int>(base + 2), normal);
                        colors.set(static_cast<int>(base), m_vertex_color);
                        colors.set(static_cast<int>(base + 1), m_vertex_color);
                        colors.set(static_cast<int>(base + 2), m_vertex_color);
                    }
                    
                    arrays[godot::Mesh::ARRAY_VERTEX] = vertices;
                    arrays[godot::Mesh::ARRAY_NORMAL] = normals;
                    arrays[godot::Mesh::ARRAY_COLOR] = colors;
                }

                void build_line_strip(const xarray_container<double>& data, godot::Array& arrays)
                {
                    size_t n = data.shape()[0];
                    godot::PackedVector3Array vertices;
                    godot::PackedColorArray colors;
                    
                    vertices.resize(static_cast<int>(n));
                    colors.resize(static_cast<int>(n));
                    
                    for (size_t i = 0; i < n; ++i)
                    {
                        vertices.set(static_cast<int>(i), godot::Vector3(
                            static_cast<float>(data(i, 0)),
                            static_cast<float>(data(i, 1)),
                            0.0f
                        ));
                        colors.set(static_cast<int>(i), m_vertex_color);
                    }
                    
                    arrays[godot::Mesh::ARRAY_VERTEX] = vertices;
                    arrays[godot::Mesh::ARRAY_COLOR] = colors;
                }

                void build_generic_points(const xarray_container<double>& data, godot::Array& arrays)
                {
                    size_t n = data.size();
                    godot::PackedVector3Array vertices;
                    godot::PackedColorArray colors;
                    
                    vertices.resize(static_cast<int>(n));
                    colors.resize(static_cast<int>(n));
                    
                    // Flatten and map to 3D using a spiral layout
                    for (size_t i = 0; i < n; ++i)
                    {
                        float val = static_cast<float>(data.flat(i));
                        float angle = static_cast<float>(i) * 0.1f;
                        float radius = std::abs(val);
                        float x = radius * std::cos(angle);
                        float z = radius * std::sin(angle);
                        float y = val;
                        vertices.set(static_cast<int>(i), godot::Vector3(x, y, z));
                        colors.set(static_cast<int>(i), m_vertex_color);
                    }
                    
                    arrays[godot::Mesh::ARRAY_VERTEX] = vertices;
                    arrays[godot::Mesh::ARRAY_COLOR] = colors;
                }

                void _on_tensor_changed()
                {
                    if (m_auto_update)
                        update_mesh();
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration helper
            // --------------------------------------------------------------------
            class XNodeRegistry
            {
            public:
                static void register_nodes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XTensorNode>();
                    godot::ClassDB::register_class<XTensorMeshNode>();
#endif
                }

                static void unregister_nodes()
                {
                }
            };

        } // namespace godot_bridge

        // Bring Godot node types into xt namespace
        using godot_bridge::XTensorNode;
        using godot_bridge::XTensorMeshNode;
        using godot_bridge::XNodeRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XNODE_HPP

// godot/xnode.hpp
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stack>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"
#include "xtu/graphics/xrenderer.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class Node;
class SceneTree;
class Viewport;
class Node2D;
class Node3D;
class Control;
class CanvasItem;
class AnimationPlayer;

// #############################################################################
// Process modes
// #############################################################################
enum class ProcessMode : uint8_t {
    INHERIT = 0,
    PAUSABLE = 1,
    WHEN_PAUSED = 2,
    ALWAYS = 3,
    DISABLED = 4
};

// #############################################################################
// Process priority (lower = earlier)
// #############################################################################
using ProcessPriority = int32_t;
static constexpr ProcessPriority PROCESS_PRIORITY_DEFAULT = 0;

// #############################################################################
// Group management
// #############################################################################
class NodeGroupManager {
private:
    static std::unordered_map<StringName, std::unordered_set<Node*>>& get_groups() {
        static std::unordered_map<StringName, std::unordered_set<Node*>> groups;
        return groups;
    }

    static std::mutex& get_mutex() {
        static std::mutex mutex;
        return mutex;
    }

public:
    static void add_to_group(Node* node, const StringName& group) {
        std::lock_guard<std::mutex> lock(get_mutex());
        get_groups()[group].insert(node);
    }

    static void remove_from_group(Node* node, const StringName& group) {
        std::lock_guard<std::mutex> lock(get_mutex());
        auto it = get_groups().find(group);
        if (it != get_groups().end()) {
            it->second.erase(node);
            if (it->second.empty()) {
                get_groups().erase(it);
            }
        }
    }

    static std::vector<Node*> get_nodes_in_group(const StringName& group) {
        std::lock_guard<std::mutex> lock(get_mutex());
        auto it = get_groups().find(group);
        if (it != get_groups().end()) {
            return std::vector<Node*>(it->second.begin(), it->second.end());
        }
        return {};
    }

    static bool is_in_group(Node* node, const StringName& group) {
        std::lock_guard<std::mutex> lock(get_mutex());
        auto it = get_groups().find(group);
        return it != get_groups().end() && it->second.count(node) > 0;
    }

    static void clear_node(Node* node) {
        std::lock_guard<std::mutex> lock(get_mutex());
        for (auto& kv : get_groups()) {
            kv.second.erase(node);
        }
    }
};

// #############################################################################
// NodePath - path to a node in the scene tree
// #############################################################################
class NodePath {
private:
    std::vector<StringName> m_names;
    std::vector<StringName> m_subnames;
    bool m_absolute = false;

public:
    NodePath() = default;
    NodePath(const std::string& path) { parse(path); }
    NodePath(const char* path) : NodePath(std::string(path)) {}

    void parse(const std::string& path) {
        m_names.clear();
        m_subnames.clear();
        m_absolute = !path.empty() && path[0] == '/';

        size_t pos = m_absolute ? 1 : 0;
        while (pos < path.size()) {
            size_t end = path.find('/', pos);
            if (end == std::string::npos) end = path.size();
            std::string segment = path.substr(pos, end - pos);

            size_t colon = segment.find(':');
            if (colon != std::string::npos) {
                m_names.push_back(StringName(segment.substr(0, colon)));
                m_subnames.push_back(StringName(segment.substr(colon + 1)));
            } else {
                m_names.push_back(StringName(segment));
                m_subnames.push_back(StringName());
            }
            pos = end + 1;
        }
    }

    std::string to_string() const {
        std::string result = m_absolute ? "/" : "";
        for (size_t i = 0; i < m_names.size(); ++i) {
            if (i > 0) result += "/";
            result += m_names[i].string();
            if (m_subnames[i]) {
                result += ":" + m_subnames[i].string();
            }
        }
        return result;
    }

    size_t get_name_count() const { return m_names.size(); }
    StringName get_name(size_t idx) const { return idx < m_names.size() ? m_names[idx] : StringName(); }
    StringName get_subname(size_t idx) const { return idx < m_subnames.size() ? m_subnames[idx] : StringName(); }
    bool is_absolute() const { return m_absolute; }
    bool is_empty() const { return m_names.empty(); }
};

// #############################################################################
// Node - base class for all scene objects
// #############################################################################
class Node : public Object {
    XTU_GODOT_REGISTER_CLASS(Node, Object)

public:
    // Internal flags for state management
    enum InternalFlags : uint32_t {
        FLAG_READY = 1 << 0,
        FLAG_INSIDE_TREE = 1 << 1,
        FLAG_PROCESSING = 1 << 2,
        FLAG_PHYSICS_PROCESSING = 1 << 3,
        FLAG_INTERNAL_PROCESSING = 1 << 4,
        FLAG_INTERNAL_PHYSICS_PROCESSING = 1 << 5,
        FLAG_PAUSED = 1 << 6,
        FLAG_EDITOR = 1 << 7,
        FLAG_SCENE_INSTANTIATED = 1 << 8,
        FLAG_OWNER_VALID = 1 << 9,
        FLAG_DISABLED = 1 << 10
    };

private:
    // Hierarchy
    Node* m_parent = nullptr;
    std::vector<Node*> m_children;
    Node* m_owner = nullptr;
    uint64_t m_scene_file_id = 0;

    // Name
    StringName m_name;

    // Processing
    ProcessMode m_process_mode = ProcessMode::INHERIT;
    ProcessPriority m_process_priority = PROCESS_PRIORITY_DEFAULT;
    ProcessPriority m_physics_process_priority = PROCESS_PRIORITY_DEFAULT;
    uint32_t m_internal_flags = 0;

    // Multiplayer
    int32_t m_multiplayer_authority = -1;

    // Scene tree reference
    SceneTree* m_tree = nullptr;

    // Deferred calls
    struct DeferredCall {
        StringName method;
        std::vector<Variant> args;
    };
    std::vector<DeferredCall> m_deferred_calls;
    mutable std::mutex m_deferred_mutex;

    // Groups (local cache)
    std::unordered_set<StringName> m_groups;

protected:
    // Virtual lifecycle methods
    virtual void _enter_tree() {}
    virtual void _exit_tree() {}
    virtual void _ready() {}
    virtual void _process(double delta) {}
    virtual void _physics_process(double delta) {}
    virtual void _input(const Variant& event) {}
    virtual void _unhandled_input(const Variant& event) {}
    virtual void _unhandled_key_input(const Variant& event) {}

    // Notification handler
    void _notification(int p_what) override {
        switch (p_what) {
            case NOTIFICATION_ENTER_TREE: _enter_tree(); break;
            case NOTIFICATION_EXIT_TREE: _exit_tree(); break;
            case NOTIFICATION_READY: _ready(); break;
            default: Object::_notification(p_what); break;
        }
    }

public:
    Node() = default;
    ~Node() override {
        // Remove from groups
        for (const auto& group : m_groups) {
            NodeGroupManager::remove_from_group(this, group);
        }
        // Detach from parent
        if (m_parent) {
            m_parent->remove_child(this);
        }
        // Remove children
        while (!m_children.empty()) {
            remove_child(m_children.back());
        }
    }

    static StringName get_class_static() { return StringName("Node"); }

    // #########################################################################
    // Name and path
    // #########################################################################
    void set_name(const StringName& name) { m_name = name; }
    StringName get_name() const { return m_name; }

    std::string get_path() const {
        if (!m_parent) return m_absolute_path;
        return m_parent->get_path() + "/" + m_name.string();
    }

    NodePath get_path_to(const Node* node) const {
        if (!node || !is_ancestor_of(node)) return NodePath();
        std::vector<StringName> names;
        const Node* current = node;
        while (current && current != this) {
            names.push_back(current->m_name);
            current = current->m_parent;
        }
        std::reverse(names.begin(), names.end());
        NodePath path;
        for (const auto& n : names) path.parse(path.to_string() + "/" + n.string());
        return path;
    }

    // #########################################################################
    // Hierarchy
    // #########################################################################
    Node* get_parent() const { return m_parent; }
    size_t get_child_count() const { return m_children.size(); }
    Node* get_child(size_t idx) const { return idx < m_children.size() ? m_children[idx] : nullptr; }
    const std::vector<Node*>& get_children() const { return m_children; }

    void add_child(Node* child, bool legible_unique_name = false) {
        if (!child || child->m_parent == this) return;
        if (child->m_parent) {
            child->m_parent->remove_child(child);
        }
        child->m_parent = this;
        m_children.push_back(child);
        if (m_tree) {
            child->_propagate_enter_tree();
        }
    }

    void remove_child(Node* child) {
        auto it = std::find(m_children.begin(), m_children.end(), child);
        if (it == m_children.end()) return;
        if (m_tree) {
            child->_propagate_exit_tree();
        }
        child->m_parent = nullptr;
        m_children.erase(it);
    }

    void move_child(Node* child, size_t to_pos) {
        auto it = std::find(m_children.begin(), m_children.end(), child);
        if (it == m_children.end() || to_pos >= m_children.size()) return;
        size_t from = std::distance(m_children.begin(), it);
        if (from < to_pos) {
            std::rotate(m_children.begin() + from, m_children.begin() + from + 1, m_children.begin() + to_pos + 1);
        } else {
            std::rotate(m_children.begin() + to_pos, m_children.begin() + from, m_children.begin() + from + 1);
        }
    }

    Node* find_child(const StringName& name, bool recursive = true, bool owned = true) const {
        for (Node* child : m_children) {
            if (child->m_name == name && (!owned || child->m_owner == m_owner)) {
                return child;
            }
            if (recursive) {
                Node* found = child->find_child(name, true, owned);
                if (found) return found;
            }
        }
        return nullptr;
    }

    Node* get_node(const NodePath& path) const {
        if (path.is_empty()) return nullptr;
        const Node* current = path.is_absolute() ? get_root() : this;
        for (size_t i = 0; i < path.get_name_count(); ++i) {
            StringName name = path.get_name(i);
            current = current->find_child(name, false, false);
            if (!current) return nullptr;
        }
        return const_cast<Node*>(current);
    }

    Node* get_root() const {
        const Node* node = this;
        while (node->m_parent) node = node->m_parent;
        return const_cast<Node*>(node);
    }

    bool is_ancestor_of(const Node* node) const {
        while (node) {
            if (node->m_parent == this) return true;
            node = node->m_parent;
        }
        return false;
    }

    // #########################################################################
    // Owner (for instanced scenes)
    // #########################################################################
    void set_owner(Node* owner) { m_owner = owner; }
    Node* get_owner() const { return m_owner; }
    bool is_owned_by(const Node* owner) const { return m_owner == owner; }

    // #########################################################################
    // Scene tree integration
    // #########################################################################
    bool is_inside_tree() const { return (m_internal_flags & FLAG_INSIDE_TREE) != 0; }
    SceneTree* get_tree() const { return m_tree; }

    void _propagate_enter_tree() {
        if (m_internal_flags & FLAG_INSIDE_TREE) return;
        m_internal_flags |= FLAG_INSIDE_TREE;
        m_tree = m_parent ? m_parent->m_tree : dynamic_cast<SceneTree*>(this);
        for (Node* child : m_children) {
            child->_propagate_enter_tree();
        }
        notification(NOTIFICATION_ENTER_TREE);
        if (!(m_internal_flags & FLAG_READY)) {
            m_internal_flags |= FLAG_READY;
            call_deferred("_ready");
        }
    }

    void _propagate_exit_tree() {
        if (!(m_internal_flags & FLAG_INSIDE_TREE)) return;
        notification(NOTIFICATION_EXIT_TREE);
        m_internal_flags &= ~FLAG_INSIDE_TREE;
        m_internal_flags &= ~FLAG_READY;
        m_tree = nullptr;
        for (Node* child : m_children) {
            child->_propagate_exit_tree();
        }
    }

    // #########################################################################
    // Processing
    // #########################################################################
    void set_process_mode(ProcessMode mode) { m_process_mode = mode; }
    ProcessMode get_process_mode() const { return m_process_mode; }

    ProcessMode get_effective_process_mode() const {
        if (m_process_mode != ProcessMode::INHERIT) return m_process_mode;
        return m_parent ? m_parent->get_effective_process_mode() : ProcessMode::PAUSABLE;
    }

    void set_process(bool enable) {
        if (enable) {
            m_internal_flags |= FLAG_PROCESSING;
        } else {
            m_internal_flags &= ~FLAG_PROCESSING;
        }
    }
    bool is_processing() const { return (m_internal_flags & FLAG_PROCESSING) != 0; }

    void set_physics_process(bool enable) {
        if (enable) {
            m_internal_flags |= FLAG_PHYSICS_PROCESSING;
        } else {
            m_internal_flags &= ~FLAG_PHYSICS_PROCESSING;
        }
    }
    bool is_physics_processing() const { return (m_internal_flags & FLAG_PHYSICS_PROCESSING) != 0; }

    void set_process_priority(ProcessPriority priority) { m_process_priority = priority; }
    ProcessPriority get_process_priority() const { return m_process_priority; }

    void set_physics_process_priority(ProcessPriority priority) { m_physics_process_priority = priority; }
    ProcessPriority get_physics_process_priority() const { return m_physics_process_priority; }

    void process(double delta) {
        if (m_internal_flags & FLAG_PROCESSING) {
            _process(delta);
        }
    }

    void physics_process(double delta) {
        if (m_internal_flags & FLAG_PHYSICS_PROCESSING) {
            _physics_process(delta);
        }
    }

    // #########################################################################
    // Deferred calls
    // #########################################################################
    void call_deferred(const StringName& method, const std::vector<Variant>& args = {}) {
        std::lock_guard<std::mutex> lock(m_deferred_mutex);
        m_deferred_calls.push_back({method, args});
    }

    template <typename... Args>
    void call_deferred(const StringName& method, Args&&... args) {
        call_deferred(method, {Variant(std::forward<Args>(args))...});
    }

    void flush_deferred_calls() {
        std::vector<DeferredCall> calls;
        {
            std::lock_guard<std::mutex> lock(m_deferred_mutex);
            calls.swap(m_deferred_calls);
        }
        for (const auto& call : calls) {
            const Variant* arg_ptrs[call.args.size()];
            for (size_t i = 0; i < call.args.size(); ++i) arg_ptrs[i] = &call.args[i];
            this->call(call.method, arg_ptrs, call.args.size());
        }
    }

    // #########################################################################
    // Groups
    // #########################################################################
    void add_to_group(const StringName& group) {
        if (m_groups.insert(group).second) {
            NodeGroupManager::add_to_group(this, group);
        }
    }

    void remove_from_group(const StringName& group) {
        if (m_groups.erase(group)) {
            NodeGroupManager::remove_from_group(this, group);
        }
    }

    bool is_in_group(const StringName& group) const {
        return m_groups.count(group) > 0;
    }

    const std::unordered_set<StringName>& get_groups() const {
        return m_groups;
    }

    // #########################################################################
    // Multiplayer authority
    // #########################################################################
    void set_multiplayer_authority(int32_t peer_id) { m_multiplayer_authority = peer_id; }
    int32_t get_multiplayer_authority() const { return m_multiplayer_authority; }
    bool is_multiplayer_authority() const { return m_multiplayer_authority == get_tree()->get_multiplayer_authority(); }

    // #########################################################################
    // Scene file tracking
    // #########################################################################
    void set_scene_file_id(uint64_t id) { m_scene_file_id = id; }
    uint64_t get_scene_file_id() const { return m_scene_file_id; }

    // #########################################################################
    // Queue operations (for deferred deletion)
    // #########################################################################
    void queue_free() {
        if (is_inside_tree() && m_tree) {
            m_tree->queue_delete(this);
        } else {
            delete this;
        }
    }

private:
    static inline const std::string m_absolute_path = "/root";
};

// #############################################################################
// SceneTree - main loop and scene manager
// #############################################################################
class SceneTree : public Node {
    XTU_GODOT_REGISTER_CLASS(SceneTree, Node)

private:
    Viewport* m_root = nullptr;
    Node* m_current_scene = nullptr;
    Node* m_edited_scene_root = nullptr;

    // Processing queues (sorted by priority)
    std::vector<Node*> m_process_nodes;
    std::vector<Node*> m_physics_process_nodes;
    std::mutex m_queue_mutex;

    // Deletion queue
    std::vector<Node*> m_delete_queue;
    std::mutex m_delete_mutex;

    // Timing
    std::chrono::steady_clock::time_point m_last_frame_time;
    double m_time_scale = 1.0;
    bool m_paused = false;

    // Multiplayer
    int32_t m_multiplayer_peer = -1;
    int32_t m_multiplayer_authority = 1;

    // Groups for parallel processing
    std::unordered_map<StringName, std::vector<Node*>> m_process_groups;

public:
    SceneTree() {
        m_last_frame_time = std::chrono::steady_clock::now();
    }

    static StringName get_class_static() { return StringName("SceneTree"); }

    // #########################################################################
    // Root and current scene
    // #########################################################################
    void set_root(Viewport* root) { m_root = root; }
    Viewport* get_root() const { return m_root; }

    void set_current_scene(Node* scene) {
        if (m_current_scene) {
            m_root->remove_child(m_current_scene);
        }
        m_current_scene = scene;
        if (scene) {
            m_root->add_child(scene);
        }
    }
    Node* get_current_scene() const { return m_current_scene; }

    void set_edited_scene_root(Node* scene) { m_edited_scene_root = scene; }
    Node* get_edited_scene_root() const { return m_edited_scene_root; }

    // #########################################################################
    // Change scene
    // #########################################################################
    void change_scene_to_file(const std::string& path) {
        // Load new scene
        Ref<Resource> res = ResourceLoader::load(path);
        if (!res.is_valid()) return;
        // TODO: Instantiate PackedScene
        Node* new_scene = nullptr; // packed_scene->instantiate();
        if (new_scene) {
            call_deferred("_change_scene", new_scene);
        }
    }

    void change_scene_to_packed(const Ref<Resource>& packed_scene) {
        // Instantiate and change
    }

    void reload_current_scene() {
        if (m_current_scene) {
            std::string path = m_current_scene->get_scene_file_path();
            if (!path.empty()) {
                change_scene_to_file(path);
            }
        }
    }

    // #########################################################################
    // Processing registration
    // #########################################################################
    void register_process_node(Node* node) {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        auto it = std::find(m_process_nodes.begin(), m_process_nodes.end(), node);
        if (it == m_process_nodes.end()) {
            m_process_nodes.push_back(node);
            // Sort by priority (descending = earlier, so we sort with custom)
            std::sort(m_process_nodes.begin(), m_process_nodes.end(),
                [](Node* a, Node* b) { return a->get_process_priority() < b->get_process_priority(); });
        }
    }

    void unregister_process_node(Node* node) {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        auto it = std::find(m_process_nodes.begin(), m_process_nodes.end(), node);
        if (it != m_process_nodes.end()) {
            m_process_nodes.erase(it);
        }
    }

    void register_physics_process_node(Node* node) {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        auto it = std::find(m_physics_process_nodes.begin(), m_physics_process_nodes.end(), node);
        if (it == m_physics_process_nodes.end()) {
            m_physics_process_nodes.push_back(node);
            std::sort(m_physics_process_nodes.begin(), m_physics_process_nodes.end(),
                [](Node* a, Node* b) { return a->get_physics_process_priority() < b->get_physics_process_priority(); });
        }
    }

    void unregister_physics_process_node(Node* node) {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        auto it = std::find(m_physics_process_nodes.begin(), m_physics_process_nodes.end(), node);
        if (it != m_physics_process_nodes.end()) {
            m_physics_process_nodes.erase(it);
        }
    }

    // #########################################################################
    // Process groups (for parallel updates)
    // #########################################################################
    void add_to_process_group(const StringName& group, Node* node) {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        m_process_groups[group].push_back(node);
    }

    void remove_from_process_group(const StringName& group, Node* node) {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        auto it = m_process_groups.find(group);
        if (it != m_process_groups.end()) {
            auto& vec = it->second;
            vec.erase(std::remove(vec.begin(), vec.end(), node), vec.end());
        }
    }

    // #########################################################################
    // Main loop
    // #########################################################################
    void process_frame() {
        auto now = std::chrono::steady_clock::now();
        double delta = std::chrono::duration<double>(now - m_last_frame_time).count();
        m_last_frame_time = now;

        if (!m_paused) {
            delta *= m_time_scale;

            // Physics process (fixed timestep handled separately)
            // Process nodes (parallel by group or serial)
            {
                std::lock_guard<std::mutex> lock(m_queue_mutex);
                parallel::parallel_for(0, m_process_nodes.size(), [&](size_t i) {
                    Node* node = m_process_nodes[i];
                    if (node->is_inside_tree() && node->is_processing()) {
                        ProcessMode mode = node->get_effective_process_mode();
                        if (mode == ProcessMode::ALWAYS || (mode == ProcessMode::PAUSABLE && !m_paused)) {
                            node->process(delta);
                        }
                    }
                });
            }

            // Flush deferred calls
            flush_all_deferred();
        }

        // Process delete queue
        {
            std::lock_guard<std::mutex> lock(m_delete_mutex);
            for (Node* node : m_delete_queue) {
                delete node;
            }
            m_delete_queue.clear();
        }

        // Render
        if (m_root) {
            m_root->render();
        }
    }

    void flush_all_deferred() {
        // Traverse tree and flush deferred calls
        std::function<void(Node*)> flush_node = [&](Node* node) {
            node->flush_deferred_calls();
            for (Node* child : node->get_children()) {
                flush_node(child);
            }
        };
        if (m_root) flush_node(m_root);
    }

    // #########################################################################
    // Timing
    // #########################################################################
    void set_time_scale(double scale) { m_time_scale = scale; }
    double get_time_scale() const { return m_time_scale; }

    void set_pause(bool paused) { m_paused = paused; }
    bool is_paused() const { return m_paused; }

    // #########################################################################
    // Deletion queue
    // #########################################################################
    void queue_delete(Node* node) {
        std::lock_guard<std::mutex> lock(m_delete_mutex);
        m_delete_queue.push_back(node);
    }

    // #########################################################################
    // Multiplayer
    // #########################################################################
    void set_multiplayer_peer(int32_t peer) { m_multiplayer_peer = peer; }
    int32_t get_multiplayer_peer() const { return m_multiplayer_peer; }
    int32_t get_multiplayer_authority() const { return m_multiplayer_authority; }
    void set_multiplayer_authority(int32_t id) { m_multiplayer_authority = id; }
    bool is_multiplayer_server() const { return m_multiplayer_authority == 1; }

    // #########################################################################
    // Quit
    // #########################################################################
    void quit(int32_t exit_code = 0) {
        // Signal quit
        emit_signal("tree_exiting");
        // Cleanup
        if (m_root) {
            delete m_root;
            m_root = nullptr;
        }
        std::exit(exit_code);
    }

protected:
    void _change_scene(Node* new_scene) {
        if (m_current_scene) {
            m_root->remove_child(m_current_scene);
            m_current_scene->queue_free();
        }
        m_current_scene = new_scene;
        if (new_scene) {
            m_root->add_child(new_scene);
        }
    }
};

// #############################################################################
// Viewport - rendering surface
// #############################################################################
class Viewport : public Node {
    XTU_GODOT_REGISTER_CLASS(Viewport, Node)

private:
    graphics::framebuffer m_framebuffer;
    graphics::renderer m_renderer;
    graphics::camera m_camera;
    bool m_own_world = false;
    uint32_t m_render_target_update_mode = 0;

public:
    static StringName get_class_static() { return StringName("Viewport"); }

    void set_size(size_t width, size_t height) {
        m_framebuffer.resize(width, height);
        m_camera.set_aspect(static_cast<float>(width) / static_cast<float>(height));
    }

    graphics::renderer& get_renderer() { return m_renderer; }
    graphics::camera& get_camera() { return m_camera; }

    void render() {
        m_renderer.render();
        // Post-processing
        m_renderer.apply_tonemap_aces();
        m_renderer.apply_gamma_correction();
    }

    const graphics::framebuffer& get_framebuffer() const { return m_framebuffer; }
};

// #############################################################################
// Node2D - 2D spatial node
// #############################################################################
class Node2D : public Node {
    XTU_GODOT_REGISTER_CLASS(Node2D, Node)

private:
    vec2f m_position;
    float m_rotation = 0.0f;
    vec2f m_scale = {1.0f, 1.0f};
    float m_skew = 0.0f;
    mat3f m_transform;
    bool m_transform_dirty = true;

    void update_transform() {
        if (!m_transform_dirty) return;
        mat3f t = translate(mat3f::identity(), m_position);
        mat3f r = rotate_z(m_rotation);
        mat3f s = scale(mat3f::identity(), m_scale);
        m_transform = t * r * s;
        m_transform_dirty = false;
    }

public:
    static StringName get_class_static() { return StringName("Node2D"); }

    void set_position(const vec2f& pos) { m_position = pos; m_transform_dirty = true; }
    vec2f get_position() const { return m_position; }

    void set_rotation(float radians) { m_rotation = radians; m_transform_dirty = true; }
    float get_rotation() const { return m_rotation; }

    void set_scale(const vec2f& scale) { m_scale = scale; m_transform_dirty = true; }
    vec2f get_scale() const { return m_scale; }

    void set_skew(float skew) { m_skew = skew; m_transform_dirty = true; }
    float get_skew() const { return m_skew; }

    mat3f get_transform() { update_transform(); return m_transform; }
    vec2f get_global_position() const {
        if (Node2D* parent = dynamic_cast<Node2D*>(m_parent)) {
            return parent->get_transform() * vec3f(m_position.x(), m_position.y(), 1.0f);
        }
        return m_position;
    }

    void look_at(const vec2f& target) {
        vec2f dir = target - m_position;
        m_rotation = std::atan2(dir.y(), dir.x());
        m_transform_dirty = true;
    }

    void move_local_x(float delta, bool scaled = false) {
        vec2f axis(std::cos(m_rotation), std::sin(m_rotation));
        if (scaled) axis *= m_scale;
        m_position += axis * delta;
        m_transform_dirty = true;
    }

    void move_local_y(float delta, bool scaled = false) {
        vec2f axis(-std::sin(m_rotation), std::cos(m_rotation));
        if (scaled) axis *= m_scale;
        m_position += axis * delta;
        m_transform_dirty = true;
    }
};

// #############################################################################
// Node3D - 3D spatial node
// #############################################################################
class Node3D : public Node {
    XTU_GODOT_REGISTER_CLASS(Node3D, Node)

private:
    vec3f m_position;
    quatf m_rotation;
    vec3f m_scale = {1.0f, 1.0f, 1.0f};
    mat4f m_transform;
    bool m_transform_dirty = true;

    void update_transform() {
        if (!m_transform_dirty) return;
        mat4f t = translate(mat4f::identity(), m_position);
        mat4f r = rotate(m_rotation);
        mat4f s = scale(mat4f::identity(), m_scale);
        m_transform = t * r * s;
        m_transform_dirty = false;
    }

public:
    static StringName get_class_static() { return StringName("Node3D"); }

    void set_position(const vec3f& pos) { m_position = pos; m_transform_dirty = true; }
    vec3f get_position() const { return m_position; }

    void set_rotation(const quatf& rot) { m_rotation = rot; m_transform_dirty = true; }
    quatf get_rotation() const { return m_rotation; }

    void set_rotation_euler(const vec3f& euler) {
        m_rotation = quatf(vec3f(1,0,0), euler.x()) * quatf(vec3f(0,1,0), euler.y()) * quatf(vec3f(0,0,1), euler.z());
        m_transform_dirty = true;
    }

    vec3f get_rotation_euler() const {
        // Convert quaternion to Euler (simplified)
        float sinr_cosp = 2.0f * (m_rotation.w() * m_rotation.x() + m_rotation.y() * m_rotation.z());
        float cosr_cosp = 1.0f - 2.0f * (m_rotation.x() * m_rotation.x() + m_rotation.y() * m_rotation.y());
        float roll = std::atan2(sinr_cosp, cosr_cosp);
        float sinp = 2.0f * (m_rotation.w() * m_rotation.y() - m_rotation.z() * m_rotation.x());
        float pitch = std::abs(sinp) >= 1.0f ? std::copysign(M_PI_2, sinp) : std::asin(sinp);
        float siny_cosp = 2.0f * (m_rotation.w() * m_rotation.z() + m_rotation.x() * m_rotation.y());
        float cosy_cosp = 1.0f - 2.0f * (m_rotation.y() * m_rotation.y() + m_rotation.z() * m_rotation.z());
        float yaw = std::atan2(siny_cosp, cosy_cosp);
        return vec3f(roll, pitch, yaw);
    }

    void set_scale(const vec3f& scale) { m_scale = scale; m_transform_dirty = true; }
    vec3f get_scale() const { return m_scale; }

    mat4f get_transform() { update_transform(); return m_transform; }

    vec3f get_global_position() const {
        if (Node3D* parent = dynamic_cast<Node3D*>(m_parent)) {
            vec4f p = parent->get_transform() * vec4f(m_position.x(), m_position.y(), m_position.z(), 1.0f);
            return vec3f(p.x(), p.y(), p.z());
        }
        return m_position;
    }

    void look_at(const vec3f& target, const vec3f& up = {0,1,0}) {
        mat4f view = look_at(m_position, target, up);
        // Extract rotation from view matrix (inverse)
        mat3f rot;
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) rot[i][j] = view[i][j];
        // Convert to quaternion
        float trace = rot[0][0] + rot[1][1] + rot[2][2];
        if (trace > 0) {
            float s = 0.5f / std::sqrt(trace + 1.0f);
            m_rotation = quatf(0.25f / s,
                (rot[2][1] - rot[1][2]) * s,
                (rot[0][2] - rot[2][0]) * s,
                (rot[1][0] - rot[0][1]) * s);
        }
        m_transform_dirty = true;
    }

    void rotate_object_local(const vec3f& axis, float angle) {
        m_rotation = quatf(axis, angle) * m_rotation;
        m_transform_dirty = true;
    }

    void translate_object_local(const vec3f& offset) {
        vec3f local_offset = m_rotation.rotate(offset);
        m_position += local_offset;
        m_transform_dirty = true;
    }
};

// #############################################################################
// CanvasItem - base for 2D rendering nodes
// #############################################################################
class CanvasItem : public Node {
    XTU_GODOT_REGISTER_CLASS(CanvasItem, Node)

public:
    static StringName get_class_static() { return StringName("CanvasItem"); }

    virtual void _draw() {}
    void update() { queue_redraw(); }
    void queue_redraw() { m_dirty = true; }

protected:
    bool m_dirty = true;
    int32_t m_z_index = 0;
    bool m_z_as_relative = true;
    bool m_visible = true;
    vec4f m_modulate = {1,1,1,1};
    vec4f m_self_modulate = {1,1,1,1};
};

// #############################################################################
// Control - GUI node
// #############################################################################
class Control : public CanvasItem {
    XTU_GODOT_REGISTER_CLASS(Control, CanvasItem)

private:
    vec2f m_position;
    vec2f m_size;
    vec2f m_min_size;
    vec4f m_margin; // left, top, right, bottom
    bool m_clip_contents = false;
    Control* m_focus_owner = nullptr;

public:
    static StringName get_class_static() { return StringName("Control"); }

    void set_position(const vec2f& pos) { m_position = pos; }
    vec2f get_position() const { return m_position; }

    void set_size(const vec2f& size) { m_size = size; }
    vec2f get_size() const { return m_size; }

    void set_min_size(const vec2f& size) { m_min_size = size; }
    vec2f get_min_size() const { return m_min_size; }

    vec2f get_combined_min_size() const { return m_min_size; }

    void grab_focus() {
        if (m_focus_owner) m_focus_owner->release_focus();
        m_focus_owner = this;
    }
    void release_focus() { m_focus_owner = nullptr; }
    bool has_focus() const { return m_focus_owner == this; }
};

// #############################################################################
// AnimationPlayer - animation playback
// #############################################################################
class AnimationPlayer : public Node {
    XTU_GODOT_REGISTER_CLASS(AnimationPlayer, Node)

public:
    static StringName get_class_static() { return StringName("AnimationPlayer"); }

    void play(const StringName& name = StringName()) {}
    void stop(bool reset = true) {}
    bool is_playing() const { return false; }
    void seek(double time, bool update = false) {}
};

} // namespace godot

// Bring into main namespace
using godot::Node;
using godot::SceneTree;
using godot::Viewport;
using godot::Node2D;
using godot::Node3D;
using godot::CanvasItem;
using godot::Control;
using godot::AnimationPlayer;
using godot::NodePath;
using godot::ProcessMode;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XNODE_HPP