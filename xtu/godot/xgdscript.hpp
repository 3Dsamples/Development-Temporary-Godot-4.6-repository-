// godot/xgdscript.hpp

#ifndef XTENSOR_XGDSCRIPT_HPP
#define XTENSOR_XGDSCRIPT_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xrandom.hpp"
#include "../math/xsorting.hpp"
#include "../math/xnorm.hpp"
#include "../math/xoptimize.hpp"
#include "../math/xinterp.hpp"
#include "../math/xcluster.hpp"
#include "../io/xio_json.hpp"
#include "xvariant.hpp"
#include "xclassdb.hpp"
#include "xresource.hpp"
#include "xnode.hpp"

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
#include <random>
#include <complex>
#include <any>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/ref_counted.hpp>
    #include <godot_cpp/classes/resource.hpp>
    #include <godot_cpp/classes/object.hpp>
    #include <godot_cpp/classes/engine.hpp>
    #include <godot_cpp/core/method_bind.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/array.hpp>
    #include <godot_cpp/variant/dictionary.hpp>
    #include <godot_cpp/variant/packed_float32_array.hpp>
    #include <godot_cpp/variant/packed_float64_array.hpp>
    #include <godot_cpp/variant/packed_int32_array.hpp>
    #include <godot_cpp/variant/packed_int64_array.hpp>
    #include <godot_cpp/variant/packed_byte_array.hpp>
    #include <godot_cpp/variant/packed_vector2_array.hpp>
    #include <godot_cpp/variant/packed_vector3_array.hpp>
    #include <godot_cpp/variant/packed_color_array.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // GDScript Tensor API - Core Tensor Object
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class GDTensor : public godot::RefCounted
            {
                GDCLASS(GDTensor, godot::RefCounted)

            public:
                enum DType
                {
                    DTYPE_FLOAT32 = 0,
                    DTYPE_FLOAT64 = 1,
                    DTYPE_INT32 = 2,
                    DTYPE_INT64 = 3,
                    DTYPE_UINT8 = 4,
                    DTYPE_BOOL = 5,
                    DTYPE_COMPLEX64 = 6,
                    DTYPE_COMPLEX128 = 7
                };

            private:
                xarray_container<double> m_data; // internal storage as double for simplicity
                DType m_dtype = DTYPE_FLOAT64;
                bool m_requires_grad = false;
                godot::Ref<GDTensor> m_grad;
                std::function<void()> m_backward_fn;

            protected:
                static void _bind_methods()
                {
                    // Creation
                    godot::ClassDB::bind_static_method("GDTensor", godot::D_METHOD("zeros", "shape", "dtype"), &GDTensor::zeros, godot::DEFVAL(DTYPE_FLOAT64));
                    godot::ClassDB::bind_static_method("GDTensor", godot::D_METHOD("ones", "shape", "dtype"), &GDTensor::ones, godot::DEFVAL(DTYPE_FLOAT64));
                    godot::ClassDB::bind_static_method("GDTensor", godot::D_METHOD("full", "shape", "fill_value", "dtype"), &GDTensor::full, godot::DEFVAL(DTYPE_FLOAT64));
                    godot::ClassDB::bind_static_method("GDTensor", godot::D_METHOD("eye", "n", "dtype"), &GDTensor::eye, godot::DEFVAL(DTYPE_FLOAT64));
                    godot::ClassDB::bind_static_method("GDTensor", godot::D_METHOD("random_uniform", "shape", "min_val", "max_val", "dtype"), &GDTensor::random_uniform, godot::DEFVAL(0.0), godot::DEFVAL(1.0), godot::DEFVAL(DTYPE_FLOAT64));
                    godot::ClassDB::bind_static_method("GDTensor", godot::D_METHOD("random_normal", "shape", "mean", "stddev", "dtype"), &GDTensor::random_normal, godot::DEFVAL(0.0), godot::DEFVAL(1.0), godot::DEFVAL(DTYPE_FLOAT64));
                    godot::ClassDB::bind_static_method("GDTensor", godot::D_METHOD("linspace", "start", "stop", "num", "endpoint", "dtype"), &GDTensor::linspace, godot::DEFVAL(true), godot::DEFVAL(DTYPE_FLOAT64));
                    godot::ClassDB::bind_static_method("GDTensor", godot::D_METHOD("arange", "start", "stop", "step", "dtype"), &GDTensor::arange, godot::DEFVAL(1.0), godot::DEFVAL(DTYPE_FLOAT64));
                    godot::ClassDB::bind_static_method("GDTensor", godot::D_METHOD("from_array", "array", "dtype"), &GDTensor::from_array, godot::DEFVAL(DTYPE_FLOAT64));
                    godot::ClassDB::bind_static_method("GDTensor", godot::D_METHOD("from_packed", "packed_array", "dtype"), &GDTensor::from_packed);
                    
                    // Properties
                    godot::ClassDB::bind_method(godot::D_METHOD("get_shape"), &GDTensor::get_shape);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_size"), &GDTensor::get_size);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_ndim"), &GDTensor::get_ndim);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_dtype"), &GDTensor::get_dtype);
                    godot::ClassDB::bind_method(godot::D_METHOD("reshape", "new_shape"), &GDTensor::reshape);
                    godot::ClassDB::bind_method(godot::D_METHOD("transpose", "axes"), &GDTensor::transpose, godot::DEFVAL(godot::Array()));
                    godot::ClassDB::bind_method(godot::D_METHOD("flatten"), &GDTensor::flatten);
                    godot::ClassDB::bind_method(godot::D_METHOD("squeeze", "axis"), &GDTensor::squeeze, godot::DEFVAL(-1));
                    godot::ClassDB::bind_method(godot::D_METHOD("unsqueeze", "axis"), &GDTensor::unsqueeze);
                    
                    // Indexing
                    godot::ClassDB::bind_method(godot::D_METHOD("get_value", "indices"), &GDTensor::get_value);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_value", "indices", "value"), &GDTensor::set_value);
                    godot::ClassDB::bind_method(godot::D_METHOD("slice", "start_indices", "end_indices"), &GDTensor::slice);
                    godot::ClassDB::bind_method(godot::D_METHOD("take", "indices", "axis"), &GDTensor::take, godot::DEFVAL(0));
                    
                    // Arithmetic
                    godot::ClassDB::bind_method(godot::D_METHOD("add", "other"), &GDTensor::add);
                    godot::ClassDB::bind_method(godot::D_METHOD("sub", "other"), &GDTensor::sub);
                    godot::ClassDB::bind_method(godot::D_METHOD("mul", "other"), &GDTensor::mul);
                    godot::ClassDB::bind_method(godot::D_METHOD("div", "other"), &GDTensor::div);
                    godot::ClassDB::bind_method(godot::D_METHOD("matmul", "other"), &GDTensor::matmul);
                    godot::ClassDB::bind_method(godot::D_METHOD("pow", "exponent"), &GDTensor::pow);
                    godot::ClassDB::bind_method(godot::D_METHOD("neg"), &GDTensor::neg);
                    
                    // Reductions
                    godot::ClassDB::bind_method(godot::D_METHOD("sum", "axis", "keepdims"), &GDTensor::sum, godot::DEFVAL(-1), godot::DEFVAL(false));
                    godot::ClassDB::bind_method(godot::D_METHOD("mean", "axis", "keepdims"), &GDTensor::mean, godot::DEFVAL(-1), godot::DEFVAL(false));
                    godot::ClassDB::bind_method(godot::D_METHOD("max", "axis", "keepdims"), &GDTensor::max, godot::DEFVAL(-1), godot::DEFVAL(false));
                    godot::ClassDB::bind_method(godot::D_METHOD("min", "axis", "keepdims"), &GDTensor::min, godot::DEFVAL(-1), godot::DEFVAL(false));
                    godot::ClassDB::bind_method(godot::D_METHOD("std", "axis", "keepdims"), &GDTensor::std, godot::DEFVAL(-1), godot::DEFVAL(false));
                    godot::ClassDB::bind_method(godot::D_METHOD("var", "axis", "keepdims"), &GDTensor::var, godot::DEFVAL(-1), godot::DEFVAL(false));
                    godot::ClassDB::bind_method(godot::D_METHOD("norm", "ord"), &GDTensor::norm, godot::DEFVAL("l2"));
                    
                    // Element-wise math
                    godot::ClassDB::bind_method(godot::D_METHOD("abs"), &GDTensor::abs);
                    godot::ClassDB::bind_method(godot::D_METHOD("sqrt"), &GDTensor::sqrt);
                    godot::ClassDB::bind_method(godot::D_METHOD("exp"), &GDTensor::exp);
                    godot::ClassDB::bind_method(godot::D_METHOD("log"), &GDTensor::log);
                    godot::ClassDB::bind_method(godot::D_METHOD("sin"), &GDTensor::sin);
                    godot::ClassDB::bind_method(godot::D_METHOD("cos"), &GDTensor::cos);
                    godot::ClassDB::bind_method(godot::D_METHOD("tan"), &GDTensor::tan);
                    godot::ClassDB::bind_method(godot::D_METHOD("asin"), &GDTensor::asin);
                    godot::ClassDB::bind_method(godot::D_METHOD("acos"), &GDTensor::acos);
                    godot::ClassDB::bind_method(godot::D_METHOD("atan"), &GDTensor::atan);
                    godot::ClassDB::bind_method(godot::D_METHOD("sinh"), &GDTensor::sinh);
                    godot::ClassDB::bind_method(godot::D_METHOD("cosh"), &GDTensor::cosh);
                    godot::ClassDB::bind_method(godot::D_METHOD("tanh"), &GDTensor::tanh);
                    godot::ClassDB::bind_method(godot::D_METHOD("ceil"), &GDTensor::ceil);
                    godot::ClassDB::bind_method(godot::D_METHOD("floor"), &GDTensor::floor);
                    godot::ClassDB::bind_method(godot::D_METHOD("round"), &GDTensor::round);
                    godot::ClassDB::bind_method(godot::D_METHOD("clip", "min_val", "max_val"), &GDTensor::clip);
                    
                    // Linear algebra
                    godot::ClassDB::bind_method(godot::D_METHOD("inv"), &GDTensor::inv);
                    godot::ClassDB::bind_method(godot::D_METHOD("det"), &GDTensor::det);
                    godot::ClassDB::bind_method(godot::D_METHOD("solve", "b"), &GDTensor::solve);
                    godot::ClassDB::bind_method(godot::D_METHOD("qr"), &GDTensor::qr);
                    godot::ClassDB::bind_method(godot::D_METHOD("svd"), &GDTensor::svd);
                    godot::ClassDB::bind_method(godot::D_METHOD("eig"), &GDTensor::eig);
                    
                    // Statistics
                    godot::ClassDB::bind_method(godot::D_METHOD("median", "axis"), &GDTensor::median, godot::DEFVAL(-1));
                    godot::ClassDB::bind_method(godot::D_METHOD("quantile", "q", "axis"), &GDTensor::quantile, godot::DEFVAL(-1));
                    godot::ClassDB::bind_method(godot::D_METHOD("percentile", "p", "axis"), &GDTensor::percentile, godot::DEFVAL(-1));
                    godot::ClassDB::bind_method(godot::D_METHOD("argmax", "axis"), &GDTensor::argmax, godot::DEFVAL(-1));
                    godot::ClassDB::bind_method(godot::D_METHOD("argmin", "axis"), &GDTensor::argmin, godot::DEFVAL(-1));
                    godot::ClassDB::bind_method(godot::D_METHOD("sort", "axis", "descending"), &GDTensor::sort, godot::DEFVAL(-1), godot::DEFVAL(false));
                    godot::ClassDB::bind_method(godot::D_METHOD("argsort", "axis", "descending"), &GDTensor::argsort, godot::DEFVAL(-1), godot::DEFVAL(false));
                    
                    // Gradients
                    godot::ClassDB::bind_method(godot::D_METHOD("requires_grad", "requires"), &GDTensor::set_requires_grad);
                    godot::ClassDB::bind_method(godot::D_METHOD("is_requires_grad"), &GDTensor::is_requires_grad);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_grad"), &GDTensor::get_grad);
                    godot::ClassDB::bind_method(godot::D_METHOD("backward"), &GDTensor::backward);
                    godot::ClassDB::bind_method(godot::D_METHOD("zero_grad"), &GDTensor::zero_grad);
                    
                    // Conversion
                    godot::ClassDB::bind_method(godot::D_METHOD("to_array"), &GDTensor::to_array);
                    godot::ClassDB::bind_method(godot::D_METHOD("to_packed_float32"), &GDTensor::to_packed_float32);
                    godot::ClassDB::bind_method(godot::D_METHOD("to_packed_float64"), &GDTensor::to_packed_float64);
                    godot::ClassDB::bind_method(godot::D_METHOD("to_packed_int32"), &GDTensor::to_packed_int32);
                    godot::ClassDB::bind_method(godot::D_METHOD("to_packed_int64"), &GDTensor::to_packed_int64);
                    godot::ClassDB::bind_method(godot::D_METHOD("to_json"), &GDTensor::to_json);
                    godot::ClassDB::bind_method(godot::D_METHOD("copy"), &GDTensor::copy);
                    godot::ClassDB::bind_method(godot::D_METHOD("cast_to", "dtype"), &GDTensor::cast_to);
                    
                    // Comparison
                    godot::ClassDB::bind_method(godot::D_METHOD("equal", "other"), &GDTensor::equal);
                    godot::ClassDB::bind_method(godot::D_METHOD("not_equal", "other"), &GDTensor::not_equal);
                    godot::ClassDB::bind_method(godot::D_METHOD("greater", "other"), &GDTensor::greater);
                    godot::ClassDB::bind_method(godot::D_METHOD("greater_equal", "other"), &GDTensor::greater_equal);
                    godot::ClassDB::bind_method(godot::D_METHOD("less", "other"), &GDTensor::less);
                    godot::ClassDB::bind_method(godot::D_METHOD("less_equal", "other"), &GDTensor::less_equal);
                    
                    // Utility
                    godot::ClassDB::bind_method(godot::D_METHOD("__str__"), &GDTensor::_to_string);
                    
                    ADD_SIGNAL(godot::MethodInfo("data_changed"));
                }

            public:
                GDTensor() = default;
                
                explicit GDTensor(const xarray_container<double>& data, DType dtype = DTYPE_FLOAT64)
                    : m_data(data), m_dtype(dtype) {}
                
                explicit GDTensor(xarray_container<double>&& data, DType dtype = DTYPE_FLOAT64)
                    : m_data(std::move(data)), m_dtype(dtype) {}

                // Static factories
                static godot::Ref<GDTensor> zeros(const godot::PackedInt64Array& shape, DType dtype)
                {
                    auto sh = packed_to_shape(shape);
                    auto data = xt::zeros<double>(sh);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = std::move(data);
                    t->m_dtype = dtype;
                    return t;
                }

                static godot::Ref<GDTensor> ones(const godot::PackedInt64Array& shape, DType dtype)
                {
                    auto sh = packed_to_shape(shape);
                    auto data = xt::ones<double>(sh);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = std::move(data);
                    t->m_dtype = dtype;
                    return t;
                }

                static godot::Ref<GDTensor> full(const godot::PackedInt64Array& shape, double fill_value, DType dtype)
                {
                    auto sh = packed_to_shape(shape);
                    auto data = xt::full<double>(sh, fill_value);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = std::move(data);
                    t->m_dtype = dtype;
                    return t;
                }

                static godot::Ref<GDTensor> eye(int64_t n, DType dtype)
                {
                    auto data = xt::eye<double>(static_cast<size_t>(n));
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = std::move(data);
                    t->m_dtype = dtype;
                    return t;
                }

                static godot::Ref<GDTensor> random_uniform(const godot::PackedInt64Array& shape, double min_val, double max_val, DType dtype)
                {
                    auto sh = packed_to_shape(shape);
                    auto data = xt::random<double>(sh, min_val, max_val);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = std::move(data);
                    t->m_dtype = dtype;
                    return t;
                }

                static godot::Ref<GDTensor> random_normal(const godot::PackedInt64Array& shape, double mean, double stddev, DType dtype)
                {
                    auto sh = packed_to_shape(shape);
                    auto data = xt::randn<double>(sh, mean, stddev);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = std::move(data);
                    t->m_dtype = dtype;
                    return t;
                }

                static godot::Ref<GDTensor> linspace(double start, double stop, int64_t num, bool endpoint, DType dtype)
                {
                    auto data = xt::linspace<double>(start, stop, static_cast<size_t>(num), endpoint);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = std::move(data);
                    t->m_dtype = dtype;
                    return t;
                }

                static godot::Ref<GDTensor> arange(double start, double stop, double step, DType dtype)
                {
                    auto data = xt::arange<double>(start, stop, step);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = std::move(data);
                    t->m_dtype = dtype;
                    return t;
                }

                static godot::Ref<GDTensor> from_array(const godot::Array& arr, DType dtype)
                {
                    auto sh = infer_shape(arr);
                    auto data = xt::zeros<double>(sh);
                    fill_from_array(data, arr, {}, 0);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = std::move(data);
                    t->m_dtype = dtype;
                    return t;
                }

                static godot::Ref<GDTensor> from_packed(const godot::Variant& packed)
                {
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    if (packed.get_type() == godot::Variant::PACKED_FLOAT32_ARRAY)
                    {
                        godot::PackedFloat32Array arr = packed;
                        t->m_data = xarray_container<double>({static_cast<size_t>(arr.size())});
                        for (int i = 0; i < arr.size(); ++i)
                            t->m_data(i) = static_cast<double>(arr[i]);
                        t->m_dtype = DTYPE_FLOAT32;
                    }
                    else if (packed.get_type() == godot::Variant::PACKED_FLOAT64_ARRAY)
                    {
                        godot::PackedFloat64Array arr = packed;
                        t->m_data = xarray_container<double>({static_cast<size_t>(arr.size())});
                        for (int i = 0; i < arr.size(); ++i)
                            t->m_data(i) = arr[i];
                        t->m_dtype = DTYPE_FLOAT64;
                    }
                    else if (packed.get_type() == godot::Variant::PACKED_INT32_ARRAY)
                    {
                        godot::PackedInt32Array arr = packed;
                        t->m_data = xarray_container<double>({static_cast<size_t>(arr.size())});
                        for (int i = 0; i < arr.size(); ++i)
                            t->m_data(i) = static_cast<double>(arr[i]);
                        t->m_dtype = DTYPE_INT32;
                    }
                    else if (packed.get_type() == godot::Variant::PACKED_INT64_ARRAY)
                    {
                        godot::PackedInt64Array arr = packed;
                        t->m_data = xarray_container<double>({static_cast<size_t>(arr.size())});
                        for (int i = 0; i < arr.size(); ++i)
                            t->m_data(i) = static_cast<double>(arr[i]);
                        t->m_dtype = DTYPE_INT64;
                    }
                    else if (packed.get_type() == godot::Variant::PACKED_BYTE_ARRAY)
                    {
                        godot::PackedByteArray arr = packed;
                        t->m_data = xarray_container<double>({static_cast<size_t>(arr.size())});
                        for (int i = 0; i < arr.size(); ++i)
                            t->m_data(i) = static_cast<double>(arr[i]);
                        t->m_dtype = DTYPE_UINT8;
                    }
                    return t;
                }

                // Properties
                godot::PackedInt64Array get_shape() const
                {
                    godot::PackedInt64Array shape;
                    for (auto s : m_data.shape())
                        shape.append(static_cast<int64_t>(s));
                    return shape;
                }

                int64_t get_size() const { return static_cast<int64_t>(m_data.size()); }
                int64_t get_ndim() const { return static_cast<int64_t>(m_data.dimension()); }
                DType get_dtype() const { return m_dtype; }

                godot::Ref<GDTensor> reshape(const godot::PackedInt64Array& new_shape) const
                {
                    auto sh = packed_to_shape(new_shape);
                    auto data = xt::reshape_view(m_data, sh);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = data;
                    t->m_dtype = m_dtype;
                    return t;
                }

                godot::Ref<GDTensor> transpose(const godot::Array& axes) const
                {
                    xarray_container<double> result;
                    if (axes.is_empty())
                        result = xt::transpose(m_data);
                    else
                    {
                        std::vector<size_t> perm;
                        for (int i = 0; i < axes.size(); ++i)
                            perm.push_back(static_cast<size_t>(static_cast<int64_t>(axes[i])));
                        result = xt::transpose(m_data, perm);
                    }
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    t->m_dtype = m_dtype;
                    return t;
                }

                godot::Ref<GDTensor> flatten() const
                {
                    auto data = xt::flatten_view(m_data);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = data;
                    t->m_dtype = m_dtype;
                    return t;
                }

                godot::Ref<GDTensor> squeeze(int64_t axis) const
                {
                    xarray_container<double> result;
                    if (axis < 0)
                        result = xt::squeeze(m_data);
                    else
                        result = xt::squeeze(m_data, static_cast<size_t>(axis));
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    t->m_dtype = m_dtype;
                    return t;
                }

                godot::Ref<GDTensor> unsqueeze(int64_t axis) const
                {
                    auto result = xt::expand_dims(m_data, static_cast<size_t>(axis));
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    t->m_dtype = m_dtype;
                    return t;
                }

                // Indexing
                double get_value(const godot::PackedInt64Array& indices) const
                {
                    std::vector<size_t> idx;
                    for (int i = 0; i < indices.size(); ++i)
                        idx.push_back(static_cast<size_t>(indices[i]));
                    if (idx.size() == m_data.dimension())
                        return m_data.element(idx);
                    return 0.0;
                }

                void set_value(const godot::PackedInt64Array& indices, double value)
                {
                    std::vector<size_t> idx;
                    for (int i = 0; i < indices.size(); ++i)
                        idx.push_back(static_cast<size_t>(indices[i]));
                    if (idx.size() == m_data.dimension())
                    {
                        m_data.element(idx) = value;
                        emit_signal("data_changed");
                    }
                }

                godot::Ref<GDTensor> slice(const godot::PackedInt64Array& start, const godot::PackedInt64Array& end) const
                {
                    std::vector<xt::xrange<size_t>> ranges;
                    size_t ndim = m_data.dimension();
                    for (size_t i = 0; i < ndim; ++i)
                    {
                        size_t s = (i < start.size()) ? static_cast<size_t>(start[static_cast<int>(i)]) : 0;
                        size_t e = (i < end.size()) ? static_cast<size_t>(end[static_cast<int>(i)]) : m_data.shape()[i];
                        ranges.emplace_back(s, e);
                    }
                    auto sliced = xt::view(m_data, ranges[0]);
                    // For multi-dim, chain views (simplified)
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = sliced;
                    t->m_dtype = m_dtype;
                    return t;
                }

                godot::Ref<GDTensor> take(const godot::PackedInt64Array& indices, int64_t axis) const
                {
                    size_t ax = static_cast<size_t>(axis < 0 ? axis + m_data.dimension() : axis);
                    std::vector<size_t> idx;
                    for (int i = 0; i < indices.size(); ++i)
                        idx.push_back(static_cast<size_t>(indices[i]));
                    auto result = xt::take(m_data, idx, ax);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    t->m_dtype = m_dtype;
                    return t;
                }

                // Arithmetic
                godot::Ref<GDTensor> add(const godot::Variant& other) const
                {
                    return binary_op(other, [](double a, double b) { return a + b; });
                }

                godot::Ref<GDTensor> sub(const godot::Variant& other) const
                {
                    return binary_op(other, [](double a, double b) { return a - b; });
                }

                godot::Ref<GDTensor> mul(const godot::Variant& other) const
                {
                    return binary_op(other, [](double a, double b) { return a * b; });
                }

                godot::Ref<GDTensor> div(const godot::Variant& other) const
                {
                    return binary_op(other, [](double a, double b) { return b != 0.0 ? a / b : 0.0; });
                }

                godot::Ref<GDTensor> matmul(const godot::Ref<GDTensor>& other) const
                {
                    if (m_data.dimension() != 2 || other->m_data.dimension() != 2)
                    {
                        godot::UtilityFunctions::printerr("matmul requires 2D matrices");
                        return godot::Ref<GDTensor>();
                    }
                    auto result = xt::linalg::dot(m_data, other->m_data);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    t->m_dtype = m_dtype;
                    return t;
                }

                godot::Ref<GDTensor> pow(double exponent) const
                {
                    auto result = xt::pow(m_data, exponent);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    t->m_dtype = m_dtype;
                    return t;
                }

                godot::Ref<GDTensor> neg() const
                {
                    auto result = -m_data;
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    t->m_dtype = m_dtype;
                    return t;
                }

                // Reductions
                godot::Ref<GDTensor> sum(int64_t axis, bool keepdims) const
                {
                    return reduce_op(axis, keepdims, [](const auto& a, const auto& axes) { return xt::sum(a, axes); });
                }

                godot::Ref<GDTensor> mean(int64_t axis, bool keepdims) const
                {
                    return reduce_op(axis, keepdims, [](const auto& a, const auto& axes) { return xt::mean(a, axes); });
                }

                godot::Ref<GDTensor> max(int64_t axis, bool keepdims) const
                {
                    return reduce_op(axis, keepdims, [](const auto& a, const auto& axes) { return xt::amax(a, axes); });
                }

                godot::Ref<GDTensor> min(int64_t axis, bool keepdims) const
                {
                    return reduce_op(axis, keepdims, [](const auto& a, const auto& axes) { return xt::amin(a, axes); });
                }

                godot::Ref<GDTensor> std(int64_t axis, bool keepdims) const
                {
                    return reduce_op(axis, keepdims, [](const auto& a, const auto& axes) { return xt::stddev(a, axes); });
                }

                godot::Ref<GDTensor> var(int64_t axis, bool keepdims) const
                {
                    return reduce_op(axis, keepdims, [](const auto& a, const auto& axes) { return xt::variance(a, axes); });
                }

                double norm(const godot::String& ord) const
                {
                    std::string o = ord.utf8().get_data();
                    return xt::norm_dispatch(m_data, o);
                }

                // Element-wise math
                #define DEFINE_UNARY_OP(name, func) \
                    godot::Ref<GDTensor> name() const { \
                        auto result = func(m_data); \
                        godot::Ref<GDTensor> t; \
                        t.instantiate(); \
                        t->m_data = result; \
                        t->m_dtype = m_dtype; \
                        return t; \
                    }

                DEFINE_UNARY_OP(abs, xt::abs)
                DEFINE_UNARY_OP(sqrt, xt::sqrt)
                DEFINE_UNARY_OP(exp, xt::exp)
                DEFINE_UNARY_OP(log, xt::log)
                DEFINE_UNARY_OP(sin, xt::sin)
                DEFINE_UNARY_OP(cos, xt::cos)
                DEFINE_UNARY_OP(tan, xt::tan)
                DEFINE_UNARY_OP(asin, xt::asin)
                DEFINE_UNARY_OP(acos, xt::acos)
                DEFINE_UNARY_OP(atan, xt::atan)
                DEFINE_UNARY_OP(sinh, xt::sinh)
                DEFINE_UNARY_OP(cosh, xt::cosh)
                DEFINE_UNARY_OP(tanh, xt::tanh)
                DEFINE_UNARY_OP(ceil, xt::ceil)
                DEFINE_UNARY_OP(floor, xt::floor)
                DEFINE_UNARY_OP(round, xt::round)
                #undef DEFINE_UNARY_OP

                godot::Ref<GDTensor> clip(double min_val, double max_val) const
                {
                    auto result = xt::clip(m_data, min_val, max_val);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    t->m_dtype = m_dtype;
                    return t;
                }

                // Linear algebra
                godot::Ref<GDTensor> inv() const
                {
                    auto result = xt::linalg::inv(m_data);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    t->m_dtype = m_dtype;
                    return t;
                }

                double det() const { return xt::linalg::det(m_data); }

                godot::Ref<GDTensor> solve(const godot::Ref<GDTensor>& b) const
                {
                    auto result = xt::linalg::solve(m_data, b->m_data);
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    t->m_dtype = m_dtype;
                    return t;
                }

                godot::Array qr() const
                {
                    auto [Q, R] = xt::linalg::qr(m_data);
                    godot::Array arr;
                    godot::Ref<GDTensor> q_t, r_t;
                    q_t.instantiate(); q_t->m_data = Q; q_t->m_dtype = m_dtype;
                    r_t.instantiate(); r_t->m_data = R; r_t->m_dtype = m_dtype;
                    arr.append(q_t);
                    arr.append(r_t);
                    return arr;
                }

                godot::Array svd() const
                {
                    auto [U, S, Vt] = xt::linalg::svd(m_data);
                    godot::Array arr;
                    godot::Ref<GDTensor> u_t, s_t, v_t;
                    u_t.instantiate(); u_t->m_data = U; u_t->m_dtype = m_dtype;
                    s_t.instantiate(); s_t->m_data = S; s_t->m_dtype = m_dtype;
                    v_t.instantiate(); v_t->m_data = Vt; v_t->m_dtype = m_dtype;
                    arr.append(u_t);
                    arr.append(s_t);
                    arr.append(v_t);
                    return arr;
                }

                godot::Array eig() const
                {
                    auto [vals, vecs] = xt::linalg::eig(m_data);
                    godot::Array arr;
                    godot::Ref<GDTensor> v1, v2;
                    v1.instantiate(); v1->m_data = xt::real(vals); v1->m_dtype = m_dtype;
                    v2.instantiate(); v2->m_data = xt::real(vecs); v2->m_dtype = m_dtype;
                    arr.append(v1);
                    arr.append(v2);
                    return arr;
                }

                // Statistics
                godot::Ref<GDTensor> median(int64_t axis) const
                {
                    if (axis < 0)
                    {
                        double med = xt::median(m_data)();
                        auto result = xarray_container<double>({});
                        result() = med;
                        godot::Ref<GDTensor> t;
                        t.instantiate();
                        t->m_data = result;
                        return t;
                    }
                    auto result = xt::median(m_data, static_cast<size_t>(axis));
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    return t;
                }

                godot::Ref<GDTensor> quantile(double q, int64_t axis) const
                {
                    if (axis < 0)
                    {
                        double val = xt::quantile(m_data, q)();
                        auto result = xarray_container<double>({});
                        result() = val;
                        godot::Ref<GDTensor> t;
                        t.instantiate();
                        t->m_data = result;
                        return t;
                    }
                    auto result = xt::quantile(m_data, q, static_cast<size_t>(axis));
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    return t;
                }

                godot::Ref<GDTensor> percentile(double p, int64_t axis) const
                {
                    return quantile(p / 100.0, axis);
                }

                godot::Ref<GDTensor> argmax(int64_t axis) const
                {
                    if (axis < 0)
                    {
                        auto flat = xt::flatten_view(m_data);
                        size_t idx = static_cast<size_t>(xt::argmax(flat)());
                        auto result = xarray_container<double>({});
                        result() = static_cast<double>(idx);
                        godot::Ref<GDTensor> t;
                        t.instantiate();
                        t->m_data = result;
                        return t;
                    }
                    auto result = xt::argmax(m_data, static_cast<size_t>(axis));
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result.cast<double>();
                    return t;
                }

                godot::Ref<GDTensor> argmin(int64_t axis) const
                {
                    if (axis < 0)
                    {
                        auto flat = xt::flatten_view(m_data);
                        size_t idx = static_cast<size_t>(xt::argmin(flat)());
                        auto result = xarray_container<double>({});
                        result() = static_cast<double>(idx);
                        godot::Ref<GDTensor> t;
                        t.instantiate();
                        t->m_data = result;
                        return t;
                    }
                    auto result = xt::argmin(m_data, static_cast<size_t>(axis));
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result.cast<double>();
                    return t;
                }

                godot::Ref<GDTensor> sort(int64_t axis, bool descending) const
                {
                    if (axis < 0)
                    {
                        auto sorted = xt::sort_flattened(m_data);
                        if (descending) std::reverse(sorted.begin(), sorted.end());
                        godot::Ref<GDTensor> t;
                        t.instantiate();
                        t->m_data = sorted;
                        return t;
                    }
                    xarray_container<double> result = m_data;
                    xt::sort(result, static_cast<size_t>(axis));
                    if (descending)
                    {
                        // Reverse along axis (simplified)
                    }
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = result;
                    return t;
                }

                godot::Ref<GDTensor> argsort(int64_t axis, bool descending) const
                {
                    auto indices = xt::argsort(m_data, static_cast<size_t>(axis));
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = indices.cast<double>();
                    return t;
                }

                // Gradients
                void set_requires_grad(bool requires) { m_requires_grad = requires; }
                bool is_requires_grad() const { return m_requires_grad; }
                godot::Ref<GDTensor> get_grad() const { return m_grad; }
                
                void backward()
                {
                    if (!m_requires_grad) return;
                    if (!m_grad.is_valid())
                        m_grad = GDTensor::ones(get_shape(), m_dtype);
                    if (m_backward_fn)
                        m_backward_fn();
                }

                void zero_grad()
                {
                    if (m_grad.is_valid())
                        m_grad = nullptr;
                }

                // Conversion
                godot::Array to_array() const
                {
                    return tensor_to_array(m_data, 0);
                }

                godot::PackedFloat32Array to_packed_float32() const
                {
                    godot::PackedFloat32Array arr;
                    arr.resize(static_cast<int>(m_data.size()));
                    for (size_t i = 0; i < m_data.size(); ++i)
                        arr.set(static_cast<int>(i), static_cast<float>(m_data.flat(i)));
                    return arr;
                }

                godot::PackedFloat64Array to_packed_float64() const
                {
                    godot::PackedFloat64Array arr;
                    arr.resize(static_cast<int>(m_data.size()));
                    for (size_t i = 0; i < m_data.size(); ++i)
                        arr.set(static_cast<int>(i), m_data.flat(i));
                    return arr;
                }

                godot::PackedInt32Array to_packed_int32() const
                {
                    godot::PackedInt32Array arr;
                    arr.resize(static_cast<int>(m_data.size()));
                    for (size_t i = 0; i < m_data.size(); ++i)
                        arr.set(static_cast<int>(i), static_cast<int32_t>(m_data.flat(i)));
                    return arr;
                }

                godot::PackedInt64Array to_packed_int64() const
                {
                    godot::PackedInt64Array arr;
                    arr.resize(static_cast<int>(m_data.size()));
                    for (size_t i = 0; i < m_data.size(); ++i)
                        arr.set(static_cast<int>(i), static_cast<int64_t>(m_data.flat(i)));
                    return arr;
                }

                godot::String to_json() const
                {
                    std::string json = xt::to_json(m_data, true);
                    return godot::String(json.c_str());
                }

                godot::Ref<GDTensor> copy() const
                {
                    godot::Ref<GDTensor> t;
                    t.instantiate();
                    t->m_data = m_data;
                    t->m_dtype = m_dtype;
                    return t;
                }

                godot::Ref<GDTensor> cast_to(DType dtype) const
                {
                    godot::Ref<GDTensor> t = copy();
                    t->m_dtype = dtype;
                    return t;
                }

                // Comparison
                #define DEFINE_COMPARE_OP(name, op) \
                    godot::Ref<GDTensor> name(const godot::Variant& other) const { \
                        if (other.get_type() == godot::Variant::OBJECT) { \
                            auto t = godot::Object::cast_to<GDTensor>(other); \
                            if (t) { \
                                auto result = xt::eval(m_data op t->m_data); \
                                godot::Ref<GDTensor> out; \
                                out.instantiate(); \
                                out->m_data = result.cast<double>(); \
                                out->m_dtype = DTYPE_BOOL; \
                                return out; \
                            } \
                        } else if (other.get_type() == godot::Variant::FLOAT || other.get_type() == godot::Variant::INT) { \
                            double val = other; \
                            auto result = xt::eval(m_data op val); \
                            godot::Ref<GDTensor> out; \
                            out.instantiate(); \
                            out->m_data = result.cast<double>(); \
                            out->m_dtype = DTYPE_BOOL; \
                            return out; \
                        } \
                        return godot::Ref<GDTensor>(); \
                    }

                DEFINE_COMPARE_OP(equal, ==)
                DEFINE_COMPARE_OP(not_equal, !=)
                DEFINE_COMPARE_OP(greater, >)
                DEFINE_COMPARE_OP(greater_equal, >=)
                DEFINE_COMPARE_OP(less, <)
                DEFINE_COMPARE_OP(less_equal, <=)
                #undef DEFINE_COMPARE_OP

                godot::String _to_string() const
                {
                    std::ostringstream oss;
                    oss << "GDTensor(shape=[";
                    for (size_t i = 0; i < m_data.dimension(); ++i)
                    {
                        if (i > 0) oss << ", ";
                        oss << m_data.shape()[i];
                    }
                    oss << "], dtype=";
                    switch (m_dtype)
                    {
                        case DTYPE_FLOAT32: oss << "float32"; break;
                        case DTYPE_FLOAT64: oss << "float64"; break;
                        case DTYPE_INT32: oss << "int32"; break;
                        case DTYPE_INT64: oss << "int64"; break;
                        case DTYPE_UINT8: oss << "uint8"; break;
                        case DTYPE_BOOL: oss << "bool"; break;
                        default: oss << "unknown";
                    }
                    oss << ")";
                    return godot::String(oss.str().c_str());
                }

            private:
                static std::vector<size_t> packed_to_shape(const godot::PackedInt64Array& arr)
                {
                    std::vector<size_t> shape;
                    for (int i = 0; i < arr.size(); ++i)
                        shape.push_back(static_cast<size_t>(arr[i]));
                    return shape;
                }

                static std::vector<size_t> infer_shape(const godot::Array& arr)
                {
                    std::vector<size_t> shape;
                    godot::Array current = arr;
                    while (true)
                    {
                        shape.push_back(static_cast<size_t>(current.size()));
                        if (current.is_empty()) break;
                        godot::Variant first = current[0];
                        if (first.get_type() == godot::Variant::ARRAY)
                            current = first;
                        else
                            break;
                    }
                    return shape;
                }

                static void fill_from_array(xarray_container<double>& data, const godot::Array& arr,
                                            const std::vector<size_t>& idx, size_t dim)
                {
                    if (dim == data.dimension())
                    {
                        // Should not happen as we don't handle scalars here
                        return;
                    }
                    for (int i = 0; i < arr.size(); ++i)
                    {
                        std::vector<size_t> new_idx = idx;
                        new_idx.push_back(static_cast<size_t>(i));
                        godot::Variant elem = arr[i];
                        if (elem.get_type() == godot::Variant::ARRAY)
                        {
                            fill_from_array(data, elem, new_idx, dim + 1);
                        }
                        else
                        {
                            double val = elem;
                            data.element(new_idx) = val;
                        }
                    }
                }

                godot::Array tensor_to_array(const xarray_container<double>& data, size_t dim) const
                {
                    godot::Array arr;
                    if (dim == data.dimension() - 1)
                    {
                        size_t size = data.shape()[dim];
                        for (size_t i = 0; i < size; ++i)
                        {
                            // Need to extract scalar - simplified
                            arr.append(0.0);
                        }
                    }
                    else
                    {
                        size_t size = data.shape()[dim];
                        for (size_t i = 0; i < size; ++i)
                        {
                            auto slice = xt::view(data, i, xt::ellipsis());
                            arr.append(tensor_to_array(slice, dim + 1));
                        }
                    }
                    return arr;
                }

                godot::Ref<GDTensor> binary_op(const godot::Variant& other,
                                               std::function<double(double,double)> op) const
                {
                    godot::Ref<GDTensor> result;
                    result.instantiate();
                    if (other.get_type() == godot::Variant::OBJECT)
                    {
                        godot::Ref<GDTensor> t = other;
                        if (t.is_valid())
                        {
                            result->m_data = xt::eval(xt::make_xfunction(op, m_data, t->m_data));
                        }
                    }
                    else if (other.get_type() == godot::Variant::FLOAT || other.get_type() == godot::Variant::INT)
                    {
                        double val = other;
                        result->m_data = xt::eval(xt::make_xfunction(
                            [op, val](double x) { return op(x, val); }, m_data));
                    }
                    result->m_dtype = m_dtype;
                    return result;
                }

                template<typename Func>
                godot::Ref<GDTensor> reduce_op(int64_t axis, bool keepdims, Func&& func) const
                {
                    godot::Ref<GDTensor> result;
                    result.instantiate();
                    if (axis < 0)
                    {
                        auto reduced = func(m_data, std::vector<size_t>{});
                        if (!keepdims)
                        {
                            xarray_container<double> scalar({});
                            scalar() = reduced();
                            result->m_data = scalar;
                        }
                        else
                        {
                            result->m_data = reduced;
                        }
                    }
                    else
                    {
                        std::vector<size_t> axes = {static_cast<size_t>(axis)};
                        auto reduced = func(m_data, axes);
                        if (!keepdims)
                            result->m_data = reduced;
                        else
                            result->m_data = xt::keep_dims_shape(reduced, axes);
                    }
                    result->m_dtype = m_dtype;
                    return result;
                }

                const xarray_container<double>& data() const { return m_data; }
                xarray_container<double>& data() { return m_data; }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // GDScript Module Initialization
            // --------------------------------------------------------------------
            class XGDScriptModule
            {
            public:
                static void register_types()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<GDTensor>();
#endif
                }

                static void unregister_types()
                {
                }
            };

        } // namespace godot_bridge

        using godot_bridge::GDTensor;
        using godot_bridge::XGDScriptModule;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XGDSCRIPT_HPP

// godot/xgdscript.hpp