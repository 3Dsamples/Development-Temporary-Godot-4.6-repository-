// godot/xresource.hpp

#ifndef XTENSOR_XRESOURCE_HPP
#define XTENSOR_XRESOURCE_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../io/xnpz.hpp"
#include "../io/xio_json.hpp"
#include "../io/ximage.hpp"
#include "xvariant.hpp"
#include "xclassdb.hpp"

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
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <filesystem>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/resource.hpp>
    #include <godot_cpp/classes/resource_loader.hpp>
    #include <godot_cpp/classes/resource_saver.hpp>
    #include <godot_cpp/classes/file_access.hpp>
    #include <godot_cpp/classes/dir_access.hpp>
    #include <godot_cpp/classes/engine.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // XTensorResource - Base Godot Resource for xtensor containers
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XTensorResource : public godot::Resource
            {
                GDCLASS(XTensorResource, godot::Resource)

            public:
                // Resource format identifiers
                enum Format
                {
                    FORMAT_NPY = 0,
                    FORMAT_NPZ = 1,
                    FORMAT_JSON = 2,
                    FORMAT_HDF5 = 3,
                    FORMAT_CUSTOM = 4
                };

            private:
                XVariant m_data;
                std::string m_name;
                std::string m_description;
                godot::Dictionary m_metadata;
                Format m_preferred_format = FORMAT_NPZ;
                int m_compression_level = 6;
                bool m_compressed = true;
                uint64_t m_version = 1;
                uint64_t m_created_timestamp = 0;
                uint64_t m_modified_timestamp = 0;
                std::string m_source_file;

            protected:
                static void _bind_methods()
                {
                    // Core data methods
                    godot::ClassDB::bind_method(godot::D_METHOD("set_data", "data"), &XTensorResource::set_data);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_data"), &XTensorResource::get_data);
                    
                    // Shape and info
                    godot::ClassDB::bind_method(godot::D_METHOD("get_shape"), &XTensorResource::get_shape);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_size"), &XTensorResource::get_size);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_dimension"), &XTensorResource::get_dimension);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_dtype"), &XTensorResource::get_dtype);
                    
                    // Metadata
                    godot::ClassDB::bind_method(godot::D_METHOD("set_metadata", "key", "value"), &XTensorResource::set_metadata);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_metadata", "key", "default"), &XTensorResource::get_metadata, godot::DEFVAL(godot::Variant()));
                    godot::ClassDB::bind_method(godot::D_METHOD("has_metadata", "key"), &XTensorResource::has_metadata);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_metadata_keys"), &XTensorResource::get_metadata_keys);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear_metadata"), &XTensorResource::clear_metadata);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_metadata_dict"), &XTensorResource::get_metadata_dict);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_metadata_dict", "dict"), &XTensorResource::set_metadata_dict);
                    
                    // Serialization
                    godot::ClassDB::bind_method(godot::D_METHOD("save", "path", "format"), &XTensorResource::save, godot::DEFVAL(FORMAT_NPZ));
                    godot::ClassDB::bind_method(godot::D_METHOD("load", "path", "format"), &XTensorResource::load, godot::DEFVAL(FORMAT_NPZ));
                    godot::ClassDB::bind_method(godot::D_METHOD("save_to_file", "path"), &XTensorResource::save_to_file);
                    godot::ClassDB::bind_method(godot::D_METHOD("load_from_file", "path"), &XTensorResource::load_from_file);
                    godot::ClassDB::bind_method(godot::D_METHOD("to_bytes"), &XTensorResource::to_bytes);
                    godot::ClassDB::bind_method(godot::D_METHOD("from_bytes", "bytes"), &XTensorResource::from_bytes);
                    
                    // Format options
                    godot::ClassDB::bind_method(godot::D_METHOD("set_preferred_format", "format"), &XTensorResource::set_preferred_format);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_preferred_format"), &XTensorResource::get_preferred_format);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_compression", "enabled"), &XTensorResource::set_compression);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_compression"), &XTensorResource::get_compression);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_compression_level", "level"), &XTensorResource::set_compression_level);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_compression_level"), &XTensorResource::get_compression_level);
                    
                    // Resource properties
                    godot::ClassDB::bind_method(godot::D_METHOD("set_name", "name"), &XTensorResource::set_name);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_name"), &XTensorResource::get_name);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_description", "desc"), &XTensorResource::set_description);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_description"), &XTensorResource::get_description);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_version"), &XTensorResource::get_version);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_created_timestamp"), &XTensorResource::get_created_timestamp);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_modified_timestamp"), &XTensorResource::get_modified_timestamp);
                    
                    // Operations
                    godot::ClassDB::bind_method(godot::D_METHOD("duplicate_resource"), &XTensorResource::duplicate_resource);
                    godot::ClassDB::bind_method(godot::D_METHOD("copy_from", "other"), &XTensorResource::copy_from);
                    
                    // Static factory methods
                    godot::ClassDB::bind_static_method("XTensorResource", godot::D_METHOD("zeros", "shape"), &XTensorResource::zeros);
                    godot::ClassDB::bind_static_method("XTensorResource", godot::D_METHOD("ones", "shape"), &XTensorResource::ones);
                    godot::ClassDB::bind_static_method("XTensorResource", godot::D_METHOD("random", "shape"), &XTensorResource::random);
                    godot::ClassDB::bind_static_method("XTensorResource", godot::D_METHOD("from_array", "array"), &XTensorResource::from_array);
                    godot::ClassDB::bind_static_method("XTensorResource", godot::D_METHOD("from_image", "image"), &XTensorResource::from_image);
                    godot::ClassDB::bind_static_method("XTensorResource", godot::D_METHOD("load_resource", "path"), &XTensorResource::load_resource);
                    
                    // Properties
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "data", godot::PROPERTY_HINT_RESOURCE_TYPE, "XVariant"), "set_data", "get_data");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "name"), "set_name", "get_name");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "description", godot::PROPERTY_HINT_MULTILINE_TEXT), "set_description", "get_description");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::DICTIONARY, "metadata"), "set_metadata_dict", "get_metadata_dict");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "preferred_format", godot::PROPERTY_HINT_ENUM, "NPY,NPZ,JSON,HDF5"), "set_preferred_format", "get_preferred_format");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "compressed"), "set_compression", "get_compression");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "compression_level", godot::PROPERTY_HINT_RANGE, "1,9,1"), "set_compression_level", "get_compression_level");
                    
                    // Signals
                    ADD_SIGNAL(godot::MethodInfo("data_changed"));
                    ADD_SIGNAL(godot::MethodInfo("metadata_changed", godot::PropertyInfo(godot::Variant::STRING, "key")));
                    
                    // Enum constants
                    BIND_ENUM_CONSTANT(FORMAT_NPY);
                    BIND_ENUM_CONSTANT(FORMAT_NPZ);
                    BIND_ENUM_CONSTANT(FORMAT_JSON);
                    BIND_ENUM_CONSTANT(FORMAT_HDF5);
                    BIND_ENUM_CONSTANT(FORMAT_CUSTOM);
                }

            public:
                XTensorResource()
                {
                    auto now = std::chrono::system_clock::now();
                    m_created_timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                        now.time_since_epoch()).count();
                    m_modified_timestamp = m_created_timestamp;
                }

                ~XTensorResource() = default;

                // Data access
                void set_data(const godot::Variant& data)
                {
                    m_data = XVariant(data);
                    m_modified_timestamp = get_current_timestamp();
                    emit_signal("data_changed");
                }

                godot::Variant get_data() const
                {
                    return m_data.variant();
                }

                godot::PackedInt64Array get_shape() const
                {
                    auto arr = m_data.to_double_array();
                    godot::PackedInt64Array shape;
                    for (auto s : arr.shape())
                        shape.append(static_cast<int64_t>(s));
                    return shape;
                }

                int64_t get_size() const
                {
                    return static_cast<int64_t>(m_data.to_double_array().size());
                }

                int64_t get_dimension() const
                {
                    return static_cast<int64_t>(m_data.to_double_array().dimension());
                }

                godot::String get_dtype() const
                {
                    return "float64"; // Default; could be extended
                }

                // Metadata
                void set_metadata(const godot::String& key, const godot::Variant& value)
                {
                    std::string k = key.utf8().get_data();
                    m_metadata[key] = value;
                    m_modified_timestamp = get_current_timestamp();
                    emit_signal("metadata_changed", key);
                }

                godot::Variant get_metadata(const godot::String& key, const godot::Variant& default_val) const
                {
                    std::string k = key.utf8().get_data();
                    if (m_metadata.has(key))
                        return m_metadata[key];
                    return default_val;
                }

                bool has_metadata(const godot::String& key) const
                {
                    return m_metadata.has(key);
                }

                godot::PackedStringArray get_metadata_keys() const
                {
                    godot::PackedStringArray keys;
                    godot::Array gd_keys = m_metadata.keys();
                    for (int i = 0; i < gd_keys.size(); ++i)
                        keys.append(gd_keys[i]);
                    return keys;
                }

                void clear_metadata()
                {
                    m_metadata.clear();
                    m_modified_timestamp = get_current_timestamp();
                }

                void set_metadata_dict(const godot::Dictionary& dict)
                {
                    m_metadata = dict;
                    m_modified_timestamp = get_current_timestamp();
                }

                godot::Dictionary get_metadata_dict() const
                {
                    return m_metadata;
                }

                // Serialization
                bool save(const godot::String& path, Format format)
                {
                    std::string p = path.utf8().get_data();
                    auto arr = m_data.to_double_array();
                    
                    try
                    {
                        switch (format)
                        {
                            case FORMAT_NPY:
                                save_npy(p, arr);
                                break;
                            case FORMAT_NPZ:
                                {
                                    std::map<std::string, xarray_container<double>> dict;
                                    dict["data"] = arr;
                                    // Also save metadata as separate arrays if simple
                                    save_npz(p, dict, m_compressed, m_compression_level);
                                }
                                break;
                            case FORMAT_JSON:
                                {
                                    std::map<std::string, xarray_container<double>> dict;
                                    dict["data"] = arr;
                                    // Convert metadata to a dict of arrays
                                    JsonArchive ar;
                                    ar.add_array("data", arr);
                                    ar.save(p);
                                }
                                break;
                            case FORMAT_HDF5:
                                {
                                    HDF5File file(p, HDF5File::Create);
                                    file.write_dataset("data", arr);
                                    // Save metadata as attributes
                                    godot::Array keys = m_metadata.keys();
                                    for (int i = 0; i < keys.size(); ++i)
                                    {
                                        std::string k = godot::String(keys[i]).utf8().get_data();
                                        godot::Variant v = m_metadata[keys[i]];
                                        if (v.get_type() == godot::Variant::STRING)
                                            file.set_attribute(k, godot::String(v).utf8().get_data());
                                        else if (v.get_type() == godot::Variant::FLOAT)
                                            file.set_attribute(k, static_cast<double>(v));
                                        else if (v.get_type() == godot::Variant::INT)
                                            file.set_attribute(k, static_cast<int64_t>(v));
                                    }
                                }
                                break;
                            default:
                                return false;
                        }
                        m_source_file = p;
                        return true;
                    }
                    catch (const std::exception& e)
                    {
                        godot::UtilityFunctions::printerr("Save failed: ", e.what());
                        return false;
                    }
                }

                bool load(const godot::String& path, Format format)
                {
                    std::string p = path.utf8().get_data();
                    
                    try
                    {
                        xarray_container<double> arr;
                        switch (format)
                        {
                            case FORMAT_NPY:
                                arr = load_npy<double>(p);
                                break;
                            case FORMAT_NPZ:
                                {
                                    auto npz = load_npz(p);
                                    if (npz.find("data") != npz.end())
                                        arr = npz["data"];
                                    else if (!npz.empty())
                                        arr = npz.begin()->second;
                                }
                                break;
                            case FORMAT_JSON:
                                {
                                    JsonArchive ar;
                                    ar.load(p);
                                    if (ar.list_arrays().size() > 0)
                                        arr = ar.get_array(ar.list_arrays()[0]);
                                }
                                break;
                            case FORMAT_HDF5:
                                {
                                    HDF5File file(p, HDF5File::ReadOnly);
                                    if (file.dataset_exists("/data"))
                                        arr = file.read_dataset<double>("/data");
                                    else
                                    {
                                        auto all = file.read_all_datasets();
                                        if (!all.empty())
                                            arr = all.begin()->second;
                                    }
                                }
                                break;
                            default:
                                return false;
                        }
                        m_data = XVariant::from_xarray(arr);
                        m_source_file = p;
                        m_modified_timestamp = get_current_timestamp();
                        emit_signal("data_changed");
                        return true;
                    }
                    catch (const std::exception& e)
                    {
                        godot::UtilityFunctions::printerr("Load failed: ", e.what());
                        return false;
                    }
                }

                bool save_to_file(const godot::String& path)
                {
                    return save(path, m_preferred_format);
                }

                bool load_from_file(const godot::String& path)
                {
                    // Auto-detect format from extension
                    std::string p = path.utf8().get_data();
                    Format fmt = FORMAT_NPZ;
                    if (p.size() > 4)
                    {
                        std::string ext = p.substr(p.size() - 4);
                        if (ext == ".npy") fmt = FORMAT_NPY;
                        else if (ext == ".npz") fmt = FORMAT_NPZ;
                        else if (ext == ".json") fmt = FORMAT_JSON;
                        else if (ext == ".h5" || ext == ".hdf5") fmt = FORMAT_HDF5;
                    }
                    return load(path, fmt);
                }

                godot::PackedByteArray to_bytes() const
                {
                    // Serialize to NPZ in memory
                    std::ostringstream oss;
                    std::map<std::string, xarray_container<double>> dict;
                    dict["data"] = m_data.to_double_array();
                    // Not directly supported; we'll use JSON for simplicity
                    JsonArchive ar;
                    ar.add_array("data", m_data.to_double_array());
                    std::string json_str = ar.to_json();
                    godot::PackedByteArray bytes;
                    bytes.resize(static_cast<int>(json_str.size()));
                    std::memcpy(bytes.ptrw(), json_str.data(), json_str.size());
                    return bytes;
                }

                void from_bytes(const godot::PackedByteArray& bytes)
                {
                    std::string json_str(reinterpret_cast<const char*>(bytes.ptr()), bytes.size());
                    JsonArchive ar;
                    ar.from_json(json_str);
                    if (ar.list_arrays().size() > 0)
                    {
                        auto arr = ar.get_array(ar.list_arrays()[0]);
                        m_data = XVariant::from_xarray(arr);
                        m_modified_timestamp = get_current_timestamp();
                        emit_signal("data_changed");
                    }
                }

                // Properties
                void set_preferred_format(Format fmt) { m_preferred_format = fmt; }
                Format get_preferred_format() const { return m_preferred_format; }
                void set_compression(bool enabled) { m_compressed = enabled; }
                bool get_compression() const { return m_compressed; }
                void set_compression_level(int level) { m_compression_level = std::clamp(level, 1, 9); }
                int get_compression_level() const { return m_compression_level; }

                void set_name(const godot::String& name) { m_name = name.utf8().get_data(); }
                godot::String get_name() const { return godot::String(m_name.c_str()); }
                void set_description(const godot::String& desc) { m_description = desc.utf8().get_data(); }
                godot::String get_description() const { return godot::String(m_description.c_str()); }

                int64_t get_version() const { return static_cast<int64_t>(m_version); }
                int64_t get_created_timestamp() const { return static_cast<int64_t>(m_created_timestamp); }
                int64_t get_modified_timestamp() const { return static_cast<int64_t>(m_modified_timestamp); }

                godot::Ref<XTensorResource> duplicate_resource() const
                {
                    godot::Ref<XTensorResource> dup;
                    dup.instantiate();
                    dup->set_data(m_data.variant());
                    dup->set_metadata_dict(m_metadata);
                    dup->set_name(get_name());
                    dup->set_description(get_description());
                    dup->set_preferred_format(m_preferred_format);
                    dup->set_compression(m_compressed);
                    dup->set_compression_level(m_compression_level);
                    return dup;
                }

                void copy_from(const godot::Ref<XTensorResource>& other)
                {
                    if (other.is_valid())
                    {
                        m_data = other->m_data;
                        m_metadata = other->m_metadata.duplicate();
                        m_name = other->m_name;
                        m_description = other->m_description;
                        m_preferred_format = other->m_preferred_format;
                        m_compressed = other->m_compressed;
                        m_compression_level = other->m_compression_level;
                        m_modified_timestamp = get_current_timestamp();
                        emit_signal("data_changed");
                    }
                }

                // Static factory methods
                static godot::Ref<XTensorResource> zeros(const godot::PackedInt64Array& shape)
                {
                    godot::Ref<XTensorResource> res;
                    res.instantiate();
                    std::vector<size_t> sh;
                    for (int i = 0; i < shape.size(); ++i)
                        sh.push_back(static_cast<size_t>(shape[i]));
                    auto arr = xt::zeros<double>(sh);
                    res->set_data(XVariant::from_xarray(arr).variant());
                    return res;
                }

                static godot::Ref<XTensorResource> ones(const godot::PackedInt64Array& shape)
                {
                    godot::Ref<XTensorResource> res;
                    res.instantiate();
                    std::vector<size_t> sh;
                    for (int i = 0; i < shape.size(); ++i)
                        sh.push_back(static_cast<size_t>(shape[i]));
                    auto arr = xt::ones<double>(sh);
                    res->set_data(XVariant::from_xarray(arr).variant());
                    return res;
                }

                static godot::Ref<XTensorResource> random(const godot::PackedInt64Array& shape)
                {
                    godot::Ref<XTensorResource> res;
                    res.instantiate();
                    std::vector<size_t> sh;
                    for (int i = 0; i < shape.size(); ++i)
                        sh.push_back(static_cast<size_t>(shape[i]));
                    auto arr = xt::random<double>(sh);
                    res->set_data(XVariant::from_xarray(arr).variant());
                    return res;
                }

                static godot::Ref<XTensorResource> from_array(const godot::Variant& array)
                {
                    godot::Ref<XTensorResource> res;
                    res.instantiate();
                    res->set_data(array);
                    return res;
                }

                static godot::Ref<XTensorResource> from_image(const godot::Ref<godot::Image>& image)
                {
                    godot::Ref<XTensorResource> res;
                    res.instantiate();
                    if (image.is_valid())
                    {
                        godot::PackedByteArray img_data = image->get_data();
                        int w = image->get_width();
                        int h = image->get_height();
                        bool has_mipmaps = image->has_mipmaps();
                        godot::Image::Format fmt = image->get_format();
                        int channels = 3;
                        if (fmt == godot::Image::FORMAT_RGBA8)
                            channels = 4;
                        xarray_container<double> arr({static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(channels)});
                        // Convert bytes to double (normalized 0-1)
                        for (int y = 0; y < h; ++y)
                        {
                            for (int x = 0; x < w; ++x)
                            {
                                godot::Color c = image->get_pixel(x, y);
                                arr(y, x, 0) = c.r;
                                arr(y, x, 1) = c.g;
                                arr(y, x, 2) = c.b;
                                if (channels == 4)
                                    arr(y, x, 3) = c.a;
                            }
                        }
                        res->set_data(XVariant::from_xarray(arr).variant());
                    }
                    return res;
                }

                static godot::Ref<XTensorResource> load_resource(const godot::String& path)
                {
                    godot::Ref<XTensorResource> res;
                    res.instantiate();
                    res->load_from_file(path);
                    return res;
                }

                // Convert to Godot Image (if appropriate)
                godot::Ref<godot::Image> to_image() const
                {
                    auto arr = m_data.to_double_array();
                    if (arr.dimension() != 3)
                    {
                        godot::UtilityFunctions::printerr("Tensor must be 3D (HxWxC) to convert to image");
                        return godot::Ref<godot::Image>();
                    }
                    size_t h = arr.shape()[0];
                    size_t w = arr.shape()[1];
                    size_t c = arr.shape()[2];
                    godot::Image::Format fmt;
                    if (c == 1) fmt = godot::Image::FORMAT_L8;
                    else if (c == 3) fmt = godot::Image::FORMAT_RGB8;
                    else if (c == 4) fmt = godot::Image::FORMAT_RGBA8;
                    else
                    {
                        godot::UtilityFunctions::printerr("Unsupported channel count");
                        return godot::Ref<godot::Image>();
                    }
                    godot::Ref<godot::Image> img;
                    img.instantiate();
                    img->create(static_cast<int>(w), static_cast<int>(h), false, fmt);
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            if (c == 1)
                            {
                                float v = static_cast<float>(arr(y, x, 0));
                                img->set_pixel(static_cast<int>(x), static_cast<int>(y), godot::Color(v, v, v));
                            }
                            else if (c == 3)
                            {
                                float r = static_cast<float>(std::clamp(arr(y, x, 0), 0.0, 1.0));
                                float g = static_cast<float>(std::clamp(arr(y, x, 1), 0.0, 1.0));
                                float b = static_cast<float>(std::clamp(arr(y, x, 2), 0.0, 1.0));
                                img->set_pixel(static_cast<int>(x), static_cast<int>(y), godot::Color(r, g, b));
                            }
                            else
                            {
                                float r = static_cast<float>(std::clamp(arr(y, x, 0), 0.0, 1.0));
                                float g = static_cast<float>(std::clamp(arr(y, x, 1), 0.0, 1.0));
                                float b = static_cast<float>(std::clamp(arr(y, x, 2), 0.0, 1.0));
                                float a = static_cast<float>(std::clamp(arr(y, x, 3), 0.0, 1.0));
                                img->set_pixel(static_cast<int>(x), static_cast<int>(y), godot::Color(r, g, b, a));
                            }
                        }
                    }
                    return img;
                }

            private:
                uint64_t get_current_timestamp() const
                {
                    auto now = std::chrono::system_clock::now();
                    return std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Resource Loader and Saver registration (for Godot's import system)
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XTensorResourceLoader : public godot::ResourceFormatLoader
            {
                GDCLASS(XTensorResourceLoader, godot::ResourceFormatLoader)

            protected:
                static void _bind_methods() {}

            public:
                virtual godot::PackedStringArray _get_recognized_extensions() const override
                {
                    godot::PackedStringArray arr;
                    arr.append("npy");
                    arr.append("npz");
                    return arr;
                }

                virtual bool _recognize_path(const godot::String& path, const godot::StringName& type) const override
                {
                    std::string p = path.utf8().get_data();
                    return p.size() > 4 && (p.substr(p.size()-4) == ".npy" || p.substr(p.size()-4) == ".npz");
                }

                virtual bool _handles_type(const godot::StringName& type) const override
                {
                    return type == godot::StringName("XTensorResource");
                }

                virtual godot::String _get_resource_type(const godot::String& path) const override
                {
                    return "XTensorResource";
                }

                virtual godot::Variant _load(const godot::String& path, const godot::String& original_path, bool use_sub_threads, int32_t cache_mode) const override
                {
                    godot::Ref<XTensorResource> res;
                    res.instantiate();
                    if (res->load_from_file(path))
                        return res;
                    return godot::Variant();
                }
            };

            class XTensorResourceSaver : public godot::ResourceFormatSaver
            {
                GDCLASS(XTensorResourceSaver, godot::ResourceFormatSaver)

            protected:
                static void _bind_methods() {}

            public:
                virtual godot::PackedStringArray _get_recognized_extensions(const godot::Ref<godot::Resource>& resource) const override
                {
                    godot::PackedStringArray arr;
                    if (resource->is_class("XTensorResource"))
                    {
                        arr.append("npy");
                        arr.append("npz");
                    }
                    return arr;
                }

                virtual bool _recognize(const godot::Ref<godot::Resource>& resource) const override
                {
                    return resource->is_class("XTensorResource");
                }

                virtual int32_t _save(const godot::Ref<godot::Resource>& resource, const godot::String& path, uint32_t flags) override
                {
                    godot::Ref<XTensorResource> tensor = resource;
                    if (tensor.is_valid())
                    {
                        return tensor->save_to_file(path) ? godot::OK : godot::ERR_FILE_CANT_WRITE;
                    }
                    return godot::ERR_INVALID_PARAMETER;
                }
            };
#endif

            // --------------------------------------------------------------------
            // Resource registry helper
            // --------------------------------------------------------------------
            class XResourceRegistry
            {
            public:
                static void register_resources()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XTensorResource>();
                    godot::ClassDB::register_class<XTensorResourceLoader>();
                    godot::ClassDB::register_class<XTensorResourceSaver>();
                    
                    // Register loader/saver with Godot
                    godot::ResourceLoader::add_resource_format_loader(memnew(XTensorResourceLoader));
                    godot::ResourceSaver::add_resource_format_saver(memnew(XTensorResourceSaver));
#endif
                }

                static void unregister_resources()
                {
                    // Cleanup if needed
                }
            };

        } // namespace godot_bridge

        // Bring Godot resource types into xt namespace
        using godot_bridge::XTensorResource;
        using godot_bridge::XTensorResourceLoader;
        using godot_bridge::XTensorResourceSaver;
        using godot_bridge::XResourceRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XRESOURCE_HPP

// godot/xresource.hpp