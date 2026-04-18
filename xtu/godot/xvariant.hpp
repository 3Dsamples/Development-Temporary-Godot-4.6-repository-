// godot/xvariant.hpp

#ifndef XTENSOR_XVARIANT_HPP
#define XTENSOR_XVARIANT_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../io/xio_json.hpp"

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
#include <complex>
#include <functional>
#include <variant>
#include <optional>

// Godot support detection
#if __has_include(<godot_cpp/core/defs.hpp>) || __has_include(<core/variant.h>)
    #define XTENSOR_HAS_GODOT 1
    #if __has_include(<godot_cpp/variant/variant.hpp>)
        #include <godot_cpp/variant/variant.hpp>
        #include <godot_cpp/variant/array.hpp>
        #include <godot_cpp/variant/dictionary.hpp>
        #include <godot_cpp/variant/packed_float32_array.hpp>
        #include <godot_cpp/variant/packed_float64_array.hpp>
        #include <godot_cpp/variant/packed_int32_array.hpp>
        #include <godot_cpp/variant/packed_int64_array.hpp>
        #include <godot_cpp/variant/packed_byte_array.hpp>
    #else
        // Fallback: define minimal Godot variant types for interface compatibility
        namespace godot
        {
            class Variant
            {
            public:
                enum Type { NIL, BOOL, INT, FLOAT, STRING, ARRAY, DICTIONARY, PACKED_FLOAT32_ARRAY, PACKED_FLOAT64_ARRAY, PACKED_INT32_ARRAY, PACKED_INT64_ARRAY, PACKED_BYTE_ARRAY, OBJECT };
                virtual ~Variant() = default;
                virtual Type get_type() const = 0;
            };
            class Array : public Variant { public: Type get_type() const override { return ARRAY; } };
            class Dictionary : public Variant { public: Type get_type() const override { return DICTIONARY; } };
            class PackedFloat32Array : public Variant { public: Type get_type() const override { return PACKED_FLOAT32_ARRAY; } };
            class PackedFloat64Array : public Variant { public: Type get_type() const override { return PACKED_FLOAT64_ARRAY; } };
            class PackedInt32Array : public Variant { public: Type get_type() const override { return PACKED_INT32_ARRAY; } };
            class PackedInt64Array : public Variant { public: Type get_type() const override { return PACKED_INT64_ARRAY; } };
            class PackedByteArray : public Variant { public: Type get_type() const override { return PACKED_BYTE_ARRAY; } };
        }
    #endif
#else
    #define XTENSOR_HAS_GODOT 0
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // Type mapping between xtensor and Godot
            // --------------------------------------------------------------------
            template<typename T>
            struct godot_type_traits
            {
                static constexpr bool is_supported = false;
            };

#if XTENSOR_HAS_GODOT
            template<> struct godot_type_traits<float>
            {
                static constexpr bool is_supported = true;
                using packed_type = godot::PackedFloat32Array;
                static godot::Variant::Type variant_type() { return godot::Variant::PACKED_FLOAT32_ARRAY; }
            };
            template<> struct godot_type_traits<double>
            {
                static constexpr bool is_supported = true;
                using packed_type = godot::PackedFloat64Array;
                static godot::Variant::Type variant_type() { return godot::Variant::PACKED_FLOAT64_ARRAY; }
            };
            template<> struct godot_type_traits<int32_t>
            {
                static constexpr bool is_supported = true;
                using packed_type = godot::PackedInt32Array;
                static godot::Variant::Type variant_type() { return godot::Variant::PACKED_INT32_ARRAY; }
            };
            template<> struct godot_type_traits<int64_t>
            {
                static constexpr bool is_supported = true;
                using packed_type = godot::PackedInt64Array;
                static godot::Variant::Type variant_type() { return godot::Variant::PACKED_INT64_ARRAY; }
            };
            template<> struct godot_type_traits<uint8_t>
            {
                static constexpr bool is_supported = true;
                using packed_type = godot::PackedByteArray;
                static godot::Variant::Type variant_type() { return godot::Variant::PACKED_BYTE_ARRAY; }
            };
            template<> struct godot_type_traits<std::string>
            {
                static constexpr bool is_supported = true;
                static godot::Variant::Type variant_type() { return godot::Variant::STRING; }
            };
            template<> struct godot_type_traits<bool>
            {
                static constexpr bool is_supported = true;
                static godot::Variant::Type variant_type() { return godot::Variant::BOOL; }
            };
#endif

            // --------------------------------------------------------------------
            // xarray to Godot Variant conversion
            // --------------------------------------------------------------------
            template<typename T>
            class xarray_to_godot
            {
            public:
#if XTENSOR_HAS_GODOT
                static godot::Variant convert(const xarray_container<T>& arr)
                {
                    if (arr.dimension() == 0)
                    {
                        // Scalar
                        if constexpr (std::is_same_v<T, bool>)
                            return godot::Variant(arr());
                        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>)
                            return godot::Variant(static_cast<int64_t>(arr()));
                        else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
                            return godot::Variant(static_cast<double>(arr()));
                        else if constexpr (std::is_same_v<T, std::string>)
                            return godot::Variant(godot::String(arr().c_str()));
                        else
                            return godot::Variant();
                    }
                    else if (arr.dimension() == 1)
                    {
                        // 1D array -> Packed*Array
                        if constexpr (godot_type_traits<T>::is_supported)
                        {
                            typename godot_type_traits<T>::packed_type packed;
                            packed.resize(static_cast<int>(arr.size()));
                            for (size_t i = 0; i < arr.size(); ++i)
                                packed.set(static_cast<int>(i), arr(i));
                            return packed;
                        }
                        else
                        {
                            // Fallback to generic Array of Variants
                            godot::Array gd_arr;
                            gd_arr.resize(static_cast<int>(arr.size()));
                            for (size_t i = 0; i < arr.size(); ++i)
                            {
                                if constexpr (std::is_arithmetic_v<T>)
                                    gd_arr[i] = static_cast<double>(arr(i));
                                else
                                    gd_arr[i] = godot::Variant();
                            }
                            return gd_arr;
                        }
                    }
                    else
                    {
                        // Multi-dimensional -> nested Array
                        godot::Array gd_arr;
                        gd_arr.resize(static_cast<int>(arr.shape()[0]));
                        for (size_t i = 0; i < arr.shape()[0]; ++i)
                        {
                            auto slice = xt::view(arr, i, xt::ellipsis());
                            gd_arr[i] = convert(slice);
                        }
                        return gd_arr;
                    }
                }
#else
                static void* convert(const xarray_container<T>&) { return nullptr; }
#endif
            };

            // --------------------------------------------------------------------
            // Godot Variant to xarray conversion
            // --------------------------------------------------------------------
            template<typename T>
            class godot_to_xarray
            {
            public:
#if XTENSOR_HAS_GODOT
                static xarray_container<T> convert(const godot::Variant& var)
                {
                    godot::Variant::Type type = var.get_type();
                    
                    // Scalar types
                    if (type == godot::Variant::BOOL && std::is_same_v<T, bool>)
                    {
                        xarray_container<T> result({});
                        result() = static_cast<T>(static_cast<const godot::Variant&>(var).operator bool());
                        return result;
                    }
                    if (type == godot::Variant::INT && (std::is_integral_v<T> || std::is_floating_point_v<T>))
                    {
                        xarray_container<T> result({});
                        result() = static_cast<T>(static_cast<const godot::Variant&>(var).operator int64_t());
                        return result;
                    }
                    if (type == godot::Variant::FLOAT && std::is_floating_point_v<T>)
                    {
                        xarray_container<T> result({});
                        result() = static_cast<T>(static_cast<const godot::Variant&>(var).operator double());
                        return result;
                    }
                    if (type == godot::Variant::STRING && std::is_same_v<T, std::string>)
                    {
                        xarray_container<T> result({});
                        result() = static_cast<const godot::Variant&>(var).operator godot::String().utf8().get_data();
                        return result;
                    }
                    
                    // Packed arrays
                    if (type == godot::Variant::PACKED_FLOAT32_ARRAY)
                    {
                        const auto& packed = static_cast<const godot::PackedFloat32Array&>(var);
                        xarray_container<T> result({static_cast<size_t>(packed.size())});
                        for (int i = 0; i < packed.size(); ++i)
                            result(i) = static_cast<T>(packed[i]);
                        return result;
                    }
                    if (type == godot::Variant::PACKED_FLOAT64_ARRAY)
                    {
                        const auto& packed = static_cast<const godot::PackedFloat64Array&>(var);
                        xarray_container<T> result({static_cast<size_t>(packed.size())});
                        for (int i = 0; i < packed.size(); ++i)
                            result(i) = static_cast<T>(packed[i]);
                        return result;
                    }
                    if (type == godot::Variant::PACKED_INT32_ARRAY)
                    {
                        const auto& packed = static_cast<const godot::PackedInt32Array&>(var);
                        xarray_container<T> result({static_cast<size_t>(packed.size())});
                        for (int i = 0; i < packed.size(); ++i)
                            result(i) = static_cast<T>(packed[i]);
                        return result;
                    }
                    if (type == godot::Variant::PACKED_INT64_ARRAY)
                    {
                        const auto& packed = static_cast<const godot::PackedInt64Array&>(var);
                        xarray_container<T> result({static_cast<size_t>(packed.size())});
                        for (int i = 0; i < packed.size(); ++i)
                            result(i) = static_cast<T>(packed[i]);
                        return result;
                    }
                    if (type == godot::Variant::PACKED_BYTE_ARRAY && std::is_same_v<T, uint8_t>)
                    {
                        const auto& packed = static_cast<const godot::PackedByteArray&>(var);
                        xarray_container<T> result({static_cast<size_t>(packed.size())});
                        for (int i = 0; i < packed.size(); ++i)
                            result(i) = packed[i];
                        return result;
                    }
                    
                    // Generic Array (possibly nested)
                    if (type == godot::Variant::ARRAY)
                    {
                        const auto& arr = static_cast<const godot::Array&>(var);
                        return parse_array_recursive<T>(arr);
                    }
                    
                    // Dictionary: treat as struct with named fields
                    if (type == godot::Variant::DICTIONARY)
                    {
                        // Return empty for now; dictionary handling requires special output type
                        return xarray_container<T>();
                    }
                    
                    return xarray_container<T>();
                }

            private:
                static xarray_container<T> parse_array_recursive(const godot::Array& arr)
                {
                    if (arr.is_empty())
                        return xarray_container<T>();
                    
                    // Check if elements are arrays (nested)
                    bool is_nested = false;
                    size_t nested_dim = 0;
                    std::vector<size_t> shape;
                    
                    godot::Variant first = arr[0];
                    if (first.get_type() == godot::Variant::ARRAY)
                    {
                        is_nested = true;
                        shape.push_back(arr.size());
                        // Recursively determine inner shape
                        auto inner = parse_array_recursive(static_cast<const godot::Array&>(first));
                        for (auto s : inner.shape())
                            shape.push_back(s);
                    }
                    else
                    {
                        shape.push_back(arr.size());
                    }
                    
                    xarray_container<T> result(shape);
                    if (is_nested)
                    {
                        for (int i = 0; i < arr.size(); ++i)
                        {
                            auto slice = parse_array_recursive(static_cast<const godot::Array&>(arr[i]));
                            auto dest = xt::view(result, i, xt::ellipsis());
                            dest = slice;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < arr.size(); ++i)
                            result(i) = static_cast<T>(static_cast<double>(arr[i]));
                    }
                    return result;
                }
#else
                static xarray_container<T> convert(const void*) { return xarray_container<T>(); }
#endif
            };

            // --------------------------------------------------------------------
            // Dictionary (Map) conversion
            // --------------------------------------------------------------------
#if XTENSOR_HAS_GODOT
            inline std::map<std::string, xarray_container<double>> dict_from_godot(const godot::Dictionary& dict)
            {
                std::map<std::string, xarray_container<double>> result;
                godot::Array keys = dict.keys();
                for (int i = 0; i < keys.size(); ++i)
                {
                    std::string key = static_cast<std::string>(godot::String(keys[i]).utf8());
                    godot::Variant value = dict[keys[i]];
                    result[key] = godot_to_xarray<double>::convert(value);
                }
                return result;
            }

            inline godot::Dictionary dict_to_godot(const std::map<std::string, xarray_container<double>>& dict)
            {
                godot::Dictionary gd_dict;
                for (const auto& p : dict)
                {
                    gd_dict[godot::String(p.first.c_str())] = xarray_to_godot<double>::convert(p.second);
                }
                return gd_dict;
            }
#endif

            // --------------------------------------------------------------------
            // xvariant class: unified interface for Godot interop
            // --------------------------------------------------------------------
            class XVariant
            {
            public:
                XVariant() = default;
#if XTENSOR_HAS_GODOT
                explicit XVariant(const godot::Variant& var) : m_variant(var) {}
                
                // Conversion to xarray
                template<typename T>
                xarray_container<T> to_xarray() const
                {
                    return godot_to_xarray<T>::convert(m_variant);
                }
                
                xarray_container<double> to_double_array() const { return to_xarray<double>(); }
                xarray_container<float> to_float_array() const { return to_xarray<float>(); }
                xarray_container<int64_t> to_int_array() const { return to_xarray<int64_t>(); }
                xarray_container<uint8_t> to_byte_array() const { return to_xarray<uint8_t>(); }
                std::map<std::string, xarray_container<double>> to_dict() const
                {
                    if (m_variant.get_type() == godot::Variant::DICTIONARY)
                        return dict_from_godot(static_cast<const godot::Dictionary&>(m_variant));
                    return {};
                }
                
                // Conversion from xarray
                template<typename T>
                static XVariant from_xarray(const xarray_container<T>& arr)
                {
                    return XVariant(xarray_to_godot<T>::convert(arr));
                }
                
                static XVariant from_dict(const std::map<std::string, xarray_container<double>>& dict)
                {
                    return XVariant(dict_to_godot(dict));
                }
                
                // Access underlying variant
                const godot::Variant& variant() const { return m_variant; }
                godot::Variant& variant() { return m_variant; }
                
                // Type information
                godot::Variant::Type get_type() const { return m_variant.get_type(); }
                bool is_array() const { return m_variant.get_type() == godot::Variant::ARRAY; }
                bool is_packed_array() const
                {
                    auto t = m_variant.get_type();
                    return t == godot::Variant::PACKED_FLOAT32_ARRAY ||
                           t == godot::Variant::PACKED_FLOAT64_ARRAY ||
                           t == godot::Variant::PACKED_INT32_ARRAY ||
                           t == godot::Variant::PACKED_INT64_ARRAY ||
                           t == godot::Variant::PACKED_BYTE_ARRAY;
                }
                bool is_dictionary() const { return m_variant.get_type() == godot::Variant::DICTIONARY; }
                bool is_scalar() const
                {
                    auto t = m_variant.get_type();
                    return t == godot::Variant::BOOL || t == godot::Variant::INT ||
                           t == godot::Variant::FLOAT || t == godot::Variant::STRING;
                }

#else
                // Placeholder implementation when Godot not available
                explicit XVariant(const void*) {}
                template<typename T> xarray_container<T> to_xarray() const { return {}; }
                xarray_container<double> to_double_array() const { return {}; }
                std::map<std::string, xarray_container<double>> to_dict() const { return {}; }
                template<typename T> static XVariant from_xarray(const xarray_container<T>&) { return XVariant(); }
                static XVariant from_dict(const std::map<std::string, xarray_container<double>>&) { return XVariant(); }
                void* variant() { return nullptr; }
#endif

                // Serialization helpers (work regardless of Godot availability)
                std::string to_json() const
                {
#if XTENSOR_HAS_GODOT
                    if (is_dictionary())
                    {
                        auto dict = to_dict();
                        JsonValue::Object jobj;
                        for (const auto& p : dict)
                            jobj[p.first] = json_detail::array_to_json_with_metadata(p.second);
                        return JsonValue(jobj).dump(2);
                    }
                    else
                    {
                        return json_detail::array_to_json_with_metadata(to_double_array()).dump(2);
                    }
#else
                    return "{}";
#endif
                }

                static XVariant from_json(const std::string& json_str)
                {
#if XTENSOR_HAS_GODOT
                    JsonValue jval = JsonParser::parse(json_str);
                    if (jval.is_object())
                    {
                        std::map<std::string, xarray_container<double>> dict;
                        for (const auto& p : jval.as_object())
                            dict[p.first] = json_detail::json_to_array_with_metadata<double>(p.second);
                        return XVariant::from_dict(dict);
                    }
                    else
                    {
                        auto arr = json_detail::json_to_array_with_metadata<double>(jval);
                        return XVariant::from_xarray(arr);
                    }
#else
                    return XVariant();
#endif
                }

            private:
#if XTENSOR_HAS_GODOT
                godot::Variant m_variant;
#endif
            };

            // --------------------------------------------------------------------
            // Convenience functions
            // --------------------------------------------------------------------
            template<typename T>
            inline XVariant to_variant(const xarray_container<T>& arr)
            {
                return XVariant::from_xarray(arr);
            }

            template<typename T>
            inline xarray_container<T> from_variant(const XVariant& var)
            {
                return var.to_xarray<T>();
            }

            inline std::map<std::string, xarray_container<double>> dict_from_variant(const XVariant& var)
            {
                return var.to_dict();
            }

            inline XVariant dict_to_variant(const std::map<std::string, xarray_container<double>>& dict)
            {
                return XVariant::from_dict(dict);
            }

            // --------------------------------------------------------------------
            // Tensor registration with Godot ClassDB (placeholder)
            // --------------------------------------------------------------------
            class XVariantRegister
            {
            public:
                static void register_types()
                {
#if XTENSOR_HAS_GODOT
                    // In actual Godot GDExtension, this would register the XVariant
                    // as a Resource or RefCounted for use in GDScript.
                    // Placeholder: registration would go here.
#endif
                }
                
                static void unregister_types()
                {
                }
            };

        } // namespace godot_bridge

        // Bring Godot bridge types into xt namespace
        using godot_bridge::XVariant;
        using godot_bridge::to_variant;
        using godot_bridge::from_variant;
        using godot_bridge::dict_from_variant;
        using godot_bridge::dict_to_variant;
        using godot_bridge::XVariantRegister;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XVARIANT_HPP

// godot/xvariant.hpp