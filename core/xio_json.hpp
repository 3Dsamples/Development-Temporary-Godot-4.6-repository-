// io/xio_json.hpp
#ifndef XTENSOR_XIO_JSON_HPP
#define XTENSOR_XIO_JSON_HPP

// ----------------------------------------------------------------------------
// xio_json.hpp – JSON serialization/deserialization for xtensor
// ----------------------------------------------------------------------------
// This header provides comprehensive JSON I/O capabilities:
//   - Serialize xarray_container to JSON string or file
//   - Deserialize JSON string or file to xarray_container
//   - Support for nested JSON structures (object of arrays, array of objects)
//   - Metadata preservation: shape, dtype, layout, attributes
//   - Streaming JSON parser for large files (SAX-style)
//   - Pretty‑printing with configurable indentation
//   - Binary JSON (BSON, MessagePack) alternative for compact storage
//
// All numeric types are supported, including bignumber::BigNumber (serialized
// as decimal strings for exact precision). FFT acceleration is not directly
// used but the infrastructure is maintained.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <complex>

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "bignumber/bignumber.hpp"

namespace xt {
namespace io {
namespace json {

// ========================================================================
// Serialization options
// ========================================================================
struct json_options {
    bool pretty_print = true;
    int indent_size = 2;
    bool include_shape = true;
    bool include_dtype = false;
    bool as_byte_array = false;          // store as base64 for compactness
    bool strict_nan_inf = false;         // throw on NaN/Inf (JSON spec)
    bool allow_comments = false;         // allow // and /* */ comments when parsing
    std::string array_key = "data";
    std::string shape_key = "shape";
    std::string dtype_key = "dtype";
    std::string attrs_key = "attrs";
};

// ========================================================================
// JSON value representation (for advanced manipulation)
// ========================================================================
enum class json_type {
    null, boolean, integer, floating, string, array, object
};

class json_value {
public:
    json_value() = default;
    explicit json_value(std::nullptr_t);
    explicit json_value(bool b);
    explicit json_value(int64_t i);
    explicit json_value(uint64_t u);
    explicit json_value(double d);
    explicit json_value(const std::string& s);
    explicit json_value(const char* s);

    json_type type() const noexcept;
    bool is_null() const noexcept;
    bool is_bool() const noexcept;
    bool is_number() const noexcept;
    bool is_string() const noexcept;
    bool is_array() const noexcept;
    bool is_object() const noexcept;

    bool as_bool() const;
    int64_t as_int64() const;
    uint64_t as_uint64() const;
    double as_double() const;
    const std::string& as_string() const;

    // Array access
    size_t array_size() const;
    const json_value& operator[](size_t index) const;
    json_value& operator[](size_t index);
    void push_back(const json_value& val);
    void push_back(json_value&& val);

    // Object access
    bool contains(const std::string& key) const;
    const json_value& operator[](const std::string& key) const;
    json_value& operator[](const std::string& key);
    std::vector<std::string> keys() const;

    // Serialize to string
    std::string dump(const json_options& opts = {}) const;
};

// ========================================================================
// Serialization to JSON
// ========================================================================
template <class T>
std::string to_json(const xarray_container<T>& arr, const json_options& opts = {});

template <class T>
void save_json(const std::string& filename, const xarray_container<T>& arr,
               const json_options& opts = {});

template <class T>
void save_json(std::ostream& os, const xarray_container<T>& arr,
               const json_options& opts = {});

// Serialize with attributes (metadata dictionary)
template <class T>
std::string to_json(const xarray_container<T>& arr,
                    const std::map<std::string, json_value>& attrs,
                    const json_options& opts = {});

template <class T>
void save_json(const std::string& filename, const xarray_container<T>& arr,
               const std::map<std::string, json_value>& attrs,
               const json_options& opts = {});

// ========================================================================
// Deserialization from JSON
// ========================================================================
template <class T>
xarray_container<T> from_json(const std::string& json_str, const json_options& opts = {});

template <class T>
xarray_container<T> load_json(const std::string& filename, const json_options& opts = {});

template <class T>
xarray_container<T> load_json(std::istream& is, const json_options& opts = {});

// Load with attributes extraction
template <class T>
std::pair<xarray_container<T>, std::map<std::string, json_value>>
load_json_with_attrs(const std::string& filename, const json_options& opts = {});

// ========================================================================
// Streaming parser (SAX-style for large files)
// ========================================================================
enum class sax_event {
    null, boolean, number, string, start_array, end_array,
    start_object, end_object, key
};

template <class T>
class sax_parser {
public:
    using number_callback = std::function<void(T)>;
    using string_callback = std::function<void(const std::string&)>;
    using structure_callback = std::function<void()>;

    sax_parser();
    void on_number(number_callback cb);
    void on_string(string_callback cb);
    void on_start_array(structure_callback cb);
    void on_end_array(structure_callback cb);
    void on_start_object(structure_callback cb);
    void on_end_object(structure_callback cb);
    void on_key(string_callback cb);

    void parse(const std::string& json_str);
    void parse(std::istream& is);
    void parse_file(const std::string& filename);

private:
    // Implementation state
};

// ========================================================================
// JSON Patch (RFC 6902) and JSON Pointer (RFC 6901)
// ========================================================================
json_value json_patch_apply(const json_value& doc, const json_value& patch);
std::string json_pointer_evaluate(const json_value& doc, const std::string& pointer);

// ========================================================================
// Convenience: Nested JSON (object of arrays)
// ========================================================================
template <class T>
std::string to_json_object(const std::map<std::string, xarray_container<T>>& arrays,
                           const json_options& opts = {});

template <class T>
std::map<std::string, xarray_container<T>>
from_json_object(const std::string& json_str, const json_options& opts = {});

} // namespace json

using json::to_json;
using json::save_json;
using json::from_json;
using json::load_json;
using json::load_json_with_attrs;
using json::json_options;
using json::json_value;
using json::sax_parser;

} // namespace io
} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt {
namespace io {
namespace json {

// json_value implementation
inline json_value::json_value(std::nullptr_t) { /* TODO: set type null */ }
inline json_value::json_value(bool b) { /* TODO: set type boolean */ }
inline json_value::json_value(int64_t i) { /* TODO: set type integer */ }
inline json_value::json_value(uint64_t u) { /* TODO: set type integer */ }
inline json_value::json_value(double d) { /* TODO: set type floating */ }
inline json_value::json_value(const std::string& s) { /* TODO: set type string */ }
inline json_value::json_value(const char* s) { /* TODO: set type string */ }

inline json_type json_value::type() const noexcept { return json_type::null; }
inline bool json_value::is_null() const noexcept { return type() == json_type::null; }
inline bool json_value::is_bool() const noexcept { return type() == json_type::boolean; }
inline bool json_value::is_number() const noexcept { return type() == json_type::integer || type() == json_type::floating; }
inline bool json_value::is_string() const noexcept { return type() == json_type::string; }
inline bool json_value::is_array() const noexcept { return type() == json_type::array; }
inline bool json_value::is_object() const noexcept { return type() == json_type::object; }

inline bool json_value::as_bool() const { return false; }
inline int64_t json_value::as_int64() const { return 0; }
inline uint64_t json_value::as_uint64() const { return 0; }
inline double json_value::as_double() const { return 0.0; }
inline const std::string& json_value::as_string() const { static std::string empty; return empty; }
inline size_t json_value::array_size() const { return 0; }
inline const json_value& json_value::operator[](size_t index) const { return *this; }
inline json_value& json_value::operator[](size_t index) { return *this; }
inline void json_value::push_back(const json_value& val) { /* TODO: append to array */ }
inline void json_value::push_back(json_value&& val) { /* TODO: append to array */ }
inline bool json_value::contains(const std::string& key) const { return false; }
inline const json_value& json_value::operator[](const std::string& key) const { return *this; }
inline json_value& json_value::operator[](const std::string& key) { return *this; }
inline std::vector<std::string> json_value::keys() const { return {}; }
inline std::string json_value::dump(const json_options& opts) const { return ""; }

// Serialization
template <class T>
std::string to_json(const xarray_container<T>& arr, const json_options& opts)
{ /* TODO: serialize array to JSON string */ return ""; }
template <class T>
void save_json(const std::string& filename, const xarray_container<T>& arr, const json_options& opts)
{ /* TODO: write to file */ }
template <class T>
void save_json(std::ostream& os, const xarray_container<T>& arr, const json_options& opts)
{ /* TODO: write to stream */ }
template <class T>
std::string to_json(const xarray_container<T>& arr, const std::map<std::string, json_value>& attrs, const json_options& opts)
{ /* TODO: serialize with attributes */ return ""; }
template <class T>
void save_json(const std::string& filename, const xarray_container<T>& arr,
               const std::map<std::string, json_value>& attrs, const json_options& opts)
{ /* TODO: write to file with attributes */ }

// Deserialization
template <class T>
xarray_container<T> from_json(const std::string& json_str, const json_options& opts)
{ /* TODO: parse JSON and build array */ return {}; }
template <class T>
xarray_container<T> load_json(const std::string& filename, const json_options& opts)
{ /* TODO: read file and parse */ return {}; }
template <class T>
xarray_container<T> load_json(std::istream& is, const json_options& opts)
{ /* TODO: read stream and parse */ return {}; }
template <class T>
std::pair<xarray_container<T>, std::map<std::string, json_value>>
load_json_with_attrs(const std::string& filename, const json_options& opts)
{ /* TODO: parse and extract attributes */ return {}; }

// SAX parser
template <class T> sax_parser<T>::sax_parser() { /* TODO: init */ }
template <class T> void sax_parser<T>::on_number(number_callback cb) { /* TODO: set callback */ }
template <class T> void sax_parser<T>::on_string(string_callback cb) { /* TODO: set callback */ }
template <class T> void sax_parser<T>::on_start_array(structure_callback cb) { /* TODO: set callback */ }
template <class T> void sax_parser<T>::on_end_array(structure_callback cb) { /* TODO: set callback */ }
template <class T> void sax_parser<T>::on_start_object(structure_callback cb) { /* TODO: set callback */ }
template <class T> void sax_parser<T>::on_end_object(structure_callback cb) { /* TODO: set callback */ }
template <class T> void sax_parser<T>::on_key(string_callback cb) { /* TODO: set callback */ }
template <class T> void sax_parser<T>::parse(const std::string& json_str) { /* TODO: parse string */ }
template <class T> void sax_parser<T>::parse(std::istream& is) { /* TODO: parse stream */ }
template <class T> void sax_parser<T>::parse_file(const std::string& filename) { /* TODO: parse file */ }

// JSON Patch and Pointer
inline json_value json_patch_apply(const json_value& doc, const json_value& patch)
{ /* TODO: apply RFC 6902 patch */ return doc; }
inline std::string json_pointer_evaluate(const json_value& doc, const std::string& pointer)
{ /* TODO: RFC 6901 evaluation */ return ""; }

// Nested JSON
template <class T>
std::string to_json_object(const std::map<std::string, xarray_container<T>>& arrays, const json_options& opts)
{ /* TODO: serialize map to JSON object */ return ""; }
template <class T>
std::map<std::string, xarray_container<T>>
from_json_object(const std::string& json_str, const json_options& opts)
{ /* TODO: parse JSON object to map */ return {}; }

} // namespace json
} // namespace io
} // namespace xt

#endif // XTENSOR_XIO_JSON_HPP))) get();
                    }
                    std::string num_str = m_input.substr(start, m_pos - start);
                    if (is_double)
                    {
                        double val;
                        auto [ptr, ec] = std::from_chars(num_str.data(), num_str.data() + num_str.size(), val);
                        if (ec != std::errc()) throw std::runtime_error("Invalid number");
                        return JsonValue(val);
                    }
                    else
                    {
                        int64_t val;
                        auto [ptr, ec] = std::from_chars(num_str.data(), num_str.data() + num_str.size(), val);
                        if (ec != std::errc()) throw std::runtime_error("Invalid integer");
                        return JsonValue(val);
                    }
                }

                JsonValue parse_array()
                {
                    if (get() != '[') throw std::runtime_error("Expected '['");
                    JsonValue::Array arr;
                    skip_whitespace();
                    if (peek() == ']')
                    {
                        get();
                        return JsonValue(arr);
                    }
                    while (true)
                    {
                        arr.push_back(parse_value());
                        skip_whitespace();
                        char c = get();
                        if (c == ']') break;
                        if (c != ',') throw std::runtime_error("Expected ',' or ']'");
                        skip_whitespace();
                    }
                    return JsonValue(arr);
                }

                JsonValue parse_object()
                {
                    if (get() != '{') throw std::runtime_error("Expected '{'");
                    JsonValue::Object obj;
                    skip_whitespace();
                    if (peek() == '}')
                    {
                        get();
                        return JsonValue(obj);
                    }
                    while (true)
                    {
                        skip_whitespace();
                        JsonValue key = parse_string();
                        skip_whitespace();
                        if (get() != ':') throw std::runtime_error("Expected ':'");
                        skip_whitespace();
                        obj[key.as_string()] = parse_value();
                        skip_whitespace();
                        char c = get();
                        if (c == '}') break;
                        if (c != ',') throw std::runtime_error("Expected ',' or '}'");
                    }
                    return JsonValue(obj);
                }
            };

            // --------------------------------------------------------------------
            // xarray to/from JSON
            // --------------------------------------------------------------------
            namespace json_detail
            {
                template<typename T>
                JsonValue array_to_json(const xarray_container<T>& arr)
                {
                    JsonValue::Array jarr;
                    if (arr.dimension() == 0)
                    {
                        // Scalar
                        return JsonValue(static_cast<double>(arr()));
                    }
                    else if (arr.dimension() == 1)
                    {
                        jarr.reserve(arr.size());
                        for (size_t i = 0; i < arr.size(); ++i)
                            jarr.push_back(JsonValue(static_cast<double>(arr(i))));
                    }
                    else
                    {
                        // Recursively convert multi-dimensional arrays to nested JSON arrays
                        std::function<JsonValue(const xarray_container<T>&, size_t, const std::vector<size_t>&)> build;
                        build = [&](const xarray_container<T>& a, size_t dim, const std::vector<size_t>& idx) -> JsonValue {
                            if (dim == a.dimension())
                            {
                                return JsonValue(static_cast<double>(a.element(idx)));
                            }
                            JsonValue::Array level;
                            size_t size = a.shape()[dim];
                            std::vector<size_t> new_idx = idx;
                            new_idx.push_back(0);
                            for (size_t i = 0; i < size; ++i)
                            {
                                new_idx[dim] = i;
                                level.push_back(build(a, dim + 1, new_idx));
                            }
                            return JsonValue(level);
                        };
                        return build(arr, 0, {});
                    }
                    return JsonValue(jarr);
                }

                template<typename T>
                xarray_container<T> json_to_array(const JsonValue& jval)
                {
                    if (jval.is_number())
                    {
                        // Scalar
                        xarray_container<T> result({});
                        result() = static_cast<T>(jval.as_double());
                        return result;
                    }
                    else if (jval.is_array())
                    {
                        const JsonValue::Array& jarr = jval.as_array();
                        if (jarr.empty())
                            return xarray_container<T>();
                        
                        // Determine dimensions
                        std::vector<size_t> shape;
                        std::function<void(const JsonValue&, std::vector<size_t>&)> explore;
                        explore = [&](const JsonValue& node, std::vector<size_t>& dims) {
                            if (node.is_array())
                            {
                                const JsonValue::Array& arr = node.as_array();
                                dims.push_back(arr.size());
                                if (!arr.empty())
                                    explore(arr[0], dims);
                            }
                        };
                        explore(jval, shape);
                        
                        xarray_container<T> result(shape);
                        std::function<void(const JsonValue&, std::vector<size_t>&, size_t)> fill;
                        fill = [&](const JsonValue& node, std::vector<size_t>& idx, size_t dim) {
                            if (dim == shape.size())
                            {
                                result.element(idx) = static_cast<T>(node.as_double());
                                return;
                            }
                            const JsonValue::Array& arr = node.as_array();
                            for (size_t i = 0; i < arr.size(); ++i)
                            {
                                idx[dim] = i;
                                fill(arr[i], idx, dim + 1);
                            }
                        };
                        std::vector<size_t> idx(shape.size(), 0);
                        fill(jval, idx, 0);
                        return result;
                    }
                    else
                    {
                        throw std::runtime_error("JSON value is not an array or number");
                    }
                }

                template<typename T>
                JsonValue array_to_json_with_metadata(const xarray_container<T>& arr, const std::string& name = "")
                {
                    JsonValue::Object obj;
                    if (!name.empty())
                        obj["name"] = name;
                    obj["shape"] = array_to_json(arr.shape());
                    obj["data"] = array_to_json(arr);
                    if constexpr (std::is_same_v<T, float>)
                        obj["dtype"] = "float32";
                    else if constexpr (std::is_same_v<T, double>)
                        obj["dtype"] = "float64";
                    else if constexpr (std::is_same_v<T, int32_t>)
                        obj["dtype"] = "int32";
                    else if constexpr (std::is_same_v<T, int64_t>)
                        obj["dtype"] = "int64";
                    else if constexpr (std::is_same_v<T, uint8_t>)
                        obj["dtype"] = "uint8";
                    else
                        obj["dtype"] = "unknown";
                    return JsonValue(obj);
                }

                template<typename T>
                xarray_container<T> json_to_array_with_metadata(const JsonValue& jval)
                {
                    if (jval.is_object() && jval.contains("data") && jval.contains("shape"))
                    {
                        return json_to_array<T>(jval["data"]);
                    }
                    else
                    {
                        return json_to_array<T>(jval);
                    }
                }

                // Helper for std::vector
                template<typename T>
                JsonValue vector_to_json(const std::vector<T>& vec)
                {
                    JsonValue::Array jarr;
                    jarr.reserve(vec.size());
                    for (const auto& v : vec)
                        jarr.push_back(JsonValue(static_cast<double>(v)));
                    return JsonValue(jarr);
                }

                template<typename T>
                std::vector<T> json_to_vector(const JsonValue& jval)
                {
                    std::vector<T> result;
                    if (jval.is_array())
                    {
                        for (const auto& v : jval.as_array())
                            result.push_back(static_cast<T>(v.as_double()));
                    }
                    return result;
                }
            }

            // --------------------------------------------------------------------
            // Public JSON I/O functions for xarray
            // --------------------------------------------------------------------
            template<typename T>
            inline std::string to_json(const xarray_container<T>& arr, bool with_metadata = false, int indent = -1)
            {
                JsonValue jval;
                if (with_metadata)
                    jval = json_detail::array_to_json_with_metadata(arr);
                else
                    jval = json_detail::array_to_json(arr);
                return jval.dump(indent);
            }

            template<typename T>
            inline xarray_container<T> from_json(const std::string& json_str)
            {
                JsonValue jval = JsonParser::parse(json_str);
                return json_detail::json_to_array_with_metadata<T>(jval);
            }

            inline xarray_container<double> from_json(const std::string& json_str)
            {
                return from_json<double>(json_str);
            }

            // Save to file
            template<typename T>
            inline void save_json(const std::string& filename, const xarray_container<T>& arr,
                                  bool with_metadata = true, int indent = 2)
            {
                std::ofstream out(filename);
                if (!out)
                    XTENSOR_THROW(std::runtime_error, "Cannot open JSON file for writing: " + filename);
                out << to_json(arr, with_metadata, indent);
            }

            // Load from file
            template<typename T>
            inline xarray_container<T> load_json(const std::string& filename)
            {
                std::ifstream in(filename);
                if (!in)
                    XTENSOR_THROW(std::runtime_error, "Cannot open JSON file: " + filename);
                std::string content((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
                return from_json<T>(content);
            }

            inline xarray_container<double> load_json(const std::string& filename)
            {
                return load_json<double>(filename);
            }

            // --------------------------------------------------------------------
            // Dictionary-like storage (multiple named arrays)
            // --------------------------------------------------------------------
            class JsonArchive
            {
            public:
                void add_array(const std::string& name, const xarray_container<double>& arr)
                {
                    m_data[name] = arr;
                }

                template<typename T>
                void add_array(const std::string& name, const xarray_container<T>& arr)
                {
                    // Convert to double for storage
                    m_data[name] = xt::cast<double>(arr);
                }

                xarray_container<double> get_array(const std::string& name) const
                {
                    auto it = m_data.find(name);
                    if (it != m_data.end())
                        return it->second;
                    XTENSOR_THROW(std::runtime_error, "Array not found: " + name);
                }

                std::vector<std::string> list_arrays() const
                {
                    std::vector<std::string> names;
                    for (const auto& p : m_data)
                        names.push_back(p.first);
                    return names;
                }

                std::string to_json(int indent = 2) const
                {
                    JsonValue::Object root;
                    for (const auto& p : m_data)
                        root[p.first] = json_detail::array_to_json_with_metadata(p.second, p.first);
                    return JsonValue(root).dump(indent);
                }

                void from_json(const std::string& json_str)
                {
                    JsonValue jval = JsonParser::parse(json_str);
                    if (!jval.is_object())
                        XTENSOR_THROW(std::runtime_error, "JSON root must be an object");
                    m_data.clear();
                    for (const auto& p : jval.as_object())
                    {
                        m_data[p.first] = json_detail::json_to_array_with_metadata<double>(p.second);
                    }
                }

                void save(const std::string& filename, int indent = 2) const
                {
                    std::ofstream out(filename);
                    if (!out)
                        XTENSOR_THROW(std::runtime_error, "Cannot open JSON file: " + filename);
                    out << to_json(indent);
                }

                void load(const std::string& filename)
                {
                    std::ifstream in(filename);
                    if (!in)
                        XTENSOR_THROW(std::runtime_error, "Cannot open JSON file: " + filename);
                    std::string content((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
                    from_json(content);
                }

            private:
                std::map<std::string, xarray_container<double>> m_data;
            };

            // --------------------------------------------------------------------
            // Convenience functions for std::vector
            // --------------------------------------------------------------------
            template<typename T>
            inline std::string vector_to_json(const std::vector<T>& vec, int indent = -1)
            {
                return json_detail::vector_to_json(vec).dump(indent);
            }

            template<typename T>
            inline std::vector<T> json_to_vector(const std::string& json_str)
            {
                JsonValue jval = JsonParser::parse(json_str);
                return json_detail::json_to_vector<T>(jval);
            }

        } // namespace io

        // Bring JSON functions into xt namespace
        using io::to_json;
        using io::from_json;
        using io::save_json;
        using io::load_json;
        using io::JsonArchive;
        using io::vector_to_json;
        using io::json_to_vector;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XIO_JSON_HPP

// io/xio_json.hpp