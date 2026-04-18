// io/xio_json.hpp

#ifndef XTENSOR_XIO_JSON_HPP
#define XTENSOR_XIO_JSON_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <complex>
#include <functional>
#include <variant>
#include <charconv>
#include <cctype>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace io
        {
            // --------------------------------------------------------------------
            // Simple JSON value type
            // --------------------------------------------------------------------
            class JsonValue
            {
            public:
                using Object = std::map<std::string, JsonValue>;
                using Array = std::vector<JsonValue>;
                using Variant = std::variant<std::nullptr_t, bool, int64_t, double, std::string, Array, Object>;

                JsonValue() : m_value(nullptr) {}
                JsonValue(std::nullptr_t) : m_value(nullptr) {}
                JsonValue(bool b) : m_value(b) {}
                JsonValue(int i) : m_value(static_cast<int64_t>(i)) {}
                JsonValue(int64_t i) : m_value(i) {}
                JsonValue(double d) : m_value(d) {}
                JsonValue(const std::string& s) : m_value(s) {}
                JsonValue(const char* s) : m_value(std::string(s)) {}
                JsonValue(const Array& arr) : m_value(arr) {}
                JsonValue(const Object& obj) : m_value(obj) {}
                JsonValue(Array&& arr) : m_value(std::move(arr)) {}
                JsonValue(Object&& obj) : m_value(std::move(obj)) {}

                // Type queries
                bool is_null() const { return std::holds_alternative<std::nullptr_t>(m_value); }
                bool is_bool() const { return std::holds_alternative<bool>(m_value); }
                bool is_int() const { return std::holds_alternative<int64_t>(m_value); }
                bool is_double() const { return std::holds_alternative<double>(m_value); }
                bool is_number() const { return is_int() || is_double(); }
                bool is_string() const { return std::holds_alternative<std::string>(m_value); }
                bool is_array() const { return std::holds_alternative<Array>(m_value); }
                bool is_object() const { return std::holds_alternative<Object>(m_value); }

                // Accessors
                bool as_bool() const { return std::get<bool>(m_value); }
                int64_t as_int() const { return std::get<int64_t>(m_value); }
                double as_double() const
                {
                    if (is_int()) return static_cast<double>(std::get<int64_t>(m_value));
                    return std::get<double>(m_value);
                }
                const std::string& as_string() const { return std::get<std::string>(m_value); }
                const Array& as_array() const { return std::get<Array>(m_value); }
                const Object& as_object() const { return std::get<Object>(m_value); }

                Array& as_array() { return std::get<Array>(m_value); }
                Object& as_object() { return std::get<Object>(m_value); }

                // Array access
                JsonValue& operator[](size_t idx) { return as_array()[idx]; }
                const JsonValue& operator[](size_t idx) const { return as_array()[idx]; }

                // Object access
                JsonValue& operator[](const std::string& key) { return as_object()[key]; }
                const JsonValue& operator[](const std::string& key) const
                {
                    static const JsonValue null_val;
                    auto it = as_object().find(key);
                    if (it != as_object().end()) return it->second;
                    return null_val;
                }

                bool contains(const std::string& key) const
                {
                    if (!is_object()) return false;
                    return as_object().find(key) != as_object().end();
                }

                size_t size() const
                {
                    if (is_array()) return as_array().size();
                    if (is_object()) return as_object().size();
                    return 0;
                }

                // Serialize to string
                std::string dump(int indent = -1) const
                {
                    std::ostringstream oss;
                    dump_internal(oss, indent, 0);
                    return oss.str();
                }

            private:
                Variant m_value;

                void dump_internal(std::ostringstream& oss, int indent, int current_indent) const
                {
                    std::string indent_str;
                    if (indent >= 0)
                        indent_str = std::string(static_cast<size_t>(current_indent * indent), ' ');

                    if (is_null())
                    {
                        oss << "null";
                    }
                    else if (is_bool())
                    {
                        oss << (as_bool() ? "true" : "false");
                    }
                    else if (is_int())
                    {
                        oss << as_int();
                    }
                    else if (is_double())
                    {
                        oss << as_double();
                    }
                    else if (is_string())
                    {
                        oss << '"';
                        for (char c : as_string())
                        {
                            switch (c)
                            {
                                case '"': oss << "\\\""; break;
                                case '\\': oss << "\\\\"; break;
                                case '\b': oss << "\\b"; break;
                                case '\f': oss << "\\f"; break;
                                case '\n': oss << "\\n"; break;
                                case '\r': oss << "\\r"; break;
                                case '\t': oss << "\\t"; break;
                                default: oss << c; break;
                            }
                        }
                        oss << '"';
                    }
                    else if (is_array())
                    {
                        const Array& arr = as_array();
                        oss << '[';
                        if (indent >= 0 && !arr.empty()) oss << '\n';
                        for (size_t i = 0; i < arr.size(); ++i)
                        {
                            if (indent >= 0) oss << indent_str << std::string(static_cast<size_t>(indent), ' ');
                            arr[i].dump_internal(oss, indent, current_indent + 1);
                            if (i + 1 < arr.size())
                                oss << ',';
                            if (indent >= 0) oss << '\n';
                        }
                        if (indent >= 0 && !arr.empty()) oss << indent_str;
                        oss << ']';
                    }
                    else if (is_object())
                    {
                        const Object& obj = as_object();
                        oss << '{';
                        if (indent >= 0 && !obj.empty()) oss << '\n';
                        size_t count = 0;
                        for (const auto& p : obj)
                        {
                            if (indent >= 0) oss << indent_str << std::string(static_cast<size_t>(indent), ' ');
                            oss << '"' << p.first << "\":";
                            if (indent >= 0) oss << ' ';
                            p.second.dump_internal(oss, indent, current_indent + 1);
                            if (++count < obj.size())
                                oss << ',';
                            if (indent >= 0) oss << '\n';
                        }
                        if (indent >= 0 && !obj.empty()) oss << indent_str;
                        oss << '}';
                    }
                }
            };

            // --------------------------------------------------------------------
            // JSON Parser
            // --------------------------------------------------------------------
            class JsonParser
            {
            public:
                static JsonValue parse(const std::string& input)
                {
                    JsonParser parser(input);
                    parser.skip_whitespace();
                    JsonValue val = parser.parse_value();
                    parser.skip_whitespace();
                    if (parser.m_pos < parser.m_input.size())
                        throw std::runtime_error("Unexpected trailing characters");
                    return val;
                }

            private:
                const std::string& m_input;
                size_t m_pos = 0;

                JsonParser(const std::string& input) : m_input(input) {}

                char peek() const { return m_pos < m_input.size() ? m_input[m_pos] : '\0'; }
                char get() { return m_pos < m_input.size() ? m_input[m_pos++] : '\0'; }
                void skip_whitespace()
                {
                    while (m_pos < m_input.size() && std::isspace(m_input[m_pos]))
                        ++m_pos;
                }

                JsonValue parse_value()
                {
                    char c = peek();
                    if (c == 'n') return parse_null();
                    if (c == 't' || c == 'f') return parse_bool();
                    if (c == '"') return parse_string();
                    if (c == '-' || std::isdigit(c)) return parse_number();
                    if (c == '[') return parse_array();
                    if (c == '{') return parse_object();
                    throw std::runtime_error("Unexpected character");
                }

                JsonValue parse_null()
                {
                    if (m_input.substr(m_pos, 4) == "null")
                    {
                        m_pos += 4;
                        return JsonValue(nullptr);
                    }
                    throw std::runtime_error("Invalid null");
                }

                JsonValue parse_bool()
                {
                    if (m_input.substr(m_pos, 4) == "true")
                    {
                        m_pos += 4;
                        return JsonValue(true);
                    }
                    if (m_input.substr(m_pos, 5) == "false")
                    {
                        m_pos += 5;
                        return JsonValue(false);
                    }
                    throw std::runtime_error("Invalid boolean");
                }

                JsonValue parse_string()
                {
                    if (get() != '"') throw std::runtime_error("Expected '\"'");
                    std::string result;
                    while (true)
                    {
                        char c = get();
                        if (c == '"') break;
                        if (c == '\\')
                        {
                            c = get();
                            switch (c)
                            {
                                case '"': result += '"'; break;
                                case '\\': result += '\\'; break;
                                case '/': result += '/'; break;
                                case 'b': result += '\b'; break;
                                case 'f': result += '\f'; break;
                                case 'n': result += '\n'; break;
                                case 'r': result += '\r'; break;
                                case 't': result += '\t'; break;
                                case 'u':
                                {
                                    // Skip unicode escape (simplified)
                                    for (int i = 0; i < 4; ++i) get();
                                    result += '?';
                                    break;
                                }
                                default: result += c; break;
                            }
                        }
                        else
                        {
                            result += c;
                        }
                    }
                    return JsonValue(result);
                }

                JsonValue parse_number()
                {
                    size_t start = m_pos;
                    bool is_double = false;
                    if (peek() == '-') get();
                    while (std::isdigit(peek())) get();
                    if (peek() == '.')
                    {
                        is_double = true;
                        get();
                        while (std::isdigit(peek())) get();
                    }
                    if (peek() == 'e' || peek() == 'E')
                    {
                        is_double = true;
                        get();
                        if (peek() == '+' || peek() == '-') get();
                        while (std::isdigit(peek())) get();
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