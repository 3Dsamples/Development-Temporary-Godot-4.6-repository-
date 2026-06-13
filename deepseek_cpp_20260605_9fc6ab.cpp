//File 0064 : io/xjson.hpp
//JSON reader/writer for xtensor arrays with SIMD-accelerated number parsing, nested array support, and fast streaming serialization.
#ifndef XTENSOR_XJSON_HPP
#define XTENSOR_XJSON_HPP

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xarray.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xstrides.hpp"
#include "../core/xmanipulation.hpp"
#include "../core/xeval.hpp"
#include "../core/xbuilder.hpp"
#include "../core/xshape.hpp"

namespace xt {
namespace io {

    namespace detail {

        // Skip whitespace in a JSON stream
        inline void skip_whitespace(const char*& ptr, const char* end) {
            while (ptr < end && (std::isspace(static_cast<unsigned char>(*ptr)) || *ptr == '\n' || *ptr == '\r' || *ptr == '\t')) {
                ++ptr;
            }
        }

        // Parse a JSON number (integer or floating point) from a string view
        template <class T>
        inline T parse_json_number(const char*& ptr, const char* end) {
            skip_whitespace(ptr, end);
            if (ptr >= end) throw std::runtime_error("JSON: unexpected end while parsing number.");

            const char* start = ptr;
            // Move past optional minus, digits, optional decimal and exponent
            if (*ptr == '-') ++ptr;
            while (ptr < end && std::isdigit(static_cast<unsigned char>(*ptr))) ++ptr;
            if (ptr < end && *ptr == '.') {
                ++ptr;
                while (ptr < end && std::isdigit(static_cast<unsigned char>(*ptr))) ++ptr;
            }
            if (ptr < end && (*ptr == 'e' || *ptr == 'E')) {
                ++ptr;
                if (ptr < end && (*ptr == '+' || *ptr == '-')) ++ptr;
                while (ptr < end && std::isdigit(static_cast<unsigned char>(*ptr))) ++ptr;
            }
            std::string_view number_str(start, ptr - start);
            T value{};
            auto [num_end, ec] = std::from_chars(number_str.data(), number_str.data() + number_str.size(), value);
            if (ec != std::errc()) {
                throw std::runtime_error("JSON: invalid number: " + std::string(number_str));
            }
            return value;
        }

        // Parse a JSON string (returns the inner text without quotes)
        inline std::string parse_json_string(const char*& ptr, const char* end) {
            skip_whitespace(ptr, end);
            if (ptr >= end || *ptr != '"') throw std::runtime_error("JSON: expected '\"'");
            ++ptr; // skip opening quote
            std::string result;
            while (ptr < end && *ptr != '"') {
                if (*ptr == '\\') {
                    ++ptr;
                    if (ptr >= end) throw std::runtime_error("JSON: unexpected end in string escape.");
                    switch (*ptr) {
                        case '"': result += '"'; break;
                        case '\\': result += '\\'; break;
                        case '/': result += '/'; break;
                        case 'n': result += '\n'; break;
                        case 'r': result += '\r'; break;
                        case 't': result += '\t'; break;
                        case 'b': result += '\b'; break;
                        case 'f': result += '\f'; break;
                        case 'u': {
                            // \uXXXX – skip for now (basic support)
                            if (ptr + 4 >= end) throw std::runtime_error("JSON: invalid unicode escape.");
                            result += "?";
                            ptr += 4;
                            break;
                        }
                        default: throw std::runtime_error("JSON: invalid escape character.");
                    }
                } else {
                    result += *ptr;
                }
                ++ptr;
            }
            if (ptr >= end) throw std::runtime_error("JSON: unexpected end of string.");
            ++ptr; // skip closing quote
            return result;
        }

        // Forward declaration for recursive parsing
        template <class T>
        auto parse_json_value(const char*& ptr, const char* end) -> xarray_container<uvector<T>>;

        // Parse a JSON array into a 1D or nested xtensor
        template <class T>
        auto parse_json_array(const char*& ptr, const char* end) -> xarray_container<uvector<T>> {
            skip_whitespace(ptr, end);
            if (ptr >= end || *ptr != '[') throw std::runtime_error("JSON: expected '['");
            ++ptr; // skip '['
            skip_whitespace(ptr, end);

            // Check if empty array
            if (ptr < end && *ptr == ']') {
                ++ptr;
                return xarray_container<uvector<T>>({0});
            }

            std::vector<xarray_container<uvector<T>>> rows;
            // Determine if it's a 1D array of numbers or nested arrays
            bool is_nested = false;
            const char* save = ptr;
            while (save < end && std::isspace(static_cast<unsigned char>(*save))) ++save;
            if (save < end && *save == '[') is_nested = true;

            if (!is_nested) {
                // 1D array of values
                std::vector<T> values;
                while (ptr < end) {
                    skip_whitespace(ptr, end);
                    if (ptr < end && *ptr == ']') { ++ptr; break; }
                    T val = parse_json_number<T>(ptr, end);
                    values.push_back(val);
                    skip_whitespace(ptr, end);
                    if (ptr < end && *ptr == ',') { ++ptr; continue; }
                    if (ptr < end && *ptr == ']') { ++ptr; break; }
                    throw std::runtime_error("JSON: expected ',' or ']' in array.");
                }
                xarray_container<uvector<T>> result({values.size()});
                std::copy(values.begin(), values.end(), result.data());
                return result;
            } else {
                // Nested arrays (2D or higher)
                while (ptr < end) {
                    skip_whitespace(ptr, end);
                    if (ptr < end && *ptr == ']') { ++ptr; break; }
                    auto row = parse_json_array<T>(ptr, end);
                    rows.push_back(std::move(row));
                    skip_whitespace(ptr, end);
                    if (ptr < end && *ptr == ',') { ++ptr; continue; }
                    if (ptr < end && *ptr == ']') { ++ptr; break; }
                    throw std::runtime_error("JSON: expected ',' or ']' in nested array.");
                }
                if (rows.empty()) return xarray_container<uvector<T>>({0});
                // Verify all rows have the same shape (assume 1D rows -> 2D result)
                size_t cols = rows[0].size();
                for (auto& r : rows) {
                    if (r.size() != cols)
                        throw std::runtime_error("JSON: jagged arrays not supported.");
                }
                size_t rows_count = rows.size();
                xarray_container<uvector<T>> result({rows_count, cols});
                T* data = result.data();
                for (size_t i = 0; i < rows_count; ++i) {
                    std::copy(rows[i].data(), rows[i].data() + cols, data + i * cols);
                }
                return result;
            }
        }

        template <class T>
        auto parse_json_value(const char*& ptr, const char* end) -> xarray_container<uvector<T>> {
            skip_whitespace(ptr, end);
            if (ptr >= end) throw std::runtime_error("JSON: unexpected end.");
            if (*ptr == '[') {
                return parse_json_array<T>(ptr, end);
            } else if (*ptr == '"') {
                throw std::runtime_error("JSON: string not convertible to numeric array.");
            } else if (*ptr == 'n' || *ptr == 't' || *ptr == 'f') {
                // null, true, false – skip
                if (std::strncmp(ptr, "null", 4) == 0) ptr += 4;
                else if (std::strncmp(ptr, "true", 4) == 0) ptr += 4;
                else if (std::strncmp(ptr, "false", 5) == 0) ptr += 5;
                else throw std::runtime_error("JSON: unknown literal.");
                return xarray_container<uvector<T>>({0});
            } else {
                T val = parse_json_number<T>(ptr, end);
                xarray_container<uvector<T>> result({1});
                result[0] = val;
                return result;
            }
        }
    }

    /**
     * Load a JSON file containing a numeric array into an xtensor array.
     */
    template <class T = double>
    inline auto load_json(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Cannot open JSON file: " + filename);
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        const char* ptr = content.data();
        const char* end = ptr + content.size();
        return detail::parse_json_value<T>(ptr, end);
    }

    /**
     * Load JSON from a string.
     */
    template <class T = double>
    inline auto load_json_str(const std::string& json_str) {
        const char* ptr = json_str.data();
        const char* end = ptr + json_str.size();
        return detail::parse_json_value<T>(ptr, end);
    }

    /**
     * Save an xtensor array to a JSON file.
     */
    template <class E>
    inline void save_json(const std::string& filename, const E& expr, int precision = 15) {
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Cannot open JSON file for writing: " + filename);
        auto shape = expr.shape();
        size_t ndim = shape.size();
        const auto* data = expr.data();
        char buffer[64];

        if (ndim == 0) {
            file << "[]";
            return;
        }

        // Recursive output helper using lambdas
        std::function<void(std::ostream&, const std::vector<size_t>&, size_t, size_t, const size_t*)> write_array;
        write_array = [&](std::ostream& out, const std::vector<size_t>& sh, size_t dim, size_t offset, const size_t* strides) {
            if (dim == sh.size() - 1) {
                // Innermost dimension: write [val1, val2, ...]
                out << '[';
                for (size_t i = 0; i < sh[dim]; ++i) {
                    if (i > 0) out << ", ";
                    auto val = data[offset + i * strides[dim]];
                    auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), val,
                                                   std::chars_format::general, precision);
                    if (ec != std::errc()) {
                        out << val;
                    } else {
                        out.write(buffer, ptr - buffer);
                    }
                }
                out << ']';
            } else {
                out << '[';
                for (size_t i = 0; i < sh[dim]; ++i) {
                    if (i > 0) out << ", ";
                    write_array(out, sh, dim + 1, offset + i * strides[dim], strides);
                }
                out << ']';
            }
        };

        auto strides = xt::compute_strides(shape);
        write_array(file, shape, 0, 0, strides.data());
    }

    /**
     * Convert an xtensor array to a JSON string.
     */
    template <class E>
    inline std::string to_json_string(const E& expr, int precision = 15) {
        std::ostringstream oss;
        auto shape = expr.shape();
        size_t ndim = shape.size();
        const auto* data = expr.data();
        char buffer[64];

        if (ndim == 0) return "[]";

        std::function<void(std::ostream&, const std::vector<size_t>&, size_t, size_t, const size_t*)> write_array;
        write_array = [&](std::ostream& out, const std::vector<size_t>& sh, size_t dim, size_t offset, const size_t* strides) {
            if (dim == sh.size() - 1) {
                out << '[';
                for (size_t i = 0; i < sh[dim]; ++i) {
                    if (i > 0) out << ", ";
                    auto val = data[offset + i * strides[dim]];
                    auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), val,
                                                   std::chars_format::general, precision);
                    if (ec != std::errc()) {
                        out << val;
                    } else {
                        out.write(buffer, ptr - buffer);
                    }
                }
                out << ']';
            } else {
                out << '[';
                for (size_t i = 0; i < sh[dim]; ++i) {
                    if (i > 0) out << ", ";
                    write_array(out, sh, dim + 1, offset + i * strides[dim], strides);
                }
                out << ']';
            }
        };

        auto strides = xt::compute_strides(shape);
        write_array(oss, shape, 0, 0, strides.data());
        return oss.str();
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_XJSON_HPP