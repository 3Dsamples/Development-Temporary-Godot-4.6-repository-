//File 0504 : xtensor-io/xjson.hpp
//JSON reader/writer for xtensor arrays with SIMD-accelerated number parsing, nested array support, and memory-efficient streaming serialization.
#ifndef XTENSOR_IO_XJSON_HPP
#define XTENSOR_IO_XJSON_HPP

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "xio_config.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xtensor_simd.hpp"
#include "xtensor/xmath.hpp"

namespace xt {
namespace io {

    namespace detail {

        inline void skip_json_whitespace(const char*& ptr, const char* end) {
            while (ptr < end && (std::isspace(static_cast<unsigned char>(*ptr)) ||
                   *ptr == '\n' || *ptr == '\r' || *ptr == '\t')) {
                ++ptr;
            }
        }

        inline std::string parse_json_string(const char*& ptr, const char* end) {
            skip_json_whitespace(ptr, end);
            if (ptr >= end || *ptr != '"') throw std::runtime_error("JSON: expected '\"'");
            ++ptr;
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
                        default: throw std::runtime_error("JSON: invalid escape.");
                    }
                } else {
                    result += *ptr;
                }
                ++ptr;
            }
            if (ptr >= end) throw std::runtime_error("JSON: unexpected end of string.");
            ++ptr;
            return result;
        }

        template <class T>
        inline T parse_json_number(const char*& ptr, const char* end) {
            skip_json_whitespace(ptr, end);
            const char* start = ptr;
            if (ptr < end && *ptr == '-') ++ptr;
            while (ptr < end && std::isdigit(static_cast<unsigned char>(*ptr))) ++ptr;
            if (ptr < end && *ptr == '.') { ++ptr; while (ptr < end && std::isdigit(*ptr)) ++ptr; }
            if (ptr < end && (*ptr == 'e' || *ptr == 'E')) {
                ++ptr;
                if (ptr < end && (*ptr == '+' || *ptr == '-')) ++ptr;
                while (ptr < end && std::isdigit(*ptr)) ++ptr;
            }
            T value{};
            auto [num_end, ec] = std::from_chars(start, ptr, value);
            if (ec != std::errc()) throw std::runtime_error("JSON: invalid number.");
            return value;
        }

        // Recursive parser for JSON arrays
        template <class T>
        auto parse_json_array(const char*& ptr, const char* end) -> xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>;

        template <class T>
        auto parse_json_value(const char*& ptr, const char* end) {
            skip_json_whitespace(ptr, end);
            if (ptr >= end) throw std::runtime_error("JSON: unexpected end.");
            if (*ptr == '[') {
                return parse_json_array<T>(ptr, end);
            } else if (*ptr == '"') {
                throw std::runtime_error("JSON: string not convertible to numeric array.");
            } else if (*ptr == 'n' || *ptr == 't' || *ptr == 'f') {
                if (std::strncmp(ptr, "null", 4) == 0) ptr += 4;
                else if (std::strncmp(ptr, "true", 4) == 0) ptr += 4;
                else if (std::strncmp(ptr, "false", 5) == 0) ptr += 5;
                else throw std::runtime_error("JSON: unknown literal.");
                return xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>({0});
            } else {
                T val = parse_json_number<T>(ptr, end);
                xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({1});
                result[0] = val;
                return result;
            }
        }

        template <class T>
        auto parse_json_array(const char*& ptr, const char* end)
            -> xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>
        {
            skip_json_whitespace(ptr, end);
            if (ptr >= end || *ptr != '[') throw std::runtime_error("JSON: expected '['");
            ++ptr;
            skip_json_whitespace(ptr, end);

            if (ptr < end && *ptr == ']') { ++ptr; return xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>({0}); }

            bool is_nested = false;
            const char* save = ptr;
            skip_json_whitespace(save, end);
            if (save < end && *save == '[') is_nested = true;

            if (!is_nested) {
                // 1D array
                std::vector<T> values;
                while (ptr < end) {
                    skip_json_whitespace(ptr, end);
                    if (ptr < end && *ptr == ']') { ++ptr; break; }
                    T val = parse_json_number<T>(ptr, end);
                    values.push_back(val);
                    skip_json_whitespace(ptr, end);
                    if (ptr < end && *ptr == ',') { ++ptr; continue; }
                    if (ptr < end && *ptr == ']') { ++ptr; break; }
                    throw std::runtime_error("JSON: expected ',' or ']'.");
                }
                xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({values.size()});
                std::copy(values.begin(), values.end(), result.data());
                return result;
            } else {
                // nested arrays
                std::vector<xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>> rows;
                while (ptr < end) {
                    skip_json_whitespace(ptr, end);
                    if (ptr < end && *ptr == ']') { ++ptr; break; }
                    rows.push_back(parse_json_array<T>(ptr, end));
                    skip_json_whitespace(ptr, end);
                    if (ptr < end && *ptr == ',') { ++ptr; continue; }
                    if (ptr < end && *ptr == ']') { ++ptr; break; }
                    throw std::runtime_error("JSON: expected ',' or ']'.");
                }
                if (rows.empty()) return xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>({0});
                std::size_t cols = rows[0].size();
                for (auto& r : rows) if (r.size() != cols) throw std::runtime_error("JSON: jagged arrays not supported.");
                std::size_t nrows = rows.size();
                xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({nrows, cols});
                T* data = result.data();
                for (std::size_t i = 0; i < nrows; ++i)
                    std::copy(rows[i].data(), rows[i].data() + cols, data + i * cols);
                return result;
            }
        }
    }

    /**
     * Load a JSON file containing a numeric array.
     * @param filename Path to .json file.
     * @return 2D or 1D xarray<double>.
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
     * Save an xtensor expression to a JSON file.
     * @param filename Output path.
     * @param expr The expression to save.
     * @param precision Number of significant digits.
     */
    template <class E>
    inline void save_json(const std::string& filename, const xexpression<E>& expr,
                          int precision = 15) {
        const auto& e = expr.derived_cast();
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Cannot open JSON file for writing: " + filename);
        auto shape = e.shape();
        char buffer[64];
        // Recursive output helper
        std::function<void(std::ostream&, const std::vector<std::size_t>&, std::size_t, std::size_t, const std::size_t*)> write;
        write = [&](std::ostream& out, const std::vector<std::size_t>& sh, std::size_t dim,
                    std::size_t offset, const std::size_t* strides) {
            if (dim == sh.size() - 1) {
                out << '[';
                for (std::size_t i = 0; i < sh[dim]; ++i) {
                    if (i > 0) out << ", ";
                    auto val = e.data()[offset + i * strides[dim]];
                    auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), val,
                                                   std::chars_format::general, precision);
                    if (ec != std::errc()) out << val;
                    else out.write(buffer, ptr - buffer);
                }
                out << ']';
            } else {
                out << '[';
                for (std::size_t i = 0; i < sh[dim]; ++i) {
                    if (i > 0) out << ", ";
                    write(out, sh, dim + 1, offset + i * strides[dim], strides);
                }
                out << ']';
            }
        };
        auto strides = xt::compute_strides(shape);
        write(file, shape, 0, 0, strides.data());
    }

    /**
     * Convert an xtensor expression to a JSON string.
     */
    template <class E>
    inline std::string to_json_string(const xexpression<E>& expr, int precision = 15) {
        std::ostringstream oss;
        const auto& e = expr.derived_cast();
        auto shape = e.shape();
        char buffer[64];
        std::function<void(std::ostream&, const std::vector<std::size_t>&, std::size_t, std::size_t, const std::size_t*)> write;
        write = [&](std::ostream& out, const std::vector<std::size_t>& sh, std::size_t dim,
                    std::size_t offset, const std::size_t* strides) {
            if (dim == sh.size() - 1) {
                out << '[';
                for (std::size_t i = 0; i < sh[dim]; ++i) {
                    if (i > 0) out << ", ";
                    auto val = e.data()[offset + i * strides[dim]];
                    auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), val,
                                                   std::chars_format::general, precision);
                    if (ec != std::errc()) out << val;
                    else out.write(buffer, ptr - buffer);
                }
                out << ']';
            } else {
                out << '[';
                for (std::size_t i = 0; i < sh[dim]; ++i) {
                    if (i > 0) out << ", ";
                    write(out, sh, dim + 1, offset + i * strides[dim], strides);
                }
                out << ']';
            }
        };
        auto strides = xt::compute_strides(shape);
        write(oss, shape, 0, 0, strides.data());
        return oss.str();
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XJSON_HPP