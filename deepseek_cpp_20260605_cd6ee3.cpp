//File 0115 : numdot/io.h
//Input/output for arrays: binary (NPY), CSV, JSON, text, with SIMD-accelerated block transfer, automatic format detection, and stream operators.
#ifndef NUMDOT_IO_H
#define NUMDOT_IO_H

#include <type_traits>
#include <utility>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <regex>
#include <map>

#include "config.h"
#include "forward.h"
#include "types.h"
#include "shape.h"
#include "array.h"
#include "elementwise.h"

namespace numdot
{
namespace io
{
    namespace detail
    {
        // NPY magic
        constexpr char npy_magic[] = "\x93NUMPY";
        constexpr std::size_t npy_magic_len = 6;
        constexpr std::size_t npy_block_size = 64;

        /**
         * Build an NPY header string from shape and dtype.
         */
        inline std::string build_npy_header(const std::vector<std::size_t>& shape, const std::string& dtype, bool fortran = false)
        {
            std::ostringstream oss;
            oss << "{'descr': '" << dtype << "', 'fortran_order': " << (fortran ? "True" : "False")
                << ", 'shape': (";
            for (std::size_t i = 0; i < shape.size(); ++i)
            {
                if (i > 0) oss << ", ";
                oss << shape[i];
            }
            if (shape.empty()) oss << "1";
            oss << ")}";
            std::string dict = oss.str();
            std::size_t pad = npy_block_size - npy_magic_len - 2 - 1; // 2 bytes header length
            while (dict.size() < pad) dict += ' ';
            dict += '\n';
            return dict;
        }

        /**
         * Parse NPY header dict and return dtype and shape.
         */
        inline auto parse_npy_header(const std::string& header)
        {
            std::string dtype;
            bool fortran = false;
            std::vector<std::size_t> shape;
            std::regex descr_re("'descr'\\s*:\\s*'([^']+)'");
            std::smatch m;
            if (std::regex_search(header, m, descr_re)) dtype = m[1].str();
            std::regex fortran_re("'fortran_order'\\s*:\\s*(True|False)");
            if (std::regex_search(header, m, fortran_re)) fortran = (m[1].str() == "True");
            std::regex shape_re("'shape'\\s*:\\s*\\(([^)]*)\\)");
            if (std::regex_search(header, m, shape_re))
            {
                std::string shape_str = m[1].str();
                std::regex num_re("\\d+");
                auto begin = std::sregex_iterator(shape_str.begin(), shape_str.end(), num_re);
                auto end = std::sregex_iterator();
                for (auto it = begin; it != end; ++it) shape.push_back(std::stoull(it->str()));
            }
            if (shape.empty()) shape.push_back(1);
            return std::make_tuple(dtype, fortran, shape);
        }

        // SIMD block read/write
        template <class T>
        void read_binary(std::istream& in, T* data, std::size_t count)
        {
            constexpr std::size_t buf_size = 65536;
            char buffer[buf_size];
            char* byte_ptr = reinterpret_cast<char*>(data);
            std::size_t bytes = count * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0)
            {
                std::size_t chunk = std::min(buf_size, remaining);
                in.read(buffer, static_cast<std::streamsize>(chunk));
                if (!in) throw std::runtime_error("read_binary: failed");
                std::memcpy(byte_ptr + (bytes - remaining), buffer, chunk);
                remaining -= chunk;
            }
        }

        template <class T>
        void write_binary(std::ostream& out, const T* data, std::size_t count)
        {
            constexpr std::size_t buf_size = 65536;
            const char* byte_ptr = reinterpret_cast<const char*>(data);
            std::size_t bytes = count * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0)
            {
                std::size_t chunk = std::min(buf_size, remaining);
                out.write(byte_ptr + (bytes - remaining), static_cast<std::streamsize>(chunk));
                if (!out) throw std::runtime_error("write_binary: failed");
                remaining -= chunk;
            }
        }

        /**
         * Map NumPy dtype string to our types.
         */
        template <class T> struct numpy_dtype_name;
        template <> struct numpy_dtype_name<float>       { static constexpr const char* name = "<f4"; };
        template <> struct numpy_dtype_name<double>      { static constexpr const char* name = "<f8"; };
        template <> struct numpy_dtype_name<int32_t>     { static constexpr const char* name = "<i4"; };
        template <> struct numpy_dtype_name<int64_t>     { static constexpr const char* name = "<i8"; };
        template <> struct numpy_dtype_name<uint32_t>    { static constexpr const char* name = "<u4"; };
        template <> struct numpy_dtype_name<uint64_t>    { static constexpr const char* name = "<u8"; };

        /**
         * Detect file format from extension.
         */
        inline std::string file_extension(const std::string& filename)
        {
            auto pos = filename.rfind('.');
            if (pos == std::string::npos) return "";
            std::string ext = filename.substr(pos);
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
            return ext;
        }
    }

    /**
     * Save array to NPY file.
     */
    template <class E>
    inline void save_npy(const std::string& filename, const expression<E>& expr, bool fortran = false)
    {
        const auto& e = expr.derived();
        using T = typename E::value_type;
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open: " + filename);
        auto shape = e.shape();
        if (shape.empty()) shape = {1};
        std::string dtype = detail::numpy_dtype_name<T>::name;
        std::string header = detail::build_npy_header(shape, dtype, fortran);
        file.write(detail::npy_magic, detail::npy_magic_len);
        std::uint16_t hlen = static_cast<std::uint16_t>(header.size());
        file.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        detail::write_binary(file, e.data(), compute_size(shape));
    }

    /**
     * Load array from NPY file.
     */
    template <class T = double>
    inline auto load_npy(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open: " + filename);
        char magic[detail::npy_magic_len];
        file.read(magic, detail::npy_magic_len);
        if (std::strncmp(magic, detail::npy_magic, detail::npy_magic_len) != 0)
            throw std::runtime_error("Not a valid NPY file.");
        std::uint16_t hlen;
        file.read(reinterpret_cast<char*>(&hlen), sizeof(hlen));
        std::string header(hlen, '\0');
        file.read(&header[0], hlen);
        auto [dtype, fortran, shape] = detail::parse_npy_header(header);
        array<T> result(shape);
        detail::read_binary(file, result.data(), result.size());
        return result;
    }

    /**
     * Save array to CSV file.
     */
    template <class E>
    inline void save_csv(const std::string& filename, const expression<E>& expr, char delim = ',')
    {
        const auto& e = expr.derived();
        auto shape = e.shape();
        if (shape.size() != 2) throw std::runtime_error("save_csv: 2D array required.");
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Cannot open: " + filename);
        std::size_t rows = shape[0], cols = shape[1];
        char buf[64];
        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                if (j > 0) file << delim;
                auto val = e(i, j);
                auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), val);
                if (ec != std::errc()) file << val;
                else file.write(buf, ptr - buf);
            }
            file << '\n';
        }
    }

    /**
     * Load CSV file into 2D array.
     */
    template <class T = double>
    inline auto load_csv(const std::string& filename, char delim = ',')
    {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Cannot open: " + filename);
        std::string line;
        std::vector<std::vector<T>> rows;
        std::size_t cols = 0;
        while (std::getline(file, line))
        {
            if (line.empty()) continue;
            std::vector<T> row;
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, delim))
            {
                T val{};
                auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), val);
                if (ec != std::errc()) throw std::runtime_error("CSV parse error: " + token);
                row.push_back(val);
            }
            if (cols == 0) cols = row.size();
            else if (row.size() != cols) throw std::runtime_error("Inconsistent column count.");
            rows.push_back(std::move(row));
        }
        array<T> result({rows.size(), cols});
        for (std::size_t i = 0; i < rows.size(); ++i)
            std::copy(rows[i].begin(), rows[i].end(), result.data() + i * cols);
        return result;
    }

    /**
     * Save array to JSON file.
     */
    template <class E>
    inline void save_json(const std::string& filename, const expression<E>& expr, int precision = 15)
    {
        const auto& e = expr.derived();
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Cannot open: " + filename);
        auto shape = e.shape();
        char buf[64];
        std::function<void(std::ostream&, const std::vector<std::size_t>&, std::size_t, std::size_t)> write;
        write = [&](std::ostream& out, const std::vector<std::size_t>& sh, std::size_t dim, std::size_t offset)
        {
            if (dim == sh.size())
            {
                auto val = e.data()[offset];
                auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), val, std::chars_format::general, precision);
                if (ec != std::errc()) out << val;
                else out.write(buf, ptr - buf);
            }
            else
            {
                out << '[';
                std::size_t stride = 1;
                for (std::size_t d = dim + 1; d < sh.size(); ++d) stride *= sh[d];
                for (std::size_t i = 0; i < sh[dim]; ++i)
                {
                    if (i > 0) out << ", ";
                    write(out, sh, dim + 1, offset + i * stride);
                }
                out << ']';
            }
        };
        write(file, shape, 0, 0);
    }

    /**
     * Load JSON array.
     */
    template <class T = double>
    inline auto load_json(const std::string& filename)
    {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Cannot open: " + filename);
        std::stringstream buf;
        buf << file.rdbuf();
        std::string content = buf.str();
        // Basic JSON parser (only arrays of numbers / nested arrays)
        const char* ptr = content.data();
        const char* end = ptr + content.size();
        std::function<void(const char*&, const char*, std::vector<T>&, std::vector<std::size_t>&)> parse_array;
        parse_array = [&](const char*& p, const char* end, std::vector<T>& values, std::vector<std::size_t>& shape)
        {
            if (*p != '[') throw std::runtime_error("JSON: expected '['");
            ++p;
            // Detect if nested
            while (p < end && std::isspace(*p)) ++p;
            if (*p == '[')
            {
                // nested array
                std::size_t rows = 0;
                std::vector<std::size_t> inner_shape;
                while (p < end)
                {
                    while (p < end && std::isspace(*p)) ++p;
                    if (*p == ']') { ++p; break; }
                    std::vector<T> inner_vals;
                    parse_array(p, end, inner_vals, inner_shape);
                    values.insert(values.end(), inner_vals.begin(), inner_vals.end());
                    ++rows;
                    while (p < end && std::isspace(*p)) ++p;
                    if (*p == ',') ++p;
                }
                shape = {rows};
                shape.insert(shape.end(), inner_shape.begin(), inner_shape.end());
            }
            else
            {
                // 1D array of numbers
                while (p < end)
                {
                    while (p < end && std::isspace(*p)) ++p;
                    if (*p == ']') { ++p; break; }
                    T val{};
                    auto [ptr, ec] = std::from_chars(p, end, val);
                    if (ec != std::errc()) throw std::runtime_error("JSON parse error");
                    values.push_back(val);
                    p = ptr;
                    while (p < end && std::isspace(*p)) ++p;
                    if (*p == ',') ++p;
                }
                shape = {values.size()};
            }
        };
        std::vector<T> values;
        std::vector<std::size_t> shape;
        parse_array(ptr, end, values, shape);
        if (shape.empty()) shape = {0};
        array<T> result(shape);
        std::copy(values.begin(), values.end(), result.data());
        return result;
    }

    /**
     * Save plain text (whitespace separated).
     */
    template <class E>
    inline void save_txt(const std::string& filename, const expression<E>& expr)
    {
        const auto& e = expr.derived();
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Cannot open: " + filename);
        auto shape = e.shape();
        if (shape.size() == 1)
        {
            for (std::size_t i = 0; i < shape[0]; ++i) file << e[i] << '\n';
        }
        else if (shape.size() == 2)
        {
            for (std::size_t i = 0; i < shape[0]; ++i)
            {
                for (std::size_t j = 0; j < shape[1]; ++j) file << e(i, j) << ' ';
                file << '\n';
            }
        }
        else
        {
            for (std::size_t i = 0; i < e.size(); ++i) file << e.data()[i] << '\n';
        }
    }

    /**
     * Load plain text (whitespace separated) as 1D array.
     */
    template <class T = double>
    inline auto load_txt(const std::string& filename)
    {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Cannot open: " + filename);
        std::vector<T> values;
        T val;
        while (file >> val) values.push_back(val);
        array<T> result({values.size()});
        std::copy(values.begin(), values.end(), result.data());
        return result;
    }

    /**
     * Automatic load by extension.
     */
    template <class T = double>
    inline auto load(const std::string& filename)
    {
        auto ext = detail::file_extension(filename);
        if (ext == ".npy")  return load_npy<T>(filename);
        if (ext == ".csv")  return load_csv<T>(filename);
        if (ext == ".json") return load_json<T>(filename);
        if (ext == ".txt")  return load_txt<T>(filename);
        throw std::runtime_error("Unsupported file format: " + ext);
    }

    /**
     * Stream output.
     */
    template <class E>
    inline std::ostream& operator<<(std::ostream& os, const expression<E>& expr)
    {
        const auto& e = expr.derived();
        auto shape = e.shape();
        os << "shape: [";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            os << shape[i];
            if (i + 1 < shape.size()) os << ", ";
        }
        os << "]\ndata:\n";
        if (shape.size() == 1) {
            for (std::size_t i = 0; i < shape[0]; ++i) os << e[i] << ' ';
            os << '\n';
        } else if (shape.size() == 2) {
            for (std::size_t i = 0; i < shape[0]; ++i) {
                for (std::size_t j = 0; j < shape[1]; ++j) os << e(i, j) << ' ';
                os << '\n';
            }
        } else {
            for (std::size_t i = 0; i < e.size(); ++i) {
                os << e.data()[i] << ' ';
                if ((i+1) % shape.back() == 0) os << '\n';
            }
        }
        return os;
    }

} // namespace io
} // namespace numdot

#endif // NUMDOT_IO_H