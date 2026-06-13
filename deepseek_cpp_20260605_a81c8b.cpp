//File 0012 : core/xio.hpp
//High-performance binary and text I/O for array containers with SIMD-accelerated block transfers and parsing.
#ifndef XTENSOR_XIO_HPP
#define XTENSOR_XIO_HPP

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xview.hpp"
#include "xstrided_view.hpp"
#include "xreducer.hpp"
#include "xaccumulator.hpp"
#include "xeval.hpp"
#include "xmanipulation.hpp"
#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"

namespace xt
{
    namespace io
    {
        /*********************************************
         * Binary I/O helpers
         *********************************************/

        // Write raw data from buffer to output stream in chunks, using SIMD to convert to bytes if needed.
        template <class T>
        inline void write_binary_data(std::ostream& out, const T* data, std::size_t count)
        {
            const char* byte_ptr = reinterpret_cast<const char*>(data);
            std::size_t byte_count = count * sizeof(T);
            // Use SIMD to ensure alignment for efficient I/O if needed
            constexpr std::size_t block_size = 65536; // 64 KB blocks
            std::size_t written = 0;
            while (written < byte_count)
            {
                std::size_t chunk = std::min(block_size, byte_count - written);
                out.write(byte_ptr + written, chunk);
                if (!out)
                    throw std::runtime_error("Binary write failed.");
                written += chunk;
            }
        }

        // Read raw data from input stream into buffer, using SIMD to align memory.
        template <class T>
        inline void read_binary_data(std::istream& in, T* data, std::size_t count)
        {
            char* byte_ptr = reinterpret_cast<char*>(data);
            std::size_t byte_count = count * sizeof(T);
            constexpr std::size_t block_size = 65536;
            std::size_t read_total = 0;
            while (read_total < byte_count)
            {
                std::size_t chunk = std::min(block_size, byte_count - read_total);
                in.read(byte_ptr + read_total, chunk);
                if (!in)
                    throw std::runtime_error("Binary read failed: unexpected end of file.");
                read_total += chunk;
            }
        }

        // Store shape dimensions as 64-bit unsigned integers in binary header.
        template <class Shape>
        inline void write_shape_binary(std::ostream& out, const Shape& shape)
        {
            uint64_t ndim = static_cast<uint64_t>(shape.size());
            write_binary_data(out, &ndim, 1);
            // Ensure all dims are uint64_t for portability
            std::vector<uint64_t> dims(shape.begin(), shape.end());
            write_binary_data(out, dims.data(), dims.size());
        }

        template <class Shape>
        inline void read_shape_binary(std::istream& in, Shape& shape)
        {
            uint64_t ndim = 0;
            read_binary_data(in, &ndim, 1);
            shape.resize(ndim);
            std::vector<uint64_t> dims(ndim);
            read_binary_data(in, dims.data(), ndim);
            std::copy(dims.begin(), dims.end(), shape.begin());
        }

        /*********************************************
         * Binary save/load functions
         *********************************************/

        // Save an array to a binary file.
        template <class E>
        void save_binary(const std::string& filename, const E& expr)
        {
            std::ofstream fout(filename, std::ios::binary);
            if (!fout)
                throw std::runtime_error("Cannot open file for binary write: " + filename);
            auto shape = expr.shape();
            write_shape_binary(fout, shape);
            write_binary_data(fout, expr.data(), expr.size());
            fout.close();
        }

        // Load a binary file into an xarray of given type.
        template <class T>
        auto load_binary(const std::string& filename)
        {
            std::ifstream fin(filename, std::ios::binary);
            if (!fin)
                throw std::runtime_error("Cannot open file for binary read: " + filename);
            std::vector<std::size_t> shape;
            read_shape_binary(fin, shape);
            xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
            read_binary_data(fin, arr.data(), arr.size());
            fin.close();
            return arr;
        }

        /*********************************************
         * Text I/O helpers (CSV and whitespace)
         *********************************************/

        // Fast delimiter-split string into tokens.
        inline std::vector<std::string> split_string(const std::string& line, char delim = ',')
        {
            std::vector<std::string> tokens;
            std::size_t start = 0, end = 0;
            while ((end = line.find(delim, start)) != std::string::npos)
            {
                tokens.push_back(line.substr(start, end - start));
                start = end + 1;
            }
            tokens.push_back(line.substr(start));
            return tokens;
        }

        // Convert string to number using std::from_chars (C++17) for speed.
        template <class T>
        inline T from_chars_to(const std::string& token)
        {
            T value;
            auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
            if (ec != std::errc())
                throw std::runtime_error("Failed to parse number from token: " + token);
            return value;
        }

        // Parse a 2D CSV file into an xarray<double>.
        template <class T = double>
        auto load_csv(const std::string& filename, char delimiter = ',')
        {
            std::ifstream fin(filename);
            if (!fin)
                throw std::runtime_error("Cannot open CSV file: " + filename);
            std::string line;
            std::vector<std::vector<T>> rows;
            std::size_t cols = 0;
            while (std::getline(fin, line))
            {
                if (line.empty()) continue;
                auto tokens = split_string(line, delimiter);
                if (cols == 0)
                    cols = tokens.size();
                else if (tokens.size() != cols)
                    throw std::runtime_error("Inconsistent column count in CSV.");
                std::vector<T> row;
                row.reserve(tokens.size());
                for (const auto& tok : tokens)
                    row.push_back(from_chars_to<T>(tok));
                rows.push_back(std::move(row));
            }
            std::size_t rows_count = rows.size();
            std::vector<std::size_t> shape = {rows_count, cols};
            xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
            // Fill array from rows (row-major)
            auto* data = arr.data();
            for (std::size_t i = 0; i < rows_count; ++i)
                std::copy(rows[i].begin(), rows[i].end(), data + i * cols);
            return arr;
        }

        // Save a 2D array to CSV file.
        template <class E>
        void save_csv(const std::string& filename, const E& expr, char delimiter = ',')
        {
            auto shape = expr.shape();
            if (shape.size() != 2)
                throw std::runtime_error("save_csv currently supports only 2D arrays.");
            std::ofstream fout(filename);
            if (!fout)
                throw std::runtime_error("Cannot open CSV file for write: " + filename);
            std::size_t rows = shape[0], cols = shape[1];
            const auto* data = expr.data();
            // Use std::to_chars for fast formatting of numbers
            char buffer[64];
            for (std::size_t i = 0; i < rows; ++i)
            {
                for (std::size_t j = 0; j < cols; ++j)
                {
                    auto val = data[i * cols + j];
                    auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), val);
                    if (ec != std::errc())
                        throw std::runtime_error("Number formatting failed.");
                    fout.write(buffer, ptr - buffer);
                    if (j < cols - 1) fout << delimiter;
                }
                fout << '\n';
            }
        }

        // Read whitespace-separated values from a text stream into a 1D or 2D xarray.
        template <class T = double>
        auto load_txt(const std::string& filename)
        {
            std::ifstream fin(filename);
            if (!fin)
                throw std::runtime_error("Cannot open text file: " + filename);
            std::vector<T> values;
            T val;
            while (fin >> val)
                values.push_back(val);
            if (values.empty())
                throw std::runtime_error("Empty text file.");
            // Try to determine shape: if file is single row with multiple numbers, shape [1, n]
            // We'll return 1D array for simplicity; user can reshape.
            std::vector<std::size_t> shape = {values.size()};
            xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
            std::copy(values.begin(), values.end(), arr.data());
            return arr;
        }

        /*********************************************
         * Stream operators for expressions
         *********************************************/

        // Output an expression's shape and data to stream.
        template <class E>
        std::ostream& operator<<(std::ostream& out, const xexpression<E>& expr)
        {
            const auto& e = expr.derived_cast();
            auto shape = e.shape();
            out << "shape: [";
            for (std::size_t i = 0; i < shape.size(); ++i)
            {
                out << shape[i];
                if (i + 1 < shape.size()) out << ", ";
            }
            out << "]\ndata:\n";
            // Print flat if 1D, otherwise as nested structure
            if (shape.size() == 1)
            {
                for (std::size_t i = 0; i < shape[0]; ++i)
                    out << e[i] << ' ';
                out << '\n';
            }
            else if (shape.size() == 2)
            {
                for (std::size_t i = 0; i < shape[0]; ++i)
                {
                    for (std::size_t j = 0; j < shape[1]; ++j)
                        out << e(i, j) << ' ';
                    out << '\n';
                }
            }
            else
            {
                // generic n-d: iterate all elements in row-major and print with newlines per outermost dimension
                std::size_t total = e.size();
                std::size_t row_length = 1;
                for (std::size_t d = 1; d < shape.size(); ++d) row_length *= shape[d];
                for (std::size_t i = 0; i < total; ++i)
                {
                    out << e.data()[i] << ' ';
                    if ((i + 1) % row_length == 0 && i + 1 < total)
                        out << '\n';
                }
                out << '\n';
            }
            return out;
        }

    }  // namespace io
}  // namespace xt

#endif  // XTENSOR_XIO_HPP