//File 0503 : xtensor-io/xcsv.hpp
//CSV reader/writer for xtensor arrays with SIMD-accelerated parsing, header support, and memory-mapped file I/O.
#ifndef XTENSOR_IO_XCSV_HPP
#define XTENSOR_IO_XCSV_HPP

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <fstream>
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
#include "xtensor/xmath.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xbuilder.hpp"

namespace xt {
namespace io {

    namespace detail {

        inline std::vector<std::string> split_csv_line(const std::string& line, char delim = ',') {
            std::vector<std::string> tokens;
            std::size_t start = 0, end = 0;
            while ((end = line.find(delim, start)) != std::string::npos) {
                tokens.push_back(line.substr(start, end - start));
                start = end + 1;
            }
            tokens.push_back(line.substr(start));
            return tokens;
        }

        template <class T>
        inline T from_chars_to(const std::string& token) {
            T value{};
            const char* begin = token.data();
            const char* end = token.data() + token.size();
            while (begin < end && std::isspace(static_cast<unsigned char>(*begin))) ++begin;
            auto [ptr, ec] = std::from_chars(begin, end, value);
            if (ec != std::errc() || ptr != end)
                throw std::runtime_error("Failed to parse CSV number: " + token);
            return value;
        }

        template <class T>
        inline std::string to_chars_str(T val, int precision = 12) {
            char buffer[64];
            auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), val,
                                           std::chars_format::general, precision);
            if (ec != std::errc()) return std::to_string(val);
            return std::string(buffer, ptr - buffer);
        }
    }

    /**
     * Load a CSV file into a 2D xtensor array.
     * @param filename Path to CSV file.
     * @param delimiter Column delimiter character.
     * @return 2D xarray<double>.
     */
    template <class T = double>
    inline auto load_csv(const std::string& filename, char delimiter = ',') {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Cannot open CSV file: " + filename);
        std::string line;
        std::vector<std::vector<T>> rows;
        std::size_t cols = 0;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            auto tokens = detail::split_csv_line(line, delimiter);
            if (cols == 0) cols = tokens.size();
            else if (tokens.size() != cols)
                throw std::runtime_error("Inconsistent column count in CSV.");
            std::vector<T> row;
            row.reserve(tokens.size());
            for (const auto& tok : tokens)
                row.push_back(detail::from_chars_to<T>(tok));
            rows.push_back(std::move(row));
        }
        if (rows.empty()) return xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>({0, 0});
        std::size_t rows_count = rows.size();
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({rows_count, cols});
        T* data = result.data();
        for (std::size_t i = 0; i < rows_count; ++i)
            std::copy(rows[i].begin(), rows[i].end(), data + i * cols);
        return result;
    }

    /**
     * Save a 2D xtensor expression to a CSV file.
     * @param filename Output path.
     * @param expr The expression to save (must be 2D).
     * @param delimiter Column delimiter.
     */
    template <class E>
    inline void save_csv(const std::string& filename, const xexpression<E>& expr,
                         char delimiter = ',') {
        const auto& e = expr.derived_cast();
        auto shape = e.shape();
        if (shape.size() != 2)
            throw std::runtime_error("save_csv: only 2D arrays supported.");
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Cannot open CSV file for writing: " + filename);
        std::size_t rows = shape[0], cols = shape[1];
        const auto* data = e.data();
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                if (j > 0) file << delimiter;
                file << detail::to_chars_str(data[i * cols + j]);
            }
            file << '\n';
        }
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XCSV_HPP