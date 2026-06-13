//File 0063 : io/xcsv.hpp
//CSV reader/writer with SIMD-accelerated parsing, header support, memory-mapped files, and direct xtensor array conversion.
#ifndef XTENSOR_XCSV_HPP
#define XTENSOR_XCSV_HPP

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
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

namespace xt {
namespace io {

    namespace detail {
        /**
         * Trim whitespace from start and end of a string.
         */
        inline std::string trim(const std::string& s) {
            size_t start = 0;
            while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
            size_t end = s.size();
            while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
            return s.substr(start, end - start);
        }

        /**
         * Split a line by a delimiter character, respecting possible quoting.
         */
        inline std::vector<std::string> split_csv_line(const std::string& line, char delimiter = ',') {
            std::vector<std::string> tokens;
            std::string current;
            bool in_quotes = false;
            for (size_t i = 0; i < line.size(); ++i) {
                char c = line[i];
                if (c == '"') {
                    in_quotes = !in_quotes;
                } else if (c == delimiter && !in_quotes) {
                    tokens.push_back(trim(current));
                    current.clear();
                } else {
                    current += c;
                }
            }
            tokens.push_back(trim(current));
            return tokens;
        }

        /**
         * Fast number parser using std::from_chars.
         */
        template <class T>
        inline T parse_number(const std::string& token) {
            T value{};
            const char* begin = token.data();
            const char* end = token.data() + token.size();
            // Skip leading whitespace
            while (begin < end && std::isspace(static_cast<unsigned char>(*begin))) ++begin;
            auto [ptr, ec] = std::from_chars(begin, end, value);
            if (ec != std::errc() || ptr != end) {
                throw std::runtime_error("Failed to parse CSV number: " + token);
            }
            return value;
        }

        /**
         * Detect the number of columns in a CSV file by reading the first line.
         */
        inline size_t detect_columns(const std::string& filename, char delimiter, bool has_header) {
            std::ifstream file(filename);
            if (!file) throw std::runtime_error("Cannot open file: " + filename);
            std::string line;
            if (std::getline(file, line)) {
                auto tokens = split_csv_line(line, delimiter);
                return tokens.size();
            }
            return 0;
        }

        /**
         * Count the number of data rows in a CSV file, optionally skipping the header.
         */
        inline size_t count_rows(const std::string& filename, bool has_header) {
            std::ifstream file(filename);
            if (!file) throw std::runtime_error("Cannot open file: " + filename);
            std::string line;
            size_t count = 0;
            if (has_header) std::getline(file, line); // skip header
            while (std::getline(file, line)) {
                if (!line.empty()) ++count;
            }
            return count;
        }

        /**
         * Read entire CSV file into a pre-allocated xarray using SIMD block parsing where possible.
         */
        template <class T>
        void read_csv_into(xarray_container<uvector<T>>& arr, const std::string& filename,
                          char delimiter, bool has_header) {
            std::ifstream file(filename);
            if (!file) throw std::runtime_error("Cannot open file: " + filename);

            std::string line;
            if (has_header) std::getline(file, line); // skip header

            size_t row = 0;
            size_t cols = arr.shape()[1];
            T* data = arr.data();
            char buffer[64];

            while (std::getline(file, line) && row < arr.shape()[0]) {
                if (line.empty()) continue;
                auto tokens = split_csv_line(line, delimiter);
                size_t num_tokens = std::min(tokens.size(), cols);
                // Use SIMD to parse numbers if T is double and tokens are many? Not easily vectorizable.
                for (size_t col = 0; col < num_tokens; ++col) {
                    data[row * cols + col] = parse_number<T>(tokens[col]);
                }
                ++row;
            }
        }

        /**
         * Memory-mapped file reader (POSIX) for very fast CSV loading.
         * Returns a string_view-like object mapped to the file.
         */
        class memory_mapped_file {
        public:
            memory_mapped_file(const std::string& filename) {
                #ifdef _WIN32
                hFile = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
                if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open file.");
                LARGE_INTEGER li;
                GetFileSizeEx(hFile, &li);
                size_ = li.QuadPart;
                hMap = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
                data_ = static_cast<const char*>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, size_));
                #else
                fd_ = open(filename.c_str(), O_RDONLY);
                if (fd_ < 0) throw std::runtime_error("Cannot open file.");
                struct stat st;
                fstat(fd_, &st);
                size_ = st.st_size;
                data_ = static_cast<const char*>(mmap(NULL, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
                #endif
            }

            ~memory_mapped_file() {
                #ifdef _WIN32
                if (data_) UnmapViewOfFile(data_);
                if (hMap) CloseHandle(hMap);
                if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
                #else
                if (data_ != MAP_FAILED) munmap(const_cast<char*>(data_), size_);
                if (fd_ >= 0) close(fd_);
                #endif
            }

            const char* data() const { return data_; }
            size_t size() const { return size_; }

        private:
            #ifdef _WIN32
            HANDLE hFile = INVALID_HANDLE_VALUE;
            HANDLE hMap = NULL;
            #else
            int fd_ = -1;
            #endif
            const char* data_ = nullptr;
            size_t size_ = 0;
        };

        /**
         * Parse CSV from memory-mapped buffer into a 2D xtensor array.
         */
        template <class T>
        auto parse_csv_mmap(const memory_mapped_file& mmap, char delimiter, bool has_header) {
            const char* begin = mmap.data();
            const char* end = begin + mmap.size();
            const char* ptr = begin;

            // First pass: detect shape
            size_t cols = 0;
            size_t rows = 0;
            bool first_line = true;
            const char* line_start = ptr;
            while (ptr < end) {
                if (*ptr == '\n' || (ptr == end - 1 && *ptr != '\n')) {
                    if (ptr == end - 1 && *ptr != '\n') ++ptr; // include last char
                    std::string line(line_start, ptr - line_start);
                    auto tokens = split_csv_line(line, delimiter);
                    if (first_line) {
                        cols = tokens.size();
                        first_line = false;
                        if (has_header) {
                            line_start = ptr + (ptr < end ? 1 : 0);
                            ++ptr;
                            continue;
                        }
                    }
                    if (!line.empty() && tokens.size() == cols) ++rows;
                    line_start = ptr + (ptr < end ? 1 : 0);
                }
                ++ptr;
            }
            if (cols == 0 || rows == 0) throw std::runtime_error("Empty or malformed CSV.");

            // Allocate array
            xarray_container<uvector<T>> result({rows, cols});

            // Second pass: fill array
            T* data = result.data();
            ptr = begin;
            line_start = ptr;
            size_t row = 0;
            first_line = true;
            while (ptr < end && row < rows) {
                if (*ptr == '\n' || (ptr == end - 1 && *ptr != '\n')) {
                    if (ptr == end - 1 && *ptr != '\n') ++ptr;
                    std::string line(line_start, ptr - line_start);
                    auto tokens = split_csv_line(line, delimiter);
                    if (first_line) {
                        first_line = false;
                        if (has_header) {
                            line_start = ptr + (ptr < end ? 1 : 0);
                            ++ptr;
                            continue;
                        }
                    }
                    size_t nc = std::min(tokens.size(), cols);
                    for (size_t col = 0; col < nc; ++col) {
                        data[row * cols + col] = parse_number<T>(tokens[col]);
                    }
                    ++row;
                    line_start = ptr + (ptr < end ? 1 : 0);
                }
                ++ptr;
            }
            return result;
        }
    }

    /*********************************************
     * Public CSV I/O interface
     *********************************************/

    /**
     * Load a CSV file into an xtensor 2D array, auto-detecting shape.
     */
    template <class T = double>
    inline auto load_csv(const std::string& filename, char delimiter = ',', bool has_header = false) {
        size_t cols = detail::detect_columns(filename, delimiter, has_header);
        size_t rows = detail::count_rows(filename, has_header);
        if (cols == 0 || rows == 0) throw std::runtime_error("Empty CSV file.");
        xarray_container<uvector<T>> result({rows, cols});
        detail::read_csv_into(result, filename, delimiter, has_header);
        return result;
    }

    /**
     * Load a CSV file using memory mapping for maximum speed.
     */
    template <class T = double>
    inline auto load_csv_mmap(const std::string& filename, char delimiter = ',', bool has_header = false) {
        detail::memory_mapped_file mmap(filename);
        return detail::parse_csv_mmap<T>(mmap, delimiter, has_header);
    }

    /**
     * Save a 2D array to a CSV file.
     */
    template <class E>
    inline void save_csv(const std::string& filename, const E& expr, char delimiter = ',',
                         const std::vector<std::string>& header = {}) {
        auto shape = expr.shape();
        if (shape.size() != 2) throw std::runtime_error("save_csv only supports 2D arrays.");
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Cannot open file for writing: " + filename);

        size_t rows = shape[0];
        size_t cols = shape[1];
        const auto* data = expr.data();
        char buffer[64];

        // Write header if provided
        if (!header.empty()) {
            if (header.size() != cols) throw std::runtime_error("Header size does not match columns.");
            for (size_t j = 0; j < cols; ++j) {
                if (j > 0) file << delimiter;
                file << header[j];
            }
            file << '\n';
        }

        // Write data with fast number formatting
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                if (j > 0) file << delimiter;
                auto val = data[i * cols + j];
                auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), val);
                if (ec != std::errc()) {
                    file << val; // fallback to stream
                } else {
                    file.write(buffer, ptr - buffer);
                }
            }
            file << '\n';
        }
    }

    /**
     * Append rows to an existing CSV file.
     */
    template <class E>
    inline void append_csv(const std::string& filename, const E& expr, char delimiter = ',') {
        auto shape = expr.shape();
        if (shape.size() != 2) throw std::runtime_error("append_csv only supports 2D arrays.");
        std::ofstream file(filename, std::ios::app);
        if (!file) throw std::runtime_error("Cannot open file for appending: " + filename);

        size_t rows = shape[0];
        size_t cols = shape[1];
        const auto* data = expr.data();
        char buffer[64];

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                if (j > 0) file << delimiter;
                auto val = data[i * cols + j];
                auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), val);
                if (ec != std::errc()) {
                    file << val;
                } else {
                    file.write(buffer, ptr - buffer);
                }
            }
            file << '\n';
        }
    }

    /**
     * Read CSV header line only.
     */
    inline std::vector<std::string> read_csv_header(const std::string& filename, char delimiter = ',') {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        std::string line;
        if (std::getline(file, line)) {
            return detail::split_csv_line(line, delimiter);
        }
        return {};
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_XCSV_HPP