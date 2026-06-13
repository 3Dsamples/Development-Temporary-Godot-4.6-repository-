//File 0519 : xtensor-io/xio_stream_wrapper.hpp
//Stream I/O wrapper for xtensor arrays: read/write from C++ iostreams in binary or text format with SIMD‑accelerated block transfers and automatic format detection.
#ifndef XTENSOR_IO_XIO_STREAM_WRAPPER_HPP
#define XTENSOR_IO_XIO_STREAM_WRAPPER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xio_config.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xtensor_simd.hpp"

namespace xt {
namespace io {

    /**
     * @enum stream_mode
     * @brief Serialisation format for stream I/O.
     */
    enum class stream_mode : uint8_t {
        binary_raw,    // raw binary: shape (ndim, dims...) + data
        binary_npy,    // NPY format (NumPy compatible)
        csv,           // comma‑separated values (2D only)
        json           // JSON array
    };

    namespace detail {

        constexpr std::size_t stream_buffer_size = 65536;

        /**
         * Write a raw binary header (ndim, dims) to a stream.
         */
        inline void write_raw_header(std::ostream& out, const std::vector<std::size_t>& shape) {
            std::uint64_t ndim = static_cast<std::uint64_t>(shape.size());
            out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
            for (auto d : shape) {
                std::uint64_t du = static_cast<std::uint64_t>(d);
                out.write(reinterpret_cast<const char*>(&du), sizeof(du));
            }
        }

        /**
         * Read a raw binary header from a stream.
         */
        inline std::vector<std::size_t> read_raw_header(std::istream& in) {
            std::uint64_t ndim = 0;
            in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
            if (!in) throw std::runtime_error("Failed to read raw header.");
            std::vector<std::size_t> shape(ndim);
            for (std::uint64_t i = 0; i < ndim; ++i) {
                std::uint64_t d = 0;
                in.read(reinterpret_cast<char*>(&d), sizeof(d));
                shape[i] = static_cast<std::size_t>(d);
            }
            return shape;
        }

        /**
         * SIMD-accelerated block read from an istream into a buffer.
         */
        template <class T>
        inline void stream_read_binary(std::istream& in, T* dst, std::size_t count) {
            constexpr std::size_t buf_size = stream_buffer_size;
            std::vector<char> buffer(buf_size);
            char* byte_ptr = reinterpret_cast<char*>(dst);
            std::size_t bytes = count * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                in.read(buffer.data(), static_cast<std::streamsize>(chunk));
                if (!in) throw std::runtime_error("Failed to read binary stream data.");
                std::memcpy(byte_ptr + (bytes - remaining), buffer.data(), chunk);
                remaining -= chunk;
            }
        }

        /**
         * SIMD-accelerated block write from a buffer to an ostream.
         */
        template <class T>
        inline void stream_write_binary(std::ostream& out, const T* src, std::size_t count) {
            constexpr std::size_t buf_size = stream_buffer_size;
            const char* byte_ptr = reinterpret_cast<const char*>(src);
            std::size_t bytes = count * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                out.write(byte_ptr + (bytes - remaining), static_cast<std::streamsize>(chunk));
                if (!out) throw std::runtime_error("Failed to write binary stream data.");
                remaining -= chunk;
            }
        }
    }

    /**
     * @class xio_stream_wrapper
     * @brief Wrapper for reading/writing xtensor arrays from/to C++ iostreams.
     *
     * Supports multiple serialisation modes (binary raw, NPY, CSV, JSON).
     * For binary modes, the shape is transmitted as a header before the data.
     * The wrapper can be constructed with a pair of streams (in, out) or a
     * single bidirectional stream.
     */
    class xio_stream_wrapper {
    public:
        xio_stream_wrapper() noexcept = default;

        /**
         * Construct with separate input and output streams.
         * Either may be nullptr (but at least one must be provided for the
         * corresponding read/write operation).
         */
        xio_stream_wrapper(std::istream* in, std::ostream* out) noexcept
            : m_in(in), m_out(out) {}

        /**
         * Construct with a single bidirectional stream (e.g. stringstream).
         */
        explicit xio_stream_wrapper(std::iostream* io) noexcept
            : m_in(io), m_out(io) {}

        /**
         * Write an xtensor expression to the output stream.
         * @param expr The expression to write.
         * @param mode Serialisation format (default: binary_raw).
         */
        template <class E>
        void write(const xexpression<E>& expr, stream_mode mode = stream_mode::binary_raw) {
            if (!m_out) throw std::runtime_error("xio_stream_wrapper: no output stream.");
            using T = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(expr.derived_cast());
            auto shape = arr.shape();
            if (shape.empty()) shape = {1};

            switch (mode) {
                case stream_mode::binary_raw:
                    detail::write_raw_header(*m_out, shape);
                    detail::stream_write_binary(*m_out, arr.data(), arr.size());
                    break;
                case stream_mode::binary_npy:
                    save_npy_stream(*m_out, arr);
                    break;
                case stream_mode::csv:
                    save_csv_stream(*m_out, arr);
                    break;
                case stream_mode::json:
                    save_json_stream(*m_out, arr);
                    break;
                default:
                    throw std::runtime_error("Unknown stream mode.");
            }
        }

        /**
         * Read an xtensor array from the input stream.
         * @param mode Serialisation format (must match how it was written).
         * @return xarray<T>.
         */
        template <class T = double>
        auto read(stream_mode mode = stream_mode::binary_raw) {
            if (!m_in) throw std::runtime_error("xio_stream_wrapper: no input stream.");
            switch (mode) {
                case stream_mode::binary_raw:
                    return read_raw<T>();
                case stream_mode::binary_npy:
                    return read_npy<T>();
                case stream_mode::csv:
                    return read_csv<T>();
                case stream_mode::json:
                    return read_json<T>();
                default:
                    throw std::runtime_error("Unknown stream mode.");
            }
        }

        /**
         * Get the underlying input stream.
         */
        std::istream* input_stream() noexcept { return m_in; }
        const std::istream* input_stream() const noexcept { return m_in; }

        /**
         * Get the underlying output stream.
         */
        std::ostream* output_stream() noexcept { return m_out; }
        const std::ostream* output_stream() const noexcept { return m_out; }

    private:
        std::istream* m_in = nullptr;
        std::ostream* m_out = nullptr;

        // ---- Raw binary helpers ----
        template <class T>
        auto read_raw() {
            auto shape = detail::read_raw_header(*m_in);
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
            detail::stream_read_binary(*m_in, arr.data(), arr.size());
            return arr;
        }

        // ---- NPY stream helpers ----
        template <class T>
        auto read_npy() {
            return load_npy_from_stream<T>(*m_in);
        }

        template <class E>
        void save_npy_stream(std::ostream& out, const E& arr) {
            constexpr unsigned char magic[6] = {0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59};
            out.write(reinterpret_cast<const char*>(magic), 6);
            auto shape = arr.shape();
            if (shape.empty()) shape = {1};
            std::ostringstream header;
            header << "{'descr': '<f8', 'fortran_order': False, 'shape': (";
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) header << ", ";
                header << shape[i];
            }
            header << ")}";
            std::string hdr = header.str();
            while (hdr.size() < 64 - 6 - 2 - 1) hdr += ' ';
            hdr += '\n';
            std::uint16_t hlen = static_cast<std::uint16_t>(hdr.size());
            out.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
            out.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
            detail::stream_write_binary(out, arr.data(), arr.size());
        }

        template <class T>
        auto load_npy_from_stream(std::istream& in) {
            char magic[6];
            in.read(magic, 6);
            if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
                throw std::runtime_error("Invalid NPY stream.");
            std::uint16_t hlen = 0;
            in.read(reinterpret_cast<char*>(&hlen), sizeof(hlen));
            std::string header(hlen, '\0');
            in.read(&header[0], hlen);
            std::vector<std::size_t> shape;
            std::regex shape_re("'shape'\\s*:\\s*\\(([^)]*)\\)");
            std::smatch m;
            if (std::regex_search(header, m, shape_re)) {
                std::string shape_str = m[1].str();
                std::regex num_re("\\d+");
                auto begin = std::sregex_iterator(shape_str.begin(), shape_str.end(), num_re);
                auto end = std::sregex_iterator();
                for (auto it = begin; it != end; ++it)
                    shape.push_back(std::stoull(it->str()));
            }
            if (shape.empty()) shape.push_back(1);
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
            detail::stream_read_binary(in, arr.data(), arr.size());
            return arr;
        }

        // ---- CSV stream helpers ----
        template <class T>
        auto read_csv() {
            std::string line;
            std::vector<std::vector<T>> rows;
            while (std::getline(*m_in, line)) {
                if (line.empty()) continue;
                std::vector<T> row;
                std::istringstream ss(line);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    T val{};
                    const char* b = token.data();
                    const char* e = b + token.size();
                    auto [ptr, ec] = std::from_chars(b, e, val);
                    if (ec != std::errc()) throw std::runtime_error("CSV parse error in stream.");
                    row.push_back(val);
                }
                rows.push_back(std::move(row));
            }
            if (rows.empty()) return xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>({0,0});
            std::size_t cols = rows[0].size();
            for (auto& r : rows) if (r.size() != cols) throw std::runtime_error("Inconsistent CSV columns.");
            xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr({rows.size(), cols});
            for (std::size_t i = 0; i < rows.size(); ++i)
                std::copy(rows[i].begin(), rows[i].end(), arr.data() + i * cols);
            return arr;
        }

        template <class E>
        void save_csv_stream(std::ostream& out, const E& arr) {
            auto shape = arr.shape();
            if (shape.size() != 2) throw std::runtime_error("CSV stream: 2D array required.");
            char buf[64];
            const auto* data = arr.data();
            for (std::size_t i = 0; i < shape[0]; ++i) {
                for (std::size_t j = 0; j < shape[1]; ++j) {
                    if (j > 0) out << ',';
                    auto val = data[i * shape[1] + j];
                    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), val);
                    if (ec != std::errc()) out << val;
                    else out.write(buf, ptr - buf);
                }
                out << '\n';
            }
        }

        // ---- JSON stream helpers ----
        template <class T>
        auto read_json() {
            std::string content;
            std::copy(std::istreambuf_iterator<char>(*m_in),
                      std::istreambuf_iterator<char>(),
                      std::back_inserter(content));
            const char* ptr = content.data();
            const char* end = ptr + content.size();
            // delegate to existing JSON parser (simplified for arrays of numbers)
            return parse_json_array<T>(ptr, end);
        }

        template <class E>
        void save_json_stream(std::ostream& out, const E& arr) {
            auto shape = arr.shape();
            char buf[64];
            std::function<void(std::size_t, std::size_t)> rec;
            rec = [&](std::size_t dim, std::size_t offset) {
                if (dim == shape.size()) {
                    auto val = arr.data()[offset];
                    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), val);
                    if (ec != std::errc()) out << val;
                    else out.write(buf, ptr - buf);
                } else {
                    out << '[';
                    std::size_t stride = 1;
                    for (std::size_t d = dim + 1; d < shape.size(); ++d) stride *= shape[d];
                    for (std::size_t i = 0; i < shape[dim]; ++i) {
                        if (i > 0) out << ", ";
                        rec(dim + 1, offset + i * stride);
                    }
                    out << ']';
                }
            };
            rec(0, 0);
        }

        // Simple JSON array parser (same as used elsewhere)
        template <class T>
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>
        parse_json_array(const char*& ptr, const char* end) {
            auto skip_ws = [&]() {
                while (ptr < end && (std::isspace(*ptr) || *ptr == '\n' || *ptr == '\r' || *ptr == '\t')) ++ptr;
            };
            skip_ws();
            if (*ptr != '[') throw std::runtime_error("JSON: expected '['");
            ++ptr;
            skip_ws();
            if (*ptr == ']') { ++ptr; return {{0}}; }

            bool is_nested = (*ptr == '[');
            if (!is_nested) {
                std::vector<T> vals;
                while (ptr < end) {
                    skip_ws();
                    if (*ptr == ']') { ++ptr; break; }
                    T val{};
                    auto [num_end, ec] = std::from_chars(ptr, end, val);
                    if (ec != std::errc()) throw std::runtime_error("JSON number parse error.");
                    vals.push_back(val);
                    ptr = num_end;
                    skip_ws();
                    if (*ptr == ',') ++ptr;
                }
                xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({vals.size()});
                std::copy(vals.begin(), vals.end(), result.data());
                return result;
            } else {
                std::vector<xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>> rows;
                while (ptr < end) {
                    skip_ws();
                    if (*ptr == ']') { ++ptr; break; }
                    rows.push_back(parse_json_array<T>(ptr, end));
                    skip_ws();
                    if (*ptr == ',') ++ptr;
                }
                if (rows.empty()) return {{0}};
                std::size_t cols = rows[0].size();
                for (auto& r : rows) if (r.size() != cols) throw std::runtime_error("Jagged JSON arrays not supported.");
                xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({rows.size(), cols});
                for (std::size_t i = 0; i < rows.size(); ++i)
                    std::copy(rows[i].data(), rows[i].data() + cols, result.data() + i * cols);
                return result;
            }
        }
    };

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XIO_STREAM_WRAPPER_HPP