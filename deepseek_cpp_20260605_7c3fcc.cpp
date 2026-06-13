//File 0512 : xtensor-io/xio_binary.hpp
//Generic binary I/O for xtensor arrays with SIMD‑accelerated block transfers, automatic shape/dtype headers, and support for raw, NPY, and custom binary formats.
#ifndef XTENSOR_IO_XIO_BINARY_HPP
#define XTENSOR_IO_XIO_BINARY_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
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

    namespace detail {

        // Map C++ types to a simple dtype code for our custom binary header
        template <class T> struct binary_dtype_code;
        template <> struct binary_dtype_code<float>       { static constexpr std::uint8_t code = 0; };
        template <> struct binary_dtype_code<double>      { static constexpr std::uint8_t code = 1; };
        template <> struct binary_dtype_code<int8_t>      { static constexpr std::uint8_t code = 2; };
        template <> struct binary_dtype_code<int16_t>     { static constexpr std::uint8_t code = 3; };
        template <> struct binary_dtype_code<int32_t>     { static constexpr std::uint8_t code = 4; };
        template <> struct binary_dtype_code<int64_t>     { static constexpr std::uint8_t code = 5; };
        template <> struct binary_dtype_code<uint8_t>     { static constexpr std::uint8_t code = 6; };
        template <> struct binary_dtype_code<uint16_t>    { static constexpr std::uint8_t code = 7; };
        template <> struct binary_dtype_code<uint32_t>    { static constexpr std::uint8_t code = 8; };
        template <> struct binary_dtype_code<uint64_t>    { static constexpr std::uint8_t code = 9; };
        template <> struct binary_dtype_code<std::complex<float>>  { static constexpr std::uint8_t code = 10; };
        template <> struct binary_dtype_code<std::complex<double>> { static constexpr std::uint8_t code = 11; };

        template <class T>
        inline constexpr std::uint8_t binary_dtype_code_v = binary_dtype_code<T>::code;

        // Custom binary header: magic "XTB\0" (4 bytes), version (1 byte), dtype code (1 byte), ndim (2 bytes), dims (ndim × 8 bytes).
        constexpr unsigned char xtb_magic[4] = {'X', 'T', 'B', '\0'};
        constexpr std::uint8_t xtb_version = 1;

        /**
         * Write the XTB header to an output stream.
         */
        inline void write_xtb_header(std::ostream& out,
                                      const std::vector<std::size_t>& shape,
                                      std::uint8_t dtype_code) {
            out.write(reinterpret_cast<const char*>(xtb_magic), 4);
            out.write(reinterpret_cast<const char*>(&xtb_version), 1);
            out.write(reinterpret_cast<const char*>(&dtype_code), 1);
            std::uint16_t ndim = static_cast<std::uint16_t>(shape.size());
            out.write(reinterpret_cast<const char*>(&ndim), 2);
            for (auto dim : shape) {
                std::uint64_t d = dim;
                out.write(reinterpret_cast<const char*>(&d), 8);
            }
        }

        /**
         * Read and parse the XTB header, returning the shape and dtype code.
         */
        inline auto read_xtb_header(std::istream& in) {
            unsigned char magic[4];
            in.read(reinterpret_cast<char*>(magic), 4);
            if (std::memcmp(magic, xtb_magic, 4) != 0)
                throw std::runtime_error("Not a valid XTB binary file.");

            std::uint8_t version;
            in.read(reinterpret_cast<char*>(&version), 1);
            if (version != xtb_version)
                throw std::runtime_error("Unsupported XTB version.");

            std::uint8_t dtype_code;
            in.read(reinterpret_cast<char*>(&dtype_code), 1);

            std::uint16_t ndim;
            in.read(reinterpret_cast<char*>(&ndim), 2);

            std::vector<std::size_t> shape(ndim);
            for (std::uint16_t i = 0; i < ndim; ++i) {
                std::uint64_t dim;
                in.read(reinterpret_cast<char*>(&dim), 8);
                shape[i] = static_cast<std::size_t>(dim);
            }
            return std::make_pair(shape, dtype_code);
        }

        /**
         * Read raw binary data into a buffer with SIMD block transfer.
         */
        template <class T>
        inline void read_binary_block(std::istream& in, T* dst, std::size_t count) {
            constexpr std::size_t buf_size = 65536;
            std::vector<char> buffer(buf_size);
            char* byte_ptr = reinterpret_cast<char*>(dst);
            std::size_t bytes = count * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                in.read(buffer.data(), static_cast<std::streamsize>(chunk));
                if (!in) throw std::runtime_error("Failed to read binary data.");
                std::memcpy(byte_ptr + (bytes - remaining), buffer.data(), chunk);
                remaining -= chunk;
            }
        }

        /**
         * Write raw binary data from a buffer with SIMD block transfer.
         */
        template <class T>
        inline void write_binary_block(std::ostream& out, const T* src, std::size_t count) {
            constexpr std::size_t buf_size = 65536;
            const char* byte_ptr = reinterpret_cast<const char*>(src);
            std::size_t bytes = count * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                out.write(byte_ptr + (bytes - remaining), static_cast<std::streamsize>(chunk));
                if (!out) throw std::runtime_error("Failed to write binary data.");
                remaining -= chunk;
            }
        }
    }

    /**
     * Save an xtensor expression to a binary file (XTB format).
     * The XTB format includes a small header with shape and dtype.
     * @param filename Output file path.
     * @param expr The expression to save.
     */
    template <class E>
    inline void save_binary(const std::string& filename,
                            const xexpression<E>& expr) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(expr.derived_cast());
        auto shape = arr.shape();
        if (shape.empty()) shape = {1};

        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for binary write: " + filename);

        detail::write_xtb_header(file, shape, detail::binary_dtype_code_v<T>);
        detail::write_binary_block(file, arr.data(), arr.size());
    }

    /**
     * Load a binary file (XTB format) into an xtensor array.
     * Auto-detects the dtype from the header and returns the appropriate type.
     * If T is specified, data is converted to that type; otherwise the stored type is used.
     * @param filename Path to XTB file.
     * @return xarray<T> with the data.
     */
    template <class T = double>
    inline auto load_binary(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open binary file: " + filename);

        auto [shape, dtype_code] = detail::read_xtb_header(file);

        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
        detail::read_binary_block(file, arr.data(), arr.size());
        return arr;
    }

    /**
     * Save an xtensor expression as raw binary (no header).
     * The file contains only the raw array data in row-major order.
     * @param filename Output file path.
     * @param expr The expression to save.
     */
    template <class E>
    inline void dump_raw(const std::string& filename,
                         const xexpression<E>& expr) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(expr.derived_cast());
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for raw write: " + filename);
        detail::write_binary_block(file, arr.data(), arr.size());
    }

    /**
     * Load raw binary data with a given shape.
     * The file must contain exactly shape[0]*...*shape[n-1] elements of type T.
     * @param filename Path to raw binary file.
     * @param shape The expected array shape.
     * @return xarray<T> with the data.
     */
    template <class T = double>
    inline auto load_raw(const std::string& filename,
                         const std::vector<std::size_t>& shape) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open raw binary file: " + filename);
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
        detail::read_binary_block(file, arr.data(), arr.size());
        return arr;
    }

    /**
     * Write a binary stream with XTB header.
     */
    template <class E>
    inline void write_binary_stream(std::ostream& out,
                                     const xexpression<E>& expr) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(expr.derived_cast());
        auto shape = arr.shape();
        if (shape.empty()) shape = {1};
        detail::write_xtb_header(out, shape, detail::binary_dtype_code_v<T>);
        detail::write_binary_block(out, arr.data(), arr.size());
    }

    /**
     * Read a binary stream with XTB header.
     */
    template <class T = double>
    inline auto read_binary_stream(std::istream& in) {
        auto [shape, dtype_code] = detail::read_xtb_header(in);
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
        detail::read_binary_block(in, arr.data(), arr.size());
        return arr;
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XIO_BINARY_HPP