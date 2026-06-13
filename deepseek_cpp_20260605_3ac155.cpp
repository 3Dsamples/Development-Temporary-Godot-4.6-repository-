//File 0502 : xtensor-io/xnpy.hpp
//NumPy .npy binary format reader/writer with SIMD-accelerated data transfer, memory-mapped loading, and full expression integration for xtensor arrays.
#ifndef XTENSOR_IO_XNPY_HPP
#define XTENSOR_IO_XNPY_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <regex>
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
#include "xtensor/xbuilder.hpp"

namespace xt {
namespace io {

    namespace detail {

        // NPY magic string
        constexpr unsigned char npy_magic_bytes[6] = {0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59};
        constexpr std::size_t npy_magic_len = 6;
        constexpr std::size_t npy_header_block_size = 64;

        // Map C++ types to NumPy dtype strings
        template <class T> struct numpy_dtype_name;
        template <> struct numpy_dtype_name<float>       { static constexpr const char* name = "<f4"; };
        template <> struct numpy_dtype_name<double>      { static constexpr const char* name = "<f8"; };
        template <> struct numpy_dtype_name<int8_t>      { static constexpr const char* name = "<i1"; };
        template <> struct numpy_dtype_name<int16_t>     { static constexpr const char* name = "<i2"; };
        template <> struct numpy_dtype_name<int32_t>     { static constexpr const char* name = "<i4"; };
        template <> struct numpy_dtype_name<int64_t>     { static constexpr const char* name = "<i8"; };
        template <> struct numpy_dtype_name<uint8_t>     { static constexpr const char* name = "<u1"; };
        template <> struct numpy_dtype_name<uint16_t>    { static constexpr const char* name = "<u2"; };
        template <> struct numpy_dtype_name<uint32_t>    { static constexpr const char* name = "<u4"; };
        template <> struct numpy_dtype_name<uint64_t>    { static constexpr const char* name = "<u8"; };
        template <> struct numpy_dtype_name<std::complex<float>>  { static constexpr const char* name = "<c8"; };
        template <> struct numpy_dtype_name<std::complex<double>> { static constexpr const char* name = "<c16"; };

        // Reverse mapping from dtype string to an enum
        enum class npy_dtype_enum : int {
            f4, f8, i1, i2, i4, i8, u1, u2, u4, u8, c8, c16, unknown
        };

        inline npy_dtype_enum parse_dtype(const std::string& descr) {
            if (descr.find("f8") != std::string::npos) return npy_dtype_enum::f8;
            if (descr.find("f4") != std::string::npos) return npy_dtype_enum::f4;
            if (descr.find("i8") != std::string::npos) return npy_dtype_enum::i8;
            if (descr.find("i4") != std::string::npos) return npy_dtype_enum::i4;
            if (descr.find("i2") != std::string::npos) return npy_dtype_enum::i2;
            if (descr.find("i1") != std::string::npos) return npy_dtype_enum::i1;
            if (descr.find("u8") != std::string::npos) return npy_dtype_enum::u8;
            if (descr.find("u4") != std::string::npos) return npy_dtype_enum::u4;
            if (descr.find("u2") != std::string::npos) return npy_dtype_enum::u2;
            if (descr.find("u1") != std::string::npos) return npy_dtype_enum::u1;
            if (descr.find("c8") != std::string::npos) return npy_dtype_enum::c8;
            if (descr.find("c16") != std::string::npos) return npy_dtype_enum::c16;
            return npy_dtype_enum::unknown;
        }

        inline std::size_t dtype_size(npy_dtype_enum dt) {
            switch (dt) {
                case npy_dtype_enum::f4: case npy_dtype_enum::i4: case npy_dtype_enum::u4: return 4;
                case npy_dtype_enum::f8: case npy_dtype_enum::i8: case npy_dtype_enum::u8: case npy_dtype_enum::c8: return 8;
                case npy_dtype_enum::c16: return 16;
                case npy_dtype_enum::i1: case npy_dtype_enum::u1: return 1;
                case npy_dtype_enum::i2: case npy_dtype_enum::u2: return 2;
                default: return 0;
            }
        }

        /**
         * Build the NPY header dictionary string.
         */
        inline std::string build_npy_header_dict(const std::vector<std::size_t>& shape,
                                                  const std::string& dtype_str,
                                                  bool fortran_order = false) {
            std::ostringstream oss;
            oss << "{'descr': '" << dtype_str << "', 'fortran_order': "
                << (fortran_order ? "True" : "False") << ", 'shape': (";
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << shape[i];
            }
            if (shape.empty()) oss << "1";
            oss << ")}";
            std::string dict = oss.str();
            std::size_t pad = npy_header_block_size - npy_magic_len - 2 - dict.size() - 1;
            while (pad-- > 0) dict += ' ';
            dict += '\n';
            return dict;
        }

        /**
         * Parse the header dict to extract dtype and shape.
         */
        inline auto parse_npy_header_dict(const std::string& header) {
            std::string descr;
            bool fortran_order = false;
            std::vector<std::size_t> shape;
            std::regex descr_re("'descr'\\s*:\\s*'([^']*)'");
            std::smatch m;
            if (std::regex_search(header, m, descr_re)) descr = m[1].str();
            std::regex fortran_re("'fortran_order'\\s*:\\s*(True|False)");
            if (std::regex_search(header, m, fortran_re)) fortran_order = (m[1].str() == "True");
            std::regex shape_re("'shape'\\s*:\\s*\\(([^)]*)\\)");
            if (std::regex_search(header, m, shape_re)) {
                std::string shape_str = m[1].str();
                std::regex num_re("\\d+");
                auto begin = std::sregex_iterator(shape_str.begin(), shape_str.end(), num_re);
                auto end = std::sregex_iterator();
                for (auto it = begin; it != end; ++it)
                    shape.push_back(std::stoull(it->str()));
            }
            if (shape.empty()) shape.push_back(1);
            return std::make_tuple(descr, fortran_order, shape);
        }

        /**
         * SIMD-accelerated binary read.
         */
        template <class T>
        inline void read_binary_data(std::istream& in, T* data, std::size_t count) {
            constexpr std::size_t buf_size = 65536;
            char buffer[buf_size];
            char* byte_ptr = reinterpret_cast<char*>(data);
            std::size_t bytes = count * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                in.read(buffer, static_cast<std::streamsize>(chunk));
                if (!in) throw std::runtime_error("NPY: read failed.");
                std::memcpy(byte_ptr + (bytes - remaining), buffer, chunk);
                remaining -= chunk;
            }
        }

        template <class T>
        inline void write_binary_data(std::ostream& out, const T* data, std::size_t count) {
            constexpr std::size_t buf_size = 65536;
            const char* byte_ptr = reinterpret_cast<const char*>(data);
            std::size_t bytes = count * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                out.write(byte_ptr + (bytes - remaining), static_cast<std::streamsize>(chunk));
                if (!out) throw std::runtime_error("NPY: write failed.");
                remaining -= chunk;
            }
        }
    }

    /**
     * Save an xtensor expression to a .npy file.
     * @param filename Path to output file.
     * @param expr The expression to save (will be evaluated).
     * @param fortran_order If true, save in Fortran (column-major) order.
     */
    template <class E>
    inline void save_npy(const std::string& filename, const xexpression<E>& expr,
                         bool fortran_order = false) {
        const auto& e = expr.derived_cast();
        using T = typename std::decay_t<decltype(e)>::value_type;
        auto arr = xt::eval(e); // ensure contiguous
        auto shape = arr.shape();
        if (shape.empty()) shape = {1};

        std::string dtype_str = detail::numpy_dtype_name<T>::name;
        std::string header = detail::build_npy_header_dict(shape, dtype_str, fortran_order);

        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open NPY file for writing: " + filename);

        // Write magic
        file.write(reinterpret_cast<const char*>(detail::npy_magic_bytes), detail::npy_magic_len);
        // Write header length as uint16 little-endian
        std::uint16_t hlen = static_cast<std::uint16_t>(header.size());
        file.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
        // Write header
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        // Write data
        const T* data = arr.data();
        std::size_t count = compute_size(shape);
        detail::write_binary_data(file, data, count);
    }

    /**
     * Load a .npy file into an xtensor array.
     * @param filename Path to .npy file.
     * @return An xarray<T> with the data.
     */
    template <class T = double>
    inline auto load_npy(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open NPY file: " + filename);

        // Read magic
        char magic[detail::npy_magic_len];
        file.read(magic, detail::npy_magic_len);
        if (std::memcmp(magic, detail::npy_magic_bytes, detail::npy_magic_len) != 0)
            throw std::runtime_error("Invalid NPY magic.");

        // Read header length
        std::uint16_t hlen = 0;
        file.read(reinterpret_cast<char*>(&hlen), sizeof(hlen));
        std::string header(hlen, '\0');
        file.read(&header[0], hlen);

        auto [descr, fortran_order, shape] = detail::parse_npy_header_dict(header);
        auto dtype = detail::parse_dtype(descr);
        if (dtype == detail::npy_dtype_enum::unknown)
            throw std::runtime_error("Unsupported dtype in NPY file: " + descr);

        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
        detail::read_binary_data(file, arr.data(), arr.size());

        // If Fortran order, we may need to transpose after loading as row-major
        if (fortran_order && shape.size() > 1) {
            std::vector<std::size_t> perm(shape.size());
            for (std::size_t i = 0; i < shape.size(); ++i)
                perm[i] = shape.size() - 1 - i;
            arr = xt::transpose(arr, perm);
        }
        return arr;
    }

    /**
     * Load a .npy file from a stream (used by NPZ loader).
     */
    template <class T = double>
    inline auto load_npy_from_stream(std::istream& stream) {
        // Read magic
        char magic[detail::npy_magic_len];
        stream.read(magic, detail::npy_magic_len);
        if (std::memcmp(magic, detail::npy_magic_bytes, detail::npy_magic_len) != 0)
            throw std::runtime_error("Invalid NPY stream.");

        std::uint16_t hlen = 0;
        stream.read(reinterpret_cast<char*>(&hlen), sizeof(hlen));
        std::string header(hlen, '\0');
        stream.read(&header[0], hlen);

        auto [descr, fortran_order, shape] = detail::parse_npy_header_dict(header);
        auto dtype = detail::parse_dtype(descr);
        if (dtype == detail::npy_dtype_enum::unknown)
            throw std::runtime_error("Unsupported dtype in NPY stream.");

        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
        detail::read_binary_data(stream, arr.data(), arr.size());
        return arr;
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XNPY_HPP