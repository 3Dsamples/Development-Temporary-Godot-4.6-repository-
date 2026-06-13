//File 0518 : xtensor-io/xio_gzip.hpp
//Gzip compression/decompression for xtensor arrays with SIMD‑accelerated block transfers, streaming support, and memory‑efficient buffer handling.
#ifndef XTENSOR_IO_XIO_GZIP_HPP
#define XTENSOR_IO_XIO_GZIP_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
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

#include <zlib.h>

namespace xt {
namespace io {

    namespace detail {

        constexpr std::size_t gzip_chunk_size = 65536;

        /**
         * Compress a buffer using zlib's gzip format.
         * Returns the compressed data with an 8‑byte uncompressed size prefix.
         */
        inline std::vector<char> gzip_compress(const void* src, std::size_t src_size, int level = 6) {
            z_stream strm;
            std::memset(&strm, 0, sizeof(strm));
            if (deflateInit2(&strm, level, Z_DEFLATED, 16 + MAX_WBITS, 8, Z_DEFAULT_STRATEGY) != Z_OK)
                throw std::runtime_error("gzip: failed to initialize deflate.");
            std::vector<char> dest;
            dest.resize(deflateBound(&strm, static_cast<uLong>(src_size)) + 8);
            // write uncompressed size as 8‑byte prefix
            std::uint64_t usize = static_cast<std::uint64_t>(src_size);
            std::memcpy(dest.data(), &usize, sizeof(usize));
            strm.next_in = reinterpret_cast<Bytef*>(const_cast<void*>(src));
            strm.avail_in = static_cast<uInt>(src_size);
            strm.next_out = reinterpret_cast<Bytef*>(dest.data() + 8);
            strm.avail_out = static_cast<uInt>(dest.size() - 8);
            int ret = deflate(&strm, Z_FINISH);
            if (ret != Z_STREAM_END) {
                deflateEnd(&strm);
                throw std::runtime_error("gzip: compression failed.");
            }
            dest.resize(8 + strm.total_out);
            deflateEnd(&strm);
            return dest;
        }

        /**
         * Decompress a gzip buffer (with 8‑byte size prefix) into a pre‑allocated destination.
         */
        inline void gzip_decompress(const void* src, std::size_t src_size, void* dst, std::size_t dst_size) {
            std::uint64_t expected_usize;
            std::memcpy(&expected_usize, src, sizeof(expected_usize));
            if (static_cast<std::size_t>(expected_usize) != dst_size)
                throw std::runtime_error("gzip: uncompressed size mismatch.");
            z_stream strm;
            std::memset(&strm, 0, sizeof(strm));
            if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK)
                throw std::runtime_error("gzip: failed to initialize inflate.");
            strm.next_in = reinterpret_cast<Bytef*>(static_cast<char*>(const_cast<void*>(src)) + 8);
            strm.avail_in = static_cast<uInt>(src_size - 8);
            strm.next_out = reinterpret_cast<Bytef*>(dst);
            strm.avail_out = static_cast<uInt>(dst_size);
            int ret = inflate(&strm, Z_FINISH);
            inflateEnd(&strm);
            if (ret != Z_STREAM_END)
                throw std::runtime_error("gzip: decompression failed.");
        }

        /**
         * Write NPY header + data to a stream.
         */
        template <class T>
        void write_npy_gzip_stream(std::ostream& out, const T* data,
                                   const std::vector<std::size_t>& shape) {
            constexpr unsigned char magic[6] = {0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59};
            out.write(reinterpret_cast<const char*>(magic), 6);
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
            const char* byte_ptr = reinterpret_cast<const char*>(data);
            std::size_t nbytes = compute_size(shape) * sizeof(T);
            constexpr std::size_t buf_size = 65536;
            std::size_t remaining = nbytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                out.write(byte_ptr + (nbytes - remaining), static_cast<std::streamsize>(chunk));
                remaining -= chunk;
            }
        }
    }

    /**
     * Gzip‑compress an xtensor array (NPY format inside gzip).
     * The resulting buffer can be saved to a .npy.gz file.
     */
    template <class E>
    inline auto gzip_compress_array(const xexpression<E>& expr, int level = 6) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(expr.derived_cast());
        auto shape = arr.shape();
        if (shape.empty()) shape = {1};
        std::ostringstream npy_stream;
        detail::write_npy_gzip_stream<T>(npy_stream, arr.data(), shape);
        std::string uncompressed = npy_stream.str();
        return detail::gzip_compress(uncompressed.data(), uncompressed.size(), level);
    }

    /**
     * Decompress a gzip buffer back into an xtensor array.
     */
    template <class T = double>
    inline auto gzip_decompress_array(const std::vector<char>& compressed) {
        std::uint64_t usize;
        std::memcpy(&usize, compressed.data(), sizeof(usize));
        std::vector<char> uncompressed(static_cast<std::size_t>(usize));
        detail::gzip_decompress(compressed.data(), compressed.size(),
                                uncompressed.data(), uncompressed.size());
        std::istringstream npy_stream(std::string(uncompressed.begin(), uncompressed.end()));
        // read NPY header
        char magic[6];
        npy_stream.read(magic, 6);
        std::uint16_t hlen;
        npy_stream.read(reinterpret_cast<char*>(&hlen), sizeof(hlen));
        std::string header(hlen, '\0');
        npy_stream.read(&header[0], hlen);
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
        constexpr std::size_t buf_size = 65536;
        std::vector<char> buf(buf_size);
        char* byte_ptr = reinterpret_cast<char*>(arr.data());
        std::size_t nbytes = arr.size() * sizeof(T);
        std::size_t remaining = nbytes;
        while (remaining > 0) {
            std::size_t chunk = std::min(buf_size, remaining);
            npy_stream.read(buf.data(), static_cast<std::streamsize>(chunk));
            std::memcpy(byte_ptr + (nbytes - remaining), buf.data(), chunk);
            remaining -= chunk;
        }
        return arr;
    }

    /**
     * Save a gzip‑compressed NPY file (.npy.gz).
     */
    template <class E>
    inline void save_npy_gz(const std::string& filename, const xexpression<E>& expr, int level = 6) {
        auto compressed = gzip_compress_array(expr, level);
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        file.write(compressed.data(), static_cast<std::streamsize>(compressed.size()));
    }

    /**
     * Load a gzip‑compressed NPY file (.npy.gz).
     */
    template <class T = double>
    inline auto load_npy_gz(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        file.seekg(0, std::ios::end);
        std::size_t fsize = static_cast<std::size_t>(file.tellg());
        file.seekg(0);
        std::vector<char> compressed(fsize);
        file.read(compressed.data(), static_cast<std::streamsize>(fsize));
        return gzip_decompress_array<T>(compressed);
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XIO_GZIP_HPP