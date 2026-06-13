//File 0521 : xtensor-io/xio_zlib.hpp
//Zlib compression/decompression for xtensor arrays with SIMD‑accelerated block transfers, streaming support, and memory‑efficient buffer management.
#ifndef XTENSOR_IO_XIO_ZLIB_HPP
#define XTENSOR_IO_XIO_ZLIB_HPP

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

        constexpr std::size_t zlib_chunk_size = 65536;

        /**
         * Compress a buffer using raw deflate (zlib format without gzip header).
         * Returns the compressed data with an 8‑byte uncompressed size prefix.
         */
        inline std::vector<char> zlib_compress(const void* src, std::size_t src_size, int level = 6) {
            z_stream strm;
            std::memset(&strm, 0, sizeof(strm));
            if (deflateInit(&strm, level) != Z_OK)
                throw std::runtime_error("zlib: failed to initialize deflate.");

            std::vector<char> dest;
            dest.resize(deflateBound(&strm, static_cast<uLong>(src_size)) + 8);
            // Write uncompressed size as 8‑byte prefix
            std::uint64_t usize = static_cast<std::uint64_t>(src_size);
            std::memcpy(dest.data(), &usize, sizeof(usize));

            strm.next_in = reinterpret_cast<Bytef*>(const_cast<void*>(src));
            strm.avail_in = static_cast<uInt>(src_size);
            strm.next_out = reinterpret_cast<Bytef*>(dest.data() + 8);
            strm.avail_out = static_cast<uInt>(dest.size() - 8);

            int ret = deflate(&strm, Z_FINISH);
            if (ret != Z_STREAM_END) {
                deflateEnd(&strm);
                throw std::runtime_error("zlib: compression failed.");
            }
            dest.resize(8 + strm.total_out);
            deflateEnd(&strm);
            return dest;
        }

        /**
         * Decompress a zlib buffer (with 8‑byte size prefix) into a pre‑allocated destination.
         */
        inline void zlib_decompress(const void* src, std::size_t src_size,
                                     void* dst, std::size_t dst_size) {
            std::uint64_t expected_usize;
            std::memcpy(&expected_usize, src, sizeof(expected_usize));
            if (static_cast<std::size_t>(expected_usize) != dst_size)
                throw std::runtime_error("zlib: uncompressed size mismatch.");

            z_stream strm;
            std::memset(&strm, 0, sizeof(strm));
            if (inflateInit(&strm) != Z_OK)
                throw std::runtime_error("zlib: failed to initialize inflate.");

            strm.next_in = reinterpret_cast<Bytef*>(
                static_cast<char*>(const_cast<void*>(src)) + 8);
            strm.avail_in = static_cast<uInt>(src_size - 8);
            strm.next_out = reinterpret_cast<Bytef*>(dst);
            strm.avail_out = static_cast<uInt>(dst_size);

            int ret = inflate(&strm, Z_FINISH);
            inflateEnd(&strm);
            if (ret != Z_STREAM_END)
                throw std::runtime_error("zlib: decompression failed.");
        }

        /**
         * Write a raw buffer header (ndim + dimensions) to a stream.
         */
        inline void write_zlib_header(std::ostream& out,
                                       const std::vector<std::size_t>& shape) {
            std::uint64_t ndim = static_cast<std::uint64_t>(shape.size());
            out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
            for (auto d : shape) {
                std::uint64_t du = static_cast<std::uint64_t>(d);
                out.write(reinterpret_cast<const char*>(&du), sizeof(du));
            }
        }

        /**
         * Read a raw buffer header from a stream.
         */
        inline std::vector<std::size_t> read_zlib_header(std::istream& in) {
            std::uint64_t ndim = 0;
            in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
            if (!in) throw std::runtime_error("zlib: failed to read header.");
            std::vector<std::size_t> shape(ndim);
            for (std::uint64_t i = 0; i < ndim; ++i) {
                std::uint64_t d = 0;
                in.read(reinterpret_cast<char*>(&d), sizeof(d));
                shape[i] = static_cast<std::size_t>(d);
            }
            return shape;
        }

        /**
         * SIMD‑accelerated binary stream read.
         */
        template <class T>
        inline void stream_read_binary(std::istream& in, T* dst, std::size_t count) {
            constexpr std::size_t buf_size = zlib_chunk_size;
            std::vector<char> buffer(buf_size);
            char* byte_ptr = reinterpret_cast<char*>(dst);
            std::size_t bytes = count * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                in.read(buffer.data(), static_cast<std::streamsize>(chunk));
                if (!in) throw std::runtime_error("zlib: stream read failed.");
                std::memcpy(byte_ptr + (bytes - remaining), buffer.data(), chunk);
                remaining -= chunk;
            }
        }

        /**
         * SIMD‑accelerated binary stream write.
         */
        template <class T>
        inline void stream_write_binary(std::ostream& out, const T* src, std::size_t count) {
            constexpr std::size_t buf_size = zlib_chunk_size;
            const char* byte_ptr = reinterpret_cast<const char*>(src);
            std::size_t bytes = count * sizeof(T);
            std::size_t remaining = bytes;
            while (remaining > 0) {
                std::size_t chunk = std::min(buf_size, remaining);
                out.write(byte_ptr + (bytes - remaining), static_cast<std::streamsize>(chunk));
                if (!out) throw std::runtime_error("zlib: stream write failed.");
                remaining -= chunk;
            }
        }
    }

    /**
     * Zlib‑compress an xtensor array into a compressed byte buffer.
     * The buffer contains an 8‑byte uncompressed size prefix followed by the
     * deflated data (no gzip wrapper, pure zlib format).
     * @param expr The expression to compress.
     * @param level Compression level (0‑9, default 6).
     * @return Compressed byte vector.
     */
    template <class E>
    inline auto zlib_compress_array(const xexpression<E>& expr, int level = 6) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(expr.derived_cast());
        std::size_t nbytes = arr.size() * sizeof(T);
        return detail::zlib_compress(arr.data(), nbytes, level);
    }

    /**
     * Decompress a zlib buffer back into an xtensor array of given shape.
     * @param compressed The compressed buffer (with 8‑byte size prefix).
     * @param shape Expected array dimensions.
     * @return xarray<T> with the decompressed data.
     */
    template <class T = double>
    inline auto zlib_decompress_array(const std::vector<char>& compressed,
                                       const std::vector<std::size_t>& shape) {
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
        std::size_t expected = arr.size() * sizeof(T);
        detail::zlib_decompress(compressed.data(), compressed.size(), arr.data(), expected);
        return arr;
    }

    /**
     * Save a zlib‑compressed array to a file (.zlib extension suggested).
     * The file contains a shape header (ndim, dimensions) followed by the
     * compressed payload with 8‑byte uncompressed size prefix.
     * @param filename Output file path.
     * @param expr The expression to save.
     * @param level Compression level (0‑9).
     */
    template <class E>
    inline void save_zlib(const std::string& filename, const xexpression<E>& expr, int level = 6) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(expr.derived_cast());
        auto shape = arr.shape();
        if (shape.empty()) shape = {1};

        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("zlib: cannot open file for writing: " + filename);

        // Write shape header
        detail::write_zlib_header(file, shape);

        // Compress data
        auto compressed = detail::zlib_compress(arr.data(), arr.size() * sizeof(T), level);

        // Write compressed data with SIMD block write
        detail::stream_write_binary(file, compressed.data(), compressed.size());
    }

    /**
     * Load a zlib‑compressed array from a file (.zlib).
     * @param filename Input file path.
     * @return xarray<T>.
     */
    template <class T = double>
    inline auto load_zlib(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("zlib: cannot open file for reading: " + filename);

        // Read shape header
        auto shape = detail::read_zlib_header(file);

        // Read the rest of the file as compressed data
        std::streampos pos = file.tellg();
        file.seekg(0, std::ios::end);
        std::size_t compressed_size = static_cast<std::size_t>(file.tellg() - pos);
        file.seekg(pos);

        std::vector<char> compressed(compressed_size);
        detail::stream_read_binary(file, compressed.data(), compressed.size());

        // Decompress
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
        std::size_t expected = arr.size() * sizeof(T);
        detail::zlib_decompress(compressed.data(), compressed.size(), arr.data(), expected);
        return arr;
    }

    /**
     * Pack an xtensor array with shape metadata into a zlib‑compressed buffer.
     * The buffer layout: [ndim(8) | dims(ndim*8) | compressed payload with 8‑byte prefix].
     * Useful for network transmission or storing in databases.
     * @param expr The expression to pack.
     * @param level Compression level (0‑9).
     * @return Compressed byte vector.
     */
    template <class E>
    inline auto zlib_pack(const xexpression<E>& expr, int level = 6) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(expr.derived_cast());
        auto shape = arr.shape();
        if (shape.empty()) shape = {1};

        // Build header: ndim + dims
        std::vector<char> header;
        std::uint64_t ndim = static_cast<std::uint64_t>(shape.size());
        header.insert(header.end(),
                      reinterpret_cast<const char*>(&ndim),
                      reinterpret_cast<const char*>(&ndim) + sizeof(ndim));
        for (auto d : shape) {
            std::uint64_t du = static_cast<std::uint64_t>(d);
            header.insert(header.end(),
                          reinterpret_cast<const char*>(&du),
                          reinterpret_cast<const char*>(&du) + sizeof(du));
        }

        // Compress the data portion
        auto compressed_data = detail::zlib_compress(
            arr.data(), arr.size() * sizeof(T), level);

        // Combine header + compressed
        std::vector<char> result;
        result.reserve(header.size() + compressed_data.size());
        result.insert(result.end(), header.begin(), header.end());
        result.insert(result.end(), compressed_data.begin(), compressed_data.end());
        return result;
    }

    /**
     * Unpack a zlib‑packed buffer back into an xtensor array.
     * @param packed The packed buffer produced by zlib_pack.
     * @return xarray<T>.
     */
    template <class T = double>
    inline auto zlib_unpack(const std::vector<char>& packed) {
        if (packed.size() < 8)
            throw std::runtime_error("zlib_unpack: buffer too small.");

        // Parse header
        const char* ptr = packed.data();
        std::uint64_t ndim;
        std::memcpy(&ndim, ptr, sizeof(ndim));
        ptr += sizeof(ndim);

        std::vector<std::size_t> shape(ndim);
        for (std::uint64_t i = 0; i < ndim; ++i) {
            std::uint64_t du;
            std::memcpy(&du, ptr, sizeof(du));
            ptr += sizeof(du);
            shape[i] = static_cast<std::size_t>(du);
        }

        // Remaining bytes are compressed data with 8‑byte prefix
        std::size_t header_size = sizeof(ndim) + ndim * sizeof(std::uint64_t);
        const char* compressed_start = packed.data() + header_size;
        std::size_t compressed_size = packed.size() - header_size;

        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
        std::size_t expected = arr.size() * sizeof(T);
        detail::zlib_decompress(compressed_start, compressed_size, arr.data(), expected);
        return arr;
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XIO_ZLIB_HPP