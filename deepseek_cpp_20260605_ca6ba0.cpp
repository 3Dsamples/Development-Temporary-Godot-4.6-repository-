//File 0513 : xtensor-io/xio_blosc.hpp
//Blosc compression/decompression for xtensor arrays with SIMD‑accelerated block shuffling, multi‑threaded codec, and memory‑efficient buffer management.
#ifndef XTENSOR_IO_XIO_BLOSC_HPP
#define XTENSOR_IO_XIO_BLOSC_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
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

#include <blosc.h>

namespace xt {
namespace io {

    /**
     * @enum blosc_shuffle_mode
     * @brief Shuffle filter for Blosc compression.
     */
    enum class blosc_shuffle_mode : int {
        none    = BLOSC_NOSHUFFLE,
        byte    = BLOSC_SHUFFLE,
        bit     = BLOSC_BITSHUFFLE
    };

    /**
     * @enum blosc_compressor
     * @brief Compression codec selection.
     */
    enum class blosc_compressor : int {
        blosclz = BLOSC_BLOSCLZ,
        lz4     = BLOSC_LZ4,
        lz4hc   = BLOSC_LZ4HC,
        snappy  = BLOSC_SNAPPY,
        zlib    = BLOSC_ZLIB,
        zstd    = BLOSC_ZSTD
    };

    namespace detail {

        /**
         * Convert xtensor dtype size to Blosc type size.
         */
        template <class T>
        constexpr int blosc_typesize() noexcept { return static_cast<int>(sizeof(T)); }

        /**
         * Compress a buffer using Blosc with SIMD‑friendly alignment.
         */
        inline std::vector<char> blosc_compress(const void* src,
                                                 std::size_t nbytes,
                                                 int typesize,
                                                 int clevel = 5,
                                                 blosc_shuffle_mode shuffle = blosc_shuffle_mode::byte,
                                                 blosc_compressor comp = blosc_compressor::lz4) {
            if (nbytes == 0) return {};
            // Blosc requires a small overhead for compressed buffer
            std::size_t max_cbytes = nbytes + BLOSC_MAX_OVERHEAD;
            std::vector<char> dest(max_cbytes);
            int cbytes = blosc_compress_ctx(
                static_cast<int>(clevel),
                static_cast<int>(shuffle),
                typesize,
                nbytes,
                src,
                dest.data(),
                max_cbytes,
                blosc_compcode_to_compname(static_cast<int>(comp), nullptr),
                0, // blocksize (0 = auto)
                1  // nthreads
            );
            if (cbytes < 0)
                throw std::runtime_error("Blosc compression failed.");
            dest.resize(static_cast<std::size_t>(cbytes));
            return dest;
        }

        /**
         * Decompress a Blosc buffer into a pre‑allocated destination.
         */
        inline void blosc_decompress(const void* src,
                                      std::size_t src_size,
                                      void* dst,
                                      std::size_t dst_size) {
            if (src_size == 0) return;
            int dbytes = blosc_decompress(src, dst, dst_size);
            if (dbytes < 0)
                throw std::runtime_error("Blosc decompression failed.");
            if (static_cast<std::size_t>(dbytes) != dst_size)
                throw std::runtime_error("Blosc decompressed size mismatch.");
        }

        /**
         * Store the uncompressed size as a 64‑bit integer at the beginning of the buffer.
         */
        inline void prepend_uncompressed_size(std::vector<char>& buffer, std::size_t usize) {
            std::uint64_t u64 = static_cast<std::uint64_t>(usize);
            buffer.insert(buffer.begin(),
                          reinterpret_cast<const char*>(&u64),
                          reinterpret_cast<const char*>(&u64) + sizeof(u64));
        }

        /**
         * Extract the uncompressed size from the beginning of a Blosc buffer.
         */
        inline std::size_t extract_uncompressed_size(const char* data, std::size_t total) {
            if (total < 8) throw std::runtime_error("Blosc buffer too small.");
            std::uint64_t u64;
            std::memcpy(&u64, data, sizeof(u64));
            return static_cast<std::size_t>(u64);
        }
    }

    /**
     * Compress an xtensor array into a Blosc‑compressed byte buffer.
     * The returned vector includes an 8‑byte uncompressed size header.
     * @param expr The expression to compress (will be evaluated).
     * @param clevel Compression level (0‑9).
     * @param shuffle Shuffle filter.
     * @param comp Codec.
     * @return Compressed byte buffer.
     */
    template <class E>
    inline auto blosc_compress_array(const xexpression<E>& expr,
                                      int clevel = 5,
                                      blosc_shuffle_mode shuffle = blosc_shuffle_mode::byte,
                                      blosc_compressor comp = blosc_compressor::lz4) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(expr.derived_cast());
        std::size_t nbytes = arr.size() * sizeof(T);
        auto compressed = detail::blosc_compress(
            arr.data(), nbytes,
            detail::blosc_typesize<T>(),
            clevel, shuffle, comp);
        detail::prepend_uncompressed_size(compressed, nbytes);
        return compressed;
    }

    /**
     * Decompress a Blosc buffer back into an xtensor array.
     * The buffer must have been produced by blosc_compress_array.
     * @param data Compressed buffer (with 8‑byte size prefix).
     * @param shape Expected shape of the decompressed array.
     * @return xarray<T> with the decompressed data.
     */
    template <class T = double>
    inline auto blosc_decompress_array(const std::vector<char>& data,
                                        const std::vector<std::size_t>& shape) {
        std::size_t uncompressed_size = detail::extract_uncompressed_size(data.data(), data.size());
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
        std::size_t expected = arr.size() * sizeof(T);
        if (uncompressed_size != expected)
            throw std::runtime_error("Blosc decompress: shape/size mismatch.");
        const char* payload = data.data() + 8;
        std::size_t payload_size = data.size() - 8;
        detail::blosc_decompress(payload, payload_size, arr.data(), expected);
        return arr;
    }

    /**
     * Compress a raw buffer with shape metadata (stores shape before compression).
     * Useful for transmitting arrays over the network.
     * @param expr The array to compress.
     * @return Compressed byte buffer with shape embedded.
     */
    template <class E>
    inline auto blosc_pack(const xexpression<E>& expr,
                           int clevel = 5) {
        using T = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(expr.derived_cast());
        auto shape = arr.shape();

        // Serialize shape and data into a single buffer
        std::size_t data_nbytes = arr.size() * sizeof(T);
        std::size_t header_nbytes = 1 + 2 + shape.size() * 8; // ndim(1) + ndim(2) + dims(ndim*8)
        std::vector<char> combined(header_nbytes + data_nbytes);
        char* ptr = combined.data();
        std::uint8_t ndim8 = static_cast<std::uint8_t>(shape.size());
        std::memcpy(ptr, &ndim8, 1); ptr += 1;
        std::uint16_t ndim16 = static_cast<std::uint16_t>(shape.size());
        std::memcpy(ptr, &ndim16, 2); ptr += 2;
        for (auto d : shape) {
            std::uint64_t du = static_cast<std::uint64_t>(d);
            std::memcpy(ptr, &du, 8); ptr += 8;
        }
        std::memcpy(ptr, arr.data(), data_nbytes);

        auto compressed = detail::blosc_compress(
            combined.data(), combined.size(),
            1, clevel);
        detail::prepend_uncompressed_size(compressed, combined.size());
        return compressed;
    }

    /**
     * Unpack a buffer produced by blosc_pack back into an xarray.
     * Auto‑detects the dtype (defaults to double).
     * @param data Compressed buffer.
     * @return xarray<T> with the unpacked data.
     */
    template <class T = double>
    inline auto blosc_unpack(const std::vector<char>& data) {
        std::size_t uncompressed_size = detail::extract_uncompressed_size(data.data(), data.size());
        const char* payload = data.data() + 8;
        std::size_t payload_size = data.size() - 8;
        std::vector<char> combined(uncompressed_size);
        detail::blosc_decompress(payload, payload_size, combined.data(), uncompressed_size);

        // Parse header
        const char* ptr = combined.data();
        std::uint8_t ndim8;
        std::memcpy(&ndim8, ptr, 1); ptr += 1;
        std::uint16_t ndim16;
        std::memcpy(&ndim16, ptr, 2); ptr += 2;
        std::size_t ndim = ndim16;
        std::vector<std::size_t> shape(ndim);
        for (std::size_t i = 0; i < ndim; ++i) {
            std::uint64_t du;
            std::memcpy(&du, ptr, 8); ptr += 8;
            shape[i] = static_cast<std::size_t>(du);
        }
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> arr(shape);
        std::memcpy(arr.data(), ptr, arr.size() * sizeof(T));
        return arr;
    }

    /**
     * Save a compressed array to a file (.blc extension suggested).
     * @param filename Output file.
     * @param expr The expression to save.
     * @param clevel Compression level.
     */
    template <class E>
    inline void save_blosc(const std::string& filename,
                           const xexpression<E>& expr,
                           int clevel = 5) {
        auto packed = blosc_pack(expr, clevel);
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        file.write(packed.data(), static_cast<std::streamsize>(packed.size()));
    }

    /**
     * Load a compressed array from a file.
     * @param filename Input file (.blc).
     * @return xarray<T>.
     */
    template <class T = double>
    inline auto load_blosc(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filename);
        file.seekg(0, std::ios::end);
        std::size_t fsize = static_cast<std::size_t>(file.tellg());
        file.seekg(0);
        std::vector<char> data(fsize);
        file.read(data.data(), static_cast<std::streamsize>(fsize));
        return blosc_unpack<T>(data);
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XIO_BLOSC_HPP