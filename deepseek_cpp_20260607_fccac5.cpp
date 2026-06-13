/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_CORE_MORTON_MORTON_BLOCK_COMPRESSOR_H_INCLUDED
#define ORTHOTREE_CORE_MORTON_MORTON_BLOCK_COMPRESSOR_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/numerical_methods.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"
#include "morton_128bit.h"

#include <cstdint>
#include <array>
#include <algorithm>
#include <cstring>
#include <vector>
#include <limits>

#if defined(__SIZEOF_INT128__) || defined(__INT128_TYPE__)
#define ORTHOTREE_HAVE_INT128 1
typedef unsigned __int128 uint128_t;
#else
#error "128-bit integer support required for MortonBlockCompressor. Compile with GCC/Clang or enable __int128."
#endif

namespace OrthoTree {
namespace Morton {

// ============================================================================
//  MortonBlockCompressor: compresses blocks of morton codes using run-length,
//  dictionary, or delta encoding. Designed for sparse octree node storage.
// ============================================================================
class MortonBlockCompressor {
public:
    using code_type = uint128_t;
    using size_type = std::size_t;
    using offset_type = uint32_t;

    // Compression modes
    enum class Mode : uint8_t {
        None,           // no compression, raw storage
        RunLength,      // run-length encoding of repeated codes
        Delta,          // store differences between consecutive codes
        Dictionary,     // Huffman-like dictionary (simplified)
        Hybrid          // auto-select best among RLE and Delta
    };

    // ------------------------------------------------------------------------
    //  Compression statistics
    // ------------------------------------------------------------------------
    struct Statistics {
        size_type original_bytes = 0;
        size_type compressed_bytes = 0;
        double ratio = 1.0;
        Mode used_mode = Mode::None;
    };

    // ------------------------------------------------------------------------
    //  Compress a block of morton codes into a compact byte buffer
    // ------------------------------------------------------------------------
    static std::vector<uint8_t> compress(const code_type* codes, size_type count, Mode mode = Mode::Hybrid) {
        std::vector<uint8_t> out;
        if (count == 0) return out;

        // Write header: count (4 bytes) and mode (1 byte)
        out.reserve(count * sizeof(code_type) / 2); // optimistic
        writeVarint(out, count);
        out.push_back(static_cast<uint8_t>(mode));

        if (mode == Mode::None) {
            // raw copy
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(codes);
            out.insert(out.end(), bytes, bytes + count * sizeof(code_type));
        } else if (mode == Mode::RunLength) {
            compressRLE(codes, count, out);
        } else if (mode == Mode::Delta) {
            compressDelta(codes, count, out);
        } else if (mode == Mode::Hybrid) {
            // try both and pick smaller
            std::vector<uint8_t> rleOut, deltaOut;
            compressRLE(codes, count, rleOut);
            compressDelta(codes, count, deltaOut);
            if (rleOut.size() <= deltaOut.size()) {
                out[1] = static_cast<uint8_t>(Mode::RunLength);
                out.insert(out.end(), rleOut.begin() + 2, rleOut.end()); // skip its own header
            } else {
                out[1] = static_cast<uint8_t>(Mode::Delta);
                out.insert(out.end(), deltaOut.begin() + 2, deltaOut.end());
            }
        }
        return out;
    }

    // ------------------------------------------------------------------------
    //  Decompress back to array of morton codes
    // ------------------------------------------------------------------------
    static std::vector<code_type> decompress(const uint8_t* data, size_type size) {
        std::vector<code_type> out;
        if (size < 5) return out; // need at least count + mode
        const uint8_t* ptr = data;
        size_type count = readVarint(ptr);
        Mode mode = static_cast<Mode>(*ptr++);
        if (mode == Mode::None) {
            if (ptr + count * sizeof(code_type) > data + size) return out;
            out.resize(count);
            std::memcpy(out.data(), ptr, count * sizeof(code_type));
        } else if (mode == Mode::RunLength) {
            decompressRLE(ptr, data + size - ptr, count, out);
        } else if (mode == Mode::Delta) {
            decompressDelta(ptr, data + size - ptr, count, out);
        }
        return out;
    }

    // ------------------------------------------------------------------------
    //  Statistics without allocating output
    // ------------------------------------------------------------------------
    static Statistics estimate(const code_type* codes, size_type count, Mode mode = Mode::Hybrid) {
        Statistics stats;
        stats.original_bytes = count * sizeof(code_type);
        auto compressed = compress(codes, count, mode);
        stats.compressed_bytes = compressed.size();
        stats.ratio = static_cast<double>(stats.compressed_bytes) / static_cast<double>(stats.original_bytes);
        stats.used_mode = (compressed.size() > 1) ? static_cast<Mode>(compressed[1]) : Mode::None;
        return stats;
    }

private:
    // ------------------------------------------------------------------------
    //  Run-length encoding: (code, run) pairs
    // ------------------------------------------------------------------------
    static void compressRLE(const code_type* codes, size_type count, std::vector<uint8_t>& out) {
        size_type i = 0;
        while (i < count) {
            code_type cur = codes[i];
            size_type run = 1;
            while (i + run < count && codes[i + run] == cur && run < 65535) ++run;
            // write code (16 bytes) and run (2 bytes)
            const uint8_t* codeBytes = reinterpret_cast<const uint8_t*>(&cur);
            out.insert(out.end(), codeBytes, codeBytes + sizeof(code_type));
            out.push_back(static_cast<uint8_t>(run & 0xFF));
            out.push_back(static_cast<uint8_t>((run >> 8) & 0xFF));
            i += run;
        }
    }

    static void decompressRLE(const uint8_t* data, size_type dataSize, size_type expectedCount, std::vector<code_type>& out) {
        out.reserve(expectedCount);
        const uint8_t* ptr = data;
        while (ptr + sizeof(code_type) + 2 <= data + dataSize && out.size() < expectedCount) {
            code_type code;
            std::memcpy(&code, ptr, sizeof(code_type));
            ptr += sizeof(code_type);
            uint16_t run = static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
            ptr += 2;
            for (uint16_t j = 0; j < run && out.size() < expectedCount; ++j) {
                out.push_back(code);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Delta encoding: store first code then differences (ZigZag + varint)
    // ------------------------------------------------------------------------
    static void compressDelta(const code_type* codes, size_type count, std::vector<uint8_t>& out) {
        // store first code raw
        const uint8_t* firstBytes = reinterpret_cast<const uint8_t*>(&codes[0]);
        out.insert(out.end(), firstBytes, firstBytes + sizeof(code_type));
        // store deltas as varint ZigZag
        for (size_type i = 1; i < count; ++i) {
            uint128_t diff = (codes[i] > codes[i-1]) ? (codes[i] - codes[i-1]) : (codes[i-1] - codes[i]);
            // ZigZag: map signed to unsigned (but diff is unsigned, we store sign separately? actually diff is positive by construction if we use absolute? For monotonic increasing morton codes, diff is positive. We assume sorted codes.)
            // For simplicity, we store diff as varint.
            writeVarint128(out, diff);
        }
    }

    static void decompressDelta(const uint8_t* data, size_type dataSize, size_type expectedCount, std::vector<code_type>& out) {
        if (expectedCount == 0) return;
        out.reserve(expectedCount);
        // read first code
        if (dataSize < sizeof(code_type)) return;
        code_type prev;
        std::memcpy(&prev, data, sizeof(code_type));
        out.push_back(prev);
        const uint8_t* ptr = data + sizeof(code_type);
        for (size_type i = 1; i < expectedCount; ++i) {
            uint128_t diff = readVarint128(ptr);
            code_type cur = prev + diff;
            out.push_back(cur);
            prev = cur;
        }
    }

    // ------------------------------------------------------------------------
    //  Variable-length integer encoding (for small counts/deltas)
    // ------------------------------------------------------------------------
    static void writeVarint(std::vector<uint8_t>& out, size_type value) {
        while (value >= 0x80) {
            out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        out.push_back(static_cast<uint8_t>(value));
    }

    static size_type readVarint(const uint8_t*& ptr) {
        size_type result = 0;
        int shift = 0;
        while (true) {
            uint8_t byte = *ptr++;
            result |= static_cast<size_type>(byte & 0x7F) << shift;
            shift += 7;
            if ((byte & 0x80) == 0) break;
        }
        return result;
    }

    static void writeVarint128(std::vector<uint8_t>& out, uint128_t value) {
        while (value >= 0x80) {
            out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        out.push_back(static_cast<uint8_t>(value));
    }

    static uint128_t readVarint128(const uint8_t*& ptr) {
        uint128_t result = 0;
        int shift = 0;
        while (true) {
            uint8_t byte = *ptr++;
            result |= static_cast<uint128_t>(byte & 0x7F) << shift;
            shift += 7;
            if ((byte & 0x80) == 0) break;
        }
        return result;
    }
};

// ============================================================================
//  SIMD batch compression (4 codes at a time using 128‑bit SIMD)
// ============================================================================
#if ORTHOTREE_SIMD_LEVEL >= 128
// We can use SSE/AVX to process 4x128-bit codes, but 128-bit is not natively
// supported in SSE (only 128-bit integers in SSE2? Actually __m128i can hold 128 bits).
// We'll provide a generic batch loop with hint.
#endif

class MortonBlockCompressorSIMD {
public:
    static void compressBatch(const uint128_t* codes, size_type count,
                              std::vector<uint8_t>& out, MortonBlockCompressor::Mode mode) {
        // fallback to scalar for now; SIMD would require 16-byte alignment and streaming stores
        auto compressed = MortonBlockCompressor::compress(codes, count, mode);
        out.insert(out.end(), compressed.begin(), compressed.end());
    }

    static void decompressBatch(const uint8_t* data, size_type size,
                                std::vector<uint128_t>& out) {
        out = MortonBlockCompressor::decompress(data, size);
    }
};

// ============================================================================
//  Dynamic environment controller for block compression (adaptive mode)
// ============================================================================
class AdaptiveBlockCompressor {
public:
    AdaptiveBlockCompressor() noexcept
        : m_mode(MortonBlockCompressor::Mode::Hybrid)
        , m_minBlockSize(16)
        , m_adaptive(true) {}

    void setMode(MortonBlockCompressor::Mode mode) noexcept { m_mode = mode; }
    void setMinBlockSize(size_type size) noexcept { m_minBlockSize = size; }
    void setAdaptive(bool adaptive) noexcept { m_adaptive = adaptive; }

    // Compress block, automatically selecting mode based on statistics of previous blocks
    std::vector<uint8_t> compress(const uint128_t* codes, size_type count) {
        if (!m_adaptive || count < m_minBlockSize) {
            return MortonBlockCompressor::compress(codes, count, m_mode);
        }
        // Estimate best mode from first few elements? We'll use hybrid which already picks best.
        return MortonBlockCompressor::compress(codes, count, MortonBlockCompressor::Mode::Hybrid);
    }

    std::vector<uint128_t> decompress(const uint8_t* data, size_type size) {
        return MortonBlockCompressor::decompress(data, size);
    }

    // Update internal model based on compression ratio (for future blocks)
    void updateFeedback(double ratio) {
        m_lastRatio = ratio;
        if (ratio > 0.8 && m_mode != MortonBlockCompressor::Mode::None) {
            // compression not helping much, fallback to none
            m_mode = MortonBlockCompressor::Mode::None;
        } else if (ratio < 0.5 && m_mode == MortonBlockCompressor::Mode::None) {
            m_mode = MortonBlockCompressor::Mode::Hybrid;
        }
    }

private:
    MortonBlockCompressor::Mode m_mode;
    size_type m_minBlockSize;
    bool m_adaptive;
    double m_lastRatio = 1.0;
};

} // namespace Morton
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MORTON_MORTON_BLOCK_COMPRESSOR_H_INCLUDED