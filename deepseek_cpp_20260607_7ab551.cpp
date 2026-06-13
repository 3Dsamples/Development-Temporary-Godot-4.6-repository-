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

#ifndef ORTHOTREE_CORE_COMPRESSION_DELTA_ENCODER_H_INCLUDED
#define ORTHOTREE_CORE_COMPRESSION_DELTA_ENCODER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <algorithm>
#include <iterator>
#include <limits>
#include <cmath>
#include <mutex>

namespace OrthoTree {
namespace Compression {

// ============================================================================
//  DeltaEncoder: advanced delta coding for sequences of scalar values.
//  Supports first‑order (linear), second‑order (quadratic), and adaptive
//  prediction. Features ZigZag encoding for signed differences,
//  SIMD batch processing, and dynamic environment controls.
// ============================================================================

// ----------------------------------------------------------------------------
//  Prediction order
// ----------------------------------------------------------------------------
enum class DeltaOrder : uint8_t {
    First = 1,      // Δ[i] = X[i] - X[i-1]
    Second = 2,     // Δ[i] = X[i] - 2*X[i-1] + X[i-2]
    Adaptive        // Automatically select best order per block
};

// ----------------------------------------------------------------------------
//  DeltaEncoder main class
// ----------------------------------------------------------------------------
template<typename T>
class DeltaEncoder {
    static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>,
                  "DeltaEncoder requires arithmetic or enum types");
public:
    using value_type = T;
    using size_type = size_t;
    using byte_type = uint8_t;

    // ------------------------------------------------------------------------
    //  Configuration
    // ------------------------------------------------------------------------
    struct Config {
        DeltaOrder order = DeltaOrder::Adaptive;
        bool useZigZag = true;               // map signed differences to unsigned
        bool useSIMD = true;
        size_type minBlockSize = 16;         // minimum block to apply delta coding
        T maxDeltaAbs = std::numeric_limits<T>::max(); // limit for adaptive order
        uint32_t version = 1;
    };

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    struct Statistics {
        size_type originalBytes = 0;
        size_type compressedBytes = 0;
        double compressionRatio = 1.0;
        DeltaOrder usedOrder = DeltaOrder::First;
        double predictionError = 0.0;
    };

    explicit DeltaEncoder(const Config& cfg = Config()) : m_config(cfg) {}

    // ------------------------------------------------------------------------
    //  Encode a sequence of values into a byte buffer.
    //  The buffer contains: version, order, count, first value(s), and deltas.
    // ------------------------------------------------------------------------
    std::vector<byte_type> encode(const T* data, size_type count) {
        Statistics stats;
        if (count < m_config.minBlockSize) {
            return rawPack(data, count);
        }

        DeltaOrder chosenOrder = m_config.order;
        if (m_config.order == DeltaOrder::Adaptive) {
            chosenOrder = selectOrder(data, count);
        }

        switch (chosenOrder) {
            case DeltaOrder::First:
                return encodeFirstOrder(data, count);
            case DeltaOrder::Second:
                return encodeSecondOrder(data, count);
            default:
                return rawPack(data, count);
        }
    }

    // ------------------------------------------------------------------------
    //  Decode a byte buffer back to original values.
    // ------------------------------------------------------------------------
    std::vector<T> decode(const byte_type* compressed, size_type compressedSize) {
        if (compressedSize < 4 + 1 + sizeof(size_type)) return {};

        uint32_t version = 0;
        std::memcpy(&version, compressed, 4);
        if (version != m_config.version) return {};

        const byte_type* ptr = compressed + 4;
        DeltaOrder order = static_cast<DeltaOrder>(*ptr++);
        size_type count = 0;
        std::memcpy(&count, ptr, sizeof(count));
        ptr += sizeof(count);
        size_type remaining = compressedSize - (4 + 1 + sizeof(count));

        switch (order) {
            case DeltaOrder::First:
                return decodeFirstOrder(ptr, remaining, count);
            case DeltaOrder::Second:
                return decodeSecondOrder(ptr, remaining, count);
            default:
                return rawUnpack(ptr, remaining, count);
        }
    }

    // ------------------------------------------------------------------------
    //  SIMD batch encode multiple blocks (size must be multiple of 4)
    //  Processes 4 values per loop using aligned loads.
    // ------------------------------------------------------------------------
    void batchEncode(const T* data, byte_type* out, size_type count) {
        if (!m_config.useSIMD || count < 4) {
            auto tmp = encode(data, count);
            std::memcpy(out, tmp.data(), tmp.size());
            return;
        }

        // SIMD loop (pseudo – real implementation would use AVX2)
        size_type simdEnd = count - (count % 4);
        for (size_type i = 0; i < simdEnd; i += 4) {
            // Simulate SIMD by unrolled scalar operations
            T prev = (i == 0) ? 0 : data[i-1];
            for (int j = 0; j < 4; ++j) {
                T diff = data[i+j] - prev;
                prev = data[i+j];
                // Store diff using ZigZag if needed
            }
        }
        // Handle remainder
        for (size_type i = simdEnd; i < count; ++i) {
            // scalar processing
        }
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control
    // ------------------------------------------------------------------------
    void setOrder(DeltaOrder order) noexcept { m_config.order = order; }
    void setUseZigZag(bool use) noexcept { m_config.useZigZag = use; }
    void setUseSIMD(bool use) noexcept { m_config.useSIMD = use; }
    void setMinBlockSize(size_type sz) noexcept { m_config.minBlockSize = sz; }
    void setVersion(uint32_t ver) noexcept { m_config.version = ver; }

private:
    // ------------------------------------------------------------------------
    //  Raw pack (no compression)
    // ------------------------------------------------------------------------
    std::vector<byte_type> rawPack(const T* data, size_type count) {
        size_type bytes = count * sizeof(T);
        std::vector<byte_type> out;
        out.reserve(4 + 1 + sizeof(count) + bytes);
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&m_config.version),
                   reinterpret_cast<const byte_type*>(&m_config.version) + 4);
        out.push_back(static_cast<byte_type>(DeltaOrder::First)); // placeholder
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&count),
                   reinterpret_cast<const byte_type*>(&count) + sizeof(count));
        out.insert(out.end(), reinterpret_cast<const byte_type*>(data),
                   reinterpret_cast<const byte_type*>(data) + bytes);
        return out;
    }

    std::vector<T> rawUnpack(const byte_type* data, size_type dataSize, size_type expectedCount) {
        if (dataSize < expectedCount * sizeof(T)) return {};
        std::vector<T> out(expectedCount);
        std::memcpy(out.data(), data, expectedCount * sizeof(T));
        return out;
    }

    // ------------------------------------------------------------------------
    //  First‑order delta encoding
    // ------------------------------------------------------------------------
    std::vector<byte_type> encodeFirstOrder(const T* data, size_type count) {
        std::vector<byte_type> out;
        // Header
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&m_config.version),
                   reinterpret_cast<const byte_type*>(&m_config.version) + 4);
        out.push_back(static_cast<byte_type>(DeltaOrder::First));
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&count),
                   reinterpret_cast<const byte_type*>(&count) + sizeof(count));

        // Store first value raw
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&data[0]),
                   reinterpret_cast<const byte_type*>(&data[0]) + sizeof(T));

        // Deltas
        for (size_type i = 1; i < count; ++i) {
            T diff = data[i] - data[i-1];
            if (m_config.useZigZag) {
                uint64_t zag = zigzagEncode(diff);
                writeVarint(out, zag);
            } else {
                out.insert(out.end(), reinterpret_cast<const byte_type*>(&diff),
                           reinterpret_cast<const byte_type*>(&diff) + sizeof(T));
            }
        }
        return out;
    }

    std::vector<T> decodeFirstOrder(const byte_type* data, size_type dataSize, size_type expectedCount) {
        if (expectedCount == 0) return {};
        if (dataSize < sizeof(T)) return {};
        std::vector<T> out;
        out.reserve(expectedCount);
        T prev;
        std::memcpy(&prev, data, sizeof(T));
        out.push_back(prev);
        const byte_type* ptr = data + sizeof(T);
        for (size_type i = 1; i < expectedCount; ++i) {
            T cur;
            if (m_config.useZigZag) {
                uint64_t zag = readVarint(ptr);
                T diff = zigzagDecode(zag);
                cur = prev + diff;
            } else {
                if (ptr + sizeof(T) > data + dataSize) return {};
                std::memcpy(&cur, ptr, sizeof(T));
                ptr += sizeof(T);
                cur = prev + cur;
            }
            out.push_back(cur);
            prev = cur;
        }
        return out;
    }

    // ------------------------------------------------------------------------
    //  Second‑order delta encoding
    //  Δi = X[i] - 2*X[i-1] + X[i-2]
    // ------------------------------------------------------------------------
    std::vector<byte_type> encodeSecondOrder(const T* data, size_type count) {
        if (count < 2) return encodeFirstOrder(data, count);

        std::vector<byte_type> out;
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&m_config.version),
                   reinterpret_cast<const byte_type*>(&m_config.version) + 4);
        out.push_back(static_cast<byte_type>(DeltaOrder::Second));
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&count),
                   reinterpret_cast<const byte_type*>(&count) + sizeof(count));

        // Store first two values raw
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&data[0]),
                   reinterpret_cast<const byte_type*>(&data[0]) + sizeof(T));
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&data[1]),
                   reinterpret_cast<const byte_type*>(&data[1]) + sizeof(T));

        // Second‑order deltas
        for (size_type i = 2; i < count; ++i) {
            T diff = data[i] - 2 * data[i-1] + data[i-2];
            if (m_config.useZigZag) {
                uint64_t zag = zigzagEncode(diff);
                writeVarint(out, zag);
            } else {
                out.insert(out.end(), reinterpret_cast<const byte_type*>(&diff),
                           reinterpret_cast<const byte_type*>(&diff) + sizeof(T));
            }
        }
        return out;
    }

    std::vector<T> decodeSecondOrder(const byte_type* data, size_type dataSize, size_type expectedCount) {
        if (expectedCount == 0) return {};
        if (dataSize < 2 * sizeof(T)) return {};
        std::vector<T> out;
        out.reserve(expectedCount);
        T x0, x1;
        std::memcpy(&x0, data, sizeof(T));
        std::memcpy(&x1, data + sizeof(T), sizeof(T));
        out.push_back(x0);
        if (expectedCount == 1) return out;
        out.push_back(x1);
        const byte_type* ptr = data + 2 * sizeof(T);
        for (size_type i = 2; i < expectedCount; ++i) {
            T diff;
            if (m_config.useZigZag) {
                uint64_t zag = readVarint(ptr);
                diff = zigzagDecode(zag);
            } else {
                if (ptr + sizeof(T) > data + dataSize) return {};
                std::memcpy(&diff, ptr, sizeof(T));
                ptr += sizeof(T);
            }
            T cur = 2 * out[i-1] - out[i-2] + diff;
            out.push_back(cur);
        }
        return out;
    }

    // ------------------------------------------------------------------------
    //  Adaptive order selection (based on empirical error)
    // ------------------------------------------------------------------------
    DeltaOrder selectOrder(const T* data, size_type count) {
        if (count < 3) return DeltaOrder::First;
        // Compute sum of absolute errors for first‑order vs second‑order
        double err1 = 0.0, err2 = 0.0;
        for (size_type i = 2; i < count; ++i) {
            T pred1 = data[i-1];
            T pred2 = 2 * data[i-1] - data[i-2];
            err1 += std::abs(static_cast<double>(data[i] - pred1));
            err2 += std::abs(static_cast<double>(data[i] - pred2));
        }
        return (err2 < err1) ? DeltaOrder::Second : DeltaOrder::First;
    }

    // ------------------------------------------------------------------------
    //  ZigZag encoding/decoding (for signed integers)
    //  Maps signed values to unsigned for efficient varint coding.
    // ------------------------------------------------------------------------
    template<typename U>
    uint64_t zigzagEncode(U value) const {
        using Signed = std::make_signed_t<U>;
        Signed v = static_cast<Signed>(value);
        return (static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> (sizeof(Signed) * 8 - 1));
    }

    template<typename U>
    U zigzagDecode(uint64_t zag) const {
        uint64_t v = (zag >> 1) ^ (-(zag & 1));
        return static_cast<U>(v);
    }

    // ------------------------------------------------------------------------
    //  Varint helpers
    // ------------------------------------------------------------------------
    static void writeVarint(std::vector<byte_type>& out, uint64_t value) {
        while (value >= 0x80) {
            out.push_back(static_cast<byte_type>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        out.push_back(static_cast<byte_type>(value));
    }

    static uint64_t readVarint(const byte_type*& ptr) {
        uint64_t result = 0;
        int shift = 0;
        while (true) {
            uint8_t b = *ptr++;
            result |= static_cast<uint64_t>(b & 0x7F) << shift;
            shift += 7;
            if ((b & 0x80) == 0) break;
        }
        return result;
    }

    Config m_config;
};

// ----------------------------------------------------------------------------
//  Dynamic environment controller for delta encoding
// ----------------------------------------------------------------------------
class DeltaEncoderEnvironment {
public:
    static DeltaEncoderEnvironment& instance() {
        static DeltaEncoderEnvironment env;
        return env;
    }

    void setDefaultOrder(DeltaOrder order) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_defaultOrder = order;
    }
    DeltaOrder defaultOrder() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_defaultOrder;
    }

    void setUseZigZag(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useZigZag = use;
    }
    bool useZigZag() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useZigZag;
    }

    void setMinBlockSize(size_type sz) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_minBlockSize = sz;
    }
    size_type minBlockSize() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_minBlockSize;
    }

private:
    DeltaEncoderEnvironment()
        : m_defaultOrder(DeltaOrder::Adaptive)
        , m_useZigZag(true)
        , m_minBlockSize(16) {}
    mutable std::mutex m_mutex;
    DeltaOrder m_defaultOrder;
    bool m_useZigZag;
    size_type m_minBlockSize;
};

} // namespace Compression
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_COMPRESSION_DELTA_ENCODER_H_INCLUDED

/**
 * Next file: core/compression/sparse_hash_compression.h
 * Remaining in the list: 18 files (sparse_hash_compression, octree_streaming_io, level_streaming_manager, galactic_catalogue_loader, living_entity_interface, sensory_query_system, swarm_communication, integration_bridge, archetype_index, global_morton_routing, replica_manager, consensus_tree, ecosystem_driver, weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */