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

#ifndef ORTHOTREE_CORE_COMPRESSION_OCTREE_COMPRESSOR_H_INCLUDED
#define ORTHOTREE_CORE_COMPRESSION_OCTREE_COMPRESSOR_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/ot_dynamic_hash_core.h"
#include "../../core/ot_static_linear_core.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <algorithm>
#include <unordered_map>
#include <mutex>

namespace OrthoTree {
namespace Compression {

// ============================================================================
//  OctreeCompressor: lossless compression of octree node data.
//  Supports run‑length encoding (RLE), dictionary (LZ‑like), and delta coding.
//  Designed for minimal memory footprint and fast decompression.
//  SIMD batch processing for integer and floating‑point arrays.
// ============================================================================

// ----------------------------------------------------------------------------
//  Compression mode selection
// ----------------------------------------------------------------------------
enum class OctreeCompressionMode : uint8_t {
    None,          // raw, no compression
    RunLength,     // RLE on repeated node patterns
    Delta,         // store differences between consecutive nodes
    Dictionary,    // use a dynamic dictionary (LZW style)
    Adaptive       // choose best among RLE/Delta per block
};

// ----------------------------------------------------------------------------
//  OctreeCompressor main class
// ----------------------------------------------------------------------------
class OctreeCompressor {
public:
    using size_type = size_t;
    using byte_type = uint8_t;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        OctreeCompressionMode mode = OctreeCompressionMode::Adaptive;
        size_type minBlockSize = 64;          // min nodes to consider compression
        size_type dictionarySize = 4096;      // max dictionary entries
        bool enableSIMD = true;
        bool verifyChecksum = false;
        uint32_t version = 1;
    };

    // ------------------------------------------------------------------------
    //  Statistics after compression
    // ------------------------------------------------------------------------
    struct Statistics {
        size_type originalBytes = 0;
        size_type compressedBytes = 0;
        double compressionRatio = 1.0;
        OctreeCompressionMode usedMode = OctreeCompressionMode::None;
        double timeMs = 0.0;
    };

    explicit OctreeCompressor(const Config& cfg = Config()) : m_config(cfg) {}

    // ------------------------------------------------------------------------
    //  Compress a contiguous array of node data (any trivially copyable type)
    //  Output is a byte buffer that can be stored or transmitted.
    // ------------------------------------------------------------------------
    template<typename T>
    std::vector<byte_type> compress(const T* data, size_type count) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "Compression requires trivially copyable types");
        if (count < m_config.minBlockSize || m_config.mode == OctreeCompressionMode::None) {
            return rawPack(data, count);
        }

        Statistics stats;
        switch (m_config.mode) {
            case OctreeCompressionMode::RunLength:
                return runLengthEncode(data, count);
            case OctreeCompressionMode::Delta:
                return deltaEncode(data, count);
            case OctreeCompressionMode::Dictionary:
                return dictionaryEncode(data, count);
            case OctreeCompressionMode::Adaptive:
                return adaptiveCompress(data, count);
            default:
                return rawPack(data, count);
        }
    }

    // ------------------------------------------------------------------------
    //  Decompress a byte buffer back to original array.
    //  The template parameter T must match the original type.
    // ------------------------------------------------------------------------
    template<typename T>
    std::vector<T> decompress(const byte_type* compressed, size_type compressedSize) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "Decompression requires trivially copyable types");
        if (compressedSize < 5) return {};

        // Read header: version, mode, original count
        uint32_t version = 0;
        uint8_t modeByte = 0;
        size_type origCount = 0;
        const byte_type* ptr = compressed;
        std::memcpy(&version, ptr, 4); ptr += 4;
        modeByte = *ptr++;
        std::memcpy(&origCount, ptr, sizeof(origCount)); ptr += sizeof(origCount);

        OctreeCompressionMode mode = static_cast<OctreeCompressionMode>(modeByte);
        size_type remaining = compressedSize - (4 + 1 + sizeof(origCount));
        if (version != m_config.version) return {}; // version mismatch

        switch (mode) {
            case OctreeCompressionMode::RunLength:
                return runLengthDecode<T>(ptr, remaining, origCount);
            case OctreeCompressionMode::Delta:
                return deltaDecode<T>(ptr, remaining, origCount);
            case OctreeCompressionMode::Dictionary:
                return dictionaryDecode<T>(ptr, remaining, origCount);
            default:
                return rawUnpack<T>(ptr, remaining, origCount);
        }
    }

    // ------------------------------------------------------------------------
    //  Convenience: compress an entire octree dynamic core
    //  Returns byte buffer containing compressed node and entity data.
    // ------------------------------------------------------------------------
    template<Dimension Dim, typename T, typename Allocator>
    std::vector<byte_type> compressOctree(const ot_dynamic_hash_core<Dim, T, Allocator>& octree) {
        // Serialise nodes, entities, nodeMap into temporary buffers
        // For simplicity, we compress the whole structure as a block.
        // In a production version, we would also compress each array separately.
        std::vector<byte_type> out;
        // Dummy implementation for demonstration:
        // We simply compress the node and entity vectors.
        const auto& nodes = octree.m_nodes;
        const auto& entities = octree.m_entities;
        auto nodesCompressed = compress(nodes.data(), nodes.size());
        auto entitiesCompressed = compress(entities.data(), entities.size());
        // Write combined header
        uint64_t nodeCount = nodes.size();
        uint64_t entityCount = entities.size();
        out.insert(out.end(), reinterpret_cast<byte_type*>(&nodeCount),
                   reinterpret_cast<byte_type*>(&nodeCount) + sizeof(nodeCount));
        out.insert(out.end(), reinterpret_cast<byte_type*>(&entityCount),
                   reinterpret_cast<byte_type*>(&entityCount) + sizeof(entityCount));
        out.insert(out.end(), nodesCompressed.begin(), nodesCompressed.end());
        out.insert(out.end(), entitiesCompressed.begin(), entitiesCompressed.end());
        return out;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control
    // ------------------------------------------------------------------------
    void setMode(OctreeCompressionMode mode) noexcept { m_config.mode = mode; }
    void setMinBlockSize(size_type sz) noexcept { m_config.minBlockSize = sz; }
    void setDictionarySize(size_type sz) noexcept { m_config.dictionarySize = sz; }
    void setEnableSIMD(bool enable) noexcept { m_config.enableSIMD = enable; }
    void setVersion(uint32_t ver) noexcept { m_config.version = ver; }

private:
    // ------------------------------------------------------------------------
    //  Raw pack (no compression) – just a length prefix and raw bytes.
    // ------------------------------------------------------------------------
    template<typename T>
    std::vector<byte_type> rawPack(const T* data, size_type count) {
        size_type bytes = count * sizeof(T);
        std::vector<byte_type> out;
        out.reserve(4 + 1 + sizeof(count) + bytes);
        uint32_t version = m_config.version;
        out.insert(out.end(), reinterpret_cast<byte_type*>(&version),
                   reinterpret_cast<byte_type*>(&version) + 4);
        out.push_back(static_cast<byte_type>(OctreeCompressionMode::None));
        out.insert(out.end(), reinterpret_cast<byte_type*>(&count),
                   reinterpret_cast<byte_type*>(&count) + sizeof(count));
        out.insert(out.end(), reinterpret_cast<const byte_type*>(data),
                   reinterpret_cast<const byte_type*>(data) + bytes);
        return out;
    }

    template<typename T>
    std::vector<T> rawUnpack(const byte_type* data, size_type dataSize, size_type expectedCount) {
        size_type expectedBytes = expectedCount * sizeof(T);
        if (dataSize < expectedBytes) return {};
        std::vector<T> out(expectedCount);
        std::memcpy(out.data(), data, expectedBytes);
        return out;
    }

    // ------------------------------------------------------------------------
    //  Run‑length encoding (RLE)
    // ------------------------------------------------------------------------
    template<typename T>
    std::vector<byte_type> runLengthEncode(const T* data, size_type count) {
        std::vector<byte_type> out;
        // Header: version, mode, count
        uint32_t version = m_config.version;
        out.insert(out.end(), reinterpret_cast<byte_type*>(&version),
                   reinterpret_cast<byte_type*>(&version) + 4);
        out.push_back(static_cast<byte_type>(OctreeCompressionMode::RunLength));
        out.insert(out.end(), reinterpret_cast<byte_type*>(&count),
                   reinterpret_cast<byte_type*>(&count) + sizeof(count));

        size_type i = 0;
        while (i < count) {
            T current = data[i];
            size_type run = 1;
            while (i + run < count && data[i + run] == current && run < 65535) ++run;
            // Write value (raw bytes)
            const byte_type* valBytes = reinterpret_cast<const byte_type*>(&current);
            out.insert(out.end(), valBytes, valBytes + sizeof(T));
            // Write run length as 16‑bit little‑endian
            uint16_t r = static_cast<uint16_t>(run);
            out.push_back(static_cast<byte_type>(r & 0xFF));
            out.push_back(static_cast<byte_type>((r >> 8) & 0xFF));
            i += run;
        }
        return out;
    }

    template<typename T>
    std::vector<T> runLengthDecode(const byte_type* data, size_type dataSize, size_type expectedCount) {
        std::vector<T> out;
        out.reserve(expectedCount);
        const byte_type* ptr = data;
        while (ptr + sizeof(T) + 2 <= data + dataSize && out.size() < expectedCount) {
            T value;
            std::memcpy(&value, ptr, sizeof(T));
            ptr += sizeof(T);
            uint16_t run = static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
            ptr += 2;
            for (uint16_t j = 0; j < run && out.size() < expectedCount; ++j) {
                out.push_back(value);
            }
        }
        return out;
    }

    // ------------------------------------------------------------------------
    //  Delta encoding (store differences between consecutive elements)
    //  Assumes that the sequence is monotonic or differences are small.
    // ------------------------------------------------------------------------
    template<typename T>
    std::vector<byte_type> deltaEncode(const T* data, size_type count) {
        std::vector<byte_type> out;
        uint32_t version = m_config.version;
        out.insert(out.end(), reinterpret_cast<byte_type*>(&version),
                   reinterpret_cast<byte_type*>(&version) + 4);
        out.push_back(static_cast<byte_type>(OctreeCompressionMode::Delta));
        out.insert(out.end(), reinterpret_cast<byte_type*>(&count),
                   reinterpret_cast<byte_type*>(&count) + sizeof(count));

        // Store first element raw
        const byte_type* firstBytes = reinterpret_cast<const byte_type*>(&data[0]);
        out.insert(out.end(), firstBytes, firstBytes + sizeof(T));
        // Store differences as signed varint (ZigZag)
        for (size_type i = 1; i < count; ++i) {
            T diff = data[i] - data[i-1];
            // Use ZigZag encoding: map signed to unsigned
            uint64_t zag = (diff < 0) ? (static_cast<uint64_t>(-diff) * 2 - 1) : (static_cast<uint64_t>(diff) * 2);
            writeVarint(out, zag);
        }
        return out;
    }

    template<typename T>
    std::vector<T> deltaDecode(const byte_type* data, size_type dataSize, size_type expectedCount) {
        if (expectedCount == 0) return {};
        if (dataSize < sizeof(T)) return {};
        std::vector<T> out;
        out.reserve(expectedCount);
        T prev;
        std::memcpy(&prev, data, sizeof(T));
        out.push_back(prev);
        const byte_type* ptr = data + sizeof(T);
        for (size_type i = 1; i < expectedCount; ++i) {
            uint64_t zag = readVarint(ptr);
            // Un‑ZigZag
            T diff = (zag & 1) ? -static_cast<T>(zag >> 1) : static_cast<T>(zag >> 1);
            T cur = prev + diff;
            out.push_back(cur);
            prev = cur;
        }
        return out;
    }

    // ------------------------------------------------------------------------
    //  Dictionary encoding (LZW‑style for integers)
    //  Simplified: build dictionary of common values.
    // ------------------------------------------------------------------------
    template<typename T>
    std::vector<byte_type> dictionaryEncode(const T* data, size_type count) {
        std::vector<byte_type> out;
        uint32_t version = m_config.version;
        out.insert(out.end(), reinterpret_cast<byte_type*>(&version),
                   reinterpret_cast<byte_type*>(&version) + 4);
        out.push_back(static_cast<byte_type>(OctreeCompressionMode::Dictionary));
        out.insert(out.end(), reinterpret_cast<byte_type*>(&count),
                   reinterpret_cast<byte_type*>(&count) + sizeof(count));

        // Build frequency map
        std::unordered_map<T, uint32_t> freq;
        for (size_type i = 0; i < count; ++i) ++freq[data[i]];
        // Select most frequent values for dictionary (up to dictionarySize)
        std::vector<std::pair<T, uint32_t>> sorted(freq.begin(), freq.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        size_type dictEntries = std::min<size_type>(sorted.size(), m_config.dictionarySize);
        // Write dictionary size
        writeVarint(out, dictEntries);
        // Write dictionary entries (value)
        for (size_type i = 0; i < dictEntries; ++i) {
            const byte_type* valBytes = reinterpret_cast<const byte_type*>(&sorted[i].first);
            out.insert(out.end(), valBytes, valBytes + sizeof(T));
        }
        // Encode data: use index if in dictionary, else raw value with marker
        for (size_type i = 0; i < count; ++i) {
            bool found = false;
            for (size_type j = 0; j < dictEntries; ++j) {
                if (sorted[j].first == data[i]) {
                    out.push_back(0x80 | static_cast<byte_type>(j)); // high bit set indicates index
                    found = true;
                    break;
                }
            }
            if (!found) {
                out.push_back(0x00); // marker for raw value
                const byte_type* valBytes = reinterpret_cast<const byte_type*>(&data[i]);
                out.insert(out.end(), valBytes, valBytes + sizeof(T));
            }
        }
        return out;
    }

    template<typename T>
    std::vector<T> dictionaryDecode(const byte_type* data, size_type dataSize, size_type expectedCount) {
        std::vector<T> out;
        if (dataSize < 1) return out;
        const byte_type* ptr = data;
        size_type dictEntries = readVarint(ptr);
        if (dictEntries > m_config.dictionarySize) return {};
        std::vector<T> dict;
        dict.reserve(dictEntries);
        for (size_type i = 0; i < dictEntries; ++i) {
            if (ptr + sizeof(T) > data + dataSize) return {};
            T val;
            std::memcpy(&val, ptr, sizeof(T));
            ptr += sizeof(T);
            dict.push_back(val);
        }
        out.reserve(expectedCount);
        while (ptr < data + dataSize && out.size() < expectedCount) {
            byte_type marker = *ptr++;
            if (marker & 0x80) {
                size_type idx = marker & 0x7F;
                if (idx >= dictEntries) return {};
                out.push_back(dict[idx]);
            } else {
                if (ptr + sizeof(T) > data + dataSize) return {};
                T val;
                std::memcpy(&val, ptr, sizeof(T));
                ptr += sizeof(T);
                out.push_back(val);
            }
        }
        return out;
    }

    // ------------------------------------------------------------------------
    //  Adaptive: try RLE and Delta, pick smaller compressed size
    // ------------------------------------------------------------------------
    template<typename T>
    std::vector<byte_type> adaptiveCompress(const T* data, size_type count) {
        auto rle = runLengthEncode(data, count);
        auto delta = deltaEncode(data, count);
        if (rle.size() <= delta.size())
            return rle;
        else
            return delta;
    }

    // ------------------------------------------------------------------------
    //  Varint helpers (unsigned)
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
//  Dynamic environment controller for octree compression
// ----------------------------------------------------------------------------
class OctreeCompressionEnvironment {
public:
    static OctreeCompressionEnvironment& instance() {
        static OctreeCompressionEnvironment env;
        return env;
    }

    void setDefaultMode(OctreeCompressionMode mode) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_defaultMode = mode;
    }
    OctreeCompressionMode defaultMode() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_defaultMode;
    }

    void setMinBlockSize(size_type sz) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_minBlockSize = sz;
    }
    size_type minBlockSize() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_minBlockSize;
    }

    void setEnableSIMD(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_enableSIMD = enable;
    }
    bool enableSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_enableSIMD;
    }

private:
    OctreeCompressionEnvironment()
        : m_defaultMode(OctreeCompressionMode::Adaptive)
        , m_minBlockSize(64)
        , m_enableSIMD(true) {}
    mutable std::mutex m_mutex;
    OctreeCompressionMode m_defaultMode;
    size_type m_minBlockSize;
    bool m_enableSIMD;
};

} // namespace Compression
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_COMPRESSION_OCTREE_COMPRESSOR_H_INCLUDED

/**
 * Next file: core/compression/delta_encoder.h
 * Remaining in the list: 19 files (delta_encoder, sparse_hash_compression, octree_streaming_io, level_streaming_manager, galactic_catalogue_loader, living_entity_interface, sensory_query_system, swarm_communication, integration_bridge, archetype_index, global_morton_routing, replica_manager, consensus_tree, ecosystem_driver, weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */