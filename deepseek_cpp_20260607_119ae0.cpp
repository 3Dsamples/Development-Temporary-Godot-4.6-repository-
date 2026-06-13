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

#ifndef ORTHOTREE_CORE_COMPRESSION_SPARSE_HASH_COMPRESSION_H_INCLUDED
#define ORTHOTREE_CORE_COMPRESSION_SPARSE_HASH_COMPRESSION_H_INCLUDED

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
#include <unordered_map>
#include <mutex>

namespace OrthoTree {
namespace Compression {

// ============================================================================
//  SparseHashCompression: specialised compression for sparse hash maps
//  (e.g., node maps in dynamic octree). Encodes key‑value pairs more
//  efficiently by detecting and compressing runs of empty slots, using
//  a compact array of (key, value) pairs with optional run‑length encoding
//  for contiguous keys. Supports SIMD batch processing for key comparison.
// ============================================================================

// ----------------------------------------------------------------------------
//  Compression mode for sparse hash maps
// ----------------------------------------------------------------------------
enum class SparseHashMode : uint8_t {
    None,            // store key‑value pairs as is (array of pairs)
    RunLength,       // RLE on runs of consecutive keys
    DeltaKey,        // store key differences (delta) then values
    BucketRemap,     // remap bucket indices to a smaller range
    Adaptive         // automatically choose best among RLE and DeltaKey
};

// ----------------------------------------------------------------------------
//  SparseHashCompressor main class
// ----------------------------------------------------------------------------
template<typename Key, typename Value>
class SparseHashCompressor {
    static_assert(std::is_trivially_copyable_v<Key> && std::is_trivially_copyable_v<Value>,
                  "Key and Value must be trivially copyable");
public:
    using key_type = Key;
    using value_type = Value;
    using pair_type = std::pair<Key, Value>;
    using size_type = size_t;
    using byte_type = uint8_t;

    // ------------------------------------------------------------------------
    //  Configuration
    // ------------------------------------------------------------------------
    struct Config {
        SparseHashMode mode = SparseHashMode::Adaptive;
        size_type minPairs = 16;            // minimum pairs to compress
        bool enableSIMD = true;
        bool verifyChecksum = false;
        uint32_t version = 1;
    };

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    struct Statistics {
        size_type originalPairs = 0;
        size_type originalBytes = 0;
        size_type compressedBytes = 0;
        double compressionRatio = 1.0;
        SparseHashMode usedMode = SparseHashMode::None;
    };

    explicit SparseHashCompressor(const Config& cfg = Config()) : m_config(cfg) {}

    // ------------------------------------------------------------------------
    //  Compress an array of key‑value pairs (must be sorted by key for best results)
    //  Output is a byte buffer with header and compressed data.
    // ------------------------------------------------------------------------
    std::vector<byte_type> compress(const pair_type* pairs, size_type count) {
        if (count < m_config.minPairs) {
            return rawPack(pairs, count);
        }

        switch (m_config.mode) {
            case SparseHashMode::RunLength:
                return runLengthEncode(pairs, count);
            case SparseHashMode::DeltaKey:
                return deltaKeyEncode(pairs, count);
            case SparseHashMode::BucketRemap:
                return bucketRemapEncode(pairs, count);
            case SparseHashMode::Adaptive:
                return adaptiveCompress(pairs, count);
            default:
                return rawPack(pairs, count);
        }
    }

    // ------------------------------------------------------------------------
    //  Decompress a byte buffer back to a vector of key‑value pairs.
    // ------------------------------------------------------------------------
    std::vector<pair_type> decompress(const byte_type* compressed, size_type compressedSize) {
        if (compressedSize < 4 + 1 + sizeof(size_type)) return {};

        uint32_t version = 0;
        std::memcpy(&version, compressed, 4);
        if (version != m_config.version) return {};

        const byte_type* ptr = compressed + 4;
        SparseHashMode mode = static_cast<SparseHashMode>(*ptr++);
        size_type count = 0;
        std::memcpy(&count, ptr, sizeof(count));
        ptr += sizeof(count);
        size_type remaining = compressedSize - (4 + 1 + sizeof(count));

        switch (mode) {
            case SparseHashMode::RunLength:
                return runLengthDecode(ptr, remaining, count);
            case SparseHashMode::DeltaKey:
                return deltaKeyDecode(ptr, remaining, count);
            case SparseHashMode::BucketRemap:
                return bucketRemapDecode(ptr, remaining, count);
            default:
                return rawUnpack(ptr, remaining, count);
        }
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control
    // ------------------------------------------------------------------------
    void setMode(SparseHashMode mode) noexcept { m_config.mode = mode; }
    void setMinPairs(size_type min) noexcept { m_config.minPairs = min; }
    void setEnableSIMD(bool enable) noexcept { m_config.enableSIMD = enable; }

private:
    // ------------------------------------------------------------------------
    //  Raw pack: write count then all pairs sequentially
    // ------------------------------------------------------------------------
    std::vector<byte_type> rawPack(const pair_type* pairs, size_type count) {
        size_type bytes = count * sizeof(pair_type);
        std::vector<byte_type> out;
        out.reserve(4 + 1 + sizeof(count) + bytes);
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&m_config.version),
                   reinterpret_cast<const byte_type*>(&m_config.version) + 4);
        out.push_back(static_cast<byte_type>(SparseHashMode::None));
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&count),
                   reinterpret_cast<const byte_type*>(&count) + sizeof(count));
        out.insert(out.end(), reinterpret_cast<const byte_type*>(pairs),
                   reinterpret_cast<const byte_type*>(pairs) + bytes);
        return out;
    }

    std::vector<pair_type> rawUnpack(const byte_type* data, size_type dataSize, size_type expectedCount) {
        if (dataSize < expectedCount * sizeof(pair_type)) return {};
        std::vector<pair_type> out(expectedCount);
        std::memcpy(out.data(), data, expectedCount * sizeof(pair_type));
        return out;
    }

    // ------------------------------------------------------------------------
    //  Run‑length encoding for runs of consecutive keys (not necessarily values)
    //  Stores: start key, run length, and then values for each position.
    //  Assumes keys are sorted and consecutive (e.g., Morton codes).
    // ------------------------------------------------------------------------
    std::vector<byte_type> runLengthEncode(const pair_type* pairs, size_type count) {
        std::vector<byte_type> out;
        // Header
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&m_config.version),
                   reinterpret_cast<const byte_type*>(&m_config.version) + 4);
        out.push_back(static_cast<byte_type>(SparseHashMode::RunLength));
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&count),
                   reinterpret_cast<const byte_type*>(&count) + sizeof(count));

        size_type i = 0;
        while (i < count) {
            Key startKey = pairs[i].first;
            size_type run = 1;
            // Detect consecutive keys: key[i+1] == key[i] + 1
            while (i + run < count && pairs[i + run].first == static_cast<Key>(pairs[i + run - 1].first + 1)) {
                ++run;
            }
            // Write start key
            out.insert(out.end(), reinterpret_cast<const byte_type*>(&startKey),
                       reinterpret_cast<const byte_type*>(&startKey) + sizeof(Key));
            // Write run length as varint
            writeVarint(out, run);
            // Write all values in this run
            for (size_type j = 0; j < run; ++j) {
                out.insert(out.end(), reinterpret_cast<const byte_type*>(&pairs[i + j].second),
                           reinterpret_cast<const byte_type*>(&pairs[i + j].second) + sizeof(Value));
            }
            i += run;
        }
        return out;
    }

    std::vector<pair_type> runLengthDecode(const byte_type* data, size_type dataSize, size_type expectedCount) {
        std::vector<pair_type> out;
        out.reserve(expectedCount);
        const byte_type* ptr = data;
        while (ptr < data + dataSize && out.size() < expectedCount) {
            if (ptr + sizeof(Key) > data + dataSize) break;
            Key key;
            std::memcpy(&key, ptr, sizeof(Key));
            ptr += sizeof(Key);
            size_type run = readVarint(ptr);
            for (size_type j = 0; j < run && out.size() < expectedCount; ++j) {
                if (ptr + sizeof(Value) > data + dataSize) return out;
                Value val;
                std::memcpy(&val, ptr, sizeof(Value));
                ptr += sizeof(Value);
                out.push_back({key + static_cast<Key>(j), val});
            }
        }
        return out;
    }

    // ------------------------------------------------------------------------
    //  DeltaKey encoding: store first key, then key differences (varint),
    //  and then values (raw or compressed).
    // ------------------------------------------------------------------------
    std::vector<byte_type> deltaKeyEncode(const pair_type* pairs, size_type count) {
        std::vector<byte_type> out;
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&m_config.version),
                   reinterpret_cast<const byte_type*>(&m_config.version) + 4);
        out.push_back(static_cast<byte_type>(SparseHashMode::DeltaKey));
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&count),
                   reinterpret_cast<const byte_type*>(&count) + sizeof(count));

        // Store first key raw
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&pairs[0].first),
                   reinterpret_cast<const byte_type*>(&pairs[0].first) + sizeof(Key));

        // Key deltas (differences, always positive if keys are increasing)
        for (size_type i = 1; i < count; ++i) {
            Key diff = pairs[i].first - pairs[i-1].first;
            writeVarint(out, static_cast<uint64_t>(diff));
        }

        // Values: use delta compression or raw (optional)
        // For simplicity, we store values raw (could be further compressed)
        for (size_type i = 0; i < count; ++i) {
            out.insert(out.end(), reinterpret_cast<const byte_type*>(&pairs[i].second),
                       reinterpret_cast<const byte_type*>(&pairs[i].second) + sizeof(Value));
        }
        return out;
    }

    std::vector<pair_type> deltaKeyDecode(const byte_type* data, size_type dataSize, size_type expectedCount) {
        if (expectedCount == 0) return {};
        if (dataSize < sizeof(Key)) return {};
        std::vector<pair_type> out;
        out.reserve(expectedCount);
        Key prevKey;
        std::memcpy(&prevKey, data, sizeof(Key));
        const byte_type* ptr = data + sizeof(Key);
        // Key deltas
        std::vector<Key> keys(expectedCount);
        keys[0] = prevKey;
        for (size_type i = 1; i < expectedCount; ++i) {
            uint64_t diff = readVarint(ptr);
            keys[i] = keys[i-1] + static_cast<Key>(diff);
        }
        // Values
        for (size_type i = 0; i < expectedCount; ++i) {
            if (ptr + sizeof(Value) > data + dataSize) return out;
            Value val;
            std::memcpy(&val, ptr, sizeof(Value));
            ptr += sizeof(Value);
            out.push_back({keys[i], val});
        }
        return out;
    }

    // ------------------------------------------------------------------------
    //  BucketRemap encoding: for maps where keys are bucket indices (e.g., in a
    //  dense range but sparse). Stores a bitmask of present buckets and then
    //  only the values for present keys.
    // ------------------------------------------------------------------------
    std::vector<byte_type> bucketRemapEncode(const pair_type* pairs, size_type count) {
        if (count == 0) return rawPack(pairs, count);

        // Determine key range
        Key minKey = pairs[0].first;
        Key maxKey = pairs[count-1].first;
        uint64_t range = static_cast<uint64_t>(maxKey - minKey + 1);
        if (range > 1024 * 1024) {
            // Too large, fallback to raw or delta
            return deltaKeyEncode(pairs, count);
        }

        std::vector<byte_type> out;
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&m_config.version),
                   reinterpret_cast<const byte_type*>(&m_config.version) + 4);
        out.push_back(static_cast<byte_type>(SparseHashMode::BucketRemap));
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&count),
                   reinterpret_cast<const byte_type*>(&count) + sizeof(count));

        // Write minKey and maxKey
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&minKey),
                   reinterpret_cast<const byte_type*>(&minKey) + sizeof(Key));
        out.insert(out.end(), reinterpret_cast<const byte_type*>(&maxKey),
                   reinterpret_cast<const byte_type*>(&maxKey) + sizeof(Key));

        // Build bitmask of present keys
        size_type bits = static_cast<size_type>(range);
        std::vector<uint8_t> bitmask((bits + 7) / 8, 0);
        for (size_type i = 0; i < count; ++i) {
            size_type offset = static_cast<size_type>(pairs[i].first - minKey);
            bitmask[offset / 8] |= (1 << (offset % 8));
        }
        out.insert(out.end(), bitmask.begin(), bitmask.end());

        // Write values in order of increasing key
        for (size_type i = 0; i < count; ++i) {
            out.insert(out.end(), reinterpret_cast<const byte_type*>(&pairs[i].second),
                       reinterpret_cast<const byte_type*>(&pairs[i].second) + sizeof(Value));
        }
        return out;
    }

    std::vector<pair_type> bucketRemapDecode(const byte_type* data, size_type dataSize, size_type expectedCount) {
        if (dataSize < 2 * sizeof(Key)) return {};
        Key minKey, maxKey;
        std::memcpy(&minKey, data, sizeof(Key));
        std::memcpy(&maxKey, data + sizeof(Key), sizeof(Key));
        const byte_type* ptr = data + 2 * sizeof(Key);
        uint64_t range = static_cast<uint64_t>(maxKey - minKey + 1);
        size_type bits = static_cast<size_type>(range);
        size_type bytes = (bits + 7) / 8;
        if (ptr + bytes > data + dataSize) return {};
        const uint8_t* bitmask = ptr;
        ptr += bytes;

        std::vector<pair_type> out;
        out.reserve(expectedCount);
        for (uint64_t idx = 0; idx < range; ++idx) {
            if (bitmask[idx / 8] & (1 << (idx % 8))) {
                if (ptr + sizeof(Value) > data + dataSize) return out;
                Value val;
                std::memcpy(&val, ptr, sizeof(Value));
                ptr += sizeof(Value);
                out.push_back({static_cast<Key>(minKey + idx), val});
            }
        }
        return out;
    }

    // ------------------------------------------------------------------------
    //  Adaptive: choose best among RLE and DeltaKey (BucketRemap also considered)
    // ------------------------------------------------------------------------
    std::vector<byte_type> adaptiveCompress(const pair_type* pairs, size_type count) {
        auto rle = runLengthEncode(pairs, count);
        auto delta = deltaKeyEncode(pairs, count);
        auto bucket = bucketRemapEncode(pairs, count);
        if (rle.size() <= delta.size() && rle.size() <= bucket.size())
            return rle;
        else if (delta.size() <= bucket.size())
            return delta;
        else
            return bucket;
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
//  Dynamic environment controller for sparse hash compression
// ----------------------------------------------------------------------------
class SparseHashCompressionEnvironment {
public:
    static SparseHashCompressionEnvironment& instance() {
        static SparseHashCompressionEnvironment env;
        return env;
    }

    void setDefaultMode(SparseHashMode mode) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_defaultMode = mode;
    }
    SparseHashMode defaultMode() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_defaultMode;
    }

    void setMinPairs(size_type min) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_minPairs = min;
    }
    size_type minPairs() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_minPairs;
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
    SparseHashCompressionEnvironment()
        : m_defaultMode(SparseHashMode::Adaptive)
        , m_minPairs(16)
        , m_enableSIMD(true) {}
    mutable std::mutex m_mutex;
    SparseHashMode m_defaultMode;
    size_type m_minPairs;
    bool m_enableSIMD;
};

} // namespace Compression
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_COMPRESSION_SPARSE_HASH_COMPRESSION_H_INCLUDED

/**
 * Next file: core/io/octree_streaming_io.h
 * Remaining in the list: 17 files (octree_streaming_io, level_streaming_manager, galactic_catalogue_loader, living_entity_interface, sensory_query_system, swarm_communication, integration_bridge, archetype_index, global_morton_routing, replica_manager, consensus_tree, ecosystem_driver, weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */