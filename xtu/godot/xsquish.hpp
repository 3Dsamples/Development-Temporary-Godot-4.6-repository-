// include/xtu/godot/xsquish.hpp
// xtensor-unified - Squish DXT Texture Compression for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XSQUISH_HPP
#define XTU_GODOT_XSQUISH_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/parallel/xparallel.hpp"

#ifdef XTU_USE_SQUISH
#include <squish.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {
namespace texture {

// #############################################################################
// DXT compression flags (matching Squish)
// #############################################################################
enum class DXTFlags : uint32_t {
    FLAG_NONE = 0,
    FLAG_DXT1 = 1 << 0,
    FLAG_DXT3 = 1 << 1,
    FLAG_DXT5 = 1 << 2,
    FLAG_COLOUR_ITERATIVE_CLUSTER_FIT = 1 << 8,
    FLAG_COLOUR_CLUSTER_FIT = 1 << 3,
    FLAG_COLOUR_RANGE_FIT = 1 << 4,
    FLAG_COLOUR_METRIC_PERCEPTUAL = 1 << 5,
    FLAG_COLOUR_METRIC_UNIFORM = 1 << 6,
    FLAG_WEIGHT_COLOUR_BY_ALPHA = 1 << 7
};

inline DXTFlags operator|(DXTFlags a, DXTFlags b) {
    return static_cast<DXTFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline bool has_flag(DXTFlags flags, DXTFlags test) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(test)) != 0;
}

// #############################################################################
// Compression quality
// #############################################################################
enum class DXTQuality : uint8_t {
    QUALITY_FAST = 0,
    QUALITY_NORMAL = 1,
    QUALITY_HIGH = 2
};

// #############################################################################
// Compression options
// #############################################################################
struct DXTCompressionOptions {
    DXTQuality quality = DXTQuality::QUALITY_NORMAL;
    bool use_perceptual_metric = true;
    bool weight_by_alpha = false;
    bool generate_mipmaps = true;
};

// #############################################################################
// SquishCompressor - DXT texture compressor
// #############################################################################
class SquishCompressor : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(SquishCompressor, RefCounted)

public:
    static StringName get_class_static() { return StringName("SquishCompressor"); }

    // #########################################################################
    // Compress RGBA to DXT1 (no alpha)
    // #########################################################################
    std::vector<uint8_t> compress_dxt1(const std::vector<uint8_t>& rgba, int width, int height,
                                        const DXTCompressionOptions& options = {}) {
#ifdef XTU_USE_SQUISH
        int flags = static_cast<int>(DXTFlags::FLAG_DXT1);
        flags |= get_quality_flags(options);
        if (options.use_perceptual_metric) {
            flags |= static_cast<int>(DXTFlags::FLAG_COLOUR_METRIC_PERCEPTUAL);
        }
        return compress_impl(rgba, width, height, flags, 8);
#else
        return {};
#endif
    }

    // #########################################################################
    // Compress RGBA to DXT3 (explicit alpha)
    // #########################################################################
    std::vector<uint8_t> compress_dxt3(const std::vector<uint8_t>& rgba, int width, int height,
                                        const DXTCompressionOptions& options = {}) {
#ifdef XTU_USE_SQUISH
        int flags = static_cast<int>(DXTFlags::FLAG_DXT3);
        flags |= get_quality_flags(options);
        if (options.use_perceptual_metric) {
            flags |= static_cast<int>(DXTFlags::FLAG_COLOUR_METRIC_PERCEPTUAL);
        }
        if (options.weight_by_alpha) {
            flags |= static_cast<int>(DXTFlags::FLAG_WEIGHT_COLOUR_BY_ALPHA);
        }
        return compress_impl(rgba, width, height, flags, 16);
#else
        return {};
#endif
    }

    // #########################################################################
    // Compress RGBA to DXT5 (interpolated alpha)
    // #########################################################################
    std::vector<uint8_t> compress_dxt5(const std::vector<uint8_t>& rgba, int width, int height,
                                        const DXTCompressionOptions& options = {}) {
#ifdef XTU_USE_SQUISH
        int flags = static_cast<int>(DXTFlags::FLAG_DXT5);
        flags |= get_quality_flags(options);
        if (options.use_perceptual_metric) {
            flags |= static_cast<int>(DXTFlags::FLAG_COLOUR_METRIC_PERCEPTUAL);
        }
        if (options.weight_by_alpha) {
            flags |= static_cast<int>(DXTFlags::FLAG_WEIGHT_COLOUR_BY_ALPHA);
        }
        return compress_impl(rgba, width, height, flags, 16);
#else
        return {};
#endif
    }

    // #########################################################################
    // Auto-select best DXT format based on alpha content
    // #########################################################################
    std::vector<uint8_t> compress_auto(const std::vector<uint8_t>& rgba, int width, int height,
                                        const DXTCompressionOptions& options = {}) {
        bool has_alpha = false;
        bool has_partial_alpha = false;

        // Analyze alpha channel
        for (size_t i = 3; i < rgba.size(); i += 4) {
            uint8_t a = rgba[i];
            if (a < 255) {
                has_alpha = true;
                if (a > 0 && a < 255) {
                    has_partial_alpha = true;
                    break;
                }
            }
        }

        if (!has_alpha) {
            return compress_dxt1(rgba, width, height, options);
        } else if (has_partial_alpha) {
            return compress_dxt5(rgba, width, height, options);
        } else {
            return compress_dxt3(rgba, width, height, options);
        }
    }

    // #########################################################################
    // Decompress DXT back to RGBA
    // #########################################################################
    std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed, int width, int height,
                                     DXTFlags format) {
#ifdef XTU_USE_SQUISH
        int flags = static_cast<int>(format);
        size_t block_size = (flags & static_cast<int>(DXTFlags::FLAG_DXT1)) ? 8 : 16;
        size_t expected_size = ((width + 3) / 4) * ((height + 3) / 4) * block_size;
        
        if (compressed.size() < expected_size) {
            return {};
        }

        std::vector<uint8_t> rgba(width * height * 4);
        squish::DecompressImage(rgba.data(), width, height, compressed.data(), flags);
        return rgba;
#else
        return {};
#endif
    }

    // #########################################################################
    // Get compressed size for given dimensions
    // #########################################################################
    static size_t get_compressed_size(int width, int height, DXTFlags format) {
        int blocks_wide = (width + 3) / 4;
        int blocks_high = (height + 3) / 4;
        int block_size = has_flag(format, DXTFlags::FLAG_DXT1) ? 8 : 16;
        return blocks_wide * blocks_high * block_size;
    }

private:
#ifdef XTU_USE_SQUISH
    int get_quality_flags(const DXTCompressionOptions& options) {
        switch (options.quality) {
            case DXTQuality::QUALITY_FAST:
                return static_cast<int>(DXTFlags::FLAG_COLOUR_RANGE_FIT);
            case DXTQuality::QUALITY_HIGH:
                return static_cast<int>(DXTFlags::FLAG_COLOUR_ITERATIVE_CLUSTER_FIT);
            default:
                return static_cast<int>(DXTFlags::FLAG_COLOUR_CLUSTER_FIT);
        }
    }

    std::vector<uint8_t> compress_impl(const std::vector<uint8_t>& rgba, int width, int height,
                                        int flags, int block_size) {
        size_t expected_rgba_size = width * height * 4;
        if (rgba.size() < expected_rgba_size) {
            return {};
        }

        size_t compressed_size = ((width + 3) / 4) * ((height + 3) / 4) * block_size;
        std::vector<uint8_t> compressed(compressed_size);

        squish::CompressImage(rgba.data(), width, height, compressed.data(), flags);

        return compressed;
    }
#endif
};

// #############################################################################
// Parallel block compressor for large textures
// #############################################################################
class ParallelSquishCompressor : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(ParallelSquishCompressor, RefCounted)

public:
    static StringName get_class_static() { return StringName("ParallelSquishCompressor"); }

    std::vector<uint8_t> compress_parallel(const std::vector<uint8_t>& rgba, int width, int height,
                                            DXTFlags format, const DXTCompressionOptions& options = {}) {
        int blocks_wide = (width + 3) / 4;
        int blocks_high = (height + 3) / 4;
        int block_size = has_flag(format, DXTFlags::FLAG_DXT1) ? 8 : 16;
        size_t compressed_size = blocks_wide * blocks_high * block_size;
        std::vector<uint8_t> compressed(compressed_size);

#ifdef XTU_USE_SQUISH
        int flags = static_cast<int>(format);
        flags |= get_quality_flags(options);
        if (options.use_perceptual_metric) {
            flags |= static_cast<int>(DXTFlags::FLAG_COLOUR_METRIC_PERCEPTUAL);
        }
        if (options.weight_by_alpha) {
            flags |= static_cast<int>(DXTFlags::FLAG_WEIGHT_COLOUR_BY_ALPHA);
        }

        // Process blocks in parallel rows
        parallel::parallel_for(0, blocks_high, [&](int by) {
            for (int bx = 0; bx < blocks_wide; ++bx) {
                uint8_t source_block[64]; // 4x4x4 bytes
                int src_x = bx * 4;
                int src_y = by * 4;

                // Extract 4x4 block
                for (int y = 0; y < 4; ++y) {
                    int py = std::min(src_y + y, height - 1);
                    for (int x = 0; x < 4; ++x) {
                        int px = std::min(src_x + x, width - 1);
                        size_t src_idx = (py * width + px) * 4;
                        size_t dst_idx = (y * 4 + x) * 4;
                        for (int c = 0; c < 4; ++c) {
                            source_block[dst_idx + c] = rgba[src_idx + c];
                        }
                    }
                }

                // Compress block
                uint8_t dest_block[16];
                squish::Compress(source_block, dest_block, flags);

                // Store compressed block
                size_t dest_offset = (by * blocks_wide + bx) * block_size;
                for (int i = 0; i < block_size; ++i) {
                    compressed[dest_offset + i] = dest_block[i];
                }
            }
        });
#endif

        return compressed;
    }

private:
#ifdef XTU_USE_SQUISH
    int get_quality_flags(const DXTCompressionOptions& options) {
        switch (options.quality) {
            case DXTQuality::QUALITY_FAST:
                return static_cast<int>(DXTFlags::FLAG_COLOUR_RANGE_FIT);
            case DXTQuality::QUALITY_HIGH:
                return static_cast<int>(DXTFlags::FLAG_COLOUR_ITERATIVE_CLUSTER_FIT);
            default:
                return static_cast<int>(DXTFlags::FLAG_COLOUR_CLUSTER_FIT);
        }
    }
#endif
};

// #############################################################################
// DXTCompressionManager - Global DXT compression manager
// #############################################################################
class DXTCompressionManager : public Object {
    XTU_GODOT_REGISTER_CLASS(DXTCompressionManager, Object)

private:
    static DXTCompressionManager* s_singleton;
    Ref<SquishCompressor> m_compressor;
    Ref<ParallelSquishCompressor> m_parallel_compressor;
    bool m_use_parallel = true;
    int m_parallel_threshold = 256; // Use parallel for textures larger than 256x256

public:
    static DXTCompressionManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("DXTCompressionManager"); }

    DXTCompressionManager() {
        s_singleton = this;
        m_compressor.instance();
        m_parallel_compressor.instance();
    }

    ~DXTCompressionManager() { s_singleton = nullptr; }

    void set_use_parallel(bool use) { m_use_parallel = use; }
    bool get_use_parallel() const { return m_use_parallel; }

    void set_parallel_threshold(int threshold) { m_parallel_threshold = threshold; }
    int get_parallel_threshold() const { return m_parallel_threshold; }

    std::vector<uint8_t> compress(const std::vector<uint8_t>& rgba, int width, int height,
                                   DXTFlags format, const DXTCompressionOptions& options = {}) {
        bool use_parallel = m_use_parallel && (width * height >= m_parallel_threshold * m_parallel_threshold);
        
        if (use_parallel) {
            return m_parallel_compressor->compress_parallel(rgba, width, height, format, options);
        } else {
            switch (format) {
                case DXTFlags::FLAG_DXT1:
                    return m_compressor->compress_dxt1(rgba, width, height, options);
                case DXTFlags::FLAG_DXT3:
                    return m_compressor->compress_dxt3(rgba, width, height, options);
                case DXTFlags::FLAG_DXT5:
                    return m_compressor->compress_dxt5(rgba, width, height, options);
                default:
                    return m_compressor->compress_auto(rgba, width, height, options);
            }
        }
    }

    std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed, int width, int height,
                                     DXTFlags format) {
        return m_compressor->decompress(compressed, width, height, format);
    }
};

} // namespace texture

// Bring into main namespace
using texture::SquishCompressor;
using texture::ParallelSquishCompressor;
using texture::DXTCompressionManager;
using texture::DXTFlags;
using texture::DXTQuality;
using texture::DXTCompressionOptions;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XSQUISH_HPP