// include/xtu/godot/xcvtt.hpp
// xtensor-unified - CVTT BC6H/BC7 Texture Compression for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XCVTT_HPP
#define XTU_GODOT_XCVTT_HPP

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

#ifdef XTU_USE_CVTT
#include <ConvectionKernels/ConvectionKernels.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {
namespace texture {

// #############################################################################
// BC compression format
// #############################################################################
enum class BCFormat : uint8_t {
    FORMAT_BC6H = 0,
    FORMAT_BC7 = 1
};

// #############################################################################
// BC compression quality
// #############################################################################
enum class BCQuality : uint8_t {
    QUALITY_VERY_FAST = 0,
    QUALITY_FAST = 1,
    QUALITY_BASIC = 2,
    QUALITY_SLOW = 3,
    QUALITY_VERY_SLOW = 4
};

// #############################################################################
// BC error metric
// #############################################################################
enum class BCErrorMetric : uint8_t {
    ERROR_METRIC_PERCEPTUAL = 0,
    ERROR_METRIC_UNIFORM = 1,
    ERROR_METRIC_RGB_AVG = 2
};

// #############################################################################
// BC compression options
// #############################################################################
struct BCCompressionOptions {
    BCQuality quality = BCQuality::QUALITY_BASIC;
    BCErrorMetric error_metric = BCErrorMetric::ERROR_METRIC_PERCEPTUAL;
    bool generate_mipmaps = true;
    bool is_signed = false;  // For BC6H signed format
    float alpha_weight = 1.0f;
    int thread_count = 0;
};

// #############################################################################
// CVTTCompressor - BC6H/BC7 texture compressor
// #############################################################################
class CVTTCompressor : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(CVTTCompressor, RefCounted)

public:
    static StringName get_class_static() { return StringName("CVTTCompressor"); }

    // #########################################################################
    // Compress RGBA to BC7 (high quality LDR)
    // #########################################################################
    std::vector<uint8_t> compress_bc7(const std::vector<uint8_t>& rgba, int width, int height,
                                       const BCCompressionOptions& options = {}) {
#ifdef XTU_USE_CVTT
        return compress_impl(rgba, width, height, BCFormat::FORMAT_BC7, options);
#else
        return {};
#endif
    }

    // #########################################################################
    // Compress RGB HDR to BC6H (high dynamic range)
    // #########################################################################
    std::vector<uint8_t> compress_bc6h(const std::vector<float>& rgb_hdr, int width, int height,
                                        const BCCompressionOptions& options = {}) {
#ifdef XTU_USE_CVTT
        return compress_impl_hdr(rgb_hdr, width, height, options);
#else
        return {};
#endif
    }

    // #########################################################################
    // Decompress BC7 back to RGBA
    // #########################################################################
    std::vector<uint8_t> decompress_bc7(const std::vector<uint8_t>& compressed, int width, int height) {
#ifdef XTU_USE_CVTT
        std::vector<uint8_t> rgba(width * height * 4);
        cvtt::DecompressBC7(compressed.data(), width, height, rgba.data());
        return rgba;
#else
        return {};
#endif
    }

    // #########################################################################
    // Decompress BC6H back to RGB HDR
    // #########################################################################
    std::vector<float> decompress_bc6h(const std::vector<uint8_t>& compressed, int width, int height) {
#ifdef XTU_USE_CVTT
        std::vector<float> rgb_hdr(width * height * 3);
        cvtt::DecompressBC6H(compressed.data(), width, height, rgb_hdr.data(), false);
        return rgb_hdr;
#else
        return {};
#endif
    }

    // #########################################################################
    // Get compressed size
    // #########################################################################
    static size_t get_compressed_size(int width, int height, BCFormat format) {
        int blocks_wide = (width + 3) / 4;
        int blocks_high = (height + 3) / 4;
        return blocks_wide * blocks_high * 16; // Both BC6H and BC7 use 16 bytes per block
    }

private:
#ifdef XTU_USE_CVTT
    cvtt::Kernels::Quality quality_to_cvtt(BCQuality quality) {
        switch (quality) {
            case BCQuality::QUALITY_VERY_FAST: return cvtt::Kernels::Quality_VeryFast;
            case BCQuality::QUALITY_FAST: return cvtt::Kernels::Quality_Fast;
            case BCQuality::QUALITY_SLOW: return cvtt::Kernels::Quality_Slow;
            case BCQuality::QUALITY_VERY_SLOW: return cvtt::Kernels::Quality_VerySlow;
            default: return cvtt::Kernels::Quality_Basic;
        }
    }

    cvtt::Kernels::ErrorMetric error_metric_to_cvtt(BCErrorMetric metric) {
        switch (metric) {
            case BCErrorMetric::ERROR_METRIC_UNIFORM: return cvtt::Kernels::ErrorMetric_Uniform;
            case BCErrorMetric::ERROR_METRIC_RGB_AVG: return cvtt::Kernels::ErrorMetric_RGBAvg;
            default: return cvtt::Kernels::ErrorMetric_Perceptual;
        }
    }

    std::vector<uint8_t> compress_impl(const std::vector<uint8_t>& rgba, int width, int height,
                                        BCFormat format, const BCCompressionOptions& options) {
        size_t expected_size = width * height * 4;
        if (rgba.size() < expected_size) return {};

        size_t compressed_size = get_compressed_size(width, height, format);
        std::vector<uint8_t> compressed(compressed_size);

        cvtt::Options cvtt_opts;
        cvtt_opts.flags = 0;
        cvtt_opts.threadCount = options.thread_count > 0 ? options.thread_count : 
                                static_cast<int>(std::thread::hardware_concurrency());
        cvtt_opts.quality = quality_to_cvtt(options.quality);
        cvtt_opts.errorMetric = error_metric_to_cvtt(options.error_metric);
        cvtt_opts.alphaWeight = options.alpha_weight;

        if (format == BCFormat::FORMAT_BC7) {
            cvtt::Kernels::CompressBC7(rgba.data(), width, height, compressed.data(), cvtt_opts);
        }

        return compressed;
    }

    std::vector<uint8_t> compress_impl_hdr(const std::vector<float>& rgb_hdr, int width, int height,
                                            const BCCompressionOptions& options) {
        size_t expected_size = width * height * 3;
        if (rgb_hdr.size() < expected_size) return {};

        size_t compressed_size = get_compressed_size(width, height, BCFormat::FORMAT_BC6H);
        std::vector<uint8_t> compressed(compressed_size);

        cvtt::Options cvtt_opts;
        cvtt_opts.flags = options.is_signed ? cvtt::Flags_BC6H_Signed : 0;
        cvtt_opts.threadCount = options.thread_count > 0 ? options.thread_count : 
                                static_cast<int>(std::thread::hardware_concurrency());
        cvtt_opts.quality = quality_to_cvtt(options.quality);

        cvtt::Kernels::CompressBC6H(rgb_hdr.data(), width, height, compressed.data(), cvtt_opts);

        return compressed;
    }
#endif
};

// #############################################################################
// Parallel BC compressor for large textures
// #############################################################################
class ParallelCVTTCompressor : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(ParallelCVTTCompressor, RefCounted)

public:
    static StringName get_class_static() { return StringName("ParallelCVTTCompressor"); }

    std::vector<uint8_t> compress_bc7_parallel(const std::vector<uint8_t>& rgba, int width, int height,
                                                const BCCompressionOptions& options = {}) {
#ifdef XTU_USE_CVTT
        int blocks_wide = (width + 3) / 4;
        int blocks_high = (height + 3) / 4;
        size_t compressed_size = blocks_wide * blocks_high * 16;
        std::vector<uint8_t> compressed(compressed_size);

        cvtt::Options cvtt_opts;
        cvtt_opts.flags = 0;
        cvtt_opts.threadCount = options.thread_count > 0 ? options.thread_count : 
                                static_cast<int>(std::thread::hardware_concurrency());
        cvtt_opts.quality = quality_to_cvtt(options.quality);
        cvtt_opts.errorMetric = error_metric_to_cvtt(options.error_metric);
        cvtt_opts.alphaWeight = options.alpha_weight;

        // Process blocks in parallel rows using CVTT's internal threading
        cvtt::Kernels::CompressBC7(rgba.data(), width, height, compressed.data(), cvtt_opts);

        return compressed;
#else
        return {};
#endif
    }

    std::vector<uint8_t> compress_bc6h_parallel(const std::vector<float>& rgb_hdr, int width, int height,
                                                 const BCCompressionOptions& options = {}) {
#ifdef XTU_USE_CVTT
        int blocks_wide = (width + 3) / 4;
        int blocks_high = (height + 3) / 4;
        size_t compressed_size = blocks_wide * blocks_high * 16;
        std::vector<uint8_t> compressed(compressed_size);

        cvtt::Options cvtt_opts;
        cvtt_opts.flags = options.is_signed ? cvtt::Flags_BC6H_Signed : 0;
        cvtt_opts.threadCount = options.thread_count > 0 ? options.thread_count : 
                                static_cast<int>(std::thread::hardware_concurrency());
        cvtt_opts.quality = quality_to_cvtt(options.quality);

        cvtt::Kernels::CompressBC6H(rgb_hdr.data(), width, height, compressed.data(), cvtt_opts);

        return compressed;
#else
        return {};
#endif
    }

private:
#ifdef XTU_USE_CVTT
    cvtt::Kernels::Quality quality_to_cvtt(BCQuality quality) {
        switch (quality) {
            case BCQuality::QUALITY_VERY_FAST: return cvtt::Kernels::Quality_VeryFast;
            case BCQuality::QUALITY_FAST: return cvtt::Kernels::Quality_Fast;
            case BCQuality::QUALITY_SLOW: return cvtt::Kernels::Quality_Slow;
            case BCQuality::QUALITY_VERY_SLOW: return cvtt::Kernels::Quality_VerySlow;
            default: return cvtt::Kernels::Quality_Basic;
        }
    }

    cvtt::Kernels::ErrorMetric error_metric_to_cvtt(BCErrorMetric metric) {
        switch (metric) {
            case BCErrorMetric::ERROR_METRIC_UNIFORM: return cvtt::Kernels::ErrorMetric_Uniform;
            case BCErrorMetric::ERROR_METRIC_RGB_AVG: return cvtt::Kernels::ErrorMetric_RGBAvg;
            default: return cvtt::Kernels::ErrorMetric_Perceptual;
        }
    }
#endif
};

// #############################################################################
// BCCompressionManager - Global BC compression manager
// #############################################################################
class BCCompressionManager : public Object {
    XTU_GODOT_REGISTER_CLASS(BCCompressionManager, Object)

private:
    static BCCompressionManager* s_singleton;
    Ref<CVTTCompressor> m_compressor;
    Ref<ParallelCVTTCompressor> m_parallel_compressor;
    bool m_use_parallel = true;

public:
    static BCCompressionManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("BCCompressionManager"); }

    BCCompressionManager() {
        s_singleton = this;
        m_compressor.instance();
        m_parallel_compressor.instance();
    }

    ~BCCompressionManager() { s_singleton = nullptr; }

    void set_use_parallel(bool use) { m_use_parallel = use; }
    bool get_use_parallel() const { return m_use_parallel; }

    std::vector<uint8_t> compress_bc7(const std::vector<uint8_t>& rgba, int width, int height,
                                       const BCCompressionOptions& options = {}) {
        if (m_use_parallel) {
            return m_parallel_compressor->compress_bc7_parallel(rgba, width, height, options);
        } else {
            return m_compressor->compress_bc7(rgba, width, height, options);
        }
    }

    std::vector<uint8_t> compress_bc6h(const std::vector<float>& rgb_hdr, int width, int height,
                                        const BCCompressionOptions& options = {}) {
        if (m_use_parallel) {
            return m_parallel_compressor->compress_bc6h_parallel(rgb_hdr, width, height, options);
        } else {
            return m_compressor->compress_bc6h(rgb_hdr, width, height, options);
        }
    }

    std::vector<uint8_t> decompress_bc7(const std::vector<uint8_t>& compressed, int width, int height) {
        return m_compressor->decompress_bc7(compressed, width, height);
    }

    std::vector<float> decompress_bc6h(const std::vector<uint8_t>& compressed, int width, int height) {
        return m_compressor->decompress_bc6h(compressed, width, height);
    }
};

} // namespace texture

// Bring into main namespace
using texture::CVTTCompressor;
using texture::ParallelCVTTCompressor;
using texture::BCCompressionManager;
using texture::BCFormat;
using texture::BCQuality;
using texture::BCErrorMetric;
using texture::BCCompressionOptions;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XCVTT_HPP