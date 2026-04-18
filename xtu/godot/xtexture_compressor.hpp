// include/xtu/godot/xtexture_compressor.hpp
// xtensor-unified - Texture compression and import for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XTEXTURE_COMPRESSOR_HPP
#define XTU_GODOT_XTEXTURE_COMPRESSOR_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class TextureCompressor;
class BasisUCompressor;
class ETCCompressor;
class ASTCCompressor;

// #############################################################################
// Texture compression format
// #############################################################################
enum class TextureCompressionFormat : uint8_t {
    FORMAT_UNCOMPRESSED = 0,
    FORMAT_BASIS_UNIVERSAL = 1,
    FORMAT_ETC2 = 2,
    FORMAT_ETC = 3,
    FORMAT_ASTC = 4,
    FORMAT_S3TC = 5,
    FORMAT_PVRTC = 6,
    FORMAT_BPTC = 7
};

// #############################################################################
// BasisU compression quality
// #############################################################################
enum class BasisUQuality : uint8_t {
    QUALITY_ETC1S = 0,
    QUALITY_UASTC = 1,
    QUALITY_HIGH = 2
};

// #############################################################################
// Compression options
// #############################################################################
struct CompressionOptions {
    TextureCompressionFormat format = TextureCompressionFormat::FORMAT_BASIS_UNIVERSAL;
    BasisUQuality quality = BasisUQuality::QUALITY_UASTC;
    bool mipmaps = true;
    int mipmap_limit = -1;
    bool srgb = true;
    bool normal_map = false;
    float quality_level = 0.8f;
    int thread_count = 0;
};

// #############################################################################
// TextureCompressor - Base class for texture compressors
// #############################################################################
class TextureCompressor : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(TextureCompressor, RefCounted)

public:
    static StringName get_class_static() { return StringName("TextureCompressor"); }

    virtual std::vector<uint8_t> compress(const std::vector<uint8_t>& image_data,
                                           int width, int height,
                                           RenderingServer::TextureFormat src_format,
                                           const CompressionOptions& options) = 0;
    virtual bool is_format_supported(TextureCompressionFormat format) const = 0;
    virtual std::vector<TextureCompressionFormat> get_supported_formats() const = 0;
};

// #############################################################################
// BasisUCompressor - Basis Universal texture compressor
// #############################################################################
class BasisUCompressor : public TextureCompressor {
    XTU_GODOT_REGISTER_CLASS(BasisUCompressor, TextureCompressor)

private:
    std::atomic<bool> m_compressing{false};
    std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("BasisUCompressor"); }

    std::vector<uint8_t> compress(const std::vector<uint8_t>& image_data,
                                   int width, int height,
                                   RenderingServer::TextureFormat src_format,
                                   const CompressionOptions& options) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_compressing = true;

        std::vector<uint8_t> result;
        
#ifdef XTU_USE_BASISU
        // Convert source data to RGBA32
        std::vector<uint8_t> rgba_data = convert_to_rgba32(image_data, width, height, src_format);
        
        // Setup BasisU parameters
        basisu::basis_compressor_params params;
        params.m_source_images.push_back(basisu::image(
            width, height,
            rgba_data.data(),
            rgba_data.data() + width * height * 4
        ));
        
        params.m_perceptual = options.srgb;
        params.m_mip_gen = options.mipmaps;
        params.m_mip_srgb = options.srgb;
        params.m_write_output_basis = true;
        params.m_pack_uastc_flags = 0;
        
        if (options.quality == BasisUQuality::QUALITY_UASTC) {
            params.m_uastc = true;
            params.m_quality_level = basisu::BASISU_QUALITY_UASTC_HIGH;
        } else {
            params.m_uastc = false;
            float q = std::clamp(options.quality_level, 0.0f, 1.0f);
            params.m_quality_level = static_cast<int>(q * 255);
        }
        
        if (options.normal_map) {
            params.m_swizzle[0] = 0; // R
            params.m_swizzle[1] = 1; // G
            params.m_swizzle[2] = 2; // B
            params.m_swizzle[3] = 3; // A
        }
        
        params.m_multithreading = true;
        params.m_thread_count = options.thread_count > 0 ? options.thread_count : 
                                std::thread::hardware_concurrency();
        
        // Compress
        basisu::basis_compressor compressor;
        if (compressor.init(params)) {
            basisu::basis_compressor::error_code ec = compressor.process();
            if (ec == basisu::basis_compressor::cEC_Success) {
                const auto& output = compressor.get_output_basis_file();
                result.assign(output.data(), output.data() + output.size());
            }
        }
#endif
        
        m_compressing = false;
        return result;
    }

    bool is_format_supported(TextureCompressionFormat format) const override {
        return format == TextureCompressionFormat::FORMAT_BASIS_UNIVERSAL;
    }

    std::vector<TextureCompressionFormat> get_supported_formats() const override {
        return {TextureCompressionFormat::FORMAT_BASIS_UNIVERSAL};
    }

    bool is_compressing() const { return m_compressing; }

private:
    std::vector<uint8_t> convert_to_rgba32(const std::vector<uint8_t>& src,
                                            int width, int height,
                                            RenderingServer::TextureFormat format) {
        std::vector<uint8_t> result(width * height * 4);
        // Conversion logic for various formats
        return result;
    }
};

// #############################################################################
// ETCCompressor - ETC/ETC2 texture compressor
// #############################################################################
class ETCCompressor : public TextureCompressor {
    XTU_GODOT_REGISTER_CLASS(ETCCompressor, TextureCompressor)

private:
    bool m_high_quality = true;

public:
    static StringName get_class_static() { return StringName("ETCCompressor"); }

    void set_high_quality(bool enable) { m_high_quality = enable; }
    bool get_high_quality() const { return m_high_quality; }

    std::vector<uint8_t> compress(const std::vector<uint8_t>& image_data,
                                   int width, int height,
                                   RenderingServer::TextureFormat src_format,
                                   const CompressionOptions& options) override {
        std::vector<uint8_t> result;
        
#ifdef XTU_USE_ETCPAK
        std::vector<uint8_t> rgba_data = convert_to_rgba32(image_data, width, height, src_format);
        
        int block_count = ((width + 3) / 4) * ((height + 3) / 4);
        result.resize(block_count * 8); // 8 bytes per block for ETC2
        
        etcpak_compress_etc2_rgba(rgba_data.data(), result.data(), width, height,
                                   m_high_quality, options.mipmaps);
#endif
        
        return result;
    }

    bool is_format_supported(TextureCompressionFormat format) const override {
        return format == TextureCompressionFormat::FORMAT_ETC2 ||
               format == TextureCompressionFormat::FORMAT_ETC;
    }

    std::vector<TextureCompressionFormat> get_supported_formats() const override {
        return {TextureCompressionFormat::FORMAT_ETC2, TextureCompressionFormat::FORMAT_ETC};
    }

private:
    std::vector<uint8_t> convert_to_rgba32(const std::vector<uint8_t>& src,
                                            int width, int height,
                                            RenderingServer::TextureFormat format) {
        std::vector<uint8_t> result(width * height * 4);
        return result;
    }
};

// #############################################################################
// ASTCCompressor - ASTC texture compressor
// #############################################################################
class ASTCCompressor : public TextureCompressor {
    XTU_GODOT_REGISTER_CLASS(ASTCCompressor, TextureCompressor)

private:
    float m_quality = 0.8f;
    int m_block_size_x = 4;
    int m_block_size_y = 4;

public:
    static StringName get_class_static() { return StringName("ASTCCompressor"); }

    void set_quality(float quality) { m_quality = std::clamp(quality, 0.0f, 1.0f); }
    float get_quality() const { return m_quality; }

    void set_block_size(int x, int y) {
        m_block_size_x = std::clamp(x, 4, 12);
        m_block_size_y = std::clamp(y, 4, 12);
    }

    std::vector<uint8_t> compress(const std::vector<uint8_t>& image_data,
                                   int width, int height,
                                   RenderingServer::TextureFormat src_format,
                                   const CompressionOptions& options) override {
        std::vector<uint8_t> result;
        
#ifdef XTU_USE_ASTCENC
        std::vector<uint8_t> rgba_data = convert_to_rgba32(image_data, width, height, src_format);
        
        astcenc_config config;
        astcenc_preset preset = m_quality > 0.7f ? ASTCENC_PRE_THOROUGH :
                                m_quality > 0.4f ? ASTCENC_PRE_MEDIUM : ASTCENC_PRE_FAST;
        
        astcenc_config_init(preset, m_block_size_x, m_block_size_y, 1, &config);
        config.tune_partition_count_limit = static_cast<unsigned int>(m_quality * 4);
        
        astcenc_image image;
        image.dim_x = width;
        image.dim_y = height;
        image.dim_z = 1;
        image.data_type = ASTCENC_TYPE_U8;
        image.data = rgba_data.data();
        
        astcenc_context* ctx;
        astcenc_context_alloc(&config, options.thread_count, &ctx);
        
        size_t compressed_size = astcenc_context_calculate_length(ctx, width, height);
        result.resize(compressed_size);
        
        astcenc_compress_image(ctx, &image, ASTCENC_PRGBA, result.data(), compressed_size, 0);
        astcenc_context_free(ctx);
#endif
        
        return result;
    }

    bool is_format_supported(TextureCompressionFormat format) const override {
        return format == TextureCompressionFormat::FORMAT_ASTC;
    }

    std::vector<TextureCompressionFormat> get_supported_formats() const override {
        return {TextureCompressionFormat::FORMAT_ASTC};
    }

private:
    std::vector<uint8_t> convert_to_rgba32(const std::vector<uint8_t>& src,
                                            int width, int height,
                                            RenderingServer::TextureFormat format) {
        std::vector<uint8_t> result(width * height * 4);
        return result;
    }
};

// #############################################################################
// TextureCompressionManager - Global texture compression manager
// #############################################################################
class TextureCompressionManager : public Object {
    XTU_GODOT_REGISTER_CLASS(TextureCompressionManager, Object)

private:
    static TextureCompressionManager* s_singleton;
    std::map<TextureCompressionFormat, Ref<TextureCompressor>> m_compressors;
    std::mutex m_mutex;
    std::vector<std::function<void(const std::vector<uint8_t>&)>> m_pending_callbacks;
    std::thread m_worker_thread;
    std::atomic<bool> m_worker_running{false};
    std::queue<std::function<void()>> m_task_queue;

public:
    static TextureCompressionManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("TextureCompressionManager"); }

    TextureCompressionManager() {
        s_singleton = this;
        register_default_compressors();
        start_worker();
    }

    ~TextureCompressionManager() {
        stop_worker();
        s_singleton = nullptr;
    }

    void register_compressor(TextureCompressionFormat format, const Ref<TextureCompressor>& compressor) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_compressors[format] = compressor;
    }

    Ref<TextureCompressor> get_compressor(TextureCompressionFormat format) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_compressors.find(format);
        return it != m_compressors.end() ? it->second : Ref<TextureCompressor>();
    }

    std::vector<uint8_t> compress_sync(const std::vector<uint8_t>& data,
                                        int width, int height,
                                        RenderingServer::TextureFormat src_format,
                                        const CompressionOptions& options) {
        Ref<TextureCompressor> compressor = get_compressor(options.format);
        if (!compressor.is_valid()) return {};
        
        return compressor->compress(data, width, height, src_format, options);
    }

    void compress_async(const std::vector<uint8_t>& data,
                        int width, int height,
                        RenderingServer::TextureFormat src_format,
                        const CompressionOptions& options,
                        std::function<void(const std::vector<uint8_t>&)> callback) {
        enqueue_task([this, data, width, height, src_format, options, callback]() {
            auto result = compress_sync(data, width, height, src_format, options);
            if (callback) {
                call_deferred("_async_callback", result, callback);
            }
        });
    }

    void _async_callback(const std::vector<uint8_t>& result,
                         std::function<void(const std::vector<uint8_t>&)> callback) {
        if (callback) callback(result);
    }

    std::vector<TextureCompressionFormat> get_supported_formats() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<TextureCompressionFormat> formats;
        for (const auto& kv : m_compressors) {
            formats.push_back(kv.first);
        }
        return formats;
    }

private:
    void register_default_compressors() {
        register_compressor(TextureCompressionFormat::FORMAT_BASIS_UNIVERSAL,
                           Ref<TextureCompressor>(new BasisUCompressor()));
        register_compressor(TextureCompressionFormat::FORMAT_ETC2,
                           Ref<TextureCompressor>(new ETCCompressor()));
        register_compressor(TextureCompressionFormat::FORMAT_ETC,
                           Ref<TextureCompressor>(new ETCCompressor()));
        register_compressor(TextureCompressionFormat::FORMAT_ASTC,
                           Ref<TextureCompressor>(new ASTCCompressor()));
    }

    void start_worker() {
        m_worker_running = true;
        m_worker_thread = std::thread([this]() {
            while (m_worker_running) {
                std::function<void()> task;
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    if (!m_task_queue.empty()) {
                        task = std::move(m_task_queue.front());
                        m_task_queue.pop();
                    }
                }
                if (task) {
                    task();
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        });
    }

    void stop_worker() {
        m_worker_running = false;
        if (m_worker_thread.joinable()) m_worker_thread.join();
    }

    void enqueue_task(std::function<void()> task) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_task_queue.push(task);
    }
};

} // namespace godot

// Bring into main namespace
using godot::TextureCompressor;
using godot::BasisUCompressor;
using godot::ETCCompressor;
using godot::ASTCCompressor;
using godot::TextureCompressionManager;
using godot::TextureCompressionFormat;
using godot::BasisUQuality;
using godot::CompressionOptions;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XTEXTURE_COMPRESSOR_HPP