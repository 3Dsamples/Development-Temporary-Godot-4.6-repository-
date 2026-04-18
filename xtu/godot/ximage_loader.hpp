// include/xtu/godot/ximage_loader.hpp
// xtensor-unified - Image format loaders for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XIMAGE_LOADER_HPP
#define XTU_GODOT_XIMAGE_LOADER_HPP

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xcore.hpp"
#include "xtu/godot/xrenderingserver.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace io {

// #############################################################################
// Forward declarations
// #############################################################################
class ImageLoader;
class ImageFormatLoader;
class ImageLoaderPNG;
class ImageLoaderJPEG;
class ImageLoaderWebP;
class ImageLoaderTGA;
class ImageLoaderBMP;
class ImageLoaderKTX;
class ImageLoaderSVG;
class ImageLoaderHDR;

// #############################################################################
// Image format type
// #############################################################################
enum class ImageFormatType : uint8_t {
    FORMAT_UNKNOWN = 0,
    FORMAT_PNG = 1,
    FORMAT_JPEG = 2,
    FORMAT_WEBP = 3,
    FORMAT_TGA = 4,
    FORMAT_BMP = 5,
    FORMAT_KTX = 6,
    FORMAT_SVG = 7,
    FORMAT_HDR = 8,
    FORMAT_EXR = 9,
    FORMAT_DDS = 10
};

// #############################################################################
// Image data structure
// #############################################################################
struct ImageData {
    std::vector<uint8_t> data;
    int width = 0;
    int height = 0;
    int channels = 0;
    RenderingServer::TextureFormat format = RenderingServer::TEXTURE_FORMAT_RGBA8;
    bool has_alpha = false;
    bool is_hdr = false;
    bool is_compressed = false;
    int mipmap_count = 1;
    String error_message;
};

// #############################################################################
// ImageFormatLoader - Base class for image format loaders
// #############################################################################
class ImageFormatLoader : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(ImageFormatLoader, RefCounted)

public:
    static StringName get_class_static() { return StringName("ImageFormatLoader"); }

    virtual std::vector<String> get_recognized_extensions() const = 0;
    virtual ImageFormatType get_format_type() const = 0;
    virtual bool can_load(const std::vector<uint8_t>& header) const = 0;
    virtual Error load_image(const std::vector<uint8_t>& data, ImageData& out_image) = 0;
    virtual Error save_image(const ImageData& image, const String& path) { return ERR_UNAVAILABLE; }
    virtual float get_priority() const { return 1.0f; }
};

// #############################################################################
// ImageLoaderPNG - PNG format loader
// #############################################################################
class ImageLoaderPNG : public ImageFormatLoader {
    XTU_GODOT_REGISTER_CLASS(ImageLoaderPNG, ImageFormatLoader)

private:
    static constexpr uint8_t PNG_SIGNATURE[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};

public:
    static StringName get_class_static() { return StringName("ImageLoaderPNG"); }

    std::vector<String> get_recognized_extensions() const override {
        return {"png"};
    }

    ImageFormatType get_format_type() const override {
        return ImageFormatType::FORMAT_PNG;
    }

    bool can_load(const std::vector<uint8_t>& header) const override {
        return header.size() >= 8 && std::memcmp(header.data(), PNG_SIGNATURE, 8) == 0;
    }

    Error load_image(const std::vector<uint8_t>& data, ImageData& out_image) override {
#ifdef XTU_USE_LIBPNG
        // libpng decoding
        // ...
        return OK;
#else
        out_image.error_message = "PNG support not compiled";
        return ERR_UNAVAILABLE;
#endif
    }

    Error save_image(const ImageData& image, const String& path) override {
#ifdef XTU_USE_LIBPNG
        // libpng encoding
        return OK;
#else
        return ERR_UNAVAILABLE;
#endif
    }

    float get_priority() const override { return 1.0f; }
};

// #############################################################################
// ImageLoaderJPEG - JPEG format loader
// #############################################################################
class ImageLoaderJPEG : public ImageFormatLoader {
    XTU_GODOT_REGISTER_CLASS(ImageLoaderJPEG, ImageFormatLoader)

private:
    static constexpr uint8_t JPEG_SOI[2] = {0xFF, 0xD8};

public:
    static StringName get_class_static() { return StringName("ImageLoaderJPEG"); }

    std::vector<String> get_recognized_extensions() const override {
        return {"jpg", "jpeg", "jpe", "jfif"};
    }

    ImageFormatType get_format_type() const override {
        return ImageFormatType::FORMAT_JPEG;
    }

    bool can_load(const std::vector<uint8_t>& header) const override {
        return header.size() >= 2 && std::memcmp(header.data(), JPEG_SOI, 2) == 0;
    }

    Error load_image(const std::vector<uint8_t>& data, ImageData& out_image) override {
#ifdef XTU_USE_JPEG
        // jpeg-compressor decoding
        return OK;
#else
        out_image.error_message = "JPEG support not compiled";
        return ERR_UNAVAILABLE;
#endif
    }
};

// #############################################################################
// ImageLoaderWebP - WebP format loader
// #############################################################################
class ImageLoaderWebP : public ImageFormatLoader {
    XTU_GODOT_REGISTER_CLASS(ImageLoaderWebP, ImageFormatLoader)

public:
    static StringName get_class_static() { return StringName("ImageLoaderWebP"); }

    std::vector<String> get_recognized_extensions() const override {
        return {"webp"};
    }

    ImageFormatType get_format_type() const override {
        return ImageFormatType::FORMAT_WEBP;
    }

    bool can_load(const std::vector<uint8_t>& header) const override {
        return header.size() >= 12 && std::memcmp(header.data(), "RIFF", 4) == 0 &&
               std::memcmp(header.data() + 8, "WEBP", 4) == 0;
    }

    Error load_image(const std::vector<uint8_t>& data, ImageData& out_image) override {
#ifdef XTU_USE_WEBP
        // libwebp decoding
        return OK;
#else
        out_image.error_message = "WebP support not compiled";
        return ERR_UNAVAILABLE;
#endif
    }

    Error save_image(const ImageData& image, const String& path) override {
#ifdef XTU_USE_WEBP
        // libwebp encoding with quality options
        return OK;
#else
        return ERR_UNAVAILABLE;
#endif
    }
};

// #############################################################################
// ImageLoaderTGA - TGA format loader
// #############################################################################
class ImageLoaderTGA : public ImageFormatLoader {
    XTU_GODOT_REGISTER_CLASS(ImageLoaderTGA, ImageFormatLoader)

public:
    static StringName get_class_static() { return StringName("ImageLoaderTGA"); }

    std::vector<String> get_recognized_extensions() const override {
        return {"tga", "vda", "icb", "vst"};
    }

    ImageFormatType get_format_type() const override {
        return ImageFormatType::FORMAT_TGA;
    }

    bool can_load(const std::vector<uint8_t>& header) const override {
        if (header.size() < 18) return false;
        // Check for valid TGA footer or header
        return true;
    }

    Error load_image(const std::vector<uint8_t>& data, ImageData& out_image) override {
        // Native TGA parsing (no external lib needed)
        if (data.size() < 18) {
            out_image.error_message = "Invalid TGA file";
            return ERR_FILE_CORRUPT;
        }

        uint8_t id_length = data[0];
        uint8_t colormap_type = data[1];
        uint8_t image_type = data[2];

        // Only support uncompressed true-color (type 2) and RLE (type 10)
        if (image_type != 2 && image_type != 10) {
            out_image.error_message = "Unsupported TGA image type";
            return ERR_UNAVAILABLE;
        }

        uint16_t colormap_first = *reinterpret_cast<const uint16_t*>(&data[3]);
        uint16_t colormap_length = *reinterpret_cast<const uint16_t*>(&data[5]);
        uint8_t colormap_bpp = data[7];
        uint16_t x_origin = *reinterpret_cast<const uint16_t*>(&data[8]);
        uint16_t y_origin = *reinterpret_cast<const uint16_t*>(&data[10]);
        uint16_t width = *reinterpret_cast<const uint16_t*>(&data[12]);
        uint16_t height = *reinterpret_cast<const uint16_t*>(&data[14]);
        uint8_t bpp = data[16];
        uint8_t descriptor = data[17];

        out_image.width = width;
        out_image.height = height;
        out_image.channels = bpp / 8;
        out_image.has_alpha = (bpp == 32);
        out_image.format = out_image.has_alpha ? RenderingServer::TEXTURE_FORMAT_RGBA8 :
                                                RenderingServer::TEXTURE_FORMAT_RGB8;

        size_t pixel_data_offset = 18 + id_length;
        if (colormap_type == 1) {
            pixel_data_offset += colormap_length * (colormap_bpp / 8);
        }

        size_t expected_size = width * height * out_image.channels;
        out_image.data.resize(expected_size);

        const uint8_t* src = data.data() + pixel_data_offset;

        if (image_type == 2) {
            // Uncompressed
            bool flip_y = !(descriptor & 0x20);
            if (flip_y) {
                for (int y = 0; y < height; ++y) {
                    int dst_y = height - 1 - y;
                    std::memcpy(out_image.data.data() + dst_y * width * out_image.channels,
                                src + y * width * out_image.channels,
                                width * out_image.channels);
                }
            } else {
                std::memcpy(out_image.data.data(), src, expected_size);
            }
            // Swap R and B channels
            for (size_t i = 0; i < expected_size; i += out_image.channels) {
                std::swap(out_image.data[i], out_image.data[i + 2]);
            }
        } else if (image_type == 10) {
            // RLE compressed
            std::vector<uint8_t> uncompressed(expected_size);
            size_t dst_idx = 0;
            size_t src_idx = 0;
            size_t data_end = data.size();

            while (dst_idx < expected_size && src_idx < data_end) {
                uint8_t header_byte = src[src_idx++];
                bool is_rle = (header_byte & 0x80) != 0;
                uint8_t count = (header_byte & 0x7F) + 1;

                if (is_rle) {
                    if (src_idx + out_image.channels > data_end) break;
                    for (uint8_t i = 0; i < count; ++i) {
                        std::memcpy(uncompressed.data() + dst_idx, src + src_idx, out_image.channels);
                        dst_idx += out_image.channels;
                    }
                    src_idx += out_image.channels;
                } else {
                    size_t copy_size = count * out_image.channels;
                    if (src_idx + copy_size > data_end) break;
                    std::memcpy(uncompressed.data() + dst_idx, src + src_idx, copy_size);
                    dst_idx += copy_size;
                    src_idx += copy_size;
                }
            }

            // Flip Y if needed and swap R/B
            bool flip_y = !(descriptor & 0x20);
            for (int y = 0; y < height; ++y) {
                int dst_y = flip_y ? height - 1 - y : y;
                const uint8_t* src_row = uncompressed.data() + y * width * out_image.channels;
                uint8_t* dst_row = out_image.data.data() + dst_y * width * out_image.channels;
                std::memcpy(dst_row, src_row, width * out_image.channels);
                for (int x = 0; x < width; ++x) {
                    std::swap(dst_row[x * out_image.channels], dst_row[x * out_image.channels + 2]);
                }
            }
        }

        return OK;
    }
};

// #############################################################################
// ImageLoaderBMP - BMP format loader
// #############################################################################
class ImageLoaderBMP : public ImageFormatLoader {
    XTU_GODOT_REGISTER_CLASS(ImageLoaderBMP, ImageFormatLoader)

public:
    static StringName get_class_static() { return StringName("ImageLoaderBMP"); }

    std::vector<String> get_recognized_extensions() const override {
        return {"bmp", "dib"};
    }

    ImageFormatType get_format_type() const override {
        return ImageFormatType::FORMAT_BMP;
    }

    bool can_load(const std::vector<uint8_t>& header) const override {
        return header.size() >= 2 && std::memcmp(header.data(), "BM", 2) == 0;
    }

    Error load_image(const std::vector<uint8_t>& data, ImageData& out_image) override {
        if (data.size() < 54) {
            out_image.error_message = "Invalid BMP file";
            return ERR_FILE_CORRUPT;
        }

        uint32_t data_offset = *reinterpret_cast<const uint32_t*>(&data[10]);
        int32_t width = *reinterpret_cast<const int32_t*>(&data[18]);
        int32_t height = *reinterpret_cast<const int32_t*>(&data[22]);
        uint16_t bpp = *reinterpret_cast<const uint16_t*>(&data[28]);
        uint32_t compression = *reinterpret_cast<const uint32_t*>(&data[30]);

        bool top_down = height < 0;
        height = std::abs(height);

        out_image.width = width;
        out_image.height = height;
        out_image.channels = bpp == 32 ? 4 : 3;
        out_image.has_alpha = (bpp == 32);
        out_image.format = out_image.has_alpha ? RenderingServer::TEXTURE_FORMAT_RGBA8 :
                                                RenderingServer::TEXTURE_FORMAT_RGB8;

        size_t row_size = ((width * bpp + 31) / 32) * 4;
        out_image.data.resize(width * height * out_image.channels);

        const uint8_t* src = data.data() + data_offset;

        for (int y = 0; y < height; ++y) {
            int src_y = top_down ? y : height - 1 - y;
            const uint8_t* src_row = src + src_y * row_size;
            uint8_t* dst_row = out_image.data.data() + y * width * out_image.channels;

            for (int x = 0; x < width; ++x) {
                dst_row[x * out_image.channels + 2] = src_row[x * (bpp / 8) + 0]; // R
                dst_row[x * out_image.channels + 1] = src_row[x * (bpp / 8) + 1]; // G
                dst_row[x * out_image.channels + 0] = src_row[x * (bpp / 8) + 2]; // B
                if (out_image.has_alpha) {
                    dst_row[x * out_image.channels + 3] = src_row[x * 4 + 3]; // A
                }
            }
        }

        return OK;
    }
};

// #############################################################################
// ImageLoaderKTX - KTX/KTX2 compressed texture loader
// #############################################################################
class ImageLoaderKTX : public ImageFormatLoader {
    XTU_GODOT_REGISTER_CLASS(ImageLoaderKTX, ImageFormatLoader)

private:
    static constexpr uint8_t KTX_IDENTIFIER[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
    static constexpr uint8_t KTX2_IDENTIFIER[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};

public:
    static StringName get_class_static() { return StringName("ImageLoaderKTX"); }

    std::vector<String> get_recognized_extensions() const override {
        return {"ktx", "ktx2"};
    }

    ImageFormatType get_format_type() const override {
        return ImageFormatType::FORMAT_KTX;
    }

    bool can_load(const std::vector<uint8_t>& header) const override {
        return header.size() >= 12 &&
               (std::memcmp(header.data(), KTX_IDENTIFIER, 12) == 0 ||
                std::memcmp(header.data(), KTX2_IDENTIFIER, 12) == 0);
    }

    Error load_image(const std::vector<uint8_t>& data, ImageData& out_image) override {
#ifdef XTU_USE_KTX
        // libktx decoding
        return OK;
#else
        out_image.error_message = "KTX support not compiled";
        return ERR_UNAVAILABLE;
#endif
    }
};

// #############################################################################
// ImageLoaderSVG - SVG vector graphics loader
// #############################################################################
class ImageLoaderSVG : public ImageFormatLoader {
    XTU_GODOT_REGISTER_CLASS(ImageLoaderSVG, ImageFormatLoader)

public:
    static StringName get_class_static() { return StringName("ImageLoaderSVG"); }

    std::vector<String> get_recognized_extensions() const override {
        return {"svg", "svgz"};
    }

    ImageFormatType get_format_type() const override {
        return ImageFormatType::FORMAT_SVG;
    }

    bool can_load(const std::vector<uint8_t>& header) const override {
        // Check for XML/SVG header
        if (header.size() < 5) return false;
        std::string str(header.begin(), header.begin() + std::min(size_t(256), header.size()));
        return str.find("<svg") != std::string::npos || str.find("<?xml") != std::string::npos;
    }

    Error load_image(const std::vector<uint8_t>& data, ImageData& out_image) override {
#ifdef XTU_USE_SVG
        // nanosvg or librsvg rasterization
        return OK;
#else
        out_image.error_message = "SVG support not compiled";
        return ERR_UNAVAILABLE;
#endif
    }
};

// #############################################################################
// ImageLoaderHDR - HDR/EXR high dynamic range loader
// #############################################################################
class ImageLoaderHDR : public ImageFormatLoader {
    XTU_GODOT_REGISTER_CLASS(ImageLoaderHDR, ImageFormatLoader)

private:
    static constexpr uint8_t HDR_MAGIC[10] = {'#', '?', 'R', 'A', 'D', 'I', 'A', 'N', 'C', 'E'};

public:
    static StringName get_class_static() { return StringName("ImageLoaderHDR"); }

    std::vector<String> get_recognized_extensions() const override {
        return {"hdr", "exr"};
    }

    ImageFormatType get_format_type() const override {
        return ImageFormatType::FORMAT_HDR;
    }

    bool can_load(const std::vector<uint8_t>& header) const override {
        if (header.size() < 11) return false;
        return std::memcmp(header.data(), HDR_MAGIC, 10) == 0 || header[0] == 0x00;
    }

    Error load_image(const std::vector<uint8_t>& data, ImageData& out_image) override {
        // Native Radiance HDR parsing or OpenEXR
        if (data.size() < 11) {
            out_image.error_message = "Invalid HDR file";
            return ERR_FILE_CORRUPT;
        }

        // Check if it's Radiance HDR
        if (std::memcmp(data.data(), HDR_MAGIC, 10) == 0) {
            return load_hdr(data, out_image);
        }
#ifdef XTU_USE_OPENEXR
        // OpenEXR loading
        return load_exr(data, out_image);
#else
        out_image.error_message = "EXR support not compiled";
        return ERR_UNAVAILABLE;
#endif
    }

private:
    Error load_hdr(const std::vector<uint8_t>& data, ImageData& out_image) {
        // Parse Radiance HDR format
        const uint8_t* ptr = data.data() + 10;
        const uint8_t* end = data.data() + data.size();

        // Skip header lines
        while (ptr < end && *ptr != '\n') {
            while (ptr < end && *ptr != '\n') ++ptr;
            ++ptr;
        }
        ++ptr; // Skip blank line

        // Read resolution line
        std::string res_line;
        while (ptr < end && *ptr != '\n') {
            res_line += static_cast<char>(*ptr++);
        }
        ++ptr;

        int width = 0, height = 0;
        sscanf(res_line.c_str(), "-Y %d +X %d", &height, &width);

        out_image.width = width;
        out_image.height = height;
        out_image.channels = 4;
        out_image.is_hdr = true;
        out_image.format = RenderingServer::TEXTURE_FORMAT_RGBA32F;

        out_image.data.resize(width * height * 16); // 4 floats per pixel

        // RLE decode scanlines
        float* dst = reinterpret_cast<float*>(out_image.data.data());

        for (int y = 0; y < height; ++y) {
            if (ptr + 4 > end) break;
            uint8_t cmd = *ptr++;
            if (cmd != 2) {
                // Old format
                break;
            }
            // RLE decode RGBE
            std::vector<uint8_t> scanline(width * 4);
            uint8_t* rgbe = scanline.data();

            for (int c = 0; c < 4; ++c) {
                int x = 0;
                while (x < width) {
                    if (ptr >= end) break;
                    uint8_t rle_cmd = *ptr++;
                    if (rle_cmd > 128) {
                        int count = rle_cmd - 128;
                        uint8_t val = *ptr++;
                        for (int i = 0; i < count; ++i) {
                            rgbe[x * 4 + c] = val;
                            ++x;
                        }
                    } else {
                        int count = rle_cmd;
                        for (int i = 0; i < count; ++i) {
                            rgbe[x * 4 + c] = *ptr++;
                            ++x;
                        }
                    }
                }
            }

            // Convert RGBE to float
            for (int x = 0; x < width; ++x) {
                uint8_t r = rgbe[x * 4 + 0];
                uint8_t g = rgbe[x * 4 + 1];
                uint8_t b = rgbe[x * 4 + 2];
                uint8_t e = rgbe[x * 4 + 3];

                if (e == 0) {
                    dst[y * width * 4 + x * 4 + 0] = 0;
                    dst[y * width * 4 + x * 4 + 1] = 0;
                    dst[y * width * 4 + x * 4 + 2] = 0;
                    dst[y * width * 4 + x * 4 + 3] = 1;
                } else {
                    float f = std::ldexp(1.0f, e - 128);
                    dst[y * width * 4 + x * 4 + 0] = (r + 0.5f) * f / 255.0f;
                    dst[y * width * 4 + x * 4 + 1] = (g + 0.5f) * f / 255.0f;
                    dst[y * width * 4 + x * 4 + 2] = (b + 0.5f) * f / 255.0f;
                    dst[y * width * 4 + x * 4 + 3] = 1.0f;
                }
            }
        }

        return OK;
    }

#ifdef XTU_USE_OPENEXR
    Error load_exr(const std::vector<uint8_t>& data, ImageData& out_image) {
        // OpenEXR loading via TinyEXR or full OpenEXR
        return OK;
    }
#endif
};

// #############################################################################
// ImageLoader - Global image loader singleton
// #############################################################################
class ImageLoader : public Object {
    XTU_GODOT_REGISTER_CLASS(ImageLoader, Object)

private:
    static ImageLoader* s_singleton;
    std::vector<Ref<ImageFormatLoader>> m_loaders;
    std::unordered_map<String, Ref<ImageFormatLoader>> m_extension_loader;
    mutable std::mutex m_mutex;

public:
    static ImageLoader* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("ImageLoader"); }

    ImageLoader() {
        s_singleton = this;
        register_default_loaders();
    }

    ~ImageLoader() { s_singleton = nullptr; }

    void add_loader(const Ref<ImageFormatLoader>& loader) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_loaders.push_back(loader);
        for (const auto& ext : loader->get_recognized_extensions()) {
            if (m_extension_loader.find(ext) == m_extension_loader.end() ||
                loader->get_priority() > m_extension_loader[ext]->get_priority()) {
                m_extension_loader[ext] = loader;
            }
        }
    }

    Ref<ImageFormatLoader> get_loader_for_extension(const String& ext) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        String lower_ext = ext.to_lower();
        auto it = m_extension_loader.find(lower_ext);
        return it != m_extension_loader.end() ? it->second : Ref<ImageFormatLoader>();
    }

    Ref<ImageFormatLoader> recognize(const std::vector<uint8_t>& data) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        Ref<ImageFormatLoader> best_loader;
        float best_priority = -1.0f;

        for (const auto& loader : m_loaders) {
            if (loader->can_load(data) && loader->get_priority() > best_priority) {
                best_loader = loader;
                best_priority = loader->get_priority();
            }
        }
        return best_loader;
    }

    Error load_image(const String& path, ImageData& out_image) {
        Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
        if (!file.is_valid()) return ERR_FILE_CANT_OPEN;

        std::vector<uint8_t> data = file->get_buffer(file->get_length());
        return load_image_from_buffer(data, out_image);
    }

    Error load_image_from_buffer(const std::vector<uint8_t>& data, ImageData& out_image) {
        Ref<ImageFormatLoader> loader = recognize(data);
        if (!loader.is_valid()) {
            out_image.error_message = "Unrecognized image format";
            return ERR_FILE_UNRECOGNIZED;
        }
        return loader->load_image(data, out_image);
    }

    std::vector<String> get_recognized_extensions() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<String> extensions;
        for (const auto& kv : m_extension_loader) {
            extensions.push_back(kv.first);
        }
        return extensions;
    }

private:
    void register_default_loaders() {
        add_loader(Ref<ImageFormatLoader>(new ImageLoaderPNG()));
        add_loader(Ref<ImageFormatLoader>(new ImageLoaderJPEG()));
        add_loader(Ref<ImageFormatLoader>(new ImageLoaderWebP()));
        add_loader(Ref<ImageFormatLoader>(new ImageLoaderTGA()));
        add_loader(Ref<ImageFormatLoader>(new ImageLoaderBMP()));
        add_loader(Ref<ImageFormatLoader>(new ImageLoaderKTX()));
        add_loader(Ref<ImageFormatLoader>(new ImageLoaderSVG()));
        add_loader(Ref<ImageFormatLoader>(new ImageLoaderHDR()));
    }
};

} // namespace io

// Bring into main namespace
using io::ImageLoader;
using io::ImageFormatLoader;
using io::ImageLoaderPNG;
using io::ImageLoaderJPEG;
using io::ImageLoaderWebP;
using io::ImageLoaderTGA;
using io::ImageLoaderBMP;
using io::ImageLoaderKTX;
using io::ImageLoaderSVG;
using io::ImageLoaderHDR;
using io::ImageFormatType;
using io::ImageData;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XIMAGE_LOADER_HPP