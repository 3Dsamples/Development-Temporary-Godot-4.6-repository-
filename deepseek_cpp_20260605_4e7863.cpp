//File 0510 : xtensor-io/ximage.hpp
//Image file reader/writer (PNG, BMP, JPEG, TIFF) via FreeImage or STB, with SIMD‑accelerated channel conversion and memory‑efficient loading for 2D/3D simulation textures.
#ifndef XTENSOR_IO_XIMAGE_HPP
#define XTENSOR_IO_XIMAGE_HPP

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

// We use a small header‑only image library (STB) to avoid heavy dependencies.
// Users can also opt into FreeImage by defining XTENSOR_IO_USE_FREEIMAGE.
#if defined(XTENSOR_IO_USE_FREEIMAGE)
    #include <FreeImage.h>
    #pragma comment(lib, "FreeImage.lib")
#else
    #define STB_IMAGE_IMPLEMENTATION
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image.h"
    #include "stb_image_write.h"
#endif

namespace xt {
namespace io {

    /**
     * @enum image_channel_order
     * @brief Specifies how colour channels are arranged.
     */
    enum class image_channel_order : uint8_t {
        hwc,  // height × width × channels (default)
        chw   // channels × height × width (3D array)
    };

    namespace detail {

        /**
         * Map number of channels to STB/FI constants.
         */
        constexpr int to_stb_channels(int n) noexcept { return n; }

        /**
         * Convert uint8 RGBA to floating‑point [0,1] with SIMD.
         */
        inline void u8_to_float(const unsigned char* src, float* dst, std::size_t count) {
            constexpr float scale = 1.0f / 255.0f;
            if constexpr (simd_enabled_v<float>) {
                using simd_type = xsimd::batch<float, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = count / simd_size;
                simd_type vscale(scale);
                for (std::size_t i = 0; i < vec_count; ++i) {
                    alignas(64) float buf[simd_size];
                    for (std::size_t k = 0; k < simd_size; ++k)
                        buf[k] = static_cast<float>(src[i * simd_size + k]);
                    simd_type v = simd_type::load_aligned(buf);
                    (v * vscale).store_unaligned(dst + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < count; ++i)
                    dst[i] = static_cast<float>(src[i]) * scale;
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    dst[i] = static_cast<float>(src[i]) * scale;
            }
        }

        /**
         * Convert float [0,1] to uint8 with SIMD.
         */
        inline void float_to_u8(const float* src, unsigned char* dst, std::size_t count) {
            constexpr float scale = 255.0f;
            if constexpr (simd_enabled_v<float>) {
                using simd_type = xsimd::batch<float, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = count / simd_size;
                simd_type vscale(scale);
                for (std::size_t i = 0; i < vec_count; ++i) {
                    simd_type v = simd_type::load_unaligned(src + i * simd_size);
                    v = v * vscale;
                    alignas(64) float buf[simd_size];
                    v.store_aligned(buf);
                    for (std::size_t k = 0; k < simd_size; ++k) {
                        float val = buf[k];
                        if (val > 255.0f) val = 255.0f;
                        if (val < 0.0f) val = 0.0f;
                        dst[i * simd_size + k] = static_cast<unsigned char>(val);
                    }
                }
                for (std::size_t i = vec_count * simd_size; i < count; ++i) {
                    float val = src[i] * scale;
                    if (val > 255.0f) val = 255.0f;
                    if (val < 0.0f) val = 0.0f;
                    dst[i] = static_cast<unsigned char>(val);
                }
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    float val = src[i] * scale;
                    if (val > 255.0f) val = 255.0f;
                    if (val < 0.0f) val = 0.0f;
                    dst[i] = static_cast<unsigned char>(val);
                }
            }
        }
    }

    /**
     * Load an image file into an xtensor array.
     * @param filename Path to image (PNG, JPG, BMP, TIFF, etc.).
     * @param order Desired channel order (HWC or CHW).
     * @param desired_channels Number of channels to force (0 = keep original).
     * @return xarray<float> with shape (height, width, channels) or (channels, height, width).
     */
    inline auto load_image(const std::string& filename,
                           image_channel_order order = image_channel_order::hwc,
                           int desired_channels = 0) {
        int width, height, channels;
        unsigned char* raw = nullptr;

#if defined(XTENSOR_IO_USE_FREEIMAGE)
        FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(filename.c_str());
        FIBITMAP* dib = FreeImage_Load(fif, filename.c_str());
        if (!dib) throw std::runtime_error("FreeImage: cannot load " + filename);
        width = FreeImage_GetWidth(dib);
        height = FreeImage_GetHeight(dib);
        channels = desired_channels ? desired_channels : (FreeImage_GetBPP(dib) / 8);
        raw = FreeImage_GetBits(dib);
        std::size_t total = static_cast<std::size_t>(width) * height * channels;
#else
        raw = stbi_load(filename.c_str(), &width, &height, &channels,
                        desired_channels);
        if (!raw) throw std::runtime_error("stb_image: cannot load " + filename);
        if (desired_channels) channels = desired_channels;
        std::size_t total = static_cast<std::size_t>(width) * height * channels;
#endif

        // Allocate output
        xarray_container<uvector<float>, DEFAULT_LAYOUT, std::vector<std::size_t>> result;
        if (order == image_channel_order::hwc) {
            result.resize({static_cast<std::size_t>(height),
                           static_cast<std::size_t>(width),
                           static_cast<std::size_t>(channels)});
        } else {
            result.resize({static_cast<std::size_t>(channels),
                           static_cast<std::size_t>(height),
                           static_cast<std::size_t>(width)});
        }

        float* dst = result.data();
        std::size_t total = static_cast<std::size_t>(width) * height * channels;
        detail::u8_to_float(raw, dst, total);

#if defined(XTENSOR_IO_USE_FREEIMAGE)
        FreeImage_Unload(dib);
#else
        stbi_image_free(raw);
#endif

        // If CHW order, permute dimensions
        if (order == image_channel_order::chw && channels > 1) {
            // Permute from HWC to CHW: (2,0,1)
            std::vector<std::size_t> perm = {2, 0, 1};
            result = xt::transpose(result, perm);
        }

        return result;
    }

    /**
     * Save an xtensor array as an image file.
     * @param filename Output path (extension determines format).
     * @param expr The array to save. If shape is (H,W,C) or (C,H,W).
     * @param order Channel order of the input array.
     * @param quality JPEG quality (1‑100), ignored for PNG/BMP.
     */
    template <class E>
    inline void save_image(const std::string& filename,
                           const xexpression<E>& expr,
                           image_channel_order order = image_channel_order::hwc,
                           int quality = 95) {
        auto arr = xt::eval(expr.derived_cast());
        auto sh = arr.shape();
        if (sh.size() < 2 || sh.size() > 3)
            throw std::runtime_error("save_image: array must be 2D (gray) or 3D (colour).");

        int height, width, channels;
        if (sh.size() == 2) {
            height = static_cast<int>(sh[0]);
            width = static_cast<int>(sh[1]);
            channels = 1;
        } else {
            if (order == image_channel_order::hwc) {
                height = static_cast<int>(sh[0]);
                width = static_cast<int>(sh[1]);
                channels = static_cast<int>(sh[2]);
            } else {
                channels = static_cast<int>(sh[0]);
                height = static_cast<int>(sh[1]);
                width = static_cast<int>(sh[2]);
            }
        }

        // If input is CHW, permute to HWC for writing
        if (order == image_channel_order::chw && channels > 1) {
            std::vector<std::size_t> perm = {1, 2, 0};
            arr = xt::transpose(arr, perm);
        }

        std::size_t total = static_cast<std::size_t>(width) * height * channels;
        std::vector<unsigned char> raw(total);
        detail::float_to_u8(arr.data(), raw.data(), total);

        // Determine format from extension
        std::string ext = filename.substr(filename.rfind('.') + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

#if defined(XTENSOR_IO_USE_FREEIMAGE)
        FIBITMAP* dib = FreeImage_ConvertFromRawBits(
            raw.data(), width, height, channels * width,
            channels * 8, 0xFF0000, 0x00FF00, 0x0000FF, false);
        FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(filename.c_str());
        FreeImage_Save(fif, dib, filename.c_str());
        FreeImage_Unload(dib);
#else
        int ok = 0;
        if (ext == "png")
            ok = stbi_write_png(filename.c_str(), width, height, channels, raw.data(), width * channels);
        else if (ext == "bmp")
            ok = stbi_write_bmp(filename.c_str(), width, height, channels, raw.data());
        else if (ext == "jpg" || ext == "jpeg")
            ok = stbi_write_jpg(filename.c_str(), width, height, channels, raw.data(), quality);
        else if (ext == "tga")
            ok = stbi_write_tga(filename.c_str(), width, height, channels, raw.data());
        else
            throw std::runtime_error("Unsupported image format: " + ext);
        if (!ok) throw std::runtime_error("Failed to write image: " + filename);
#endif
    }

    /**
     * Query image metadata without loading pixel data.
     * @return Tuple of (width, height, channels).
     */
    inline auto image_info(const std::string& filename) {
        int width, height, channels;
#if defined(XTENSOR_IO_USE_FREEIMAGE)
        FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(filename.c_str());
        FIBITMAP* dib = FreeImage_Load(fif, filename.c_str());
        if (!dib) throw std::runtime_error("FreeImage: cannot load " + filename);
        width = FreeImage_GetWidth(dib);
        height = FreeImage_GetHeight(dib);
        channels = FreeImage_GetBPP(dib) / 8;
        FreeImage_Unload(dib);
#else
        if (!stbi_info(filename.c_str(), &width, &height, &channels))
            throw std::runtime_error("Cannot query image: " + filename);
#endif
        return std::make_tuple(width, height, channels);
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XIMAGE_HPP