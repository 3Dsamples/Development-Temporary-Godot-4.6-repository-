// include/xtu/godot/xdenoise.hpp
// xtensor-unified - OpenImageDenoise wrapper for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XDENOISE_HPP
#define XTU_GODOT_XDENOISE_HPP

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
#include "xtu/parallel/xparallel.hpp"

#ifdef XTU_USE_OIDN
#include <OpenImageDenoise/oidn.hpp>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {
namespace denoise {

// #############################################################################
// Denoiser quality levels
// #############################################################################
enum class DenoiserQuality : uint8_t {
    QUALITY_FAST = 0,
    QUALITY_BALANCED = 1,
    QUALITY_HIGH = 2
};

// #############################################################################
// Denoiser filter types
// #############################################################################
enum class DenoiserFilterType : uint8_t {
    FILTER_RT = 0,           // Ray tracing denoiser
    FILTER_RT_LIGHTMAP = 1,  // Lightmap-specific denoiser
    FILTER_RT_ALBEDO = 2,    // With albedo guide
    FILTER_RT_NORMAL = 3     // With normal guide
};

// #############################################################################
// Denoiser configuration
// #############################################################################
struct DenoiserConfig {
    DenoiserQuality quality = DenoiserQuality::QUALITY_BALANCED;
    DenoiserFilterType filter_type = DenoiserFilterType::FILTER_RT;
    bool hdr = true;
    bool clean_aux = true;
    int max_memory_mb = 6000;
    int num_threads = 0;
};

// #############################################################################
// OIDNDenoiser - OpenImageDenoise wrapper
// #############################################################################
class OIDNDenoiser : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(OIDNDenoiser, RefCounted)

private:
#ifdef XTU_USE_OIDN
    oidn::DeviceRef m_device;
#endif
    DenoiserConfig m_config;
    bool m_initialized = false;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("OIDNDenoiser"); }

    OIDNDenoiser() {
        initialize();
    }

    ~OIDNDenoiser() {
        cleanup();
    }

    bool initialize() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized) return true;

#ifdef XTU_USE_OIDN
        try {
            m_device = oidn::newDevice();
            const char* error = nullptr;
            if (m_device.getError(error) != oidn::Error::None) {
                return false;
            }
            m_device.set("numThreads", m_config.num_threads > 0 ? m_config.num_threads : 
                                        static_cast<int>(std::thread::hardware_concurrency()));
            m_device.commit();
            m_initialized = true;
            return true;
        } catch (...) {
            return false;
        }
#else
        return false;
#endif
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(m_mutex);
#ifdef XTU_USE_OIDN
        m_device.release();
#endif
        m_initialized = false;
    }

    bool is_initialized() const { return m_initialized; }

    void set_config(const DenoiserConfig& config) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_config = config;
    }

    DenoiserConfig get_config() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_config;
    }

    void set_quality(DenoiserQuality quality) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_config.quality = quality;
    }

    void set_hdr(bool hdr) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_config.hdr = hdr;
    }

    void set_filter_type(DenoiserFilterType type) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_config.filter_type = type;
    }

    // #########################################################################
    // Denoise color only
    // #########################################################################
    std::vector<float> denoise_color(const std::vector<float>& color, int width, int height) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_initialized) return color;

        std::vector<float> result(color.size());

#ifdef XTU_USE_OIDN
        oidn::FilterRef filter = m_device.newFilter("RT");
        filter.setImage("color", color.data(), oidn::Format::Float3, width, height);
        filter.setImage("output", result.data(), oidn::Format::Float3, width, height);
        filter.set("hdr", m_config.hdr);
        filter.set("quality", quality_to_oidn(m_config.quality));
        filter.commit();
        filter.execute();
#endif

        return result;
    }

    // #########################################################################
    // Denoise with albedo guide
    // #########################################################################
    std::vector<float> denoise_with_albedo(const std::vector<float>& color,
                                            const std::vector<float>& albedo,
                                            int width, int height) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_initialized) return color;

        std::vector<float> result(color.size());

#ifdef XTU_USE_OIDN
        oidn::FilterRef filter = m_device.newFilter("RT");
        filter.setImage("color", color.data(), oidn::Format::Float3, width, height);
        filter.setImage("albedo", albedo.data(), oidn::Format::Float3, width, height);
        filter.setImage("output", result.data(), oidn::Format::Float3, width, height);
        filter.set("hdr", m_config.hdr);
        filter.set("quality", quality_to_oidn(m_config.quality));
        filter.commit();
        filter.execute();
#endif

        return result;
    }

    // #########################################################################
    // Denoise with albedo and normal guides
    // #########################################################################
    std::vector<float> denoise_full(const std::vector<float>& color,
                                     const std::vector<float>& albedo,
                                     const std::vector<float>& normal,
                                     int width, int height) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_initialized) return color;

        std::vector<float> result(color.size());

#ifdef XTU_USE_OIDN
        oidn::FilterRef filter = m_device.newFilter("RT");
        filter.setImage("color", color.data(), oidn::Format::Float3, width, height);
        filter.setImage("albedo", albedo.data(), oidn::Format::Float3, width, height);
        filter.setImage("normal", normal.data(), oidn::Format::Float3, width, height);
        filter.setImage("output", result.data(), oidn::Format::Float3, width, height);
        filter.set("hdr", m_config.hdr);
        filter.set("cleanAux", m_config.clean_aux);
        filter.set("quality", quality_to_oidn(m_config.quality));
        filter.commit();
        filter.execute();
#endif

        return result;
    }

    // #########################################################################
    // Denoise lightmap
    // #########################################################################
    std::vector<float> denoise_lightmap(const std::vector<float>& lightmap,
                                         int width, int height) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_initialized) return lightmap;

        std::vector<float> result(lightmap.size());

#ifdef XTU_USE_OIDN
        oidn::FilterRef filter = m_device.newFilter("RTLightmap");
        filter.setImage("color", lightmap.data(), oidn::Format::Float3, width, height);
        filter.setImage("output", result.data(), oidn::Format::Float3, width, height);
        filter.set("quality", quality_to_oidn(m_config.quality));
        filter.commit();
        filter.execute();
#endif

        return result;
    }

    // #########################################################################
    // Denoise with progress callback
    // #########################################################################
    void denoise_async(const std::vector<float>& color,
                       const std::vector<float>& albedo,
                       const std::vector<float>& normal,
                       int width, int height,
                       std::function<void(const std::vector<float>&)> callback,
                       std::function<void(float)> progress = nullptr) {
        std::thread([=]() {
            if (progress) progress(0.1f);
            auto result = denoise_full(color, albedo, normal, width, height);
            if (progress) progress(1.0f);
            if (callback) callback(result);
        }).detach();
    }

private:
#ifdef XTU_USE_OIDN
    static oidn::Quality quality_to_oidn(DenoiserQuality quality) {
        switch (quality) {
            case DenoiserQuality::QUALITY_FAST: return oidn::Quality::Fast;
            case DenoiserQuality::QUALITY_HIGH: return oidn::Quality::High;
            default: return oidn::Quality::Balanced;
        }
    }
#endif
};

// #############################################################################
// DenoiserPool - Pool of denoisers for parallel processing
// #############################################################################
class DenoiserPool : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(DenoiserPool, RefCounted)

private:
    std::vector<Ref<OIDNDenoiser>> m_denoisers;
    DenoiserConfig m_config;
    mutable std::mutex m_mutex;
    size_t m_next = 0;

public:
    static StringName get_class_static() { return StringName("DenoiserPool"); }

    void initialize(int count, const DenoiserConfig& config = {}) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_config = config;
        m_denoisers.clear();
        m_denoisers.reserve(count);
        for (int i = 0; i < count; ++i) {
            Ref<OIDNDenoiser> denoiser;
            denoiser.instance();
            denoiser->set_config(config);
            if (denoiser->initialize()) {
                m_denoisers.push_back(denoiser);
            }
        }
    }

    Ref<OIDNDenoiser> acquire() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_denoisers.empty()) {
            Ref<OIDNDenoiser> denoiser;
            denoiser.instance();
            denoiser->set_config(m_config);
            denoiser->initialize();
            return denoiser;
        }
        Ref<OIDNDenoiser> denoiser = m_denoisers[m_next];
        m_next = (m_next + 1) % m_denoisers.size();
        return denoiser;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_denoisers.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_denoisers.clear();
        m_next = 0;
    }
};

// #############################################################################
// SimpleBilateralDenoiser - Fallback bilateral filter
// #############################################################################
class SimpleBilateralDenoiser : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(SimpleBilateralDenoiser, RefCounted)

private:
    float m_spatial_sigma = 3.0f;
    float m_range_sigma = 0.1f;
    int m_kernel_radius = 3;

public:
    static StringName get_class_static() { return StringName("SimpleBilateralDenoiser"); }

    void set_spatial_sigma(float sigma) { m_spatial_sigma = sigma; }
    void set_range_sigma(float sigma) { m_range_sigma = sigma; }
    void set_kernel_radius(int radius) { m_kernel_radius = radius; }

    std::vector<float> denoise(const std::vector<float>& image, int width, int height, int channels = 3) {
        std::vector<float> result(image.size());

        parallel::parallel_for(0, height, [&](int y) {
            for (int x = 0; x < width; ++x) {
                std::vector<float> sum(channels, 0.0f);
                float total_weight = 0.0f;

                size_t center_idx = (y * width + x) * channels;
                std::vector<float> center_pixel(image.begin() + center_idx,
                                                 image.begin() + center_idx + channels);

                for (int dy = -m_kernel_radius; dy <= m_kernel_radius; ++dy) {
                    for (int dx = -m_kernel_radius; dx <= m_kernel_radius; ++dx) {
                        int nx = std::clamp(x + dx, 0, width - 1);
                        int ny = std::clamp(y + dy, 0, height - 1);

                        float spatial_dist = (dx * dx + dy * dy) / (2.0f * m_spatial_sigma * m_spatial_sigma);
                        float spatial_weight = std::exp(-spatial_dist);

                        size_t neighbor_idx = (ny * width + nx) * channels;
                        float range_dist = 0.0f;
                        for (int c = 0; c < channels; ++c) {
                            float diff = image[neighbor_idx + c] - center_pixel[c];
                            range_dist += diff * diff;
                        }
                        range_dist /= (2.0f * m_range_sigma * m_range_sigma);
                        float range_weight = std::exp(-range_dist);

                        float weight = spatial_weight * range_weight;
                        total_weight += weight;

                        for (int c = 0; c < channels; ++c) {
                            sum[c] += image[neighbor_idx + c] * weight;
                        }
                    }
                }

                if (total_weight > 0.0f) {
                    size_t out_idx = (y * width + x) * channels;
                    for (int c = 0; c < channels; ++c) {
                        result[out_idx + c] = sum[c] / total_weight;
                    }
                }
            }
        });

        return result;
    }
};

} // namespace denoise

// Bring into main namespace
using denoise::OIDNDenoiser;
using denoise::DenoiserPool;
using denoise::SimpleBilateralDenoiser;
using denoise::DenoiserQuality;
using denoise::DenoiserFilterType;
using denoise::DenoiserConfig;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XDENOISE_HPP