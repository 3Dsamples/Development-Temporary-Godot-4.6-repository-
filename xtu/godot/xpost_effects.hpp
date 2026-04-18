// include/xtu/godot/xpost_effects.hpp
// xtensor-unified - Post-processing effects for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XPOST_EFFECTS_HPP
#define XTU_GODOT_XPOST_EFFECTS_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/graphics/xgraphics.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace rendering {

// #############################################################################
// Part 1: SSAO - Screen Space Ambient Occlusion
// #############################################################################

class SSAOEffect : public Resource {
    XTU_GODOT_REGISTER_CLASS(SSAOEffect, Resource)

public:
    enum Quality {
        QUALITY_VERY_LOW,
        QUALITY_LOW,
        QUALITY_MEDIUM,
        QUALITY_HIGH,
        QUALITY_ULTRA
    };

    enum Algorithm {
        ALGORITHM_SSAO,
        ALGORITHM_HBAO,
        ALGORITHM_GTAO
    };

private:
    bool m_enabled = false;
    Algorithm m_algorithm = ALGORITHM_GTAO;
    Quality m_quality = QUALITY_MEDIUM;
    float m_radius = 1.0f;
    float m_intensity = 2.0f;
    float m_power = 1.5f;
    float m_detail = 0.5f;
    float m_horizon = 0.06f;
    float m_sharpness = 0.98f;
    float m_light_affect = 0.0f;
    float m_ao_channel_affect = 0.0f;
    bool m_half_size = true;
    bool m_blur_enabled = true;
    float m_blur_sharpness = 0.9f;

public:
    static StringName get_class_static() { return StringName("SSAOEffect"); }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void set_algorithm(Algorithm algo) { m_algorithm = algo; }
    Algorithm get_algorithm() const { return m_algorithm; }

    void set_quality(Quality quality) { m_quality = quality; }
    Quality get_quality() const { return m_quality; }

    void set_radius(float radius) { m_radius = radius; }
    float get_radius() const { return m_radius; }

    void set_intensity(float intensity) { m_intensity = intensity; }
    float get_intensity() const { return m_intensity; }

    void set_power(float power) { m_power = power; }
    float get_power() const { return m_power; }

    void set_half_size(bool half) { m_half_size = half; }
    bool get_half_size() const { return m_half_size; }

    void set_blur_enabled(bool enabled) { m_blur_enabled = enabled; }
    bool is_blur_enabled() const { return m_blur_enabled; }

    // Apply SSAO to scene
    RID render(RID source_depth, RID source_normal, RID target) {
        // Placeholder - would integrate with RenderingDevice compute shaders
        return RID();
    }
};

// #############################################################################
// Part 2: SSIL - Screen Space Indirect Lighting
// #############################################################################

class SSILEffect : public Resource {
    XTU_GODOT_REGISTER_CLASS(SSILEffect, Resource)

private:
    bool m_enabled = false;
    float m_radius = 5.0f;
    float m_intensity = 1.0f;
    float m_sharpness = 0.98f;
    float m_normal_rejection = 1.0f;
    bool m_half_size = true;

public:
    static StringName get_class_static() { return StringName("SSILEffect"); }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void set_radius(float radius) { m_radius = radius; }
    float get_radius() const { return m_radius; }

    void set_intensity(float intensity) { m_intensity = intensity; }
    float get_intensity() const { return m_intensity; }

    RID render(RID source_color, RID source_depth, RID source_normal, RID target) {
        return RID();
    }
};

// #############################################################################
// Part 3: Glow - Bloom effect
// #############################################################################

class GlowEffect : public Resource {
    XTU_GODOT_REGISTER_CLASS(GlowEffect, Resource)

public:
    enum BlendMode {
        BLEND_ADDITIVE,
        BLEND_SCREEN,
        BLEND_SOFTLIGHT,
        BLEND_REPLACE,
        BLEND_MIX
    };

private:
    bool m_enabled = false;
    BlendMode m_blend_mode = BLEND_SOFTLIGHT;
    float m_intensity = 0.8f;
    float m_strength = 1.0f;
    float m_mix = 0.01f;
    float m_bloom = 0.0f;
    float m_hdr_bleed_threshold = 1.0f;
    float m_hdr_bleed_scale = 2.0f;
    float m_hdr_luminance_cap = 12.0f;
    std::vector<float> m_levels = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    Ref<Texture2D> m_glow_map;

public:
    static StringName get_class_static() { return StringName("GlowEffect"); }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void set_blend_mode(BlendMode mode) { m_blend_mode = mode; }
    BlendMode get_blend_mode() const { return m_blend_mode; }

    void set_intensity(float intensity) { m_intensity = std::max(0.0f, intensity); }
    float get_intensity() const { return m_intensity; }

    void set_strength(float strength) { m_strength = std::max(0.0f, strength); }
    float get_strength() const { return m_strength; }

    void set_bloom(float bloom) { m_bloom = bloom; }
    float get_bloom() const { return m_bloom; }

    void set_hdr_bleed_threshold(float threshold) { m_hdr_bleed_threshold = threshold; }
    float get_hdr_bleed_threshold() const { return m_hdr_bleed_threshold; }

    void set_level(int idx, float value) {
        if (idx >= 0 && idx < static_cast<int>(m_levels.size())) {
            m_levels[idx] = value;
        }
    }

    float get_level(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_levels.size()) ? m_levels[idx] : 1.0f;
    }

    RID render(RID source, RID target, const vec2i& size) {
        return RID();
    }
};

// #############################################################################
// Part 4: DepthOfField - Depth of field effect
// #############################################################################

class DepthOfFieldEffect : public Resource {
    XTU_GODOT_REGISTER_CLASS(DepthOfFieldEffect, Resource)

public:
    enum Quality {
        QUALITY_LOW,
        QUALITY_MEDIUM,
        QUALITY_HIGH
    };

    enum BokehShape {
        BOKEH_CIRCLE,
        BOKEH_HEXAGON,
        BOKEH_OCTAGON
    };

private:
    bool m_enabled = false;
    Quality m_quality = QUALITY_MEDIUM;
    BokehShape m_bokeh_shape = BOKEH_CIRCLE;
    float m_focus_distance = 10.0f;
    float m_focus_range = 5.0f;
    float m_blur_amount = 0.1f;
    float m_bokeh_bias = 1.0f;
    bool m_use_physical_camera = true;

public:
    static StringName get_class_static() { return StringName("DepthOfFieldEffect"); }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void set_focus_distance(float distance) { m_focus_distance = distance; }
    float get_focus_distance() const { return m_focus_distance; }

    void set_focus_range(float range) { m_focus_range = range; }
    float get_focus_range() const { return m_focus_range; }

    void set_blur_amount(float amount) { m_blur_amount = amount; }
    float get_blur_amount() const { return m_blur_amount; }

    RID render(RID source, RID depth, RID target, const vec2i& size) {
        return RID();
    }
};

// #############################################################################
// Part 5: MotionBlur - Motion blur effect
// #############################################################################

class MotionBlurEffect : public Resource {
    XTU_GODOT_REGISTER_CLASS(MotionBlurEffect, Resource)

private:
    bool m_enabled = false;
    float m_intensity = 0.5f;
    float m_max_samples = 32.0f;
    bool m_use_camera_motion = true;
    bool m_use_object_motion = true;

public:
    static StringName get_class_static() { return StringName("MotionBlurEffect"); }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void set_intensity(float intensity) { m_intensity = std::clamp(intensity, 0.0f, 1.0f); }
    float get_intensity() const { return m_intensity; }

    RID render(RID source, RID velocity, RID depth, RID target, const vec2i& size) {
        return RID();
    }
};

// #############################################################################
// Part 6: Tonemap - Tone mapping
// #############################################################################

class TonemapEffect : public Resource {
    XTU_GODOT_REGISTER_CLASS(TonemapEffect, Resource)

public:
    enum Mode {
        MODE_LINEAR,
        MODE_REINHARDT,
        MODE_FILMIC,
        MODE_ACES,
        MODE_AGX
    };

private:
    bool m_enabled = true;
    Mode m_mode = MODE_ACES;
    float m_exposure = 1.0f;
    float m_white = 1.0f;
    bool m_auto_exposure_enabled = false;
    float m_auto_exposure_min_luma = 0.05f;
    float m_auto_exposure_max_luma = 8.0f;
    float m_auto_exposure_speed = 0.5f;
    float m_auto_exposure_scale = 0.4f;

public:
    static StringName get_class_static() { return StringName("TonemapEffect"); }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void set_mode(Mode mode) { m_mode = mode; }
    Mode get_mode() const { return m_mode; }

    void set_exposure(float exposure) { m_exposure = std::max(0.0f, exposure); }
    float get_exposure() const { return m_exposure; }

    void set_white(float white) { m_white = std::max(0.0f, white); }
    float get_white() const { return m_white; }

    void set_auto_exposure_enabled(bool enabled) { m_auto_exposure_enabled = enabled; }
    bool is_auto_exposure_enabled() const { return m_auto_exposure_enabled; }

    RID render(RID source, RID target, float delta) {
        return RID();
    }
};

// #############################################################################
// Part 7: ColorCorrection - Color grading
// #############################################################################

class ColorCorrectionEffect : public Resource {
    XTU_GODOT_REGISTER_CLASS(ColorCorrectionEffect, Resource)

private:
    bool m_enabled = false;
    float m_brightness = 1.0f;
    float m_contrast = 1.0f;
    float m_saturation = 1.0f;
    Ref<Texture2D> m_lut;
    float m_lut_strength = 1.0f;
    vec4f m_color_multiplier = {1, 1, 1, 1};
    vec4f m_color_offset = {0, 0, 0, 0};

public:
    static StringName get_class_static() { return StringName("ColorCorrectionEffect"); }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void set_brightness(float brightness) { m_brightness = brightness; }
    float get_brightness() const { return m_brightness; }

    void set_contrast(float contrast) { m_contrast = contrast; }
    float get_contrast() const { return m_contrast; }

    void set_saturation(float saturation) { m_saturation = saturation; }
    float get_saturation() const { return m_saturation; }

    void set_lut(const Ref<Texture2D>& lut) { m_lut = lut; }
    Ref<Texture2D> get_lut() const { return m_lut; }

    void set_lut_strength(float strength) { m_lut_strength = std::clamp(strength, 0.0f, 1.0f); }
    float get_lut_strength() const { return m_lut_strength; }

    RID render(RID source, RID target) {
        return RID();
    }
};

// #############################################################################
// Part 8: PostProcessStack - Combined post-processing pipeline
// #############################################################################

class PostProcessStack : public Resource {
    XTU_GODOT_REGISTER_CLASS(PostProcessStack, Resource)

private:
    Ref<SSAOEffect> m_ssao;
    Ref<SSILEffect> m_ssil;
    Ref<GlowEffect> m_glow;
    Ref<DepthOfFieldEffect> m_dof;
    Ref<MotionBlurEffect> m_motion_blur;
    Ref<TonemapEffect> m_tonemap;
    Ref<ColorCorrectionEffect> m_color_correction;

    std::vector<RID> m_temp_textures;
    int m_width = 0;
    int m_height = 0;

public:
    static StringName get_class_static() { return StringName("PostProcessStack"); }

    PostProcessStack() {
        m_ssao.instance();
        m_ssil.instance();
        m_glow.instance();
        m_dof.instance();
        m_motion_blur.instance();
        m_tonemap.instance();
        m_color_correction.instance();
    }

    Ref<SSAOEffect> get_ssao() { return m_ssao; }
    Ref<SSILEffect> get_ssil() { return m_ssil; }
    Ref<GlowEffect> get_glow() { return m_glow; }
    Ref<DepthOfFieldEffect> get_dof() { return m_dof; }
    Ref<MotionBlurEffect> get_motion_blur() { return m_motion_blur; }
    Ref<TonemapEffect> get_tonemap() { return m_tonemap; }
    Ref<ColorCorrectionEffect> get_color_correction() { return m_color_correction; }

    void resize(int width, int height) {
        m_width = width;
        m_height = height;
        // Recreate temporary textures
        for (RID& tex : m_temp_textures) {
            RenderingServer::get_singleton()->free_rid(tex);
        }
        m_temp_textures.clear();
    }

    RID render(RID source_color, RID source_depth, RID source_normal, RID source_velocity, RID target, float delta) {
        if (!target) return RID();

        RID current_source = source_color;
        RID current_target = target;

        // SSAO
        if (m_ssao->is_enabled()) {
            RID ao_target = get_temp_texture(m_width, m_height, RenderingServer::TEXTURE_FORMAT_R8);
            m_ssao->render(source_depth, source_normal, ao_target);
            // Composite AO onto source
            // ...
        }

        // SSIL
        if (m_ssil->is_enabled()) {
            // ...
        }

        // Motion Blur (before tonemap for better quality)
        if (m_motion_blur->is_enabled()) {
            // ...
        }

        // Depth of Field
        if (m_dof->is_enabled()) {
            // ...
        }

        // Glow / Bloom
        if (m_glow->is_enabled()) {
            // ...
        }

        // Tonemap
        if (m_tonemap->is_enabled()) {
            m_tonemap->render(current_source, current_target, delta);
            std::swap(current_source, current_target);
        }

        // Color Correction (always last)
        if (m_color_correction->is_enabled()) {
            m_color_correction->render(current_source, current_target);
            std::swap(current_source, current_target);
        }

        return current_target;
    }

private:
    RID get_temp_texture(int width, int height, RenderingServer::TextureFormat format) {
        RID tex = RenderingServer::get_singleton()->texture_create();
        // Create texture with appropriate size and format
        m_temp_textures.push_back(tex);
        return tex;
    }
};

} // namespace rendering

// Bring into main namespace
using rendering::SSAOEffect;
using rendering::SSILEffect;
using rendering::GlowEffect;
using rendering::DepthOfFieldEffect;
using rendering::MotionBlurEffect;
using rendering::TonemapEffect;
using rendering::ColorCorrectionEffect;
using rendering::PostProcessStack;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XPOST_EFFECTS_HPP