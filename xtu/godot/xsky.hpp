// include/xtu/godot/xsky.hpp
// xtensor-unified - Sky and Environment system for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XSKY_HPP
#define XTU_GODOT_XSKY_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
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
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/godot/xlighting.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"
#include "xtu/graphics/xmaterial.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class WorldEnvironment;
class Environment;
class Sky;
class SkyMaterial;
class ProceduralSkyMaterial;
class PhysicalSkyMaterial;
class PanoramaSkyMaterial;

// #############################################################################
// Environment tone mapper types
// #############################################################################
enum class EnvironmentToneMapper : uint8_t {
    TONE_MAPPER_LINEAR = 0,
    TONE_MAPPER_REINHARDT = 1,
    TONE_MAPPER_FILMIC = 2,
    TONE_MAPPER_ACES = 3,
    TONE_MAPPER_AGX = 4
};

// #############################################################################
// Environment glow blend modes
// #############################################################################
enum class EnvironmentGlowBlendMode : uint8_t {
    GLOW_BLEND_MODE_ADDITIVE = 0,
    GLOW_BLEND_MODE_SCREEN = 1,
    GLOW_BLEND_MODE_SOFTLIGHT = 2,
    GLOW_BLEND_MODE_REPLACE = 3,
    GLOW_BLEND_MODE_MIX = 4
};

// #############################################################################
// Environment fog modes
// #############################################################################
enum class EnvironmentFogMode : uint8_t {
    FOG_MODE_EXPONENTIAL = 0,
    FOG_MODE_EXPONENTIAL_SQUARED = 1,
    FOG_MODE_LINEAR = 2,
    FOG_MODE_DEPTH = 3
};

// #############################################################################
// Environment SSAO quality levels
// #############################################################################
enum class EnvironmentSSAOQuality : uint8_t {
    SSAO_QUALITY_VERY_LOW = 0,
    SSAO_QUALITY_LOW = 1,
    SSAO_QUALITY_MEDIUM = 2,
    SSAO_QUALITY_HIGH = 3,
    SSAO_QUALITY_ULTRA = 4
};

// #############################################################################
// Environment SDFGI quality levels
// #############################################################################
enum class EnvironmentSDFGIQuality : uint8_t {
    SDFGI_QUALITY_LOW = 0,
    SDFGI_QUALITY_MEDIUM = 1,
    SDFGI_QUALITY_HIGH = 2
};

// #############################################################################
// Sky mode types
// #############################################################################
enum class SkyMode : uint8_t {
    SKY_MODE_AUTOMATIC = 0,
    SKY_MODE_CUSTOM = 1,
    SKY_MODE_CLEAR_COLOR = 2
};

// #############################################################################
// Sky radiance size
// #############################################################################
enum class SkyRadianceSize : uint8_t {
    RADIANCE_SIZE_32 = 0,
    RADIANCE_SIZE_64 = 1,
    RADIANCE_SIZE_128 = 2,
    RADIANCE_SIZE_256 = 3,
    RADIANCE_SIZE_512 = 4,
    RADIANCE_SIZE_1024 = 5,
    RADIANCE_SIZE_2048 = 6
};

// #############################################################################
// Environment - Rendering environment settings resource
// #############################################################################
class Environment : public Resource {
    XTU_GODOT_REGISTER_CLASS(Environment, Resource)

public:
    // Background settings
    enum BGMode : uint8_t {
        BG_MODE_CLEAR_COLOR = 0,
        BG_MODE_COLOR = 1,
        BG_MODE_SKY = 2,
        BG_MODE_CANVAS = 3,
        BG_MODE_KEEP = 4,
        BG_MODE_CAMERA_FEED = 5
    };

private:
    // Background
    BGMode m_background_mode = BG_MODE_CLEAR_COLOR;
    Color m_background_color = {0, 0, 0, 1};
    Ref<Sky> m_background_sky;
    float m_background_energy_multiplier = 1.0f;
    float m_background_intensity = 1.0f;
    int m_background_canvas_max_layer = 0;

    // Ambient light
    Color m_ambient_light_color = {0, 0, 0, 1};
    float m_ambient_light_energy = 1.0f;
    float m_ambient_light_sky_contribution = 1.0f;
    Color m_ambient_light_source = {0, 0, 0, 1};
    Ref<Sky> m_ambient_light_sky;

    // Reflected light
    Color m_reflected_light_source = {0, 0, 0, 1};

    // Tonemap
    EnvironmentToneMapper m_tonemap_mode = EnvironmentToneMapper::TONE_MAPPER_LINEAR;
    float m_tonemap_exposure = 1.0f;
    float m_tonemap_white = 1.0f;

    // Auto exposure
    bool m_auto_exposure_enabled = false;
    float m_auto_exposure_scale = 0.4f;
    float m_auto_exposure_speed = 0.5f;
    float m_auto_exposure_min_luma = 0.05f;
    float m_auto_exposure_max_luma = 8.0f;

    // Glow
    bool m_glow_enabled = false;
    std::vector<float> m_glow_levels = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float m_glow_intensity = 0.8f;
    float m_glow_strength = 1.0f;
    float m_glow_mix = 0.01f;
    float m_glow_bloom = 0.0f;
    EnvironmentGlowBlendMode m_glow_blend_mode = EnvironmentGlowBlendMode::GLOW_BLEND_MODE_SOFTLIGHT;
    float m_glow_hdr_bleed_threshold = 1.0f;
    float m_glow_hdr_bleed_scale = 2.0f;
    float m_glow_hdr_luminance_cap = 12.0f;
    float m_glow_map_strength = 0.8f;
    Ref<Texture2D> m_glow_map;

    // Fog
    bool m_fog_enabled = false;
    EnvironmentFogMode m_fog_mode = EnvironmentFogMode::FOG_MODE_EXPONENTIAL;
    Color m_fog_light_color = {0.5f, 0.6f, 0.7f, 1.0f};
    float m_fog_light_energy = 1.0f;
    float m_fog_sun_scatter = 0.0f;
    float m_fog_density = 0.01f;
    float m_fog_height = 0.0f;
    float m_fog_height_density = 0.0f;
    float m_fog_aerial_perspective = 0.0f;
    Color m_fog_sky_affect = {1, 1, 1, 1};
    float m_fog_depth_begin = 10.0f;
    float m_fog_depth_end = 100.0f;
    float m_fog_depth_curve = 1.0f;
    bool m_fog_transmit_enabled = false;
    float m_fog_transmit_curve = 1.0f;

    // Volumetric fog
    bool m_volumetric_fog_enabled = false;
    float m_volumetric_fog_density = 0.05f;
    Color m_volumetric_fog_albedo = {1, 1, 1, 1};
    Color m_volumetric_fog_emission = {0, 0, 0, 1};
    float m_volumetric_fog_emission_energy = 1.0f;
    float m_volumetric_fog_anisotropy = 0.2f;
    float m_volumetric_fog_length = 64.0f;
    float m_volumetric_fog_detail_spread = 2.0f;
    float m_volumetric_fog_gi_inject = 1.0f;
    bool m_volumetric_fog_temporal_reprojection = true;
    float m_volumetric_fog_temporal_reprojection_amount = 0.9f;
    float m_volumetric_fog_ambient_inject = 0.0f;
    Ref<Texture3D> m_volumetric_fog_density_texture;

    // SSAO
    bool m_ssao_enabled = false;
    float m_ssao_radius = 1.0f;
    float m_ssao_intensity = 2.0f;
    float m_ssao_power = 1.5f;
    float m_ssao_detail = 0.5f;
    float m_ssao_horizon = 0.06f;
    float m_ssao_sharpness = 0.98f;
    float m_ssao_light_affect = 0.0f;
    float m_ssao_ao_channel_affect = 0.0f;
    EnvironmentSSAOQuality m_ssao_quality = EnvironmentSSAOQuality::SSAO_QUALITY_MEDIUM;
    bool m_ssao_half_size = true;
    Color m_ssao_color = {0, 0, 0, 1};
    bool m_ssao_blur = true;

    // SSIL
    bool m_ssil_enabled = false;
    float m_ssil_radius = 5.0f;
    float m_ssil_intensity = 1.0f;
    float m_ssil_sharpness = 0.98f;
    float m_ssil_normal_rejection = 1.0f;
    EnvironmentSSAOQuality m_ssil_quality = EnvironmentSSAOQuality::SSAO_QUALITY_MEDIUM;
    bool m_ssil_half_size = true;

    // SDFGI
    bool m_sdfgi_enabled = false;
    int m_sdfgi_cascades = 4;
    float m_sdfgi_min_cell_size = 0.2f;
    EnvironmentSDFGIQuality m_sdfgi_quality = EnvironmentSDFGIQuality::SDFGI_QUALITY_MEDIUM;
    bool m_sdfgi_use_occlusion = true;
    float m_sdfgi_bounce_feedback = 0.5f;
    bool m_sdfgi_read_sky_light = true;
    float m_sdfgi_energy = 1.0f;
    float m_sdfgi_normal_bias = 1.1f;
    float m_sdfgi_probe_bias = 1.1f;
    bool m_sdfgi_y_scale = true;

    // Adjustments
    bool m_adjustments_enabled = false;
    float m_adjustments_brightness = 1.0f;
    float m_adjustments_contrast = 1.0f;
    float m_adjustments_saturation = 1.0f;
    Ref<Texture2D> m_adjustments_color_correction;

public:
    static StringName get_class_static() { return StringName("Environment"); }

    void set_background_mode(BGMode mode) { m_background_mode = mode; }
    BGMode get_background_mode() const { return m_background_mode; }

    void set_background_color(const Color& color) { m_background_color = color; }
    Color get_background_color() const { return m_background_color; }

    void set_sky(const Ref<Sky>& sky) { m_background_sky = sky; }
    Ref<Sky> get_sky() const { return m_background_sky; }

    void set_ambient_light_color(const Color& color) { m_ambient_light_color = color; }
    Color get_ambient_light_color() const { return m_ambient_light_color; }

    void set_ambient_light_energy(float energy) { m_ambient_light_energy = energy; }
    float get_ambient_light_energy() const { return m_ambient_light_energy; }

    void set_ambient_light_sky_contribution(float contrib) { m_ambient_light_sky_contribution = contrib; }
    float get_ambient_light_sky_contribution() const { return m_ambient_light_sky_contribution; }

    void set_tonemapper(EnvironmentToneMapper mapper) { m_tonemap_mode = mapper; }
    EnvironmentToneMapper get_tonemapper() const { return m_tonemap_mode; }

    void set_tonemap_exposure(float exposure) { m_tonemap_exposure = exposure; }
    float get_tonemap_exposure() const { return m_tonemap_exposure; }

    void set_tonemap_white(float white) { m_tonemap_white = white; }
    float get_tonemap_white() const { return m_tonemap_white; }

    void set_glow_enabled(bool enabled) { m_glow_enabled = enabled; }
    bool is_glow_enabled() const { return m_glow_enabled; }

    void set_glow_intensity(float intensity) { m_glow_intensity = intensity; }
    float get_glow_intensity() const { return m_glow_intensity; }

    void set_glow_strength(float strength) { m_glow_strength = strength; }
    float get_glow_strength() const { return m_glow_strength; }

    void set_glow_blend_mode(EnvironmentGlowBlendMode mode) { m_glow_blend_mode = mode; }
    EnvironmentGlowBlendMode get_glow_blend_mode() const { return m_glow_blend_mode; }

    void set_fog_enabled(bool enabled) { m_fog_enabled = enabled; }
    bool is_fog_enabled() const { return m_fog_enabled; }

    void set_fog_mode(EnvironmentFogMode mode) { m_fog_mode = mode; }
    EnvironmentFogMode get_fog_mode() const { return m_fog_mode; }

    void set_fog_light_color(const Color& color) { m_fog_light_color = color; }
    Color get_fog_light_color() const { return m_fog_light_color; }

    void set_fog_density(float density) { m_fog_density = density; }
    float get_fog_density() const { return m_fog_density; }

    void set_fog_height(float height) { m_fog_height = height; }
    float get_fog_height() const { return m_fog_height; }

    void set_fog_depth_begin(float begin) { m_fog_depth_begin = begin; }
    float get_fog_depth_begin() const { return m_fog_depth_begin; }

    void set_fog_depth_end(float end) { m_fog_depth_end = end; }
    float get_fog_depth_end() const { return m_fog_depth_end; }

    void set_volumetric_fog_enabled(bool enabled) { m_volumetric_fog_enabled = enabled; }
    bool is_volumetric_fog_enabled() const { return m_volumetric_fog_enabled; }

    void set_volumetric_fog_density(float density) { m_volumetric_fog_density = density; }
    float get_volumetric_fog_density() const { return m_volumetric_fog_density; }

    void set_volumetric_fog_albedo(const Color& albedo) { m_volumetric_fog_albedo = albedo; }
    Color get_volumetric_fog_albedo() const { return m_volumetric_fog_albedo; }

    void set_ssao_enabled(bool enabled) { m_ssao_enabled = enabled; }
    bool is_ssao_enabled() const { return m_ssao_enabled; }

    void set_ssao_radius(float radius) { m_ssao_radius = radius; }
    float get_ssao_radius() const { return m_ssao_radius; }

    void set_ssao_intensity(float intensity) { m_ssao_intensity = intensity; }
    float get_ssao_intensity() const { return m_ssao_intensity; }

    void set_ssao_quality(EnvironmentSSAOQuality quality) { m_ssao_quality = quality; }
    EnvironmentSSAOQuality get_ssao_quality() const { return m_ssao_quality; }

    void set_ssil_enabled(bool enabled) { m_ssil_enabled = enabled; }
    bool is_ssil_enabled() const { return m_ssil_enabled; }

    void set_sdfgi_enabled(bool enabled) { m_sdfgi_enabled = enabled; }
    bool is_sdfgi_enabled() const { return m_sdfgi_enabled; }

    void set_sdfgi_cascades(int cascades) { m_sdfgi_cascades = cascades; }
    int get_sdfgi_cascades() const { return m_sdfgi_cascades; }

    void set_sdfgi_energy(float energy) { m_sdfgi_energy = energy; }
    float get_sdfgi_energy() const { return m_sdfgi_energy; }

    void set_adjustments_brightness(float brightness) { m_adjustments_brightness = brightness; }
    void set_adjustments_contrast(float contrast) { m_adjustments_contrast = contrast; }
    void set_adjustments_saturation(float saturation) { m_adjustments_saturation = saturation; }
};

// #############################################################################
// SkyMaterial - Base class for sky materials
// #############################################################################
class SkyMaterial : public Material {
    XTU_GODOT_REGISTER_CLASS(SkyMaterial, Material)

public:
    static StringName get_class_static() { return StringName("SkyMaterial"); }
};

// #############################################################################
// ProceduralSkyMaterial - Generated sky with sun and clouds
// #############################################################################
class ProceduralSkyMaterial : public SkyMaterial {
    XTU_GODOT_REGISTER_CLASS(ProceduralSkyMaterial, SkyMaterial)

private:
    Color m_sky_top_color = {0.385f, 0.454f, 0.55f, 1.0f};
    Color m_sky_horizon_color = {0.646f, 0.656f, 0.67f, 1.0f};
    float m_sky_curve = 0.15f;
    float m_sky_energy_multiplier = 1.0f;
    Color m_ground_bottom_color = {0.2f, 0.169f, 0.133f, 1.0f};
    Color m_ground_horizon_color = {0.329f, 0.318f, 0.298f, 1.0f};
    float m_ground_curve = 0.02f;
    float m_ground_energy_multiplier = 1.0f;
    float m_sun_angle_max = 30.0f;
    float m_sun_curve = 0.15f;
    bool m_use_debanding = true;
    Ref<Texture2D> m_cloud_texture;
    Ref<Texture2D> m_cloud_noise_texture;
    Color m_cloud_color = {1, 1, 1, 1};
    float m_cloud_coverage = 0.5f;
    float m_cloud_speed = 0.01f;
    float m_cloud_height = 1000.0f;

public:
    static StringName get_class_static() { return StringName("ProceduralSkyMaterial"); }

    void set_sky_top_color(const Color& color) { m_sky_top_color = color; }
    Color get_sky_top_color() const { return m_sky_top_color; }

    void set_sky_horizon_color(const Color& color) { m_sky_horizon_color = color; }
    Color get_sky_horizon_color() const { return m_sky_horizon_color; }

    void set_ground_bottom_color(const Color& color) { m_ground_bottom_color = color; }
    Color get_ground_bottom_color() const { return m_ground_bottom_color; }

    void set_sun_angle_max(float angle) { m_sun_angle_max = angle; }
    float get_sun_angle_max() const { return m_sun_angle_max; }

    void set_cloud_coverage(float coverage) { m_cloud_coverage = coverage; }
    float get_cloud_coverage() const { return m_cloud_coverage; }
};

// #############################################################################
// PhysicalSkyMaterial - Physically-based atmosphere scattering
// #############################################################################
class PhysicalSkyMaterial : public SkyMaterial {
    XTU_GODOT_REGISTER_CLASS(PhysicalSkyMaterial, SkyMaterial)

private:
    float m_rayleigh_coefficient = 2.0f;
    Color m_rayleigh_color = {0.3f, 0.405f, 0.6f, 1.0f};
    float m_mie_coefficient = 0.005f;
    float m_mie_eccentricity = 0.8f;
    Color m_mie_color = {0.69f, 0.729f, 0.812f, 1.0f};
    float m_turbidity = 10.0f;
    float m_sun_disk_scale = 1.0f;
    Color m_ground_color = {0.1f, 0.07f, 0.034f, 1.0f};
    float m_energy_multiplier = 1.0f;
    bool m_use_debanding = true;
    float m_night_sky_energy = 0.1f;

public:
    static StringName get_class_static() { return StringName("PhysicalSkyMaterial"); }

    void set_rayleigh_coefficient(float coeff) { m_rayleigh_coefficient = coeff; }
    float get_rayleigh_coefficient() const { return m_rayleigh_coefficient; }

    void set_rayleigh_color(const Color& color) { m_rayleigh_color = color; }
    Color get_rayleigh_color() const { return m_rayleigh_color; }

    void set_mie_coefficient(float coeff) { m_mie_coefficient = coeff; }
    float get_mie_coefficient() const { return m_mie_coefficient; }

    void set_mie_eccentricity(float ecc) { m_mie_eccentricity = ecc; }
    float get_mie_eccentricity() const { return m_mie_eccentricity; }

    void set_mie_color(const Color& color) { m_mie_color = color; }
    Color get_mie_color() const { return m_mie_color; }

    void set_turbidity(float turbidity) { m_turbidity = turbidity; }
    float get_turbidity() const { return m_turbidity; }

    void set_sun_disk_scale(float scale) { m_sun_disk_scale = scale; }
    float get_sun_disk_scale() const { return m_sun_disk_scale; }

    void set_ground_color(const Color& color) { m_ground_color = color; }
    Color get_ground_color() const { return m_ground_color; }

    void set_energy_multiplier(float multiplier) { m_energy_multiplier = multiplier; }
    float get_energy_multiplier() const { return m_energy_multiplier; }
};

// #############################################################################
// PanoramaSkyMaterial - HDRI panoramic sky
// #############################################################################
class PanoramaSkyMaterial : public SkyMaterial {
    XTU_GODOT_REGISTER_CLASS(PanoramaSkyMaterial, SkyMaterial)

private:
    Ref<Texture2D> m_panorama_texture;
    float m_energy_multiplier = 1.0f;
    bool m_use_debanding = true;

public:
    static StringName get_class_static() { return StringName("PanoramaSkyMaterial"); }

    void set_panorama(const Ref<Texture2D>& texture) { m_panorama_texture = texture; }
    Ref<Texture2D> get_panorama() const { return m_panorama_texture; }

    void set_energy_multiplier(float multiplier) { m_energy_multiplier = multiplier; }
    float get_energy_multiplier() const { return m_energy_multiplier; }
};

// #############################################################################
// Sky - Sky resource
// #############################################################################
class Sky : public Resource {
    XTU_GODOT_REGISTER_CLASS(Sky, Resource)

public:
    enum RadianceSize : uint8_t {
        RADIANCE_SIZE_32 = 0,
        RADIANCE_SIZE_64 = 1,
        RADIANCE_SIZE_128 = 2,
        RADIANCE_SIZE_256 = 3,
        RADIANCE_SIZE_512 = 4,
        RADIANCE_SIZE_1024 = 5,
        RADIANCE_SIZE_2048 = 6
    };

    enum ProcessMode : uint8_t {
        PROCESS_MODE_AUTOMATIC = 0,
        PROCESS_MODE_QUALITY = 1,
        PROCESS_MODE_INCREMENTAL = 2,
        PROCESS_MODE_REALTIME = 3
    };

private:
    Ref<SkyMaterial> m_sky_material;
    RadianceSize m_radiance_size = RadianceSize::RADIANCE_SIZE_512;
    ProcessMode m_process_mode = ProcessMode::PROCESS_MODE_AUTOMATIC;
    float m_sky_rotation = 0.0f;

public:
    static StringName get_class_static() { return StringName("Sky"); }

    void set_sky_material(const Ref<SkyMaterial>& material) { m_sky_material = material; }
    Ref<SkyMaterial> get_sky_material() const { return m_sky_material; }

    void set_radiance_size(RadianceSize size) { m_radiance_size = size; }
    RadianceSize get_radiance_size() const { return m_radiance_size; }

    void set_process_mode(ProcessMode mode) { m_process_mode = mode; }
    ProcessMode get_process_mode() const { return m_process_mode; }

    void set_sky_rotation(float rotation) { m_sky_rotation = rotation; }
    float get_sky_rotation() const { return m_sky_rotation; }
};

// #############################################################################
// WorldEnvironment - Environment node for scene
// #############################################################################
class WorldEnvironment : public Node {
    XTU_GODOT_REGISTER_CLASS(WorldEnvironment, Node)

private:
    Ref<Environment> m_environment;
    Ref<CameraAttributes> m_camera_attributes;

public:
    static StringName get_class_static() { return StringName("WorldEnvironment"); }

    void set_environment(const Ref<Environment>& env) { m_environment = env; }
    Ref<Environment> get_environment() const { return m_environment; }

    void set_camera_attributes(const Ref<CameraAttributes>& attr) { m_camera_attributes = attr; }
    Ref<CameraAttributes> get_camera_attributes() const { return m_camera_attributes; }

    void _enter_tree() override {
        // Register environment with RenderingServer
    }

    void _exit_tree() override {
        // Unregister environment
    }
};

// #############################################################################
// CameraAttributes - Camera-specific rendering attributes
// #############################################################################
class CameraAttributes : public Resource {
    XTU_GODOT_REGISTER_CLASS(CameraAttributes, Resource)

public:
    static StringName get_class_static() { return StringName("CameraAttributes"); }
};

// #############################################################################
// CameraAttributesPractical - Physical camera settings
// #############################################################################
class CameraAttributesPractical : public CameraAttributes {
    XTU_GODOT_REGISTER_CLASS(CameraAttributesPractical, CameraAttributes)

private:
    float m_exposure_multiplier = 1.0f;
    float m_exposure_sensitivity = 1.0f;
    float m_f_stop = 8.0f;
    float m_shutter_speed = 100.0f;
    float m_iso = 100.0f;
    bool m_auto_exposure = true;
    float m_auto_exposure_min = 0.1f;
    float m_auto_exposure_max = 10.0f;

public:
    static StringName get_class_static() { return StringName("CameraAttributesPractical"); }

    void set_f_stop(float f_stop) { m_f_stop = f_stop; }
    float get_f_stop() const { return m_f_stop; }

    void set_shutter_speed(float speed) { m_shutter_speed = speed; }
    float get_shutter_speed() const { return m_shutter_speed; }

    void set_iso(float iso) { m_iso = iso; }
    float get_iso() const { return m_iso; }

    void set_auto_exposure(bool enabled) { m_auto_exposure = enabled; }
    bool get_auto_exposure() const { return m_auto_exposure; }
};

// #############################################################################
// CameraAttributesPhysical - Physically-based camera
// #############################################################################
class CameraAttributesPhysical : public CameraAttributes {
    XTU_GODOT_REGISTER_CLASS(CameraAttributesPhysical, CameraAttributes)

private:
    float m_focal_length = 35.0f;
    float m_focus_distance = 10.0f;
    float m_f_stop = 2.8f;
    int m_blade_count = 6;
    float m_blade_rotation = 0.0f;
    float m_bokeh_bias = 1.0f;
    bool m_auto_exposure = false;

public:
    static StringName get_class_static() { return StringName("CameraAttributesPhysical"); }

    void set_focal_length(float length) { m_focal_length = length; }
    float get_focal_length() const { return m_focal_length; }

    void set_focus_distance(float distance) { m_focus_distance = distance; }
    float get_focus_distance() const { return m_focus_distance; }

    void set_f_stop(float f_stop) { m_f_stop = f_stop; }
    float get_f_stop() const { return m_f_stop; }
};

} // namespace godot

// Bring into main namespace
using godot::WorldEnvironment;
using godot::Environment;
using godot::Sky;
using godot::SkyMaterial;
using godot::ProceduralSkyMaterial;
using godot::PhysicalSkyMaterial;
using godot::PanoramaSkyMaterial;
using godot::CameraAttributes;
using godot::CameraAttributesPractical;
using godot::CameraAttributesPhysical;
using godot::EnvironmentToneMapper;
using godot::EnvironmentGlowBlendMode;
using godot::EnvironmentFogMode;
using godot::EnvironmentSSAOQuality;
using godot::EnvironmentSDFGIQuality;
using godot::SkyMode;
using godot::SkyRadianceSize;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XSKY_HPP