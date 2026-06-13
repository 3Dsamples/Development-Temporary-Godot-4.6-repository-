// light_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <vector>
#include <array>
#include <memory>
#include <variant>
#include <optional>

namespace lighting {

// ============================================================================
// Enums & forward declarations
// ============================================================================

enum class LightType : uint8_t {
    DIRECTIONAL,
    POINT,
    SPOT,
    RECTANGLE,
    DISC,
    SPHERE,
    LINE
};

enum class ShadowTechnique : uint8_t {
    PCF,
    PCSS,
    VSM,
    CHS
};

enum class IBLQuality : uint8_t {
    LOW,
    MEDIUM,
    HIGH,
    ULTRA
};

// Forward declares for advanced systems (defined in .cpp)
class ScreenSpaceReflections;
class ScreenSpaceGlobalIllumination;
class ScreenSpaceAmbientOcclusion;
class LightProbeGrid;
class ReflectionProbe;
class CubemapSkybox;
class PlanarReflection;
class DistanceFieldAO;
class RayTracedGI;
class RayTracedReflections;
class PathTracer;
class Denoiser;
class VoxelConeTracer;
class CascadedShadowMap;
class ShadowEvaluator;

// ============================================================================
// LightGeometry – area light description
// ============================================================================

struct LightGeometry {
    LightType type = LightType::POINT;
    double position[3] = {0.0, 0.0, 0.0};
    double direction[3] = {0.0, -1.0, 0.0};
    double up[3] = {0.0, 0.0, 1.0};
    double size_x = 1.0;
    double size_y = 1.0;
    double radius = 0.5;
    double length = 1.0;

    LightGeometry() = default;
    LightGeometry(LightType t, const double* pos, const double* dir = nullptr, const double* up_vec = nullptr);
    void sample_point(float rng1, float rng2, double* out_pos, double* out_normal, float& pdf) const;
};

// ============================================================================
// Light3D – main light node, supports all real-time techniques
// ============================================================================

class Light3D : public Node3D {
public:
    Light3D();
    ~Light3D();

    // ------------------------------------------------------------------------
    // Basic light parameters
    // ------------------------------------------------------------------------
    void set_light_type(LightType type);
    LightType get_light_type() const;
    void set_color(float r, float g, float b);
    void get_color(float& r, float& g, float& b) const;
    void set_intensity(float intensity);
    float get_intensity() const;
    void set_range(float range);
    float get_range() const;

    // ------------------------------------------------------------------------
    // Spot light specific
    // ------------------------------------------------------------------------
    void set_spot_angle(float degrees);
    float get_spot_angle() const;
    void set_spot_attenuation(float attenuation);
    float get_spot_attenuation() const;

    // ------------------------------------------------------------------------
    // Area light geometry (rectangle / disc / sphere)
    // ------------------------------------------------------------------------
    void set_area_geometry(const LightGeometry& geom);
    const LightGeometry& get_area_geometry() const;

    // ------------------------------------------------------------------------
    // Shadows – advanced techniques (CSM, PCSS, VSM, CHS)
    // ------------------------------------------------------------------------
    void set_shadow_enabled(bool enabled);
    bool is_shadow_enabled() const;
    void set_shadow_technique(ShadowTechnique tech);
    ShadowTechnique get_shadow_technique() const;
    void set_shadow_map_resolution(int resolution);
    int get_shadow_map_resolution() const;
    void set_csm_cascade_count(int count);
    int get_csm_cascade_count() const;
    void set_csm_split_lambda(float lambda);
    float get_csm_split_lambda() const;
    void set_pcss_light_size(float size);
    float get_pcss_light_size() const;
    void set_vsm_exponent(float exp);
    float get_vsm_exponent() const;

    // ------------------------------------------------------------------------
    // IBL (environment map)
    // ------------------------------------------------------------------------
    void set_environment_cubemap(const std::array<std::vector<float>, 6>& faces, int resolution, IBLQuality quality);
    void clear_environment();

    // ------------------------------------------------------------------------
    // Reflection probes (dynamic cubemap captures)
    // ------------------------------------------------------------------------
    void add_reflection_probe(const double* position, float update_rate_hz, int resolution);
    void remove_reflection_probe(int index);
    void update_reflection_probes(double delta_time, void* render_callback); // render_callback is function pointer

    // ------------------------------------------------------------------------
    // Light probes (irradiance volumes)
    // ------------------------------------------------------------------------
    void set_light_probe_grid(const double* bounds_min, const double* bounds_max, int res_x, int res_y, int res_z);
    void clear_light_probe_grid();
    void update_light_probes(); // recalc SH from scene

    // ------------------------------------------------------------------------
    // Distance Field Ambient Occlusion (DFAO)
    // ------------------------------------------------------------------------
    void set_dfao_enabled(bool enabled);
    bool is_dfao_enabled() const;
    void set_dfao_resolution(float cell_size);
    float get_dfao_resolution() const;

    // ------------------------------------------------------------------------
    // Ray-Traced Global Illumination (RTGI)
    // ------------------------------------------------------------------------
    void set_rtgi_enabled(bool enabled);
    bool is_rtgi_enabled() const;
    void set_rtgi_samples_per_pixel(int samples);
    int get_rtgi_samples_per_pixel() const;
    void set_rtgi_max_bounces(int bounces);
    int get_rtgi_max_bounces() const;

    // ------------------------------------------------------------------------
    // Ray-Traced Reflections
    // ------------------------------------------------------------------------
    void set_raytraced_reflections_enabled(bool enabled);
    bool is_raytraced_reflections_enabled() const;
    void set_raytraced_reflection_roughness_threshold(float threshold);
    float get_raytraced_reflection_roughness_threshold() const;

    // ------------------------------------------------------------------------
    // Path tracing (full global illumination)
    // ------------------------------------------------------------------------
    void set_path_tracing_enabled(bool enabled);
    bool is_path_tracing_enabled() const;
    void set_path_tracing_max_bounces(int bounces);
    int get_path_tracing_max_bounces() const;

    // ------------------------------------------------------------------------
    // Denoising (spatial‑temporal)
    // ------------------------------------------------------------------------
    void set_denoising_enabled(bool enabled);
    bool is_denoising_enabled() const;
    void set_denoiser_parameters(float spatial_sigma, float color_sigma, float depth_sigma);
    void get_denoiser_parameters(float& spatial_sigma, float& color_sigma, float& depth_sigma) const;

    // ------------------------------------------------------------------------
    // Voxel Cone Tracing
    // ------------------------------------------------------------------------
    void set_voxel_cone_tracing_enabled(bool enabled);
    bool is_voxel_cone_tracing_enabled() const;
    void set_voxel_grid_resolution(int resolution);
    int get_voxel_grid_resolution() const;

    // ------------------------------------------------------------------------
    // Screen‑space techniques (SSR, SSGI, SSAO)
    // ------------------------------------------------------------------------
    void set_ssr_enabled(bool enabled);
    bool is_ssr_enabled() const;
    void set_ssgi_enabled(bool enabled);
    bool is_ssgi_enabled() const;
    void set_ssao_enabled(bool enabled);
    bool is_ssao_enabled() const;
    void set_ssao_radius(float radius);
    float get_ssao_radius() const;

    // ------------------------------------------------------------------------
    // Planar reflections (mirror plane)
    // ------------------------------------------------------------------------
    void set_planar_reflection_plane(const double* normal, const double* point, int resolution);
    void clear_planar_reflection();

    // ------------------------------------------------------------------------
    // Skybox (cubemap)
    // ------------------------------------------------------------------------
    void set_skybox_cubemap(const std::array<std::vector<float>, 6>& faces, int resolution, float intensity);
    void clear_skybox();

    // ------------------------------------------------------------------------
    // Rendering & update (called by engine)
    // ------------------------------------------------------------------------
    void prepare_lighting_for_frame(const double* camera_position, const double* camera_forward, double delta_time);
    void render_gbuffer();          // fills internal G‑buffers (to be called by renderer)
    void compute_lighting(float* out_color, int width, int height); // final HDR output

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

// ============================================================================
// Helper: Precomputed irradiance cache (for light probes)
// ============================================================================

class IrradianceCache {
public:
    IrradianceCache(size_t max_entries = 10000);
    ~IrradianceCache();

    void insert(const double* world_pos, const double* normal, const float* irradiance);
    bool lookup(const double* world_pos, const double* normal, float* out_irradiance, double threshold = 0.1) const;
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

// ============================================================================
// Helper: Photon map (for caustics / full GI)
// ============================================================================

class PhotonMap {
public:
    PhotonMap(size_t max_photons = 500000);
    ~PhotonMap();

    void add_photon(const double* pos, const double* dir, const float* color, float power);
    void build_kdtree();   // call after adding photons
    void estimate_radiance(const double* point, const double* normal, float radius, float* out_radiance, int num_neighbors = 50) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting