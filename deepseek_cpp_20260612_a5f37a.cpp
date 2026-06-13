// Name : lighting enhancement
// File : scene/3d/lighting_techniques_ext.h 61 of 61
// Description : High-performance screen-space, baked, and ray-traced lighting techniques.
//               Implements SSR, SSGI, SSAO, lightmaps, light/reflection probes,
//               cubemap sky, planar reflections, DFAO, RTGI, ray-traced reflections,
//               path tracing, denoising, voxel cone tracing, CSM, VSM, PCSS.
#pragma once

#include "scene/3d/node_3d_ext.h"
#include "scene/3d/camera_3d_ext.h"
#include "scene/3d/light_3d_ext.h"
#include "servers/rendering_server.h"
#include "servers/physics_server_3d.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/templates/vector.h"

namespace lighting {

// ----------------------------------------------------------------------------
// Screen-Space Reflections (SSR)
// ----------------------------------------------------------------------------
class ScreenSpaceReflections {
public:
    void set_ray_steps(int p_steps);
    void set_binary_search_steps(int p_steps);
    void set_temporal_blend(float p_blend);
    void render(const RID &p_color, const RID &p_depth, const RID &p_normal,
                const Camera3DExt *p_camera, RID &out_reflections);
private:
    int ray_steps = 32;
    int binary_steps = 8;
    float temporal_blend = 0.95f;
};

// ----------------------------------------------------------------------------
// Screen-Space Global Illumination (SSGI)
// ----------------------------------------------------------------------------
class ScreenSpaceGlobalIllumination {
public:
    void set_num_samples(int p_samples);
    void set_radius(float p_radius);
    void render(const RID &p_color, const RID &p_depth, const RID &p_normal,
                const Camera3DExt *p_camera, RID &out_indirect);
private:
    int num_samples = 8;
    float radius = 1.0f;
};

// ----------------------------------------------------------------------------
// Screen-Space Ambient Occlusion (SSAO)
// ----------------------------------------------------------------------------
class ScreenSpaceAmbientOcclusion {
public:
    void set_radius(float p_radius);
    void set_intensity(float p_intensity);
    void set_samples(int p_samples);
    void render(const RID &p_depth, const RID &p_normal,
                const Camera3DExt *p_camera, RID &out_ao);
private:
    float radius = 0.5f;
    float intensity = 1.0f;
    int samples = 32;
};

// ----------------------------------------------------------------------------
// Lightmaps (baked GI)
// ----------------------------------------------------------------------------
class LightmapBaker {
public:
    void set_quality(int p_quality); // 0=low,1=medium,2=high
    void set_bounce_count(int p_bounces);
    void set_texel_per_unit(float p_texels);
    void bake(World3D *p_world, RID &out_lightmap_atlas);
    RID get_lightmap_atlas() const;
private:
    int quality = 1;
    int bounce_count = 2;
    float texel_per_unit = 64.0f;
    RID atlas_rid;
};

// ----------------------------------------------------------------------------
// Light Probe (irradiance volume)
// ----------------------------------------------------------------------------
class LightProbe {
public:
    void set_position(const Vector3 &p_pos);
    void set_influence_radius(float p_radius);
    void set_sh_coefficients(const Vector<float> &p_sh9); // 27 floats
    Vector<float> get_sh_coefficients() const;
    void update_from_scene(World3D *p_world);
    RID get_probe_rid() const;
private:
    RID probe_rid;
    Vector<float> sh_coeffs;
};

// ----------------------------------------------------------------------------
// Reflection Probe (cubemap capture)
// ----------------------------------------------------------------------------
class ReflectionProbe {
public:
    void set_position(const Vector3 &p_pos);
    void set_extents(const Vector3 &p_extents);
    void set_resolution(int p_res);
    void set_update_mode(int p_mode); // 0=static,1=dynamic,2=always
    void capture_async();
    RID get_cubemap_rid() const;
private:
    RID probe_rid;
    RID cubemap_rid;
    Vector3 extents;
    int resolution = 256;
};

// ----------------------------------------------------------------------------
// Cubemap Skybox
// ----------------------------------------------------------------------------
class CubemapSky {
public:
    void set_cubemap(const RID &p_cubemap);
    void set_intensity(float p_intensity);
    RID get_sky_rid() const;
private:
    RID sky_rid;
    float intensity = 1.0f;
};

// ----------------------------------------------------------------------------
// Planar Reflections
// ----------------------------------------------------------------------------
class PlanarReflection {
public:
    void set_plane(const Plane &p_plane);
    void set_resolution(int p_res);
    void render(const Camera3DExt *p_camera, World3D *p_world, RID &out_texture);
private:
    Plane plane;
    int resolution = 512;
    RID reflection_rid;
};

// ----------------------------------------------------------------------------
// Distance Field Ambient Occlusion (DFAO)
// ----------------------------------------------------------------------------
class DistanceFieldAO {
public:
    void build_from_world(World3D *p_world);
    void set_resolution(float p_cell_size);
    float compute_occlusion(const Vector3 &p_point, const Vector3 &p_normal) const;
    RID get_sdf_texture() const;
private:
    RID sdf_texture;
    float cell_size = 0.1f;
};

// ----------------------------------------------------------------------------
// Ray-Traced Global Illumination (RTGI)
// ----------------------------------------------------------------------------
class RayTracedGI {
public:
    void set_num_bounces(int p_bounces);
    void set_num_samples(int p_samples);
    void set_denoise(bool p_denoise);
    void render(const RID &p_color, const RID &p_depth, const RID &p_normal,
                const Camera3DExt *p_camera, World3D *p_world, RID &out_indirect);
private:
    int bounces = 1;
    int samples = 2;
    bool denoise = true;
};

// ----------------------------------------------------------------------------
// Ray-Traced Reflections
// ----------------------------------------------------------------------------
class RayTracedReflections {
public:
    void set_roughness_threshold(float p_threshold);
    void set_num_samples(int p_samples);
    void render(const RID &p_color, const RID &p_depth, const RID &p_normal,
                const Camera3DExt *p_camera, World3D *p_world, RID &out_reflections);
private:
    float roughness_threshold = 0.2f;
    int samples = 1;
};

// ----------------------------------------------------------------------------
// Path Tracer (full global illumination)
// ----------------------------------------------------------------------------
class PathTracer {
public:
    void set_max_bounces(int p_bounces);
    void set_num_samples(int p_samples);
    void set_resolution_scale(float p_scale);
    void render(const Camera3DExt *p_camera, World3D *p_world, RID &out_color);
private:
    int max_bounces = 4;
    int samples = 16;
    float resolution_scale = 0.5f;
};

// ----------------------------------------------------------------------------
// Denoiser (spatial-temporal)
// ----------------------------------------------------------------------------
class Denoiser {
public:
    void set_spatial_sigma(float p_sigma);
    void set_color_sigma(float p_sigma);
    void set_depth_sigma(float p_sigma);
    void denoise(const RID &p_color, const RID &p_depth, const RID &p_normal,
                 const RID &p_motion, RID &out_denoised);
private:
    float spatial_sigma = 1.5f;
    float color_sigma = 0.1f;
    float depth_sigma = 0.05f;
    RID history_buffer;
};

// ----------------------------------------------------------------------------
// Voxel Cone Tracing
// ----------------------------------------------------------------------------
class VoxelConeTracer {
public:
    void set_voxel_resolution(int p_res);
    void set_world_extents(const AABB &p_bounds);
    void build_from_world(World3D *p_world);
    void trace(const RID &p_color, const RID &p_depth, const RID &p_normal,
               const Camera3DExt *p_camera, RID &out_indirect);
private:
    int resolution = 128;
    AABB world_bounds;
    RID voxel_grid;
};

// ----------------------------------------------------------------------------
// Cascaded Shadow Maps (CSM)
// ----------------------------------------------------------------------------
class CascadedShadowMaps {
public:
    void set_num_cascades(int p_num);
    void set_split_lambda(float p_lambda);
    void set_resolution(int p_res);
    void update(const Light3DExt *p_light, const Camera3DExt *p_camera);
    RID get_shadow_atlas() const;
private:
    int num_cascades = 4;
    float split_lambda = 0.5f;
    int resolution = 2048;
    RID shadow_atlas;
};

// ----------------------------------------------------------------------------
// Percentage-Closer Soft Shadows (PCSS)
// ----------------------------------------------------------------------------
class PCSS {
public:
    void set_light_size(float p_size);
    void set_samples(int p_samples);
    float compute_shadow(const RID &p_shadow_map, const Vector3 &p_uv,
                         float p_receiver_depth, float p_light_size) const;
private:
    float light_size = 1.0f;
    int samples = 16;
};

// ----------------------------------------------------------------------------
// Variance Shadow Maps (VSM)
// ----------------------------------------------------------------------------
class VSM {
public:
    void set_exponent(float p_exp);
    float compute_shadow(const RID &p_depth_texture, const RID &p_depth2_texture,
                         const Vector3 &p_uv, float p_receiver_depth) const;
private:
    float exponent = 40.0f;
};

// ----------------------------------------------------------------------------
// High-level lighting system that orchestrates all techniques
// ----------------------------------------------------------------------------
class LightingTechniquesManager {
public:
    LightingTechniquesManager();
    ~LightingTechniquesManager();

    void set_screen_space_enabled(bool p_enabled);
    void set_ssr_enabled(bool p_enabled);
    void set_ssgi_enabled(bool p_enabled);
    void set_ssao_enabled(bool p_enabled);
    void set_lightmap_enabled(bool p_enabled);
    void set_reflection_probes_enabled(bool p_enabled);
    void set_rtgi_enabled(bool p_enabled);
    void set_raytraced_reflections_enabled(bool p_enabled);
    void set_path_tracing_enabled(bool p_enabled);
    void set_voxel_cone_tracing_enabled(bool p_enabled);
    void set_csm_enabled(bool p_enabled);

    void process_frame(const RID &p_color, const RID &p_depth, const RID &p_normal,
                       const RID &p_motion, const Camera3DExt *p_camera,
                       World3D *p_world, RID &out_final);

private:
    // Sub‑systems
    ScreenSpaceReflections ssr;
    ScreenSpaceGlobalIllumination ssgi;
    ScreenSpaceAmbientOcclusion ssao;
    LightmapBaker lightmap_baker;
    Vector<ReflectionProbe> reflection_probes;
    RayTracedGI rtgi;
    RayTracedReflections rt_reflections;
    PathTracer path_tracer;
    Denoiser denoiser;
    VoxelConeTracer vct;
    CascadedShadowMaps csm;
    PCSS pcss;
    VSM vsm;

    bool screen_space_enabled = true;
    bool ssr_enabled = true;
    bool ssgi_enabled = true;
    bool ssao_enabled = true;
    bool lightmap_enabled = false;
    bool reflection_probes_enabled = true;
    bool rtgi_enabled = false;
    bool raytraced_reflections_enabled = false;
    bool path_tracing_enabled = false;
    bool voxel_cone_tracing_enabled = false;
    bool csm_enabled = true;

    // Internal buffers
    RID temp_buffer;
    RID indirect_buffer;
    RID reflection_buffer;
    RID ao_buffer;
};

} // namespace lighting