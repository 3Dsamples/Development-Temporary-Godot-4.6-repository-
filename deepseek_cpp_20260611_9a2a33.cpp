// lightmap_gi.cpp
#include "lightmap_gi.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>
#include <unordered_map>

namespace lighting {

// ============================================================================
// Lightmap baking helper (simplified CPU version for demonstration)
// In a real engine, this would be GPU compute with path tracing / radiosity.
// ============================================================================
struct BakeWork {
    std::vector<float> lightmap_data; // RGBA or RGB (depending on directional)
    int width, height;
    std::atomic<float> progress{0.0f};
    std::atomic<bool> cancel{false};
    std::thread worker;
};

// ============================================================================
// Implementation
// ============================================================================
struct LightmapGI::Impl {
    // Baking parameters
    int bake_quality = 1;          // 0=low,1=medium,2=high,3=ultra
    bool bake_denoiser = true;
    int bake_max_bounces = 4;
    int bake_light_resolution = 4; // texels per world unit
    int atlas_size = 4096;
    bool use_compression = true;
    int texture_format = 2;        // BC6H for HDR

    // Runtime settings
    int gi_mode = 1;               // static by default
    float energy = 1.0f;
    float bias = 0.0f;
    float normal_bias = 0.0f;
    bool interior = false;
    bool directional_enabled = false;
    float directional_half_lobe = 0.5f;
    bool probe_capture_enabled = false;
    bool use_mipmaps = true;

    // Baking state
    std::unique_ptr<BakeWork> bake_work;
    bool is_baking_flag = false;
    bool dirty = true;

    // Render server handles
    int64_t lightmap_texture = -1;
    int64_t dir_texture = -1;
    int64_t instance_rid = -1;

    ~Impl() {
        if (bake_work && bake_work->worker.joinable()) {
            bake_work->cancel = true;
            bake_work->worker.join();
        }
    }

    void start_bake();
    void run_bake();
    void apply_lightmap_to_render_server();
};

LightmapGI::LightmapGI() : pimpl(std::make_unique<Impl>()) {}
LightmapGI::~LightmapGI() = default;

void LightmapGI::set_bake_quality(int quality) { pimpl->bake_quality = std::clamp(quality, 0, 3); }
int LightmapGI::get_bake_quality() const { return pimpl->bake_quality; }
void LightmapGI::set_bake_denoiser(bool enable) { pimpl->bake_denoiser = enable; }
bool LightmapGI::is_bake_denoiser_enabled() const { return pimpl->bake_denoiser; }
void LightmapGI::set_bake_max_bounces(int bounces) { pimpl->bake_max_bounces = std::max(1, bounces); }
int LightmapGI::get_bake_max_bounces() const { return pimpl->bake_max_bounces; }
void LightmapGI::set_bake_light_resolution(int resolution) { pimpl->bake_light_resolution = std::max(1, resolution); }
int LightmapGI::get_bake_light_resolution() const { return pimpl->bake_light_resolution; }
void LightmapGI::set_atlas_size(int size) { pimpl->atlas_size = std::max(64, size); }
int LightmapGI::get_atlas_size() const { return pimpl->atlas_size; }
void LightmapGI::set_use_compression(bool enable) { pimpl->use_compression = enable; }
bool LightmapGI::is_compression_enabled() const { return pimpl->use_compression; }
void LightmapGI::set_texture_format(int format) { pimpl->texture_format = std::clamp(format, 0, 3); }
int LightmapGI::get_texture_format() const { return pimpl->texture_format; }
void LightmapGI::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
int LightmapGI::get_gi_mode() const { return pimpl->gi_mode; }
void LightmapGI::set_energy(float multiplier) { pimpl->energy = multiplier; }
float LightmapGI::get_energy() const { return pimpl->energy; }
void LightmapGI::set_bias(float bias) { pimpl->bias = bias; }
float LightmapGI::get_bias() const { return pimpl->bias; }
void LightmapGI::set_normal_bias(float bias) { pimpl->normal_bias = bias; }
float LightmapGI::get_normal_bias() const { return pimpl->normal_bias; }
void LightmapGI::set_interior(bool interior) { pimpl->interior = interior; }
bool LightmapGI::is_interior() const { return pimpl->interior; }
void LightmapGI::set_directional_enabled(bool enable) { pimpl->directional_enabled = enable; }
bool LightmapGI::is_directional_enabled() const { return pimpl->directional_enabled; }
void LightmapGI::set_directional_half_lobe(float angle) { pimpl->directional_half_lobe = std::clamp(angle, 0.0f, 1.0f); }
float LightmapGI::get_directional_half_lobe() const { return pimpl->directional_half_lobe; }
void LightmapGI::set_probe_capture_enabled(bool enable) { pimpl->probe_capture_enabled = enable; }
bool LightmapGI::is_probe_capture_enabled() const { return pimpl->probe_capture_enabled; }
void LightmapGI::set_use_mipmaps(bool enable) { pimpl->use_mipmaps = enable; }
bool LightmapGI::are_mipmaps_enabled() const { return pimpl->use_mipmaps; }

void LightmapGI::bake() {
    if (pimpl->is_baking_flag) return;
    pimpl->start_bake();
}
bool LightmapGI::is_baking() const { return pimpl->is_baking_flag; }
float LightmapGI::get_bake_progress() const {
    return pimpl->bake_work ? pimpl->bake_work->progress.load() : 0.0f;
}
void LightmapGI::cancel_bake() {
    if (pimpl->bake_work) pimpl->bake_work->cancel = true;
}
void LightmapGI::clear_lightmaps() {
    if (pimpl->is_baking_flag) cancel_bake();
    pimpl->dirty = true;
    // In real engine, free textures and inform render server
}

void LightmapGI::Impl::start_bake() {
    is_baking_flag = true;
    bake_work = std::make_unique<BakeWork>();
    bake_work->worker = std::thread([this]() { run_bake(); });
}

void LightmapGI::Impl::run_bake() {
    // Simulate baking (real engine would collect static meshes, compute texel positions,
    // gather lights, shoot rays, accumulate radiance)
    // For demonstration, we fill with dummy data.
    int atlas_pixels = atlas_size * atlas_size;
    int channels = directional_enabled ? 4 : 3;
    bake_work->lightmap_data.resize(atlas_pixels * channels);
    // Simulate progress
    for (int i = 0; i <= 100; ++i) {
        if (bake_work->cancel) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        bake_work->progress = i / 100.0f;
    }
    if (!bake_work->cancel) {
        // Fill with example color (warm)
        for (size_t j = 0; j < bake_work->lightmap_data.size(); ++j) {
            bake_work->lightmap_data[j] = (j % 3 == 0) ? 0.7f : 0.5f; // rough
        }
        // After baking, apply to render server
        apply_lightmap_to_render_server();
    }
    is_baking_flag = false;
    bake_work.reset();
}

void LightmapGI::Impl::apply_lightmap_to_render_server() {
    if (bake_work->lightmap_data.empty()) return;
    // Create or update lightmap texture in RenderingServer
    // For each static geometry instance, assign lightmap UVs and texture.
    // Also update directional lightmap if enabled.
    // In a real engine, this would set a global GI data resource.
    dirty = false;
}

void LightmapGI::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (pimpl->dirty && !pimpl->is_baking_flag) {
        // If no lightmap data, we might still need to clear previous
        // For now, nothing.
    }
    // Update runtime settings in render server (energy, bias, normal_bias, etc.)
}

void LightmapGI::process(double delta) {
    Node3D::process(delta);
    // Not used for lightmap, but could update runtime blending
}

} // namespace lighting