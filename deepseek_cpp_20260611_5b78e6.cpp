// lightmap_probe.cpp
#include "lightmap_probe.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace lighting {

// ============================================================================
// Implementation
// ============================================================================
struct LightmapProbe::Impl {
    int capture_size = 128;      // cubemap face resolution
    double capture_extents[3] = {10.0, 10.0, 10.0}; // influence radius (half extents)
    float sh_coeffs[27] = {0};    // 3 bands, 9 coefficients per RGB (27 floats)
    int64_t specular_cubemap = -1;
    bool interior = false;
    float energy = 1.0f;

    bool dirty = true;
    int64_t probe_rid = -1;       // handle in rendering server

    ~Impl() {
        // free resources
    }
};

LightmapProbe::LightmapProbe() : pimpl(std::make_unique<Impl>()) {
    // default SH coefficients set to average gray
    for (int i = 0; i < 27; ++i) pimpl->sh_coeffs[i] = 0.0f;
    // l0 term (constant) set to 0.5
    for (int c = 0; c < 3; ++c) pimpl->sh_coeffs[c*9] = 0.5f;
}
LightmapProbe::~LightmapProbe() = default;

void LightmapProbe::set_capture_size(int resolution) {
    pimpl->capture_size = std::max(16, resolution);
    pimpl->dirty = true;
}
int LightmapProbe::get_capture_size() const { return pimpl->capture_size; }

void LightmapProbe::set_capture_extents(const double* extents) {
    memcpy(pimpl->capture_extents, extents, 3*sizeof(double));
    pimpl->dirty = true;
}
void LightmapProbe::get_capture_extents(double* out_extents) const {
    memcpy(out_extents, pimpl->capture_extents, 3*sizeof(double));
}

void LightmapProbe::set_sh_coefficients(const float* coeffs_27) {
    memcpy(pimpl->sh_coeffs, coeffs_27, 27*sizeof(float));
    pimpl->dirty = true;
}
void LightmapProbe::get_sh_coefficients(float* out_coeffs_27) const {
    memcpy(out_coeffs_27, pimpl->sh_coeffs, 27*sizeof(float));
}

void LightmapProbe::set_specular_cubemap(int64_t texture_rid) {
    pimpl->specular_cubemap = texture_rid;
    pimpl->dirty = true;
}
int64_t LightmapProbe::get_specular_cubemap() const { return pimpl->specular_cubemap; }

void LightmapProbe::set_interior(bool interior) { pimpl->interior = interior; pimpl->dirty = true; }
bool LightmapProbe::is_interior() const { return pimpl->interior; }
void LightmapProbe::set_energy(float multiplier) { pimpl->energy = multiplier; pimpl->dirty = true; }
float LightmapProbe::get_energy() const { return pimpl->energy; }

void LightmapProbe::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (!pimpl->dirty) return;

    // Create or update probe in RenderingServer
    if (pimpl->probe_rid == -1) {
        // pimpl->probe_rid = RenderingServer::lightmap_probe_create();
    }
    // Set probe data: position (from global transform), extents, SH coefficients,
    // specular cubemap texture, interior flag, energy.
    Transform3D global = get_global_transform();
    // RenderingServer::lightmap_probe_set_position(probe_rid, global.origin);
    // RenderingServer::lightmap_probe_set_extents(probe_rid, pimpl->capture_extents);
    // RenderingServer::lightmap_probe_set_sh(probe_rid, pimpl->sh_coeffs);
    // if specular cubemap valid: RenderingServer::lightmap_probe_set_cubemap(...)
    // RenderingServer::lightmap_probe_set_interior(probe_rid, pimpl->interior);
    // RenderingServer::lightmap_probe_set_energy(probe_rid, pimpl->energy);
    // Also set visibility and culling

    // For GI integration: if energy > 0, this probe influences the global lighting.
    // The rendering server will combine probes when lighting dynamic objects.

    pimpl->dirty = false;
}

void LightmapProbe::process(double delta) {
    Node3D::process(delta);
    // Probes do not need per‑frame processing unless dynamic capture.
}

} // namespace lighting