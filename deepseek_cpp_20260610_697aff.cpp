// gpu_particles_attractor_3d.cpp
#include "gpu_particles_attractor_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace lighting {

struct GPUParticlesAttractor3D::Impl {
    AttractorShape shape = AttractorShape::SPHERE;
    double size[3] = {1.0, 1.0, 1.0};   // half extents for sphere/box
    double direction[3] = {0.0, 1.0, 0.0};
    float strength = 1.0f;
    float falloff = 1.0f;               // 0=constant,1=linear,2=quadratic
    float attenuation = 1.0f;
    float max_distance = 100.0f;

    bool axis_x = true, axis_y = true, axis_z = true;

    // Visual representation
    bool visible = true;
    bool cast_shadow = false;            // attractor visual rarely casts shadow
    int gi_mode = 0;                     // off by default
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    // Render server handles
    int64_t attractor_rid = -1;
    bool dirty = true;
};

GPUParticlesAttractor3D::GPUParticlesAttractor3D() : pimpl(std::make_unique<Impl>()) {}
GPUParticlesAttractor3D::~GPUParticlesAttractor3D() = default;

void GPUParticlesAttractor3D::set_shape(AttractorShape shape) { pimpl->shape = shape; pimpl->dirty = true; }
AttractorShape GPUParticlesAttractor3D::get_shape() const { return pimpl->shape; }
void GPUParticlesAttractor3D::set_size(const double* size) { memcpy(pimpl->size, size, 3*sizeof(double)); pimpl->dirty = true; }
void GPUParticlesAttractor3D::get_size(double* out_size) const { memcpy(out_size, pimpl->size, 3*sizeof(double)); }
void GPUParticlesAttractor3D::set_direction(const double* dir) { memcpy(pimpl->direction, dir, 3*sizeof(double)); pimpl->dirty = true; }
void GPUParticlesAttractor3D::get_direction(double* out_dir) const { memcpy(out_dir, pimpl->direction, 3*sizeof(double)); }

void GPUParticlesAttractor3D::set_strength(float strength) { pimpl->strength = strength; pimpl->dirty = true; }
float GPUParticlesAttractor3D::get_strength() const { return pimpl->strength; }
void GPUParticlesAttractor3D::set_falloff(float falloff) { pimpl->falloff = falloff; pimpl->dirty = true; }
float GPUParticlesAttractor3D::get_falloff() const { return pimpl->falloff; }
void GPUParticlesAttractor3D::set_attenuation(float attenuation) { pimpl->attenuation = attenuation; pimpl->dirty = true; }
float GPUParticlesAttractor3D::get_attenuation() const { return pimpl->attenuation; }
void GPUParticlesAttractor3D::set_max_distance(float max_dist) { pimpl->max_distance = max_dist; pimpl->dirty = true; }
float GPUParticlesAttractor3D::get_max_distance() const { return pimpl->max_distance; }

void GPUParticlesAttractor3D::set_axis_enabled(bool x, bool y, bool z) {
    pimpl->axis_x = x; pimpl->axis_y = y; pimpl->axis_z = z;
    pimpl->dirty = true;
}
void GPUParticlesAttractor3D::get_axis_enabled(bool& x, bool& y, bool& z) const {
    x = pimpl->axis_x; y = pimpl->axis_y; z = pimpl->axis_z;
}

void GPUParticlesAttractor3D::set_visible(bool visible) { pimpl->visible = visible; }
bool GPUParticlesAttractor3D::is_visible() const { return pimpl->visible; }
void GPUParticlesAttractor3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool GPUParticlesAttractor3D::get_cast_shadow() const { return pimpl->cast_shadow; }
void GPUParticlesAttractor3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; }
int GPUParticlesAttractor3D::get_gi_mode() const { return pimpl->gi_mode; }
void GPUParticlesAttractor3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void GPUParticlesAttractor3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void GPUParticlesAttractor3D::process(double delta) {
    Node3D::process(delta);
    // Nothing per‑frame, just sync
}

void GPUParticlesAttractor3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (!pimpl->dirty) return;
    // Create or update attractor in rendering server / particle system.
    // For each active GPU particles system, we would register this attractor.
    // In a real engine, we would call RenderingServer::particles_attractor_create().
    if (pimpl->attractor_rid == -1) {
        // pimpl->attractor_rid = RenderingServer::particles_attractor_create();
    }
    // Set parameters: shape, size, direction, strength, falloff, attenuation, max_distance, axes.
    // Also set visual representation (if visible) – often a debug wireframe.
    // For GI: if gi_mode > 0 and emissive_intensity > 0, register as emissive source.
    if (pimpl->gi_mode > 0 && pimpl->emissive_intensity > 0.0f) {
        // Inject emissive contribution into light probe or VCT.
    }
    pimpl->dirty = false;
}

} // namespace lighting