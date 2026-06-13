// gpu_particles_3d.cpp
#include "gpu_particles_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

namespace lighting {

struct GPUParticles3D::Impl {
    bool emitting = true;
    bool one_shot = false;
    bool restart_requested = false;
    ParticleParameters params;
    ParticleDrawOrder draw_order = ParticleDrawOrder::INDICES;
    bool trail_enabled = false;
    float trail_length = 0.3f;
    bool collision_enabled = false;
    float collision_radius = 0.05f;
    uint32_t collision_mask = 0xFFFFFFFF;
    int64_t sub_emitter_instance = -1;
    float sub_emitter_at_end = 0.0f;

    // Internal state for simulation (on GPU, but CPU mirror for bounds)
    double bounds_min[3] = {-10, -10, -10};
    double bounds_max[3] = {10, 10, 10};
    bool bounds_dirty = true;

    // Random generator for seeding GPU random (CPU side)
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist01;

    // GPU buffer handles (fake IDs)
    int64_t particle_buffer = -1;
    int64_t parameter_buffer = -1;

    Impl() : rng(std::random_device{}()), dist01(0.0f, 1.0f) {}
};

GPUParticles3D::GPUParticles3D() : pimpl(std::make_unique<Impl>()) {}
GPUParticles3D::~GPUParticles3D() = default;

void GPUParticles3D::set_emitting(bool emitting) { pimpl->emitting = emitting; }
bool GPUParticles3D::is_emitting() const { return pimpl->emitting; }
void GPUParticles3D::restart() { pimpl->restart_requested = true; }
void GPUParticles3D::set_one_shot(bool one_shot) { pimpl->one_shot = one_shot; }
bool GPUParticles3D::is_one_shot() const { return pimpl->one_shot; }
void GPUParticles3D::set_lifetime(float seconds) { pimpl->params.lifetime = seconds; }
float GPUParticles3D::get_lifetime() const { return pimpl->params.lifetime; }

void GPUParticles3D::set_parameters(const ParticleParameters& params) {
    pimpl->params = params;
    pimpl->bounds_dirty = true;
}
const ParticleParameters& GPUParticles3D::get_parameters() const { return pimpl->params; }

void GPUParticles3D::set_draw_order(ParticleDrawOrder order) { pimpl->draw_order = order; }
ParticleDrawOrder GPUParticles3D::get_draw_order() const { return pimpl->draw_order; }
void GPUParticles3D::set_trail_enabled(bool enabled) { pimpl->trail_enabled = enabled; }
bool GPUParticles3D::is_trail_enabled() const { return pimpl->trail_enabled; }
void GPUParticles3D::set_trail_length(float seconds) { pimpl->trail_length = seconds; }
float GPUParticles3D::get_trail_length() const { return pimpl->trail_length; }

void GPUParticles3D::set_collision_enabled(bool enabled) { pimpl->collision_enabled = enabled; }
bool GPUParticles3D::is_collision_enabled() const { return pimpl->collision_enabled; }
void GPUParticles3D::set_collision_radius(float radius) { pimpl->collision_radius = radius; }
float GPUParticles3D::get_collision_radius() const { return pimpl->collision_radius; }
void GPUParticles3D::set_collision_mask(uint32_t mask) { pimpl->collision_mask = mask; }
uint32_t GPUParticles3D::get_collision_mask() const { return pimpl->collision_mask; }

void GPUParticles3D::set_sub_emitter(int64_t particle_instance_id, float at_end) {
    pimpl->sub_emitter_instance = particle_instance_id;
    pimpl->sub_emitter_at_end = at_end;
}
void GPUParticles3D::clear_sub_emitter() { pimpl->sub_emitter_instance = -1; }

void GPUParticles3D::set_shader_uniform(const char* name, const float* value, int size) {
    // Placeholder: would bind to GPU shader
}
void GPUParticles3D::clear_shader_uniform(const char* name) { }

void GPUParticles3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    // Send particle parameters to GPU buffer
    // Update emission state, restart if requested
    if (pimpl->restart_requested) {
        // GPU would clear particle buffers
        pimpl->restart_requested = false;
    }
    // Update bounds from GPU (simulated)
    if (pimpl->bounds_dirty) {
        // Conservative bounds from emission shape
        double radius = 10.0;
        double center[3] = {0,0,0};
        pimpl->bounds_min[0] = center[0] - radius;
        pimpl->bounds_min[1] = center[1] - radius;
        pimpl->bounds_min[2] = center[2] - radius;
        pimpl->bounds_max[0] = center[0] + radius;
        pimpl->bounds_max[1] = center[1] + radius;
        pimpl->bounds_max[2] = center[2] + radius;
        set_aabb(pimpl->bounds_min, pimpl->bounds_max);
        pimpl->bounds_dirty = false;
    }
}

void GPUParticles3D::update_particle_buffers() {
    // Update GPU constant buffer with current parameters
}

void GPUParticles3D::_update_bounding_volume() {
    // Recompute from active particles (CPU side would read back from GPU, but expensive)
    // Here we just mark dirty, synchronize_render_server will update.
    pimpl->bounds_dirty = true;
}

} // namespace lighting