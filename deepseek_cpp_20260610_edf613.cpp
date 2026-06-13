// reflection_probe_3d.cpp
#include "reflection_probe_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace lighting {

struct ReflectionProbe3D::Impl {
    ReflectionProbeUpdateMode update_mode = ReflectionProbeUpdateMode::ON_MOVE;
    int resolution = 256;
    float update_rate = 2.0f;          // Hz
    double time_since_last_update = 0.0;

    Vector3 extents = Vector3(10.0f, 10.0f, 10.0f);
    Vector3 origin_offset = Vector3(0.0f, 0.0f, 0.0f);
    float intensity = 1.0f;
    float blend_distance = 1.0f;

    uint32_t cull_mask = 0xFFFFFFFF;
    float gi_importance = 1.0f;
    bool cast_shadow = true;
    float ambient_contribution = 0.2f;

    bool include_dynamic_objects = true;

    // Internal state
    Vector3 last_position;
    bool dirty = true;
    bool cubemap_ready = false;
    int64_t cubemap_texture = -1;   // RenderingServer texture RID

    // Temporary capture data
    std::array<std::vector<float>, 6> captured_faces; // HDR linear floats
};

ReflectionProbe3D::ReflectionProbe3D() : pimpl(std::make_unique<Impl>()) {}
ReflectionProbe3D::~ReflectionProbe3D() = default;

void ReflectionProbe3D::set_update_mode(ReflectionProbeUpdateMode mode) {
    pimpl->update_mode = mode;
    if (mode == ReflectionProbeUpdateMode::ONCE && !pimpl->cubemap_ready)
        update_cubemap();
}
ReflectionProbeUpdateMode ReflectionProbe3D::get_update_mode() const { return pimpl->update_mode; }
void ReflectionProbe3D::set_resolution(int resolution) {
    pimpl->resolution = std::max(16, std::min(4096, resolution));
    pimpl->dirty = true;
}
int ReflectionProbe3D::get_resolution() const { return pimpl->resolution; }
void ReflectionProbe3D::set_update_rate(float hz) { pimpl->update_rate = std::max(0.1f, hz); }
float ReflectionProbe3D::get_update_rate() const { return pimpl->update_rate; }

void ReflectionProbe3D::set_extents(const Vector3& extents) {
    pimpl->extents = extents;
    pimpl->dirty = true;
}
Vector3 ReflectionProbe3D::get_extents() const { return pimpl->extents; }
void ReflectionProbe3D::set_origin_offset(const Vector3& offset) { pimpl->origin_offset = offset; }
Vector3 ReflectionProbe3D::get_origin_offset() const { return pimpl->origin_offset; }
void ReflectionProbe3D::set_intensity(float intensity) { pimpl->intensity = intensity; }
float ReflectionProbe3D::get_intensity() const { return pimpl->intensity; }
void ReflectionProbe3D::set_blend_distance(float distance) { pimpl->blend_distance = distance; }
float ReflectionProbe3D::get_blend_distance() const { return pimpl->blend_distance; }

void ReflectionProbe3D::set_cull_mask(uint32_t mask) { pimpl->cull_mask = mask; }
uint32_t ReflectionProbe3D::get_cull_mask() const { return pimpl->cull_mask; }
void ReflectionProbe3D::set_gi_importance(float importance) { pimpl->gi_importance = importance; }
float ReflectionProbe3D::get_gi_importance() const { return pimpl->gi_importance; }
void ReflectionProbe3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; }
bool ReflectionProbe3D::get_cast_shadow() const { return pimpl->cast_shadow; }
void ReflectionProbe3D::set_ambient_contribution(float amount) { pimpl->ambient_contribution = amount; }
float ReflectionProbe3D::get_ambient_contribution() const { return pimpl->ambient_contribution; }

void ReflectionProbe3D::set_include_dynamic_objects(bool include) { pimpl->include_dynamic_objects = include; }
bool ReflectionProbe3D::get_include_dynamic_objects() const { return pimpl->include_dynamic_objects; }

void ReflectionProbe3D::update_cubemap() {
    if (!is_visible()) return;
    // In a real engine: render scene from probe position (six faces)
    // using the rendering server. Here we simulate capture.
    // For each face, capture into pimpl->captured_faces[face].
    // Then upload to GPU texture.
    pimpl->cubemap_ready = true;
    pimpl->dirty = false;
    pimpl->time_since_last_update = 0.0;
}

bool ReflectionProbe3D::is_cubemap_ready() const { return pimpl->cubemap_ready; }
int64_t ReflectionProbe3D::get_cubemap_texture_rid() const { return pimpl->cubemap_texture; }

void ReflectionProbe3D::_transform_changed() {
    VisualInstance3D::_transform_changed();
    if (pimpl->update_mode == ReflectionProbeUpdateMode::ON_MOVE) {
        Vector3 current_pos = get_global_transform().origin;
        if (pimpl->last_position.distance_to(current_pos) > 0.01f) {
            pimpl->last_position = current_pos;
            pimpl->dirty = true;
        }
    }
}

void ReflectionProbe3D::synchronize_render_server(double delta) {
    VisualInstance3D::synchronize_render_server(delta);
    // Update capture timer
    if (pimpl->update_mode == ReflectionProbeUpdateMode::ALWAYS) {
        pimpl->time_since_last_update += delta;
        if (pimpl->time_since_last_update >= (1.0 / pimpl->update_rate)) {
            update_cubemap();
        }
    } else if (pimpl->update_mode == ReflectionProbeUpdateMode::ON_MOVE && pimpl->dirty) {
        update_cubemap();
    }
    // Send probe parameters to rendering server (influence volume, intensity, etc.)
    // Also associate cubemap texture with this probe.
}

} // namespace lighting