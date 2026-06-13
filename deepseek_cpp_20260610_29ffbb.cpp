// static_body_3d.cpp
#include "static_body_3d.h"
#include <cstring>
#include <algorithm>

namespace lighting {

struct StaticBody3D::Impl {
    double constant_linear_velocity[3] = {0,0,0};
    double constant_angular_velocity[3] = {0,0,0};

    int lightmap_index = -1;
    double lightmap_uv_scale[2] = {1.0, 1.0};
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    int gi_mode = 1; // 1 = static (pre‑baked or lightmapped)
    bool cast_shadow = true;
    bool receive_shadow = true;
    bool lightmap_shadow_receiver = true;

    // Rendering server handle for static GI data
    int64_t gi_rid = -1;
};

StaticBody3D::StaticBody3D() : pimpl(std::make_unique<Impl>()) {
    set_body_mode(BodyMode::STATIC);
}
StaticBody3D::~StaticBody3D() = default;

void StaticBody3D::set_constant_linear_velocity(const double* velocity) {
    memcpy(pimpl->constant_linear_velocity, velocity, 3*sizeof(double));
}
void StaticBody3D::get_constant_linear_velocity(double* out_velocity) const {
    memcpy(out_velocity, pimpl->constant_linear_velocity, 3*sizeof(double));
}
void StaticBody3D::set_constant_angular_velocity(const double* velocity) {
    memcpy(pimpl->constant_angular_velocity, velocity, 3*sizeof(double));
}
void StaticBody3D::get_constant_angular_velocity(double* out_velocity) const {
    memcpy(out_velocity, pimpl->constant_angular_velocity, 3*sizeof(double));
}

void StaticBody3D::set_lightmap_index(int index) {
    pimpl->lightmap_index = index;
}
int StaticBody3D::get_lightmap_index() const { return pimpl->lightmap_index; }

void StaticBody3D::set_lightmap_uv_scale(const double* scale) {
    memcpy(pimpl->lightmap_uv_scale, scale, 2*sizeof(double));
}
void StaticBody3D::get_lightmap_uv_scale(double* out_scale) const {
    memcpy(out_scale, pimpl->lightmap_uv_scale, 2*sizeof(double));
}

void StaticBody3D::set_emissive_lighting(const float* emissive_color, float intensity) {
    memcpy(pimpl->emissive_color, emissive_color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void StaticBody3D::get_emissive_lighting(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void StaticBody3D::set_gi_mode(int mode) {
    pimpl->gi_mode = mode;
}
int StaticBody3D::get_gi_mode() const { return pimpl->gi_mode; }

void StaticBody3D::set_cast_shadow(bool cast) {
    pimpl->cast_shadow = cast;
    GeometryInstance3D::set_cast_shadow(cast);
}
void StaticBody3D::set_receive_shadow(bool receive) {
    pimpl->receive_shadow = receive;
    GeometryInstance3D::set_receive_shadow(receive);
}
void StaticBody3D::set_lightmap_shadow_receiver(bool receive) {
    pimpl->lightmap_shadow_receiver = receive;
}
bool StaticBody3D::is_lightmap_shadow_receiver() const { return pimpl->lightmap_shadow_receiver; }

void StaticBody3D::update_physics(double delta_time) {
    // Static bodies do not move by simulation, but can have constant velocity
    // if they are kinematic. However, being static, we ignore physics update.
    if (get_body_mode() == BodyMode::KINEMATIC) {
        // Apply constant velocity to transform (if any)
        double delta[3] = {
            pimpl->constant_linear_velocity[0] * delta_time,
            pimpl->constant_linear_velocity[1] * delta_time,
            pimpl->constant_linear_velocity[2] * delta_time
        };
        if (delta[0] != 0.0 || delta[1] != 0.0 || delta[2] != 0.0) {
            Transform3D t = get_global_transform();
            t.origin[0] += delta[0];
            t.origin[1] += delta[1];
            t.origin[2] += delta[2];
            set_global_transform(t);
        }
        // Angular velocity would rotate basis (not implemented)
    }
}

void StaticBody3D::synchronize_render_server(double delta) {
    PhysicsBody3D::synchronize_render_server(delta);
    // Notify rendering server about lightmap index, emissive, and GI mode.
    // For static lighting, we also potentially update the global illumination
    // contribution (e.g., for light probes or light propagation volumes).
    if (pimpl->gi_mode == 1) { // static GI
        // Placeholder: register this body as static GI contributor
    }
}

} // namespace lighting