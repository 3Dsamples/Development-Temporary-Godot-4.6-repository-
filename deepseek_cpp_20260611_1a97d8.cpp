// Name : lighting enhancement
// File : scene/3d/node_3d_ext.cpp 2 of 60
// Description : Implementation of Node3DExt with rendering server sync, dirty transform caching,
//               physics interpolation, and real‑time lighting flag updates.
#include "node_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/transform_3d.h"
#include "core/object/object.h"

struct Node3DExt::Impl {
    RID render_instance_id;
    Transform3D current_global_transform;
    Transform3D previous_physics_transform;
    Transform3D interpolated_transform;
    bool transform_dirty = true;
    bool visibility_dirty = true;
    bool lighting_params_dirty = true;
    bool physics_interpolated = false;
    double physics_fraction = 0.0;
    bool always_visible = false;
    bool cast_shadow = true;
    int gi_mode = 1; // 0=off,1=static,2=dynamic
    float gi_contribution = 1.0f;
    Color emissive_color;
    float emissive_intensity = 0.0f;
    bool visible = true;

    Impl() {
        render_instance_id = RenderingServer::get_singleton()->instance_create();
    }

    ~Impl() {
        if (render_instance_id.is_valid()) {
            RenderingServer::get_singleton()->free(render_instance_id);
        }
    }
};

Node3DExt::Node3DExt() {
    pimpl = new Impl();
}

Node3DExt::~Node3DExt() {
    delete pimpl;
}

void Node3DExt::set_transform(const Transform3D &p_transform) {
    Node3D::set_transform(p_transform);
    pimpl->transform_dirty = true;
    _update_render_server_transform();
}

void Node3DExt::set_global_transform(const Transform3D &p_global) {
    Node3D::set_global_transform(p_global);
    pimpl->transform_dirty = true;
    _update_render_server_transform();
}

void Node3DExt::set_visible(bool p_visible) {
    Node3D::set_visible(p_visible);
    pimpl->visible = p_visible;
    pimpl->visibility_dirty = true;
    sync_render_server_visibility();
}

void Node3DExt::set_cast_shadow(bool p_cast) {
    pimpl->cast_shadow = p_cast;
    pimpl->lighting_params_dirty = true;
    sync_render_server_lighting_params();
}

bool Node3DExt::get_cast_shadow() const { return pimpl->cast_shadow; }

void Node3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    pimpl->lighting_params_dirty = true;
    sync_render_server_lighting_params();
}

int Node3DExt::get_gi_mode() const { return pimpl->gi_mode; }

void Node3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
    pimpl->lighting_params_dirty = true;
    sync_render_server_lighting_params();
}

float Node3DExt::get_gi_contribution() const { return pimpl->gi_contribution; }

void Node3DExt::set_emissive(const Color &p_color, float p_intensity) {
    pimpl->emissive_color = p_color;
    pimpl->emissive_intensity = p_intensity;
    pimpl->lighting_params_dirty = true;
    sync_render_server_lighting_params();
}

Color Node3DExt::get_emissive() const { return pimpl->emissive_color; }
float Node3DExt::get_emissive_intensity() const { return pimpl->emissive_intensity; }

RID Node3DExt::get_render_instance_id() const { return pimpl->render_instance_id; }

void Node3DExt::sync_render_server_transform() {
    if (!pimpl->transform_dirty) return;
    Transform3D global = get_global_transform();
    pimpl->current_global_transform = global;
    RenderingServer::get_singleton()->instance_set_transform(pimpl->render_instance_id, global);
    pimpl->transform_dirty = false;
}

void Node3DExt::sync_render_server_visibility() {
    if (!pimpl->visibility_dirty) return;
    bool visible = pimpl->visible && (pimpl->always_visible || true); // frustum culling handled by server
    RenderingServer::get_singleton()->instance_set_visible(pimpl->render_instance_id, visible);
    pimpl->visibility_dirty = false;
}

void Node3DExt::sync_render_server_lighting_params() {
    if (!pimpl->lighting_params_dirty) return;
    RenderingServer::get_singleton()->instance_set_cast_shadow(pimpl->render_instance_id, pimpl->cast_shadow);
    RenderingServer::get_singleton()->instance_set_gi_mode(pimpl->render_instance_id, pimpl->gi_mode);
    RenderingServer::get_singleton()->instance_set_gi_contribution(pimpl->render_instance_id, pimpl->gi_contribution);
    RenderingServer::get_singleton()->instance_set_emissive(pimpl->render_instance_id, pimpl->emissive_color, pimpl->emissive_intensity);
    pimpl->lighting_params_dirty = false;
}

void Node3DExt::set_physics_interpolated(bool p_enabled) {
    pimpl->physics_interpolated = p_enabled;
}

bool Node3DExt::is_physics_interpolated() const {
    return pimpl->physics_interpolated;
}

void Node3DExt::set_physics_fraction(double p_frac) {
    pimpl->physics_fraction = p_frac;
}

void Node3DExt::apply_interpolated_transform() {
    if (!pimpl->physics_interpolated) return;
    // interpolate between previous physics transform and current global transform
    Transform3D current = get_global_transform();
    Transform3D interp = pimpl->previous_physics_transform.interpolate_with(current, pimpl->physics_fraction);
    RenderingServer::get_singleton()->instance_set_transform(pimpl->render_instance_id, interp);
}

void Node3DExt::set_always_visible(bool p_visible) {
    pimpl->always_visible = p_visible;
    pimpl->visibility_dirty = true;
    sync_render_server_visibility();
}

bool Node3DExt::is_always_visible() const { return pimpl->always_visible; }

void Node3DExt::_transform_changed() {
    Node3D::_transform_changed();
    pimpl->transform_dirty = true;
    if (pimpl->physics_interpolated) {
        // store previous transform for interpolation
        pimpl->previous_physics_transform = pimpl->current_global_transform;
    }
    _update_render_server_transform();
}

void Node3DExt::_update_render_server_transform() {
    sync_render_server_transform();
}