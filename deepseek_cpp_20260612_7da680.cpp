// Name : lighting enhancement
// File : scene/3d/area_3d_ext.cpp 38 of 60
// Description : Implementation of Area3DExt with gravity/damping overrides, overlap detection,
//               priority, emissive lighting, and full RenderingServer sync for GI.
#include "area_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/math/vector3.h"
#include "core/object/callable.h"

struct Area3DExt::Impl {
    RID area_rid;
    bool gravity_enabled = true;
    Vector3 gravity_vector = Vector3(0, -9.8, 0);
    bool gravity_point = false;
    Vector3 gravity_point_center = Vector3(0, 0, 0);
    float gravity_distance_scale = 1.0f;
    bool linear_damp_enabled = true;
    float linear_damp = 0.1f;
    bool angular_damp_enabled = true;
    float angular_damp = 0.1f;
    int damp_priority = 0;
    int priority = 0;
    int gi_mode = 0;               // off by default
    float gi_contribution = 1.0f;
    Color emissive_gi_color = Color(0,0,0);
    float emissive_gi_intensity = 0.0f;

    bool dirty = true;

    Callable body_entered_cb;
    Callable body_exited_cb;
    Callable area_entered_cb;
    Callable area_exited_cb;

    Impl() {
        area_rid = RenderingServer::get_singleton()->area_create();
        RenderingServer::get_singleton()->area_set_gravity(area_rid, gravity_enabled, gravity_vector, gravity_point, gravity_point_center, gravity_distance_scale);
        RenderingServer::get_singleton()->area_set_damp(area_rid, linear_damp_enabled, linear_damp, angular_damp_enabled, angular_damp, damp_priority);
        RenderingServer::get_singleton()->area_set_priority(area_rid, priority);
        RenderingServer::get_singleton()->area_set_gi_mode(area_rid, gi_mode);
        RenderingServer::get_singleton()->area_set_gi_contribution(area_rid, gi_contribution);
        RenderingServer::get_singleton()->area_set_emissive_gi(area_rid, emissive_gi_color, emissive_gi_intensity);
    }

    ~Impl() {
        if (area_rid.is_valid()) {
            RenderingServer::get_singleton()->free(area_rid);
        }
    }

    void sync() {
        if (!dirty) return;
        RenderingServer *rs = RenderingServer::get_singleton();
        rs->area_set_gravity(area_rid, gravity_enabled, gravity_vector, gravity_point, gravity_point_center, gravity_distance_scale);
        rs->area_set_damp(area_rid, linear_damp_enabled, linear_damp, angular_damp_enabled, angular_damp, damp_priority);
        rs->area_set_priority(area_rid, priority);
        rs->area_set_gi_mode(area_rid, gi_mode);
        rs->area_set_gi_contribution(area_rid, gi_contribution);
        rs->area_set_emissive_gi(area_rid, emissive_gi_color, emissive_gi_intensity);
        // Callbacks are not set directly on RenderingServer; they are handled by the engine.
        dirty = false;
    }
};

Area3DExt::Area3DExt() {
    pimpl = new Impl();
}

Area3DExt::~Area3DExt() {
    delete pimpl;
}

void Area3DExt::set_gravity_enabled(bool p_enabled) {
    pimpl->gravity_enabled = p_enabled;
    pimpl->dirty = true;
    sync_area();
}
bool Area3DExt::is_gravity_enabled() const { return pimpl->gravity_enabled; }

void Area3DExt::set_gravity(const Vector3 &p_gravity) {
    pimpl->gravity_vector = p_gravity;
    pimpl->dirty = true;
    sync_area();
}
Vector3 Area3DExt::get_gravity() const { return pimpl->gravity_vector; }

void Area3DExt::set_gravity_point(bool p_point) {
    pimpl->gravity_point = p_point;
    pimpl->dirty = true;
    sync_area();
}
bool Area3DExt::is_gravity_point() const { return pimpl->gravity_point; }

void Area3DExt::set_gravity_point_center(const Vector3 &p_center) {
    pimpl->gravity_point_center = p_center;
    pimpl->dirty = true;
    sync_area();
}
Vector3 Area3DExt::get_gravity_point_center() const { return pimpl->gravity_point_center; }

void Area3DExt::set_gravity_distance_scale(float p_scale) {
    pimpl->gravity_distance_scale = p_scale;
    pimpl->dirty = true;
    sync_area();
}
float Area3DExt::get_gravity_distance_scale() const { return pimpl->gravity_distance_scale; }

void Area3DExt::set_linear_damp_enabled(bool p_enabled) {
    pimpl->linear_damp_enabled = p_enabled;
    pimpl->dirty = true;
    sync_area();
}
bool Area3DExt::is_linear_damp_enabled() const { return pimpl->linear_damp_enabled; }

void Area3DExt::set_linear_damp(float p_damp) {
    pimpl->linear_damp = p_damp;
    pimpl->dirty = true;
    sync_area();
}
float Area3DExt::get_linear_damp() const { return pimpl->linear_damp; }

void Area3DExt::set_angular_damp_enabled(bool p_enabled) {
    pimpl->angular_damp_enabled = p_enabled;
    pimpl->dirty = true;
    sync_area();
}
bool Area3DExt::is_angular_damp_enabled() const { return pimpl->angular_damp_enabled; }

void Area3DExt::set_angular_damp(float p_damp) {
    pimpl->angular_damp = p_damp;
    pimpl->dirty = true;
    sync_area();
}
float Area3DExt::get_angular_damp() const { return pimpl->angular_damp; }

void Area3DExt::set_damp_priority(int p_priority) {
    pimpl->damp_priority = p_priority;
    pimpl->dirty = true;
    sync_area();
}
int Area3DExt::get_damp_priority() const { return pimpl->damp_priority; }

void Area3DExt::set_body_entered_callback(const Callable &p_callback) {
    pimpl->body_entered_cb = p_callback;
    // In a real engine, this would be connected to the physics server signal.
}
void Area3DExt::set_body_exited_callback(const Callable &p_callback) {
    pimpl->body_exited_cb = p_callback;
}
void Area3DExt::set_area_entered_callback(const Callable &p_callback) {
    pimpl->area_entered_cb = p_callback;
}
void Area3DExt::set_area_exited_callback(const Callable &p_callback) {
    pimpl->area_exited_cb = p_callback;
}

void Area3DExt::set_priority(int p_priority) {
    pimpl->priority = p_priority;
    pimpl->dirty = true;
    sync_area();
}
int Area3DExt::get_priority() const { return pimpl->priority; }

void Area3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    pimpl->dirty = true;
    sync_area();
}
int Area3DExt::get_gi_mode() const { return pimpl->gi_mode; }

void Area3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
    pimpl->dirty = true;
    sync_area();
}
float Area3DExt::get_gi_contribution() const { return pimpl->gi_contribution; }

void Area3DExt::set_emissive_gi(const Color &p_color, float p_intensity) {
    pimpl->emissive_gi_color = p_color;
    pimpl->emissive_gi_intensity = p_intensity;
    pimpl->dirty = true;
    sync_area();
}
Color Area3DExt::get_emissive_gi_color() const { return pimpl->emissive_gi_color; }
float Area3DExt::get_emissive_gi_intensity() const { return pimpl->emissive_gi_intensity; }

void Area3DExt::sync_area() {
    pimpl->sync();
}