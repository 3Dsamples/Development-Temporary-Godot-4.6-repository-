// area_3d.cpp
#include "area_3d.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

namespace lighting {

// ============================================================================
// Area3D implementation
// ============================================================================
struct Area3D::Impl {
    // Gravity
    bool gravity_enabled = true;
    double gravity_vector[3] = {0.0, -9.8, 0.0};
    bool gravity_point = false;
    double gravity_point_center[3] = {0.0, 0.0, 0.0};
    float gravity_distance_scale = 1.0f;

    // Damping
    bool linear_damp_enabled = true;
    float linear_damp = 0.1f;
    bool angular_damp_enabled = true;
    float angular_damp = 0.1f;
    int damp_priority = 0;

    // Callbacks
    std::function<void(int64_t,int64_t,int,int)> body_entered_cb;
    std::function<void(int64_t,int64_t,int,int)> body_exited_cb;
    std::function<void(int64_t,int64_t,int,int)> area_entered_cb;
    std::function<void(int64_t,int64_t,int,int)> area_exited_cb;

    int priority = 0;

    // GI
    int gi_mode = 0;              // off by default, area not visible
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0.0f,0.0f,0.0f};
    float emissive_intensity = 0.0f;

    // Overlap state (simulated, in real engine populated by physics server)
    std::vector<int64_t> overlapping_bodies;
    std::vector<int64_t> overlapping_areas;

    // Physics server handle
    int64_t area_rid = -1;
    bool dirty = true;

    ~Impl() {
        if (area_rid != -1) {
            // PhysicsServer3D::area_free(area_rid);
        }
    }

    void update_physics_server();
};

Area3D::Area3D() : pimpl(std::make_unique<Impl>()) {}
Area3D::~Area3D() = default;

void Area3D::set_gravity_enabled(bool enabled) { pimpl->gravity_enabled = enabled; pimpl->dirty = true; }
bool Area3D::is_gravity_enabled() const { return pimpl->gravity_enabled; }
void Area3D::set_gravity(const double* gravity_vector) { memcpy(pimpl->gravity_vector, gravity_vector, 3*sizeof(double)); pimpl->dirty = true; }
void Area3D::get_gravity(double* out_gravity) const { memcpy(out_gravity, pimpl->gravity_vector, 3*sizeof(double)); }
void Area3D::set_gravity_point(bool point) { pimpl->gravity_point = point; pimpl->dirty = true; }
bool Area3D::is_gravity_point() const { return pimpl->gravity_point; }
void Area3D::set_gravity_point_center(const double* center) { memcpy(pimpl->gravity_point_center, center, 3*sizeof(double)); pimpl->dirty = true; }
void Area3D::get_gravity_point_center(double* out_center) const { memcpy(out_center, pimpl->gravity_point_center, 3*sizeof(double)); }
void Area3D::set_gravity_distance_scale(float scale) { pimpl->gravity_distance_scale = std::max(0.0f, scale); pimpl->dirty = true; }
float Area3D::get_gravity_distance_scale() const { return pimpl->gravity_distance_scale; }

void Area3D::set_linear_damp_enabled(bool enabled) { pimpl->linear_damp_enabled = enabled; pimpl->dirty = true; }
bool Area3D::is_linear_damp_enabled() const { return pimpl->linear_damp_enabled; }
void Area3D::set_linear_damp(float damp) { pimpl->linear_damp = damp; pimpl->dirty = true; }
float Area3D::get_linear_damp() const { return pimpl->linear_damp; }
void Area3D::set_angular_damp_enabled(bool enabled) { pimpl->angular_damp_enabled = enabled; pimpl->dirty = true; }
bool Area3D::is_angular_damp_enabled() const { return pimpl->angular_damp_enabled; }
void Area3D::set_angular_damp(float damp) { pimpl->angular_damp = damp; pimpl->dirty = true; }
float Area3D::get_angular_damp() const { return pimpl->angular_damp; }
void Area3D::set_damp_priority(int priority) { pimpl->damp_priority = priority; pimpl->dirty = true; }
int Area3D::get_damp_priority() const { return pimpl->damp_priority; }

void Area3D::set_body_entered_callback(BodyCallback callback) { pimpl->body_entered_cb = callback; }
void Area3D::set_body_exited_callback(BodyCallback callback) { pimpl->body_exited_cb = callback; }
void Area3D::set_area_entered_callback(BodyCallback callback) { pimpl->area_entered_cb = callback; }
void Area3D::set_area_exited_callback(BodyCallback callback) { pimpl->area_exited_cb = callback; }

bool Area3D::overlaps_body(int64_t body_rid) const {
    return std::find(pimpl->overlapping_bodies.begin(), pimpl->overlapping_bodies.end(), body_rid) != pimpl->overlapping_bodies.end();
}
std::vector<int64_t> Area3D::get_overlapping_bodies() const { return pimpl->overlapping_bodies; }
std::vector<int64_t> Area3D::get_overlapping_areas() const { return pimpl->overlapping_areas; }

void Area3D::set_priority(int priority) { pimpl->priority = priority; pimpl->dirty = true; }
int Area3D::get_priority() const { return pimpl->priority; }

void Area3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; pimpl->dirty = true; }
int Area3D::get_gi_mode() const { return pimpl->gi_mode; }
void Area3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; pimpl->dirty = true; }
float Area3D::get_gi_contribution() const { return pimpl->gi_contribution; }
void Area3D::set_emissive_gi(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
    pimpl->dirty = true;
}
void Area3D::get_emissive_gi(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void Area3D::Impl::update_physics_server() {
    if (area_rid == -1) {
        // area_rid = PhysicsServer3D::area_create();
    }
    // Set transform
    Transform3D global = get_global_transform();
    // PhysicsServer3D::area_set_transform(area_rid, global);
    // Set shape data (from CollisionObject3D)
    // PhysicsServer3D::area_set_shape_list(area_rid, shapes);
    // Set gravity parameters
    // PhysicsServer3D::area_set_gravity(area_rid, gravity_enabled, gravity_vector, gravity_point, gravity_point_center, gravity_distance_scale);
    // Set damping
    // PhysicsServer3D::area_set_damp(area_rid, linear_damp_enabled, linear_damp, angular_damp_enabled, angular_damp, damp_priority);
    // Set priority
    // PhysicsServer3D::area_set_priority(area_rid, priority);
    // Set collision masks (inherited)
    // Enable/disable
    // Set callbacks (body_entered etc.) – signals
}

void Area3D::update_area() {
    if (!pimpl->dirty) return;
    pimpl->update_physics_server();
    pimpl->dirty = false;
}

void Area3D::synchronize_render_server(double delta) {
    CollisionObject3D::synchronize_render_server(delta);
    update_area();
    // If emissive_gi is set and gi_mode > 0, add this area's contribution to GI
    if (pimpl->gi_mode > 0 && pimpl->emissive_intensity > 0.0f) {
        // Register as an emissive volume for light propagation volumes or light probes
        // For dynamic GI, each frame we may need to add or update.
        // Placeholder: inject into GI system.
    }
}

void Area3D::process(double delta) {
    CollisionObject3D::process(delta);
    if (is_transform_dirty()) {
        pimpl->dirty = true;
    }
    // Overlap information would be updated by physics server and stored in overlapping_bodies/areas.
    // For demo, we simulate by clearing and re‑detecting each frame (not done for performance).
}

} // namespace lighting