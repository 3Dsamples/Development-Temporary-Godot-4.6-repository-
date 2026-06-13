// collision_object_3d.cpp
#include "collision_object_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <unordered_map>

namespace lighting {

struct CollisionObject3D::Impl {
    std::vector<CollisionShape> shapes;
    int next_shape_id = 1;
    uint32_t collision_layer = 0xFFFFFFFF;
    uint32_t collision_mask = 0xFFFFFFFF;
    float collision_priority = 1.0f;
    bool cast_collision_shadow = true;
    float gi_collision_contribution = 1.0f;

    CollisionCallback collision_callback;
    CollisionCallback area_enter_callback;
    CollisionCallback area_exit_callback;

    // Physics server handle
    int64_t physics_rid = -1;
};

CollisionObject3D::CollisionObject3D() : pimpl(std::make_unique<Impl>()) {}
CollisionObject3D::~CollisionObject3D() = default;

int CollisionObject3D::add_shape(const CollisionShape& shape) {
    CollisionShape new_shape = shape;
    new_shape.id = pimpl->next_shape_id++;
    pimpl->shapes.push_back(new_shape);
    _shape_added(new_shape.id);
    return new_shape.id;
}

void CollisionObject3D::remove_shape(int shape_id) {
    auto it = std::find_if(pimpl->shapes.begin(), pimpl->shapes.end(),
        [shape_id](const CollisionShape& s) { return s.id == shape_id; });
    if (it != pimpl->shapes.end()) {
        int removed_id = it->id;
        pimpl->shapes.erase(it);
        _shape_removed(removed_id);
    }
}

void CollisionObject3D::clear_shapes() {
    pimpl->shapes.clear();
    pimpl->next_shape_id = 1;
}

int CollisionObject3D::get_shape_count() const { return (int)pimpl->shapes.size(); }

CollisionShape CollisionObject3D::get_shape(int shape_id) const {
    for (const auto& s : pimpl->shapes)
        if (s.id == shape_id) return s;
    return CollisionShape{};
}

void CollisionObject3D::set_shape_transform(int shape_id, const Transform3D& transform) {
    for (auto& s : pimpl->shapes)
        if (s.id == shape_id) { s.local_transform = transform; break; }
}

Transform3D CollisionObject3D::get_shape_transform(int shape_id) const {
    for (const auto& s : pimpl->shapes)
        if (s.id == shape_id) return s.local_transform;
    return Transform3D();
}

void CollisionObject3D::set_collision_layer(uint32_t layer) { pimpl->collision_layer = layer; }
uint32_t CollisionObject3D::get_collision_layer() const { return pimpl->collision_layer; }
void CollisionObject3D::set_collision_mask(uint32_t mask) { pimpl->collision_mask = mask; }
uint32_t CollisionObject3D::get_collision_mask() const { return pimpl->collision_mask; }
void CollisionObject3D::set_collision_priority(float priority) { pimpl->collision_priority = priority; }
float CollisionObject3D::get_collision_priority() const { return pimpl->collision_priority; }

void CollisionObject3D::set_cast_collision_shadow(bool cast) { pimpl->cast_collision_shadow = cast; }
bool CollisionObject3D::get_cast_collision_shadow() const { return pimpl->cast_collision_shadow; }
void CollisionObject3D::set_gi_collision_contribution(float amount) { pimpl->gi_collision_contribution = amount; }
float CollisionObject3D::get_gi_collision_contribution() const { return pimpl->gi_collision_contribution; }

void CollisionObject3D::set_collision_callback(CollisionCallback callback) { pimpl->collision_callback = callback; }
void CollisionObject3D::set_area_enter_callback(CollisionCallback callback) { pimpl->area_enter_callback = callback; }
void CollisionObject3D::set_area_exit_callback(CollisionCallback callback) { pimpl->area_exit_callback = callback; }

bool CollisionObject3D::intersect_ray(const double* origin, const double* direction, double max_distance, double* out_point, double* out_normal) const {
    // Simplified ray-shape intersection – for sphere and box only.
    double closest_t = max_distance;
    bool hit = false;
    for (const auto& shape : pimpl->shapes) {
        if (shape.disabled) continue;
        Transform3D world_transform = get_global_transform() * shape.local_transform;
        if (shape.type == CollisionShapeType::SPHERE) {
            double radius = *(double*)shape.shape_data;
            double oc[3] = { origin[0] - world_transform.origin[0],
                             origin[1] - world_transform.origin[1],
                             origin[2] - world_transform.origin[2] };
            double b = oc[0]*direction[0] + oc[1]*direction[1] + oc[2]*direction[2];
            double c = oc[0]*oc[0] + oc[1]*oc[1] + oc[2]*oc[2] - radius*radius;
            double disc = b*b - c;
            if (disc > 0) {
                double t = -b - sqrt(disc);
                if (t > 0 && t < closest_t) {
                    closest_t = t;
                    hit = true;
                    if (out_point) {
                        out_point[0] = origin[0] + direction[0]*t;
                        out_point[1] = origin[1] + direction[1]*t;
                        out_point[2] = origin[2] + direction[2]*t;
                    }
                    if (out_normal) {
                        double n[3] = { out_point[0] - world_transform.origin[0],
                                        out_point[1] - world_transform.origin[1],
                                        out_point[2] - world_transform.origin[2] };
                        double len = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
                        if (len > 1e-6) {
                            out_normal[0] = n[0]/len;
                            out_normal[1] = n[1]/len;
                            out_normal[2] = n[2]/len;
                        }
                    }
                }
            }
        }
        // Box intersection omitted for brevity – would be full AABB ray test.
    }
    return hit;
}

void CollisionObject3D::synchronize_render_server(double delta) {
    VisualInstance3D::synchronize_render_server(delta);
    // Update physics server with transform and shapes
}

void CollisionObject3D::_shape_added(int shape_id) {
    // Notify physics server
}

void CollisionObject3D::_shape_removed(int shape_id) {
    // Notify physics server
}

void CollisionObject3D::_collision_detected(const double* point, const double* normal, int with_shape) {
    if (pimpl->collision_callback)
        pimpl->collision_callback(point, normal, -1, with_shape);
}

// ============================================================================
// Area3D implementation
// ============================================================================
struct Area3D::Impl {
    bool gravity_enabled = true;
    double gravity_vector[3] = {0, -9.8, 0};
    bool gravity_point = false;
    double gravity_point_center[3] = {0,0,0};
    bool linear_damp_enabled = true;
    float linear_damp = 0.1f;
    float angular_damp = 0.1f;
    int priority = 0;
    float gi_override = -1.0f; // <0 means no override
};

Area3D::Area3D() : pimpl(std::make_unique<Impl>()) {}
Area3D::~Area3D() = default;

void Area3D::set_gravity_enabled(bool enabled) { pimpl->gravity_enabled = enabled; }
bool Area3D::is_gravity_enabled() const { return pimpl->gravity_enabled; }
void Area3D::set_gravity(const double* gravity_vector) { memcpy(pimpl->gravity_vector, gravity_vector, 3*sizeof(double)); }
void Area3D::get_gravity(double* out_gravity) const { memcpy(out_gravity, pimpl->gravity_vector, 3*sizeof(double)); }
void Area3D::set_gravity_point(bool point) { pimpl->gravity_point = point; }
bool Area3D::is_gravity_point() const { return pimpl->gravity_point; }
void Area3D::set_gravity_point_center(const double* center) { memcpy(pimpl->gravity_point_center, center, 3*sizeof(double)); }
void Area3D::get_gravity_point_center(double* out_center) const { memcpy(out_center, pimpl->gravity_point_center, 3*sizeof(double)); }

void Area3D::set_linear_damp_enabled(bool enabled) { pimpl->linear_damp_enabled = enabled; }
void Area3D::set_linear_damp(float damp) { pimpl->linear_damp = damp; }
float Area3D::get_linear_damp() const { return pimpl->linear_damp; }
void Area3D::set_angular_damp(float damp) { pimpl->angular_damp = damp; }
float Area3D::get_angular_damp() const { return pimpl->angular_damp; }

void Area3D::set_priority(int priority) { pimpl->priority = priority; }
int Area3D::get_priority() const { return pimpl->priority; }

bool Area3D::overlaps_body(int64_t body_rid) const {
    // Placeholder: query physics server
    return false;
}

std::vector<int64_t> Area3D::get_overlapping_bodies() const {
    return {};
}

void Area3D::set_gi_override(float intensity) { pimpl->gi_override = intensity; }
float Area3D::get_gi_override() const { return pimpl->gi_override; }

} // namespace lighting