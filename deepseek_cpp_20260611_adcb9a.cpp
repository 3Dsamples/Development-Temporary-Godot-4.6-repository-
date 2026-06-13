// character_body_3d.cpp
#include "character_body_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace lighting {

// ============================================================================
// Collision information for slide
// ============================================================================
struct SlideCollision {
    double position[3];
    double normal[3];
    double velocity[3];
    int shape_index;
    int collider_id;
};

// ============================================================================
// CharacterBody3D implementation
// ============================================================================
struct CharacterBody3D::Impl {
    double velocity[3] = {0,0,0};
    double floor_velocity[3] = {0,0,0};
    double up_direction[3] = {0,1,0};
    int max_slides = 4;
    float floor_max_angle = 1.396f;         // 80 degrees in radians
    bool floor_stop_on_slope = true;
    float wall_min_angle = 0.0f;

    std::vector<SlideCollision> slide_collisions;
    bool on_floor = false;
    bool on_wall = false;
    bool on_ceiling = false;
    int slide_count = 0;

    // Lighting flags
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 2;               // dynamic
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    // Internal state
    double last_position[3] = {0,0,0};
    double current_velocity[3] = {0,0,0};

    void resolve_collision(SlideCollision& col, double* in_velocity, double* out_velocity);
    bool test_floor(double* velocity, double* normal);
};

CharacterBody3D::CharacterBody3D() : pimpl(std::make_unique<Impl>()) {
    set_body_mode(BodyMode::KINEMATIC);
}
CharacterBody3D::~CharacterBody3D() = default;

void CharacterBody3D::set_velocity(const double* velocity) {
    memcpy(pimpl->velocity, velocity, 3*sizeof(double));
}
const double* CharacterBody3D::get_velocity() const { return pimpl->velocity; }
void CharacterBody3D::set_max_slides(int max_slides) { pimpl->max_slides = std::max(1, max_slides); }
int CharacterBody3D::get_max_slides() const { return pimpl->max_slides; }
void CharacterBody3D::set_floor_max_angle(float radians) { pimpl->floor_max_angle = radians; }
float CharacterBody3D::get_floor_max_angle() const { return pimpl->floor_max_angle; }
void CharacterBody3D::set_floor_stop_on_slope(bool enabled) { pimpl->floor_stop_on_slope = enabled; }
bool CharacterBody3D::get_floor_stop_on_slope() const { return pimpl->floor_stop_on_slope; }
void CharacterBody3D::set_up_direction(const double* up) { memcpy(pimpl->up_direction, up, 3*sizeof(double)); }
const double* CharacterBody3D::get_up_direction() const { return pimpl->up_direction; }
void CharacterBody3D::set_wall_min_angle(float radians) { pimpl->wall_min_angle = radians; }
float CharacterBody3D::get_wall_min_angle() const { return pimpl->wall_min_angle; }

void CharacterBody3D::move_and_slide() {
    move_and_slide_with_step(1.0/60.0); // assume 60 fps step
}
void CharacterBody3D::move_and_slide_with_step(double delta) {
    pimpl->slide_collisions.clear();
    pimpl->on_floor = false;
    pimpl->on_wall = false;
    pimpl->on_ceiling = false;
    memcpy(pimpl->current_velocity, pimpl->velocity, 3*sizeof(double));

    Transform3D global = get_global_transform();
    memcpy(pimpl->last_position, global.origin, 3*sizeof(double));
    double remaining_motion[3] = {pimpl->current_velocity[0]*delta,
                                  pimpl->current_velocity[1]*delta,
                                  pimpl->current_velocity[2]*delta};

    for (int slide = 0; slide < pimpl->max_slides; ++slide) {
        // Simulate collision detection (simplified: use sphere cast against world)
        // In real engine, we would query PhysicsServer. For demo, we assume a ground plane at y=0.
        double new_pos[3] = {pimpl->last_position[0] + remaining_motion[0],
                             pimpl->last_position[1] + remaining_motion[1],
                             pimpl->last_position[2] + remaining_motion[2]};
        // Check floor (Y axis)
        if (remaining_motion[1] < 0 && new_pos[1] < 0.0) {
            // hit floor
            SlideCollision col;
            col.position[0] = new_pos[0];
            col.position[1] = 0.0;
            col.position[2] = new_pos[2];
            col.normal[0] = 0; col.normal[1] = 1; col.normal[2] = 0;
            memcpy(col.velocity, pimpl->floor_velocity, 3*sizeof(double));
            pimpl->slide_collisions.push_back(col);
            remaining_motion[1] = 0.0;
            // adjust velocity after collision
            double vel_along_normal = pimpl->current_velocity[0]*col.normal[0] +
                                      pimpl->current_velocity[1]*col.normal[1] +
                                      pimpl->current_velocity[2]*col.normal[2];
            if (vel_along_normal < 0) {
                pimpl->current_velocity[0] -= vel_along_normal * col.normal[0];
                pimpl->current_velocity[1] -= vel_along_normal * col.normal[1];
                pimpl->current_velocity[2] -= vel_along_normal * col.normal[2];
            }
            pimpl->on_floor = true;
        }
        // Check walls (simplified: stop if X or Z hits boundaries)
        if (remaining_motion[0] != 0) {
            // dummy
        }
        // Update position
        global.origin[0] += remaining_motion[0];
        global.origin[1] += remaining_motion[1];
        global.origin[2] += remaining_motion[2];
        set_global_transform(global);
        // If no more motion, break
        if (remaining_motion[0]==0 && remaining_motion[1]==0 && remaining_motion[2]==0) break;
        // For next slide, set last position to new position
        memcpy(pimpl->last_position, global.origin, 3*sizeof(double));
    }
    pimpl->slide_count = (int)pimpl->slide_collisions.size();
    // Update floor velocity
    if (pimpl->on_floor && !pimpl->slide_collisions.empty()) {
        memcpy(pimpl->floor_velocity, pimpl->slide_collisions.back().velocity, 3*sizeof(double));
    }
}

int CharacterBody3D::get_slide_count() const { return pimpl->slide_count; }
void CharacterBody3D::get_slide_collision(int idx, double* out_position, double* out_normal, double* out_velocity) const {
    if (idx >= 0 && idx < pimpl->slide_count) {
        const auto& col = pimpl->slide_collisions[idx];
        memcpy(out_position, col.position, 3*sizeof(double));
        memcpy(out_normal, col.normal, 3*sizeof(double));
        memcpy(out_velocity, col.velocity, 3*sizeof(double));
    }
}

bool CharacterBody3D::is_on_floor() const { return pimpl->on_floor; }
bool CharacterBody3D::is_on_wall() const { return pimpl->on_wall; }
bool CharacterBody3D::is_on_ceiling() const { return pimpl->on_ceiling; }
bool CharacterBody3D::is_on_floor_only() const { return pimpl->on_floor && !pimpl->on_wall && !pimpl->on_ceiling; }
bool CharacterBody3D::is_on_wall_only() const { return pimpl->on_wall && !pimpl->on_floor && !pimpl->on_ceiling; }
bool CharacterBody3D::is_on_ceiling_only() const { return pimpl->on_ceiling && !pimpl->on_floor && !pimpl->on_wall; }

void CharacterBody3D::set_floor_velocity(const double* velocity) {
    memcpy(pimpl->floor_velocity, velocity, 3*sizeof(double));
}
const double* CharacterBody3D::get_floor_velocity() const { return pimpl->floor_velocity; }

void CharacterBody3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; PhysicsBody3D::set_cast_shadow(cast); }
void CharacterBody3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void CharacterBody3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; PhysicsBody3D::set_gi_mode(mode); }
void CharacterBody3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void CharacterBody3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void CharacterBody3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void CharacterBody3D::update_physics(double delta_time) {
    PhysicsBody3D::update_physics(delta_time);
    // The user is expected to call move_and_slide() manually each frame.
    // We do not call it automatically.
}

void CharacterBody3D::synchronize_render_server(double delta) {
    PhysicsBody3D::synchronize_render_server(delta);
    // Update any attached visual instance with new transform (already done by parent)
    // Also update GI contribution if emissive
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register as emissive source for GI (placeholder)
    }
}

} // namespace lighting