// Name : lighting enhancement
// File : scene/3d/character_body_3d_ext.cpp 36 of 60
// Description : Implementation of CharacterBody3DExt with move_and_slide, floor/wall detection,
//               slide collision resolution, and full lighting integration.
#include "character_body_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include <cmath>
#include <vector>

// ------------------------------------------------------------------------
// Simple collision shape: plane (ground) and axis-aligned boxes (walls)
// In a real engine, this would query the physics server.
// For demonstration, we define a few static obstacles.
// ------------------------------------------------------------------------
struct SimpleCollisionWorld {
    // Ground plane at y = 0
    static bool intersect_ground(const Vector3 &from, const Vector3 &to, double &t, Vector3 &normal) {
        if (from.y <= 0.0 && to.y <= 0.0) return false;
        if (from.y >= 0.0 && to.y >= 0.0) return false;
        t = -from.y / (to.y - from.y);
        if (t < 0.0 || t > 1.0) return false;
        normal = Vector3(0, 1, 0);
        return true;
    }

    // Box obstacles (position, half extents)
    struct Box {
        Vector3 center;
        Vector3 half_size;
    };
    static std::vector<Box> boxes;

    static bool intersect_box(const Vector3 &from, const Vector3 &to, const Box &box, double &t, Vector3 &normal) {
        // AABB ray intersection (slab method)
        Vector3 dir = to - from;
        double tmin = -1e30, tmax = 1e30;
        for (int axis = 0; axis < 3; ++axis) {
            double inv_dir = (dir[axis] == 0.0) ? 1e30 : 1.0 / dir[axis];
            double t1 = (box.center[axis] - box.half_size[axis] - from[axis]) * inv_dir;
            double t2 = (box.center[axis] + box.half_size[axis] - from[axis]) * inv_dir;
            if (t1 > t2) std::swap(t1, t2);
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) return false;
        }
        if (tmin < 0.0 || tmin > 1.0) return false;
        t = tmin;
        // Compute normal (axis of minimal entry)
        Vector3 contact = from + dir * t;
        for (int axis = 0; axis < 3; ++axis) {
            if (Math::abs(contact[axis] - (box.center[axis] + box.half_size[axis])) < 1e-4) {
                normal = Vector3();
                normal[axis] = 1.0;
                return true;
            }
            if (Math::abs(contact[axis] - (box.center[axis] - box.half_size[axis])) < 1e-4) {
                normal = Vector3();
                normal[axis] = -1.0;
                return true;
            }
        }
        return true;
    }
};
std::vector<SimpleCollisionWorld::Box> SimpleCollisionWorld::boxes; // will be populated in constructor

struct CharacterBody3DExt::Impl {
    Vector3 velocity;
    int max_slides = 4;
    float floor_max_angle = Math::deg_to_rad(45.0f);
    bool floor_stop_on_slope = true;
    Vector3 up_direction = Vector3(0, 1, 0);
    float wall_min_angle = 0.0f; // radians (0 = no wall detection)
    Vector3 floor_velocity;

    // Runtime state
    bool on_floor = false;
    bool on_wall = false;
    bool on_ceiling = false;
    int slide_count = 0;
    struct SlideCollision {
        Vector3 position;
        Vector3 normal;
        Vector3 velocity;
    };
    std::vector<SlideCollision> slide_collisions;

    // Helper: test if a point is inside any static collision (for penetration recovery)
    bool test_penetration(const Vector3 &point, float radius) {
        // Ground
        if (point.y - radius < 0.0) return true;
        // Boxes
        for (const auto &box : SimpleCollisionWorld::boxes) {
            AABB box_aabb(box.center - box.half_size, box.half_size * 2.0);
            if (box_aabb.encloses(point)) return true;
        }
        return false;
    }

    // Sweep sphere from "from" to "to", radius r, return first hit t (0..1) and normal
    bool sweep_test(const Vector3 &from, const Vector3 &to, float radius, double &t_out, Vector3 &normal_out) {
        Vector3 dir = to - from;
        double hit_t = 1.0;
        Vector3 hit_normal;
        bool hit = false;

        // Ground
        double ground_t;
        Vector3 ground_n;
        if (SimpleCollisionWorld::intersect_ground(from, to, ground_t, ground_n)) {
            if (ground_t < hit_t) {
                hit_t = ground_t;
                hit_normal = ground_n;
                hit = true;
            }
        }
        // Boxes
        for (const auto &box : SimpleCollisionWorld::boxes) {
            // Expand box by radius
            Vector3 expanded_half = box.half_size + Vector3(radius, radius, radius);
            Vector3 expanded_center = box.center;
            // Ray intersection with expanded box
            double t_box;
            Vector3 normal_box;
            if (intersect_ray_aabb(from, dir, expanded_center - expanded_half, expanded_center + expanded_half, t_box, normal_box)) {
                if (t_box < hit_t) {
                    hit_t = t_box;
                    hit_normal = normal_box;
                    hit = true;
                }
            }
        }
        if (hit) {
            t_out = hit_t;
            normal_out = hit_normal;
            return true;
        }
        return false;
    }

    bool intersect_ray_aabb(const Vector3 &origin, const Vector3 &dir, const Vector3 &aabb_min, const Vector3 &aabb_max, double &t, Vector3 &normal) {
        Vector3 inv_dir(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
        double tmin = -1e30, tmax = 1e30;
        for (int i = 0; i < 3; ++i) {
            double t1 = (aabb_min[i] - origin[i]) * inv_dir[i];
            double t2 = (aabb_max[i] - origin[i]) * inv_dir[i];
            if (t1 > t2) std::swap(t1, t2);
            if (t1 > tmin) tmin = t1;
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) return false;
        }
        if (tmin < 0.0 || tmin > 1.0) return false;
        t = tmin;
        // compute normal
        Vector3 contact = origin + dir * t;
        for (int i = 0; i < 3; ++i) {
            if (Math::abs(contact[i] - aabb_min[i]) < 1e-4) {
                normal = Vector3();
                normal[i] = -1.0;
                return true;
            }
            if (Math::abs(contact[i] - aabb_max[i]) < 1e-4) {
                normal = Vector3();
                normal[i] = 1.0;
                return true;
            }
        }
        return true;
    }

    bool is_floor(const Vector3 &normal) const {
        return normal.angle_to(up_direction) <= floor_max_angle + 1e-4;
    }

    bool is_wall(const Vector3 &normal) const {
        if (wall_min_angle <= 0.0) return false;
        double angle_to_up = normal.angle_to(up_direction);
        return angle_to_up > floor_max_angle && angle_to_up < Math_PI - floor_max_angle;
    }

    bool is_ceiling(const Vector3 &normal) const {
        double angle_to_up = normal.angle_to(up_direction);
        return angle_to_up > Math_PI - floor_max_angle;
    }
};

CharacterBody3DExt::CharacterBody3DExt() {
    pimpl = new Impl();
    // Add some example obstacles
    if (SimpleCollisionWorld::boxes.empty()) {
        SimpleCollisionWorld::Box box1;
        box1.center = Vector3(2, 0.5, 2);
        box1.half_size = Vector3(0.5, 0.5, 0.5);
        SimpleCollisionWorld::boxes.push_back(box1);
        SimpleCollisionWorld::Box box2;
        box2.center = Vector3(-2, 0.5, -2);
        box2.half_size = Vector3(0.5, 0.5, 0.5);
        SimpleCollisionWorld::boxes.push_back(box2);
    }
}

CharacterBody3DExt::~CharacterBody3DExt() {
    delete pimpl;
}

void CharacterBody3DExt::set_velocity(const Vector3 &p_velocity) {
    pimpl->velocity = p_velocity;
}
Vector3 CharacterBody3DExt::get_velocity() const { return pimpl->velocity; }

void CharacterBody3DExt::set_max_slides(int p_max_slides) {
    pimpl->max_slides = p_max_slides;
}
int CharacterBody3DExt::get_max_slides() const { return pimpl->max_slides; }

void CharacterBody3DExt::set_floor_max_angle(float p_radians) {
    pimpl->floor_max_angle = p_radians;
}
float CharacterBody3DExt::get_floor_max_angle() const { return pimpl->floor_max_angle; }

void CharacterBody3DExt::set_floor_stop_on_slope(bool p_enabled) {
    pimpl->floor_stop_on_slope = p_enabled;
}
bool CharacterBody3DExt::get_floor_stop_on_slope() const { return pimpl->floor_stop_on_slope; }

void CharacterBody3DExt::set_up_direction(const Vector3 &p_up) {
    pimpl->up_direction = p_up.normalized();
}
Vector3 CharacterBody3DExt::get_up_direction() const { return pimpl->up_direction; }

void CharacterBody3DExt::set_wall_min_angle(float p_radians) {
    pimpl->wall_min_angle = p_radians;
}
float CharacterBody3DExt::get_wall_min_angle() const { return pimpl->wall_min_angle; }

void CharacterBody3DExt::move_and_slide() {
    move_and_slide_with_step(1.0 / 60.0);
}

void CharacterBody3DExt::move_and_slide_with_step(double p_delta) {
    if (p_delta <= 0.0) return;
    pimpl->slide_collisions.clear();
    pimpl->on_floor = false;
    pimpl->on_wall = false;
    pimpl->on_ceiling = false;

    Transform3D global = get_global_transform();
    Vector3 start_pos = global.origin;
    Vector3 remaining_velocity = pimpl->velocity * p_delta;
    Vector3 original_velocity = remaining_velocity;

    for (int slide = 0; slide < pimpl->max_slides; ++slide) {
        double closest_t = 1.0;
        Vector3 closest_normal;
        bool hit = false;
        // Sweep test
        Vector3 end_pos = start_pos + remaining_velocity;
        double t;
        Vector3 normal;
        if (pimpl->sweep_test(start_pos, end_pos, 0.5, t, normal)) {
            hit = true;
            if (t < closest_t) {
                closest_t = t;
                closest_normal = normal;
            }
        }
        if (!hit) {
            // No collision: move full
            start_pos += remaining_velocity;
            break;
        }
        // Move to collision point (with small margin)
        start_pos += remaining_velocity * (closest_t - 0.001);
        // Record collision
        pimpl->slide_collisions.push_back({start_pos, closest_normal, remaining_velocity});
        // Compute remaining velocity after bounce/slide
        Vector3 vel_along_normal = remaining_velocity.dot(closest_normal);
        remaining_velocity = remaining_velocity - vel_along_normal * closest_normal;
        // Apply friction (optional: reduce parallel component)
        // For now, we just slide without loss.
        // Update floor/wall/ceiling flags
        if (pimpl->is_floor(closest_normal)) pimpl->on_floor = true;
        if (pimpl->is_wall(closest_normal)) pimpl->on_wall = true;
        if (pimpl->is_ceiling(closest_normal)) pimpl->on_ceiling = true;
        // Stop if velocity is nearly zero
        if (remaining_velocity.length_squared() < 1e-6) break;
        // Reduce max slides (prevent infinite loop)
        if (slide == pimpl->max_slides - 1) {
            start_pos += remaining_velocity; // final push
        }
    }
    // Final transform
    global.origin = start_pos;
    set_global_transform(global);
    pimpl->slide_count = pimpl->slide_collisions.size();

    // Apply floor velocity if on floor and moving platform
    if (pimpl->on_floor && pimpl->floor_velocity != Vector3()) {
        Vector3 new_origin = get_global_transform().origin + pimpl->floor_velocity * p_delta;
        Transform3D new_global = get_global_transform();
        new_global.origin = new_origin;
        set_global_transform(new_global);
    }
}

int CharacterBody3DExt::get_slide_count() const { return pimpl->slide_collisions.size(); }
void CharacterBody3DExt::get_slide_collision(int p_idx, Vector3 &r_position, Vector3 &r_normal, Vector3 &r_velocity) const {
    if (p_idx < 0 || p_idx >= (int)pimpl->slide_collisions.size()) return;
    r_position = pimpl->slide_collisions[p_idx].position;
    r_normal = pimpl->slide_collisions[p_idx].normal;
    r_velocity = pimpl->slide_collisions[p_idx].velocity;
}

bool CharacterBody3DExt::is_on_floor() const { return pimpl->on_floor; }
bool CharacterBody3DExt::is_on_wall() const { return pimpl->on_wall; }
bool CharacterBody3DExt::is_on_ceiling() const { return pimpl->on_ceiling; }
bool CharacterBody3DExt::is_on_floor_only() const { return pimpl->on_floor && !pimpl->on_wall && !pimpl->on_ceiling; }
bool CharacterBody3DExt::is_on_wall_only() const { return pimpl->on_wall && !pimpl->on_floor && !pimpl->on_ceiling; }
bool CharacterBody3DExt::is_on_ceiling_only() const { return pimpl->on_ceiling && !pimpl->on_floor && !pimpl->on_wall; }

void CharacterBody3DExt::set_floor_velocity(const Vector3 &p_velocity) {
    pimpl->floor_velocity = p_velocity;
}
Vector3 CharacterBody3DExt::get_floor_velocity() const { return pimpl->floor_velocity; }

void CharacterBody3DExt::set_cast_shadow(bool p_cast) {
    PhysicsBody3DExt::set_cast_shadow(p_cast);
}
void CharacterBody3DExt::set_gi_mode(int p_mode) {
    PhysicsBody3DExt::set_gi_mode(p_mode);
}
void CharacterBody3DExt::set_gi_contribution(float p_amount) {
    PhysicsBody3DExt::set_gi_contribution(p_amount);
}
void CharacterBody3DExt::set_emissive(const Color &p_color, float p_intensity) {
    PhysicsBody3DExt::set_emissive(p_color, p_intensity);
}

void CharacterBody3DExt::sync_character_body() {
    PhysicsBody3DExt::sync_physics_body();
}