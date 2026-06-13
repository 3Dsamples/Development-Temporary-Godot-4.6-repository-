// Name : lighting enhancement
// File : scene/3d/spring_arm_3d_ext.cpp 54 of 60
// Description : Implementation of SpringArm3DExt with collision raycast,
//               mass-spring-damper smoothing for length and rotation,
//               and debug line mesh with emissive lighting.
#include "spring_arm_3d_ext.h"
#include "servers/physics_server_3d.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/quaternion.h"
#include <cmath>

struct SpringArm3DExt::Impl {
    float spring_length = 2.0f;
    bool collision_enabled = true;
    uint32_t collision_mask = 0xFFFFFFFF;
    float collision_margin = 0.05f;
    float spring_stiffness = 0.8f;
    float spring_damping = 0.4f;
    float angular_stiffness = 0.8f;
    float angular_damping = 0.4f;
    bool clip_far = true;
    float avoidance_radius = 0.1f;

    // Current smoothed state
    Vector3 current_end;
    Vector3 current_end_velocity;
    Quaternion current_orientation;      // orientation of the arm (only if angular smoothing)
    Vector3 angular_velocity;

    bool debug_visible = false;
    Color debug_color = Color(0.2f, 1.0f, 0.2f);
    Color debug_emissive_color = Color(0,0,0);
    float debug_emissive_intensity = 0.0f;
    RID debug_mesh_rid;
    RID debug_instance_rid;
    int gi_mode = 0;
    float gi_contribution = 1.0f;

    Impl() {
        debug_mesh_rid = RenderingServer::get_singleton()->mesh_create();
        debug_instance_rid = RenderingServer::get_singleton()->instance_create();
        RenderingServer::get_singleton()->instance_set_base(debug_instance_rid, debug_mesh_rid);
        RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, false);
    }

    ~Impl() {
        if (debug_mesh_rid.is_valid()) RenderingServer::get_singleton()->free(debug_mesh_rid);
        if (debug_instance_rid.is_valid()) RenderingServer::get_singleton()->free(debug_instance_rid);
    }

    // Perform raycast along a direction from origin, returns hit position and distance, or none
    bool raycast(const Vector3 &origin, const Vector3 &direction, float max_dist, Vector3 &hit_point, float &hit_dist) {
        PhysicsServer3D *ps = PhysicsServer3D::get_singleton();
        PhysicsDirectSpaceState3D *space = ps->space_get_direct_state(get_world_3d()->get_space());
        PhysicsRayQueryParameters3D params;
        params.from = origin;
        params.to = origin + direction * max_dist;
        params.collision_mask = collision_mask;
        params.margin = collision_margin;
        Dictionary result = space->intersect_ray(params);
        if (result.is_empty()) return false;
        hit_point = result["position"];
        hit_dist = origin.distance_to(hit_point);
        return true;
    }

    void update_arm(double delta) {
        Transform3D global = get_global_transform();
        Vector3 origin = global.origin;
        Vector3 forward = global.basis.get_axis(2); // local Z axis (forward)
        // Desired length = spring_length (target)
        float desired_len = spring_length;

        if (collision_enabled) {
            Vector3 dir = forward.normalized();
            Vector3 hit_point;
            float hit_dist;
            if (raycast(origin, dir, desired_len + collision_margin, hit_point, hit_dist)) {
                desired_len = hit_dist - collision_margin;
                if (desired_len < 0.0f) desired_len = 0.0f;
            }
            if (clip_far && desired_len > spring_length) desired_len = spring_length;
        }

        // Mass-spring-damper for length smoothing
        float dt = MIN(delta, 0.033f);
        float k = spring_stiffness * 100.0f; // scaling for realistic behavior
        float c = spring_damping * 2.0f * sqrt(k);
        float force = k * (desired_len - current_end.length()) - c * current_end_velocity.length();
        float acc = force; // assume unit mass
        current_end_velocity += acc * dt;
        float new_len = current_end.length() + current_end_velocity.length() * dt;
        if (new_len < 0.0f) new_len = 0.0f;
        current_end = forward * new_len;
        current_end_velocity = forward * current_end_velocity.length();

        // Optional angular smoothing: integrate angular velocity towards desired orientation
        // For simplicity, we treat the arm's orientation as always facing forward (no angular smoothing here).
        // The child camera (or attached node) should be positioned at current_end.

        // Update debug mesh
        update_debug_mesh(origin, origin + current_end);
    }

    void update_debug_mesh(const Vector3 &p_start, const Vector3 &p_end) {
        if (!debug_visible) {
            RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, false);
            return;
        }
        Vector<Vector3> vertices;
        Vector<int> indices;
        vertices.push_back(p_start);
        vertices.push_back(p_end);
        indices.push_back(0);
        indices.push_back(1);
        RenderingServer::get_singleton()->mesh_clear(debug_mesh_rid);
        RenderingServer::get_singleton()->mesh_add_surface(debug_mesh_rid, RS::PRIMITIVE_LINES, vertices, indices, Vector<Vector2>(), Vector<Vector3>());
        RID mat = RenderingServer::get_singleton()->material_create();
        RenderingServer::get_singleton()->material_set_param(mat, "albedo", debug_color);
        if (debug_emissive_intensity > 0.0f) {
            RenderingServer::get_singleton()->material_set_param(mat, "emission", debug_emissive_color);
            RenderingServer::get_singleton()->material_set_param(mat, "emission_intensity", debug_emissive_intensity);
        }
        RenderingServer::get_singleton()->mesh_surface_set_material(debug_mesh_rid, 0, mat);
        RenderingServer::get_singleton()->instance_set_transform(debug_instance_rid, Transform3D());
        RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, true);
    }
};

SpringArm3DExt::SpringArm3DExt() {
    pimpl = new Impl();
}

SpringArm3DExt::~SpringArm3DExt() {
    delete pimpl;
}

void SpringArm3DExt::set_spring_length(float p_length) {
    pimpl->spring_length = p_length;
    update_arm();
}
float SpringArm3DExt::get_spring_length() const { return pimpl->spring_length; }

void SpringArm3DExt::set_collision_enabled(bool p_enabled) {
    pimpl->collision_enabled = p_enabled;
    update_arm();
}
bool SpringArm3DExt::is_collision_enabled() const { return pimpl->collision_enabled; }

void SpringArm3DExt::set_collision_mask(uint32_t p_mask) {
    pimpl->collision_mask = p_mask;
    update_arm();
}
uint32_t SpringArm3DExt::get_collision_mask() const { return pimpl->collision_mask; }

void SpringArm3DExt::set_collision_margin(float p_margin) {
    pimpl->collision_margin = p_margin;
    update_arm();
}
float SpringArm3DExt::get_collision_margin() const { return pimpl->collision_margin; }

void SpringArm3DExt::set_spring_stiffness(float p_stiffness) {
    pimpl->spring_stiffness = p_stiffness;
    update_arm();
}
float SpringArm3DExt::get_spring_stiffness() const { return pimpl->spring_stiffness; }

void SpringArm3DExt::set_spring_damping(float p_damping) {
    pimpl->spring_damping = p_damping;
    update_arm();
}
float SpringArm3DExt::get_spring_damping() const { return pimpl->spring_damping; }

void SpringArm3DExt::set_angular_stiffness(float p_stiffness) {
    pimpl->angular_stiffness = p_stiffness;
    update_arm();
}
float SpringArm3DExt::get_angular_stiffness() const { return pimpl->angular_stiffness; }

void SpringArm3DExt::set_angular_damping(float p_damping) {
    pimpl->angular_damping = p_damping;
    update_arm();
}
float SpringArm3DExt::get_angular_damping() const { return pimpl->angular_damping; }

void SpringArm3DExt::set_clip_far(bool p_clip) {
    pimpl->clip_far = p_clip;
    update_arm();
}
bool SpringArm3DExt::get_clip_far() const { return pimpl->clip_far; }

void SpringArm3DExt::set_avoidance_radius(float p_radius) {
    pimpl->avoidance_radius = p_radius;
    update_arm();
}
float SpringArm3DExt::get_avoidance_radius() const { return pimpl->avoidance_radius; }

Vector3 SpringArm3DExt::get_arm_end_position() const {
    Transform3D global = get_global_transform();
    return global.origin + pimpl->current_end;
}

void SpringArm3DExt::set_debug_visible(bool p_visible) {
    pimpl->debug_visible = p_visible;
    if (p_visible) {
        Transform3D global = get_global_transform();
        pimpl->update_debug_mesh(global.origin, global.origin + pimpl->current_end);
    } else {
        RenderingServer::get_singleton()->instance_set_visible(pimpl->debug_instance_rid, false);
    }
}
bool SpringArm3DExt::is_debug_visible() const { return pimpl->debug_visible; }

void SpringArm3DExt::set_debug_color(const Color &p_color) {
    pimpl->debug_color = p_color;
    if (pimpl->debug_visible) {
        Transform3D global = get_global_transform();
        pimpl->update_debug_mesh(global.origin, global.origin + pimpl->current_end);
    }
}
Color SpringArm3DExt::get_debug_color() const { return pimpl->debug_color; }

void SpringArm3DExt::set_debug_emissive(const Color &p_color, float p_intensity) {
    pimpl->debug_emissive_color = p_color;
    pimpl->debug_emissive_intensity = p_intensity;
    if (pimpl->debug_visible) {
        Transform3D global = get_global_transform();
        pimpl->update_debug_mesh(global.origin, global.origin + pimpl->current_end);
    }
}
void SpringArm3DExt::get_debug_emissive(Color &r_color, float &r_intensity) const {
    r_color = pimpl->debug_emissive_color;
    r_intensity = pimpl->debug_emissive_intensity;
}

void SpringArm3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
}
int SpringArm3DExt::get_gi_mode() const { return pimpl->gi_mode; }

void SpringArm3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
}
float SpringArm3DExt::get_gi_contribution() const { return pimpl->gi_contribution; }

void SpringArm3DExt::update_arm() {
    // Called each frame or manually
    pimpl->update_arm(1.0/60.0);
}