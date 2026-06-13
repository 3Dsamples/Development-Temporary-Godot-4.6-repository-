// spring_arm_3d.cpp
#include "spring_arm_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace lighting {

// ============================================================================
// Helper: Raycast against scene physics (simplified – in real engine uses
// PhysicsServer3D, here we simulate by checking a dummy ground plane)
// ============================================================================
static bool raycast(const double* origin, const double* direction, double max_dist,
                    uint32_t mask, double margin, double* out_hit_point) {
    // Simplified: check intersection with plane y = 0
    if (direction[1] >= 0) return false; // pointing upward
    double t = -origin[1] / direction[1];
    if (t < 0 || t > max_dist) return false;
    out_hit_point[0] = origin[0] + direction[0] * t;
    out_hit_point[1] = 0;
    out_hit_point[2] = origin[2] + direction[2] * t;
    return true;
}

// ============================================================================
// SpringArm3D implementation
// ============================================================================
struct SpringArm3D::Impl {
    double spring_length = 2.0;
    bool collision_enabled = true;
    uint32_t collision_mask = 0xFFFFFFFF;
    double collision_margin = 0.05;
    float spring_stiffness = 0.8f;
    float spring_damping = 0.4f;
    float angular_stiffness = 0.8f;
    float angular_damping = 0.4f;
    bool clip_far = true;
    float avoidance_radius = 0.1f;

    bool debug_visible = false;
    float debug_color[3] = {0.2f, 1.0f, 0.2f};
    bool emissive_debug = false;
    float emissive_intensity = 0.2f;

    // Current arm target (world position) after collision adjustment
    double arm_end_position[3] = {0,0,0};
    // Desired arm length (without collision)
    double desired_length = 2.0;
    // Current smoothed length (for spring)
    double current_length = 2.0;
    double current_velocity = 0.0; // for linear spring
    // Angular smoothing for direction (quaternion representation)
    double current_rot[9] = {1,0,0, 0,1,0, 0,0,1};
    double current_rot_vel[3] = {0,0,0};

    // Debug mesh for line
    int64_t debug_mesh_rid = -1;
    int64_t debug_instance_rid = -1;
    bool dirty = true;

    void update_arm(double delta);
    void update_debug_mesh();
};

SpringArm3D::SpringArm3D() : pimpl(std::make_unique<Impl>()) {}
SpringArm3D::~SpringArm3D() = default;

void SpringArm3D::set_spring_length(double length) {
    pimpl->spring_length = std::max(0.0, length);
    pimpl->desired_length = pimpl->spring_length;
    pimpl->dirty = true;
}
double SpringArm3D::get_spring_length() const { return pimpl->spring_length; }

void SpringArm3D::set_collision_enabled(bool enabled) { pimpl->collision_enabled = enabled; }
bool SpringArm3D::is_collision_enabled() const { return pimpl->collision_enabled; }
void SpringArm3D::set_collision_mask(uint32_t mask) { pimpl->collision_mask = mask; }
uint32_t SpringArm3D::get_collision_mask() const { return pimpl->collision_mask; }
void SpringArm3D::set_collision_margin(double margin) { pimpl->collision_margin = margin; }
double SpringArm3D::get_collision_margin() const { return pimpl->collision_margin; }

void SpringArm3D::set_spring_stiffness(float stiffness) {
    pimpl->spring_stiffness = std::clamp(stiffness, 0.0f, 1.0f);
}
float SpringArm3D::get_spring_stiffness() const { return pimpl->spring_stiffness; }
void SpringArm3D::set_spring_damping(float damping) {
    pimpl->spring_damping = std::clamp(damping, 0.0f, 1.0f);
}
float SpringArm3D::get_spring_damping() const { return pimpl->spring_damping; }

void SpringArm3D::set_angular_stiffness(float stiffness) {
    pimpl->angular_stiffness = std::clamp(stiffness, 0.0f, 1.0f);
}
float SpringArm3D::get_angular_stiffness() const { return pimpl->angular_stiffness; }
void SpringArm3D::set_angular_damping(float damping) {
    pimpl->angular_damping = std::clamp(damping, 0.0f, 1.0f);
}
float SpringArm3D::get_angular_damping() const { return pimpl->angular_damping; }

void SpringArm3D::set_clip_far(bool clip) { pimpl->clip_far = clip; }
bool SpringArm3D::get_clip_far() const { return pimpl->clip_far; }
void SpringArm3D::set_avoidance_radius(float radius) { pimpl->avoidance_radius = std::max(0.0f, radius); }
float SpringArm3D::get_avoidance_radius() const { return pimpl->avoidance_radius; }

void SpringArm3D::set_debug_visible(bool visible) { pimpl->debug_visible = visible; pimpl->dirty = true; }
bool SpringArm3D::is_debug_visible() const { return pimpl->debug_visible; }
void SpringArm3D::set_debug_color(const float* rgb) { memcpy(pimpl->debug_color, rgb, 3*sizeof(float)); pimpl->dirty = true; }
void SpringArm3D::get_debug_color(float* out_rgb) const { memcpy(out_rgb, pimpl->debug_color, 3*sizeof(float)); }
void SpringArm3D::set_emissive_debug(bool enable, float intensity) {
    pimpl->emissive_debug = enable;
    pimpl->emissive_intensity = intensity;
    pimpl->dirty = true;
}
bool SpringArm3D::is_emissive_debug() const { return pimpl->emissive_debug; }

void SpringArm3D::get_arm_end_position(double* out_pos) const {
    memcpy(out_pos, pimpl->arm_end_position, 3*sizeof(double));
}

void SpringArm3D::Impl::update_arm(double delta) {
    // Get current global transform of the spring arm
    Transform3D global = get_global_transform();
    double origin[3] = {global.origin[0], global.origin[1], global.origin[2]};
    // Direction: the spring arm's local forward (usually -Z for Godot, but we assume +Z for simplicity)
    double dir[3] = {global.basis[6], global.basis[7], global.basis[8]}; // column-major? we use row-major. For row-major, Z axis = basis[8], basis[9], basis[10]. Let's assume row-major: indices 8,9,10.
    double forward[3] = {global.basis[8], global.basis[9], global.basis[10]};
    // Normalize
    double len = sqrt(forward[0]*forward[0] + forward[1]*forward[1] + forward[2]*forward[2]);
    if (len > 1e-6) {
        forward[0] /= len; forward[1] /= len; forward[2] /= len;
    }

    // Desired length = spring_length (target)
    double target_len = spring_length;
    if (collision_enabled) {
        // Cast ray from origin along forward direction to find collision
        double hit_point[3];
        bool hit = raycast(origin, forward, target_len + margin, collision_mask, collision_margin, hit_point);
        if (hit) {
            double hit_dist = sqrt((hit_point[0]-origin[0])*(hit_point[0]-origin[0]) +
                                   (hit_point[1]-origin[1])*(hit_point[1]-origin[1]) +
                                   (hit_point[2]-origin[2])*(hit_point[2]-origin[2]));
            target_len = hit_dist - margin;
            if (target_len < 0.0) target_len = 0.0;
        }
        if (clip_far && target_len > spring_length) target_len = spring_length;
    }
    desired_length = target_len;

    // Spring physics (mass‑spring‑damper)
    double dt = std::min(delta, 0.033);
    double stiffness_s = spring_stiffness;
    double damping_s = spring_damping;
    // Critical damping factor: 2*sqrt(k)
    double k = stiffness_s * 100.0; // scaling
    double c = damping_s * 2.0 * sqrt(k);
    double force = k * (desired_length - current_length) - c * current_velocity;
    double acc = force; // assume unit mass
    current_velocity += acc * dt;
    current_length += current_velocity * dt;
    if (current_length < 0.0) current_length = 0.0;

    // Compute final end point
    arm_end_position[0] = origin[0] + forward[0] * current_length;
    arm_end_position[1] = origin[1] + forward[1] * current_length;
    arm_end_position[2] = origin[2] + forward[2] * current_length;

    // Optionally apply angular smoothing to the spring arm's own orientation
    // (to smooth camera rotation, not implemented here)
}

void SpringArm3D::Impl::update_debug_mesh() {
    if (!debug_visible) {
        if (debug_instance_rid != -1) {
            // RenderingServer::instance_set_visible(debug_instance_rid, false);
        }
        return;
    }

    // Generate a line mesh from origin to arm_end_position
    Transform3D global = get_global_transform();
    double start[3] = {global.origin[0], global.origin[1], global.origin[2]};
    double end[3] = {arm_end_position[0], arm_end_position[1], arm_end_position[2]};

    std::vector<double> vertices = {start[0], start[1], start[2], end[0], end[1], end[2]};
    std::vector<int> indices = {0,1};

    // Create or update mesh with line primitive
    if (debug_mesh_rid == -1) {
        // debug_mesh_rid = RenderingServer::mesh_create();
    }
    // RenderingServer::mesh_add_surface(debug_mesh_rid, RenderingServer::PRIMITIVE_LINES, vertices, indices);
    // Set material: unlit with debug_color, optionally emissive.
    // Transform: identity (vertices are in world space, but better to use local space).
    // For simplicity, we use world space vertices and identity instance transform.
    if (debug_instance_rid == -1) {
        // debug_instance_rid = RenderingServer::instance_create();
    }
    // RenderingServer::instance_set_base(debug_instance_rid, debug_mesh_rid);
    // RenderingServer::instance_set_transform(debug_instance_rid, Transform3D::identity());
    // RenderingServer::instance_set_visible(debug_instance_rid, true);
}

void SpringArm3D::process(double delta) {
    Node3D::process(delta);
    pimpl->update_arm(delta);
    // If the spring arm has a child node (e.g., camera), we could automatically
    // update its transform to the arm_end_position. Godot does this automatically
    // by design, but we can also do it manually: if we find a child Node3D,
    // set its global transform to arm_end_position with same rotation as arm.
    // For simplicity, we rely on the user to attach a child.
    // Mark transform dirty so that children get updated? Not needed if we directly set child.
    // We'll just update the spring arm's own transform to point to the end? No.
    if (pimpl->debug_visible) pimpl->update_debug_mesh();
}

void SpringArm3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (pimpl->dirty) {
        if (pimpl->debug_visible) pimpl->update_debug_mesh();
        pimpl->dirty = false;
    }
    // Update debug instance transform if using local space mesh.
}

} // namespace lighting