// include/xtu/godot/xsoftbody3d.hpp
// xtensor-unified - Soft body physics for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XSOFTBODY3D_HPP
#define XTU_GODOT_XSOFTBODY3D_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xphysics3d.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xmesh.hpp"
#include "xtu/parallel/xparallel.hpp"
#include "xtu/math/xlinalg.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class SoftBody3D;
class SoftBodyMesh;

// #############################################################################
// Soft body node types
// #############################################################################
enum class SoftBodyNodeType : uint8_t {
    NODE_TYPE_DYNAMIC = 0,
    NODE_TYPE_STATIC = 1,
    NODE_TYPE_PINNED = 2
};

// #############################################################################
// Soft body pin mode
// #############################################################################
enum class SoftBodyPinMode : uint8_t {
    PIN_MODE_DISABLED = 0,
    PIN_MODE_PERMANENT = 1,
    PIN_MODE_ATTACHED = 2
};

// #############################################################################
// Soft body update mode
// #############################################################################
enum class SoftBodyUpdateMode : uint8_t {
    UPDATE_MODE_DISABLED = 0,
    UPDATE_MODE_DEFORM = 1,
    UPDATE_MODE_DEFORM_STORE = 2
};

// #############################################################################
// Soft body collision mask flags
// #############################################################################
enum class SoftBodyCollisionMask : uint32_t {
    COLLISION_MASK_DEFAULT = 1 << 0,
    COLLISION_MASK_RIGID = 1 << 1,
    COLLISION_MASK_STATIC = 1 << 2,
    COLLISION_MASK_KINEMATIC = 1 << 3,
    COLLISION_MASK_CHARACTER = 1 << 4,
    COLLISION_MASK_ALL = 0xFFFFFFFF
};

// #############################################################################
// Soft body simulation parameters
// #############################################################################
struct SoftBodyParameters {
    float total_mass = 1.0f;
    float linear_stiffness = 0.5f;
    float area_angular_stiffness = 0.5f;
    float volume_stiffness = 0.5f;
    float pressure_coefficient = 0.0f;
    float damping_coefficient = 0.01f;
    float drag_coefficient = 0.0f;
    float pose_matching_coefficient = 0.0f;
    float collision_margin = 0.01f;
    int32_t simulation_precision = 5;
    float linear_damping = 0.01f;
    float angular_damping = 0.01f;
};

// #############################################################################
// Soft body node (vertex in the soft body mesh)
// #############################################################################
struct SoftBodyNode {
    vec3f position;
    vec3f previous_position;
    vec3f velocity;
    vec3f force;
    vec3f normal;
    float inv_mass = 1.0f;
    float area = 0.0f;
    SoftBodyNodeType type = SoftBodyNodeType::NODE_TYPE_DYNAMIC;
    uint32_t index = 0;
    bool is_pinned = false;
    NodePath attached_node_path;
    vec3f attached_local_offset;

    void apply_force(const vec3f& f) { force += f; }
    void apply_impulse(const vec3f& impulse) { velocity += impulse * inv_mass; }
};

// #############################################################################
// Soft body link (spring between two nodes)
// #############################################################################
struct SoftBodyLink {
    uint32_t node_a;
    uint32_t node_b;
    float rest_length;
    float stiffness = 1.0f;
    float damping = 0.1f;
    bool can_collide = true;

    float get_current_length(const std::vector<SoftBodyNode>& nodes) const {
        return (nodes[node_b].position - nodes[node_a].position).length();
    }
};

// #############################################################################
// Soft body face (triangle for volume preservation)
// #############################################################################
struct SoftBodyFace {
    uint32_t node_a;
    uint32_t node_b;
    uint32_t node_c;
    vec3f normal;
    float area;
    float rest_volume_contribution = 0.0f;

    void update_normal_and_area(const std::vector<SoftBodyNode>& nodes) {
        vec3f ab = nodes[node_b].position - nodes[node_a].position;
        vec3f ac = nodes[node_c].position - nodes[node_a].position;
        normal = normalize(cross(ab, ac));
        area = cross(ab, ac).length() * 0.5f;
    }

    vec3f get_center(const std::vector<SoftBodyNode>& nodes) const {
        return (nodes[node_a].position + nodes[node_b].position + nodes[node_c].position) / 3.0f;
    }
};

// #############################################################################
// Soft body tetrahedron (for volume preservation)
// #############################################################################
struct SoftBodyTetra {
    uint32_t node_a;
    uint32_t node_b;
    uint32_t node_c;
    uint32_t node_d;
    float rest_volume;
    mat3f inverse_rest_shape;

    float get_current_volume(const std::vector<SoftBodyNode>& nodes) const {
        vec3f ab = nodes[node_b].position - nodes[node_a].position;
        vec3f ac = nodes[node_c].position - nodes[node_a].position;
        vec3f ad = nodes[node_d].position - nodes[node_a].position;
        return std::abs(dot(cross(ab, ac), ad)) / 6.0f;
    }

    void compute_inverse_rest_shape(const std::vector<SoftBodyNode>& nodes) {
        vec3f ab = nodes[node_b].position - nodes[node_a].position;
        vec3f ac = nodes[node_c].position - nodes[node_a].position;
        vec3f ad = nodes[node_d].position - nodes[node_a].position;
        mat3f rest_shape = mat3f(
            ab.x(), ac.x(), ad.x(),
            ab.y(), ac.y(), ad.y(),
            ab.z(), ac.z(), ad.z()
        );
        inverse_rest_shape = inverse(rest_shape);
    }
};

// #############################################################################
// SoftBodyMesh - Soft body mesh resource
// #############################################################################
class SoftBodyMesh : public Resource {
    XTU_GODOT_REGISTER_CLASS(SoftBodyMesh, Resource)

private:
    std::vector<vec3f> m_vertices;
    std::vector<int32_t> m_indices;
    std::vector<vec3f> m_normals;
    std::vector<int32_t> m_pinned_vertices;
    aabb m_bounds;

public:
    static StringName get_class_static() { return StringName("SoftBodyMesh"); }

    void set_vertices(const std::vector<vec3f>& vertices) { m_vertices = vertices; }
    const std::vector<vec3f>& get_vertices() const { return m_vertices; }

    void set_indices(const std::vector<int32_t>& indices) { m_indices = indices; }
    const std::vector<int32_t>& get_indices() const { return m_indices; }

    void set_pinned_vertices(const std::vector<int32_t>& pinned) { m_pinned_vertices = pinned; }
    const std::vector<int32_t>& get_pinned_vertices() const { return m_pinned_vertices; }

    size_t get_vertex_count() const { return m_vertices.size(); }
    size_t get_face_count() const { return m_indices.size() / 3; }

    void compute_bounds() {
        m_bounds = aabb();
        for (const auto& v : m_vertices) m_bounds.extend(v);
    }

    aabb get_bounds() const { return m_bounds; }

    void generate_edges(std::vector<std::pair<uint32_t, uint32_t>>& edges) const {
        std::unordered_set<uint64_t> edge_set;
        for (size_t i = 0; i < m_indices.size(); i += 3) {
            uint32_t a = m_indices[i];
            uint32_t b = m_indices[i + 1];
            uint32_t c = m_indices[i + 2];
            add_edge(edge_set, a, b);
            add_edge(edge_set, b, c);
            add_edge(edge_set, c, a);
        }
        for (uint64_t e : edge_set) {
            edges.emplace_back(static_cast<uint32_t>(e >> 32), static_cast<uint32_t>(e & 0xFFFFFFFF));
        }
    }

private:
    void add_edge(std::unordered_set<uint64_t>& edges, uint32_t a, uint32_t b) {
        if (a > b) std::swap(a, b);
        edges.insert((static_cast<uint64_t>(a) << 32) | b);
    }
};

// #############################################################################
// SoftBody3D - Deformable physics body
// #############################################################################
class SoftBody3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(SoftBody3D, Node3D)

public:
    enum DisableMode : uint8_t {
        DISABLE_MODE_REMOVE = 0,
        DISABLE_MODE_KEEP_ACTIVE = 1,
        DISABLE_MODE_SLEEP = 2
    };

private:
    Ref<SoftBodyMesh> m_mesh;
    std::vector<SoftBodyNode> m_nodes;
    std::vector<SoftBodyLink> m_links;
    std::vector<SoftBodyFace> m_faces;
    std::vector<SoftBodyTetra> m_tetras;
    SoftBodyParameters m_params;
    RID m_space_rid;
    RID m_body_rid;
    float m_time_scale = 1.0f;
    float m_simulation_precision = 5.0f;
    float m_total_mass = 1.0f;
    float m_linear_stiffness = 0.5f;
    float m_pressure_coefficient = 0.0f;
    float m_damping_coefficient = 0.01f;
    float m_drag_coefficient = 0.0f;
    float m_pose_matching_coefficient = 0.0f;
    vec3f m_gravity = {0, -9.8f, 0};
    vec3f m_wind_velocity;
    bool m_ray_pickable = true;
    bool m_sleeping = false;
    float m_sleep_threshold = 0.05f;
    float m_sleep_time = 0.0f;
    DisableMode m_disable_mode = DISABLE_MODE_REMOVE;
    bool m_enabled = true;
    std::mutex m_mutex;
    std::vector<mat4f> m_rest_poses;

public:
    static StringName get_class_static() { return StringName("SoftBody3D"); }

    SoftBody3D() {
        m_body_rid = PhysicsServer3D::get_singleton()->create_body();
        PhysicsServer3D::get_singleton()->body_set_mode(m_body_rid, PhysicsServer3D::BODY_MODE_RIGID);
    }

    ~SoftBody3D() {
        PhysicsServer3D::get_singleton()->free_rid(m_body_rid);
    }

    void set_mesh(const Ref<SoftBodyMesh>& mesh) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_mesh = mesh;
        rebuild_from_mesh();
    }

    Ref<SoftBodyMesh> get_mesh() const { return m_mesh; }

    void set_total_mass(float mass) {
        m_total_mass = mass;
        update_masses();
    }
    float get_total_mass() const { return m_total_mass; }

    void set_linear_stiffness(float stiffness) { m_linear_stiffness = stiffness; }
    float get_linear_stiffness() const { return m_linear_stiffness; }

    void set_pressure_coefficient(float coeff) { m_pressure_coefficient = coeff; }
    float get_pressure_coefficient() const { return m_pressure_coefficient; }

    void set_damping_coefficient(float coeff) { m_damping_coefficient = coeff; }
    float get_damping_coefficient() const { return m_damping_coefficient; }

    void set_drag_coefficient(float coeff) { m_drag_coefficient = coeff; }
    float get_drag_coefficient() const { return m_drag_coefficient; }

    void set_pose_matching_coefficient(float coeff) { m_pose_matching_coefficient = coeff; }
    float get_pose_matching_coefficient() const { return m_pose_matching_coefficient; }

    void set_simulation_precision(int precision) { m_simulation_precision = static_cast<float>(precision); }
    int get_simulation_precision() const { return static_cast<int>(m_simulation_precision); }

    void set_gravity(const vec3f& gravity) { m_gravity = gravity; }
    vec3f get_gravity() const { return m_gravity; }

    void set_wind_velocity(const vec3f& wind) { m_wind_velocity = wind; }
    vec3f get_wind_velocity() const { return m_wind_velocity; }

    void set_ray_pickable(bool pickable) { m_ray_pickable = pickable; }
    bool is_ray_pickable() const { return m_ray_pickable; }

    void set_disable_mode(DisableMode mode) { m_disable_mode = mode; }
    DisableMode get_disable_mode() const { return m_disable_mode; }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void add_collision_exception_with(Node* node) {
        if (auto* body = dynamic_cast<PhysicsBody3DNode*>(node)) {
            // Add exception
        }
    }

    void remove_collision_exception_with(Node* node) {}

    void pin_vertex(int vertex_index, Node* attach_to = nullptr, const vec3f& local_offset = vec3f(0)) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (vertex_index < 0 || vertex_index >= static_cast<int>(m_nodes.size())) return;
        SoftBodyNode& node = m_nodes[vertex_index];
        node.type = SoftBodyNodeType::NODE_TYPE_PINNED;
        node.is_pinned = true;
        if (attach_to) {
            node.attached_node_path = attach_to->get_path();
            node.attached_local_offset = local_offset;
        }
    }

    void unpin_vertex(int vertex_index) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (vertex_index < 0 || vertex_index >= static_cast<int>(m_nodes.size())) return;
        SoftBodyNode& node = m_nodes[vertex_index];
        node.type = SoftBodyNodeType::NODE_TYPE_DYNAMIC;
        node.is_pinned = false;
    }

    bool is_vertex_pinned(int vertex_index) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return vertex_index >= 0 && vertex_index < static_cast<int>(m_nodes.size()) && m_nodes[vertex_index].is_pinned;
    }

    int get_vertex_count() const { return static_cast<int>(m_nodes.size()); }

    vec3f get_vertex_position(int vertex_index) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (vertex_index < 0 || vertex_index >= static_cast<int>(m_nodes.size())) return vec3f(0);
        return m_nodes[vertex_index].position;
    }

    void set_vertex_position(int vertex_index, const vec3f& position) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (vertex_index < 0 || vertex_index >= static_cast<int>(m_nodes.size())) return;
        m_nodes[vertex_index].position = position;
        m_nodes[vertex_index].previous_position = position;
    }

    vec3f get_vertex_velocity(int vertex_index) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (vertex_index < 0 || vertex_index >= static_cast<int>(m_nodes.size())) return vec3f(0);
        return m_nodes[vertex_index].velocity;
    }

    void apply_force_to_vertex(int vertex_index, const vec3f& force) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (vertex_index >= 0 && vertex_index < static_cast<int>(m_nodes.size())) {
            m_nodes[vertex_index].apply_force(force);
        }
    }

    void apply_impulse_to_vertex(int vertex_index, const vec3f& impulse) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (vertex_index >= 0 && vertex_index < static_cast<int>(m_nodes.size())) {
            m_nodes[vertex_index].apply_impulse(impulse);
        }
    }

    void _physics_process(double delta) override {
        if (!m_enabled || !m_mesh.is_valid()) return;
        std::lock_guard<std::mutex> lock(m_mutex);
        float dt = static_cast<float>(delta) * m_time_scale;
        if (dt <= 0.0f) return;

        int substeps = static_cast<int>(m_simulation_precision);
        float sub_dt = dt / static_cast<float>(substeps);

        for (int s = 0; s < substeps; ++s) {
            // Update pinned nodes from attached objects
            update_pinned_nodes();
            // Verlet integration
            integrate(sub_dt);
            // Apply forces
            apply_forces(sub_dt);
            // Solve constraints
            for (int iter = 0; iter < 5; ++iter) {
                solve_links(sub_dt);
                solve_volume(sub_dt);
                solve_collisions(sub_dt);
            }
            // Update velocities
            update_velocities(sub_dt);
        }

        // Check sleep state
        check_sleep(dt);
        // Update visual mesh
        update_visual_mesh();
    }

private:
    void rebuild_from_mesh() {
        if (!m_mesh.is_valid()) return;
        m_nodes.clear();
        m_links.clear();
        m_faces.clear();
        m_tetras.clear();

        // Create nodes from vertices
        const auto& vertices = m_mesh->get_vertices();
        m_nodes.resize(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            m_nodes[i].position = vertices[i];
            m_nodes[i].previous_position = vertices[i];
            m_nodes[i].index = static_cast<uint32_t>(i);
        }

        // Create faces
        const auto& indices = m_mesh->get_indices();
        for (size_t i = 0; i < indices.size(); i += 3) {
            SoftBodyFace face;
            face.node_a = indices[i];
            face.node_b = indices[i + 1];
            face.node_c = indices[i + 2];
            face.update_normal_and_area(m_nodes);
            m_faces.push_back(face);
        }

        // Create links (edges)
        std::vector<std::pair<uint32_t, uint32_t>> edges;
        m_mesh->generate_edges(edges);
        for (const auto& edge : edges) {
            SoftBodyLink link;
            link.node_a = edge.first;
            link.node_b = edge.second;
            link.rest_length = (m_nodes[link.node_b].position - m_nodes[link.node_a].position).length();
            m_links.push_back(link);
        }

        // Pin specified vertices
        for (int32_t idx : m_mesh->get_pinned_vertices()) {
            if (idx >= 0 && idx < static_cast<int32_t>(m_nodes.size())) {
                m_nodes[idx].type = SoftBodyNodeType::NODE_TYPE_PINNED;
                m_nodes[idx].is_pinned = true;
            }
        }

        update_masses();
    }

    void update_masses() {
        if (m_nodes.empty()) return;
        float mass_per_node = m_total_mass / static_cast<float>(m_nodes.size());
        for (auto& node : m_nodes) {
            if (node.type == SoftBodyNodeType::NODE_TYPE_DYNAMIC) {
                node.inv_mass = 1.0f / mass_per_node;
            } else {
                node.inv_mass = 0.0f;
            }
        }
    }

    void update_pinned_nodes() {
        mat4f global_transform = get_global_transform();
        for (auto& node : m_nodes) {
            if (node.is_pinned && !node.attached_node_path.is_empty()) {
                Node* attached = get_node(node.attached_node_path);
                if (attached && attached->is_inside_tree()) {
                    vec3f world_pos;
                    if (auto* n3d = dynamic_cast<Node3D*>(attached)) {
                        world_pos = n3d->get_global_position() + node.attached_local_offset;
                    }
                    node.position = inverse(global_transform) * vec4f(world_pos.x(), world_pos.y(), world_pos.z(), 1.0f);
                    node.previous_position = node.position;
                }
            }
        }
    }

    void integrate(float dt) {
        for (auto& node : m_nodes) {
            if (node.type != SoftBodyNodeType::NODE_TYPE_DYNAMIC) continue;
            vec3f temp = node.position;
            vec3f acceleration = node.force * node.inv_mass + m_gravity;
            node.position += (node.position - node.previous_position) * (1.0f - m_damping_coefficient) + acceleration * dt * dt;
            node.previous_position = temp;
            node.force = vec3f(0);
        }
    }

    void apply_forces(float dt) {
        vec3f relative_wind = m_wind_velocity;
        for (auto& face : m_faces) {
            vec3f center = face.get_center(m_nodes);
            vec3f face_velocity = (m_nodes[face.node_a].velocity + m_nodes[face.node_b].velocity + m_nodes[face.node_c].velocity) / 3.0f;
            vec3f rel_vel = relative_wind - face_velocity;
            float vn = dot(rel_vel, face.normal);
            if (vn > 0) {
                vec3f force = face.normal * face.area * vn * m_drag_coefficient;
                m_nodes[face.node_a].apply_force(force / 3.0f);
                m_nodes[face.node_b].apply_force(force / 3.0f);
                m_nodes[face.node_c].apply_force(force / 3.0f);
            }
        }
    }

    void solve_links(float dt) {
        parallel::parallel_for(0, m_links.size(), [&](size_t i) {
            SoftBodyLink& link = m_links[i];
            SoftBodyNode& a = m_nodes[link.node_a];
            SoftBodyNode& b = m_nodes[link.node_b];
            if (a.type == SoftBodyNodeType::NODE_TYPE_STATIC && b.type == SoftBodyNodeType::NODE_TYPE_STATIC) return;
            vec3f delta = b.position - a.position;
            float current_length = delta.length();
            if (current_length < 1e-6f) return;
            vec3f direction = delta / current_length;
            float correction = (current_length - link.rest_length) * link.stiffness * m_linear_stiffness;
            float total_inv_mass = a.inv_mass + b.inv_mass;
            if (total_inv_mass > 0) {
                vec3f impulse = direction * correction / total_inv_mass;
                if (a.type == SoftBodyNodeType::NODE_TYPE_DYNAMIC) a.position += impulse * a.inv_mass;
                if (b.type == SoftBodyNodeType::NODE_TYPE_DYNAMIC) b.position -= impulse * b.inv_mass;
            }
            // Damping
            vec3f vel_diff = b.velocity - a.velocity;
            float damping_force = dot(vel_diff, direction) * link.damping;
            if (a.type == SoftBodyNodeType::NODE_TYPE_DYNAMIC) a.velocity += direction * damping_force * a.inv_mass;
            if (b.type == SoftBodyNodeType::NODE_TYPE_DYNAMIC) b.velocity -= direction * damping_force * b.inv_mass;
        });
    }

    void solve_volume(float dt) {
        if (m_pressure_coefficient <= 0.0f) return;
        for (auto& face : m_faces) {
            face.update_normal_and_area(m_nodes);
        }
        float total_volume = 0.0f;
        for (const auto& face : m_faces) {
            vec3f center = face.get_center(m_nodes);
            total_volume += dot(center, face.normal) * face.area / 3.0f;
        }
        if (total_volume <= 0.0f) return;
        float pressure = m_pressure_coefficient / total_volume;
        for (auto& face : m_faces) {
            vec3f force = face.normal * face.area * pressure;
            m_nodes[face.node_a].apply_force(force / 3.0f);
            m_nodes[face.node_b].apply_force(force / 3.0f);
            m_nodes[face.node_c].apply_force(force / 3.0f);
        }
    }

    void solve_collisions(float dt) {
        // Collision with physics bodies handled by PhysicsServer
        // Simplified ground plane collision
        for (auto& node : m_nodes) {
            if (node.type != SoftBodyNodeType::NODE_TYPE_DYNAMIC) continue;
            if (node.position.y() < 0.0f) {
                node.position.y() = 0.0f;
                if (node.velocity.y() < 0) node.velocity.y() *= -0.5f;
            }
        }
    }

    void update_velocities(float dt) {
        float inv_dt = 1.0f / dt;
        for (auto& node : m_nodes) {
            if (node.type != SoftBodyNodeType::NODE_TYPE_DYNAMIC) continue;
            node.velocity = (node.position - node.previous_position) * inv_dt;
        }
    }

    void check_sleep(float dt) {
        float max_velocity = 0.0f;
        for (const auto& node : m_nodes) {
            max_velocity = std::max(max_velocity, node.velocity.length());
        }
        if (max_velocity < m_sleep_threshold) {
            m_sleep_time += dt;
            if (m_sleep_time > 0.5f) m_sleeping = true;
        } else {
            m_sleep_time = 0.0f;
            m_sleeping = false;
        }
    }

    void update_visual_mesh() {
        // Update rendering mesh with deformed vertices
    }
};

} // namespace godot

// Bring into main namespace
using godot::SoftBody3D;
using godot::SoftBodyMesh;
using godot::SoftBodyNode;
using godot::SoftBodyLink;
using godot::SoftBodyFace;
using godot::SoftBodyTetra;
using godot::SoftBodyParameters;
using godot::SoftBodyNodeType;
using godot::SoftBodyPinMode;
using godot::SoftBodyUpdateMode;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XSOFTBODY3D_HPP