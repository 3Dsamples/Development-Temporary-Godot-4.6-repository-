// include/xtu/godot/xocclusion.hpp
// xtensor-unified - Occlusion culling system for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XOCCLUSION_HPP
#define XTU_GODOT_XOCCLUSION_HPP

#include <algorithm>
#include <atomic>
#include <bitset>
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
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"
#include "xtu/graphics/xintersection.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class Occluder3D;
class OccluderInstance3D;
class RoomManager;
class Room;
class Portal;
class OcclusionCuller;

// #############################################################################
// Occluder shape types
// #############################################################################
enum class OccluderShapeType : uint8_t {
    SHAPE_SPHERE = 0,
    SHAPE_BOX = 1,
    SHAPE_POLYGON = 2
};

// #############################################################################
// Portal connection flags
// #############################################################################
enum class PortalFlags : uint32_t {
    FLAG_NONE = 0,
    FLAG_TWO_WAY = 1 << 0,
    FLAG_AUTOMATIC = 1 << 1,
    FLAG_DISABLED = 1 << 2
};

inline PortalFlags operator|(PortalFlags a, PortalFlags b) {
    return static_cast<PortalFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline bool has_flag(PortalFlags flags, PortalFlags test) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(test)) != 0;
}

// #############################################################################
// Room priority for sorting
// #############################################################################
enum class RoomPriority : uint8_t {
    PRIORITY_LOW = 0,
    PRIORITY_MEDIUM = 1,
    PRIORITY_HIGH = 2,
    PRIORITY_ALWAYS_VISIBLE = 3
};

// #############################################################################
// Occluder polygon vertex
// #############################################################################
struct OccluderPolygon {
    std::vector<vec3f> vertices;
    std::vector<int32_t> indices;
    bool double_sided = true;

    plane get_plane() const {
        if (vertices.size() < 3) return plane();
        return plane(vertices[0], vertices[1], vertices[2]);
    }

    aabb get_aabb() const {
        aabb bounds;
        for (const auto& v : vertices) bounds.extend(v);
        return bounds;
    }
};

// #############################################################################
// Occluder3D - Occlusion shape resource
// #############################################################################
class Occluder3D : public Resource {
    XTU_GODOT_REGISTER_CLASS(Occluder3D, Resource)

private:
    std::vector<OccluderPolygon> m_polygons;
    OccluderShapeType m_shape_type = OccluderShapeType::SHAPE_POLYGON;
    vec3f m_sphere_center;
    float m_sphere_radius = 1.0f;
    vec3f m_box_extents = {1, 1, 1};

public:
    static StringName get_class_static() { return StringName("Occluder3D"); }

    void set_polygons(const std::vector<OccluderPolygon>& polygons) {
        m_polygons = polygons;
        m_shape_type = OccluderShapeType::SHAPE_POLYGON;
    }

    const std::vector<OccluderPolygon>& get_polygons() const { return m_polygons; }

    int get_polygon_count() const { return static_cast<int>(m_polygons.size()); }

    void set_as_sphere(const vec3f& center, float radius) {
        m_shape_type = OccluderShapeType::SHAPE_SPHERE;
        m_sphere_center = center;
        m_sphere_radius = radius;
        m_polygons.clear();
    }

    void set_as_box(const vec3f& extents) {
        m_shape_type = OccluderShapeType::SHAPE_BOX;
        m_box_extents = extents;
        m_polygons.clear();
    }

    OccluderShapeType get_shape_type() const { return m_shape_type; }

    aabb get_local_aabb() const {
        if (m_shape_type == OccluderShapeType::SHAPE_SPHERE) {
            return aabb(m_sphere_center - vec3f(m_sphere_radius),
                        m_sphere_center + vec3f(m_sphere_radius));
        } else if (m_shape_type == OccluderShapeType::SHAPE_BOX) {
            return aabb(-m_box_extents, m_box_extents);
        } else {
            aabb bounds;
            for (const auto& poly : m_polygons) {
                bounds.extend(poly.get_aabb());
            }
            return bounds;
        }
    }
};

// #############################################################################
// Portal - Connection between rooms
// #############################################################################
class Portal : public Node3D {
    XTU_GODOT_REGISTER_CLASS(Portal, Node3D)

private:
    vec2f m_size = {2, 2};
    PortalFlags m_flags = PortalFlags::FLAG_TWO_WAY;
    Room* m_room_a = nullptr;
    Room* m_room_b = nullptr;
    bool m_active = true;
    plane m_plane;
    std::vector<vec3f> m_world_vertices;
    aabb m_world_aabb;

public:
    static StringName get_class_static() { return StringName("Portal"); }

    void set_size(const vec2f& size) { m_size = size; update_geometry(); }
    vec2f get_size() const { return m_size; }

    void set_two_way(bool enabled) {
        if (enabled) m_flags = m_flags | PortalFlags::FLAG_TWO_WAY;
        else m_flags = static_cast<PortalFlags>(static_cast<uint32_t>(m_flags) & ~static_cast<uint32_t>(PortalFlags::FLAG_TWO_WAY));
    }
    bool is_two_way() const { return has_flag(m_flags, PortalFlags::FLAG_TWO_WAY); }

    void set_disabled(bool disabled) {
        if (disabled) m_flags = m_flags | PortalFlags::FLAG_DISABLED;
        else m_flags = static_cast<PortalFlags>(static_cast<uint32_t>(m_flags) & ~static_cast<uint32_t>(PortalFlags::FLAG_DISABLED));
    }
    bool is_disabled() const { return has_flag(m_flags, PortalFlags::FLAG_DISABLED); }

    void set_active(bool active) { m_active = active; }
    bool is_active() const { return m_active; }

    Room* get_room_a() const { return m_room_a; }
    Room* get_room_b() const { return m_room_b; }
    void set_rooms(Room* a, Room* b) { m_room_a = a; m_room_b = b; }

    Room* get_other_room(const Room* from) const {
        if (from == m_room_a) return m_room_b;
        if (from == m_room_b) return m_room_a;
        return nullptr;
    }

    const plane& get_world_plane() const { return m_plane; }
    const std::vector<vec3f>& get_world_vertices() const { return m_world_vertices; }
    const aabb& get_world_aabb() const { return m_world_aabb; }

    void update_geometry() {
        mat4f transform = get_global_transform();
        vec3f pos = vec3f(transform[3][0], transform[3][1], transform[3][2]);
        vec3f right = vec3f(transform[0][0], transform[0][1], transform[0][2]) * m_size.x() * 0.5f;
        vec3f up = vec3f(transform[1][0], transform[1][1], transform[1][2]) * m_size.y() * 0.5f;
        vec3f normal = vec3f(transform[2][0], transform[2][1], transform[2][2]);

        m_world_vertices = {
            pos - right - up,
            pos + right - up,
            pos + right + up,
            pos - right + up
        };

        m_plane = plane(m_world_vertices[0], m_world_vertices[1], m_world_vertices[2]);
        m_world_aabb = aabb();
        for (const auto& v : m_world_vertices) m_world_aabb.extend(v);
    }

    bool is_visible_from(const vec3f& view_pos, const frustum& view_frustum) const {
        if (!m_active || is_disabled()) return false;
        // Frustum culling
        if (!view_frustum.intersects(m_world_aabb)) return false;
        // Back-face culling (if portal faces away)
        vec3f to_viewer = view_pos - m_world_vertices[0];
        if (dot(m_plane.normal, to_viewer) < 0) return false;
        return true;
    }

    void _ready() override { update_geometry(); }
    void _notification(int p_what) override {
        if (p_what == NOTIFICATION_TRANSFORM_CHANGED) update_geometry();
    }
};

// #############################################################################
// Room - Spatial partition container
// #############################################################################
class Room : public Node3D {
    XTU_GODOT_REGISTER_CLASS(Room, Node3D)

private:
    aabb m_bounds;
    RoomPriority m_priority = RoomPriority::PRIORITY_MEDIUM;
    std::vector<Portal*> m_portals;
    std::vector<OccluderInstance3D*> m_occluders;
    std::vector<Node3D*> m_contained_objects;
    bool m_bounds_auto = true;
    bool m_active = true;
    Color m_debug_color = {1, 1, 1, 1};

public:
    static StringName get_class_static() { return StringName("Room"); }

    void set_bounds(const aabb& bounds) { m_bounds = bounds; m_bounds_auto = false; }
    aabb get_bounds() const { return m_bounds; }

    void set_priority(RoomPriority priority) { m_priority = priority; }
    RoomPriority get_priority() const { return m_priority; }

    void set_active(bool active) { m_active = active; }
    bool is_active() const { return m_active; }

    void add_portal(Portal* portal) {
        if (std::find(m_portals.begin(), m_portals.end(), portal) == m_portals.end()) {
            m_portals.push_back(portal);
        }
    }

    void remove_portal(Portal* portal) {
        auto it = std::find(m_portals.begin(), m_portals.end(), portal);
        if (it != m_portals.end()) m_portals.erase(it);
    }

    const std::vector<Portal*>& get_portals() const { return m_portals; }

    void add_occluder(OccluderInstance3D* occluder) {
        if (std::find(m_occluders.begin(), m_occluders.end(), occluder) == m_occluders.end()) {
            m_occluders.push_back(occluder);
        }
    }

    const std::vector<OccluderInstance3D*>& get_occluders() const { return m_occluders; }

    void add_object(Node3D* obj) {
        if (std::find(m_contained_objects.begin(), m_contained_objects.end(), obj) == m_contained_objects.end()) {
            m_contained_objects.push_back(obj);
        }
    }

    const std::vector<Node3D*>& get_objects() const { return m_contained_objects; }

    void update_bounds_from_children() {
        if (!m_bounds_auto) return;
        m_bounds = aabb();
        for (Node* child = get_child(0); child; child = child->get_next_sibling()) {
            if (auto* n3d = dynamic_cast<Node3D*>(child)) {
                aabb child_bounds = n3d->get_global_transform() * n3d->get_aabb();
                m_bounds.extend(child_bounds);
            }
        }
    }

    bool contains_point(const vec3f& point) const {
        return m_bounds.contains(point);
    }

    bool is_visible_from(const vec3f& view_pos, const frustum& view_frustum) const {
        if (!m_active) return false;
        return view_frustum.intersects(m_bounds);
    }

    void _ready() override {
        if (m_bounds_auto) update_bounds_from_children();
        // Register with RoomManager
        RoomManager::get_singleton()->register_room(this);
    }

    void _exit_tree() override {
        RoomManager::get_singleton()->unregister_room(this);
    }

    void _notification(int p_what) override {
        if (p_what == NOTIFICATION_TRANSFORM_CHANGED && m_bounds_auto) {
            update_bounds_from_children();
        }
    }
};

// #############################################################################
// OccluderInstance3D - Instanced occluder in the scene
// #############################################################################
class OccluderInstance3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(OccluderInstance3D, Node3D)

private:
    Ref<Occluder3D> m_occluder;
    std::vector<OccluderPolygon> m_world_polygons;
    aabb m_world_aabb;
    bool m_enabled = true;
    uint32_t m_cull_mask = 0xFFFFFFFF;

public:
    static StringName get_class_static() { return StringName("OccluderInstance3D"); }

    void set_occluder(const Ref<Occluder3D>& occluder) {
        m_occluder = occluder;
        update_world_polygons();
    }

    Ref<Occluder3D> get_occluder() const { return m_occluder; }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void set_cull_mask(uint32_t mask) { m_cull_mask = mask; }
    uint32_t get_cull_mask() const { return m_cull_mask; }

    const std::vector<OccluderPolygon>& get_world_polygons() const { return m_world_polygons; }
    const aabb& get_world_aabb() const { return m_world_aabb; }

    void update_world_polygons() {
        m_world_polygons.clear();
        m_world_aabb = aabb();
        if (!m_occluder.is_valid()) return;
        mat4f transform = get_global_transform();
        for (const auto& poly : m_occluder->get_polygons()) {
            OccluderPolygon world_poly;
            world_poly.double_sided = poly.double_sided;
            for (const auto& v : poly.vertices) {
                vec4f tv = transform * vec4f(v.x(), v.y(), v.z(), 1.0f);
                world_poly.vertices.push_back(vec3f(tv.x(), tv.y(), tv.z()));
            }
            world_poly.indices = poly.indices;
            m_world_polygons.push_back(world_poly);
            m_world_aabb.extend(world_poly.get_aabb());
        }
    }

    void _ready() override {
        update_world_polygons();
        Room* parent_room = find_parent_room();
        if (parent_room) parent_room->add_occluder(this);
    }

    void _exit_tree() override {
        Room* parent_room = find_parent_room();
        if (parent_room) parent_room->remove_occluder(this);
    }

    void _notification(int p_what) override {
        if (p_what == NOTIFICATION_TRANSFORM_CHANGED) update_world_polygons();
    }

private:
    Room* find_parent_room() const {
        Node* parent = get_parent();
        while (parent) {
            if (auto* room = dynamic_cast<Room*>(parent)) return room;
            parent = parent->get_parent();
        }
        return nullptr;
    }
};

// #############################################################################
// RoomManager - Global room/portal manager singleton
// #############################################################################
class RoomManager : public Node {
    XTU_GODOT_REGISTER_CLASS(RoomManager, Node)

private:
    static RoomManager* s_singleton;
    std::vector<Room*> m_rooms;
    std::unordered_map<Node3D*, Room*> m_object_room_map;
    std::unordered_set<Room*> m_visible_rooms;
    std::mutex m_mutex;
    bool m_active = true;
    float m_portal_depth_limit = 16.0f;
    bool m_debug_enabled = false;

public:
    static RoomManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("RoomManager"); }

    RoomManager() { s_singleton = this; }
    ~RoomManager() { s_singleton = nullptr; }

    void register_room(Room* room) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (std::find(m_rooms.begin(), m_rooms.end(), room) == m_rooms.end()) {
            m_rooms.push_back(room);
        }
    }

    void unregister_room(Room* room) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = std::find(m_rooms.begin(), m_rooms.end(), room);
        if (it != m_rooms.end()) m_rooms.erase(it);
    }

    const std::vector<Room*>& get_rooms() const { return m_rooms; }

    Room* find_room_at_point(const vec3f& point) const {
        for (Room* room : m_rooms) {
            if (room->is_active() && room->contains_point(point)) {
                return room;
            }
        }
        return nullptr;
    }

    void set_active(bool active) { m_active = active; }
    bool is_active() const { return m_active; }

    void set_debug_enabled(bool enabled) { m_debug_enabled = enabled; }
    bool is_debug_enabled() const { return m_debug_enabled; }

    std::vector<Node3D*> cull_objects(const graphics::camera& camera, const std::vector<Node3D*>& objects) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<Node3D*> visible;
        if (!m_active || m_rooms.empty()) return objects;

        vec3f cam_pos = camera.get_position();
        frustum cam_frustum = frustum(camera.get_projection_matrix() * camera.get_view_matrix());

        Room* start_room = find_room_at_point(cam_pos);
        if (!start_room) {
            // Camera outside all rooms - use traditional frustum culling
            for (Node3D* obj : objects) {
                if (cam_frustum.intersects(obj->get_global_aabb())) {
                    visible.push_back(obj);
                }
            }
            return visible;
        }

        m_visible_rooms.clear();
        traverse_portals(start_room, cam_pos, cam_frustum, 0);
        m_visible_rooms.insert(start_room);

        // Collect visible objects from all visible rooms
        for (Room* room : m_visible_rooms) {
            for (Node3D* obj : room->get_objects()) {
                aabb world_aabb = obj->get_global_aabb();
                // Occlusion test
                if (cam_frustum.intersects(world_aabb) && !is_occluded(world_aabb, cam_pos, room)) {
                    visible.push_back(obj);
                }
            }
        }
        return visible;
    }

    void traverse_portals(Room* room, const vec3f& view_pos, const frustum& view_frustum, int depth) {
        if (depth > 16) return; // Prevent infinite recursion
        for (Portal* portal : room->get_portals()) {
            if (!portal->is_active() || portal->is_disabled()) continue;
            if (!portal->is_visible_from(view_pos, view_frustum)) continue;
            Room* other = portal->get_other_room(room);
            if (other && other->is_active() && m_visible_rooms.find(other) == m_visible_rooms.end()) {
                // Create reduced frustum through portal
                frustum portal_frustum = clip_frustum_to_portal(view_frustum, portal);
                m_visible_rooms.insert(other);
                traverse_portals(other, view_pos, portal_frustum, depth + 1);
            }
        }
    }

    frustum clip_frustum_to_portal(const frustum& f, Portal* portal) {
        // Simplified: return original frustum for now
        return f;
    }

    bool is_occluded(const aabb& object_bounds, const vec3f& view_pos, Room* room) {
        // Test against occluders in the same room
        for (OccluderInstance3D* occluder : room->get_occluders()) {
            if (!occluder->is_enabled()) continue;
            if (test_occlusion(object_bounds, view_pos, occluder)) {
                return true;
            }
        }
        return false;
    }

    bool test_occlusion(const aabb& object_bounds, const vec3f& view_pos, OccluderInstance3D* occluder) {
        // Simplified occlusion test
        aabb occluder_aabb = occluder->get_world_aabb();
        if (!occluder_aabb.intersects(object_bounds)) return false;
        // Ray casting from view to object corners
        return false;
    }

    void debug_draw() {
        if (!m_debug_enabled) return;
        // Draw room bounds and portals
    }

    void _process(double delta) override {
        if (m_debug_enabled) debug_draw();
    }
};

// #############################################################################
// OcclusionCuller - Software occlusion culling helper
// #############################################################################
class OcclusionCuller {
private:
    struct OcclusionQuery {
        aabb bounds;
        uint32_t id;
        bool visible = true;
    };

    std::vector<OcclusionQuery> m_queries;
    std::vector<uint32_t> m_visible_indices;
    std::mutex m_mutex;

public:
    void add_query(uint32_t id, const aabb& bounds) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queries.push_back({bounds, id, true});
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queries.clear();
    }

    std::vector<uint32_t> cull(const vec3f& view_pos, const std::vector<OccluderInstance3D*>& occluders) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_visible_indices.clear();
        // Simple frustum culling (occlusion requires depth buffer)
        for (const auto& q : m_queries) {
            bool occluded = false;
            for (OccluderInstance3D* occ : occluders) {
                if (!occ->is_enabled()) continue;
                // Test if query bounds are fully behind occluder
                // Placeholder: simplified visibility test
            }
            if (!occluded) m_visible_indices.push_back(q.id);
        }
        return m_visible_indices;
    }
};

} // namespace godot

// Bring into main namespace
using godot::Occluder3D;
using godot::OccluderInstance3D;
using godot::Room;
using godot::Portal;
using godot::RoomManager;
using godot::OcclusionCuller;
using godot::OccluderShapeType;
using godot::PortalFlags;
using godot::RoomPriority;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XOCCLUSION_HPP