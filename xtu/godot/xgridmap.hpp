// include/xtu/godot/xgridmap.hpp
// xtensor-unified - GridMap 3D tile system for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XGRIDMAP_HPP
#define XTU_GODOT_XGRIDMAP_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xphysics3d.hpp"
#include "xtu/godot/xnavigation_mesh.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xmesh.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class GridMap;
class GridMapLibrary;
class GridMapItem;

// #############################################################################
// GridMap orientation (rotation)
// #############################################################################
enum class GridMapOrientation : uint8_t {
    ORIENTATION_0 = 0,
    ORIENTATION_90 = 1,
    ORIENTATION_180 = 2,
    ORIENTATION_270 = 3
};

// #############################################################################
// GridMap octant (for navigation mesh baking)
// #############################################################################
enum class GridMapOctant : uint8_t {
    OCTANT_0 = 0,
    OCTANT_1 = 1,
    OCTANT_2 = 2,
    OCTANT_3 = 3,
    OCTANT_4 = 4,
    OCTANT_5 = 5,
    OCTANT_6 = 6,
    OCTANT_7 = 7
};

// #############################################################################
// GridMap cell data
// #############################################################################
struct GridMapCell {
    int32_t item_id = -1;
    GridMapOrientation orientation = GridMapOrientation::ORIENTATION_0;
    uint8_t octant = 0;
    bool valid = false;
};

// #############################################################################
// GridMapLibrary - Collection of mesh items for grid map
// #############################################################################
class GridMapLibrary : public Resource {
    XTU_GODOT_REGISTER_CLASS(GridMapLibrary, Resource)

public:
    struct Item {
        String name;
        Ref<Mesh> mesh;
        Ref<Material> material;
        Ref<Material> material_override;
        Ref<Texture2D> icon;
        vec3f mesh_offset;
        vec3f mesh_scale = {1, 1, 1};
        vec3f mesh_rotation;
        Ref<Shape3D> collision_shape;
        Ref<NavigationMesh> navigation_mesh;
        float navigation_mesh_transform_scale = 1.0f;
        bool valid = false;
    };

private:
    std::vector<Item> m_items;
    std::unordered_map<String, int32_t> m_name_to_id;
    std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("GridMapLibrary"); }

    int32_t add_item(const String& name) {
        std::lock_guard<std::mutex> lock(m_mutex);
        int32_t id = static_cast<int32_t>(m_items.size());
        Item item;
        item.name = name;
        item.valid = true;
        m_items.push_back(item);
        m_name_to_id[name] = id;
        return id;
    }

    void remove_item(int32_t id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (id >= 0 && id < static_cast<int32_t>(m_items.size())) {
            m_name_to_id.erase(m_items[id].name);
            m_items[id].valid = false;
        }
    }

    int32_t find_item_by_name(const String& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_name_to_id.find(name);
        return it != m_name_to_id.end() ? it->second : -1;
    }

    int32_t get_item_count() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return static_cast<int32_t>(m_items.size());
    }

    Item* get_item(int32_t id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        return (id >= 0 && id < static_cast<int32_t>(m_items.size())) ? &m_items[id] : nullptr;
    }

    const Item* get_item(int32_t id) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return (id >= 0 && id < static_cast<int32_t>(m_items.size())) ? &m_items[id] : nullptr;
    }

    void set_item_mesh(int32_t id, const Ref<Mesh>& mesh) {
        if (auto* item = get_item(id)) item->mesh = mesh;
    }

    void set_item_material(int32_t id, const Ref<Material>& material) {
        if (auto* item = get_item(id)) item->material = material;
    }

    void set_item_collision_shape(int32_t id, const Ref<Shape3D>& shape) {
        if (auto* item = get_item(id)) item->collision_shape = shape;
    }

    void set_item_navigation_mesh(int32_t id, const Ref<NavigationMesh>& mesh) {
        if (auto* item = get_item(id)) item->navigation_mesh = mesh;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_items.clear();
        m_name_to_id.clear();
    }
};

// #############################################################################
// GridMap - 3D grid-based map
// #############################################################################
class GridMap : public Node3D {
    XTU_GODOT_REGISTER_CLASS(GridMap, Node3D)

private:
    Ref<GridMapLibrary> m_library;
    std::unordered_map<vec3i, GridMapCell> m_cells;
    vec3f m_cell_size = {2, 2, 2};
    vec3i m_cell_center_offset;
    float m_bake_navigation = true;
    bool m_centered = true;
    uint32_t m_collision_layer = 1;
    uint32_t m_collision_mask = 1;
    float m_collision_priority = 1.0f;
    bool m_use_kinematic_bodies = false;
    
    mutable std::mutex m_mutex;
    std::unordered_map<vec3i, RID> m_collision_bodies;
    std::unordered_map<vec3i, RID> m_mesh_instances;
    RID m_navigation_region;
    bool m_dirty = true;

public:
    static StringName get_class_static() { return StringName("GridMap"); }

    GridMap() {
        m_navigation_region = NavigationServer3D::get_singleton()->region_create();
    }

    ~GridMap() {
        clear_collision();
        clear_mesh_instances();
        NavigationServer3D::get_singleton()->free_rid(m_navigation_region);
    }

    void set_library(const Ref<GridMapLibrary>& library) {
        m_library = library;
        mark_dirty();
    }

    Ref<GridMapLibrary> get_library() const { return m_library; }

    void set_cell_size(const vec3f& size) {
        m_cell_size = vec3f(std::max(0.01f, size.x()), std::max(0.01f, size.y()), std::max(0.01f, size.z()));
        mark_dirty();
    }

    vec3f get_cell_size() const { return m_cell_size; }

    void set_centered(bool centered) { m_centered = centered; mark_dirty(); }
    bool is_centered() const { return m_centered; }

    void set_cell_item(const vec3i& pos, int32_t item_id, GridMapOrientation orientation = GridMapOrientation::ORIENTATION_0) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (item_id < 0) {
            m_cells.erase(pos);
        } else {
            GridMapCell cell;
            cell.item_id = item_id;
            cell.orientation = orientation;
            cell.valid = true;
            m_cells[pos] = cell;
        }
        mark_dirty();
    }

    int32_t get_cell_item(const vec3i& pos) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cells.find(pos);
        return it != m_cells.end() ? it->second.item_id : -1;
    }

    GridMapOrientation get_cell_orientation(const vec3i& pos) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cells.find(pos);
        return it != m_cells.end() ? it->second.orientation : GridMapOrientation::ORIENTATION_0;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cells.clear();
        mark_dirty();
    }

    std::vector<vec3i> get_used_cells() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<vec3i> result;
        result.reserve(m_cells.size());
        for (const auto& kv : m_cells) {
            result.push_back(kv.first);
        }
        return result;
    }

    aabb get_cell_aabb(const vec3i& pos) const {
        vec3f origin = map_to_world(pos);
        vec3f half = m_cell_size * 0.5f;
        if (!m_centered) {
            origin += half;
        }
        return aabb(origin - half, origin + half);
    }

    vec3f map_to_world(const vec3i& pos) const {
        vec3f result = vec3f(pos.x() * m_cell_size.x(), pos.y() * m_cell_size.y(), pos.z() * m_cell_size.z());
        if (m_centered) {
            result += m_cell_size * 0.5f;
        }
        return result;
    }

    vec3i world_to_map(const vec3f& pos) const {
        vec3f adjusted = pos;
        if (m_centered) {
            adjusted -= m_cell_size * 0.5f;
        }
        return vec3i(
            static_cast<int32_t>(std::floor(adjusted.x() / m_cell_size.x())),
            static_cast<int32_t>(std::floor(adjusted.y() / m_cell_size.y())),
            static_cast<int32_t>(std::floor(adjusted.z() / m_cell_size.z()))
        );
    }

    void update_meshes() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_dirty) return;
        
        clear_mesh_instances();
        clear_collision();
        
        if (!m_library.is_valid()) return;
        
        for (const auto& kv : m_cells) {
            const vec3i& pos = kv.first;
            const GridMapCell& cell = kv.second;
            
            auto* item = m_library->get_item(cell.item_id);
            if (!item || !item->valid || !item->mesh.is_valid()) continue;
            
            create_mesh_instance(pos, cell, *item);
            create_collision_body(pos, cell, *item);
        }
        
        if (m_bake_navigation) {
            bake_navigation_mesh();
        }
        
        m_dirty = false;
    }

    void set_bake_navigation(bool enable) { m_bake_navigation = enable; mark_dirty(); }
    bool is_baking_navigation() const { return m_bake_navigation; }

    void set_collision_layer(uint32_t layer) { m_collision_layer = layer; }
    uint32_t get_collision_layer() const { return m_collision_layer; }

    void set_collision_mask(uint32_t mask) { m_collision_mask = mask; }
    uint32_t get_collision_mask() const { return m_collision_mask; }

    void _ready() override {
        Node3D::_ready();
        update_meshes();
    }

    void _notification(int p_what) override {
        if (p_what == NOTIFICATION_TRANSFORM_CHANGED) {
            mark_dirty();
            update_meshes();
        }
    }

private:
    void mark_dirty() { m_dirty = true; }

    void clear_mesh_instances() {
        for (auto& kv : m_mesh_instances) {
            RenderingServer::get_singleton()->free_rid(kv.second);
        }
        m_mesh_instances.clear();
    }

    void clear_collision() {
        for (auto& kv : m_collision_bodies) {
            PhysicsServer3D::get_singleton()->free_rid(kv.second);
        }
        m_collision_bodies.clear();
    }

    void create_mesh_instance(const vec3i& pos, const GridMapCell& cell, const GridMapLibrary::Item& item) {
        RID mesh_rid = RenderingServer::get_singleton()->mesh_create();
        
        mat4f transform = mat4f::identity();
        transform = translate(transform, map_to_world(pos) + item.mesh_offset);
        transform = rotate(transform, static_cast<float>(cell.orientation) * 90.0f, vec3f(0, 1, 0));
        transform = scale(transform, item.mesh_scale);
        
        RenderingServer::get_singleton()->mesh_add_surface_from_mesh(mesh_rid, item.mesh->get_rid());
        RenderingServer::get_singleton()->instance_set_transform(mesh_rid, transform);
        
        if (item.material.is_valid()) {
            RenderingServer::get_singleton()->instance_set_material(mesh_rid, item.material->get_rid());
        }
        
        m_mesh_instances[pos] = mesh_rid;
    }

    void create_collision_body(const vec3i& pos, const GridMapCell& cell, const GridMapLibrary::Item& item) {
        if (!item.collision_shape.is_valid()) return;
        
        RID body = PhysicsServer3D::get_singleton()->create_body();
        PhysicsServer3D::get_singleton()->body_set_mode(body, PhysicsServer3D::BODY_MODE_STATIC);
        PhysicsServer3D::get_singleton()->body_set_collision_layer(body, m_collision_layer);
        PhysicsServer3D::get_singleton()->body_set_collision_mask(body, m_collision_mask);
        
        RID shape = PhysicsServer3D::get_singleton()->shape_create(PhysicsServer3D::SHAPE_BOX);
        PhysicsServer3D::get_singleton()->shape_set_data(shape, item.collision_shape->get_data());
        
        mat4f transform = mat4f::identity();
        transform = translate(transform, map_to_world(pos) + item.mesh_offset);
        transform = rotate(transform, static_cast<float>(cell.orientation) * 90.0f, vec3f(0, 1, 0));
        
        PhysicsServer3D::get_singleton()->body_add_shape(body, shape, transform);
        PhysicsServer3D::get_singleton()->body_set_space(body, get_world_3d()->get_space());
        
        m_collision_bodies[pos] = body;
    }

    void bake_navigation_mesh() {
        NavigationMeshSourceGeometryData3D source;
        for (const auto& kv : m_cells) {
            const vec3i& pos = kv.first;
            const GridMapCell& cell = kv.second;
            auto* item = m_library->get_item(cell.item_id);
            if (!item || !item->valid) continue;
            
            if (item->navigation_mesh.is_valid()) {
                mat4f transform = mat4f::identity();
                transform = translate(transform, map_to_world(pos) + item->mesh_offset);
                source.add_mesh(item->mesh, transform);
            }
        }
        
        NavigationMeshBakeParams params;
        NavigationMeshResult result = NavigationMeshGenerator::get_singleton()->bake(source, params);
        
        if (result.valid) {
            Ref<NavigationMesh> nav_mesh;
            nav_mesh.instance();
            nav_mesh->set_vertices(result.vertices);
            nav_mesh->set_polygons(result.indices);
            NavigationServer3D::get_singleton()->region_set_navigation_mesh(m_navigation_region, *nav_mesh);
            NavigationServer3D::get_singleton()->region_set_transform(m_navigation_region, get_global_transform());
        }
    }
};

} // namespace godot

// Bring into main namespace
using godot::GridMap;
using godot::GridMapLibrary;
using godot::GridMapOrientation;
using godot::GridMapOctant;
using godot::GridMapCell;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XGRIDMAP_HPP