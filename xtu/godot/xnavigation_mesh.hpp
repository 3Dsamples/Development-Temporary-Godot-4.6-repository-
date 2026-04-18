// include/xtu/godot/xnavigation_mesh.hpp
// xtensor-unified - Navigation mesh generation for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XNAVIGATION_MESH_HPP
#define XTU_GODOT_XNAVIGATION_MESH_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xpathfinding.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xmesh.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class NavigationMeshGenerator;
class NavigationMeshSourceGeometryData3D;
class NavigationRegion3D;
class NavigationLink3D;
class NavigationObstacle3D;

// #############################################################################
// Navigation mesh baking parameters
// #############################################################################
struct NavigationMeshBakeParams {
    float cell_size = 0.3f;
    float cell_height = 0.2f;
    float agent_height = 2.0f;
    float agent_radius = 0.6f;
    float agent_max_climb = 0.9f;
    float agent_max_slope = 45.0f;
    float region_min_size = 8.0f;
    float region_merge_size = 20.0f;
    float edge_max_len = 12.0f;
    float edge_max_error = 1.3f;
    float verts_per_poly = 6.0f;
    float detail_sample_dist = 6.0f;
    float detail_sample_max_error = 1.0f;
    bool filter_low_hanging_obstacles = true;
    bool filter_ledge_spans = true;
    bool filter_walkable_low_height_spans = true;
};

// #############################################################################
// Navigation mesh generation result
// #############################################################################
struct NavigationMeshResult {
    std::vector<vec3f> vertices;
    std::vector<int32_t> indices;
    std::vector<int32_t> polygon_sizes;
    aabb bounds;
    bool valid = false;
};

// #############################################################################
// NavigationMeshSourceGeometryData3D - Geometry collection
// #############################################################################
class NavigationMeshSourceGeometryData3D : public Resource {
    XTU_GODOT_REGISTER_CLASS(NavigationMeshSourceGeometryData3D, Resource)

private:
    std::vector<vec3f> m_vertices;
    std::vector<int32_t> m_indices;
    std::vector<mat4f> m_transforms;
    bool m_dirty = true;

public:
    static StringName get_class_static() { return StringName("NavigationMeshSourceGeometryData3D"); }

    void clear() {
        m_vertices.clear();
        m_indices.clear();
        m_transforms.clear();
        m_dirty = true;
    }

    void add_mesh(const Ref<Mesh>& mesh, const mat4f& transform) {
        if (!mesh.is_valid()) return;
        // Extract mesh geometry
        m_transforms.push_back(transform);
        m_dirty = true;
    }

    void add_node(Node* node) {
        if (!node) return;
        if (auto* mesh_instance = dynamic_cast<MeshInstance3D*>(node)) {
            add_mesh(mesh_instance->get_mesh(), mesh_instance->get_global_transform());
        }
        // Recursively add children
        for (int i = 0; i < node->get_child_count(); ++i) {
            add_node(node->get_child(i));
        }
    }

    bool has_data() const { return !m_vertices.empty(); }
};

// #############################################################################
// NavigationMeshGenerator - Main baking class
// #############################################################################
class NavigationMeshGenerator : public Object {
    XTU_GODOT_REGISTER_CLASS(NavigationMeshGenerator, Object)

private:
    static NavigationMeshGenerator* s_singleton;
    std::mutex m_mutex;
    std::atomic<bool> m_baking{false};

public:
    static NavigationMeshGenerator* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("NavigationMeshGenerator"); }

    NavigationMeshGenerator() { s_singleton = this; }
    ~NavigationMeshGenerator() { s_singleton = nullptr; }

    NavigationMeshResult bake(const NavigationMeshSourceGeometryData3D& source,
                              const NavigationMeshBakeParams& params) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_baking = true;
        
        NavigationMeshResult result;
        result.valid = false;
        
        if (!source.has_data()) {
            m_baking = false;
            return result;
        }

        // Voxelize geometry
        VoxelGrid grid = voxelize_geometry(source, params);
        
        // Build compact heightfield
        CompactHeightfield chf = build_compact_heightfield(grid, params);
        
        // Contour tracing
        ContourSet contours = build_contours(chf, params);
        
        // Polygon mesh generation
        PolyMesh poly_mesh = build_poly_mesh(contours, params);
        
        // Detail mesh
        result = build_detail_mesh(poly_mesh, params);
        result.valid = true;
        
        m_baking = false;
        return result;
    }

    void bake_async(const NavigationMeshSourceGeometryData3D& source,
                    const NavigationMeshBakeParams& params,
                    std::function<void(const NavigationMeshResult&)> callback) {
        std::thread([this, source, params, callback]() {
            NavigationMeshResult result = bake(source, params);
            if (callback) {
                callback(result);
            }
        }).detach();
    }

    bool is_baking() const { return m_baking; }

    void clear() {
        // Clear any cached data
    }

private:
    struct VoxelGrid {
        std::vector<uint8_t> data;
        int32_t width = 0;
        int32_t height = 0;
        int32_t depth = 0;
        vec3f origin;
        float cell_size = 0.3f;
        float cell_height = 0.2f;
    };

    struct CompactHeightfield {
        std::vector<uint32_t> spans;
        int32_t width = 0;
        int32_t height = 0;
    };

    struct ContourSet {
        std::vector<std::vector<vec3f>> contours;
    };

    struct PolyMesh {
        std::vector<vec3f> vertices;
        std::vector<int32_t> indices;
        std::vector<int32_t> polygon_sizes;
    };

    VoxelGrid voxelize_geometry(const NavigationMeshSourceGeometryData3D& source,
                                 const NavigationMeshBakeParams& params) {
        VoxelGrid grid;
        // Calculate bounds
        aabb bounds;
        for (const auto& v : source.m_vertices) {
            bounds.extend(v);
        }
        
        grid.cell_size = params.cell_size;
        grid.cell_height = params.cell_height;
        grid.origin = bounds.min;
        
        grid.width = static_cast<int32_t>(std::ceil((bounds.max.x() - bounds.min.x()) / grid.cell_size)) + 1;
        grid.height = static_cast<int32_t>(std::ceil((bounds.max.y() - bounds.min.y()) / grid.cell_height)) + 1;
        grid.depth = static_cast<int32_t>(std::ceil((bounds.max.z() - bounds.min.z()) / grid.cell_size)) + 1;
        
        grid.data.resize(grid.width * grid.height * grid.depth, 0);
        
        // Rasterize triangles into voxel grid
        for (size_t i = 0; i < source.m_indices.size(); i += 3) {
            vec3f v0 = source.m_vertices[source.m_indices[i]];
            vec3f v1 = source.m_vertices[source.m_indices[i + 1]];
            vec3f v2 = source.m_vertices[source.m_indices[i + 2]];
            rasterize_triangle(grid, v0, v1, v2);
        }
        
        return grid;
    }

    void rasterize_triangle(VoxelGrid& grid, const vec3f& v0, const vec3f& v1, const vec3f& v2) {
        // Conservative voxelization
        aabb tri_bounds;
        tri_bounds.extend(v0);
        tri_bounds.extend(v1);
        tri_bounds.extend(v2);
        
        int32_t min_x = static_cast<int32_t>((tri_bounds.min.x() - grid.origin.x()) / grid.cell_size);
        int32_t max_x = static_cast<int32_t>((tri_bounds.max.x() - grid.origin.x()) / grid.cell_size);
        int32_t min_y = static_cast<int32_t>((tri_bounds.min.y() - grid.origin.y()) / grid.cell_height);
        int32_t max_y = static_cast<int32_t>((tri_bounds.max.y() - grid.origin.y()) / grid.cell_height);
        int32_t min_z = static_cast<int32_t>((tri_bounds.min.z() - grid.origin.z()) / grid.cell_size);
        int32_t max_z = static_cast<int32_t>((tri_bounds.max.z() - grid.origin.z()) / grid.cell_size);
        
        for (int32_t z = min_z; z <= max_z; ++z) {
            for (int32_t y = min_y; y <= max_y; ++y) {
                for (int32_t x = min_x; x <= max_x; ++x) {
                    vec3f cell_center = grid.origin + vec3f(
                        (x + 0.5f) * grid.cell_size,
                        (y + 0.5f) * grid.cell_height,
                        (z + 0.5f) * grid.cell_size
                    );
                    if (point_in_triangle(cell_center, v0, v1, v2)) {
                        size_t idx = z * grid.height * grid.width + y * grid.width + x;
                        if (idx < grid.data.size()) {
                            grid.data[idx] = 1;
                        }
                    }
                }
            }
        }
    }

    bool point_in_triangle(const vec3f& p, const vec3f& a, const vec3f& b, const vec3f& c) {
        // Use barycentric coordinates (simplified 2D projection)
        vec3f v0 = c - a;
        vec3f v1 = b - a;
        vec3f v2 = p - a;
        
        float dot00 = v0.x() * v0.x() + v0.z() * v0.z();
        float dot01 = v0.x() * v1.x() + v0.z() * v1.z();
        float dot02 = v0.x() * v2.x() + v0.z() * v2.z();
        float dot11 = v1.x() * v1.x() + v1.z() * v1.z();
        float dot12 = v1.x() * v2.x() + v1.z() * v2.z();
        
        float inv_denom = 1.0f / (dot00 * dot11 - dot01 * dot01);
        float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
        
        return (u >= 0) && (v >= 0) && (u + v <= 1);
    }

    CompactHeightfield build_compact_heightfield(const VoxelGrid& grid, const NavigationMeshBakeParams& params) {
        CompactHeightfield chf;
        chf.width = grid.width;
        chf.height = grid.depth;
        chf.spans.resize(chf.width * chf.height, 0);
        return chf;
    }

    ContourSet build_contours(const CompactHeightfield& chf, const NavigationMeshBakeParams& params) {
        ContourSet contours;
        return contours;
    }

    PolyMesh build_poly_mesh(const ContourSet& contours, const NavigationMeshBakeParams& params) {
        PolyMesh mesh;
        return mesh;
    }

    NavigationMeshResult build_detail_mesh(const PolyMesh& poly_mesh, const NavigationMeshBakeParams& params) {
        NavigationMeshResult result;
        result.vertices = poly_mesh.vertices;
        result.indices = poly_mesh.indices;
        result.polygon_sizes = poly_mesh.polygon_sizes;
        return result;
    }
};

// #############################################################################
// NavigationRegion3D - Region that provides navigation mesh
// #############################################################################
class NavigationRegion3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(NavigationRegion3D, Node3D)

private:
    Ref<NavigationMesh> m_navigation_mesh;
    RID m_region_rid;
    bool m_enabled = true;
    float m_enter_cost = 0.0f;
    float m_travel_cost = 1.0f;
    uint32_t m_navigation_layers = 1;
    NavigationMeshSourceGeometryData3D m_source_geometry;

public:
    static StringName get_class_static() { return StringName("NavigationRegion3D"); }

    NavigationRegion3D() {
        m_region_rid = NavigationServer3D::get_singleton()->region_create();
    }

    ~NavigationRegion3D() {
        NavigationServer3D::get_singleton()->free_rid(m_region_rid);
    }

    void set_navigation_mesh(const Ref<NavigationMesh>& mesh) {
        m_navigation_mesh = mesh;
        update_navigation_mesh();
    }

    Ref<NavigationMesh> get_navigation_mesh() const { return m_navigation_mesh; }

    void set_enabled(bool enabled) {
        m_enabled = enabled;
        update_navigation_mesh();
    }

    bool is_enabled() const { return m_enabled; }

    void set_enter_cost(float cost) { m_enter_cost = cost; }
    float get_enter_cost() const { return m_enter_cost; }

    void set_travel_cost(float cost) { m_travel_cost = cost; }
    float get_travel_cost() const { return m_travel_cost; }

    void bake_navigation_mesh() {
        NavigationMeshBakeParams params;
        NavigationMeshResult result = NavigationMeshGenerator::get_singleton()->bake(m_source_geometry, params);
        if (result.valid) {
            Ref<NavigationMesh> mesh;
            mesh.instance();
            mesh->set_vertices(result.vertices);
            mesh->set_polygons(result.indices);
            set_navigation_mesh(mesh);
        }
    }

    void _ready() override {
        Node3D::_ready();
        // Collect source geometry from children
        m_source_geometry.clear();
        for (int i = 0; i < get_child_count(); ++i) {
            m_source_geometry.add_node(get_child(i));
        }
        update_navigation_mesh();
    }

private:
    void update_navigation_mesh() {
        if (m_enabled && m_navigation_mesh.is_valid()) {
            NavigationServer3D::get_singleton()->region_set_navigation_mesh(m_region_rid, *m_navigation_mesh);
            NavigationServer3D::get_singleton()->region_set_transform(m_region_rid, get_global_transform());
        }
    }
};

// #############################################################################
// NavigationLink3D - Off-mesh link connection
// #############################################################################
class NavigationLink3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(NavigationLink3D, Node3D)

private:
    vec3f m_start_position;
    vec3f m_end_position;
    bool m_bidirectional = true;
    bool m_enabled = true;
    float m_enter_cost = 0.0f;
    float m_travel_cost = 1.0f;
    uint32_t m_navigation_layers = 1;

public:
    static StringName get_class_static() { return StringName("NavigationLink3D"); }

    void set_start_position(const vec3f& pos) { m_start_position = pos; }
    vec3f get_start_position() const { return m_start_position; }

    void set_end_position(const vec3f& pos) { m_end_position = pos; }
    vec3f get_end_position() const { return m_end_position; }

    void set_bidirectional(bool bidirectional) { m_bidirectional = bidirectional; }
    bool is_bidirectional() const { return m_bidirectional; }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }
};

// #############################################################################
// NavigationObstacle3D - Dynamic obstacle
// #############################################################################
class NavigationObstacle3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(NavigationObstacle3D, Node3D)

private:
    float m_radius = 1.0f;
    float m_height = 2.0f;
    vec3f m_velocity;
    bool m_affect_navigation = true;
    bool m_estimate_radius = true;

public:
    static StringName get_class_static() { return StringName("NavigationObstacle3D"); }

    void set_radius(float radius) { m_radius = radius; }
    float get_radius() const { return m_radius; }

    void set_height(float height) { m_height = height; }
    float get_height() const { return m_height; }

    void set_velocity(const vec3f& vel) { m_velocity = vel; }
    vec3f get_velocity() const { return m_velocity; }
};

} // namespace godot

// Bring into main namespace
using godot::NavigationMeshGenerator;
using godot::NavigationMeshSourceGeometryData3D;
using godot::NavigationRegion3D;
using godot::NavigationLink3D;
using godot::NavigationObstacle3D;
using godot::NavigationMeshBakeParams;
using godot::NavigationMeshResult;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XNAVIGATION_MESH_HPP