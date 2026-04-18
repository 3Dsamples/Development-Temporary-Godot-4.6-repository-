// include/xtu/godot/xcsg.hpp
// xtensor-unified - Constructive Solid Geometry (CSG) for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XCSG_HPP
#define XTU_GODOT_XCSG_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xphysics3d.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xmesh.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class CSGShape3D;
class CSGCombiner3D;
class CSGPrimitive3D;
class CSGBox3D;
class CSGSphere3D;
class CSGCylinder3D;
class CSGTorus3D;
class CSGPolygon3D;
class CSGMesh3D;

// #############################################################################
// CSG operation types
// #############################################################################
enum class CSGOperation : uint8_t {
    OPERATION_UNION = 0,
    OPERATION_INTERSECTION = 1,
    OPERATION_SUBTRACTION = 2
};

// #############################################################################
// CSG shape flags
// #############################################################################
enum class CSGShapeFlags : uint32_t {
    FLAG_NONE = 0,
    FLAG_USE_COLLISION = 1 << 0,
    FLAG_SMOOTH_FACES = 1 << 1,
    FLAG_UPDATE_ALWAYS = 1 << 2
};

inline CSGShapeFlags operator|(CSGShapeFlags a, CSGShapeFlags b) {
    return static_cast<CSGShapeFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

// #############################################################################
// CSG mesh data (internal representation)
// #############################################################################
struct CSGMeshData {
    std::vector<vec3f> vertices;
    std::vector<vec3f> normals;
    std::vector<vec2f> uvs;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> material_indices;
    std::vector<StringName> materials;
    aabb bounds;
    bool valid = false;
};

// #############################################################################
// CSGShape3D - Base class for all CSG shapes
// #############################################################################
class CSGShape3D : public Node3D {
    XTU_GODOT_REGISTER_CLASS(CSGShape3D, Node3D)

protected:
    CSGOperation m_operation = CSGOperation::OPERATION_UNION;
    CSGShapeFlags m_flags = CSGShapeFlags::FLAG_USE_COLLISION;
    float m_snap = 0.001f;
    bool m_calculate_tangents = true;
    int32_t m_collision_layer = 1;
    int32_t m_collision_mask = 1;
    float m_collision_priority = 1.0f;
    
    mutable std::mutex m_mutex;
    CSGMeshData m_mesh_data;
    bool m_dirty = true;
    RID m_mesh_rid;
    RID m_collision_rid;

public:
    static StringName get_class_static() { return StringName("CSGShape3D"); }

    CSGShape3D() {
        m_mesh_rid = RenderingServer::get_singleton()->mesh_create();
        m_collision_rid = PhysicsServer3D::get_singleton()->create_body();
        PhysicsServer3D::get_singleton()->body_set_mode(m_collision_rid, PhysicsServer3D::BODY_MODE_STATIC);
    }

    ~CSGShape3D() {
        RenderingServer::get_singleton()->free_rid(m_mesh_rid);
        PhysicsServer3D::get_singleton()->free_rid(m_collision_rid);
    }

    void set_operation(CSGOperation op) {
        m_operation = op;
        mark_dirty();
    }

    CSGOperation get_operation() const { return m_operation; }

    void set_use_collision(bool enable) {
        if (enable) m_flags = m_flags | CSGShapeFlags::FLAG_USE_COLLISION;
        else m_flags = static_cast<CSGShapeFlags>(static_cast<uint32_t>(m_flags) & ~static_cast<uint32_t>(CSGShapeFlags::FLAG_USE_COLLISION));
        mark_dirty();
    }

    bool get_use_collision() const { return (static_cast<uint32_t>(m_flags) & static_cast<uint32_t>(CSGShapeFlags::FLAG_USE_COLLISION)) != 0; }

    void set_snap(float snap) { m_snap = std::max(0.0001f, snap); mark_dirty(); }
    float get_snap() const { return m_snap; }

    void set_calculate_tangents(bool enable) { m_calculate_tangents = enable; mark_dirty(); }
    bool get_calculate_tangents() const { return m_calculate_tangents; }

    void set_collision_layer(int32_t layer) { m_collision_layer = layer; mark_dirty(); }
    int32_t get_collision_layer() const { return m_collision_layer; }

    void set_collision_mask(int32_t mask) { m_collision_mask = mask; mark_dirty(); }
    int32_t get_collision_mask() const { return m_collision_mask; }

    virtual CSGMeshData build_mesh_data() const = 0;

    void update_mesh() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_dirty) return;
        
        // Collect mesh data from this shape and all children
        CSGMeshData final_data;
        build_combined_mesh(this, final_data);
        
        if (final_data.valid && !final_data.vertices.empty()) {
            // Update rendering mesh
            graphics::vertex_buffer vb;
            vb.set_layout(graphics::vertex_layout::pos3_norm3_uv2());
            vb.resize(final_data.vertices.size());
            for (size_t i = 0; i < final_data.vertices.size(); ++i) {
                float* v = reinterpret_cast<float*>(vb.vertex(i));
                v[0] = final_data.vertices[i].x();
                v[1] = final_data.vertices[i].y();
                v[2] = final_data.vertices[i].z();
                if (i < final_data.normals.size()) {
                    v[3] = final_data.normals[i].x();
                    v[4] = final_data.normals[i].y();
                    v[5] = final_data.normals[i].z();
                }
                if (i < final_data.uvs.size()) {
                    v[6] = final_data.uvs[i].x();
                    v[7] = final_data.uvs[i].y();
                }
            }
            
            graphics::index_buffer ib;
            ib.resize(final_data.indices.size());
            std::copy(final_data.indices.begin(), final_data.indices.end(), ib.data());
            
            RenderingServer::get_singleton()->mesh_add_surface(m_mesh_rid, vb, ib);
            RenderingServer::get_singleton()->mesh_set_aabb(m_mesh_rid, final_data.bounds);
        }
        
        m_mesh_data = final_data;
        m_dirty = false;
    }

    void update_collision() {
        if (!get_use_collision()) return;
        // Create collision shape from mesh data
    }

    RID get_mesh_rid() const { return m_mesh_rid; }
    const CSGMeshData& get_mesh_data() const { return m_mesh_data; }

    void mark_dirty() {
        m_dirty = true;
        for (int i = 0; i < get_child_count(); ++i) {
            if (auto* child = dynamic_cast<CSGShape3D*>(get_child(i))) {
                child->mark_dirty();
            }
        }
    }

    void _ready() override {
        Node3D::_ready();
        update_mesh();
        update_collision();
    }

    void _notification(int p_what) override {
        if (p_what == NOTIFICATION_TRANSFORM_CHANGED) {
            mark_dirty();
            update_mesh();
        }
    }

private:
    void build_combined_mesh(CSGShape3D* root, CSGMeshData& out_data) {
        // Recursively collect and combine meshes using CSG operations
        CSGMeshData accumulated = root->build_mesh_data();
        
        for (int i = 0; i < root->get_child_count(); ++i) {
            if (auto* child = dynamic_cast<CSGShape3D*>(root->get_child(i))) {
                CSGMeshData child_data;
                build_combined_mesh(child, child_data);
                accumulated = apply_csg_operation(accumulated, child_data, child->get_operation());
            }
        }
        
        out_data = std::move(accumulated);
    }

    CSGMeshData apply_csg_operation(const CSGMeshData& a, const CSGMeshData& b, CSGOperation op) {
        // Use manifold library for boolean operations
        // Simplified implementation for demonstration
        CSGMeshData result;
        if (op == CSGOperation::OPERATION_UNION) {
            result = a;
            result.vertices.insert(result.vertices.end(), b.vertices.begin(), b.vertices.end());
            // Offset indices for b
            size_t offset = a.vertices.size();
            for (uint32_t idx : b.indices) {
                result.indices.push_back(idx + static_cast<uint32_t>(offset));
            }
        } else if (op == CSGOperation::OPERATION_INTERSECTION) {
            // Intersection logic
        } else if (op == CSGOperation::OPERATION_SUBTRACTION) {
            // Subtraction logic
        }
        result.valid = true;
        result.bounds = a.bounds;
        result.bounds.extend(b.bounds);
        return result;
    }
};

// #############################################################################
// CSGCombiner3D - Combines multiple CSG shapes
// #############################################################################
class CSGCombiner3D : public CSGShape3D {
    XTU_GODOT_REGISTER_CLASS(CSGCombiner3D, CSGShape3D)

public:
    static StringName get_class_static() { return StringName("CSGCombiner3D"); }

    CSGMeshData build_mesh_data() const override {
        CSGMeshData data;
        data.valid = true;
        return data;
    }
};

// #############################################################################
// CSGPrimitive3D - Base class for primitive shapes
// #############################################################################
class CSGPrimitive3D : public CSGShape3D {
    XTU_GODOT_REGISTER_CLASS(CSGPrimitive3D, CSGShape3D)

protected:
    bool m_flip_faces = false;

public:
    static StringName get_class_static() { return StringName("CSGPrimitive3D"); }

    void set_flip_faces(bool flip) { m_flip_faces = flip; mark_dirty(); }
    bool get_flip_faces() const { return m_flip_faces; }
};

// #############################################################################
// CSGBox3D - Box primitive
// #############################################################################
class CSGBox3D : public CSGPrimitive3D {
    XTU_GODOT_REGISTER_CLASS(CSGBox3D, CSGPrimitive3D)

private:
    vec3f m_size = {2, 2, 2};
    vec3f m_material_offset = {0, 0, 0};

public:
    static StringName get_class_static() { return StringName("CSGBox3D"); }

    void set_size(const vec3f& size) { m_size = size; mark_dirty(); }
    vec3f get_size() const { return m_size; }

    void set_material_offset(const vec3f& offset) { m_material_offset = offset; mark_dirty(); }
    vec3f get_material_offset() const { return m_material_offset; }

    CSGMeshData build_mesh_data() const override {
        CSGMeshData data;
        vec3f half = m_size * 0.5f;
        
        // 8 vertices, 12 triangles (36 indices)
        data.vertices = {
            {-half.x(), -half.y(), -half.z()}, { half.x(), -half.y(), -half.z()},
            { half.x(),  half.y(), -half.z()}, {-half.x(),  half.y(), -half.z()},
            {-half.x(), -half.y(),  half.z()}, { half.x(), -half.y(),  half.z()},
            { half.x(),  half.y(),  half.z()}, {-half.x(),  half.y(),  half.z()}
        };
        
        data.normals = {
            {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1},
            {0, 0,  1}, {0, 0,  1}, {0, 0,  1}, {0, 0,  1},
            {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0},
            {0,  1, 0}, {0,  1, 0}, {0,  1, 0}, {0,  1, 0},
            {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0},
            { 1, 0, 0}, { 1, 0, 0}, { 1, 0, 0}, { 1, 0, 0}
        };
        
        data.uvs = {
            {0, 0}, {1, 0}, {1, 1}, {0, 1},
            {0, 0}, {1, 0}, {1, 1}, {0, 1},
            {0, 0}, {1, 0}, {1, 1}, {0, 1},
            {0, 0}, {1, 0}, {1, 1}, {0, 1},
            {0, 0}, {1, 0}, {1, 1}, {0, 1},
            {0, 0}, {1, 0}, {1, 1}, {0, 1}
        };
        
        data.indices = {
            0, 1, 2, 0, 2, 3,   // front
            4, 6, 5, 4, 7, 6,   // back
            0, 4, 5, 0, 5, 1,   // bottom
            2, 6, 7, 2, 7, 3,   // top
            0, 3, 7, 0, 7, 4,   // left
            1, 5, 6, 1, 6, 2    // right
        };
        
        if (m_flip_faces) {
            for (size_t i = 0; i < data.indices.size(); i += 3) {
                std::swap(data.indices[i], data.indices[i + 1]);
            }
        }
        
        data.bounds = aabb(-half, half);
        data.valid = true;
        return data;
    }
};

// #############################################################################
// CSGSphere3D - Sphere primitive
// #############################################################################
class CSGSphere3D : public CSGPrimitive3D {
    XTU_GODOT_REGISTER_CLASS(CSGSphere3D, CSGPrimitive3D)

private:
    float m_radius = 1.0f;
    int32_t m_radial_segments = 12;
    int32_t m_rings = 6;
    bool m_smooth_faces = true;

public:
    static StringName get_class_static() { return StringName("CSGSphere3D"); }

    void set_radius(float radius) { m_radius = std::max(0.001f, radius); mark_dirty(); }
    float get_radius() const { return m_radius; }

    void set_radial_segments(int32_t segments) { m_radial_segments = std::max(3, segments); mark_dirty(); }
    int32_t get_radial_segments() const { return m_radial_segments; }

    void set_rings(int32_t rings) { m_rings = std::max(2, rings); mark_dirty(); }
    int32_t get_rings() const { return m_rings; }

    void set_smooth_faces(bool smooth) { m_smooth_faces = smooth; mark_dirty(); }
    bool get_smooth_faces() const { return m_smooth_faces; }

    CSGMeshData build_mesh_data() const override {
        CSGMeshData data;
        // Generate sphere vertices (simplified)
        data.valid = true;
        data.bounds = aabb(vec3f(-m_radius), vec3f(m_radius));
        return data;
    }
};

// #############################################################################
// CSGCylinder3D - Cylinder primitive
// #############################################################################
class CSGCylinder3D : public CSGPrimitive3D {
    XTU_GODOT_REGISTER_CLASS(CSGCylinder3D, CSGPrimitive3D)

private:
    float m_radius = 1.0f;
    float m_height = 2.0f;
    int32_t m_radial_segments = 12;
    int32_t m_height_segments = 1;
    bool m_cone = false;

public:
    static StringName get_class_static() { return StringName("CSGCylinder3D"); }

    void set_radius(float radius) { m_radius = std::max(0.001f, radius); mark_dirty(); }
    float get_radius() const { return m_radius; }

    void set_height(float height) { m_height = std::max(0.001f, height); mark_dirty(); }
    float get_height() const { return m_height; }

    void set_radial_segments(int32_t segments) { m_radial_segments = std::max(3, segments); mark_dirty(); }
    int32_t get_radial_segments() const { return m_radial_segments; }

    void set_cone(bool cone) { m_cone = cone; mark_dirty(); }
    bool is_cone() const { return m_cone; }

    CSGMeshData build_mesh_data() const override {
        CSGMeshData data;
        data.valid = true;
        float hh = m_height * 0.5f;
        data.bounds = aabb(vec3f(-m_radius, -hh, -m_radius), vec3f(m_radius, hh, m_radius));
        return data;
    }
};

// #############################################################################
// CSGTorus3D - Torus primitive
// #############################################################################
class CSGTorus3D : public CSGPrimitive3D {
    XTU_GODOT_REGISTER_CLASS(CSGTorus3D, CSGPrimitive3D)

private:
    float m_inner_radius = 0.5f;
    float m_outer_radius = 1.0f;
    int32_t m_radial_segments = 24;
    int32_t m_ring_segments = 12;

public:
    static StringName get_class_static() { return StringName("CSGTorus3D"); }

    void set_inner_radius(float radius) { m_inner_radius = std::max(0.001f, radius); mark_dirty(); }
    float get_inner_radius() const { return m_inner_radius; }

    void set_outer_radius(float radius) { m_outer_radius = std::max(0.001f, radius); mark_dirty(); }
    float get_outer_radius() const { return m_outer_radius; }

    CSGMeshData build_mesh_data() const override {
        CSGMeshData data;
        data.valid = true;
        float r = m_outer_radius + m_inner_radius;
        data.bounds = aabb(vec3f(-r), vec3f(r));
        return data;
    }
};

// #############################################################################
// CSGPolygon3D - Extruded polygon
// #############################################################################
class CSGPolygon3D : public CSGPrimitive3D {
    XTU_GODOT_REGISTER_CLASS(CSGPolygon3D, CSGPrimitive3D)

private:
    std::vector<vec2f> m_polygon;
    float m_depth = 1.0f;
    bool m_spin = false;
    int32_t m_spin_degrees = 360;
    int32_t m_spin_sides = 8;
    vec3f m_offset;

public:
    static StringName get_class_static() { return StringName("CSGPolygon3D"); }

    void set_polygon(const std::vector<vec2f>& poly) { m_polygon = poly; mark_dirty(); }
    const std::vector<vec2f>& get_polygon() const { return m_polygon; }

    void set_depth(float depth) { m_depth = std::max(0.001f, depth); mark_dirty(); }
    float get_depth() const { return m_depth; }

    void set_spin(bool spin) { m_spin = spin; mark_dirty(); }
    bool get_spin() const { return m_spin; }

    CSGMeshData build_mesh_data() const override {
        CSGMeshData data;
        data.valid = true;
        return data;
    }
};

// #############################################################################
// CSGMesh3D - Mesh-based CSG shape
// #############################################################################
class CSGMesh3D : public CSGPrimitive3D {
    XTU_GODOT_REGISTER_CLASS(CSGMesh3D, CSGPrimitive3D)

private:
    Ref<Mesh> m_mesh;

public:
    static StringName get_class_static() { return StringName("CSGMesh3D"); }

    void set_mesh(const Ref<Mesh>& mesh) { m_mesh = mesh; mark_dirty(); }
    Ref<Mesh> get_mesh() const { return m_mesh; }

    CSGMeshData build_mesh_data() const override {
        CSGMeshData data;
        data.valid = true;
        return data;
    }
};

} // namespace godot

// Bring into main namespace
using godot::CSGShape3D;
using godot::CSGCombiner3D;
using godot::CSGPrimitive3D;
using godot::CSGBox3D;
using godot::CSGSphere3D;
using godot::CSGCylinder3D;
using godot::CSGTorus3D;
using godot::CSGPolygon3D;
using godot::CSGMesh3D;
using godot::CSGOperation;
using godot::CSGShapeFlags;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XCSG_HPP