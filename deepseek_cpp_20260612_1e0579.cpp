// Name : lighting enhancement
// File : scene/3d/vehicle_wheel_3d_ext.cpp 58 of 60
// Description : Implementation of VehicleWheel3DExt with visual cylinder mesh,
//               suspension compression, rotation, steering angle, and full
//               RenderingServer sync for shadows and GI.
#include "vehicle_wheel_3d_ext.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/math/quaternion.h"
#include <cmath>
#include <vector>

struct VehicleWheel3DExt::Impl {
    RID mesh_rid;                 // cylinder mesh for visual wheel
    RID instance_rid;             // visual instance
    RID material_rid;             // wheel material (can be emissive)

    // Geometry
    float radius = 0.4f;
    float width = 0.2f;
    float suspension_rest_length = 0.5f;
    float suspension_travel = 0.2f;
    float suspension_stiffness = 20.0f;
    float suspension_damping = 2.0f;
    float friction = 1.0f;
    float roll_influence = 1.0f;

    // Runtime state
    float compression = 0.0f;          // 0 = extended, 1 = compressed
    float rotation_angle = 0.0f;       // radians (wheel spin)
    float steering_angle = 0.0f;       // radians (for front wheels)
    bool use_as_steering = false;

    // Lighting flags
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 2;                   // dynamic
    float gi_contribution = 1.0f;
    Color emissive_color = Color(0,0,0);
    float emissive_intensity = 0.0f;

    bool dirty = true;

    Impl() {
        mesh_rid = RenderingServer::get_singleton()->mesh_create();
        instance_rid = RenderingServer::get_singleton()->instance_create();
        RenderingServer::get_singleton()->instance_set_base(instance_rid, mesh_rid);
        material_rid = RenderingServer::get_singleton()->material_create();
        RenderingServer::get_singleton()->material_set_param(material_rid, "albedo", Color(0.2,0.2,0.2));
        RenderingServer::get_singleton()->material_set_param(material_rid, "roughness", 0.8f);
        RenderingServer::get_singleton()->material_set_param(material_rid, "metalness", 0.0f);
        generate_mesh();
    }

    ~Impl() {
        if (mesh_rid.is_valid()) RenderingServer::get_singleton()->free(mesh_rid);
        if (instance_rid.is_valid()) RenderingServer::get_singleton()->free(instance_rid);
        if (material_rid.is_valid()) RenderingServer::get_singleton()->free(material_rid);
    }

    void generate_mesh() {
        // Create a cylinder mesh (wheel) with radius and width.
        // Number of radial segments = 32, height segments = 2 (top and bottom caps optional)
        int radial_segments = 32;
        float half_width = width * 0.5f;

        std::vector<Vector3> vertices;
        std::vector<int> indices;
        std::vector<Vector3> normals;
        std::vector<Vector2> uvs;

        // Side vertices: two rings (bottom and top) for each angle
        for (int i = 0; i <= radial_segments; ++i) {
            float angle = i * 2.0f * Math_PI / radial_segments;
            float x = radius * cos(angle);
            float z = radius * sin(angle);
            // bottom edge
            vertices.push_back(Vector3(x, -half_width, z));
            // top edge
            vertices.push_back(Vector3(x,  half_width, z));
            // normals (point outward)
            Vector3 n = Vector3(cos(angle), 0.0f, sin(angle)).normalized();
            normals.push_back(n);
            normals.push_back(n);
            // UVs (u = angle / 2pi, v = 0 or 1)
            float u = (float)i / radial_segments;
            uvs.push_back(Vector2(u, 0.0f));
            uvs.push_back(Vector2(u, 1.0f));
        }
        // Indices for side quads
        for (int i = 0; i < radial_segments; ++i) {
            int i0 = i * 2;
            int i1 = i * 2 + 1;
            int i2 = (i+1) * 2;
            int i3 = (i+1) * 2 + 1;
            indices.push_back(i0); indices.push_back(i1); indices.push_back(i2);
            indices.push_back(i1); indices.push_back(i3); indices.push_back(i2);
        }

        // Add caps (optional: simple discs, but for wheel we might want a tire pattern).
        // For simplicity, add caps as two triangles (no extra vertices, but we need to generate disk vertices).
        int cap_base = vertices.size();
        // Bottom cap center
        vertices.push_back(Vector3(0, -half_width, 0));
        normals.push_back(Vector3(0, -1, 0));
        uvs.push_back(Vector2(0.5, 0.5));
        // Bottom cap outer ring vertices
        for (int i = 0; i <= radial_segments; ++i) {
            float angle = i * 2.0f * Math_PI / radial_segments;
            float x = radius * cos(angle);
            float z = radius * sin(angle);
            vertices.push_back(Vector3(x, -half_width, z));
            normals.push_back(Vector3(0, -1, 0));
            uvs.push_back(Vector2((cos(angle)+1.0f)*0.5f, (sin(angle)+1.0f)*0.5f));
        }
        // Indices for bottom cap (triangles)
        for (int i = 0; i < radial_segments; ++i) {
            indices.push_back(cap_base);
            indices.push_back(cap_base + 1 + i);
            indices.push_back(cap_base + 1 + i + 1);
        }
        // Top cap
        cap_base = vertices.size();
        vertices.push_back(Vector3(0, half_width, 0));
        normals.push_back(Vector3(0, 1, 0));
        uvs.push_back(Vector2(0.5, 0.5));
        for (int i = 0; i <= radial_segments; ++i) {
            float angle = i * 2.0f * Math_PI / radial_segments;
            float x = radius * cos(angle);
            float z = radius * sin(angle);
            vertices.push_back(Vector3(x, half_width, z));
            normals.push_back(Vector3(0, 1, 0));
            uvs.push_back(Vector2((cos(angle)+1.0f)*0.5f, (sin(angle)+1.0f)*0.5f));
        }
        for (int i = 0; i < radial_segments; ++i) {
            indices.push_back(cap_base);
            indices.push_back(cap_base + 1 + i + 1);
            indices.push_back(cap_base + 1 + i);
        }

        // Build mesh
        RenderingServer::get_singleton()->mesh_clear(mesh_rid);
        RenderingServer::get_singleton()->mesh_add_surface(mesh_rid, RS::PRIMITIVE_TRIANGLES,
                                                           vertices, indices, uvs, normals);
        // Set material on surface
        RenderingServer::get_singleton()->mesh_surface_set_material(mesh_rid, 0, material_rid);
    }

    void update_wheel_transform() {
        // Compute local transform of wheel relative to vehicle body.
        // Assumes wheel is attached to VehicleBody3D, and we want to set the transform
        // of this node (the wheel) based on suspension compression, steering, and rotation.
        Transform3D local;
        // Suspension compression: move wheel up/down along local Y axis (Godot uses Y up)
        float y_offset = -suspension_rest_length + compression * suspension_travel;
        local.origin = Vector3(0, y_offset, 0);
        // Steering (rotate around Y)
        if (use_as_steering) {
            Quaternion steer_quat = Quaternion(Vector3(0,1,0), steering_angle);
            local.basis = Basis(steer_quat);
        } else {
            local.basis = Basis();
        }
        // Rotation around X (wheel spin)
        Quaternion spin_quat = Quaternion(Vector3(1,0,0), rotation_angle);
        local.basis = local.basis * Basis(spin_quat);
        set_transform(local);
        // Also update the visual instance's transform (the node's global transform
        // is set by the scene tree, but we need to sync with rendering server.
        // The node's transform will be propagated automatically when set_transform is called.
    }

    void sync() {
        if (dirty) {
            generate_mesh();
            dirty = false;
        }
        // Update material parameters (emissive, etc.)
        RenderingServer *rs = RenderingServer::get_singleton();
        if (emissive_intensity > 0.0f) {
            rs->material_set_param(material_rid, "emission", emissive_color);
            rs->material_set_param(material_rid, "emission_intensity", emissive_intensity);
        } else {
            rs->material_set_param(material_rid, "emission_intensity", 0.0f);
        }
        // Sync instance flags
        rs->instance_set_cast_shadow(instance_rid, cast_shadow);
        rs->instance_set_receive_shadows(instance_rid, receive_shadow);
        rs->instance_set_gi_mode(instance_rid, gi_mode);
        rs->instance_set_gi_contribution(instance_rid, gi_contribution);
        rs->instance_set_emissive(instance_rid, emissive_color, emissive_intensity);
        // Update transform (the node's global transform is already updated via set_transform,
        // but we also need to set the instance's transform to the node's global.
        rs->instance_set_transform(instance_rid, get_global_transform());
    }
};

VehicleWheel3DExt::VehicleWheel3DExt() {
    pimpl = new Impl();
}

VehicleWheel3DExt::~VehicleWheel3DExt() {
    delete pimpl;
}

void VehicleWheel3DExt::set_radius(float p_radius) {
    pimpl->radius = p_radius;
    pimpl->dirty = true;
    sync_wheel();
}
float VehicleWheel3DExt::get_radius() const { return pimpl->radius; }

void VehicleWheel3DExt::set_width(float p_width) {
    pimpl->width = p_width;
    pimpl->dirty = true;
    sync_wheel();
}
float VehicleWheel3DExt::get_width() const { return pimpl->width; }

void VehicleWheel3DExt::set_suspension_rest_length(float p_length) {
    pimpl->suspension_rest_length = p_length;
}
float VehicleWheel3DExt::get_suspension_rest_length() const { return pimpl->suspension_rest_length; }

void VehicleWheel3DExt::set_suspension_travel(float p_travel) {
    pimpl->suspension_travel = p_travel;
}
float VehicleWheel3DExt::get_suspension_travel() const { return pimpl->suspension_travel; }

void VehicleWheel3DExt::set_suspension_stiffness(float p_stiffness) {
    pimpl->suspension_stiffness = p_stiffness;
}
float VehicleWheel3DExt::get_suspension_stiffness() const { return pimpl->suspension_stiffness; }

void VehicleWheel3DExt::set_suspension_damping(float p_damping) {
    pimpl->suspension_damping = p_damping;
}
float VehicleWheel3DExt::get_suspension_damping() const { return pimpl->suspension_damping; }

void VehicleWheel3DExt::set_friction(float p_friction) {
    pimpl->friction = p_friction;
}
float VehicleWheel3DExt::get_friction() const { return pimpl->friction; }

void VehicleWheel3DExt::set_roll_influence(float p_influence) {
    pimpl->roll_influence = p_influence;
}
float VehicleWheel3DExt::get_roll_influence() const { return pimpl->roll_influence; }

void VehicleWheel3DExt::set_compression(float p_compression) {
    pimpl->compression = p_compression;
    pimpl->update_wheel_transform();
}
float VehicleWheel3DExt::get_compression() const { return pimpl->compression; }

void VehicleWheel3DExt::set_rotation_angle(float p_radians) {
    pimpl->rotation_angle = p_radians;
    pimpl->update_wheel_transform();
}
float VehicleWheel3DExt::get_rotation_angle() const { return pimpl->rotation_angle; }

void VehicleWheel3DExt::set_steering_angle(float p_radians) {
    pimpl->steering_angle = p_radians;
    pimpl->update_wheel_transform();
}
float VehicleWheel3DExt::get_steering_angle() const { return pimpl->steering_angle; }

void VehicleWheel3DExt::set_use_as_steering(bool p_steering) {
    pimpl->use_as_steering = p_steering;
    pimpl->update_wheel_transform();
}
bool VehicleWheel3DExt::is_steering_wheel() const { return pimpl->use_as_steering; }

void VehicleWheel3DExt::set_cast_shadow(bool p_cast) {
    pimpl->cast_shadow = p_cast;
    sync_wheel();
}
void VehicleWheel3DExt::set_receive_shadow(bool p_receive) {
    pimpl->receive_shadow = p_receive;
    sync_wheel();
}
void VehicleWheel3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
    sync_wheel();
}
void VehicleWheel3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
    sync_wheel();
}
void VehicleWheel3DExt::set_emissive(const Color &p_color, float p_intensity) {
    pimpl->emissive_color = p_color;
    pimpl->emissive_intensity = p_intensity;
    sync_wheel();
}
Color VehicleWheel3DExt::get_emissive() const { return pimpl->emissive_color; }
float VehicleWheel3DExt::get_emissive_intensity() const { return pimpl->emissive_intensity; }

void VehicleWheel3DExt::sync_wheel() {
    pimpl->sync();
}