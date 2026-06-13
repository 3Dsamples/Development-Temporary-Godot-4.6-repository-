// vehicle_wheel_3d.cpp
#include "vehicle_wheel_3d.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace lighting {

// ============================================================================
// Helper: generate cylinder mesh (approximation of wheel)
// ============================================================================
static void generate_wheel_mesh(float radius, float width, int radial_segments,
                                std::vector<double>& out_vertices,
                                std::vector<int>& out_indices,
                                std::vector<float>& out_normals,
                                std::vector<float>& out_uvs) {
    out_vertices.clear();
    out_indices.clear();
    out_normals.clear();
    out_uvs.clear();

    int seg = std::max(6, radial_segments);
    float half_w = width * 0.5f;
    float angle_step = 2.0f * M_PI / seg;

    // vertices: side and two caps (simplified: just side cylinder)
    for (int i = 0; i <= seg; ++i) {
        float angle = i * angle_step;
        float cx = cos(angle) * radius;
        float cz = sin(angle) * radius;
        // bottom edge
        out_vertices.push_back(cx);
        out_vertices.push_back(-half_w);
        out_vertices.push_back(cz);
        // normal pointing outward (radial)
        float nx = cx / radius;
        float nz = cz / radius;
        out_normals.push_back(nx);
        out_normals.push_back(0.0f);
        out_normals.push_back(nz);
        out_uvs.push_back((float)i / seg);
        out_uvs.push_back(0.0f);
        // top edge
        out_vertices.push_back(cx);
        out_vertices.push_back(half_w);
        out_vertices.push_back(cz);
        out_normals.push_back(nx);
        out_normals.push_back(0.0f);
        out_normals.push_back(nz);
        out_uvs.push_back((float)i / seg);
        out_uvs.push_back(1.0f);
    }
    // indices for side quads (2 triangles per quad)
    for (int i = 0; i < seg; ++i) {
        int bottom_left = i * 2;
        int bottom_right = (i+1) * 2;
        int top_left = bottom_left + 1;
        int top_right = bottom_right + 1;
        out_indices.push_back(bottom_left);
        out_indices.push_back(bottom_right);
        out_indices.push_back(top_left);
        out_indices.push_back(top_left);
        out_indices.push_back(bottom_right);
        out_indices.push_back(top_right);
    }
    // Caps (simplified: add disc at bottom and top – omitted for brevity)
}

// ============================================================================
// VehicleWheel3D implementation
// ============================================================================
struct VehicleWheel3D::Impl {
    float radius = 0.4f;
    float width = 0.2f;
    float suspension_rest_length = 0.5f;
    float suspension_travel = 0.2f;
    float suspension_stiffness = 20.0f;
    float suspension_damping = 2.0f;
    float friction = 1.0f;
    float roll_influence = 1.0f;

    bool use_as_steering = false;

    // runtime state
    float compression = 0.0f;        // 0 = extended, 1 = compressed
    float rotation_angle = 0.0f;     // radians (for rotation)
    float steering_angle = 0.0f;     // radians (for steering)

    // Lighting flags
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 2;                 // dynamic by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    bool dirty = true;
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;
    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<float> normals;
    std::vector<float> uvs;

    void regenerate_mesh();
    void update_render_server();
};

VehicleWheel3D::VehicleWheel3D() : pimpl(std::make_unique<Impl>()) {}
VehicleWheel3D::~VehicleWheel3D() = default;

void VehicleWheel3D::set_radius(float radius) {
    pimpl->radius = std::max(0.01f, radius);
    pimpl->dirty = true;
}
float VehicleWheel3D::get_radius() const { return pimpl->radius; }
void VehicleWheel3D::set_width(float width) {
    pimpl->width = std::max(0.01f, width);
    pimpl->dirty = true;
}
float VehicleWheel3D::get_width() const { return pimpl->width; }
void VehicleWheel3D::set_suspension_rest_length(float length) {
    pimpl->suspension_rest_length = std::max(0.01f, length);
}
float VehicleWheel3D::get_suspension_rest_length() const { return pimpl->suspension_rest_length; }
void VehicleWheel3D::set_suspension_travel(float travel) {
    pimpl->suspension_travel = std::max(0.0f, travel);
}
float VehicleWheel3D::get_suspension_travel() const { return pimpl->suspension_travel; }
void VehicleWheel3D::set_suspension_stiffness(float stiffness) { pimpl->suspension_stiffness = stiffness; }
float VehicleWheel3D::get_suspension_stiffness() const { return pimpl->suspension_stiffness; }
void VehicleWheel3D::set_suspension_damping(float damping) { pimpl->suspension_damping = damping; }
float VehicleWheel3D::get_suspension_damping() const { return pimpl->suspension_damping; }
void VehicleWheel3D::set_friction(float friction) { pimpl->friction = friction; }
float VehicleWheel3D::get_friction() const { return pimpl->friction; }
void VehicleWheel3D::set_roll_influence(float influence) { pimpl->roll_influence = influence; }
float VehicleWheel3D::get_roll_influence() const { return pimpl->roll_influence; }

void VehicleWheel3D::set_compression(float compression) {
    pimpl->compression = std::clamp(compression, 0.0f, 1.0f);
    pimpl->dirty = true;
}
float VehicleWheel3D::get_compression() const { return pimpl->compression; }
void VehicleWheel3D::set_rotation_angle(float radians) {
    pimpl->rotation_angle = radians;
    pimpl->dirty = true;
}
float VehicleWheel3D::get_rotation_angle() const { return pimpl->rotation_angle; }
void VehicleWheel3D::set_steering_angle(float radians) {
    pimpl->steering_angle = radians;
    pimpl->dirty = true;
}
float VehicleWheel3D::get_steering_angle() const { return pimpl->steering_angle; }

void VehicleWheel3D::set_use_as_steering(bool steering) { pimpl->use_as_steering = steering; }
bool VehicleWheel3D::is_steering_wheel() const { return pimpl->use_as_steering; }

void VehicleWheel3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void VehicleWheel3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void VehicleWheel3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void VehicleWheel3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void VehicleWheel3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void VehicleWheel3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void VehicleWheel3D::Impl::regenerate_mesh() {
    generate_wheel_mesh(radius, width, 24, vertices, indices, normals, uvs);
    // compute bounding box from vertices
    double min_x = vertices[0], max_x = vertices[0];
    double min_y = vertices[1], max_y = vertices[1];
    double min_z = vertices[2], max_z = vertices[2];
    for (size_t i = 3; i < vertices.size(); i += 3) {
        min_x = std::min(min_x, vertices[i]);
        max_x = std::max(max_x, vertices[i]);
        min_y = std::min(min_y, vertices[i+1]);
        max_y = std::max(max_y, vertices[i+1]);
        min_z = std::min(min_z, vertices[i+2]);
        max_z = std::max(max_z, vertices[i+2]);
    }
    set_aabb(&min_x, &max_x);
    double dx = max_x-min_x, dy = max_y-min_y, dz = max_z-min_z;
    set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz)*0.5);
}

void VehicleWheel3D::Impl::update_render_server() {
    if (mesh_rid == -1) {
        // mesh_rid = RenderingServer::mesh_create();
    }
    // RenderingServer::mesh_add_surface_from_arrays(mesh_rid, PRIMITIVE_TRIANGLES, vertices, normals, uvs, indices);
    if (instance_rid == -1) {
        // instance_rid = RenderingServer::instance_create();
    }
    // RenderingServer::instance_set_base(instance_rid, mesh_rid);
}

void VehicleWheel3D::update_wheel_mesh() {
    if (!pimpl->dirty) return;
    pimpl->regenerate_mesh();
    pimpl->update_render_server();
    pimpl->dirty = false;
}

void VehicleWheel3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    if (pimpl->dirty) {
        update_wheel_mesh();
    }
    // Compute wheel transform based on suspension compression, steering, and rotation.
    Transform3D local;
    // Translation: wheel center is at (0, -rest_length + compression*travel, 0) assuming local Y up
    double y_offset = -pimpl->suspension_rest_length + pimpl->compression * pimpl->suspension_travel;
    local.origin[0] = 0;
    local.origin[1] = y_offset;
    local.origin[2] = 0;
    // Steering rotation around Y axis (for front wheels)
    if (pimpl->use_as_steering) {
        double cs = cos(pimpl->steering_angle);
        double ss = sin(pimpl->steering_angle);
        // rotate basis
        double basis[9] = {cs, 0, ss, 0,1,0, -ss,0,cs};
        memcpy(local.basis, basis, 9*sizeof(double));
    } else {
        local.basis[0]=1; local.basis[1]=0; local.basis[2]=0;
        local.basis[3]=0; local.basis[4]=1; local.basis[5]=0;
        local.basis[6]=0; local.basis[7]=0; local.basis[8]=1;
    }
    // Rotation around X axis (wheel spin)
    double cs = cos(pimpl->rotation_angle);
    double ss = sin(pimpl->rotation_angle);
    double rot_basis[9] = {1,0,0, 0,cs,ss, 0,-ss,cs};
    // Combine: new_basis = local.basis * rot_basis (matrix multiplication)
    double new_basis[9];
    for (int i=0;i<3;++i) {
        for (int j=0;j<3;++j) {
            new_basis[i*3+j] = local.basis[i*3+0] * rot_basis[0*3+j] +
                               local.basis[i*3+1] * rot_basis[1*3+j] +
                               local.basis[i*3+2] * rot_basis[2*3+j];
        }
    }
    memcpy(local.basis, new_basis, 9*sizeof(double));
    set_transform(local);
    // Update instance transform if needed (but node's transform already set)
}

void VehicleWheel3D::process(double delta) {
    GeometryInstance3D::process(delta);
}

} // namespace lighting