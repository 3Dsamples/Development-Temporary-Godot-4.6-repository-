// csg_sphere_3d.cpp
#include "csg_sphere_3d.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numbers>

namespace lighting {

// ============================================================================
// Helper: generate sphere mesh with radius, radial segments, rings, hemisphere
// ============================================================================
static void generate_sphere_mesh(double radius, int radial_segments, int rings,
                                 bool hemisphere, bool smooth_shading,
                                 std::vector<double>& out_vertices,
                                 std::vector<int>& out_indices,
                                 std::vector<float>& out_normals,
                                 std::vector<float>& out_uvs) {
    out_vertices.clear();
    out_indices.clear();
    out_normals.clear();
    out_uvs.clear();

    int rad_seg = std::max(3, radial_segments);
    int ring_cnt = std::max(2, rings);
    double phi_step = std::numbers::pi / ring_cnt;       // polar angle step (0 to pi)
    double theta_step = 2.0 * std::numbers::pi / rad_seg; // azimuthal angle step
    int max_rings = ring_cnt;
    if (hemisphere) {
        max_rings = ring_cnt / 2;   // only top half (phi from 0 to pi/2)
        phi_step = (std::numbers::pi * 0.5) / max_rings;
    }

    // Generate vertices and normals
    for (int i = 0; i <= max_rings; ++i) {
        double phi = i * phi_step;
        double sin_phi = std::sin(phi);
        double cos_phi = std::cos(phi);
        double y = radius * cos_phi;
        for (int j = 0; j <= rad_seg; ++j) {
            double theta = j * theta_step;
            double sin_theta = std::sin(theta);
            double cos_theta = std::cos(theta);
            double x = radius * sin_phi * cos_theta;
            double z = radius * sin_phi * sin_theta;
            out_vertices.push_back(x);
            out_vertices.push_back(y);
            out_vertices.push_back(z);
            // normal (same as direction from origin, for smooth shading)
            double nx = x / radius;
            double ny = y / radius;
            double nz = z / radius;
            out_normals.push_back((float)nx);
            out_normals.push_back((float)ny);
            out_normals.push_back((float)nz);
            // UVs (u = theta / 2pi, v = phi / pi)
            out_uvs.push_back((float)(theta / (2.0 * std::numbers::pi)));
            out_uvs.push_back((float)(phi / std::numbers::pi));
        }
    }

    // Indices (two triangles per quad)
    for (int i = 0; i < max_rings; ++i) {
        int row_start = i * (rad_seg + 1);
        int next_row_start = (i + 1) * (rad_seg + 1);
        for (int j = 0; j < rad_seg; ++j) {
            int top_left = row_start + j;
            int top_right = row_start + j + 1;
            int bottom_left = next_row_start + j;
            int bottom_right = next_row_start + j + 1;
            out_indices.push_back(top_left);
            out_indices.push_back(bottom_left);
            out_indices.push_back(top_right);
            out_indices.push_back(top_right);
            out_indices.push_back(bottom_left);
            out_indices.push_back(bottom_right);
        }
    }

    // If not smooth shading, we need to duplicate vertices per face to have flat normals.
    // For simplicity, we skip and rely on smooth normals (which look good for spheres).
}

// ============================================================================
// CSGSphere3D implementation
// ============================================================================
struct CSGSphere3D::Impl {
    double radius = 1.0;
    int radial_segments = 32;
    int rings = 16;
    bool hemisphere = false;
    bool smooth_shading = true;

    int material_id = -1;          // default material
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 1;               // static by default
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    bool mesh_dirty = true;
    // Cached mesh data
    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<float> normals;
    std::vector<float> uvs;
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;

    ~Impl() {
        // free render server resources
    }

    void regenerate_mesh() {
        generate_sphere_mesh(radius, radial_segments, rings, hemisphere,
                             smooth_shading, vertices, indices, normals, uvs);
    }
};

CSGSphere3D::CSGSphere3D() : pimpl(std::make_unique<Impl>()) {
    set_csg_operation(CSGOperation::UNION);
}
CSGSphere3D::~CSGSphere3D() = default;

void CSGSphere3D::set_radius(double radius) {
    pimpl->radius = radius;
    pimpl->mesh_dirty = true;
    update_csg_mesh();
}
double CSGSphere3D::get_radius() const { return pimpl->radius; }

void CSGSphere3D::set_radial_segments(int segments) {
    pimpl->radial_segments = std::max(3, segments);
    pimpl->mesh_dirty = true;
    update_csg_mesh();
}
int CSGSphere3D::get_radial_segments() const { return pimpl->radial_segments; }

void CSGSphere3D::set_rings(int rings) {
    pimpl->rings = std::max(2, rings);
    pimpl->mesh_dirty = true;
    update_csg_mesh();
}
int CSGSphere3D::get_rings() const { return pimpl->rings; }

void CSGSphere3D::set_hemisphere(bool enable) {
    pimpl->hemisphere = enable;
    pimpl->mesh_dirty = true;
    update_csg_mesh();
}
bool CSGSphere3D::is_hemisphere() const { return pimpl->hemisphere; }

void CSGSphere3D::set_material(int material_id) {
    pimpl->material_id = material_id;
}
void CSGSphere3D::set_smooth_shading(bool smooth) {
    pimpl->smooth_shading = smooth;
    pimpl->mesh_dirty = true;
    update_csg_mesh();
}
bool CSGSphere3D::is_smooth_shading() const { return pimpl->smooth_shading; }

void CSGSphere3D::set_cast_shadow(bool cast) {
    pimpl->cast_shadow = cast;
    GeometryInstance3D::set_cast_shadow(cast);
}
void CSGSphere3D::set_receive_shadow(bool receive) {
    pimpl->receive_shadow = receive;
}
void CSGSphere3D::set_gi_mode(int mode) {
    pimpl->gi_mode = mode;
    GeometryInstance3D::set_gi_mode(mode);
}
void CSGSphere3D::set_gi_contribution(float amount) {
    pimpl->gi_contribution = amount;
}
void CSGSphere3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void CSGSphere3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void CSGSphere3D::update_csg_mesh() {
    if (!pimpl->mesh_dirty) return;
    pimpl->regenerate_mesh();
    // Compute bounding box
    double aabb_min[3] = {-pimpl->radius, (pimpl->hemisphere ? 0.0 : -pimpl->radius), -pimpl->radius};
    double aabb_max[3] = { pimpl->radius,  pimpl->radius,  pimpl->radius};
    set_aabb(aabb_min, aabb_max);
    double dx = aabb_max[0]-aabb_min[0];
    double dy = aabb_max[1]-aabb_min[1];
    double dz = aabb_max[2]-aabb_min[2];
    set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz) * 0.5);

    // Upload mesh data to rendering server (simplified)
    _set_mesh_data(pimpl->vertices, pimpl->indices, pimpl->normals, pimpl->uvs);

    pimpl->mesh_dirty = false;
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register emissive surface for global illumination
    }
}

} // namespace lighting