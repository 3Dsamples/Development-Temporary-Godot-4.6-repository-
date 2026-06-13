// csg_cylinder_3d.cpp
#include "csg_cylinder_3d.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numbers>

namespace lighting {

// ============================================================================
// Helper: generate cylinder/cone mesh with radial segments, caps
// ============================================================================
static void generate_cylinder_mesh(double height, double radius_top, double radius_bottom,
                                   int radial_seg, bool caps_enabled, bool smooth_shading,
                                   std::vector<double>& out_vertices,
                                   std::vector<int>& out_indices,
                                   std::vector<float>& out_normals) {
    out_vertices.clear();
    out_indices.clear();
    out_normals.clear();

    int seg = std::max(3, radial_seg);
    double half_h = height * 0.5;
    double angle_step = 2.0 * std::numbers::pi / seg;

    // Body vertices (two rings: top and bottom)
    int base_idx = 0;
    for (int i = 0; i <= seg; ++i) {
        double angle = i * angle_step;
        double x = std::cos(angle);
        double z = std::sin(angle);
        // Bottom ring
        double r_bot = radius_bottom;
        out_vertices.push_back(x * r_bot);
        out_vertices.push_back(-half_h);
        out_vertices.push_back(z * r_bot);
        // Normal for smooth shading (pointing outward radially, but also up/down? not perfect)
        double nx = x;
        double nz = z;
        double ny = 0.0;
        // if cone, adjust normal to be perpendicular to cone surface
        if (radius_top != radius_bottom) {
            double slope = (radius_top - radius_bottom) / height;
            // normal = (x, -slope, z) normalized (for outward facing)
            nx = x;
            nz = z;
            ny = -slope;
            double len = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (len > 1e-6) { nx /= len; ny /= len; nz /= len; }
        }
        out_normals.push_back((float)nx);
        out_normals.push_back((float)ny);
        out_normals.push_back((float)nz);
        // Top ring
        double r_top = radius_top;
        out_vertices.push_back(x * r_top);
        out_vertices.push_back(half_h);
        out_vertices.push_back(z * r_top);
        out_normals.push_back((float)nx);
        out_normals.push_back((float)ny);
        out_normals.push_back((float)nz);
    }
    // Indices for body (two triangles per quad)
    for (int i = 0; i < seg; ++i) {
        int bottom_left = i * 2;
        int bottom_right = (i+1) * 2;
        int top_left = bottom_left + 1;
        int top_right = bottom_right + 1;
        // triangle 1
        out_indices.push_back(bottom_left);
        out_indices.push_back(bottom_right);
        out_indices.push_back(top_left);
        // triangle 2
        out_indices.push_back(top_left);
        out_indices.push_back(bottom_right);
        out_indices.push_back(top_right);
    }

    // Caps (if enabled)
    if (caps_enabled) {
        // Bottom cap: fan around center
        int bottom_center_idx = (int)out_vertices.size() / 3;
        out_vertices.push_back(0.0);
        out_vertices.push_back(-half_h);
        out_vertices.push_back(0.0);
        out_normals.push_back(0.0f);
        out_normals.push_back(-1.0f);
        out_normals.push_back(0.0f);
        for (int i = 0; i <= seg; ++i) {
            double angle = i * angle_step;
            double x = std::cos(angle) * radius_bottom;
            double z = std::sin(angle) * radius_bottom;
            out_vertices.push_back(x);
            out_vertices.push_back(-half_h);
            out_vertices.push_back(z);
            out_normals.push_back(0.0f);
            out_normals.push_back(-1.0f);
            out_normals.push_back(0.0f);
        }
        int cap_start = bottom_center_idx + 1;
        for (int i = 0; i < seg; ++i) {
            out_indices.push_back(bottom_center_idx);
            out_indices.push_back(cap_start + i);
            out_indices.push_back(cap_start + i + 1);
        }
        // Top cap
        int top_center_idx = (int)out_vertices.size() / 3;
        out_vertices.push_back(0.0);
        out_vertices.push_back(half_h);
        out_vertices.push_back(0.0);
        out_normals.push_back(0.0f);
        out_normals.push_back(1.0f);
        out_normals.push_back(0.0f);
        for (int i = 0; i <= seg; ++i) {
            double angle = i * angle_step;
            double x = std::cos(angle) * radius_top;
            double z = std::sin(angle) * radius_top;
            out_vertices.push_back(x);
            out_vertices.push_back(half_h);
            out_vertices.push_back(z);
            out_normals.push_back(0.0f);
            out_normals.push_back(1.0f);
            out_normals.push_back(0.0f);
        }
        cap_start = top_center_idx + 1;
        for (int i = 0; i < seg; ++i) {
            out_indices.push_back(top_center_idx);
            out_indices.push_back(cap_start + i + 1);
            out_indices.push_back(cap_start + i);
        }
    }
}

// ============================================================================
// CSGCylinder3D implementation
// ============================================================================
struct CSGCylinder3D::Impl {
    double height = 2.0;
    double radius_top = 1.0;
    double radius_bottom = 1.0;
    int radial_segments = 32;
    bool cone = false;
    bool caps_enabled = true;
    bool smooth_shading = true;

    int materials[3] = {-1, -1, -1}; // 0=body,1=top cap,2=bottom cap
    int default_material = -1;

    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 1;          // static by default for CSG
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    bool mesh_dirty = true;
    // Cached mesh data
    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<float> normals;
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;

    ~Impl() {
        // free mesh resources
    }

    void regenerate_mesh() {
        generate_cylinder_mesh(height, radius_top, radius_bottom, radial_segments,
                               caps_enabled, smooth_shading, vertices, indices, normals);
    }
};

CSGCylinder3D::CSGCylinder3D() : pimpl(std::make_unique<Impl>()) {
    set_csg_operation(CSGOperation::UNION);
}
CSGCylinder3D::~CSGCylinder3D() = default;

void CSGCylinder3D::set_height(double height) { pimpl->height = height; pimpl->mesh_dirty = true; update_csg_mesh(); }
double CSGCylinder3D::get_height() const { return pimpl->height; }
void CSGCylinder3D::set_radius(double radius) { pimpl->radius_top = radius; pimpl->radius_bottom = radius; pimpl->mesh_dirty = true; update_csg_mesh(); }
void CSGCylinder3D::set_radius_top(double radius) { pimpl->radius_top = radius; if (pimpl->cone && radius==0) pimpl->cone = true; pimpl->mesh_dirty = true; update_csg_mesh(); }
double CSGCylinder3D::get_radius_top() const { return pimpl->radius_top; }
void CSGCylinder3D::set_radius_bottom(double radius) { pimpl->radius_bottom = radius; pimpl->mesh_dirty = true; update_csg_mesh(); }
double CSGCylinder3D::get_radius_bottom() const { return pimpl->radius_bottom; }
void CSGCylinder3D::set_radial_segments(int segments) { pimpl->radial_segments = std::max(3, segments); pimpl->mesh_dirty = true; update_csg_mesh(); }
int CSGCylinder3D::get_radial_segments() const { return pimpl->radial_segments; }
void CSGCylinder3D::set_cone(bool is_cone) { pimpl->cone = is_cone; if (is_cone) pimpl->radius_top = 0.0; else pimpl->radius_top = pimpl->radius_bottom; pimpl->mesh_dirty = true; update_csg_mesh(); }
bool CSGCylinder3D::is_cone() const { return pimpl->cone; }
void CSGCylinder3D::set_caps_enabled(bool enabled) { pimpl->caps_enabled = enabled; pimpl->mesh_dirty = true; update_csg_mesh(); }
bool CSGCylinder3D::are_caps_enabled() const { return pimpl->caps_enabled; }
void CSGCylinder3D::set_smooth_shading(bool smooth) { pimpl->smooth_shading = smooth; pimpl->mesh_dirty = true; update_csg_mesh(); }
bool CSGCylinder3D::is_smooth_shading() const { return pimpl->smooth_shading; }

void CSGCylinder3D::set_material(int material_id) { pimpl->default_material = material_id; }
void CSGCylinder3D::set_material_side(int side, int material_id) {
    if (side >= 0 && side < 3) pimpl->materials[side] = material_id;
}

void CSGCylinder3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void CSGCylinder3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void CSGCylinder3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void CSGCylinder3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void CSGCylinder3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void CSGCylinder3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void CSGCylinder3D::update_csg_mesh() {
    if (!pimpl->mesh_dirty) return;
    pimpl->regenerate_mesh();
    // Update bounding box
    double max_radius = std::max(pimpl->radius_top, pimpl->radius_bottom);
    double aabb_min[3] = {-max_radius, -pimpl->height*0.5, -max_radius};
    double aabb_max[3] = { max_radius,  pimpl->height*0.5,  max_radius};
    set_aabb(aabb_min, aabb_max);
    double dx = aabb_max[0]-aabb_min[0];
    double dy = aabb_max[1]-aabb_min[1];
    double dz = aabb_max[2]-aabb_min[2];
    set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz) * 0.5);

    pimpl->mesh_dirty = false;
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // register emissive surface for GI
    }
}

} // namespace lighting