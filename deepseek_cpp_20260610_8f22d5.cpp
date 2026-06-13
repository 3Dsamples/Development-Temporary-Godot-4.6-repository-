// csg_box_3d.cpp
#include "csg_box_3d.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

namespace lighting {

// ============================================================================
// Helper: generate box vertices with subdivision
// ============================================================================
static void generate_box_mesh(double size_x, double size_y, double size_z,
                              int subdiv, bool hollow, double wall_thickness,
                              std::vector<double>& out_vertices,
                              std::vector<int>& out_indices,
                              std::vector<float>& out_normals) {
    // Simplified: generate a standard 8‑vertex box with 12 triangles (6 faces).
    // Subdivision adds intermediate vertices (not implemented for brevity – full code would loop).
    double half_x = size_x * 0.5;
    double half_y = size_y * 0.5;
    double half_z = size_z * 0.5;

    // 8 vertices
    double v[8][3] = {
        {-half_x, -half_y, -half_z}, // 0
        { half_x, -half_y, -half_z}, // 1
        { half_x, -half_y,  half_z}, // 2
        {-half_x, -half_y,  half_z}, // 3
        {-half_x,  half_y, -half_z}, // 4
        { half_x,  half_y, -half_z}, // 5
        { half_x,  half_y,  half_z}, // 6
        {-half_x,  half_y,  half_z}  // 7
    };

    // face indices (12 triangles, each face 2 triangles)
    int faces[6][6] = {
        {0,1,2, 0,2,3}, // bottom (Y-)
        {4,7,6, 4,6,5}, // top    (Y+)
        {0,4,5, 0,5,1}, // front? depends on orientation, but works
        {1,5,6, 1,6,2},
        {2,6,7, 2,7,3},
        {3,7,4, 3,4,0}
    };

    // normals per face (simplified)
    float normals[6][3] = {
        {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}, {1,0,0}, {-1,0,0}
    };

    out_vertices.clear();
    out_indices.clear();
    out_normals.clear();

    if (hollow) {
        // For hollow box, we generate inner faces with reduced size
        // (not implemented – full code would subtract inner volume)
    }

    // Add vertices and indices for each face
    for (int f = 0; f < 6; ++f) {
        int base_idx = (int)out_vertices.size() / 3;
        // vertices for this face (4 unique vertices, but we add all 6 per triangle to keep simple)
        for (int v_idx : {faces[f][0], faces[f][1], faces[f][2], faces[f][3], faces[f][4], faces[f][5]}) {
            out_vertices.push_back(v[v_idx][0]);
            out_vertices.push_back(v[v_idx][1]);
            out_vertices.push_back(v[v_idx][2]);
            out_normals.push_back(normals[f][0]);
            out_normals.push_back(normals[f][1]);
            out_normals.push_back(normals[f][2]);
        }
        // indices for this face (sequential after base)
        out_indices.push_back(base_idx);
        out_indices.push_back(base_idx+1);
        out_indices.push_back(base_idx+2);
        out_indices.push_back(base_idx+3);
        out_indices.push_back(base_idx+4);
        out_indices.push_back(base_idx+5);
    }
}

// ============================================================================
// CSGBox3D implementation
// ============================================================================
struct CSGBox3D::Impl {
    double size[3] = {2.0, 2.0, 2.0};   // width, height, depth
    int subdivision = 0;
    bool hollow = false;
    double wall_thickness = 0.1;

    int materials[6] = {-1, -1, -1, -1, -1, -1}; // per face material IDs
    int default_material = -1;

    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 1;          // static by default for CSG
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    bool mesh_dirty = true;
    // Cached mesh data (to be sent to rendering server)
    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<float> normals;
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;

    ~Impl() {
        // free mesh resources
    }

    void regenerate_mesh();
};

CSGBox3D::CSGBox3D() : pimpl(std::make_unique<Impl>()) {
    set_csg_operation(CSGOperation::UNION); // from CSGShape3D base
}
CSGBox3D::~CSGBox3D() = default;

void CSGBox3D::set_size(const double* size) {
    memcpy(pimpl->size, size, 3*sizeof(double));
    pimpl->mesh_dirty = true;
    update_csg_mesh();
}
void CSGBox3D::get_size(double* out_size) const { memcpy(out_size, pimpl->size, 3*sizeof(double)); }

void CSGBox3D::set_material(int material_id) {
    pimpl->default_material = material_id;
    pimpl->mesh_dirty = true; // material may affect rendering
}
void CSGBox3D::set_material_face(int face, int material_id) {
    if (face >= 0 && face < 6) {
        pimpl->materials[face] = material_id;
    }
}

void CSGBox3D::set_subdivision_level(int level) { pimpl->subdivision = level; pimpl->mesh_dirty = true; }
int CSGBox3D::get_subdivision_level() const { return pimpl->subdivision; }
void CSGBox3D::set_hollow(bool hollow) { pimpl->hollow = hollow; pimpl->mesh_dirty = true; }
bool CSGBox3D::is_hollow() const { return pimpl->hollow; }
void CSGBox3D::set_wall_thickness(double thickness) { pimpl->wall_thickness = thickness; pimpl->mesh_dirty = true; }
double CSGBox3D::get_wall_thickness() const { return pimpl->wall_thickness; }

void CSGBox3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void CSGBox3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void CSGBox3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void CSGBox3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void CSGBox3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void CSGBox3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void CSGBox3D::Impl::regenerate_mesh() {
    generate_box_mesh(size[0], size[1], size[2], subdivision, hollow, wall_thickness,
                      vertices, indices, normals);
    // In a real engine, we would create a new mesh resource in RenderingServer
    // and assign materials per face. For now, we just mark as ready.
}

void CSGBox3D::update_csg_mesh() {
    if (!pimpl->mesh_dirty) return;
    pimpl->regenerate_mesh();
    // Notify parent CSG tree (if any) that geometry has changed.
    // Also update the visual instance: replace mesh RID.
    // For lighting, we must also update the bounding box for GI.
    double aabb_min[3] = {-pimpl->size[0]*0.5, -pimpl->size[1]*0.5, -pimpl->size[2]*0.5};
    double aabb_max[3] = { pimpl->size[0]*0.5,  pimpl->size[1]*0.5,  pimpl->size[2]*0.5};
    set_aabb(aabb_min, aabb_max);
    double dx = aabb_max[0]-aabb_min[0];
    double dy = aabb_max[1]-aabb_min[1];
    double dz = aabb_max[2]-aabb_min[2];
    set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz) * 0.5);

    pimpl->mesh_dirty = false;
    // Also, if emissive, contribute to GI system (inject into lightprobe or VCT)
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register this shape as an emissive surface for global illumination
    }
}

} // namespace lighting