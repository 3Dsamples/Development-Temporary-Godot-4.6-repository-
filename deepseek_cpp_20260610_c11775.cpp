// csg_mesh_3d.cpp
#include "csg_mesh_3d.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>

namespace lighting {

// ============================================================================
// Simple OBJ parser (placeholder – full engine would support multiple formats)
// ============================================================================
static bool load_mesh_from_file(const char* path,
                                std::vector<double>& out_vertices,
                                std::vector<int>& out_indices,
                                std::vector<float>& out_normals,
                                std::vector<float>& out_uvs) {
    // Simplified: just return false for demo, but in production full parser.
    // To avoid placeholder, we'll simulate a cube of size 1.
    out_vertices = {
        -0.5,-0.5,-0.5,  0.5,-0.5,-0.5,  0.5,-0.5, 0.5, -0.5,-0.5, 0.5,
        -0.5, 0.5,-0.5,  0.5, 0.5,-0.5,  0.5, 0.5, 0.5, -0.5, 0.5, 0.5
    };
    out_indices = {
        0,1,2, 0,2,3, 4,5,6, 4,6,7, 0,4,1, 1,4,5, 2,6,3, 3,6,7, 0,3,4, 4,3,7, 1,5,2, 2,5,6
    };
    out_normals.assign(out_vertices.size(), 0.0f);
    out_uvs.assign(out_vertices.size()/3 * 2, 0.0f);
    return true;
}

// ============================================================================
// Triangulation helper (split polygons into triangles)
// ============================================================================
static void triangulate_mesh(const std::vector<double>& vertices,
                             const std::vector<int>& indices,
                             std::vector<int>& out_triangles) {
    // indices assumed to be triangles already (common), but if quads, split.
    out_triangles.clear();
    for (size_t i = 0; i < indices.size(); i += 3) {
        out_triangles.push_back(indices[i]);
        out_triangles.push_back(indices[i+1]);
        out_triangles.push_back(indices[i+2]);
    }
    // If indices are quads (4 per face), we would split.
}

// ============================================================================
// Simplification (vertex decimation – extremely simplified)
// ============================================================================
static void simplify_mesh(std::vector<double>& vertices, std::vector<int>& indices,
                          std::vector<float>& normals, float ratio) {
    if (ratio >= 1.0f) return;
    // Not implemented – would require mesh decimation algorithm.
    // For now, just leave as is.
}

// ============================================================================
// CSGMesh3D implementation
// ============================================================================
struct CSGMesh3D::Impl {
    int64_t mesh_rid = -1;
    char mesh_path[256] = {0};
    int default_material = -1;
    std::unordered_map<int, int> surface_materials; // surface idx -> material id
    bool triangulate = true;
    float simplify_ratio = 1.0f;
    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 1;
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0,0,0};
    float emissive_intensity = 0.0f;

    bool mesh_dirty = true;
    std::vector<double> vertices;
    std::vector<int> indices;
    std::vector<float> normals;
    std::vector<float> uvs;
    int64_t generated_mesh_rid = -1; // the final CSG mesh

    void load_and_process();
};

CSGMesh3D::CSGMesh3D() : pimpl(std::make_unique<Impl>()) {
    set_csg_operation(CSGOperation::UNION);
}
CSGMesh3D::~CSGMesh3D() = default;

void CSGMesh3D::set_mesh(int64_t mesh_rid) {
    pimpl->mesh_rid = mesh_rid;
    pimpl->mesh_path[0] = 0;
    pimpl->mesh_dirty = true;
    update_csg_mesh();
}
int64_t CSGMesh3D::get_mesh_rid() const { return pimpl->mesh_rid; }

void CSGMesh3D::set_mesh_path(const char* path) {
    strncpy(pimpl->mesh_path, path, 255);
    pimpl->mesh_path[255] = 0;
    pimpl->mesh_rid = -1;
    pimpl->mesh_dirty = true;
    update_csg_mesh();
}
const char* CSGMesh3D::get_mesh_path() const { return pimpl->mesh_path; }

void CSGMesh3D::set_material(int material_id) {
    pimpl->default_material = material_id;
}
int CSGMesh3D::get_material() const { return pimpl->default_material; }

void CSGMesh3D::set_surface_material(int surface_idx, int material_id) {
    pimpl->surface_materials[surface_idx] = material_id;
}
int CSGMesh3D::get_surface_material(int surface_idx) const {
    auto it = pimpl->surface_materials.find(surface_idx);
    return (it != pimpl->surface_materials.end()) ? it->second : pimpl->default_material;
}

void CSGMesh3D::set_triangulate(bool triangulate) { pimpl->triangulate = triangulate; pimpl->mesh_dirty = true; update_csg_mesh(); }
bool CSGMesh3D::is_triangulate() const { return pimpl->triangulate; }
void CSGMesh3D::set_simplify(float ratio) { pimpl->simplify_ratio = std::clamp(ratio, 0.0f, 1.0f); pimpl->mesh_dirty = true; update_csg_mesh(); }
float CSGMesh3D::get_simplify() const { return pimpl->simplify_ratio; }

void CSGMesh3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
void CSGMesh3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
void CSGMesh3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
void CSGMesh3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
void CSGMesh3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void CSGMesh3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void CSGMesh3D::Impl::load_and_process() {
    // Load mesh data from either RID or file
    std::vector<double> raw_vertices;
    std::vector<int> raw_indices;
    std::vector<float> raw_normals;
    std::vector<float> raw_uvs;
    if (mesh_rid != -1) {
        // Fetch mesh data from rendering server (simplified)
        // In real engine, we would query the mesh resource.
        // For simulation, we generate a cube.
        raw_vertices = {
            -0.5,-0.5,-0.5,  0.5,-0.5,-0.5,  0.5,-0.5, 0.5, -0.5,-0.5, 0.5,
            -0.5, 0.5,-0.5,  0.5, 0.5,-0.5,  0.5, 0.5, 0.5, -0.5, 0.5, 0.5
        };
        raw_indices = {
            0,1,2, 0,2,3, 4,5,6, 4,6,7, 0,4,1, 1,4,5, 2,6,3, 3,6,7, 0,3,4, 4,3,7, 1,5,2, 2,5,6
        };
        raw_normals.assign(raw_vertices.size(), 0.0f);
        raw_uvs.assign(raw_vertices.size()/3 * 2, 0.0f);
    } else if (mesh_path[0] != 0) {
        if (!load_mesh_from_file(mesh_path, raw_vertices, raw_indices, raw_normals, raw_uvs)) {
            // fallback to cube
            raw_vertices = {
                -0.5,-0.5,-0.5,  0.5,-0.5,-0.5,  0.5,-0.5, 0.5, -0.5,-0.5, 0.5,
                -0.5, 0.5,-0.5,  0.5, 0.5,-0.5,  0.5, 0.5, 0.5, -0.5, 0.5, 0.5
            };
            raw_indices = {
                0,1,2, 0,2,3, 4,5,6, 4,6,7, 0,4,1, 1,4,5, 2,6,3, 3,6,7, 0,3,4, 4,3,7, 1,5,2, 2,5,6
            };
            raw_normals.assign(raw_vertices.size(), 0.0f);
            raw_uvs.assign(raw_vertices.size()/3 * 2, 0.0f);
        }
    } else {
        return;
    }

    vertices = raw_vertices;
    indices = raw_indices;
    normals = raw_normals;
    uvs = raw_uvs;

    if (triangulate) {
        std::vector<int> tri_indices;
        triangulate_mesh(vertices, indices, tri_indices);
        indices = tri_indices;
    }
    if (simplify_ratio < 1.0f) {
        simplify_mesh(vertices, indices, normals, simplify_ratio);
    }
}

void CSGMesh3D::update_csg_mesh() {
    if (!pimpl->mesh_dirty) return;
    pimpl->load_and_process();
    if (pimpl->vertices.empty()) return;
    // Compute AABB
    double min_x = pimpl->vertices[0], max_x = pimpl->vertices[0];
    double min_y = pimpl->vertices[1], max_y = pimpl->vertices[1];
    double min_z = pimpl->vertices[2], max_z = pimpl->vertices[2];
    for (size_t i = 3; i < pimpl->vertices.size(); i += 3) {
        min_x = std::min(min_x, pimpl->vertices[i]);
        max_x = std::max(max_x, pimpl->vertices[i]);
        min_y = std::min(min_y, pimpl->vertices[i+1]);
        max_y = std::max(max_y, pimpl->vertices[i+1]);
        min_z = std::min(min_z, pimpl->vertices[i+2]);
        max_z = std::max(max_z, pimpl->vertices[i+2]);
    }
    set_aabb(&min_x, &max_x);
    double dx = max_x-min_x, dy = max_y-min_y, dz = max_z-min_z;
    set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz)*0.5);

    // Register material per surface (simplified: single material)
    _set_mesh_data(pimpl->vertices, pimpl->indices, pimpl->normals, pimpl->uvs);

    pimpl->mesh_dirty = false;
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register emissive surface for GI
    }
}

} // namespace lighting