// csg_shape_3d.cpp
#include "csg_shape_3d.h"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace lighting {

// ============================================================================
// CSGShape3D implementation
// ============================================================================
struct CSGShape3D::Impl {
    CSGOperation operation = CSGOperation::UNION;
    CSGShape3D* parent = nullptr;
    std::vector<CSGShape3D*> children;

    int material_id = -1;        // -1 = use default from scene
    Transform3D csg_transform;   // local transform for this CSG primitive

    bool cast_shadow = true;
    bool receive_shadow = true;
    int gi_mode = 1;             // static by default for CSG geometry
    float gi_contribution = 1.0f;
    float emissive_color[3] = {0.0f, 0.0f, 0.0f};
    float emissive_intensity = 0.0f;

    // Render server handles
    int64_t mesh_rid = -1;
    int64_t instance_rid = -1;

    bool tree_dirty = true;      // marks that CSG tree needs rebuild
};

CSGShape3D::CSGShape3D() : pimpl(std::make_unique<Impl>()) {
    pimpl->csg_transform.set_identity();
}
CSGShape3D::~CSGShape3D() = default;

void CSGShape3D::set_csg_operation(CSGOperation op) {
    pimpl->operation = op;
    if (pimpl->parent) pimpl->parent->rebuild_csg_tree();
}
CSGOperation CSGShape3D::get_csg_operation() const { return pimpl->operation; }

void CSGShape3D::set_csg_parent(CSGShape3D* parent) {
    if (pimpl->parent == parent) return;
    if (pimpl->parent) pimpl->parent->remove_csg_child(this);
    pimpl->parent = parent;
    if (parent) parent->add_csg_child(this);
}
CSGShape3D* CSGShape3D::get_csg_parent() const { return pimpl->parent; }

void CSGShape3D::add_csg_child(CSGShape3D* child) {
    if (child && std::find(pimpl->children.begin(), pimpl->children.end(), child) == pimpl->children.end()) {
        pimpl->children.push_back(child);
        rebuild_csg_tree();
    }
}
void CSGShape3D::remove_csg_child(CSGShape3D* child) {
    auto it = std::find(pimpl->children.begin(), pimpl->children.end(), child);
    if (it != pimpl->children.end()) {
        pimpl->children.erase(it);
        rebuild_csg_tree();
    }
}
const std::vector<CSGShape3D*>& CSGShape3D::get_csg_children() const { return pimpl->children; }

void CSGShape3D::set_material(int material_id) { pimpl->material_id = material_id; }
int CSGShape3D::get_material() const { return pimpl->material_id; }

void CSGShape3D::set_csg_transform(const Transform3D& transform) {
    pimpl->csg_transform = transform;
    rebuild_csg_tree();
}
Transform3D CSGShape3D::get_csg_transform() const { return pimpl->csg_transform; }

void CSGShape3D::set_cast_shadow(bool cast) { pimpl->cast_shadow = cast; GeometryInstance3D::set_cast_shadow(cast); }
bool CSGShape3D::get_cast_shadow() const { return pimpl->cast_shadow; }
void CSGShape3D::set_receive_shadow(bool receive) { pimpl->receive_shadow = receive; }
bool CSGShape3D::get_receive_shadow() const { return pimpl->receive_shadow; }
void CSGShape3D::set_gi_mode(int mode) { pimpl->gi_mode = mode; GeometryInstance3D::set_gi_mode(mode); }
int CSGShape3D::get_gi_mode() const { return pimpl->gi_mode; }
void CSGShape3D::set_gi_contribution(float amount) { pimpl->gi_contribution = amount; }
float CSGShape3D::get_gi_contribution() const { return pimpl->gi_contribution; }

void CSGShape3D::set_emissive(const float* color, float intensity) {
    memcpy(pimpl->emissive_color, color, 3*sizeof(float));
    pimpl->emissive_intensity = intensity;
}
void CSGShape3D::get_emissive(float* out_color, float& out_intensity) const {
    memcpy(out_color, pimpl->emissive_color, 3*sizeof(float));
    out_intensity = pimpl->emissive_intensity;
}

void CSGShape3D::rebuild_csg_tree() {
    // In a real engine, this would traverse the CSG tree and combine meshes
    // using boolean operations (union, intersection, subtraction) to produce
    // a final mesh. For performance, this is often done on the CPU and then
    // sent to the GPU as a single mesh. Here we mark dirty and let the derived
    // class handle it.
    pimpl->tree_dirty = true;
    update_csg_mesh();
}

void CSGShape3D::_set_mesh_data(const std::vector<double>& vertices,
                                const std::vector<int>& indices,
                                const std::vector<float>& normals,
                                const std::vector<float>& uvs) {
    // Create or update mesh resource in RenderingServer.
    // This should also upload vertex buffers (positions, normals, UVs) and index buffer.
    // For now, we just update bounding box and visual instance.
    if (vertices.empty()) return;
    // Compute AABB from vertices
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
    double dx = max_x - min_x;
    double dy = max_y - min_y;
    double dz = max_z - min_z;
    set_bounding_sphere_radius(std::sqrt(dx*dx+dy*dy+dz*dz) * 0.5);
}

void CSGShape3D::process(double delta) {
    GeometryInstance3D::process(delta);
    if (pimpl->tree_dirty) {
        update_csg_mesh();
        pimpl->tree_dirty = false;
    }
}

void CSGShape3D::synchronize_render_server(double delta) {
    GeometryInstance3D::synchronize_render_server(delta);
    // If emissive, contribute to global illumination
    if (pimpl->emissive_intensity > 0.0f && pimpl->gi_mode > 0) {
        // Register as emissive surface for GI (inject into lightprobe, VCT, or LPV)
    }
}

} // namespace lighting