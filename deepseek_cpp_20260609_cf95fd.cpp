// mesh_instance_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <string>

namespace lighting {

// ============================================================================
// MeshInstance3D – renders a mesh with optional skeleton & blend shapes
// Supports LOD, skinning, morph targets, lightmapping, and advanced shadows.
// ============================================================================

class MeshInstance3D : public GeometryInstance3D {
public:
    MeshInstance3D();
    ~MeshInstance3D();

    // ------------------------------------------------------------------------
    // Mesh resource management
    // ------------------------------------------------------------------------
    void set_mesh(int64_t mesh_rid); // RenderingServer mesh ID
    int64_t get_mesh_rid() const;
    void set_mesh_path(const char* path); // load from file
    const char* get_mesh_path() const;

    // ------------------------------------------------------------------------
    // Skinning (skeleton) support
    // ------------------------------------------------------------------------
    void set_skeleton(int64_t skeleton_rid);
    int64_t get_skeleton_rid() const;
    void set_skin(int64_t skin_rid);   // defines vertex-to-bone mapping
    int64_t get_skin_rid() const;

    // ------------------------------------------------------------------------
    // Blend shapes (morph targets)
    // ------------------------------------------------------------------------
    void set_blend_shape_count(int count);
    int get_blend_shape_count() const;
    void set_blend_shape_value(int index, float weight);
    float get_blend_shape_value(int index) const;
    void set_blend_shape_names(const char** names, int count);
    const char* get_blend_shape_name(int index) const;

    // ------------------------------------------------------------------------
    // LOD (level-of-detail) – multiple meshes per distance
    // ------------------------------------------------------------------------
    void set_lod_mesh_rid(float distance, int64_t mesh_rid);
    void clear_lod_meshes();
    int64_t get_active_lod_mesh(double camera_distance) const;

    // ------------------------------------------------------------------------
    // Shadow mesh (simplified geometry for cascaded shadows)
    // ------------------------------------------------------------------------
    void set_shadow_mesh_rid(int64_t mesh_rid);
    int64_t get_shadow_mesh_rid() const;

    // ------------------------------------------------------------------------
    // Lightmap baked data
    // ------------------------------------------------------------------------
    void set_lightmap_uvs(const std::vector<float>& uvs); // per vertex
    void set_lightmap_texture(int64_t texture_rid);
    int64_t get_lightmap_texture() const;

    // ------------------------------------------------------------------------
    // GPU instancing (multi‑mesh)
    // ------------------------------------------------------------------------
    void set_instance_count(int count);
    int get_instance_count() const;
    void set_instance_transform(int idx, const Transform3D& transform);
    Transform3D get_instance_transform(int idx) const;

    // ------------------------------------------------------------------------
    // Visibility & culling overrides
    // ------------------------------------------------------------------------
    void set_ignore_frustum_culling(bool ignore) override;
    bool get_ignore_frustum_culling() const override;

    // ------------------------------------------------------------------------
    // Render server synchronization (called each frame)
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;

protected:
    void _update_render_instance_transform() override;
    void _update_bounding_volume() override; // compute AABB from mesh

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting