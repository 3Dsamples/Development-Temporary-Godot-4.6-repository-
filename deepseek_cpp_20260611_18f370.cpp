// Name : lighting enhancement
// File : scene/3d/mesh_instance_3d_ext.h 11 of 60
// Description : Extended mesh instance with mesh resource management, blend shapes,
//               skinning, LOD, shadow settings, and full RenderingServer synchronization.
#pragma once

#include "scene/3d/mesh_instance_3d.h"
#include "servers/rendering_server.h"

class MeshInstance3DExt : public MeshInstance3D {
    GDCLASS(MeshInstance3DExt, MeshInstance3D);

public:
    MeshInstance3DExt();
    ~MeshInstance3DExt();

    // ------------------------------------------------------------------------
    // Mesh and material management (overrides)
    // ------------------------------------------------------------------------
    void set_mesh(const RID &p_mesh) override;
    RID get_mesh() const override;
    void set_material_override(const RID &p_material) override;
    void set_surface_material(int p_surface, const RID &p_material) override;

    // ------------------------------------------------------------------------
    // Skinning (skeleton and blend shapes)
    // ------------------------------------------------------------------------
    void set_skeleton(const RID &p_skeleton);
    RID get_skeleton() const;
    void set_skin(const RID &p_skin);
    RID get_skin() const;
    void set_blend_shape_value(int p_shape, float p_value) override;
    float get_blend_shape_value(int p_shape) const override;

    // ------------------------------------------------------------------------
    // LOD and distance culling
    // ------------------------------------------------------------------------
    void set_lod_distance(float p_distance) override;
    float get_lod_distance() const override;

    // ------------------------------------------------------------------------
    // Shadow and GI (overrides)
    // ------------------------------------------------------------------------
    void set_cast_shadow(int p_cast_shadow) override;
    void set_receive_shadow(bool p_receive) override;
    void set_gi_mode(int p_mode) override;
    void set_gi_contribution(float p_amount) override;

    // ------------------------------------------------------------------------
    // Rendering server synchronization (push all parameters)
    // ------------------------------------------------------------------------
    void sync_mesh();
    void sync_skeleton();
    void sync_blend_shapes();
    void sync_all();

private:
    struct Impl;
    Impl *pimpl;
};