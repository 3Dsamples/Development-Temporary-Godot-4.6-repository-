// Name : lighting enhancement
// File : scene/3d/occluder_instance_3d_ext.h 27 of 60
// Description : Extended occluder instance for occlusion culling, with shape (box/sphere/mesh),
//               size, transform, and full RenderingServer synchronization.
#pragma once

#include "scene/3d/occluder_instance_3d.h"
#include "servers/rendering_server.h"

class OccluderInstance3DExt : public OccluderInstance3D {
    GDCLASS(OccluderInstance3DExt, OccluderInstance3D);

public:
    OccluderInstance3DExt();
    ~OccluderInstance3DExt();

    // ------------------------------------------------------------------------
    // Occluder shape and size
    // ------------------------------------------------------------------------
    void set_shape(int p_shape);                // 0 = box, 1 = sphere, 2 = mesh
    int get_shape() const;
    void set_size(const Vector3 &p_size);       // half extents for box, radius for sphere
    Vector3 get_size() const;
    void set_mesh(const RID &p_mesh);           // for mesh‑based occluder
    RID get_mesh() const;

    // ------------------------------------------------------------------------
    // Occlusion culling parameters
    // ------------------------------------------------------------------------
    void set_enabled(bool p_enabled);
    bool is_enabled() const;
    void set_cull_mask(uint32_t p_mask);
    uint32_t get_cull_mask() const;

    // ------------------------------------------------------------------------
    // Debug visualization (draw occluder wireframe)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool p_visible);
    bool is_debug_visible() const;
    void set_debug_color(const Color &p_color);
    Color get_debug_color() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void sync_occluder();

private:
    struct Impl;
    Impl *pimpl;
};