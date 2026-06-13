// Name : lighting enhancement
// File : scene/3d/voxel_gi_ext.h 25 of 60
// Description : Extended VoxelGI node with voxel grid resolution, bake volume,
//               dynamic updates, and full RenderingServer sync.
#pragma once

#include "scene/3d/voxel_gi.h"
#include "servers/rendering_server.h"

class VoxelGIExt : public VoxelGI {
    GDCLASS(VoxelGIExt, VoxelGI);

public:
    VoxelGIExt();
    ~VoxelGIExt();

    // ------------------------------------------------------------------------
    // Voxel grid parameters
    // ------------------------------------------------------------------------
    void set_resolution(int p_resolution);          // e.g., 64, 128, 256
    int get_resolution() const;
    void set_size(const Vector3 &p_size);           // world size (half extents)
    Vector3 get_size() const;
    void set_extents(const Vector3 &p_extents);     // alias for size
    Vector3 get_extents() const;

    // ------------------------------------------------------------------------
    // Baking and propagation
    // ------------------------------------------------------------------------
    void set_bounce_count(int p_bounces);
    int get_bounce_count() const;
    void set_energy(float p_energy);
    float get_energy() const;
    void set_normal_bias(float p_bias);
    float get_normal_bias() const;

    // ------------------------------------------------------------------------
    // Real‑time update
    // ------------------------------------------------------------------------
    void set_dynamic(bool p_dynamic);
    bool is_dynamic() const;
    void set_update_frequency(float p_fps);
    float get_update_frequency() const;
    void request_bake();                            // force re‑bake (async)

    // ------------------------------------------------------------------------
    // Debug visualization
    // ------------------------------------------------------------------------
    void set_debug_visible(bool p_visible);
    bool is_debug_visible() const;
    void set_debug_color(const Color &p_color);
    Color get_debug_color() const;

    // ------------------------------------------------------------------------
    // Global illumination influence
    // ------------------------------------------------------------------------
    void set_gi_contribution(float p_amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void sync_voxel_gi();

private:
    struct Impl;
    Impl *pimpl;
};