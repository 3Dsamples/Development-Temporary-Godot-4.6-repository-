// occluder_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// Occluder3D – provides occlusion culling data to the rendering engine.
// Static occluders can be simple shapes (box, sphere) or arbitrary meshes.
// Supports dynamic updates (transforms, shape changes) and debug visualization.
// ============================================================================

enum class OccluderShape : uint8_t {
    NONE,
    SPHERE,
    BOX,
    MESH
};

class Occluder3D : public Node3D {
public:
    Occluder3D();
    ~Occluder3D();

    // ------------------------------------------------------------------------
    // Occluder geometry
    // ------------------------------------------------------------------------
    void set_shape(OccluderShape shape);
    OccluderShape get_shape() const;
    void set_size(const double* size);      // half extents for box, radius for sphere
    void get_size(double* out_size) const;
    void set_mesh(int64_t mesh_rid);        // custom triangle mesh
    int64_t get_mesh_rid() const;
    void set_vertices(const std::vector<double>& vertices, const std::vector<int>& indices);
    void clear_mesh();

    // ------------------------------------------------------------------------
    // Occlusion culling parameters
    // ------------------------------------------------------------------------
    void set_enabled(bool enabled);
    bool is_enabled() const;
    void set_force_occluder(bool force);    // treat as occluder even if not in view
    bool is_force_occluder() const;
    void set_occlusion_layer(uint32_t layer);
    uint32_t get_occlusion_layer() const;

    // ------------------------------------------------------------------------
    // Performance: approximate world bounds for acceleration structures
    // ------------------------------------------------------------------------
    void set_bounds_override(const double* min, const double* max);
    void clear_bounds_override();

    // ------------------------------------------------------------------------
    // Debug visualization (lit, can be emissive)
    // ------------------------------------------------------------------------
    void set_debug_visible(bool visible);
    bool is_debug_visible() const;
    void set_debug_color(const float* rgb);
    void get_debug_color(float* out_rgb) const;
    void set_emissive_debug(bool enable, float intensity = 0.2f);
    bool is_emissive_debug() const;

    // ------------------------------------------------------------------------
    // Lighting (debug mesh can affect GI if emissive)
    // ------------------------------------------------------------------------
    void set_gi_contribution(float amount);
    float get_gi_contribution() const;
    void set_cast_shadow(bool cast);         // for debug mesh
    bool get_cast_shadow() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting