// multi_mesh_instance_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// MultiMeshInstance3D – renders many instances of the same mesh with different
// transforms, colors, and per‑instance flags. Supports dynamic updates,
// GPU instancing, and full lighting (shadows, GI, emissive per instance).
// Optimized for tens of thousands of instances with low CPU overhead.
// ============================================================================

class MultiMeshInstance3D : public GeometryInstance3D {
public:
    MultiMeshInstance3D();
    ~MultiMeshInstance3D();

    // ------------------------------------------------------------------------
    // Base mesh (shared among all instances)
    // ------------------------------------------------------------------------
    void set_mesh(int64_t mesh_rid);
    int64_t get_mesh() const;

    // ------------------------------------------------------------------------
    // Instance management
    // ------------------------------------------------------------------------
    void set_instance_count(int count);
    int get_instance_count() const;
    void set_instance_transform(int idx, const Transform3D& transform);
    Transform3D get_instance_transform(int idx) const;
    void set_instance_color(int idx, const float* rgba);
    void get_instance_color(int idx, float* out_rgba) const;
    void set_instance_custom_data(int idx, const float* data, int size); // e.g., for shader uniforms
    void get_instance_custom_data(int idx, float* out_data, int max_size) const;

    // ------------------------------------------------------------------------
    // Batch updates (performance)
    // ------------------------------------------------------------------------
    void set_all_transforms(const std::vector<Transform3D>& transforms);
    void set_all_colors(const std::vector<float>& colors_rgba); // 4 floats per instance
    void flush_changes(); // send accumulated updates to GPU

    // ------------------------------------------------------------------------
    // Per‑instance lighting overrides
    // ------------------------------------------------------------------------
    void set_instance_cast_shadow(int idx, bool cast);
    bool get_instance_cast_shadow(int idx) const;
    void set_instance_gi_mode(int idx, int mode); // 0=off,1=static,2=dynamic
    int get_instance_gi_mode(int idx) const;
    void set_instance_gi_contribution(int idx, float amount);
    float get_instance_gi_contribution(int idx) const;
    void set_instance_emissive(int idx, const float* color, float intensity);
    void get_instance_emissive(int idx, float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Global overrides (applied to all instances unless per‑instance overridden)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity) override;

    // ------------------------------------------------------------------------
    // Rendering server sync
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting