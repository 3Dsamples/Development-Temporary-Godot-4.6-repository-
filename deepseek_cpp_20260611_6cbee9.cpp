// multi_mesh_instance_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// MultiMeshInstance3D – renders a single mesh many times with different
// transforms and optional per‑instance colors / custom data.
// Highly efficient using GPU instancing; ideal for grass, trees, particles.
// Integrated with lighting: each instance can cast shadows and receive GI
// individually, with per‑instance emissive and lightmap data.
// ============================================================================

class MultiMeshInstance3D : public GeometryInstance3D {
public:
    MultiMeshInstance3D();
    ~MultiMeshInstance3D();

    // ------------------------------------------------------------------------
    // MultiMesh resource (holds mesh, instance count, transforms)
    // ------------------------------------------------------------------------
    void set_multimesh(int64_t multimesh_rid);
    int64_t get_multimesh_rid() const;

    // ------------------------------------------------------------------------
    // Instance management (if not using a separate MultiMesh resource)
    // ------------------------------------------------------------------------
    void set_instance_count(int count);
    int get_instance_count() const;
    void set_instance_transform(int idx, const Transform3D& transform);
    Transform3D get_instance_transform(int idx) const;
    void set_instance_color(int idx, float r, float g, float b, float a = 1.0f);
    void get_instance_color(int idx, float* out_rgba) const;
    void set_instance_custom_data(int idx, const float* data, int size); // up to 4 floats
    void get_instance_custom_data(int idx, float* out_data, int size) const;

    // ------------------------------------------------------------------------
    // Rendering settings
    // ------------------------------------------------------------------------
    void set_use_color(bool use);
    bool is_using_color() const;
    void set_use_custom_data(bool use);
    bool is_using_custom_data() const;
    void set_color_format(int format); // 0 = RGBA, 1 = RGB, 2 = SRGB
    int get_color_format() const;

    // ------------------------------------------------------------------------
    // Per‑instance lighting overrides (shadow, GI, emissive)
    // ------------------------------------------------------------------------
    void set_instance_cast_shadow(int idx, bool cast);
    bool get_instance_cast_shadow(int idx) const;
    void set_instance_gi_mode(int idx, int mode); // 0=off,1=static,2=dynamic
    int get_instance_gi_mode(int idx) const;
    void set_instance_emissive(int idx, const float* color, float intensity);
    void get_instance_emissive(int idx, float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Global defaults (applied to all instances unless overridden)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity) override;

    // ------------------------------------------------------------------------
    // GPU buffer updates (call after many changes)
    // ------------------------------------------------------------------------
    void update_instances();

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting