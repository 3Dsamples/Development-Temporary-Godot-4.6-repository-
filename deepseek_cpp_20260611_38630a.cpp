// multi_mesh_instance_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// MultiMesh – container for multiple mesh instances with optimized GPU rendering.
// Stores per‑instance transforms, colors, and custom data. Supports large
// numbers of instances (trees, grass, particles) with single draw calls.
// ============================================================================
class MultiMesh : public Resource {
public:
    MultiMesh();
    ~MultiMesh();

    void set_mesh(int64_t mesh_rid);
    int64_t get_mesh_rid() const;

    void set_instance_count(int count);
    int get_instance_count() const;

    void set_instance_transform(int index, const Transform3D& transform);
    Transform3D get_instance_transform(int index) const;

    void set_instance_color(int index, const float* color);
    void get_instance_color(int index, float* out_color) const;

    void set_instance_custom_data(int index, const float* data, int data_size); // up to 4 floats
    void get_instance_custom_data(int index, float* out_data, int data_size) const;

    void set_transform_array(const std::vector<Transform3D>& transforms);
    void set_color_array(const std::vector<float>& colors); // RGBA per instance

    void set_visible_instance_range(int start, int end); // subset of instances to draw
    void get_visible_instance_range(int& start, int& end) const;

    void set_buffer_dirty();
    void update_buffers(); // upload to GPU

    // Performance: buffer management
    void set_buffer_usage_hint(int hint); // 0 = static, 1 = dynamic, 2 = stream

    int64_t get_multi_mesh_rid() const; // RenderingServer handle

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

// ============================================================================
// MultiMeshInstance3D – node that visualizes a MultiMesh resource.
// Supports all lighting features (shadows, GI, emissive) and frustum culling
// at instance level (if enabled).
// ============================================================================
class MultiMeshInstance3D : public GeometryInstance3D {
public:
    MultiMeshInstance3D();
    ~MultiMeshInstance3D();

    void set_multi_mesh(const std::shared_ptr<MultiMesh>& multi_mesh);
    std::shared_ptr<MultiMesh> get_multi_mesh() const;

    // Culling: if true, instances outside camera frustum are skipped (requires instance bounds).
    void set_frustum_culling_enabled(bool enabled);
    bool is_frustum_culling_enabled() const;

    // Provide AABB per instance for culling (optional, if not set, use mesh AABB).
    void set_instance_aabb(int index, const double* min, const double* max);
    void clear_instance_aabb(int index);

    // Overrides from GeometryInstance3D
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity) override;
    void get_emissive(float* out_color, float& out_intensity) const override;

    // Force update of all instances (call after changing MultiMesh data)
    void update_multi_mesh();

    // Node overrides
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting