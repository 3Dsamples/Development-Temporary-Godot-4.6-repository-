// visual_instance_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <array>

namespace lighting {

// ============================================================================
// VisualInstance3D – base for all renderable 3D objects
// Provides: visibility, layers, bounding box, instance ID for render server,
//           LOD levels, occlusion culling support, and GPU instancing hints.
// ============================================================================

enum class LODLevel : uint8_t {
    VERY_HIGH,
    HIGH,
    MEDIUM,
    LOW,
    VERY_LOW
};

class VisualInstance3D : public Node3D {
public:
    VisualInstance3D();
    ~VisualInstance3D();

    // ------------------------------------------------------------------------
    // Visibility & layers
    // ------------------------------------------------------------------------
    void set_visible(bool visible) override;
    bool is_visible() const override;
    void set_layer_mask(uint32_t mask);
    uint32_t get_layer_mask() const;
    void set_shadow_casting_enabled(bool enabled);
    bool is_shadow_casting_enabled() const;
    void set_gi_mode(int mode); // 0=static, 1=dynamic, 2=off
    int get_gi_mode() const;

    // ------------------------------------------------------------------------
    // Bounding volume (AABB) for frustum culling & LOD
    // ------------------------------------------------------------------------
    void set_aabb(const double* min, const double* max);
    void get_aabb(double* out_min, double* out_max) const;
    void set_bounding_sphere_radius(double radius);
    double get_bounding_sphere_radius() const;
    void update_bounding_volume(); // recompute from geometry (overridden by subclasses)

    // ------------------------------------------------------------------------
    // Level of detail (LOD)
    // ------------------------------------------------------------------------
    void set_lod_level(LODLevel level);
    LODLevel get_lod_level() const;
    void set_lod_distances(const double* distances); // array of 4 distances (very_high -> very_low)
    void get_lod_distances(double* out_distances) const;
    LODLevel compute_lod_from_distance(double distance) const;

    // ------------------------------------------------------------------------
    // Occlusion culling (hook for GPU occlusion queries)
    // ------------------------------------------------------------------------
    void set_occlusion_culling_enabled(bool enabled);
    bool is_occlusion_culling_enabled() const;
    void set_occlusion_query_id(int id);
    int get_occlusion_query_id() const;
    void set_last_occlusion_result(bool visible);
    bool get_last_occlusion_result() const;

    // ------------------------------------------------------------------------
    // GPU instancing (for many identical objects)
    // ------------------------------------------------------------------------
    void set_instancing_enabled(bool enabled);
    bool is_instancing_enabled() const;
    void set_instancing_group_id(int group);
    int get_instancing_group_id() const;
    void set_instance_transform_index(int index);
    int get_instance_transform_index() const;

    // ------------------------------------------------------------------------
    // Material overrides
    // ------------------------------------------------------------------------
    void set_material_override(const char* material_path); // would load material resource
    void clear_material_override();
    bool has_material_override() const;

    // ------------------------------------------------------------------------
    // Sorting & render order
    // ------------------------------------------------------------------------
    void set_render_priority(int priority);
    int get_render_priority() const;
    void set_transparency_sorting(bool enabled);
    bool get_transparency_sorting() const;

    // ------------------------------------------------------------------------
    // Light / reflection probe influence
    // ------------------------------------------------------------------------
    void set_lightmap_uv_scale(const double* scale); // for baked lighting
    void get_lightmap_uv_scale(double* out_scale) const;
    void set_reflection_probe_id(int probe_id);
    int get_reflection_probe_id() const;

    // ------------------------------------------------------------------------
    // Rendering server synchronization
    // ------------------------------------------------------------------------
    void notify_visibility_changed();
    void synchronize_render_server(double delta) override;
    int64_t get_render_instance_id() const; // handle to RenderingServer instance

protected:
    // Called when bounding volume needs recomputation (e.g., after transform)
    void _transform_changed() override;
    virtual void _update_render_instance_transform();
    virtual void _update_render_instance_visibility();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting