// area_3d.h
#pragma once

#include "collision_object_3d.h"
#include <cstdint>
#include <memory>
#include <functional>
#include <vector>

namespace lighting {

// ============================================================================
// Area3D – region that detects overlapping bodies and can override physics
// parameters (gravity, damping). Used for triggers, zones, and physics effects.
// Can also contribute to global illumination (GI) as an emissive volume or
// override ambient GI intensity.
// ============================================================================

class Area3D : public CollisionObject3D {
public:
    Area3D();
    ~Area3D();

    // ------------------------------------------------------------------------
    // Gravity override
    // ------------------------------------------------------------------------
    void set_gravity_enabled(bool enabled);
    bool is_gravity_enabled() const;
    void set_gravity(const double* gravity_vector);
    void get_gravity(double* out_gravity) const;
    void set_gravity_point(bool point);
    bool is_gravity_point() const;
    void set_gravity_point_center(const double* center);
    void get_gravity_point_center(double* out_center) const;
    void set_gravity_distance_scale(float scale);
    float get_gravity_distance_scale() const;

    // ------------------------------------------------------------------------
    // Linear / angular damping overrides
    // ------------------------------------------------------------------------
    void set_linear_damp_enabled(bool enabled);
    bool is_linear_damp_enabled() const;
    void set_linear_damp(float damp);
    float get_linear_damp() const;
    void set_angular_damp_enabled(bool enabled);
    bool is_angular_damp_enabled() const;
    void set_angular_damp(float damp);
    float get_angular_damp() const;
    void set_damp_priority(int priority);
    int get_damp_priority() const;

    // ------------------------------------------------------------------------
    // Overlap detection and signals
    // ------------------------------------------------------------------------
    using BodyCallback = std::function<void(int64_t body_rid, int64_t body_instance_id, int body_shape, int area_shape)>;
    void set_body_entered_callback(BodyCallback callback);
    void set_body_exited_callback(BodyCallback callback);
    void set_area_entered_callback(BodyCallback callback);
    void set_area_exited_callback(BodyCallback callback);
    bool overlaps_body(int64_t body_rid) const;
    std::vector<int64_t> get_overlapping_bodies() const;
    std::vector<int64_t> get_overlapping_areas() const;

    // ------------------------------------------------------------------------
    // Priority (for overlapping areas)
    // ------------------------------------------------------------------------
    void set_priority(int priority);
    int get_priority() const;

    // ------------------------------------------------------------------------
    // Global illumination contribution (area can affect GI intensity or add light)
    // ------------------------------------------------------------------------
    void set_gi_mode(int mode) override;      // 0=off,1=static,2=dynamic
    int get_gi_mode() const;
    void set_gi_contribution(float amount);
    float get_gi_contribution() const;
    void set_emissive_gi(const float* color, float intensity);
    void get_emissive_gi(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Physics server synchronization
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

    // ------------------------------------------------------------------------
    // Force update (call after shape changes)
    // ------------------------------------------------------------------------
    void update_area();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting