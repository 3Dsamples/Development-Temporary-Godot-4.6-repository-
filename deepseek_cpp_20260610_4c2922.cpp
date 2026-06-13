// collision_object_3d.h
#pragma once

#include "visual_instance_3d.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>

namespace lighting {

// ============================================================================
// CollisionObject3D – base for all physics-aware objects
// Manages collision shapes, layers/masks, and collision signals.
// ============================================================================

enum class CollisionShapeType : uint8_t {
    SPHERE,
    BOX,
    CAPSULE,
    CYLINDER,
    CONVEX_POLYHEDRON,
    CONCAVE_MESH,
    HEIGHTFIELD
};

struct CollisionShape {
    CollisionShapeType type;
    void* shape_data; // Shape-specific parameters (radius, extents, etc.)
    Transform3D local_transform;
    bool disabled = false;
    int id = -1;
};

class CollisionObject3D : public VisualInstance3D {
public:
    CollisionObject3D();
    ~CollisionObject3D();

    // ------------------------------------------------------------------------
    // Shape management
    // ------------------------------------------------------------------------
    int add_shape(const CollisionShape& shape);
    void remove_shape(int shape_id);
    void clear_shapes();
    int get_shape_count() const;
    CollisionShape get_shape(int shape_id) const;
    void set_shape_transform(int shape_id, const Transform3D& transform);
    Transform3D get_shape_transform(int shape_id) const;

    // ------------------------------------------------------------------------
    // Layer & mask (collision and hit detection)
    // ------------------------------------------------------------------------
    void set_collision_layer(uint32_t layer);
    uint32_t get_collision_layer() const;
    void set_collision_mask(uint32_t mask);
    uint32_t get_collision_mask() const;
    void set_collision_priority(float priority);
    float get_collision_priority() const;

    // ------------------------------------------------------------------------
    // Lighting integration (dynamic shadows from collision objects)
    // ------------------------------------------------------------------------
    void set_cast_collision_shadow(bool cast);
    bool get_cast_collision_shadow() const;
    void set_gi_collision_contribution(float amount);
    float get_gi_collision_contribution() const;

    // ------------------------------------------------------------------------
    // Collision signals (for raycast & area overlap)
    // ------------------------------------------------------------------------
    using CollisionCallback = std::function<void(const double* point, const double* normal, int shape_id, int other_shape_id)>;
    void set_collision_callback(CollisionCallback callback);
    void set_area_enter_callback(CollisionCallback callback);
    void set_area_exit_callback(CollisionCallback callback);

    // ------------------------------------------------------------------------
    // Raycasting directly on this object (without physics server)
    // ------------------------------------------------------------------------
    bool intersect_ray(const double* origin, const double* direction, double max_distance, double* out_point, double* out_normal) const;

    // ------------------------------------------------------------------------
    // Physics server synchronization
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;

protected:
    virtual void _shape_added(int shape_id);
    virtual void _shape_removed(int shape_id);
    virtual void _collision_detected(const double* point, const double* normal, int with_shape);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

// ============================================================================
// Area3D – detection zone (overlaps, gravity/effects)
// ============================================================================

class Area3D : public CollisionObject3D {
public:
    Area3D();
    ~Area3D();

    void set_gravity_enabled(bool enabled);
    bool is_gravity_enabled() const;
    void set_gravity(const double* gravity_vector);
    void get_gravity(double* out_gravity) const;
    void set_gravity_point(bool point);
    bool is_gravity_point() const;
    void set_gravity_point_center(const double* center);
    void get_gravity_point_center(double* out_center) const;

    void set_linear_damp_enabled(bool enabled);
    void set_linear_damp(float damp);
    float get_linear_damp() const;
    void set_angular_damp(float damp);
    float get_angular_damp() const;

    void set_priority(int priority);
    int get_priority() const;

    // Overlap query
    bool overlaps_body(int64_t body_rid) const;
    std::vector<int64_t> get_overlapping_bodies() const;

    // Lighting area influence (modulate GI)
    void set_gi_override(float intensity);
    float get_gi_override() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting