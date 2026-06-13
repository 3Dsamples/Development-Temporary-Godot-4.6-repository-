// path_3d.h
#pragma once

#include "node_3d.h"
#include <memory>
#include <vector>
#include <cstdint>

namespace lighting {

// ============================================================================
// Path3D – 3D curve path (Catmull‑Rom / Bezier / linear)
// Can be used for camera tracks, moving platforms, or particle trails.
// Supports point interpolation, global transforms, and baking into points.
// ============================================================================

enum class CurveType : uint8_t {
    LINEAR,
    CATMULL_ROM,
    BEZIER,
    CUBIC_SPLINE
};

struct PathPoint {
    Vector3 position;
    Vector3 in_tangent;   // for Bezier
    Vector3 out_tangent;
    float tilt = 0.0f;    // twist angle in degrees
    bool smooth = true;
};

class Path3D : public Node3D {
public:
    Path3D();
    ~Path3D();

    // ------------------------------------------------------------------------
    // Point management
    // ------------------------------------------------------------------------
    void add_point(const PathPoint& point);
    void insert_point(int index, const PathPoint& point);
    void remove_point(int index);
    void clear_points();
    int get_point_count() const;
    void set_point(int index, const PathPoint& point);
    PathPoint get_point(int index) const;

    // ------------------------------------------------------------------------
    // Path parameters
    // ------------------------------------------------------------------------
    void set_curve_type(CurveType type);
    CurveType get_curve_type() const;
    void set_closed(bool closed);
    bool is_closed() const;
    void set_curve_resolution(int steps); // number of interpolated segments between points
    int get_curve_resolution() const;

    // ------------------------------------------------------------------------
    // Evaluation (world space)
    // ------------------------------------------------------------------------
    Vector3 get_point_at_ratio(float t) const;      // t in [0,1]
    Vector3 get_tangent_at_ratio(float t) const;
    float get_total_length() const;
    Vector3 get_up_direction_at_ratio(float t) const; // for camera alignment
    float get_closest_ratio(const Vector3& world_pos, int max_iterations = 16) const;

    // ------------------------------------------------------------------------
    // Baking to discrete points (for high performance)
    // ------------------------------------------------------------------------
    void bake_points(); // pre‑compute interpolated points
    const std::vector<Vector3>& get_baked_points() const;
    const std::vector<float>& get_baked_distances() const;

    // ------------------------------------------------------------------------
    // Lighting and rendering (visualization of path, optional)
    // ------------------------------------------------------------------------
    void set_visible(bool visible) override;
    void set_material(const char* material_path);
    void set_line_width(float width);
    void set_color(float r, float g, float b);

    // ------------------------------------------------------------------------
    // Render server sync (for debug visualization)
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

// ============================================================================
// PathFollow3D – moves a node along a Path3D
// Supports offset, rotation, and absolute/relative positioning.
// ============================================================================

class PathFollow3D : public Node3D {
public:
    PathFollow3D();
    ~PathFollow3D();

    void set_path(Path3D* path);
    Path3D* get_path() const;

    void set_ratio(float ratio);           // 0..1
    float get_ratio() const;
    void set_offset(float distance);       // offset in world units
    float get_offset() const;
    void set_rotation_mode(int mode);      // 0=none, 1=orient to path, 2=orient with tilt
    int get_rotation_mode() const;

    void set_cubic_interpolation(bool enable);
    bool is_cubic_interpolation_enabled() const;

    // ------------------------------------------------------------------------
    // Update transform (call each frame, or during physics)
    // ------------------------------------------------------------------------
    void update_position();

    // ------------------------------------------------------------------------
    // Advance along path automatically (for moving platforms)
    // ------------------------------------------------------------------------
    void set_auto_advance(bool enable, float speed = 1.0f);
    void set_loop(bool loop);
    void advance(float delta_time);

    // ------------------------------------------------------------------------
    // Lighting influence (if the followed object casts shadows)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast);
    bool get_cast_shadow() const;
    void set_gi_mode(int mode);
    int get_gi_mode() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting