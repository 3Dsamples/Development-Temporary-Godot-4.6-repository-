// path_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// Path3D – defines a 3D curve (CatmullRom or Bezier) for movement or extrusion
// Supports lighting of debug visualization and integrated with GI for dynamic paths.
// ============================================================================

enum class PathCurveType : uint8_t {
    CATMULL_ROM,
    CUBIC_BEZIER,
    LINEAR
};

struct PathPoint {
    double position[3];
    double tangent_in[3];   // for Bezier
    double tangent_out[3];
    float tilt;             // in radians
    bool enabled = true;
};

class Path3D : public Node3D {
public:
    Path3D();
    ~Path3D();

    // ------------------------------------------------------------------------
    // Curve manipulation
    // ------------------------------------------------------------------------
    void set_curve_type(PathCurveType type);
    PathCurveType get_curve_type() const;
    int add_point(const PathPoint& point);
    void remove_point(int index);
    void set_point(int index, const PathPoint& point);
    PathPoint get_point(int index) const;
    int get_point_count() const;
    void clear_points();
    void update_curve();                // recompute internal cache

    // ------------------------------------------------------------------------
    // Evaluation along path (t in [0,1])
    // ------------------------------------------------------------------------
    void evaluate_point(double t, double* out_position) const;
    void evaluate_tangent(double t, double* out_tangent) const;
    void evaluate_tilt(double t, float* out_tilt) const;
    double get_length() const;          // total arc length
    double get_arc_length(double t) const; // length from start to t (normalized)

    // ------------------------------------------------------------------------
    // Lighting integration – path can be rendered as debug geometry with lighting
    // ------------------------------------------------------------------------
    void set_debug_material(const char* material_path);
    void set_debug_line_width(float width);
    void set_debug_color(float r, float g, float b);
    void set_cast_shadow(bool cast);
    bool get_cast_shadow() const;

    // ------------------------------------------------------------------------
    // GI contribution (dynamic paths can influence light probes)
    // ------------------------------------------------------------------------
    void set_gi_contribution(float amount);
    float get_gi_contribution() const;

    // ------------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting