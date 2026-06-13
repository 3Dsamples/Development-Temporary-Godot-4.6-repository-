// path_3d.h
#pragma once

#include "node_3d.h"
#include <memory>
#include <vector>
#include <cstdint>

namespace lighting {

// ============================================================================
// Curve3D – 3D curve made of bezier points (simplified)
// ============================================================================

struct CurvePoint {
    double position[3];
    double forward[3];   // incoming control point
    double backward[3];  // outgoing control point
    double tilt;         // bank angle in radians
};

class Curve3D {
public:
    Curve3D();
    ~Curve3D();

    void add_point(const double* pos, const double* in = nullptr, const double* out = nullptr, double tilt = 0.0);
    void clear_points();
    int get_point_count() const;
    void get_point(int idx, double* out_pos, double* out_in, double* out_out, double& out_tilt) const;
    void set_point(int idx, const double* pos, const double* in, const double* out, double tilt);
    void remove_point(int idx);

    // Evaluation
    double get_length() const; // approximate total length
    void sample_at(double t, double* out_pos, double* out_tangent, double& out_tilt) const; // t in [0,1]
    double get_closest_offset(const double* world_pos) const; // nearest point on curve

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

// ============================================================================
// Path3D – visual representation of a curve for debug / navigation
// Supports lighting (emissive material for glowing paths) and shadows.
// ============================================================================

class Path3D : public Node3D {
public:
    Path3D();
    ~Path3D();

    // Curve access
    void set_curve(std::shared_ptr<Curve3D> curve);
    std::shared_ptr<Curve3D> get_curve() const;

    // Visibility of the path (debug drawing)
    void set_path_visible(bool visible);
    bool is_path_visible() const;
    void set_path_color(const float* rgb);
    void get_path_color(float* out_rgb) const;
    void set_path_width(float width);
    float get_path_width() const;

    // Lighting & shadowing for the debug path (if used as a visible element)
    void set_cast_shadow(bool cast);
    bool get_cast_shadow() const;
    void set_receive_shadow(bool receive);
    bool get_receive_shadow() const;
    void set_gi_mode(int mode);
    int get_gi_mode() const;

    // Emissive light from path (e.g., neon sign)
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // Rendering
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting