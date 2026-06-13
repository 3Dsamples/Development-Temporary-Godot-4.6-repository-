// csg_cylinder_3d.h
#pragma once

#include "csg_shape_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// CSGCylinder3D – CSG cylinder primitive (can also be cone, truncated cone).
// Supports radius at top and bottom, height, radial segments, caps, and
// smooth/flat shading. Integrated with lighting: casts shadows, receives GI,
// emissive surfaces. Optimized for real‑time CSG with dirty flag propagation.
// ============================================================================

class CSGCylinder3D : public CSGShape3D {
public:
    CSGCylinder3D();
    ~CSGCylinder3D();

    // ------------------------------------------------------------------------
    // Cylinder geometry
    // ------------------------------------------------------------------------
    void set_height(double height);
    double get_height() const;
    void set_radius(double radius);             // sets both top and bottom
    void set_radius_top(double radius);
    double get_radius_top() const;
    void set_radius_bottom(double radius);
    double get_radius_bottom() const;
    void set_radial_segments(int segments);     // 3–64
    int get_radial_segments() const;
    void set_cone(bool is_cone);                // if true, radius_top = 0
    bool is_cone() const;

    // ------------------------------------------------------------------------
    // Caps (end caps)
    // ------------------------------------------------------------------------
    void set_caps_enabled(bool enabled);
    bool are_caps_enabled() const;

    // ------------------------------------------------------------------------
    // Material assignment
    // ------------------------------------------------------------------------
    void set_material(int material_id);         // material for all parts
    void set_material_side(int side, int material_id); // side: 0=body,1=top cap,2=bottom cap
    void set_smooth_shading(bool smooth);
    bool is_smooth_shading() const;

    // ------------------------------------------------------------------------
    // Lighting & GI overrides
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;        // 0=off,1=static,2=dynamic
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // CSG interface
    // ------------------------------------------------------------------------
    void update_csg_mesh() override;            // regenerate geometry when parameters change

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting