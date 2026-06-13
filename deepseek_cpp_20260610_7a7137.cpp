// csg_sphere_3d.h
#pragma once

#include "csg_shape_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// CSGSphere3D – CSG sphere primitive.
// Supports radius, detail (radial and rings), smooth/flat shading,
// hemisphere (half sphere), and material assignment.
// Integrated with lighting: casts shadows, receives GI, can be emissive.
// Optimized for real‑time CSG with dirty flag propagation.
// ============================================================================

class CSGSphere3D : public CSGShape3D {
public:
    CSGSphere3D();
    ~CSGSphere3D();

    // ------------------------------------------------------------------------
    // Sphere geometry
    // ------------------------------------------------------------------------
    void set_radius(double radius);
    double get_radius() const;
    void set_radial_segments(int segments);   // longitudinal divisions (around equator)
    int get_radial_segments() const;
    void set_rings(int rings);                // latitudinal divisions (from pole to pole)
    int get_rings() const;
    void set_hemisphere(bool enable);         // if true, only top half (y >= 0)
    bool is_hemisphere() const;

    // ------------------------------------------------------------------------
    // Material assignment
    // ------------------------------------------------------------------------
    void set_material(int material_id);       // material for entire sphere
    void set_smooth_shading(bool smooth);
    bool is_smooth_shading() const;

    // ------------------------------------------------------------------------
    // Lighting & GI overrides
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;      // 0=off,1=static,2=dynamic
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // CSG interface
    // ------------------------------------------------------------------------
    void update_csg_mesh() override;          // regenerate geometry when parameters change

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting