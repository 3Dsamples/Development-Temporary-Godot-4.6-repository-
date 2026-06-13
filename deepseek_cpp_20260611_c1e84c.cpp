// csg_polygon_3d.h
#pragma once

#include "csg_shape_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// CSGPolygon3D – CSG polygon primitive, extruded or turned into a prism.
// Can be used to create 2D shapes extruded along Z or rotated (lathe).
// Supports polygon triangulation, depth, and material assignment.
// Integrated with lighting: casts shadows, receives GI, can be emissive.
// ============================================================================

class CSGPolygon3D : public CSGShape3D {
public:
    CSGPolygon3D();
    ~CSGPolygon3D();

    // ------------------------------------------------------------------------
    // Polygon definition (2D points in local XY plane)
    // ------------------------------------------------------------------------
    void set_polygon(const std::vector<double>& points); // [x0,y0, x1,y1, ...]
    std::vector<double> get_polygon() const;

    // ------------------------------------------------------------------------
    // Extrusion / mode
    // ------------------------------------------------------------------------
    void set_depth(double depth);            // extrusion depth (0 = flat)
    double get_depth() const;
    void set_extrusion_mode(int mode);       // 0=flat, 1=extruded, 2=lathe
    int get_extrusion_mode() const;
    void set_lathe_angle(double angle);      // rotation angle (radians) for lathe mode
    double get_lathe_angle() const;
    void set_lathe_smooth(bool smooth);
    bool is_lathe_smooth() const;

    // ------------------------------------------------------------------------
    // Triangulation options
    // ------------------------------------------------------------------------
    void set_use_ear_clipping(bool use);
    bool is_ear_clipping() const;
    void set_max_convex_pieces(int pieces);
    int get_max_convex_pieces() const;

    // ------------------------------------------------------------------------
    // Material
    // ------------------------------------------------------------------------
    void set_material(int material_id);
    int get_material() const;

    // ------------------------------------------------------------------------
    // Lighting & GI
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // CSG interface
    // ------------------------------------------------------------------------
    void update_csg_mesh() override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting