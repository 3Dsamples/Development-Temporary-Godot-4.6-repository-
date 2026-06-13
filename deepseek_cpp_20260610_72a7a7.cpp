// csg_combined_3d.h
#pragma once

#include "csg_shape_3d.h"
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// CSGCombined3D – combines multiple CSG shapes using boolean operations.
// Supports union, intersection, subtraction. Uses BSP tree for accurate
// mesh boolean evaluation. All child CSG shapes are transformed and merged
// into a single mesh with proper material assignment.
// Integrated with lighting: casts shadows, receives GI, emissive surfaces.
// Optimized for real‑time CSG with dirty flag propagation.
// ============================================================================

class CSGCombined3D : public CSGShape3D {
public:
    CSGCombined3D();
    ~CSGCombined3D();

    // ------------------------------------------------------------------------
    // CSG tree management (inherited from base, but we override child handling)
    // ------------------------------------------------------------------------
    void add_csg_child(CSGShape3D* child) override;
    void remove_csg_child(CSGShape3D* child) override;
    void rebuild_csg_tree() override;

    // ------------------------------------------------------------------------
    // Lighting & GI overrides (apply to combined mesh)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // CSG interface – recalculates the combined mesh.
    // ------------------------------------------------------------------------
    void update_csg_mesh() override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting