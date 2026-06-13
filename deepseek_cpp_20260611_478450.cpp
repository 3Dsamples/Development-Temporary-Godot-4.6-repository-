// csg_mesh_3d.h
#pragma once

#include "csg_shape_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// CSGMesh3D – CSG primitive that loads an arbitrary mesh (from file or resource)
// and uses it in CSG operations. Supports mesh triangulation, material assignment,
// and transformation. Integrated with lighting: casts shadows, receives GI,
// emissive surfaces. Optimized for real‑time CSG with dirty flag propagation.
// ============================================================================

class CSGMesh3D : public CSGShape3D {
public:
    CSGMesh3D();
    ~CSGMesh3D();

    // ------------------------------------------------------------------------
    // Mesh source
    // ------------------------------------------------------------------------
    void set_mesh(int64_t mesh_rid);          // from RenderingServer
    int64_t get_mesh_rid() const;
    void set_mesh_path(const char* path);     // load from asset file (e.g., .obj, .gltf)
    const char* get_mesh_path() const;

    // ------------------------------------------------------------------------
    // Material assignment (override per surface)
    // ------------------------------------------------------------------------
    void set_material(int material_id);
    int get_material() const;
    void set_surface_material(int surface_idx, int material_id);
    int get_surface_material(int surface_idx) const;

    // ------------------------------------------------------------------------
    // Mesh processing (triangulation, simplification)
    // ------------------------------------------------------------------------
    void set_triangulate(bool triangulate);
    bool is_triangulate() const;
    void set_simplify(float ratio);           // 0..1, 1 = no simplification
    float get_simplify() const;

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
    void update_csg_mesh() override;           // loads mesh data and processes

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting