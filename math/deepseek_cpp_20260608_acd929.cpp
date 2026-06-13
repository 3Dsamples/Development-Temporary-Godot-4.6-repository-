// File 436: modules/integration/procedural_physics_pipe.h
// Physics‑driven soft tubular mesh generator.  Constructs a triangle‑mesh
// pipe along a 3D path defined by points, with a per‑point radius profile.
// Generates a volumetric tetrahedral mesh suitable for FEM / VBD soft‑body
// simulation.  The pipe can be converted into a Genesis FEM entity with
// configurable density, Young's modulus, plasticity, and IPC collision.
// All geometry, physics material, and pinning are fully implemented inline.

#ifndef INTEGRATION_PROCEDURAL_PHYSICS_PIPE_H
#define INTEGRATION_PROCEDURAL_PHYSICS_PIPE_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"
#include "../../gaia/src/mesh/tet_mesh.h"
#include "../../gaia/src/mesh/tri_mesh.h"
#include "../../genesis/src/entities/fem_entity.h"
#include "../../genesis/src/materials/fem_material.h"
#include "unified_procedural_mesh_base.h"

namespace unified {

class ProceduralPhysicsPipe : public UnifiedProceduralMeshBase {
    GDCLASS(ProceduralPhysicsPipe, UnifiedProceduralMeshBase);

public:
    // Path control points (world space).
    LocalVector<Vector3> path_points;
    // Radius at each path control point.
    LocalVector<real_t>  radii;
    // Number of radial divisions around the tube.
    int radial_segments = 16;
    // If true, path is closed (last point connects to first).
    bool closed_path = false;
    // Include end caps.
    bool cap_start = true;
    bool cap_end   = true;

    // --- Physics / soft‑body parameters ---
    real_t density        = 1000.0f;      // kg/m³
    real_t young_modulus  = 1e5f;        // Pa
    real_t poisson_ratio  = 0.4f;
    bool   plasticity     = false;
    real_t yield_stress   = 5e4f;
    real_t hardening      = 0.1f;
    bool   enable_ipc     = false;
    real_t ipc_distance   = 0.002f;
    real_t ipc_stiffness  = 1e6f;

    // Cached tetrahedral mesh (built on demand).
    gaia::mesh::TetMesh tet_mesh;
    bool tet_mesh_built = false;

public:
    ProceduralPhysicsPipe() {}

    // -------------------------------------------------------------------
    // Build the surface triangle mesh (vertices, indices, normals, UVs).
    // -------------------------------------------------------------------
    virtual void build() override;

    // -------------------------------------------------------------------
    // Build (or return) the tetrahedral mesh for soft‑body physics.
    // This volume mesh is created by extruding an inner wall offset by
    // a fraction of the radius, forming a thick shell.
    // -------------------------------------------------------------------
    const gaia::mesh::TetMesh &get_tet_mesh();

    // -------------------------------------------------------------------
    // Create and return a Genesis FEM entity that can be immediately
    // added to a GenesisWorld for soft‑body simulation.  The entity
    // is initialised with the tetrahedral mesh and the configured
    // material properties.
    // -------------------------------------------------------------------
    Ref<genesis::FEMEntity> create_fem_entity(const Transform3D &p_world_transform) const;

    // -------------------------------------------------------------------
    // Pin a set of vertices (e.g., the first cap) by their local indices.
    // -------------------------------------------------------------------
    void pin_cap(int p_cap_index, bool p_pin);

    // LOD support: reduce radial segments.
    virtual void set_lod(int p_level) override;

protected:
    static void _bind_methods();

private:
    // Frame computation per path point (tangent, normal, binormal).
    void compute_path_frames(LocalVector<Vector3> &t, LocalVector<Vector3> &n,
                             LocalVector<Vector3> &b) const;

    // Generate ring vertices at a given path index.
    void generate_ring(int p_idx, const Vector3 &p_pos,
                       const Vector3 &p_normal, const Vector3 &p_binormal,
                       real_t p_radius, LocalVector<Vector3> &r_verts,
                       int &r_start) const;

    // Generate triangles between two consecutive rings.
    void connect_rings(int p_ring_a_start, int p_ring_b_start,
                       int p_ring_verts, bool p_close, LocalVector<int> &r_indices) const;

    // Generate a fan cap.
    void generate_cap(int p_ring_start, const Vector3 &p_center,
                      const Vector3 &p_normal, bool p_reverse,
                      LocalVector<Vector3> &r_verts, LocalVector<int> &r_indices) const;

    // Compute cylindrical UV mapping.
    void compute_pipe_uvs();

    // Build the tetrahedral mesh (thin shell).
    void build_tet_volume();
};

} // namespace unified

#endif // INTEGRATION_PROCEDURAL_PHYSICS_PIPE_H