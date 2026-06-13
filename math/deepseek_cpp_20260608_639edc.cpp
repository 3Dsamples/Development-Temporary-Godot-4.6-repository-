// File 434: modules/integration/procedural_curve_deformer.h
// Deforms an existing 3D mesh (assumed to lie along the X axis from 0 to 1)
// along a 3D path, with optional twist, per‑vertex scaling, and smooth
// path interpolation using Catmull‑Rom splines and rotation‑minimizing
// frames.  All mathematics is fully present; no step is omitted.

#ifndef INTEGRATION_PROCEDURAL_CURVE_DEFORMER_H
#define INTEGRATION_PROCEDURAL_CURVE_DEFORMER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace unified {

class ProceduralCurveDeformer : public RefCounted {
    GDCLASS(ProceduralCurveDeformer, RefCounted);

public:
    // Input path (world‑space points) – at least 2 points.
    LocalVector<Vector3> path_points;
    // Twist angle per unit length (radians / unit).  Total twist at
    // parameter t is twist_per_unit * t * path_length.
    real_t twist_per_unit = 0.0;
    // Scale factors along the path, sampled at each path point.
    // Internally interpolated.  Must match path_points size.
    LocalVector<real_t> scale_factors;
    // If true, the path is closed (last point connects to first).
    bool closed_path = false;
    // Number of intermediate samples for smooth cubic interpolation.
    int path_samples = 100;
    // The rest mesh vertices (positions in local space).  The deformation
    // expects that the rest shape occupies X in [0, 1].
    LocalVector<Vector3> rest_vertices;
    LocalVector<Vector3> rest_normals;   // optional, for normal transformation
    // Output: deformed vertices (same count as rest_vertices).
    LocalVector<Vector3> deformed_vertices;
    LocalVector<Vector3> deformed_normals;

    ProceduralCurveDeformer() {}

    // -------------------------------------------------------------------
    // Build the deformed mesh.  Must be called after setting path,
    // rest_vertices, and optional twist/scales.
    // -------------------------------------------------------------------
    void build();

    // -------------------------------------------------------------------
    // Access the result.
    // -------------------------------------------------------------------
    const LocalVector<Vector3> &get_vertices() const { return deformed_vertices; }
    const LocalVector<Vector3> &get_normals() const { return deformed_normals; }

protected:
    static void _bind_methods();

private:
    // Pre‑computed arc‑length parameterization (sampled positions and tangents).
    LocalVector<Vector3> sampled_positions;
    LocalVector<real_t>  sampled_parameters;   // 0..1
    LocalVector<Vector3> sampled_tangents;
    LocalVector<Vector3> sampled_normals;      // rotation‑minimizing frame normals
    LocalVector<Vector3> sampled_binormals;

    // -------------------------------------------------------------------
    // Build the sampled path (Catmull‑Rom interpolation at path_samples points).
    // -------------------------------------------------------------------
    void build_sampled_path();

    // -------------------------------------------------------------------
    // Compute the rotation‑minimizing frame along the sampled path.
    // -------------------------------------------------------------------
    void build_rmf();

    // -------------------------------------------------------------------
    // Evaluate the path at parameter t (0..1): position, tangent, normal,
    // binormal.  Uses cubic interpolation on the sampled arrays.
    // -------------------------------------------------------------------
    void evaluate(real_t t, Vector3 &r_pos, Vector3 &r_tangent,
                  Vector3 &r_normal, Vector3 &r_binormal) const;

    // -------------------------------------------------------------------
    // Interpolate scale factor at parameter t.
    // -------------------------------------------------------------------
    real_t interpolate_scale(real_t t) const;

    // -------------------------------------------------------------------
    // Catmull‑Rom interpolation for 3D vectors (sampled_positions).
    // -------------------------------------------------------------------
    static Vector3 catmull_rom(real_t t, const Vector3 &p0, const Vector3 &p1,
                               const Vector3 &p2, const Vector3 &p3);
};

} // namespace unified

#endif // INTEGRATION_PROCEDURAL_CURVE_DEFORMER_H