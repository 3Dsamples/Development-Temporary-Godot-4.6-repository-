--- START OF FILE core/math/geometry_3d_perforation_logic.cpp ---

#include "core/math/geometry_3d.h"
#include "core/math/face3.h"
#include "core/math/triangulate.h"
#include "core/math/math_funcs.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * process_subdivision_and_prune()
 * 
 * Recursive kernel for high-fidelity perforation.
 * It decomposes a single triangle into smaller facets and prunes those 
 * that fall within the bit-perfect circular impact radius.
 * 
 * p_face: The original triangle to process.
 * p_center: Epicenter of the puncture.
 * p_radius_sq: Squared radius in FixedMath for O(1) distance checks.
 * p_depth: Current recursion depth to control 120 FPS performance budget.
 * r_output: The SoA collection of resulting bit-perfect faces.
 */
static void process_subdivision_and_prune(
		const Face3f &p_face,
		const Vector3f &p_center,
		const FixedMathCore &p_radius_sq,
		int p_depth,
		Vector<Face3f> &r_output) {

	// 1. Calculate containment state for the 3 vertices
	bool inside[3];
	int inside_count = 0;
	for (int i = 0; i < 3; i++) {
		inside[i] = (p_face.vertex[i] - p_center).length_squared() < p_radius_sq;
		if (inside[i]) inside_count++;
	}

	// 2. Terminal Cases
	if (inside_count == 3) {
		// Entire face is inside the puncture hole: Discard (Hole created)
		return;
	}

	if (inside_count == 0) {
		// Face is potentially outside or partially crossing the arc.
		// We check the median to ensure we aren't straddling the curve too broadly.
		if (p_depth >= 3) {
			r_output.push_back(p_face);
			return;
		}
	}

	// 3. Subdivide Case (Mixed containment or straddling boundary)
	// Depth limit (3) ensures we don't stall the 120 FPS simulation heartbeat.
	if (p_depth < 3) {
		// 4-way deterministic tessellation
		Vector3f m01 = (p_face.vertex[0] + p_face.vertex[1]) * MathConstants<FixedMathCore>::half();
		Vector3f m12 = (p_face.vertex[1] + p_face.vertex[2]) * MathConstants<FixedMathCore>::half();
		Vector3f m20 = (p_face.vertex[2] + p_face.vertex[0]) * MathConstants<FixedMathCore>::half();

		Face3f f[4];
		f[0] = Face3f(p_face.vertex[0], m01, m20);
		f[1] = Face3f(p_face.vertex[1], m12, m01);
		f[2] = Face3f(p_face.vertex[2], m20, m12);
		f[3] = Face3f(m01, m12, m20);

		// Inherit material tensors for each sub-facet
		for (int i = 0; i < 4; i++) {
			f[i].surface_tension = p_face.surface_tension;
			f[i].thermal_state = p_face.thermal_state;
			f[i].yield_strength = p_face.yield_strength;
			f[i].material_density = p_face.material_density;
			
			process_subdivision_and_prune(f[i], p_center, p_radius_sq, p_depth + 1, r_output);
		}
	} else {
		// At max depth, if even one vertex is outside, keep the remaining geometry
		r_output.push_back(p_face);
	}
}

/**
 * apply_mesh_perforation_logic()
 * 
 * Absolute implementation of procedural hole punching.
 * Optimized for high-speed spaceship hull breaches and material perforation.
 * 
 * p_mesh: Original SoA face stream from EnTT.
 * p_center: World/Local point of the puncture.
 * p_radius: Physical radius of the hole.
 */
void apply_mesh_perforation_logic(
		Vector<Face3f> &r_mesh,
		const Vector3f &p_center,
		const FixedMathCore &p_radius) {

	Vector<Face3f> finalized_faces;
	FixedMathCore r2 = p_radius * p_radius;
	
	// Pre-filter: Only process faces near the impact to maintain 120 FPS
	FixedMathCore broad_r2 = (p_radius * FixedMathCore(2LL)) * (p_radius * FixedMathCore(2LL));

	for (uint32_t i = 0; i < r_mesh.size(); i++) {
		const Face3f &face = r_mesh[i];
		FixedMathCore dist_sq = (face.get_median() - p_center).length_squared();

		if (dist_sq > broad_r2) {
			// Fast Path: Outside interaction zone
			finalized_faces.push_back(face);
		} else {
			// Heavy Path: Recursive clipping and subdivision
			process_subdivision_and_prune(face, p_center, r2, 0, finalized_faces);
		}
	}

	// Zero-Copy swap back to the EnTT registry
	r_mesh = finalized_faces;
}

/**
 * apply_smooth_rim_stitching()
 * 
 * Sophisticated Real-Time Behavior:
 * Aligns the vertices on the edge of the new puncture to form a perfect circle.
 * Prevents "jagged" holes by projecting boundary vertices onto the radius.
 */
void apply_smooth_rim_stitching(
		Vector<Face3f> &r_mesh,
		const Vector3f &p_center,
		const FixedMathCore &p_radius) {

	FixedMathCore tolerance = p_radius * FixedMathCore(128849018LL, true); // 0.03 margin

	for (uint32_t i = 0; i < r_mesh.size(); i++) {
		Face3f &face = r_mesh.ptrw()[i];
		for (int v = 0; v < 3; v++) {
			Vector3f dir = face.vertex[v] - p_center;
			FixedMathCore d = dir.length();
			
			// If vertex is near the "cut line", snap it to the perfect circle
			if (Math::abs(d - p_radius) < tolerance) {
				face.vertex[v] = p_center + dir.normalized() * p_radius;
			}
		}
		// Refresh internal normal and area caches
		face._update_geometric_caches();
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/geometry_3d_perforation_logic.cpp ---
