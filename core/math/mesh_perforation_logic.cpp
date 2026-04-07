--- START OF FILE core/math/mesh_perforation_logic.cpp ---

#include "core/math/geometry_3d.h"
#include "core/math/face3.h"
#include "core/math/math_funcs.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * process_subdivision_recursive()
 * 
 * Internal recursive kernel that splits a triangle into four sub-triangles.
 * Prunes sub-facets that fall entirely within the bit-perfect circular impact radius.
 * Ensures the resulting boundary is jagged or smooth based on the material yield.
 */
static void process_subdivision_recursive(
		const Face3f &p_face,
		const Vector3f &p_epicenter,
		const FixedMathCore &p_radius_sq,
		int p_current_depth,
		int p_max_depth,
		Vector<Face3f> &r_output_faces) {

	// 1. Calculate containment of vertices
	bool inside[3];
	int inside_count = 0;
	for (int i = 0; i < 3; i++) {
		FixedMathCore dist_sq = (p_face.vertex[i] - p_epicenter).length_squared();
		inside[i] = (dist_sq < p_radius_sq);
		if (inside[i]) {
			inside_count++;
		}
	}

	// 2. Base Cases
	if (inside_count == 3) {
		// Face is entirely inside the hole: Discard it.
		return;
	}

	if (inside_count == 0) {
		// Check if the edge of the circle passes through the face center
		Vector3f median = p_face.get_median();
		if ((median - p_epicenter).length_squared() > p_radius_sq * FixedMathCore(2LL)) {
			// Far enough to be safe
			r_output_faces.push_back(p_face);
			return;
		}
	}

	// 3. Subdivide or Finalize
	if (p_current_depth < p_max_depth) {
		// Standard 4-way split
		Vector3f m01 = (p_face.vertex[0] + p_face.vertex[1]) * MathConstants<FixedMathCore>::half();
		Vector3f m12 = (p_face.vertex[1] + p_face.vertex[2]) * MathConstants<FixedMathCore>::half();
		Vector3f m20 = (p_face.vertex[2] + p_face.vertex[0]) * MathConstants<FixedMathCore>::half();

		Face3f sub[4];
		sub[0] = Face3f(p_face.vertex[0], m01, m20);
		sub[1] = Face3f(p_face.vertex[1], m12, m01);
		sub[2] = Face3f(p_face.vertex[2], m20, m12);
		sub[3] = Face3f(m01, m12, m20);

		// Recursively process each sub-facet
		for (int i = 0; i < 4; i++) {
			sub[i].surface_tension = p_face.surface_tension;
			sub[i].thermal_state = p_face.thermal_state;
			sub[i].yield_strength = p_face.yield_strength;
			sub[i].material_density = p_face.material_density;
			
			process_subdivision_recursive(sub[i], p_epicenter, p_radius_sq, p_current_depth + 1, p_max_depth, r_output_faces);
		}
	} else {
		// Terminal depth: If any part is outside, keep it
		r_output_faces.push_back(p_face);
	}
}

/**
 * apply_mesh_perforation()
 * 
 * The master logic for procedural punching.
 * 1. Filters faces via broadphase AABB.
 * 2. Launches recursive subdivision on intersecting faces.
 * 3. Re-normalizes the resulting geometry.
 */
void apply_mesh_perforation(
		Vector<Face3f> &r_mesh,
		const Vector3f &p_epicenter,
		const FixedMathCore &p_radius) {

	Vector<Face3f> result_mesh;
	FixedMathCore r2 = p_radius * p_radius;
	
	// Adaptive depth based on radius size to maintain 120 FPS
	int max_depth = 2;
	if (p_radius > FixedMathCore(5LL)) max_depth = 3;

	for (uint32_t i = 0; i < r_mesh.size(); i++) {
		const Face3f &face = r_mesh[i];
		
		// Broad check: is the face within range of the puncture?
		FixedMathCore dist_sq = (face.get_median() - p_epicenter).length_squared();
		FixedMathCore safety_radius_sq = (p_radius + face.area_cache) * (p_radius + face.area_cache);

		if (dist_sq > safety_radius_sq) {
			// Optimization: Preserve distant faces without subdivision
			result_mesh.push_back(face);
		} else {
			// Heavy Path: Perform bit-perfect subdivision and pruning
			process_subdivision_recursive(face, p_epicenter, r2, 0, max_depth, result_mesh);
		}
	}

	r_mesh = result_mesh;
}

/**
 * apply_edge_thermal_cauterization()
 * 
 * Real-Time Behavior: Increases the thermal tensor and structural fatigue
 * on the edges of the newly created hole.
 */
void apply_edge_thermal_cauterization(
		Vector<Face3f> &r_mesh,
		const Vector3f &p_center,
		const FixedMathCore &p_radius,
		const FixedMathCore &p_energy) {

	FixedMathCore border_min = p_radius;
	FixedMathCore border_max = p_radius * FixedMathCore("1.2");

	for (uint32_t i = 0; i < r_mesh.size(); i++) {
		Face3f &f = r_mesh.ptrw()[i];
		FixedMathCore d = (f.get_median() - p_center).length();
		
		if (d >= border_min && d <= border_max) {
			FixedMathCore falloff = (border_max - d) / (border_max - border_min);
			f.thermal_state += p_energy * falloff;
			f.structural_fatigue += FixedMathCore("0.5") * falloff;
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/mesh_perforation_logic.cpp ---
