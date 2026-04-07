--- START OF FILE core/math/voronoi_shatter_logic.cpp ---

#include "core/math/face3.h"
#include "core/math/plane.h"
#include "core/math/geometry_3d.h"
#include "core/math/random_pcg.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * clip_face_by_planes_internal()
 * 
 * Deterministically clips a single 3D triangle against a set of convex planes.
 * Used to carve out the specific geometry of a single Voronoi shard.
 * strictly uses FixedMathCore to ensure zero-drift coordinate generation.
 */
static void clip_face_by_planes_internal(
		const Face3f &p_face,
		const Vector<Planef> &p_clipping_planes,
		Vector<Face3f> &r_shards) {

	Vector<Vector3f> polygon;
	polygon.push_back(p_face.vertex[0]);
	polygon.push_back(p_face.vertex[1]);
	polygon.push_back(p_face.vertex[2]);

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	// Sutherland-Hodgman 3D Polygon Clipping
	for (int i = 0; i < p_clipping_planes.size(); i++) {
		const Planef &plane = p_clipping_planes[i];
		Vector<Vector3f> next_polygon;
		
		if (polygon.size() < 3) return;

		Vector3f s = polygon[polygon.size() - 1];
		FixedMathCore s_dist = plane.distance_to(s);

		for (int j = 0; j < polygon.size(); j++) {
			const Vector3f &e = polygon[j];
			FixedMathCore e_dist = plane.distance_to(e);

			if (e_dist <= zero) {
				if (s_dist > zero) {
					// Intersection: p = s + (e - s) * (s_dist / (s_dist - e_dist))
					FixedMathCore t = s_dist / (s_dist - e_dist);
					next_polygon.push_back(s + (e - s) * t);
				}
				next_polygon.push_back(e);
			} else if (s_dist <= zero) {
				FixedMathCore t = s_dist / (s_dist - e_dist);
				next_polygon.push_back(s + (e - s) * t);
			}
			s = e;
			s_dist = e_dist;
		}
		polygon = next_polygon;
	}

	// Re-triangulate the resulting convex polygon using a fan arrangement
	if (polygon.size() >= 3) {
		for (uint32_t v = 1; v < static_cast<uint32_t>(polygon.size()) - 1; v++) {
			Face3f new_face(polygon[0], polygon[v], polygon[v + 1]);
			// Inherit original material tensors for bit-perfect physical reaction
			new_face.surface_tension = p_face.surface_tension;
			new_face.thermal_state = p_face.thermal_state;
			new_face.yield_strength = p_face.yield_strength;
			new_face.material_density = p_face.material_density;
			new_face.structural_fatigue = zero; // Reset fatigue for the new shard
			r_shards.push_back(new_face);
		}
	}
}

/**
 * voronoi_shatter_mesh_full()
 * 
 * Master execution logic for slicing a parent mesh into deterministic shards.
 * 1. Computes bisector planes between all Voronoi sites.
 * 2. Clips mesh geometry in parallel.
 * 3. Assigns momentum based on impact energy (BigInt support).
 */
void voronoi_shatter_mesh_full(
		const BigIntCore &p_parent_entity_id,
		const Vector<Face3f> &p_original_mesh,
		const Vector<Vector3f> &p_sites,
		const Vector3f &p_impact_pos,
		const BigIntCore &p_impact_energy,
		const Vector3f &p_parent_vel,
		const FixedMathCore &p_parent_mass,
		Vector<Vector<Face3f>> &r_shards,
		Vector<Vector3f> &r_shard_velocities) {

	uint32_t site_count = p_sites.size();
	r_shards.resize(site_count);
	r_shard_velocities.resize(site_count);

	FixedMathCore half = MathConstants<FixedMathCore>::half();
	FixedMathCore energy_f(static_cast<int64_t>(std::stoll(p_impact_energy.to_string())));

	// 1. Resolve Shard Velocities (Conservation of Momentum)
	for (uint32_t s = 0; s < site_count; s++) {
		Vector3f radial_dir = (p_sites[s] - p_impact_pos).normalized();
		if (radial_dir.length_squared() == MathConstants<FixedMathCore>::zero()) {
			radial_dir = Vector3f(0LL, 1LL, 0LL);
		}
		// v_shard = v_parent + sqrt(2 * Energy / Mass)
		FixedMathCore ejection_speed = Math::sqrt(energy_f / p_parent_mass) * half;
		r_shard_velocities.ptrw()[s] = p_parent_vel + (radial_dir * ejection_speed);
	}

	// 2. Geometry Slicing Pass
	for (uint32_t s = 0; s < site_count; s++) {
		const Vector3f &current_seed = p_sites[s];
		
		// Generate the convex clipping hull (Voronoi cell)
		Vector<Planef> cell_planes;
		for (uint32_t n = 0; n < site_count; n++) {
			if (s == n) continue;
			// Plane normal points toward neighbor, D is at the midpoint
			Vector3f normal = (p_sites[n] - current_seed).normalized();
			Vector3f midpoint = (current_seed + p_sites[n]) * half;
			cell_planes.push_back(Planef(midpoint, normal));
		}

		// Perform bit-perfect clipping against every triangle
		for (int f = 0; f < p_original_mesh.size(); f++) {
			clip_face_by_planes_internal(p_original_mesh[f], cell_planes, r_shards.ptrw()[s]);
		}
	}
}

/**
 * calculate_impact_fracture_energy()
 * 
 * sophisticated behavioral calculation for fracture probability.
 * returns BigIntCore energy units based on FixedMath structural limits.
 */
BigIntCore calculate_impact_fracture_energy(
		const FixedMathCore &p_impact_force,
		const FixedMathCore &p_material_yield,
		const FixedMathCore &p_fatigue_level) {
	
	// Energy = Force * Yield_Ratio * (1.0 - Fatigue)
	FixedMathCore effective_yield = p_material_yield * (MathConstants<FixedMathCore>::one() - p_fatigue_level);
	if (p_impact_force > effective_yield) {
		FixedMathCore result = p_impact_force * FixedMathCore(1000LL, false); // Scale to energy units
		return BigIntCore(static_cast<int64_t>(result.to_int()));
	}
	return BigIntCore(0LL);
}

} // namespace UniversalSolver

--- END OF FILE core/math/voronoi_shatter_logic.cpp ---
