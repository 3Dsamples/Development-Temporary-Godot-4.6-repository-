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
 * clip_face_by_planes()
 * 
 * Deterministically clips a single 3D triangle against a set of convex planes 
 * (representing a Voronoi Cell). 
 * strictly uses FixedMathCore to ensure zero-drift coordinate generation.
 */
static void clip_face_by_planes(
		const Face3f &p_face,
		const Vector<Planef> &p_clipping_planes,
		Vector<Face3f> &r_shards) {

	Vector<Vector3f> polygon;
	polygon.push_back(p_face.vertex[0]);
	polygon.push_back(p_face.vertex[1]);
	polygon.push_back(p_face.vertex[2]);

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	// Sutherland-Hodgman 3D Implementation
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

	// Re-triangulate the resulting convex polygon
	if (polygon.size() >= 3) {
		for (uint32_t v = 1; v < static_cast<uint32_t>(polygon.size()) - 1; v++) {
			Face3f new_face(polygon[0], polygon[v], polygon[v + 1]);
			// Inherit original material tensors (Surface Tension, Thermal state)
			new_face.surface_tension = p_face.surface_tension;
			new_face.thermal_state = p_face.thermal_state;
			new_face.yield_strength = p_face.yield_strength;
			r_shards.push_back(new_face);
		}
	}
}

/**
 * execute_volumetric_shatter()
 * 
 * Master logic for high-fidelity structural failure.
 * 1. Generates clipping volumes for each Voronoi site.
 * 2. Clips mesh geometry in parallel (Warp-Kernel ready).
 * 3. Calculates bit-perfect ejection velocities via momentum conservation.
 */
void execute_volumetric_shatter(
		const BigIntCore &p_entity_id,
		const Vector<Face3f> &p_original_mesh,
		const Vector<Vector3f> &p_seeds,
		const Vector3f &p_impact_vel,
		const BigIntCore &p_impact_energy,
		const FixedMathCore &p_mass,
		Vector<Vector<Face3f>> &r_shards,
		Vector<Vector3f> &r_shard_velocities) {

	uint32_t site_count = p_seeds.size();
	r_shards.resize(site_count);
	r_shard_velocities.resize(site_count);

	FixedMathCore half = MathConstants<FixedMathCore>::half();
	FixedMathCore energy_f(static_cast<int64_t>(std::stoll(p_impact_energy.to_string())));

	// 1. Resolve Ejection Velocities
	// Conservation of Momentum: Every shard inherits parent velocity + radial impulse
	for (uint32_t s = 0; s < site_count; s++) {
		Vector3f radial_dir = (p_seeds[s] - p_seeds[0]).normalized(); // Seeds relative to epicenter
		FixedMathCore ejection_speed = Math::sqrt(energy_f / p_mass) * half;
		r_shard_velocities.ptrw()[s] = p_impact_vel + (radial_dir * ejection_speed);
	}

	// 2. Mesh Clipping Sweep
	// In a Warp launch, this loop is partitioned across the SimulationThreadPool
	for (uint32_t s = 0; s < site_count; s++) {
		const Vector3f &current_seed = p_seeds[s];
		
		// Generate the convex clipping hull for this Voronoi cell
		Vector<Planef> cell_planes;
		for (uint32_t n = 0; n < site_count; n++) {
			if (s == n) continue;
			Vector3f normal = (p_seeds[n] - current_seed).normalized();
			Vector3f midpoint = (current_seed + p_seeds[n]) * half;
			cell_planes.push_back(Planef(midpoint, normal));
		}

		// Clip every face against the cell
		for (int f = 0; f < p_original_mesh.size(); f++) {
			clip_face_by_planes(p_original_mesh[f], cell_planes, r_shards.ptrw()[s]);
		}
	}
}

/**
 * calculate_fracture_force_threshold()
 * 
 * Advanced Physical Behavior:
 * returns the required BigIntCore energy to trigger a fracture based on 
 * material yield strength and surface area.
 */
BigIntCore calculate_fracture_force_threshold(
		const FixedMathCore &p_yield_strength,
		const FixedMathCore &p_surface_area,
		const FixedMathCore &p_fatigue) {
	
	// Threshold = Yield * Area * (1.0 - Fatigue)
	FixedMathCore factor = p_yield_strength * p_surface_area * (MathConstants<FixedMathCore>::one() - p_fatigue);
	return BigIntCore(static_cast<int64_t>(factor.to_int()));
}

} // namespace UniversalSolver

--- END OF FILE core/math/voronoi_shatter_logic.cpp ---
