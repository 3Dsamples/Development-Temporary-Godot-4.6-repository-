--- START OF FILE core/math/voronoi_solver_geometry.cpp ---

#include "core/math/voronoi_solver.h"
#include "core/math/math_funcs.h"
#include "core/math/face3.h"
#include "core/math/plane.h"
#include "core/math/noise_simplex.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * generate_jagged_bisector_plane()
 * 
 * Computes a bit-perfect clipping plane between two Voronoi sites.
 * Sophisticated Feature: Injects deterministic Simplex Noise to warp the plane.
 * This ensures that shards have "natural" rough edges instead of laser-flat faces.
 */
static _FORCE_INLINE_ Planef generate_jagged_bisector_plane(
		const Vector3f &p_site_a,
		const Vector3f &p_site_b,
		const SimplexNoisef &p_noise,
		const BigIntCore &p_fracture_seed) {

	Vector3f normal = (p_site_b - p_site_a).normalized();
	Vector3f midpoint = (p_site_a + p_site_b) * MathConstants<FixedMathCore>::half();

	// Sample noise based on the midpoint to perturb the plane's 'D' coefficient
	// Uses raw bits to ensure no FPU involvement
	FixedMathCore noise_val = p_noise.sample_3d(midpoint.x, midpoint.y, midpoint.z);
	
	// Amplitude of jaggedness scales with the distance between sites
	FixedMathCore amplitude = (p_site_b - p_site_a).length() * FixedMathCore(214748364LL, true); // 0.05 scaling
	
	midpoint += normal * (noise_val * amplitude);

	// Hessian Normal Form: ax + by + cz = d
	return Planef(midpoint, normal);
}

/**
 * Warp Kernel: MeshSlicingKernel
 * 
 * Slices a parent mesh into shards.
 * 1. For each Voronoi site, generates a convex hull of jagged planes.
 * 2. Clips every triangle of the original mesh against the hull.
 * 3. Reconstructs resulting polygons into valid Face3 tensors.
 */
void mesh_slicing_kernel(
		const Face3f *p_source_faces,
		uint64_t p_source_count,
		const Vector3f *p_sites,
		uint32_t p_site_count,
		const SimplexNoisef &p_noise,
		const BigIntCore &p_seed,
		Vector<Vector<Face3f>> &r_output_shards,
		uint32_t p_shard_start,
		uint32_t p_shard_end) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore half = MathConstants<FixedMathCore>::half();

	for (uint32_t s = p_shard_start; s < p_shard_end; s++) {
		const Vector3f &target_site = p_sites[s];
		
		// 1. Build the clipping volume for this Voronoi Cell
		Vector<Planef> cell_boundary;
		for (uint32_t n = 0; n < p_site_count; n++) {
			if (s == n) continue;
			cell_boundary.push_back(generate_jagged_bisector_plane(target_site, p_sites[n], p_noise, p_seed));
		}

		// 2. Clip all source faces against the cell volume
		for (uint64_t f = 0; f < p_source_count; f++) {
			const Face3f &original_face = p_source_faces[f];
			
			Vector<Vector3f> polygon;
			polygon.push_back(original_face.vertex[0]);
			polygon.push_back(original_face.vertex[1]);
			polygon.push_back(original_face.vertex[2]);

			// Sutherland-Hodgman loop (Bit-Perfect)
			for (int p = 0; p < cell_boundary.size(); p++) {
				const Planef &plane = cell_boundary[p];
				Vector<Vector3f> next_polygon;
				if (polygon.size() < 3) break;

				Vector3f S = polygon[polygon.size() - 1];
				FixedMathCore dist_s = plane.distance_to(S);

				for (int i = 0; i < polygon.size(); i++) {
					const Vector3f &E = polygon[i];
					FixedMathCore dist_e = plane.distance_to(E);

					if (dist_e <= zero) {
						if (dist_s > zero) {
							FixedMathCore t = dist_s / (dist_s - dist_e);
							next_polygon.push_back(S + (E - S) * t);
						}
						next_polygon.push_back(E);
					} else if (dist_s <= zero) {
						FixedMathCore t = dist_s / (dist_s - dist_e);
						next_polygon.push_back(S + (E - S) * t);
					}
					S = E;
					dist_s = dist_e;
				}
				polygon = next_polygon;
			}

			// 3. Re-triangulate the clipped convex polygon
			if (polygon.size() >= 3) {
				for (uint32_t v = 1; v < static_cast<uint32_t>(polygon.size()) - 1; v++) {
					Face3f shard_face(polygon[0], polygon[v], polygon[v + 1]);
					
					// Material state inheritance: ensure shards carry parent thermal/fatigue tensors
					shard_face.thermal_state = original_face.thermal_state;
					shard_face.surface_tension = original_face.surface_tension;
					shard_face.material_density = original_face.material_density;
					shard_face.structural_fatigue = zero; // Energy released in fracture

					r_output_shards.ptrw()[s].push_back(shard_face);
				}
			}
		}
	}
}

/**
 * execute_parallel_shatter_reconstruction()
 * 
 * Master orchestrator for shard geometry generation.
 * Partitions the shard-list across physical CPU cores to maintain 120 FPS.
 */
void execute_parallel_shatter_reconstruction(
		const Face3f *p_mesh_data,
		uint64_t p_mesh_face_count,
		const Vector<Vector3f> &p_seeds,
		const BigIntCore &p_fracture_seed,
		Vector<Vector<Face3f>> &r_shards) {

	SimplexNoisef fracture_noise;
	fracture_noise.seed(p_fracture_seed);

	uint32_t shard_count = p_seeds.size();
	r_shards.resize(shard_count);

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint32_t chunk = shard_count / workers;
	if (chunk == 0) chunk = 1;

	for (uint32_t w = 0; w < workers; w++) {
		uint32_t start = w * chunk;
		uint32_t end = (w == workers - 1) ? shard_count : (start + chunk);
		if (start >= shard_count) break;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &p_seeds, &r_shards, &fracture_noise]() {
			mesh_slicing_kernel(
				p_mesh_data,
				p_mesh_face_count,
				p_seeds.ptr(),
				shard_count,
				fracture_noise,
				p_fracture_seed,
				r_shards,
				start,
				end
			);
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/voronoi_solver_geometry.cpp ---
