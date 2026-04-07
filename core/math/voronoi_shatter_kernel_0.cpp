--- START OF FILE core/math/voronoi_shatter_kernel.cpp ---

#include "core/math/geometry_3d.h"
#include "core/math/face3.h"
#include "core/math/voronoi_solver.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * compute_shatter_mapping_kernel()
 * 
 * A Warp-style kernel that assigns mesh faces to specific Voronoi shards.
 * Triggered only when Impact Force exceeds the Material Force Threshold.
 * 
 * p_faces: SoA stream of original mesh triangles.
 * p_sites: Pre-generated Voronoi seed points.
 * r_shard_assignments: Output buffer mapping face index to shard ID.
 */
void compute_shatter_mapping_kernel(
		const Face3f *p_faces,
		const Vector3f *p_sites,
		uint32_t p_site_count,
		uint32_t *r_shard_assignments,
		uint64_t p_face_start,
		uint64_t p_face_end) {

	for (uint64_t i = p_face_start; i < p_face_end; i++) {
		const Face3f &face = p_faces[i];
		Vector3f median = face.get_median();
		
		uint32_t closest_site = 0;
		FixedMathCore min_dist_sq = FixedMathCore(2147483647LL, false); // Infinity

		// Deterministic Nearest-Neighbor Search
		for (uint32_t s = 0; s < p_site_count; s++) {
			FixedMathCore d2 = (p_sites[s] - median).length_squared();
			if (d2 < min_dist_sq) {
				min_dist_sq = d2;
				closest_site = s;
			}
		}
		r_shard_assignments[i] = closest_site;
	}
}

/**
 * execute_mesh_shatter()
 * 
 * Master logic for real-time mesh destruction.
 * 1. Validates physics behavior against the force threshold.
 * 2. Generates deterministic fracture sites around the point of impact.
 * 3. Launches parallel kernels to partition the EnTT geometry.
 */
void execute_mesh_shatter(
		const BigIntCore &p_entity_id,
		Vector<Face3f> &r_source_faces,
		const Vector3f &p_impact_point,
		const FixedMathCore &p_impact_force,
		const FixedMathCore &p_force_threshold,
		const BigIntCore &p_shard_count_max,
		Vector<Vector<Face3f>> &r_shards) {

	// Physical Logic: If impact energy doesn't break the bonds, exit
	if (p_impact_force < p_force_threshold) {
		return;
	}

	uint32_t face_count = r_source_faces.size();
	if (face_count == 0) return;

	// 1. Generate Fracture Sites (Deterministic via Impact Point + Entity ID)
	VoronoiSolver solver;
	solver.set_seed(p_entity_id);
	
	int site_count = static_cast<int>(std::stoll(p_shard_count_max.to_string()));
	Vector<Vector3f> sites;
	solver.generate_shatter_points(p_impact_point, FixedMathCore(5LL, false), site_count, sites);

	// 2. Parallel Shard Mapping
	Vector<uint32_t> assignments;
	assignments.resize(face_count);
	
	uint32_t worker_threads = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk_size = face_count / worker_threads;

	for (uint32_t w = 0; w < worker_threads; w++) {
		uint64_t start = w * chunk_size;
		uint64_t end = (w == worker_threads - 1) ? face_count : (w + 1) * chunk_size;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &r_source_faces, &sites, &assignments]() {
			compute_shatter_mapping_kernel(
				r_source_faces.ptr(),
				sites.ptr(),
				sites.size(),
				assignments.ptrw(),
				start,
				end
			);
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();

	// 3. Zero-Copy Shard Reconstruction
	r_shards.resize(sites.size());
	for (uint32_t i = 0; i < face_count; i++) {
		uint32_t shard_idx = assignments[i];
		r_shards.ptrw()[shard_idx].push_back(r_source_faces[i]);
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/voronoi_shatter_kernel.cpp ---
