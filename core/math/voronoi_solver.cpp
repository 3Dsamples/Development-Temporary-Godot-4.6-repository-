--- START OF FILE core/math/voronoi_solver.cpp ---

#include "core/math/voronoi_solver.h"
#include "core/math/math_funcs.h"
#include "core/templates/hash_funcs.h"

VoronoiSolver::VoronoiSolver() {
	cell_size = FixedMathCore(5LL, false); // Default 5.0 unit cells
	global_seed = BigIntCore(12345LL);
}

VoronoiSolver::~VoronoiSolver() {}

void VoronoiSolver::set_seed(const BigIntCore &p_seed) {
	global_seed = p_seed;
}

void VoronoiSolver::set_cell_size(const FixedMathCore &p_size) {
	cell_size = p_size;
}

/**
 * _get_site_in_cell()
 * 
 * Generates a deterministic jittered point within a specific grid cell.
 * Hashes the BigInt sector and local cell indices to ensure the point
 * is identical across all simulation nodes.
 */
ET_SIMD_INLINE void VoronoiSolver::_get_site_in_cell(int64_t p_cx, int64_t p_cy, int64_t p_cz, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, VoronoiSite &r_site) const {
	uint32_t h = global_seed.hash();
	h = hash_murmur3_one_32(p_sx.hash(), h);
	h = hash_murmur3_one_32(p_sy.hash(), h);
	h = hash_murmur3_one_32(p_sz.hash(), h);
	h = hash_murmur3_one_64(static_cast<uint64_t>(p_cx), h);
	h = hash_murmur3_one_64(static_cast<uint64_t>(p_cy), h);
	h = hash_murmur3_one_64(static_cast<uint64_t>(p_cz), h);

	RandomPCG pcg;
	pcg.seed(h);

	r_site.site_id = static_cast<uint64_t>(h);
	r_site.sx = p_sx;
	r_site.sy = p_sy;
	r_site.sz = p_sz;

	// Jitter the point within the cell bounds [0, cell_size]
	FixedMathCore jx = pcg.randf() * cell_size;
	FixedMathCore jy = pcg.randf() * cell_size;
	FixedMathCore jz = pcg.randf() * cell_size;

	r_site.position = Vector3f(
		FixedMathCore(p_cx) * cell_size + jx,
		FixedMathCore(p_cy) * cell_size + jy,
		FixedMathCore(p_cz) * cell_size + jz
	);
	
	r_site.weight = pcg.randf(); // For varied crack widths
}

/**
 * get_nearest_site()
 * 
 * Performs a 3x3x3 neighborhood search to find the closest seed point.
 * Essential for O(1) "Shard Identification" during high-velocity impacts.
 */
void VoronoiSolver::get_nearest_site(const Vector3f &p_local_pos, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, VoronoiSite &r_site) const {
	int64_t gx = Math::floor(p_local_pos.x / cell_size).to_int();
	int64_t gy = Math::floor(p_local_pos.y / cell_size).to_int();
	int64_t gz = Math::floor(p_local_pos.z / cell_size).to_int();

	FixedMathCore min_dist_sq = FixedMathCore(2147483647LL, false); // Infinity

	for (int64_t x = -1; x <= 1; x++) {
		for (int64_t y = -1; y <= 1; y++) {
			for (int64_t z = -1; z <= 1; z++) {
				VoronoiSite s;
				_get_site_in_cell(gx + x, gy + y, gz + z, p_sx, p_sy, p_sz, s);
				
				FixedMathCore d2 = (s.position - p_local_pos).length_squared();
				if (d2 < min_dist_sq) {
					min_dist_sq = d2;
					r_site = s;
				}
			}
		}
	}
}

/**
 * get_edge_proximity()
 * 
 * Computes the distance to the boundary between the two closest sites.
 * Returns a value where 1.0 is exactly on the edge.
 * Foundation for deterministic "Cracking" patterns in the Universal Solver.
 */
FixedMathCore VoronoiSolver::get_edge_proximity(const Vector3f &p_local_pos, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz) const {
	int64_t gx = Math::floor(p_local_pos.x / cell_size).to_int();
	int64_t gy = Math::floor(p_local_pos.y / cell_size).to_int();
	int64_t gz = Math::floor(p_local_pos.z / cell_size).to_int();

	FixedMathCore d1_sq = FixedMathCore(2147483647LL, false);
	FixedMathCore d2_sq = FixedMathCore(2147483647LL, false);

	for (int64_t x = -1; x <= 1; x++) {
		for (int64_t y = -1; y <= 1; y++) {
			for (int64_t z = -1; z <= 1; z++) {
				VoronoiSite s;
				_get_site_in_cell(gx + x, gy + y, gz + z, p_sx, p_sy, p_sz, s);
				FixedMathCore d_sq = (s.position - p_local_pos).length_squared();

				if (d_sq < d1_sq) {
					d2_sq = d1_sq;
					d1_sq = d_sq;
				} else if (d_sq < d2_sq) {
					d2_sq = d_sq;
				}
			}
		}
	}

	// Normalized distance between first and second closest: 1.0 - (d2 - d1)
	FixedMathCore edge_dist = Math::sqrt(d2_sq) - Math::sqrt(d1_sq);
	FixedMathCore proximity = MathConstants<FixedMathCore>::one() - (edge_dist / cell_size);
	return CLAMP(proximity, MathConstants<FixedMathCore>::zero(), MathConstants<FixedMathCore>::one());
}

/**
 * generate_shatter_points()
 * 
 * Produces a deterministic point cloud centered on an impact epicenter.
 * Used to initialize dynamic fragment meshes during destruction events.
 */
void VoronoiSolver::generate_shatter_points(const Vector3f &p_epicenter, FixedMathCore p_radius, int p_count, Vector<Vector3f> &r_points) {
	RandomPCG pcg;
	pcg.seed(p_epicenter.x.get_raw() ^ p_epicenter.z.get_raw());

	for (int i = 0; i < p_count; i++) {
		// Spherical distribution via bit-perfect trig
		FixedMathCore theta = pcg.randf() * Math::tau();
		FixedMathCore phi = pcg.randf() * Math::pi();
		FixedMathCore r = pcg.randf() * p_radius;

		FixedMathCore sin_phi = Math::sin(phi);
		Vector3f point(
			r * sin_phi * Math::cos(theta),
			r * Math::cos(phi),
			r * sin_phi * Math::sin(theta)
		);

		r_points.push_back(p_epicenter + point);
	}
}

--- END OF FILE core/math/voronoi_solver.cpp ---
