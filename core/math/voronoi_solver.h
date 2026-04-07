--- START OF FILE core/math/voronoi_solver.h ---

#ifndef VORONOI_SOLVER_H
#define VORONOI_SOLVER_H

#include "core/typedefs.h"
#include "core/math/vector3.h"
#include "core/templates/vector.h"
#include "core/templates/hash_map.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"
#include "core/math/random_pcg.h"

/**
 * VoronoiSolver
 * 
 * High-performance deterministic Voronoi cell generator.
 * Used for procedural shattering, tectonic cracking, and galactic region partitioning.
 * Optimized for Warp-style parallel sweeps over EnTT component streams.
 */
class ET_ALIGN_32 VoronoiSolver {
public:
	/**
	 * VoronoiSite
	 * Represents a seed point for a Voronoi cell.
	 */
	struct ET_ALIGN_32 VoronoiSite {
		Vector3f position;        // Local to sector
		BigIntCore sx, sy, sz;    // Galactic sector coordinates
		uint64_t site_id;         // Deterministic hash ID
		FixedMathCore weight;     // For Weighted/Power Voronoi (cracking variation)
	};

private:
	FixedMathCore cell_size;      // Size of the jitter grid
	BigIntCore global_seed;

	/**
	 * _get_site_in_cell()
	 * Deterministically generates a single site within a grid cell using PCG.
	 */
	ET_SIMD_INLINE void _get_site_in_cell(int64_t p_cx, int64_t p_cy, int64_t p_cz, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, VoronoiSite &r_site) const;

public:
	// ------------------------------------------------------------------------
	// Deterministic Query API
	// ------------------------------------------------------------------------

	/**
	 * get_nearest_site()
	 * Returns the closest Voronoi site for a given point.
	 * Essential for determining which "shard" an impact point belongs to.
	 */
	void get_nearest_site(const Vector3f &p_local_pos, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, VoronoiSite &r_site) const;

	/**
	 * get_edge_proximity()
	 * Calculates the distance to the nearest Voronoi edge.
	 * Used for "Cracking" shaders and physical structural failure lines.
	 * Returns FixedMathCore in range [0, 1] where 0 is the center and 1 is the edge.
	 */
	FixedMathCore get_edge_proximity(const Vector3f &p_local_pos, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz) const;

	/**
	 * generate_shatter_points()
	 * Generates a batch of points around an epicenter for procedural fracturing.
	 * Returns points as Vector3f for local mesh clipping.
	 */
	void generate_shatter_points(const Vector3f &p_epicenter, FixedMathCore p_radius, int p_count, Vector<Vector3f> &r_points);

	// ------------------------------------------------------------------------
	// Lifecycle
	// ------------------------------------------------------------------------

	void set_seed(const BigIntCore &p_seed);
	void set_cell_size(const FixedMathCore &p_size);

	VoronoiSolver();
	~VoronoiSolver();
};

#endif // VORONOI_SOLVER_H

--- END OF FILE core/math/voronoi_solver.h ---
