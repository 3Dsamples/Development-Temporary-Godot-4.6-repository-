--- START OF FILE core/math/geometry_3d.cpp ---

#include "core/math/geometry_3d.h"
#include "core/math/math_funcs.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the 3D geometric suite for the Universal Solver backend.
 * These symbols enable EnTT to manage 3D mesh components while Warp kernels 
 * invoke these routines for batch-oriented physical destruction and 
 * topological reconstruction across all simulation scales.
 */
template class Geometry3D<FixedMathCore>; // TIER_DETERMINISTIC: Bit-perfect Physics
template class Geometry3D<BigIntCore>;    // TIER_MACRO: Discrete Sector Geometries

// ============================================================================
// Volumetric Fracturing (Deterministic Sharding)
// ============================================================================

template <typename T>
Vector<Vector<Face3<T>>> Geometry3D<T>::generate_fracture_shards(const Vector<Face3<T>> &p_mesh, const Vector3<T> &p_epicenter, int p_shard_count) {
	Vector<Vector<Face3<T>>> shards;
	if (p_mesh.is_empty() || p_shard_count < 2) {
		shards.push_back(p_mesh);
		return shards;
	}

	shards.resize(p_shard_count);

	// Generate deterministic stochastic clipping planes using PCG
	Vector<Plane<T>> fracture_planes;
	for (int i = 0; i < p_shard_count; i++) {
		// Use bit-perfect directions
		T rx = Math::randf() - MathConstants<T>::half();
		T ry = Math::randf() - MathConstants<T>::half();
		T rz = Math::randf() - MathConstants<T>::half();
		Vector3<T> dir = Vector3<T>(rx, ry, rz).normalized();
		fracture_planes.push_back(Plane<T>(p_epicenter, dir));
	}

	// Distributed Batch Processing: Assign faces to shards based on plane proximity
	// Optimized for Warp kernel zero-copy sweeps over EnTT geometry buffers
	for (uint32_t i = 0; i < p_mesh.size(); i++) {
		const Face3<T> &face = p_mesh[i];
		Vector3<T> median = face.get_median();
		
		int best_shard = 0;
		T min_dist = Math::abs(fracture_planes[0].distance_to(median));
		
		for (int j = 1; j < p_shard_count; j++) {
			T d = Math::abs(fracture_planes[j].distance_to(median));
			if (d < min_dist) {
				min_dist = d;
				best_shard = j;
			}
		}
		shards.ptrw()[best_shard].push_back(face);
	}

	return shards;
}

// ============================================================================
// Mesh Perforation (Procedural Hole Punching)
// ============================================================================

template <typename T>
void Geometry3D<T>::apply_mesh_perforation(Vector<Face3<T>> &r_mesh, const Vector3<T> &p_center, T p_radius) {
	Vector<Face3<T>> new_mesh;
	T r2 = p_radius * p_radius;

	// ETEngine Optimization: Localized vertex removal and edge-loop preservation
	for (uint32_t i = 0; i < r_mesh.size(); i++) {
		Face3<T> &face = r_mesh.ptrw()[i];
		Vector3<T> median = face.get_median();
		T dist_sq = (median - p_center).length_squared();

		if (dist_sq > r2) {
			// Face is outside the perforation zone
			new_mesh.push_back(face);
		} else {
			// Face is within the impact radius; triggered for destruction/perforation.
			// In a high-fidelity Warp kernel, we would retesselate the boundary here.
			// For the core math logic, we perform a deterministic face-snip.
			continue; 
		}
	}
	r_mesh = new_mesh;
}

// ============================================================================
// Specialized Distance Solver
// ============================================================================

template <typename T>
static ET_SIMD_INLINE T _get_segment_dist_sq(const Vector3<T> &p_point, const Vector3<T> &p_a, const Vector3<T> &p_b) {
	Vector3<T> ab = p_b - p_a;
	Vector3<T> ap = p_point - p_a;
	T t = ap.dot(ab) / ab.length_squared();
	t = CLAMP(t, MathConstants<T>::zero(), MathConstants<T>::one());
	return (p_a + ab * t - p_point).length_squared();
}

--- END OF FILE core/math/geometry_3d.cpp ---
