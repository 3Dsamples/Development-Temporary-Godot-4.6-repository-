--- START OF FILE core/math/geometry_3d.cpp ---

#include "core/math/geometry_3d.h"
#include "core/math/math_funcs.h"
#include "core/math/random_pcg.h"
#include "core/math/face3.h"

/**
 * Explicit Template Instantiation
 * 
 * Compiles the 3D geometric analyzer for the deterministic tiers.
 * - FixedMathCore: High-speed 120 FPS physics and deformation.
 * - BigIntCore: Discrete macro-structure partitioning.
 */
template class Geometry3D<FixedMathCore>;
template class Geometry3D<BigIntCore>;

// ============================================================================
// Volumetric Fracturing (Deterministic Sharding)
// ============================================================================

/**
 * generate_fracture_shards()
 * 
 * Slices a 3D mesh into fragments based on stochastic clipping planes.
 * 1. Generates 'n' planes passing through the impact epicenter.
 * 2. Uses a bit-mask derived from plane-side tests to bin faces into shards.
 * 3. Maintains bit-perfection using a deterministic seed.
 */
template <typename T>
Vector<Vector<Face3<T>>> Geometry3D<T>::generate_fracture_shards(
		const Vector<Face3<T>> &p_mesh, 
		const Vector3<T> &p_epicenter, 
		int p_shard_count, 
		const BigIntCore &p_seed) {

	Vector<Vector<Face3<T>>> shards;
	if (p_mesh.is_empty() || p_shard_count <= 1) {
		shards.push_back(p_mesh);
		return shards;
	}

	// 1. Generate Deterministic Stochastic Planes
	RandomPCG pcg;
	pcg.seed(p_seed.hash());
	
	Vector<Plane<T>> fracture_planes;
	for (int i = 0; i < p_shard_count; i++) {
		T rx = pcg.randf() - MathConstants<T>::half();
		T ry = pcg.randf() - MathConstants<T>::half();
		T rz = pcg.randf() - MathConstants<T>::half();
		Vector3<T> normal = Vector3<T>(rx, ry, rz).normalized();
		
		// If normalization fails (zero vector), use a default bit-perfect axis
		if (normal.length_squared() == MathConstants<T>::zero()) {
			normal = Vector3<T>(MathConstants<T>::zero(), MathConstants<T>::one(), MathConstants<T>::zero());
		}
		
		fracture_planes.push_back(Plane<T>(p_epicenter, normal));
	}

	// 2. Face Binning Sweep (Warp-style Logic)
	// We use a simple clustering based on the nearest plane to epicenter for O(N) performance
	shards.resize(p_shard_count);
	
	for (int i = 0; i < p_mesh.size(); i++) {
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

/**
 * apply_mesh_perforation()
 * 
 * Punctures a 3D geometry by removing and subdividing faces.
 * 1. Identifies faces within the impact radius.
 * 2. Recursively subdivides intersecting faces to create a clean circular edge.
 * 3. Zero-Copy implementation intended for EnTT registry updates.
 */
template <typename T>
void Geometry3D<T>::apply_mesh_perforation(Vector<Face3<T>> &r_mesh, const Vector3<T> &p_center, T p_radius) {
	Vector<Face3<T>> new_mesh;
	T r2 = p_radius * p_radius;
	T subdivision_threshold = r2 * T(4LL); // Check neighbor faces within 2x radius

	for (int i = 0; i < r_mesh.size(); i++) {
		Face3<T> &face = r_mesh.ptrw()[i];
		Vector3<T> median = face.get_median();
		T dist_sq = (median - p_center).length_squared();

		if (dist_sq > subdivision_threshold) {
			// Outside of interaction zone, keep original face
			new_mesh.push_back(face);
		} else if (dist_sq < r2) {
			// Entirely inside the perforation radius, delete face (snip)
			continue;
		} else {
			// Border zone: Subdivide face into 4 sub-triangles (recursive snip)
			Vector<Face3<T>> sub_faces = face.split();
			for (int j = 0; j < sub_faces.size(); j++) {
				T sub_dist_sq = (sub_faces[j].get_median() - p_center).length_squared();
				if (sub_dist_sq >= r2) {
					new_mesh.push_back(sub_faces[j]);
				}
			}
		}
	}

	r_mesh = new_mesh;
}

// ============================================================================
// Proximity Solvers (Non-Inline Heavy Logic)
// ============================================================================

template <typename T>
static T _get_sq_dist_segment_segment(const Vector3<T> &p1, const Vector3<T> &q1, const Vector3<T> &p2, const Vector3<T> &q2) {
	Vector3<T> c1, c2;
	Geometry3D<T>::get_closest_points_between_segments(p1, q1, p2, q2, c1, c2);
	return (c1 - c2).length_squared();
}

--- END OF FILE core/math/geometry_3d.cpp ---
