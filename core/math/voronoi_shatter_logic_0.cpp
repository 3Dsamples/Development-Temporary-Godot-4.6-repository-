--- START OF FILE core/math/voronoi_shatter_logic.cpp ---

#include "core/math/geometry_3d.h"
#include "core/math/face3.h"
#include "core/math/plane.h"
#include "core/math/triangulate.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * clip_polygon_by_plane()
 * 
 * Deterministic Sutherland-Hodgman algorithm for 3D polygons.
 * clips a list of vertices against a plane, returning a new list of vertices.
 * strictly uses FixedMathCore to ensure no drift during fracture edge generation.
 */
static void clip_polygon_by_plane(
		const Vector<Vector3f> &p_vertices,
		const Planef &p_plane,
		Vector<Vector3f> &r_output) {

	r_output.clear();
	uint32_t vertex_count = p_vertices.size();
	if (vertex_count == 0) return;

	Vector3f s = p_vertices[vertex_count - 1];
	FixedMathCore s_dist = p_plane.distance_to(s);

	for (uint32_t i = 0; i < vertex_count; i++) {
		const Vector3f &e = p_vertices[i];
		FixedMathCore e_dist = p_plane.distance_to(e);

		if (e_dist <= MathConstants<FixedMathCore>::zero()) {
			if (s_dist > MathConstants<FixedMathCore>::zero()) {
				// Calculate intersection point: s + (e - s) * (dist_s / (dist_s - dist_e))
				FixedMathCore t = s_dist / (s_dist - e_dist);
				r_output.push_back(s + (e - s) * t);
			}
			r_output.push_back(e);
		} else if (s_dist <= MathConstants<FixedMathCore>::zero()) {
			FixedMathCore t = s_dist / (s_dist - e_dist);
			r_output.push_back(s + (e - s) * t);
		}
		s = e;
		s_dist = e_dist;
	}
}

/**
 * generate_voronoi_cell_clipping_planes()
 * 
 * Computes the perpendicular bisector planes between a target seed and its neighbors.
 * These planes define the convex Voronoi cell volume.
 */
static void generate_voronoi_cell_clipping_planes(
		const Vector3f &p_target_seed,
		const Vector<Vector3f> &p_neighbor_seeds,
		Vector<Planef> &r_planes) {

	r_planes.clear();
	for (int i = 0; i < p_neighbor_seeds.size(); i++) {
		const Vector3f &neighbor = p_neighbor_seeds[i];
		// Normal points from target to neighbor
		Vector3f normal = (neighbor - p_target_seed).normalized();
		// Plane origin is the midpoint
		Vector3f midpoint = (p_target_seed + neighbor) * MathConstants<FixedMathCore>::half();
		// D = dot(normal, midpoint)
		r_planes.push_back(Planef(midpoint, normal));
	}
}

/**
 * voronoi_shatter_mesh()
 * 
 * Master logic for slicing a mesh into shards.
 * 1. For each Voronoi site, compute the clipping volume (planes).
 * 2. Clip every face of the original mesh against these planes.
 * 3. Re-triangulate resulting polygons into new shard Face3 streams.
 * 4. Ensures zero-copy compatibility for EnTT registry insertion.
 */
void voronoi_shatter_mesh(
		const Vector<Face3f> &p_source_mesh,
		const Vector<Vector3f> &p_seeds,
		const BigIntCore &p_entity_id,
		Vector<Vector<Face3f>> &r_shards) {

	r_shards.resize(p_seeds.size());
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	for (uint32_t s = 0; s < p_seeds.size(); s++) {
		const Vector3f &current_seed = p_seeds[s];
		
		// 1. Gather neighboring seeds for clipping plane generation
		Vector<Vector3f> neighbors;
		for (uint32_t n = 0; n < p_seeds.size(); n++) {
			if (s == n) continue;
			neighbors.push_back(p_seeds[n]);
		}

		Vector<Planef> clipping_planes;
		generate_voronoi_cell_clipping_planes(current_seed, neighbors, clipping_planes);

		// 2. Clip original faces against this Voronoi cell
		for (uint32_t f = 0; f < p_source_mesh.size(); f++) {
			const Face3f &original_face = p_source_mesh[f];
			
			Vector<Vector3f> polygon_verts;
			polygon_verts.push_back(original_face.vertex[0]);
			polygon_verts.push_back(original_face.vertex[1]);
			polygon_verts.push_back(original_face.vertex[2]);

			Vector<Vector3f> working_poly;
			for (int p = 0; p < clipping_planes.size(); p++) {
				clip_polygon_by_plane(polygon_verts, clipping_planes[p], working_poly);
				polygon_verts = working_poly;
				if (polygon_verts.size() < 3) break;
			}

			// 3. Re-triangulate clipped polygon into Face3 components
			if (polygon_verts.size() >= 3) {
				// Fan triangulation for convex Voronoi fragments
				for (uint32_t v = 1; v < polygon_verts.size() - 1; v++) {
					Face3f new_face(polygon_verts[0], polygon_verts[v], polygon_verts[v + 1]);
					
					// Inherit Material Tensors
					new_face.surface_tension = original_face.surface_tension;
					new_face.material_density = original_face.material_density;
					new_face.thermal_state = original_face.thermal_state;
					
					r_shards.ptrw()[s].push_back(new_face);
				}
			}
		}
	}
}

/**
 * apply_fracture_impulse()
 * 
 * calculates the ejection velocity for a shard.
 * v = parent_v + (energy / shard_mass) * normalization(seed - epicenter)
 */
Vector3f calculate_shard_velocity(
		const Vector3f &p_parent_vel,
		const Vector3f &p_epicenter,
		const Vector3f &p_shard_seed,
		const FixedMathCore &p_total_energy,
		const FixedMathCore &p_shard_mass) {
	
	Vector3f dir = (p_shard_seed - p_epicenter).normalized();
	if (dir.length_squared() == MathConstants<FixedMathCore>::zero()) {
		dir = Vector3f(0LL, 1LL, 0LL);
	}

	FixedMathCore impulse_mag = p_total_energy / (p_shard_mass + FixedMathCore(1LL, true));
	return p_parent_vel + dir * impulse_mag;
}

} // namespace UniversalSolver

--- END OF FILE core/math/voronoi_shatter_logic.cpp ---
