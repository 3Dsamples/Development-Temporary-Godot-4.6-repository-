// File 119: modules/gaia/src/vbd_cloth/contact_relations.h
// Builds and maintains candidate contact pairs between a deformable triangle mesh
// (cloth) and a set of rigid body colliders. Uses spatial hashing to accelerate
// proximity queries. Each pair consists of a cloth vertex and a rigid triangle.

#ifndef GAIA_VBD_CLOTH_CONTACT_RELATIONS_H
#define GAIA_VBD_CLOTH_CONTACT_RELATIONS_H

#include "../vbd_cloth/vbd_base_tri_mesh.h"   // VBDBaseTriMesh (for cloth vertex positions)
#include "../collision_detector/collision_object.h" // CollisionObject, ConvexShape etc.
#include "../spatial_query/spatial_hash.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"

namespace gaia::vbd_cloth {

struct ContactPair {
	int cloth_vertex;               // index into the cloth mesh
	int rigid_body_index;           // index into the rigid body array (or -1 for world)
	int rigid_triangle_v0;          // vertex indices of the triangle in the rigid collider
	int rigid_triangle_v1;
	int rigid_triangle_v2;
	real_t distance;                // current separating distance (negative = penetration)
};

class ContactRelations {
public:
	// Clear all pairs.
	void clear() { pairs.clear(); }

	/**
	 * Build candidate contact pairs between a cloth mesh and a set of rigid
	 * collision objects (each providing a triangle surface). Only pairs with
	 * distance less than `max_distance` are kept.
	 *
	 * @param cloth_mesh        The deformable triangle mesh.
	 * @param rigid_objects     Array of rigid collision objects (each with a triangle surface).
	 * @param max_distance      Maximum proximity to consider.
	 */
	void build_pairs(const VBDBaseTriMesh &cloth_mesh,
					 const LocalVector<collision::CollisionObject *> &rigid_objects,
					 real_t max_distance = 0.1) {
		clear();
		// Build spatial hash for cloth vertices
		spatial::SpatialHash hash(max_distance);
		for (int i = 0; i < cloth_mesh.vertex_count(); ++i) {
			hash.insert(i, cloth_mesh.get_vertex(i).pos);
		}

		// For each rigid object, extract its world-space triangle soup and test
		// nearby cloth vertices.
		for (int ri = 0; ri < rigid_objects.size(); ++ri) {
			const collision::CollisionObject *obj = rigid_objects[ri];
			if (!obj->is_active()) continue;

			// We assume a convex or tri-mesh collider that exposes a list of triangles.
			// For demonstration, we call a virtual method get_triangles(std::vector<...>) that
			// every collider must implement. If it's a simple sphere, we skip.
			LocalVector<Vector3> tri_verts; // serialized: 3 per triangle
			LocalVector<uint32_t> tri_indices; // indices if indexed
			obj->get_shape()->get_triangle_mesh(tri_verts, tri_indices);
			if (tri_verts.is_empty() && tri_indices.is_empty()) continue;

			// Transform to world space
			const Transform3D &xform = obj->get_transform();
			// For indexed mesh, process each triangle
			int num_tris = tri_indices.size() / 3;
			if (num_tris == 0) num_tris = tri_verts.size() / 3;
			for (int t = 0; t < num_tris; ++t) {
				Vector3 v0, v1, v2;
				if (!tri_indices.is_empty()) {
					v0 = xform.xform(tri_verts[tri_indices[t*3]]);
					v1 = xform.xform(tri_verts[tri_indices[t*3+1]]);
					v2 = xform.xform(tri_verts[tri_indices[t*3+2]]);
				} else {
					v0 = xform.xform(tri_verts[t*3]);
					v1 = xform.xform(tri_verts[t*3+1]);
					v2 = xform.xform(tri_verts[t*3+2]);
				}
				// Compute AABB of this triangle
				AABB tri_aabb(v0, Vector3());
				tri_aabb.expand_to(v1);
				tri_aabb.expand_to(v2);
				tri_aabb = tri_aabb.grow(max_distance);

				// Query cloth vertices near this triangle
				LocalVector<int32_t> candidate_verts;
				// Approximate: query points in the bounding box (use hash query with center and neighbors)
				Vector3 center = tri_aabb.get_center();
				candidate_verts.clear();
				hash.query(center, candidate_verts, true);

				// For each candidate cloth vertex, compute distance to triangle
				for (int vi : candidate_verts) {
					const Vector3 &p = cloth_mesh.get_vertex(vi).pos;
					// Skip pinned vertices if desired? Keep for now.
					real_t dist = point_triangle_distance(p, v0, v1, v2);
					if (dist < max_distance) {
						ContactPair cp;
						cp.cloth_vertex = vi;
						cp.rigid_body_index = ri;
						cp.rigid_triangle_v0 = t*3;
						cp.rigid_triangle_v1 = t*3+1;
						cp.rigid_triangle_v2 = t*3+2;
						cp.distance = dist;
						pairs.push_back(cp);
					}
				}
			}
		}
	}

	// Access the list of contact pairs.
	const LocalVector<ContactPair> &get_pairs() const { return pairs; }

private:
	// Squared distance from point to triangle (returns actual distance not squared).
	static real_t point_triangle_distance(const Vector3 &p, const Vector3 &a, const Vector3 &b, const Vector3 &c) {
		Vector3 ab = b - a;
		Vector3 ac = c - a;
		Vector3 ap = p - a;
		real_t d1 = ab.dot(ap);
		real_t d2 = ac.dot(ap);
		if (d1 <= 0 && d2 <= 0) return p.distance_to(a); // closest to a

		Vector3 bp = p - b;
		real_t d3 = ab.dot(bp);
		real_t d4 = ac.dot(bp);
		if (d3 >= 0 && d4 <= d3) return p.distance_to(b); // closest to b

		real_t vc = d1 * d4 - d3 * d2;
		if (vc <= 0 && d1 >= 0 && d3 <= 0) {
			real_t v = d1 / (d1 - d3);
			return p.distance_to(a + ab * v);
		}

		Vector3 cp = p - c;
		real_t d5 = ab.dot(cp);
		real_t d6 = ac.dot(cp);
		if (d6 >= 0 && d5 <= d6) return p.distance_to(c); // closest to c

		real_t vb = d5 * d2 - d1 * d6;
		if (vb <= 0 && d2 >= 0 && d6 <= 0) {
			real_t w = d2 / (d2 - d6);
			return p.distance_to(a + ac * w);
		}

		real_t va = d3 * d6 - d5 * d4;
		if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
			real_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
			return p.distance_to(b + (c - b) * w);
		}

		// Inside face
		real_t denom = 1.0 / (va + vb + vc);
		real_t v = vb * denom;
		real_t w = vc * denom;
		real_t u = 1.0 - v - w;
		return p.distance_to(a * u + b * v + c * w);
	}

	LocalVector<ContactPair> pairs;
};

} // namespace gaia::vbd_cloth

#endif // GAIA_VBD_CLOTH_CONTACT_RELATIONS_H