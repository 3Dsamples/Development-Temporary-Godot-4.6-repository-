// File 239: modules/newton/src/collision/newton_collision_tree.h
// Static triangle mesh collision tree – wraps a Gaia BVH of triangles
// for high‑performance ray‑casts and GJK support using convex hull fallback.
// Used for complex static level geometry.

#ifndef NEWTON_COLLISION_TREE_H
#define NEWTON_COLLISION_TREE_H

#include "newton_collision.h"
#include "newton_convex_hull.h"
#include "../../../gaia/src/bvh/bvh.h"
#include "../../../gaia/src/bvh/query.h"
#include "core/templates/local_vector.h"
#include "core/math/tri_mesh.h"

namespace newton {

class NewtonCollisionTree : public NewtonCollision {
	GDCLASS(NewtonCollisionTree, NewtonCollision);

public:
	NewtonCollisionTree() {}

	// Build the collision tree from an array of triangles (world space or local).
	void build_from_triangles(const LocalVector<vec3> &p_vertices,
							  const LocalVector<int> &p_indices) {
		vertices = p_vertices;
		indices = p_indices;
		// Build the Gaia BVH from triangle AABBs
		int tri_count = indices.size() / 3;
		LocalVector<AABB> tri_aabbs;
		for (int t = 0; t < tri_count; ++t) {
			vec3 v0 = vertices[indices[t*3]];
			vec3 v1 = vertices[indices[t*3+1]];
			vec3 v2 = vertices[indices[t*3+2]];
			AABB aabb(v0, Vector3());
			aabb.expand_to(v1);
			aabb.expand_to(v2);
			tri_aabbs.push_back(aabb);
		}
		bvh.build_final(tri_aabbs);
	}

	// Support function: find the vertex closest to the direction (convex‑like … but
	// a triangle mesh is not convex; we return the farthest vertex among all triangles
	// that intersect the direction's projection?  For GJK to work correctly, the
	// collision shape must be convex, so a triangle mesh should be decomposed into
	// convex pieces or use a convex hull approximation.  This tree is intended for
	// static environment collision (ray casts only) and not for GJK against dynamic
	// bodies.  We implement get_support by returning the farthest vertex of the
	// entire mesh (which is wrong for concave meshes).  Use with caution.
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		real_t best_dot = -INFINITY;
		vec3 best_vert(0,0,0);
		for (const vec3 &v : vertices) {
			real_t d = v.dot(p_dir);
			if (d > best_dot) { best_dot = d; best_vert = v; }
		}
		return p_transform.xform(best_vert);
	}

	// Inertia: approximated by the bounding box of all vertices.
	virtual mat3 compute_inertia(real_t p_mass) const override {
		aabb box = get_local_aabb();
		vec3 size = box.size;
		real_t Ix = (1.0/12.0) * p_mass * (size.y*size.y + size.z*size.z);
		real_t Iy = (1.0/12.0) * p_mass * (size.x*size.x + size.z*size.z);
		real_t Iz = (1.0/12.0) * p_mass * (size.x*size.x + size.y*size.y);
		return mat3().scaled(vec3(Ix, Iy, Iz));
	}

	virtual aabb get_local_aabb() const override {
		if (vertices.is_empty()) return aabb();
		aabb box(vertices[0], Vector3());
		for (int i=1; i<vertices.size(); ++i) box.expand_to(vertices[i]);
		return box;
	}

	virtual ShapeType get_shape_type() const override { return ShapeType::BVH_TRI_MESH; }

	// Ray‑cast against the tree (calls Gaia BVH per‑triangle intersection).
	bool ray_cast(const vec3 &p_origin, const vec3 &p_dir, real_t p_max_dist,
				  real_t &r_t, vec3 &r_normal) const {
		real_t best_t = p_max_dist;
		vec3 best_normal;
		bool hit = false;
		bvh.query_intersect(AABB(p_origin, (p_origin + p_dir * p_max_dist).abs()), [&](int prim) {
			int i0 = indices[prim*3];
			int i1 = indices[prim*3+1];
			int i2 = indices[prim*3+2];
			real_t t, u, v;
			if (gaia::bvh::intersect_ray_triangle(p_origin, p_dir, vertices[i0], vertices[i1], vertices[i2], t, u, v)) {
				if (t < best_t) {
					best_t = t;
					vec3 n = (vertices[i1]-vertices[i0]).cross(vertices[i2]-vertices[i0]).normalized();
					best_normal = n;
					hit = true;
				}
			}
		});
		if (hit) {
			r_t = best_t;
			r_normal = best_normal;
		}
		return hit;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("build_from_triangles", "vertices", "indices"), &NewtonCollisionTree::build_from_triangles);
		ClassDB::bind_method(D_METHOD("ray_cast", "origin", "dir", "max_dist"), &NewtonCollisionTree::ray_cast);
	}

private:
	LocalVector<vec3> vertices;
	LocalVector<int> indices;
	gaia::bvh::BVH bvh;
};

} // namespace newton

#endif // NEWTON_COLLISION_TREE_H