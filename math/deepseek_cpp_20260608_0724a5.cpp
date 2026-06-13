// File 277: modules/vienna/src/collision/vienna_trimesh.h
// Triangle mesh collision shape – a static mesh built from a list of triangles.
// Uses Gaia BVH for fast ray-cast queries and GJK support via convex hull fallback.
// Inertia is approximated by the bounding box of the mesh.

#ifndef VIENNA_COLLISION_TRIMESH_H
#define VIENNA_COLLISION_TRIMESH_H

#include "vienna_shape.h"
#include "../../../gaia/src/bvh/bvh.h"
#include "../../../gaia/src/bvh/query.h"
#include "core/templates/local_vector.h"

namespace vienna {

class ViennaTriMesh : public ViennaShape {
	GDCLASS(ViennaTriMesh, ViennaShape);

public:
	ViennaTriMesh() {}

	// Build the collision mesh from a vertex array and a triangle index array.
	void build(const LocalVector<vec3> &p_vertices, const LocalVector<int> &p_indices) {
		vertices = p_vertices;
		indices = p_indices;
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
		// Compute local AABB from all vertices
		if (!vertices.is_empty()) {
			local_aabb = AABB(vertices[0], Vector3());
			for (int i = 1; i < vertices.size(); ++i) local_aabb.expand_to(vertices[i]);
		}
	}

	// Number of triangles
	int get_triangle_count() const { return indices.size() / 3; }

	// Ray-cast against the triangle mesh; returns true on hit and fills t, normal.
	bool ray_cast(const vec3 &p_origin, const vec3 &p_direction, real_t p_max_dist,
				  real_t &r_t, vec3 &r_normal) const {
		real_t best_t = p_max_dist;
		vec3 best_normal(0, 1, 0);
		bool found = false;
		vec3 dir = p_direction.normalized();
		bvh.query_intersect(AABB(p_origin, p_origin + dir * p_max_dist), [&](int prim) {
			int i0 = indices[prim*3];
			int i1 = indices[prim*3+1];
			int i2 = indices[prim*3+2];
			real_t t, u, v;
			if (gaia::bvh::intersect_ray_triangle(p_origin, dir, vertices[i0], vertices[i1], vertices[i2], t, u, v)) {
				if (t < best_t) {
					best_t = t;
					vec3 n = (vertices[i1]-vertices[i0]).cross(vertices[i2]-vertices[i0]).normalized();
					best_normal = n;
					found = true;
				}
			}
		});
		if (found) {
			r_t = best_t;
			r_normal = best_normal;
		}
		return found;
	}

	// Support point: return the vertex farthest along the direction (convex approximation).
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir);
		real_t best_dot = -INFINITY;
		vec3 best_vert(0, 0, 0);
		for (const vec3 &v : vertices) {
			real_t d = v.dot(local_dir);
			if (d > best_dot) { best_dot = d; best_vert = v; }
		}
		return p_transform.xform(best_vert);
	}

	// Inertia: approximated as a bounding box.
	virtual mat3 compute_inertia(real_t p_mass) const override {
		vec3 size = local_aabb.size;
		real_t Ix = (1.0/12.0) * p_mass * (size.y*size.y + size.z*size.z);
		real_t Iy = (1.0/12.0) * p_mass * (size.x*size.x + size.z*size.z);
		real_t Iz = (1.0/12.0) * p_mass * (size.x*size.x + size.y*size.y);
		return mat3().scaled(vec3(Ix, Iy, Iz));
	}

	virtual aabb get_local_aabb() const override { return local_aabb; }

	virtual ShapeType get_shape_type() const override { return ShapeType::TRI_MESH; }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("build", "vertices", "indices"), &ViennaTriMesh::build);
		ClassDB::bind_method(D_METHOD("ray_cast", "origin", "direction", "max_dist"), &ViennaTriMesh::ray_cast);
		ClassDB::bind_method(D_METHOD("get_triangle_count"), &ViennaTriMesh::get_triangle_count);
	}

private:
	LocalVector<vec3> vertices;
	LocalVector<int> indices;
	gaia::bvh::BVH bvh;
	aabb local_aabb;
};

} // namespace vienna

#endif // VIENNA_COLLISION_TRIMESH_H