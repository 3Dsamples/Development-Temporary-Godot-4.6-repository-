// File 211: modules/newton/src/collision/newton_convex_hull.h
// Convex hull collision shape: stores a list of vertices in local space.
// Provides support point, inertia via tensor voting, and local AABB.
// Suitable for arbitrary convex meshes used by Newton Dynamics.

#ifndef NEWTON_COLLISION_CONVEX_HULL_H
#define NEWTON_COLLISION_CONVEX_HULL_H

#include "newton_collision.h"
#include "core/templates/local_vector.h"

namespace newton {

class NewtonCollisionConvexHull : public NewtonCollision {
	GDCLASS(NewtonCollisionConvexHull, NewtonCollision);

public:
	NewtonCollisionConvexHull() {}

	// Add a vertex to the hull.
	void add_vertex(const vec3 &p_vertex) {
		vertices.push_back(p_vertex);
		if (vertices.size() == 1) {
			local_aabb = aabb(p_vertex, vec3());
		} else {
			local_aabb.expand_to(p_vertex);
		}
	}

	// Remove all vertices.
	void clear_vertices() {
		vertices.clear();
		local_aabb = aabb();
	}

	int get_vertex_count() const { return vertices.size(); }
	const vec3 &get_vertex(int p_idx) const { return vertices[p_idx]; }

	virtual ShapeType get_shape_type() const override { return ShapeType::CONVEX_HULL; }

	// Support point: return the vertex farthest along the direction.
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir);
		real_t best_dot = -INFINITY;
		int best_idx = 0;
		for (int i = 0; i < vertices.size(); ++i) {
			real_t d = vertices[i].dot(local_dir);
			if (d > best_dot) {
				best_dot = d;
				best_idx = i;
			}
		}
		return p_transform.xform(vertices[best_idx]);
	}

	// Inertia tensor: approximate using a cloud of point masses at vertices.
	virtual mat3 compute_inertia(real_t p_mass) const override {
		if (vertices.is_empty()) return mat3().scaled(vec3(1, 1, 1));
		real_t inv_count = 1.0f / (real_t)vertices.size();
		real_t point_mass = p_mass * inv_count;
		mat3 inertia;
		inertia.scale(vec3(0, 0, 0));
		for (const vec3 &v : vertices) {
			real_t x = v.x, y = v.y, z = v.z;
			vec3 diag(y * y + z * z, x * x + z * z, x * x + y * y);
			mat3 contrib;
			contrib.set(diag.x, 0, 0, 0, diag.y, 0, 0, 0, diag.z);
			// off-diagonal terms: -m*x*y etc.
			contrib[0][1] = -point_mass * x * y;
			contrib[1][0] = -point_mass * x * y;
			contrib[0][2] = -point_mass * x * z;
			contrib[2][0] = -point_mass * x * z;
			contrib[1][2] = -point_mass * y * z;
			contrib[2][1] = -point_mass * y * z;
			inertia += contrib;
		}
		return inertia;
	}

	// Local AABB (updated when vertices are added).
	virtual aabb get_local_aabb() const override {
		return local_aabb;
	}

	// For compound support, we can provide direct vertex access.
	const LocalVector<vec3> &get_vertices() const { return vertices; }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("add_vertex", "vertex"), &NewtonCollisionConvexHull::add_vertex);
		ClassDB::bind_method(D_METHOD("clear_vertices"), &NewtonCollisionConvexHull::clear_vertices);
		ClassDB::bind_method(D_METHOD("get_vertex_count"), &NewtonCollisionConvexHull::get_vertex_count);
		ClassDB::bind_method(D_METHOD("get_vertex", "idx"), &NewtonCollisionConvexHull::get_vertex);
	}

private:
	LocalVector<vec3> vertices;
	aabb local_aabb;
};

} // namespace newton

#endif // NEWTON_COLLISION_CONVEX_HULL_H