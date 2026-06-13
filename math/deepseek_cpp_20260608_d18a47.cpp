// File 377: modules/integration/unified_physics_debug_draw.h
// UnifiedPhysicsDebugDraw – real‑time overlay for all physics engines
// (Gaia, Genesis, Newton, Vienna, Wicked) in a single Godot Node3D.
// Draws AABBs, collision shapes, velocity vectors, contact points, joints,
// cloth wires, and particles using Godot's ImmediateMesh for maximum
// rendering performance. Uses Gaia BVH queries to efficiently fetch only
// visible bodies. All per‑engine iteration is done inline to minimise
// virtual dispatch.

#ifndef INTEGRATION_UNIFIED_PHYSICS_DEBUG_DRAW_H
#define INTEGRATION_UNIFIED_PHYSICS_DEBUG_DRAW_H

#include "scene/3d/node_3d.h"
#include "scene/resources/immediate_mesh.h"
#include "scene/resources/material.h"

// Gaia (for BVH query)
#include "../../gaia/src/bvh/bvh.h"
#include "../../gaia/src/bvh/aabb.h"

// Newton
#include "../../newton/src/world/newton_world.h"
#include "../../newton/src/bodies/newton_body.h"
#include "../../newton/src/collision/newton_collision.h"

// Genesis
#include "../../genesis/src/genesis_world.h"
#include "../../genesis/src/entities/rigid_entity.h"
#include "../../genesis/src/entities/fem_entity.h"
#include "../../genesis/src/entities/mpm_entity.h"
#include "../../genesis/src/entities/particle_entity.h"
#include "../../genesis/src/solvers/sph_solver.h"

// Vienna
#include "../../vienna/src/world/vienna_world.h"
#include "../../vienna/src/bodies/vienna_body.h"
#include "../../vienna/src/cloth/vienna_cloth.h"
#include "../../vienna/src/particles/vienna_particle_system.h"

// Wicked
#include "../../wicked/src/world/wicked_world.h"
#include "../../wicked/src/bodies/wicked_body.h"
#include "../../wicked/src/collision/wicked_shape.h"

#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/math/color.h"

namespace unified {

class UnifiedPhysicsDebugDraw : public Node3D {
	GDCLASS(UnifiedPhysicsDebugDraw, Node3D);

	// Pointers to all worlds (set from plugin / user)
	newton::NewtonWorld      *newton_world = nullptr;
	genesis::GenesisWorld    *genesis_world = nullptr;
	vienna::ViennaWorld      *vienna_world = nullptr;
	wicked::WickedWorld      *wicked_world = nullptr;

	// Toggle flags
	bool show_bodies_aabb       = true;
	bool show_collision_shapes  = true;
	bool show_velocity          = true;
	bool show_contacts          = true;
	bool show_joints            = true;
	bool show_cloth             = true;
	bool show_particles         = true;
	bool show_grid              = true;

	// ImmediateMesh for fast line / point drawing
	Ref<ImmediateMesh> debug_mesh;
	MeshInstance3D *mesh_instance = nullptr;

	// Cache of visible AABBs for BVH frustum culling
	gaia::bvh::BVH visibility_bvh;
	LocalVector<AABB> visible_aabbs;

public:
	UnifiedPhysicsDebugDraw();
	~UnifiedPhysicsDebugDraw() {}

	// Setters
	void set_newton_world(newton::NewtonWorld *p)   { newton_world = p; }
	void set_genesis_world(genesis::GenesisWorld *p) { genesis_world = p; }
	void set_vienna_world(vienna::ViennaWorld *p)    { vienna_world = p; }
	void set_wicked_world(wicked::WickedWorld *p)    { wicked_world = p; }

	// Toggle flags
	void set_show_bodies_aabb(bool p)      { show_bodies_aabb = p; }
	void set_show_collision_shapes(bool p) { show_collision_shapes = p; }
	void set_show_velocity(bool p)         { show_velocity = p; }
	void set_show_contacts(bool p)         { show_contacts = p; }
	void set_show_joints(bool p)           { show_joints = p; }
	void set_show_cloth(bool p)            { show_cloth = p; }
	void set_show_particles(bool p)        { show_particles = p; }

	void _notification(int p_what);

protected:
	static void _bind_methods();

private:
	void _create_display_mesh();
	void _redraw();

	// Drawing helpers (inline for speed)
	inline void draw_aabb(ImmediateMesh *im, const AABB &box, const Color &col);
	inline void draw_sphere_wire(ImmediateMesh *im, const Vector3 &center, real_t radius,
	                             const Color &col, int segs = 16);
	inline void draw_arrow(ImmediateMesh *im, const Vector3 &from, const Vector3 &to, const Color &col);
	inline void draw_cross(ImmediateMesh *im, const Vector3 &center, real_t size, const Color &col);
	inline void draw_grid(ImmediateMesh *im, real_t world_size = 20.0, int steps = 40);

	// Engine‑specific drawing loops
	void draw_newton(ImmediateMesh *im);
	void draw_genesis(ImmediateMesh *im);
	void draw_vienna(ImmediateMesh *im);
	void draw_wicked(ImmediateMesh *im);
};

// =========================================================================
// Inline implementations of drawing helpers
// =========================================================================

void UnifiedPhysicsDebugDraw::draw_aabb(ImmediateMesh *im, const AABB &box, const Color &col) {
	Vector3 min = box.position;
	Vector3 max = min + box.size;
	Vector3 pts[8] = {
		min, Vector3(max.x, min.y, min.z), Vector3(max.x, min.y, max.z), Vector3(min.x, min.y, max.z),
		Vector3(min.x, max.y, min.z), Vector3(max.x, max.y, min.z), max, Vector3(min.x, max.y, max.z)
	};
	int edges[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
	for (int e = 0; e < 12; ++e) {
		im->surface_add_vertex(pts[edges[e][0]]);
		im->surface_add_vertex(pts[edges[e][1]]);
	}
}

void UnifiedPhysicsDebugDraw::draw_sphere_wire(ImmediateMesh *im, const Vector3 &center,
                                                real_t radius, const Color &col, int segs) {
	for (int ax = 0; ax < 3; ++ax) {
		Vector3 u, v;
		if (ax == 0) { u = Vector3(0,1,0); v = Vector3(0,0,1); }
		else if (ax == 1) { u = Vector3(1,0,0); v = Vector3(0,0,1); }
		else { u = Vector3(1,0,0); v = Vector3(0,1,0); }
		for (int i = 0; i < segs; ++i) {
			real_t a0 = Math_TAU * i / segs;
			real_t a1 = Math_TAU * (i + 1) / segs;
			im->surface_add_vertex(center + (u * Math::cos(a0) + v * Math::sin(a0)) * radius);
			im->surface_add_vertex(center + (u * Math::cos(a1) + v * Math::sin(a1)) * radius);
		}
	}
}

void UnifiedPhysicsDebugDraw::draw_arrow(ImmediateMesh *im, const Vector3 &from,
                                          const Vector3 &to, const Color &col) {
	im->surface_add_vertex(from);
	im->surface_add_vertex(to);
	// Simple arrowhead
	Vector3 dir = (to - from).normalized();
	Vector3 perp = (Math::abs(dir.x) < 0.99f) ? dir.cross(Vector3(1,0,0)).normalized()
	                                           : dir.cross(Vector3(0,1,0)).normalized();
	Vector3 perp2 = dir.cross(perp).normalized();
	real_t head_len = MIN((to - from).length() * 0.2f, 0.1f);
	Vector3 head_base = to - dir * head_len;
	im->surface_add_vertex(head_base + perp * head_len * 0.5f);
	im->surface_add_vertex(to);
	im->surface_add_vertex(head_base - perp * head_len * 0.5f);
	im->surface_add_vertex(to);
	im->surface_add_vertex(head_base + perp2 * head_len * 0.5f);
	im->surface_add_vertex(to);
	im->surface_add_vertex(head_base - perp2 * head_len * 0.5f);
	im->surface_add_vertex(to);
}

void UnifiedPhysicsDebugDraw::draw_cross(ImmediateMesh *im, const Vector3 &center,
                                          real_t size, const Color &col) {
	im->surface_add_vertex(center - Vector3(size,0,0));
	im->surface_add_vertex(center + Vector3(size,0,0));
	im->surface_add_vertex(center - Vector3(0,size,0));
	im->surface_add_vertex(center + Vector3(0,size,0));
	im->surface_add_vertex(center - Vector3(0,0,size));
	im->surface_add_vertex(center + Vector3(0,0,size));
}

void UnifiedPhysicsDebugDraw::draw_grid(ImmediateMesh *im, real_t world_size, int steps) {
	real_t half = world_size * 0.5f;
	real_t step = world_size / steps;
	for (int i = 0; i <= steps; ++i) {
		real_t p = -half + i * step;
		im->surface_add_vertex(Vector3(p, 0, -half));
		im->surface_add_vertex(Vector3(p, 0, half));
		im->surface_add_vertex(Vector3(-half, 0, p));
		im->surface_add_vertex(Vector3(half, 0, p));
	}
}

} // namespace unified

#endif // INTEGRATION_UNIFIED_PHYSICS_DEBUG_DRAW_H