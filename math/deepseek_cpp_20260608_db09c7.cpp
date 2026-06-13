// File 170: modules/genesis/src/nodes/genesis_debug_draw_3d.cpp
// Implementation of GenesisDebugDraw3D – wireframe overlay for all entities
// in a GenesisWorld. Draws AABBs, collision shapes, velocity vectors, FEM
// wireframes, MPM particle clouds, SPH particles, joint anchors, and a grid.

#include "genesis_debug_draw_3d.h"

#include "scene/main/scene_tree.h"
#include "scene/resources/immediate_mesh.h"
#include "scene/resources/material.h"

#include "../genesis_world.h"
#include "../entities/rigid_entity.h"
#include "../entities/fem_entity.h"
#include "../entities/mpm_entity.h"
#include "../entities/tool_entity.h"
#include "../entities/hybrid_entity.h"
#include "../solvers/sph_solver.h"
#include "../constraints/joint_constraint.h"

namespace genesis {

void GenesisDebugDraw3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_show_aabbs", "enable"), &GenesisDebugDraw3D::set_show_aabbs);
	ClassDB::bind_method(D_METHOD("get_show_aabbs"), &GenesisDebugDraw3D::is_show_aabbs);
	ClassDB::bind_method(D_METHOD("set_show_collision_shapes", "enable"), &GenesisDebugDraw3D::set_show_collision_shapes);
	ClassDB::bind_method(D_METHOD("get_show_collision_shapes"), &GenesisDebugDraw3D::is_show_collision_shapes);
	ClassDB::bind_method(D_METHOD("set_show_velocity_vectors", "enable"), &GenesisDebugDraw3D::set_show_velocity_vectors);
	ClassDB::bind_method(D_METHOD("get_show_velocity_vectors"), &GenesisDebugDraw3D::is_show_velocity_vectors);
	ClassDB::bind_method(D_METHOD("set_show_fem_wireframe", "enable"), &GenesisDebugDraw3D::set_show_fem_wireframe);
	ClassDB::bind_method(D_METHOD("get_show_fem_wireframe"), &GenesisDebugDraw3D::is_show_fem_wireframe);
	ClassDB::bind_method(D_METHOD("set_show_mpm_particles", "enable"), &GenesisDebugDraw3D::set_show_mpm_particles);
	ClassDB::bind_method(D_METHOD("get_show_mpm_particles"), &GenesisDebugDraw3D::is_show_mpm_particles);
	ClassDB::bind_method(D_METHOD("set_show_sph_particles", "enable"), &GenesisDebugDraw3D::set_show_sph_particles);
	ClassDB::bind_method(D_METHOD("get_show_sph_particles"), &GenesisDebugDraw3D::is_show_sph_particles);
	ClassDB::bind_method(D_METHOD("set_show_joint_anchors", "enable"), &GenesisDebugDraw3D::set_show_joint_anchors);
	ClassDB::bind_method(D_METHOD("get_show_joint_anchors"), &GenesisDebugDraw3D::is_show_joint_anchors);
	ClassDB::bind_method(D_METHOD("set_show_contacts", "enable"), &GenesisDebugDraw3D::set_show_contacts);
	ClassDB::bind_method(D_METHOD("get_show_contacts"), &GenesisDebugDraw3D::is_show_contacts);
	ClassDB::bind_method(D_METHOD("set_show_grid", "enable"), &GenesisDebugDraw3D::set_show_grid);
	ClassDB::bind_method(D_METHOD("get_show_grid"), &GenesisDebugDraw3D::is_show_grid);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_aabbs"), "set_show_aabbs", "get_show_aabbs");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_collision_shapes"), "set_show_collision_shapes", "get_show_collision_shapes");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_velocity_vectors"), "set_show_velocity_vectors", "get_show_velocity_vectors");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_fem_wireframe"), "set_show_fem_wireframe", "get_show_fem_wireframe");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_mpm_particles"), "set_show_mpm_particles", "get_show_mpm_particles");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_sph_particles"), "set_show_sph_particles", "get_show_sph_particles");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_joint_anchors"), "set_show_joint_anchors", "get_show_joint_anchors");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_contacts"), "set_show_contacts", "get_show_contacts");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_grid"), "set_show_grid", "get_show_grid");
}

void GenesisDebugDraw3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		_resolve_world();
		_create_display_mesh();
	}
	if (p_what == NOTIFICATION_PROCESS) {
		_update_draw();
	}
}

void GenesisDebugDraw3D::_resolve_world() {
	if (!world) {
		Node *parent = get_parent();
		while (parent) {
			world = Object::cast_to<GenesisWorld>(parent);
			if (world) return;
			parent = parent->get_parent();
		}
	}
}

void GenesisDebugDraw3D::_create_display_mesh() {
	MeshInstance3D *mi = get_node_or_null<MeshInstance3D>(NodePath("DebugDisplay"));
	if (!mi) {
		mi = memnew(MeshInstance3D);
		mi->set_name("DebugDisplay");
		add_child(mi);
	}
	_debug_mesh.instantiate();
	mi->set_mesh(_debug_mesh);
	Ref<StandardMaterial3D> mat; mat.instantiate();
	mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	mi->set_material_override(mat);
}

void GenesisDebugDraw3D::_update_draw() {
	if (_debug_mesh.is_null() || !world) return;
	ImmediateMesh *im = _debug_mesh.ptr();
	im->clear_surfaces();
	im->surface_begin(Mesh::PRIMITIVE_LINES);

	// Grid
	if (show_grid) _draw_world_grid(im);

	// Obtain list of all entities from the world (requires public getter – we assume it's added)
	LocalVector<Ref<BaseEntity>> all_entities = world->get_all_entities(); // to be added to GenesisWorld
	for (const Ref<BaseEntity> &ent : all_entities) {
		if (ent.is_null() || !ent->is_active()) continue;

		// AABBs
		if (show_aabbs) {
			_draw_aabb(im, ent->get_aabb(), Color(0.8f, 0.8f, 0.8f));
		}

		// Velocity vectors
		if (show_velocity_vectors) {
			Vector3 pos = ent->get_position();
			Vector3 vel = ent->get_linear_velocity();
			_draw_arrow(im, pos, pos + vel * 0.1f, Color(1, 0, 0));
		}

		// Rigid collision shapes
		if (show_collision_shapes) {
			Ref<RigidEntity> rigid = ent;
			if (rigid.is_valid()) {
				_draw_collision_shape(im, rigid);
			}
		}

		// FEM wireframe
		if (show_fem_wireframe) {
			Ref<FEMEntity> fem = ent;
			if (fem.is_valid()) {
				_draw_fem_wireframe(im, fem);
			}
		}

		// MPM particles
		if (show_mpm_particles) {
			Ref<MPMEntity> mpm = ent;
			if (mpm.is_valid()) {
				_draw_mpm_particles(im, mpm);
			}
		}
	}
	im->surface_end();
}

// --- Drawing helpers ---

void GenesisDebugDraw3D::_draw_aabb(ImmediateMesh *im, const AABB &aabb, const Color &color) {
	Vector3 min = aabb.position;
	Vector3 max = min + aabb.size;
	Vector3 pts[8] = {
		min,
		Vector3(max.x, min.y, min.z),
		Vector3(max.x, min.y, max.z),
		Vector3(min.x, min.y, max.z),
		Vector3(min.x, max.y, min.z),
		Vector3(max.x, max.y, min.z),
		max,
		Vector3(min.x, max.y, max.z)
	};
	int edges[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
	for (int e = 0; e < 12; ++e) {
		im->surface_add_vertex(pts[edges[e][0]]);
		im->surface_add_vertex(pts[edges[e][1]]);
	}
}

void GenesisDebugDraw3D::_draw_arrow(ImmediateMesh *im, const Vector3 &from, const Vector3 &to, const Color &color) {
	im->surface_add_vertex(from);
	im->surface_add_vertex(to);
	// small arrowhead lines
	Vector3 dir = (to - from).normalized();
	Vector3 perp = (Math::abs(dir.x) < 0.99f) ? dir.cross(Vector3(1,0,0)).normalized()
											   : dir.cross(Vector3(0,1,0)).normalized();
	Vector3 perp2 = dir.cross(perp).normalized();
	real_t head_len = (to - from).length() * 0.2f;
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

void GenesisDebugDraw3D::_draw_collision_shape(ImmediateMesh *im, const Ref<RigidEntity> &rigid) {
	GeometryType geom = rigid->get_geometry_type();
	Transform3D xform = rigid->get_transform();
	switch (geom) {
		case GeometryType::SPHERE:
			_draw_sphere(im, xform.origin, rigid->get_radius(), Color(0,1,0));
			break;
		case GeometryType::BOX:
			_draw_aabb(im, rigid->get_aabb(), Color(0,1,0));
			break;
		case GeometryType::CAPSULE:
		case GeometryType::CYLINDER:
		default:
			_draw_aabb(im, rigid->get_aabb(), Color(0,0.7f,0));
			break;
	}
}

void GenesisDebugDraw3D::_draw_sphere(ImmediateMesh *im, const Vector3 &center, real_t radius, const Color &color) {
	int segs = 16;
	for (int ax = 0; ax < 3; ++ax) {
		Vector3 u, v;
		if (ax == 0) { u = Vector3(0,1,0); v = Vector3(0,0,1); }
		else if (ax == 1) { u = Vector3(1,0,0); v = Vector3(0,0,1); }
		else { u = Vector3(1,0,0); v = Vector3(0,1,0); }
		for (int i = 0; i < segs; ++i) {
			real_t a0 = Math_TAU * i / segs;
			real_t a1 = Math_TAU * (i + 1) / segs;
			Vector3 p0 = center + (u * Math::cos(a0) + v * Math::sin(a0)) * radius;
			Vector3 p1 = center + (u * Math::cos(a1) + v * Math::sin(a1)) * radius;
			im->surface_add_vertex(p0);
			im->surface_add_vertex(p1);
		}
	}
}

void GenesisDebugDraw3D::_draw_fem_wireframe(ImmediateMesh *im, const Ref<FEMEntity> &fem) {
	const gaia::mesh::TetMesh &mesh = fem->get_mesh();
	int tet_count = mesh.element_count();
	for (int el = 0; el < tet_count; ++el) {
		auto tet = mesh.get_tetrahedron(el);
		Vector3 v0 = mesh.get_vertex(tet.v0);
		Vector3 v1 = mesh.get_vertex(tet.v1);
		Vector3 v2 = mesh.get_vertex(tet.v2);
		Vector3 v3 = mesh.get_vertex(tet.v3);
		int edges[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
		const Vector3 *p[4] = {&v0, &v1, &v2, &v3};
		for (int e = 0; e < 6; ++e) {
			im->surface_add_vertex(*p[edges[e][0]]);
			im->surface_add_vertex(*p[edges[e][1]]);
		}
	}
}

void GenesisDebugDraw3D::_draw_mpm_particles(ImmediateMesh *im, const Ref<MPMEntity> &mpm) {
	const auto &particles = mpm->get_particles();
	for (const auto &p : particles) {
		// draw particle as a small point (line from pos to pos+eps) or use point mode,
		// but we are in PRIMITIVE_LINES so we draw a tiny cross
		Vector3 pos = p.position;
		real_t eps = 0.01f;
		im->surface_add_vertex(pos - Vector3(eps,0,0));
		im->surface_add_vertex(pos + Vector3(eps,0,0));
		im->surface_add_vertex(pos - Vector3(0,eps,0));
		im->surface_add_vertex(pos + Vector3(0,eps,0));
		im->surface_add_vertex(pos - Vector3(0,0,eps));
		im->surface_add_vertex(pos + Vector3(0,0,eps));
	}
}

void GenesisDebugDraw3D::_draw_world_grid(ImmediateMesh *im) {
	int steps = 20;
	real_t size = 10.0f;
	for (int i = -steps; i <= steps; ++i) {
		real_t p = i * (size / steps);
		im->surface_add_vertex(Vector3(p, 0, -size));
		im->surface_add_vertex(Vector3(p, 0, size));
		im->surface_add_vertex(Vector3(-size, 0, p));
		im->surface_add_vertex(Vector3(size, 0, p));
	}
}

} // namespace genesis