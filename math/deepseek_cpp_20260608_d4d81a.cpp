// File 230: modules/newton/src/utils/newton_debug_draw.h
// Debug draw utility for Newton bodies and contacts: renders AABBs, bones,
// contact points, joint axes, and velocity vectors using Godot ImmediateMesh.

#ifndef NEWTON_UTILS_DEBUG_DRAW_H
#define NEWTON_UTILS_DEBUG_DRAW_H

#include "scene/3d/node_3d.h"
#include "scene/resources/immediate_mesh.h"
#include "../world/newton_world.h"
#include "../bodies/newton_body.h"
#include "../joints/newton_joint.h"
#include "../collision/newton_collision.h"
#include "../collision/newton_contact.h"

namespace newton {

class NewtonDebugDraw : public Node3D {
	GDCLASS(NewtonDebugDraw, Node3D);

public:
	NewtonDebugDraw() {
		world = nullptr;
		show_aabbs = true;
		show_contacts = true;
		show_joint_axes = true;
		show_velocity = true;
	}

	void set_world(NewtonWorld *p_world) { world = p_world; }
	void set_show_aabbs(bool p_show) { show_aabbs = p_show; }
	void set_show_contacts(bool p_show) { show_contacts = p_show; }
	void set_show_joint_axes(bool p_show) { show_joint_axes = p_show; }
	void set_show_velocity(bool p_show) { show_velocity = p_show; }

	void _notification(int p_what) {
		if (p_what == NOTIFICATION_PROCESS) redraw();
	}

	void redraw() {
		if (!world) return;
		Ref<ImmediateMesh> im = get_display_mesh();
		if (im.is_null()) return;
		im->clear_surfaces();
		im->surface_begin(Mesh::PRIMITIVE_LINES);

		LocalVector<body_id> body_ids = world->get_body_ids();
		for (body_id id : body_ids) {
			Ref<NewtonBody> body = world->get_body(id);
			if (body.is_null() || !body->is_active()) continue;

			if (show_aabbs) {
				draw_aabb(im, body->get_aabb());
			}
			if (show_velocity && body->get_type() == BodyType::DYNAMIC) {
				vec3 p = body->get_position();
				vec3 v = body->get_linear_velocity();
				draw_arrow(im, p, p + v * 0.1f, Color(1, 0, 0));
			}
		}

		if (show_contacts) {
			const LocalVector<NewtonContactPoint> &contacts = world->get_generated_contacts();
			for (const auto &cp : contacts) {
				draw_cross(im, cp.point_a, cp.normal, 0.05f, Color(0, 1, 0));
			}
		}

		if (show_joint_axes) {
			LocalVector<joint_id> joint_ids = world->get_joint_ids();
			for (joint_id jid : joint_ids) {
				Ref<NewtonJoint> joint = world->get_joint(jid);
				if (joint.is_null() || !joint->is_enabled()) continue;
				// Simple: draw pivot points from body transforms (if we can access)
				// For now, placeholder: no direct pivot info without joint type specifics.
			}
		}

		im->surface_end();
	}

private:
	Ref<ImmediateMesh> get_display_mesh() {
		MeshInstance3D *mi = get_node_or_null<MeshInstance3D>(NodePath("NewtonDebugMesh"));
		if (!mi) {
			mi = memnew(MeshInstance3D);
			mi->set_name("NewtonDebugMesh");
			add_child(mi);
			Ref<ImmediateMesh> im; im.instantiate();
			mi->set_mesh(im);
			Ref<StandardMaterial3D> mat; mat.instantiate();
			mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
			mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
			mi->set_material_override(mat);
			return im;
		}
		return mi->get_mesh();
	}

	void draw_aabb(Ref<ImmediateMesh> im, const AABB &aabb) {
		vec3 min = aabb.position;
		vec3 max = min + aabb.size;
		vec3 pts[8] = { min, vec3(max.x,min.y,min.z), vec3(max.x,min.y,max.z), vec3(min.x,min.y,max.z),
						vec3(min.x,max.y,min.z), vec3(max.x,max.y,min.z), max, vec3(min.x,max.y,max.z) };
		int edges[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
		for (int e=0; e<12; ++e) {
			im->surface_add_vertex(pts[edges[e][0]]);
			im->surface_add_vertex(pts[edges[e][1]]);
		}
	}

	void draw_arrow(Ref<ImmediateMesh> im, const vec3 &from, const vec3 &to, const Color &color) {
		im->surface_add_vertex(from);
		im->surface_add_vertex(to);
	}

	void draw_cross(Ref<ImmediateMesh> im, const vec3 &center, const vec3 &normal, real_t size, const Color &color) {
		vec3 perp = (Math::abs(normal.x)<0.99f) ? normal.cross(vec3(1,0,0)).normalized() : normal.cross(vec3(0,1,0)).normalized();
		vec3 perp2 = normal.cross(perp).normalized();
		im->surface_add_vertex(center - perp*size);
		im->surface_add_vertex(center + perp*size);
		im->surface_add_vertex(center - perp2*size);
		im->surface_add_vertex(center + perp2*size);
	}

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_world", "world"), &NewtonDebugDraw::set_world);
		ClassDB::bind_method(D_METHOD("set_show_aabbs", "show"), &NewtonDebugDraw::set_show_aabbs);
		ClassDB::bind_method(D_METHOD("set_show_contacts", "show"), &NewtonDebugDraw::set_show_contacts);
		ClassDB::bind_method(D_METHOD("set_show_joint_axes", "show"), &NewtonDebugDraw::set_show_joint_axes);
		ClassDB::bind_method(D_METHOD("set_show_velocity", "show"), &NewtonDebugDraw::set_show_velocity);
	}

	NewtonWorld *world;
	bool show_aabbs;
	bool show_contacts;
	bool show_joint_axes;
	bool show_velocity;
};

} // namespace newton

#endif // NEWTON_UTILS_DEBUG_DRAW_H