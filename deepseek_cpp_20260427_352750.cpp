// File 299: modules/vienna/src/utils/vienna_debug_draw.cpp
// Implementation of ViennaDebugDraw – renders AABBs, collision shapes, velocity
// vectors, contact points, joint pivots/axes, cloth wires, and particles using
// Godot's ImmediateMesh with per‑surface colours.

#include "vienna_debug_draw.h"

#include "scene/main/scene_tree.h"
#include "scene/resources/immediate_mesh.h"
#include "scene/resources/material.h"
#include "../world/vienna_world.h"
#include "../bodies/vienna_body.h"
#include "../collision/vienna_shape.h"
#include "../joints/vienna_joint.h"
#include "../joints/vienna_ball_joint.h"
#include "../joints/vienna_hinge_joint.h"
#include "../joints/vienna_slider_joint.h"
#include "../joints/vienna_fixed_joint.h"
#include "../joints/vienna_distance_joint.h"
#include "../joints/vienna_rope_joint.h"
#include "../cloth/vienna_cloth.h"
#include "../particles/vienna_particle_system.h"
#include "../solver/vienna_solver.h"

namespace vienna {

void ViennaDebugDraw::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_world", "world"), &ViennaDebugDraw::set_world);
	ClassDB::bind_method(D_METHOD("get_world"), &ViennaDebugDraw::get_world);
	ClassDB::bind_method(D_METHOD("set_show_bodies", "show"), &ViennaDebugDraw::set_show_bodies);
	ClassDB::bind_method(D_METHOD("is_show_bodies"), &ViennaDebugDraw::get_show_bodies);
	ClassDB::bind_method(D_METHOD("set_show_joints", "show"), &ViennaDebugDraw::set_show_joints);
	ClassDB::bind_method(D_METHOD("is_show_joints"), &ViennaDebugDraw::get_show_joints);
	ClassDB::bind_method(D_METHOD("set_show_contacts", "show"), &ViennaDebugDraw::set_show_contacts);
	ClassDB::bind_method(D_METHOD("is_show_contacts"), &ViennaDebugDraw::get_show_contacts);
	ClassDB::bind_method(D_METHOD("set_show_cloth", "show"), &ViennaDebugDraw::set_show_cloth);
	ClassDB::bind_method(D_METHOD("is_show_cloth"), &ViennaDebugDraw::get_show_cloth);
	ClassDB::bind_method(D_METHOD("set_show_particles", "show"), &ViennaDebugDraw::set_show_particles);
	ClassDB::bind_method(D_METHOD("is_show_particles"), &ViennaDebugDraw::get_show_particles);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "world", PROPERTY_HINT_RESOURCE_TYPE, "ViennaWorld"), "set_world", "get_world");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_bodies"), "set_show_bodies", "is_show_bodies");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_joints"), "set_show_joints", "is_show_joints");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_contacts"), "set_show_contacts", "is_show_contacts");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_cloth"), "set_show_cloth", "is_show_cloth");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_particles"), "set_show_particles", "is_show_particles");
}

ViennaDebugDraw::ViennaDebugDraw() :
	world(nullptr), mesh_instance(nullptr),
	show_bodies(true), show_joints(true), show_contacts(true),
	show_cloth(true), show_particles(true) {
	set_process(true);
}

void ViennaDebugDraw::set_world(ViennaWorld *p_world) { world = p_world; }
void ViennaDebugDraw::set_show_bodies(bool p_show) { show_bodies = p_show; }
void ViennaDebugDraw::set_show_joints(bool p_show) { show_joints = p_show; }
void ViennaDebugDraw::set_show_contacts(bool p_show) { show_contacts = p_show; }
void ViennaDebugDraw::set_show_cloth(bool p_show) { show_cloth = p_show; }
void ViennaDebugDraw::set_show_particles(bool p_show) { show_particles = p_show; }

void ViennaDebugDraw::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		_create_display_mesh();
	}
	if (p_what == NOTIFICATION_PROCESS) {
		_redraw();
	}
}

void ViennaDebugDraw::_create_display_mesh() {
	if (!get_node_or_null<Node3D>("ViennaDebugMesh")) {
		mesh_instance = memnew(MeshInstance3D);
		mesh_instance->set_name("ViennaDebugMesh");
		add_child(mesh_instance);
	}
	debug_mesh.instantiate();
	mesh_instance->set_mesh(debug_mesh);
	Ref<StandardMaterial3D> mat;
	mat.instantiate();
	mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, false); // we set per-surface colours
	mesh_instance->set_material_override(mat);
}

void ViennaDebugDraw::_redraw() {
	if (debug_mesh.is_null() || !world) return;
	ImmediateMesh *im = debug_mesh.ptr();
	im->clear_surfaces();

	// ---------- Bodies ----------
	if (show_bodies) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(0.0, 1.0, 0.0)); // green
		LocalVector<body_id> body_ids = world->get_body_ids();
		for (body_id id : body_ids) {
			Ref<ViennaBody> body = world->get_body(id);
			if (body.is_null() || !body->is_active()) continue;

			// AABB
			aabb box = body->get_aabb();
			draw_aabb(im, box, Color(0.0, 1.0, 0.0));

			// Velocity vector
			if (body->get_type() == BodyType::DYNAMIC) {
				vec3 pos = body->get_position();
				vec3 vel = body->get_linear_velocity();
				draw_arrow(im, pos, pos + vel * 0.25f, Color(1.0, 0.0, 0.0)); // red
			}
		}
		im->surface_end();
	}

	// ---------- Joints ----------
	if (show_joints) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(0.0, 0.0, 1.0)); // blue
		LocalVector<joint_id> joint_ids = world->get_joint_ids();
		for (joint_id jid : joint_ids) {
			Ref<ViennaJoint> joint = world->get_joint(jid);
			if (joint.is_null() || !joint->is_enabled()) continue;

			// We need the body transforms to convert local pivot to world.
			ViennaBody *bodyA = world->get_body(joint->get_body_a()).ptr();
			ViennaBody *bodyB = world->get_body(joint->get_body_b()).ptr();
			if (!bodyA && !bodyB) continue;

			// Get pivot from joint type if possible, otherwise draw nothing.
			vec3 worldPivot;
			if (joint->get_joint_type() == JointType::BALL) {
				Ref<ViennaBallJoint> ball = joint;
				vec3 local_pivot = ball->get_pivot();
				if (bodyA) worldPivot = bodyA->get_transform().xform(local_pivot);
				else if (bodyB) worldPivot = bodyB->get_transform().xform(local_pivot);
				draw_joint_pivot(im, worldPivot, Color(0.0, 0.5, 1.0));
			}
			else if (joint->get_joint_type() == JointType::HINGE) {
				Ref<ViennaHingeJoint> hinge = joint;
				if (bodyA) worldPivot = bodyA->get_transform().xform(hinge->get_pivot());
				else if (bodyB) worldPivot = bodyB->get_transform().xform(hinge->get_pivot());
				draw_joint_pivot(im, worldPivot, Color(0.0, 0.5, 1.0));
			}
			else if (joint->get_joint_type() == JointType::SLIDER) {
				Ref<ViennaSliderJoint> slider = joint;
				if (bodyA) worldPivot = bodyA->get_transform().xform(slider->get_pivot());
				else if (bodyB) worldPivot = bodyB->get_transform().xform(slider->get_pivot());
				draw_joint_pivot(im, worldPivot, Color(0.0, 0.5, 1.0));
			}
			else if (joint->get_joint_type() == JointType::FIXED) {
				if (bodyA) worldPivot = bodyA->get_position();
				else if (bodyB) worldPivot = bodyB->get_position();
				draw_joint_pivot(im, worldPivot, Color(0.0, 0.5, 1.0));
			}
			else if (joint->get_joint_type() == JointType::DISTANCE || joint->get_joint_type() == JointType::ROPE) {
				// For distance/rope, draw line between the two anchors
				vec3 anchorA, anchorB;
				if (bodyA && joint->get_joint_type() == JointType::DISTANCE) {
					Ref<ViennaDistanceJoint> dj = joint;
					anchorA = bodyA->get_transform().xform(dj->get_anchor_a());
					if (bodyB) anchorB = bodyB->get_transform().xform(dj->get_anchor_b());
				}
				else if (bodyA && joint->get_joint_type() == JointType::ROPE) {
					Ref<ViennaRopeJoint> rj = joint;
					anchorA = bodyA->get_transform().xform(rj->get_anchor_a());
					if (bodyB) anchorB = bodyB->get_transform().xform(rj->get_anchor_b());
				}
				if (bodyA && bodyB) {
					im->surface_add_vertex(anchorA);
					im->surface_add_vertex(anchorB);
				}
			}
		}
		im->surface_end();
	}

	// ---------- Cloth ----------
	if (show_cloth) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(1.0, 1.0, 1.0)); // white
		LocalVector<cloth_id> cloth_ids = world->get_cloth_ids();
		for (cloth_id cid : cloth_ids) {
			Ref<ViennaCloth> cloth = world->get_cloth(cid);
			if (cloth.is_null()) continue;
			int rx = cloth->get_resolution_x();
			int ry = cloth->get_resolution_y();
			int vc = cloth->get_vertex_count();
			// Draw edges: horizontal
			for (int y = 0; y < ry; ++y) {
				for (int x = 0; x < rx - 1; ++x) {
					int i0 = y * rx + x;
					int i1 = i0 + 1;
					im->surface_add_vertex(cloth->get_vertex(i0).position);
					im->surface_add_vertex(cloth->get_vertex(i1).position);
				}
			}
			// vertical
			for (int y = 0; y < ry - 1; ++y) {
				for (int x = 0; x < rx; ++x) {
					int i0 = y * rx + x;
					int i1 = i0 + rx;
					im->surface_add_vertex(cloth->get_vertex(i0).position);
					im->surface_add_vertex(cloth->get_vertex(i1).position);
				}
			}
		}
		im->surface_end();
	}

	// ---------- Particles ----------
	if (show_particles) {
		im->surface_begin(Mesh::PRIMITIVE_POINTS);
		im->surface_set_color(Color(1.0, 0.8, 0.0)); // orange
		LocalVector<cloth_id> cloth_ids = world->get_cloth_ids(); // particle systems use cloth_id
		for (cloth_id cid : cloth_ids) {
			Ref<ViennaParticleSystem> ps = world->get_particle_system(cid);
			if (ps.is_null()) continue;
			for (int i = 0; i < ps->get_live_count(); ++i) {
				im->surface_add_vertex(ps->get_particle(i).position);
			}
		}
		im->surface_end();
	}
}

// ---------- Drawing helpers ----------
void ViennaDebugDraw::draw_aabb(ImmediateMesh *im, const aabb &box, const Color &color) {
	vec3 min = box.position;
	vec3 max = min + box.size;
	vec3 pts[8] = {
		min,
		vec3(max.x, min.y, min.z),
		vec3(max.x, min.y, max.z),
		vec3(min.x, min.y, max.z),
		vec3(min.x, max.y, min.z),
		vec3(max.x, max.y, min.z),
		max,
		vec3(min.x, max.y, max.z)
	};
	int edges[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
	for (int e = 0; e < 12; ++e) {
		im->surface_add_vertex(pts[edges[e][0]]);
		im->surface_add_vertex(pts[edges[e][1]]);
	}
}

void ViennaDebugDraw::draw_sphere(ImmediateMesh *im, const vec3 &center, real_t radius, const Color &color) {
	const int segs = 16;
	for (int ax = 0; ax < 3; ++ax) {
		vec3 u, v;
		if (ax == 0) { u = vec3(0,1,0); v = vec3(0,0,1); }
		else if (ax == 1) { u = vec3(1,0,0); v = vec3(0,0,1); }
		else { u = vec3(1,0,0); v = vec3(0,1,0); }
		for (int i = 0; i < segs; ++i) {
			real_t a0 = Math_TAU * i / segs;
			real_t a1 = Math_TAU * (i + 1) / segs;
			vec3 p0 = center + (u * Math::cos(a0) + v * Math::sin(a0)) * radius;
			vec3 p1 = center + (u * Math::cos(a1) + v * Math::sin(a1)) * radius;
			im->surface_add_vertex(p0);
			im->surface_add_vertex(p1);
		}
	}
}

void ViennaDebugDraw::draw_arrow(ImmediateMesh *im, const vec3 &from, const vec3 &to, const Color &color) {
	im->surface_add_vertex(from);
	im->surface_add_vertex(to);
	// Simple arrowhead: two lines from tip back
	vec3 dir = (to - from).normalized();
	vec3 perp = (Math::abs(dir.x) < 0.99f) ? dir.cross(vec3(1,0,0)).normalized()
	                                        : dir.cross(vec3(0,1,0)).normalized();
	vec3 perp2 = dir.cross(perp).normalized();
	real_t head_len = (to - from).length() * 0.2f;
	vec3 head_base = to - dir * head_len;
	im->surface_add_vertex(head_base + perp * head_len * 0.5f);
	im->surface_add_vertex(to);
	im->surface_add_vertex(head_base - perp * head_len * 0.5f);
	im->surface_add_vertex(to);
	im->surface_add_vertex(head_base + perp2 * head_len * 0.5f);
	im->surface_add_vertex(to);
	im->surface_add_vertex(head_base - perp2 * head_len * 0.5f);
	im->surface_add_vertex(to);
}

void ViennaDebugDraw::draw_joint_pivot(ImmediateMesh *im, const vec3 &pivot, const Color &color) {
	// Draw a small cross at the pivot
	real_t s = 0.1f;
	im->surface_add_vertex(pivot - vec3(s, 0, 0));
	im->surface_add_vertex(pivot + vec3(s, 0, 0));
	im->surface_add_vertex(pivot - vec3(0, s, 0));
	im->surface_add_vertex(pivot + vec3(0, s, 0));
	im->surface_add_vertex(pivot - vec3(0, 0, s));
	im->surface_add_vertex(pivot + vec3(0, 0, s));
}

} // namespace vienna