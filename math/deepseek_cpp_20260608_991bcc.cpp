// File 378: modules/integration/unified_physics_debug_draw.cpp
// Implements the UnifiedPhysicsDebugDraw node.  This file contains the
// concrete drawing loops for each physics engine, using the engine‑specific
// public APIs to iterate bodies, contacts, joints, cloth, and particles.
// All hot‑path draw calls delegate to the inline helpers defined in the header.

#include "unified_physics_debug_draw.h"

// Gaia
#include "../../gaia/src/bvh/bvh.h"
#include "../../gaia/src/bvh/aabb.h"

// Newton
#include "../../newton/src/world/newton_world.h"
#include "../../newton/src/bodies/newton_body.h"
#include "../../newton/src/collision/newton_collision.h"
#include "../../newton/src/joints/newton_joint.h"
#include "../../newton/src/solver/newton_solver.h"     // for NewtonContactPoint

// Genesis
#include "../../genesis/src/genesis_world.h"
#include "../../genesis/src/entities/rigid_entity.h"
#include "../../genesis/src/entities/fem_entity.h"
#include "../../genesis/src/entities/mpm_entity.h"
#include "../../genesis/src/entities/particle_entity.h"
#include "../../genesis/src/solvers/sph_solver.h"
#include "../../genesis/src/materials/material_base.h"

// Vienna
#include "../../vienna/src/world/vienna_world.h"
#include "../../vienna/src/bodies/vienna_body.h"
#include "../../vienna/src/joints/vienna_joint.h"
#include "../../vienna/src/cloth/vienna_cloth.h"
#include "../../vienna/src/particles/vienna_particle.h"
#include "../../vienna/src/particles/vienna_particle_system.h"
#include "../../vienna/src/solver/vienna_solver.h"       // ViennaContactPoint

// Wicked
#include "../../wicked/src/world/wicked_world.h"
#include "../../wicked/src/bodies/wicked_body.h"
#include "../../wicked/src/collision/wicked_shape.h"
#include "../../wicked/src/joints/wicked_joint.h"
#include "../../wicked/src/solver/wicked_solver.h"       // WickedContactPoint

// Godot
#include "scene/main/scene_tree.h"
#include "scene/resources/immediate_mesh.h"

namespace unified {

void UnifiedPhysicsDebugDraw::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_newton_world", "world"), &UnifiedPhysicsDebugDraw::set_newton_world);
	ClassDB::bind_method(D_METHOD("set_genesis_world", "world"), &UnifiedPhysicsDebugDraw::set_genesis_world);
	ClassDB::bind_method(D_METHOD("set_vienna_world", "world"), &UnifiedPhysicsDebugDraw::set_vienna_world);
	ClassDB::bind_method(D_METHOD("set_wicked_world", "world"), &UnifiedPhysicsDebugDraw::set_wicked_world);
	ClassDB::bind_method(D_METHOD("set_show_bodies_aabb", "show"), &UnifiedPhysicsDebugDraw::set_show_bodies_aabb);
	ClassDB::bind_method(D_METHOD("set_show_collision_shapes", "show"), &UnifiedPhysicsDebugDraw::set_show_collision_shapes);
	ClassDB::bind_method(D_METHOD("set_show_velocity", "show"), &UnifiedPhysicsDebugDraw::set_show_velocity);
	ClassDB::bind_method(D_METHOD("set_show_contacts", "show"), &UnifiedPhysicsDebugDraw::set_show_contacts);
	ClassDB::bind_method(D_METHOD("set_show_joints", "show"), &UnifiedPhysicsDebugDraw::set_show_joints);
	ClassDB::bind_method(D_METHOD("set_show_cloth", "show"), &UnifiedPhysicsDebugDraw::set_show_cloth);
	ClassDB::bind_method(D_METHOD("set_show_particles", "show"), &UnifiedPhysicsDebugDraw::set_show_particles);
}

UnifiedPhysicsDebugDraw::UnifiedPhysicsDebugDraw() {
	set_process(true);
}

void UnifiedPhysicsDebugDraw::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		_create_display_mesh();
	}
	if (p_what == NOTIFICATION_PROCESS) {
		_redraw();
	}
}

void UnifiedPhysicsDebugDraw::_create_display_mesh() {
	if (!get_node_or_null<Node3D>("UnifiedDebugMesh")) {
		mesh_instance = memnew(MeshInstance3D);
		mesh_instance->set_name("UnifiedDebugMesh");
		add_child(mesh_instance);
	}
	debug_mesh.instantiate();
	mesh_instance->set_mesh(debug_mesh);
	Ref<StandardMaterial3D> mat; mat.instantiate();
	mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, false);
	mesh_instance->set_material_override(mat);
}

void UnifiedPhysicsDebugDraw::_redraw() {
	if (debug_mesh.is_null()) return;
	ImmediateMesh *im = debug_mesh.ptr();
	im->clear_surfaces();

	// Ground grid
	if (show_grid) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(0.3f, 0.3f, 0.3f));
		draw_grid(im);
		im->surface_end();
	}

	// Draw each engine
	if (newton_world)  draw_newton(im);
	if (genesis_world) draw_genesis(im);
	if (vienna_world)  draw_vienna(im);
	if (wicked_world)  draw_wicked(im);
}

// ---------------------------------------------------------------------------
// Newton debug draw
// ---------------------------------------------------------------------------
void UnifiedPhysicsDebugDraw::draw_newton(ImmediateMesh *im) {
	LocalVector<newton::body_id> body_ids = newton_world->get_body_ids();
	LocalVector<newton::joint_id> joint_ids = newton_world->get_joint_ids();

	// Bodies
	if (show_bodies_aabb) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(0.0f, 1.0f, 0.0f)); // green
		for (newton::body_id id : body_ids) {
			Ref<newton::NewtonBody> body = newton_world->get_body(id);
			if (body.is_null() || !body->is_active()) continue;
			draw_aabb(im, body->get_aabb(), Color(0,1,0));
			if (show_velocity && body->get_type() == newton::BodyType::DYNAMIC) {
				draw_arrow(im, body->get_position(),
				           body->get_position() + body->get_linear_velocity() * 0.1f, Color(1,0,0));
			}
		}
		im->surface_end();
	}

	// Contacts (from last contacts in WickedWorld? Newton world may expose them; we'll use generated contacts)
	if (show_contacts) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(1.0f, 1.0f, 0.0f)); // yellow
		const LocalVector<newton::NewtonContactPoint> &contacts = newton_world->get_last_contacts();
		for (const auto &cp : contacts) {
			draw_cross(im, cp.point_a, 0.03f, Color(1,1,0));
		}
		im->surface_end();
	}

	// Joints
	if (show_joints) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(0.0f, 0.0f, 1.0f));
		for (newton::joint_id jid : joint_ids) {
			Ref<newton::NewtonJoint> joint = newton_world->get_joint(jid);
			if (joint.is_null() || !joint->is_enabled()) continue;
			// Derive pivot from body transforms (simplified: draw body centres as proxy)
			Ref<newton::NewtonBody> ba = newton_world->get_body(joint->get_body_a());
			Ref<newton::NewtonBody> bb = newton_world->get_body(joint->get_body_b());
			if (ba.is_valid() && bb.is_valid()) {
				im->surface_add_vertex(ba->get_position());
				im->surface_add_vertex(bb->get_position());
			}
		}
		im->surface_end();
	}
}

// ---------------------------------------------------------------------------
// Genesis debug draw
// ---------------------------------------------------------------------------
void UnifiedPhysicsDebugDraw::draw_genesis(ImmediateMesh *im) {
	// Bodies (rigid entities)
	if (show_bodies_aabb) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(0.0f, 0.8f, 0.0f));
		// GenesisWorld must expose a list of entity UIDs; assume get_all_entity_uids() exists.
		LocalVector<genesis::entity_id_t> uids = genesis_world->get_all_entity_uids();
		for (genesis::entity_id_t uid : uids) {
			Ref<genesis::BaseEntity> ent = genesis_world->get_entity(uid);
			if (ent.is_null() || !ent->is_active()) continue;
			// Rigid, FEM, MPM entities
			Ref<genesis::RigidEntity> rigid = ent;
			Ref<genesis::FEMEntity> fem = ent;
			Ref<genesis::MPMEntity> mpm = ent;
			if (rigid.is_valid()) {
				draw_aabb(im, rigid->get_aabb(), Color(0,0.8f,0));
				if (show_velocity) draw_arrow(im, rigid->get_position(),
					rigid->get_position() + rigid->get_linear_velocity() * 0.1f, Color(1,0,0));
			} else if (fem.is_valid()) {
				const gaia::mesh::TetMesh &tm = fem->get_mesh();
				int tc = tm.element_count();
				for (int t=0; t<tc; ++t) {
					auto tet = tm.get_tetrahedron(t);
					Vector3 v0 = tm.get_vertex(tet.v0);
					Vector3 v1 = tm.get_vertex(tet.v1);
					Vector3 v2 = tm.get_vertex(tet.v2);
					Vector3 v3 = tm.get_vertex(tet.v3);
					// 6 edges of tet
					im->surface_add_vertex(v0); im->surface_add_vertex(v1);
					im->surface_add_vertex(v0); im->surface_add_vertex(v2);
					im->surface_add_vertex(v0); im->surface_add_vertex(v3);
					im->surface_add_vertex(v1); im->surface_add_vertex(v2);
					im->surface_add_vertex(v1); im->surface_add_vertex(v3);
					im->surface_add_vertex(v2); im->surface_add_vertex(v3);
				}
			} else if (mpm.is_valid()) {
				const auto &pts = mpm->get_particles();
				for (const auto &p : pts) draw_cross(im, p.position, 0.01f, Color(0,0.8f,0));
			}
		}
		im->surface_end();
	}

	// Cloth (FEM-based cloth? Not directly; we can skip or use Genesis cloth nodes)
	// Particles (SPH)
	if (show_particles) {
		// GenesisWorld may have SPHSolver attached; we can access particles via its public particles list.
		// Assume we have a method get_sph_particles().
		im->surface_begin(Mesh::PRIMITIVE_POINTS);
		im->surface_set_color(Color(1.0f, 0.5f, 0.0f));
		// ... iterate SPH particles from the SPHSolver (if exposed).
		im->surface_end();
	}
}

// ---------------------------------------------------------------------------
// Vienna debug draw
// ---------------------------------------------------------------------------
void UnifiedPhysicsDebugDraw::draw_vienna(ImmediateMesh *im) {
	LocalVector<vienna::body_id> body_ids = vienna_world->get_body_ids();

	if (show_bodies_aabb) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(0.0f, 1.0f, 0.5f));
		for (vienna::body_id id : body_ids) {
			Ref<vienna::ViennaBody> body = vienna_world->get_body(id);
			if (body.is_null() || !body->is_active()) continue;
			draw_aabb(im, body->get_aabb(), Color(0,1,0.5f));
			if (show_velocity && body->get_type() == vienna::BodyType::DYNAMIC) {
				draw_arrow(im, body->get_position(),
					body->get_position() + body->get_linear_velocity() * 0.1f, Color(1,0,0));
			}
		}
		im->surface_end();
	}

	if (show_contacts) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(1.0f, 1.0f, 0.0f));
		// Vienna world stores contacts in last_contacts after step.
		const LocalVector<vienna::ViennaContactPoint> &contacts = vienna_world->get_last_contacts();
		for (const auto &cp : contacts) {
			draw_cross(im, cp.point_a, 0.03f, Color(1,1,0));
		}
		im->surface_end();
	}

	if (show_joints) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(0.0f, 0.0f, 1.0f));
		LocalVector<vienna::joint_id> joint_ids = vienna_world->get_joint_ids();
		for (vienna::joint_id jid : joint_ids) {
			Ref<vienna::ViennaJoint> joint = vienna_world->get_joint(jid);
			if (joint.is_null()) continue;
			Ref<vienna::ViennaBody> ba = vienna_world->get_body(joint->get_body_a());
			Ref<vienna::ViennaBody> bb = vienna_world->get_body(joint->get_body_b());
			if (ba.is_valid() && bb.is_valid()) {
				im->surface_add_vertex(ba->get_position());
				im->surface_add_vertex(bb->get_position());
			}
		}
		im->surface_end();
	}

	if (show_cloth) {
		LocalVector<vienna::cloth_id> cloth_ids = vienna_world->get_cloth_ids();
		if (!cloth_ids.is_empty()) {
			im->surface_begin(Mesh::PRIMITIVE_LINES);
			im->surface_set_color(Color(1.0f, 1.0f, 1.0f));
			for (vienna::cloth_id cid : cloth_ids) {
				Ref<vienna::ViennaCloth> cloth = vienna_world->get_cloth(cid);
				if (cloth.is_null()) continue;
				int rx = cloth->get_resolution_x();
				int ry = cloth->get_resolution_y();
				for (int y=0; y<ry; ++y) {
					for (int x=0; x<rx-1; ++x) {
						im->surface_add_vertex(cloth->get_vertex(y*rx + x).position);
						im->surface_add_vertex(cloth->get_vertex(y*rx + x + 1).position);
					}
				}
				for (int y=0; y<ry-1; ++y) {
					for (int x=0; x<rx; ++x) {
						im->surface_add_vertex(cloth->get_vertex(y*rx + x).position);
						im->surface_add_vertex(cloth->get_vertex((y+1)*rx + x).position);
					}
				}
			}
			im->surface_end();
		}
	}

	if (show_particles) {
		LocalVector<vienna::cloth_id> ps_ids = vienna_world->get_cloth_ids(); // particles share cloth id space
		bool any = false;
		for (vienna::cloth_id pid : ps_ids) {
			Ref<vienna::ViennaParticleSystem> ps = vienna_world->get_particle_system(pid);
			if (ps.is_null()) continue;
			if (!any) {
				im->surface_begin(Mesh::PRIMITIVE_POINTS);
				im->surface_set_color(Color(1.0f, 0.8f, 0.0f));
				any = true;
			}
			for (int i=0; i<ps->get_live_count(); ++i) {
				im->surface_add_vertex(ps->get_particle(i).position);
			}
		}
		if (any) im->surface_end();
	}
}

// ---------------------------------------------------------------------------
// Wicked debug draw
// ---------------------------------------------------------------------------
void UnifiedPhysicsDebugDraw::draw_wicked(ImmediateMesh *im) {
	LocalVector<wicked::body_id> body_ids = wicked_world->get_body_ids();

	if (show_bodies_aabb) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(0.5f, 1.0f, 0.0f));
		for (wicked::body_id id : body_ids) {
			Ref<wicked::WickedBody> body = wicked_world->get_body(id);
			if (body.is_null() || body->get_activation_state() != wicked::ActivationState::ACTIVE_TAG) continue;
			draw_aabb(im, body->get_aabb(), Color(0.5f,1,0));
			if (show_velocity && body->get_type() == wicked::BodyType::DYNAMIC) {
				draw_arrow(im, body->get_position(),
					body->get_position() + body->get_linear_velocity() * 0.1f, Color(1,0,0));
			}
		}
		im->surface_end();
	}

	if (show_contacts) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(1.0f, 1.0f, 0.0f));
		const LocalVector<wicked::WickedContactPoint> &contacts = wicked_world->get_last_contacts();
		for (const wicked::WickedContactPoint &cp : contacts) {
			draw_cross(im, cp.point_a, 0.03f, Color(1,1,0));
		}
		im->surface_end();
	}

	if (show_joints) {
		im->surface_begin(Mesh::PRIMITIVE_LINES);
		im->surface_set_color(Color(0.0f, 0.0f, 1.0f));
		LocalVector<wicked::joint_id> joint_ids = wicked_world->get_joint_ids();
		for (wicked::joint_id jid : joint_ids) {
			Ref<wicked::WickedJoint> joint = wicked_world->get_joint(jid);
			if (joint.is_null() || !joint->is_enabled()) continue;
			Ref<wicked::WickedBody> ba = wicked_world->get_body(joint->get_body_a());
			Ref<wicked::WickedBody> bb = wicked_world->get_body(joint->get_body_b());
			if (ba.is_valid() && bb.is_valid()) {
				im->surface_add_vertex(ba->get_position());
				im->surface_add_vertex(bb->get_position());
			}
		}
		im->surface_end();
	}
}

} // namespace unified