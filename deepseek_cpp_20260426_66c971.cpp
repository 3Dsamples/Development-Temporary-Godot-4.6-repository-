// File 177: modules/newton/src/world/newton_world.cpp
// NewtonWorld implementation – simulation loop, broad‑phase collision,
// island construction, constraint solving, and integration.

#include "newton_world.h"

#include "../bodies/newton_body.h"
#include "../joints/newton_joint.h"
#include "../materials/newton_material.h"
#include "../solver/newton_solver.h"
#include "../solver/newton_island.h"

#include "../../../gaia/src/collision_detector/broad_phase.h"
#include "../../../gaia/src/collision_detector/narrow_phase.h"
#include "../../../gaia/src/collision_detector/contact.h"

#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace newton {

void NewtonWorld::_bind_methods() {
	ClassDB::bind_method(D_METHOD("step", "dt"), &NewtonWorld::step);
	ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &NewtonWorld::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity"), &NewtonWorld::get_gravity);
	ClassDB::bind_method(D_METHOD("set_solver_iterations", "iter"), &NewtonWorld::set_solver_iterations);
	ClassDB::bind_method(D_METHOD("get_solver_iterations"), &NewtonWorld::get_solver_iterations);
	ClassDB::bind_method(D_METHOD("set_solver_method", "method"), &NewtonWorld::set_solver_method);
	ClassDB::bind_method(D_METHOD("get_solver_method"), &NewtonWorld::get_solver_method);
	ClassDB::bind_method(D_METHOD("set_sleep_speed_thresholds", "linear", "angular"), &NewtonWorld::set_sleep_speed_thresholds);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_iterations"), "set_solver_iterations", "get_solver_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_method"), "set_solver_method", "get_solver_method");
}

NewtonWorld::NewtonWorld() {
	solver.instantiate();
	island_manager.instantiate();
}

NewtonWorld::~NewtonWorld() {
	// cleanup if needed
}

void NewtonWorld::set_gravity(const vec3 &p_gravity) { gravity = p_gravity; }

void NewtonWorld::set_solver_iterations(int p_iter) {
	solver_iterations = CLAMP(p_iter, 1, MAX_SOLVER_ITERATIONS);
}

void NewtonWorld::set_sleep_speed_thresholds(real_t linear, real_t angular) {
	sleep_linear_threshold = MAX(linear, 0.0);
	sleep_angular_threshold = MAX(angular, 0.0);
}

body_id NewtonWorld::create_body(const Ref<NewtonBody> &p_body) {
	ERR_FAIL_COND_V(p_body.is_null(), 0);
	body_id id = next_body_id++;
	bodies[id] = p_body;
	// Insert into broad-phase with its AABB
	broad_phase.add_object(id, p_body->get_aabb());
	return id;
}

void NewtonWorld::destroy_body(body_id p_id) {
	bodies.erase(p_id);
	broad_phase.remove_object(p_id);
	sleep_counters.erase(p_id);
}

Ref<NewtonBody> NewtonWorld::get_body(body_id p_id) const {
	HashMap<body_id, Ref<NewtonBody>>::ConstIterator it = bodies.find(p_id);
	return it ? it->value : Ref<NewtonBody>();
}

joint_id NewtonWorld::create_joint(const Ref<NewtonJoint> &p_joint) {
	ERR_FAIL_COND_V(p_joint.is_null(), 0);
	joint_id id = next_joint_id++;
	joints[id] = p_joint;
	return id;
}

void NewtonWorld::destroy_joint(joint_id p_id) { joints.erase(p_id); }

Ref<NewtonJoint> NewtonWorld::get_joint(joint_id p_id) const {
	HashMap<joint_id, Ref<NewtonJoint>>::ConstIterator it = joints.find(p_id);
	return it ? it->value : Ref<NewtonJoint>();
}

material_id NewtonWorld::create_material(const Ref<NewtonMaterial> &p_material) {
	ERR_FAIL_COND_V(p_material.is_null(), 0);
	material_id id = next_material_id++;
	materials[id] = p_material;
	return id;
}

void NewtonWorld::destroy_material(material_id p_id) { materials.erase(p_id); }

Ref<NewtonMaterial> NewtonWorld::get_material(material_id p_id) const {
	HashMap<material_id, Ref<NewtonMaterial>>::ConstIterator it = materials.find(p_id);
	return it ? it->value : Ref<NewtonMaterial>();
}

void NewtonWorld::step(real_t p_dt) {
	// 1. Apply forces and integrate velocities (unconstrained)
	for (KeyValue<body_id, Ref<NewtonBody>> &kv : bodies) {
		NewtonBody *body = kv.value.ptr();
		if (!body->is_active()) continue;
		if (body->is_gravity_enabled() && body->get_type() == BodyType::DYNAMIC) {
			body->apply_force(gravity * body->get_mass(), body->get_position());
		}
		body->integrate_velocity(p_dt);
	}

	// 2. Broad‑phase collision detection
	detect_collisions();

	// 3. Build islands (bodies connected by contacts / joints)
	build_islands();

	// 4. Solve islands
	solve_islands(p_dt);

	// 5. Integrate positions
	integrate(p_dt);

	// 6. Update sleep state
	update_sleep_state();

	world_time += p_dt;
}

void NewtonWorld::detect_collisions() {
	// Update AABBs of all active bodies in the broad-phase
	for (KeyValue<body_id, Ref<NewtonBody>> &kv : bodies) {
		if (kv.value->is_active()) {
			broad_phase.update_object(kv.key, kv.value->get_aabb(), true);
		} else {
			broad_phase.update_object(kv.key, kv.value->get_aabb(), false);
		}
	}
	// Collect overlapping pairs (the broad-phase will rebuild internally)
	LocalVector<std::pair<body_id, body_id>> pairs;
	broad_phase.find_pairs([](uint32_t hA, uint32_t hB, void *userdata) {
		auto *vec = static_cast<LocalVector<std::pair<body_id, body_id>>*>(userdata);
		vec->push_back({ (body_id)hA, (body_id)hB });
	}, &pairs);

	// Generate contacts for each overlapping pair (narrow‑phase)
	// We'll use Gaia's narrow‑phase GJK to produce contact points
	for (const auto &pair : pairs) {
		generate_contacts(pair.first, pair.second);
	}
}

void NewtonWorld::generate_contacts(body_id a, body_id b) {
	// Placeholder: contact generation is done later in the solver using GJK.
	// The actual narrow‑phase will be called during constraint solving.
	// We store the pair as a candidate; the solver will fetch them.
	// For now we just record pairs in the island manager.
	island_manager->add_contact_pair(a, b);
}

void NewtonWorld::build_islands() {
	island_manager->build(bodies, joints);
}

void NewtonWorld::solve_islands(real_t dt) {
	solver->set_iterations(solver_iterations);
	solver->solve_islands(island_manager, bodies, joints, materials, dt);
}

void NewtonWorld::integrate(real_t dt) {
	for (KeyValue<body_id, Ref<NewtonBody>> &kv : bodies) {
		NewtonBody *body = kv.value.ptr();
		if (!body->is_active() || body->get_type() != BodyType::DYNAMIC) continue;
		body->integrate_position(dt);
	}
}

void NewtonWorld::update_sleep_state() {
	for (KeyValue<body_id, Ref<NewtonBody>> &kv : bodies) {
		NewtonBody *body = kv.value.ptr();
		if (!body->is_active() || body->get_type() != BodyType::DYNAMIC) continue;
		real_t lin_speed = body->get_linear_velocity().length();
		real_t ang_speed = body->get_angular_velocity().length();
		if (lin_speed < sleep_linear_threshold && ang_speed < sleep_angular_threshold) {
			body->increment_sleep_counter();
			if (body->get_sleep_counter() > sleep_frames) {
				body->set_active(false);
				body->reset_sleep_counter();
			}
		} else {
			body->reset_sleep_counter();
		}
	}
}

} // namespace newton