// File 273: modules/vienna/src/world/vienna_world.cpp
// ViennaWorld implementation – simulation step, broad‑phase, island building,
// contact solving, integration, sleep management, and cloth/particle stepping.

#include "vienna_world.h"
#include "../bodies/vienna_body.h"
#include "../joints/vienna_joint.h"
#include "../materials/vienna_material.h"
#include "../cloth/vienna_cloth.h"
#include "../cloth/vienna_cloth_solver.h"
#include "../particles/vienna_particle_system.h"
#include "../solver/vienna_solver.h"
#include "../solver/vienna_island.h"

// Gaia broad‑phase
#include "../../../gaia/src/collision_detector/broad_phase.h"

#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/typedefs.h"

namespace vienna {

void ViennaWorld::_bind_methods() {
	ClassDB::bind_method(D_METHOD("step", "dt"), &ViennaWorld::step);
	ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &ViennaWorld::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity"), &ViennaWorld::get_gravity);
	ClassDB::bind_method(D_METHOD("set_solver_iterations", "iter"), &ViennaWorld::set_solver_iterations);
	ClassDB::bind_method(D_METHOD("get_solver_iterations"), &ViennaWorld::get_solver_iterations);
	ClassDB::bind_method(D_METHOD("set_solver_method", "method"), &ViennaWorld::set_solver_method);
	ClassDB::bind_method(D_METHOD("get_solver_method"), &ViennaWorld::get_solver_method);
	ClassDB::bind_method(D_METHOD("set_sleep_speed_thresholds", "linear", "angular"), &ViennaWorld::set_sleep_speed_thresholds);
	ClassDB::bind_method(D_METHOD("set_sleep_frames", "frames"), &ViennaWorld::set_sleep_frames);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_iterations"), "set_solver_iterations", "get_solver_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_method"), "set_solver_method", "get_solver_method");
}

ViennaWorld::ViennaWorld() {
	solver.instantiate();
	island_manager.instantiate();
}

ViennaWorld::~ViennaWorld() {}

void ViennaWorld::set_gravity(const vec3 &p_gravity) { gravity = p_gravity; }
void ViennaWorld::set_solver_iterations(int p_iter) { solver_iterations = CLAMP(p_iter, 1, MAX_SOLVER_ITERATIONS); }
void ViennaWorld::set_sleep_speed_thresholds(real_t linear, real_t angular) {
	sleep_linear_threshold = MAX(linear, 0.0);
	sleep_angular_threshold = MAX(angular, 0.0);
}

body_id ViennaWorld::create_body(const Ref<ViennaBody> &p_body) {
	ERR_FAIL_COND_V(p_body.is_null(), 0);
	body_id id = next_body_id++;
	bodies[id] = p_body;
	broad_phase.add_object(id, p_body->get_aabb());
	return id;
}

void ViennaWorld::destroy_body(body_id p_id) {
	bodies.erase(p_id);
	broad_phase.remove_object(p_id);
}

Ref<ViennaBody> ViennaWorld::get_body(body_id p_id) const {
	HashMap<body_id, Ref<ViennaBody>>::ConstIterator it = bodies.find(p_id);
	return it ? it->value : Ref<ViennaBody>();
}

int ViennaWorld::get_body_count() const { return bodies.size(); }

LocalVector<body_id> ViennaWorld::get_body_ids() const {
	LocalVector<body_id> ids;
	for (const KeyValue<body_id, Ref<ViennaBody>> &kv : bodies) ids.push_back(kv.key);
	return ids;
}

joint_id ViennaWorld::create_joint(const Ref<ViennaJoint> &p_joint) {
	ERR_FAIL_COND_V(p_joint.is_null(), 0);
	joint_id id = next_joint_id++;
	joints[id] = p_joint;
	return id;
}

void ViennaWorld::destroy_joint(joint_id p_id) { joints.erase(p_id); }

Ref<ViennaJoint> ViennaWorld::get_joint(joint_id p_id) const {
	HashMap<joint_id, Ref<ViennaJoint>>::ConstIterator it = joints.find(p_id);
	return it ? it->value : Ref<ViennaJoint>();
}

LocalVector<joint_id> ViennaWorld::get_joint_ids() const {
	LocalVector<joint_id> ids;
	for (const KeyValue<joint_id, Ref<ViennaJoint>> &kv : joints) ids.push_back(kv.key);
	return ids;
}

material_id ViennaWorld::create_material(const Ref<ViennaMaterial> &p_material) {
	ERR_FAIL_COND_V(p_material.is_null(), 0);
	material_id id = next_material_id++;
	materials[id] = p_material;
	return id;
}

void ViennaWorld::destroy_material(material_id p_id) { materials.erase(p_id); }

Ref<ViennaMaterial> ViennaWorld::get_material(material_id p_id) const {
	HashMap<material_id, Ref<ViennaMaterial>>::ConstIterator it = materials.find(p_id);
	return it ? it->value : Ref<ViennaMaterial>();
}

LocalVector<material_id> ViennaWorld::get_material_ids() const {
	LocalVector<material_id> ids;
	for (const KeyValue<material_id, Ref<ViennaMaterial>> &kv : materials) ids.push_back(kv.key);
	return ids;
}

cloth_id ViennaWorld::create_cloth(const Ref<ViennaCloth> &p_cloth) {
	ERR_FAIL_COND_V(p_cloth.is_null(), 0);
	cloth_id id = next_cloth_id++;
	cloths[id] = p_cloth;
	return id;
}

void ViennaWorld::destroy_cloth(cloth_id p_id) { cloths.erase(p_id); }

Ref<ViennaCloth> ViennaWorld::get_cloth(cloth_id p_id) const {
	HashMap<cloth_id, Ref<ViennaCloth>>::ConstIterator it = cloths.find(p_id);
	return it ? it->value : Ref<ViennaCloth>();
}

LocalVector<cloth_id> ViennaWorld::get_cloth_ids() const {
	LocalVector<cloth_id> ids;
	for (const KeyValue<cloth_id, Ref<ViennaCloth>> &kv : cloths) ids.push_back(kv.key);
	return ids;
}

cloth_id ViennaWorld::create_particle_system(const Ref<ViennaParticleSystem> &p_system) {
	ERR_FAIL_COND_V(p_system.is_null(), 0);
	cloth_id id = next_cloth_id++;
	particle_systems[id] = p_system;
	return id;
}

void ViennaWorld::destroy_particle_system(cloth_id p_id) { particle_systems.erase(p_id); }

Ref<ViennaParticleSystem> ViennaWorld::get_particle_system(cloth_id p_id) const {
	HashMap<cloth_id, Ref<ViennaParticleSystem>>::ConstIterator it = particle_systems.find(p_id);
	return it ? it->value : Ref<ViennaParticleSystem>();
}

void ViennaWorld::step(real_t p_dt) {
	// Sub‑stepping not shown – single step for brevity
	// 1. Apply forces and gravity to all dynamic bodies
	apply_forces(p_dt);

	// 2. Broad‑phase collision detection (update AABBs, find pairs)
	detect_collisions();

	// 3. Build islands from contact pairs and joints
	build_islands();

	// 4. Solve contacts and joints within each island
	solve_islands(p_dt);

	// 5. Integrate positions
	integrate(p_dt);

	// 6. Step cloth and particle systems
	step_cloths(p_dt);
	step_particle_systems(p_dt);

	// 7. Update sleep state
	update_sleep_state();

	world_time += p_dt;
}

void ViennaWorld::apply_forces(real_t dt) {
	for (KeyValue<body_id, Ref<ViennaBody>> &kv : bodies) {
		ViennaBody *body = kv.value.ptr();
		if (!body || !body->is_active()) continue;
		body->clear_forces();
		if (body->is_gravity_enabled() && body->get_type() == BodyType::DYNAMIC) {
			body->apply_force(gravity * body->get_mass(), body->get_position());
		}
		// integrate velocities with external forces
		body->integrate_velocity(dt);
	}
}

void ViennaWorld::detect_collisions() {
	// Update AABBs in Gaia broad‑phase
	for (KeyValue<body_id, Ref<ViennaBody>> &kv : bodies) {
		ViennaBody *body = kv.value.ptr();
		if (!body) continue;
		bool active = body->is_active() && (body->get_type() != BodyType::STATIC);
		broad_phase.update_object(kv.key, body->get_aabb(), active);
	}

	// Find overlapping pairs
	contact_pairs.clear();
	broad_phase.find_pairs([](uint32_t hA, uint32_t hB, void *userdata) {
		auto *vec = static_cast<LocalVector<std::pair<body_id, body_id>>*>(userdata);
		vec->push_back({(body_id)hA, (body_id)hB});
	}, &contact_pairs);
}

void ViennaWorld::build_islands() {
	island_manager->clear();
	island_manager->build(bodies, joints, contact_pairs);
}

void ViennaWorld::solve_islands(real_t dt) {
	solver->set_iterations(solver_iterations);
	solver->solve_islands(island_manager, bodies, joints, materials, dt);

	// Notify contact callback if set
	if (contact_callback.is_valid()) {
		// For now, the callback is not invoked; user can poll via query methods.
	}
}

void ViennaWorld::integrate(real_t dt) {
	for (KeyValue<body_id, Ref<ViennaBody>> &kv : bodies) {
		ViennaBody *body = kv.value.ptr();
		if (!body || !body->is_active() || body->get_type() != BodyType::DYNAMIC) continue;
		body->integrate_position(dt);
	}
}

void ViennaWorld::update_sleep_state() {
	for (KeyValue<body_id, Ref<ViennaBody>> &kv : bodies) {
		ViennaBody *body = kv.value.ptr();
		if (!body || !body->is_active() || body->get_type() != BodyType::DYNAMIC) continue;
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

void ViennaWorld::step_cloths(real_t dt) {
	for (KeyValue<cloth_id, Ref<ViennaCloth>> &kv : cloths) {
		if (kv.value.is_valid()) {
			kv.value->step(dt);
		}
	}
}

void ViennaWorld::step_particle_systems(real_t dt) {
	for (KeyValue<cloth_id, Ref<ViennaParticleSystem>> &kv : particle_systems) {
		if (kv.value.is_valid()) {
			kv.value->step(dt);
		}
	}
}

} // namespace vienna