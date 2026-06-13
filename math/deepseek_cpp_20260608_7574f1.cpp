// File 255: modules/newton/src/nodes/newton_world_node.cpp
// NewtonWorldNode implementation – creates a NewtonWorld, applies config,
// runs physics each _physics_process, and provides access to the world.

#include "newton_world_node.h"

#include "../world/newton_world.h"
#include "../world/newton_world_config.h"
#include "../core/newton_constants.h"

#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace newton {

void NewtonWorldNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_config", "config"), &NewtonWorldNode::set_config);
	ClassDB::bind_method(D_METHOD("get_config"), &NewtonWorldNode::get_config);
	ClassDB::bind_method(D_METHOD("set_active", "active"), &NewtonWorldNode::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &NewtonWorldNode::is_active);
	ClassDB::bind_method(D_METHOD("get_newton_world"), &NewtonWorldNode::get_newton_world);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "config", PROPERTY_HINT_RESOURCE_TYPE, "NewtonWorldConfig"), "set_config", "get_config");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");
}

NewtonWorldNode::NewtonWorldNode() :
	world(nullptr),
	active(true) {
	world = memnew(NewtonWorld);
	set_physics_process(true);
}

NewtonWorldNode::~NewtonWorldNode() {
	if (world) {
		memdelete(world);
		world = nullptr;
	}
}

void NewtonWorldNode::set_config(const Ref<NewtonWorldConfig> &p_config) {
	config = p_config;
	if (config.is_valid()) {
		_apply_config();
	}
}

Ref<NewtonWorldConfig> NewtonWorldNode::get_config() const {
	return config;
}

void NewtonWorldNode::set_active(bool p_active) {
	active = p_active;
}

bool NewtonWorldNode::is_active() const {
	return active;
}

void NewtonWorldNode::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		// Apply config if set before entering tree
		if (config.is_valid()) {
			_apply_config();
		}
	}
	if (p_what == NOTIFICATION_PHYSICS_PROCESS) {
		if (active && world) {
			real_t dt = get_physics_process_delta_time();
			world->step(dt);
		}
	}
}

void NewtonWorldNode::_apply_config() {
	if (!world) return;
	world->set_gravity(config->get_gravity());
	world->set_solver_iterations(config->get_solver_iterations());
	world->set_solver_method(config->get_solver_method());
	world->set_sleep_speed_thresholds(config->get_sleep_linear_speed(),
									  config->get_sleep_angular_speed());
	world->set_sleep_frames(config->get_sleep_frames());
	// CCD and broad-phase settings are not exposed yet; add later if needed.
}

} // namespace newton