// File 307: modules/vienna/src/nodes/vienna_world_node_3d.cpp
// Implementation of ViennaWorldNode3D – creates a ViennaWorld, applies settings,
// and steps the simulation in _physics_process.

#include "vienna_world_node_3d.h"
#include "../world/vienna_world.h"
#include "../core/vienna_constants.h"
#include "core/typedefs.h"

namespace vienna {

void ViennaWorldNode3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_vienna_world"), &ViennaWorldNode3D::get_vienna_world);
	ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &ViennaWorldNode3D::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity"), &ViennaWorldNode3D::get_gravity);
	ClassDB::bind_method(D_METHOD("set_solver_iterations", "iterations"), &ViennaWorldNode3D::set_solver_iterations);
	ClassDB::bind_method(D_METHOD("get_solver_iterations"), &ViennaWorldNode3D::get_solver_iterations);
	ClassDB::bind_method(D_METHOD("set_sleep_frames", "frames"), &ViennaWorldNode3D::set_sleep_frames);
	ClassDB::bind_method(D_METHOD("get_sleep_frames"), &ViennaWorldNode3D::get_sleep_frames);
	ClassDB::bind_method(D_METHOD("set_active", "active"), &ViennaWorldNode3D::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &ViennaWorldNode3D::is_active);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_iterations", PROPERTY_HINT_RANGE, "1,256,1"), "set_solver_iterations", "get_solver_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sleep_frames", PROPERTY_HINT_RANGE, "1,100,1"), "set_sleep_frames", "get_sleep_frames");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");
}

ViennaWorldNode3D::ViennaWorldNode3D() :
	world(nullptr),
	active(true),
	gravity(0.0, -9.80665, 0.0),
	solver_iterations(16),
	sleep_frames(10) {
	world = memnew(ViennaWorld);
	world->set_gravity(gravity);
	world->set_solver_iterations(solver_iterations);
	world->set_sleep_frames(sleep_frames);
	set_physics_process(true);
}

ViennaWorldNode3D::~ViennaWorldNode3D() {
	if (world) {
		memdelete(world);
		world = nullptr;
	}
}

void ViennaWorldNode3D::set_gravity(const vec3 &p_gravity) {
	gravity = p_gravity;
	if (world) world->set_gravity(p_gravity);
}

vec3 ViennaWorldNode3D::get_gravity() const { return gravity; }

void ViennaWorldNode3D::set_solver_iterations(int p_iters) {
	solver_iterations = CLAMP(p_iters, 1, 256);
	if (world) world->set_solver_iterations(solver_iterations);
}

int ViennaWorldNode3D::get_solver_iterations() const { return solver_iterations; }

void ViennaWorldNode3D::set_sleep_frames(int p_frames) {
	sleep_frames = MAX(p_frames, 1);
	if (world) world->set_sleep_frames(sleep_frames);
}

int ViennaWorldNode3D::get_sleep_frames() const { return sleep_frames; }

void ViennaWorldNode3D::set_active(bool p_active) { active = p_active; }
bool ViennaWorldNode3D::is_active() const { return active; }

void ViennaWorldNode3D::set_config(const Ref<Resource> &p_config) { config = p_config; }
Ref<Resource> ViennaWorldNode3D::get_config() const { return config; }

void ViennaWorldNode3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_PHYSICS_PROCESS) {
		if (active && world) {
			real_t dt = get_physics_process_delta_time();
			world->step(dt);
		}
	}
}

} // namespace vienna