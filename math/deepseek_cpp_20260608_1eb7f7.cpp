// File 314: modules/vienna/src/settings/vienna_physics_settings.cpp
// Implementation of ViennaPhysicsSettings – default values and property bindings.

#include "vienna_physics_settings.h"
#include "../core/vienna_constants.h"

namespace vienna {

ViennaPhysicsSettings::ViennaPhysicsSettings() :
	gravity(0.0, DEFAULT_GRAVITY, 0.0),
	solver_iterations(DEFAULT_SOLVER_ITERATIONS),
	broad_phase_algo(BroadPhaseAlgorithm::BVH),
	sleep_linear_speed(DEFAULT_SLEEP_LINEAR),
	sleep_angular_speed(DEFAULT_SLEEP_ANGULAR),
	sleep_frames(DEFAULT_SLEEP_FRAMES),
	sub_steps(1),
	collision_margin(0.001),
	default_friction(DEFAULT_FRICTION),
	default_restitution(DEFAULT_RESTITUTION),
	default_softness(DEFAULT_SOFTNESS) {}

void ViennaPhysicsSettings::set_gravity(const Vector3 &p_g) { gravity = p_g; }
void ViennaPhysicsSettings::set_solver_iterations(int p_iters) { solver_iterations = CLAMP(p_iters, 1, MAX_SOLVER_ITERATIONS); }
void ViennaPhysicsSettings::set_broad_phase_algorithm(int p_algo) { broad_phase_algo = (BroadPhaseAlgorithm)CLAMP(p_algo, 0, 2); }
void ViennaPhysicsSettings::set_sleep_linear_speed(real_t p_speed) { sleep_linear_speed = MAX(p_speed, 0.0); }
void ViennaPhysicsSettings::set_sleep_angular_speed(real_t p_speed) { sleep_angular_speed = MAX(p_speed, 0.0); }
void ViennaPhysicsSettings::set_sleep_frames(int p_frames) { sleep_frames = MAX(p_frames, 1); }
void ViennaPhysicsSettings::set_sub_steps(int p_sub) { sub_steps = MAX(p_sub, 1); }
void ViennaPhysicsSettings::set_collision_margin(real_t p_margin) { collision_margin = MAX(p_margin, 0.0); }
void ViennaPhysicsSettings::set_default_friction(real_t p_fric) { default_friction = CLAMP(p_fric, 0.0, 10.0); }
void ViennaPhysicsSettings::set_default_restitution(real_t p_rest) { default_restitution = CLAMP(p_rest, 0.0, 1.0); }
void ViennaPhysicsSettings::set_default_softness(real_t p_soft) { default_softness = MAX(p_soft, 0.0); }

void ViennaPhysicsSettings::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_gravity", "gravity"), &ViennaPhysicsSettings::set_gravity);
	ClassDB::bind_method(D_METHOD("get_gravity"), &ViennaPhysicsSettings::get_gravity);
	ClassDB::bind_method(D_METHOD("set_solver_iterations", "iterations"), &ViennaPhysicsSettings::set_solver_iterations);
	ClassDB::bind_method(D_METHOD("get_solver_iterations"), &ViennaPhysicsSettings::get_solver_iterations);
	ClassDB::bind_method(D_METHOD("set_broad_phase_algorithm", "algo"), &ViennaPhysicsSettings::set_broad_phase_algorithm);
	ClassDB::bind_method(D_METHOD("get_broad_phase_algorithm"), &ViennaPhysicsSettings::get_broad_phase_algorithm);
	ClassDB::bind_method(D_METHOD("set_sleep_linear_speed", "speed"), &ViennaPhysicsSettings::set_sleep_linear_speed);
	ClassDB::bind_method(D_METHOD("get_sleep_linear_speed"), &ViennaPhysicsSettings::get_sleep_linear_speed);
	ClassDB::bind_method(D_METHOD("set_sleep_angular_speed", "speed"), &ViennaPhysicsSettings::set_sleep_angular_speed);
	ClassDB::bind_method(D_METHOD("get_sleep_angular_speed"), &ViennaPhysicsSettings::get_sleep_angular_speed);
	ClassDB::bind_method(D_METHOD("set_sleep_frames", "frames"), &ViennaPhysicsSettings::set_sleep_frames);
	ClassDB::bind_method(D_METHOD("get_sleep_frames"), &ViennaPhysicsSettings::get_sleep_frames);
	ClassDB::bind_method(D_METHOD("set_sub_steps", "sub_steps"), &ViennaPhysicsSettings::set_sub_steps);
	ClassDB::bind_method(D_METHOD("get_sub_steps"), &ViennaPhysicsSettings::get_sub_steps);
	ClassDB::bind_method(D_METHOD("set_collision_margin", "margin"), &ViennaPhysicsSettings::set_collision_margin);
	ClassDB::bind_method(D_METHOD("get_collision_margin"), &ViennaPhysicsSettings::get_collision_margin);
	ClassDB::bind_method(D_METHOD("set_default_friction", "friction"), &ViennaPhysicsSettings::set_default_friction);
	ClassDB::bind_method(D_METHOD("get_default_friction"), &ViennaPhysicsSettings::get_default_friction);
	ClassDB::bind_method(D_METHOD("set_default_restitution", "restitution"), &ViennaPhysicsSettings::set_default_restitution);
	ClassDB::bind_method(D_METHOD("get_default_restitution"), &ViennaPhysicsSettings::get_default_restitution);
	ClassDB::bind_method(D_METHOD("set_default_softness", "softness"), &ViennaPhysicsSettings::set_default_softness);
	ClassDB::bind_method(D_METHOD("get_default_softness"), &ViennaPhysicsSettings::get_default_softness);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "solver_iterations", PROPERTY_HINT_RANGE, "1,256,1"), "set_solver_iterations", "get_solver_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "broad_phase_algorithm", PROPERTY_HINT_ENUM, "BruteForce,SAP,BVH"), "set_broad_phase_algorithm", "get_broad_phase_algorithm");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sleep_linear_speed"), "set_sleep_linear_speed", "get_sleep_linear_speed");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sleep_angular_speed"), "set_sleep_angular_speed", "get_sleep_angular_speed");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sleep_frames", PROPERTY_HINT_RANGE, "1,100,1"), "set_sleep_frames", "get_sleep_frames");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sub_steps", PROPERTY_HINT_RANGE, "1,10,1"), "set_sub_steps", "get_sub_steps");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_margin"), "set_collision_margin", "get_collision_margin");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_friction"), "set_default_friction", "get_default_friction");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_restitution"), "set_default_restitution", "get_default_restitution");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_softness"), "set_default_softness", "get_default_softness");
}

} // namespace vienna