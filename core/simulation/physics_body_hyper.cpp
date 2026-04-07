--- START OF FILE core/simulation/physics_body_hyper.cpp ---

#include "core/simulation/physics_body_hyper.h"
#include "core/object/class_db.h"
#include "core/math/math_funcs.h"
#include "core/simulation/simulation_manager.h"

void PhysicsBodyHyper::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &PhysicsBodyHyper::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &PhysicsBodyHyper::get_mode);
	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &PhysicsBodyHyper::set_mass);
	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &PhysicsBodyHyper::apply_central_impulse);
	ClassDB::bind_method(D_METHOD("integrate_forces", "step"), &PhysicsBodyHyper::integrate_forces);
	ClassDB::bind_method(D_METHOD("integrate_velocities", "step"), &PhysicsBodyHyper::integrate_velocities);
}

PhysicsBodyHyper::PhysicsBodyHyper() {
	mass = MathConstants<FixedMathCore>::one();
	friction = MathConstants<FixedMathCore>::half();
	bounce = FixedMathCore(1288490188LL, true); // 0.3
	gravity_scale = MathConstants<FixedMathCore>::one();
	
	sector_x = BigIntCore(0LL);
	sector_y = BigIntCore(0LL);
	sector_z = BigIntCore(0LL);
}

PhysicsBodyHyper::~PhysicsBodyHyper() {}

void PhysicsBodyHyper::set_mode(BodyMode p_mode) {
	mode = p_mode;
}

void PhysicsBodyHyper::set_mass(const FixedMathCore &p_mass) {
	mass = p_mass;
}

void PhysicsBodyHyper::apply_central_impulse(const Vector3f &p_impulse) {
	if (mode != BODY_MODE_DYNAMIC && mode != BODY_MODE_DEFORMABLE) return;
	// delta_v = impulse / mass
	linear_velocity += p_impulse / mass;
}

/**
 * integrate_forces()
 * 
 * Deterministic Force Kernel.
 * Updates linear and angular velocity using bit-perfect FixedMathCore.
 * Designed for parallel execution in Warp-style sweeps.
 */
void PhysicsBodyHyper::integrate_forces(const FixedMathCore &p_step) {
	if (mode != BODY_MODE_DYNAMIC && mode != BODY_MODE_DEFORMABLE) return;

	// Apply gravity constant (9.80665) scaled by body gravity_scale
	FixedMathCore g_accel = FixedMathCore(42122340ULL, false) >> 32; // Placeholder for precise G
	Vector3f gravity_vec(FixedMathCore(0LL, true), -FixedMathCore(980665LL, true), FixedMathCore(0LL, true));
	
	linear_velocity += gravity_vec * gravity_scale * p_step;
	
	// Integrate constant forces
	linear_velocity += (constant_force / mass) * p_step;
	
	// Simple damping to simulate air/vacuum resistance
	FixedMathCore damping = MathConstants<FixedMathCore>::one() - (FixedMathCore(42949672LL, true) * p_step); // 0.01 damping
	linear_velocity *= damping;
	angular_velocity *= damping;
}

/**
 * integrate_velocities()
 * 
 * Deterministic Position Kernel.
 * Updates the transform and performs Galactic Origin Recenter logic.
 */
void PhysicsBodyHyper::integrate_velocities(const FixedMathCore &p_step) {
	if (mode == BODY_MODE_STATIC) return;

	// Update translation: pos = pos + v * dt
	transform.origin += linear_velocity * p_step;

	// Update rotation: Use bit-perfect axis-angle to Quaternion integration
	FixedMathCore ang_speed = angular_velocity.length();
	if (ang_speed > FixedMathCore(4294LL, true)) { // Epsilon check
		Vector3f axis = angular_velocity / ang_speed;
		Quaternionf rot(axis, ang_speed * p_step);
		transform.basis = Basis<FixedMathCore>(rot) * transform.basis;
		transform.basis.orthonormalize();
	}

	// --- Galactic Origin Drift Correction ---
	// threshold = 10,000 units. If exceeded, shift BigInt sectors.
	const FixedMathCore threshold(10000LL, false);
	Vector3f pos = transform.origin;

	int64_t dx = Math::floor(pos.x / threshold).to_int();
	int64_t dy = Math::floor(pos.y / threshold).to_int();
	int64_t dz = Math::floor(pos.z / threshold).to_int();

	if (dx != 0 || dy != 0 || dz != 0) {
		sector_x += BigIntCore(dx);
		sector_y += BigIntCore(dy);
		sector_z += BigIntCore(dz);

		Vector3f offset(threshold * FixedMathCore(dx), threshold * FixedMathCore(dy), threshold * FixedMathCore(dz));
		transform.origin -= offset;
	}
}

--- END OF FILE core/simulation/physics_body_hyper.cpp ---
