--- START OF FILE core/simulation/physics_body_hyper.h ---

#ifndef PHYSICS_BODY_HYPER_H
#define PHYSICS_BODY_HYPER_H

#include "core/object/object.h"
#include "core/templates/rid.h"
#include "core/math/transform_3d.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * PhysicsBodyHyper
 * 
 * The foundational physical entity for the Universal Solver.
 * Manages motion, mass, and material state using strictly deterministic math.
 * Aligned to 32 bytes for SIMD-optimized EnTT component streams.
 */
class ET_ALIGN_32 PhysicsBodyHyper : public Object {
	GDCLASS(PhysicsBodyHyper, Object);

public:
	enum BodyMode {
		BODY_MODE_STATIC,
		BODY_MODE_KINEMATIC,
		BODY_MODE_DYNAMIC,
		BODY_MODE_DEFORMABLE
	};

private:
	RID self;
	BodyMode mode = BODY_MODE_STATIC;

	// --- Deterministic State (FixedMathCore) ---
	Vector3f linear_velocity;
	Vector3f angular_velocity;
	Vector3f constant_force;
	Vector3f constant_torque;
	
	FixedMathCore mass;
	FixedMathCore friction;
	FixedMathCore bounce;
	FixedMathCore gravity_scale;

	// --- Spatial State (Sector-Aware) ---
	Transform3Df transform;
	BigIntCore sector_x, sector_y, sector_z;

	// --- Simulation Flags ---
	bool ccd_enabled = true;
	bool active = true;
	bool sleeping = false;

protected:
	static void _bind_methods();

public:
	// ------------------------------------------------------------------------
	// Configuration API
	// ------------------------------------------------------------------------

	void set_mode(BodyMode p_mode);
	_FORCE_INLINE_ BodyMode get_mode() const { return mode; }

	void set_mass(const FixedMathCore &p_mass);
	_FORCE_INLINE_ FixedMathCore get_mass() const { return mass; }

	// ------------------------------------------------------------------------
	// Dynamics API (Warp-Kernel Compatible)
	// ------------------------------------------------------------------------

	void set_linear_velocity(const Vector3f &p_velocity);
	_FORCE_INLINE_ Vector3f get_linear_velocity() const { return linear_velocity; }

	void apply_central_impulse(const Vector3f &p_impulse);
	void apply_impulse(const Vector3f &p_impulse, const Vector3f &p_position);
	void apply_torque_impulse(const Vector3f &p_torque);

	// ------------------------------------------------------------------------
	// Galactic Positioning
	// ------------------------------------------------------------------------

	void set_transform(const Transform3Df &p_transform);
	_FORCE_INLINE_ Transform3Df get_transform() const { return transform; }

	void set_sector(const BigIntCore &p_x, const BigIntCore &p_y, const BigIntCore &p_z);
	void get_sector(BigIntCore &r_x, BigIntCore &r_y, BigIntCore &r_z) const;

	// ------------------------------------------------------------------------
	// Simulation Hooks
	// ------------------------------------------------------------------------

	void set_ccd_enabled(bool p_enable);
	_FORCE_INLINE_ bool is_ccd_enabled() const { return ccd_enabled; }

	/**
	 * integrate_forces()
	 * The deterministic integration kernel.
	 * Updates velocities based on accumulated FixedMath forces.
	 */
	void integrate_forces(const FixedMathCore &p_step);

	/**
	 * integrate_velocities()
	 * The deterministic position kernel.
	 * Updates transform and handles galactic sector transitions.
	 */
	void integrate_velocities(const FixedMathCore &p_step);

	PhysicsBodyHyper();
	virtual ~PhysicsBodyHyper();
};

#endif // PHYSICS_BODY_HYPER_H

--- END OF FILE core/simulation/physics_body_hyper.h ---
