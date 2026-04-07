--- START OF FILE core/simulation/physics_server_hyper.h ---

#ifndef PHYSICS_SERVER_HYPER_H
#define PHYSICS_SERVER_HYPER_H

#include "core/object/object.h"
#include "core/templates/rid_owner.h"
#include "core/math/transform_3d.h"
#include "core/math/spatial_partition.h"
#include "core/math/collision_solver.h"
#include "core/math/dynamic_mesh.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * PhysicsServerHyper
 * 
 * The master simulation server for the Scale-Aware pipeline.
 * Manages bit-perfect physical state and orchestrates Warp-style parallel sweeps.
 */
class PhysicsServerHyper : public Object {
	GDCLASS(PhysicsServerHyper, Object);

	static PhysicsServerHyper *singleton;

public:
	enum BodyMode {
		BODY_MODE_STATIC,
		BODY_MODE_KINEMATIC,
		BODY_MODE_DYNAMIC,
		BODY_MODE_DEFORMABLE // Supports Balloon, Flesh, and Fracture effects
	};

	struct ET_ALIGN_32 Body {
		RID self;
		BodyMode mode = BODY_MODE_STATIC;

		// --- Spatial State (Galactic Aware) ---
		Transform3Df transform;
		BigIntCore sector_x;
		BigIntCore sector_y;
		BigIntCore sector_z;

		// --- Kinematic Tensors (FixedMathCore) ---
		Vector3f linear_velocity;
		Vector3f angular_velocity;
		Vector3f constant_force;
		Vector3f constant_torque;

		// --- Physical Properties ---
		FixedMathCore mass;
		FixedMathCore inv_mass;
		FixedMathCore friction;
		FixedMathCore bounce;
		FixedMathCore gravity_scale;

		// --- Deformable/Material State (Balloon & Flesh) ---
		Ref<DynamicMesh> mesh_data;
		FixedMathCore structural_integrity;
		FixedMathCore fatigue_accum;
		FixedMathCore yield_strength;
		FixedMathCore thermal_state; // Temperature in Kelvin

		// --- Simulation Flags ---
		bool active = true;
		bool ccd_enabled = false;
		bool sleeping = false;

		_FORCE_INLINE_ Body() {
			mass = MathConstants<FixedMathCore>::one();
			inv_mass = MathConstants<FixedMathCore>::one();
			gravity_scale = MathConstants<FixedMathCore>::one();
			structural_integrity = MathConstants<FixedMathCore>::one();
		}
	};

	struct Joint {
		RID self;
		RID body_a;
		RID body_b;
		Transform3Df local_a;
		Transform3Df local_b;
		FixedMathCore breaking_threshold;
		bool active = true;
	};

private:
	// Contiguous paged memory for Body/Joint components (EnTT style)
	RID_Owner<Body> body_owner;
	RID_Owner<Joint> joint_owner;

	// Galactic Broadphase
	SpatialPartition<RID> broadphase;

	FixedMathCore global_gravity;
	Vector3f gravity_vector;

protected:
	static void _bind_methods();

public:
	static _FORCE_INLINE_ PhysicsServerHyper *get_singleton() { return singleton; }

	// ------------------------------------------------------------------------
	// Lifecycle API
	// ------------------------------------------------------------------------

	virtual RID body_create();
	virtual void body_set_mode(RID p_body, BodyMode p_mode);
	virtual void body_set_transform(RID p_body, const Transform3Df &p_transform);
	virtual void body_set_sector(RID p_body, const BigIntCore &p_x, const BigIntCore &p_y, const BigIntCore &p_z);

	// ------------------------------------------------------------------------
	// Dynamics & Material API
	// ------------------------------------------------------------------------

	virtual void body_set_mass(RID p_body, const FixedMathCore &p_mass);
	virtual void body_apply_impulse(RID p_body, const Vector3f &p_impulse, const Vector3f &p_position = Vector3f_ZERO);
	virtual void body_set_material_yield(RID p_body, const FixedMathCore &p_yield);
	virtual void body_set_mesh(RID p_body, const Ref<DynamicMesh> &p_mesh);

	// ------------------------------------------------------------------------
	// Warp-Style Parallel Sweep Orchestration
	// ------------------------------------------------------------------------

	/**
	 * execute_gravity_sweep()
	 * Wave 1: N-Body Gravitational resolution using BigInt masses.
	 */
	void execute_gravity_sweep(const FixedMathCore &p_delta);

	/**
	 * execute_integration_sweep()
	 * Wave 2: Lorentz-corrected velocity and position integration.
	 * Handles Balloon/Flesh elastic restoration forces.
	 */
	void execute_integration_sweep(const FixedMathCore &p_delta);

	/**
	 * execute_collision_resolution()
	 * Wave 3: Narrow-phase contact resolve and CCD sub-stepping.
	 */
	void execute_collision_resolution(const FixedMathCore &p_delta);

	/**
	 * execute_galactic_sync()
	 * Wave 4: Re-anchors origins to BigInt sectors and updates broadphase.
	 */
	void execute_galactic_sync();

	// ------------------------------------------------------------------------
	// Spatial Query API
	// ------------------------------------------------------------------------

	virtual bool ray_cast(const Vector3f &p_from, const Vector3f &p_to, CollisionSolverf::CollisionResult &r_hit, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz);

	PhysicsServerHyper();
	virtual ~PhysicsServerHyper();
};

#endif // PHYSICS_SERVER_HYPER_H

--- END OF FILE core/simulation/physics_server_hyper.h ---
