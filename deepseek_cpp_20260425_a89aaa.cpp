// File 11: modules/gaia/src/framework/body.h

#ifndef GAIA_FRAMEWORK_BODY_H
#define GAIA_FRAMEWORK_BODY_H

#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"

namespace gaia {

// ---------------------------------------------------------------------------
// RigidBody – a simple rigid body with velocity-level dynamics.
// ---------------------------------------------------------------------------
class RigidBody {
public:
	enum BodyType {
		STATIC,
		DYNAMIC,
		KINEMATIC
	};

	RigidBody() :
		position(0, 0, 0),
		rotation(Basis()),
		linear_velocity(0, 0, 0),
		angular_velocity(0, 0, 0),
		force_accum(0, 0, 0),
		torque_accum(0, 0, 0),
		mass(1.0),
		inverse_mass(1.0),
		inertia_local(Basis().scaled(Vector3(1, 1, 1))),
		inverse_inertia_local(Basis().scaled(Vector3(1, 1, 1))),
		inverse_inertia_world(Basis().scaled(Vector3(1, 1, 1))),
		type(DYNAMIC),
		enable_gravity(true),
		linear_damping(0.0),
		angular_damping(0.0),
		handle(0) {}

	// --- Setters/getters ---
	void set_position(const Vector3 &p) { position = p; }
	const Vector3 &get_position() const { return position; }

	void set_rotation(const Basis &r) { rotation = r; }
	const Basis &get_rotation() const { return rotation; }

	void set_transform(const Transform3D &t) {
		position = t.origin;
		rotation = t.basis;
	}
	Transform3D get_transform() const { return Transform3D(rotation, position); }

	void set_linear_velocity(const Vector3 &v) { linear_velocity = v; }
	Vector3 get_linear_velocity() const { return linear_velocity; }

	void set_angular_velocity(const Vector3 &w) { angular_velocity = w; }
	Vector3 get_angular_velocity() const { return angular_velocity; }

	void set_mass(real_t m) {
		mass = MAX(m, 0.0);
		inverse_mass = (mass > 0.0 && type == DYNAMIC) ? 1.0 / mass : 0.0;
	}
	real_t get_mass() const { return mass; }
	real_t get_inverse_mass() const { return inverse_mass; }

	void set_inertia(const Basis &inertia_local) {
		inertia_local = inertia_local;
		// ensure invertibility
		Vector3 diag = inertia_local.get_scale();
		inverse_inertia_local = Basis().scaled(Vector3(
			(diag.x > 0.0) ? 1.0 / diag.x : 0.0,
			(diag.y > 0.0) ? 1.0 / diag.y : 0.0,
			(diag.z > 0.0) ? 1.0 / diag.z : 0.0
		));
		compute_world_inverse_inertia();
	}
	Basis get_inertia_local() const { return inertia_local; }
	Basis get_inverse_inertia_world() const { return inverse_inertia_world; }

	void set_type(BodyType t) {
		type = t;
		if (type != DYNAMIC) {
			inverse_mass = 0.0;
			inverse_inertia_local = Basis();
			inverse_inertia_world = Basis();
		} else {
			// recalc mass if needed
			set_mass(mass);
			set_inertia(inertia_local);
		}
	}
	BodyType get_type() const { return type; }

	void set_gravity_enabled(bool b) { enable_gravity = b; }
	bool is_gravity_enabled() const { return enable_gravity; }

	void set_linear_damping(real_t d) { linear_damping = CLAMP(d, 0.0, 1.0); }
	real_t get_linear_damping() const { return linear_damping; }

	void set_angular_damping(real_t d) { angular_damping = CLAMP(d, 0.0, 1.0); }
	real_t get_angular_damping() const { return angular_damping; }

	void set_handle(uint32_t h) { handle = h; }
	uint32_t get_handle() const { return handle; }

	// --- Force accumulation ---
	void apply_force(const Vector3 &force, const Vector3 &world_point = Vector3()) {
		force_accum += force;
		if (world_point.length_squared() > 0.0) {
			Vector3 r = world_point - position;
			torque_accum += r.cross(force);
		}
	}

	void apply_impulse(const Vector3 &impulse, const Vector3 &world_point = Vector3()) {
		if (inverse_mass > 0.0) {
			linear_velocity += impulse * inverse_mass;
		}
		if (!world_point.is_zero_approx()) {
			Vector3 r = world_point - position;
			angular_velocity += inverse_inertia_world.xform(r.cross(impulse));
		}
	}

	void clear_forces() {
		force_accum = Vector3(0, 0, 0);
		torque_accum = Vector3(0, 0, 0);
	}

	// --- Integration steps (to be called by World) ---
	void integrate_velocity(real_t dt) {
		if (type != DYNAMIC) return;

		linear_velocity += force_accum * (inverse_mass * dt);
		angular_velocity += inverse_inertia_world.xform(torque_accum * dt);
		// Apply damping
		linear_velocity *= 1.0 - linear_damping * dt;
		angular_velocity *= 1.0 - angular_damping * dt;
		clear_forces();
	}

	void integrate_position(real_t dt) {
		if (type != DYNAMIC) return;

		position += linear_velocity * dt;
		// Update rotation from angular velocity
		real_t angle = angular_velocity.length();
		if (angle > CMP_EPSILON) {
			Vector3 axis = angular_velocity / angle;
			Basis rot(axis, angle * dt);
			rotation = rot * rotation;
			rotation.orthonormalize();
		}
		compute_world_inverse_inertia();
	}

private:
	void compute_world_inverse_inertia() {
		Basis rot = rotation;

		inverse_inertia_world = rot * inverse_inertia_local * rot.transposed();
	}

	Vector3 position;
	Basis rotation;
	Vector3 linear_velocity;
	Vector3 angular_velocity;
	Vector3 force_accum;
	Vector3 torque_accum;
	real_t mass;
	real_t inverse_mass;
	Basis inertia_local;
	Basis inverse_inertia_local;
	Basis inverse_inertia_world;
	BodyType type;
	bool enable_gravity;
	real_t linear_damping;
	real_t angular_damping;
	uint32_t handle;
};

// ---------------------------------------------------------------------------
// SoftBody – a container for a tetrahedral/triangle mesh for deformable
// simulation. Actual solver logic is in PBD/VBD modules.
// ---------------------------------------------------------------------------
class SoftBody {
public:
	SoftBody() : handle(0), total_mass(1.0) {}

	// Mesh data (to be filled by modules/gaia/src/mesh/*)
	LocalVector<Vector3> rest_positions;   // undeformed positions
	LocalVector<Vector3> positions;        // current world positions
	LocalVector<Vector3> velocities;       // per-vertex velocity

	// Tetrahedral elements (indices into positions)
	LocalVector<uint32_t> tetrahedra;      // 4 indices per tet

	// Constraints (distance bending etc.) are owned by the solver, not here.
	void set_handle(uint32_t h) { handle = h; }
	uint32_t get_handle() const { return handle; }

	void resize(int vertex_count) {
		rest_positions.resize(vertex_count);
		positions.resize(vertex_count);
		velocities.resize(vertex_count);
		for (int i = 0; i < vertex_count; ++i) velocities[i] = Vector3();
	}

	void set_rest_position(int idx, const Vector3 &pos) {
		ERR_FAIL_INDEX(idx, rest_positions.size());
		rest_positions[idx] = pos;
		// positions default to rest unless set otherwise
		if (positions[idx].is_zero_approx()) positions[idx] = pos;
	}

	void set_position(int idx, const Vector3 &pos) {
		ERR_FAIL_INDEX(idx, positions.size());
		positions[idx] = pos;
	}

	Vector3 get_position(int idx) const {
		ERR_FAIL_INDEX_V(idx, positions.size(), Vector3());
		return positions[idx];
	}

	Vector3 get_velocity(int idx) const {
		ERR_FAIL_INDEX_V(idx, velocities.size(), Vector3());
		return velocities[idx];
	}

	void set_velocity(int idx, const Vector3 &vel) {
		ERR_FAIL_INDEX(idx, velocities.size());
		velocities[idx] = vel;
	}

	void set_total_mass(real_t m) { total_mass = MAX(m, 0.0); }
	real_t get_total_mass() const { return total_mass; }

	// Can be later extended with per-vertex masses.

private:
	uint32_t handle;
	real_t total_mass;
};

} // namespace gaia

#endif // GAIA_FRAMEWORK_BODY_H