// File 82: modules/genesis/src/states/entity_state.h
// Entity state – stores the full kinematic and material state for saving/loading and
// differentiable physics checkpoints.

#ifndef GENESIS_STATES_ENTITY_STATE_H
#define GENESIS_STATES_ENTITY_STATE_H

#include "core/io/resource.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "../entities/base_entity.h"
#include "../core/genesis_types.h"

namespace genesis {

/**
 * Holds a snapshot of an entity's state (position, rotation, velocity,
 * internal forces, deformation gradients for FEM/MPM, etc.).
 * Used for initialisation, saving checkpoints, and gradient-based
 * optimisation.
 */
class EntityState : public Resource {
	GDCLASS(EntityState, Resource);

public:
	EntityState() :
		time(0.0),
		position(Vector3()),
		rotation(Basis()),
		linear_velocity(Vector3()),
		angular_velocity(Vector3()) {}

	// --- Basic kinematic state ---
	void set_time(real_t p_t) { time = p_t; }
	real_t get_time() const { return time; }

	void set_position(const Vector3 &p_pos) { position = p_pos; }
	Vector3 get_position() const { return position; }

	void set_rotation(const Basis &p_basis) { rotation = p_basis; }
	Basis get_rotation() const { return rotation; }

	void set_linear_velocity(const Vector3 &p_vel) { linear_velocity = p_vel; }
	Vector3 get_linear_velocity() const { return linear_velocity; }

	void set_angular_velocity(const Vector3 &p_vel) { angular_velocity = p_vel; }
	Vector3 get_angular_velocity() const { return angular_velocity; }

	// --- Optional per‑vertex / per‑particle state for deformable bodies ---
	void set_vertex_positions(const LocalVector<Vector3> &p_positions) {
		vertex_positions = p_positions;
	}
	const LocalVector<Vector3> &get_vertex_positions() const { return vertex_positions; }

	void set_vertex_velocities(const LocalVector<Vector3> &p_velocities) {
		vertex_velocities = p_velocities;
	}
	const LocalVector<Vector3> &get_vertex_velocities() const { return vertex_velocities; }

	// --- For MPM / FEM: per‑element deformation gradients and plastic strain ---
	void set_deformation_gradients(const LocalVector<Basis> &p_F) { deformation_gradients = p_F; }
	const LocalVector<Basis> &get_deformation_gradients() const { return deformation_gradients; }

	void set_plastic_strains(const LocalVector<real_t> &p_eps) { plastic_strains = p_eps; }
	const LocalVector<real_t> &get_plastic_strains() const { return plastic_strains; }

	// --- Damage fields (MPM) ---
	void set_damages(const LocalVector<real_t> &p_d) { damages = p_d; }
	const LocalVector<real_t> &get_damages() const { return damages; }

	// --- Populate from a BaseEntity (virtual – overridden per entity type) ---
	virtual void capture_from(const Ref<BaseEntity> &p_entity) {
		ERR_FAIL_COND(p_entity.is_null());
		time = 0.0; // caller should set time
		position = p_entity->get_position();
		rotation = p_entity->get_rotation();
		linear_velocity = p_entity->get_linear_velocity();
		angular_velocity = p_entity->get_angular_velocity();
		// Derived classes will add vertex / F / damage data.
	}

	// --- Apply state to an entity ---
	virtual void apply_to(const Ref<BaseEntity> &p_entity) const {
		ERR_FAIL_COND(p_entity.is_null());
		p_entity->set_position(position);
		p_entity->set_rotation(rotation);
		p_entity->set_linear_velocity(linear_velocity);
		p_entity->set_angular_velocity(angular_velocity);
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_time", "t"), &EntityState::set_time);
		ClassDB::bind_method(D_METHOD("get_time"), &EntityState::get_time);
		ClassDB::bind_method(D_METHOD("set_position", "position"), &EntityState::set_position);
		ClassDB::bind_method(D_METHOD("get_position"), &EntityState::get_position);
		ClassDB::bind_method(D_METHOD("set_rotation", "rotation"), &EntityState::set_rotation);
		ClassDB::bind_method(D_METHOD("get_rotation"), &EntityState::get_rotation);
		ClassDB::bind_method(D_METHOD("set_linear_velocity", "velocity"), &EntityState::set_linear_velocity);
		ClassDB::bind_method(D_METHOD("get_linear_velocity"), &EntityState::get_linear_velocity);
		ClassDB::bind_method(D_METHOD("set_angular_velocity", "velocity"), &EntityState::set_angular_velocity);
		ClassDB::bind_method(D_METHOD("get_angular_velocity"), &EntityState::get_angular_velocity);
		ClassDB::bind_method(D_METHOD("set_vertex_positions", "positions"), &EntityState::set_vertex_positions);
		ClassDB::bind_method(D_METHOD("get_vertex_positions"), &EntityState::get_vertex_positions);
		ClassDB::bind_method(D_METHOD("set_vertex_velocities", "velocities"), &EntityState::set_vertex_velocities);
		ClassDB::bind_method(D_METHOD("get_vertex_velocities"), &EntityState::get_vertex_velocities);
		ClassDB::bind_method(D_METHOD("set_deformation_gradients", "F_list"), &EntityState::set_deformation_gradients);
		ClassDB::bind_method(D_METHOD("get_deformation_gradients"), &EntityState::get_deformation_gradients);
		ClassDB::bind_method(D_METHOD("set_plastic_strains", "eps_list"), &EntityState::set_plastic_strains);
		ClassDB::bind_method(D_METHOD("get_plastic_strains"), &EntityState::get_plastic_strains);
		ClassDB::bind_method(D_METHOD("set_damages", "damages"), &EntityState::set_damages);
		ClassDB::bind_method(D_METHOD("get_damages"), &EntityState::get_damages);
		ClassDB::bind_method(D_METHOD("capture_from", "entity"), &EntityState::capture_from);
		ClassDB::bind_method(D_METHOD("apply_to", "entity"), &EntityState::apply_to);

		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time"), "set_time", "get_time");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "position"), "set_position", "get_position");
		ADD_PROPERTY(PropertyInfo(Variant::BASIS, "rotation"), "set_rotation", "get_rotation");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	}

private:
	real_t time;
	Vector3 position;
	Basis rotation;
	Vector3 linear_velocity;
	Vector3 angular_velocity;
	LocalVector<Vector3> vertex_positions;
	LocalVector<Vector3> vertex_velocities;
	LocalVector<Basis> deformation_gradients;
	LocalVector<real_t> plastic_strains;
	LocalVector<real_t> damages;
};

} // namespace genesis

#endif // GENESIS_STATES_ENTITY_STATE_H