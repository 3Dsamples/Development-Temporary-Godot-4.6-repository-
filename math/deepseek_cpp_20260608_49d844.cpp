// File 123: modules/genesis/src/entities/emitter_entity.h
// Emitter entity: continuously spawns particles for SPH, MPM, or PBD
// solvers. Parameters define the emission region, rate, velocity, and
// per‑particle properties.

#ifndef GENESIS_ENTITIES_EMITTER_ENTITY_H
#define GENESIS_ENTITIES_EMITTER_ENTITY_H

#include "base_entity.h"
#include "particle_entity.h"
#include "../core/genesis_types.h"
#include "core/math/vector3.h"
#include "core/math/random_number_generator.h"

namespace genesis {

class EmitterEntity : public BaseEntity {
	GDCLASS(EmitterEntity, BaseEntity);

public:
	enum EmitterShape {
		POINT,
		BOX,
		SPHERE,
		CYLINDER
	};

	EmitterEntity() : BaseEntity() {
		solver_type = SolverType::SPH; // can be reassigned
	}

	// --- Emission shape ---
	void set_shape(EmitterShape p_shape) { shape = p_shape; }
	EmitterShape get_shape() const { return shape; }

	// --- Emission volume ---
	void set_box_extents(const Vector3 &p_ext) { box_extents = p_ext.abs(); }
	Vector3 get_box_extents() const { return box_extents; }

	void set_sphere_radius(real_t p_r) { sphere_radius = MAX(p_r, 0.0); }
	real_t get_sphere_radius() const { return sphere_radius; }

	void set_cylinder_height(real_t p_h) { cylinder_height = MAX(p_h, 0.0); }
	real_t get_cylinder_height() const { return cylinder_height; }
	void set_cylinder_radius(real_t p_r) { cylinder_radius = MAX(p_r, 0.0); }
	real_t get_cylinder_radius() const { return cylinder_radius; }

	// --- Emission rate ---
	void set_particles_per_second(real_t p_rate) { rate = MAX(p_rate, 0.0); }
	real_t get_particles_per_second() const { return rate; }

	void set_particle_mass(real_t p_m) { mass = MAX(p_m, 0.0); }
	real_t get_particle_mass() const { return mass; }

	void set_particle_radius(real_t p_r) { radius = MAX(p_r, 1e-6); }
	real_t get_particle_radius() const { return radius; }

	void set_particle_velocity(const Vector3 &p_vel) { velocity = p_vel; }
	Vector3 get_particle_velocity() const { return velocity; }

	void set_random_velocity_spread(real_t p_spread) { vel_spread = MAX(p_spread, 0.0); }
	real_t get_random_velocity_spread() const { return vel_spread; }

	// --- Enable / disable emission ---
	void set_emitting(bool p_emit) { emitting = p_emit; }
	bool is_emitting() const { return emitting; }

	// --- Emit particles into a target ParticleEntity ---
	void emit(Ref<ParticleEntity> p_target, real_t dt) {
		ERR_FAIL_COND(p_target.is_null());
		if (!emitting || rate <= 0) return;

		time_accum += dt;
		int to_spawn = int(time_accum * rate);
		time_accum -= to_spawn / rate;

		for (int i = 0; i < to_spawn; ++i) {
			Vector3 pos = random_position();
			Vector3 vel = velocity + random_unit_vector() * vel_spread;
			p_target->add_particle(pos, vel, mass, radius);
		}
	}

	// Reset accumulated time.
	void reset_timer() { time_accum = 0.0; }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_shape", "shape"), &EmitterEntity::set_shape);
		ClassDB::bind_method(D_METHOD("get_shape"), &EmitterEntity::get_shape);
		ClassDB::bind_method(D_METHOD("set_box_extents", "extents"), &EmitterEntity::set_box_extents);
		ClassDB::bind_method(D_METHOD("get_box_extents"), &EmitterEntity::get_box_extents);
		ClassDB::bind_method(D_METHOD("set_sphere_radius", "r"), &EmitterEntity::set_sphere_radius);
		ClassDB::bind_method(D_METHOD("get_sphere_radius"), &EmitterEntity::get_sphere_radius);
		ClassDB::bind_method(D_METHOD("set_cylinder_height", "h"), &EmitterEntity::set_cylinder_height);
		ClassDB::bind_method(D_METHOD("get_cylinder_height"), &EmitterEntity::get_cylinder_height);
		ClassDB::bind_method(D_METHOD("set_cylinder_radius", "r"), &EmitterEntity::set_cylinder_radius);
		ClassDB::bind_method(D_METHOD("get_cylinder_radius"), &EmitterEntity::get_cylinder_radius);
		ClassDB::bind_method(D_METHOD("set_particles_per_second", "rate"), &EmitterEntity::set_particles_per_second);
		ClassDB::bind_method(D_METHOD("get_particles_per_second"), &EmitterEntity::get_particles_per_second);
		ClassDB::bind_method(D_METHOD("set_particle_mass", "mass"), &EmitterEntity::set_particle_mass);
		ClassDB::bind_method(D_METHOD("get_particle_mass"), &EmitterEntity::get_particle_mass);
		ClassDB::bind_method(D_METHOD("set_particle_radius", "r"), &EmitterEntity::set_particle_radius);
		ClassDB::bind_method(D_METHOD("get_particle_radius"), &EmitterEntity::get_particle_radius);
		ClassDB::bind_method(D_METHOD("set_particle_velocity", "vel"), &EmitterEntity::set_particle_velocity);
		ClassDB::bind_method(D_METHOD("get_particle_velocity"), &EmitterEntity::get_particle_velocity);
		ClassDB::bind_method(D_METHOD("set_random_velocity_spread", "spread"), &EmitterEntity::set_random_velocity_spread);
		ClassDB::bind_method(D_METHOD("get_random_velocity_spread"), &EmitterEntity::get_random_velocity_spread);
		ClassDB::bind_method(D_METHOD("set_emitting", "emit"), &EmitterEntity::set_emitting);
		ClassDB::bind_method(D_METHOD("is_emitting"), &EmitterEntity::is_emitting);
		ClassDB::bind_method(D_METHOD("emit", "target", "dt"), &EmitterEntity::emit);
		ADD_PROPERTY(PropertyInfo(Variant::INT, "shape", PROPERTY_HINT_ENUM, "Point,Box,Sphere,Cylinder"), "set_shape", "get_shape");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "box_extents"), "set_box_extents", "get_box_extents");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sphere_radius"), "set_sphere_radius", "get_sphere_radius");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cylinder_height"), "set_cylinder_height", "get_cylinder_height");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cylinder_radius"), "set_cylinder_radius", "get_cylinder_radius");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rate"), "set_particles_per_second", "get_particles_per_second");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_particle_mass", "get_particle_mass");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_particle_radius", "get_particle_radius");
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "velocity"), "set_particle_velocity", "get_particle_velocity");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "vel_spread"), "set_random_velocity_spread", "get_random_velocity_spread");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "emitting"), "set_emitting", "is_emitting");
	}

private:
	Vector3 random_position() const {
		// Generate a point uniformly inside the emitter volume in local space, then transform to world.
		Vector3 local_pos;
		RandomNumberGenerator rng;
		rng.randomize();
		switch (shape) {
			case POINT: local_pos = Vector3(); break;
			case BOX:
				local_pos = Vector3(rng.randf_range(-box_extents.x, box_extents.x),
								   rng.randf_range(-box_extents.y, box_extents.y),
								   rng.randf_range(-box_extents.z, box_extents.z));
				break;
			case SPHERE:
				// Rejection method
				while (1) {
					Vector3 p(rng.randf_range(-1,1), rng.randf_range(-1,1), rng.randf_range(-1,1));
					if (p.length_squared() <= 1.0) { local_pos = p * sphere_radius; break; }
				}
				break;
			case CYLINDER:
				real_t angle = rng.randf_range(0, Math_TAU);
				real_t r = cylinder_radius * Math::sqrt(rng.randf());
				local_pos = Vector3(r * Math::cos(angle), rng.randf_range(-cylinder_height * 0.5, cylinder_height * 0.5), r * Math::sin(angle));
				break;
		}
		return get_transform().xform(local_pos);
	}

	Vector3 random_unit_vector() const {
		RandomNumberGenerator rng;
		rng.randomize();
		real_t theta = rng.randf_range(0, Math_TAU);
		real_t phi = Math::acos(1.0 - 2.0 * rng.randf());
		return Vector3(Math::sin(phi) * Math::cos(theta), Math::sin(phi) * Math::sin(theta), Math::cos(phi));
	}

	EmitterShape shape = POINT;
	Vector3 box_extents = Vector3(0.5, 0.5, 0.5);
	real_t sphere_radius = 0.5;
	real_t cylinder_height = 1.0;
	real_t cylinder_radius = 0.5;
	real_t rate = 100.0;
	real_t mass = 1.0;
	real_t radius = 0.05;
	Vector3 velocity = Vector3(0, -1, 0);
	real_t vel_spread = 0.0;
	bool emitting = true;
	real_t time_accum = 0.0;
};

} // namespace genesis

#endif // GENESIS_ENTITIES_EMITTER_ENTITY_H