// File 122: modules/genesis/src/entities/particle_entity.h
// Particle entity for SPH, granular, or PBD particle systems.
// Stores a dynamic set of particles each with position, velocity, mass,
// radius, and an optional phase flag. Used by SPHSolver and particle emitters.

#ifndef GENESIS_ENTITIES_PARTICLE_ENTITY_H
#define GENESIS_ENTITIES_PARTICLE_ENTITY_H

#include "base_entity.h"
#include "../core/genesis_types.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"

namespace genesis {

class ParticleEntity : public BaseEntity {
	GDCLASS(ParticleEntity, BaseEntity);

public:
	struct Particle {
		Vector3 position;
		Vector3 velocity;
		real_t mass;
		real_t radius;
		int phase;             // 0 = fluid, 1 = boundary, 2 = granular
		Particle() : position(), velocity(), mass(1.0), radius(0.05), phase(0) {}
		Particle(const Vector3 &p, const Vector3 &v, real_t m, real_t r, int ph = 0) :
			position(p), velocity(v), mass(m), radius(r), phase(ph) {}
	};

	ParticleEntity() : BaseEntity() {
		solver_type = SolverType::SPH;
	}

	// Add a single particle.
	void add_particle(const Vector3 &p_pos, const Vector3 &p_vel, real_t p_mass = 1.0,
					  real_t p_radius = 0.05, int p_phase = 0) {
		particles.push_back(Particle(p_pos, p_vel, p_mass, p_radius, p_phase));
	}

	// Remove all particles.
	void clear_particles() { particles.clear(); }

	// Access the underlying array.
	LocalVector<Particle> &get_particles() { return particles; }
	const LocalVector<Particle> &get_particles() const { return particles; }

	int particle_count() const { return particles.size(); }

	// Total mass of all particles.
	virtual real_t get_mass() const override {
		real_t total = 0.0;
		for (const Particle &p : particles) total += p.mass;
		return total;
	}

	// AABB encompassing all particles.
	virtual AABB get_aabb() const override {
		if (particles.is_empty()) return AABB(transform.origin, Vector3());
		Vector3 minv(INFINITY, INFINITY, INFINITY);
		Vector3 maxv(-INFINITY, -INFINITY, -INFINITY);
		for (const Particle &p : particles) {
			minv = minv.min(p.position);
			maxv = maxv.max(p.position);
		}
		return AABB(minv, maxv - minv);
	}

	// Override: particles do not have rigid inertia.
	virtual real_t get_inertia_scalar() const override { return 0.0; }

	// No standard integration; solver handles it.
	virtual void integrate_velocity(real_t dt) override {}
	virtual void integrate_position(real_t dt) override {}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("add_particle", "pos", "vel", "mass", "radius", "phase"),
				&ParticleEntity::add_particle, DEFVAL(1.0), DEFVAL(0.05), DEFVAL(0));
		ClassDB::bind_method(D_METHOD("clear_particles"), &ParticleEntity::clear_particles);
		ClassDB::bind_method(D_METHOD("particle_count"), &ParticleEntity::particle_count);
	}

private:
	LocalVector<Particle> particles;
};

} // namespace genesis

#endif // GENESIS_ENTITIES_PARTICLE_ENTITY_H