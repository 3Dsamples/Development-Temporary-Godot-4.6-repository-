// File 162: modules/genesis/src/entities/particle_entity.cpp
// Implements ParticleEntity – a collection of particles for SPH / granular
// simulations. Provides add, clear, AABB access and mass computation.

#include "particle_entity.h"

#include "core/math/aabb.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace genesis {

ParticleEntity::ParticleEntity() {
	solver_type = SolverType::SPH;
}

void ParticleEntity::add_particle(const Vector3 &p_pos, const Vector3 &p_vel,
								  real_t p_mass, real_t p_radius, int p_phase) {
	Particle p;
	p.position = p_pos;
	p.velocity = p_vel;
	p.mass = MAX(p_mass, 0.0);
	p.radius = MAX(p_radius, 1e-6);
	p.phase = p_phase;
	particles.push_back(p);
}

void ParticleEntity::clear_particles() {
	particles.clear();
}

LocalVector<ParticleEntity::Particle> &ParticleEntity::get_particles() {
	return particles;
}

const LocalVector<ParticleEntity::Particle> &ParticleEntity::get_particles() const {
	return particles;
}

int ParticleEntity::particle_count() const {
	return particles.size();
}

real_t ParticleEntity::get_mass() const {
	real_t total = 0.0;
	for (const Particle &p : particles) total += p.mass;
	return total;
}

AABB ParticleEntity::get_aabb() const {
	if (particles.is_empty()) return AABB(transform.origin, Vector3());
	Vector3 minv(INFINITY, INFINITY, INFINITY);
	Vector3 maxv(-INFINITY, -INFINITY, -INFINITY);
	for (const Particle &p : particles) {
		minv = minv.min(p.position);
		maxv = maxv.max(p.position);
	}
	return AABB(minv, maxv - minv);
}

real_t ParticleEntity::get_inertia_scalar() const {
	return 0.0;  // particles have no rigid inertia
}

void ParticleEntity::integrate_velocity(real_t dt) {
	// Solver handles integration directly
}

void ParticleEntity::integrate_position(real_t dt) {
	// Solver handles integration directly
}

} // namespace genesis