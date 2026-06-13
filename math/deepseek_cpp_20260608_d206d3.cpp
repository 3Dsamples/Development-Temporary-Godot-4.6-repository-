// File 291: modules/vienna/src/particles/vienna_particle.h
// ViennaParticle – a single particle used by the particle system.
// Stores position, velocity, mass, radius, life, and colour.

#ifndef VIENNA_PARTICLES_PARTICLE_H
#define VIENNA_PARTICLES_PARTICLE_H

#include "core/math/vector3.h"
#include "core/math/color.h"
#include "../core/vienna_types.h"

namespace vienna {

class ViennaParticle {
public:
	vec3 position;
	vec3 velocity;
	real_t mass;
	real_t radius;
	real_t life;          // remaining seconds, <=0 means dead
	real_t max_life;
	Color color;
	bool active;

	ViennaParticle() :
		position(), velocity(), mass(1.0), radius(0.1),
		life(0.0), max_life(1.0), color(1,1,1,1), active(false) {}

	void init(const vec3 &p_pos, const vec3 &p_vel, real_t p_mass, real_t p_radius, real_t p_life, const Color &p_color = Color(1,1,1,1)) {
		position = p_pos;
		velocity = p_vel;
		mass = p_mass;
		radius = p_radius;
		life = p_life;
		max_life = p_life;
		color = p_color;
		active = true;
	}

	void kill() {
		life = 0.0;
		active = false;
	}
};

} // namespace vienna

#endif // VIENNA_PARTICLES_PARTICLE_H