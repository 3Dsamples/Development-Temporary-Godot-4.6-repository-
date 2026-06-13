// File 167: modules/genesis/src/entities/emitter_entity.cpp
// Emitter entity methods: particle spawning logic for all emitter shapes.
// Generates random positions/velocities and adds them to a ParticleEntity.

#include "emitter_entity.h"

#include "particle_entity.h"
#include "../core/genesis_types.h"
#include "core/math/vector3.h"
#include "core/math/random_number_generator.h"
#include "core/typedefs.h"

namespace genesis {

void EmitterEntity::emit(Ref<ParticleEntity> p_target, real_t dt) {
	ERR_FAIL_COND(p_target.is_null());
	if (!emitting || rate <= 0.0f) return;

	time_accum += dt;
	int to_spawn = int(time_accum * rate);
	time_accum -= real_t(to_spawn) / rate;

	RandomNumberGenerator rng;
	rng.randomize();
	Transform3D xform = get_transform();

	for (int i = 0; i < to_spawn; ++i) {
		// Generate random position inside the emitter volume in local space.
		Vector3 local_pos = random_local_position(rng);
		// Transform to world space.
		Vector3 world_pos = xform.xform(local_pos);

		// Target velocity: base direction plus random spread.
		Vector3 vel = velocity + random_unit_sphere(rng) * vel_spread;

		p_target->add_particle(world_pos, vel, mass, radius, 0);
	}
}

Vector3 EmitterEntity::random_local_position(RandomNumberGenerator &rng) const {
	switch (shape) {
		case POINT:
			return Vector3();
		case BOX: {
			Vector3 he = box_extents;
			return Vector3(
				rng.randf_range(-he.x, he.x),
				rng.randf_range(-he.y, he.y),
				rng.randf_range(-he.z, he.z));
		}
		case SPHERE: {
			// Uniform inside sphere (rejection method).
			while (true) {
				Vector3 p(rng.randf_range(-1.0f, 1.0f),
						  rng.randf_range(-1.0f, 1.0f),
						  rng.randf_range(-1.0f, 1.0f));
				if (p.length_squared() <= 1.0f)
					return p * sphere_radius;
			}
		}
		case CYLINDER: {
			real_t angle = rng.randf_range(0.0f, Math_TAU);
			real_t r = cylinder_radius * Math::sqrt(rng.randf());
			real_t h = rng.randf_range(-cylinder_height * 0.5f, cylinder_height * 0.5f);
			return Vector3(r * Math::cos(angle), h, r * Math::sin(angle));
		}
		default:
			return Vector3();
	}
}

Vector3 EmitterEntity::random_unit_sphere(RandomNumberGenerator &rng) const {
	// Uniformly random direction using spherical coordinates.
	real_t theta = rng.randf_range(0.0f, Math_TAU);
	real_t phi = Math::acos(1.0f - 2.0f * rng.randf());
	return Vector3(
		Math::sin(phi) * Math::cos(theta),
		Math::sin(phi) * Math::sin(theta),
		Math::cos(phi));
}

} // namespace genesis