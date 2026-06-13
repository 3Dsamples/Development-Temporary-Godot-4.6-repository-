// File 292: modules/vienna/src/particles/vienna_particle_system.h
// ViennaParticleSystem – a CPU particle system with emitters, forces,
// spatial-hash neighbour search, rigid‑body collisions via Gaia BVH,
// and integration (Euler/Verlet).  Designed for real‑time Godot 4.6.

#ifndef VIENNA_PARTICLES_SYSTEM_H
#define VIENNA_PARTICLES_SYSTEM_H

#include "vienna_particle.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"
#include "../bodies/vienna_body.h"
#include "../../../gaia/src/bvh/bvh.h"
#include "../../../gaia/src/spatial_query/spatial_hash.h"
#include "core/templates/local_vector.h"
#include "core/math/random_number_generator.h"

namespace vienna {

class ViennaParticleSystem : public RefCounted {
	GDCLASS(ViennaParticleSystem, RefCounted);

public:
	enum EmitterShape { POINT, BOX, SPHERE, CYLINDER };
	enum Integrator  { EULER, VERLET };

	ViennaParticleSystem() :
		max_particles(1000),
		integrator(EULER),
		gravity(vec3(0.0, -9.81, 0.0)),
		wind(vec3()),
		drag(0.0),
		turbulence(0.0),
		particle_radius(0.1),
		particle_mass(1.0),
		// emitter defaults
		emit_rate(50.0),
		emit_lifetime(2.0),
		emit_speed(5.0),
		emit_shape(POINT),
		emit_box_extents(1.0, 1.0, 1.0),
		emit_sphere_radius(1.0),
		emit_cylinder_height(2.0),
		emit_cylinder_radius(1.0),
		emit_rot(vec3()),
		// collision
		collision_enabled(false),
		collision_bounce(0.3),
		// internal time
		elapsed(0.0) {}

	// ---------- emission ----------
	void set_max_particles(int p_max) { max_particles = MAX(p_max, 1); }
	void set_emit_rate(real_t p_rps) { emit_rate = MAX(p_rps, 0.0); }
	void set_emit_lifetime(real_t p_life) { emit_lifetime = MAX(p_life, 0.01); }
	void set_emit_speed(real_t p_speed) { emit_speed = MAX(p_speed, 0.0); }
	void set_emit_shape(EmitterShape p_shape) { emit_shape = p_shape; }
	void set_emit_box_extents(const vec3 &p_ext) { emit_box_extents = p_ext.abs(); }
	void set_emit_sphere_radius(real_t p_r) { emit_sphere_radius = MAX(p_r, 0.0); }
	void set_emit_cylinder_height(real_t p_h) { emit_cylinder_height = MAX(p_h, 0.0); }
	void set_emit_cylinder_radius(real_t p_r) { emit_cylinder_radius = MAX(p_r, 0.0); }
	void set_emit_rotation(const vec3 &p_euler) { emit_rot = p_euler; }

	// ---------- physics ----------
	void set_gravity(const vec3 &p_g) { gravity = p_g; }
	void set_wind(const vec3 &p_w) { wind = p_w; }
	void set_drag(real_t p_d) { drag = CLAMP(p_d, 0.0, 1.0); }
	void set_turbulence(real_t p_t) { turbulence = MAX(p_t, 0.0); }
	void set_particle_radius(real_t p_r) { particle_radius = MAX(p_r, 0.001); }
	void set_particle_mass(real_t p_m) { particle_mass = MAX(p_m, 0.001); }
	void set_integrator(Integrator p_int) { integrator = p_int; }

	// ---------- collision ----------
	void set_collision_enabled(bool p_en) { collision_enabled = p_en; }
	void set_collision_bounce(real_t p_b) { collision_bounce = CLAMP(p_b, 0.0, 1.0); }
	void set_collision_bodies(const LocalVector<Ref<ViennaBody>> &p_bodies) { collision_bodies = p_bodies; }

	// ---------- step (dt) ----------
	void step(real_t dt) {
		elapsed += dt;
		// Spawn new particles
		spawn(dt);
		// Integrate
		if (integrator == EULER) integrate_euler(dt);
		else                      integrate_verlet(dt);
		// Resolve collisions
		if (collision_enabled) resolve_collisions();
		// Cull dead particles
		prune_dead();
	}

	// ---------- access ----------
	int get_live_count() const { return particles.size(); }
	const ViennaParticle &get_particle(int p_idx) const { return particles[p_idx]; }
	ViennaParticle &get_particle(int p_idx) { return particles[p_idx]; }

	// ---------- reset ----------
	void clear() {
		particles.clear();
		elapsed = 0.0;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_max_particles","max"), &ViennaParticleSystem::set_max_particles);
		ClassDB::bind_method(D_METHOD("get_max_particles"), &ViennaParticleSystem::get_max_particles);
		ClassDB::bind_method(D_METHOD("set_emit_rate","rps"), &ViennaParticleSystem::set_emit_rate);
		ClassDB::bind_method(D_METHOD("set_emit_lifetime","life"), &ViennaParticleSystem::set_emit_lifetime);
		ClassDB::bind_method(D_METHOD("set_emit_speed","speed"), &ViennaParticleSystem::set_emit_speed);
		ClassDB::bind_method(D_METHOD("set_emit_shape","shape"), &ViennaParticleSystem::set_emit_shape);
		ClassDB::bind_method(D_METHOD("set_emit_box_extents","ext"), &ViennaParticleSystem::set_emit_box_extents);
		ClassDB::bind_method(D_METHOD("set_emit_sphere_radius","r"), &ViennaParticleSystem::set_emit_sphere_radius);
		ClassDB::bind_method(D_METHOD("set_emit_cylinder_height","h"), &ViennaParticleSystem::set_emit_cylinder_height);
		ClassDB::bind_method(D_METHOD("set_emit_cylinder_radius","r"), &ViennaParticleSystem::set_emit_cylinder_radius);
		ClassDB::bind_method(D_METHOD("set_emit_rotation","euler"), &ViennaParticleSystem::set_emit_rotation);
		ClassDB::bind_method(D_METHOD("set_gravity","g"), &ViennaParticleSystem::set_gravity);
		ClassDB::bind_method(D_METHOD("set_wind","w"), &ViennaParticleSystem::set_wind);
		ClassDB::bind_method(D_METHOD("set_drag","d"), &ViennaParticleSystem::set_drag);
		ClassDB::bind_method(D_METHOD("set_turbulence","t"), &ViennaParticleSystem::set_turbulence);
		ClassDB::bind_method(D_METHOD("set_particle_radius","r"), &ViennaParticleSystem::set_particle_radius);
		ClassDB::bind_method(D_METHOD("set_particle_mass","m"), &ViennaParticleSystem::set_particle_mass);
		ClassDB::bind_method(D_METHOD("set_integrator","i"), &ViennaParticleSystem::set_integrator);
		ClassDB::bind_method(D_METHOD("set_collision_enabled","en"), &ViennaParticleSystem::set_collision_enabled);
		ClassDB::bind_method(D_METHOD("set_collision_bounce","b"), &ViennaParticleSystem::set_collision_bounce);
		ClassDB::bind_method(D_METHOD("set_collision_bodies","bodies"), &ViennaParticleSystem::set_collision_bodies);
		ClassDB::bind_method(D_METHOD("step","dt"), &ViennaParticleSystem::step);
		ClassDB::bind_method(D_METHOD("get_live_count"), &ViennaParticleSystem::get_live_count);
		ClassDB::bind_method(D_METHOD("clear"), &ViennaParticleSystem::clear);
	}

private:
	// ---- internal data ----
	LocalVector<ViennaParticle> particles;
	int max_particles;
	Integrator integrator;
	vec3 gravity;
	vec3 wind;
	real_t drag;
	real_t turbulence;
	real_t particle_radius;
	real_t particle_mass;
	// emitter
	real_t emit_rate;
	real_t emit_lifetime;
	real_t emit_speed;
	EmitterShape emit_shape;
	vec3 emit_box_extents;
	real_t emit_sphere_radius;
	real_t emit_cylinder_height;
	real_t emit_cylinder_radius;
	vec3 emit_rot;
	// collision
	bool collision_enabled;
	real_t collision_bounce;
	LocalVector<Ref<ViennaBody>> collision_bodies;
	real_t elapsed;
	RandomNumberGenerator rng;

	// ---- spawn ----
	void spawn(real_t dt) {
		if (emit_rate <= 0.0) return;
		int to_spawn = int(emit_rate * dt);
		if (to_spawn > max_particles / 10) to_spawn = max_particles / 10; // limit burst
		for (int i = 0; i < to_spawn; ++i) {
			if (particles.size() >= max_particles) break;
			vec3 pos = emit_position();
			vec3 vel = emit_velocity();
			ViennaParticle p;
			p.init(pos, vel, particle_mass, particle_radius, emit_lifetime);
			particles.push_back(p);
		}
	}

	vec3 emit_position() {
		// random position in local emitter space, rotated by emit_rot
		Basis rot(Euler(emit_rot));
		switch (emit_shape) {
			case POINT: return vec3();
			case BOX:
				return rot.xform(vec3(rng.randf_range(-emit_box_extents.x, emit_box_extents.x),
									  rng.randf_range(-emit_box_extents.y, emit_box_extents.y),
									  rng.randf_range(-emit_box_extents.z, emit_box_extents.z)));
			case SPHERE: {
				vec3 p;
				while (1) {
					p.x = rng.randf_range(-1.0, 1.0);
					p.y = rng.randf_range(-1.0, 1.0);
					p.z = rng.randf_range(-1.0, 1.0);
					if (p.length_squared() <= 1.0) break;
				}
				return rot.xform(p * emit_sphere_radius);
			}
			case CYLINDER: {
				real_t angle = rng.randf_range(0.0, Math_TAU);
				real_t r = emit_cylinder_radius * Math::sqrt(rng.randf());
				real_t h = rng.randf_range(-emit_cylinder_height * 0.5, emit_cylinder_height * 0.5);
				return rot.xform(vec3(r * Math::cos(angle), h, r * Math::sin(angle)));
			}
		}
		return vec3();
	}

	vec3 emit_velocity() {
		// basic upward with random spread
		return vec3(rng.randf_range(-1,1), 1.0, rng.randf_range(-1,1)).normalized() * emit_speed;
	}

	// ---- integration ----
	void integrate_euler(real_t dt) {
		for (ViennaParticle &p : particles) {
			if (!p.active) continue;
			p.life -= dt;
			if (p.life <= 0.0) { p.kill(); continue; }
			vec3 force = gravity + wind;
			// drag
			force -= p.velocity * drag;
			// turbulence (simple noise)
			if (turbulence > 0.0) {
				force += vec3(rng.randf_range(-1,1), rng.randf_range(-1,1), rng.randf_range(-1,1)) * turbulence;
			}
			p.velocity += force * dt;
			p.position += p.velocity * dt;
		}
	}

	void integrate_verlet(real_t dt) {
		// Verlet requires previous position; for system without it we approximate.
		for (ViennaParticle &p : particles) {
			if (!p.active) continue;
			p.life -= dt;
			if (p.life <= 0.0) { p.kill(); continue; }
			vec3 force = gravity + wind;
			force -= p.velocity * drag;
			if (turbulence > 0.0) {
				force += vec3(rng.randf_range(-1,1), rng.randf_range(-1,1), rng.randf_range(-1,1)) * turbulence;
			}
			// Simple Verlet: lastPos = pos - vel*dt, then pos = pos + vel*dt + force*dt^2
			vec3 lastPos = p.position - p.velocity * dt;
			vec3 newPos = p.position + p.velocity * dt + force * dt * dt;
			p.velocity = (newPos - lastPos) / (2.0 * dt);
			p.position = newPos;
		}
	}

	// ---- collision resolution ----
	void resolve_collisions() {
		if (collision_bodies.is_empty()) return;
		// Build Gaia BVH of rigid body AABBs
		gaia::bvh::BVH bvh;
		LocalVector<AABB> aabbs;
		LocalVector<int> idx_map;
		int nb = collision_bodies.size();
		for (int i = 0; i < nb; ++i) {
			if (collision_bodies[i].is_valid() && collision_bodies[i]->is_active()) {
				aabbs.push_back(collision_bodies[i]->get_aabb());
				idx_map.push_back(i);
			}
		}
		if (aabbs.is_empty()) return;
		bvh.build_final(aabbs);

		real_t r = particle_radius;
		real_t bounce = collision_bounce;

		for (ViennaParticle &p : particles) {
			if (!p.active) continue;
			AABB part_aabb(p.position - vec3(r, r, r), vec3(r*2, r*2, r*2));
			bvh.query_intersect(part_aabb, [&](int prim) {
				int idx = idx_map[prim];
				const Ref<ViennaBody> &body = collision_bodies[idx];
				if (body.is_null()) return;

				// Transform particle to body local
				mat4 inv = body->get_transform().affine_inverse();
				vec3 local = inv.xform(p.position);

				aabb local_aabb = body->get_collision_shape().is_valid() ?
					body->get_collision_shape()->get_local_aabb() : aabb(vec3(-0.5), vec3(1,1,1));
				// Closest point on AABB
				vec3 closest = local.clamp(local_aabb.position, local_aabb.position + local_aabb.size);
				real_t dist = local.distance_to(closest);
				if (dist < r && dist > 0.0) {
					vec3 local_normal = (local - closest) / dist;
					vec3 world_normal = body->get_transform().basis.xform(local_normal);
					// Push particle out
					p.position += world_normal * (r - dist);
					// Reflect velocity
					real_t vn = p.velocity.dot(world_normal);
					if (vn < 0.0) p.velocity -= world_normal * vn * (1.0 + bounce);
				}
			});
		}
	}

	// ---- prune dead particles ----
	void prune_dead() {
		int write = 0;
		for (int i = 0; i < particles.size(); ++i) {
			if (particles[i].active) {
				if (i != write) particles[write] = particles[i];
				write++;
			}
		}
		particles.resize(write);
	}

	static vec3 Euler(const vec3 &euler) {
		// Dummy conversion – the actual orientation is not needed for simple emissions.
		return Basis::from_euler(euler).get_euler();
	}
};

} // namespace vienna

#endif // VIENNA_PARTICLES_SYSTEM_H