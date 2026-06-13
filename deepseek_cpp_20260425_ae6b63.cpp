// File 73: modules/genesis/src/entities/mpm_entity.h
// MPM entity – a continuum represented as a set of particles for the MPM solver.

#ifndef GENESIS_ENTITIES_MPM_ENTITY_H
#define GENESIS_ENTITIES_MPM_ENTITY_H

#include "base_entity.h"
#include "../materials/mpm_material.h"
#include "../core/genesis_types.h"
#include "../core/genesis_constants.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"

namespace genesis {

class MPMMaterial;

class MPMEntity : public BaseEntity {
	GDCLASS(MPMEntity, BaseEntity);

public:
	struct Particle {
		Vector3 position;
		Vector3 velocity;
		real_t mass;
		real_t volume0;          // initial volume
		Basis F;                 // deformation gradient
		real_t Jp;               // Jacobian for plasticity
		real_t damage;           // accumulated damage (0..1)

		Particle() : velocity(), mass(1.0), volume0(1.0), F(Basis()), Jp(1.0), damage(0.0) {}
		Particle(const Vector3 &p_pos, const Vector3 &p_vel, real_t p_mass, real_t p_vol) :
			position(p_pos), velocity(p_vel), mass(p_mass), volume0(p_vol),
			F(Basis()), Jp(1.0), damage(0.0) {}
	};

	MPMEntity() : grid_resolution(64), cell_size(0.05) {
		solver_type = SolverType::MPM;
		// Particles are added later via add_particle()
	}

	// --- Particle access ---
	void add_particle(const Vector3 &pos, const Vector3 &vel, real_t mass, real_t volume) {
		particles.push_back(Particle(pos, vel, mass, volume));
	}
	void clear_particles() { particles.clear(); }
	int particle_count() const { return particles.size(); }

	Particle &get_particle(int idx) { return particles[idx]; }
	const Particle &get_particle(int idx) const { return particles[idx]; }

	LocalVector<Particle> &get_particles() { return particles; }
	const LocalVector<Particle> &get_particles() const { return particles; }

	// --- Grid settings (may be overridden by solver) ---
	void set_grid_resolution(int p_res) { grid_resolution = MAX(p_res, 1); }
	int get_grid_resolution() const { return grid_resolution; }

	void set_cell_size(real_t p_dx) { cell_size = MAX(p_dx, 1e-6); }
	real_t get_cell_size() const { return cell_size; }

	// --- AABB (computed from particles) ---
	virtual AABB get_aabb() const override {
		if (particles.is_empty()) return AABB(transform.origin, Vector3());
		Vector3 min(INFINITY, INFINITY, INFINITY);
		Vector3 max(-INFINITY, -INFINITY, -INFINITY);
		for (const Particle &p : particles) {
			min = min.min(p.position);
			max = max.max(p.position);
		}
		return AABB(min, max - min);
	}

	// --- Mass (sum of particle masses) ---
	virtual real_t get_mass() const override {
		real_t total = 0.0;
		for (const Particle &p : particles) total += p.mass;
		return total;
	}

	// --- No rigid-type inertia needed ---
	virtual real_t get_inertia_scalar() const override { return 0.0; }

	// --- Apply force to all particles (e.g., gravity done by solver) ---
	void apply_force_to_particles(const Vector3 &force_per_unit_mass) {
		// forces are applied as accelerations in the solver; stored externally not here.
	}

	// --- Initialize from options ---
	virtual void init_from_options(const genesis::options::Options &opts) override {
		BaseEntity::init_from_options(opts);
		grid_resolution = opts.get_int("mpm.grid_res", grid_resolution);
		cell_size = opts.get_real("mpm.cell_size", cell_size);
		// Particles are typically loaded from a mesh file or programmatically added.
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("add_particle", "pos", "vel", "mass", "volume"), &MPMEntity::add_particle);
		ClassDB::bind_method(D_METHOD("clear_particles"), &MPMEntity::clear_particles);
		ClassDB::bind_method(D_METHOD("particle_count"), &MPMEntity::particle_count);
		ClassDB::bind_method(D_METHOD("get_particle", "idx"), &MPMEntity::get_particle);
		ClassDB::bind_method(D_METHOD("set_grid_resolution", "res"), &MPMEntity::set_grid_resolution);
		ClassDB::bind_method(D_METHOD("get_grid_resolution"), &MPMEntity::get_grid_resolution);
		ClassDB::bind_method(D_METHOD("set_cell_size", "dx"), &MPMEntity::set_cell_size);
		ClassDB::bind_method(D_METHOD("get_cell_size"), &MPMEntity::get_cell_size);

		ADD_PROPERTY(PropertyInfo(Variant::INT, "grid_resolution", PROPERTY_HINT_RANGE, "1,512,1"), "set_grid_resolution", "get_grid_resolution");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cell_size", PROPERTY_HINT_RANGE, "0.001,10,0.001"), "set_cell_size", "get_cell_size");
	}

private:
	LocalVector<Particle> particles;
	int grid_resolution;
	real_t cell_size;
};

} // namespace genesis

#endif // GENESIS_ENTITIES_MPM_ENTITY_H