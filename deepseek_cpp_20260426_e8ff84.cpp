// File 163: modules/genesis/src/entities/mpm_entity.cpp
// Implements MPMEntity – holds a collection of material point method particles,
// provides accessors for grid resolution, cell size, particle list, and
// physical properties (mass, AABB).

#include "mpm_entity.h"

#include "core/math/aabb.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace genesis {

MPMEntity::MPMEntity() :
	grid_resolution(64),
	cell_size(0.05) {
	solver_type = SolverType::MPM;
}

void MPMEntity::add_particle(const Vector3 &pos, const Vector3 &vel,
							 real_t mass, real_t volume) {
	Particle p;
	p.position = pos;
	p.velocity = vel;
	p.mass = MAX(mass, 0.0);
	p.volume0 = MAX(volume, CMP_EPSILON);
	p.F = Basis();             // identity deformation gradient
	p.Jp = 1.0;               // no plastic deformation initially
	p.damage = 0.0;
	particles.push_back(p);
}

void MPMEntity::clear_particles() {
	particles.clear();
}

int MPMEntity::particle_count() const {
	return particles.size();
}

MPMEntity::Particle &MPMEntity::get_particle(int idx) {
	return particles[idx];
}

const MPMEntity::Particle &MPMEntity::get_particle(int idx) const {
	return particles[idx];
}

LocalVector<MPMEntity::Particle> &MPMEntity::get_particles() {
	return particles;
}

const LocalVector<MPMEntity::Particle> &MPMEntity::get_particles() const {
	return particles;
}

void MPMEntity::set_grid_resolution(int p_res) {
	grid_resolution = MAX(p_res, 1);
}

int MPMEntity::get_grid_resolution() const {
	return grid_resolution;
}

void MPMEntity::set_cell_size(real_t p_dx) {
	cell_size = MAX(p_dx, 1e-6);
}

real_t MPMEntity::get_cell_size() const {
	return cell_size;
}

AABB MPMEntity::get_aabb() const {
	if (particles.is_empty())
		return AABB(transform.origin, Vector3());
	Vector3 minv(INFINITY, INFINITY, INFINITY);
	Vector3 maxv(-INFINITY, -INFINITY, -INFINITY);
	for (const Particle &p : particles) {
		minv = minv.min(p.position);
		maxv = maxv.max(p.position);
	}
	return AABB(minv, maxv - minv);
}

real_t MPMEntity::get_mass() const {
	real_t total = 0.0;
	for (const Particle &p : particles)
		total += p.mass;
	return total;
}

real_t MPMEntity::get_inertia_scalar() const {
	return 0.0;   // not a rigid body
}

void MPMEntity::apply_force_to_particles(const Vector3 &force_per_unit_mass) {
	// This is just a convenience; the MPM solver applies forces directly.
	// We store nothing here – the solver uses gravity and internal forces.
}

void MPMEntity::init_from_options(const genesis::options::Options &opts) {
	BaseEntity::init_from_options(opts);
	grid_resolution = opts.get_int("mpm.grid_res", grid_resolution);
	cell_size = opts.get_real("mpm.cell_size", cell_size);
}

} // namespace genesis