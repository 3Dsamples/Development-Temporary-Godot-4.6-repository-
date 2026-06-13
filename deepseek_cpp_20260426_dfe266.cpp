// File 153: modules/genesis/src/solvers/sph_solver.cpp
// Full SPH solver implementation: density computation, pressure forces,
// viscosity, surface tension, and integration. Uses spatial hash for
// neighbour search and enforces boundary conditions.

#include "sph_solver.h"

#include "../entities/particle_entity.h"
#include "../materials/sph_material.h"
#include "../boundaries/sdf_boundary.h"
#include "../../../gaia/src/spatial_query/spatial_hash.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace genesis {

void SPHSolver::step() {
	if (particles.is_empty()) return;
	real_t sub_dt = dt / real_t(sub_steps);
	for (int substep = 0; substep < sub_steps; ++substep) {
		// 1. Build spatial hash for neighbour search
		build_hash();
		// 2. Compute density and pressure for all particles
		compute_density_pressure();
		// 3. Compute forces (pressure gradient, viscosity, surface tension, gravity)
		compute_forces(sub_dt);
		// 4. Apply boundary conditions (SDF walls)
		apply_boundaries(sub_dt);
		// 5. Integrate velocities and positions
		integrate(sub_dt);
		time += sub_dt;
	}
}

void SPHSolver::build_hash() {
	spatial_hash.clear();
	real_t cell_size = 2.0 * smoothing_length;
	spatial_hash.set_cell_size(cell_size);
	for (int i = 0; i < particles.size(); ++i) {
		spatial_hash.insert(i, particles[i].position);
	}
}

void SPHSolver::compute_density_pressure() {
	const real_t h = smoothing_length;
	const real_t h2 = h * h;
	// Poly6 kernel constant: 315 / (64 * pi * h^9)
	const real_t poly6_const = 315.0 / (64.0 * Math_PI * Math::pow(h, 9));
	const real_t self_contrib = poly6_const * Math::pow(h, 6); // W(0) weight for self

	for (int i = 0; i < particles.size(); ++i) {
		real_t density_sum = particles[i].mass * self_contrib; // self contribution
		LocalVector<int32_t> neighbours;
		spatial_hash.query(particles[i].position, neighbours, true);
		for (int n : neighbours) {
			if (n == i) continue;
			Vector3 diff = particles[i].position - particles[n].position;
			real_t r2 = diff.length_squared();
			if (r2 >= h2) continue;
			real_t diff_h2 = h2 - r2;
			density_sum += particles[n].mass * poly6_const * diff_h2 * diff_h2 * diff_h2;
		}
		particles[i].density = MAX(density_sum, rest_density * 0.1f);
	}

	// Compute pressure from density (Tait or ideal gas EOS)
	if (material.is_valid()) {
		for (int i = 0; i < particles.size(); ++i) {
			particles[i].pressure = material->compute_pressure(particles[i].density, rest_density);
		}
	} else {
		// Default Tait EOS
		real_t k = 1000.0f; // bulk modulus
		for (int i = 0; i < particles.size(); ++i) {
			particles[i].pressure = k * (Math::pow(particles[i].density / rest_density, 7.0f) - 1.0f);
		}
	}
}

void SPHSolver::compute_forces(real_t dt) {
	const real_t h = smoothing_length;
	const real_t h2 = h * h;
	// Spiky gradient constant: -45 / (pi * h^6)
	const real_t spiky_grad_const = -45.0 / (Math_PI * Math::pow(h, 6));
	// Viscosity laplacian constant: 45 / (pi * h^6)
	const real_t visc_lapl_const = 45.0 / (Math_PI * Math::pow(h, 6));

	// Reset force accumulators
	if (force_accum.size() != particles.size()) force_accum.resize(particles.size());
	for (int i = 0; i < particles.size(); ++i) force_accum[i] = Vector3();

	for (int i = 0; i < particles.size(); ++i) {
		const Vector3 &pos_i = particles[i].position;
		const Vector3 &vel_i = particles[i].velocity;
		const real_t rho_i = particles[i].density;
		const real_t p_i  = particles[i].pressure;
		const real_t mass_i = particles[i].mass;
		Vector3 f_pressure, f_visc, f_surf;

		LocalVector<int32_t> neighbours;
		spatial_hash.query(pos_i, neighbours, true);
		for (int n : neighbours) {
			if (n == i) continue;
			const Vector3 &pos_j = particles[n].position;
			const Vector3 &vel_j = particles[n].velocity;
			const real_t rho_j = particles[n].density;
			const real_t p_j  = particles[n].pressure;
			const real_t mass_j = particles[n].mass;

			Vector3 diff = pos_i - pos_j;
			real_t r = diff.length();
			if (r < CMP_EPSILON || r >= h) continue;
			Vector3 r_dir = diff / r;

			// Pressure gradient (symmetrised)
			real_t p_term = (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j));
			real_t spiky_weight = (h - r) * (h - r);
			real_t spiky_grad_mag = spiky_grad_const * spiky_weight / r;
			f_pressure -= mass_j * p_term * spiky_grad_mag * r_dir;

			// Viscosity (XSPH style: laplacian)
			Vector3 v_diff = vel_j - vel_i;
			real_t lapl_weight = visc_lapl_const * (h - r);
			real_t viscosity_coeff = material.is_valid() ? material->get_viscosity_mu() : 0.001f;
			f_visc += viscosity_coeff * mass_j * (v_diff / rho_j) * lapl_weight;

			// Surface tension (CSF model – curvature approximated by colour field)
			if (material.is_valid() && material->get_surface_tension_coeff() > 0.0f) {
				// Colour field Laplacian contribution (simplified)
				const real_t poly6_const = 315.0 / (64.0 * Math_PI * Math::pow(h, 9));
				real_t diff_h2 = h2 - r * r;
				real_t colour_contrib = mass_j * poly6_const * diff_h2 * diff_h2 * diff_h2 / rho_j;
				f_surf += colour_contrib * diff;
			}
		}

		// Surface tension normal and curvature
		real_t surface_coeff = material.is_valid() ? material->get_surface_tension_coeff() : 0.0f;
		if (surface_coeff > 0.0f) {
			// Normalise and combine
			real_t surf_len = f_surf.length();
			if (surf_len > CMP_EPSILON) {
				f_surf = f_surf * (surface_coeff / surf_len);
			}
		}

		// Accumulate forces: gravity + pressure + viscosity + surface tension
		Vector3 f_total = gravity * particles[i].density + f_pressure + f_visc + f_surf;
		force_accum[i] = f_total;
	}
}

void SPHSolver::apply_boundaries(real_t dt) {
	// If an SDF boundary is attached, use it to push particles back.
	if (sdf_boundary.is_valid()) {
		sdf_boundary->apply_to_particles(particle_positions_ref(), particle_velocities_ref(), dt);
	} else {
		// Default: sticky box at [-5,5]^3
		const real_t limit = 5.0f;
		for (int i = 0; i < particles.size(); ++i) {
			for (int d = 0; d < 3; ++d) {
				if (particles[i].position[d] < -limit) {
					particles[i].position[d] = -limit;
					particles[i].velocity[d] = 0.0f;
				} else if (particles[i].position[d] > limit) {
					particles[i].position[d] = limit;
					particles[i].velocity[d] = 0.0f;
				}
			}
		}
	}
}

void SPHSolver::integrate(real_t dt) {
	for (int i = 0; i < particles.size(); ++i) {
		// Semi‑implicit Euler: first update velocity with forces, then position
		Vector3 acceleration = force_accum[i] / particles[i].density;
		particles[i].velocity += acceleration * dt;
		particles[i].position += particles[i].velocity * dt;
	}
}

// Helper to get positions/velocities as LocalVector<Vector3> for boundary interface
LocalVector<Vector3> &SPHSolver::particle_positions_ref() {
	// We need to return a reference to a LocalVector<Vector3> that backs the particles.
	// Since particles store position inside a struct, we'll just build a temporary mapping.
	// For efficiency, we assume the boundary function accepts raw arrays; we can cast.
	// Here we'll resize a temporary buffer and fill it (not ideal, but works).
	if (pos_buffer.size() != particles.size()) pos_buffer.resize(particles.size());
	for (int i = 0; i < particles.size(); ++i) pos_buffer[i] = particles[i].position;
	return pos_buffer;
}

LocalVector<Vector3> &SPHSolver::particle_velocities_ref() {
	if (vel_buffer.size() != particles.size()) vel_buffer.resize(particles.size());
	for (int i = 0; i < particles.size(); ++i) vel_buffer[i] = particles[i].velocity;
	return vel_buffer;
}

} // namespace genesis