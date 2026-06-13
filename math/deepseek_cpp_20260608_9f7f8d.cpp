// File 154: modules/genesis/src/solvers/sf_solver.cpp
// Stable Fluids solver implementation: advection (semi-Lagrangian & MacCormack),
// diffusion, pressure projection, buoyancy, and boundary conditions.
// Operates on a staggered MAC grid with scalar fields for density and temperature.

#include "sf_solver.h"

#include "../materials/sf_material.h"
#include "../boundaries/boundary_conditions.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace genesis {

void SFSolver::step() {
	if (N <= 0) return;
	real_t sub_dt = dt / real_t(sub_steps);
	for (int substep = 0; substep < sub_steps; ++substep) {
		vel_step(sub_dt);
		scalar_step(sub_dt);
		time += sub_dt;
		// Swap buffers for next substep
		SWAP(u, u_prev);
		SWAP(v, v_prev);
		SWAP(w, w_prev);
		SWAP(density, density_prev);
		SWAP(temperature, temperature_prev);
	}
}

void SFSolver::vel_step(real_t dt) {
	// 1. Add buoyancy forces
	add_buoyancy(dt);
	// 2. Diffuse velocity (viscosity) if non-zero
	const real_t visc = material.is_valid() ? material->get_kinematic_viscosity() : 0.0f;
	if (visc > 0.0f) {
		diffuse_velocity(dt, visc);
	} else {
		SWAP(u, u_prev);
		SWAP(v, v_prev);
		SWAP(w, w_prev);
	}
	// 3. Advect velocity (MacCormack or semi-Lagrangian)
	const int adv_method = material.is_valid() ? material->get_advection_method() : 0;
	if (adv_method == 1) {
		advect_velocity_maccormack(dt);
	} else {
		advect_velocity(dt);
	}
	// 4. Pressure projection (incompressibility)
	project(dt);
}

void SFSolver::add_buoyancy(real_t dt) {
	if (!material.is_valid()) return;
	const real_t alpha = material->get_buoyancy_alpha();
	const real_t beta  = material->get_buoyancy_beta();
	const real_t T_amb = material->get_ambient_temperature();

	for (int k = 1; k <= N; ++k) {
		for (int j = 1; j <= N; ++j) {
			for (int i = 1; i <= N; ++i) {
				int idx = IX(i, j, k);
				real_t buoy = alpha * density[idx] + beta * (temperature[idx] - T_amb);
				// Apply upward along Y axis
				v[idx] += buoy * dt;
			}
		}
	}
	set_boundary_velocity();
}

void SFSolver::diffuse_velocity(real_t dt, real_t visc) {
	// Copy current velocity to previous buffers
	for (int i = 0; i < u.size(); ++i) {
		u_prev[i] = u[i];
		v_prev[i] = v[i];
		w_prev[i] = w[i];
	}
	// Explicit Euler diffusion: u_new = u + dt * visc * laplacian(u)
	for (int k = 1; k <= N; ++k) {
		for (int j = 1; j <= N; ++j) {
			for (int i = 1; i <= N; ++i) {
				int idx = IX(i, j, k);
				u[idx] += dt * visc * laplacian(i, j, k, u_prev, 0);
				v[idx] += dt * visc * laplacian(i, j, k, v_prev, 1);
				w[idx] += dt * visc * laplacian(i, j, k, w_prev, 2);
			}
		}
	}
	set_boundary_velocity();
}

void SFSolver::advect_velocity(real_t dt) {
	// Save current velocity as previous
	for (int i = 0; i < u.size(); ++i) {
		u_prev[i] = u[i];
		v_prev[i] = v[i];
		w_prev[i] = w[i];
	}
	// Back-trace each grid cell centre and interpolate
	for (int k = 1; k <= N; ++k) {
		for (int j = 1; j <= N; ++j) {
			for (int i = 1; i <= N; ++i) {
				int idx = IX(i, j, k);
				// Backtrace position
				real_t x = (real_t)i - dt * u_prev[idx] * inv_h;
				real_t y = (real_t)j - dt * v_prev[idx] * inv_h;
				real_t z = (real_t)k - dt * w_prev[idx] * inv_h;
				x = CLAMP(x, 0.5f, N + 0.5f);
				y = CLAMP(y, 0.5f, N + 0.5f);
				z = CLAMP(z, 0.5f, N + 0.5f);
				u[idx] = interpolate_velocity(x, y, z, 0);
				v[idx] = interpolate_velocity(x, y, z, 1);
				w[idx] = interpolate_velocity(x, y, z, 2);
			}
		}
	}
	set_boundary_velocity();
}

void SFSolver::advect_velocity_maccormack(real_t dt) {
	// Copy current velocity
	for (int i = 0; i < u.size(); ++i) {
		u_prev[i] = u[i];
		v_prev[i] = v[i];
		w_prev[i] = w[i];
	}
	const int total = size * size * size;
	LocalVector<real_t> u_hat(total), v_hat(total), w_hat(total);

	// Forward advection (same as semi-Lagrangian)
	for (int k = 1; k <= N; ++k) {
		for (int j = 1; j <= N; ++j) {
			for (int i = 1; i <= N; ++i) {
				int idx = IX(i, j, k);
				real_t x = (real_t)i - dt * u_prev[idx] * inv_h;
				real_t y = (real_t)j - dt * v_prev[idx] * inv_h;
				real_t z = (real_t)k - dt * w_prev[idx] * inv_h;
				x = CLAMP(x, 0.5f, N + 0.5f);
				y = CLAMP(y, 0.5f, N + 0.5f);
				z = CLAMP(z, 0.5f, N + 0.5f);
				u_hat[idx] = interpolate_velocity(x, y, z, 0);
				v_hat[idx] = interpolate_velocity(x, y, z, 1);
				w_hat[idx] = interpolate_velocity(x, y, z, 2);
			}
		}
	}

	// Backward advection of forward solution
	LocalVector<real_t> u_hat_hat(total), v_hat_hat(total), w_hat_hat(total);
	for (int k = 1; k <= N; ++k) {
		for (int j = 1; j <= N; ++j) {
			for (int i = 1; i <= N; ++i) {
				int idx = IX(i, j, k);
				real_t x = (real_t)i + dt * u_hat[idx] * inv_h;
				real_t y = (real_t)j + dt * v_hat[idx] * inv_h;
				real_t z = (real_t)k + dt * w_hat[idx] * inv_h;
				x = CLAMP(x, 0.5f, N + 0.5f);
				y = CLAMP(y, 0.5f, N + 0.5f);
				z = CLAMP(z, 0.5f, N + 0.5f);
				u_hat_hat[idx] = interpolate_velocity(x, y, z, 0);
				v_hat_hat[idx] = interpolate_velocity(x, y, z, 1);
				w_hat_hat[idx] = interpolate_velocity(x, y, z, 2);
			}
		}
	}

	// MacCormack correction: u = u_hat + 0.5 * (u_prev - u_hat_hat)
	for (int i = 0; i < total; ++i) {
		u[i] = u_hat[i] + 0.5f * (u_prev[i] - u_hat_hat[i]);
		v[i] = v_hat[i] + 0.5f * (v_prev[i] - v_hat_hat[i]);
		w[i] = w_hat[i] + 0.5f * (w_prev[i] - w_hat_hat[i]);
	}
	set_boundary_velocity();
}

void SFSolver::project(real_t dt) {
	// Compute negative divergence
	for (int k = 1; k <= N; ++k) {
		for (int j = 1; j <= N; ++j) {
			for (int i = 1; i <= N; ++i) {
				int idx = IX(i, j, k);
				divergence[idx] = -0.5f * h * (
					(u[IX(i+1,j,k)] - u[IX(i-1,j,k)]) +
					(v[IX(i,j+1,k)] - v[IX(i,j-1,k)]) +
					(w[IX(i,j,k+1)] - w[IX(i,j,k-1)]) );
				pressure[idx] = 0.0f;
			}
		}
	}
	set_boundary_pressure();

	// Solve Poisson equation with Gauss-Seidel
	const int max_iter = material.is_valid() ? material->get_pressure_solver_iter() : 20;
	for (int iter = 0; iter < max_iter; ++iter) {
		for (int k = 1; k <= N; ++k) {
			for (int j = 1; j <= N; ++j) {
				for (int i = 1; i <= N; ++i) {
					int idx = IX(i, j, k);
					pressure[idx] = (divergence[idx] +
						pressure[IX(i-1,j,k)] + pressure[IX(i+1,j,k)] +
						pressure[IX(i,j-1,k)] + pressure[IX(i,j+1,k)] +
						pressure[IX(i,j,k-1)] + pressure[IX(i,j,k+1)]) / 6.0f;
				}
			}
		}
		set_boundary_pressure();
	}

	// Subtract pressure gradient from velocity
	for (int k = 1; k <= N; ++k) {
		for (int j = 1; j <= N; ++j) {
			for (int i = 1; i <= N; ++i) {
				int idx = IX(i, j, k);
				u[idx] -= 0.5f * (pressure[IX(i+1,j,k)] - pressure[IX(i-1,j,k)]) * inv_h;
				v[idx] -= 0.5f * (pressure[IX(i,j+1,k)] - pressure[IX(i,j-1,k)]) * inv_h;
				w[idx] -= 0.5f * (pressure[IX(i,j,k+1)] - pressure[IX(i,j,k-1)]) * inv_h;
			}
		}
	}
	set_boundary_velocity();
}

void SFSolver::scalar_step(real_t dt) {
	// Advect density (semi-Lagrangian)
	SWAP(density, density_prev);
	for (int k = 1; k <= N; ++k) {
		for (int j = 1; j <= N; ++j) {
			for (int i = 1; i <= N; ++i) {
				int idx = IX(i, j, k);
				real_t x = (real_t)i - dt * u[idx] * inv_h;
				real_t y = (real_t)j - dt * v[idx] * inv_h;
				real_t z = (real_t)k - dt * w[idx] * inv_h;
				x = CLAMP(x, 0.5f, N + 0.5f);
				y = CLAMP(y, 0.5f, N + 0.5f);
				z = CLAMP(z, 0.5f, N + 0.5f);
				density[idx] = interpolate_scalar(x, y, z, density_prev);
			}
		}
	}

	// Advect temperature (semi-Lagrangian)
	SWAP(temperature, temperature_prev);
	for (int k = 1; k <= N; ++k) {
		for (int j = 1; j <= N; ++j) {
			for (int i = 1; i <= N; ++i) {
				int idx = IX(i, j, k);
				real_t x = (real_t)i - dt * u[idx] * inv_h;
				real_t y = (real_t)j - dt * v[idx] * inv_h;
				real_t z = (real_t)k - dt * w[idx] * inv_h;
				x = CLAMP(x, 0.5f, N + 0.5f);
				y = CLAMP(y, 0.5f, N + 0.5f);
				z = CLAMP(z, 0.5f, N + 0.5f);
				temperature[idx] = interpolate_scalar(x, y, z, temperature_prev);
			}
		}
	}
}

// --- Boundary helpers ---
void SFSolver::set_boundary_velocity() {
	for (int k = 0; k <= N+1; ++k) {
		for (int j = 0; j <= N+1; ++j) {
			// X faces (i=0 and i=N+1)
			u[IX(0,j,k)] = 0.0f; u[IX(N+1,j,k)] = 0.0f;
			v[IX(0,j,k)] = -v[IX(1,j,k)]; v[IX(N+1,j,k)] = -v[IX(N,j,k)];
			w[IX(0,j,k)] = -w[IX(1,j,k)]; w[IX(N+1,j,k)] = -w[IX(N,j,k)];
		}
	}
	for (int k = 0; k <= N+1; ++k) {
		for (int i = 0; i <= N+1; ++i) {
			// Y faces
			v[IX(i,0,k)] = 0.0f; v[IX(i,N+1,k)] = 0.0f;
			u[IX(i,0,k)] = -u[IX(i,1,k)]; u[IX(i,N+1,k)] = -u[IX(i,N,k)];
			w[IX(i,0,k)] = -w[IX(i,1,k)]; w[IX(i,N+1,k)] = -w[IX(i,N,k)];
		}
	}
	for (int j = 0; j <= N+1; ++j) {
		for (int i = 0; i <= N+1; ++i) {
			// Z faces
			w[IX(i,j,0)] = 0.0f; w[IX(i,j,N+1)] = 0.0f;
			u[IX(i,j,0)] = -u[IX(i,j,1)]; u[IX(i,j,N+1)] = -u[IX(i,j,N)];
			v[IX(i,j,0)] = -v[IX(i,j,1)]; v[IX(i,j,N+1)] = -v[IX(i,j,N)];
		}
	}
}

void SFSolver::set_boundary_pressure() {
	for (int j = 1; j <= N; ++j) {
		for (int k = 1; k <= N; ++k) {
			pressure[IX(0,j,k)] = pressure[IX(1,j,k)];
			pressure[IX(N+1,j,k)] = pressure[IX(N,j,k)];
		}
	}
	for (int i = 1; i <= N; ++i) {
		for (int k = 1; k <= N; ++k) {
			pressure[IX(i,0,k)] = pressure[IX(i,1,k)];
			pressure[IX(i,N+1,k)] = pressure[IX(i,N,k)];
		}
	}
	for (int i = 1; i <= N; ++i) {
		for (int j = 1; j <= N; ++j) {
			pressure[IX(i,j,0)] = pressure[IX(i,j,1)];
			pressure[IX(i,j,N+1)] = pressure[IX(i,j,N)];
		}
	}
}

// --- Interpolation helpers (trilinear) ---
real_t SFSolver::interpolate_velocity(real_t x, real_t y, real_t z, int comp) const {
	int i0 = (int)x, j0 = (int)y, k0 = (int)z;
	int i1 = i0+1, j1 = j0+1, k1 = k0+1;
	real_t s1 = x - i0, s0 = 1.0f - s1;
	real_t t1 = y - j0, t0 = 1.0f - t1;
	real_t u1 = z - k0, u0 = 1.0f - u1;
	i0 = CLAMP(i0, 0, N+1); i1 = CLAMP(i1, 0, N+1);
	j0 = CLAMP(j0, 0, N+1); j1 = CLAMP(j1, 0, N+1);
	k0 = CLAMP(k0, 0, N+1); k1 = CLAMP(k1, 0, N+1);
	const real_t *arr = u.ptr();
	if (comp == 1) arr = v.ptr();
	if (comp == 2) arr = w.ptr();
	return s0*(t0*(u0*arr[IX(i0,j0,k0)] + u1*arr[IX(i0,j0,k1)]) +
			   t1*(u0*arr[IX(i0,j1,k0)] + u1*arr[IX(i0,j1,k1)])) +
		   s1*(t0*(u0*arr[IX(i1,j0,k0)] + u1*arr[IX(i1,j0,k1)]) +
			   t1*(u0*arr[IX(i1,j1,k0)] + u1*arr[IX(i1,j1,k1)]));
}

real_t SFSolver::interpolate_scalar(real_t x, real_t y, real_t z, const LocalVector<real_t> &field) const {
	int i0 = (int)x, j0 = (int)y, k0 = (int)z;
	int i1 = i0+1, j1 = j0+1, k1 = k0+1;
	real_t s1 = x - i0, s0 = 1.0f - s1;
	real_t t1 = y - j0, t0 = 1.0f - t1;
	real_t u1 = z - k0, u0 = 1.0f - u1;
	i0 = CLAMP(i0, 0, N+1); i1 = CLAMP(i1, 0, N+1);
	j0 = CLAMP(j0, 0, N+1); j1 = CLAMP(j1, 0, N+1);
	k0 = CLAMP(k0, 0, N+1); k1 = CLAMP(k1, 0, N+1);
	return s0*(t0*(u0*field[IX(i0,j0,k0)] + u1*field[IX(i0,j0,k1)]) +
			   t1*(u0*field[IX(i0,j1,k0)] + u1*field[IX(i0,j1,k1)])) +
		   s1*(t0*(u0*field[IX(i1,j0,k0)] + u1*field[IX(i1,j0,k1)]) +
			   t1*(u0*field[IX(i1,j1,k0)] + u1*field[IX(i1,j1,k1)]));
}

} // namespace genesis