// File 71: modules/genesis/src/solvers/sf_solver.h
// Stable Fluids (Eulerian grid) solver for smoke, fire, and liquid.
// Solves the incompressible Navier-Stokes equations using operator splitting:
// advection, diffusion, pressure projection, and vorticity confinement.
// Fully integrated with SFMaterial parameters.

#ifndef GENESIS_SOLVERS_SF_SOLVER_H
#define GENESIS_SOLVERS_SF_SOLVER_H

#include "base_solver.h"
#include "../materials/sf_material.h"
#include "../boundaries/boundary_conditions.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"

namespace genesis {

class SFSolver : public BaseSolver {
	GDCLASS(SFSolver, BaseSolver);

public:
	// Grid dimensions and spacing
	int N;             // number of cells per axis (same for all)
	int size;          // N+2 for padded buffer
	real_t h;          // cell size
	real_t inv_h;      // 1/h

	// Staggered MAC grid (velocity stored at cell faces)
	LocalVector<real_t> u;     // x-velocity (size^3)
	LocalVector<real_t> v;     // y-velocity
	LocalVector<real_t> w;     // z-velocity
	LocalVector<real_t> u_prev, v_prev, w_prev;

	// Scalar fields (centered)
	LocalVector<real_t> density;
	LocalVector<real_t> density_prev;
	LocalVector<real_t> temperature;
	LocalVector<real_t> temperature_prev;
	LocalVector<real_t> pressure;
	LocalVector<real_t> divergence;

	// Material reference
	Ref<SFMaterial> material;

	SFSolver() : BaseSolver(), N(64), h(1.0/64.0), inv_h(64.0) {
		resize_grid(64);
	}

	void set_grid_resolution(int n) {
		N = MAX(n, 4);
		h = material.is_valid() ? material->get_domain_size() / N : 1.0 / N;
		inv_h = 1.0 / h;
		resize_grid(N);
	}

	void set_material(const Ref<SFMaterial> &p_mat) {
		material = p_mat;
		if (material.is_valid()) {
			set_grid_resolution(material->get_grid_resolution());
			h = material->get_domain_size() / N;
			inv_h = 1.0 / h;
		}
	}

	// Resize all grid buffers to (N+2)*(N+2)*(N+2)
	void resize_grid(int n) {
		N = n;
		size = N + 2;
		int total = size * size * size;
		u.resize(total);
		v.resize(total);
		w.resize(total);
		u_prev.resize(total);
		v_prev.resize(total);
		w_prev.resize(total);
		density.resize(total);
		density_prev.resize(total);
		temperature.resize(total);
		temperature_prev.resize(total);
		pressure.resize(total);
		divergence.resize(total);
		clear_fields();
	}

	void clear_fields() {
		memset(u.ptr(), 0, u.size() * sizeof(real_t));
		memset(v.ptr(), 0, v.size() * sizeof(real_t));
		memset(w.ptr(), 0, w.size() * sizeof(real_t));
		memset(density.ptr(), 0, density.size() * sizeof(real_t));
		memset(temperature.ptr(), 0, temperature.size() * sizeof(real_t));
	}

	// Index helpers (MAC grid)
	inline int IX(int i, int j, int k) const { return i + size * (j + size * k); }

	// Set initial velocity / density sources (e.g., from emitters)
	void add_velocity(int i, int j, int k, real_t du, real_t dv, real_t dw) {
		int idx = IX(i, j, k);
		if (i >= 0 && i < size && j >= 0 && j < size && k >= 0 && k < size) {
			u[idx] += du;
			v[idx] += dv;
			w[idx] += dw;
		}
	}

	void add_density(int i, int j, int k, real_t d) {
		int idx = IX(i, j, k);
		if (i >= 0 && i < size && j >= 0 && j < size && k >= 0 && k < size) {
			density[idx] += d;
		}
	}

	// Main step
	virtual void step() override {
		real_t sub_dt = dt / real_t(sub_steps);
		for (int substep = 0; substep < sub_steps; ++substep) {
			vel_step(sub_dt);
			scalar_step(sub_dt);
			time += sub_dt;
			// swap buffers for next step
			SWAP(u, u_prev);
			SWAP(v, v_prev);
			SWAP(w, w_prev);
			SWAP(density, density_prev);
			SWAP(temperature, temperature_prev);
		}
	}

	virtual void solve(real_t p_sub_dt) override {}

private:
	// --- Velocity solver ---
	void vel_step(real_t dt) {
		// 1. Add force (buoyancy, gravity, user forces)
		add_buoyancy(dt);
		// 2. Diffuse (viscosity)
		if (material.is_valid() && material->get_kinematic_viscosity() > 0) {
			real_t visc = material->get_kinematic_viscosity();
			diffuse_velocity(dt, visc);
		}
		// 3. Advect (semi-Lagrangian or MacCormack)
		if (material.is_valid() && material->get_advection_method() == 1) {
			advect_velocity_maccormack(dt);
		} else {
			advect_velocity(dt);
		}
		// 4. Project (pressure Poisson)
		project(dt);
	}

	void add_buoyancy(real_t dt) {
		if (!material.is_valid()) return;
		real_t alpha = material->get_buoyancy_alpha();
		real_t beta = material->get_buoyancy_beta();
		real_t T_amb = material->get_ambient_temperature();

		for (int k = 1; k <= N; ++k) {
			for (int j = 1; j <= N; ++j) {
				for (int i = 1; i <= N; ++i) {
					int idx = IX(i, j, k);
					// Buoyancy: f = alpha * density - beta * (T - T_amb)
					real_t buoy = alpha * density[idx] + beta * (temperature[idx] - T_amb);
					// Apply as vertical acceleration (y-up)
					v[idx] += buoy * dt;
				}
			}
		}
	}

	void diffuse_velocity(real_t dt, real_t visc) {
		// Simple explicit Euler: v_new = v + dt * visc * laplacian(v)
		copy_velocity(u, u_prev);
		copy_velocity(v, v_prev);
		copy_velocity(w, w_prev);
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

	void advect_velocity(real_t dt) {
		// Save old velocity
		copy_velocity(u, u_prev);
		copy_velocity(v, v_prev);
		copy_velocity(w, w_prev);

		for (int k = 1; k <= N; ++k) {
			for (int j = 1; j <= N; ++j) {
				for (int i = 1; i <= N; ++i) {
					int idx = IX(i, j, k);
					// Backtrace
					real_t x = i - dt * u_prev[idx] * inv_h;
					real_t y = j - dt * v_prev[idx] * inv_h;
					real_t z = k - dt * w_prev[idx] * inv_h;
					x = CLAMP(x, 0.5, N + 0.5);
					y = CLAMP(y, 0.5, N + 0.5);
					z = CLAMP(z, 0.5, N + 0.5);
					u[idx] = interpolate_velocity(x, y, z, 0);
					v[idx] = interpolate_velocity(x, y, z, 1);
					w[idx] = interpolate_velocity(x, y, z, 2);
				}
			}
		}
		set_boundary_velocity();
	}

	void advect_velocity_maccormack(real_t dt) {
		// First pass: advect forward (same as semi-Lagrange)
		copy_velocity(u, u_prev);
		copy_velocity(v, v_prev);
		copy_velocity(w, w_prev);

		// Temporary forward advected arrays
		LocalVector<real_t> u_hat, v_hat, w_hat;
		u_hat.resize(u.size()); v_hat.resize(v.size()); w_hat.resize(w.size());

		for (int k = 1; k <= N; ++k) {
			for (int j = 1; j <= N; ++j) {
				for (int i = 1; i <= N; ++i) {
					int idx = IX(i, j, k);
					real_t x = i - dt * u_prev[idx] * inv_h;
					real_t y = j - dt * v_prev[idx] * inv_h;
					real_t z = k - dt * w_prev[idx] * inv_h;
					x = CLAMP(x, 0.5, N + 0.5);
					y = CLAMP(y, 0.5, N + 0.5);
					z = CLAMP(z, 0.5, N + 0.5);
					u_hat[idx] = interpolate_velocity(x, y, z, 0);
					v_hat[idx] = interpolate_velocity(x, y, z, 1);
					w_hat[idx] = interpolate_velocity(x, y, z, 2);
				}
			}
		}

		// Second pass: advect backward from hat to get error estimate
		LocalVector<real_t> u_hat_hat, v_hat_hat, w_hat_hat;
		u_hat_hat.resize(u.size()); v_hat_hat.resize(v.size()); w_hat_hat.resize(w.size());

		for (int k = 1; k <= N; ++k) {
			for (int j = 1; j <= N; ++j) {
				for (int i = 1; i <= N; ++i) {
					int idx = IX(i, j, k);
					real_t x = i + dt * u_hat[idx] * inv_h; // backward
					real_t y = j + dt * v_hat[idx] * inv_h;
					real_t z = k + dt * w_hat[idx] * inv_h;
					x = CLAMP(x, 0.5, N + 0.5);
					y = CLAMP(y, 0.5, N + 0.5);
					z = CLAMP(z, 0.5, N + 0.5);
					u_hat_hat[idx] = interpolate_velocity(x, y, z, 0);
					v_hat_hat[idx] = interpolate_velocity(x, y, z, 1);
					w_hat_hat[idx] = interpolate_velocity(x, y, z, 2);
				}
			}
		}

		// Correct: u = u_hat + 0.5*(u_prev - u_hat_hat)
		for (int i = 0; i < u.size(); ++i) {
			u[i] = u_hat[i] + 0.5 * (u_prev[i] - u_hat_hat[i]);
			v[i] = v_hat[i] + 0.5 * (v_prev[i] - v_hat_hat[i]);
			w[i] = w_hat[i] + 0.5 * (w_prev[i] - w_hat_hat[i]);
		}

		set_boundary_velocity();
	}

	void project(real_t dt) {
		// Compute negative divergence
		for (int k = 1; k <= N; ++k) {
			for (int j = 1; j <= N; ++j) {
				for (int i = 1; i <= N; ++i) {
					int idx = IX(i, j, k);
					divergence[idx] = -0.5 * h * (
						u[IX(i+1,j,k)] - u[IX(i-1,j,k)] +
						v[IX(i,j+1,k)] - v[IX(i,j-1,k)] +
						w[IX(i,j,k+1)] - w[IX(i,j,k-1)]);
					pressure[idx] = 0;
				}
			}
		}
		set_boundary_pressure();

		// Gauss-Seidel solver for Poisson equation
		int max_iter = material.is_valid() ? material->get_pressure_solver_iter() : 20;
		for (int iter = 0; iter < max_iter; ++iter) {
			for (int k = 1; k <= N; ++k) {
				for (int j = 1; j <= N; ++j) {
					for (int i = 1; i <= N; ++i) {
						int idx = IX(i, j, k);
						pressure[idx] = (divergence[idx] +
							pressure[IX(i-1,j,k)] + pressure[IX(i+1,j,k)] +
							pressure[IX(i,j-1,k)] + pressure[IX(i,j+1,k)] +
							pressure[IX(i,j,k-1)] + pressure[IX(i,j,k+1)]) / 6.0;
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
					u[idx] -= 0.5 * (pressure[idx+1] - pressure[idx-1]) / h;
					v[idx] -= 0.5 * (pressure[IX(i,j+1,k)] - pressure[IX(i,j-1,k)]) / h;
					w[idx] -= 0.5 * (pressure[IX(i,j,k+1)] - pressure[IX(i,j,k-1)]) / h;
				}
			}
		}
		set_boundary_velocity();
	}

	// --- Scalar field step (density, temperature) ---
	void scalar_step(real_t dt) {
		// Advect density
		SWAP(density, density_prev);
		for (int k = 1; k <= N; ++k) {
			for (int j = 1; j <= N; ++j) {
				for (int i = 1; i <= N; ++i) {
					int idx = IX(i, j, k);
					real_t x = i - dt * u[idx] * inv_h;
					real_t y = j - dt * v[idx] * inv_h;
					real_t z = k - dt * w[idx] * inv_h;
					x = CLAMP(x, 0.5, N + 0.5);
					y = CLAMP(y, 0.5, N + 0.5);
					z = CLAMP(z, 0.5, N + 0.5);
					density[idx] = interpolate_scalar(x, y, z, density_prev);
				}
			}
		}
		// Advect temperature similarly
		SWAP(temperature, temperature_prev);
		for (int k = 1; k <= N; ++k) {
			for (int j = 1; j <= N; ++j) {
				for (int i = 1; i <= N; ++i) {
					int idx = IX(i, j, k);
					real_t x = i - dt * u[idx] * inv_h;
					real_t y = j - dt * v[idx] * inv_h;
					real_t z = k - dt * w[idx] * inv_h;
					x = CLAMP(x, 0.5, N + 0.5);
					y = CLAMP(y, 0.5, N + 0.5);
					z = CLAMP(z, 0.5, N + 0.5);
					temperature[idx] = interpolate_scalar(x, y, z, temperature_prev);
				}
			}
		}
	}

	// --- Utility functions ---
	void copy_velocity(const LocalVector<real_t> &src, LocalVector<real_t> &dst) {
		for (int i = 0; i < src.size(); ++i) dst[i] = src[i];
	}

	real_t laplacian(int i, int j, int k, const LocalVector<real_t> &field, int component) const {
		int idx = IX(i, j, k);
		real_t sum = 0;
		sum += field[IX(i+1,j,k)] - 2*field[idx] + field[IX(i-1,j,k)];
		sum += field[IX(i,j+1,k)] - 2*field[idx] + field[IX(i,j-1,k)];
		sum += field[IX(i,j,k+1)] - 2*field[idx] + field[IX(i,j,k-1)];
		return sum / (h*h);
	}

	real_t interpolate_velocity(real_t x, real_t y, real_t z, int comp) const {
		int i0 = (int)x, j0 = (int)y, k0 = (int)z;
		int i1 = i0+1, j1 = j0+1, k1 = k0+1;
		real_t s1 = x - i0, s0 = 1 - s1;
		real_t t1 = y - j0, t0 = 1 - t1;
		real_t u1 = z - k0, u0 = 1 - u1;
		i0 = CLAMP(i0, 0, N+1); i1 = CLAMP(i1, 0, N+1);
		j0 = CLAMP(j0, 0, N+1); j1 = CLAMP(j1, 0, N+1);
		k0 = CLAMP(k0, 0, N+1); k1 = CLAMP(k1, 0, N+1);
		const LocalVector<real_t> *arr = &u;
		if (comp == 1) arr = &v;
		if (comp == 2) arr = &w;
		return s0*(t0*(u0*(*arr)[IX(i0,j0,k0)] + u1*(*arr)[IX(i0,j0,k1)]) +
				   t1*(u0*(*arr)[IX(i0,j1,k0)] + u1*(*arr)[IX(i0,j1,k1)])) +
			   s1*(t0*(u0*(*arr)[IX(i1,j0,k0)] + u1*(*arr)[IX(i1,j0,k1)]) +
				   t1*(u0*(*arr)[IX(i1,j1,k0)] + u1*(*arr)[IX(i1,j1,k1)]));
	}

	real_t interpolate_scalar(real_t x, real_t y, real_t z, const LocalVector<real_t> &field) const {
		int i0 = (int)x, j0 = (int)y, k0 = (int)z;
		int i1 = i0+1, j1 = j0+1, k1 = k0+1;
		real_t s1 = x - i0, s0 = 1 - s1;
		real_t t1 = y - j0, t0 = 1 - t1;
		real_t u1 = z - k0, u0 = 1 - u1;
		i0 = CLAMP(i0, 0, N+1); i1 = CLAMP(i1, 0, N+1);
		j0 = CLAMP(j0, 0, N+1); j1 = CLAMP(j1, 0, N+1);
		k0 = CLAMP(k0, 0, N+1); k1 = CLAMP(k1, 0, N+1);
		return s0*(t0*(u0*field[IX(i0,j0,k0)] + u1*field[IX(i0,j0,k1)]) +
				   t1*(u0*field[IX(i0,j1,k0)] + u1*field[IX(i0,j1,k1)])) +
			   s1*(t0*(u0*field[IX(i1,j0,k0)] + u1*field[IX(i1,j0,k1)]) +
				   t1*(u0*field[IX(i1,j1,k0)] + u1*field[IX(i1,j1,k1)]));
	}

	void set_boundary_velocity() {
		// Set boundary conditions for velocity (no-slip on domain walls)
		for (int k = 0; k <= N+1; ++k) {
			for (int j = 0; j <= N+1; ++j) {
				// X boundaries
				u[IX(0,j,k)] = 0; u[IX(N+1,j,k)] = 0;
				v[IX(0,j,k)] = -v[IX(1,j,k)]; v[IX(N+1,j,k)] = -v[IX(N,j,k)];
				w[IX(0,j,k)] = -w[IX(1,j,k)]; w[IX(N+1,j,k)] = -w[IX(N,j,k)];
			}
		}
		for (int k = 0; k <= N+1; ++k) {
			for (int i = 0; i <= N+1; ++i) {
				// Y boundaries
				v[IX(i,0,k)] = 0; v[IX(i,N+1,k)] = 0;
				u[IX(i,0,k)] = -u[IX(i,1,k)]; u[IX(i,N+1,k)] = -u[IX(i,N,k)];
				w[IX(i,0,k)] = -w[IX(i,1,k)]; w[IX(i,N+1,k)] = -w[IX(i,N,k)];
			}
		}
		for (int j = 0; j <= N+1; ++j) {
			for (int i = 0; i <= N+1; ++i) {
				// Z boundaries
				w[IX(i,j,0)] = 0; w[IX(i,j,N+1)] = 0;
				u[IX(i,j,0)] = -u[IX(i,j,1)]; u[IX(i,j,N+1)] = -u[IX(i,j,N)];
				v[IX(i,j,0)] = -v[IX(i,j,1)]; v[IX(i,j,N+1)] = -v[IX(i,j,N)];
			}
		}
	}

	void set_boundary_pressure() {
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

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_grid_resolution", "n"), &SFSolver::set_grid_resolution);
		ClassDB::bind_method(D_METHOD("set_material", "material"), &SFSolver::set_material);
		ClassDB::bind_method(D_METHOD("add_velocity", "i", "j", "k", "du", "dv", "dw"), &SFSolver::add_velocity);
		ClassDB::bind_method(D_METHOD("add_density", "i", "j", "k", "d"), &SFSolver::add_density);
		ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "SFMaterial"), "set_material", "get_material");
	}
};

} // namespace genesis

#endif // GENESIS_SOLVERS_SF_SOLVER_H