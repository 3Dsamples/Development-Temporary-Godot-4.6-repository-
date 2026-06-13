// File 110: modules/gaia/src/solver_utils/chebyshev_accelerator.h
// Chebyshev semi-iterative accelerator for solving Ax = b.
// Given eigenvalue bounds [lambda_min, lambda_max] of the symmetric positive-definite
// matrix A, this routine accelerates a given fixed-point iteration (e.g., Jacobi)
// using Chebyshev polynomials. Used as a linear-solver preconditioner in implicit
// FEM/VBD Newton steps.

#ifndef GAIA_SOLVER_UTILS_CHEBYSHEV_ACCELERATOR_H
#define GAIA_SOLVER_UTILS_CHEBYSHEV_ACCELERATOR_H

#include "core/typedefs.h"
#include "core/templates/local_vector.h"
#include <cmath>

namespace gaia::solver_utils {

/**
 * ChebyshevAccelerator solves Ax = b using the Chebyshev semi-iterative method.
 *
 * The caller provides:
 *  - `b`: right-hand side vector (size n)
 *  - `x0`: initial guess (size n) – will be updated in-place.
 *  - `apply_A`: a callable with signature void(const LocalVector<real_t>&, LocalVector<real_t>&)
 *               that computes y = A * x.
 *  - `lambda_min`, `lambda_max`: estimates of the smallest and largest eigenvalues of A.
 *
 * The method performs `iterations` steps and stores the result in `x0`.
 */
class ChebyshevAccelerator {
public:
	// Default parameters
	int max_iterations = 50;
	real_t eigenvalue_lower = 0.1;
	real_t eigenvalue_upper = 1.0;
	real_t relative_tolerance = 1e-6;

	ChebyshevAccelerator() {}

	// Set eigenvalue bounds (must be positive and lambda_max > lambda_min).
	void set_eigenvalue_bounds(real_t p_lambda_min, real_t p_lambda_max) {
		ERR_FAIL_COND(p_lambda_min <= 0 || p_lambda_max <= p_lambda_min);
		eigenvalue_lower = p_lambda_min;
		eigenvalue_upper = p_lambda_max;
	}

	// Perform Chebyshev acceleration.
	// `x` is both the initial guess and the final solution.
	// `b` is the RHS.
	// `apply_A` is a function (or lambda) that computes y = A * x.
	template <typename ApplyFunc>
	void solve(LocalVector<real_t> &x, const LocalVector<real_t> &b, ApplyFunc apply_A) const {
		int n = b.size();
		ERR_FAIL_COND(x.size() != n);

		LocalVector<real_t> r(n);  // residual
		LocalVector<real_t> p(n);  // search direction
		LocalVector<real_t> Ap(n); // matrix-vector product

		// Precompute constants for the Chebyshev iteration
		real_t d = (eigenvalue_upper + eigenvalue_lower) * 0.5;
		real_t c = (eigenvalue_upper - eigenvalue_lower) * 0.5;

		// Initial residual: r0 = b - A x0
		apply_A(x, Ap);
		for (int i = 0; i < n; ++i) {
			r[i] = b[i] - Ap[i];
		}

		// p0 = r0
		for (int i = 0; i < n; ++i) {
			p[i] = r[i];
		}

		real_t rho_prev = 1.0 / d; // rho_{-1} = 1/d in standard formulation
		real_t alpha, beta;

		for (int k = 0; k < max_iterations; ++k) {
			// Compute Ap = A * p
			apply_A(p, Ap);

			// alpha_k = (r_k, r_k) / (p_k, A p_k) – but Chebyshev uses fixed coefficients.
			// Instead, we use the three-term recurrence:
			//   x_{k+1} = x_k + alpha_k * p_k
			//   r_{k+1} = r_k - alpha_k * A p_k
			//   p_{k+1} = r_{k+1} + beta_k * p_k
			//
			// For the Chebyshev method, alpha_k and beta_k are computed from
			// the eigenvalue bounds. We follow the standard algorithm (e.g., Gutknecht & Röllin).

			real_t rho = (k == 0) ? 1.0 : (1.0 / (d - (c * c * rho_prev * 0.25)));
			real_t gamma = (k == 0) ? 1.0 : (rho * rho * c * c * 0.25);

			alpha = rho;
			beta = gamma;

			// Update solution and residual
			for (int i = 0; i < n; ++i) {
				x[i] += alpha * p[i];
				r[i] -= alpha * Ap[i];
			}

			// Check residual norm
			real_t r_norm_sq = 0.0;
			for (int i = 0; i < n; ++i) r_norm_sq += r[i] * r[i];
			if (r_norm_sq < relative_tolerance * relative_tolerance)
				break;

			// Update search direction
			for (int i = 0; i < n; ++i) {
				p[i] = r[i] + beta * p[i];
			}

			rho_prev = rho;
		}
	}
};

} // namespace gaia::solver_utils

#endif // GAIA_SOLVER_UTILS_CHEBYSHEV_ACCELERATOR_H