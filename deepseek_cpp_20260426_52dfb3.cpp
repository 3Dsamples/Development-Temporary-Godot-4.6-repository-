// File 135: modules/gaia/src/solver_utils/gd_solver_utilities.h
// Gradient‑descent and conjugate‑gradient linear solvers for symmetric
// positive‑definite systems arising from implicit FEM / VBD.
// These operate on arrays of Vector3 (per‑vertex unknowns) and use
// matrix‑free apply functions.

#ifndef GAIA_SOLVER_UTILS_GD_SOLVER_UTILITIES_H
#define GAIA_SOLVER_UTILS_GD_SOLVER_UTILITIES_H

#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"
#include <cmath>

namespace gaia::solver_utils {

/**
 * Conjugate Gradient solver for A x = b.
 * `x` is the initial guess and the solution on exit.
 * `apply_A` is a callable with signature:
 *   void(const LocalVector<Vector3>& src, LocalVector<Vector3>& dst)
 * It computes y = A * src.
 */
template <typename ApplyFunc>
class ConjugateGradient {
public:
	int    max_iterations = 200;
	real_t relative_tolerance = 1e-6;
	real_t absolute_tolerance = 1e-12;

	// Solve A x = b. Returns the number of iterations, or -1 if failed.
	int solve(LocalVector<Vector3> &x,
			  const LocalVector<Vector3> &b,
			  ApplyFunc apply_A) const {
		int n = b.size();
		ERR_FAIL_COND_V(x.size() != n, -1);

		LocalVector<Vector3> r(n), p(n), Ap(n), z(n);
		// r = b - A*x
		apply_A(x, Ap);
		real_t r_norm_sq = 0.0;
		for (int i = 0; i < n; ++i) {
			r[i] = b[i] - Ap[i];
			r_norm_sq += r[i].length_squared();
		}
		real_t b_norm_sq = 0.0;
		for (int i = 0; i < n; ++i) b_norm_sq += b[i].length_squared();
		real_t tol = MAX(relative_tolerance * relative_tolerance * b_norm_sq,
						 absolute_tolerance * absolute_tolerance);

		if (r_norm_sq <= tol) return 0; // already converged

		// p = r (no preconditioner; use identity)
		for (int i = 0; i < n; ++i) p[i] = r[i];
		real_t rs_old = r_norm_sq;

		for (int iter = 1; iter <= max_iterations; ++iter) {
			// Ap = A * p
			apply_A(p, Ap);
			// alpha = (r^T r) / (p^T Ap)
			real_t pAp = 0.0;
			for (int i = 0; i < n; ++i) pAp += p[i].dot(Ap[i]);
			if (pAp <= CMP_EPSILON) break; // A is singular or not SPD

			real_t alpha = rs_old / pAp;
			// x += alpha * p
			// r -= alpha * Ap
			real_t r_norm_sq_new = 0.0;
			for (int i = 0; i < n; ++i) {
				x[i] += alpha * p[i];
				r[i] -= alpha * Ap[i];
				r_norm_sq_new += r[i].length_squared();
			}
			if (r_norm_sq_new <= tol) return iter; // converged

			// beta = (r_new^T r_new) / (r_old^T r_old)
			real_t beta = r_norm_sq_new / rs_old;
			// p = r + beta * p
			for (int i = 0; i < n; ++i) {
				p[i] = r[i] + beta * p[i];
			}
			rs_old = r_norm_sq_new;
		}
		return -1; // did not converge
	}
};

/**
 * Diagonal Jacobi preconditioner for CG.
 * `diag` contains the diagonal entries (3x3 blocks) of A.
 * The apply function is the same A.
 */
template <typename ApplyFunc>
class DiagonalPreconditionedCG {
public:
	int    max_iterations = 200;
	real_t relative_tolerance = 1e-6;
	real_t absolute_tolerance = 1e-12;

	int solve(LocalVector<Vector3> &x,
			  const LocalVector<Vector3> &b,
			  ApplyFunc apply_A,
			  const LocalVector<Basis> &diag_inv) const {
		int n = b.size();
		ERR_FAIL_COND_V(x.size() != n || diag_inv.size() != n, -1);

		LocalVector<Vector3> r(n), p(n), Ap(n), z(n);
		apply_A(x, Ap);
		real_t r_norm_sq = 0.0;
		for (int i = 0; i < n; ++i) {
			r[i] = b[i] - Ap[i];
			r_norm_sq += r[i].length_squared();
		}
		real_t b_norm_sq = 0.0;
		for (int i = 0; i < n; ++i) b_norm_sq += b[i].length_squared();
		real_t tol = MAX(relative_tolerance * relative_tolerance * b_norm_sq,
						 absolute_tolerance * absolute_tolerance);

		if (r_norm_sq <= tol) return 0;

		// z = M^{-1} r  (M = diag(A))
		for (int i = 0; i < n; ++i) z[i] = diag_inv[i].xform(r[i]);
		// p = z
		for (int i = 0; i < n; ++i) p[i] = z[i];
		real_t rz_old = 0.0;
		for (int i = 0; i < n; ++i) rz_old += r[i].dot(z[i]);

		for (int iter = 1; iter <= max_iterations; ++iter) {
			apply_A(p, Ap);
			real_t pAp = 0.0;
			for (int i = 0; i < n; ++i) pAp += p[i].dot(Ap[i]);
			if (pAp <= CMP_EPSILON) break;

			real_t alpha = rz_old / pAp;
			real_t r_norm_sq_new = 0.0;
			for (int i = 0; i < n; ++i) {
				x[i] += alpha * p[i];
				r[i] -= alpha * Ap[i];
				r_norm_sq_new += r[i].length_squared();
			}
			if (r_norm_sq_new <= tol) return iter;

			for (int i = 0; i < n; ++i) z[i] = diag_inv[i].xform(r[i]);
			real_t rz_new = 0.0;
			for (int i = 0; i < n; ++i) rz_new += r[i].dot(z[i]);
			real_t beta = rz_new / rz_old;
			for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
			rz_old = rz_new;
		}
		return -1;
	}
};

} // namespace gaia::solver_utils

#endif // GAIA_SOLVER_UTILS_GD_SOLVER_UTILITIES_H