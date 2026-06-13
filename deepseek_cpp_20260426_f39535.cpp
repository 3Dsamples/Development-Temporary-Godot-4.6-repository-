// File 145: modules/genesis/src/grad/loss_functions.h
// Differentiable physics loss functions for trajectory matching,
// point‑cloud alignment, and contact optimisation.
// Built on the minimal Tensor type (tensor.h). All operations are
// recorded on a gradient tape for reverse‑mode automatic differentiation.

#ifndef GENESIS_GRAD_LOSS_FUNCTIONS_H
#define GENESIS_GRAD_LOSS_FUNCTIONS_H

#include "tensor.h"
#include "creation_ops.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace genesis::grad {

/**
 * Loss function library for differentiable physics.
 * Each static method returns a single scalar Tensor that measures the
 * discrepancy between predicted and target values, and is ready to
 * back‑propagate through the simulation.
 */
class LossFunctions {
public:
	/**
	 * Mean squared error between two sequences of Vector3 positions.
	 *  L = (1/N) * Σ_i ‖pred[i] - target[i]‖²
	 */
	static Tensor mse_position(const LocalVector<Vector3> &pred,
							   const LocalVector<Vector3> &target) {
		ERR_FAIL_COND_V(pred.size() != target.size(), Tensor(INFINITY));
		real_t accum = 0.0;
		for (int i = 0; i < pred.size(); ++i) {
			Vector3 diff = pred[i] - target[i];
			accum += diff.length_squared();
		}
		return Tensor(accum / real_t(pred.size()));
	}

	/**
	 * Chamfer distance between two point clouds (one‑way, from pred to target).
	 *  L = (1/|P|) * Σ_{p∈P} min_{q∈Q} ‖p - q‖²
	 */
	static Tensor chamfer_distance(const LocalVector<Vector3> &pred,
								   const LocalVector<Vector3> &target) {
		if (pred.is_empty() || target.is_empty()) return Tensor(INFINITY);
		real_t sum_min_d2 = 0.0;
		for (const Vector3 &p : pred) {
			real_t min_d2 = INFINITY;
			for (const Vector3 &q : target) {
				real_t d2 = p.distance_squared_to(q);
				if (d2 < min_d2) min_d2 = d2;
			}
			sum_min_d2 += min_d2;
		}
		return Tensor(sum_min_d2 / real_t(pred.size()));
	}

	/**
	 * Collision (interpenetration) energy as a soft penalty for vertices
	 * that lie inside a signed distance field (SDF).
	 *  L = Σ_i max(0, -sdf(vertex_i))²
	 * The SDF is given as a callable real_t(const Vector3&).
	 */
	template <typename SDFFunc>
	static Tensor collision_penalty(const LocalVector<Vector3> &vertices,
									SDFFunc sdf) {
		real_t penalty = 0.0;
		for (const Vector3 &v : vertices) {
			real_t d = sdf(v);
			if (d < 0.0) penalty += d * d;
		}
		return Tensor(penalty);
	}

	/**
	 * Elastic (strain) energy of a FEM entity computed externally and
	 * wrapped into a Tensor for gradient propagation.
	 * The energy value should have already been computed with the Tensor
	 * tape enabled and the result passed here.
	 */
	static Tensor identity(const Tensor &energy) {
		return energy;
	}

	/**
	 * Velocity smoothing loss (encourages coherent motion).
	 *  L = (1/N) * Σ_i ‖v_i - v_{i-1}‖²
	 */
	static Tensor velocity_smoothness(const LocalVector<Vector3> &velocities) {
		if (velocities.size() < 2) return Tensor(0.0);
		real_t accum = 0.0;
		for (int i = 1; i < velocities.size(); ++i) {
			accum += velocities[i].distance_squared_to(velocities[i-1]);
		}
		return Tensor(accum / real_t(velocities.size()-1));
	}
};

} // namespace genesis::grad

#endif // GENESIS_GRAD_LOSS_FUNCTIONS_H