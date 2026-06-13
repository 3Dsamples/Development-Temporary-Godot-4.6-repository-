// File 81: modules/genesis/src/grad/creation_ops.h
// Tensor creation and reduction operations for differentiable physics.
// Works with the minimalist Tensor type defined in tensor.h.

#ifndef GENESIS_GRAD_CREATION_OPS_H
#define GENESIS_GRAD_CREATION_OPS_H

#include "tensor.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"

namespace genesis::grad {

/**
 * Factory and helper functions that produce Tensors from common data.
 */
namespace creation {

	inline Tensor constant(real_t value) { return Tensor(value, 0.0); }
	inline Tensor zero() { return Tensor(0.0, 0.0); }
	inline Tensor one()  { return Tensor(1.0, 0.0); }

	inline Tensor from_vector(const Vector3 &v, int component) {
		ERR_FAIL_INDEX_V(component, 3, Tensor());
		return Tensor(v[component], 0.0);
	}

	// Convert a whole Vector3 into three separate Tensors (returns array of 3)
	inline void vector_to_tensors(const Vector3 &v, Tensor out[3]) {
		out[0] = Tensor(v.x, 0.0);
		out[1] = Tensor(v.y, 0.0);
		out[2] = Tensor(v.z, 0.0);
	}

	// Sum of a list of tensors (recorded on a tape if desired)
	inline Tensor sum(const LocalVector<Tensor> &tensors) {
		real_t s = 0.0;
		for (const Tensor &t : tensors) s += t.value;
		return Tensor(s, 0.0);
	}

	// Simple moving average
	inline Tensor moving_average(const LocalVector<Tensor> &tensors) {
		if (tensors.is_empty()) return zero();
		real_t avg = 0.0;
		for (const Tensor &t : tensors) avg += t.value;
		return Tensor(avg / real_t(tensors.size()), 0.0);
	}

	// MSE loss between two lists of tensors (prediction vs target)
	inline Tensor mse_loss(const LocalVector<Tensor> &pred,
						   const LocalVector<Tensor> &target) {
		ERR_FAIL_COND_V(pred.size() != target.size(), Tensor(INFINITY, 0.0));
		real_t loss = 0.0;
		for (int i = 0; i < pred.size(); ++i) {
			real_t diff = pred[i].value - target[i].value;
			loss += diff * diff;
		}
		return Tensor(loss / real_t(pred.size()), 0.0);
	}

} // namespace creation
} // namespace genesis::grad

#endif // GENESIS_GRAD_CREATION_OPS_H