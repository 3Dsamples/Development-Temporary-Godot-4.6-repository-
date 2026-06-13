// File 147: modules/gaia/src/vbd_physics/vbd_physics_compute_cpu.h
// Dispatcher for VBD element solve: selects CPU parallel or optional CUDA
// kernel depending on build configuration. When CUDA is enabled, this
// header provides the kernel launch wrapper; otherwise it falls back to
// the CPU implementation in vbd_physics_compute.h.

#ifndef GAIA_VBD_PHYSICS_COMPUTE_CPU_H
#define GAIA_VBD_PHYSICS_COMPUTE_CPU_H

#include "vbd_physics_compute.h"          // CPU implementation
#include "../parallelization/cuda_utilities.h"   // cuda::malloc_device, etc.

namespace gaia::vbd {

class VBDPhysicsComputeDispatcher {
public:
	/**
	 * Solve a colour group of elements. If CUDA is available, the
	 * element data is uploaded, a kernel is launched, and results are
	 * downloaded. Otherwise the CPU fallback is used.
	 *
	 * @param d_positions      pointer to device positions (or nullptr if CPU)
	 * @param soft_body        the soft body (CPU data)
	 * @param elements         indices of elements to process
	 * @param vbd_constraints  all constraints (element‑wise)
	 * @param compliance       XPBD compliance
	 * @param dt               time step
	 */
	static void solve_colour_group(
#ifdef CUDA_ENABLED
			float* d_positions,
#endif
			gaia::SoftBody* soft_body,
			const LocalVector<int>& elements,
			const LocalVector<genesis::VBDConstraint*>& vbd_constraints,
			real_t compliance,
			real_t dt)
	{
#ifdef CUDA_ENABLED
		if (d_positions) {
			// Copy current positions to device
			int n_verts = soft_body->positions.size();
			cuda::memcpy_host_to_device(d_positions, soft_body->positions.ptr(),
										n_verts * sizeof(Vector3));
			// Launch kernel (placeholder: actual kernel not included here)
			// kernel_solve_blocks<<<...>>>(d_positions, ...);
			cuda::sync();
			// Copy result back
			cuda::memcpy_device_to_host(soft_body->positions.ptrw(), d_positions,
										n_verts * sizeof(Vector3));
			return;
		}
#endif
		// CPU fallback
		VBDPhysicsCompute::solve_colour_group(elements, vbd_constraints, compliance, dt);
	}
};

} // namespace gaia::vbd

#endif // GAIA_VBD_PHYSICS_COMPUTE_CPU_H