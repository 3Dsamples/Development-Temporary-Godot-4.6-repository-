--- START OF FILE core/math/flesh_tensor_physics.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * FleshLayer Coefficients (Deterministic Constants)
 * Defined in FixedMathCore to ensure identical deformation across hardware.
 */
struct FleshTensors {
	FixedMathCore skin_stiffness;      // Outer layer resistance
	FixedMathCore subcutaneous_fat;    // Damping and volume bulk
	FixedMathCore muscle_fiber_tension; // Anisotropic restoration
	FixedMathCore compression_limit;   // Non-linear limit (Balloon effect)
};

/**
 * Warp Kernel: SubDermalDynamicsKernel
 * 
 * Processes a batch of vertices to calculate multi-layer flesh response.
 * Uses a non-linear Hooke's law: force increases exponentially as 
 * displacement approaches the compression_limit.
 */
void subdermal_dynamics_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		const Vector3f &p_rest_position,
		const Vector3f &p_normal,
		const FleshTensors &p_tensors,
		const FixedMathCore &p_delta) {

	// 1. Calculate Displacement Vector
	Vector3f displacement = r_position - p_rest_position;
	FixedMathCore dist = displacement.length();
	
	if (dist.get_raw() == 0) return;

	// 2. Non-Linear Elasticity (The "Balloon" resistance)
	// As the displacement increases, the stiffness grows to prevent mesh collapse.
	// factor = stiffness / (limit - current_dist)
	FixedMathCore safety_margin = p_tensors.compression_limit - dist;
	if (safety_margin <= FixedMathCore(429496LL, true)) { // 0.0001 safety
		safety_margin = FixedMathCore(429496LL, true);
	}

	FixedMathCore effective_stiffness = p_tensors.skin_stiffness / safety_margin;
	Vector3f restoration_force = displacement * (-effective_stiffness);

	// 3. Muscle Fiber Anisotropy
	// Restoration is stronger along the surface normal to maintain "perkiness"
	FixedMathCore normal_projection = displacement.dot(p_normal);
	Vector3f normal_force = p_normal * (-normal_projection * p_tensors.muscle_fiber_tension);
	
	// 4. Viscoelastic Damping (Fat Layer absorption)
	// Simulates the energy loss of jiggling flesh.
	FixedMathCore damping_factor = p_tensors.subcutaneous_fat * p_delta;
	r_velocity *= (MathConstants<FixedMathCore>::one() - damping_factor);

	// 5. Integration (Semi-Implicit Euler)
	Vector3f total_accel = (restoration_force + normal_force); // Mass assumed unit per vertex
	r_velocity += total_accel * p_delta;
	r_position += r_velocity * p_delta;
}

/**
 * resolve_flesh_deformation_sweep()
 * 
 * Master parallel sweep for anatomical body simulation.
 * Iterates through EnTT SoA streams for vertices, rest-states, and normals.
 */
void resolve_flesh_deformation_sweep(
		Vector3f *r_positions,
		Vector3f *r_velocities,
		const Vector3f *p_rest_positions,
		const Vector3f *p_normals,
		uint64_t p_count,
		const StringName &p_flesh_type,
		const FixedMathCore &p_delta) {

	// Set Layer Tensors based on Body Type
	FleshTensors tensors;
	if (p_flesh_type == SNAME("breast")) {
		tensors.skin_stiffness = FixedMathCore(2147483648LL, true);      // 0.5
		tensors.subcutaneous_fat = FixedMathCore(858993459LL, true);    // 0.2
		tensors.muscle_fiber_tension = FixedMathCore(429496729LL, true); // 0.1
		tensors.compression_limit = FixedMathCore(2LL, false);           // 2.0 units
	} else if (p_flesh_type == SNAME("buttock")) {
		tensors.skin_stiffness = FixedMathCore(4294967296LL, false);     // 1.0
		tensors.subcutaneous_fat = FixedMathCore(1288490188LL, true);    // 0.3
		tensors.muscle_fiber_tension = FixedMathCore(2147483648LL, true); // 0.5
		tensors.compression_limit = FixedMathCore(1LL, false);           // 1.0 unit
	} else {
		// Generic muscle
		tensors.skin_stiffness = FixedMathCore(10LL, false);
		tensors.subcutaneous_fat = FixedMathCore(2147483648LL, true);
		tensors.muscle_fiber_tension = FixedMathCore(5LL, false);
		tensors.compression_limit = FixedMathCore(429496729LL, true);    // 0.1 (Rigid)
	}

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = p_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &tensors]() {
			for (uint64_t i = start; i < end; i++) {
				subdermal_dynamics_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_positions[i],
					r_velocities[i],
					p_rest_positions[i],
					p_normals[i],
					tensors,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_pinch_tensor_interaction()
 * 
 * Simulates a two-point compression (Pinch). 
 * Directly modifies the velocity stream to create localized "bulging"
 * around the pinch points to maintain volume integrity.
 */
void apply_pinch_tensor_interaction(
		Vector3f *r_velocities,
		const Vector3f *p_positions,
		uint64_t p_count,
		const Vector3f &p_finger_a,
		const Vector3f &p_finger_b,
		const FixedMathCore &p_pinch_force) {

	Vector3f midpoint = (p_finger_a + p_finger_b) * MathConstants<FixedMathCore>::half();
	FixedMathCore pinch_dist = (p_finger_a - p_finger_b).length();
	FixedMathCore radius = pinch_dist * FixedMathCore(2LL, false);

	for (uint64_t i = 0; i < p_count; i++) {
		Vector3f to_mid = midpoint - p_positions[i];
		FixedMathCore d2 = to_mid.length_squared();
		
		if (d2 < (radius * radius)) {
			FixedMathCore d = Math::sqrt(d2);
			FixedMathCore weight = (radius - d) / radius;
			
			// Move toward midpoint (Compression)
			r_velocities[i] += to_mid.normalized() * (p_pinch_force * weight);
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/flesh_tensor_physics.cpp ---
