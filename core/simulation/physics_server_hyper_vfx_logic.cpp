--- START OF FILE core/simulation/physics_server_hyper_vfx_logic.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ParticlePhysicsKernel
 * 
 * Performs the fundamental physical update for a single VFX particle.
 * 1. Kinematics: integrates velocity and position using bit-perfect FixedMath.
 * 2. Lifetime: Decays the particle's existence based on delta.
 * 3. Drag: Applies atmospheric or vacuum resistance tensors.
 */
void particle_physics_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		FixedMathCore &r_lifetime,
		const FixedMathCore &p_drag_coeff,
		const Vector3f &p_gravity,
		const FixedMathCore &p_delta) {

	if (r_lifetime.get_raw() <= 0) return;

	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Resolve Velocity (Euler Integration)
	// a = gravity + drag
	FixedMathCore speed = r_velocity.length();
	Vector3f f_drag;
	if (speed.get_raw() > 0) {
		// F_drag = -v * v * coefficient
		f_drag = r_velocity.normalized() * (speed * speed * p_drag_coeff * (-one));
	}
	
	Vector3f acceleration = p_gravity + f_drag;
	r_velocity += acceleration * p_delta;

	// 2. Resolve Position
	r_position += r_velocity * p_delta;

	// 3. Resolve Lifetime
	r_lifetime -= p_delta;
}

/**
 * Warp Kernel: IonizationSpectralKernel
 * 
 * Sophisticated Behavior: Simulates the "Glow" of high-energy particles.
 * 1. Temperature: Particles heat up based on velocity (Friction).
 * 2. Incandescence: Color shifts from Red to Blue based on thermal state.
 * 3. Anime Style: Snaps spectral energy to sharp bands for cel-shaded effects.
 */
void ionization_spectral_kernel(
		const BigIntCore &p_index,
		Vector3f &r_spectral_radiance,
		const Vector3f &p_velocity,
		const FixedMathCore &p_lifetime,
		const FixedMathCore &p_max_lifetime,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Calculate Kinetic Temperature approx
	FixedMathCore energy = p_velocity.length_squared() * FixedMathCore("0.001");
	FixedMathCore life_ratio = p_lifetime / p_max_lifetime;
	
	// 2. Resolve Spectral Color (Red-Hot to Blue-Ionized)
	FixedMathCore blue_factor = wp::clamp(energy, zero, one);
	Vector3f color_tensor = wp::lerp(Vector3f(one, zero, zero), Vector3f(zero, one, one), blue_factor);

	// 3. --- Sophisticated Behavior: Anime Light-Banding ---
	if (p_is_anime) {
		// Technique: "Luminance Tiers". 
		// Instead of a smooth fade, particles flicker and snap between intensity levels.
		FixedMathCore pulse = wp::sin(p_lifetime * FixedMathCore(20LL));
		FixedMathCore intensity = life_ratio * (one + pulse * FixedMathCore("0.2"));
		
		FixedMathCore band = wp::step(FixedMathCore("0.7"), intensity) * one + 
		                    wp::step(FixedMathCore("0.3"), intensity) * FixedMathCore("0.4");
		
		r_spectral_radiance = color_tensor * band;
	} else {
		// Realistic Path: smooth attenuation
		r_spectral_radiance = color_tensor * life_ratio;
	}
}

/**
 * Warp Kernel: RelativisticTrailKernel
 * 
 * Special Feature: Length contraction and aberration for particle trails 
 * emitted by high-speed spaceships.
 */
void relativistic_trail_kernel(
		Vector3f &r_pos,
		const Vector3f &p_velocity,
		const FixedMathCore &p_c_sq) {
	
	FixedMathCore v2 = p_velocity.length_squared();
	FixedMathCore gamma = MathConstants<FixedMathCore>::one() / (MathConstants<FixedMathCore>::one() - (v2 / p_c_sq)).square_root();
	
	// Apply length contraction along the velocity vector
	Vector3f n = p_velocity.normalized();
	FixedMathCore projection = r_pos.dot(n);
	r_pos -= n * (projection * (MathConstants<FixedMathCore>::one() - (MathConstants<FixedMathCore>::one() / gamma)));
}

/**
 * execute_vfx_logic_wave()
 * 
 * Master orchestrator for parallel 120 FPS effect simulation.
 * Partitions EnTT component streams for Position, Velocity, and Radiance.
 */
void PhysicsServerHyper::execute_vfx_logic_wave(const FixedMathCore &p_delta) {
	KernelRegistry &registry = get_registry();
	auto &pos_stream = registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &life_stream = registry.get_stream<FixedMathCore>(COMPONENT_LIFETIME);
	auto &rad_stream = registry.get_stream<Vector3f>(COMPONENT_SPECTRAL_RADIANCE);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	FixedMathCore c_sq = PHYSICS_C * PHYSICS_C;
	FixedMathCore drag("0.01");

	// PASS 1: Parallel Physics Integration
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &vel_stream, &life_stream]() {
			for (uint64_t i = start; i < end; i++) {
				particle_physics_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					pos_stream[i],
					vel_stream[i],
					life_stream[i],
					drag,
					gravity_vector,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	// PASS 2: Parallel Spectral Resolve
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &rad_stream, &vel_stream, &life_stream]() {
			for (uint64_t i = start; i < end; i++) {
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 8 == 0);

				ionization_spectral_kernel(
					handle,
					rad_stream[i],
					vel_stream[i],
					life_stream[i],
					FixedMathCore(5LL, false), // 5.0s max life
					anime_mode
				);

				// If moving at relativistic speed, warp the trail length
				if (vel_stream[i].length_squared() > (c_sq * FixedMathCore("0.01"))) {
					relativistic_trail_kernel(pos_stream[i], vel_stream[i], c_sq);
				}
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_vfx_logic.cpp ---
