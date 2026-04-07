--- START OF FILE core/simulation/physics_server_hyper_acoustics_engine.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/collision_solver.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: AcousticWaveformIntegrationKernel
 * 
 * Resolves the physical state of a sound wave at the listener's position.
 * 1. Phase Alignment: Calculates the exact time-of-flight phase shift.
 * 2. Interference: Performs deterministic superposition of concurrent waves.
 * 3. Diffraction: Checks for geometric obstructions and applies the bending tensor.
 */
void acoustic_waveform_integration_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_final_amplitude,
		const Vector3f &p_listener_pos,
		const Vector3f &p_source_pos,
		const FixedMathCore &p_source_amp,
		const FixedMathCore &p_source_freq,
		const FixedMathCore &p_speed_of_sound,
		const BigIntCore &p_global_tick,
		const Face3f *p_obstructions,
		uint64_t p_obstruction_count,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Precise Distance Resolve
	Vector3f path_vec = p_listener_pos - p_source_pos;
	FixedMathCore distance = path_vec.length();
	if (distance.get_raw() == 0) return;

	// 2. Obstruction & Diffraction logic
	// Huygens Principle approximation: If blocked, sound "bends" around the nearest edge.
	FixedMathCore diffraction_loss = one;
	Vector3f ray_dir = path_vec / distance;
	
	for (uint64_t i = 0; i < p_obstruction_count; i++) {
		const Face3f &f = p_obstructions[i];
		Vector3f intersect;
		if (f.intersects_ray(p_source_pos, ray_dir, &intersect)) {
			FixedMathCore dist_to_hit = (intersect - p_source_pos).length();
			if (dist_to_hit < distance) {
				// Blocked: Apply frequency-dependent diffraction loss
				// High frequencies (Anime "Shing") are blocked more than low (Realistic "Thud")
				diffraction_loss = p_is_anime ? FixedMathCore(1288490188LL, true) : FixedMathCore(429496730LL, true); // 0.3 vs 0.1
				break;
			}
		}
	}

	// 3. Deterministic Phase Calculation
	// Phase = (2 * pi * freq * distance / speed) + (2 * pi * freq * time)
	FixedMathCore time_f(static_cast<int64_t>(std::stoll(p_global_tick.to_string())));
	FixedMathCore phase_offset = (Math::tau() * p_source_freq * distance) / p_speed_of_sound;
	FixedMathCore current_phase = (Math::tau() * p_source_freq * time_f) + phase_offset;

	// 4. Wave Interference (Superposition)
	// Output = Amplitude * sin(phase) * diffraction
	FixedMathCore wave_val = p_source_amp * wp::sin(current_phase) * diffraction_loss;
	
	// Inverse Square Law Attenuation
	r_final_amplitude += wave_val / (distance * distance + one);

	// --- Sophisticated Behavior: Sonic Boom / Anime Punch ---
	if (p_is_anime && p_source_amp > FixedMathCore(10LL, false)) {
		// Distort the waveform for high-energy events (Clipping/Crunch)
		if (r_final_amplitude > one) r_final_amplitude = one + (r_final_amplitude - one) * FixedMathCore(214748364LL, true); 
	}
}

/**
 * execute_acoustic_engine_sweep()
 * 
 * Orchestrates the parallel 120 FPS audio simulation.
 * Maps sources to listeners across galactic sectors using EnTT SoA.
 */
void PhysicsServerHyper::execute_acoustic_engine_sweep(const FixedMathCore &p_delta) {
	auto &registry = get_kernel_registry();
	uint64_t source_count = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_FREQ).size();
	if (source_count == 0) return;

	// ETEngine Strategy: Partition world geometry for high-speed diffraction checks
	// (Assumes localized Face3f obstruction stream provided by broadphase)

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = source_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? source_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &registry]() {
			for (uint64_t i = start; i < end; i++) {
				BigIntCore idx(static_cast<int64_t>(i));
				bool anime_style = (idx.hash() % 7 == 0);

				acoustic_waveform_integration_kernel(
					idx,
					registry.get_stream<FixedMathCore>(COMPONENT_FINAL_AUDIO_AMP)[i],
					registry.get_stream<Vector3f>(COMPONENT_LISTENER_POS)[0], // Primary listener
					registry.get_stream<Vector3f>(COMPONENT_SOURCE_POS)[i],
					registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_AMP)[i],
					registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_FREQ)[i],
					FixedMathCore(343LL, false), // Speed of sound
					SimulationManager::get_singleton()->get_total_frames(),
					registry.get_stream<Face3f>(COMPONENT_GEOMETRY).get_base_ptr(),
					registry.get_stream<Face3f>(COMPONENT_GEOMETRY).size(),
					anime_style
				);
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_acoustics_engine.cpp ---
