--- START OF FILE core/simulation/physics_server_hyper_acoustics_engine.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/collision_solver.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: AcousticWaveFieldKernel
 * 
 * Computes the physical state of the pressure field at a listener's coordinate.
 * 1. Euclidean Distance: Resolves the exact distance in bit-perfect FixedMath across sectors.
 * 2. Phase Resolve: Computes wave phase phi = (2 * pi * f * t) - (2 * pi * f * r / c).
 * 3. Diffraction: Checks for geometric occluders and applies frequency-dependent loss.
 * 4. Interference: Sums amplitudes using deterministic superposition.
 */
void acoustic_wave_field_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_final_amplitude,
		FixedMathCore &r_final_frequency,
		const Vector3f &p_listener_pos,
		const Vector3f &p_listener_vel,
		const BigIntCore &p_listener_sx, const BigIntCore &p_listener_sy, const BigIntCore &p_listener_sz,
		const Vector3f &p_source_pos,
		const Vector3f &p_source_vel,
		const BigIntCore &p_source_sx, const BigIntCore &p_source_sy, const BigIntCore &p_source_sz,
		const FixedMathCore &p_base_freq,
		const FixedMathCore &p_base_amp,
		const BigIntCore &p_global_tick,
		const FixedMathCore &p_speed_of_sound,
		const FixedMathCore &p_c_sq,
		const Face3f *p_obstructions,
		uint64_t p_obstruction_count,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Resolve Galactic Distance (Bit-Perfect Sector Alignment)
	Vector3f rel_vec = wp::calculate_galactic_relative_pos(
		p_listener_pos, p_listener_sx, p_listener_sy, p_listener_sz,
		p_source_pos, p_source_sx, p_source_sy, p_source_sz,
		FixedMathCore(10000LL, false) // 10k sector size
	);
	FixedMathCore r = rel_vec.length();
	if (unlikely(r.get_raw() == 0)) return;

	// 2. Relativistic Doppler Resolve
	// f_obs = f_src * (c_sound + v_obs) / (c_sound - v_src) * (1 / Lorentz_Gamma)
	Vector3f radial_dir = rel_vec / r;
	FixedMathCore v_o = p_listener_vel.dot(radial_dir);
	FixedMathCore v_s = p_source_vel.dot(radial_dir);

	FixedMathCore freq_num = p_speed_of_sound + v_o;
	FixedMathCore freq_den = p_speed_of_sound - v_s;
	
	// Safety: If source moves faster than sound (Sonic Boom), clamp denominator
	if (freq_den <= zero) freq_den = FixedMathCore(429496LL, true); // 0.0001
	
	FixedMathCore doppler_mult = freq_num / freq_den;
	
	// Lorentz Factor Correction for High-Speed Ships
	FixedMathCore beta2 = wp::min(p_source_vel.length_squared() / p_c_sq, FixedMathCore(4294967290LL, true));
	FixedMathCore inv_gamma = (one - beta2).square_root();
	
	FixedMathCore perceived_freq = p_base_freq * doppler_mult * inv_gamma;

	// 3. Deterministic Phase Calculation
	// Convert BigInt ticks to FixedMath time: t = frames / 120
	FixedMathCore time_f(static_cast<int64_t>(std::stoll(p_global_tick.to_string())));
	time_f /= FixedMathCore(120LL);

	// Phase = 2 * PI * f * (t - r/c)
	FixedMathCore phase = MathConstants<FixedMathCore>::two_pi() * perceived_freq * (time_f - (r / p_speed_of_sound));

	// 4. Knife-Edge Diffraction logic
	// Fresnel Zone Approximation: applies signal reduction if the path is obstructed.
	FixedMathCore diffraction_loss = one;
	for (uint64_t i = 0; i < p_obstruction_count; i++) {
		const Face3f &f = p_obstructions[i];
		Vector3f intersect;
		if (f.intersects_ray(p_source_pos, radial_dir, &intersect)) {
			FixedMathCore dist_to_hit = (intersect - p_source_pos).length();
			if (dist_to_hit < r) {
				// Blocked: Apply frequency-dependent loss (Higher freq = higher loss)
				FixedMathCore loss_coeff = perceived_freq * FixedMathCore(429496LL, true); // 0.0001 scale
				diffraction_loss = one / (one + loss_coeff);
				break; 
			}
		}
	}

	// 5. Wave Superposition (Deterministic Interference)
	// Inverse Square Law: A = A0 * sin(phase) * loss / r
	FixedMathCore wave_val = (p_base_amp * wp::sin(phase) * diffraction_loss) / r;

	// --- Sophisticated Behavior: Anime Pitch & Power Tensors ---
	if (p_is_anime) {
		// Anime Technique: "Harmonic Snapping". 
		// Quantizes frequency into perfect musical octaves (440Hz base) for stylized impact.
		FixedMathCore base_hz(440LL); 
		FixedMathCore octaves = perceived_freq / base_hz;
		perceived_freq = Math::snapped(octaves, one) * base_hz;

		// Waveshape saturation for high-amplitude shockwaves (Non-linear crunch)
		if (wp::abs(wave_val) > one) {
			FixedMathCore over = wp::abs(wave_val) - one;
			wave_val = wp::sign(wave_val) * (one + (over * FixedMathCore(214748364LL, true))); // 0.05 saturation
		}
	}

	r_final_amplitude += wave_val;
	r_final_frequency = perceived_freq;
}

/**
 * execute_parallel_acoustic_wave_sweep()
 * 
 * Orchestrates the parallel 120 FPS audio simulation wave.
 * Partitions the EnTT registry into worker batches for SIMD interference resolve.
 */
void PhysicsServerHyper::execute_parallel_acoustic_wave_sweep(const FixedMathCore &p_delta) {
	KernelRegistry &registry = get_kernel_registry();
	auto &src_freq_stream = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_BASE_FREQ);
	uint64_t count = src_freq_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	FixedMathCore c_sound(343LL); 
	FixedMathCore c_light_sq = PHYSICS_C * PHYSICS_C;
	BigIntCore current_tick = SimulationManager::get_singleton()->get_total_frames();

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &registry]() {
			auto &out_amp = registry.get_stream<FixedMathCore>(COMPONENT_FINAL_AUDIO_AMP);
			auto &out_freq = registry.get_stream<FixedMathCore>(COMPONENT_FINAL_AUDIO_FREQ);
			auto &src_pos = registry.get_stream<Vector3f>(COMPONENT_SOURCE_POS);
			auto &src_vel = registry.get_stream<Vector3f>(COMPONENT_SOURCE_VEL);
			auto &src_sx = registry.get_stream<BigIntCore>(COMPONENT_SOURCE_SX);
			auto &src_sy = registry.get_stream<BigIntCore>(COMPONENT_SOURCE_SY);
			auto &src_sz = registry.get_stream<BigIntCore>(COMPONENT_SOURCE_SZ);
			auto &src_amp = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_BASE_AMP);

			for (uint64_t i = start; i < end; i++) {
				// Deterministic Style Selection
				bool anime_mode = (i % 7 == 0);

				acoustic_wave_field_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					out_amp[i],
					out_freq[i],
					observer_pos, observer_vel, observer_sx, observer_sy, observer_sz,
					src_pos[i], src_vel[i], src_sx[i], src_sy[i], src_sz[i],
					src_freq_stream[i], src_amp[i],
					current_tick, c_sound, c_light_sq,
					nullptr, 0, // Obstruction integration via localized broadphase
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_acoustics_wave.cpp ---
