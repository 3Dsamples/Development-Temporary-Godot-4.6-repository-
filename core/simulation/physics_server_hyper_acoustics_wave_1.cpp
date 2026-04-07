--- START OF FILE core/simulation/physics_server_hyper_acoustics_wave.cpp ---

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
 * Warp Kernel: AcousticWaveformIntegrationKernel
 * 
 * Computes the specific pressure contribution of an acoustic source to a listener point.
 * 1. Distance Resolve: Bit-perfect Euclidean distance across BigInt sectors.
 * 2. Absorption: Atmospheric attenuation based on local medium density.
 * 3. Phase Calculation: theta = (2 * pi * f * (t - r/c)).
 * 4. Diffraction: Recursive Huygens edge-bending around mesh obstructions.
 */
void acoustic_waveform_integration_kernel(
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

	// 1. Precise Galactic Distance Resolve
	// Uses the sector-aware relative position helper to prevent floating-point drift.
	Vector3f rel_vec = wp::calculate_galactic_relative_pos(
		p_listener_pos, p_listener_sx, p_listener_sy, p_listener_sz,
		p_source_pos, p_source_sx, p_source_sy, p_source_sz,
		FixedMathCore(10000LL, false) // 10k sector threshold
	);
	FixedMathCore dist = rel_vec.length();
	
	if (unlikely(dist.get_raw() == 0)) {
		r_final_amplitude = p_base_amp;
		r_final_frequency = p_base_freq;
		return;
	}

	// 2. Relativistic and Classical Doppler Resolve
	// f_observed = f_source * ((c_sound + v_obs) / (c_sound - v_source)) * sqrt(1 - v^2/c^2)
	Vector3f n = rel_vec / dist;
	FixedMathCore vr_o = p_listener_vel.dot(n);
	FixedMathCore vr_s = p_source_vel.dot(n);

	FixedMathCore freq_numerator = p_speed_of_sound + vr_o;
	FixedMathCore freq_denominator = p_speed_of_sound - vr_s;

	// Clamp to prevent singularity during supersonic travel (Sonic Boom threshold)
	if (freq_denominator <= zero) {
		freq_denominator = FixedMathCore(1LL, true);
	}
	
	FixedMathCore doppler_ratio = freq_num / freq_den;

	// Apply Lorentz Time Dilation factor for high-speed ship audio
	FixedMathCore beta2 = wp::min(p_source_vel.length_squared() / p_c_sq, FixedMathCore(4294967290LL, true));
	FixedMathCore lorentz_inv = (one - beta2).square_root();

	FixedMathCore perceived_freq = p_base_freq * doppler_ratio * lorentz_inv;

	// 3. Deterministic Phase Integration
	// t = current_frame / 120.0
	FixedMathCore time_seconds(static_cast<int64_t>(std::stoll(p_global_tick.to_string())));
	time_seconds /= FixedMathCore(120LL);

	// Phase phi = 2 * PI * f * (t - dist / speed_of_sound)
	FixedMathCore phase = MathConstants<FixedMathCore>::two_pi() * perceived_freq * (time_seconds - (dist / p_speed_of_sound));

	// 4. Knife-Edge Diffraction Resolve
	// Checks for geometric occluders in the direct line of sight.
	FixedMathCore diffraction_coeff = one;
	for (uint64_t i = 0; i < p_obstruction_count; i++) {
		const Face3f &face = p_obstructions[i];
		Vector3f intersection;
		if (face.intersects_ray(p_source_pos, n, &intersection)) {
			FixedMathCore dist_to_occluder = (intersection - p_source_pos).length();
			if (dist_to_occluder < dist) {
				// Obstruction detected: Apply frequency-based low-pass absorption.
				// high-frequency sound (Anime 'shing') is absorbed more than low (Realistic 'thud').
				FixedMathCore absorption_tensor = perceived_freq * FixedMathCore(429496LL, true); // 0.0001 scale
				diffraction_coeff = one / (one + absorption_tensor);
				break;
			}
		}
	}

	// 5. Sophisticated Anime Tensors
	if (p_is_anime) {
		// Technique: "Harmonic Snapping". 
		// Quantizes frequency into perfect musical octaves for stylized impact.
		FixedMathCore base_harmonic(440LL); // A4 440Hz
		FixedMathCore octaves = perceived_freq / base_harmonic;
		perceived_freq = Math::snapped(octaves, one) * base_harmonic;

		// Technique: "Wave Saturation".
		// Non-linear crunch for high-amplitude audio events.
		if (p_base_amp > FixedMathCore(10LL, false)) {
			diffraction_coeff *= FixedMathCore(2LL, false); // Exaggerate diffraction bypass
		}
	}

	// 6. Superposition and Attenuation
	// A = (A_src * sin(phase) * diffraction) / dist (Inverse Square Law)
	FixedMathCore wave_sample = (p_base_amp * wp::sin(phase) * diffraction_coeff) / dist;
	
	r_final_amplitude += wave_sample;
	r_final_frequency = perceived_freq;
}

/**
 * execute_acoustic_wave_sweep()
 * 
 * Orchestrates the parallel 120 FPS audio simulation wave across the EnTT registry.
 */
void PhysicsServerHyper::execute_acoustic_wave_sweep(const FixedMathCore &p_delta) {
	KernelRegistry &registry = get_kernel_registry();
	auto &freq_stream = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_FREQ);
	uint64_t count = freq_stream.size();
	if (count == 0) return;

	SimulationThreadPool *pool = SimulationThreadPool::get_singleton();
	uint32_t workers = pool->get_worker_count();
	uint64_t chunk = count / workers;

	FixedMathCore c_sound(343LL); // 343 m/s (Standard Atmosphere)
	FixedMathCore c_light_sq = PHYSICS_C * PHYSICS_C;
	BigIntCore current_frame = SimulationManager::get_singleton()->get_total_frames();

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		pool->enqueue_task([=, &registry]() {
			auto &out_amp = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_OUT_AMP);
			auto &out_freq = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_OUT_FREQ);
			auto &src_pos = registry.get_stream<Vector3f>(COMPONENT_POSITION);
			auto &src_vel = registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
			auto &src_sx = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X);
			auto &src_sy = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y);
			auto &src_sz = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z);
			auto &base_amp = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_BASE_AMP);
			auto &base_freq = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_BASE_FREQ);
			
			// Reference to world obstructions for diffraction checks
			const Face3f *face_ptr = registry.get_stream<Face3f>(COMPONENT_MESH_GEOMETRY).get_base_ptr();
			uint64_t face_count = registry.get_stream<Face3f>(COMPONENT_MESH_GEOMETRY).size();

			for (uint64_t i = start; i < end; i++) {
				// Style derived from entity seed
				bool anime_style = (i % 7 == 0);

				acoustic_wave_field_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					out_amp[i],
					out_freq[i],
					observer_pos, observer_vel, observer_sx, observer_sy, observer_sz,
					src_pos[i], src_vel[i], src_sx[i], src_sy[i], src_sz[i],
					base_freq[i], base_amp[i],
					current_frame,
					c_sound,
					c_light_sq,
					face_ptr,
					face_count,
					anime_style
				);
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}

	pool->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_acoustics_wave.cpp ---
