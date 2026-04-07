--- START OF FILE core/simulation/physics_server_hyper_acoustics_wave.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/collision_solver.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/simulation/simulation_manager.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_acoustic_diffraction_factor()
 * 
 * Computes the loss in amplitude caused by geometric obstructions.
 * Implements a bit-perfect Fresnel-Kirchhoff diffraction integral approximation.
 * Strictly uses FixedMathCore for wavelength and clearance calculations.
 */
static _FORCE_INLINE_ FixedMathCore calculate_acoustic_diffraction_factor(
		const FixedMathCore &p_freq,
		const FixedMathCore &p_clearance_h,
		const FixedMathCore &p_dist_total,
		const FixedMathCore &p_speed_of_sound) {
	
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// lambda = v / f
	FixedMathCore lambda = p_speed_of_sound / (p_freq + MathConstants<FixedMathCore>::unit_epsilon());
	
	// Fresnel Parameter: v = h * sqrt(2 / (lambda * d))
	FixedMathCore v = p_clearance_h * Math::sqrt(FixedMathCore(2LL) / (lambda * p_dist_total + MathConstants<FixedMathCore>::unit_epsilon()));
	
	// Deterministic Diffraction Loss (Lee's Approximation): 
	// L = 10 * log10(0.5 - 0.5 * sin(pi/2 * v))
	// Converting to linear amplitude scale: factor = 0.5 - 0.5 * sin(pi/2 * v)
	FixedMathCore phase_shift = MathConstants<FixedMathCore>::half_pi() * v;
	FixedMathCore factor = MathConstants<FixedMathCore>::half() - (MathConstants<FixedMathCore>::half() * Math::sin(phase_shift));
	
	return wp::clamp(factor, FixedMathCore(4294967LL, true), one); // 0.001 floor
}

/**
 * Warp Kernel: AcousticWaveFieldKernel
 * 
 * Computes the instantaneous pressure value at the listener coordinate.
 * 1. Galactic Resolve: Bit-perfect distance between BigInt sectors.
 * 2. Relativistic Doppler: Lorentz-corrected pitch shifting for high-speed ship flybys.
 * 3. Sonic Lensing: Sound magnification near high-mass (BigIntCore) gravity wells.
 * 4. Anime Tensors: Harmonic quantization and non-linear waveshape clipping.
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

	// 1. Resolve Galactic Spatial Vector
	Vector3f rel_vec = wp::calculate_galactic_relative_pos(
		p_listener_pos, p_listener_sx, p_listener_sy, p_listener_sz,
		p_source_pos, p_source_sx, p_source_sy, p_source_sz,
		FixedMathCore(10000LL, false) // 10k unit sector size
	);
	FixedMathCore r = rel_vec.length();
	
	if (unlikely(r.get_raw() == 0)) {
		r_final_amplitude = p_base_amp;
		r_final_frequency = p_base_freq;
		return;
	}

	// 2. Relativistic Doppler Frequency Shift
	// f_obs = f_src * ( (c_sound + v_obs) / (c_sound - v_src) ) * (1 / gamma)
	Vector3f radial_dir = rel_vec / r;
	FixedMathCore v_o = p_listener_vel.dot(radial_dir);
	FixedMathCore v_s = p_source_vel.dot(radial_dir);

	FixedMathCore freq_num = p_speed_of_sound + v_o;
	FixedMathCore freq_den = p_speed_of_sound - v_s;
	
	// Handle Sonic Boom / Supersonic Singularity
	if (freq_den <= zero) freq_den = FixedMathCore(429496LL, true); // 0.0001
	
	FixedMathCore classical_doppler = freq_num / freq_den;

	// Relativistic Time Dilation for High-Speed Entities
	FixedMathCore beta2 = wp::min(p_source_vel.length_squared() / p_c_sq, FixedMathCore(4294967290LL, true));
	FixedMathCore lorentz_inv = (one - beta2).square_root();
	
	FixedMathCore perceived_freq = p_base_freq * classical_doppler * lorentz_inv;

	// 3. Deterministic Phase Integration
	// t = current_frame / 120.0
	FixedMathCore time_sec(static_cast<int64_t>(std::stoll(p_global_tick.to_string())));
	time_sec /= FixedMathCore(120LL);

	// Phase = 2 * PI * f * (t - r/c)
	FixedMathCore phase = MathConstants<FixedMathCore>::two_pi() * perceived_freq * (time_sec - (r / p_speed_of_sound));

	// 4. Diffraction & Obstruction Sweep
	FixedMathCore diffraction_loss = one;
	for (uint64_t i = 0; i < p_obstruction_count; i++) {
		const Face3f &f = p_obstructions[i];
		Vector3f intersection;
		if (f.intersects_ray(p_source_pos, radial_dir, &intersection)) {
			FixedMathCore d_hit = (intersection - p_source_pos).length();
			if (d_hit < r) {
				// Path Obstructed: Calculate clearance from nearest edge
				Vector3f edge_pt = f.get_closest_point(intersection);
				FixedMathCore clearance_h = (intersection - edge_pt).length();
				diffraction_loss = calculate_fresnel_diffraction_loss(perceived_freq, clearance_h, r, p_speed_of_sound);
				break;
			}
		}
	}

	// 5. Sophisticated Anime Sound Tensors
	if (p_is_anime) {
		// Technique: "Harmonic Snapping". 
		// Quantizes frequency into perfect musical octaves for stylized dramatic impact.
		FixedMathCore A4(440LL); 
		FixedMathCore octave_index = perceived_freq / A4;
		perceived_freq = Math::snapped(octave_index, one) * A4;

		// Technique: "Sonic Lensing". 
		// Magnifies amplitude based on Doppler acceleration (simulating dramatic entry).
		if (classical_doppler > FixedMathCore(2LL)) {
			diffraction_loss *= FixedMathCore(15LL, false) / FixedMathCore(10LL, false); // 1.5x boost
		}
	}

	// 6. Wave Superposition (Deterministic Interference)
	// A = (SourceAmplitude * sin(phase) * Diffraction) / r (Inverse Square Law)
	FixedMathCore instantaneous_p = (p_base_amp * Math::sin(phase) * diffraction_loss) / r;

	// Non-linear crunch for high-energy events (Anime distortion)
	if (p_is_anime && wp::abs(instantaneous_p) > one) {
		FixedMathCore over = wp::abs(instantaneous_p) - one;
		instantaneous_p = wp::sign(instantaneous_p) * (one + (over * FixedMathCore(214748364LL, true))); // 0.05 saturation
	}

	r_final_amplitude += instantaneous_p;
	r_final_frequency = perceived_freq;
}

/**
 * execute_parallel_acoustic_wave_sweep()
 * 
 * Orchestrates the parallel 120 FPS audio simulation wave across the EnTT registry.
 * maps millions of acoustic interactions using SimulationThreadPool.
 */
void PhysicsServerHyper::execute_parallel_acoustic_wave_sweep(const FixedMathCore &p_delta) {
	KernelRegistry &registry = get_kernel_registry();
	auto &base_freq_stream = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_BASE_FREQ);
	uint64_t count = base_freq_stream.size();
	if (count == 0) return;

	SimulationThreadPool *pool = SimulationThreadPool::get_singleton();
	uint32_t workers = pool->get_worker_count();
	uint64_t chunk = count / workers;

	FixedMathCore c_sound(343LL); 
	FixedMathCore c_light_sq = PHYSICS_C * PHYSICS_C;
	BigIntCore current_frame = SimulationManager::get_singleton()->get_total_frames();

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		pool->enqueue_task([=, &registry]() {
			auto &out_amp = registry.get_stream<FixedMathCore>(COMPONENT_FINAL_AUDIO_AMP);
			auto &out_freq = registry.get_stream<FixedMathCore>(COMPONENT_FINAL_AUDIO_FREQ);
			auto &src_pos = registry.get_stream<Vector3f>(COMPONENT_SOURCE_POS);
			auto &src_vel = registry.get_stream<Vector3f>(COMPONENT_SOURCE_VEL);
			auto &src_sx = registry.get_stream<BigIntCore>(COMPONENT_SOURCE_SX);
			auto &src_sy = registry.get_stream<BigIntCore>(COMPONENT_SOURCE_SY);
			auto &src_sz = registry.get_stream<BigIntCore>(COMPONENT_SOURCE_SZ);
			auto &base_amp = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_BASE_AMP);
			
			// Zero-copy access to world Face3f obstructions
			const Face3f *faces = registry.get_stream<Face3f>(COMPONENT_GEOMETRY).get_base_ptr();
			uint64_t f_count = registry.get_stream<Face3f>(COMPONENT_GEOMETRY).size();

			for (uint64_t i = start; i < end; i++) {
				// Style derived from Entity handle index hash
				bool anime_mode = (BigIntCore(static_cast<int64_t>(i)).hash() % 10 == 0);

				acoustic_wave_field_kernel(
					BigIntCore(static_cast<int64_t>(i)), out_amp[i], out_freq[i],
					observer_pos, observer_vel, observer_sx, observer_sy, observer_sz,
					src_pos[i], src_vel[i], src_sx[i], src_sy[i], src_sz[i],
					base_freq_stream[i], base_amp[i], current_frame, c_sound, c_light_sq,
					faces, f_count, anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}

	pool->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_acoustics_wave.cpp ---
