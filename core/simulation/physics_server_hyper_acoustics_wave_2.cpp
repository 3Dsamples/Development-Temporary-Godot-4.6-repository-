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
 * calculate_fresnel_diffraction_loss()
 * 
 * Deterministic approximation of the Huygens-Fresnel principle for sound bending.
 * Calculates a reduction factor [0..1] based on frequency and obstruction distance.
 */
static _FORCE_INLINE_ FixedMathCore calculate_fresnel_diffraction_loss(
		const FixedMathCore &p_freq,
		const FixedMathCore &p_dist_to_edge,
		const FixedMathCore &p_speed_of_sound) {
	
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore wavelength = p_speed_of_sound / (p_freq + MathConstants<FixedMathCore>::unit_epsilon());
	
	// Fresnel parameter v = h * sqrt(2 / (lambda * d))
	// where h is the path clearance (dist_to_edge).
	FixedMathCore v = p_dist_to_edge * Math::sqrt(FixedMathCore(2LL) / (wavelength * p_dist_to_edge + MathConstants<FixedMathCore>::unit_epsilon()));
	
	// Deterministic path loss approximation: L = 0.5 - 0.5 * sin(pi/2 * v)
	FixedMathCore loss = MathConstants<FixedMathCore>::half() - MathConstants<FixedMathCore>::half() * Math::sin(MathConstants<FixedMathCore>::half_pi() * v);
	return wp::max(FixedMathCore(42949673LL, true), loss); // 0.01 floor
}

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
	Vector3f rel_vec = wp::calculate_galactic_relative_pos(
		p_listener_pos, p_listener_sx, p_listener_sy, p_listener_sz,
		p_source_pos, p_source_sx, p_source_sy, p_source_sz,
		FixedMathCore(10000LL, false)
	);
	FixedMathCore r = rel_vec.length();
	
	if (unlikely(r.get_raw() == 0)) {
		r_final_amplitude = p_base_amp;
		r_final_frequency = p_base_freq;
		return;
	}

	// 2. Relativistic Doppler Resolve
	Vector3f n = rel_vec / r;
	FixedMathCore v_o = p_listener_vel.dot(n);
	FixedMathCore v_s = p_source_vel.dot(n);
	FixedMathCore freq_den = p_speed_of_sound - v_s;
	if (freq_den <= zero) freq_den = FixedMathCore(429496LL, true); // 0.0001 floor
	FixedMathCore doppler_ratio = (p_speed_of_sound + v_o) / freq_den;

	FixedMathCore beta2 = wp::min(p_source_vel.length_squared() / p_c_sq, FixedMathCore(4294967290LL, true));
	FixedMathCore lorentz_inv = (one - beta2).square_root();
	FixedMathCore perceived_freq = p_base_freq * doppler_ratio * lorentz_inv;

	// 3. Phase Integration
	FixedMathCore time_f(static_cast<int64_t>(std::stoll(p_global_tick.to_string())));
	time_f /= FixedMathCore(120LL);
	FixedMathCore phase = MathConstants<FixedMathCore>::two_pi() * perceived_freq * (time_seconds - (r / p_speed_of_sound));

	// 4. Diffraction Resolve
	FixedMathCore diffraction_loss = one;
	for (uint64_t i = 0; i < p_obstruction_count; i++) {
		const Face3f &face = p_obstructions[i];
		Vector3f intersect;
		if (face.intersects_ray(p_source_pos, n, &intersect)) {
			if ((intersect - p_source_pos).length() < r) {
				Vector3f edge_p = face.get_closest_point(intersect);
				FixedMathCore h = (intersect - edge_p).length();
				diffraction_loss = calculate_fresnel_diffraction_loss(perceived_freq, h, p_speed_of_sound);
				break;
			}
		}
	}

	// 5. Sophisticated Anime Behavior
	if (p_is_anime) {
		FixedMathCore base_harmonic(440LL);
		perceived_freq = Math::snapped(perceived_freq / base_harmonic, one) * base_harmonic;
	}

	// 6. Superposition (Interference)
	FixedMathCore wave_sample = (p_base_amp * wp::sin(phase) * diffraction_loss) / r;
	r_final_amplitude += wave_sample;
	r_final_frequency = perceived_freq;
}

/**
 * execute_parallel_acoustic_wave_sweep()
 * Orchestrates the parallel 120 FPS audio simulation wave across the EnTT registry.
 */
void PhysicsServerHyper::execute_parallel_acoustic_wave_sweep(const FixedMathCore &p_delta) {
	auto &registry = get_kernel_registry();
	auto &freq_stream = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_BASE_FREQ);
	uint64_t count = freq_stream.size();
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
			const Face3f *face_ptr = registry.get_stream<Face3f>(COMPONENT_MESH_GEOMETRY).get_base_ptr();
			uint64_t face_count = registry.get_stream<Face3f>(COMPONENT_MESH_GEOMETRY).size();

			for (uint64_t i = start; i < end; i++) {
				acoustic_wave_field_kernel(
					BigIntCore(static_cast<int64_t>(i)), out_amp[i], out_freq[i],
					observer_pos, observer_vel, observer_sx, observer_sy, observer_sz,
					src_pos[i], src_vel[i], src_sx[i], src_sy[i], src_sz[i],
					freq_stream[i], base_amp[i], current_frame, c_sound, c_light_sq,
					face_ptr, face_count, (i % 7 == 0)
				);
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}
	pool->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_acoustics_wave.cpp ---
