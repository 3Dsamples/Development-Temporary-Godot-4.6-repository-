--- START OF FILE core/simulation/physics_server_hyper_acoustics_engine.cpp ---

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
 * calculate_diffraction_tensor()
 * 
 * Deterministic approximation of the Huygens-Fresnel principle.
 * Computes the amplitude reduction factor [0..1] when a sound path is 
 * partially obstructed by geometry (Face3f).
 */
static _FORCE_INLINE_ FixedMathCore calculate_diffraction_tensor(
		const FixedMathCore &p_freq,
		const FixedMathCore &p_clearance_dist,
		const FixedMathCore &p_total_dist,
		const FixedMathCore &p_speed_of_sound) {
	
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore lambda = p_speed_of_sound / (p_freq + MathConstants<FixedMathCore>::unit_epsilon());
	
	// Fresnel Parameter: v = h * sqrt(2 / (lambda * d))
	FixedMathCore v = p_clearance_dist * Math::sqrt(FixedMathCore(2LL) / (lambda * p_total_dist + MathConstants<FixedMathCore>::unit_epsilon()));
	
	// Deterministic Loss Curve: factor = 0.5 - 0.5 * sin(pi/2 * v)
	FixedMathCore phase_shift = MathConstants<FixedMathCore>::half_pi() * v;
	FixedMathCore factor = MathConstants<FixedMathCore>::half() - (MathConstants<FixedMathCore>::half() * phase_shift.sin());
	
	return wp::clamp(factor, FixedMathCore(4294967LL, true), one); // 0.001 floor
}

/**
 * Warp Kernel: AcousticWaveFieldKernel
 * 
 * Computes the instantaneous pressure sample at the listener coordinate.
 * 1. Galactic Distance: Bit-perfect resolve between different BigInt sectors.
 * 2. Relativistic Doppler: Lorentz-corrected pitch for high-speed spaceship flybys.
 * 3. Diffraction: Checks for geometric occluders and applies the Fresnel tensor.
 * 4. Superposition: Sums all wave amplitudes using bit-perfect sin() integration.
 */
void acoustic_wave_field_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_final_pressure,
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

	// 1. Resolve Precise Galactic Distance
	Vector3f rel_vec = wp::calculate_galactic_relative_pos(
		p_listener_pos, p_listener_sx, p_listener_sy, p_listener_sz,
		p_source_pos, p_source_sx, p_source_sy, p_source_sz,
		FixedMathCore(10000LL, false) // 10k sector size
	);
	FixedMathCore r = rel_vec.length();
	
	if (unlikely(r.get_raw() == 0)) {
		r_final_pressure = p_base_amp;
		r_final_frequency = p_base_freq;
		return;
	}

	// 2. Relativistic Doppler Frequency Shift
	// f_obs = f_src * ( (c_s + v_obs) / (c_s - v_src) ) * (1 / gamma)
	Vector3f radial_dir = rel_vec / r;
	FixedMathCore v_o = p_listener_vel.dot(radial_dir);
	FixedMathCore v_s = p_source_vel.dot(radial_dir);

	FixedMathCore freq_num = p_speed_of_sound + v_o;
	FixedMathCore freq_den = p_speed_of_sound - v_s;
	
	// Singularity Guard: Source speed approaching sound speed (Mach 1)
	if (freq_den <= zero) freq_den = FixedMathCore(429496LL, true); 
	
	FixedMathCore doppler_ratio = freq_num / freq_den;

	// Time Dilation factor for entities at relativistic velocities
	FixedMathCore beta2 = wp::min(p_source_vel.length_squared() / p_c_sq, FixedMathCore(4294967290LL, true));
	FixedMathCore inv_gamma = (one - beta2).square_root();
	
	FixedMathCore perceived_freq = p_base_freq * doppler_ratio * inv_gamma;

	// 3. Deterministic Phase Integration
	// t = current_frame / 120.0 (Seconds)
	FixedMathCore time_sec(static_cast<int64_t>(std::stoll(p_global_tick.to_string())));
	time_sec /= FixedMathCore(120LL);

	// phi = 2 * PI * f * (t - r/c_s)
	FixedMathCore phase = MathConstants<FixedMathCore>::two_pi() * perceived_freq * (time_sec - (r / p_speed_of_sound));

	// 4. Obstruction & Diffraction Resolve
	FixedMathCore diffraction_loss = one;
	for (uint64_t i = 0; i < p_obstruction_count; i++) {
		const Face3f &f = p_obstructions[i];
		Vector3f intersection;
		if (f.intersects_ray(p_source_pos, radial_dir, &intersection)) {
			FixedMathCore d_hit = (intersection - p_source_pos).length();
			if (d_hit < r) {
				// Blocked: calculate clearance distance from nearest edge point
				Vector3f edge_pt = f.get_closest_point(intersection);
				FixedMathCore clearance = (intersection - edge_pt).length();
				diffraction_loss = calculate_diffraction_tensor(perceived_freq, clearance, r, p_speed_of_sound);
				break;
			}
		}
	}

	// 5. Sophisticated Anime Behavior
	if (p_is_anime) {
		// Harmonic Snapping: force frequencies into perfect octaves for stylized "Whistle/Thrum"
		FixedMathCore base_h(440LL); 
		FixedMathCore octave_val = perceived_freq / base_h;
		perceived_freq = Math::snapped(octave_val, one) * base_h;
	}

	// 6. Final Superposition Resolve
	// Pressure = (A_source * sin(phase) * Loss) / r (Inverse Square Attenuation)
	FixedMathCore sample_val = (p_base_amp * phase.sin() * diffraction_loss) / r;

	// Wave Clipping: non-linear distortion for high-energy combat shockwaves
	if (p_is_anime && wp::abs(sample_val) > one) {
		FixedMathCore extra = wp::abs(sample_val) - one;
		sample_val = wp::sign(sample_val) * (one + (extra * FixedMathCore(214748364LL, true))); // 0.05x saturation
	}

	r_final_pressure += sample_val;
	r_final_frequency = perceived_freq;
}

/**
 * execute_parallel_acoustic_wave_sweep()
 * 
 * Master 120 FPS orchestrator for the auditory simulation wave.
 * Partitions the acoustic EnTT component streams into worker batches.
 */
void PhysicsServerHyper::execute_parallel_acoustic_wave_sweep(const FixedMathCore &p_delta) {
	KernelRegistry &registry = get_kernel_registry();
	auto &base_freq_stream = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_BASE_FREQ);
	uint64_t count = base_freq_stream.size();
	if (count == 0) return;

	SimulationThreadPool *pool = SimulationThreadPool::get_singleton();
	uint32_t workers = pool->get_worker_count();
	uint64_t chunk = count / workers;

	FixedMathCore speed_sound(343LL); 
	FixedMathCore light_c_sq = PHYSICS_C * PHYSICS_C;
	BigIntCore frame_tick = SimulationManager::get_singleton()->get_total_frames();

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		pool->enqueue_task([=, &registry]() {
			auto &p_final = registry.get_stream<FixedMathCore>(COMPONENT_FINAL_PRESSURE);
			auto &f_final = registry.get_stream<FixedMathCore>(COMPONENT_FINAL_FREQUENCY);
			auto &src_pos = registry.get_stream<Vector3f>(COMPONENT_SOURCE_POS);
			auto &src_vel = registry.get_stream<Vector3f>(COMPONENT_SOURCE_VEL);
			auto &src_sx  = registry.get_stream<BigIntCore>(COMPONENT_SOURCE_SX);
			auto &src_sy  = registry.get_stream<BigIntCore>(COMPONENT_SOURCE_SY);
			auto &src_sz  = registry.get_stream<BigIntCore>(COMPONENT_SOURCE_SZ);
			auto &base_amp = registry.get_stream<FixedMathCore>(COMPONENT_AUDIO_BASE_AMP);
			
			// Zero-copy access to world obstruction faces
			const Face3f *mesh_faces = registry.get_stream<Face3f>(COMPONENT_COLLISION_GEOMETRY).get_base_ptr();
			uint64_t mesh_face_count = registry.get_stream<Face3f>(COMPONENT_COLLISION_GEOMETRY).size();

			for (uint64_t i = start; i < end; i++) {
				// Anime style determination (Linked to Entity ID hash)
				BigIntCore handle(static_cast<int64_t>(i));
				bool anime_style = (handle.hash() % 9 == 0);

				acoustic_wave_field_kernel(
					handle, p_final[i], f_final[i],
					observer_pos, observer_vel, observer_sx, observer_sy, observer_sz,
					src_pos[i], src_vel[i], src_sx[i], src_sy[i], src_sz[i],
					base_freq_stream[i], base_amp[i], frame_tick, speed_sound, light_c_sq,
					mesh_faces, mesh_face_count, anime_style
				);
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}

	pool->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_acoustics_wave.cpp ---
