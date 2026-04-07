--- START OF FILE core/math/spectral_tensor_kernel.cpp ---

#include "core/math/spectral_tensor_kernel.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

// ============================================================================
// Deterministic Microfacet Kernels (FixedMathCore Q32.32)
// ============================================================================

/**
 * calculate_distribution_ggx()
 * 
 * Trowbridge-Reitz GGX Normal Distribution Function (D).
 * D(h) = alpha^2 / (pi * ((n.h)^2 * (alpha^2 - 1) + 1)^2)
 */
static _FORCE_INLINE_ FixedMathCore calculate_distribution_ggx(FixedMathCore p_dot_nh, FixedMathCore p_alpha) {
	FixedMathCore alpha2 = p_alpha * p_alpha;
	FixedMathCore pi = MathConstants<FixedMathCore>::pi();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	FixedMathCore denom_inner = (p_dot_nh * p_dot_nh) * (alpha2 - one) + one;
	FixedMathCore denominator = pi * (denom_inner * denom_inner);

	if (unlikely(denominator.get_raw() == 0)) return one;
	return alpha2 / denominator;
}

/**
 * calculate_geometry_smith()
 * 
 * Smith's Schlick-GGX Geometry Shadowing Function (G).
 * G(v, l, n, k) = G1(v) * G1(l)
 */
static _FORCE_INLINE_ FixedMathCore calculate_geometry_smith(FixedMathCore p_dot_nv, FixedMathCore p_dot_nl, FixedMathCore p_k) {
	auto g1 = [&](FixedMathCore dot) -> FixedMathCore {
		FixedMathCore one = MathConstants<FixedMathCore>::one();
		return dot / (dot * (one - p_k) + p_k);
	};
	return g1(p_dot_nv) * g1(p_dot_nl);
}

/**
 * calculate_fresnel_schlick()
 * 
 * Schlick's approximation for Fresnel reflectance (F).
 * F = F0 + (1 - F0) * (1 - cos_theta)^5
 */
static _FORCE_INLINE_ Vector3f calculate_fresnel_schlick(FixedMathCore p_cos_theta, const Vector3f &p_f0) {
	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore sph = one - p_cos_theta;
	FixedMathCore sph2 = sph * sph;
	FixedMathCore factor = sph2 * sph2 * sph; // (1 - cos_theta)^5

	Vector3f complement(one - p_f0.x, one - p_f0.y, one - p_f0.z);
	return p_f0 + complement * factor;
}

// ============================================================================
// Warp Kernels (EnTT SoA Parallel Execution)
// ============================================================================

/**
 * Warp Kernel: BRDFResolutionKernel
 * 
 * Performs parallel spectral resolve for a batch of surface samples.
 * 1. Resolves PBR material tensors (Metallic/Roughness).
 * 2. Computes spectral energy composition.
 * 3. Injects sophisticated Anime Banding for cel-shaded highlights.
 */
void brdf_resolution_kernel(
		const BigIntCore &p_index,
		SpectralRadiance &r_out_radiance,
		const MaterialTensor &p_mat,
		const Vector3f &p_normal,
		const Vector3f &p_view_dir,
		const Vector3f &p_light_dir,
		const Vector3f &p_light_energy,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	Vector3f N = p_normal;
	Vector3f V = p_view_dir;
	Vector3f L = p_light_dir;
	Vector3f H = (V + L).normalized();

	FixedMathCore dot_nl = wp::max(N.dot(L), zero);
	FixedMathCore dot_nv = wp::max(N.dot(V), zero);
	FixedMathCore dot_nh = wp::max(N.dot(H), zero);
	FixedMathCore dot_vh = wp::max(V.dot(H), zero);

	if (dot_nl.get_raw() == 0) {
		r_out_radiance.energy = Vector3f_ZERO;
		return;
	}

	// 1. Resolve Specular Term (Cook-Torrance)
	FixedMathCore alpha = p_mat.roughness * p_mat.roughness;
	FixedMathCore k_pbr = (p_mat.roughness + one);
	k_pbr = (k_pbr * k_pbr) / FixedMathCore(8LL, false);

	FixedMathCore D = calculate_distribution_ggx(dot_nh, alpha);
	FixedMathCore G = calculate_geometry_smith(dot_nv, dot_nl, k_pbr);
	
	// F0 calculation: Dielectrics ~0.04, Metals = Albedo
	Vector3f f0_base(FixedMathCore(171798691LL, true)); // 0.04
	Vector3f F0 = wp::lerp(f0_base, p_mat.albedo, p_mat.metallic);
	Vector3f F = calculate_fresnel_schlick(dot_vh, F0);

	// Specular = (D * G * F) / (4 * dot_nl * dot_nv)
	FixedMathCore denom = FixedMathCore(4LL, false) * dot_nl * dot_nv + MathConstants<FixedMathCore>::unit_epsilon();
	Vector3f specular = (F * (D * G)) / denom;

	// 2. Resolve Diffuse Term (Energy Conserving Lambert)
	Vector3f k_diffuse = Vector3f(one - F.x, one - F.y, one - F.z) * (one - p_mat.metallic);
	Vector3f diffuse = p_mat.albedo / MathConstants<FixedMathCore>::pi();

	// 3. --- Sophisticated Behavior: Anime Light Snap ---
	if (p_is_anime) {
		// Anime Technique: "Threshold Shadowing". 
		// Instead of smooth falloff, we snap the dot_nl to discrete bands.
		FixedMathCore threshold = p_mat.anime_shading_threshold;
		if (dot_nl > threshold) {
			dot_nl = one;
		} else {
			dot_nl = FixedMathCore(858993459LL, true); // 0.2 shadow band
		}
		
		// Style Enhancement: Saturated highlights
		specular *= FixedMathCore(5LL, false);
	}

	// 4. Final Radiance Summation
	r_out_radiance.energy = (k_diffuse * diffuse + specular) * p_light_energy * dot_nl;
	r_out_radiance.energy += p_mat.emission;
}

/**
 * execute_spectral_brdf_sweep()
 * 
 * Master orchestrator for parallel 120 FPS light transport.
 * Processes billions of light-matter interactions using Warp architecture.
 */
void execute_spectral_brdf_sweep(
		KernelRegistry &p_registry,
		const Vector3f &p_sun_dir,
		const Vector3f &p_sun_energy) {

	auto &mat_stream = p_registry.get_stream<MaterialTensor>(COMPONENT_MATERIAL);
	auto &norm_stream = p_registry.get_stream<Vector3f>(COMPONENT_NORMAL);
	auto &view_stream = p_registry.get_stream<Vector3f>(COMPONENT_VIEW_DIR);
	auto &rad_stream = p_registry.get_stream<SpectralRadiance>(COMPONENT_RADIANCE);

	uint64_t count = mat_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &mat_stream, &norm_stream, &view_stream, &rad_stream]() {
			for (uint64_t i = start; i < end; i++) {
				// Style derived from Entity ID handle hash
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 6 == 0);

				brdf_resolution_kernel(
					handle,
					rad_stream[i],
					mat_stream[i],
					norm_stream[i],
					view_stream[i],
					p_sun_dir,
					p_sun_energy,
					anime_mode
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/spectral_tensor_kernel.cpp ---
