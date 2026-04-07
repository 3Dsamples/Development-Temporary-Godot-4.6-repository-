--- START OF FILE core/math/kepler_orbit_logic.cpp ---

#include "core/math/kepler_orbit_solver.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * nbody_gravitational_perturbation_kernel()
 * 
 * A high-performance Warp kernel that calculates the sum of gravitational 
 * forces acting on a batch of entities. 
 * Optimized for EnTT Structure of Arrays (SoA) access.
 * 
 * p_positions: SoA stream of body positions.
 * p_masses: SoA stream of body masses (BigIntCore).
 * r_accelerations: Output SoA stream for resulting force vectors.
 */
void nbody_gravitational_perturbation_kernel(
		const Vector3f *p_positions,
		const BigIntCore *p_masses,
		Vector3f *r_accelerations,
		uint64_t p_count,
		const FixedMathCore &p_G) {

	// ETEngine Strategy: Use a tiled approach for the O(N^2) problem
	// to maximize L1/L2 cache hits for FixedMathCore coordinates.
	for (uint64_t i = 0; i < p_count; i++) {
		Vector3f total_accel;
		Vector3f pos_i = p_positions[i];

		for (uint64_t j = 0; j < p_count; j++) {
			if (i == j) continue;

			Vector3f diff = p_positions[j] - pos_i;
			FixedMathCore dist_sq = diff.length_squared();

			// Softening factor to prevent division by zero in dense clusters
			FixedMathCore epsilon(4294LL, true); // 0.000001
			dist_sq += epsilon;

			// Acceleration a = G * M / r^2
			// Convert BigInt mass to FixedMath for the kernel math
			FixedMathCore m_j(static_cast<int64_t>(std::stoll(p_masses[j].to_string())));
			FixedMathCore magnitude = (p_G * m_j) / dist_sq;

			total_accel += diff.normalized() * magnitude;
		}
		r_accelerations[i] = total_accel;
	}
}

/**
 * apply_atmospheric_drag_kernel()
 * 
 * Simulates physical interaction behavior for arbitrary bodies entering 
 * planetary atmospheres. Calculates drag force based on local density.
 */
void apply_atmospheric_drag_kernel(
		Vector3f *r_velocities,
		const Vector3f *p_positions,
		const FixedMathCore *p_drag_coeffs,
		const FixedMathCore *p_atm_densities,
		uint64_t p_count,
		const FixedMathCore &p_delta) {

	for (uint64_t i = 0; i < p_count; i++) {
		Vector3f &vel = r_velocities[i];
		FixedMathCore speed_sq = vel.length_squared();
		if (speed_sq.get_raw() == 0) continue;

		// Drag Equation: Fd = 1/2 * rho * v^2 * Cd * A
		// rho (density) provided by AtmosphericScattering logic
		FixedMathCore drag_mag = MathConstants<FixedMathCore>::half() * p_atm_densities[i] * speed_sq * p_drag_coeffs[i];
		
		Vector3f drag_force = -vel.normalized() * drag_mag;
		vel += drag_force * p_delta;
	}
}

/**
 * celestial_tide_deformation_kernel()
 * 
 * Advanced Feature: Calculates tidal forces that trigger structural 
 * fatigue in deformable bodies near massive gravity wells.
 */
void celestial_tide_deformation_kernel(
		FixedMathCore *r_fatigue_levels,
		const Vector3f *p_positions,
		const BigIntCore &p_parent_mass,
		const Vector3f &p_parent_pos,
		uint64_t p_count,
		const FixedMathCore &p_G) {

	FixedMathCore m_parent(static_cast<int64_t>(std::stoll(p_parent_mass.to_string())));

	for (uint64_t i = 0; i < p_count; i++) {
		FixedMathCore dist = (p_positions[i] - p_parent_pos).length();
		// Tidal Force gradient approximation: F_tide ~ 2GM / r^3
		FixedMathCore dist_cubed = dist * dist * dist;
		FixedMathCore tide_stress = (FixedMathCore(2LL, false) * p_G * m_parent) / dist_cubed;

		// Accumulate fatigue in the material tensor
		r_fatigue_levels[i] += tide_stress * FixedMathCore(42949LL, true); // Scaled
	}
}

--- END OF FILE core/math/kepler_orbit_logic.cpp ---
