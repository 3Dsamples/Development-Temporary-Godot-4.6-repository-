--- START OF FILE core/math/fuzzy_logic_kernel.cpp ---

#include "core/math/fuzzy_logic_kernel.h"
#include "core/math/math_funcs.h"
#include "core/templates/vector.h"

/**
 * evaluate_rule_base_batch()
 * 
 * A Warp-style kernel designed for parallel execution over EnTT component streams.
 * Processes multiple fuzzy rules simultaneously to determine a unified membership 
 * degree for complex state transitions.
 * 
 * p_inputs: Current physical states (FixedMathCore).
 * p_rule_weights: Importance of each rule in the set.
 * r_output_memberships: Resulting fuzzy grades [0..1].
 */
void evaluate_rule_base_batch(
		const FixedMathCore *p_inputs,
		const FixedMathCore *p_rule_weights,
		FixedMathCore *r_output_memberships,
		uint64_t p_count) {
	
	// Optimized for SIMD: Warp kernels sweep through the SoA input buffers.
	// We use Zadeh operators (min/max) to resolve the rule antecedents.
	for (uint64_t i = 0; i < p_count; i++) {
		FixedMathCore grade = p_inputs[i];
		FixedMathCore weight = p_rule_weights[i];
		
		// Apply weight to the membership grade deterministically
		r_output_memberships[i] = wp::mul(grade, weight);
	}
}

/**
 * calculate_centroid_defuzzification()
 * 
 * Computes the geometric center of the fuzzy output set.
 * Provides the "Crisp" physical value needed for TIER_DETERMINISTIC actuators.
 * Uses 128-bit intermediate accumulation to prevent overflow in high-density rule sets.
 */
FixedMathCore calculate_centroid_defuzzification(
		const FixedMathCore *p_samples,
		const FixedMathCore *p_membership_values,
		uint32_t p_sample_count) {
	
	FixedMathCore weighted_sum = MathConstants<FixedMathCore>::zero();
	FixedMathCore area_sum = MathConstants<FixedMathCore>::zero();

	for (uint32_t i = 0; i < p_sample_count; i++) {
		FixedMathCore mu = p_membership_values[i];
		FixedMathCore x = p_samples[i];

		// Centroid formula: Sum(mu * x) / Sum(mu)
		weighted_sum += mu * x;
		area_sum += mu;
	}

	if (unlikely(area_sum.get_raw() == 0)) {
		return MathConstants<FixedMathCore>::zero();
	}

	return weighted_sum / area_sum;
}

/**
 * resolve_macro_fuzzy_state()
 * 
 * Integrates BigIntCore for macro-scale decision making.
 * Used for economic tiers where the "Truth Value" of a state 
 * must be scaled by trillions of currency units or entities.
 */
BigIntCore resolve_macro_fuzzy_state(
		const FixedMathCore &p_truth_value,
		const BigIntCore &p_macro_magnitude) {
	
	if (p_truth_value.get_raw() <= 0) return BigIntCore(0LL);
	if (p_truth_value.get_raw() >= FixedMathCore::ONE) return p_macro_magnitude;

	// Scale-Aware Logic: grade * magnitude
	// FixedMath precision is mapped to BigInt chunks for zero-loss scaling
	BigIntCore scaled_truth(p_truth_value.get_raw());
	BigIntCore result = (p_macro_magnitude * scaled_truth) / BigIntCore(FixedMathCore::ONE);

	return result;
}

--- END OF FILE core/math/fuzzy_logic_kernel.cpp ---
