--- START OF FILE core/math/fuzzy_logic_kernel.cpp ---

#include "core/math/fuzzy_logic_kernel.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_triangle_membership()
 * 
 * Deterministic fuzzification using a triangular function.
 * Returns a grade in FixedMathCore [0..1].
 */
static _FORCE_INLINE_ FixedMathCore calculate_triangle_membership(
		FixedMathCore p_x, 
		FixedMathCore p_a, 
		FixedMathCore p_b, 
		FixedMathCore p_c) {
	
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	if (p_x <= p_a || p_x >= p_c) return zero;
	if (p_x == p_b) return one;

	if (p_x < p_b) {
		return (p_x - p_a) / (p_b - p_a);
	} else {
		return (p_c - p_x) / (p_c - p_b);
	}
}

/**
 * calculate_trapezoid_membership()
 * 
 * Deterministic fuzzification using a trapezoidal function.
 * Used for "Plateau" states like "Nominal Temperature Range".
 */
static _FORCE_INLINE_ FixedMathCore calculate_trapezoid_membership(
		FixedMathCore p_x, 
		FixedMathCore p_a, 
		FixedMathCore p_b, 
		FixedMathCore p_c, 
		FixedMathCore p_d) {
	
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	FixedMathCore term1 = (p_x - p_a) / (p_b - p_a + MathConstants<FixedMathCore>::unit_epsilon());
	FixedMathCore term2 = (p_d - p_x) / (p_d - p_c + MathConstants<FixedMathCore>::unit_epsilon());

	FixedMathCore result = wp::min(term1, one);
	result = wp::min(result, term2);

	return wp::max(zero, result);
}

/**
 * Warp Kernel: FuzzyInferenceKernel
 * 
 * Executes a batch of fuzzy rules on an entity's physical state.
 * 1. Fuzzification: Translates heat/stress/distance into membership degrees.
 * 2. Rule Evaluation: Combines antecedents using Zadeh AND (min) / OR (max).
 * 3. Aggregation: Collects fuzzy outputs for the defuzzification pass.
 */
void fuzzy_inference_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_fuzzy_state_grade,
		const FixedMathCore &p_input_val,
		const FixedMathCore *p_set_points, // [a, b, c, d]
		uint32_t p_rule_type,
		bool p_is_anime) {

	FixedMathCore grade;

	if (p_rule_type == 0) { // Triangular
		grade = calculate_triangle_membership(p_input_val, p_set_points[0], p_set_points[1], p_set_points[2]);
	} else { // Trapezoidal
		grade = calculate_trapezoid_membership(p_input_val, p_set_points[0], p_set_points[1], p_set_points[2], p_set_points[3]);
	}

	// --- Sophisticated Real-Time Behavior: Anime Snap-Logic ---
	if (p_is_anime) {
		// Anime AI Technique: "Binary Certainty".
		// Instead of smooth fuzzy transitions, AI decisions 'snap' to 1.0 or 0.0 
		// when exceeding a confidence threshold, creating sharp dramatic reactions.
		FixedMathCore snap_threshold(2147483648LL, true); // 0.5
		grade = wp::step(snap_threshold, grade);
	}

	r_fuzzy_state_grade = grade;
}

/**
 * Warp Kernel: CentroidDefuzzifierKernel
 * 
 * Converts fuzzy results back to a "Crisp" physical value (Actuation).
 * Formula: x_crisp = Sum(mu_i * x_i) / Sum(mu_i)
 * Essential for resolving yielding intensity or AI movement speed.
 */
void centroid_defuzzifier_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_crisp_output,
		const FixedMathCore *p_membership_grades,
		const FixedMathCore *p_centroids,
		uint32_t p_rule_count) {

	FixedMathCore numerator = MathConstants<FixedMathCore>::zero();
	FixedMathCore denominator = MathConstants<FixedMathCore>::zero();

	for (uint32_t i = 0; i < p_rule_count; i++) {
		FixedMathCore mu = p_membership_grades[i];
		numerator += mu * p_centroids[i];
		denominator += mu;
	}

	if (unlikely(denominator.get_raw() == 0)) {
		r_crisp_output = MathConstants<FixedMathCore>::zero();
		return;
	}

	r_crisp_output = numerator / denominator;
}

/**
 * execute_fuzzy_logic_sweep()
 * 
 * Master orchestrator for parallel AI and material state resolution.
 * Partitions EnTT behavior components into worker batches.
 */
void execute_fuzzy_logic_sweep(
		KernelRegistry &p_registry,
		const BigIntCore &p_entity_count,
		const FixedMathCore &p_delta) {

	auto &input_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_PHYSICAL_INPUT);
	auto &grade_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_FUZZY_GRADE);
	auto &output_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_CRISP_OUTPUT);

	uint64_t total = static_cast<uint64_t>(std::stoll(p_entity_count.to_string()));
	if (total == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	// PASS 1: Parallel Fuzzification & Inference
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &input_stream, &grade_stream]() {
			// Deterministic Set Points (e.g., AI "Aggression" zones)
			FixedMathCore set_points[4] = { FixedMathCore(0LL), FixedMathCore(5LL), FixedMathCore(10LL), FixedMathCore(15LL) };

			for (uint64_t i = start; i < end; i++) {
				BigIntCore handle = BigIntCore(static_cast<int64_t>(i));
				bool anime_mode = (handle.hash() % 10 == 0);

				fuzzy_inference_kernel(handle, grade_stream[i], input_stream[i], set_points, 1, anime_mode);
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	// PASS 2: Parallel Centroid Defuzzification
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &grade_stream, &output_stream]() {
			// Deterministic Centroids (Action Mappings)
			FixedMathCore action_centroids[1] = { FixedMathCore(100LL, false) };

			for (uint64_t i = start; i < end; i++) {
				centroid_defuzzifier_kernel(BigIntCore(static_cast<int64_t>(i)), output_stream[i], &grade_stream[i], action_centroids, 1);
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * resolve_macro_fuzzy_priority()
 * 
 * Sophisticated Feature: Integrates BigIntCore for high-mass decision making.
 * Scales the fuzzy result by trillions of currency units or massive populations.
 */
BigIntCore resolve_macro_fuzzy_priority(
		const FixedMathCore &p_fuzzy_grade,
		const BigIntCore &p_macro_magnitude) {
	
	if (p_fuzzy_grade.get_raw() <= 0) return BigIntCore(0LL);
	
	// result = (grade * magnitude) / 1.0 (Fixed point scale shift)
	BigIntCore grade_bi(p_fuzzy_grade.get_raw());
	BigIntCore result = (p_macro_magnitude * grade_bi) / BigIntCore(FixedMathCore::ONE_RAW);
	
	return result;
}

} // namespace UniversalSolver

--- END OF FILE core/math/fuzzy_logic_kernel.cpp ---
