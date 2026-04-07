--- START OF FILE core/math/geometry_2d_physics.cpp ---

#include "core/math/geometry_2d.h"
#include "core/math/math_funcs.h"
#include "core/math/random_pcg.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Explicit Template Instantiation
 * 
 * Compiles the 2D physics logic for the deterministic FixedMathCore tier.
 * These symbols enable Warp kernels to perform parallel vertex-level 
 * deformation and structural failure analysis on 2D polygons at 120 FPS.
 */
template class Geometry2D<FixedMathCore>;

/**
 * apply_impact_crater()
 * 
 * Simulates a localized high-energy impact on a 2D surface.
 * Displaces vertices based on a quadratic falloff model.
 * 
 * r_vertices: The SoA vertex stream from an EnTT component.
 * p_impact_point: The epicenter in deterministic coordinates.
 * p_force: Magnitude of the displacement.
 * p_radius: Area of effect.
 */
template <typename T>
void Geometry2D<T>::apply_impact_crater(Vector<Vector2<T>> &r_vertices, const Vector2<T> &p_point, T p_force, T p_radius) {
	T r2 = p_radius * p_radius;
	uint32_t count = r_vertices.size();
	Vector2<T> *ptr = r_vertices.ptrw();

	for (uint32_t i = 0; i < count; i++) {
		Vector2<T> diff = ptr[i] - p_point;
		T d2 = diff.length_squared();
		
		if (d2 < r2) {
			T dist = Math::sqrt(d2);
			// falloff = (radius - distance) / radius
			T weight = (p_radius - dist) / p_radius;
			// Quadratic intensity for realistic impact curves
			T displacement = p_force * (weight * weight);
			
			if (dist > T(42949LL, true)) { // Epsilon check
				ptr[i] += diff.normalized() * displacement;
			} else {
				// Handle direct center hit with deterministic nudge
				ptr[i].x += displacement;
			}
		}
	}
}

/**
 * apply_procedural_tear()
 * 
 * Simulates structural failure under tension. 
 * If an edge in the 2D manifold is stretched beyond its elastic limit, 
 * this kernel splits the edge and inserts a jagged vertex.
 */
template <typename T>
void Geometry2D<T>::apply_procedural_tear(Vector<Vector2<T>> &r_polygon, uint32_t p_edge_index, T p_energy) {
	uint32_t v_count = r_polygon.size();
	if (unlikely(v_count < 3 || p_edge_index >= v_count)) {
		return;
	}

	uint32_t next_idx = (p_edge_index + 1) % v_count;
	Vector2<T> v1 = r_polygon[p_edge_index];
	Vector2<T> v2 = r_polygon[next_idx];

	// Calculate midpoint and normal for the tear direction
	Vector2<T> edge_vec = v2 - v1;
	Vector2<T> mid = v1 + (edge_vec * MathConstants<T>::half());
	Vector2<T> normal = edge_vec.perpendicular().normalized();

	// Use deterministic RandomPCG for jaggedness
	RandomPCG pcg;
	pcg.seed(static_cast<uint64_t>(v1.x.get_raw() ^ v2.y.get_raw()));
	
	T noise = pcg.randf() * p_energy;
	Vector2<T> tear_point = mid - (normal * noise);

	// Insert the new vertex into the EnTT-managed polygon stream
	r_polygon.insert(next_idx, tear_point);
}

/**
 * apply_thermal_buckling()
 * 
 * Pseudo-3D effect in 2D space. Simulates material expansion due to 
 * high-temperature delta (e.g., laser heating or atmospheric friction).
 * Vertices are pushed outward from the heat source, creating warping.
 */
template <typename T>
void Geometry2D<T>::apply_thermal_buckling(Vector<Vector2<T>> &r_vertices, const Vector2<T> &p_origin, T p_temp_delta) {
	uint32_t count = r_vertices.size();
	Vector2<T> *ptr = r_vertices.ptrw();
	
	// Thermal expansion coefficient (Simplified bit-perfect constant)
	T expansion_coeff(42949LL, true); // 0.00001

	for (uint32_t i = 0; i < count; i++) {
		Vector2<T> diff = ptr[i] - p_origin;
		T dist = diff.length();
		
		// Intensity decays linearly with distance from heat center
		T thermal_influence = p_temp_delta / (dist + MathConstants<T>::one());
		T expansion = thermal_influence * expansion_coeff;
		
		if (dist > T(42949LL, true)) {
			ptr[i] += diff.normalized() * expansion;
		}
	}
}

/**
 * apply_torsional_screw_2d()
 * 
 * Simulates mechanical "Screwing" or twisting force on a 2D surface.
 * Rotates vertices around a center point based on torque magnitude.
 */
template <typename T>
void Geometry2D<T>::apply_torsional_screw_2d(Vector<Vector2<T>> &r_polygon, const Vector2<T> &p_center, T p_torque) {
	uint32_t count = r_polygon.size();
	Vector2<T> *ptr = r_polygon.ptrw();

	for (uint32_t i = 0; i < count; i++) {
		Vector2<T> rel = ptr[i] - p_center;
		T dist = rel.length();
		
		// Angular displacement proportional to torque and distance
		T angle = p_torque * dist;
		
		// Bit-perfect FixedMath rotation
		ptr[i] = p_center + rel.rotated(angle);
	}
}

--- END OF FILE core/math/geometry_2d_physics.cpp ---
