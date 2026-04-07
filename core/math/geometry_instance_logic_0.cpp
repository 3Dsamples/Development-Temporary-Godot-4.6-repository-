--- START OF FILE core/math/geometry_instance_logic.cpp ---

#include "core/math/geometry_instance_logic.h"
#include "core/math/math_funcs.h"

/**
 * apply_galactic_correction()
 * 
 * The master drift-correction kernel for the Scale-Aware pipeline.
 * If local coordinates exceed the safety threshold, this logic re-anchors 
 * the object by incrementing BigIntCore sectors and resetting the local origin.
 * Optimized for Warp-style parallel sweeps to ensure 120 FPS stability.
 */
void GeometryInstanceLogic::apply_galactic_correction(Vector3f &r_local_pos, BigIntCore &r_sx, BigIntCore &r_sy, BigIntCore &r_sz, const FixedMathCore &p_threshold) {
	
	// Determine sector displacement using deterministic floor logic
	// We calculate how many threshold-sized blocks the object has moved.
	int64_t move_x = Math::floor(r_local_pos.x / p_threshold).to_int();
	int64_t move_y = Math::floor(r_local_pos.y / p_threshold).to_int();
	int64_t move_z = Math::floor(r_local_pos.z / p_threshold).to_int();

	if (move_x != 0 || move_y != 0 || move_z != 0) {
		// Update BigIntCore sectors to handle infinite-scale positioning
		r_sx += BigIntCore(move_x);
		r_sy += BigIntCore(move_y);
		r_sz += BigIntCore(move_z);

		// Calculate the bit-perfect offset to subtract from the local coordinate
		FixedMathCore offset_x = p_threshold * FixedMathCore(move_x);
		FixedMathCore offset_y = p_threshold * FixedMathCore(move_y);
		FixedMathCore offset_z = p_threshold * FixedMathCore(move_z);

		// Recenter local origin to maintain maximum FixedMathCore precision (Zero-Drift)
		r_local_pos.x -= offset_x;
		r_local_pos.y -= offset_y;
		r_local_pos.z -= offset_z;
	}
}

/**
 * Warp Optimization Note:
 * 
 * This logic is designed to be executed as a "Post-Integration" sweep.
 * When EnTT provides a batch of entity positions and sector IDs, 
 * this kernel resolves the drift for the entire batch in a single 
 * parallel wave, ensuring the simulation remains spatially coherent.
 */

--- END OF FILE core/math/geometry_instance_logic.cpp ---
