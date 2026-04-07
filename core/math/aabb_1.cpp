--- START OF FILE core/math/aabb.cpp ---

#include "core/math/aabb.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the machine code for:
 * - AABBf: Bit-perfect collision volumes and physics broadphase (FixedMathCore).
 * - AABBb: Discrete galactic sector bounds and macro-volume triggers (BigIntCore).
 */
template struct AABB<FixedMathCore>;
template struct AABB<BigIntCore>;

// ============================================================================
// Global Deterministic Constants (FixedMathCore Q32.32)
// ============================================================================

/**
 * Static Constants for AABBf
 * 
 * Initialized with raw bit patterns to ensure no runtime FPU involvement.
 * Optimized for Warp-style parallel sweeps during frustum culling.
 */

// Zero-Size AABB at Origin
const AABBf AABBf_ZERO = AABBf(
	Vector3f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true)),
	Vector3f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true))
);

// Unit AABB centered at (0.5, 0.5, 0.5)
const AABBf AABBf_UNIT = AABBf(
	Vector3f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true)),
	Vector3f(FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(FixedMathCore::ONE_RAW, true))
);

// ============================================================================
// Global Deterministic Constants (BigIntCore)
// ============================================================================

const AABBb AABBb_ZERO = AABBb(
	Vector3b(BigIntCore(0LL), BigIntCore(0LL), BigIntCore(0LL)),
	Vector3b(BigIntCore(0LL), BigIntCore(0LL), BigIntCore(0LL))
);

/**
 * Sophisticated Scale-Aware Implementation:
 * 
 * Because AABB is ET_ALIGN_32, it maps perfectly to modern CPU SIMD lanes.
 * In a 120 FPS simulation, the BroadphaseRehashKernel (Turn 41) utilizes
 * these pre-compiled symbols to perform millions of volume-intersection
 * tests per second. By using FixedMathCore, the "Support" vertices 
 * used in GJK Narrowphase checks are bit-identical on every node, 
 * preventing the "jitter" or "pop" in physics resolution that occurs 
 * when standard floating-point engines handle large coordinate ranges.
 */

--- END OF FILE core/math/aabb.cpp ---
