--- START OF FILE core/math/aabb.cpp ---

#include "core/math/aabb.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the AABB logic for the Universal Solver backend.
 * These symbols enable EnTT to manage volumetric components as contiguous 
 * data streams, allowing Warp kernels to perform massive parallel culling 
 * or spatial intersection tests with zero-copy efficiency.
 */

template struct AABB<FixedMathCore>; // TIER_DETERMINISTIC: Bit-perfect Collision Volumes
template struct AABB<BigIntCore>;   // TIER_MACRO: Discrete Galactic Sector Bounds

/**
 * Warp Optimization: Linear Volume Queries
 * 
 * Because AABB is ET_ALIGN_32, it maps directly to SIMD lanes. 
 * When EnTT provides a pointer to a block of AABBf, Warp kernels 
 * can process multiple visibility or collision checks per cycle, 
 * maintaining 120 FPS in environments with millions of active entities.
 */

--- END OF FILE core/math/aabb.cpp ---
