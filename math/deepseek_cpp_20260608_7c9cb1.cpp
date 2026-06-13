// File 02: modules/gaia/src/bvh/morton_code.h

#ifndef GAIA_BVH_MORTON_CODE_H
#define GAIA_BVH_MORTON_CODE_H

#include "core/typedefs.h"

// Morton code (Z-order curve) generation for 3D spatial sorting.
// Used by the LBVH builder to map 3D AABB centroids into 1D keys.

namespace gaia::bvh {

/**
 * Bit interleaving: expand a 10-bit value into a 30-bit field with
 * one input bit followed by two zero bits (used for 30-bit Morton code).
 */
inline uint32_t expand_bits_10(uint32_t v) {
	// Original Gaia used classic part1by1 formula for 32-bit.
	// Here we produce 30-bit expanded pattern: bits at positions 0,3,6...
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

/**
 * Compute 30-bit Morton code from three 10-bit coordinates.
 * Suitable for scenes with up to 1024^3 spatial grid cells.
 */
inline uint32_t morton_code_30(uint32_t x, uint32_t y, uint32_t z) {
	x = MIN(x, 0x3FFu); // 10 bits
	y = MIN(y, 0x3FFu);
	z = MIN(z, 0x3FFu);
	return (expand_bits_10(z) << 2) | (expand_bits_10(y) << 1) | expand_bits_10(x);
}

/**
 * Bit interleaving: expand a 21-bit value into a 63-bit field with
 * one input bit followed by two zero bits (for 64-bit Morton code).
 */
inline uint64_t expand_bits_21(uint64_t v) {
	v = (v * 0x0000000100000001ull) & 0xFF000000FF000000ull;
	v = (v * 0x0000010000000100ull) & 0x0F000F000F000F00ull;
	v = (v * 0x0001000000010000ull) & 0xC30C30C30C30C30Cull;
	v = (v * 0x0004000000040000ull) & 0x4924924924924924ull; // Corrected mask for 63-bit expansion
	return v;
}

/**
 * Compute 64-bit Morton code from three 21-bit coordinates.
 * Supports huge worlds with up to 2^21 grid cells per axis.
 */
inline uint64_t morton_code_64(uint64_t x, uint64_t y, uint64_t z) {
	x = MIN(x, 0x1FFFFFull); // 21 bits
	y = MIN(y, 0x1FFFFFull);
	z = MIN(z, 0x1FFFFFull);
	return (expand_bits_21(z) << 2) | (expand_bits_21(y) << 1) | expand_bits_21(x);
}

} // namespace gaia::bvh

#endif // GAIA_BVH_MORTON_CODE_H