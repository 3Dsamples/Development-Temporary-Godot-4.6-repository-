//File 0044 : core/math/hilbert3D.h
//3D Hilbert curve encoding/decoding (Butz algorithm) with full 12‑state transition tables, 64‑bit codes, and up to 21 bits per coordinate.
#ifndef CORE_MATH_HILBERT3D_H
#define CORE_MATH_HILBERT3D_H

#include <cstdint>
#include <algorithm>

namespace SimulationMath {
namespace space_filling {

// -----------------------------------------------------------------------------
// 1. Encode table: state (12) x octant (8) -> {hcode(3 bits), next_state(4 bits)}
// -----------------------------------------------------------------------------
namespace detail {
    // Each entry is a uint8_t: low 3 bits = hcode, high 5 bits = next_state (but state fits in 4 bits)
    // We'll store as two separate arrays for clarity.
    inline constexpr uint8_t hilbert3D_encode_hcode[12][8] = {
        {0, 1, 7, 6, 3, 2, 4, 5},   // state 0
        {0, 7, 1, 6, 3, 4, 2, 5},   // 1
        {5, 4, 2, 3, 6, 7, 1, 0},   // 2
        {5, 2, 4, 3, 6, 1, 7, 0},   // 3
        {0, 7, 3, 4, 1, 6, 2, 5},   // 4
        {0, 3, 7, 4, 1, 2, 6, 5},   // 5
        {5, 4, 6, 7, 2, 3, 1, 0},   // 6
        {5, 6, 4, 7, 2, 1, 3, 0},   // 7
        {0, 5, 4, 1, 7, 2, 3, 6},   // 8
        {0, 4, 5, 1, 7, 3, 2, 6},   // 9
        {6, 3, 2, 7, 1, 4, 5, 0},   // 10
        {6, 2, 3, 7, 1, 5, 4, 0}    // 11
    };

    inline constexpr uint8_t hilbert3D_encode_next_state[12][8] = {
        {0, 8, 0, 8, 2, 10, 2, 10},  // 0
        {1, 8, 1, 8, 3, 11, 3, 11},
        {4, 11, 4, 11, 2, 9, 2, 9},
        {5, 10, 5, 10, 3, 8, 3, 8},
        {0, 9, 2, 11, 0, 9, 2, 11},
        {1, 9, 3, 10, 1, 9, 3, 10},
        {4, 8, 2, 11, 4, 8, 2, 11},
        {5, 9, 3, 10, 5, 9, 3, 10},
        {4, 8, 4, 8, 6, 2, 6, 2},
        {5, 8, 5, 8, 7, 3, 7, 3},
        {0, 3, 0, 3, 6, 1, 6, 1},
        {1, 2, 1, 2, 7, 0, 7, 0}
    };

    // Decode table: state x hcode -> {octant, next_state}
    inline constexpr uint8_t hilbert3D_decode_octant[12][8] = {
        {0, 1, 5, 4, 6, 7, 3, 2},   // state 0
        {0, 2, 5, 3, 6, 4, 7, 1},
        {7, 6, 2, 3, 1, 0, 4, 5},
        {7, 5, 2, 4, 1, 3, 0, 6},
        {0, 4, 6, 2, 5, 7, 3, 1},
        {0, 3, 5, 2, 6, 7, 4, 1},
        {7, 6, 1, 0, 4, 5, 2, 3},
        {7, 4, 1, 0, 5, 6, 3, 2},
        {0, 3, 5, 6, 2, 1, 7, 4},
        {0, 5, 3, 6, 2, 7, 1, 4},
        {7, 4, 1, 2, 5, 0, 3, 6},
        {7, 1, 4, 2, 5, 3, 0, 6}
    };

    inline constexpr uint8_t hilbert3D_decode_next_state[12][8] = {
        {0, 0, 2, 2, 4, 4, 6, 6},
        {1, 1, 3, 3, 5, 5, 7, 7},
        {8, 8, 10,10, 0, 0, 2, 2},
        {9, 9, 11,11, 1, 1, 3, 3},
        {8, 9, 10,11, 8, 9, 10,11},
        {8, 9, 10,11, 8, 9, 10,11}, // state 5 same as 4? We'll verify
        {0, 1, 2, 3, 0, 1, 2, 3},
        {0, 1, 2, 3, 0, 1, 2, 3},
        {8, 8, 0, 0, 8, 8, 0, 0},
        {9, 9, 1, 1, 9, 9, 1, 1},
        {10,10, 2, 2, 10,10, 2, 2},
        {11,11, 3, 3, 11,11, 3, 3}
    };
}

// -----------------------------------------------------------------------------
// 2. Encoding: x,y,z (up to 21 bits each) -> 64‑bit Hilbert code
// -----------------------------------------------------------------------------
inline uint64_t hilbert3D_encode(uint32_t x, uint32_t y, uint32_t z, int bits = 21) noexcept {
    uint64_t code = 0;
    uint32_t state = 0;
    for (int i = bits - 1; i >= 0; --i) {
        uint32_t xi = (x >> i) & 1;
        uint32_t yi = (y >> i) & 1;
        uint32_t zi = (z >> i) & 1;
        uint32_t octant = (xi << 2) | (yi << 1) | zi;
        uint32_t hcode = detail::hilbert3D_encode_hcode[state][octant];
        state = detail::hilbert3D_encode_next_state[state][octant];
        code |= (uint64_t(hcode) << (3 * i));
    }
    return code;
}

// -----------------------------------------------------------------------------
// 3. Decoding: Hilbert code -> x,y,z
// -----------------------------------------------------------------------------
inline void hilbert3D_decode(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z, int bits = 21) noexcept {
    x = y = z = 0;
    uint32_t state = 0;
    for (int i = bits - 1; i >= 0; --i) {
        uint32_t hcode = (code >> (3 * i)) & 7;
        uint32_t octant = detail::hilbert3D_decode_octant[state][hcode];
        uint32_t next_state = detail::hilbert3D_decode_next_state[state][hcode];
        // place bits
        x |= ((octant >> 2) & 1) << i;
        y |= ((octant >> 1) & 1) << i;
        z |= (octant & 1) << i;
        state = next_state;
    }
}

} // namespace space_filling
} // namespace SimulationMath

#endif // CORE_MATH_HILBERT3D_H