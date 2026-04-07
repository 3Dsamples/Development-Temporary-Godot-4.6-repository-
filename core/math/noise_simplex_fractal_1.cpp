--- START OF FILE core/math/noise_simplex_fractal.cpp ---

#include "core/math/noise_simplex_fractal.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the machine code for the simulation tiers:
 * - SimplexNoiseFractalf: Bit-perfect multi-octave noise for physics, 
 *   visuals, and procedural generation (FixedMathCore).
 * 
 * This allows the Universal Solver's background worker threads to invoke 
 * fractal sampling logic with zero-copy efficiency from the EnTT registry.
 */

template struct SimplexNoiseFractal<FixedMathCore>;

/**
 * Global Deterministic Procedural Constants
 * 
 * Pre-defined configurations for common natural phenomena.
 * strictly using raw bits to ensure no runtime FPU drift.
 */

// Cloud Noise: 6 Octaves, 0.5 persistence, 2.18 lacunarity
void get_cloud_noise_preset(SimplexNoiseFractalf &r_noise) {
	r_noise.set_parameters(
		6, 
		FixedMathCore(2147483648LL, true), // 0.5
		FixedMathCore(9363028377LL, true)  // 2.18
	);
}

// Terrain/Crust Noise: 8 Octaves, 0.45 persistence, 2.0 lacunarity
void get_terrain_noise_preset(SimplexNoiseFractalf &r_noise) {
	r_noise.set_parameters(
		8, 
		FixedMathCore(1932735283LL, true), // 0.45
		FixedMathCore(2LL, false)           // 2.0
	);
}

/**
 * Architectural Coherence Validation:
 * 
 * Because SimplexNoiseFractal is ET_ALIGN_32 and its parameters are stored
 * as FixedMathCore tensors, the Warp kernels can sample entire batches 
 * of noise in parallel. In a 120 FPS simulation, this allows for 
 * real-time "Domain Warping" where one noise field displaces another 
 * to create turbulent fluid effects or organic flesh textures. 
 * By using BigIntCore hashes for seeding, we guarantee that 
 * procedural galaxies are identical across all client instances.
 */

--- END OF FILE core/math/noise_simplex_fractal.cpp ---
