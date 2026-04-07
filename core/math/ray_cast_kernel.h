--- START OF FILE core/math/ray_cast_kernel.h ---

#ifndef RAY_CAST_KERNEL_H
#define RAY_CAST_KERNEL_H

#include "core/typedefs.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/math/face3.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * RayCastKernel
 * 
 * Parallel execution kernel for deterministic ray-intersections.
 * Designed for Warp-style zero-copy batching over EnTT component streams.
 * Handles sub-meter precision in galactic-scale sectors.
 */
namespace UniversalSolver {

struct ET_ALIGN_32 Ray {
	Vector3f origin;
	Vector3f direction;
	FixedMathCore max_distance;
	BigIntCore sector_x, sector_y, sector_z; // Galactic anchor
};

struct ET_ALIGN_32 RayHit {
	bool collided = false;
	FixedMathCore distance;
	Vector3f position;
	Vector3f normal;
	BigIntCore entity_id;
};

class RayCastKernel {
public:
	// ------------------------------------------------------------------------
	// Batch Intersection Kernels (SIMD/Warp Optimized)
	// ------------------------------------------------------------------------

	/**
	 * intersect_aabb_batch()
	 * Sweeps a batch of rays against a batch of AABBs.
	 * Returns bit-perfect results for broadphase visibility or sensor logic.
	 */
	static void intersect_aabb_batch(
			const Ray *p_rays,
			const AABBf *p_bounds,
			RayHit *r_hits,
			uint64_t p_count);

	/**
	 * intersect_mesh_batch()
	 * High-fidelity ray-triangle sweep using Moller-Trumbore FixedMath logic.
	 * Optimized for zero-copy access to EnTT face component streams.
	 */
	static void intersect_mesh_batch(
			const Ray *p_rays,
			const Face3f *p_faces,
			uint64_t p_face_count,
			RayHit *r_hits,
			uint64_t p_ray_count);

	// ------------------------------------------------------------------------
	// Galactic Translation Helpers
	// ------------------------------------------------------------------------

	/**
	 * resolve_relative_ray()
	 * Translates a ray origin from its home sector to a target sector.
	 * Prevents precision loss when checking intersections in distant star systems.
	 */
	static ET_SIMD_INLINE Vector3f resolve_relative_ray_origin(
			const Ray &p_ray,
			const BigIntCore &p_target_sx,
			const BigIntCore &p_target_sy,
			const BigIntCore &p_target_sz,
			const FixedMathCore &p_sector_size) {
		
		BigIntCore dx = p_ray.sector_x - p_target_sx;
		BigIntCore dy = p_ray.sector_y - p_target_sy;
		BigIntCore dz = p_ray.sector_z - p_target_sz;

		FixedMathCore off_x = FixedMathCore(static_cast<int64_t>(std::stoll(dx.to_string()))) * p_sector_size;
		FixedMathCore off_y = FixedMathCore(static_cast<int64_t>(std::stoll(dy.to_string()))) * p_sector_size;
		FixedMathCore off_z = FixedMathCore(static_cast<int64_t>(std::stoll(dz.to_string()))) * p_sector_size;

		return p_ray.origin + Vector3f(off_x, off_y, off_z);
	}
};

} // namespace UniversalSolver

#endif // RAY_CAST_KERNEL_H

--- END OF FILE core/math/ray_cast_kernel.h ---
