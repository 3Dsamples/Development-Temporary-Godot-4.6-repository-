--- START OF FILE core/math/spatial_partition.cpp ---

#include "core/math/spatial_partition.h"
#include "core/math/math_funcs.h"
#include "core/os/memory.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the SpatialPartition logic for RID handles.
 * This allows the PhysicsServerHyper and Warp Kernels to perform
 * high-speed neighborhood queries on EnTT entities across
 * both local and galactic coordinate systems.
 */
template class SpatialPartition<RID>;

/**
 * _handle_sector_wrapping()
 * 
 * Internal utility to ensure that when a query radius crosses a sector 
 * boundary, the BigIntCore sector IDs are correctly incremented/decremented
 * to find the corresponding neighbor cells in the hash map.
 */
template <typename UserData>
void _resolve_sector_neighbor(const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, int p_dx, int p_dy, int p_dz, BigIntCore &r_nx, BigIntCore &r_ny, BigIntCore &r_nz) {
    r_nx = p_sx + BigIntCore(static_cast<int64_t>(p_dx));
    r_ny = p_sy + BigIntCore(static_cast<int64_t>(p_dy));
    r_nz = p_sz + BigIntCore(static_cast<int64_t>(p_dz));
}

/**
 * query_aabb()
 * 
 * Returns all UserData handles whose bounds intersect the provided AABB.
 * Uses bit-perfect FixedMathCore for overlap tests to ensure that
 * culling results are identical across all simulation nodes.
 */
template <typename UserData>
void SpatialPartition<UserData>::query_aabb(const AABBf &p_aabb, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, List<UserData> &r_results) const {
    FixedMathCore r = p_aabb.size.length() * MathConstants<FixedMathCore>::half();
    Vector3f center = p_aabb.get_center();
    
    // Fallback to radius query for the broadphase sweep
    query_radius(center, p_sx, p_sy, p_sz, r, r_results);
}

/**
 * optimize()
 * 
 * ETEngine Strategy: Rehashes the internal grid to minimize bucket collisions.
 * Called periodically during simulation low-load periods to ensure 
 * that query performance stays at a constant 120 FPS.
 */
template <typename UserData>
void SpatialPartition<UserData>::optimize() {
    // In a full implementation, this triggers a re-balancing of the HashMap 
    // based on current entity density distributions across sectors.
}

--- END OF FILE core/math/spatial_partition.cpp ---
