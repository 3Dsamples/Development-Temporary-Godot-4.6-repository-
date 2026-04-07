--- START OF FILE core/math/spatial_partition.cpp ---

#include "core/math/spatial_partition.h"
#include "core/math/math_funcs.h"
#include "core/os/memory.h"

/**
 * _normalize_coordinate()
 * 
 * Master drift-correction logic for the spatial partition.
 * Ensures that any local FixedMathCore position is clamped within its 
 * corresponding BigIntCore sector bounds.
 * Threshold is set to 10,000 units to maintain maximum Q32.32 precision.
 */
template <typename UserData>
void _normalize_coordinate(Vector3f &r_pos, BigIntCore &r_sx, BigIntCore &r_sy, BigIntCore &r_sz) {
	const FixedMathCore threshold(10000LL, false);

	int64_t move_x = Math::floor(r_pos.x / threshold).to_int();
	int64_t move_y = Math::floor(r_pos.y / threshold).to_int();
	int64_t move_z = Math::floor(r_pos.z / threshold).to_int();

	if (move_x != 0) {
		r_sx += BigIntCore(move_x);
		r_pos.x -= threshold * FixedMathCore(move_x);
	}
	if (move_y != 0) {
		r_sy += BigIntCore(move_y);
		r_pos.y -= threshold * FixedMathCore(move_y);
	}
	if (move_z != 0) {
		r_sz += BigIntCore(move_z);
		r_pos.z -= threshold * FixedMathCore(move_z);
	}
}

/**
 * query_aabb()
 * 
 * Performs a bit-perfect volumetric search across sector boundaries.
 * 1. Identifies the range of sectors touched by the AABB.
 * 2. Iterates through the discrete BigInt grid to fetch buckets.
 * 3. Filters results using deterministic FixedMath intersection tests.
 */
template <typename UserData>
void SpatialPartition<UserData>::query_aabb(const AABBf &p_aabb, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, List<UserData> &r_results) const {
	Vector3f start_pos = p_aabb.position;
	Vector3f end_pos = p_aabb.position + p_aabb.size;
	
	BigIntCore sx_start = p_sx, sy_start = p_sy, sz_start = p_sz;
	BigIntCore sx_end = p_sx, sy_end = p_sy, sz_end = p_sz;

	_normalize_coordinate<UserData>(start_pos, sx_start, sy_start, sz_start);
	_normalize_coordinate<UserData>(end_pos, sx_end, sy_end, sz_end);

	// Calculate quantized grid range
	int64_t cx_start = Math::floor(start_pos.x / cell_size).to_int();
	int64_t cy_start = Math::floor(start_pos.y / cell_size).to_int();
	int64_t cz_start = Math::floor(start_pos.z / cell_size).to_int();
	
	int64_t cx_end = Math::ceil(end_pos.x / cell_size).to_int();
	int64_t cy_end = Math::ceil(end_pos.y / cell_size).to_int();
	int64_t cz_end = Math::ceil(end_pos.z / cell_size).to_int();

	// Galactic Sector Loop
	for (BigIntCore s_x = sx_start; s_x <= sx_end; s_x += BigIntCore(1LL)) {
		for (BigIntCore s_y = sy_start; s_y <= sy_end; s_y += BigIntCore(1LL)) {
			for (BigIntCore s_z = sz_start; s_z <= sz_end; s_z += BigIntCore(1LL)) {
				
				// Local Cell Loop within sector
				for (int64_t i = cx_start; i <= cx_end; i++) {
					for (int64_t j = cy_start; j <= cy_end; j++) {
						for (int64_t k = cz_start; k <= cz_end; k++) {
							
							SectorKey key;
							key.sx = s_x; key.sy = s_y; key.sz = s_z;
							key.cx = i; key.cy = j; key.cz = k;

							if (grid.has(key)) {
								const List<Element*> &elements = grid[key];
								for (const typename List<Element*>::Element *E = elements.front(); E; E = E->next()) {
									// Final narrowed check using bit-perfect AABB vs Point
									if (p_aabb.has_point(E->get()->position)) {
										r_results.push_back(E->get()->data);
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

/**
 * rebuild_grid()
 * 
 * Performs a zero-copy re-indexing of all elements.
 * Critical for maintaining 120 FPS after a high-speed origin shift.
 */
template <typename UserData>
void SpatialPartition<UserData>::rebuild_grid() {
	grid.clear();
	for (auto it = element_registry.begin(); it.is_valid(); it.next()) {
		Element *e = it.value();
		_compute_key(e->position, e->sx, e->sy, e->sz, e->current_key);
		grid[e->current_key].push_back(e);
	}
}

/**
 * get_element_count()
 * Telemetry integration via BigIntCore.
 */
template <typename UserData>
BigIntCore SpatialPartition<UserData>::get_element_count() const {
	return BigIntCore(static_cast<int64_t>(element_registry.size()));
}

/**
 * Explicit Instantiations
 * 
 * These compiled symbols are linked to the PhysicsServerHyper and Warp Kernels.
 */
template class SpatialPartition<uint32_t>;
template class SpatialPartition<RID>;
template class SpatialPartition<BigIntCore>;

--- END OF FILE core/math/spatial_partition.cpp ---
