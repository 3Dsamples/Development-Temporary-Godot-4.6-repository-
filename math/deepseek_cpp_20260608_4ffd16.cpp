// File 31: modules/gaia/src/spatial_query/spatial_hash.h

#ifndef GAIA_SPATIAL_QUERY_SPATIAL_HASH_H
#define GAIA_SPATIAL_QUERY_SPATIAL_HASH_H

#include "core/math/vector3.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace gaia::spatial {

/**
 * A spatial hash grid for fast neighbour searches.
 *
 * Maps 3D points to integer cell coordinates and stores arbitrary
 * integer keys (e.g., vertex indices) in cells. Provides O(1) average
 * queries for a given cell.
 */
class SpatialHash {
public:
	// Cell size controls the grid resolution.
	SpatialHash(real_t p_cell_size) : cell_size(p_cell_size), inv_cell_size(1.0 / p_cell_size) {
		ERR_FAIL_COND(p_cell_size <= 0);
	}

	// Insert a spatial key at position.
	void insert(int32_t p_key, const Vector3 &p_position) {
		uint64_t cell = compute_cell_index(p_position);
		grid[cell].push_back(p_key);
	}

	// Remove a key from the cell that contains its last known position.
	// Since we don't track position per key, the caller must provide the
	// exact position that was used at insertion, or iterate over all cells.
	// For efficient removal, we use a simpler clear-rebuild approach.
	void remove(int32_t p_key, const Vector3 &p_position) {
		uint64_t cell = compute_cell_index(p_position);
		HashMap<uint64_t, LocalVector<int32_t>>::Iterator it = grid.find(cell);
		if (!it) return;
		LocalVector<int32_t> &vec = it->value;
		for (int i = 0; i < vec.size(); ++i) {
			if (vec[i] == p_key) {
				vec.remove_at_unordered(i);
				if (vec.is_empty()) {
					grid.remove(it);
				}
				return;
			}
		}
	}

	// Clear all entries.
	void clear() { grid.clear(); }

	// Return all keys that share the same cell as the given position,
	// including the queried position's own cell.
	// Optionally also include neighbouring cells (9 or 27).
	void query(const Vector3 &p_position, LocalVector<int32_t> &r_keys,
			   bool p_include_neighbors = true) const {
		uint64_t base_cell = compute_cell_index(p_position);
		// Always query own cell
		HashMap<uint64_t, LocalVector<int32_t>>::ConstIterator it = grid.find(base_cell);
		if (it) {
			r_keys.push_back(it->value.ptr(), it->value.size());
		}
		if (!p_include_neighbors) return;

		// Compute neighbour offsets (3x3x3 = 27, excluding center if already done)
		static const int offsets[3] = { -1, 0, 1 };
		Vector3 world_pos = cell_to_world(base_cell);
		for (int dx : offsets) {
			for (int dy : offsets) {
				for (int dz : offsets) {
					if (dx == 0 && dy == 0 && dz == 0) continue;
					Vector3 offset_world = world_pos + Vector3(dx, dy, dz) * cell_size;
					uint64_t neighbor_cell = compute_cell_index(offset_world);
					HashMap<uint64_t, LocalVector<int32_t>>::ConstIterator n_it = grid.find(neighbor_cell);
					if (n_it) {
						r_keys.push_back(n_it->value.ptr(), n_it->value.size());
					}
				}
			}
		}
	}

	// Return all cell indices that are occupied (for debug / iteration).
	void get_occupied_cells(LocalVector<uint64_t> &r_cells) const {
		for (const KeyValue<uint64_t, LocalVector<int32_t>> &kv : grid) {
			r_cells.push_back(kv.key);
		}
	}

	// Get number of occupied cells.
	int get_cell_count() const { return grid.size(); }

	// Cell size accessors.
	real_t get_cell_size() const { return cell_size; }
	void set_cell_size(real_t p_size) {
		ERR_FAIL_COND(p_size <= 0);
		if (p_size == cell_size) return;
		cell_size = p_size;
		inv_cell_size = 1.0 / p_size;
		// Changing cell size invalidates the hash; user must rebuild.
		grid.clear();
	}

private:
	// Convert a world position to an integer cell coordinate (key = Morton-like or simple hash).
	// We use a simple 3D integer packing into uint64_t: x (21 bits), y (21 bits), z (21 bits).
	// This allows grid up to ±2^20 cells in each axis.
	uint64_t compute_cell_index(const Vector3 &p_pos) const {
		int64_t cx = int64_t(Math::floor(p_pos.x * inv_cell_size));
		int64_t cy = int64_t(Math::floor(p_pos.y * inv_cell_size));
		int64_t cz = int64_t(Math::floor(p_pos.z * inv_cell_size));
		// Pack into uint64_t: each 21 bits (we allow negative using 2s complement, but for
		// hash map we'll use unsigned conversion).
		uint64_t ux = (uint64_t)(cx & 0x1FFFFF);
		uint64_t uy = (uint64_t)(cy & 0x1FFFFF);
		uint64_t uz = (uint64_t)(cz & 0x1FFFFF);
		return (ux) | (uy << 21) | (uz << 42);
	}

	// Convert cell index back to the world position of the cell center.
	Vector3 cell_to_world(uint64_t p_cell) const {
		uint64_t mask = 0x1FFFFF;
		int64_t cx = (int64_t)(p_cell & mask);
		int64_t cy = (int64_t)((p_cell >> 21) & mask);
		int64_t cz = (int64_t)((p_cell >> 42) & mask);
		// If the highest relevant bit is set, extend sign appropriately.
		if (cx & 0x100000) cx |= ~0x1FFFFF;
		if (cy & 0x100000) cy |= ~0x1FFFFF;
		if (cz & 0x100000) cz |= ~0x1FFFFF;
		return Vector3(real_t(cx) * cell_size + cell_size * 0.5,
					   real_t(cy) * cell_size + cell_size * 0.5,
					   real_t(cz) * cell_size + cell_size * 0.5);
	}

	real_t cell_size;
	real_t inv_cell_size;
	HashMap<uint64_t, LocalVector<int32_t>> grid;
};

} // namespace gaia::spatial

#endif // GAIA_SPATIAL_QUERY_SPATIAL_HASH_H