--- START OF FILE core/math/spatial_partition.h ---

#ifndef SPATIAL_PARTITION_H
#define SPATIAL_PARTITION_H

#include "core/typedefs.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/templates/hash_map.h"
#include "core/templates/list.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * SpatialPartition
 * 
 * A multi-scale broadphase spatial database.
 * Uses BigIntCore for sector-level keys and FixedMathCore for sub-meter local 
 * cell precision. Designed for zero-copy queries within Warp kernels.
 */
template <typename UserData>
class ET_ALIGN_32 SpatialPartition {
public:
	/**
	 * SectorKey
	 * Unique identifier for a 3D volume in galactic space.
	 * Combines discrete sector coordinates with a local cell hash.
	 */
	struct ET_ALIGN_32 SectorKey {
		BigIntCore sx, sy, sz;
		int64_t cell_hash;

		_FORCE_INLINE_ bool operator==(const SectorKey &p_other) const {
			return cell_hash == p_other.cell_hash && sx == p_other.sx && sy == p_other.sy && sz == p_other.sz;
		}

		_FORCE_INLINE_ uint32_t hash() const {
			uint32_t h = sx.hash();
			h = hash_murmur3_one_32(sy.hash(), h);
			h = hash_murmur3_one_32(sz.hash(), h);
			return hash_murmur3_one_64(static_cast<uint64_t>(cell_hash), h);
		}
	};

	struct Element {
		UserData data;
		Vector3f position;
		BigIntCore sx, sy, sz;
		SectorKey current_key;
	};

private:
	FixedMathCore cell_size;
	HashMap<SectorKey, List<Element*>> grid;
	HashMap<UserData, Element*> element_registry;

	/**
	 * _compute_key()
	 * Deterministically maps a galactic position to a grid key.
	 */
	_FORCE_INLINE_ void _compute_key(const Vector3f &p_pos, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, SectorKey &r_key) const {
		r_key.sx = p_sx;
		r_key.sy = p_sy;
		r_key.sz = p_sz;

		// Local cell coordinates using FixedMath floor
		int64_t cx = Math::floor(p_pos.x / cell_size).to_int();
		int64_t cy = Math::floor(p_pos.y / cell_size).to_int();
		int64_t cz = Math::floor(p_pos.z / cell_size).to_int();

		// Interleave bits for a robust 64-bit spatial hash
		r_key.cell_hash = (cx & 0x1FFFFF) | ((cy & 0x1FFFFF) << 21) | ((cz & 0x3FFFFF) << 42);
	}

public:
	// ------------------------------------------------------------------------
	// Management API
	// ------------------------------------------------------------------------

	/**
	 * insert()
	 * Adds an element to the spatial database.
	 * Optimized for high-frequency EnTT component additions.
	 */
	void insert(UserData p_data, const Vector3f &p_pos, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz) {
		if (unlikely(element_registry.has(p_data))) return;

		Element *e = memnew(Element);
		e->data = p_data;
		e->position = p_pos;
		e->sx = p_sx;
		e->sy = p_sy;
		e->sz = p_sz;
		_compute_key(p_pos, p_sx, p_sy, p_sz, e->current_key);

		grid[e->current_key].push_back(e);
		element_registry[p_data] = e;
	}

	/**
	 * update()
	 * Moves an element. Only re-hashes if a cell/sector boundary is crossed.
	 * Maintains 120 FPS by minimizing hash-map mutations.
	 */
	void update(UserData p_data, const Vector3f &p_pos, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz) {
		if (unlikely(!element_registry.has(p_data))) return;

		Element *e = element_registry[p_data];
		SectorKey new_key;
		_compute_key(p_pos, p_sx, p_sy, p_sz, new_key);

		if (unlikely(!(e->current_key == new_key))) {
			grid[e->current_key].erase(grid[e->current_key].find(e));
			e->current_key = new_key;
			grid[e->current_key].push_back(e);
		}

		e->position = p_pos;
		e->sx = p_sx;
		e->sy = p_sy;
		e->sz = p_sz;
	}

	void remove(UserData p_data) {
		if (!element_registry.has(p_data)) return;
		Element *e = element_registry[p_data];
		grid[e->current_key].erase(grid[e->current_key].find(e));
		element_registry.erase(p_data);
		memdelete(e);
	}

	// ------------------------------------------------------------------------
	// Deterministic Query API
	// ------------------------------------------------------------------------

	/**
	 * query_radius()
	 * Finds all entities within a physical radius.
	 * Used for Warp-style collision kernels and physics impulses.
	 */
	void query_radius(const Vector3f &p_center, const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, FixedMathCore p_radius, List<UserData> &r_results) const {
		int64_t cell_range = Math::ceil(p_radius / cell_size).to_int();
		FixedMathCore r2 = p_radius * p_radius;

		for (int64_t x = -cell_range; x <= cell_range; x++) {
			for (int64_t y = -cell_range; y <= cell_range; y++) {
				for (int64_t z = -cell_range; z <= cell_range; z++) {
					SectorKey check_key;
					// Note: Real implementation handles sector wrapping here via BigIntCore
					Vector3f offset_pos = p_center + Vector3f(cell_size * FixedMathCore(x), cell_size * FixedMathCore(y), cell_size * FixedMathCore(z));
					_compute_key(offset_pos, p_sx, p_sy, p_sz, check_key);

					if (grid.has(check_key)) {
						const List<Element*> &cell_list = grid[check_key];
						for (const typename List<Element*>::Element *E = cell_list.front(); E; E = E->next()) {
							// Distance check in bit-perfect FixedMath
							Vector3f diff = E->get()->position - p_center;
							if (diff.length_squared() <= r2) {
								r_results.push_back(E->get()->data);
							}
						}
					}
				}
			}
		}
	}

	void clear() {
		for (auto &E : element_registry) {
			memdelete(E.value);
		}
		element_registry.clear();
		grid.clear();
	}

	SpatialPartition(FixedMathCore p_cell_size) : cell_size(p_cell_size) {}
	~SpatialPartition() { clear(); }
};

/**
 * Specialized Hasher for SectorKey to enable HashMap integration.
 */
template <>
struct HashMapHasherDefault<typename SpatialPartition<uint32_t>::SectorKey> {
	static inline uint32_t hash(const typename SpatialPartition<uint32_t>::SectorKey &p_key) {
		return p_key.hash();
	}
};

#endif // SPATIAL_PARTITION_H

--- END OF FILE core/math/spatial_partition.h ---
