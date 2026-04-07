--- START OF FILE core/simulation/galactic_coordinator.h ---

#ifndef GALACTIC_COORDINATOR_H
#define GALACTIC_COORDINATOR_H

#include "core/object/object.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * GalacticCoordinator
 * 
 * The central authority for high-precision spatial re-centering and sector management.
 * 1. Tracks the "Global Observer" sector (usually the player or camera).
 * 2. Manages sector-based paging for EnTT components.
 * 3. Provides bit-perfect coordinate translation between simulation tiers.
 * 
 * Aligned to 32 bytes for SIMD-accelerated coordinate mapping.
 */
class ET_ALIGN_32 GalacticCoordinator : public Object {
	GDCLASS(GalacticCoordinator, Object);

	static GalacticCoordinator *singleton;

public:
	/**
	 * SectorID
	 * Deterministic 3D grid coordinate for a massive galactic volume.
	 */
	struct ET_ALIGN_32 SectorID {
		BigIntCore x, y, z;

		_FORCE_INLINE_ bool operator==(const SectorID &p_other) const {
			return x == p_other.x && y == p_other.y && z == p_other.z;
		}

		_FORCE_INLINE_ uint32_t hash() const {
			uint32_t h = x.hash();
			h = hash_murmur3_one_32(y.hash(), h);
			h = hash_murmur3_one_32(z.hash(), h);
			return h;
		}
	};

private:
	// The currently centered sector in world space (0,0,0)
	SectorID anchor_sector;
	
	// Size of a single sector in FixedMath units (e.g., 10,000.0)
	FixedMathCore sector_size;
	
	// Active sector management
	HashMap<SectorID, bool> active_sectors;

	// Floating Origin offset for visual rendering sync
	Vector3f floating_origin_offset;

protected:
	static void _bind_methods();

public:
	static _FORCE_INLINE_ GalacticCoordinator *get_singleton() { return singleton; }

	// ------------------------------------------------------------------------
	// Galactic Coordinate API
	// ------------------------------------------------------------------------

	/**
	 * world_to_galactic()
	 * Converts a local FixedMath position to an absolute BigInt+Fixed pair.
	 */
	void world_to_galactic(const Vector3f &p_world_pos, SectorID &r_sector, Vector3f &r_local_offset) const;

	/**
	 * galactic_to_world()
	 * Converts absolute coordinates to local FixedMath relative to the current anchor.
	 * Critical for 120 FPS jitter-free rendering of distant objects.
	 */
	Vector3f galactic_to_world(const SectorID &p_sector, const Vector3f &p_local_offset) const;

	// ------------------------------------------------------------------------
	// Origin Management (120 FPS Heartbeat Hooks)
	// ------------------------------------------------------------------------

	/**
	 * update_floating_origin()
	 * Checks if the primary observer has moved beyond the sector threshold.
	 * If so, it triggers a "Simulation Wave" to shift all entity positions.
	 */
	void update_floating_origin(const Vector3f &p_observer_pos);

	/**
	 * set_anchor_sector()
	 * Force-recenters the universe on a specific BigInt coordinate.
	 * used for long-range teleportation or initial spawns.
	 */
	void set_anchor_sector(const BigIntCore &p_x, const BigIntCore &p_y, const BigIntCore &p_z);

	// ------------------------------------------------------------------------
	// Data Accessors
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ FixedMathCore get_sector_size() const { return sector_size; }
	_FORCE_INLINE_ SectorID get_anchor_sector() const { return anchor_sector; }
	
	void set_sector_size(const FixedMathCore &p_size);

	GalacticCoordinator();
	~GalacticCoordinator();
};

/**
 * Custom Hasher for SectorID to enable high-speed HashMap lookup.
 */
template <>
struct HashMapHasherDefault<GalacticCoordinator::SectorID> {
	static _FORCE_INLINE_ uint32_t hash(const GalacticCoordinator::SectorID &p_id) {
		return p_id.hash();
	}
};

#endif // GALACTIC_COORDINATOR_H

--- END OF FILE core/simulation/galactic_coordinator.h ---
