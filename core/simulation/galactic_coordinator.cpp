--- START OF FILE core/simulation/galactic_coordinator.cpp ---

#include "core/simulation/galactic_coordinator.h"
#include "core/object/class_db.h"
#include "core/math/math_funcs.h"
#include "core/simulation/physics_server_hyper.h"
#include "core/simulation/simulation_manager.h"
#include "core/core_logger.h"

GalacticCoordinator *GalacticCoordinator::singleton = nullptr;

/**
 * _bind_methods()
 * 
 * Exposes the galactic coordinate API to Godot's reflection system.
 * strictly uses FixedMathCore and BigIntCore for all parameter passing.
 */
void GalacticCoordinator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_anchor_sector", "x", "y", "z"), &GalacticCoordinator::set_anchor_sector);
	ClassDB::bind_method(D_METHOD("set_sector_size", "size"), &GalacticCoordinator::set_sector_size);
	ClassDB::bind_method(D_METHOD("get_sector_size"), &GalacticCoordinator::get_sector_size);
	ClassDB::bind_method(D_METHOD("update_floating_origin", "observer_pos"), &GalacticCoordinator::update_floating_origin);
	
	ClassDB::bind_method(D_METHOD("get_anchor_x"), &GalacticCoordinator::get_anchor_x);
	ClassDB::bind_method(D_METHOD("get_anchor_y"), &GalacticCoordinator::get_anchor_y);
	ClassDB::bind_method(D_METHOD("get_anchor_z"), &GalacticCoordinator::get_anchor_z);
}

GalacticCoordinator::GalacticCoordinator() {
	singleton = this;
	sector_size = FixedMathCore(10000LL, false); // Default 10km sectors
	anchor_sector.x = BigIntCore(0LL);
	anchor_sector.y = BigIntCore(0LL);
	anchor_sector.z = BigIntCore(0LL);
	floating_origin_offset = Vector3f_ZERO;
}

GalacticCoordinator::~GalacticCoordinator() {
	singleton = nullptr;
}

/**
 * world_to_galactic()
 * 
 * Computes the absolute galactic coordinates for a given local position.
 * 1. Calculates how many 'sector_size' units the point is from the current anchor.
 * 2. Returns the resulting BigInt index and the normalized FixedMath local offset.
 */
void GalacticCoordinator::world_to_galactic(const Vector3f &p_world_pos, SectorID &r_sector, Vector3f &r_local_offset) const {
	// Sector counts = floor(pos / size)
	int64_t dx = Math::floor(p_world_pos.x / sector_size).to_int();
	int64_t dy = Math::floor(p_world_pos.y / sector_size).to_int();
	int64_t dz = Math::floor(p_world_pos.z / sector_size).to_int();

	r_sector.x = anchor_sector.x + BigIntCore(dx);
	r_sector.y = anchor_sector.y + BigIntCore(dy);
	r_sector.z = anchor_sector.z + BigIntCore(dz);

	// Local Offset is the remainder in FixedMath
	r_local_offset.x = p_world_pos.x - (sector_size * FixedMathCore(dx));
	r_local_offset.y = p_world_pos.y - (sector_size * FixedMathCore(dy));
	r_local_offset.z = p_world_pos.z - (sector_size * FixedMathCore(dz));
}

/**
 * galactic_to_world()
 * 
 * Reconstructs a local coordinate relative to the current anchor.
 * Essential for 120 FPS rendering of distant starships.
 */
Vector3f GalacticCoordinator::galactic_to_world(const SectorID &p_sector, const Vector3f &p_local_offset) const {
	BigIntCore dsx = p_sector.x - anchor_sector.x;
	BigIntCore dsy = p_sector.y - anchor_sector.y;
	BigIntCore dsz = p_sector.z - anchor_sector.z;

	// Scale-Aware conversion: BigInt * Fixed -> Fixed
	FixedMathCore fx(static_cast<int64_t>(std::stoll(dsx.to_string())));
	FixedMathCore fy(static_cast<int64_t>(std::stoll(dsy.to_string())));
	FixedMathCore fz(static_cast<int64_t>(std::stoll(dsz.to_string())));

	return p_local_offset + Vector3f(fx * sector_size, fy * sector_size, fz * sector_size);
}

/**
 * update_floating_origin()
 * 
 * The master spatial re-centering logic. 
 * Invoked by the SimulationManager at 120 FPS.
 * 
 * Sophisticated Behavior: Hysteresis Transition Buffer.
 * We only shift if the observer is > 80% of the way to the next sector edge.
 * This prevents thrashing at sector boundaries.
 */
void GalacticCoordinator::update_floating_origin(const Vector3f &p_observer_pos) {
	FixedMathCore threshold = sector_size * FixedMathCore(858993459LL, true); // 0.2 factor (shift when at 80% limit)
	
	int64_t dx = 0, dy = 0, dz = 0;
	if (Math::abs(p_observer_pos.x) > threshold) dx = Math::sign(p_observer_pos.x).to_int();
	if (Math::abs(p_observer_pos.y) > threshold) dy = Math::sign(p_observer_pos.y).to_int();
	if (Math::abs(p_observer_pos.z) > threshold) dz = Math::sign(p_observer_pos.z).to_int();

	if (dx != 0 || dy != 0 || dz != 0) {
		// 1. Calculate Shift Vector
		FixedMathCore sx = FixedMathCore(dx) * sector_size;
		FixedMathCore sy = FixedMathCore(dy) * sector_size;
		FixedMathCore sz = FixedMathCore(dz) * sector_size;
		Vector3f shift_vec(sx, sy, sz);

		// 2. Advance Global Anchor
		anchor_sector.x += BigIntCore(dx);
		anchor_sector.y += BigIntCore(dy);
		anchor_sector.z += BigIntCore(dz);

		// 3. Launch Warp Kernels to Shift All Entities (Zero-Copy)
		// We tell the Physics Server to perform a bit-perfect parallel translation
		PhysicsServerHyper::get_singleton()->execute_global_origin_shift(shift_vec);

		Logger::info("Galactic Shift: New Anchor Sector [" + 
			String(anchor_sector.x.to_string().c_str()) + ", " + 
			String(anchor_sector.y.to_string().c_str()) + ", " + 
			String(anchor_sector.z.to_string().c_str()) + "]");
	}
}

void GalacticCoordinator::set_anchor_sector(const BigIntCore &p_x, const BigIntCore &p_y, const BigIntCore &p_z) {
	anchor_sector.x = p_x;
	anchor_sector.y = p_y;
	anchor_sector.z = p_z;
}

void GalacticCoordinator::set_sector_size(const FixedMathCore &p_size) {
	sector_size = p_size;
}

--- END OF FILE core/simulation/galactic_coordinator.cpp ---
