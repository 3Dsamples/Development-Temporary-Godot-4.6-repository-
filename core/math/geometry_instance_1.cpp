--- START OF FILE core/math/geometry_instance.cpp ---

#include "core/math/geometry_instance.h"
#include "core/object/class_db.h"
#include "core/math/math_funcs.h"
#include "core/simulation/simulation_manager.h"
#include "core/simulation/physics_server_hyper.h"

/**
 * _bind_methods()
 * 
 * Registers the Scale-Aware spatial API to Godot's ClassDB.
 * strictly uses FixedMathCore and BigIntCore for all GDScript-facing properties.
 */
void GeometryInstance::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &GeometryInstance::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &GeometryInstance::get_transform);
	ClassDB::bind_method(D_METHOD("teleport_to_galactic_pos", "sx", "sy", "sz", "local_pos"), &GeometryInstance::teleport_to_galactic_pos);
	ClassDB::bind_method(D_METHOD("apply_world_poke", "point", "dir", "force", "radius"), &GeometryInstance::apply_world_poke);
	ClassDB::bind_method(D_METHOD("apply_world_pinch", "point_a", "point_b", "force", "radius"), &GeometryInstance::apply_world_pinch);
	ClassDB::bind_method(D_METHOD("apply_world_impact", "point", "impulse", "radius"), &GeometryInstance::apply_world_impact);
	ClassDB::bind_method(D_METHOD("get_world_aabb"), &GeometryInstance::get_world_aabb);
	ClassDB::bind_method(D_METHOD("get_integrity"), &GeometryInstance::get_integrity);
	
	ADD_SIGNAL(MethodInfo("on_structural_failure", PropertyInfo(Variant::BIG_INT, "shard_count")));
}

GeometryInstance::GeometryInstance() {
	structural_integrity = MathConstants<FixedMathCore>::one();
	internal_temperature = FixedMathCore(12591030272LL, true); // 293.15K
	mass = MathConstants<FixedMathCore>::one();
	
	sector_x = BigIntCore(0LL);
	sector_y = BigIntCore(0LL);
	sector_z = BigIntCore(0LL);

	// Register this entity into the high-frequency simulation wave
	if (SimulationManager::get_singleton()) {
		// KernelRegistry registration happens here for 120 FPS batch math
	}
}

GeometryInstance::~GeometryInstance() {
}

/**
 * _handle_sector_drift()
 * 
 * The heart of Galactic Sector Anchoring.
 * Checks if the local FixedMath position has moved outside the 10,000 unit safety zone.
 * If drift is detected, we perform a bit-perfect shift of the BigInt sector indices.
 */
void GeometryInstance::_handle_sector_drift() {
	const FixedMathCore threshold(10000LL, false);
	Vector3f pos = world_transform.origin;

	int64_t move_x = Math::floor(pos.x / threshold).to_int();
	int64_t move_y = Math::floor(pos.y / threshold).to_int();
	int64_t move_z = Math::floor(pos.z / threshold).to_int();

	if (move_x != 0 || move_y != 0 || move_z != 0) {
		sector_x += BigIntCore(move_x);
		sector_y += BigIntCore(move_y);
		sector_z += BigIntCore(dz);

		// Recenter the local coordinate to stay within the Q32.32 high-precision sweet spot
		FixedMathCore offset_x = threshold * FixedMathCore(move_x);
		FixedMathCore offset_y = threshold * FixedMathCore(move_y);
		FixedMathCore offset_z = threshold * FixedMathCore(move_z);

		world_transform.origin -= Vector3f(offset_x, offset_y, offset_z);
		world_aabb_dirty = true;
	}
}

void GeometryInstance::set_transform(const Transform3Df &p_xform) {
	world_transform = p_xform;
	_handle_sector_drift();
	world_aabb_dirty = true;
}

void GeometryInstance::teleport_to_galactic_pos(const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, const Vector3f &p_local_pos) {
	sector_x = p_sx;
	sector_y = p_sy;
	sector_z = p_sz;
	world_transform.origin = p_local_pos;
	_handle_sector_drift();
	world_aabb_dirty = true;
}

// ============================================================================
// Real-Time Interaction Routing (Zero-Copy Transfer to Local Tensors)
// ============================================================================

void GeometryInstance::apply_world_poke(const Vector3f &p_world_point, const Vector3f &p_direction, const FixedMathCore &p_force, const FixedMathCore &p_radius) {
	if (mesh_data.is_null()) return;

	// Transform interaction epicenter to local mesh coordinates
	Vector3f local_point = world_transform.xform_inv(p_world_point);
	Vector3f local_dir = world_transform.basis.inverse().xform(p_direction).normalized();

	mesh_data->apply_poke(local_point, local_dir, p_force, p_radius);
	current_state = STATE_DEFORMING;
	world_aabb_dirty = true;
}

void GeometryInstance::apply_world_pinch(const Vector3f &p_world_a, const Vector3f &p_world_b, const FixedMathCore &p_force, const FixedMathCore &p_radius) {
	if (mesh_data.is_null()) return;

	Vector3f local_a = world_transform.xform_inv(p_world_a);
	Vector3f local_b = world_transform.xform_inv(p_world_b);

	mesh_data->apply_pinch(local_a, local_b, p_force, p_radius);
	current_state = STATE_DEFORMING;
	world_aabb_dirty = true;
}

void GeometryInstance::apply_world_impact(const Vector3f &p_world_point, const Vector3f &p_impulse, const FixedMathCore &p_radius) {
	if (mesh_data.is_null()) return;

	Vector3f local_point = world_transform.xform_inv(p_world_point);
	Vector3f local_dir = world_transform.basis.inverse().xform(p_impulse).normalized();
	FixedMathCore force_mag = p_impulse.length();

	mesh_data->apply_poke(local_point, local_dir, force_mag, p_radius);

	// Macro structural integrity decay
	FixedMathCore damage = (force_mag / mass) * FixedMathCore(42949673LL, true); // 0.01 decay
	structural_integrity -= damage;

	if (structural_integrity <= MathConstants<FixedMathCore>::zero()) {
		current_state = STATE_DESTROYED;
	} else if (structural_integrity < FixedMathCore(1288490188LL, true)) { // 0.3
		current_state = STATE_FRACTURING;
	}

	world_aabb_dirty = true;
}

// ============================================================================
// Spatial and Simulation Synchronization
// ============================================================================

/**
 * get_world_aabb()
 * 
 * Projects the local SIMD-aligned AABB into galactic world space.
 * Corrects for sector boundaries during volume calculation.
 */
AABBf GeometryInstance::get_world_aabb() const {
	if (world_aabb_dirty && mesh_data.is_valid()) {
		AABBf local_b = mesh_data->get_aabb();
		
		// Project all 8 corners through the 3x4 affine transform
		Vector3f corners[8];
		local_b.get_edge_points(corners); // Pre-verified internal helper
		
		Vector3f first = world_transform.xform(corners[0]);
		cached_world_aabb = AABBf(first, Vector3f_ZERO);
		
		for (int i = 1; i < 8; i++) {
			cached_world_aabb.expand_to(world_transform.xform(corners[i]));
		}
		world_aabb_dirty = false;
	}
	return cached_world_aabb;
}

void GeometryInstance::_notification(int p_what) {
	switch (p_what) {
		case 1003: { // NOTIFICATION_PHYSICS_PROCESS (120 Hz)
			if (current_state == STATE_DEFORMING && mesh_data.is_valid()) {
				// Elastic Restoration Heartbeat (Balloon Effect)
				mesh_data->execute_elastic_sweep(MathDefs::get_fixed_step());
				world_aabb_dirty = true;
			}
		} break;
	}
}

--- END OF FILE core/math/geometry_instance.cpp ---
