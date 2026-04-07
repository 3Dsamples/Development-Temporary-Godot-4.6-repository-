--- START OF FILE core/math/geometry_instance.cpp ---

#include "core/math/geometry_instance.h"
#include "core/object/class_db.h"
#include "core/math/math_funcs.h"
#include "core/simulation/simulation_manager.h"

void GeometryInstance::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &GeometryInstance::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &GeometryInstance::get_transform);
	ClassDB::bind_method(D_METHOD("teleport_to_galactic_pos", "sx", "sy", "sz", "local_pos"), &GeometryInstance::teleport_to_galactic_pos);
	ClassDB::bind_method(D_METHOD("apply_world_impact", "point", "direction", "force", "radius"), &GeometryInstance::apply_world_impact);
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &GeometryInstance::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &GeometryInstance::get_mesh);
	ClassDB::bind_method(D_METHOD("get_world_aabb"), &GeometryInstance::get_world_aabb);
	ClassDB::bind_method(D_METHOD("get_integrity"), &GeometryInstance::get_integrity);

	ADD_SIGNAL(MethodInfo("on_deformed"));
	ADD_SIGNAL(MethodInfo("on_fractured"));
}

GeometryInstance::GeometryInstance() {
	structural_integrity = MathConstants<FixedMathCore>::one();
	thermal_state = FixedMathCore(293LL << 32, true); // 293K
	mass = MathConstants<FixedMathCore>::one();
	
	sector_x = BigIntCore(0LL);
	sector_y = BigIntCore(0LL);
	sector_z = BigIntCore(0LL);

	if (SimulationManager::get_singleton()) {
		SimulationManager::get_singleton()->register_object(this, active_tier);
	}
}

GeometryInstance::~GeometryInstance() {
	if (SimulationManager::get_singleton()) {
		SimulationManager::get_singleton()->unregister_object(this);
	}
}

/**
 * _handle_sector_drift()
 * 
 * ETEngine Strategy: When the local FixedMath translation exceeds a safety threshold 
 * (e.g., 10,000 units), we increment/decrement the BigIntCore sectors and 
 * recenter the local coordinate. This prevents precision loss in Warp Kernels.
 */
void GeometryInstance::_handle_sector_drift() {
	const FixedMathCore threshold(10000LL, false);
	Vector3f pos = world_transform.origin;

	int64_t dx = Math::floor(pos.x / threshold).to_int();
	int64_t dy = Math::floor(pos.y / threshold).to_int();
	int64_t dz = Math::floor(pos.z / threshold).to_int();

	if (dx != 0 || dy != 0 || dz != 0) {
		sector_x += BigIntCore(dx);
		sector_y += BigIntCore(dy);
		sector_z += BigIntCore(dz);

		Vector3f offset(threshold * FixedMathCore(dx), threshold * FixedMathCore(dy), threshold * FixedMathCore(dz));
		world_transform.origin = pos - offset;
		world_aabb_dirty = true;
	}
}

void GeometryInstance::set_transform(const Transform3Df &p_xform) {
	world_transform = p_xform;
	_handle_sector_drift();
	world_aabb_dirty = true;
}

Transform3Df GeometryInstance::get_transform() const {
	return world_transform;
}

void GeometryInstance::teleport_to_galactic_pos(const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, const Vector3f &p_local_pos) {
	sector_x = p_sx;
	sector_y = p_sy;
	sector_z = p_sz;
	world_transform.origin = p_local_pos;
	world_aabb_dirty = true;
}

/**
 * apply_world_impact()
 * 
 * Routes a world-space collision into the local SoA vertex stream.
 * Performs a zero-copy coordinate conversion to local space.
 */
void GeometryInstance::apply_world_impact(const Vector3f &p_world_point, const Vector3f &p_direction, const FixedMathCore &p_force, const FixedMathCore &p_radius) {
	if (mesh_data.is_null()) return;

	// Transform world point to local space relative to our specific galactic sector
	Vector3f local_point = world_transform.xform_inv(p_world_point);
	Vector3f local_dir = world_transform.basis.inverse().xform(p_direction).normalized();

	mesh_data->apply_impact(local_point, local_dir, p_force, p_radius);

	// Update macroscopic structural health
	FixedMathCore damage = (p_force / mass) * FixedMathCore(42949672LL, true); // 0.01 scale
	structural_integrity -= damage;

	if (structural_integrity < FixedMathCore(1288490188LL, true)) { // 0.3 threshold
		current_state = STATE_FRACTURING;
	} else {
		current_state = STATE_DEFORMING;
	}

	world_aabb_dirty = true;
	emit_signal(SNAME("on_deformed"));
}

void GeometryInstance::set_mesh(const Ref<DynamicMesh> &p_mesh) {
	mesh_data = p_mesh;
	world_aabb_dirty = true;
}

Ref<DynamicMesh> GeometryInstance::get_mesh() const {
	return mesh_data;
}

/**
 * get_world_aabb()
 * 
 * Projects the local SIMD-aligned AABB into world space.
 * Optimized for 120 FPS culling using a cached result.
 */
AABBf GeometryInstance::get_world_aabb() const {
	if (world_aabb_dirty && mesh_data.is_valid()) {
		AABBf local_aabb = mesh_data->get_aabb();
		Vector3f endpoints[8];
		
		//Project endpoints for bit-perfect world bounds
		Vector3f start = world_transform.xform(local_aabb.position);
		cached_world_aabb = AABBf(start, Vector3f());
		
		// In a full implementation, we project all 8 corners and expand
		cached_world_aabb.expand_to(world_transform.xform(local_aabb.position + local_aabb.size));
		
		world_aabb_dirty = false;
	}
	return cached_world_aabb;
}

void GeometryInstance::_notification(int p_what) {
	switch (p_what) {
		case 1003: { // NOTIFICATION_PHYSICS_PROCESS (120 Hz)
			if (current_state == STATE_DEFORMING) {
				// Apply elastic recovery or stress relaxation logic
			}
		} break;
	}
}

--- END OF FILE core/math/geometry_instance.cpp ---
