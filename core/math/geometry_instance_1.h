--- START OF FILE core/math/geometry_instance.h ---

#ifndef GEOMETRY_INSTANCE_H
#define GEOMETRY_INSTANCE_H

#include "core/object/object.h"
#include "core/math/dynamic_mesh.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * GeometryInstance
 * 
 * The master orchestrator for physical entities in the SceneTree.
 * Replaces standard nodes with a Scale-Aware, bit-perfect spatial handle.
 */
class ET_ALIGN_32 GeometryInstance : public Object {
	GDCLASS(GeometryInstance, Object);

public:
	enum SimulationState {
		STATE_STABLE,      // Nominal integrity
		STATE_DEFORMING,   // Active elastic/plastic displacement
		STATE_FRACTURING,  // Critical fatigue detected
		STATE_DESTROYED    // Converted to independent physics shards
	};

private:
	// The simulated geometric resource (Balloon/Flesh/Rigid)
	Ref<DynamicMesh> mesh_data;
	
	// Spatial State (Sector-aware for Galactic Scale)
	Transform3Df world_transform;
	BigIntCore sector_x, sector_y, sector_z;

	// Physical Interaction Tensors
	FixedMathCore structural_integrity;
	FixedMathCore internal_temperature;
	FixedMathCore mass;

	SimulationState current_state = STATE_STABLE;
	
	// ETEngine: Bounding volume cache for 120 FPS culling
	mutable AABBf cached_world_aabb;
	mutable bool world_aabb_dirty = true;

	/**
	 * _handle_sector_drift()
	 * Automatically re-anchors the BigInt sectors if FixedMath precision is threatened.
	 */
	void _handle_sector_drift();

protected:
	static void _bind_methods();
	virtual void _notification(int p_what);

public:
	// ------------------------------------------------------------------------
	// Spatial API (Relativistic & Galactic)
	// ------------------------------------------------------------------------

	void set_transform(const Transform3Df &p_xform);
	_FORCE_INLINE_ Transform3Df get_transform() const { return world_transform; }

	/**
	 * teleport_to_galactic_pos()
	 * Directly anchors an object to a BigIntCore coordinate system.
	 */
	void teleport_to_galactic_pos(const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, const Vector3f &p_local_pos);

	_FORCE_INLINE_ void get_galactic_sector(BigIntCore &r_x, BigIntCore &r_y, BigIntCore &r_z) const {
		r_x = sector_x; r_y = sector_y; r_z = sector_z;
	}

	// ------------------------------------------------------------------------
	// Real-Time Interaction Routing (Balloon & Flesh)
	// ------------------------------------------------------------------------

	/**
	 * apply_world_poke()
	 * Routes a world-space interaction into local mesh vertex tensors.
	 */
	void apply_world_poke(const Vector3f &p_world_point, const Vector3f &p_direction, const FixedMathCore &p_force, const FixedMathCore &p_radius);

	/**
	 * apply_world_pinch()
	 * Sophisticated Behavior: Multi-point compression for flesh (Breast/Buttock logic).
	 */
	void apply_world_pinch(const Vector3f &p_world_a, const Vector3f &p_world_b, const FixedMathCore &p_force, const FixedMathCore &p_radius);

	/**
	 * apply_world_impact()
	 * High-velocity collision resolution for spaceships and projectiles.
	 */
	void apply_world_impact(const Vector3f &p_world_point, const Vector3f &p_impulse, const FixedMathCore &p_radius);

	// ------------------------------------------------------------------------
	// Simulation Lifecycle
	// ------------------------------------------------------------------------

	void set_mesh(const Ref<DynamicMesh> &p_mesh);
	_FORCE_INLINE_ Ref<DynamicMesh> get_mesh() const { return mesh_data; }

	AABBf get_world_aabb() const;

	_FORCE_INLINE_ FixedMathCore get_integrity() const { return structural_integrity; }
	_FORCE_INLINE_ SimulationState get_state() const { return current_state; }

	/**
	 * update_simulation_lod()
	 * Importance-based LOD management for 120 FPS throughput.
	 */
	void update_simulation_lod(const Vector3f &p_observer_pos);

	GeometryInstance();
	virtual ~GeometryInstance();
};

#endif // GEOMETRY_INSTANCE_H

--- END OF FILE core/math/geometry_instance.h ---
