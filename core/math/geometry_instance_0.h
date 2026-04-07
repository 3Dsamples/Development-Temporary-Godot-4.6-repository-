--- START OF FILE core/math/geometry_instance.h ---

#ifndef GEOMETRY_INSTANCE_H
#define GEOMETRY_INSTANCE_H

#include "core/object/object.h"
#include "core/math/dynamic_mesh.h"
#include "core/math/transform_3d.h"
#include "core/simulation/simulation_manager.h"
#include "core/templates/rid.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * GeometryInstance
 * 
 * The master orchestrator for physical entities in the scene.
 * Bridges spatial positioning with high-fidelity geometric simulation.
 * Optimized for Zero-Copy data flow between SceneTree and Warp kernels.
 */
class ET_ALIGN_32 GeometryInstance : public Object {
	GDCLASS(GeometryInstance, Object);

public:
	enum SimulationState {
		STATE_STABLE,      // Nominal integrity
		STATE_DEFORMING,   // Active vertex displacement
		STATE_FRACTURING,  // Critical fatigue, nearing destruction
		STATE_DESTROYED    // Converted to independent physics shards
	};

private:
	// The simulated geometric resource
	Ref<DynamicMesh> mesh_data;
	
	// Spatial State (Sector-aware)
	Transform3Df world_transform;
	BigIntCore sector_x, sector_y, sector_z;

	// Physical Parameters
	FixedMathCore structural_integrity;
	FixedMathCore thermal_state;
	FixedMathCore mass;

	SimulationState current_state = STATE_STABLE;
	SimulationManager::SimulationTier active_tier = SimulationManager::TIER_DETERMINISTIC;

	// ETEngine: Bounding volume cache for 120 FPS culling
	mutable AABBf cached_world_aabb;
	mutable bool world_aabb_dirty = true;

	void _handle_sector_drift();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	// ------------------------------------------------------------------------
	// Spatial API (Galactic Scale)
	// ------------------------------------------------------------------------

	void set_transform(const Transform3Df &p_xform);
	Transform3Df get_transform() const;

	/**
	 * teleport_to_galactic_pos()
	 * Directly sets the BigIntCore sectors and local FixedMath offset.
	 */
	void teleport_to_galactic_pos(const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, const Vector3f &p_local_pos);

	void get_galactic_sector(BigIntCore &r_x, BigIntCore &r_y, BigIntCore &r_z) const;

	// ------------------------------------------------------------------------
	// Hyper-Simulation Action Routing
	// ------------------------------------------------------------------------

	/**
	 * apply_world_impact()
	 * Converts a world-space collision into local mesh deformation tensors.
	 */
	void apply_world_impact(const Vector3f &p_world_point, const Vector3f &p_direction, const FixedMathCore &p_force, const FixedMathCore &p_radius);

	/**
	 * apply_world_bend()
	 * Routes a global bending force into the local vertex array.
	 */
	void apply_world_bend(const Vector3f &p_world_pivot, const Vector3f &p_world_axis, const FixedMathCore &p_angle);

	// ------------------------------------------------------------------------
	// Simulation Lifecycle
	// ------------------------------------------------------------------------

	void set_mesh(const Ref<DynamicMesh> &p_mesh);
	Ref<DynamicMesh> get_mesh() const;

	AABBf get_world_aabb() const;

	_FORCE_INLINE_ FixedMathCore get_integrity() const { return structural_integrity; }
	_FORCE_INLINE_ SimulationState get_state() const { return current_state; }

	/**
	 * update_simulation_lod()
	 * Adjusts mesh density and task priority based on importance.
	 */
	void update_simulation_lod(const Vector3f &p_observer_pos);

	GeometryInstance();
	virtual ~GeometryInstance();
};

#endif // GEOMETRY_INSTANCE_H

--- END OF FILE core/math/geometry_instance.h ---
