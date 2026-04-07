--- START OF FILE core/math/dynamic_mesh.h ---

#ifndef DYNAMIC_MESH_H
#define DYNAMIC_MESH_H

#include "core/object/ref_counted.h"
#include "core/templates/vector.h"
#include "core/math/face3.h"
#include "core/math/aabb.h"
#include "core/math/spatial_partition.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * DynamicMesh
 * 
 * A high-fidelity physical geometry resource.
 * Stores vertex data in SIMD-aligned SoA streams for Warp kernel sweeps.
 * Manages material tensors for deterministic deformation and destruction.
 */
class ET_ALIGN_32 DynamicMesh : public RefCounted {
	GDCLASS(DynamicMesh, RefCounted);

public:
	// --- Internal Simulation Tensors ---
	struct ET_ALIGN_32 VertexState {
		Vector3f position;
		Vector3f velocity;
		Vector3f normal;
		FixedMathCore fatigue;      // Structural integrity [0..1]
		FixedMathCore temperature;  // Thermal state for buckling
		FixedMathCore mass;
	};

private:
	// SoA Storage for EnTT-style batch processing
	Vector<VertexState> vertices;
	Vector<int> indices;
	Vector<Face3f> face_tensors;

	// Broadphase for localized deformation lookups
	SpatialPartition<uint32_t> local_partition;

	// Material Constants (Deterministic)
	FixedMathCore yield_strength;
	FixedMathCore elasticity;
	FixedMathCore thermal_conductivity;

	// Galactic Sector Anchoring
	BigIntCore sector_x, sector_y, sector_z;

	mutable AABBf cached_aabb;
	mutable bool aabb_dirty = true;

	void _rebuild_spatial_index();
	void _update_aabb() const;

protected:
	static void _bind_methods();

public:
	// ------------------------------------------------------------------------
	// Geometric Manipulation API (Warp-Kernel Ready)
	// ------------------------------------------------------------------------

	/**
	 * apply_impact()
	 * Simulates plastic/elastic cratering.
	 * Displaces vertices and accumulates fatigue deterministically.
	 */
	void apply_impact(const Vector3f &p_local_point, const Vector3f &p_direction, const FixedMathCore &p_force, const FixedMathCore &p_radius);

	/**
	 * apply_torsional_screw()
	 * Twists the mesh geometry around a torque axis.
	 */
	void apply_torsional_screw(const Vector3f &p_axis_origin, const Vector3f &p_axis_dir, const FixedMathCore &p_torque, const FixedMathCore &p_radius);

	/**
	 * apply_structural_bend()
	 * Fold geometry along a deterministic hinge.
	 */
	void apply_structural_bend(const Vector3f &p_pivot_origin, const Vector3f &p_axis_dir, const FixedMathCore &p_angle, const FixedMathCore &p_radius);

	/**
	 * punch_hole()
	 * Procedural mesh perforation. Removes faces and re-triangulates.
	 */
	void punch_hole(const Vector3f &p_local_center, const FixedMathCore &p_radius);

	// ------------------------------------------------------------------------
	// Life-Cycle & Data Access
	// ------------------------------------------------------------------------

	void set_base_geometry(const Vector<Vector3f> &p_vertices, const Vector<int> &p_indices);
	
	_FORCE_INLINE_ uint64_t get_vertex_count() const { return static_cast<uint64_t>(vertices.size()); }
	_FORCE_INLINE_ uint64_t get_face_count() const { return static_cast<uint64_t>(face_tensors.size()); }

	void set_material_properties(const FixedMathCore &p_yield, const FixedMathCore &p_elasticity, const FixedMathCore &p_thermal);

	AABBf get_aabb() const;

	/**
	 * update_lod()
	 * Dynamic Automatic LOD based on importance/distance.
	 */
	void update_lod(const Vector3f &p_observer_pos, const FixedMathCore &p_importance);

	DynamicMesh();
	virtual ~DynamicMesh();
};

#endif // DYNAMIC_MESH_H

--- END OF FILE core/math/dynamic_mesh.h ---
