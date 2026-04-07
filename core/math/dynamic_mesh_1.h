--- START OF FILE core/math/dynamic_mesh.h ---

#ifndef DYNAMIC_MESH_H
#define DYNAMIC_MESH_H

#include "core/io/resource.h"
#include "core/math/vector3.h"
#include "core/math/face3.h"
#include "core/math/aabb.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * DynamicMesh
 * 
 * A deterministic physical geometry resource.
 * Aligned to 32 bytes for EnTT SoA optimization and Warp Parallel Kernels.
 * strictly uses FixedMathCore for all vertex and material calculations.
 */
class ET_ALIGN_32 DynamicMesh : public Resource {
	GDCLASS(DynamicMesh, Resource);

public:
	// --- Internal Simulation Tensors (SIMD Aligned) ---
	struct ET_ALIGN_32 VertexState {
		Vector3f position;      // Current displaced position
		Vector3f rest_position; // Original shape anchor
		Vector3f velocity;      // Internal oscillation velocity
		Vector3f normal;        // Surface normal for pressure/lighting
		FixedMathCore fatigue;  // Accumulated damage [0..1]
		FixedMathCore temperature; // Thermal energy for buckling/melting
		FixedMathCore mass;     // Local vertex mass
	};

	struct ET_ALIGN_32 FaceState {
		uint32_t indices[3];
		FixedMathCore surface_tension; // Stiffness of the local skin
		FixedMathCore expansion_coeff;  // Thermal sensitivity
	};

private:
	// SoA Component Streams managed for Zero-Copy Warp execution
	Vector<VertexState> vertex_stream;
	Vector<FaceState> face_stream;

	// Material Parameters
	FixedMathCore stiffness;        // Global Hooke's k
	FixedMathCore damping;          // Viscoelastic absorption
	FixedMathCore pressure_coeff;   // Balloon internal pressure
	FixedMathCore rest_volume;      // Target volume for preservation

	mutable AABBf local_aabb;
	mutable bool aabb_dirty = true;

	void _recalculate_aabb() const;

protected:
	static void _bind_methods();

public:
	// ------------------------------------------------------------------------
	// Deformation API (Real-Time Sophisticated Interactions)
	// ------------------------------------------------------------------------

	/**
	 * apply_poke()
	 * Indents the surface at a specific local point.
	 */
	void apply_poke(const Vector3f &p_local_point, const Vector3f &p_dir, const FixedMathCore &p_force, const FixedMathCore &p_radius);

	/**
	 * apply_pull()
	 * Grabs and stretches the mesh surface.
	 */
	void apply_pull(const Vector3f &p_local_point, const Vector3f &p_target_world, const FixedMathCore &p_radius);

	/**
	 * apply_pinch()
	 * Compresses material between two points (Flesh behavior).
	 */
	void apply_pinch(const Vector3f &p_point_a, const Vector3f &p_point_b, const FixedMathCore &p_force, const FixedMathCore &p_radius);

	// ------------------------------------------------------------------------
	// Deterministic Simulation Waves (120 FPS Kernels)
	// ------------------------------------------------------------------------

	/**
	 * execute_elastic_sweep()
	 * The "Balloon Effect" wave. Returns vertices to rest position based on stiffness.
	 */
	void execute_elastic_sweep(const FixedMathCore &p_delta);

	/**
	 * execute_volume_preservation()
	 * Adjusts internal pressure based on signed tetrahedral volume delta.
	 */
	void execute_volume_preservation(const FixedMathCore &p_delta);

	/**
	 * execute_thermal_conduction()
	 * Propagates heat through the mesh and applies buckling displacement.
	 */
	void execute_thermal_sweep(const FixedMathCore &p_delta);

	// ------------------------------------------------------------------------
	// Lifecycle & Data Management
	// ------------------------------------------------------------------------

	void set_initial_geometry(const Vector<Vector3f> &p_vertices, const Vector<int> &p_indices);
	void set_material_tensors(const FixedMathCore &p_stiffness, const FixedMathCore &p_damping, const FixedMathCore &p_pressure);

	_FORCE_INLINE_ uint64_t get_vertex_count() const { return static_cast<uint64_t>(vertex_stream.size()); }
	_FORCE_INLINE_ AABBf get_aabb() const {
		if (aabb_dirty) _recalculate_aabb();
		return local_aabb;
	}

	/**
	 * update_lod_level()
	 * Dynamic Automatic LOD: Subdivides or collapses based on observer distance.
	 */
	void update_lod_level(const Vector3f &p_observer_local, const FixedMathCore &p_threshold);

	DynamicMesh();
	virtual ~DynamicMesh();
};

#endif // DYNAMIC_MESH_H

--- END OF FILE core/math/dynamic_mesh.h ---
