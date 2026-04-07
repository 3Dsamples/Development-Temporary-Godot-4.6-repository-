--- START OF FILE core/math/geometry_instance_hierarchy.cpp ---

#include "core/math/geometry_instance.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/kernel_registry.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: HierarchySectorPropagationKernel
 * 
 * Computes the absolute world-space transform and sector indices for a child entity.
 * 1. Basis Concatenation: ChildBasis = ParentBasis * LocalBasis.
 * 2. Position Calculation: ChildPos = ParentPos + (ParentBasis * LocalPos).
 * 3. Sector Drift Resolve: If ChildPos > SectorSize, increment BigInt sector and recenter.
 */
void hierarchy_sector_propagation_kernel(
		const BigIntCore &p_index,
		Transform3Df &r_child_world_xform,
		BigIntCore &r_child_sx, BigIntCore &r_child_sy, BigIntCore &r_child_sz,
		const Transform3Df &p_child_local_xform,
		const Transform3Df &p_parent_world_xform,
		const BigIntCore &p_parent_sx, const BigIntCore &p_parent_sy, const BigIntCore &p_parent_sz,
		const FixedMathCore &p_sector_size) {

	// 1. Recursive Basis Resolution
	// strictly deterministic matrix multiplication using FixedMathCore
	r_child_world_xform.basis = p_parent_world_xform.basis * p_child_local_xform.basis;

	// 2. Relative Position Resolve in Parent Basis
	Vector3f local_offset = p_parent_world_xform.basis.xform(p_child_local_xform.origin);
	Vector3f world_pos_unnormalized = p_parent_world_xform.origin + local_offset;

	// 3. Inherit Base Sectors
	r_child_sx = p_parent_sx;
	r_child_sy = p_parent_sy;
	r_child_sz = p_parent_sz;

	// 4. Resolve Sophisticated Sector Drift
	// Checks if the child has drifted relative to the parent into an adjacent BigInt sector.
	// Essential for high-speed ship attachments or multi-sector starbases.
	int64_t dx = Math::floor(world_pos_unnormalized.x / p_sector_size).to_int();
	int64_t dy = Math::floor(world_pos_unnormalized.y / p_sector_size).to_int();
	int64_t dz = Math::floor(world_pos_unnormalized.z / p_sector_size).to_int();

	if (dx != 0 || dy != 0 || dz != 0) {
		r_child_sx += BigIntCore(dx);
		r_child_sy += BigIntCore(dy);
		r_child_sz += BigIntCore(dz);

		FixedMathCore off_x = p_sector_size * FixedMathCore(dx);
		FixedMathCore off_y = p_sector_size * FixedMathCore(dy);
		FixedMathCore off_z = p_sector_size * FixedMathCore(dz);
		
		// Recenter to maintain maximum Q32.32 precision (Zero-Drift)
		r_child_world_xform.origin = world_pos_unnormalized - Vector3f(off_x, off_y, off_z);
	} else {
		r_child_world_xform.origin = world_pos_unnormalized;
	}
}

/**
 * execute_hierarchy_propagation_wave()
 * 
 * The master 120 FPS wave for hierarchical synchronization.
 * 1. Organizes entities by depth (Breadth-First order).
 * 2. Parallelizes the resolution of each depth-tier using Warp kernels.
 * 3. Ensures parents are bit-perfectly resolved before children.
 */
void execute_hierarchy_propagation_wave(
		KernelRegistry &p_registry,
		const FixedMathCore &p_sector_size) {

	// EnTT Strategy: Hierarchy is stored as a directed acyclic graph in SoA.
	// We iterate through depth layers [1..MaxDepth].
	uint32_t max_depth = p_registry.get_max_hierarchy_depth();
	
	for (uint32_t current_depth = 1; current_depth <= max_depth; current_depth++) {
		auto depth_view = p_registry.get_entities_at_depth(current_depth);
		uint64_t count = depth_view.size();
		if (count == 0) continue;

		uint32_t worker_count = SimulationThreadPool::get_singleton()->get_worker_count();
		uint64_t chunk_size = count / worker_count;

		for (uint32_t w = 0; w < worker_count; w++) {
			uint64_t start = w * chunk_size;
			uint64_t end = (w == worker_count - 1) ? count : (start + chunk_size);

			SimulationThreadPool::get_singleton()->enqueue_task([=, &p_registry]() {
				for (uint64_t i = start; i < end; i++) {
					BigIntCore child_entity = depth_view[i];
					BigIntCore parent_entity = p_registry.get_parent(child_entity);

					// Zero-Copy Direct Memory Access to SoA streams
					hierarchy_sector_propagation_kernel(
						child_entity,
						p_registry.get_component<Transform3Df>(child_entity, COMPONENT_WORLD_XFORM),
						p_registry.get_component<BigIntCore>(child_entity, COMPONENT_SECTOR_X),
						p_registry.get_component<BigIntCore>(child_entity, COMPONENT_SECTOR_Y),
						p_registry.get_component<BigIntCore>(child_entity, COMPONENT_SECTOR_Z),
						p_registry.get_component<Transform3Df>(child_entity, COMPONENT_LOCAL_XFORM),
						p_registry.get_component<Transform3Df>(parent_entity, COMPONENT_WORLD_XFORM),
						p_registry.get_component<BigIntCore>(parent_entity, COMPONENT_SECTOR_X),
						p_registry.get_component<BigIntCore>(parent_entity, COMPONENT_SECTOR_Y),
						p_registry.get_component<BigIntCore>(parent_entity, COMPONENT_SECTOR_Z),
						p_sector_size
					);
				}
			}, SimulationThreadPool::PRIORITY_CRITICAL);
		}
		
		// Barrier: Wait for the current depth tier to synchronize before descending
		SimulationThreadPool::get_singleton()->wait_for_all();
	}
}

/**
 * apply_kinematic_attachment_sync()
 * 
 * Sophisticated Interaction: Attachment Momentum Transfer.
 * Ensures that if a ship rotates at high speed, attached sub-components (robots, turrets)
 * experience centripetal forces derived from the bit-perfect parent basis.
 */
void apply_kinematic_attachment_sync(
		Vector3f &r_child_velocity,
		const Vector3f &p_parent_linear_vel,
		const Vector3f &p_parent_angular_vel,
		const Vector3f &p_child_local_pos) {
	
	// v_world = v_parent + omega_parent x r_child
	r_child_velocity = p_parent_linear_vel + p_parent_angular_vel.cross(p_child_local_pos);
}

} // namespace UniversalSolver

--- END OF FILE core/math/geometry_instance_hierarchy.cpp ---
