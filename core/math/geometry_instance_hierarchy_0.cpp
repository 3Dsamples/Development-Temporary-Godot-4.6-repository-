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
 * Warp Kernel: HierarchyPropagationKernel
 * 
 * Computes the world-space state of a child entity based on its parent's world-space state.
 * Specifically handles the hand-off between local FixedMathCore coordinates and BigIntCore sectors.
 * 
 * r_child_world_xform: Output world-space transform.
 * r_child_sector: Output BigIntCore sectors (X, Y, Z).
 * p_child_local_xform: Input transform relative to parent.
 * p_parent_world_xform: World transform of the parent.
 * p_parent_sector: Parent's BigIntCore sectors.
 */
void hierarchy_propagation_kernel(
		Transform3Df &r_child_world_xform,
		BigIntCore &r_child_sx, BigIntCore &r_child_sy, BigIntCore &r_child_sz,
		const Transform3Df &p_child_local_xform,
		const Transform3Df &p_parent_world_xform,
		const BigIntCore &p_parent_sx, const BigIntCore &p_parent_sy, const BigIntCore &p_parent_sz,
		const FixedMathCore &p_sector_size) {

	// 1. Compute local Basis/Rotation: WorldBasis = ParentWorldBasis * ChildLocalBasis
	r_child_world_xform.basis = p_parent_world_xform.basis * p_child_local_xform.basis;

	// 2. Compute translation in Parent's local FixedMath space
	Vector3f local_offset = p_parent_world_xform.xform(p_child_local_xform.origin);
	
	// 3. Galactic Sector Addition
	// Start with parent's absolute sector coordinates
	r_child_sx = p_parent_sx;
	r_child_sy = p_parent_sy;
	r_child_sz = p_parent_sz;

	// 4. Resolve Sector Crossing
	// If the child's offset relative to the parent pushes it into a new sector,
	// we perform a bit-perfect drift correction.
	int64_t dx = Math::floor(local_offset.x / p_sector_size).to_int();
	int64_t dy = Math::floor(local_offset.y / p_sector_size).to_int();
	int64_t dz = Math::floor(local_offset.z / p_sector_size).to_int();

	if (dx != 0 || dy != 0 || dz != 0) {
		r_child_sx += BigIntCore(dx);
		r_child_sy += BigIntCore(dy);
		r_child_sz += BigIntCore(dz);

		Vector3f sector_offset(
			p_sector_size * FixedMathCore(dx),
			p_sector_size * FixedMathCore(dy),
			p_sector_size * FixedMathCore(dz)
		);
		r_child_world_xform.origin = local_offset - sector_offset;
	} else {
		r_child_world_xform.origin = local_offset;
	}
}

/**
 * propagate_hierarchies_parallel()
 * 
 * Orchestrates the hierarchical update across the EnTT registry.
 * Processes entities level-by-level (Breadth-First) to ensure parents are 
 * always resolved before their children.
 */
void propagate_hierarchies_parallel(KernelRegistry &p_registry, const FixedMathCore &p_sector_size) {
	// ETEngine Strategy: Use a 'Depth' component to group entities into batches.
	// This allows Warp kernels to run in parallel on all entities of the same depth.
	
	uint32_t max_depth = p_registry.get_max_hierarchy_depth();
	
	for (uint32_t d = 1; d <= max_depth; d++) {
		auto view = p_registry.get_entities_at_depth(d);
		uint64_t count = view.size();
		if (count == 0) continue;

		uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
		uint64_t chunk = count / workers;

		for (uint32_t w = 0; w < workers; w++) {
			uint64_t start = w * chunk;
			uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

			SimulationThreadPool::get_singleton()->enqueue_task([=, &p_registry]() {
				for (uint64_t i = start; i < end; i++) {
					BigIntCore entity = view[i];
					BigIntCore parent = p_registry.get_parent(entity);

					// Zero-Copy access to parent and child components in the EnTT registry
					hierarchy_propagation_kernel(
						p_registry.get_component<Transform3Df>(entity, COMPONENT_WORLD_XFORM),
						p_registry.get_component<BigIntCore>(entity, COMPONENT_SECTOR_X),
						p_registry.get_component<BigIntCore>(entity, COMPONENT_SECTOR_Y),
						p_registry.get_component<BigIntCore>(entity, COMPONENT_SECTOR_Z),
						p_registry.get_component<Transform3Df>(entity, COMPONENT_LOCAL_XFORM),
						p_registry.get_component<Transform3Df>(parent, COMPONENT_WORLD_XFORM),
						p_registry.get_component<BigIntCore>(parent, COMPONENT_SECTOR_X),
						p_registry.get_component<BigIntCore>(parent, COMPONENT_SECTOR_Y),
						p_registry.get_component<BigIntCore>(parent, COMPONENT_SECTOR_Z),
						p_sector_size
					);
				}
			}, SimulationThreadPool::PRIORITY_CRITICAL);
		}
		// Barrier ensures parents of depth 'd' are finished before children of 'd+1' start
		SimulationThreadPool::get_singleton()->wait_for_all();
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/geometry_instance_hierarchy.cpp ---
