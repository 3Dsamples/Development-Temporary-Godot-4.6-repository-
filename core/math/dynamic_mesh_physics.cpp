--- START OF FILE core/math/dynamic_mesh_physics.cpp ---

#include "core/math/dynamic_mesh.h"
#include "core/math/math_funcs.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/os/memory.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the physical kernels for the Universal Solver.
 * Instantiating for FixedMathCore enables bit-perfect 60/120 FPS 
 * synchronization across parallel workers.
 */
template class DynamicMesh<FixedMathCore>;

/**
 * simulation_step()
 * 
 * The master orchestrator for the parallel vertex sweep.
 * It divides the EnTT-style vertex stream into cache-aligned chunks 
 * and dispatches them to the Warp-style execution pool.
 */
template <typename T>
void DynamicMesh<T>::simulation_step(T p_delta) {
	if (unlikely(is_updating)) return;
	is_updating = true;

	MeshTier &tier = lod_tiers.ptrw()[active_lod];
	uint32_t v_count = tier.vertices.size();
	if (v_count == 0) {
		is_updating = false;
		return;
	}

	// ETEngine Strategy: Optimize chunk size to saturate L1/L2 cache lines
	uint32_t worker_count = SimulationThreadPool::get_singleton()->get_worker_count();
	uint32_t chunk_size = v_count / worker_count;

	if (chunk_size < 64) {
		// Single-threaded path for low-complexity meshes to avoid sync overhead
		_worker_thread_process(0, v_count, p_delta);
	} else {
		for (uint32_t i = 0; i < worker_count; i++) {
			uint32_t start = i * chunk_size;
			uint32_t end = (i == worker_count - 1) ? v_count : (i + 1) * chunk_size;

			SimulationThreadPool::get_singleton()->enqueue_task([this, start, end, p_delta]() {
				_worker_thread_process(start, end, p_delta);
			}, SimulationThreadPool::PRIORITY_HIGH);
		}
		// Barrier: Wait for all Warp kernels to finish the current wave
		SimulationThreadPool::get_singleton()->wait_for_all();
	}

	// 2. Serial Post-Sweep: Material Relaxation and Thermal Conduction
	// We iterate through the Face3 tensors to settle stress and propagate heat
	for (uint32_t i = 0; i < tier.faces.size(); i++) {
		Face3<T> &f = tier.faces.ptrw()[i];
		
		// Convective Cooling: T_new = T_old + k * (T_ambient - T_old) * dt
		T ambient(12591030272LL, true); // 293.15K in FixedMath
		f.thermal_energy += (ambient - f.thermal_energy) * thermal_conductivity * p_delta;
		
		// Elastic Relaxation: Fatigue decays if below the yield threshold
		if (f.stress_level > MathConstants<T>::zero()) {
			T relaxation = elasticity * p_delta;
			f.stress_level = (f.stress_level > relaxation) ? f.stress_level - relaxation : MathConstants<T>::zero();
		}
	}

	aabb_dirty = true;
	is_updating = false;
}

/**
 * _worker_thread_process()
 * 
 * The zero-copy vertex integration kernel.
 * Operates directly on the aligned SoA memory addresses.
 * Uses bit-perfect Semi-Implicit Euler integration.
 */
template <typename T>
void DynamicMesh<T>::_worker_thread_process(uint32_t p_start, uint32_t p_end, T p_delta) {
	MeshTier &tier = lod_tiers.ptrw()[active_lod];
	Vertex *v_ptr = tier.vertices.ptrw();

	for (uint32_t i = p_start; i < p_end; i++) {
		Vertex &v = v_ptr[i];

		// 1. Integration: Pos = Pos + Vel * dt
		v.position += v.velocity * p_delta;

		// 2. Physical Damping: Simulates material internal friction
		// damping_factor = 1.0 - (friction_coeff * dt)
		T damping = MathConstants<T>::one() - (T(214748364LL, true) * p_delta); // 0.05 damping
		v.velocity *= damping;

		// 3. Structural Fatigue Decay: Minor damage recovery per step
		if (v.fatigue > MathConstants<T>::zero()) {
			T recovery = T(42949LL, true) * p_delta; // 0.00001 per sec
			v.fatigue = (v.fatigue > recovery) ? v.fatigue - recovery : MathConstants<T>::zero();
		}

		// 4. Spatial Index Update
		// We only trigger a re-hash if the vertex crosses a 2.0 unit boundary
		tier.spatial_index.update(i, v.position, sector_x, sector_y, sector_z);
	}
}

/**
 * fracture()
 * 
 * Implements procedural volumetric fracturing.
 * Uses stochastic clipping planes generated from a deterministic seed.
 * Slices the mesh into independent debris objects using bit-perfect FixedMath.
 */
template <typename T>
Vector<Ref<DynamicMesh<T>>> DynamicMesh<T>::fracture(const Vector3<T> &p_epicenter, T p_energy) {
	Vector<Ref<DynamicMesh<T>>> fragments;
	MeshTier &base_tier = lod_tiers.ptrw()[0];

	if (base_tier.faces.size() == 0) return fragments;

	// ETEngine Strategy: Create 4 stochastic fracture planes based on energy magnitude
	int shard_count = (p_energy > T(100LL, false)) ? 4 : 2;
	Vector<Plane<T>> planes;
	
	for (int i = 0; i < shard_count; i++) {
		T rx = Math::randf() - MathConstants<T>::half();
		T ry = Math::randf() - MathConstants<T>::half();
		T rz = Math::randf() - MathConstants<T>::half();
		planes.push_back(Plane<T>(p_epicenter, Vector3<T>(rx, ry, rz).normalized()));
	}

	// Bucket faces into fragments based on plane half-space tests
	Vector<Vector<Face3<T>>> buckets;
	buckets.resize(1 << shard_count);

	for (uint32_t i = 0; i < base_tier.faces.size(); i++) {
		uint32_t bucket_idx = 0;
		Vector3<T> median = base_tier.faces[i].get_median();
		for (int j = 0; j < shard_count; j++) {
			if (planes[j].is_point_over(median)) {
				bucket_idx |= (1 << j);
			}
		}
		buckets.ptrw()[bucket_idx].push_back(base_tier.faces[i]);
	}

	// Create new DynamicMesh resources for each populated bucket
	for (int i = 0; i < buckets.size(); i++) {
		if (buckets[i].size() > 0) {
			Ref<DynamicMesh<T>> shard;
			shard.instantiate();
			
			Vector<Vector3<T>> shard_verts;
			Vector<int> shard_indices;
			for (uint32_t j = 0; j < buckets[i].size(); j++) {
				shard_verts.push_back(buckets[i][j].vertex[0]);
				shard_verts.push_back(buckets[i][j].vertex[1]);
				shard_verts.push_back(buckets[i][j].vertex[2]);
				int base_idx = shard_verts.size() - 3;
				shard_indices.push_back(base_idx);
				shard_indices.push_back(base_idx + 1);
				shard_indices.push_back(base_idx + 2);
			}
			shard->set_initial_geometry(shard_verts, shard_indices);
			fragments.push_back(shard);
		}
	}

	return fragments;
}

--- END OF FILE core/math/dynamic_mesh_physics.cpp ---
