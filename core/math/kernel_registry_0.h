--- START OF FILE core/math/kernel_registry.h ---

#ifndef KERNEL_REGISTRY_H
#define KERNEL_REGISTRY_H

#include "core/typedefs.h"
#include "core/templates/vector.h"
#include "core/templates/hash_map.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * ComponentStream
 * 
 * A cache-aligned Structure of Arrays (SoA) container.
 * Specifically engineered for NVIDIA Warp kernels to sweep through
 * FixedMathCore components without pointer chasing or cache misses.
 */
template <typename T>
struct ET_ALIGN_32 ComponentStream {
	Vector<T> data;
	Vector<BigIntCore> entity_map;

	ET_SIMD_INLINE T *get_base_ptr() { return data.ptrw(); }
	ET_SIMD_INLINE const T *get_base_ptr() const { return data.ptr(); }
	ET_SIMD_INLINE uint64_t size() const { return (uint64_t)data.size(); }

	void push(const BigIntCore &p_entity, const T &p_component) {
		entity_map.push_back(p_entity);
		data.push_back(p_component);
	}
};

/**
 * KernelRegistry
 * 
 * The central EnTT-based data organizer for the Universal Solver.
 * Manages Entity IDs as BigIntCore to support galactic-scale entity counts.
 * Provides zero-copy access to aligned memory for Warp-style batch math.
 */
class KernelRegistry {
private:
	// Use BigIntCore as the primary Entity handle for infinite scalability
	BigIntCore next_entity_id;
	
	// Type-erased storage map for heterogeneous components
	struct IStorage { virtual ~IStorage() {} };
	
	template <typename T>
	struct Storage : public IStorage {
		ComponentStream<T> stream;
	};

	HashMap<uint64_t, IStorage*> registries;

	template <typename T>
	uint64_t _get_type_id() const {
		static const uint8_t type_marker = 0;
		return reinterpret_cast<uint64_t>(&type_marker);
	}

public:
	/**
	 * create_entity()
	 * Generates a unique BigIntCore ID. 
	 * Ensures O(1) allocation for 120 FPS simulation heartbeats.
	 */
	BigIntCore create_entity() {
		BigIntCore current = next_entity_id;
		next_entity_id += BigIntCore(1LL);
		return current;
	}

	/**
	 * assign()
	 * Injects a component into the SoA stream. 
	 * Aligned for Warp kernel zero-copy execution.
	 */
	template <typename T>
	void assign(const BigIntCore &p_entity, const T &p_data) {
		uint64_t tid = _get_type_id<T>();
		if (!registries.has(tid)) {
			registries[tid] = memnew(Storage<T>());
		}
		static_cast<Storage<T>*>(registries[tid])->stream.push(p_entity, p_data);
	}

	/**
	 * get_stream()
	 * Returns the raw component stream for Warp kernel dispatch.
	 */
	template <typename T>
	ComponentStream<T>& get_stream() {
		uint64_t tid = _get_type_id<T>();
		CRASH_COND_MSG(!registries.has(tid), "Universal Solver Error: Component type not registered.");
		return static_cast<Storage<T>*>(registries[tid])->stream;
	}

	KernelRegistry() : next_entity_id(0LL) {}
	~KernelRegistry() {
		for (auto &E : registries) {
			memdelete(E.value);
		}
		registries.clear();
	}
};

#endif // KERNEL_REGISTRY_H

--- END OF FILE core/math/kernel_registry.h ---
