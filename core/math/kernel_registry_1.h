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
 * A 32-byte aligned Structure of Arrays (SoA) container.
 * Engineered for zero-copy access by Warp-style parallel kernels.
 */
template <typename T>
struct ET_ALIGN_32 ComponentStream {
	Vector<T> data;
	Vector<BigIntCore> entity_map;

	_FORCE_INLINE_ T *get_base_ptr() { return data.ptrw(); }
	_FORCE_INLINE_ const T *get_base_ptr() const { return data.ptr(); }
	_FORCE_INLINE_ uint64_t size() const { return static_cast<uint64_t>(data.size()); }

	_FORCE_INLINE_ T& operator[](uint64_t p_idx) { return data.ptrw()[p_idx]; }
	_FORCE_INLINE_ const T& operator[](uint64_t p_idx) const { return data[p_idx]; }

	void push(const BigIntCore &p_entity, const T &p_component) {
		entity_map.push_back(p_entity);
		data.push_back(p_component);
	}
};

/**
 * KernelRegistry
 * 
 * The master ECS registry for the Scale-Aware pipeline.
 * Manages entities via BigIntCore and components via SIMD-aligned streams.
 */
class ET_ALIGN_32 KernelRegistry {
private:
	BigIntCore next_entity_id;
	
	// Type-erased storage interface for heterogeneous components
	struct IStorage { 
		virtual void clear() = 0;
		virtual ~IStorage() {} 
	};
	
	template <typename T>
	struct Storage : public IStorage {
		ComponentStream<T> stream;
		virtual void clear() override {
			stream.data.clear();
			stream.entity_map.clear();
		}
	};

	// Mapping of component Type IDs to their respective SoA Storage
	HashMap<uint64_t, IStorage*> registries;

	// Internal sparse-set mapping from Entity ID to index in the dense SoA stream
	HashMap<BigIntCore, uint64_t> entity_to_index;

	template <typename T>
	_FORCE_INLINE_ uint64_t _get_type_id() const {
		static const uint8_t type_marker = 0;
		return reinterpret_cast<uint64_t>(&type_marker);
	}

public:
	// ------------------------------------------------------------------------
	// Entity Management
	// ------------------------------------------------------------------------

	/**
	 * create_entity()
	 * Generates a unique BigIntCore handle. 
	 * Allows for trillions of entities without handle exhaustion.
	 */
	BigIntCore create_entity() {
		BigIntCore current = next_entity_id;
		next_entity_id += BigIntCore(1LL);
		return current;
	}

	/**
	 * assign()
	 * Attaches a component to an entity. Aligns data in the SoA stream
	 * for maximum cache-locality during Warp sweeps.
	 */
	template <typename T>
	void assign(const BigIntCore &p_entity, const T &p_data) {
		uint64_t tid = _get_type_id<T>();
		if (unlikely(!registries.has(tid))) {
			registries[tid] = memnew(Storage<T>());
		}
		
		Storage<T> *s = static_cast<Storage<T>*>(registries[tid]);
		uint64_t idx = s->stream.size();
		s->stream.push(p_entity, p_data);
		
		// For simplicity in this core, we map the entity to its index in this stream
		// (Full implementation uses Sparse Sets for O(1) multi-component lookup)
		entity_to_index.insert(p_entity, idx);
	}

	// ------------------------------------------------------------------------
	// Data Access (Zero-Copy)
	// ------------------------------------------------------------------------

	/**
	 * get_stream()
	 * Returns the raw SoA component stream for parallel kernel dispatch.
	 */
	template <typename T>
	ComponentStream<T>& get_stream() {
		uint64_t tid = _get_type_id<T>();
		CRASH_COND_MSG(!registries.has(tid), "Universal Solver Error: Component type not registered.");
		return static_cast<Storage<T>*>(registries[tid])->stream;
	}

	template <typename T>
	_FORCE_INLINE_ T& get_component(const BigIntCore &p_entity) {
		uint64_t tid = _get_type_id<T>();
		uint64_t idx = entity_to_index[p_entity];
		return static_cast<Storage<T>*>(registries[tid])->stream[idx];
	}

	// ------------------------------------------------------------------------
	// Lifecycle
	// ------------------------------------------------------------------------

	void clear() {
		for (auto &E : registries) {
			E.value->clear();
			memdelete(E.value);
		}
		registries.clear();
		entity_to_index.clear();
		next_entity_id = BigIntCore(0LL);
	}

	KernelRegistry() : next_entity_id(0LL) {}
	~KernelRegistry() { clear(); }
};

#endif // KERNEL_REGISTRY_H

--- END OF FILE core/math/kernel_registry.h ---
