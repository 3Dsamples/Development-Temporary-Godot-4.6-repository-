--- START OF FILE core/math/kernel_registry.cpp ---

#include "core/math/kernel_registry.h"
#include "core/os/memory.h"
#include "core/core_logger.h"

/**
 * KernelRegistry Implementation
 * 
 * Manages the heterogeneous storage of simulation components.
 * Ported to support arbitrary-precision entity handles to ensure
 * that EnTT-style organization works at galactic scales.
 */

namespace UniversalSolver {

/**
 * _get_type_id_internal()
 * 
 * Provides a unique 64-bit identifier for C++ types without relying
 * on RTTI (which can be non-deterministic across compilers).
 */
uint64_t KernelRegistry::_get_type_id_internal(const char* p_type_name) const {
	// DJB2 Hash of the type name string to generate a stable ID
	uint32_t hash = 5381;
	const char* ptr = p_type_name;
	while (*ptr) {
		hash = ((hash << 5) + hash) + (*ptr++);
	}
	return static_cast<uint64_t>(hash);
}

/**
 * create_entity()
 * 
 * Generates a new unique entity handle.
 * Since we use BigIntCore, we never fear handle exhaustion, even in
 * simulations spawning trillions of micro-particles per second.
 */
BigIntCore KernelRegistry::create_entity() {
	std::lock_guard<std::mutex> lock(registry_mutex);
	BigIntCore current = next_entity_id;
	next_entity_id += BigIntCore(1LL);
	return current;
}

/**
 * remove_all_components()
 * 
 * Gracefully clears the registry. 
 * This is called during simulation resets to ensure that all SoA buffers
 * are deallocated through ETEngine's memory tracking system.
 */
void KernelRegistry::clear() {
	std::lock_guard<std::mutex> lock(registry_mutex);
	
	for (auto &E : registries) {
		// IStorage destructor handles the deletion of specific ComponentStreams
		memdelete(E.value);
	}
	registries.clear();
	next_entity_id = BigIntCore(0LL);
	
	Logger::info("Universal Solver: Kernel Registry cleared and synchronized.");
}

/**
 * report_memory_usage()
 * 
 * ETEngine Telemetry Integration.
 * Traverses all ComponentStreams to calculate the total byte-weight 
 * of the simulation state using BigIntCore.
 */
BigIntCore KernelRegistry::get_total_memory_usage() const {
	BigIntCore total_bytes(0LL);
	
	for (auto it = registries.begin(); it.is_valid(); it.next()) {
		// Memory::get_static_mem_size returns bytes, we accumulate into BigInt
		total_bytes += BigIntCore(static_cast<int64_t>(Memory::get_static_mem_size(it.value())));
	}
	
	return total_bytes;
}

KernelRegistry::KernelRegistry() {
	next_entity_id = BigIntCore(0LL);
}

KernelRegistry::~KernelRegistry() {
	clear();
}

} // namespace UniversalSolver

--- END OF FILE core/math/kernel_registry.cpp ---
