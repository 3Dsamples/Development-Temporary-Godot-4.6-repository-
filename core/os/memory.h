--- START OF FILE core/os/memory.h ---

#ifndef MEMORY_H
#define MEMORY_H

#include "core/typedefs.h"
#include "src/big_int_core.h"
#include <cstddef>
#include <new>

/**
 * Memory Class
 * 
 * Centralized memory management for the Scale-Aware pipeline.
 * Aligns allocations to 32 bytes for Warp-Kernel SIMD compatibility.
 * Tracks usage via BigIntCore to support infinite-scale allocation telemetry.
 */
class Memory {
	static BigIntCore mem_usage;
	static BigIntCore mem_max_usage;
	static BigIntCore alloc_count;

public:
	// ------------------------------------------------------------------------
	// Telemetry & Statistics
	// ------------------------------------------------------------------------
	static BigIntCore get_mem_usage();
	static BigIntCore get_mem_max_usage();
	static BigIntCore get_alloc_count();

	// ------------------------------------------------------------------------
	// Deterministic Allocators
	// ------------------------------------------------------------------------
	static void *alloc_static(uint64_t p_bytes, bool p_pad_align = true);
	static void *realloc_static(void *p_ptr, uint64_t p_bytes, bool p_pad_align = true);
	static void free_static(void *p_ptr);

	static uint64_t get_static_mem_size(void *p_ptr);
};

// ============================================================================
// Global Memory Macros (Deterministic Lifecycle)
// ============================================================================

/**
 * memnew()
 * 
 * High-performance object instantiation. 
 * Ensures every simulation object is 32-byte aligned for zero-copy EnTT SoA.
 */
template <typename T>
_FORCE_INLINE_ T *post_initialize(T *p_obj) {
	return p_obj;
}

#define memnew(m_class) post_initialize(new (Memory::alloc_static(sizeof(m_class))) m_class)

/**
 * memnew_placement()
 * 
 * Placement new for reconstructing math components (FixedMathCore/BigIntCore)
 * directly within pre-allocated Warp kernel buffers.
 */
#define memnew_placement(m_placement, m_class) new (m_placement) m_class

/**
 * memdelete()
 * 
 * Deterministic destruction. 
 * Unregisters the object from the Scale-Aware telemetry before freeing memory.
 */
template <typename T>
void memdelete(T *p_ptr) {
	if (unlikely(!p_ptr)) return;
	p_ptr->~T();
	Memory::free_static(p_ptr);
}

/**
 * memalloc() / memfree()
 * 
 * Raw buffer management for massive simulation data arrays.
 * Size is handled as uint64_t to prevent overflow in large world sectors.
 */
#define memalloc(m_size) Memory::alloc_static(m_size)
#define memrealloc(m_ptr, m_size) Memory::realloc_static(m_ptr, m_size)
#define memfree(m_ptr) Memory::free_static(m_ptr)

#endif // MEMORY_H

--- END OF FILE core/os/memory.h ---
