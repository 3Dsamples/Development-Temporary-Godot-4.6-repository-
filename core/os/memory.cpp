--- START OF FILE core/os/memory.cpp ---

#include "core/os/memory.h"
#include "core/error/error_macros.h"
#include "src/big_int_core.h"
#include <cstdlib>
#include <cstring>
#include <mutex>

// ============================================================================
// Static Member Initialization (Scale-Aware Telemetry)
// ============================================================================

BigIntCore Memory::mem_usage = BigIntCore(0LL);
BigIntCore Memory::mem_max_usage = BigIntCore(0LL);
BigIntCore Memory::alloc_count = BigIntCore(0LL);

static std::mutex memory_mutex;

// Header stored before every allocation to track size and maintain alignment
struct ET_ALIGN_32 MemoryHeader {
	uint64_t size;
	uint64_t pad[3]; // Padding to ensure 32-byte alignment for the following data
};

// ============================================================================
// Telemetry API
// ============================================================================

BigIntCore Memory::get_mem_usage() {
	std::lock_guard<std::mutex> lock(memory_mutex);
	return mem_usage;
}

BigIntCore Memory::get_mem_max_usage() {
	std::lock_guard<std::mutex> lock(memory_mutex);
	return mem_max_usage;
}

BigIntCore Memory::get_alloc_count() {
	std::lock_guard<std::mutex> lock(memory_mutex);
	return alloc_count;
}

// ============================================================================
// Deterministic Allocators (SIMD Aligned)
// ============================================================================

/**
 * alloc_static()
 * 
 * Allocates a 32-byte aligned block of memory. 
 * Optimized for Warp-style parallel kernels and EnTT SoA component pools.
 */
void *Memory::alloc_static(uint64_t p_bytes, bool p_pad_align) {
	if (unlikely(p_bytes == 0)) return nullptr;

	uint64_t total_size = p_bytes + sizeof(MemoryHeader);
	
	// Ensure 32-byte alignment for SIMD/Warp hardware-agnostic execution
	void *ptr = nullptr;
#if defined(_MSC_VER) || defined(__MINGW32__)
	ptr = _aligned_malloc(total_size, 32);
#else
	if (posix_memalign(&ptr, 32, total_size) != 0) ptr = nullptr;
#endif

	CRASH_COND_MSG(!ptr, "Universal Solver Fatal: Out of memory during galactic-scale allocation.");

	MemoryHeader *header = static_cast<MemoryHeader *>(ptr);
	header->size = p_bytes;

	{
		std::lock_guard<std::mutex> lock(memory_mutex);
		mem_usage += BigIntCore(static_cast<int64_t>(p_bytes));
		if (mem_usage > mem_max_usage) {
			mem_max_usage = mem_usage;
		}
		alloc_count += BigIntCore(1LL);
	}

	return static_cast<void *>(header + 1);
}

/**
 * realloc_static()
 * 
 * Resizes an existing aligned block. Maintains data integrity for 
 * high-precision math components during dynamic registry expansion.
 */
void *Memory::realloc_static(void *p_ptr, uint64_t p_bytes, bool p_pad_align) {
	if (unlikely(!p_ptr)) return alloc_static(p_bytes, p_pad_align);
	if (unlikely(p_bytes == 0)) {
		free_static(p_ptr);
		return nullptr;
	}

	MemoryHeader *header = static_cast<MemoryHeader *>(p_ptr) - 1;
	uint64_t old_size = header->size;
	if (old_size == p_bytes) return p_ptr;

	void *new_ptr = alloc_static(p_bytes, p_pad_align);
	if (new_ptr) {
		std::memcpy(new_ptr, p_ptr, old_size < p_bytes ? old_size : p_bytes);
		free_static(p_ptr);
	}

	return new_ptr;
}

/**
 * free_static()
 * 
 * Frees an aligned block and updates the BigIntCore telemetry.
 */
void Memory::free_static(void *p_ptr) {
	if (unlikely(!p_ptr)) return;

	MemoryHeader *header = static_cast<MemoryHeader *>(p_ptr) - 1;
	uint64_t size = header->size;

	{
		std::lock_guard<std::mutex> lock(memory_mutex);
		mem_usage -= BigIntCore(static_cast<int64_t>(size));
		alloc_count -= BigIntCore(1LL);
	}

#if defined(_MSC_VER) || defined(__MINGW32__)
	_aligned_free(header);
#else
	std::free(header);
#endif
}

uint64_t Memory::get_static_mem_size(void *p_ptr) {
	if (unlikely(!p_ptr)) return 0;
	MemoryHeader *header = static_cast<MemoryHeader *>(p_ptr) - 1;
	return header->size;
}

--- END OF FILE core/os/memory.cpp ---
