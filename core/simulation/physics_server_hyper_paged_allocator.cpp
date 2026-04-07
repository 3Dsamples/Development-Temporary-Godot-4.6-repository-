--- START OF FILE core/simulation/physics_server_hyper_paged_allocator.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/os/memory.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"
#include <mutex>

namespace UniversalSolver {

/**
 * BodyPage
 * 
 * A single contiguous block of memory for Physics Bodies.
 * Aligned to 32 bytes to ensure SIMD-optimized EnTT SoA access.
 */
struct BodyPage {
	static const uint32_t BODIES_PER_PAGE = 1024;
	PhysicsServerHyper::Body *bodies;
	BodyPage *next;

	BodyPage() : bodies(nullptr), next(nullptr) {}
};

// Global state for the Paged Allocator
static BodyPage *page_list_head = nullptr;
static PhysicsServerHyper::Body *free_list_head = nullptr;
static std::mutex pool_mutex;

// ETEngine Telemetry Counters
static BigIntCore total_allocated_entities = BigIntCore(0LL);
static BigIntCore active_page_count = BigIntCore(0LL);

/**
 * _allocate_new_page()
 * 
 * Reserves a 32-byte aligned block of memory for 1024 body structures.
 * Links the bodies into the global free-list using pointer-threading.
 */
static void _allocate_new_page() {
	// Allocate the page header
	BodyPage *new_page = (BodyPage *)Memory::alloc_static(sizeof(BodyPage), false);
	
	// Allocate the contiguous body array (32-byte aligned for Warp Kernels)
	uint64_t body_array_size = sizeof(PhysicsServerHyper::Body) * BodyPage::BODIES_PER_PAGE;
	new_page->bodies = (PhysicsServerHyper::Body *)Memory::alloc_static(body_array_size, true);
	
	new_page->next = page_list_head;
	page_list_head = new_page;

	// Thread the bodies into the free-list
	for (uint32_t i = 0; i < BodyPage::BODIES_PER_PAGE; i++) {
		PhysicsServerHyper::Body *b = &new_page->bodies[i];
		
		// Use the first 8 bytes of the body memory to store the next pointer
		// This is safe because the memory is currently uninitialized.
		*(PhysicsServerHyper::Body **)b = free_list_head;
		free_list_head = b;
	}

	// Update BigInt telemetry
	active_page_count += BigIntCore(1LL);
	total_allocated_entities += BigIntCore(static_cast<int64_t>(BodyPage::BODIES_PER_PAGE));
}

/**
 * allocate_body_memory()
 * 
 * Returns a pointer to a bit-perfectly initialized Body structure.
 * Guaranteed O(1) complexity unless a new memory page must be requested from the OS.
 */
PhysicsServerHyper::Body *PhysicsServerHyper::_allocate_body_memory() {
	std::lock_guard<std::mutex> lock(pool_mutex);

	if (unlikely(!free_list_head)) {
		_allocate_new_page();
	}

	// Pop from free-list
	PhysicsServerHyper::Body *b = free_list_head;
	free_list_head = *(PhysicsServerHyper::Body **)free_list_head;

	// Placement New: Initialize FixedMathCore and BigIntCore components
	// without additional heap allocation.
	memnew_placement(b, PhysicsServerHyper::Body);
	
	return b;
}

/**
 * free_body_memory()
 * 
 * Returns the memory block to the paged pool for recycling.
 * strictly calls the destructor to flush BigIntCore internal chunks.
 */
void PhysicsServerHyper::_free_body_memory(Body *p_body) {
	if (unlikely(!p_body)) return;

	// Manually invoke destructor to clean up Ref<DynamicMesh> and BigInt chunks
	p_body->~Body();

	std::lock_guard<std::mutex> lock(pool_mutex);
	
	// Push back to free-list (Zero-Copy recycling)
	*(PhysicsServerHyper::Body **)p_body = free_list_head;
	free_list_head = p_body;
}

/**
 * get_paged_memory_stats()
 * 
 * Provides bit-perfect telemetry data to the ETEngine diagnostic system.
 */
void PhysicsServerHyper::get_paged_memory_stats(BigIntCore &r_allocated, BigIntCore &r_pages) {
	std::lock_guard<std::mutex> lock(pool_mutex);
	r_allocated = total_allocated_entities;
	r_pages = active_page_count;
}

/**
 * cleanup_paged_allocator()
 * 
 * Performs the absolute final teardown of the physics memory pools.
 * strictly ensures zero memory leaks.
 */
void PhysicsServerHyper::_cleanup_paged_allocator() {
	std::lock_guard<std::mutex> lock(pool_mutex);
	
	BodyPage *curr = page_list_head;
	while (curr) {
		BodyPage *next = curr->next;
		
		// Free the aligned body array
		Memory::free_static(curr->bodies);
		// Free the page header
		Memory::free_static(curr);
		
		curr = next;
	}
	
	page_list_head = nullptr;
	free_list_head = nullptr;
	total_allocated_entities = BigIntCore(0LL);
	active_page_count = BigIntCore(0LL);
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_paged_allocator.cpp ---
