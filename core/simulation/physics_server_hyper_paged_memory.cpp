--- START OF FILE core/simulation/physics_server_hyper_paged_memory.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/os/memory.h"
#include "src/big_int_core.h"
#include <mutex>

namespace UniversalSolver {

/**
 * PagedMemoryPool
 * 
 * Internal structure to manage Body components in the EnTT Sparse-Set backend.
 * Uses BigIntCore to index pages and track total allocated capacity.
 * Engineered for cache-line alignment to maximize Warp kernel throughput.
 */
struct BodyPage {
	static const uint32_t BODIES_PER_PAGE = 1024;
	PhysicsServerHyper::Body *data;
	BodyPage *next;

	BodyPage() : data(nullptr), next(nullptr) {}
};

static BodyPage *page_list_head = nullptr;
static PhysicsServerHyper::Body *free_list_head = nullptr;
static std::mutex pool_mutex;

static BigIntCore total_allocated_bodies = BigIntCore(0LL);
static BigIntCore active_page_count = BigIntCore(0LL);

/**
 * _allocate_new_page()
 * 
 * Reserves a 32-byte aligned contiguous block of memory for 1024 bodies.
 * Uses BigIntCore to update the global memory telemetry for ETEngine.
 */
static void _allocate_new_page() {
	BodyPage *new_page = (BodyPage *)Memory::alloc_static(sizeof(BodyPage), false);
	new_page->data = (PhysicsServerHyper::Body *)Memory::alloc_static(sizeof(PhysicsServerHyper::Body) * BodyPage::BODIES_PER_PAGE, true);
	new_page->next = page_list_head;
	page_list_head = new_page;

	// Link all bodies in the new page into the free list
	for (uint32_t i = 0; i < BodyPage::BODIES_PER_PAGE; i++) {
		PhysicsServerHyper::Body *b = &new_page->data[i];
		// We use the 'next' pointer or similar casting logic to maintain the free list
		// In a zero-copy system, we cast the body memory to a pointer-link
		*(PhysicsServerHyper::Body **)b = free_list_head;
		free_list_head = b;
	}

	active_page_count += BigIntCore(1LL);
	total_allocated_bodies += BigIntCore(static_cast<int64_t>(BodyPage::BODIES_PER_PAGE));
}

/**
 * allocate_body_memory()
 * 
 * Returns a pointer to a clean Body structure.
 * Guaranteed O(1) unless a new page allocation is triggered.
 */
PhysicsServerHyper::Body *PhysicsServerHyper::_allocate_body_memory() {
	std::lock_guard<std::mutex> lock(pool_mutex);

	if (unlikely(!free_list_head)) {
		_allocate_new_page();
	}

	PhysicsServerHyper::Body *b = free_list_head;
	free_list_head = *(PhysicsServerHyper::Body **)free_list_head;

	// Placement new to initialize FixedMathCore and BigIntCore components
	memnew_placement(b, Body);
	
	return b;
}

/**
 * free_body_memory()
 * 
 * Returns the body to the paged pool.
 * Destructs simulation tensors before re-linking to the free list.
 */
void PhysicsServerHyper::_free_body_memory(Body *p_body) {
	if (unlikely(!p_body)) return;

	// Call destructor manually to flush BigIntCore chunks
	p_body->~Body();

	std::lock_guard<std::mutex> lock(pool_mutex);
	*(PhysicsServerHyper::Body **)p_body = free_list_head;
	free_list_head = p_body;
}

/**
 * get_paged_memory_stats()
 * 
 * ETEngine Telemetry: Provides BigIntCore metrics for memory pressure.
 */
void PhysicsServerHyper::get_paged_memory_stats(BigIntCore &r_allocated, BigIntCore &r_pages) {
	std::lock_guard<std::mutex> lock(pool_mutex);
	r_allocated = total_allocated_bodies;
	r_pages = active_page_count;
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_paged_memory.cpp ---
