// File 374: modules/integration/unified_warm_start_cache.h
// High‑performance thread‑safe warm‑starting cache for the unified physics
// pipeline.  Stores accumulated normal and tangential impulses per body pair
// across frames, enabling all engines (Newton, Vienna, Wicked, Genesis) to
// benefit from persistent contact data.  Uses a lock‑free open‑addressing
// hash table with atomic operations and a fixed ring buffer per pair to
// store multiple contact‑point impulses.  All hot‑path methods are inline
// for minimal overhead during contact solving.

#ifndef INTEGRATION_UNIFIED_WARM_START_CACHE_H
#define INTEGRATION_UNIFIED_WARM_START_CACHE_H

#include "core/typedefs.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include <atomic>

namespace unified {

// --------------------------------------------------------------------------
// Per‑contact impulse data stored for warm‑starting
// --------------------------------------------------------------------------
struct WarmStartImpulse {
	Vector3 point;            // contact point in world space (used to match across frames)
	real_t normal_impulse;    // accumulated normal impulse
	Vector3 tangent1_impulse; // tangential impulses (2D)
	Vector3 tangent2_impulse;
	uint32_t age;             // frames since last update (for eviction)
};

// --------------------------------------------------------------------------
// Per body‑pair data: fixed‑size ring buffer of contact impulses
// --------------------------------------------------------------------------
struct WarmStartPair {
	static constexpr int MAX_CONTACTS = 8;   // max contacts per manifold
	WarmStartImpulse impulses[MAX_CONTACTS];
	int count;
	uint32_t last_frame;                      // when this pair was last updated
	uint64_t key;                             // body pair key (a<<32 | b, a<b)
};

// --------------------------------------------------------------------------
// Lock‑free hash table based on open addressing with linear probing.
// The table size is a power of two for fast masking.
// --------------------------------------------------------------------------
class WarmStartCache {
public:
	static constexpr uint32_t DEFAULT_CAPACITY = 16384;   // 16K pairs
	static constexpr uint32_t MAX_AGE = 8;                // frames before eviction

private:
	struct Slot {
		std::atomic<uint64_t> key { 0 };      // 0 = empty
		WarmStartPair pair;                   // valid only when key != 0
	};

	LocalVector<Slot> table;
	uint32_t capacity_mask;
	uint32_t current_frame;

	// -----------------------------------------------------------------------
	// Hash function for 64‑bit key (body pair ID)
	// -----------------------------------------------------------------------
	inline uint32_t hash_key(uint64_t p_key) const {
		// Murmur3‑inspired mixing
		uint64_t k = p_key;
		k ^= k >> 33;
		k *= 0xff51afd7ed558ccdULL;
		k ^= k >> 33;
		k *= 0xc4ceb9fe1a85ec53ULL;
		k ^= k >> 33;
		return (uint32_t)(k & capacity_mask);
	}

public:
	WarmStartCache() : capacity_mask(0), current_frame(0) {
		resize(DEFAULT_CAPACITY);
	}

	// Resize the table (must be called before any physics step).
	void resize(uint32_t p_new_capacity) {
		// Round up to next power of two.
		uint32_t cap = 1;
		while (cap < p_new_capacity) cap <<= 1;
		table.resize(cap);
		capacity_mask = cap - 1;
		for (uint32_t i = 0; i < cap; ++i) {
			table[i].key.store(0, std::memory_order_relaxed);
		}
	}

	// Advance the frame counter (call once per physics step).
	void advance_frame() { current_frame++; }

	// -----------------------------------------------------------------------
	// Retrieve (and optionally update) warm‑starting impulses for a body pair.
	// `contacts` is an output buffer (size MAX_CONTACTS) that will be filled.
	// Returns the number of contacts that were found (<= MAX_CONTACTS).
	// -----------------------------------------------------------------------
	inline int get_impulses(uint64_t p_body_a, uint64_t p_body_b,
							WarmStartImpulse *contacts, int max_contacts) const {
		if (max_contacts <= 0) return 0;
		uint64_t key = build_key(p_body_a, p_body_b);
		uint32_t idx = hash_key(key);
		const Slot *slot = &table[idx];
		for (uint32_t probe = 0; probe < capacity_mask + 1; ++probe) {
			uint64_t k = slot->key.load(std::memory_order_acquire);
			if (k == 0) return 0;          // not found
			if (k == key) {
				const WarmStartPair &pair = slot->pair;
				if (pair.last_frame + MAX_AGE < current_frame) {
					// Expired; remove the entry? No, caller will overwrite.
					return 0;
				}
				int n = MIN(pair.count, max_contacts);
				for (int i = 0; i < n; ++i) contacts[i] = pair.impulses[i];
				return n;
			}
			idx = (idx + 1) & capacity_mask;
			slot = &table[idx];
		}
		return 0;  // table full or not found
	}

	// -----------------------------------------------------------------------
	// Store new impulses for a body pair (overwrites existing entry).
	// -----------------------------------------------------------------------
	inline void set_impulses(uint64_t p_body_a, uint64_t p_body_b,
							 const WarmStartImpulse *p_impulses, int p_count) {
		if (p_count <= 0 || p_count > WarmStartPair::MAX_CONTACTS) return;
		uint64_t key = build_key(p_body_a, p_body_b);
		uint32_t idx = hash_key(key);
		Slot *slot = &table[idx];
		for (uint32_t probe = 0; probe < capacity_mask + 1; ++probe) {
			uint64_t expected = 0;
			// Try to claim an empty slot or overwrite an existing key.
			if (slot->key.compare_exchange_strong(expected, key,
				std::memory_order_acq_rel, std::memory_order_acquire)) {
				// We claimed an empty slot; fill the pair.
				fill_pair(slot->pair, current_frame, key, p_impulses, p_count);
				return;
			}
			if (expected == key) {
				// Already exists; update.
				fill_pair(slot->pair, current_frame, key, p_impulses, p_count);
				return;
			}
			idx = (idx + 1) & capacity_mask;
			slot = &table[idx];
		}
		// Table full – not expected; we could evict oldest or resize.
	}

	// -----------------------------------------------------------------------
	// Clear all entries (e.g., when the world is reset).
	// -----------------------------------------------------------------------
	void clear() {
		for (uint32_t i = 0; i < table.size(); ++i) {
			table[i].key.store(0, std::memory_order_relaxed);
		}
		current_frame = 0;
	}

private:
	inline static uint64_t build_key(uint64_t a, uint64_t b) {
		if (a > b) { uint64_t tmp = a; a = b; b = tmp; }
		return (a << 32) | b;
	}

	inline void fill_pair(WarmStartPair &pair, uint32_t frame, uint64_t key,
						  const WarmStartImpulse *imp, int count) {
		pair.count = MIN(count, WarmStartPair::MAX_CONTACTS);
		for (int i = 0; i < pair.count; ++i) {
			pair.impulses[i] = imp[i];
		}
		pair.last_frame = frame;
		pair.key = key;
	}
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_WARM_START_CACHE_H