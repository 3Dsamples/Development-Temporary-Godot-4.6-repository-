// File 136: modules/genesis/src/states/cache.h
// Solver state cache: stores a ring buffer of SolverState snapshots
// indexed by simulation time. Used for checkpointing, warm‑starting,
// rolling back, and computing finite‑difference gradients.

#ifndef GENESIS_STATES_CACHE_H
#define GENESIS_STATES_CACHE_H

#include "core/io/resource.h"
#include "solver_state.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace genesis {

class SolverStateCache : public Resource {
	GDCLASS(SolverStateCache, Resource);

public:
	SolverStateCache() : capacity(100), oldest_index(0), count(0) {}

	// Maximum number of snapshots kept.
	void set_capacity(int p_cap) {
		capacity = MAX(p_cap, 1);
		resize_buffer();
	}
	int get_capacity() const { return capacity; }

	// Store a snapshot at the current simulation time.
	void push(const Ref<SolverState> &p_state) {
		ERR_FAIL_COND(p_state.is_null());
		if (count < capacity) {
			buffer.push_back(p_state);
			count++;
		} else {
			buffer[oldest_index] = p_state;
			oldest_index = (oldest_index + 1) % capacity;
		}
	}

	// Get the most recent snapshot.
	Ref<SolverState> latest() const {
		if (count == 0) return Ref<SolverState>();
		int idx = (oldest_index + count - 1) % buffer.size();
		return buffer[idx];
	}

	// Get snapshot by its simulation time (closest match within tolerance).
	Ref<SolverState> get_at_time(real_t p_time, real_t p_tolerance = 1e-6) const {
		int best = -1;
		real_t best_diff = INFINITY;
		for (int i = 0; i < count; ++i) {
			int idx = (oldest_index + i) % buffer.size();
			real_t t = buffer[idx]->get_time();
			real_t diff = Math::abs(t - p_time);
			if (diff < best_diff && diff <= p_tolerance) {
				best_diff = diff;
				best = idx;
			}
		}
		return (best >= 0) ? buffer[best] : Ref<SolverState>();
	}

	// Return all stored times in chronological order.
	PackedFloat64Array get_times() const {
		PackedFloat64Array times;
		times.resize(count);
		for (int i = 0; i < count; ++i) {
			int idx = (oldest_index + i) % buffer.size();
			times.set(i, buffer[idx]->get_time());
		}
		return times;
	}

	// Clear all snapshots.
	void clear() {
		buffer.clear();
		oldest_index = 0;
		count = 0;
	}

	// Number of stored snapshots.
	int get_count() const { return count; }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_capacity", "cap"), &SolverStateCache::set_capacity);
		ClassDB::bind_method(D_METHOD("get_capacity"), &SolverStateCache::get_capacity);
		ClassDB::bind_method(D_METHOD("push", "state"), &SolverStateCache::push);
		ClassDB::bind_method(D_METHOD("latest"), &SolverStateCache::latest);
		ClassDB::bind_method(D_METHOD("get_at_time", "time", "tolerance"), &SolverStateCache::get_at_time, DEFVAL(1e-6));
		ClassDB::bind_method(D_METHOD("get_times"), &SolverStateCache::get_times);
		ClassDB::bind_method(D_METHOD("clear"), &SolverStateCache::clear);
		ClassDB::bind_method(D_METHOD("get_count"), &SolverStateCache::get_count);
		ADD_PROPERTY(PropertyInfo(Variant::INT, "capacity"), "set_capacity", "get_capacity");
	}

private:
	void resize_buffer() {
		// If capacity changed, discard overflow from oldest.
		if (buffer.size() == capacity) return;
		LocalVector<Ref<SolverState>> new_buf;
		new_buf.resize(capacity);
		int to_copy = MIN(count, capacity);
		for (int i = 0; i < to_copy; ++i) {
			int old_idx = (oldest_index + (count - to_copy) + i) % buffer.size();
			new_buf[i] = buffer[old_idx];
		}
		buffer = new_buf;
		oldest_index = 0;
		count = to_copy;
	}

	int capacity;
	int oldest_index;
	int count;
	LocalVector<Ref<SolverState>> buffer;
};

} // namespace genesis

#endif // GENESIS_STATES_CACHE_H