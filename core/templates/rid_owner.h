--- START OF FILE core/templates/rid_owner.h ---

#ifndef RID_OWNER_H
#define RID_OWNER_H

#include "core/templates/rid.h"
#include "core/templates/hash_map.h"
#include "core/os/memory.h"
#include "src/big_int_core.h"
#include <mutex>

/**
 * RID_Owner
 * 
 * Manages the lifecycle and mapping of resources identified by RIDs.
 * Uses BigIntCore to support an infinite number of resource handles.
 * Thread-safe implementation to allow concurrent access from parallel Warp kernels.
 */
template <typename T>
class RID_Owner : public RID_Owner_Base {
	// Internal mapping of BigInt IDs to resource pointers
	HashMap<BigIntCore, T*> id_map;
	BigIntCore last_id;
	mutable std::mutex map_mutex;

public:
	/**
	 * make_rid()
	 * Creates a new unique RID and registers the associated resource.
	 * Atomic ID increment ensures determinism in multi-threaded environments.
	 */
	RID make_rid(T *p_ptr) {
		std::lock_guard<std::mutex> lock(map_mutex);
		last_id += BigIntCore(1LL);
		BigIntCore new_id = last_id;
		id_map.insert(new_id, p_ptr);
		return RID(new_id);
	}

	/**
	 * get_or_null()
	 * Retrieves the resource pointer from an RID handle in O(1).
	 * Vital for zero-copy access inside high-frequency math loops.
	 */
	_FORCE_INLINE_ T *get_or_null(const RID &p_rid) {
		if (unlikely(!p_rid.is_valid())) {
			return nullptr;
		}

		std::lock_guard<std::mutex> lock(map_mutex);
		if (id_map.has(p_rid.get_id())) {
			return id_map[p_rid.get_id()];
		}
		return nullptr;
	}

	/**
	 * free()
	 * Unregisters the RID and removes the resource from the database.
	 */
	virtual void free(RID p_rid) override {
		if (!p_rid.is_valid()) {
			return;
		}

		std::lock_guard<std::mutex> lock(map_mutex);
		id_map.erase(p_rid.get_id());
	}

	/**
	 * owns()
	 * Returns true if the RID is currently registered in this owner.
	 */
	virtual bool owns(RID p_rid) const override {
		if (!p_rid.is_valid()) {
			return false;
		}

		std::lock_guard<std::mutex> lock(map_mutex);
		return id_map.has(p_rid.get_id());
	}

	_FORCE_INLINE_ uint64_t get_rid_count() const {
		std::lock_guard<std::mutex> lock(map_mutex);
		return static_cast<uint64_t>(std::stoll(id_map.size().to_string()));
	}

	void clear() {
		std::lock_guard<std::mutex> lock(map_mutex);
		id_map.clear();
	}

	RID_Owner() : last_id(0LL) {}
	virtual ~RID_Owner() { clear(); }
};

#endif // RID_OWNER_H

--- END OF FILE core/templates/rid_owner.h ---
