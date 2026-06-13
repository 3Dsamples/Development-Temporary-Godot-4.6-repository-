// File 234: modules/newton/src/contacts/newton_contact_report.h
// NewtonContactReport – user‑callback system for collision events.
// Allows registering a callback for a specific pair of bodies (or all bodies)
// that fires after each physics step with the contact manifold information.

#ifndef NEWTON_CONTACTS_REPORT_H
#define NEWTON_CONTACTS_REPORT_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"
#include "../collision/newton_contact.h"

namespace newton {

class NewtonContactReport : public RefCounted {
	GDCLASS(NewtonContactReport, RefCounted);

public:
	// Callback type: receives the two body IDs and the list of contact points for the pair.
	typedef void (*ContactCallback)(body_id, body_id, const LocalVector<NewtonContactPoint> &, void *);

	NewtonContactReport() {}

	// Register a callback for a specific body pair.  If pair_already_registered, it will be updated.
	void register_pair(body_id p_body_a, body_id p_body_b, ContactCallback p_callback, void *p_userdata = nullptr) {
		uint64_t key = build_key(p_body_a, p_body_b);
		PairCallback pc;
		pc.callback = p_callback;
		pc.userdata = p_userdata;
		pair_callbacks[key] = pc;
	}

	// Register a global callback that fires for every contact pair.
	void register_global_callback(ContactCallback p_callback, void *p_userdata = nullptr) {
		GlobalCallback gc;
		gc.callback = p_callback;
		gc.userdata = p_userdata;
		global_callbacks.push_back(gc);
	}

	// Remove a specific pair callback.
	void unregister_pair(body_id p_body_a, body_id p_body_b) {
		uint64_t key = build_key(p_body_a, p_body_b);
		pair_callbacks.erase(key);
	}

	// Remove a global callback (identified by pointer and userdata).
	void unregister_global_callback(ContactCallback p_callback, void *p_userdata) {
		for (int i = global_callbacks.size() - 1; i >= 0; --i) {
			if (global_callbacks[i].callback == p_callback && global_callbacks[i].userdata == p_userdata) {
				global_callbacks.remove_at(i);
			}
		}
	}

	// Called internally by the world after solving, to notify listeners.
	void report_contacts(const LocalVector<NewtonContactPoint> &all_contacts) {
		// Group contacts by pair (body_a, body_b)
		HashMap<uint64_t, LocalVector<NewtonContactPoint>> grouped;
		for (const NewtonContactPoint &cp : all_contacts) {
			uint64_t key = build_key(cp.body_a, cp.body_b);
			grouped[key].push_back(cp);
		}

		// Notify per‑pair callbacks.
		for (const KeyValue<uint64_t, PairCallback> &kv : pair_callbacks) {
			const PairCallback &pc = kv.value;
			if (pc.callback) {
				// Extract the two IDs from the key.
				body_id a = (body_id)(kv.key >> 32);
				body_id b = (body_id)(kv.key & 0xFFFFFFFFull);
				if (grouped.has(kv.key)) {
					pc.callback(a, b, grouped[kv.key], pc.userdata);
				}
			}
		}

		// Notify global callbacks (all contacts).
		for (const GlobalCallback &gc : global_callbacks) {
			if (gc.callback) {
				// Send the raw all_contacts; the user must filter by body IDs.
				gc.callback(0, 0, all_contacts, gc.userdata);
			}
		}
	}

private:
	// Build a 64‑bit key from two body IDs (order independent).
	static uint64_t build_key(body_id a, body_id b) {
		if (a > b) SWAP(a, b);
		return ((uint64_t)a << 32) | (uint64_t)b;
	}

	struct PairCallback {
		ContactCallback callback = nullptr;
		void *userdata = nullptr;
	};

	struct GlobalCallback {
		ContactCallback callback = nullptr;
		void *userdata = nullptr;
	};

	HashMap<uint64_t, PairCallback> pair_callbacks;
	LocalVector<GlobalCallback> global_callbacks;
};

} // namespace newton

#endif // NEWTON_CONTACTS_REPORT_H