// File 238: modules/newton/src/contacts/newton_contact_modifier.h
// User‑defined callback for filtering / altering contact points before
// they are solved.  This is called once per overlapping pair after
// narrow‑phase GJK but before the sequential‑impulse solver.
// It allows per‑contact friction, restitution, softness, and enables
// contact disabling.

#ifndef NEWTON_CONTACTS_MODIFIER_H
#define NEWTON_CONTACTS_MODIFIER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "../core/newton_types.h"
#include "../collision/newton_contact.h"

namespace newton {

class NewtonContactModifier : public RefCounted {
	GDCLASS(NewtonContactModifier, RefCounted);

public:
	// Signature of a callback that receives the two body IDs and the
	// mutable contact manifold for that pair.  The callback can modify
	// any field (normal, penetration, friction, restitution) or even
	// remove contacts from the list.
	typedef void (*ModifyContactsCallback)(body_id, body_id, LocalVector<NewtonContactPoint> &, void *);

	NewtonContactModifier() {}

	// Register a per‑pair modifier.
	void register_pair(body_id a, body_id b, ModifyContactsCallback p_callback, void *p_userdata = nullptr) {
		uint64_t key = build_pair_key(a, b);
		PairMod mod;
		mod.callback = p_callback;
		mod.userdata = p_userdata;
		pair_modifiers[key] = mod;
	}

	// Register a global modifier that fires for every contact pair.
	void register_global(ModifyContactsCallback p_callback, void *p_userdata = nullptr) {
		GlobalMod gm;
		gm.callback = p_callback;
		gm.userdata = p_userdata;
		global_mods.push_back(gm);
	}

	// Remove a per‑pair modifier.
	void unregister_pair(body_id a, body_id b) {
		pair_modifiers.erase(build_pair_key(a, b));
	}

	// Remove a global modifier identified by its callback pointer and userdata.
	void unregister_global(ModifyContactsCallback p_callback, void *p_userdata) {
		for (int i = global_mods.size() - 1; i >= 0; --i) {
			if (global_mods[i].callback == p_callback && global_mods[i].userdata == p_userdata) {
				global_mods.remove_at(i);
			}
		}
	}

	// Called internally by the world after narrow‑phase and before solving.
	// Iterates over all contact pairs and invokes registered callbacks.
	void modify_contacts(LocalVector<NewtonContactPoint> &all_contacts) {
		// Group contacts by pair key.
		HashMap<uint64_t, LocalVector<NewtonContactPoint>> grouped;
		for (const NewtonContactPoint &cp : all_contacts) {
			uint64_t key = build_pair_key(cp.body_a, cp.body_b);
			grouped[key].push_back(cp);
		}

		// Apply per‑pair modifiers.
		for (KeyValue<uint64_t, LocalVector<NewtonContactPoint>> &kv : grouped) {
			uint64_t key = kv.key;
			if (pair_modifiers.has(key)) {
				const PairMod &mod = pair_modifiers[key];
				body_id a = (body_id)(key >> 32);
				body_id b = (body_id)(key & 0xFFFFFFFFull);
				mod.callback(a, b, kv.value, mod.userdata);
			}
			// Apply global modifiers to each pair's manifold.
			for (const GlobalMod &gm : global_mods) {
				body_id a = (body_id)(kv.key >> 32);
				body_id b = (body_id)(kv.key & 0xFFFFFFFFull);
				gm.callback(a, b, kv.value, gm.userdata);
			}
		}

		// Rebuild the all_contacts list from the modified grouped vectors.
		all_contacts.clear();
		for (const KeyValue<uint64_t, LocalVector<NewtonContactPoint>> &kv : grouped) {
			for (const NewtonContactPoint &cp : kv.value) {
				all_contacts.push_back(cp);
			}
		}
	}

private:
	static uint64_t build_pair_key(body_id a, body_id b) {
		if (a > b) SWAP(a, b);
		return ((uint64_t)a << 32) | (uint64_t)b;
	}

	struct PairMod {
		ModifyContactsCallback callback = nullptr;
		void *userdata = nullptr;
	};
	struct GlobalMod {
		ModifyContactsCallback callback = nullptr;
		void *userdata = nullptr;
	};

	HashMap<uint64_t, PairMod> pair_modifiers;
	LocalVector<GlobalMod> global_mods;
};

} // namespace newton

#endif // NEWTON_CONTACTS_MODIFIER_H