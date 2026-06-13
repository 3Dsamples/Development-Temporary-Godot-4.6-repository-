// File 255: modules/newton/src/collision/newton_collision_filter.h
// NewtonCollisionFilter – implements collision layers and masks for bodies.
// Each body can belong to up to 32 layers and collide with multiple masks.
// Filters are evaluated before narrow-phase GJK and can disable contacts.

#ifndef NEWTON_COLLISION_FILTER_H
#define NEWTON_COLLISION_FILTER_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "../core/newton_types.h"

namespace newton {

class NewtonCollisionFilter : public RefCounted {
	GDCLASS(NewtonCollisionFilter, RefCounted);

public:
	NewtonCollisionFilter() : default_layer(1), default_mask(0xFFFFFFFF) {}

	// Set the collision layer for a single body.  Layer is a 32‑bit mask.
	void set_body_layer(body_id p_id, uint32_t p_layer) {
		body_layers[p_id] = p_layer;
	}
	uint32_t get_body_layer(body_id p_id) const {
		HashMap<body_id, uint32_t>::ConstIterator it = body_layers.find(p_id);
		return it ? it->value : default_layer;
	}

	// Set the collision mask for a single body (which layers it can collide with).
	void set_body_mask(body_id p_id, uint32_t p_mask) {
		body_masks[p_id] = p_mask;
	}
	uint32_t get_body_mask(body_id p_id) const {
		HashMap<body_id, uint32_t>::ConstIterator it = body_masks.find(p_id);
		return it ? it->value : default_mask;
	}

	// Global settings for bodies not explicitly configured.
	void set_default_layer(uint32_t p_layer) { default_layer = p_layer; }
	uint32_t get_default_layer() const { return default_layer; }
	void set_default_mask(uint32_t p_mask) { default_mask = p_mask; }
	uint32_t get_default_mask() const { return default_mask; }

	// Return true if body `a` should collide with body `b`.
	// A collision is allowed if (layer_a & mask_b) != 0 and (layer_b & mask_a) != 0.
	bool can_collide(body_id a, body_id b) const {
		uint32_t layer_a = get_body_layer(a);
		uint32_t mask_a  = get_body_mask(a);
		uint32_t layer_b = get_body_layer(b);
		uint32_t mask_b  = get_body_mask(b);
		return (layer_a & mask_b) && (layer_b & mask_a);
	}

	// Explicitly disable collision between two specific bodies.
	void disable_pair(body_id a, body_id b) {
		if (a > b) SWAP(a, b);
		disabled_pairs.insert(std::make_pair(a, b));
	}
	void enable_pair(body_id a, body_id b) {
		if (a > b) SWAP(a, b);
		disabled_pairs.erase(std::make_pair(a, b));
	}
	bool is_pair_disabled(body_id a, body_id b) const {
		if (a > b) SWAP(a, b);
		return disabled_pairs.has(std::make_pair(a, b));
	}

	// Main filter call: return true if the pair should be sent to narrow-phase.
	bool filter_pair(body_id a, body_id b) const {
		if (is_pair_disabled(a, b)) return false;
		return can_collide(a, b);
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_body_layer", "id", "layer"), &NewtonCollisionFilter::set_body_layer);
		ClassDB::bind_method(D_METHOD("get_body_layer", "id"), &NewtonCollisionFilter::get_body_layer);
		ClassDB::bind_method(D_METHOD("set_body_mask", "id", "mask"), &NewtonCollisionFilter::set_body_mask);
		ClassDB::bind_method(D_METHOD("get_body_mask", "id"), &NewtonCollisionFilter::get_body_mask);
		ClassDB::bind_method(D_METHOD("set_default_layer", "layer"), &NewtonCollisionFilter::set_default_layer);
		ClassDB::bind_method(D_METHOD("get_default_layer"), &NewtonCollisionFilter::get_default_layer);
		ClassDB::bind_method(D_METHOD("set_default_mask", "mask"), &NewtonCollisionFilter::set_default_mask);
		ClassDB::bind_method(D_METHOD("get_default_mask"), &NewtonCollisionFilter::get_default_mask);
		ClassDB::bind_method(D_METHOD("disable_pair", "a", "b"), &NewtonCollisionFilter::disable_pair);
		ClassDB::bind_method(D_METHOD("enable_pair", "a", "b"), &NewtonCollisionFilter::enable_pair);
		ClassDB::bind_method(D_METHOD("is_pair_disabled", "a", "b"), &NewtonCollisionFilter::is_pair_disabled);
		ClassDB::bind_method(D_METHOD("filter_pair", "a", "b"), &NewtonCollisionFilter::filter_pair);
	}

private:
	HashMap<body_id, uint32_t> body_layers;
	HashMap<body_id, uint32_t> body_masks;
	uint32_t default_layer;
	uint32_t default_mask;
	HashSet<std::pair<body_id, body_id>> disabled_pairs;
};

} // namespace newton

#endif // NEWTON_COLLISION_FILTER_H