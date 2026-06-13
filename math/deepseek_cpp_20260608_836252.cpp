// File 248: modules/newton/src/materials/newton_material_pair.h
// MaterialPair – combines two materials (each with friction, restitution,
// softness) into a single set of contact properties.  Material pairs can be
// overridden per body pair and are used by the solver for contact resolution.

#ifndef NEWTON_MATERIALS_PAIR_H
#define NEWTON_MATERIALS_PAIR_H

#include "../core/newton_types.h"
#include "newton_material.h"

namespace newton {

class NewtonMaterialPair : public RefCounted {
	GDCLASS(NewtonMaterialPair, RefCounted);

public:
	NewtonMaterialPair() :
		combined_friction(0.5),
		combined_restitution(0.0),
		combined_softness(0.001),
		default_friction(0.5),
		default_restitution(0.0),
		default_softness(0.001) {}

	// Called by the world to set the default material that is used when
	// no per‑pair override exists.
	void set_default_material(const Ref<NewtonMaterial> &p_mat) {
		if (p_mat.is_valid()) {
			default_friction = p_mat->get_dynamic_friction();
			default_restitution = p_mat->get_restitution();
			default_softness = p_mat->get_softness();
		}
	}

	// Set the combined properties for a specific pair of material IDs.
	void set_pair(material_id a, material_id b,
				  real_t p_friction, real_t p_restitution, real_t p_softness) {
		uint64_t key = build_key(a, b);
		PairData pd;
		pd.friction = p_friction;
		pd.restitution = p_restitution;
		pd.softness = p_softness;
		pair_map[key] = pd;
	}

	// Remove a previously set pair override.
	void remove_pair(material_id a, material_id b) {
		pair_map.erase(build_key(a, b));
	}

	// Combine two material IDs into a single set of contact properties.
	// If an explicit pair is set, use it; otherwise, use the maximum of
	// the two materials' properties.
	void combine(material_id a, material_id b,
				 const HashMap<material_id, Ref<NewtonMaterial>> &materials,
				 real_t &r_friction, real_t &r_restitution, real_t &r_softness) {
		uint64_t key = build_key(a, b);
		HashMap<uint64_t, PairData>::ConstIterator it = pair_map.find(key);
		if (it) {
			r_friction = it->value.friction;
			r_restitution = it->value.restitution;
			r_softness = it->value.softness;
			return;
		}

		// Default combination: maximum of the two individual materials,
		// or the global default if no materials exist.
		if (materials.is_empty()) {
			r_friction = default_friction;
			r_restitution = default_restitution;
			r_softness = default_softness;
			return;
		}

		Ref<NewtonMaterial> matA = materials.has(a) ? materials[a] : Ref<NewtonMaterial>();
		Ref<NewtonMaterial> matB = materials.has(b) ? materials[b] : Ref<NewtonMaterial>();

		if (matA.is_valid() && matB.is_valid()) {
			r_friction = MAX(matA->get_dynamic_friction(), matB->get_dynamic_friction());
			r_restitution = MAX(matA->get_restitution(), matB->get_restitution());
			r_softness = MAX(matA->get_softness(), matB->get_softness());
		} else if (matA.is_valid()) {
			r_friction = matA->get_dynamic_friction();
			r_restitution = matA->get_restitution();
			r_softness = matA->get_softness();
		} else if (matB.is_valid()) {
			r_friction = matB->get_dynamic_friction();
			r_restitution = matB->get_restitution();
			r_softness = matB->get_softness();
		} else {
			r_friction = default_friction;
			r_restitution = default_restitution;
			r_softness = default_softness;
		}
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_default_material", "material"), &NewtonMaterialPair::set_default_material);
		ClassDB::bind_method(D_METHOD("set_pair", "mat_a", "mat_b", "friction", "restitution", "softness"), &NewtonMaterialPair::set_pair);
		ClassDB::bind_method(D_METHOD("remove_pair", "mat_a", "mat_b"), &NewtonMaterialPair::remove_pair);
		ClassDB::bind_method(D_METHOD("combine", "mat_a", "mat_b", "materials"), &NewtonMaterialPair::combine);
	}

private:
	static uint64_t build_key(material_id a, material_id b) {
		if (a > b) SWAP(a, b);
		return ((uint64_t)a << 32) | (uint64_t)b;
	}

	struct PairData {
		real_t friction;
		real_t restitution;
		real_t softness;
	};

	HashMap<uint64_t, PairData> pair_map;
	real_t combined_friction;      // latest combined result
	real_t combined_restitution;
	real_t combined_softness;
	real_t default_friction;
	real_t default_restitution;
	real_t default_softness;
};

} // namespace newton

#endif // NEWTON_MATERIALS_PAIR_H