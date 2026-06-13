// File 288: modules/vienna/src/materials/vienna_material.h
// ViennaMaterial – defines per‑material collision properties (friction,
// restitution, softness).  Materials are assigned to bodies and combined
// by the solver when two bodies collide.

#ifndef VIENNA_MATERIALS_VIENNA_MATERIAL_H
#define VIENNA_MATERIALS_VIENNA_MATERIAL_H

#include "core/object/ref_counted.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"

namespace vienna {

class ViennaMaterial : public RefCounted {
	GDCLASS(ViennaMaterial, RefCounted);

public:
	ViennaMaterial() :
		static_friction(0.5),
		dynamic_friction(0.3),
		restitution(0.0),
		softness(0.001),
		contact_skin_thickness(0.001) {}

	void set_static_friction(real_t p_val) { static_friction = CLAMP(p_val, 0.0, 10.0); }
	real_t get_static_friction() const { return static_friction; }

	void set_dynamic_friction(real_t p_val) { dynamic_friction = CLAMP(p_val, 0.0, static_friction); }
	real_t get_dynamic_friction() const { return dynamic_friction; }

	void set_restitution(real_t p_val) { restitution = CLAMP(p_val, 0.0, 1.0); }
	real_t get_restitution() const { return restitution; }

	void set_softness(real_t p_val) { softness = MAX(p_val, 0.0); }
	real_t get_softness() const { return softness; }

	void set_contact_skin_thickness(real_t p_val) { contact_skin_thickness = MAX(p_val, 0.0); }
	real_t get_contact_skin_thickness() const { return contact_skin_thickness; }

	// Combine two materials (caller passes material references from the world).
	// The result uses the maximum of the two materials for friction and restitution,
	// and the sum of softness values.
	static void combine(const ViennaMaterial *p_matA, const ViennaMaterial *p_matB,
						real_t &r_friction, real_t &r_restitution, real_t &r_softness) {
		if (p_matA && p_matB) {
			r_friction = MAX(p_matA->get_dynamic_friction(), p_matB->get_dynamic_friction());
			r_restitution = MAX(p_matA->get_restitution(), p_matB->get_restitution());
			r_softness = p_matA->get_softness() + p_matB->get_softness();
		} else if (p_matA) {
			r_friction = p_matA->get_dynamic_friction();
			r_restitution = p_matA->get_restitution();
			r_softness = p_matA->get_softness();
		} else if (p_matB) {
			r_friction = p_matB->get_dynamic_friction();
			r_restitution = p_matB->get_restitution();
			r_softness = p_matB->get_softness();
		} else {
			r_friction = DEFAULT_FRICTION;
			r_restitution = DEFAULT_RESTITUTION;
			r_softness = DEFAULT_SOFTNESS;
		}
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_static_friction", "friction"), &ViennaMaterial::set_static_friction);
		ClassDB::bind_method(D_METHOD("get_static_friction"), &ViennaMaterial::get_static_friction);
		ClassDB::bind_method(D_METHOD("set_dynamic_friction", "friction"), &ViennaMaterial::set_dynamic_friction);
		ClassDB::bind_method(D_METHOD("get_dynamic_friction"), &ViennaMaterial::get_dynamic_friction);
		ClassDB::bind_method(D_METHOD("set_restitution", "restitution"), &ViennaMaterial::set_restitution);
		ClassDB::bind_method(D_METHOD("get_restitution"), &ViennaMaterial::get_restitution);
		ClassDB::bind_method(D_METHOD("set_softness", "softness"), &ViennaMaterial::set_softness);
		ClassDB::bind_method(D_METHOD("get_softness"), &ViennaMaterial::get_softness);
		ClassDB::bind_method(D_METHOD("set_contact_skin_thickness", "thickness"), &ViennaMaterial::set_contact_skin_thickness);
		ClassDB::bind_method(D_METHOD("get_contact_skin_thickness"), &ViennaMaterial::get_contact_skin_thickness);

		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "static_friction", PROPERTY_HINT_RANGE, "0,10,0.01"), "set_static_friction", "get_static_friction");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dynamic_friction", PROPERTY_HINT_RANGE, "0,10,0.01"), "set_dynamic_friction", "get_dynamic_friction");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "restitution", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_restitution", "get_restitution");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "softness", PROPERTY_HINT_RANGE, "0,1,0.0001"), "set_softness", "get_softness");
		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "contact_skin_thickness", PROPERTY_HINT_RANGE, "0,0.1,0.0001"), "set_contact_skin_thickness", "get_contact_skin_thickness");
	}

	real_t static_friction;
	real_t dynamic_friction;
	real_t restitution;
	real_t softness;
	real_t contact_skin_thickness;
};

} // namespace vienna

#endif // VIENNA_MATERIALS_VIENNA_MATERIAL_H