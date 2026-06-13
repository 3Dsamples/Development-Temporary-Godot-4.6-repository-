// File 186: modules/newton/src/materials/newton_material.h
// NewtonMaterial – defines per‑material‑pair contact properties:
// friction, restitution, softness (spring‑damper coefficient), and
// callbacks for custom contact behaviour.
// Materials are mapped to body pairs during collision.

#ifndef NEWTON_MATERIALS_NEWTON_MATERIAL_H
#define NEWTON_MATERIALS_NEWTON_MATERIAL_H

#include "core/object/ref_counted.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"

namespace newton {

class NewtonMaterial : public RefCounted {
	GDCLASS(NewtonMaterial, RefCounted);

public:
	NewtonMaterial() :
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

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_static_friction", "friction"), &NewtonMaterial::set_static_friction);
		ClassDB::bind_method(D_METHOD("get_static_friction"), &NewtonMaterial::get_static_friction);
		ClassDB::bind_method(D_METHOD("set_dynamic_friction", "friction"), &NewtonMaterial::set_dynamic_friction);
		ClassDB::bind_method(D_METHOD("get_dynamic_friction"), &NewtonMaterial::get_dynamic_friction);
		ClassDB::bind_method(D_METHOD("set_restitution", "restitution"), &NewtonMaterial::set_restitution);
		ClassDB::bind_method(D_METHOD("get_restitution"), &NewtonMaterial::get_restitution);
		ClassDB::bind_method(D_METHOD("set_softness", "softness"), &NewtonMaterial::set_softness);
		ClassDB::bind_method(D_METHOD("get_softness"), &NewtonMaterial::get_softness);
		ClassDB::bind_method(D_METHOD("set_contact_skin_thickness", "thickness"), &NewtonMaterial::set_contact_skin_thickness);
		ClassDB::bind_method(D_METHOD("get_contact_skin_thickness"), &NewtonMaterial::get_contact_skin_thickness);

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

} // namespace newton

#endif // NEWTON_MATERIALS_NEWTON_MATERIAL_H