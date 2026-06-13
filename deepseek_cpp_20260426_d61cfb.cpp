// File 164: modules/genesis/src/materials/material_base.cpp
// Implementation of the base material class: property binding, clone,
// and the solver options interface.

#include "material_base.h"

#include "core/variant/variant.h"

namespace genesis {

void GenesisMaterial::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_density", "density"), &GenesisMaterial::set_density);
	ClassDB::bind_method(D_METHOD("get_density"), &GenesisMaterial::get_density);
	ClassDB::bind_method(D_METHOD("set_young_modulus", "young_modulus"), &GenesisMaterial::set_young_modulus);
	ClassDB::bind_method(D_METHOD("get_young_modulus"), &GenesisMaterial::get_young_modulus);
	ClassDB::bind_method(D_METHOD("set_poisson_ratio", "poisson_ratio"), &GenesisMaterial::set_poisson_ratio);
	ClassDB::bind_method(D_METHOD("get_poisson_ratio"), &GenesisMaterial::get_poisson_ratio);
	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &GenesisMaterial::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &GenesisMaterial::get_friction);
	ClassDB::bind_method(D_METHOD("set_restitution", "restitution"), &GenesisMaterial::set_restitution);
	ClassDB::bind_method(D_METHOD("get_restitution"), &GenesisMaterial::get_restitution);
	ClassDB::bind_method(D_METHOD("set_damping", "damping"), &GenesisMaterial::set_damping);
	ClassDB::bind_method(D_METHOD("get_damping"), &GenesisMaterial::get_damping);
	ClassDB::bind_method(D_METHOD("set_material_type", "type"), &GenesisMaterial::set_material_type);
	ClassDB::bind_method(D_METHOD("get_material_type"), &GenesisMaterial::get_material_type);
	ClassDB::bind_method(D_METHOD("get_lame_lambda"), &GenesisMaterial::get_lame_lambda);
	ClassDB::bind_method(D_METHOD("get_lame_mu"), &GenesisMaterial::get_lame_mu);
	ClassDB::bind_method(D_METHOD("get_bulk_modulus"), &GenesisMaterial::get_bulk_modulus);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "density", PROPERTY_HINT_RANGE, "0,100000,0.1"), "set_density", "get_density");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "young_modulus", PROPERTY_HINT_RANGE, "0,1e12,1"), "set_young_modulus", "get_young_modulus");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "poisson_ratio", PROPERTY_HINT_RANGE, "0,0.49,0.01"), "set_poisson_ratio", "get_poisson_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_friction", "get_friction");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "restitution", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_restitution", "get_restitution");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "damping", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_damping", "get_damping");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "material_type"), "set_material_type", "get_material_type");
	// Note: Lame parameters are read‑only → no setter properties.
	ClassDB::bind_method(D_METHOD("get_lame_lambda"), &GenesisMaterial::get_lame_lambda);
	ClassDB::bind_method(D_METHOD("get_lame_mu"), &GenesisMaterial::get_lame_mu);
	ClassDB::bind_method(D_METHOD("get_bulk_modulus"), &GenesisMaterial::get_bulk_modulus);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lame_lambda"), "", "get_lame_lambda");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "lame_mu"), "", "get_lame_mu");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bulk_modulus"), "", "get_bulk_modulus");
}

// The virtual duplicate method is implemented here for the base class.
Ref<GenesisMaterial> GenesisMaterial::duplicate(bool p_subresources) const {
	Ref<GenesisMaterial> mat = memnew(GenesisMaterial);
	mat->density = density;
	mat->young_modulus = young_modulus;
	mat->poisson_ratio = poisson_ratio;
	mat->friction = friction;
	mat->restitution = restitution;
	mat->damping = damping;
	mat->material_type = material_type;
	return mat;
}

} // namespace genesis