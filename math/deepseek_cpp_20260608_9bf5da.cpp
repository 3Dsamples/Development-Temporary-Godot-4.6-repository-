// File 165: modules/genesis/src/materials/material_library.cpp
// Implementation of MaterialLibrary – pre‑populated presets and lookup methods.

#include "material_library.h"

#include "material_base.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"

namespace genesis {

MaterialLibrary::MaterialLibrary() {
	create_defaults();
}

void MaterialLibrary::add_material(const String &p_name, const Material &p_mat) {
	library[p_name.to_lower()] = p_mat;
}

bool MaterialLibrary::get_material(const String &p_name, Material &r_mat) const {
	HashMap<String, Material>::ConstIterator it = library.find(p_name.to_lower());
	if (it) {
		r_mat = it->value;
		return true;
	}
	return false;
}

Material MaterialLibrary::get_material_or_default(const String &p_name, const Material &p_fallback) const {
	Material out;
	if (get_material(p_name, out)) return out;
	return p_fallback;
}

bool MaterialLibrary::has_material(const String &p_name) const {
	return library.has(p_name.to_lower());
}

void MaterialLibrary::remove_material(const String &p_name) {
	library.erase(p_name.to_lower());
}

List<String> MaterialLibrary::get_material_names() const {
	List<String> names;
	for (const KeyValue<String, Material> &kv : library) {
		names.push_back(kv.key);
	}
	return names;
}

void MaterialLibrary::reset() {
	library.clear();
	create_defaults();
}

void MaterialLibrary::create_defaults() {
	// Default (generic) material
	Material m_default;
	m_default.name = "default";
	m_default.density = 1000.0f;
	m_default.young_modulus = 1e6f;
	m_default.poisson_ratio = 0.3f;
	m_default.damping_coefficient = 0.0f;
	m_default.friction = 0.5f;
	m_default.restitution = 0.0f;
	add_material("default", m_default);

	// Rubber
	Material rubber;
	rubber.name = "rubber";
	rubber.density = 1200.0f;
	rubber.young_modulus = 1e5f;
	rubber.poisson_ratio = 0.47f;
	rubber.friction = 1.0f;
	rubber.restitution = 0.1f;
	rubber.damping_coefficient = 0.1f;
	add_material("rubber", rubber);

	// Steel
	Material steel;
	steel.name = "steel";
	steel.density = 7800.0f;
	steel.young_modulus = 2e11f;
	steel.poisson_ratio = 0.28f;
	steel.friction = 0.6f;
	steel.restitution = 0.0f;
	steel.damping_coefficient = 0.0f;
	add_material("steel", steel);

	// Cloth (soft)
	Material cloth;
	cloth.name = "cloth";
	cloth.density = 300.0f;
	cloth.young_modulus = 1e4f;
	cloth.poisson_ratio = 0.45f;
	cloth.friction = 0.5f;
	cloth.restitution = 0.0f;
	cloth.damping_coefficient = 0.2f;
	add_material("cloth", cloth);

	// Concrete
	Material concrete;
	concrete.name = "concrete";
	concrete.density = 2400.0f;
	concrete.young_modulus = 3e10f;
	concrete.poisson_ratio = 0.2f;
	concrete.friction = 0.7f;
	concrete.restitution = 0.0f;
	concrete.damping_coefficient = 0.0f;
	add_material("concrete", concrete);
}

} // namespace genesis