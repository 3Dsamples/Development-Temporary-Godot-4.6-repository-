--- START OF FILE core/io/resource.cpp ---

#include "core/io/resource.h"
#include "core/object/class_db.h"

/**
 * _bind_methods()
 * 
 * Exposes the deterministic resource API to Godot.
 * Natively supports BigIntCore for resource size metadata.
 */
void Resource::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_path", "path"), &Resource::set_path);
	ClassDB::bind_method(D_METHOD("get_path"), &Resource::get_path);
	ClassDB::bind_method(D_METHOD("set_name", "name"), &Resource::set_name);
	ClassDB::bind_method(D_METHOD("get_name"), &Resource::get_name);
	
	ClassDB::bind_method(D_METHOD("set_resource_size", "size"), &Resource::set_resource_size);
	ClassDB::bind_method(D_METHOD("get_resource_size"), &Resource::get_resource_size);
	
	ClassDB::bind_method(D_METHOD("set_deterministic", "enabled"), &Resource::set_deterministic);
	ClassDB::bind_method(D_METHOD("get_deterministic"), &Resource::get_deterministic);
	
	ClassDB::bind_method(D_METHOD("duplicate", "subresources"), &Resource::duplicate, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("emit_changed"), &Resource::emit_changed);

	ADD_SIGNAL(MethodInfo("changed"));
}

void Resource::set_path(const String &p_path) {
	path_cache = p_path;
	_resource_path_changed();
}

String Resource::get_path() const {
	return path_cache;
}

void Resource::set_name(const String &p_name) {
	name_cache = p_name;
	emit_changed();
}

String Resource::get_name() const {
	return name_cache;
}

void Resource::_resource_path_changed() {
	// Virtual stub for language-specific resource path updates
}

void Resource::set_resource_size(const BigIntCore &p_size) {
	resource_size = p_size;
}

BigIntCore Resource::get_resource_size() const {
	return resource_size;
}

void Resource::set_deterministic(bool p_enabled) {
	is_deterministic = p_enabled;
}

bool Resource::get_deterministic() const {
	return is_deterministic;
}

/**
 * duplicate()
 * 
 * High-performance resource cloning.
 * Optimized for EnTT-based entity templates where thousands of 
 * deterministic resources are duplicated per frame for Warp kernels.
 */
Ref<Resource> Resource::duplicate(bool p_subresources) const {
	StringName res_class = get_class();
	Resource *res = static_cast<Resource *>(ClassDB::instantiate(res_class));
	ERR_FAIL_NULL_V(res, Ref<Resource>());

	res->set_name(get_name());
	res->set_resource_size(get_resource_size());
	res->set_deterministic(get_deterministic());

	return Ref<Resource>(res);
}

void Resource::emit_changed() {
	emit_signalp(SNAME("changed"), nullptr, 0);
}

Resource::Resource() {
	resource_size = BigIntCore(0LL);
	is_deterministic = true;
}

Resource::~Resource() {
}

--- END OF FILE core/io/resource.cpp ---
