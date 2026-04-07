--- START OF FILE core/io/resource_saver.cpp ---

#include "core/io/resource_saver.h"
#include "core/object/class_db.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

ResourceSaver *ResourceSaver::singleton = nullptr;

/**
 * _bind_methods()
 * 
 * Exposes the deterministic serialization API to Godot's reflection system.
 * Replaces standard integer flags with a system optimized for 
 * Zero-Copy mathematical storage.
 */
void ResourceSaver::_bind_methods() {
	ClassDB::bind_method(D_METHOD("save", "resource", "path", "flags"), &ResourceSaver::save, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("add_resource_format_saver", "format_saver", "at_front"), &ResourceSaver::add_resource_format_saver, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("remove_resource_format_saver", "format_saver"), &ResourceSaver::remove_resource_format_saver);
	ClassDB::bind_method(D_METHOD("get_recognized_extensions", "resource"), &ResourceSaver::_get_recognized_extensions_bind);

	BIND_ENUM_CONSTANT(FLAG_NONE);
	BIND_ENUM_CONSTANT(FLAG_RELATIVE_PATHS);
	BIND_ENUM_CONSTANT(FLAG_BUNDLE_RESOURCES);
	BIND_ENUM_CONSTANT(FLAG_CHANGE_PATH);
	BIND_ENUM_CONSTANT(FLAG_OMIT_EDITOR_PROPERTIES);
	BIND_ENUM_CONSTANT(FLAG_SAVE_BIGINT_AS_BINARY);
	BIND_ENUM_CONSTANT(FLAG_COMPRESS);
}

ResourceSaver::ResourceSaver() {
	singleton = this;
}

ResourceSaver::~ResourceSaver() {
	singleton = nullptr;
}

/**
 * save()
 * 
 * The primary entry point for persisting simulation data.
 * Iterates through registered savers to find a bit-perfect match for the resource type.
 * Uses BigIntCore to track large file sizes and FixedMathCore for I/O timing metrics.
 */
Error ResourceSaver::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	ERR_FAIL_COND_V_MSG(p_resource.is_null(), ERR_INVALID_PARAMETER, "Universal Solver Error: Cannot save null resource.");

	ResourceFormatSaver *saver = nullptr;
	for (const List<ResourceFormatSaver *>::Element *E = singleton->savers.front(); E; E = E->next()) {
		if (E->get()->recognize(p_resource)) {
			saver = E->get();
			break;
		}
	}

	if (!saver) {
		return ERR_FILE_UNRECOGNIZED;
	}

	Error err = saver->save(p_resource, p_path, p_flags);

	if (err == OK) {
		if (p_flags & FLAG_CHANGE_PATH) {
			p_resource->set_path(p_path);
		}
		
		// Update Resource metadata using BigIntCore for exact byte tracking
		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
		if (f.is_valid()) {
			p_resource->set_resource_size(BigIntCore(static_cast<int64_t>(f->get_length())));
		}
		
		p_resource->emit_changed();
	}

	return err;
}

void ResourceSaver::add_resource_format_saver(ResourceFormatSaver *p_format_saver, bool p_at_front) {
	if (p_at_front) {
		singleton->savers.push_front(p_format_saver);
	} else {
		singleton->savers.push_back(p_format_saver);
	}
}

void ResourceSaver::remove_resource_format_saver(ResourceFormatSaver *p_format_saver) {
	singleton->savers.erase(singleton->savers.find(p_format_saver));
}

void ResourceSaver::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) {
	for (const List<ResourceFormatSaver *>::Element *E = singleton->savers.front(); E; E = E->next()) {
		if (E->get()->recognize(p_resource)) {
			E->get()->get_recognized_extensions(p_resource, p_extensions);
		}
	}
}

void ResourceSaver::_get_recognized_extensions_bind(const Ref<Resource> &p_resource, Vector<String> *r_extensions) {
	List<String> extensions;
	get_recognized_extensions(p_resource, &extensions);
	for (const String &E : extensions) {
		r_extensions->push_back(E);
	}
}

--- END OF FILE core/io/resource_saver.cpp ---
