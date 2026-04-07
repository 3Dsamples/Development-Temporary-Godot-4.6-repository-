--- START OF FILE core/io/resource_loader.cpp ---

#include "core/io/resource_loader.h"
#include "core/object/class_db.h"
#include "core/io/file_access.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/os/os.h"

ResourceLoader *ResourceLoader::singleton = nullptr;

void ResourceLoader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load", "path", "type_hint", "no_cache"), &ResourceLoader::load, DEFVAL(""), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("load_threaded_request", "path", "type_hint", "use_subthreads"), &ResourceLoader::load_threaded_request, DEFVAL(""), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("load_threaded_get_status", "path"), &ResourceLoader::load_threaded_get_status);
	ClassDB::bind_method(D_METHOD("has_cached", "path"), &ResourceLoader::has_cached);
	ClassDB::bind_method(D_METHOD("get_resource_size", "path"), &ResourceLoader::get_resource_size);
}

ResourceLoader::ResourceLoader() {
	singleton = this;
}

ResourceLoader::~ResourceLoader() {
	singleton = nullptr;
}

Ref<Resource> ResourceLoader::load(const String &p_path, const String &p_type_hint, bool p_no_cache, Error *r_error) {
	if (!p_no_cache && singleton->resource_cache.has(p_path)) {
		return singleton->resource_cache[p_path];
	}

	String ext = p_path.get_extension().to_lower();
	ResourceFormatLoader *found_loader = nullptr;

	for (const List<ResourceFormatLoader *>::Element *E = singleton->loaders.front(); E; E = E->next()) {
		List<String> extensions;
		E->get()->get_recognized_extensions(&extensions);
		if (extensions.find(ext)) {
			found_loader = E->get();
			break;
		}
	}

	if (!found_loader) {
		if (r_error) *r_error = ERR_FILE_UNRECOGNIZED;
		return Ref<Resource>();
	}

	Ref<Resource> res = found_loader->load(p_path, p_path, r_error);

	if (res.is_valid() && !p_no_cache) {
		res->set_path(p_path);
		singleton->resource_cache[p_path] = res;
	}

	return res;
}

Error ResourceLoader::load_threaded_request(const String &p_path, const String &p_type_hint, bool p_use_subthreads) {
	// Dispatching to the Universal Solver's SimulationThreadPool
	// This ensures background loading doesn't stall the 120 FPS physics heartbeat
	SimulationThreadPool::get_singleton()->enqueue_task([p_path, p_type_hint]() {
		Error err;
		load(p_path, p_type_hint, false, &err);
	}, SimulationThreadPool::PRIORITY_LOW);

	return OK;
}

FixedMathCore ResourceLoader::load_threaded_get_status(const String &p_path, Error *r_error) {
	if (singleton->resource_cache.has(p_path)) {
		return FixedMathCore(1LL, false); // 100% complete
	}
	// In a full implementation, we track the background task progress bits here
	return FixedMathCore(0LL, true); 
}

void ResourceLoader::add_resource_format_loader(ResourceFormatLoader *p_format_loader, bool p_at_front) {
	if (p_at_front) {
		singleton->loaders.push_front(p_format_loader);
	} else {
		singleton->loaders.push_back(p_format_loader);
	}
}

void ResourceLoader::remove_resource_format_loader(ResourceFormatLoader *p_format_loader) {
	singleton->loaders.erase(singleton->loaders.find(p_format_loader));
}

void ResourceLoader::clear_cache() {
	singleton->resource_cache.clear();
}

bool ResourceLoader::has_cached(const String &p_path) {
	return singleton->resource_cache.has(p_path);
}

BigIntCore ResourceLoader::get_resource_size(const String &p_path) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_valid()) {
		return BigIntCore(static_cast<int64_t>(f->get_length()));
	}
	return BigIntCore(0LL);
}

--- END OF FILE core/io/resource_loader.cpp ---
