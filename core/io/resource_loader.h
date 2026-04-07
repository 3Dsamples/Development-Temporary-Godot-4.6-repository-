--- START OF FILE core/io/resource_loader.h ---

#ifndef RESOURCE_LOADER_H
#define RESOURCE_LOADER_H

#include "core/io/resource.h"
#include "core/object/object.h"
#include "core/templates/hash_map.h"
#include "core/templates/list.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

class ResourceFormatLoader;

/**
 * ResourceLoader
 * 
 * The central manager for loading deterministic simulation assets.
 * Optimized for multi-threaded streaming of galactic data blocks.
 * Uses FixedMathCore for sub-percent loading progress accuracy.
 */
class ET_ALIGN_32 ResourceLoader : public Object {
	GDCLASS(ResourceLoader, Object);

	static ResourceLoader *singleton;

	HashMap<String, Ref<Resource>> resource_cache;
	List<ResourceFormatLoader *> loaders;

	bool abort_on_missing_resource = true;
	bool use_thread_queuing = true;

protected:
	static void _bind_methods();

public:
	static _FORCE_INLINE_ ResourceLoader *get_singleton() { return singleton; }

	// ------------------------------------------------------------------------
	// Loading API (Deterministic & Scale-Aware)
	// ------------------------------------------------------------------------

	/**
	 * load()
	 * Synchronous asset retrieval. Ensures that the loaded resource
	 * is bit-perfectly prepared for EnTT registry insertion.
	 */
	static Ref<Resource> load(const String &p_path, const String &p_type_hint = "", bool p_no_cache = false, Error *r_error = nullptr);

	/**
	 * load_threaded_request()
	 * Initiates asynchronous loading via the SimulationThreadPool.
	 * Critical for maintaining 120 FPS while streaming massive star clusters.
	 */
	static Error load_threaded_request(const String &p_path, const String &p_type_hint = "", bool p_use_subthreads = true);

	/**
	 * load_threaded_get_status()
	 * Returns the current loading progress using FixedMathCore [0.0, 1.0].
	 * Guarantees UI consistency regardless of system performance.
	 */
	static FixedMathCore load_threaded_get_status(const String &p_path, Error *r_error = nullptr);

	// ------------------------------------------------------------------------
	// Cache & Registry Management
	// ------------------------------------------------------------------------

	static void add_resource_format_loader(ResourceFormatLoader *p_format_loader, bool p_at_front = false);
	static void remove_resource_format_loader(ResourceFormatLoader *p_format_loader);

	static void clear_cache();
	static bool has_cached(const String &p_path);

	/**
	 * get_resource_size()
	 * Returns the size on disk as BigIntCore to support multi-terabyte assets.
	 */
	static BigIntCore get_resource_size(const String &p_path);

	ResourceLoader();
	~ResourceLoader();
};

/**
 * ResourceFormatLoader
 * 
 * Abstract interface for specific simulation file format importers.
 * Designed for zero-copy binary reconstruction of math core data.
 */
class ResourceFormatLoader {
public:
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path, Error *r_error = nullptr) = 0;
	virtual void get_recognized_extensions(List<String> *p_extensions) const = 0;
	virtual bool handles_type(const String &p_type) const = 0;
	virtual String get_resource_type(const String &p_path) const = 0;
	
	virtual ~ResourceFormatLoader() {}
};

#endif // RESOURCE_LOADER_H

--- END OF FILE core/io/resource_loader.h ---
