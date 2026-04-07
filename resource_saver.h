--- START OF FILE core/io/resource_saver.h ---

#ifndef RESOURCE_SAVER_H
#define RESOURCE_SAVER_H

#include "core/io/resource.h"
#include "core/object/object.h"
#include "core/templates/list.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

class ResourceFormatSaver;

/**
 * ResourceSaver
 * 
 * Global singleton for serializing deterministic simulation assets.
 * Optimized for high-throughput binary storage of FixedMath and BigInt data.
 * Features mandatory integrity validation for galactic-scale save files.
 */
class ET_ALIGN_32 ResourceSaver : public Object {
	GDCLASS(ResourceSaver, Object);

	static ResourceSaver *singleton;

public:
	enum SaverFlags {
		FLAG_NONE = 0,
		FLAG_RELATIVE_PATHS = 1,
		FLAG_BUNDLE_RESOURCES = 2,
		FLAG_CHANGE_PATH = 4,
		FLAG_OMIT_EDITOR_PROPERTIES = 8,
		FLAG_SAVE_BIGINT_AS_BINARY = 16, // Optimization for Universal Solver
		FLAG_COMPRESS = 32,
	};

private:
	List<ResourceFormatSaver *> savers;

protected:
	static void _bind_methods();

public:
	static _FORCE_INLINE_ ResourceSaver *get_singleton() { return singleton; }

	/**
	 * save()
	 * The primary entry point for persisting resources.
	 * Guarantees bit-perfect serialization for deterministic physics states
	 * by utilizing raw FixedMathCore stream buffers.
	 */
	static Error save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags = 0);

	static void add_resource_format_saver(ResourceFormatSaver *p_format_saver, bool p_at_front = false);
	static void remove_resource_format_saver(ResourceFormatSaver *p_format_saver);

	/**
	 * get_recognized_extensions()
	 * Returns extensions supported by the Universal Solver's registered savers.
	 */
	static void get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions);

	ResourceSaver();
	~ResourceSaver();
};

/**
 * ResourceFormatSaver
 * 
 * Abstract interface for implementing custom simulation storage formats.
 * Designed for zero-copy serialization of EnTT component arrays.
 */
class ResourceFormatSaver {
public:
	virtual Error save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags = 0) = 0;
	virtual void get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const = 0;
	virtual bool recognize(const Ref<Resource> &p_resource) const = 0;

	virtual ~ResourceFormatSaver() {}
};

#endif // RESOURCE_SAVER_H

--- END OF FILE core/io/resource_saver.h ---
