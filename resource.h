--- START OF FILE core/io/resource.h ---

#ifndef RESOURCE_H
#define RESOURCE_H

#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * Resource
 * 
 * Base class for all loadable simulation data.
 * Merges Godot's asset lifecycle with Universal Solver determinism.
 * Aligned to 32 bytes to facilitate EnTT-based resource handle batching.
 */
class ET_ALIGN_32 Resource : public RefCounted {
	GDCLASS(Resource, RefCounted);

	String path_cache;
	String name_cache;
	
	// Simulation metadata using BigIntCore to support multi-petabyte archive offsets
	BigIntCore resource_size;
	
	// Deterministic flag for Warp kernel validation
	bool is_deterministic = true;

protected:
	static void _bind_methods();
	virtual void _resource_path_changed();

public:
	// ------------------------------------------------------------------------
	// Resource Identity API
	// ------------------------------------------------------------------------

	void set_path(const String &p_path);
	String get_path() const;

	void set_name(const String &p_name);
	String get_name() const;

	// ------------------------------------------------------------------------
	// Hyper-Simulation Metadata
	// ------------------------------------------------------------------------

	/**
	 * set_resource_size()
	 * Uses BigIntCore to track the byte weight of high-precision simulation assets.
	 */
	void set_resource_size(const BigIntCore &p_size);
	BigIntCore get_resource_size() const;

	/**
	 * set_deterministic()
	 * Flags the resource for use in TIER_DETERMINISTIC Warp kernels.
	 */
	void set_deterministic(bool p_enabled);
	bool get_deterministic() const;

	// ------------------------------------------------------------------------
	// Logic and State
	// ------------------------------------------------------------------------

	/**
	 * duplicate()
	 * Creates a deep copy of the resource. 
	 * Ensures all FixedMathCore and BigIntCore components are cloned bit-perfectly.
	 */
	virtual Ref<Resource> duplicate(bool p_subresources = false) const;

	void emit_changed();

	Resource();
	virtual ~Resource();
};

#endif // RESOURCE_H

--- END OF FILE core/io/resource.h ---
