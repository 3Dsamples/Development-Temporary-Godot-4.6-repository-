--- START OF FILE core/object/object.h ---

#ifndef OBJECT_H
#define OBJECT_H

#include "core/typedefs.h"
#include "core/string/string_name.h"
#include "core/templates/list.h"
#include "core/templates/hash_map.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

class Variant;
class ScriptInstance;

/**
 * Object
 * 
 * The base class for all simulated entities.
 * Features EnTT Sparse-Set handle integration and Warp-Kernel affinity.
 * Aligned for SIMD-accelerated metadata lookups and 120 FPS state updates.
 */
class ET_ALIGN_32 Object {
public:
	enum ConnectFlags {
		CONNECT_DEFERRED = 1,
		CONNECT_PERSIST = 2,
		CONNECT_ONE_SHOT = 4,
		CONNECT_REFERENCE_COUNTED = 8,
	};

private:
	// Unique ID for EnTT Sparse-Set Registry integration
	uint64_t _instance_id;
	
	// Simulation metadata
	HashMap<StringName, Variant> _metadata;
	ScriptInstance *_script_instance = nullptr;

	// Signal system management
	struct Connection {
		Object *target = nullptr;
		StringName method;
		uint32_t flags = 0;
	};
	HashMap<StringName, List<Connection>> _connections;

	bool _block_signals = false;
	bool _predelete_ok = true;

	// ETEngine: Simulation affinity tag (Deterministic, Macro, or Standard)
	uint8_t _sim_tier = 1; // Default to TIER_DETERMINISTIC

protected:
	virtual void _notification(int p_notification) {}
	static void _bind_methods();

public:
	// ------------------------------------------------------------------------
	// Lifecycle & Identity
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ uint64_t get_instance_id() const { return _instance_id; }
	
	virtual String get_class() const { return "Object"; }
	virtual String get_parent_class() const { return ""; }
	static String get_class_static() { return "Object"; }
	static String get_parent_class_static() { return ""; }
	
	virtual bool is_class(const String &p_class) const { return p_class == "Object"; }

	// ------------------------------------------------------------------------
	// Deterministic Property System
	// ------------------------------------------------------------------------

	virtual bool set(const StringName &p_name, const Variant &p_value);
	virtual bool get(const StringName &p_name, Variant &r_ret) const;
	
	/**
	 * set_meta() / get_meta()
	 * High-speed metadata storage. Optimized for BigIntCore entity handles 
	 * and FixedMathCore physical constants.
	 */
	void set_meta(const StringName &p_name, const Variant &p_value);
	Variant get_meta(const StringName &p_name) const;
	bool has_meta(const StringName &p_name) const;

	// ------------------------------------------------------------------------
	// Signal & Connection API
	// ------------------------------------------------------------------------

	Error connect(const StringName &p_signal, Object *p_target, const StringName &p_method, uint32_t p_flags = 0);
	void disconnect(const StringName &p_signal, Object *p_target, const StringName &p_method);
	void emit_signal(const StringName &p_name, const Variant **p_args, int p_argcount);

	// ------------------------------------------------------------------------
	// Hyper-Simulation Synchronization
	// ------------------------------------------------------------------------

	/**
	 * notification()
	 * The primary execution hook for 120 FPS simulation heartbeats.
	 */
	void notification(int p_notification, bool p_reversed = false);

	_FORCE_INLINE_ void set_sim_tier(uint8_t p_tier) { _sim_tier = p_tier; }
	_FORCE_INLINE_ uint8_t get_sim_tier() const { return _sim_tier; }

	// ------------------------------------------------------------------------
	// Scripting & ECS Integration
	// ------------------------------------------------------------------------

	void set_script(const Variant &p_script);
	Variant get_script() const;
	_FORCE_INLINE_ ScriptInstance *get_script_instance() const { return _script_instance; }

	Object();
	virtual ~Object();
};

/**
 * GDCLASS Macro
 * 
 * Injects the required reflection logic into child classes for 
 * ClassDB and EnTT registry mapping.
 */
#define GDCLASS(m_class, m_inherits) \
private: \
	void operator=(const m_class &p_rval) {} \
public: \
	virtual String get_class() const override { return #m_class; } \
	virtual String get_parent_class() const override { return #m_inherits; } \
	static String get_class_static() { return #m_class; } \
	static String get_parent_class_static() { return #m_inherits; } \
	virtual bool is_class(const String &p_class) const override { return (p_class == #m_class) || m_inherits::is_class(p_class); }

#endif // OBJECT_H

--- END OF FILE core/object/object.h ---
