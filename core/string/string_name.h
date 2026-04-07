--- START OF FILE core/string/string_name.h ---

#ifndef STRING_NAME_H
#define STRING_NAME_H

#include "core/string/ustring.h"
#include "core/templates/safe_refcount.h"

/**
 * StringName
 * 
 * High-performance unique string handles. 
 * Essential for EnTT registry keys and Warp kernel metadata.
 * Uses a global pool to ensure that identical strings share the same memory address,
 * making equality checks a simple pointer comparison for 120 FPS efficiency.
 */
class ET_ALIGN_32 StringName {
	struct _Data {
		SafeRefCount refcount;
		String string;
		uint32_t hash;
		_Data *next;

		_Data() : hash(0), next(nullptr) {}
	};

	_Data *_data = nullptr;

	friend void register_core_types();
	friend void unregister_core_types();

public:
	// ------------------------------------------------------------------------
	// Constructors & Destructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE StringName() {}
	StringName(const StringName &p_from);
	StringName(const String &p_string);
	StringName(const char *p_contents);
	
	~StringName();

	// ------------------------------------------------------------------------
	// Operators (O(1) Comparison)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ bool operator==(const StringName &p_other) const { return _data == p_other._data; }
	_FORCE_INLINE_ bool operator!=(const StringName &p_other) const { return _data != p_other._data; }
	_FORCE_INLINE_ bool operator<(const StringName &p_other) const { return _data < p_other._data; }

	void operator=(const StringName &p_from);

	// ------------------------------------------------------------------------
	// Logic & Data Access
	// ------------------------------------------------------------------------
	operator String() const;
	
	_FORCE_INLINE_ uint32_t hash() const { return _data ? _data->hash : 0; }
	_FORCE_INLINE_ bool is_empty() const { return _data == nullptr; }

	// ------------------------------------------------------------------------
	// Static Pool Management
	// ------------------------------------------------------------------------
	static void setup();
	static void cleanup();
};

/**
 * SNAME Macro
 * 
 * Provides a thread-safe, static cache for StringNames.
 * Used heavily in EnTT component tagging and Warp execution paths.
 */
#define SNAME(m_arg) ([]() { static StringName s(m_arg); return s; }())

#endif // STRING_NAME_H

--- END OF FILE core/string/string_name.h ---
