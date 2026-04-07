--- START OF FILE core/object/class_db.h ---

#ifndef CLASS_DB_H
#define CLASS_DB_H

#include "core/object/method_bind.h"
#include "core/object/object.h"
#include "core/templates/hash_map.h"
#include "core/templates/list.h"
#include "core/string/string_name.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * ClassDB
 * 
 * The master registry for engine reflection. 
 * Updated to support Warp-Kernel method binding and EnTT component mapping.
 * Natively supports BigIntCore and FixedMathCore for all property metadata.
 */
class ClassDB {
public:
	struct PropertyInfo {
		uint32_t type; // maps to Variant::Type
		StringName name;
		StringName class_name;
		uint32_t hint = 0;
		String hint_string;
		uint32_t usage = 7; // PROPERTY_USAGE_DEFAULT

		PropertyInfo() : type(0) {}
		PropertyInfo(uint32_t p_type, const StringName &p_name, uint32_t p_hint = 0, const String &p_hint_string = "", uint32_t p_usage = 7) :
				type(p_type), name(p_name), hint(p_hint), hint_string(p_hint_string), usage(p_usage) {}
	};

	struct MethodInfo {
		StringName name;
		PropertyInfo return_val;
		List<PropertyInfo> arguments;
		uint32_t flags = 1; // METHOD_FLAG_NORMAL

		MethodInfo() {}
		MethodInfo(const StringName &p_name) : name(p_name) {}
	};

private:
	struct ClassInfo {
		StringName name;
		StringName inherits;
		HashMap<StringName, MethodBind *> method_map;
		List<PropertyInfo> property_list;
		HashMap<StringName, MethodInfo> signal_map;
		
		Object *(*creation_func)() = nullptr;
		bool is_virtual = false;

		ClassInfo() {}
	};

	static HashMap<StringName, ClassInfo> classes;

public:
	// ------------------------------------------------------------------------
	// Class Registration API
	// ------------------------------------------------------------------------

	template <typename T>
	static void register_class() {
		StringName name = T::get_class_static();
		StringName inherits = T::get_parent_class_static();
		ClassInfo ci;
		ci.name = name;
		ci.inherits = inherits;
		ci.creation_func = []() -> Object * { return memnew(T); };
		classes[name] = ci;
		T::_bind_methods();
	}

	template <typename T>
	static void register_virtual_class() {
		StringName name = T::get_class_static();
		StringName inherits = T::get_parent_class_static();
		ClassInfo ci;
		ci.name = name;
		ci.inherits = inherits;
		ci.is_virtual = true;
		classes[name] = ci;
		T::_bind_methods();
	}

	// ------------------------------------------------------------------------
	// Method Binding API (Batch & Warp Optimized)
	// ------------------------------------------------------------------------

	static void bind_method(const StringName &p_class, MethodBind *p_method);

	template <typename T, typename M>
	static MethodBind *bind_method(const StringName &p_method_name, M p_method) {
		MethodBind *mb = create_method_bind(p_method);
		mb->set_name(p_method_name);
		mb->set_instance_class(T::get_class_static());
		bind_method(T::get_class_static(), mb);
		return mb;
	}

	static void add_property(const StringName &p_class, const PropertyInfo &p_pinfo, const StringName &p_setter, const StringName &p_getter);
	static void add_signal(const StringName &p_class, const MethodInfo &p_signal);

	// ------------------------------------------------------------------------
	// Retrieval API
	// ------------------------------------------------------------------------

	static Object *instantiate(const StringName &p_class);
	static bool class_exists(const StringName &p_class);
	static StringName get_parent_class(const StringName &p_class);
	static bool is_parent_class(const StringName &p_class, const StringName &p_inherits);

	static MethodBind *get_method(const StringName &p_class, const StringName &p_method);
	
	static void init();
	static void cleanup();
};

#endif // CLASS_DB_H

--- END OF FILE core/object/class_db.h ---
