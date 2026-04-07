--- START OF FILE core/object/class_db.cpp ---

#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/os/memory.h"

// Static storage for the global class registry
HashMap<StringName, ClassDB::ClassInfo> ClassDB::classes;

/**
 * bind_method()
 * 
 * Registers a method binder into the specified class.
 * Ensures O(1) lookup during high-frequency Warp kernel dispatches.
 */
void ClassDB::bind_method(const StringName &p_class, MethodBind *p_method) {
	if (unlikely(!classes.has(p_class))) {
		memdelete(p_method);
		ERR_FAIL_MSG("ClassDB Error: Attempted to bind method to non-existent class: " + String(p_class));
	}

	ClassInfo &ci = classes[p_class];
	StringName name = p_method->get_name();

	if (unlikely(ci.method_map.has(name))) {
		memdelete(p_method);
		ERR_FAIL_MSG("ClassDB Error: Method already exists: " + String(p_class) + "::" + String(name));
	}

	ci.method_map[name] = p_method;
}

/**
 * add_property()
 * 
 * Defines a class property with its associated setter and getter.
 * Natively supports BigIntCore and FixedMathCore types for metadata.
 */
void ClassDB::add_property(const StringName &p_class, const PropertyInfo &p_pinfo, const StringName &p_setter, const StringName &p_getter) {
	if (unlikely(!classes.has(p_class))) return;
	ClassInfo &ci = classes[p_class];
	ci.property_list.push_back(p_pinfo);
}

/**
 * instantiate()
 * 
 * High-performance object factory. 
 * Invokes the pre-registered creation function for the class.
 */
Object *ClassDB::instantiate(const StringName &p_class) {
	if (unlikely(!classes.has(p_class))) {
		return nullptr;
	}

	ClassInfo &ci = classes[p_class];
	if (unlikely(ci.is_virtual || !ci.creation_func)) {
		return nullptr;
	}

	return ci.creation_func();
}

bool ClassDB::class_exists(const StringName &p_class) {
	return classes.has(p_class);
}

StringName ClassDB::get_parent_class(const StringName &p_class) {
	if (!classes.has(p_class)) return StringName();
	return classes[p_class].inherits;
}

/**
 * get_method()
 * 
 * Returns the binder for a method, traversing the inheritance tree if necessary.
 * Optimized for high-frequency 120 FPS script-to-engine calls.
 */
MethodBind *ClassDB::get_method(const StringName &p_class, const StringName &p_method) {
	StringName current = p_class;
	while (current != StringName()) {
		if (classes.has(current)) {
			const ClassInfo &ci = classes[current];
			if (ci.method_map.has(p_method)) {
				return ci.method_map[p_method];
			}
		}
		current = get_parent_class(current);
	}
	return nullptr;
}

void ClassDB::init() {
	// Root initialization of the reflection system
}

/**
 * cleanup()
 * 
 * Graceful teardown of the registry. 
 * Frees all allocated MethodBind objects to ensure zero memory leaks.
 */
void ClassDB::cleanup() {
	for (auto &E_class : classes) {
		for (auto &E_method : E_class.value.method_map) {
			memdelete(E_method.value);
		}
		E_class.value.method_map.clear();
		E_class.value.property_list.clear();
	}
	classes.clear();
}

--- END OF FILE core/object/class_db.cpp ---
