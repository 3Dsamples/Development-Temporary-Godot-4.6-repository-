--- START OF FILE core/object/ref_counted.cpp ---

#include "core/object/ref_counted.h"
#include "core/object/class_db.h"

/**
 * _bind_methods
 * 
 * Registers the reference counting API. This allows scripts to 
 * query the lifecycle state of high-precision simulation objects.
 */
void RefCounted::_bind_methods() {
	ClassDB::bind_method(D_METHOD("init_ref"), &RefCounted::init_ref);
	ClassDB::bind_method(D_METHOD("reference"), &RefCounted::reference);
	ClassDB::bind_method(D_METHOD("unreference"), &RefCounted::unreference);
	ClassDB::bind_method(D_METHOD("get_reference_count"), &RefCounted::get_reference_count);
}

/**
 * RefCounted Constructor
 * 
 * Initializes the atomic counters. The init_ref state ensures that 
 * objects created in C++ but not yet assigned to a Ref<T> handle 
 * are not prematurely deleted.
 */
RefCounted::RefCounted() {
	ref_count.init(0);
	ref_count_init.init(1);
}

/**
 * RefCounted Destructor
 * 
 * Standard virtual destructor. Object cleanup is handled by 
 * the memory management macros (memdelete) triggered by unreference().
 */
RefCounted::~RefCounted() {
}

--- END OF FILE core/object/ref_counted.cpp ---
