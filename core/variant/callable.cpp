--- START OF FILE core/variant/callable.cpp ---

#include "core/variant/callable.h"
#include "core/object/object.h"
#include "core/object/message_queue.h"
#include "core/variant/variant.h"

/**
 * Callable Constructor
 * 
 * Binds an Object instance to a specific method handle.
 */
Callable::Callable(Object *p_object, const StringName &p_method) {
	object = p_object;
	method = p_method;
	_id = p_object ? p_object->get_instance_id() : 0;
}

/**
 * call()
 * 
 * Synchronous execution of the delegate.
 * Uses the low-level callp interface to achieve Zero-Copy argument passing,
 * ensuring that heavy FixedMathCore or BigIntCore data is processed 
 * with maximum throughput.
 */
void Callable::call(const Variant **p_args, int p_argcount, Variant &r_ret, CallError &r_error) const {
	if (unlikely(!object)) {
		r_error.error = CALL_ERROR_INSTANCE_IS_NULL;
		return;
	}

	r_ret = object->callp(method, p_args, p_argcount, r_error);
}

/**
 * call_deferred()
 * 
 * Schedules the method call in the MessageQueue.
 * Essential for synchronizing multi-threaded EnTT logic with the 
 * main 120 FPS simulation thread.
 */
void Callable::call_deferred(const Variant **p_args, int p_argcount) const {
	if (unlikely(!object)) {
		return;
	}

	MessageQueue::get_singleton()->push_call(object, method, p_args, p_argcount);
}

bool Callable::is_valid() const {
	if (!object) return false;
	// In a full implementation, we would check if the instance ID is still alive
	// in the global Object database or EnTT registry.
	return true;
}

uint64_t Callable::get_object_id() const {
	return _id;
}

/**
 * hash()
 * 
 * Generates a unique hash for the delegate.
 * Optimized for O(1) comparison within Warp-Kernel dispatch tables.
 */
uint32_t Callable::hash() const {
	uint32_t h = hash_murmur3_one_64(_id);
	return hash_murmur3_one_32(method.hash(), h);
}

--- END OF FILE core/variant/callable.cpp ---
