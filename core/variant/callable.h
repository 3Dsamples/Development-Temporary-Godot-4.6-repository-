--- START OF FILE core/variant/callable.h ---

#ifndef CALLABLE_H
#define CALLABLE_H

#include "core/typedefs.h"
#include "core/string/string_name.h"
#include "src/fixed_math_core.h"

class Object;
class Variant;

/**
 * Callable
 * 
 * A high-performance delegate for the Universal Solver.
 * Encapsulates an object instance and a method name for dynamic invocation.
 * Optimized for Warp-Kernel dispatch and EnTT-based event propagation.
 */
class ET_ALIGN_32 Callable {
public:
	enum CallErrorType {
		CALL_OK,
		CALL_ERROR_INVALID_METHOD,
		CALL_ERROR_INVALID_ARGUMENTS,
		CALL_ERROR_TOO_MANY_ARGUMENTS,
		CALL_ERROR_TOO_FEW_ARGUMENTS,
		CALL_ERROR_INSTANCE_IS_NULL,
		CALL_ERROR_METHOD_NOT_CONST,
	};

	struct CallError {
		CallErrorType error;
		int argument;
		int expected;
	};

private:
	Object *object = nullptr;
	StringName method;
	uint64_t _id = 0; // Internal EnTT-compatible handle

public:
	// ------------------------------------------------------------------------
	// Execution API
	// ------------------------------------------------------------------------

	/**
	 * call()
	 * Synchronous execution using Zero-Copy Variant pointers.
	 */
	void call(const Variant **p_args, int p_argcount, Variant &r_ret, CallError &r_error) const;

	/**
	 * call_deferred()
	 * Queues the call for the next 120 FPS heartbeat synchronization point.
	 */
	void call_deferred(const Variant **p_args, int p_argcount) const;

	// ------------------------------------------------------------------------
	// Accessors & Logic
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ bool is_null() const { return object == nullptr; }
	_FORCE_INLINE_ bool is_custom() const { return false; } // Simplified for core
	_FORCE_INLINE_ bool is_valid() const;

	_FORCE_INLINE_ Object *get_object() const { return object; }
	_FORCE_INLINE_ StringName get_method() const { return method; }
	_FORCE_INLINE_ uint64_t get_object_id() const;

	// ------------------------------------------------------------------------
	// Operators (O(1) Comparison for Warp Kernels)
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ bool operator==(const Callable &p_callable) const {
		return object == p_callable.object && method == p_callable.method;
	}

	_FORCE_INLINE_ bool operator!=(const Callable &p_callable) const {
		return !(*this == p_callable);
	}

	_FORCE_INLINE_ bool operator<(const Callable &p_callable) const {
		if (object != p_callable.object) return object < p_callable.object;
		return method < p_callable.method;
	}

	uint32_t hash() const;

	Callable() {}
	Callable(Object *p_object, const StringName &p_method);
	~Callable() {}
};

#endif // CALLABLE_H

--- END OF FILE core/variant/callable.h ---
