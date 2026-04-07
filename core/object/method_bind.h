--- START OF FILE core/object/method_bind.h ---

#ifndef METHOD_BIND_H
#define METHOD_BIND_H

#include "core/typedefs.h"
#include "core/variant/variant.h"
#include "core/string/string_name.h"
#include <type_traits>

class Object;

/**
 * MethodBind
 * 
 * Abstract base for wrapping C++ member functions.
 * Optimized for high-frequency Warp kernel dispatch and EnTT registry interaction.
 */
class MethodBind {
	StringName name;
	StringName instance_class;
	int argument_count = 0;
	bool _const = false;
	bool _returns = false;

public:
	_FORCE_INLINE_ void set_name(const StringName &p_name) { name = p_name; }
	_FORCE_INLINE_ StringName get_name() const { return name; }
	_FORCE_INLINE_ void set_instance_class(const StringName &p_class) { instance_class = p_class; }
	_FORCE_INLINE_ StringName get_instance_class() const { return instance_class; }

	_FORCE_INLINE_ int get_argument_count() const { return argument_count; }
	_FORCE_INLINE_ bool has_return() const { return _returns; }
	_FORCE_INLINE_ bool is_const() const { return _const; }

	/**
	 * call()
	 * Dynamic invocation using Variant arguments.
	 */
	virtual Variant call(Object *p_object, const Variant **p_args, int p_argcount, Callable::CallError &r_error) = 0;

	/**
	 * ptrcall()
	 * Zero-copy invocation using raw pointers. 
	 * Foundation for 120 FPS batch-oriented math in EnTT sweeps.
	 */
	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) = 0;

	MethodBind() {}
	virtual ~MethodBind() {}

protected:
	_FORCE_INLINE_ void set_argument_count(int p_count) { argument_count = p_count; }
	_FORCE_INLINE_ void set_returns(bool p_returns) { _returns = p_returns; }
	_FORCE_INLINE_ void set_const(bool p_const) { _const = p_const; }
};

/**
 * MethodBindT
 * 
 * Variadic template implementation for member function binding.
 * Optimized to handle FixedMathCore and BigIntCore as first-class citizens.
 */
template <typename T, typename R, typename... Args>
class MethodBindT : public MethodBind {
	typedef R (T::*P)(Args...);
	P method;

	template <size_t... Is>
	_FORCE_INLINE_ Variant _call_impl(T *p_obj, const Variant **p_args, std::index_sequence<Is...>) {
		if constexpr (std::is_void_v<R>) {
			(p_obj->*method)((*p_args[Is]).operator typename std::decay<Args>::type()...);
			return Variant();
		} else {
			return Variant((p_obj->*method)((*p_args[Is]).operator typename std::decay<Args>::type()...));
		}
	}

	template <size_t... Is>
	_FORCE_INLINE_ void _ptrcall_impl(T *p_obj, const void **p_args, void *r_ret, std::index_sequence<Is...>) {
		if constexpr (std::is_void_v<R>) {
			(p_obj->*method)(*static_cast<const typename std::decay<Args>::type *>(p_args[Is])...);
		} else {
			*static_cast<R *>(r_ret) = (p_obj->*method)(*static_cast<const typename std::decay<Args>::type *>(p_args[Is])...);
		}
	}

public:
	virtual Variant call(Object *p_object, const Variant **p_args, int p_argcount, Callable::CallError &r_error) override {
		r_error.error = Callable::CallError::CALL_OK;
		if (p_argcount != sizeof...(Args)) {
			r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENTS;
			r_error.expected = sizeof...(Args);
			return Variant();
		}
		return _call_impl(static_cast<T *>(p_object), p_args, std::index_sequence_for<Args...>{});
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) override {
		_ptrcall_impl(static_cast<T *>(p_object), p_args, r_ret, std::index_sequence_for<Args...>{});
	}

	MethodBindT(P p_method) :
			method(p_method) {
		set_argument_count(sizeof...(Args));
		set_returns(!std::is_void_v<R>);
		set_const(false);
	}
};

// Helper for non-const method creation
template <typename T, typename R, typename... Args>
MethodBind *create_method_bind(R (T::*p_method)(Args...)) {
	return memnew((MethodBindT<T, R, Args...>)(p_method));
}

// Specialization for const methods (critical for spatial queries)
template <typename T, typename R, typename... Args>
class MethodBindConstT : public MethodBind {
	typedef R (T::*P)(Args...) const;
	P method;

public:
	virtual Variant call(Object *p_object, const Variant **p_args, int p_argcount, Callable::CallError &r_error) override {
		return _call_impl(static_cast<const T *>(p_object), p_args, std::index_sequence_for<Args...>{});
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) override {
		_ptrcall_impl(static_cast<const T *>(p_object), p_args, r_ret, std::index_sequence_for<Args...>{});
	}

	MethodBindConstT(P p_method) : method(p_method) {
		set_argument_count(sizeof...(Args));
		set_returns(!std::is_void_v<R>);
		set_const(true);
	}

private:
	template <size_t... Is>
	_FORCE_INLINE_ Variant _call_impl(const T *p_obj, const Variant **p_args, std::index_sequence<Is...>) {
		if constexpr (std::is_void_v<R>) {
			(p_obj->*method)((*p_args[Is]).operator typename std::decay<Args>::type()...);
			return Variant();
		} else {
			return Variant((p_obj->*method)((*p_args[Is]).operator typename std::decay<Args>::type()...));
		}
	}

	template <size_t... Is>
	_FORCE_INLINE_ void _ptrcall_impl(const T *p_obj, const void **p_args, void *r_ret, std::index_sequence<Is...>) {
		if constexpr (std::is_void_v<R>) {
			(p_obj->*method)(*static_cast<const typename std::decay<Args>::type *>(p_args[Is])...);
		} else {
			*static_cast<R *>(r_ret) = (p_obj->*method)(*static_cast<const typename std::decay<Args>::type *>(p_args[Is])...);
		}
	}
};

template <typename T, typename R, typename... Args>
MethodBind *create_method_bind(R (T::*p_method)(Args...) const) {
	return memnew((MethodBindConstT<T, R, Args...>)(p_method));
}

#endif // METHOD_BIND_H

--- END OF FILE core/object/method_bind.h ---
