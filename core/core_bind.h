--- START OF FILE core/core_bind.h ---

#ifndef CORE_BIND_H
#define CORE_BIND_H

#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/variant/variant.h"
#include "src/big_number.h"

/**
 * _OS
 * 
 * Internal binding for the Operating System abstraction.
 * Upgraded to provide high-precision simulation ticks as BigNumber instances.
 */
class _OS : public Object {
	GDCLASS(_OS, Object);

	static _OS *singleton;

protected:
	static void _bind_methods();

public:
	static _OS *get_singleton();

	// Standard OS functionality
	String get_name() const;
	String get_version() const;
	
	// Hyper-Simulation Timing API
	Ref<BigNumber> get_ticks_msec_big() const;
	Ref<BigNumber> get_ticks_usec_big() const;
	Ref<FixedNumber> get_precise_time_step() const;

	void delay_usec(uint32_t p_usec) const;

	_OS();
	~_OS();
};

/**
 * _Engine
 * 
 * Internal binding for the global Engine singleton.
 * Provides the "Universal Solver" factory methods to instantiate 
 * high-precision components directly within the SceneTree.
 */
class _Engine : public Object {
	GDCLASS(_Engine, Object);

	static _Engine *singleton;

protected:
	static void _bind_methods();

public:
	static _Engine *get_singleton();

	void set_physics_ticks_per_second(int p_ips);
	int get_physics_ticks_per_second() const;

	void set_max_fps(int p_fps);
	int get_max_fps() const;

	void set_time_scale(double p_scale);
	double get_time_scale() const;

	// Universal Solver Object Factories
	Ref<BigNumber> create_big_number(const String &p_value) const;
	Ref<FixedNumber> create_fixed_number(const String &p_value) const;
	Ref<FixedNumber> create_fixed_from_float(double p_value) const;

	bool is_editor_hint() const;

	_Engine();
	~_Engine();
};

#endif // CORE_BIND_H

--- END OF FILE core/core_bind.h ---
