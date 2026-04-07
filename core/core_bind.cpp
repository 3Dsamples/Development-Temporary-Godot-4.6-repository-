--- START OF FILE core/core_bind.cpp ---

#include "core/core_bind.h"
#include "core/os/os.h"
#include "core/config/engine.h"
#include "src/big_number.h"

// ============================================================================
// _OS Implementation
// ============================================================================

_OS *_OS::singleton = nullptr;

_OS * _OS::get_singleton() {
	return singleton;
}

void _OS::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_name"), &_OS::get_name);
	ClassDB::bind_method(D_METHOD("get_version"), &_OS::get_version);
	ClassDB::bind_method(D_METHOD("get_ticks_msec_big"), &_OS::get_ticks_msec_big);
	ClassDB::bind_method(D_METHOD("get_ticks_usec_big"), &_OS::get_ticks_usec_big);
	ClassDB::bind_method(D_METHOD("get_precise_time_step"), &_OS::get_precise_time_step);
	ClassDB::bind_method(D_METHOD("delay_usec", "usec"), &_OS::delay_usec);
}

_OS::_OS() {
	singleton = this;
}

_OS::~_OS() {
	singleton = nullptr;
}

String _OS::get_name() const {
	return OS::get_singleton()->get_name();
}

String _OS::get_version() const {
	return OS::get_singleton()->get_version();
}

Ref<BigNumber> _OS::get_ticks_msec_big() const {
	Ref<BigNumber> bn;
	bn.instantiate();
	bn->set_value_from_int(static_cast<int64_t>(OS::get_singleton()->get_ticks_msec()));
	return bn;
}

Ref<BigNumber> _OS::get_ticks_usec_big() const {
	Ref<BigNumber> bn;
	bn.instantiate();
	bn->set_value_from_int(static_cast<int64_t>(OS::get_singleton()->get_ticks_usec()));
	return bn;
}

Ref<FixedNumber> _OS::get_precise_time_step() const {
	Ref<FixedNumber> fn;
	fn.instantiate();
	// Convert micro-ticks to a deterministic fixed-point representation of seconds
	double sec = static_cast<double>(OS::get_singleton()->get_ticks_usec()) / 1000000.0;
	fn->set_value_from_float(sec);
	return fn;
}

void _OS::delay_usec(uint32_t p_usec) const {
	OS::get_singleton()->delay_usec(p_usec);
}

// ============================================================================
// _Engine Implementation
// ============================================================================

_Engine *_Engine::singleton = nullptr;

_Engine * _Engine::get_singleton() {
	return singleton;
}

void _Engine::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_physics_ticks_per_second", "ips"), &_Engine::set_physics_ticks_per_second);
	ClassDB::bind_method(D_METHOD("get_physics_ticks_per_second"), &_Engine::get_physics_ticks_per_second);
	ClassDB::bind_method(D_METHOD("set_max_fps", "max_fps"), &_Engine::set_max_fps);
	ClassDB::bind_method(D_METHOD("get_max_fps"), &_Engine::get_max_fps);
	ClassDB::bind_method(D_METHOD("set_time_scale", "time_scale"), &_Engine::set_time_scale);
	ClassDB::bind_method(D_METHOD("get_time_scale"), &_Engine::get_time_scale);
	
	ClassDB::bind_method(D_METHOD("create_big_number", "value"), &_Engine::create_big_number);
	ClassDB::bind_method(D_METHOD("create_fixed_number", "value"), &_Engine::create_fixed_number);
	ClassDB::bind_method(D_METHOD("create_fixed_from_float", "value"), &_Engine::create_fixed_from_float);
	
	ClassDB::bind_method(D_METHOD("is_editor_hint"), &_Engine::is_editor_hint);
}

_Engine::_Engine() {
	singleton = this;
}

_Engine::~_Engine() {
	singleton = nullptr;
}

void _Engine::set_physics_ticks_per_second(int p_ips) {
	Engine::get_singleton()->set_physics_ticks_per_second(p_ips);
}

int _Engine::get_physics_ticks_per_second() const {
	return Engine::get_singleton()->get_physics_ticks_per_second();
}

void _Engine::set_max_fps(int p_fps) {
	Engine::get_singleton()->set_max_fps(p_fps);
}

int _Engine::get_max_fps() const {
	return Engine::get_singleton()->get_max_fps();
}

void _Engine::set_time_scale(double p_scale) {
	Engine::get_singleton()->set_time_scale(p_scale);
}

double _Engine::get_time_scale() const {
	return Engine::get_singleton()->get_time_scale();
}

Ref<BigNumber> _Engine::create_big_number(const String &p_value) const {
	Ref<BigNumber> bn;
	bn.instantiate();
	bn->set_value_from_string(p_value);
	return bn;
}

Ref<FixedNumber> _Engine::create_fixed_number(const String &p_value) const {
	Ref<FixedNumber> fn;
	fn.instantiate();
	fn->set_value_from_string(p_value);
	return fn;
}

Ref<FixedNumber> _Engine::create_fixed_from_float(double p_value) const {
	Ref<FixedNumber> fn;
	fn.instantiate();
	fn->set_value_from_float(p_value);
	return fn;
}

bool _Engine::is_editor_hint() const {
	return Engine::get_singleton()->is_editor_hint();
}

--- END OF FILE core/core_bind.cpp ---
