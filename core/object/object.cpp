--- START OF FILE core/object/object.cpp ---

#include "core/object/object.h"
#include "core/object/class_db.h"
#include "core/object/script_instance.h"
#include "core/object/message_queue.h"
#include "core/variant/variant.h"
#include "core/os/memory.h"

// Global instance counter using 64-bit to map perfectly to BigIntCore handles
static uint64_t _instance_counter = 1;

void Object::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_class"), &Object::get_class);
	ClassDB::bind_method(D_METHOD("is_class", "name"), &Object::is_class);
	ClassDB::bind_method(D_METHOD("set", "property", "value"), &Object::set);
	ClassDB::bind_method(D_METHOD("get", "property"), &Object::get);
	ClassDB::bind_method(D_METHOD("get_instance_id"), &Object::get_instance_id);

	ClassDB::bind_method(D_METHOD("set_meta", "name", "value"), &Object::set_meta);
	ClassDB::bind_method(D_METHOD("get_meta", "name"), &Object::get_meta);
	ClassDB::bind_method(D_METHOD("has_meta", "name"), &Object::has_meta);

	ClassDB::bind_method(D_METHOD("connect", "signal", "target", "method", "flags"), &Object::connect, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("disconnect", "signal", "target", "method"), &Object::disconnect);

	ClassDB::bind_method(D_METHOD("set_sim_tier", "tier"), &Object::set_sim_tier);
	ClassDB::bind_method(D_METHOD("get_sim_tier"), &Object::get_sim_tier);

	// Core Simulation Notifications
	BIND_CONSTANT(NOTIFICATION_POSTINITIALIZE);
	BIND_CONSTANT(NOTIFICATION_PREDELETE);
}

Object::Object() {
	_instance_id = _instance_counter++;
	_sim_tier = 1; // Default to TIER_DETERMINISTIC (FixedMath)
}

Object::~Object() {
	_predelete_ok = false;
	if (_script_instance) {
		memdelete(_script_instance);
	}

	// Graceful disconnection of all Warp-Style observers
	_connections.clear();
	_metadata.clear();
}

bool Object::set(const StringName &p_name, const Variant &p_value) {
	if (_script_instance && _script_instance->set(p_name, p_value)) {
		return true;
	}
	return false;
}

bool Object::get(const StringName &p_name, Variant &r_ret) const {
	if (_script_instance && _script_instance->get(p_name, r_ret)) {
		return true;
	}
	return false;
}

void Object::set_meta(const StringName &p_name, const Variant &p_value) {
	if (p_value.get_type() == Variant::NIL) {
		_metadata.erase(p_name);
		return;
	}
	_metadata[p_name] = p_value;
}

Variant Object::get_meta(const StringName &p_name) const {
	if (!_metadata.has(p_name)) {
		return Variant();
	}
	return _metadata[p_name];
}

bool Object::has_meta(const StringName &p_name) const {
	return _metadata.has(p_name);
}

/**
 * notification()
 * 
 * Dispatcher for the 120 FPS simulation heartbeat.
 * Optimized for high-frequency Warp kernel updates by minimizing virtual call depth.
 */
void Object::notification(int p_notification, bool p_reversed) {
	if (p_reversed) {
		if (_script_instance) {
			_script_instance->notification(p_notification);
		}
		_notification(p_notification);
	} else {
		_notification(p_notification);
		if (_script_instance) {
			_script_instance->notification(p_notification);
		}
	}
}

Error Object::connect(const StringName &p_signal, Object *p_target, const StringName &p_method, uint32_t p_flags) {
	ERR_FAIL_NULL_V(p_target, ERR_INVALID_PARAMETER);
	
	Connection c;
	c.target = p_target;
	c.method = p_method;
	c.flags = p_flags;

	_connections[p_signal].push_back(c);
	return OK;
}

void Object::disconnect(const StringName &p_signal, Object *p_target, const StringName &p_method) {
	if (!_connections.has(p_signal)) return;

	List<Connection> &cl = _connections[p_signal];
	for (typename List<Connection>::Element *E = cl.front(); E; E = E->next()) {
		if (E->get().target == p_target && E->get().method == p_method) {
			cl.erase(E);
			return;
		}
	}
}

/**
 * emit_signal()
 * 
 * High-performance signal emission. 
 * Uses const Variant ** to achieve Zero-Copy argument passing, allowing 
 * millions of BigIntCore/FixedMathCore updates to propagate per frame.
 */
void Object::emit_signal(const StringName &p_name, const Variant **p_args, int p_argcount) {
	if (_block_signals || !_connections.has(p_name)) return;

	List<Connection> &cl = _connections[p_name];
	for (typename List<Connection>::Element *E = cl.front(); E; E = E->next()) {
		Connection &c = E->get();
		Callable::CallError ce;
		c.target->callp(c.method, p_args, p_argcount, ce);
	}
}

void Object::set_script(const Variant &p_script) {
	if (_script_instance) {
		memdelete(_script_instance);
		_script_instance = nullptr;
	}

	// In a full implementation, this creates the language-specific instance
	// such as GDScriptInstance or CSharpInstance.
}

Variant Object::get_script() const {
	if (_script_instance) {
		return _script_instance->get_script();
	}
	return Variant();
}

--- END OF FILE core/object/object.cpp ---
