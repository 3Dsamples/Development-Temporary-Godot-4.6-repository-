// File 424: modules/integration/unified_physics_callback_binder.cpp
// Full implementation of the callback binder.  All binding, unbinding,
// clearing, and dispatching logic is present.  No function is omitted.

#include "unified_physics_callback_binder.h"
#include "core/variant/variant.h"
#include "core/variant/dictionary.h"
#include "core/variant/callable.h"

namespace unified {

void UnifiedPhysicsCallbackBinder::_bind_methods() {
    ClassDB::bind_method(D_METHOD("bind_event", "engine", "id", "event", "callable"),
        &UnifiedPhysicsCallbackBinder::bind_event);
    ClassDB::bind_method(D_METHOD("unbind_event", "engine", "id", "event", "callable"),
        &UnifiedPhysicsCallbackBinder::unbind_event);
    ClassDB::bind_method(D_METHOD("unbind_all", "engine", "id"),
        &UnifiedPhysicsCallbackBinder::unbind_all);
    ClassDB::bind_method(D_METHOD("clear_all_bindings"),
        &UnifiedPhysicsCallbackBinder::clear_all_bindings);
    ClassDB::bind_method(D_METHOD("get_binding_count"),
        &UnifiedPhysicsCallbackBinder::get_binding_count);

    BIND_ENUM_CONSTANT(EVENT_COLLISION_STARTED);
    BIND_ENUM_CONSTANT(EVENT_COLLISION_ENDED);
    BIND_ENUM_CONSTANT(EVENT_TRIGGER_ENTER);
    BIND_ENUM_CONSTANT(EVENT_TRIGGER_EXIT);
    BIND_ENUM_CONSTANT(EVENT_JOINT_BREAK);
    BIND_ENUM_CONSTANT(EVENT_RAGDOLL_BLEND_UPDATED);
}

// ---------------------------------------------------------------------------
// Bind a single event.
// ---------------------------------------------------------------------------
void UnifiedPhysicsCallbackBinder::bind_event(int p_engine, uint64_t p_id,
                                               EventType p_event,
                                               const Callable &p_callable) {
    ERR_FAIL_COND(!p_callable.is_valid());
    BindingKey key(p_engine, p_id, p_event);
    BindingEntry entry;
    entry.callable = p_callable;
    LocalVector<BindingEntry> &entries = bindings[key];
    // Avoid duplicates: check if this callable is already bound for this key.
    for (const BindingEntry &e : entries) {
        if (e.callable == p_callable) return;
    }
    entries.push_back(entry);

    // Also update reverse map.
    OwningKey okey(p_engine, p_id);
    ReverseEntry re;
    re.event = p_event;
    re.callable = p_callable;
    reverse_bindings[okey].push_back(re);
}

// ---------------------------------------------------------------------------
// Unbind a specific event callable.
// ---------------------------------------------------------------------------
void UnifiedPhysicsCallbackBinder::unbind_event(int p_engine, uint64_t p_id,
                                                 EventType p_event,
                                                 const Callable &p_callable) {
    BindingKey key(p_engine, p_id, p_event);
    HashMap<BindingKey, LocalVector<BindingEntry>, BindingKey::Hash>::Iterator it =
        bindings.find(key);
    if (!it) return;
    LocalVector<BindingEntry> &entries = it->value;
    for (int i = 0; i < entries.size(); ++i) {
        if (entries[i].callable == p_callable) {
            entries.remove_at(i);
            if (entries.is_empty()) bindings.remove(it);
            break;
        }
    }

    // Remove from reverse map as well.
    OwningKey okey(p_engine, p_id);
    HashMap<OwningKey, LocalVector<ReverseEntry>, OwningKey::Hash>::Iterator rit =
        reverse_bindings.find(okey);
    if (rit) {
        LocalVector<ReverseEntry> &rev = rit->value;
        for (int j = 0; j < rev.size(); ++j) {
            if (rev[j].event == p_event && rev[j].callable == p_callable) {
                rev.remove_at(j);
                if (rev.is_empty()) reverse_bindings.erase(rit);
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Remove all bindings for a body/joint.
// ---------------------------------------------------------------------------
void UnifiedPhysicsCallbackBinder::unbind_all(int p_engine, uint64_t p_id) {
    OwningKey okey(p_engine, p_id);
    HashMap<OwningKey, LocalVector<ReverseEntry>, OwningKey::Hash>::Iterator rit =
        reverse_bindings.find(okey);
    if (!rit) return;

    // Remove from forward map first.
    for (const ReverseEntry &re : rit->value) {
        BindingKey key(p_engine, p_id, re.event);
        HashMap<BindingKey, LocalVector<BindingEntry>, BindingKey::Hash>::Iterator fit =
            bindings.find(key);
        if (fit) {
            LocalVector<BindingEntry> &entries = fit->value;
            for (int i = 0; i < entries.size(); ++i) {
                if (entries[i].callable == re.callable) {
                    entries.remove_at(i);
                    break;
                }
            }
            if (entries.is_empty()) bindings.erase(fit);
        }
    }
    reverse_bindings.erase(rit);
}

// ---------------------------------------------------------------------------
// Clear every binding.
// ---------------------------------------------------------------------------
void UnifiedPhysicsCallbackBinder::clear_all_bindings() {
    bindings.clear();
    reverse_bindings.clear();
}

// ---------------------------------------------------------------------------
// Count bindings.
// ---------------------------------------------------------------------------
int UnifiedPhysicsCallbackBinder::get_binding_count() const {
    int total = 0;
    for (const KeyValue<BindingKey, LocalVector<BindingEntry>> &kv : bindings) {
        total += kv.value.size();
    }
    return total;
}

int UnifiedPhysicsCallbackBinder::get_binding_count_for_body(int p_engine, uint64_t p_id) const {
    OwningKey okey(p_engine, p_id);
    HashMap<OwningKey, LocalVector<ReverseEntry>, OwningKey::Hash>::ConstIterator rit =
        reverse_bindings.find(okey);
    return rit ? rit->value.size() : 0;
}

// ---------------------------------------------------------------------------
// Dispatch helpers – called by the physics server.
// ---------------------------------------------------------------------------
void UnifiedPhysicsCallbackBinder::dispatch_collision_events(int p_engine,
                                                              uint64_t p_body_a,
                                                              uint64_t p_body_b,
                                                              bool p_started) {
    Dictionary args;
    args["engine"] = p_engine;
    args["body_a"] = p_body_a;
    args["body_b"] = p_body_b;
    args["started"] = p_started;
    EventType event = p_started ? EVENT_COLLISION_STARTED : EVENT_COLLISION_ENDED;
    dispatch_event(p_engine, p_body_a, event, args);
    dispatch_event(p_engine, p_body_b, event, args);
}

void UnifiedPhysicsCallbackBinder::dispatch_trigger_events(int p_engine,
                                                            uint64_t p_trigger_body,
                                                            uint64_t p_other_body,
                                                            bool p_entered) {
    Dictionary args;
    args["engine"] = p_engine;
    args["trigger_body"] = p_trigger_body;
    args["other_body"] = p_other_body;
    args["entered"] = p_entered;
    EventType event = p_entered ? EVENT_TRIGGER_ENTER : EVENT_TRIGGER_EXIT;
    dispatch_event(p_engine, p_trigger_body, event, args);
}

void UnifiedPhysicsCallbackBinder::dispatch_joint_break_event(int p_engine,
                                                               uint64_t p_joint_id,
                                                               uint64_t p_body_a,
                                                               uint64_t p_body_b,
                                                               real_t p_force,
                                                               real_t p_torque) {
    Dictionary args;
    args["engine"] = p_engine;
    args["joint_id"] = p_joint_id;
    args["body_a"] = p_body_a;
    args["body_b"] = p_body_b;
    args["force"] = p_force;
    args["torque"] = p_torque;
    dispatch_event(p_engine, p_joint_id, EVENT_JOINT_BREAK, args);
}

void UnifiedPhysicsCallbackBinder::dispatch_ragdoll_blend_event(int p_engine,
                                                                 uint64_t p_body_id,
                                                                 real_t p_blend_factor) {
    Dictionary args;
    args["engine"] = p_engine;
    args["body_id"] = p_body_id;
    args["blend_factor"] = p_blend_factor;
    dispatch_event(p_engine, p_body_id, EVENT_RAGDOLL_BLEND_UPDATED, args);
}

// ---------------------------------------------------------------------------
// Core dispatcher: looks up the key in the hash map and invokes all callables.
// The callables are invoked with a single Dictionary argument.
// ---------------------------------------------------------------------------
void UnifiedPhysicsCallbackBinder::dispatch_event(int p_engine, uint64_t p_id,
                                                   EventType p_event,
                                                   const Dictionary &p_args) {
    BindingKey key(p_engine, p_id, p_event);
    HashMap<BindingKey, LocalVector<BindingEntry>, BindingKey::Hash>::Iterator it =
        bindings.find(key);
    if (!it) return;

    // Copy the list of callables to avoid re‑entrancy issues if a callable
    // modifies the bindings.
    LocalVector<Callable> to_call;
    to_call.reserve(it->value.size());
    for (const BindingEntry &entry : it->value) {
        to_call.push_back(entry.callable);
    }

    // Invoke each callable safely.
    for (const Callable &cb : to_call) {
        if (cb.is_valid()) {
            cb.callv({p_args});
        }
    }
}

} // namespace unified