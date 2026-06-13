// File 423: modules/integration/unified_physics_callback_binder.h
// High‑performance event‑to‑signal bridge that maps engine‑unique body/joint
// IDs to user‑defined Callables (GDScript, C#).  It intercepts collision,
// trigger, joint‑break, and ragdoll‑blend events from the unified physics
// pipeline and dispatches them to the connected callables.  Maintains a
// lock‑free dispatch table for asynchronous safety and zero heap allocation
// during the physics step.

#ifndef INTEGRATION_UNIFIED_PHYSICS_CALLBACK_BINDER_H
#define INTEGRATION_UNIFIED_PHYSICS_CALLBACK_BINDER_H

#include "core/object/ref_counted.h"
#include "core/variant/variant.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace unified {

class UnifiedPhysicsCallbackBinder : public RefCounted {
    GDCLASS(UnifiedPhysicsCallbackBinder, RefCounted);

public:
    // -------------------------------------------------------------------
    // Event types that can be bound to a body or joint ID.
    // -------------------------------------------------------------------
    enum EventType {
        EVENT_COLLISION_STARTED = 0,
        EVENT_COLLISION_ENDED,
        EVENT_TRIGGER_ENTER,
        EVENT_TRIGGER_EXIT,
        EVENT_JOINT_BREAK,
        EVENT_RAGDOLL_BLEND_UPDATED
    };

    // -------------------------------------------------------------------
    // Bind a specific event for a body (or joint) to a callable.
    // `p_engine` is the engine index (0=Newton,1=Genesis,2=Vienna,3=Wicked).
    // `p_id` is the body or joint ID in that engine.
    // `p_callable` will be invoked with a Dictionary argument.
    // -------------------------------------------------------------------
    void bind_event(int p_engine, uint64_t p_id, EventType p_event, const Callable &p_callable);

    // Remove a previously bound event.
    void unbind_event(int p_engine, uint64_t p_id, EventType p_event, const Callable &p_callable);

    // Remove all bindings for a given body/joint.
    void unbind_all(int p_engine, uint64_t p_id);

    // Clear every binding.
    void clear_all_bindings();

    // -------------------------------------------------------------------
    // Called internally by the physics server after each step.
    // The server will pass pre‑formatted event arrays (dictionaries).
    // This is faster than querying per‑body in script.
    // -------------------------------------------------------------------
    void dispatch_collision_events(int p_engine, uint64_t p_body_a, uint64_t p_body_b,
                                   bool p_started);
    void dispatch_trigger_events(int p_engine, uint64_t p_trigger_body,
                                 uint64_t p_other_body, bool p_entered);
    void dispatch_joint_break_event(int p_engine, uint64_t p_joint_id,
                                    uint64_t p_body_a, uint64_t p_body_b,
                                    real_t p_force, real_t p_torque);
    void dispatch_ragdoll_blend_event(int p_engine, uint64_t p_body_id,
                                      real_t p_blend_factor);

    // -------------------------------------------------------------------
    // Debug: number of registered bindings.
    // -------------------------------------------------------------------
    int get_binding_count() const;
    int get_binding_count_for_body(int p_engine, uint64_t p_id) const;

protected:
    static void _bind_methods();

private:
    // -------------------------------------------------------------------
    // Key for the dispatch table: combines engine, body ID, and event type.
    // -------------------------------------------------------------------
    struct BindingKey {
        int      engine;
        uint64_t id;
        EventType event;

        BindingKey() : engine(0), id(0), event(EVENT_COLLISION_STARTED) {}
        BindingKey(int eng, uint64_t body_id, EventType evt)
            : engine(eng), id(body_id), event(evt) {}

        bool operator==(const BindingKey &o) const {
            return engine == o.engine && id == o.id && event == o.event;
        }

        struct Hash {
            uint64_t operator()(const BindingKey &k) const {
                uint64_t h = (uint64_t(k.engine) << 56) | (k.id & 0xFFFFFFFFFFFFFFULL);
                h ^= (uint64_t(k.event) << 48);
                h ^= h >> 33;
                h *= 0xff51afd7ed558ccdULL;
                h ^= h >> 33;
                h *= 0xc4ceb9fe1a85ec53ULL;
                h ^= h >> 33;
                return h;
            }
        };
    };

    // -------------------------------------------------------------------
    // Per‑binding stored callables (can be multiple for the same key).
    // -------------------------------------------------------------------
    struct BindingEntry {
        Callable callable;
    };

    // Map from BindingKey to a list of bound callables.
    HashMap<BindingKey, LocalVector<BindingEntry>, BindingKey::Hash> bindings;

    // Reverse map: (engine, id) -> list of (event, callable) for efficient unbind.
    struct OwningKey {
        int engine;
        uint64_t id;
        OwningKey() : engine(0), id(0) {}
        OwningKey(int eng, uint64_t bid) : engine(eng), id(bid) {}
        bool operator==(const OwningKey &o) const { return engine==o.engine && id==o.id; }
        struct Hash {
            uint64_t operator()(const OwningKey &k) const {
                return (uint64_t(k.engine) << 56) | (k.id & 0xFFFFFFFFFFFFFFULL);
            }
        };
    };
    struct ReverseEntry {
        EventType event;
        Callable  callable;
    };
    HashMap<OwningKey, LocalVector<ReverseEntry>, OwningKey::Hash> reverse_bindings;

    // -------------------------------------------------------------------
    // Thread‑safe dispatch helper: copies the callables and invokes them
    // outside the hot path (avoids locking).
    // -------------------------------------------------------------------
    void dispatch_event(int p_engine, uint64_t p_id, EventType p_event, const Dictionary &p_args);
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_PHYSICS_CALLBACK_BINDER_H