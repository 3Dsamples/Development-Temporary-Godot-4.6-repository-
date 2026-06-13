// File 321: modules/vienna/src/callbacks/vienna_contact_callbacks.h
// High‑performance collision event system for ViennaPhysicsEngine.
// Bodies can have callbacks attached that fire when they collide with
// another body.  The system uses a signal‑like interface with fast
// function pointers, avoiding heap allocations during the physics step.

#ifndef VIENNA_CALLBACKS_CONTACT_H
#define VIENNA_CALLBACKS_CONTACT_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "../core/vienna_types.h"
#include "../core/vienna_constants.h"

namespace vienna {

class ViennaBody;

class ViennaContactCallbacks : public RefCounted {
    GDCLASS(ViennaContactCallbacks, RefCounted);

public:
    // Callback type: invoked with the two body IDs and the world‑space
    // contact data (point, normal, impulse, …).  The callback may be
    // a static function, a member function, or a Callable.
    typedef void (*ContactCallback)(body_id, body_id, const vec3 &point, const vec3 &normal, real_t impulse, void *);

    ViennaContactCallbacks() {}

    // Register a callback for collisions involving a specific body.
    void bind(body_id p_body, ContactCallback p_callback, void *p_userdata = nullptr) {
        bound_callbacks[p_body] = { p_callback, p_userdata };
    }

    // Register a global callback that fires for every collision in the world.
    void bind_global(ContactCallback p_callback, void *p_userdata = nullptr) {
        global_callbacks.push_back({ p_callback, p_userdata });
    }

    // Remove callbacks.
    void unbind(body_id p_body) { bound_callbacks.erase(p_body); }
    void unbind_global(ContactCallback p_callback, void *p_userdata) {
        for (int i = global_callbacks.size() - 1; i >= 0; --i) {
            if (global_callbacks[i].callback == p_callback &&
                global_callbacks[i].userdata == p_userdata) {
                global_callbacks.remove_at(i);
            }
        }
    }

    // Called internally by the world after the solver step to notify
    // all matching callbacks about each contact point that was solved.
    void notify_contact(body_id a, body_id b, const vec3 &point, const vec3 &normal, real_t impulse) {
        // Invoke per‑body callbacks for `a` and `b`.
        if (bound_callbacks.has(a)) {
            const Bound &bnd = bound_callbacks[a];
            bnd.callback(a, b, point, normal, impulse, bnd.userdata);
        }
        if (bound_callbacks.has(b)) {
            const Bound &bnd = bound_callbacks[b];
            bnd.callback(b, a, point, normal, impulse, bnd.userdata);
        }
        // Invoke global callbacks.
        for (const Global &g : global_callbacks) {
            g.callback(a, b, point, normal, impulse, g.userdata);
        }
    }

    // Clear all callbacks.
    void clear() {
        bound_callbacks.clear();
        global_callbacks.clear();
    }

protected:
    static void _bind_methods() {
        // For GDScript usage, a Callable interface can be added later.
    }

private:
    struct Bound {
        ContactCallback callback;
        void *userdata;
    };
    struct Global {
        ContactCallback callback;
        void *userdata;
    };

    HashMap<body_id, Bound> bound_callbacks;
    LocalVector<Global> global_callbacks;
};

} // namespace vienna

#endif // VIENNA_CALLBACKS_CONTACT_H