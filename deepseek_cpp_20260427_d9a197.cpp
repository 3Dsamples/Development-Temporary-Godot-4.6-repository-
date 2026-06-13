// File 339: modules/wicked/src/materials/wicked_material.h
// WickedMaterial – per‑material collision properties (friction, restitution,
// rolling friction, spinning friction, softness). Materials are combined by
// the solver when two bodies collide, using maximum values for safety.
// All accessors are inline for performance.

#ifndef WICKED_MATERIALS_WICKED_MATERIAL_H
#define WICKED_MATERIALS_WICKED_MATERIAL_H

#include "core/object/ref_counted.h"
#include "../core/wicked_types.h"
#include "../core/wicked_constants.h"

namespace wicked {

class WickedMaterial : public RefCounted {
    GDCLASS(WickedMaterial, RefCounted);

public:
    WickedMaterial() :
        static_friction(DEFAULT_FRICTION),
        dynamic_friction(DEFAULT_FRICTION),
        restitution(DEFAULT_RESTITUTION),
        rolling_friction(DEFAULT_ROLLING_FRICTION),
        spinning_friction(DEFAULT_SPINNING_FRICTION),
        softness(DEFAULT_SOFTNESS) {}

    // --- Friction ---
    inline void set_static_friction(real_t p_val) { static_friction = CLAMP(p_val, 0.0, 10.0); }
    inline real_t get_static_friction() const { return static_friction; }

    inline void set_dynamic_friction(real_t p_val) { dynamic_friction = CLAMP(p_val, 0.0, static_friction); }
    inline real_t get_dynamic_friction() const { return dynamic_friction; }

    // --- Restitution ---
    inline void set_restitution(real_t p_val) { restitution = CLAMP(p_val, 0.0, 1.0); }
    inline real_t get_restitution() const { return restitution; }

    // --- Rolling friction (torque opposing rolling) ---
    inline void set_rolling_friction(real_t p_val) { rolling_friction = MAX(p_val, 0.0); }
    inline real_t get_rolling_friction() const { return rolling_friction; }

    // --- Spinning friction (torque opposing twisting) ---
    inline void set_spinning_friction(real_t p_val) { spinning_friction = MAX(p_val, 0.0); }
    inline real_t get_spinning_friction() const { return spinning_friction; }

    // --- Softness (compliance, similar to CFM) ---
    inline void set_softness(real_t p_val) { softness = MAX(p_val, 0.0); }
    inline real_t get_softness() const { return softness; }

    // Combine two materials, returning the computed contact properties.
    // The combination uses the maximum of each individual property.
    static inline void combine(const WickedMaterial *p_a, const WickedMaterial *p_b,
                                real_t &r_friction, real_t &r_restitution,
                                real_t &r_rolling, real_t &r_spinning, real_t &r_softness) {
        if (p_a && p_b) {
            r_friction   = MAX(p_a->dynamic_friction, p_b->dynamic_friction);
            r_restitution = MAX(p_a->restitution, p_b->restitution);
            r_rolling    = MAX(p_a->rolling_friction, p_b->rolling_friction);
            r_spinning   = MAX(p_a->spinning_friction, p_b->spinning_friction);
            r_softness   = MAX(p_a->softness, p_b->softness);
        } else if (p_a) {
            r_friction   = p_a->dynamic_friction;
            r_restitution = p_a->restitution;
            r_rolling    = p_a->rolling_friction;
            r_spinning   = p_a->spinning_friction;
            r_softness   = p_a->softness;
        } else if (p_b) {
            r_friction   = p_b->dynamic_friction;
            r_restitution = p_b->restitution;
            r_rolling    = p_b->rolling_friction;
            r_spinning   = p_b->spinning_friction;
            r_softness   = p_b->softness;
        } else {
            r_friction   = DEFAULT_FRICTION;
            r_restitution = DEFAULT_RESTITUTION;
            r_rolling    = DEFAULT_ROLLING_FRICTION;
            r_spinning   = DEFAULT_SPINNING_FRICTION;
            r_softness   = DEFAULT_SOFTNESS;
        }
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_static_friction", "friction"), &WickedMaterial::set_static_friction);
        ClassDB::bind_method(D_METHOD("get_static_friction"), &WickedMaterial::get_static_friction);
        ClassDB::bind_method(D_METHOD("set_dynamic_friction", "friction"), &WickedMaterial::set_dynamic_friction);
        ClassDB::bind_method(D_METHOD("get_dynamic_friction"), &WickedMaterial::get_dynamic_friction);
        ClassDB::bind_method(D_METHOD("set_restitution", "restitution"), &WickedMaterial::set_restitution);
        ClassDB::bind_method(D_METHOD("get_restitution"), &WickedMaterial::get_restitution);
        ClassDB::bind_method(D_METHOD("set_rolling_friction", "friction"), &WickedMaterial::set_rolling_friction);
        ClassDB::bind_method(D_METHOD("get_rolling_friction"), &WickedMaterial::get_rolling_friction);
        ClassDB::bind_method(D_METHOD("set_spinning_friction", "friction"), &WickedMaterial::set_spinning_friction);
        ClassDB::bind_method(D_METHOD("get_spinning_friction"), &WickedMaterial::get_spinning_friction);
        ClassDB::bind_method(D_METHOD("set_softness", "softness"), &WickedMaterial::set_softness);
        ClassDB::bind_method(D_METHOD("get_softness"), &WickedMaterial::get_softness);

        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "static_friction", PROPERTY_HINT_RANGE, "0,10,0.01"), "set_static_friction", "get_static_friction");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dynamic_friction", PROPERTY_HINT_RANGE, "0,10,0.01"), "set_dynamic_friction", "get_dynamic_friction");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "restitution", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_restitution", "get_restitution");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rolling_friction", PROPERTY_HINT_RANGE, "0,10,0.01"), "set_rolling_friction", "get_rolling_friction");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "spinning_friction", PROPERTY_HINT_RANGE, "0,10,0.01"), "set_spinning_friction", "get_spinning_friction");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "softness", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_softness", "get_softness");
    }

private:
    real_t static_friction;
    real_t dynamic_friction;
    real_t restitution;
    real_t rolling_friction;
    real_t spinning_friction;
    real_t softness;
};

} // namespace wicked

#endif // WICKED_MATERIALS_WICKED_MATERIAL_H