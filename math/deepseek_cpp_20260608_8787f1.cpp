// File 381: modules/integration/unified_physics_engine_registry.h
// UnifiedPhysicsEngineRegistry – factory and manager for all supported
// physics backends (Newton, Genesis, Vienna, Wicked, and the Godot default).
// Provides a single interface to create, configure, and step each engine,
// and to swap between them at runtime.  Registers each engine's types with
// Godot's ClassDB and provides global settings via ProjectSettings.
// All public methods are static and thread‑safe.

#ifndef INTEGRATION_UNIFIED_PHYSICS_ENGINE_REGISTRY_H
#define INTEGRATION_UNIFIED_PHYSICS_ENGINE_REGISTRY_H

#include "core/config/project_settings.h"
#include "core/object/class_db.h"
#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/string/ustring.h"

// Engine headers (for factory functions)
namespace newton   { class NewtonWorld; }
namespace genesis  { class GenesisWorld; }
namespace vienna   { class ViennaWorld; }
namespace wicked   { class WickedWorld; }

namespace unified {

class UnifiedPhysicsEngineRegistry : public RefCounted {
    GDCLASS(UnifiedPhysicsEngineRegistry, RefCounted);

public:
    // Supported physics engines
    enum EngineID {
        ENGINE_GODOT_DEFAULT = 0,
        ENGINE_NEWTON         = 1,
        ENGINE_GENESIS        = 2,
        ENGINE_VIENNA         = 3,
        ENGINE_WICKED         = 4,
        ENGINE_UNIFIED        = 5,   // the combined, multi‑engine pipeline
        ENGINE_COUNT
    };

    // Descriptive information about each engine
    struct EngineInfo {
        String name;
        String description;
        String version;
        bool    supports_soft_body;
        bool    supports_cloth;
        bool    supports_particles;
        bool    supports_vehicles;
        bool    supports_ccd;
    };

private:
    static EngineInfo engine_infos[ENGINE_COUNT];
    static bool       _initialised;

public:
    UnifiedPhysicsEngineRegistry() {}

    // One‑time initialisation: registers all engine classes and settings.
    // Called by module initialisation.
    static void initialise() {
        if (_initialised) return;
        _initialised = true;

        // Populate static info
        engine_infos[ENGINE_GODOT_DEFAULT] = {"GodotPhysics3D", "Default Godot physics", "4.6", false, false, false, false, false};
        engine_infos[ENGINE_NEWTON]        = {"Newton Dynamics", "High‑performance rigid body solver", "4.0", false, false, false, true, true};
        engine_infos[ENGINE_GENESIS]       = {"Genesis", "Multi‑solver: FEM, MPM, SPH, PBD, SF", "1.0", true, true, true, false, false};
        engine_infos[ENGINE_VIENNA]        = {"Vienna Physics", "Rigid body, cloth, particles", "1.0", false, true, true, false, false};
        engine_infos[ENGINE_WICKED]        = {"Wicked Engine", "Rigid body (Bullet‑like)", "1.0", false, false, false, true, true};
        engine_infos[ENGINE_UNIFIED]       = {"Unified Physics", "Gaia + Genesis + Newton + Vienna + Wicked", "1.0", true, true, true, true, true};

        // Define project settings
        GLOBAL_DEF("physics/unified/enabled", true);
        GLOBAL_DEF("physics/unified/primary_engine", "Newton");
        GLOBAL_DEF("physics/unified/secondary_engines", Array());
        ProjectSettings::get_singleton()->set_custom_property_info(
            "physics/unified/primary_engine",
            PropertyInfo(Variant::STRING, "physics/unified/primary_engine",
                         PROPERTY_HINT_ENUM, "GodotDefault,Newton,Genesis,Vienna,Wicked,Unified"));
        GLOBAL_DEF("physics/unified/threads", 4);
        GLOBAL_DEF("physics/unified/enable_ccd", true);
        GLOBAL_DEF("physics/unified/enable_vehicles", true);
        GLOBAL_DEF("physics/unified/enable_cloth", true);
        GLOBAL_DEF("physics/unified/enable_particles", true);
        GLOBAL_DEF("physics/unified/enable_debug_draw", false);
    }

    // Get the info for an engine.
    static const EngineInfo &get_engine_info(EngineID p_id) {
        ERR_FAIL_INDEX_V((int)p_id, ENGINE_COUNT, engine_infos[0]);
        return engine_infos[p_id];
    }

    // Create a new world instance for the given engine.
    // Returns nullptr if the engine is not available (e.g., module not compiled).
    static void *create_world(EngineID p_id) {
        switch (p_id) {
            case ENGINE_NEWTON:  return memnew(newton::NewtonWorld);
            case ENGINE_GENESIS: return memnew(genesis::GenesisWorld);
            case ENGINE_VIENNA:  return memnew(vienna::ViennaWorld);
            case ENGINE_WICKED:  return memnew(wicked::WickedWorld);
            case ENGINE_UNIFIED: // unified world is handled externally
            default: return nullptr;
        }
    }

    // Destroy a world created by create_world.
    static void destroy_world(EngineID p_id, void *p_world) {
        if (!p_world) return;
        switch (p_id) {
            case ENGINE_NEWTON:  memdelete(static_cast<newton::NewtonWorld *>(p_world)); break;
            case ENGINE_GENESIS: memdelete(static_cast<genesis::GenesisWorld *>(p_world)); break;
            case ENGINE_VIENNA:  memdelete(static_cast<vienna::ViennaWorld *>(p_world)); break;
            case ENGINE_WICKED:  memdelete(static_cast<wicked::WickedWorld *>(p_world)); break;
            default: break;
        }
    }

    // Step function pointer type (for uniform dispatch).
    typedef void (*WorldStepFunc)(void *, real_t);

    // Get the step function for an engine (each engine implements its own step).
    static WorldStepFunc get_step_function(EngineID p_id) {
        switch (p_id) {
            case ENGINE_NEWTON:  return newton_step;
            case ENGINE_GENESIS: return genesis_step;
            case ENGINE_VIENNA:  return vienna_step;
            case ENGINE_WICKED:  return wicked_step;
            default: return nullptr;
        }
    }

    // Read the current primary engine from project settings.
    static EngineID get_primary_engine() {
        String name = GLOBAL_GET("physics/unified/primary_engine");
        return engine_from_name(name);
    }

    // Resolve engine name to ID.
    static EngineID engine_from_name(const String &p_name) {
        for (int i = 0; i < ENGINE_COUNT; ++i) {
            if (engine_infos[i].name == p_name) return (EngineID)i;
        }
        return ENGINE_NEWTON; // default
    }

    // Check if an engine is enabled in the project settings.
    static bool is_engine_enabled(EngineID p_id) {
        if (p_id == ENGINE_GODOT_DEFAULT) return false; // never use Godot default when unified is active
        if (!GLOBAL_GET("physics/unified/enabled")) return false;
        if (p_id == ENGINE_UNIFIED) return true;
        String primary = GLOBAL_GET("physics/unified/primary_engine");
        if (primary == engine_infos[p_id].name) return true;
        Array secondary = GLOBAL_GET("physics/unified/secondary_engines");
        for (int i = 0; i < secondary.size(); ++i) {
            if (secondary[i].operator String() == engine_infos[p_id].name) return true;
        }
        return false;
    }

    // Get the number of threads to use for physics.
    static int get_num_threads() {
        return MAX(GLOBAL_GET("physics/unified/threads").operator int(), 1);
    }

private:
    // Engine‑specific step wrappers (cast void* and call world->step(dt))
    static void newton_step(void *w, real_t dt) {
        static_cast<newton::NewtonWorld *>(w)->step(dt);
    }
    static void genesis_step(void *w, real_t dt) {
        static_cast<genesis::GenesisWorld *>(w)->simulate_step(dt);
    }
    static void vienna_step(void *w, real_t dt) {
        static_cast<vienna::ViennaWorld *>(w)->step(dt);
    }
    static void wicked_step(void *w, real_t dt) {
        static_cast<wicked::WickedWorld *>(w)->step(dt);
    }
};

// Static member definitions
UnifiedPhysicsEngineRegistry::EngineInfo UnifiedPhysicsEngineRegistry::engine_infos[ENGINE_COUNT];
bool UnifiedPhysicsEngineRegistry::_initialised = false;

} // namespace unified

#endif // INTEGRATION_UNIFIED_PHYSICS_ENGINE_REGISTRY_H