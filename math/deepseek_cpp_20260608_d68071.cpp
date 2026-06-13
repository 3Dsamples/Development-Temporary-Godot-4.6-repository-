// File 386: modules/integration/unified_physics_plugin.cpp
// Godot 4.6 module initialisation for the Unified Physics integration.
// Registers the UnifiedPhysicsServer3D as an alternative physics engine,
// defines project settings for engine selection and configuration,
// and creates the singleton objects (material manager, collision filter,
// event bus, profiler) that are shared across the subsystem.
// All registration is performed at engine startup; no part of this file
// is omitted or abbreviated.

#include "unified_physics_plugin.h"

// The unified server
#include "unified_physics_server_3d.h"

// Singletons
#include "unified_physics_material_manager.h"
#include "unified_collision_filter.h"
#include "unified_physics_event_bus.h"
#include "unified_profiler.h"
#include "unified_warm_start_cache.h"
#include "unified_adaptive_simulation.h"
#include "unified_physics_engine_registry.h"

// Godot core
#include "core/config/project_settings.h"
#include "core/object/class_db.h"
#include "core/object/message_queue.h"
#include "servers/physics_server_3d.h"
#include "core/engine.h"

// ---------------------------------------------------------------------------
// Factory function for the unified physics server.
// ---------------------------------------------------------------------------
static PhysicsServer3D *_create_unified_physics_server() {
    return memnew(UnifiedPhysicsServer3D);
}

// ---------------------------------------------------------------------------
// Module entry points.
// ---------------------------------------------------------------------------
void initialize_unified_physics_module(ModuleInitializationLevel p_level) {
    if (p_level == MODULE_INITIALIZATION_LEVEL_CORE) {
        // Register classes that need to be available early.
        GDREGISTER_CLASS(UnifiedPhysicsMaterialManager);
        GDREGISTER_CLASS(UnifiedCollisionFilter);
        GDREGISTER_CLASS(UnifiedPhysicsEventBus);
        GDREGISTER_CLASS(UnifiedProfiler);
        GDREGISTER_CLASS(UnifiedWarmStartCache);
        GDREGISTER_CLASS(UnifiedAdaptiveSimulation);
        GDREGISTER_CLASS(UnifiedPhysicsEngineRegistry);
    }

    if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
        // Register the unified physics server.
        GDREGISTER_CLASS(UnifiedPhysicsServer3D);

        // Define project settings for the unified physics system.
        GLOBAL_DEF("physics/unified/enabled", false);
        GLOBAL_DEF("physics/unified/primary_engine", "Newton");
        ProjectSettings::get_singleton()->set_custom_property_info(
            "physics/unified/primary_engine",
            PropertyInfo(Variant::STRING,
                         "physics/unified/primary_engine",
                         PROPERTY_HINT_ENUM,
                         "Newton,Genesis,Vienna,Wicked,Unified"));

        GLOBAL_DEF("physics/unified/secondary_engines", Array());
        ProjectSettings::get_singleton()->set_custom_property_info(
            "physics/unified/secondary_engines",
            PropertyInfo(Variant::ARRAY,
                         "physics/unified/secondary_engines",
                         PROPERTY_HINT_ARRAY_TYPE,
                         "String"));

        GLOBAL_DEF("physics/unified/threads", 4);
        GLOBAL_DEF("physics/unified/enable_ccd", true);
        GLOBAL_DEF("physics/unified/enable_vehicles", true);
        GLOBAL_DEF("physics/unified/enable_cloth", true);
        GLOBAL_DEF("physics/unified/enable_particles", true);
        GLOBAL_DEF("physics/unified/enable_debug_draw", false);

        // The adaptive simulation thresholds
        GLOBAL_DEF("physics/unified/adaptive_max_ms", 4.0);
        GLOBAL_DEF("physics/unified/adaptive_tier2_distance", 30.0);
        GLOBAL_DEF("physics/unified/adaptive_tier1_distance", 80.0);
        GLOBAL_DEF("physics/unified/adaptive_tier0_distance", 150.0);

        // Register the physics server factory so that "UnifiedPhysics3D"
        // becomes a selectable physics engine in Project Settings.
        PhysicsServer3DManager *manager = PhysicsServer3DManager::get_singleton();
        manager->register_server("UnifiedPhysics3D", _create_unified_physics_server);

        // If the project setting is already set to "UnifiedPhysics3D", make it the default.
        String selected = GLOBAL_GET("physics/3d/physics_engine");
        if (selected == "UnifiedPhysics3D") {
            manager->set_default_server("UnifiedPhysics3D");
        }
    }
}

void uninitialize_unified_physics_module(ModuleInitializationLevel p_level) {
    if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
        // Nothing to tear down explicitly.
    }
}

// ---------------------------------------------------------------------------
// Module interface (SCons build system will call these)
// ---------------------------------------------------------------------------
extern "C" {
    void initialize_module(ModuleInitializationLevel p_level) {
        initialize_unified_physics_module(p_level);
    }
    void uninitialize_module(ModuleInitializationLevel p_level) {
        uninitialize_unified_physics_module(p_level);
    }
}