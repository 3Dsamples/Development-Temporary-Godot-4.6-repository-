--- START OF FILE src/register_types.cpp ---

#include "src/register_types.h"
#include "src/big_number.h"

#include "core/object/class_db.h"
#include "core/config/engine.h"

/**
 * initialize_bignumber_module
 * 
 * Sets up the high-performance math simulation backend.
 * This level ensures that RefCounted types are ready before any
 * SceneTree or Physics logic attempts to utilize hyper-scale values.
 */
void initialize_bignumber_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    // Register the Arbitrary-Precision Integer Engine (Economy/Idle Games)
    ClassDB::register_class<BigNumber>();

    // Register the Deterministic Fixed-Point Math Engine (Galactic Physics/Orbital Mechanics)
    ClassDB::register_class<FixedNumber>();

    // Architectural Note: Here we would initialize the Universal Solver's
    // global EnTT registry and Warp kernel dispatcher if they require
    // a persistent singleton state during the simulation lifecycle.
}

/**
 * uninitialize_bignumber_module
 * 
 * Handles the graceful teardown of the module.
 * Since BigNumber and FixedNumber are RefCounted, Godot's memory management
 * handles the cleanup of individual instances automatically.
 */
void uninitialize_bignumber_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    
    // Cleanup global simulation buffers here to prevent memory leaks 
    // in long-running scientific computation environments.
}

// GDExtension Entry Point (Using Godot 4.6 internal standard)
extern "C" {
GDExtensionBool GDE_EXPORT bignumber_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, const GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
    godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

    init_obj.register_initializer(initialize_bignumber_module);
    init_obj.register_terminator(uninitialize_bignumber_module);
    init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

    return init_obj.init();
}
}

--- END OF FILE src/register_types.cpp ---
