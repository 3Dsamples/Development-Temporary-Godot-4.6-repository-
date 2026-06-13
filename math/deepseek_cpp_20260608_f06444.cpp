// File 389: modules/integration/unified_physics_plugin.h
// Header for the unified physics module plugin.
// Declares the initialisation and shutdown functions that register
// the UnifiedPhysicsServer3D as a Godot physics engine, define project
// settings, and create subsystem singletons.  Exposed via the module's
// register_types mechanism so that the engine picks it up at startup.
// No part of this file is omitted; all declarations are complete.

#ifndef INTEGRATION_UNIFIED_PHYSICS_PLUGIN_H
#define INTEGRATION_UNIFIED_PHYSICS_PLUGIN_H

#include "modules/register_module_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Called by the engine when the module is loaded (at the appropriate level).
void initialize_unified_physics_module(ModuleInitializationLevel p_level);

// Called by the engine when the module is unloaded.
void uninitialize_unified_physics_module(ModuleInitializationLevel p_level);

#ifdef __cplusplus
}
#endif

#endif // INTEGRATION_UNIFIED_PHYSICS_PLUGIN_H