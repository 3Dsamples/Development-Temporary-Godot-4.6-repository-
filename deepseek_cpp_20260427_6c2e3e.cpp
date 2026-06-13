// File 263: modules/integration/register_types.cpp
// Registers the UnifiedPhysicsServer3D and helper classes with Godot's
// ClassDB so they can be used throughout the engine and GDScript.

#include "register_types.h"

#include "unified_physics_server_3d.h"

// Include the individual module registration headers so subordinate classes
// are also registered.  These are called from the unified init.
#include "../../../gaia/register_types.h"
#include "../../../genesis/register_types.h"
#include "../../../newton/register_types.h"

#ifdef TOOLS_ENABLED
#include "core/config/engine.h"
#endif

void initialize_integration_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_CORE) {
		// Register the three sub‑modules first (they register at SCENE level in their own init).
		initialize_gaia_module(p_level);
		initialize_genesis_module(p_level);
		initialize_newton_module(p_level);
	}
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// Register the unified physics server class.
		GDREGISTER_CLASS(unified::UnifiedPhysicsServer3D);
	}
}

void uninitialize_integration_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// Nothing specific to clean up.
	}
	if (p_level == MODULE_INITIALIZATION_LEVEL_CORE) {
		uninitialize_newton_module(p_level);
		uninitialize_genesis_module(p_level);
		uninitialize_gaia_module(p_level);
	}
}