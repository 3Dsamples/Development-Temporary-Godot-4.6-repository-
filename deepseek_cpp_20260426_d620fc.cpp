// File 151: modules/gaia/genesis_plugin.cpp
// Plugin entry point for combined Gaia + Genesis physics engine extension.
// Registers all classes, solvers, sensors, and nodes with Godot's ClassDB.
// This file is compiled once at startup; the module initialisation functions
// are called by the engine during scene, editor, and server initialisation.

#include "register_types.h"          // Gaia registration header
#include "../genesis/register_types.h" // Genesis registration header

#include "core/config/project_settings.h"
#include "core/object/class_db.h"

// Forward declare the two module entry points from register_types.cpp
extern void initialize_gaia_module(ModuleInitializationLevel p_level);
extern void uninitialize_gaia_module(ModuleInitializationLevel p_level);

extern void initialize_genesis_module(ModuleInitializationLevel p_level);
extern void uninitialize_genesis_module(ModuleInitializationLevel p_level);

// Combined initialiser called by Godot's module system.
void initialize_gaia_genesis_module(ModuleInitializationLevel p_level) {
	initialize_gaia_module(p_level);
	initialize_genesis_module(p_level);

	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// Set default physics parameters in ProjectSettings if desired.
		// GLOBAL_DEF("physics/gaia/vbd_iterations", 50);
		// GLOBAL_DEF("physics/genesis/solver_iterations", 10);
	}
}

void uninitialize_gaia_genesis_module(ModuleInitializationLevel p_level) {
	uninitialize_genesis_module(p_level);
	uninitialize_gaia_module(p_level);
}