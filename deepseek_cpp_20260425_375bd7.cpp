// File 45: modules/gaia/register_types.cpp

#include "register_types.h"

#include "src/framework/sim_framework.h" // SimulationWorld (RefCounted)

#ifdef TOOLS_ENABLED
#include "core/config/engine.h"
#endif

void initialize_gaia_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// Register classes with Godot's ClassDB.
		// Physics server extension and other nodes can be added here later.
		GDREGISTER_CLASS(gaia::framework::SimulationWorld);
	}
}

void uninitialize_gaia_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// Cleanup if needed
	}
}