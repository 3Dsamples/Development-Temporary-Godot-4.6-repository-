--- START OF FILE src/register_types.h ---

#ifndef BIG_NUMBER_REGISTER_TYPES_H
#define BIG_NUMBER_REGISTER_TYPES_H

#include <godot_cpp/core/class_db.hpp>

using namespace godot;

// ============================================================================
// GDExtension Lifecycle Hooks
// ============================================================================

/**
 * initialize_bignumber_module()
 * 
 * Invoked by the Godot engine when the extension is loaded.
 * Registers the Arbitrary-Precision (BigNumber) and Deterministic (FixedNumber)
 * mathematical backends into the engine's reflection database (ClassDB).
 */
void initialize_bignumber_module(ModuleInitializationLevel p_level);

/**
 * uninitialize_bignumber_module()
 * 
 * Invoked by the Godot engine when the extension is unloaded.
 * Handles the graceful teardown of any static memory pools or shared 
 * ECS/Warp registry resources allocated by the Universal Solver.
 */
void uninitialize_bignumber_module(ModuleInitializationLevel p_level);

#endif // BIG_NUMBER_REGISTER_TYPES_H

--- END OF FILE src/register_types.h ---
