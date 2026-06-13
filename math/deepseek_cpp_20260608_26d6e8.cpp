// File 215: modules/newton/register_types.h
// Newton Dynamics 4.0 module registration header.
// Declares initialisation and cleanup functions called by Godot engine.

#ifndef NEWTON_REGISTER_TYPES_H
#define NEWTON_REGISTER_TYPES_H

#include "modules/register_module_types.h"

void initialize_newton_module(ModuleInitializationLevel p_level);
void uninitialize_newton_module(ModuleInitializationLevel p_level);

#endif // NEWTON_REGISTER_TYPES_H