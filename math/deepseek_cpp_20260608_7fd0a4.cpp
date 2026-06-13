// File 302: modules/vienna/register_types.h
// Registration header for the ViennaPhysicsEngine module.
// Declares initialisation and shutdown functions for the module system.

#ifndef VIENNA_REGISTER_TYPES_H
#define VIENNA_REGISTER_TYPES_H

#include "modules/register_module_types.h"

void initialize_vienna_module(ModuleInitializationLevel p_level);
void uninitialize_vienna_module(ModuleInitializationLevel p_level);

#endif // VIENNA_REGISTER_TYPES_H