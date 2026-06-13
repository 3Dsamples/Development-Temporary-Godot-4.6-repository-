// File 262: modules/integration/register_types.h
// Integration module – combines Newton, Genesis, and Gaia into the
// UnifiedPhysicsServer3D.  This header declares init/uninit functions.

#ifndef INTEGRATION_REGISTER_TYPES_H
#define INTEGRATION_REGISTER_TYPES_H

#include "modules/register_module_types.h"

void initialize_integration_module(ModuleInitializationLevel p_level);
void uninitialize_integration_module(ModuleInitializationLevel p_level);

#endif // INTEGRATION_REGISTER_TYPES_H