// File 84: modules/genesis/register_types.h

#ifndef GENESIS_REGISTER_TYPES_H
#define GENESIS_REGISTER_TYPES_H

#include "modules/register_module_types.h"

void initialize_genesis_module(ModuleInitializationLevel p_level);
void uninitialize_genesis_module(ModuleInitializationLevel p_level);

#endif // GENESIS_REGISTER_TYPES_H