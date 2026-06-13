// File 49: modules/genesis/genesis.h
// Umbrella header for the Genesis physics engine module
// This module extends Godot 4.6 with Genesis's advanced multi-solver physics engine,
// differentiable physics, and IPC collision coupling.

#ifndef GENESIS_H
#define GENESIS_H

// --- Core Macros & Configuration ---
#include "genesis_macros.h"

// --- Options System ---
#include "src/options/options_system.h"

// --- Types ---
#include "src/core/genesis_types.h"
#include "src/core/genesis_constants.h"

// --- Materials ---
#include "src/materials/material_base.h"
#include "src/materials/fem_material.h"
#include "src/materials/mpm_material.h"
#include "src/materials/pbd_material.h"
#include "src/materials/sph_material.h"
#include "src/materials/sf_material.h"

// --- Entities ---
#include "src/entities/base_entity.h"
#include "src/entities/rigid_entity.h"
#include "src/entities/fem_entity.h"
#include "src/entities/mpm_entity.h"
#include "src/entities/tool_entity.h"

// --- Solvers ---
#include "src/solvers/base_solver.h"
#include "src/solvers/rigid_solver.h"
#include "src/solvers/fem_solver.h"
#include "src/solvers/mpm_solver.h"
#include "src/solvers/pbd_solver.h"
#include "src/solvers/sph_solver.h"

// --- Collision ---
#include "src/collision/collider.h"
#include "src/collision/gjk.h"
#include "src/collision/ipc_coupler.h"

// --- Boundaries ---
#include "src/boundaries/boundary_conditions.h"

// --- Sensors ---
#include "src/sensors/base_sensor.h"
#include "src/sensors/camera_sensor.h"
#include "src/sensors/contact_force_sensor.h"
#include "src/sensors/imu_sensor.h"

// --- Grad (Differentiable Physics) ---
#include "src/grad/tensor.h"
#include "src/grad/creation_ops.h"

// --- States ---
#include "src/states/entity_state.h"
#include "src/states/solver_state.h"

#endif // GENESIS_H