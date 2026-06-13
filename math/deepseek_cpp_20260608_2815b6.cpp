// File 172: modules/newton/newton.h
// Umbrella header for the Newton Dynamics 4.0 module.
// Provides high‑performance rigid body dynamics, collision detection,
// constraint solving, joints, and continuous collision.
// Designed as a drop‑in replacement for the default Godot physics server
// and fully interoperable with Gaia/Genesis solvers.

#ifndef NEWTON_H
#define NEWTON_H

// --- Core types and utilities ---
#include "src/core/newton_types.h"
#include "src/core/newton_constants.h"

// --- World management ---
#include "src/world/newton_world.h"

// --- Bodies ---
#include "src/bodies/newton_body.h"

// --- Collision shapes ---
#include "src/collision/newton_collision.h"
#include "src/collision/newton_compound_collision.h"

// --- Joints ---
#include "src/joints/newton_joint.h"
#include "src/joints/newton_hinge_joint.h"
#include "src/joints/newton_slider_joint.h"
#include "src/joints/newton_universal_joint.h"
#include "src/joints/newton_corkscrew_joint.h"
#include "src/joints/newton_kinematic_controller.h"

// --- Solver ---
#include "src/solver/newton_solver.h"
#include "src/solver/newton_island.h"

// --- Materials ---
#include "src/materials/newton_material.h"

// --- Vehicles ---
#include "src/vehicles/newton_vehicle.h"

// --- Serialisation / Utils ---
#include "src/utils/newton_serializer.h"

// Newton module entry points
void initialize_newton_module(ModuleInitializationLevel p_level);
void uninitialize_newton_module(ModuleInitializationLevel p_level);

#endif // NEWTON_H