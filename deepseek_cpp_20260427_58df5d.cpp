// File 267: modules/vienna/vienna.h
// Umbrella header for ViennaPhysicsEngine module.
// Provides cloth, particles, rigid bodies, joints, collision detection,
// and constraint solving, all adapted for Godot 4.6 and interoperable
// with Gaia, Genesis, and Newton dynamics.

#ifndef VIENNA_H
#define VIENNA_H

// --- Core ---
#include "src/core/vienna_types.h"
#include "src/core/vienna_constants.h"

// --- World ---
#include "src/world/vienna_world.h"

// --- Bodies ---
#include "src/bodies/vienna_body.h"

// --- Shapes ---
#include "src/collision/vienna_shape.h"
#include "src/collision/vienna_compound_shape.h"
#include "src/collision/vienna_heightfield.h"
#include "src/collision/vienna_trimesh.h"

// --- Joints ---
#include "src/joints/vienna_joint.h"
#include "src/joints/vienna_ball_joint.h"
#include "src/joints/vienna_hinge_joint.h"
#include "src/joints/vienna_slider_joint.h"
#include "src/joints/vienna_fixed_joint.h"
#include "src/joints/vienna_distance_joint.h"
#include "src/joints/vienna_rope_joint.h"

// --- Cloth ---
#include "src/cloth/vienna_cloth.h"
#include "src/cloth/vienna_cloth_solver.h"

// --- Particles ---
#include "src/particles/vienna_particle.h"
#include "src/particles/vienna_particle_system.h"

// --- Solver ---
#include "src/solver/vienna_solver.h"
#include "src/solver/vienna_island.h"

// --- Materials ---
#include "src/materials/vienna_material.h"

// --- Utilities ---
#include "src/utils/vienna_serializer.h"
#include "src/utils/vienna_debug_draw.h"
#include "src/utils/vienna_mesh_loader.h"

// Module entry points
void initialize_vienna_module(ModuleInitializationLevel p_level);
void uninitialize_vienna_module(ModuleInitializationLevel p_level);

#endif // VIENNA_H