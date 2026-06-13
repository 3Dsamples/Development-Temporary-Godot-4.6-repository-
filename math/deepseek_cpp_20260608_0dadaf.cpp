// File 326: modules/wicked/wicked.h
// Umbrella header for the WickedEngine‑inspired Bullet physics module.
// Provides high‑performance rigid body dynamics, collision shapes,
// joints, and constraint solving via the Bullet Physics library.
// Designed for native integration into Godot 4.6 and interoperable
// with the existing Gaia, Genesis, Newton, and Vienna modules.

#ifndef WICKED_H
#define WICKED_H

#include "src/core/wicked_types.h"
#include "src/core/wicked_constants.h"
#include "src/world/wicked_world.h"
#include "src/bodies/wicked_body.h"
#include "src/collision/wicked_shape.h"
#include "src/joints/wicked_joint.h"
#include "src/joints/wicked_ball_joint.h"
#include "src/joints/wicked_hinge_joint.h"
#include "src/joints/wicked_slider_joint.h"
#include "src/joints/wicked_fixed_joint.h"
#include "src/joints/wicked_cone_twist_joint.h"
#include "src/joints/wicked_generic_6dof_joint.h"
#include "src/solver/wicked_solver.h"
#include "src/solver/wicked_island.h"
#include "src/materials/wicked_material.h"
#include "src/vehicles/wicked_raycast_vehicle.h"
#include "src/servers/wicked_physics_server_3d.h"
#include "src/utils/wicked_mesh_loader.h"

void initialize_wicked_module(ModuleInitializationLevel p_level);
void uninitialize_wicked_module(ModuleInitializationLevel p_level);

#endif // WICKED_H