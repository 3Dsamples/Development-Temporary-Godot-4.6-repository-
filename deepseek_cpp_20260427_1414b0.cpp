// File 216: modules/newton/register_types.cpp
// Newton Dynamics 4.0 module registration implementation.
// Registers all Newton classes with Godot's ClassDB so they can be used
// from GDScript, C#, and the editor.

#include "register_types.h"

// Core
#include "src/world/newton_world.h"
#include "src/bodies/newton_body.h"
#include "src/materials/newton_material.h"

// Collision shapes
#include "src/collision/newton_collision.h"
#include "src/collision/newton_collision_cylinder.h"
#include "src/collision/newton_collision_cone.h"
#include "src/collision/newton_compound_collision.h"
#include "src/collision/newton_convex_hull.h"
#include "src/collision/newton_heightfield_collision.h"

// Joints
#include "src/joints/newton_joint.h"
#include "src/joints/newton_ball_joint.h"
#include "src/joints/newton_hinge_joint.h"
#include "src/joints/newton_slider_joint.h"
#include "src/joints/newton_universal_joint.h"
#include "src/joints/newton_corkscrew_joint.h"
#include "src/joints/newton_fixed_joint.h"
#include "src/joints/newton_kinematic_controller.h"

// Solver / islands
#include "src/solver/newton_solver.h"
#include "src/solver/newton_island.h"

// Vehicles
#include "src/vehicles/newton_vehicle.h"

// Physics server (optional – registers itself later if used as drop‑in)
#include "src/servers/newton_physics_server_3d.h"

#ifdef TOOLS_ENABLED
#include "core/config/engine.h"
#endif

void initialize_newton_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// Core
		GDREGISTER_CLASS(newton::NewtonWorld);
		GDREGISTER_CLASS(newton::NewtonBody);
		GDREGISTER_CLASS(newton::NewtonMaterial);

		// Collision shapes
		GDREGISTER_CLASS(newton::NewtonCollision);
		GDREGISTER_CLASS(newton::NewtonCollisionSphere);
		GDREGISTER_CLASS(newton::NewtonCollisionBox);
		GDREGISTER_CLASS(newton::NewtonCollisionCapsule);
		GDREGISTER_CLASS(newton::NewtonCollisionCylinder);
		GDREGISTER_CLASS(newton::NewtonCollisionCone);
		GDREGISTER_CLASS(newton::NewtonCompoundCollision);
		GDREGISTER_CLASS(newton::NewtonCollisionConvexHull);
		GDREGISTER_CLASS(newton::NewtonHeightfieldCollision);

		// Joints
		GDREGISTER_CLASS(newton::NewtonJoint);
		GDREGISTER_CLASS(newton::NewtonBallJoint);
		GDREGISTER_CLASS(newton::NewtonHingeJoint);
		GDREGISTER_CLASS(newton::NewtonSliderJoint);
		GDREGISTER_CLASS(newton::NewtonUniversalJoint);
		GDREGISTER_CLASS(newton::NewtonCorkscrewJoint);
		GDREGISTER_CLASS(newton::NewtonFixedJoint);
		GDREGISTER_CLASS(newton::NewtonKinematicController);

		// Solver
		GDREGISTER_CLASS(newton::NewtonSolver);
		GDREGISTER_CLASS(newton::NewtonIsland);

		// Vehicles
		GDREGISTER_CLASS(newton::NewtonVehicle);
	}
}

void uninitialize_newton_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// Nothing to clean up explicitly.
	}
}