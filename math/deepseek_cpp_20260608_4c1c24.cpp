// File 354: modules/wicked/register_types.cpp
#include "register_types.h"

// Core
#include "src/core/wicked_types.h"
#include "src/core/wicked_constants.h"

// World
#include "src/world/wicked_world.h"

// Bodies
#include "src/bodies/wicked_body.h"

// Shapes
#include "src/collision/wicked_shape.h"

// Joints
#include "src/joints/wicked_joint.h"
#include "src/joints/wicked_ball_joint.h"
#include "src/joints/wicked_hinge_joint.h"
#include "src/joints/wicked_slider_joint.h"
#include "src/joints/wicked_fixed_joint.h"
#include "src/joints/wicked_cone_twist_joint.h"
#include "src/joints/wicked_generic_6dof_joint.h"

// Solver
#include "src/solver/wicked_solver.h"
#include "src/solver/wicked_island.h"

// Materials
#include "src/materials/wicked_material.h"

// Vehicles
#include "src/vehicles/wicked_raycast_vehicle.h"

// Server
#include "src/servers/wicked_physics_server_3d.h"

#ifdef TOOLS_ENABLED
#include "core/config/engine.h"
#endif

void initialize_wicked_module(ModuleInitializationLevel p_level) {
    if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
        // Core
        GDREGISTER_CLASS(wicked::WickedWorld);
        GDREGISTER_CLASS(wicked::WickedBody);
        // Shapes
        GDREGISTER_CLASS(wicked::WickedShape);
        GDREGISTER_CLASS(wicked::WickedShapeSphere);
        GDREGISTER_CLASS(wicked::WickedShapeBox);
        GDREGISTER_CLASS(wicked::WickedShapeCapsule);
        GDREGISTER_CLASS(wicked::WickedShapeCylinder);
        GDREGISTER_CLASS(wicked::WickedShapeCone);
        GDREGISTER_CLASS(wicked::WickedShapeConvexHull);
        GDREGISTER_CLASS(wicked::WickedShapeTriMesh);
        GDREGISTER_CLASS(wicked::WickedShapeHeightfield);
        GDREGISTER_CLASS(wicked::WickedShapeCompound);
        // Joints
        GDREGISTER_CLASS(wicked::WickedJoint);
        GDREGISTER_CLASS(wicked::WickedBallJoint);
        GDREGISTER_CLASS(wicked::WickedHingeJoint);
        GDREGISTER_CLASS(wicked::WickedSliderJoint);
        GDREGISTER_CLASS(wicked::WickedFixedJoint);
        GDREGISTER_CLASS(wicked::WickedConeTwistJoint);
        GDREGISTER_CLASS(wicked::WickedGeneric6DOFJoint);
        // Solver
        GDREGISTER_CLASS(wicked::WickedSolver);
        GDREGISTER_CLASS(wicked::WickedIsland);
        // Materials
        GDREGISTER_CLASS(wicked::WickedMaterial);
        // Vehicles
        GDREGISTER_CLASS(wicked::WickedRaycastVehicle);
        // Server
        GDREGISTER_CLASS(wicked::WickedPhysicsServer3D);
    }
}

void uninitialize_wicked_module(ModuleInitializationLevel p_level) {}