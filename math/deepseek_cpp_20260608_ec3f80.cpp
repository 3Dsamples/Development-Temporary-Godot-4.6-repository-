// File 303: modules/vienna/register_types.cpp
// Registers all ViennaPhysicsEngine classes with Godot's ClassDB.
// After this, the types are available in GDScript, C#, and the editor.

#include "register_types.h"

// Core
#include "src/core/vienna_types.h"
#include "src/core/vienna_constants.h"

// World
#include "src/world/vienna_world.h"

// Bodies
#include "src/bodies/vienna_body.h"

// Shapes (collision primitives)
#include "src/collision/vienna_shape.h"
#include "src/collision/vienna_compound_shape.h"
#include "src/collision/vienna_heightfield.h"
#include "src/collision/vienna_trimesh.h"

// Joints
#include "src/joints/vienna_joint.h"
#include "src/joints/vienna_ball_joint.h"
#include "src/joints/vienna_hinge_joint.h"
#include "src/joints/vienna_slider_joint.h"
#include "src/joints/vienna_fixed_joint.h"
#include "src/joints/vienna_distance_joint.h"
#include "src/joints/vienna_rope_joint.h"

// Solver
#include "src/solver/vienna_solver.h"
#include "src/solver/vienna_island.h"
#include "src/solver/vienna_parallel_solver.h"

// Materials
#include "src/materials/vienna_material.h"

// Cloth
#include "src/cloth/vienna_cloth.h"
#include "src/cloth/vienna_cloth_solver.h"

// Particles
#include "src/particles/vienna_particle.h"
#include "src/particles/vienna_particle_system.h"

// Utilities (if they need to be resources)
#include "src/utils/vienna_debug_draw.h"
#include "src/utils/vienna_serializer.h"

// Server (the drop‑in physics server)
#include "src/servers/vienna_physics_server_3d.h"

#ifdef TOOLS_ENABLED
#include "core/config/engine.h"
#endif

void initialize_vienna_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// World
		GDREGISTER_CLASS(vienna::ViennaWorld);

		// Bodies
		GDREGISTER_CLASS(vienna::ViennaBody);

		// Shapes
		GDREGISTER_CLASS(vienna::ViennaShape);
		GDREGISTER_CLASS(vienna::ViennaShapeSphere);
		GDREGISTER_CLASS(vienna::ViennaShapeBox);
		GDREGISTER_CLASS(vienna::ViennaShapeCapsule);
		GDREGISTER_CLASS(vienna::ViennaShapeCylinder);
		GDREGISTER_CLASS(vienna::ViennaShapeCone);
		GDREGISTER_CLASS(vienna::ViennaShapeConvexHull);
		GDREGISTER_CLASS(vienna::ViennaCompoundShape);
		GDREGISTER_CLASS(vienna::ViennaHeightfield);
		GDREGISTER_CLASS(vienna::ViennaTriMesh);

		// Joints
		GDREGISTER_CLASS(vienna::ViennaJoint);
		GDREGISTER_CLASS(vienna::ViennaBallJoint);
		GDREGISTER_CLASS(vienna::ViennaHingeJoint);
		GDREGISTER_CLASS(vienna::ViennaSliderJoint);
		GDREGISTER_CLASS(vienna::ViennaFixedJoint);
		GDREGISTER_CLASS(vienna::ViennaDistanceJoint);
		GDREGISTER_CLASS(vienna::ViennaRopeJoint);

		// Solver & Islands
		GDREGISTER_CLASS(vienna::ViennaSolver);
		GDREGISTER_CLASS(vienna::ViennaIsland);
		GDREGISTER_CLASS(vienna::ViennaParallelSolver);

		// Materials
		GDREGISTER_CLASS(vienna::ViennaMaterial);

		// Cloth
		GDREGISTER_CLASS(vienna::ViennaCloth);
		GDREGISTER_CLASS(vienna::ViennaClothSolver);

		// Particles
		GDREGISTER_CLASS(vienna::ViennaParticleSystem);   // ViennaParticle is a struct, not a resource, so it's not registered as a class

		// Utility nodes / resources
		GDREGISTER_CLASS(vienna::ViennaDebugDraw);

		// Server
		GDREGISTER_CLASS(vienna::ViennaPhysicsServer3D);
	}
}

void uninitialize_vienna_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// Nothing to clean up explicitly
	}
}