// File 86: modules/genesis/register_types.cpp

#include "register_types.h"

#include "src/materials/material_base.h"
#include "src/materials/fem_material.h"
#include "src/materials/mpm_material.h"
#include "src/materials/pbd_material.h"
#include "src/materials/sph_material.h"
#include "src/materials/sf_material.h"

#include "src/entities/base_entity.h"
#include "src/entities/rigid_entity.h"
#include "src/entities/fem_entity.h"
#include "src/entities/mpm_entity.h"
#include "src/entities/tool_entity.h"

#include "src/solvers/base_solver.h"
#include "src/solvers/rigid_solver.h"
#include "src/solvers/fem_solver.h"
#include "src/solvers/mpm_solver.h"
#include "src/solvers/sph_solver.h"
#include "src/solvers/sf_solver.h"
#include "src/solvers/pbd_solver.h"
#include "src/solvers/kinematic_solver.h"

#include "src/collision/collider.h"
#include "src/collision/gjk.h"
#include "src/collision/ipc_coupler.h"

#include "src/boundaries/boundary_conditions.h"

#include "src/sensors/base_sensor.h"
#include "src/sensors/camera_sensor.h"
#include "src/sensors/contact_force_sensor.h"
#include "src/sensors/imu_sensor.h"

#include "src/states/entity_state.h"
#include "src/states/solver_state.h"

#include "src/grad/tensor.h"
#include "src/grad/creation_ops.h"

#ifdef TOOLS_ENABLED
#include "core/config/engine.h"
#endif

void initialize_genesis_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// Materials
		GDREGISTER_CLASS(genesis::GenesisMaterial);
		GDREGISTER_CLASS(genesis::FEMMaterial);
		GDREGISTER_CLASS(genesis::MPMMaterial);
		GDREGISTER_CLASS(genesis::PBDMaterial);
		GDREGISTER_CLASS(genesis::SPHMaterial);
		GDREGISTER_CLASS(genesis::SFMaterial);

		// Entities
		GDREGISTER_CLASS(genesis::BaseEntity);
		GDREGISTER_CLASS(genesis::RigidEntity);
		GDREGISTER_CLASS(genesis::FEMEntity);
		GDREGISTER_CLASS(genesis::MPMEntity);
		GDREGISTER_CLASS(genesis::ToolEntity);

		// Solvers
		GDREGISTER_CLASS(genesis::BaseSolver);
		GDREGISTER_CLASS(genesis::RigidSolver);
		GDREGISTER_CLASS(genesis::FEMSolver);
		GDREGISTER_CLASS(genesis::MPMSolver);
		GDREGISTER_CLASS(genesis::SPHSolver);
		GDREGISTER_CLASS(genesis::SFSolver);
		GDREGISTER_CLASS(genesis::GenesisPBDSolver);
		GDREGISTER_CLASS(genesis::KinematicSolver);

		// Collision
		// (Collider and GJK are not registered as Godot classes yet, but could be)
		// Boundary conditions, sensors, states, grad are also resources
		GDREGISTER_CLASS(genesis::BaseSensor);
		GDREGISTER_CLASS(genesis::CameraSensor);
		GDREGISTER_CLASS(genesis::ContactForceSensor);
		GDREGISTER_CLASS(genesis::IMUSensor);

		GDREGISTER_CLASS(genesis::EntityState);
		GDREGISTER_CLASS(genesis::SolverState);
	}
}

void uninitialize_genesis_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		// Nothing to clean up explicitly
	}
}