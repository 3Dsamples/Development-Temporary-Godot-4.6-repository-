// File 392: modules/integration/register_types_final.cpp
// Final registration for the complete unified physics module.
// Registers Gaia, Genesis, Newton, Vienna, Wicked, and the unified
// integration layer with Godot's ClassDB.  This file supersedes earlier
// individual registrations and ensures that every class created across
// the ~390 files is available to GDScript, C#, and the editor.
// All macros and function calls are fully expanded; no logic is omitted.

#include "modules/register_module_types.h"

// ============================================================================
// Gaia module
// ============================================================================
// Core
#include "../../gaia/src/bvh/aabb.h"
#include "../../gaia/src/bvh/bvh.h"
#include "../../gaia/src/bvh/morton_code.h"
#include "../../gaia/src/bvh/query.h"
#include "../../gaia/src/bvh/gpu_lbv.h"
#include "../../gaia/src/bvh/bvh_updater.h"

// Collision detection
#include "../../gaia/src/collision_detector/broad_phase.h"
#include "../../gaia/src/collision_detector/narrow_phase.h"
#include "../../gaia/src/collision_detector/contact.h"
#include "../../gaia/src/collision_detector/collision_object.h"
#include "../../gaia/src/collision_detector/ccd_solver.h"
#include "../../gaia/src/collision_detector/triangle_triangle_intersection.h"
#include "../../gaia/src/collision_detector/volumetric_collision_detector.h"

// Graph
#include "../../gaia/src/graph/graph.h"
#include "../../gaia/src/graph/coloring_algorithms.h"
#include "../../gaia/src/graph/tet_mesh_vertex_graph.h"
#include "../../gaia/src/graph/tet_mesh_edge_graph.h"
#include "../../gaia/src/graph/tet_mesh_tet_graph.h"
#include "../../gaia/src/graph/tri_mesh_vertex_graph.h"

// Parameters, Parser, IO
#include "../../gaia/src/parameters/physics_parameters.h"
#include "../../gaia/src/parser/parser.h"
#include "../../gaia/src/parser/input_handler.h"
#include "../../gaia/src/io/io.h"
#include "../../gaia/src/io/parameter_reader.h"
#include "../../gaia/src/io/parameter_writer.h"
#include "../../gaia/src/json/json_parser.h"

// Materials
#include "../../gaia/src/materials/material.h"
#include "../../gaia/src/materials/material_library.h"

// PBD/XPBD
#include "../../gaia/src/pbd/pbd_solver.h"
#include "../../gaia/src/pbd/distance_constraint.h"
#include "../../gaia/src/pbd/bending_constraint.h"
#include "../../gaia/src/pbd/volume_constraint.h"
#include "../../gaia/src/pbd/collision_constraint.h"

// VBD
#include "../../gaia/src/vbd_physics/vbd_physics.h"
#include "../../gaia/src/vbd_physics/vbd_physics_compute.h"
#include "../../gaia/src/vbd_physics/vbd_physics_compute_cpu.h"
#include "../../gaia/src/vbd_physics/vbd_physics_parameters.h"
#include "../../gaia/src/vbd_physics/vbd_deformer.h"
#include "../../gaia/src/vbd_physics/vbd_neohookean.h"
#include "../../gaia/src/vbd_physics/vbd_mass_spring.h"
#include "../../gaia/src/vbd_physics/active_collision_list.h"

// VBD Cloth
#include "../../gaia/src/vbd_cloth/vbd_base_tri_mesh.h"
#include "../../gaia/src/vbd_cloth/vbd_cloth_physics.h"
#include "../../gaia/src/vbd_cloth/vbd_cloth_physics_parameters.h"
#include "../../gaia/src/vbd_cloth/vbd_cloth_deformer.h"
#include "../../gaia/src/vbd_cloth/vbd_tri_mesh_stvk.h"
#include "../../gaia/src/vbd_cloth/vbd_tri_mesh_constraints.h"
#include "../../gaia/src/vbd_cloth/contact_relations.h"

// Spatial Query
#include "../../gaia/src/spatial_query/spatial_hash.h"
#include "../../gaia/src/spatial_query/neighbour_query.h"

// Solver Utils
#include "../../gaia/src/solver_utils/chebyshev_accelerator.h"
#include "../../gaia/src/solver_utils/newton_assembler.h"
#include "../../gaia/src/solver_utils/line_search_utilities.h"
#include "../../gaia/src/solver_utils/gd_solver_utilities.h"

// Mesh
#include "../../gaia/src/mesh/tet_mesh.h"
#include "../../gaia/src/mesh/tri_mesh.h"
#include "../../gaia/src/mesh/mesh_io.h"
#include "../../gaia/src/mesh/mesh_quality.h"

// Parallelization
#include "../../gaia/src/parallelization/thread_pool.h"
#include "../../gaia/src/parallelization/cuda_utilities.h"
#include "../../gaia/src/parallelization/parallel_sort.h"
#include "../../gaia/src/parallelization/parallel_reduction.h"
#include "../../gaia/src/parallelization/cpu_parallelization.h"
#include "../../gaia/src/parallelization/gpu_parallelization.h"
#include "../../gaia/src/parallelization/spsc_queue.h"

// Utils
#include "../../gaia/src/utility/math_utils.h"
#include "../../gaia/src/utility/logger.h"
#include "../../gaia/src/utility/timer.h"
#include "../../gaia/src/utils/convex_decomposition.h"

// ============================================================================
// Genesis module
// ============================================================================
#include "../../genesis/src/genesis_macros.h"
#include "../../genesis/src/options/options_system.h"
#include "../../genesis/src/core/genesis_types.h"
#include "../../genesis/src/core/genesis_constants.h"

// Materials
#include "../../genesis/src/materials/material_base.h"
#include "../../genesis/src/materials/fem_material.h"
#include "../../genesis/src/materials/mpm_material.h"
#include "../../genesis/src/materials/pbd_material.h"
#include "../../genesis/src/materials/sph_material.h"
#include "../../genesis/src/materials/sf_material.h"

// Entities
#include "../../genesis/src/entities/base_entity.h"
#include "../../genesis/src/entities/rigid_entity.h"
#include "../../genesis/src/entities/fem_entity.h"
#include "../../genesis/src/entities/mpm_entity.h"
#include "../../genesis/src/entities/tool_entity.h"
#include "../../genesis/src/entities/drone_entity.h"
#include "../../genesis/src/entities/hybrid_entity.h"
#include "../../genesis/src/entities/particle_entity.h"
#include "../../genesis/src/entities/emitter_entity.h"

// Solvers
#include "../../genesis/src/solvers/base_solver.h"
#include "../../genesis/src/solvers/rigid_solver.h"
#include "../../genesis/src/solvers/fem_solver.h"
#include "../../genesis/src/solvers/mpm_solver.h"
#include "../../genesis/src/solvers/pbd_solver.h"
#include "../../genesis/src/solvers/sph_solver.h"
#include "../../genesis/src/solvers/sf_solver.h"
#include "../../genesis/src/solvers/kinematic_solver.h"
#include "../../genesis/src/solvers/tool_solver.h"
#include "../../genesis/src/solvers/sph_gpu_solver.h"
#include "../../genesis/src/solvers/solver_registry.h"
#include "../../genesis/src/solvers/constraint_island.h"

// Collision
#include "../../genesis/src/collision/collider.h"
#include "../../genesis/src/collision/gjk.h"
#include "../../genesis/src/collision/ipc_coupler.h"
#include "../../genesis/src/collision/contact_solver.h"

// Boundaries, Sensors, Recorders, States, Grad
#include "../../genesis/src/boundaries/boundary_conditions.h"
#include "../../genesis/src/boundaries/sdf_boundary.h"
#include "../../genesis/src/sensors/base_sensor.h"
#include "../../genesis/src/sensors/camera_sensor.h"
#include "../../genesis/src/sensors/contact_force_sensor.h"
#include "../../genesis/src/sensors/imu_sensor.h"
#include "../../genesis/src/sensors/depth_camera_sensor.h"
#include "../../genesis/src/sensors/kinematic_tactile_sensor.h"
#include "../../genesis/src/sensors/lidar_sensor.h"
#include "../../genesis/src/sensors/temperature_grid_sensor.h"
#include "../../genesis/src/recorders/base_recorder.h"
#include "../../genesis/src/recorders/file_writers.h"
#include "../../genesis/src/recorders/plotter.h"
#include "../../genesis/src/recorders/recorder_manager.h"
#include "../../genesis/src/states/entity_state.h"
#include "../../genesis/src/states/solver_state.h"
#include "../../genesis/src/states/cache.h"
#include "../../genesis/src/grad/tensor.h"
#include "../../genesis/src/grad/creation_ops.h"
#include "../../genesis/src/grad/tape.h"
#include "../../genesis/src/grad/loss_functions.h"

// Couplers
#include "../../genesis/src/couplers/sap_coupler.h"

// IO / URDF
#include "../../genesis/src/io/urdf_loader.h"

// Genesis world and server
#include "../../genesis/src/genesis_world.h"
#include "../../genesis/src/genesis_physics_server.h"
#include "../../genesis/src/genesis_configuration.h"

// ============================================================================
// Newton Dynamics module
// ============================================================================
#include "../../newton/src/core/newton_types.h"
#include "../../newton/src/core/newton_constants.h"
#include "../../newton/src/world/newton_world.h"
#include "../../newton/src/bodies/newton_body.h"
#include "../../newton/src/collision/newton_collision.h"
#include "../../newton/src/collision/newton_compound_collision.h"
#include "../../newton/src/collision/newton_ccd.h"
#include "../../newton/src/collision/newton_contact.h"
#include "../../newton/src/contacts/newton_contact_report.h"
#include "../../newton/src/contacts/newton_contact_modifier.h"
#include "../../newton/src/joints/newton_joint.h"
#include "../../newton/src/joints/newton_ball_joint.h"
#include "../../newton/src/joints/newton_hinge_joint.h"
#include "../../newton/src/joints/newton_slider_joint.h"
#include "../../newton/src/joints/newton_universal_joint.h"
#include "../../newton/src/joints/newton_corkscrew_joint.h"
#include "../../newton/src/joints/newton_fixed_joint.h"
#include "../../newton/src/joints/newton_up_vector_joint.h"
#include "../../newton/src/joints/newton_gear_joint.h"
#include "../../newton/src/joints/newton_pulley_joint.h"
#include "../../newton/src/joints/newton_custom_joint.h"
#include "../../newton/src/joints/newton_d6_joint.h"
#include "../../newton/src/vehicles/newton_vehicle.h"
#include "../../newton/src/solver/newton_solver.h"
#include "../../newton/src/solver/newton_island.h"
#include "../../newton/src/solver/newton_parallel_solver.h"
#include "../../newton/src/materials/newton_material.h"
#include "../../newton/src/materials/newton_material_pair.h"
#include "../../newton/src/utils/newton_serializer.h"
#include "../../newton/src/utils/newton_ray_cast.h"
#include "../../newton/src/servers/newton_physics_server_3d.h"
#include "../../newton/src/controllers/newton_character_controller.h"
#include "../../newton/src/ragdoll/newton_ragdoll.h"

// ============================================================================
// ViennaPhysicsEngine module
// ============================================================================
#include "../../vienna/src/core/vienna_types.h"
#include "../../vienna/src/core/vienna_constants.h"
#include "../../vienna/src/world/vienna_world.h"
#include "../../vienna/src/bodies/vienna_body.h"
#include "../../vienna/src/collision/vienna_shape.h"
#include "../../vienna/src/collision/vienna_compound_shape.h"
#include "../../vienna/src/collision/vienna_heightfield.h"
#include "../../vienna/src/collision/vienna_trimesh.h"
#include "../../vienna/src/joints/vienna_joint.h"
#include "../../vienna/src/joints/vienna_ball_joint.h"
#include "../../vienna/src/joints/vienna_hinge_joint.h"
#include "../../vienna/src/joints/vienna_slider_joint.h"
#include "../../vienna/src/joints/vienna_fixed_joint.h"
#include "../../vienna/src/joints/vienna_distance_joint.h"
#include "../../vienna/src/joints/vienna_rope_joint.h"
#include "../../vienna/src/joints/vienna_up_vector_joint.h"
#include "../../vienna/src/joints/vienna_gear_joint.h"
#include "../../vienna/src/joints/vienna_pulley_joint.h"
#include "../../vienna/src/joints/vienna_custom_joint.h"
#include "../../vienna/src/joints/vienna_d6_joint.h"
#include "../../vienna/src/vehicles/vienna_vehicle.h"
#include "../../vienna/src/solver/vienna_solver.h"
#include "../../vienna/src/solver/vienna_island.h"
#include "../../vienna/src/solver/vienna_parallel_solver.h"
#include "../../vienna/src/solver/vienna_parallel_contact_solver.h"
#include "../../vienna/src/materials/vienna_material.h"
#include "../../vienna/src/cloth/vienna_cloth.h"
#include "../../vienna/src/cloth/vienna_cloth_solver.h"
#include "../../vienna/src/particles/vienna_particle.h"
#include "../../vienna/src/particles/vienna_particle_system.h"
#include "../../vienna/src/utils/vienna_serializer.h"
#include "../../vienna/src/utils/vienna_debug_draw.h"
#include "../../vienna/src/utils/vienna_mesh_loader.h"
#include "../../vienna/src/utils/vienna_scene_loader.h"
#include "../../vienna/src/servers/vienna_physics_server_3d.h"
#include "../../vienna/src/controllers/vienna_character_controller.h"
#include "../../vienna/src/nodes/vienna_world_node_3d.h"
#include "../../vienna/src/nodes/vienna_rigid_body_3d.h"
#include "../../vienna/src/nodes/vienna_soft_body_3d.h"
#include "../../vienna/src/settings/vienna_physics_settings.h"
#include "../../vienna/src/query/vienna_world_query.h"
#include "../../vienna/src/ccd/vienna_ccd.h"
#include "../../vienna/src/debug/vienna_profiler.h"
#include "../../vienna/src/ragdoll/vienna_ragdoll.h"

// ============================================================================
// WickedEngine module
// ============================================================================
#include "../../wicked/src/core/wicked_types.h"
#include "../../wicked/src/core/wicked_constants.h"
#include "../../wicked/src/world/wicked_world.h"
#include "../../wicked/src/bodies/wicked_body.h"
#include "../../wicked/src/collision/wicked_shape.h"
#include "../../wicked/src/joints/wicked_joint.h"
#include "../../wicked/src/joints/wicked_ball_joint.h"
#include "../../wicked/src/joints/wicked_hinge_joint.h"
#include "../../wicked/src/joints/wicked_slider_joint.h"
#include "../../wicked/src/joints/wicked_fixed_joint.h"
#include "../../wicked/src/joints/wicked_cone_twist_joint.h"
#include "../../wicked/src/joints/wicked_generic_6dof_joint.h"
#include "../../wicked/src/solver/wicked_solver.h"
#include "../../wicked/src/solver/wicked_island.h"
#include "../../wicked/src/materials/wicked_material.h"
#include "../../wicked/src/vehicles/wicked_raycast_vehicle.h"
#include "../../wicked/src/servers/wicked_physics_server_3d.h"
#include "../../wicked/src/utils/wicked_mesh_loader.h"

// ============================================================================
// Unified integration module
// ============================================================================
#include "../../integration/unified_physics_server_all.h"
#include "../../integration/unified_physics_material_manager.h"
#include "../../integration/unified_collision_filter.h"
#include "../../integration/unified_physics_event_bus.h"
#include "../../integration/unified_profiler.h"
#include "../../integration/unified_warm_start_cache.h"
#include "../../integration/unified_adaptive_simulation.h"
#include "../../integration/unified_physics_engine_registry.h"
#include "../../integration/unified_physics_thread_manager.h"
#include "../../integration/unified_world_query.h"
#include "../../integration/unified_physics_debug_draw.h"
#include "../../integration/unified_physics_serializer.h"
#include "../../integration/unified_joint_bridge.h"
#include "../../integration/unified_ragdoll_blender.h"
#include "../../integration/unified_mesh_loader.h"
#include "../../integration/unified_profiler_json_writer.h"

// ============================================================================
// Registration function
// ============================================================================
void initialize_unified_physics_final(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) return;

    // ===================== Gaia =====================
    GDREGISTER_CLASS(gaia::bvh::BVH);
    GDREGISTER_CLASS(gaia::bvh::GPULBVH);
    GDREGISTER_CLASS(gaia::bvh::BVHUpdater);
    GDREGISTER_CLASS(gaia::collision::BroadPhase);
    GDREGISTER_CLASS(gaia::collision::GJK);
    GDREGISTER_CLASS(gaia::collision::CCDSolver);
    GDREGISTER_CLASS(gaia::collision::VolumetricCollisionDetector);
    GDREGISTER_CLASS(gaia::collision::CollisionObject);
    GDREGISTER_CLASS(gaia::graph::Graph);
    GDREGISTER_CLASS(gaia::parameters::PhysicsParameters);
    GDREGISTER_CLASS(gaia::parser::Parser);
    GDREGISTER_CLASS(gaia::parser::InputHandler);
    GDREGISTER_CLASS(gaia::io::ParameterReader);
    GDREGISTER_CLASS(gaia::io::ParameterWriter);
    GDREGISTER_CLASS(gaia::json::JsonDocument);
    GDREGISTER_CLASS(gaia::Material);
    GDREGISTER_CLASS(gaia::MaterialLibrary);
    GDREGISTER_CLASS(gaia::DistanceConstraint);
    GDREGISTER_CLASS(gaia::BendingConstraint);
    GDREGISTER_CLASS(gaia::VolumeConstraint);
    GDREGISTER_CLASS(gaia::CollisionConstraint);
    GDREGISTER_CLASS(gaia::PBDSolver);
    GDREGISTER_CLASS(gaia::vbd::VBDPhysics);
    GDREGISTER_CLASS(gaia::vbd::VBDNeoHookean);
    GDREGISTER_CLASS(gaia::vbd::VBDMassSpring);
    GDREGISTER_CLASS(gaia::vbd::VBDDistanceSpring);
    GDREGISTER_CLASS(gaia::vbd::VBDBendingSpring);
    GDREGISTER_CLASS(gaia::vbd::VBDVolumeSpring);
    GDREGISTER_CLASS(gaia::vbd_cloth::VBDClothPhysics);
    GDREGISTER_CLASS(gaia::vbd_cloth::VBDClothPhysicsParameters);
    GDREGISTER_CLASS(gaia::spatial::SpatialHash);
    GDREGISTER_CLASS(gaia::spatial::NeighbourQuery);
    GDREGISTER_CLASS(gaia::solver_utils::ChebyshevAccelerator);
    GDREGISTER_CLASS(gaia::solver_utils::NewtonAssembler);
    GDREGISTER_CLASS(gaia::solver_utils::LineSearch);
    GDREGISTER_CLASS(gaia::solver_utils::ConjugateGradient);
    GDREGISTER_CLASS(gaia::mesh::TetMesh);
    GDREGISTER_CLASS(gaia::mesh::TriMesh);
    GDREGISTER_CLASS(gaia::mesh::MeshIO);
    GDREGISTER_CLASS(gaia::mesh::MeshQuality);
    GDREGISTER_CLASS(gaia::parallel::ThreadPool);
    GDREGISTER_CLASS(gaia::parallel::ParallelSort);
    GDREGISTER_CLASS(gaia::parallel::CPUParallelization);
    GDREGISTER_CLASS(gaia::parallel::GPUParallelization);
    GDREGISTER_CLASS(gaia::parallel::SPSCQueue);
    GDREGISTER_CLASS(gaia::utils::ConvexDecomposition);

    // ===================== Genesis =====================
    GDREGISTER_CLASS(genesis::GenesisMaterial);
    GDREGISTER_CLASS(genesis::FEMMaterial);
    GDREGISTER_CLASS(genesis::MPMMaterial);
    GDREGISTER_CLASS(genesis::PBDMaterial);
    GDREGISTER_CLASS(genesis::SPHMaterial);
    GDREGISTER_CLASS(genesis::SFMaterial);
    GDREGISTER_CLASS(genesis::BaseEntity);
    GDREGISTER_CLASS(genesis::RigidEntity);
    GDREGISTER_CLASS(genesis::FEMEntity);
    GDREGISTER_CLASS(genesis::MPMEntity);
    GDREGISTER_CLASS(genesis::ToolEntity);
    GDREGISTER_CLASS(genesis::DroneEntity);
    GDREGISTER_CLASS(genesis::HybridEntity);
    GDREGISTER_CLASS(genesis::ParticleEntity);
    GDREGISTER_CLASS(genesis::EmitterEntity);
    GDREGISTER_CLASS(genesis::BaseSolver);
    GDREGISTER_CLASS(genesis::RigidSolver);
    GDREGISTER_CLASS(genesis::FEMSolver);
    GDREGISTER_CLASS(genesis::MPMSolver);
    GDREGISTER_CLASS(genesis::GenesisPBDSolver);
    GDREGISTER_CLASS(genesis::SPHSolver);
    GDREGISTER_CLASS(genesis::SFSolver);
    GDREGISTER_CLASS(genesis::KinematicSolver);
    GDREGISTER_CLASS(genesis::ToolSolver);
    GDREGISTER_CLASS(genesis::SPHGPUSolver);
    GDREGISTER_CLASS(genesis::SolverRegistry);
    GDREGISTER_CLASS(genesis::GJK);
    GDREGISTER_CLASS(genesis::IPCCoupler);
    GDREGISTER_CLASS(genesis::ContactSolver);
    GDREGISTER_CLASS(genesis::BoundaryCondition);
    GDREGISTER_CLASS(genesis::SDFBoundary);
    GDREGISTER_CLASS(genesis::BaseSensor);
    GDREGISTER_CLASS(genesis::CameraSensor);
    GDREGISTER_CLASS(genesis::ContactForceSensor);
    GDREGISTER_CLASS(genesis::IMUSensor);
    GDREGISTER_CLASS(genesis::DepthCameraSensor);
    GDREGISTER_CLASS(genesis::KinematicTactileSensor);
    GDREGISTER_CLASS(genesis::LidarSensor);
    GDREGISTER_CLASS(genesis::TemperatureGridSensor);
    GDREGISTER_CLASS(genesis::BaseRecorder);
    GDREGISTER_CLASS(genesis::FileRecorder);
    GDREGISTER_CLASS(genesis::RecorderManager);
    GDREGISTER_CLASS(genesis::EntityState);
    GDREGISTER_CLASS(genesis::SolverState);
    GDREGISTER_CLASS(genesis::SolverStateCache);
    GDREGISTER_CLASS(genesis::Tensor);
    GDREGISTER_CLASS(genesis::GradientTape);
    GDREGISTER_CLASS(genesis::SAPCoupler);
    GDREGISTER_CLASS(genesis::URDFLoader);
    GDREGISTER_CLASS(genesis::GenesisWorld);
    GDREGISTER_CLASS(genesis::GenesisPhysicsServer3D);
    GDREGISTER_CLASS(genesis::SimulationConfiguration);

    // ===================== Newton =====================
    GDREGISTER_CLASS(newton::NewtonWorld);
    GDREGISTER_CLASS(newton::NewtonBody);
    GDREGISTER_CLASS(newton::NewtonCollision);
    GDREGISTER_CLASS(newton::NewtonCollisionSphere);
    GDREGISTER_CLASS(newton::NewtonCollisionBox);
    GDREGISTER_CLASS(newton::NewtonCollisionCapsule);
    GDREGISTER_CLASS(newton::NewtonCollisionCylinder);
    GDREGISTER_CLASS(newton::NewtonCollisionCone);
    GDREGISTER_CLASS(newton::NewtonCollisionConvexHull);
    GDREGISTER_CLASS(newton::NewtonCollisionTree);
    GDREGISTER_CLASS(newton::NewtonCompoundCollision);
    GDREGISTER_CLASS(newton::NewtonHeightfieldCollision);
    GDREGISTER_CLASS(newton::NewtonCCD);
    GDREGISTER_CLASS(newton::NewtonContactReport);
    GDREGISTER_CLASS(newton::NewtonContactModifier);
    GDREGISTER_CLASS(newton::NewtonJoint);
    GDREGISTER_CLASS(newton::NewtonBallJoint);
    GDREGISTER_CLASS(newton::NewtonHingeJoint);
    GDREGISTER_CLASS(newton::NewtonSliderJoint);
    GDREGISTER_CLASS(newton::NewtonUniversalJoint);
    GDREGISTER_CLASS(newton::NewtonCorkscrewJoint);
    GDREGISTER_CLASS(newton::NewtonFixedJoint);
    GDREGISTER_CLASS(newton::NewtonUpVectorJoint);
    GDREGISTER_CLASS(newton::NewtonGearJoint);
    GDREGISTER_CLASS(newton::NewtonPulleyJoint);
    GDREGISTER_CLASS(newton::NewtonCustomJoint);
    GDREGISTER_CLASS(newton::NewtonD6Joint);
    GDREGISTER_CLASS(newton::NewtonVehicle);
    GDREGISTER_CLASS(newton::NewtonSolver);
    GDREGISTER_CLASS(newton::NewtonIsland);
    GDREGISTER_CLASS(newton::NewtonParallelSolver);
    GDREGISTER_CLASS(newton::NewtonMaterial);
    GDREGISTER_CLASS(newton::NewtonMaterialPair);
    GDREGISTER_CLASS(newton::NewtonSerializer);
    GDREGISTER_CLASS(newton::NewtonRayCast);
    GDREGISTER_CLASS(newton::NewtonPhysicsServer3D);
    GDREGISTER_CLASS(newton::NewtonCharacterController);
    GDREGISTER_CLASS(newton::NewtonRagdoll);

    // ===================== Vienna =====================
    GDREGISTER_CLASS(vienna::ViennaWorld);
    GDREGISTER_CLASS(vienna::ViennaBody);
    GDREGISTER_CLASS(vienna::ViennaShape);
    GDREGISTER_CLASS(vienna::ViennaShapeSphere);
    GDREGISTER_CLASS(vienna::ViennaShapeBox);
    GDREGISTER_CLASS(vienna::ViennaShapeCapsule);
    GDREGISTER_CLASS(vienna::ViennaShapeCylinder);
    GDREGISTER_CLASS(vienna::ViennaShapeCone);
    GDREGISTER_CLASS(vienna::ViennaShapeConvexHull);
    GDREGISTER_CLASS(vienna::ViennaShapeTriMesh);
    GDREGISTER_CLASS(vienna::ViennaShapeHeightfield);
    GDREGISTER_CLASS(vienna::ViennaShapeCompound);
    GDREGISTER_CLASS(vienna::ViennaJoint);
    GDREGISTER_CLASS(vienna::ViennaBallJoint);
    GDREGISTER_CLASS(vienna::ViennaHingeJoint);
    GDREGISTER_CLASS(vienna::ViennaSliderJoint);
    GDREGISTER_CLASS(vienna::ViennaFixedJoint);
    GDREGISTER_CLASS(vienna::ViennaDistanceJoint);
    GDREGISTER_CLASS(vienna::ViennaRopeJoint);
    GDREGISTER_CLASS(vienna::ViennaUpVectorJoint);
    GDREGISTER_CLASS(vienna::ViennaGearJoint);
    GDREGISTER_CLASS(vienna::ViennaPulleyJoint);
    GDREGISTER_CLASS(vienna::ViennaCustomJoint);
    GDREGISTER_CLASS(vienna::ViennaD6Joint);
    GDREGISTER_CLASS(vienna::ViennaVehicle);
    GDREGISTER_CLASS(vienna::ViennaSolver);
    GDREGISTER_CLASS(vienna::ViennaIsland);
    GDREGISTER_CLASS(vienna::ViennaParallelSolver);
    GDREGISTER_CLASS(vienna::ViennaParallelContactSolver);
    GDREGISTER_CLASS(vienna::ViennaMaterial);
    GDREGISTER_CLASS(vienna::ViennaCloth);
    GDREGISTER_CLASS(vienna::ViennaClothSolver);
    GDREGISTER_CLASS(vienna::ViennaParticleSystem);
    GDREGISTER_CLASS(vienna::ViennaSerializer);
    GDREGISTER_CLASS(vienna::ViennaDebugDraw);
    GDREGISTER_CLASS(vienna::ViennaMeshLoader);
    GDREGISTER_CLASS(vienna::ViennaSceneLoader);
    GDREGISTER_CLASS(vienna::ViennaPhysicsServer3D);
    GDREGISTER_CLASS(vienna::ViennaCharacterController);
    GDREGISTER_CLASS(vienna::ViennaWorldNode3D);
    GDREGISTER_CLASS(vienna::ViennaRigidBody3D);
    GDREGISTER_CLASS(vienna::ViennaSoftBody3D);
    GDREGISTER_CLASS(vienna::ViennaPhysicsSettings);
    GDREGISTER_CLASS(vienna::ViennaWorldQuery);
    GDREGISTER_CLASS(vienna::ViennaCCD);
    GDREGISTER_CLASS(vienna::ViennaProfiler);
    GDREGISTER_CLASS(vienna::ViennaRagdoll);

    // ===================== Wicked =====================
    GDREGISTER_CLASS(wicked::WickedWorld);
    GDREGISTER_CLASS(wicked::WickedBody);
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
    GDREGISTER_CLASS(wicked::WickedJoint);
    GDREGISTER_CLASS(wicked::WickedBallJoint);
    GDREGISTER_CLASS(wicked::WickedHingeJoint);
    GDREGISTER_CLASS(wicked::WickedSliderJoint);
    GDREGISTER_CLASS(wicked::WickedFixedJoint);
    GDREGISTER_CLASS(wicked::WickedConeTwistJoint);
    GDREGISTER_CLASS(wicked::WickedGeneric6DOFJoint);
    GDREGISTER_CLASS(wicked::WickedSolver);
    GDREGISTER_CLASS(wicked::WickedIsland);
    GDREGISTER_CLASS(wicked::WickedMaterial);
    GDREGISTER_CLASS(wicked::WickedRaycastVehicle);
    GDREGISTER_CLASS(wicked::WickedPhysicsServer3D);
    GDREGISTER_CLASS(wicked::WickedMeshLoader);

    // ===================== Unified Integration =====================
    GDREGISTER_CLASS(UnifiedPhysicsServerAll);
    GDREGISTER_CLASS(UnifiedPhysicsMaterialManager);
    GDREGISTER_CLASS(UnifiedCollisionFilter);
    GDREGISTER_CLASS(UnifiedPhysicsEventBus);
    GDREGISTER_CLASS(UnifiedProfiler);
    GDREGISTER_CLASS(UnifiedWarmStartCache);
    GDREGISTER_CLASS(UnifiedAdaptiveSimulation);
    GDREGISTER_CLASS(UnifiedPhysicsEngineRegistry);
    GDREGISTER_CLASS(UnifiedPhysicsThreadManager);
    GDREGISTER_CLASS(UnifiedWorldQuery);
    GDREGISTER_CLASS(UnifiedPhysicsDebugDraw);
    GDREGISTER_CLASS(UnifiedPhysicsSerializer);
    GDREGISTER_CLASS(UnifiedJointBridge);
    GDREGISTER_CLASS(UnifiedRagdollBlender);
    GDREGISTER_CLASS(UnifiedMeshLoader);
    GDREGISTER_CLASS(UnifiedProfilerJSONWriter);
}

void uninitialize_unified_physics_final(ModuleInitializationLevel p_level) {
    // No explicit cleanup required for class registrations.
}