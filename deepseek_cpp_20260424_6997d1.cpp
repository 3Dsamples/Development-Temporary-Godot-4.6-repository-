// genesis/engine/__init__.h

#pragma once

//------------------------------------------------------------------------------
// Genesis Engine Module
// This header serves as the main entry point for the physics simulation engine.
// It provides common definitions and forward declarations used across the engine.
//------------------------------------------------------------------------------

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Forward declarations of major engine components
//------------------------------------------------------------------------------
class Simulator;
class Scene;
class Entity;
class BaseEntity;
class Solver;
class BaseSolver;
class BVH;
class Mesh;
class ForceField;
class ContactManager;
class ConstraintSolver;
class Integrator;

//------------------------------------------------------------------------------
// Smart pointer aliases for engine objects
//------------------------------------------------------------------------------
using SimulatorPtr = std::shared_ptr<Simulator>;
using ScenePtr = std::shared_ptr<Scene>;
using EntityPtr = std::shared_ptr<Entity>;
using BaseEntityPtr = std::shared_ptr<BaseEntity>;
using SolverPtr = std::shared_ptr<Solver>;
using BaseSolverPtr = std::shared_ptr<BaseSolver>;
using BVHPtr = std::shared_ptr<BVH>;
using MeshPtr = std::shared_ptr<Mesh>;
using ForceFieldPtr = std::shared_ptr<ForceField>;
using ContactManagerPtr = std::shared_ptr<ContactManager>;
using ConstraintSolverPtr = std::shared_ptr<ConstraintSolver>;

//------------------------------------------------------------------------------
// Engine configuration flags
//------------------------------------------------------------------------------
enum class EngineFeature : uint32_t {
    NONE                = 0,
    GPU_ACCELERATION    = 1 << 0,
    MULTITHREADED       = 1 << 1,
    DOUBLE_PRECISION    = 1 << 2,
    COLLISION_DETECTION = 1 << 3,
    CONTINUOUS_COLLISION = 1 << 4,
    SOFT_BODY           = 1 << 5,
    FLUID               = 1 << 6,
    CLOTH               = 1 << 7,
    ROPE                = 1 << 8,
    FEM                 = 1 << 9,
    MPM                 = 1 << 10,
    SPH                 = 1 << 11,
    PBD                 = 1 << 12,
    RIGID_BODY          = 1 << 13,
    KINEMATIC           = 1 << 14,
    PARTICLE_SYSTEM     = 1 << 15,
    ALL                 = 0xFFFFFFFF
};

inline constexpr EngineFeature operator|(EngineFeature a, EngineFeature b) {
    return static_cast<EngineFeature>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline constexpr bool has_feature(EngineFeature flags, EngineFeature feature) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(feature)) != 0;
}

//------------------------------------------------------------------------------
// Engine initialization parameters
//------------------------------------------------------------------------------
struct EngineConfig {
    EngineFeature features = EngineFeature::ALL;
    uint32_t num_threads = 0;                 // 0 = auto-detect
    uint32_t gpu_device_id = 0;
    bool enable_profiling = false;
    bool enable_logging = true;
    bool enable_validation = false;
    std::string log_file_path;
};

//------------------------------------------------------------------------------
// Global engine state functions (mirroring module-level functions in Python)
//------------------------------------------------------------------------------

// Initialize the engine with given configuration.
void initialize(const EngineConfig& config = EngineConfig{});

// Shutdown the engine and release resources.
void shutdown();

// Check if engine is initialized.
bool is_initialized();

// Get current engine configuration.
const EngineConfig& get_config();

// Set global gravity vector (applies to all scenes unless overridden).
void set_gravity(double gx, double gy, double gz);
void set_gravity(const std::array<double, 3>& g);
std::array<double, 3> get_gravity();

// Set global time step.
void set_time_step(double dt);
double get_time_step();

// Global profiling controls.
void enable_profiling(bool enable);
bool is_profiling_enabled();

// Get engine version (re-export from genesis::version).
std::string get_version_string();

//------------------------------------------------------------------------------
// Utility functions
//------------------------------------------------------------------------------

// Get the number of available CPU cores.
int get_num_cpu_cores();

// Get available GPU devices information.
struct GPUDeviceInfo {
    std::string name;
    size_t total_memory_bytes;
    int compute_capability_major;
    int compute_capability_minor;
};
std::vector<GPUDeviceInfo> get_available_gpus();

} // namespace engine
} // namespace genesis