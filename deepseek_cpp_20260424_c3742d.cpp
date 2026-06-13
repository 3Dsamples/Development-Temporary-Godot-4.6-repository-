// genesis/engine/solvers/base_solver.h

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <chrono>
#include "genesis/datatypes.h"
#include "genesis/engine/entities/base_entity.h"

namespace genesis {
namespace engine {

// Forward declarations
class BaseEntity;
class Scene;

//------------------------------------------------------------------------------
// Solver configuration parameters
//------------------------------------------------------------------------------
struct SolverConfig {
    // Time stepping
    int substeps = 1;
    int iterations = 5;
    
    // Constraint solver parameters
    double constraint_solver_tolerance = 1e-6;
    double contact_offset = 0.005;
    double restitution = 0.5;
    double friction_coefficient = 0.5;
    
    // Damping
    double velocity_damping = 0.0;
    double position_damping = 0.0;
    
    // Solver-specific flags
    bool enable_self_collision = true;
    bool enable_stabilization = true;
    bool use_warm_starting = true;
    
    // Parallelization
    int num_threads = 0;  // 0 = auto
    
    // Debugging
    bool debug_visualization = false;
    std::string debug_output_path;
    
    // Custom parameters (key-value store for solver-specific settings)
    std::unordered_map<std::string, double> custom_params;
};

//------------------------------------------------------------------------------
// Abstract base class for all physics solvers.
// Defines the interface that concrete solvers must implement.
//------------------------------------------------------------------------------
class BaseSolver {
public:
    // Constructor with optional name
    explicit BaseSolver(const std::string& name = "BaseSolver");
    virtual ~BaseSolver();

    // Prevent copying (polymorphic base)
    BaseSolver(const BaseSolver&) = delete;
    BaseSolver& operator=(const BaseSolver&) = delete;

    // Move allowed
    BaseSolver(BaseSolver&&) = default;
    BaseSolver& operator=(BaseSolver&&) = default;

    //--- Identification ---
    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }

    // Get the type of solver (e.g., "PBD", "MPM", "SPH")
    virtual std::string solver_type() const = 0;

    //--- Configuration ---
    void set_config(const SolverConfig& config);
    const SolverConfig& config() const { return config_; }
    
    // Set/get custom parameter
    void set_param(const std::string& key, double value);
    double get_param(const std::string& key, double default_value = 0.0) const;

    //--- Initialization ---
    // Initialize solver for a given set of entities (called once before simulation)
    virtual void initialize(const std::vector<std::shared_ptr<BaseEntity>>& entities) = 0;

    // Reset solver state (e.g., clear accumulated impulses)
    virtual void reset();

    //--- Stepping ---
    // Perform one simulation step for the given entities over time dt.
    // This is the main entry point for the solver.
    virtual void step(double dt, const std::vector<std::shared_ptr<BaseEntity>>& entities) = 0;

    //--- Entity management ---
    // Called when an entity is added to the scene (optional override)
    virtual void on_entity_added(std::shared_ptr<BaseEntity> entity) {}

    // Called when an entity is removed from the scene (optional override)
    virtual void on_entity_removed(std::shared_ptr<BaseEntity> entity) {}

    //--- Statistics ---
    struct Stats {
        double last_step_time_ms = 0.0;
        double avg_step_time_ms = 0.0;
        size_t constraint_count = 0;
        size_t iteration_count = 0;
        size_t collision_pairs = 0;
        double residual = 0.0;
    };
    const Stats& stats() const { return stats_; }

    // Reset statistics counters
    virtual void reset_stats();

    //--- Debug visualization support ---
    virtual void debug_draw() const {}

    // Enable/disable debug visualization
    void set_debug_enabled(bool enabled) { debug_enabled_ = enabled; }
    bool debug_enabled() const { return debug_enabled_; }

    //--- Solver state queries ---
    virtual bool is_converged() const { return true; }
    virtual double get_residual() const { return 0.0; }

protected:
    std::string name_;
    SolverConfig config_;
    Stats stats_;
    bool debug_enabled_ = false;
    bool initialized_ = false;

    // Helper for timing
    struct ScopedTimer {
        BaseSolver* solver;
        std::chrono::high_resolution_clock::time_point start;
        ScopedTimer(BaseSolver* s) : solver(s), start(std::chrono::high_resolution_clock::now()) {}
        ~ScopedTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            solver->stats_.last_step_time_ms = ms;
            // Exponential moving average
            const double alpha = 0.1;
            solver->stats_.avg_step_time_ms = solver->stats_.avg_step_time_ms * (1.0 - alpha) + ms * alpha;
        }
    };

    // Utility to filter entities by dynamic/kinematic/etc.
    static std::vector<std::shared_ptr<BaseEntity>> filter_dynamic(
        const std::vector<std::shared_ptr<BaseEntity>>& entities);
    static std::vector<std::shared_ptr<BaseEntity>> filter_static(
        const std::vector<std::shared_ptr<BaseEntity>>& entities);
};

} // namespace engine
} // namespace genesis