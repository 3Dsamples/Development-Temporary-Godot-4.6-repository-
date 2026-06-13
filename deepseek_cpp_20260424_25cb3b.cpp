// genesis/engine/simulator.h

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include "genesis/datatypes.h"
#include "genesis/engine/scene.h"

namespace genesis {
namespace engine {

// Forward declarations
class Simulator;
struct SimulatorConfig;

//------------------------------------------------------------------------------
// Simulator configuration
//------------------------------------------------------------------------------
struct SimulatorConfig {
    // Time control
    double target_realtime_factor = 1.0;      // 1.0 = realtime, 0 = run as fast as possible
    double max_dt_per_step = 0.1;             // Maximum time step allowed
    bool fixed_timestep = true;
    
    // Threading
    int num_threads = 0;                      // 0 = auto
    bool parallel_solver = true;
    
    // GPU
    bool use_gpu = false;
    int gpu_device_id = 0;
    
    // Profiling
    bool enable_profiling = false;
    std::string profile_output_file;
    
    // Headless mode (no visualization)
    bool headless = true;
    
    // Recording
    bool record_simulation = false;
    std::string record_path;
    int record_fps = 60;
};

//------------------------------------------------------------------------------
// Simulation statistics
//------------------------------------------------------------------------------
struct SimulationStats {
    double simulation_time = 0.0;              // Total simulated time
    double wall_time = 0.0;                    // Total wall-clock time
    uint64_t step_count = 0;
    double avg_step_time_ms = 0.0;
    double avg_realtime_factor = 0.0;
    size_t active_entities = 0;
    size_t active_particles = 0;
    size_t active_constraints = 0;
};

//------------------------------------------------------------------------------
// Simulator class: manages multiple scenes and the main simulation loop.
// Provides high-level control for stepping, running, and realtime synchronization.
//------------------------------------------------------------------------------
class Simulator {
public:
    // Constructor / Destructor
    explicit Simulator(const SimulatorConfig& config = SimulatorConfig{});
    ~Simulator();

    // Prevent copy
    Simulator(const Simulator&) = delete;
    Simulator& operator=(const Simulator&) = delete;

    //--- Configuration ---
    void set_config(const SimulatorConfig& config);
    const SimulatorConfig& config() const { return config_; }

    //--- Scene management ---
    // Create a new scene and add to simulator
    std::shared_ptr<Scene> create_scene(const std::string& name = "default",
                                        const SceneConfig& scene_config = SceneConfig{});
    
    // Add an existing scene
    void add_scene(std::shared_ptr<Scene> scene);
    void remove_scene(const std::string& name);
    void remove_scene(size_t index);
    std::shared_ptr<Scene> get_scene(const std::string& name) const;
    std::shared_ptr<Scene> get_scene(size_t index) const;
    size_t scene_count() const { return scenes_.size(); }
    void clear_scenes();

    // Set the active scene (for single-scene operations)
    void set_active_scene(const std::string& name);
    void set_active_scene(size_t index);
    std::shared_ptr<Scene> active_scene() const { return active_scene_; }

    //--- Simulation control ---
    // Reset all scenes
    void reset();
    
    // Step all scenes by given delta time
    void step(double dt = -1.0);
    
    // Run simulation for a given duration (blocking)
    void run(double duration = std::numeric_limits<double>::max());
    
    // Run simulation asynchronously
    void start();
    void stop();
    bool is_running() const { return running_.load(); }
    
    // Wait for async simulation to finish (if started)
    void wait();

    //--- Realtime synchronization ---
    void set_realtime(bool enable);
    bool realtime_enabled() const { return target_realtime_factor_ > 0.0; }

    //--- Callbacks ---
    using StepCallback = std::function<void(Simulator*, double dt)>;
    using PreStepCallback = std::function<void(Simulator*)>;
    using PostStepCallback = std::function<void(Simulator*)>;
    
    void set_step_callback(StepCallback cb) { step_callback_ = std::move(cb); }
    void set_pre_step_callback(PreStepCallback cb) { pre_step_callback_ = std::move(cb); }
    void set_post_step_callback(PostStepCallback cb) { post_step_callback_ = std::move(cb); }

    //--- Statistics ---
    SimulationStats get_stats() const;
    void reset_stats();

    //--- Profiling ---
    void start_profiling();
    void stop_profiling();
    void save_profile(const std::string& filename = "");

    //--- Recording ---
    void start_recording(const std::string& path = "", int fps = 60);
    void stop_recording();
    bool is_recording() const { return recording_; }

    //--- Debug ---
    void set_debug(bool enabled) { debug_mode_ = enabled; }

private:
    SimulatorConfig config_;
    std::vector<std::shared_ptr<Scene>> scenes_;
    std::unordered_map<std::string, std::shared_ptr<Scene>> scenes_by_name_;
    std::shared_ptr<Scene> active_scene_;

    // Async simulation thread
    std::unique_ptr<std::thread> sim_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    std::mutex sim_mutex_;
    std::condition_variable sim_cv_;
    double run_duration_remaining_ = 0.0;

    // Realtime synchronization
    double target_realtime_factor_;
    std::chrono::steady_clock::time_point last_step_time_;

    // Callbacks
    StepCallback step_callback_;
    PreStepCallback pre_step_callback_;
    PostStepCallback post_step_callback_;

    // Statistics
    mutable std::mutex stats_mutex_;
    SimulationStats stats_;

    // Profiling
    bool profiling_ = false;
    std::vector<double> step_times_ms_;

    // Recording
    bool recording_ = false;
    std::string record_path_;
    int record_fps_;

    bool debug_mode_ = false;

    // Internal methods
    void run_loop();
    void record_frame(double sim_time);
    void update_stats(double dt, double step_time_ms);
};

} // namespace engine
} // namespace genesis