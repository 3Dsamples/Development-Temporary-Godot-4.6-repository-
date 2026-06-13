// genesis/engine/entities/restoration_controller.cpp
#include "genesis/engine/entities/restoration_controller.h" // Corresponding header
#include "genesis/engine/entities/fem_entity.h"        // FEMEntity for node access
#include <algorithm>                                   // std::max, std::min
#include <cmath>                                       // std::sqrt
#include <numeric>                                     // std::accumulate

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Construction / Destruction
//------------------------------------------------------------------------------
RestorationController::RestorationController() = default;
RestorationController::~RestorationController() = default;

//------------------------------------------------------------------------------
// Binding
//------------------------------------------------------------------------------
void RestorationController::attach(std::shared_ptr<FEMEntity> entity) {
    entity_ = entity;                               // Store weak pointer
    if (auto fem = entity_.lock()) {
        // Capture reference positions (current rest shape, assumed to be the
        // original undeformed shape)
        reference_positions_ = fem->node_positions(); // Copy all node positions
    }
}

void RestorationController::detach() {
    entity_.reset();                                // Release weak reference
    reference_positions_.clear();                   // Discard reference
}

bool RestorationController::is_attached() const {
    return !entity_.expired();                      // True if entity still exists
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
void RestorationController::set_config(const RestorationConfig& config) {
    config_ = config;                               // Copy new settings
}

const RestorationConfig& RestorationController::config() const {
    return config_;                                 // Read-only access
}

//------------------------------------------------------------------------------
// External interaction notification
//------------------------------------------------------------------------------
void RestorationController::notify_interaction_active(bool active) {
    interaction_active_ = active;                   // Record external interaction state
    if (!active) {
        // Interaction just ceased; reset the timer for activation delay
        time_since_last_interaction_ = 0.0;
    }
}

//------------------------------------------------------------------------------
// Main update – called each simulation step
//------------------------------------------------------------------------------
void RestorationController::update(double dt) {
    auto fem = entity_.lock();                      // Acquire strong reference
    if (!fem) return;                               // Entity no longer exists

    // If auto‑modulation is enabled, detect whether external forces are still
    // being applied by checking if any nodal force is non‑zero (simple heuristic).
    // Alternatively, rely on explicit notify_interaction_active from InteractionManager.
    if (config_.auto_modulate) {
        // We can also check if the entity has any external forces applied
        // (this is a crude measure; in a complete system the InteractionManager
        // would call notify_interaction_active directly).
        // For automatic detection without notification, we compute the total
        // external force magnitude on the entity.
        double total_ext_force = 0.0;
        size_t n_nodes = fem->node_count();
        for (size_t i = 0; i < n_nodes; ++i) {
            total_ext_force += fem->node_external_force(i).norm();
        }
        // If total external force exceeds a small threshold, assume interaction
        // is active; otherwise not.
        const double interaction_threshold = 0.01;   // N, small value
        interaction_active_ = (total_ext_force > interaction_threshold);
        
        // If interaction just stopped, reset the timer
        static bool prev_interaction_active = true;  // assume active initially
        if (!interaction_active_ && prev_interaction_active) {
            time_since_last_interaction_ = 0.0;      // start activation delay timer
        }
        prev_interaction_active = interaction_active_;
    }

    // Update timer for activation delay
    if (!interaction_active_) {
        time_since_last_interaction_ += dt;          // Accumulate time since last interaction
    }

    // Determine whether to apply restoring forces
    bool apply_restoration = false;
    if (!interaction_active_ && time_since_last_interaction_ >= config_.activation_delay) {
        // Interaction is not active and delay has passed -> start/continue restoring
        apply_restoration = true;
    }

    // Apply forces if needed
    if (apply_restoration) {
        apply_restoring_forces();                    // Push nodes toward reference
        currently_restoring_ = true;
    } else {
        currently_restoring_ = false;
    }

    // Invoke status callback if registered
    if (status_callback_) {
        double avg_force = 0.0;
        if (currently_restoring_) {
            // Compute average restoration force magnitude for feedback
            size_t n = fem->node_count();
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i) {
                datatypes::Vector3 disp = fem->node_position(i) - reference_positions_[i];
                sum += std::min(config_.max_force_per_node,
                               config_.stiffness * disp.norm());
            }
            if (n > 0) avg_force = sum / n;
        }
        status_callback_(avg_force, currently_restoring_);
    }
}

//------------------------------------------------------------------------------
// Apply restoring forces to all nodes
//------------------------------------------------------------------------------
void RestorationController::apply_restoring_forces() {
    auto fem = entity_.lock();
    if (!fem) return;

    double k = config_.stiffness;
    double d = config_.damping;
    size_t n = fem->node_count();

    for (size_t i = 0; i < n; ++i) {
        datatypes::Vector3 current_pos = fem->node_position(i);
        datatypes::Vector3 rest_pos = reference_positions_[i];
        datatypes::Vector3 displacement = current_pos - rest_pos; // vector from rest to current

        // Spring force: -k * displacement
        datatypes::Vector3 spring_force = displacement * (-k);

        // Damping force: -d * velocity (assuming velocity is available)
        datatypes::Vector3 velocity = fem->node_velocity(i);
        datatypes::Vector3 damp_force = velocity * (-d);

        datatypes::Vector3 total_force = spring_force + damp_force;

        // Clamp force magnitude
        double mag = total_force.norm();
        if (mag > config_.max_force_per_node) {
            total_force *= config_.max_force_per_node / mag;
        }

        // Add force to the entity's nodal external force accumulator
        fem->add_node_force(i, total_force);
    }
}

//------------------------------------------------------------------------------
// Compute average displacement (for internal use)
//------------------------------------------------------------------------------
double RestorationController::compute_average_displacement() const {
    auto fem = entity_.lock();
    if (!fem) return 0.0;

    double sum = 0.0;
    size_t n = fem->node_count();
    for (size_t i = 0; i < n; ++i) {
        sum += (fem->node_position(i) - reference_positions_[i]).norm();
    }
    return (n > 0) ? (sum / n) : 0.0;
}

//------------------------------------------------------------------------------
// Hook setter
//------------------------------------------------------------------------------
void RestorationController::set_status_callback(std::function<void(double, bool)> callback) {
    status_callback_ = callback;                    // Store callback function
}

} // namespace engine
} // namespace genesis