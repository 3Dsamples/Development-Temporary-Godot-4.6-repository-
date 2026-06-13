// genesis/engine/interaction_manager.h
#pragma once

//------------------------------------------------------------------------------
// InteractionManager – processes high‑level deformation commands (poke, punch,
// bend, grab) and applies appropriate forces to deformable entities (FEM, PBD,
// MPM, or SoftTissue).  Works per‑scene or globally.  Handles transient force
// profiles, automatic release, and callback notifications.
//------------------------------------------------------------------------------

#include <memory>                                     // std::shared_ptr, std::weak_ptr
#include <vector>                                     // std::vector
#include <unordered_map>                              // std::unordered_map
#include <functional>                                 // std::function
#include <chrono>                                     // std::chrono for timestamps

namespace genesis {
namespace engine {

class BaseEntity;
class Scene;
class SoftTissueEntity;    // Forward declarations
class FEMEntity;
class PBDEntity;
class MPMEntity;

//------------------------------------------------------------------------------
// Interaction description – a single poke/punch/bend request
//------------------------------------------------------------------------------
struct InteractionEvent {
    enum class Type : uint8_t {
        POKE = 0,
        PUNCH = 1,
        BEND = 2,
        GRAB = 3,
        RELEASE = 4
    };

    Type type = Type::POKE;
    uint64_t target_entity_id = 0;      // if zero, use closest hit
    datatypes::Vector3 world_point;     // location of interaction
    datatypes::Vector3 direction;       // direction of force (poke/punch) or bend axis normal
    double magnitude = 1.0;             // force (N) or moment (Nm)
    double duration = 0.0;              // seconds (0 = instant, negative = sustained until release)
    double timestamp = 0.0;             // simulation time when created
};

//------------------------------------------------------------------------------
// Active interaction state – tracks ongoing forces until released
//------------------------------------------------------------------------------
struct ActiveInteraction {
    InteractionEvent event;
    double remaining_time = 0.0;        // seconds left (or negative = infinite)
    uint64_t affected_entity_id = 0;    // resolved entity
    std::vector<uint32_t> affected_node_indices; // for FEM/PBD, nodes currently loaded
};

//------------------------------------------------------------------------------
// InteractionManager class
//------------------------------------------------------------------------------
class InteractionManager {
public:
    //----------------------------------------------------------------------
    // Construction
    //----------------------------------------------------------------------
    InteractionManager();
    ~InteractionManager();

    //----------------------------------------------------------------------
    // Main update – called every simulation step (from Scene or solver)
    //----------------------------------------------------------------------
    void update(double dt, Scene& scene);

    //----------------------------------------------------------------------
    // Add an interaction (poke, punch, bend, grab)
    //----------------------------------------------------------------------
    // Add a force at a world point; the manager resolves which entity is hit
    void add_interaction(const InteractionEvent& event);

    // Convenience: poke with given force and duration
    void poke(const datatypes::Vector3& world_point,
              const datatypes::Vector3& direction,
              double force, double duration = 0.0);

    // Convenience: punch (high force, short duration)
    void punch(const datatypes::Vector3& world_point,
               const datatypes::Vector3& direction,
               double force, double duration = 0.1);

    // Convenience: bend around an axis (applied to entities in contact)
    void bend(const datatypes::Vector3& axis_origin,
              const datatypes::Vector3& axis_direction,
              double moment, double duration = -1.0); // sustained until released

    // Convenience: grab a point and pull it toward a target (not implemented here, placeholder)
    void grab(uint64_t entity_id, const datatypes::Vector3& grab_point);

    // Release all interactions for a given entity (or all if entity_id=0)
    void release(uint64_t entity_id = 0);

    //----------------------------------------------------------------------
    // Callback for external listeners (haptics, sound, visual effects)
    //----------------------------------------------------------------------
    void set_event_callback(std::function<void(const InteractionEvent&, const std::string&)> callback);
    void clear_event_callback();

    //----------------------------------------------------------------------
    // Statistics
    //----------------------------------------------------------------------
    size_t active_interaction_count() const { return active_interactions_.size(); }

private:
    // Active interactions
    std::vector<ActiveInteraction> active_interactions_;

    // Optional event callback (event + "started"/"updated"/"ended")
    std::function<void(const InteractionEvent&, const std::string&)> event_callback_;

    // Internal methods
    void resolve_target_entity(ActiveInteraction& interaction, Scene& scene);
    void apply_poke_punch(ActiveInteraction& interaction, double dt);
    void apply_bend(ActiveInteraction& interaction, Scene& scene, double dt);
    void apply_grab(ActiveInteraction& interaction, Scene& scene, double dt);
    void remove_expired_interactions();

    // Helper: spread force over nearby nodes of a deformable entity
    void spread_force_to_nodes(std::shared_ptr<BaseEntity> entity,
                               const datatypes::Vector3& world_point,
                               const datatypes::Vector3& force,
                               std::vector<uint32_t>& out_affected_nodes);
};

} // namespace engine
} // namespace genesis