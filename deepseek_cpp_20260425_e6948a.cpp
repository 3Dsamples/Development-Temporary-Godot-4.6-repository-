// genesis/engine/interaction_manager.h
#pragma once

#include <memory>
#include <vector>
#include <functional>
#include "genesis/datatypes.h"

namespace genesis {
namespace engine {

class BaseEntity;
class Scene;

struct InteractionEvent {
    enum class Type : uint8_t { POKE=0, PUNCH=1, BEND=2, GRAB=3, RELEASE=4 };
    Type type = Type::POKE;
    uint64_t target_entity_id = 0;
    datatypes::Vector3 world_point;
    datatypes::Vector3 direction;
    double magnitude = 1.0;
    double duration = 0.0;
    double timestamp = 0.0;
};

struct ActiveInteraction {
    InteractionEvent event;
    double remaining_time = 0.0;
    uint64_t affected_entity_id = 0;
    std::vector<uint32_t> affected_node_indices;
};

class InteractionManager {
public:
    InteractionManager();
    ~InteractionManager();

    void update(double dt, Scene& scene);

    void add_interaction(const InteractionEvent& event);
    void poke(const datatypes::Vector3& world_point, const datatypes::Vector3& direction,
              double force, double duration = 0.0);
    void punch(const datatypes::Vector3& world_point, const datatypes::Vector3& direction,
               double force, double duration = 0.1);
    void bend(const datatypes::Vector3& axis_origin, const datatypes::Vector3& axis_direction,
              double moment, double duration = -1.0);
    void grab(uint64_t entity_id, const datatypes::Vector3& grab_point);
    void release(uint64_t entity_id = 0);

    void set_event_callback(std::function<void(const InteractionEvent&, const std::string&)> callback);
    void clear_event_callback();

    size_t active_interaction_count() const { return active_interactions_.size(); }

private:
    std::vector<ActiveInteraction> active_interactions_;
    std::function<void(const InteractionEvent&, const std::string&)> event_callback_;
    void remove_expired_interactions();
};

} // namespace engine
} // namespace genesis