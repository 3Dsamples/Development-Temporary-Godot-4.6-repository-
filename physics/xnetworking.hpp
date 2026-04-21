// network/xnetworking.hpp
#ifndef XTENSOR_XNETWORKING_HPP
#define XTENSOR_XNETWORKING_HPP

// ----------------------------------------------------------------------------
// xnetworking.hpp – Real‑time network synchronization
// ----------------------------------------------------------------------------
// Provides deterministic networking for physics and animation:
//   - Rollback netcode with input prediction and reconciliation
//   - Delta compression and state snapshots
//   - Interest management (relevance filtering)
//   - Lag compensation for server‑authoritative physics
//   - Deterministic lockstep simulation support
//   - Bit‑packed serialization for bandwidth efficiency
//   - Integration with BigNumber for exact state replication
//
// Targets 120 fps tick rate with minimal latency.
// ----------------------------------------------------------------------------

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "xserialization.hpp"
#include "bignumber/bignumber.hpp"
#include <chrono>
#include <functional>
#include <queue>
#include <unordered_map>

namespace xt {
namespace network {

using frame_id = uint32_t;
using player_id = uint32_t;
using timestamp = std::chrono::steady_clock::time_point;

// ========================================================================
// Network configuration
// ========================================================================
struct network_config {
    uint32_t tick_rate = 60;           // simulation frames per second
    uint32_t input_delay = 3;          // frames of input buffering
    uint32_t max_rollback_frames = 8;  // maximum rollback depth
    bool use_delta_compression = true;
    bool enable_prediction = true;
    float interpolation_delay = 0.1f;  // seconds
};

// ========================================================================
// Input state (for rollback)
// ========================================================================
struct player_input {
    frame_id frame;
    uint32_t buttons;          // bitmask
    xarray_container<float> move_direction;  // (2,) or (3,)
    xarray_container<float> look_direction;
    timestamp received_at;
};

// ========================================================================
// Game state snapshot
// ========================================================================
class state_snapshot {
public:
    frame_id frame;
    timestamp time;
    std::vector<uint8_t> data;  // serialized state

    template <class T>
    void serialize(const T& state);
    template <class T>
    void deserialize(T& state) const;
};

// ========================================================================
// Rollback manager
// ========================================================================
template <class State>
class rollback_manager {
public:
    using state_type = State;
    using step_func = std::function<void(state_type&, const std::vector<player_input>&, float dt)>;
    using save_func = std::function<state_snapshot(const state_type&)>;
    using load_func = std::function<void(state_type&, const state_snapshot&)>;

    rollback_manager(const network_config& cfg, step_func step, save_func save, load_func load);

    // Called every frame with local inputs
    void advance_frame(frame_id current_frame, const player_input& local_input);

    // Receive remote inputs (may be late)
    void receive_input(const player_input& input);

    // Get current authoritative state
    const state_type& current_state() const;
    state_type& current_state();

    // Prediction for local player
    state_type predicted_state(frame_id frame) const;

private:
    network_config m_cfg;
    step_func m_step;
    save_func m_save;
    load_func m_load;
    state_type m_current_state;
    frame_id m_current_frame;
    std::unordered_map<player_id, std::queue<player_input>> m_input_queues;
    std::vector<state_snapshot> m_snapshots;
    std::vector<player_input> m_pending_inputs;

    void rollback_to(frame_id target_frame);
    void resimulate_from(frame_id from_frame);
};

// ========================================================================
// Network entity (replicated object)
// ========================================================================
class network_entity {
public:
    uint32_t network_id;
    bool is_owned;           // locally authoritative
    bool is_relevant;        // should be sent to clients
    uint32_t priority;       // for interest management

    virtual ~network_entity() = default;
    virtual void serialize_delta(std::vector<uint8_t>& out, const network_entity* baseline) = 0;
    virtual void deserialize_delta(const uint8_t* data, size_t size) = 0;
    virtual state_snapshot full_snapshot() const = 0;
    virtual void apply_snapshot(const state_snapshot& snap) = 0;
};

// ========================================================================
// Replication manager
// ========================================================================
class replication_manager {
public:
    replication_manager();

    void register_entity(std::shared_ptr<network_entity> entity);
    void unregister_entity(uint32_t network_id);

    // Generate state update for a client
    std::vector<uint8_t> generate_update(player_id target_client,
                                          const std::vector<uint32_t>& acked_entities);

    // Apply received update
    void apply_update(const uint8_t* data, size_t size);

    // Interest management
    void set_relevance_function(std::function<bool(const network_entity*, player_id)> func);

private:
    std::unordered_map<uint32_t, std::shared_ptr<network_entity>> m_entities;
    std::unordered_map<uint32_t, state_snapshot> m_last_acked;
    std::function<bool(const network_entity*, player_id)> m_relevance_func;
};

// ========================================================================
// Physics synchronization helper
// ========================================================================
template <class T, class PhysicsState>
class physics_replication : public network_entity {
public:
    physics_replication(uint32_t id, PhysicsState* state);

    void serialize_delta(std::vector<uint8_t>& out, const network_entity* baseline) override;
    void deserialize_delta(const uint8_t* data, size_t size) override;
    state_snapshot full_snapshot() const override;
    void apply_snapshot(const state_snapshot& snap) override;

    // Interpolation between snapshots
    void set_interpolation_delay(float delay);
    PhysicsState interpolated_state(timestamp now) const;

private:
    PhysicsState* m_state;
    std::deque<state_snapshot> m_snapshot_buffer;
    float m_interp_delay;
};

// ========================================================================
// Lag compensation (server‑side)
// ========================================================================
template <class T>
class lag_compensator {
public:
    using rewind_func = std::function<void(T)>;
    using restore_func = std::function<void()>;

    lag_compensator(rollback_manager<T>* rollback);

    // Rewind world to a client's perceived time and execute action
    void compensate(player_id client, const std::function<void(T&)>& action);

private:
    rollback_manager<T>* m_rollback;
};

} // namespace network

using network::rollback_manager;
using network::replication_manager;
using network::network_entity;
using network::physics_replication;
using network::lag_compensator;
using network::network_config;

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs
// ----------------------------------------------------------------------------
namespace xt {
namespace network {

// state_snapshot
template <class T> void state_snapshot::serialize(const T& state) {}
template <class T> void state_snapshot::deserialize(T& state) const {}

// rollback_manager
template <class S>
rollback_manager<S>::rollback_manager(const network_config& cfg, step_func step, save_func save, load_func load)
    : m_cfg(cfg), m_step(step), m_save(save), m_load(load), m_current_frame(0) {}
template <class S> void rollback_manager<S>::advance_frame(frame_id f, const player_input& local) {}
template <class S> void rollback_manager<S>::receive_input(const player_input& input) {}
template <class S> const S& rollback_manager<S>::current_state() const { return m_current_state; }
template <class S> S& rollback_manager<S>::current_state() { return m_current_state; }
template <class S> S rollback_manager<S>::predicted_state(frame_id frame) const { return m_current_state; }
template <class S> void rollback_manager<S>::rollback_to(frame_id frame) {}
template <class S> void rollback_manager<S>::resimulate_from(frame_id frame) {}

// replication_manager
inline replication_manager::replication_manager() {}
inline void replication_manager::register_entity(std::shared_ptr<network_entity> e) {}
inline void replication_manager::unregister_entity(uint32_t id) {}
inline std::vector<uint8_t> replication_manager::generate_update(player_id target, const std::vector<uint32_t>& acked) { return {}; }
inline void replication_manager::apply_update(const uint8_t* data, size_t size) {}
inline void replication_manager::set_relevance_function(std::function<bool(const network_entity*, player_id)> f) { m_relevance_func = f; }

// physics_replication
template <class T, class S> physics_replication<T,S>::physics_replication(uint32_t id, S* state) : m_state(state), m_interp_delay(0.1f) {}
template <class T, class S> void physics_replication<T,S>::serialize_delta(std::vector<uint8_t>& out, const network_entity* base) {}
template <class T, class S> void physics_replication<T,S>::deserialize_delta(const uint8_t* data, size_t size) {}
template <class T, class S> state_snapshot physics_replication<T,S>::full_snapshot() const { return {}; }
template <class T, class S> void physics_replication<T,S>::apply_snapshot(const state_snapshot& snap) {}
template <class T, class S> void physics_replication<T,S>::set_interpolation_delay(float d) { m_interp_delay = d; }
template <class T, class S> S physics_replication<T,S>::interpolated_state(timestamp now) const { return *m_state; }

// lag_compensator
template <class T> lag_compensator<T>::lag_compensator(rollback_manager<T>* rb) : m_rollback(rb) {}
template <class T> void lag_compensator<T>::compensate(player_id client, const std::function<void(T&)>& action) {}

} // namespace network
} // namespace xt

#endif // XTENSOR_XNETWORKING_HPP