// include/xtu/godot/xmultiplayer_advanced.hpp
// xtensor-unified - Advanced multiplayer replication for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XMULTIPLAYER_ADVANCED_HPP
#define XTU_GODOT_XMULTIPLAYER_ADVANCED_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xnetworking.hpp"
#include "xtu/parallel/xparallel.hpp"
#include "xtu/io/xio_json.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class SceneReplicationConfig;
class MultiplayerSpawner;
class MultiplayerSynchronizer;
class CustomMultiplayerPeer;
class ReplicationEditor;

// #############################################################################
// Replication authority mode
// #############################################################################
enum class ReplicationAuthority : uint8_t {
    AUTHORITY_SERVER = 0,
    AUTHORITY_CLIENT = 1,
    AUTHORITY_SHARED = 2
};

// #############################################################################
// Synchronization mode
// #############################################################################
enum class SynchronizationMode : uint8_t {
    SYNC_ALWAYS = 0,
    SYNC_ON_CHANGE = 1,
    SYNC_INTERVAL = 2,
    SYNC_MANUAL = 3
};

// #############################################################################
// Interpolation mode
// #############################################################################
enum class InterpolationMode : uint8_t {
    INTERP_NONE = 0,
    INTERP_LINEAR = 1,
    INTERP_CUBIC = 2,
    INTERP_HERMITE = 3
};

// #############################################################################
// Compression type
// #############################################################################
enum class CompressionType : uint8_t {
    COMPRESS_NONE = 0,
    COMPRESS_QUANTIZE = 1,
    COMPRESS_DELTA = 2,
    COMPRESS_RLE = 3
};

// #############################################################################
// Replicated property configuration
// #############################################################################
struct ReplicatedProperty {
    StringName name;
    VariantType type = VariantType::NIL;
    SynchronizationMode sync_mode = SynchronizationMode::SYNC_ON_CHANGE;
    InterpolationMode interp_mode = InterpolationMode::INTERP_LINEAR;
    CompressionType compression = CompressionType::COMPRESS_NONE;
    float sync_interval = 0.0f;
    float interp_delay = 0.1f;
    bool reliable = true;
    int priority = 0;
};

// #############################################################################
// Snapshot data for state interpolation
// #############################################################################
struct StateSnapshot {
    uint32_t tick = 0;
    double timestamp = 0.0;
    std::unordered_map<StringName, Variant> properties;
};

// #############################################################################
// SceneReplicationConfig - Replication configuration resource
// #############################################################################
class SceneReplicationConfig : public Resource {
    XTU_GODOT_REGISTER_CLASS(SceneReplicationConfig, Resource)

private:
    std::vector<ReplicatedProperty> m_properties;
    std::unordered_map<StringName, size_t> m_property_index;
    ReplicationAuthority m_authority = ReplicationAuthority::AUTHORITY_SERVER;
    float m_default_sync_interval = 0.05f;
    bool m_spawnable = true;
    int32_t m_spawn_limit = 0;
    std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("SceneReplicationConfig"); }

    void set_authority(ReplicationAuthority auth) { m_authority = auth; }
    ReplicationAuthority get_authority() const { return m_authority; }

    void set_spawnable(bool spawnable) { m_spawnable = spawnable; }
    bool is_spawnable() const { return m_spawnable; }

    void set_spawn_limit(int32_t limit) { m_spawn_limit = limit; }
    int32_t get_spawn_limit() const { return m_spawn_limit; }

    void add_property(const ReplicatedProperty& prop) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_property_index[prop.name] = m_properties.size();
        m_properties.push_back(prop);
    }

    void remove_property(const StringName& name) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_property_index.find(name);
        if (it != m_property_index.end()) {
            m_properties.erase(m_properties.begin() + it->second);
            rebuild_index();
        }
    }

    const ReplicatedProperty* get_property(const StringName& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_property_index.find(name);
        return it != m_property_index.end() ? &m_properties[it->second] : nullptr;
    }

    std::vector<ReplicatedProperty> get_properties() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_properties;
    }

    bool has_property(const StringName& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_property_index.find(name) != m_property_index.end();
    }

private:
    void rebuild_index() {
        m_property_index.clear();
        for (size_t i = 0; i < m_properties.size(); ++i) {
            m_property_index[m_properties[i].name] = i;
        }
    }
};

// #############################################################################
// MultiplayerSpawner - Network object spawning with prediction
// #############################################################################
class MultiplayerSpawner : public Node {
    XTU_GODOT_REGISTER_CLASS(MultiplayerSpawner, Node)

private:
    NodePath m_spawn_path;
    Ref<SceneReplicationConfig> m_replication_config;
    int32_t m_spawn_limit = 0;
    float m_spawn_rate = 1.0f;
    bool m_auto_spawn = true;
    int32_t m_prediction_buffer = 3;
    std::queue<float> m_spawn_times;
    std::unordered_map<int32_t, Node*> m_spawned_nodes;
    std::unordered_map<int32_t, std::queue<StateSnapshot>> m_prediction_history;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("MultiplayerSpawner"); }

    void set_spawn_path(const NodePath& path) { m_spawn_path = path; }
    NodePath get_spawn_path() const { return m_spawn_path; }

    void set_replication_config(const Ref<SceneReplicationConfig>& config) { m_replication_config = config; }
    Ref<SceneReplicationConfig> get_replication_config() const { return m_replication_config; }

    void set_spawn_limit(int32_t limit) { m_spawn_limit = limit; }
    int32_t get_spawn_limit() const { return m_spawn_limit; }

    void set_spawn_rate(float rate) { m_spawn_rate = rate; }
    float get_spawn_rate() const { return m_spawn_rate; }

    void set_auto_spawn(bool enabled) { m_auto_spawn = enabled; }
    bool is_auto_spawn_enabled() const { return m_auto_spawn; }

    void set_prediction_buffer(int32_t frames) { m_prediction_buffer = frames; }
    int32_t get_prediction_buffer() const { return m_prediction_buffer; }

    Node* spawn(const Variant& data = Variant()) {
        if (!is_multiplayer_authority()) return nullptr;
        
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_spawn_limit > 0 && static_cast<int32_t>(m_spawned_nodes.size()) >= m_spawn_limit) {
            return nullptr;
        }

        Node* parent = get_node_or_null(m_spawn_path);
        if (!parent) parent = this;

        // Instantiate the scene
        Node* instance = nullptr;
        // ... instantiation logic
        
        if (instance) {
            int32_t network_id = generate_network_id();
            instance->set_meta("network_id", network_id);
            m_spawned_nodes[network_id] = instance;
            parent->add_child(instance);
            emit_signal("spawned", instance, data);
        }

        return instance;
    }

    void despawn(Node* node) {
        if (!node) return;
        int32_t network_id = node->get_meta("network_id").as<int32_t>();
        if (network_id > 0) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_spawned_nodes.erase(network_id);
            m_prediction_history.erase(network_id);
        }
        node->queue_free();
        emit_signal("despawned", node);
    }

    void despawn_all() {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto& kv : m_spawned_nodes) {
            kv.second->queue_free();
        }
        m_spawned_nodes.clear();
        m_prediction_history.clear();
    }

    Node* get_spawned_node(int32_t network_id) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_spawned_nodes.find(network_id);
        return it != m_spawned_nodes.end() ? it->second : nullptr;
    }

    std::vector<Node*> get_spawned_nodes() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<Node*> result;
        for (const auto& kv : m_spawned_nodes) {
            result.push_back(kv.second);
        }
        return result;
    }

    void predict_state(int32_t network_id, uint32_t tick, const std::unordered_map<StringName, Variant>& state) {
        std::lock_guard<std::mutex> lock(m_mutex);
        StateSnapshot snapshot;
        snapshot.tick = tick;
        snapshot.timestamp = OS::get_singleton()->get_ticks_msec() / 1000.0;
        snapshot.properties = state;
        
        auto& history = m_prediction_history[network_id];
        history.push(snapshot);
        while (history.size() > static_cast<size_t>(m_prediction_buffer)) {
            history.pop();
        }
    }

    void apply_prediction(int32_t network_id, uint32_t target_tick) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_prediction_history.find(network_id);
        if (it == m_prediction_history.end()) return;

        Node* node = get_spawned_node(network_id);
        if (!node) return;

        auto& history = it->second;
        // Find best matching snapshot
        StateSnapshot best_match;
        for (auto& snapshot : history.get_container()) {
            if (snapshot.tick <= target_tick) {
                best_match = snapshot;
            }
        }

        for (const auto& kv : best_match.properties) {
            node->set(kv.first, kv.second);
        }
    }

    bool is_multiplayer_authority() const {
        return MultiplayerAPI::get_singleton()->is_server();
    }

private:
    int32_t generate_network_id() {
        static std::atomic<int32_t> next_id{1};
        return next_id++;
    }
};

// #############################################################################
// MultiplayerSynchronizer - Advanced state synchronization
// #############################################################################
class MultiplayerSynchronizer : public Node {
    XTU_GODOT_REGISTER_CLASS(MultiplayerSynchronizer, Node)

private:
    NodePath m_root_path;
    Ref<SceneReplicationConfig> m_config;
    float m_replication_interval = 0.0f;
    float m_time_since_last_sync = 0.0f;
    uint32_t m_tick = 0;
    bool m_public_visibility = true;
    bool m_delta_compression = true;
    bool m_interpolation_enabled = true;
    float m_interpolation_delay = 0.1f;
    std::unordered_map<StringName, Variant> m_last_synced_values;
    std::queue<StateSnapshot> m_interpolation_buffer;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("MultiplayerSynchronizer"); }

    void set_root_path(const NodePath& path) { m_root_path = path; }
    NodePath get_root_path() const { return m_root_path; }

    void set_replication_config(const Ref<SceneReplicationConfig>& config) { m_config = config; }
    Ref<SceneReplicationConfig> get_replication_config() const { return m_config; }

    void set_replication_interval(float interval) { m_replication_interval = interval; }
    float get_replication_interval() const { return m_replication_interval; }

    void set_delta_compression(bool enabled) { m_delta_compression = enabled; }
    bool is_delta_compression_enabled() const { return m_delta_compression; }

    void set_interpolation_enabled(bool enabled) { m_interpolation_enabled = enabled; }
    bool is_interpolation_enabled() const { return m_interpolation_enabled; }

    void set_interpolation_delay(float delay) { m_interpolation_delay = delay; }
    float get_interpolation_delay() const { return m_interpolation_delay; }

    void set_visibility_public(bool visible) { m_public_visibility = visible; }
    bool is_visibility_public() const { return m_public_visibility; }

    void sync_state() {
        if (!m_config.is_valid()) return;

        Node* root = m_root_path.is_empty() ? this : get_node_or_null(m_root_path);
        if (!root) return;

        std::unordered_map<StringName, Variant> current_state;
        bool has_changes = false;

        for (const auto& prop : m_config->get_properties()) {
            Variant value = root->get(prop.name);
            current_state[prop.name] = value;

            if (prop.sync_mode == SynchronizationMode::SYNC_ON_CHANGE) {
                auto it = m_last_synced_values.find(prop.name);
                if (it == m_last_synced_values.end() || it->second != value) {
                    has_changes = true;
                }
            }
        }

        if (has_changes || m_replication_interval <= 0.0f) {
            send_state_update(current_state);
            m_last_synced_values = current_state;
        }
    }

    void receive_state_update(int32_t from_peer, const std::vector<uint8_t>& data) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        StateSnapshot snapshot;
        snapshot.tick = ++m_tick;
        snapshot.timestamp = OS::get_singleton()->get_ticks_msec() / 1000.0;
        snapshot.properties = deserialize_state(data);

        if (m_interpolation_enabled) {
            m_interpolation_buffer.push(snapshot);
            while (m_interpolation_buffer.size() > 30) {
                m_interpolation_buffer.pop();
            }
        } else {
            apply_snapshot(snapshot);
        }
    }

    void _process(double delta) override {
        if (is_multiplayer_authority()) {
            m_time_since_last_sync += delta;
            if (m_replication_interval > 0.0f && m_time_since_last_sync >= m_replication_interval) {
                sync_state();
                m_time_since_last_sync = 0.0f;
            }
        } else if (m_interpolation_enabled) {
            interpolate_state(delta);
        }
    }

    bool is_multiplayer_authority() const {
        if (m_config.is_valid() && m_config->get_authority() == ReplicationAuthority::AUTHORITY_CLIENT) {
            return !MultiplayerAPI::get_singleton()->is_server();
        }
        return MultiplayerAPI::get_singleton()->is_server();
    }

private:
    void send_state_update(const std::unordered_map<StringName, Variant>& state) {
        std::vector<uint8_t> data = serialize_state(state);
        // Send via MultiplayerAPI
    }

    std::vector<uint8_t> serialize_state(const std::unordered_map<StringName, Variant>& state) const {
        std::vector<uint8_t> result;
        // Binary serialization with compression
        return result;
    }

    std::unordered_map<StringName, Variant> deserialize_state(const std::vector<uint8_t>& data) const {
        std::unordered_map<StringName, Variant> result;
        // Binary deserialization
        return result;
    }

    void apply_snapshot(const StateSnapshot& snapshot) {
        Node* root = m_root_path.is_empty() ? this : get_node_or_null(m_root_path);
        if (!root) return;

        for (const auto& kv : snapshot.properties) {
            root->set(kv.first, kv.second);
        }
    }

    void interpolate_state(float delta) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_interpolation_buffer.size() < 2) return;

        double render_time = OS::get_singleton()->get_ticks_msec() / 1000.0 - m_interpolation_delay;
        
        const StateSnapshot* from = nullptr;
        const StateSnapshot* to = nullptr;

        auto& container = m_interpolation_buffer.get_container();
        for (size_t i = 0; i < container.size() - 1; ++i) {
            if (container[i].timestamp <= render_time && container[i + 1].timestamp >= render_time) {
                from = &container[i];
                to = &container[i + 1];
                break;
            }
        }

        if (!from || !to) return;

        float t = static_cast<float>((render_time - from->timestamp) / (to->timestamp - from->timestamp));
        t = std::clamp(t, 0.0f, 1.0f);

        Node* root = m_root_path.is_empty() ? this : get_node_or_null(m_root_path);
        if (!root) return;

        for (const auto& prop : m_config->get_properties()) {
            auto it_from = from->properties.find(prop.name);
            auto it_to = to->properties.find(prop.name);
            if (it_from == from->properties.end() || it_to == to->properties.end()) continue;

            Variant interpolated = interpolate_value(it_from->second, it_to->second, t, prop.interp_mode);
            root->set(prop.name, interpolated);
        }
    }

    Variant interpolate_value(const Variant& a, const Variant& b, float t, InterpolationMode mode) const {
        if (a.get_type() != b.get_type()) return b;

        switch (mode) {
            case InterpolationMode::INTERP_LINEAR:
                if (a.is_num()) {
                    return Variant(a.as<double>() * (1.0 - t) + b.as<double>() * t);
                } else if (a.get_type() == VariantType::VECTOR2) {
                    return Variant(a.as<vec2f>() * (1.0f - t) + b.as<vec2f>() * t);
                } else if (a.get_type() == VariantType::VECTOR3) {
                    return Variant(a.as<vec3f>() * (1.0f - t) + b.as<vec3f>() * t);
                } else if (a.get_type() == VariantType::QUATERNION) {
                    return Variant(quatf::slerp(a.as<quatf>(), b.as<quatf>(), t));
                }
                break;
            case InterpolationMode::INTERP_CUBIC:
                // Cubic interpolation
                break;
            default:
                break;
        }
        return t < 0.5f ? a : b;
    }
};

// #############################################################################
// CustomMultiplayerPeer - Custom transport layer
// #############################################################################
class CustomMultiplayerPeer : public MultiplayerPeer {
    XTU_GODOT_REGISTER_CLASS(CustomMultiplayerPeer, MultiplayerPeer)

private:
    std::function<void(const std::vector<uint8_t>&, int32_t)> m_send_callback;
    std::function<void()> m_connect_callback;
    std::function<void()> m_disconnect_callback;
    std::queue<std::pair<std::vector<uint8_t>, int32_t>> m_incoming_packets;
    int32_t m_unique_id = 0;
    MultiplayerPeerConnectionStatus m_status = MultiplayerPeerConnectionStatus::CONNECTION_DISCONNECTED;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("CustomMultiplayerPeer"); }

    void set_send_callback(std::function<void(const std::vector<uint8_t>&, int32_t)> cb) { m_send_callback = cb; }
    void set_connect_callback(std::function<void()> cb) { m_connect_callback = cb; }
    void set_disconnect_callback(std::function<void()> cb) { m_disconnect_callback = cb; }

    void set_unique_id(int32_t id) { m_unique_id = id; }
    int32_t get_unique_id() const override { return m_unique_id; }

    void set_connection_status(MultiplayerPeerConnectionStatus status) {
        m_status = status;
        if (status == MultiplayerPeerConnectionStatus::CONNECTION_CONNECTED && m_connect_callback) {
            m_connect_callback();
        } else if (status == MultiplayerPeerConnectionStatus::CONNECTION_DISCONNECTED && m_disconnect_callback) {
            m_disconnect_callback();
        }
    }

    MultiplayerPeerConnectionStatus get_connection_status() const override { return m_status; }

    void deliver_packet(const std::vector<uint8_t>& data, int32_t from_peer) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_incoming_packets.push({data, from_peer});
    }

    int32_t get_available_packet_count() const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        return static_cast<int32_t>(m_incoming_packets.size());
    }

    Error get_packet(const uint8_t** buffer, int32_t& size) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_incoming_packets.empty()) return ERR_UNAVAILABLE;
        
        auto& packet = m_incoming_packets.front();
        *buffer = packet.first.data();
        size = static_cast<int32_t>(packet.first.size());
        return OK;
    }

    Error put_packet(const uint8_t* buffer, int32_t size) override {
        if (m_send_callback) {
            std::vector<uint8_t> data(buffer, buffer + size);
            m_send_callback(data, m_target_peer);
            return OK;
        }
        return ERR_UNAVAILABLE;
    }

    int32_t get_packet_peer() const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_incoming_packets.empty() ? 0 : m_incoming_packets.front().second;
    }

    void close() override {
        set_connection_status(MultiplayerPeerConnectionStatus::CONNECTION_DISCONNECTED);
        std::lock_guard<std::mutex> lock(m_mutex);
        while (!m_incoming_packets.empty()) m_incoming_packets.pop();
    }

    bool is_server() const override { return m_unique_id == 1; }
};

} // namespace godot

// Bring into main namespace
using godot::SceneReplicationConfig;
using godot::MultiplayerSpawner;
using godot::MultiplayerSynchronizer;
using godot::CustomMultiplayerPeer;
using godot::ReplicationAuthority;
using godot::SynchronizationMode;
using godot::InterpolationMode;
using godot::CompressionType;
using godot::ReplicatedProperty;
using godot::StateSnapshot;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XMULTIPLAYER_ADVANCED_HPP