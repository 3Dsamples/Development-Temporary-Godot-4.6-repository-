// include/xtu/godot/xnetworking.hpp
// xtensor-unified - Multiplayer networking for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XNETWORKING_HPP
#define XTU_GODOT_XNETWORKING_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
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
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class MultiplayerAPI;
class SceneMultiplayer;
class MultiplayerPeer;
class MultiplayerSpawner;
class MultiplayerSynchronizer;

// #############################################################################
// RPC mode types
// #############################################################################
enum class RPCMode : uint8_t {
    RPC_MODE_DISABLED = 0,
    RPC_MODE_ANY_PEER = 1,
    RPC_MODE_AUTHORITY = 2,
    RPC_MODE_REMOTE = 3,
    RPC_MODE_MASTER = 4,
    RPC_MODE_PUPPET = 5
};

// #############################################################################
// Multiplayer peer connection status
// #############################################################################
enum class MultiplayerPeerConnectionStatus : uint8_t {
    CONNECTION_DISCONNECTED = 0,
    CONNECTION_CONNECTING = 1,
    CONNECTION_CONNECTED = 2
};

// #############################################################################
// Multiplayer peer transfer mode
// #############################################################################
enum class MultiplayerPeerTransferMode : uint8_t {
    TRANSFER_MODE_UNRELIABLE = 0,
    TRANSFER_MODE_UNRELIABLE_ORDERED = 1,
    TRANSFER_MODE_RELIABLE = 2
};

// #############################################################################
// Networked multiplayer peer interface
// #############################################################################
class MultiplayerPeer : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(MultiplayerPeer, RefCounted)

private:
    MultiplayerPeerConnectionStatus m_connection_status = MultiplayerPeerConnectionStatus::CONNECTION_DISCONNECTED;
    int32_t m_unique_id = 0;
    bool m_refuse_connections = false;
    MultiplayerPeerTransferMode m_transfer_mode = MultiplayerPeerTransferMode::TRANSFER_MODE_RELIABLE;
    int32_t m_transfer_channel = 0;

public:
    static StringName get_class_static() { return StringName("MultiplayerPeer"); }

    virtual Error create_server(int32_t port, int32_t max_clients = 32, const String& bind_address = "*") {
        return ERR_UNAVAILABLE;
    }

    virtual Error create_client(const String& address, int32_t port) {
        return ERR_UNAVAILABLE;
    }

    virtual void close() {
        m_connection_status = MultiplayerPeerConnectionStatus::CONNECTION_DISCONNECTED;
    }

    virtual void poll() {}

    virtual int32_t get_available_packet_count() const { return 0; }

    virtual Error get_packet(const uint8_t** buffer, int32_t& size) {
        return ERR_UNAVAILABLE;
    }

    virtual Error put_packet(const uint8_t* buffer, int32_t size) {
        return ERR_UNAVAILABLE;
    }

    virtual int32_t get_packet_peer() const { return 0; }
    virtual MultiplayerPeerTransferMode get_packet_mode() const { return m_transfer_mode; }
    virtual int32_t get_packet_channel() const { return 0; }

    virtual void set_transfer_mode(MultiplayerPeerTransferMode mode) { m_transfer_mode = mode; }
    virtual MultiplayerPeerTransferMode get_transfer_mode() const { return m_transfer_mode; }

    virtual void set_transfer_channel(int32_t channel) { m_transfer_channel = channel; }
    virtual int32_t get_transfer_channel() const { return m_transfer_channel; }

    virtual void set_target_peer(int32_t id) { m_target_peer = id; }
    virtual int32_t get_target_peer() const { return m_target_peer; }

    virtual int32_t get_unique_id() const { return m_unique_id; }
    virtual bool is_server() const { return m_unique_id == 1; }

    virtual void set_refuse_connections(bool refuse) { m_refuse_connections = refuse; }
    virtual bool is_refusing_connections() const { return m_refuse_connections; }

    virtual MultiplayerPeerConnectionStatus get_connection_status() const { return m_connection_status; }

    virtual void disconnect_peer(int32_t id, bool force = false) {}

    virtual String get_peer_address(int32_t id) const { return String(); }
    virtual int32_t get_peer_port(int32_t id) const { return 0; }

    virtual int32_t get_max_packet_size() const { return 1200; }

protected:
    int32_t m_target_peer = 0;
};

// #############################################################################
// ENetMultiplayerPeer - ENet implementation
// #############################################################################
class ENetMultiplayerPeer : public MultiplayerPeer {
    XTU_GODOT_REGISTER_CLASS(ENetMultiplayerPeer, MultiplayerPeer)

private:
    struct Peer {
        uint64_t host;
        uint32_t connect_id;
        int32_t id;
    };

    void* m_host = nullptr;
    std::unordered_map<int32_t, Peer> m_peers;
    std::queue<std::vector<uint8_t>> m_incoming_packets;
    std::mutex m_mutex;
    int32_t m_max_clients = 32;
    int32_t m_in_bandwidth = 0;
    int32_t m_out_bandwidth = 0;
    String m_bind_address = "*";

public:
    static StringName get_class_static() { return StringName("ENetMultiplayerPeer"); }

    Error create_server(int32_t port, int32_t max_clients = 32, const String& bind_address = "*") override {
        m_max_clients = max_clients;
        m_bind_address = bind_address;
        // ENet initialization
        return OK;
    }

    Error create_client(const String& address, int32_t port) override {
        // ENet client connect
        return OK;
    }

    void close() override {
        MultiplayerPeer::close();
        // ENet cleanup
    }

    void poll() override {
        // Process ENet events
    }

    Error put_packet(const uint8_t* buffer, int32_t size) override {
        // Send via ENet
        return OK;
    }

    void disconnect_peer(int32_t id, bool force = false) override {
        // Disconnect specific peer
    }
};

// #############################################################################
// WebRTCMultiplayerPeer - WebRTC implementation
// #############################################################################
class WebRTCMultiplayerPeer : public MultiplayerPeer {
    XTU_GODOT_REGISTER_CLASS(WebRTCMultiplayerPeer, MultiplayerPeer)

private:
    std::unordered_map<int32_t, void*> m_connections;
    std::vector<String> m_stun_servers;
    std::vector<String> m_turn_servers;

public:
    static StringName get_class_static() { return StringName("WebRTCMultiplayerPeer"); }

    void add_ice_server(const String& server) {
        if (server.begins_with("stun:")) {
            m_stun_servers.push_back(server);
        } else if (server.begins_with("turn:")) {
            m_turn_servers.push_back(server);
        }
    }

    Error create_server(int32_t port, int32_t max_clients = 32, const String& bind_address = "*") override {
        // WebRTC signaling server
        return OK;
    }

    Error create_client(const String& address, int32_t port) override {
        // WebRTC client
        return OK;
    }

    Error create_data_channel(int32_t peer_id, const String& label) {
        return OK;
    }
};

// #############################################################################
// SceneMultiplayer - Multiplayer API implementation
// #############################################################################
class SceneMultiplayer : public MultiplayerAPI {
    XTU_GODOT_REGISTER_CLASS(SceneMultiplayer, MultiplayerAPI)

private:
    Ref<MultiplayerPeer> m_peer;
    Node* m_root_node = nullptr;
    std::unordered_map<int32_t, Node*> m_network_nodes;
    std::unordered_set<StringName> m_rpc_methods;
    std::mutex m_mutex;
    bool m_allow_object_decoding = false;
    bool m_server_relay = true;

public:
    static StringName get_class_static() { return StringName("SceneMultiplayer"); }

    void set_multiplayer_peer(const Ref<MultiplayerPeer>& peer) {
        m_peer = peer;
    }

    Ref<MultiplayerPeer> get_multiplayer_peer() const { return m_peer; }

    void set_root_node(Node* node) { m_root_node = node; }
    Node* get_root_node() const { return m_root_node; }

    int32_t get_unique_id() const {
        return m_peer.is_valid() ? m_peer->get_unique_id() : 0;
    }

    bool is_server() const {
        return m_peer.is_valid() && m_peer->is_server();
    }

    void set_allow_object_decoding(bool allow) { m_allow_object_decoding = allow; }
    bool is_object_decoding_allowed() const { return m_allow_object_decoding; }

    void set_server_relay(bool enabled) { m_server_relay = enabled; }
    bool is_server_relay_enabled() const { return m_server_relay; }

    void poll() {
        if (!m_peer.is_valid()) return;
        m_peer->poll();
        while (m_peer->get_available_packet_count() > 0) {
            const uint8_t* buffer;
            int32_t size;
            if (m_peer->get_packet(&buffer, &size) == OK) {
                process_packet(m_peer->get_packet_peer(), buffer, size);
            }
        }
    }

    Error rpc(int32_t peer_id, Object* object, const StringName& method, const std::vector<Variant>& args) {
        if (!m_peer.is_valid()) return ERR_UNCONFIGURED;
        // Serialize and send RPC
        return OK;
    }

    void register_rpc_method(const StringName& method) {
        m_rpc_methods.insert(method);
    }

private:
    void process_packet(int32_t from_peer, const uint8_t* data, int32_t size) {
        // Deserialize and execute RPC
    }
};

// #############################################################################
// MultiplayerAPI - Global multiplayer singleton
// #############################################################################
class MultiplayerAPI : public Object {
    XTU_GODOT_REGISTER_CLASS(MultiplayerAPI, Object)

private:
    static MultiplayerAPI* s_singleton;
    Ref<SceneMultiplayer> m_default_multiplayer;

public:
    static MultiplayerAPI* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("MultiplayerAPI"); }

    MultiplayerAPI() {
        s_singleton = this;
        m_default_multiplayer.instance();
    }

    ~MultiplayerAPI() { s_singleton = nullptr; }

    void set_default_multiplayer(const Ref<SceneMultiplayer>& mp) { m_default_multiplayer = mp; }
    Ref<SceneMultiplayer> get_default_multiplayer() const { return m_default_multiplayer; }

    bool has_multiplayer_peer() const {
        return m_default_multiplayer.is_valid() && m_default_multiplayer->get_multiplayer_peer().is_valid();
    }

    Ref<MultiplayerPeer> get_multiplayer_peer() const {
        return m_default_multiplayer.is_valid() ? m_default_multiplayer->get_multiplayer_peer() : Ref<MultiplayerPeer>();
    }

    int32_t get_unique_id() const {
        return m_default_multiplayer.is_valid() ? m_default_multiplayer->get_unique_id() : 0;
    }

    bool is_server() const {
        return m_default_multiplayer.is_valid() && m_default_multiplayer->is_server();
    }

    Vector<int32_t> get_network_connected_peers() const {
        Vector<int32_t> peers;
        // Collect connected peer IDs
        return peers;
    }

    void rpc(int32_t peer_id, Object* object, const StringName& method, const std::vector<Variant>& args = {}) {
        if (m_default_multiplayer.is_valid()) {
            m_default_multiplayer->rpc(peer_id, object, method, args);
        }
    }
};

// #############################################################################
// MultiplayerSpawner - Network object spawning
// #############################################################################
class MultiplayerSpawner : public Node {
    XTU_GODOT_REGISTER_CLASS(MultiplayerSpawner, Node)

private:
    NodePath m_spawn_path;
    int32_t m_spawn_limit = 0;
    float m_spawn_rate = 1.0f;
    bool m_auto_spawn = true;
    std::queue<float> m_spawn_times;
    std::unordered_map<int32_t, Node*> m_spawned_nodes;

public:
    static StringName get_class_static() { return StringName("MultiplayerSpawner"); }

    void set_spawn_path(const NodePath& path) { m_spawn_path = path; }
    NodePath get_spawn_path() const { return m_spawn_path; }

    void set_spawn_limit(int32_t limit) { m_spawn_limit = limit; }
    int32_t get_spawn_limit() const { return m_spawn_limit; }

    void set_spawn_rate(float rate) { m_spawn_rate = rate; }
    float get_spawn_rate() const { return m_spawn_rate; }

    void set_auto_spawn(bool auto_spawn) { m_auto_spawn = auto_spawn; }
    bool get_auto_spawn() const { return m_auto_spawn; }

    Node* spawn(const Variant& data = Variant()) {
        if (!is_multiplayer_authority()) return nullptr;
        // Instantiate and spawn
        return nullptr;
    }

    void despawn(Node* node) {}
    void despawn_all() {}

    bool is_multiplayer_authority() const {
        return MultiplayerAPI::get_singleton()->is_server();
    }

    void _process(double delta) override {
        if (!m_auto_spawn || !is_multiplayer_authority()) return;
        // Spawn logic
    }
};

// #############################################################################
// MultiplayerSynchronizer - State synchronization
// #############################################################################
class MultiplayerSynchronizer : public Node {
    XTU_GODOT_REGISTER_CLASS(MultiplayerSynchronizer, Node)

private:
    NodePath m_root_path;
    float m_replication_interval = 0.0f;
    float m_delta_interval = 0.0f;
    bool m_public_visibility = true;
    bool m_replicate_visibility = true;
    float m_time_since_last_sync = 0.0f;
    std::unordered_set<StringName> m_synced_properties;

public:
    static StringName get_class_static() { return StringName("MultiplayerSynchronizer"); }

    void set_root_path(const NodePath& path) { m_root_path = path; }
    NodePath get_root_path() const { return m_root_path; }

    void set_replication_interval(float interval) { m_replication_interval = interval; }
    float get_replication_interval() const { return m_replication_interval; }

    void set_delta_interval(float interval) { m_delta_interval = interval; }
    float get_delta_interval() const { return m_delta_interval; }

    void set_visibility_public(bool visible) { m_public_visibility = visible; }
    bool is_visibility_public() const { return m_public_visibility; }

    void add_synced_property(const StringName& property) {
        m_synced_properties.insert(property);
    }

    void remove_synced_property(const StringName& property) {
        m_synced_properties.erase(property);
    }

    bool has_synced_property(const StringName& property) const {
        return m_synced_properties.find(property) != m_synced_properties.end();
    }

    bool is_multiplayer_authority() const {
        return MultiplayerAPI::get_singleton()->is_server();
    }

    void _process(double delta) override {
        if (!is_multiplayer_authority()) return;
        m_time_since_last_sync += delta;
        if (m_replication_interval > 0 && m_time_since_last_sync >= m_replication_interval) {
            sync_state();
            m_time_since_last_sync = 0.0f;
        }
    }

private:
    void sync_state() {
        // Serialize and broadcast synced properties
    }
};

} // namespace godot

// Bring into main namespace
using godot::MultiplayerAPI;
using godot::SceneMultiplayer;
using godot::MultiplayerPeer;
using godot::ENetMultiplayerPeer;
using godot::WebRTCMultiplayerPeer;
using godot::MultiplayerSpawner;
using godot::MultiplayerSynchronizer;
using godot::RPCMode;
using godot::MultiplayerPeerConnectionStatus;
using godot::MultiplayerPeerTransferMode;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XNETWORKING_HPP