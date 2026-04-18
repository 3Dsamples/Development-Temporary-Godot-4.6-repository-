// include/xtu/godot/xnetwork_profiler.hpp
// xtensor-unified - Network profiling and debugging for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XNETWORK_PROFILER_HPP
#define XTU_GODOT_XNETWORK_PROFILER_HPP

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xeditor.hpp"
#include "xtu/godot/xgui.hpp"
#include "xtu/godot/xnetworking.hpp"
#include "xtu/io/xio_json.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace multiplayer {

// #############################################################################
// Forward declarations
// #############################################################################
class NetworkProfiler;
class LatencySimulator;
class MultiplayerDebugger;
class MultiplayerEditorPlugin;

// #############################################################################
// Network packet type
// #############################################################################
enum class NetworkPacketType : uint8_t {
    PACKET_UNKNOWN = 0,
    PACKET_RPC = 1,
    PACKET_SYNC = 2,
    PACKET_SPAWN = 3,
    PACKET_DESPAWN = 4,
    PACKET_INPUT = 5,
    PACKET_CUSTOM = 6
};

// #############################################################################
// Network statistics
// #############################################################################
struct NetworkStats {
    uint64_t bytes_sent = 0;
    uint64_t bytes_received = 0;
    uint64_t packets_sent = 0;
    uint64_t packets_received = 0;
    uint64_t packets_lost = 0;
    uint64_t rpcs_sent = 0;
    uint64_t rpcs_received = 0;
    float avg_latency_ms = 0.0f;
    float jitter_ms = 0.0f;
    float packet_loss_rate = 0.0f;
    float bandwidth_sent_kbps = 0.0f;
    float bandwidth_received_kbps = 0.0f;
    uint64_t timestamp = 0;
};

// #############################################################################
// Packet log entry
// #############################################################################
struct PacketLogEntry {
    NetworkPacketType type = NetworkPacketType::PACKET_UNKNOWN;
    int32_t peer_id = 0;
    size_t size = 0;
    uint64_t timestamp = 0;
    float latency_ms = 0.0f;
    String method;  // For RPC
    String path;    // For sync/spawn
};

// #############################################################################
// NetworkProfiler - Network statistics collector
// #############################################################################
class NetworkProfiler : public Object {
    XTU_GODOT_REGISTER_CLASS(NetworkProfiler, Object)

private:
    static NetworkProfiler* s_singleton;
    
    std::unordered_map<int32_t, NetworkStats> m_peer_stats;
    std::unordered_map<int32_t, std::deque<float>> m_peer_latency_history;
    std::deque<PacketLogEntry> m_packet_log;
    std::deque<NetworkStats> m_stats_history;
    
    std::atomic<uint64_t> m_total_bytes_sent{0};
    std::atomic<uint64_t> m_total_bytes_received{0};
    std::atomic<uint64_t> m_total_packets_sent{0};
    std::atomic<uint64_t> m_total_packets_received{0};
    std::atomic<uint64_t> m_total_rpcs_sent{0};
    std::atomic<uint64_t> m_total_rpcs_received{0};
    
    mutable std::mutex m_mutex;
    size_t m_max_log_entries = 1000;
    size_t m_max_stats_history = 300;  // 5 seconds at 60 FPS
    bool m_enabled = true;
    uint64_t m_last_reset_time = 0;
    uint64_t m_last_bandwidth_update = 0;

public:
    static NetworkProfiler* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("NetworkProfiler"); }

    NetworkProfiler() {
        s_singleton = this;
        m_last_reset_time = OS::get_singleton()->get_ticks_msec();
        m_last_bandwidth_update = m_last_reset_time;
    }

    ~NetworkProfiler() { s_singleton = nullptr; }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void reset() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_peer_stats.clear();
        m_peer_latency_history.clear();
        m_packet_log.clear();
        m_stats_history.clear();
        m_total_bytes_sent = 0;
        m_total_bytes_received = 0;
        m_total_packets_sent = 0;
        m_total_packets_received = 0;
        m_total_rpcs_sent = 0;
        m_total_rpcs_received = 0;
        m_last_reset_time = OS::get_singleton()->get_ticks_msec();
    }

    void record_packet_sent(int32_t peer_id, size_t size, NetworkPacketType type = NetworkPacketType::PACKET_UNKNOWN,
                            const String& method = "", const String& path = "") {
        if (!m_enabled) return;
        
        std::lock_guard<std::mutex> lock(m_mutex);
        m_total_bytes_sent += size;
        m_total_packets_sent++;
        
        auto& stats = m_peer_stats[peer_id];
        stats.bytes_sent += size;
        stats.packets_sent++;
        
        if (type == NetworkPacketType::PACKET_RPC) {
            m_total_rpcs_sent++;
            stats.rpcs_sent++;
        }
        
        PacketLogEntry entry;
        entry.type = type;
        entry.peer_id = peer_id;
        entry.size = size;
        entry.timestamp = OS::get_singleton()->get_ticks_msec();
        entry.method = method;
        entry.path = path;
        
        m_packet_log.push_back(entry);
        while (m_packet_log.size() > m_max_log_entries) {
            m_packet_log.pop_front();
        }
    }

    void record_packet_received(int32_t peer_id, size_t size, NetworkPacketType type = NetworkPacketType::PACKET_UNKNOWN,
                                const String& method = "", const String& path = "") {
        if (!m_enabled) return;
        
        std::lock_guard<std::mutex> lock(m_mutex);
        m_total_bytes_received += size;
        m_total_packets_received++;
        
        auto& stats = m_peer_stats[peer_id];
        stats.bytes_received += size;
        stats.packets_received++;
        
        if (type == NetworkPacketType::PACKET_RPC) {
            m_total_rpcs_received++;
            stats.rpcs_received++;
        }
        
        PacketLogEntry entry;
        entry.type = type;
        entry.peer_id = peer_id;
        entry.size = size;
        entry.timestamp = OS::get_singleton()->get_ticks_msec();
        entry.method = method;
        entry.path = path;
        
        m_packet_log.push_back(entry);
        while (m_packet_log.size() > m_max_log_entries) {
            m_packet_log.pop_front();
        }
    }

    void record_latency(int32_t peer_id, float latency_ms) {
        if (!m_enabled) return;
        
        std::lock_guard<std::mutex> lock(m_mutex);
        auto& history = m_peer_latency_history[peer_id];
        history.push_back(latency_ms);
        while (history.size() > 60) {
            history.pop_front();
        }
        
        auto& stats = m_peer_stats[peer_id];
        if (history.size() >= 2) {
            float sum = 0.0f;
            float prev = history.front();
            float jitter_sum = 0.0f;
            for (size_t i = 0; i < history.size(); ++i) {
                sum += history[i];
                if (i > 0) {
                    jitter_sum += std::abs(history[i] - prev);
                    prev = history[i];
                }
            }
            stats.avg_latency_ms = sum / history.size();
            stats.jitter_ms = jitter_sum / (history.size() - 1);
        }
    }

    void record_packet_loss(int32_t peer_id, bool lost) {
        if (!m_enabled) return;
        
        std::lock_guard<std::mutex> lock(m_mutex);
        auto& stats = m_peer_stats[peer_id];
        if (lost) stats.packets_lost++;
        
        uint64_t total = stats.packets_sent + stats.packets_lost;
        if (total > 0) {
            stats.packet_loss_rate = static_cast<float>(stats.packets_lost) / total;
        }
    }

    void update_bandwidth() {
        std::lock_guard<std::mutex> lock(m_mutex);
        uint64_t now = OS::get_singleton()->get_ticks_msec();
        float delta_sec = (now - m_last_bandwidth_update) / 1000.0f;
        if (delta_sec < 0.1f) return;
        
        NetworkStats snapshot;
        snapshot.bytes_sent = m_total_bytes_sent;
        snapshot.bytes_received = m_total_bytes_received;
        snapshot.packets_sent = m_total_packets_sent;
        snapshot.packets_received = m_total_packets_received;
        snapshot.rpcs_sent = m_total_rpcs_sent;
        snapshot.rpcs_received = m_total_rpcs_received;
        snapshot.timestamp = now;
        
        if (!m_stats_history.empty()) {
            const NetworkStats& prev = m_stats_history.back();
            float dt = (snapshot.timestamp - prev.timestamp) / 1000.0f;
            if (dt > 0.0f) {
                snapshot.bandwidth_sent_kbps = (snapshot.bytes_sent - prev.bytes_sent) * 8.0f / (dt * 1000.0f);
                snapshot.bandwidth_received_kbps = (snapshot.bytes_received - prev.bytes_received) * 8.0f / (dt * 1000.0f);
            }
        }
        
        m_stats_history.push_back(snapshot);
        while (m_stats_history.size() > m_max_stats_history) {
            m_stats_history.pop_front();
        }
        
        m_last_bandwidth_update = now;
    }

    NetworkStats get_total_stats() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        NetworkStats total;
        total.bytes_sent = m_total_bytes_sent;
        total.bytes_received = m_total_bytes_received;
        total.packets_sent = m_total_packets_sent;
        total.packets_received = m_total_packets_received;
        total.rpcs_sent = m_total_rpcs_sent;
        total.rpcs_received = m_total_rpcs_received;
        
        if (!m_stats_history.empty()) {
            total.bandwidth_sent_kbps = m_stats_history.back().bandwidth_sent_kbps;
            total.bandwidth_received_kbps = m_stats_history.back().bandwidth_received_kbps;
        }
        return total;
    }

    NetworkStats get_peer_stats(int32_t peer_id) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_peer_stats.find(peer_id);
        return it != m_peer_stats.end() ? it->second : NetworkStats();
    }

    std::vector<PacketLogEntry> get_packet_log(size_t max_entries = 100) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<PacketLogEntry> result;
        size_t start = m_packet_log.size() > max_entries ? m_packet_log.size() - max_entries : 0;
        for (size_t i = start; i < m_packet_log.size(); ++i) {
            result.push_back(m_packet_log[i]);
        }
        return result;
    }

    std::vector<NetworkStats> get_stats_history() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return std::vector<NetworkStats>(m_stats_history.begin(), m_stats_history.end());
    }

    void clear_logs() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_packet_log.clear();
    }
};

// #############################################################################
// LatencySimulator - Network condition simulator
// #############################################################################
class LatencySimulator : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(LatencySimulator, RefCounted)

public:
    struct DelayedPacket {
        std::vector<uint8_t> data;
        int32_t peer_id = 0;
        uint64_t deliver_time = 0;
    };

private:
    bool m_enabled = false;
    float m_latency_ms = 0.0f;
    float m_jitter_ms = 0.0f;
    float m_packet_loss_rate = 0.0f;
    float m_duplicate_rate = 0.0f;
    float m_reorder_rate = 0.0f;
    size_t m_bandwidth_limit_bps = 0;
    
    std::priority_queue<DelayedPacket, std::vector<DelayedPacket>,
        std::function<bool(const DelayedPacket&, const DelayedPacket&)>> m_delayed_packets;
    std::mt19937 m_rng;
    mutable std::mutex m_mutex;
    
    std::function<void(const std::vector<uint8_t>&, int32_t)> m_output_callback;

public:
    static StringName get_class_static() { return StringName("LatencySimulator"); }

    LatencySimulator() 
        : m_delayed_packets([](const DelayedPacket& a, const DelayedPacket& b) {
            return a.deliver_time > b.deliver_time;
        })
        , m_rng(std::random_device{}()) {}

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void set_latency(float ms) { m_latency_ms = std::max(0.0f, ms); }
    float get_latency() const { return m_latency_ms; }

    void set_jitter(float ms) { m_jitter_ms = std::max(0.0f, ms); }
    float get_jitter() const { return m_jitter_ms; }

    void set_packet_loss_rate(float rate) { m_packet_loss_rate = std::clamp(rate, 0.0f, 1.0f); }
    float get_packet_loss_rate() const { return m_packet_loss_rate; }

    void set_duplicate_rate(float rate) { m_duplicate_rate = std::clamp(rate, 0.0f, 1.0f); }
    float get_duplicate_rate() const { return m_duplicate_rate; }

    void set_reorder_rate(float rate) { m_reorder_rate = std::clamp(rate, 0.0f, 1.0f); }
    float get_reorder_rate() const { return m_reorder_rate; }

    void set_bandwidth_limit(size_t bps) { m_bandwidth_limit_bps = bps; }
    size_t get_bandwidth_limit() const { return m_bandwidth_limit_bps; }

    void set_output_callback(std::function<void(const std::vector<uint8_t>&, int32_t)> cb) {
        m_output_callback = cb;
    }

    void send_packet(const std::vector<uint8_t>& data, int32_t peer_id) {
        if (!m_enabled) {
            if (m_output_callback) m_output_callback(data, peer_id);
            return;
        }

        std::lock_guard<std::mutex> lock(m_mutex);

        // Packet loss
        std::uniform_real_distribution<float> loss_dist(0.0f, 1.0f);
        if (loss_dist(m_rng) < m_packet_loss_rate) {
            NetworkProfiler::get_singleton()->record_packet_loss(peer_id, true);
            return;
        }

        // Duplicate
        if (loss_dist(m_rng) < m_duplicate_rate) {
            DelayedPacket dup;
            dup.data = data;
            dup.peer_id = peer_id;
            dup.deliver_time = compute_delivery_time();
            m_delayed_packets.push(dup);
        }

        // Reorder (send immediately but mark as reordered)
        if (loss_dist(m_rng) < m_reorder_rate && m_output_callback) {
            m_output_callback(data, peer_id);
        }

        DelayedPacket packet;
        packet.data = data;
        packet.peer_id = peer_id;
        packet.deliver_time = compute_delivery_time();
        m_delayed_packets.push(packet);
    }

    void process() {
        if (!m_enabled) return;

        std::lock_guard<std::mutex> lock(m_mutex);
        uint64_t now = OS::get_singleton()->get_ticks_msec();

        while (!m_delayed_packets.empty() && m_delayed_packets.top().deliver_time <= now) {
            DelayedPacket packet = m_delayed_packets.top();
            m_delayed_packets.pop();

            if (m_output_callback) {
                m_output_callback(packet.data, packet.peer_id);
            }
        }
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        while (!m_delayed_packets.empty()) {
            m_delayed_packets.pop();
        }
    }

private:
    uint64_t compute_delivery_time() const {
        uint64_t now = OS::get_singleton()->get_ticks_msec();
        float delay = m_latency_ms;

        if (m_jitter_ms > 0.0f) {
            std::normal_distribution<float> jitter_dist(0.0f, m_jitter_ms);
            delay += jitter_dist(m_rng);
        }

        delay = std::max(0.0f, delay);
        return now + static_cast<uint64_t>(delay);
    }
};

// #############################################################################
// MultiplayerDebugger - Remote debugging interface
// #############################################################################
class MultiplayerDebugger : public Object {
    XTU_GODOT_REGISTER_CLASS(MultiplayerDebugger, Object)

private:
    static MultiplayerDebugger* s_singleton;
    bool m_enabled = false;
    int m_port = 6008;
    std::vector<int32_t> m_tracked_peers;
    std::function<void(const String&)> m_send_callback;
    mutable std::mutex m_mutex;

public:
    static MultiplayerDebugger* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("MultiplayerDebugger"); }

    MultiplayerDebugger() { s_singleton = this; }
    ~MultiplayerDebugger() { s_singleton = nullptr; }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    void set_port(int port) { m_port = port; }
    int get_port() const { return m_port; }

    void set_send_callback(std::function<void(const String&)> cb) { m_send_callback = cb; }

    void track_peer(int32_t peer_id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (std::find(m_tracked_peers.begin(), m_tracked_peers.end(), peer_id) == m_tracked_peers.end()) {
            m_tracked_peers.push_back(peer_id);
        }
    }

    void untrack_peer(int32_t peer_id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = std::find(m_tracked_peers.begin(), m_tracked_peers.end(), peer_id);
        if (it != m_tracked_peers.end()) {
            m_tracked_peers.erase(it);
        }
    }

    String get_state_json() const {
        io::json::JsonValue json;
        
        // Network stats
        NetworkStats total = NetworkProfiler::get_singleton()->get_total_stats();
        io::json::JsonValue stats_json;
        stats_json["bytes_sent"] = io::json::JsonValue(static_cast<double>(total.bytes_sent));
        stats_json["bytes_received"] = io::json::JsonValue(static_cast<double>(total.bytes_received));
        stats_json["packets_sent"] = io::json::JsonValue(static_cast<double>(total.packets_sent));
        stats_json["packets_received"] = io::json::JsonValue(static_cast<double>(total.packets_received));
        stats_json["rpcs_sent"] = io::json::JsonValue(static_cast<double>(total.rpcs_sent));
        stats_json["rpcs_received"] = io::json::JsonValue(static_cast<double>(total.rpcs_received));
        stats_json["bandwidth_sent_kbps"] = io::json::JsonValue(total.bandwidth_sent_kbps);
        stats_json["bandwidth_received_kbps"] = io::json::JsonValue(total.bandwidth_received_kbps);
        json["stats"] = stats_json;

        // Peer stats
        io::json::JsonValue peers_arr;
        std::lock_guard<std::mutex> lock(m_mutex);
        for (int32_t peer_id : m_tracked_peers) {
            NetworkStats peer_stats = NetworkProfiler::get_singleton()->get_peer_stats(peer_id);
            io::json::JsonValue peer_json;
            peer_json["id"] = io::json::JsonValue(static_cast<double>(peer_id));
            peer_json["bytes_sent"] = io::json::JsonValue(static_cast<double>(peer_stats.bytes_sent));
            peer_json["bytes_received"] = io::json::JsonValue(static_cast<double>(peer_stats.bytes_received));
            peer_json["avg_latency_ms"] = io::json::JsonValue(peer_stats.avg_latency_ms);
            peer_json["packet_loss_rate"] = io::json::JsonValue(peer_stats.packet_loss_rate);
            peers_arr.as_array().push_back(peer_json);
        }
        json["peers"] = peers_arr;

        return json.dump().c_str();
    }

    void process() {
        if (!m_enabled || !m_send_callback) return;

        static uint64_t last_send = 0;
        uint64_t now = OS::get_singleton()->get_ticks_msec();
        if (now - last_send > 500) {  // Send every 500ms
            m_send_callback(get_state_json());
            last_send = now;
        }
    }
};

// #############################################################################
// MultiplayerEditorPlugin - Editor UI for network debugging
// #############################################################################
#ifdef XTU_GODOT_EDITOR_ENABLED
class MultiplayerEditorPlugin : public EditorPlugin {
    XTU_GODOT_REGISTER_CLASS(MultiplayerEditorPlugin, EditorPlugin)

private:
    VBoxContainer* m_panel = nullptr;
    Tree* m_stats_tree = nullptr;
    Tree* m_peers_tree = nullptr;
    Tree* m_packet_log_tree = nullptr;
    Button* m_enable_btn = nullptr;
    Button* m_clear_btn = nullptr;
    Label* m_bandwidth_label = nullptr;
    GraphEdit* m_bandwidth_graph = nullptr;
    LatencySimulator* m_latency_simulator = nullptr;

public:
    static StringName get_class_static() { return StringName("MultiplayerEditorPlugin"); }

    StringName get_plugin_name() const override { return StringName("NetworkProfiler"); }

    void _enter_tree() override {
        EditorPlugin::_enter_tree();
        build_ui();
        m_latency_simulator = new LatencySimulator();
        NetworkProfiler::get_singleton()->set_enabled(true);
    }

    void _exit_tree() override {
        NetworkProfiler::get_singleton()->set_enabled(false);
        remove_control_from_container(m_panel);
        delete m_latency_simulator;
        EditorPlugin::_exit_tree();
    }

    void _process(double delta) override {
        if (!is_visible() || !m_panel->is_visible()) return;
        NetworkProfiler::get_singleton()->update_bandwidth();
        refresh_stats();
    }

private:
    void build_ui() {
        m_panel = new VBoxContainer();

        // Toolbar
        HBoxContainer* toolbar = new HBoxContainer();
        m_enable_btn = new Button();
        m_enable_btn->set_text("Pause");
        m_enable_btn->set_toggle_mode(true);
        m_enable_btn->connect("toggled", this, "on_enable_toggled");
        toolbar->add_child(m_enable_btn);

        m_clear_btn = new Button();
        m_clear_btn->set_text("Clear");
        m_clear_btn->connect("pressed", this, "on_clear");
        toolbar->add_child(m_clear_btn);

        m_bandwidth_label = new Label();
        m_bandwidth_label->set_text("↓ 0 KB/s  ↑ 0 KB/s");
        toolbar->add_child(m_bandwidth_label);
        m_panel->add_child(toolbar);

        // Tab container
        TabContainer* tabs = new TabContainer();
        tabs->set_v_size_flags(SIZE_EXPAND | SIZE_FILL);

        // Stats tab
        m_stats_tree = new Tree();
        m_stats_tree->set_columns(2);
        m_stats_tree->set_column_title(0, "Metric");
        m_stats_tree->set_column_title(1, "Value");
        tabs->add_child(m_stats_tree);
        tabs->set_tab_title(0, "Statistics");

        // Peers tab
        m_peers_tree = new Tree();
        m_peers_tree->set_columns(5);
        m_peers_tree->set_column_title(0, "ID");
        m_peers_tree->set_column_title(1, "Sent");
        m_peers_tree->set_column_title(2, "Received");
        m_peers_tree->set_column_title(3, "Latency");
        m_peers_tree->set_column_title(4, "Loss");
        tabs->add_child(m_peers_tree);
        tabs->set_tab_title(1, "Peers");

        // Packet log tab
        m_packet_log_tree = new Tree();
        m_packet_log_tree->set_columns(5);
        m_packet_log_tree->set_column_title(0, "Time");
        m_packet_log_tree->set_column_title(1, "Peer");
        m_packet_log_tree->set_column_title(2, "Type");
        m_packet_log_tree->set_column_title(3, "Size");
        m_packet_log_tree->set_column_title(4, "Details");
        tabs->add_child(m_packet_log_tree);
        tabs->set_tab_title(2, "Packet Log");

        // Latency simulator tab
        VBoxContainer* sim_container = new VBoxContainer();
        // Add simulator controls...
        tabs->add_child(sim_container);
        tabs->set_tab_title(3, "Simulator");

        m_panel->add_child(tabs);
        add_control_to_container(m_panel, "bottom");
    }

    void refresh_stats() {
        NetworkStats stats = NetworkProfiler::get_singleton()->get_total_stats();

        // Update stats tree
        m_stats_tree->clear();
        TreeItem* root = m_stats_tree->create_item();
        add_stat_row(root, "Bytes Sent", String::num(stats.bytes_sent));
        add_stat_row(root, "Bytes Received", String::num(stats.bytes_received));
        add_stat_row(root, "Packets Sent", String::num(stats.packets_sent));
        add_stat_row(root, "Packets Received", String::num(stats.packets_received));
        add_stat_row(root, "RPCs Sent", String::num(stats.rpcs_sent));
        add_stat_row(root, "RPCs Received", String::num(stats.rpcs_received));
        add_stat_row(root, "Send Rate", String::num(stats.bandwidth_sent_kbps, 2) + " KB/s");
        add_stat_row(root, "Receive Rate", String::num(stats.bandwidth_received_kbps, 2) + " KB/s");

        m_bandwidth_label->set_text("↓ " + String::num(stats.bandwidth_received_kbps, 1) +
                                    " KB/s  ↑ " + String::num(stats.bandwidth_sent_kbps, 1) + " KB/s");
    }

    void add_stat_row(TreeItem* parent, const String& name, const String& value) {
        TreeItem* item = m_stats_tree->create_item(parent);
        item->set_text(0, name);
        item->set_text(1, value);
    }

    void on_enable_toggled(bool pressed) {
        NetworkProfiler::get_singleton()->set_enabled(!pressed);
        m_enable_btn->set_text(pressed ? "Resume" : "Pause");
    }

    void on_clear() {
        NetworkProfiler::get_singleton()->reset();
        refresh_stats();
    }

    bool is_visible() const { return m_panel && m_panel->is_visible(); }
};
#endif // XTU_GODOT_EDITOR_ENABLED

} // namespace multiplayer

// Bring into main namespace
using multiplayer::NetworkProfiler;
using multiplayer::LatencySimulator;
using multiplayer::MultiplayerDebugger;
using multiplayer::NetworkPacketType;
using multiplayer::NetworkStats;
using multiplayer::PacketLogEntry;
#ifdef XTU_GODOT_EDITOR_ENABLED
using multiplayer::MultiplayerEditorPlugin;
#endif

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XNETWORK_PROFILER_HPP