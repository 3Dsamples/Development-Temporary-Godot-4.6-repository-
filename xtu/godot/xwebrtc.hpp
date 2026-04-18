// include/xtu/godot/xwebrtc.hpp
// xtensor-unified - WebRTC integration for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XWEBRTC_HPP
#define XTU_GODOT_XWEBRTC_HPP

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnetworking.hpp"

#ifdef XTU_USE_WEBRTC
#include <api/peer_connection_interface.h>
#include <api/data_channel_interface.h>
#include <rtc_base/thread.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class WebRTCMultiplayerPeer;
class WebRTCDataChannel;
class WebRTCPeerConnection;
class WebRTCLibrary;

// #############################################################################
// WebRTC data channel state
// #############################################################################
enum class WebRTCDataChannelState : uint8_t {
    STATE_CONNECTING = 0,
    STATE_OPEN = 1,
    STATE_CLOSING = 2,
    STATE_CLOSED = 3
};

// #############################################################################
// WebRTC peer connection state
// #############################################################################
enum class WebRTCPeerConnectionState : uint8_t {
    STATE_NEW = 0,
    STATE_CONNECTING = 1,
    STATE_CONNECTED = 2,
    STATE_DISCONNECTED = 3,
    STATE_FAILED = 4,
    STATE_CLOSED = 5
};

// #############################################################################
// WebRTC ICE gathering state
// #############################################################################
enum class WebRTCICEGatheringState : uint8_t {
    GATHERING_NEW = 0,
    GATHERING_GATHERING = 1,
    GATHERING_COMPLETE = 2
};

// #############################################################################
// WebRTC SDP type
// #############################################################################
enum class WebRTCSDPType : uint8_t {
    SDP_OFFER = 0,
    SDP_PRANSWER = 1,
    SDP_ANSWER = 2,
    SDP_ROLLBACK = 3
};

// #############################################################################
// WebRTCDataChannel - Data channel for peer-to-peer communication
// #############################################################################
class WebRTCDataChannel : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(WebRTCDataChannel, RefCounted)

private:
    String m_label;
    bool m_ordered = true;
    int m_max_packet_life_time = -1;
    int m_max_retransmits = -1;
    String m_protocol;
    bool m_negotiated = false;
    int m_id = -1;
    WebRTCDataChannelState m_state = WebRTCDataChannelState::STATE_CLOSED;
    std::queue<std::vector<uint8_t>> m_incoming_packets;
    mutable std::mutex m_mutex;
#ifdef XTU_USE_WEBRTC
    rtc::scoped_refptr<webrtc::DataChannelInterface> m_channel;
#endif

public:
    static StringName get_class_static() { return StringName("WebRTCDataChannel"); }

    void set_label(const String& label) { m_label = label; }
    String get_label() const { return m_label; }

    void set_ordered(bool ordered) { m_ordered = ordered; }
    bool is_ordered() const { return m_ordered; }

    WebRTCDataChannelState get_state() const { return m_state; }

    void send(const std::vector<uint8_t>& data) {
#ifdef XTU_USE_WEBRTC
        if (m_channel && m_state == WebRTCDataChannelState::STATE_OPEN) {
            webrtc::DataBuffer buffer(rtc::CopyOnWriteBuffer(data.data(), data.size()), true);
            m_channel->Send(buffer);
        }
#endif
    }

    void send_text(const String& text) {
        std::string str = text.to_std_string();
        std::vector<uint8_t> data(str.begin(), str.end());
        send(data);
    }

    int get_available_packet_count() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return static_cast<int>(m_incoming_packets.size());
    }

    std::vector<uint8_t> receive() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_incoming_packets.empty()) return {};
        auto data = std::move(m_incoming_packets.front());
        m_incoming_packets.pop();
        return data;
    }

    void close() {
#ifdef XTU_USE_WEBRTC
        if (m_channel) {
            m_channel->Close();
            m_state = WebRTCDataChannelState::STATE_CLOSED;
        }
#endif
    }

    void on_message(const std::vector<uint8_t>& data) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_incoming_packets.push(data);
        call_deferred("emit_signal", "data_received", data);
    }

    void on_state_change(WebRTCDataChannelState state) {
        m_state = state;
        call_deferred("emit_signal", "state_changed", static_cast<int>(state));
    }

#ifdef XTU_USE_WEBRTC
    void set_internal_channel(rtc::scoped_refptr<webrtc::DataChannelInterface> channel) {
        m_channel = channel;
        m_label = String(channel->label().c_str());
    }
#endif
};

// #############################################################################
// WebRTCPeerConnection - Peer connection management
// #############################################################################
class WebRTCPeerConnection : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(WebRTCPeerConnection, RefCounted)

private:
    std::map<String, String> m_configuration;
    WebRTCPeerConnectionState m_connection_state = WebRTCPeerConnectionState::STATE_NEW;
    WebRTCICEGatheringState m_ice_gathering_state = WebRTCICEGatheringState::GATHERING_NEW;
    std::vector<Ref<WebRTCDataChannel>> m_data_channels;
    mutable std::mutex m_mutex;
#ifdef XTU_USE_WEBRTC
    rtc::scoped_refptr<webrtc::PeerConnectionInterface> m_peer_connection;
    std::unique_ptr<rtc::Thread> m_worker_thread;
    std::unique_ptr<rtc::Thread> m_signaling_thread;
#endif

public:
    static StringName get_class_static() { return StringName("WebRTCPeerConnection"); }

    void set_configuration(const std::map<String, String>& config) { m_configuration = config; }

    bool initialize() {
#ifdef XTU_USE_WEBRTC
        m_worker_thread = rtc::Thread::Create();
        m_worker_thread->Start();
        m_signaling_thread = rtc::Thread::Create();
        m_signaling_thread->Start();
        
        webrtc::PeerConnectionInterface::RTCConfiguration rtc_config;
        // Parse configuration
        for (const auto& kv : m_configuration) {
            if (kv.first == "ice_servers") {
                // Parse ICE servers
            }
        }
        
        webrtc::PeerConnectionInterface::IceServer ice_server;
        ice_server.urls.push_back("stun:stun.l.google.com:19302");
        rtc_config.servers.push_back(ice_server);
        
        m_peer_connection = WebRTCLibrary::get_singleton()->create_peer_connection(rtc_config, this);
        return m_peer_connection != nullptr;
#else
        return false;
#endif
    }

    Ref<WebRTCDataChannel> create_data_channel(const String& label, const std::map<String, Variant>& options = {}) {
        Ref<WebRTCDataChannel> channel;
        channel.instance();
        channel->set_label(label);
        
#ifdef XTU_USE_WEBRTC
        webrtc::DataChannelInit config;
        config.ordered = options.count("ordered") ? options.at("ordered").as<bool>() : true;
        config.maxRetransmitTime = options.count("max_packet_life_time") ? options.at("max_packet_life_time").as<int>() : -1;
        config.maxRetransmits = options.count("max_retransmits") ? options.at("max_retransmits").as<int>() : -1;
        config.protocol = options.count("protocol") ? options.at("protocol").as<String>().to_std_string() : "";
        config.negotiated = options.count("negotiated") ? options.at("negotiated").as<bool>() : false;
        config.id = options.count("id") ? options.at("id").as<int>() : -1;
        
        auto rtc_channel = m_peer_connection->CreateDataChannel(label.to_std_string(), &config);
        channel->set_internal_channel(rtc_channel);
#endif
        
        std::lock_guard<std::mutex> lock(m_mutex);
        m_data_channels.push_back(channel);
        return channel;
    }

    void create_offer() {
#ifdef XTU_USE_WEBRTC
        webrtc::PeerConnectionInterface::RTCOfferAnswerOptions options;
        m_peer_connection->CreateOffer(this, options);
#endif
    }

    void create_answer() {
#ifdef XTU_USE_WEBRTC
        webrtc::PeerConnectionInterface::RTCOfferAnswerOptions options;
        m_peer_connection->CreateAnswer(this, options);
#endif
    }

    void set_local_description(WebRTCSDPType type, const String& sdp) {
#ifdef XTU_USE_WEBRTC
        webrtc::SessionDescriptionInterface* desc = webrtc::CreateSessionDescription(
            convert_sdp_type(type), sdp.to_std_string());
        m_peer_connection->SetLocalDescription(this, desc);
#endif
    }

    void set_remote_description(WebRTCSDPType type, const String& sdp) {
#ifdef XTU_USE_WEBRTC
        webrtc::SessionDescriptionInterface* desc = webrtc::CreateSessionDescription(
            convert_sdp_type(type), sdp.to_std_string());
        m_peer_connection->SetRemoteDescription(this, desc);
#endif
    }

    void add_ice_candidate(const String& sdp_mid, int sdp_mline_index, const String& candidate) {
#ifdef XTU_USE_WEBRTC
        webrtc::SdpParseError error;
        std::unique_ptr<webrtc::IceCandidateInterface> ice_candidate(
            webrtc::CreateIceCandidate(sdp_mid.to_std_string(), sdp_mline_index, candidate.to_std_string(), &error));
        if (ice_candidate) {
            m_peer_connection->AddIceCandidate(ice_candidate.get());
        }
#endif
    }

    void close() {
#ifdef XTU_USE_WEBRTC
        if (m_peer_connection) {
            m_peer_connection->Close();
            m_peer_connection = nullptr;
        }
        m_worker_thread->Stop();
        m_signaling_thread->Stop();
#endif
    }

    WebRTCPeerConnectionState get_connection_state() const { return m_connection_state; }
    WebRTCICEGatheringState get_ice_gathering_state() const { return m_ice_gathering_state; }

#ifdef XTU_USE_WEBRTC
    static webrtc::SdpType convert_sdp_type(WebRTCSDPType type) {
        switch (type) {
            case WebRTCSDPType::SDP_OFFER: return webrtc::SdpType::kOffer;
            case WebRTCSDPType::SDP_PRANSWER: return webrtc::SdpType::kPrAnswer;
            case WebRTCSDPType::SDP_ANSWER: return webrtc::SdpType::kAnswer;
            case WebRTCSDPType::SDP_ROLLBACK: return webrtc::SdpType::kRollback;
            default: return webrtc::SdpType::kOffer;
        }
    }

    void on_success(webrtc::SessionDescriptionInterface* desc) {
        std::string sdp;
        desc->ToString(&sdp);
        call_deferred("emit_signal", "session_description_created", 
                      static_cast<int>(desc->GetType() == webrtc::SdpType::kOffer ? WebRTCSDPType::SDP_OFFER : WebRTCSDPType::SDP_ANSWER),
                      String(sdp.c_str()));
    }

    void on_ice_candidate(const webrtc::IceCandidateInterface* candidate) {
        std::string sdp;
        candidate->ToString(&sdp);
        call_deferred("emit_signal", "ice_candidate_created",
                      String(candidate->sdp_mid().c_str()),
                      candidate->sdp_mline_index(),
                      String(sdp.c_str()));
    }
#endif
};

// #############################################################################
// WebRTCLibrary - Global WebRTC initialization
// #############################################################################
class WebRTCLibrary : public Object {
    XTU_GODOT_REGISTER_CLASS(WebRTCLibrary, Object)

private:
    static WebRTCLibrary* s_singleton;
    bool m_initialized = false;
    std::mutex m_mutex;
#ifdef XTU_USE_WEBRTC
    std::unique_ptr<rtc::Thread> m_network_thread;
    std::unique_ptr<rtc::Thread> m_worker_thread;
    rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface> m_factory;
#endif

public:
    static WebRTCLibrary* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("WebRTCLibrary"); }

    WebRTCLibrary() { s_singleton = this; }
    ~WebRTCLibrary() { cleanup(); s_singleton = nullptr; }

    bool initialize() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized) return true;
        
#ifdef XTU_USE_WEBRTC
        m_network_thread = rtc::Thread::CreateWithSocketServer();
        m_network_thread->Start();
        m_worker_thread = rtc::Thread::Create();
        m_worker_thread->Start();
        
        webrtc::PeerConnectionFactoryDependencies deps;
        deps.network_thread = m_network_thread.get();
        deps.worker_thread = m_worker_thread.get();
        deps.signaling_thread = rtc::Thread::Current();
        
        m_factory = webrtc::CreateModularPeerConnectionFactory(std::move(deps));
        m_initialized = m_factory != nullptr;
        return m_initialized;
#else
        return false;
#endif
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(m_mutex);
#ifdef XTU_USE_WEBRTC
        m_factory = nullptr;
        if (m_network_thread) m_network_thread->Stop();
        if (m_worker_thread) m_worker_thread->Stop();
#endif
        m_initialized = false;
    }

    bool is_initialized() const { return m_initialized; }

#ifdef XTU_USE_WEBRTC
    rtc::scoped_refptr<webrtc::PeerConnectionInterface> create_peer_connection(
        const webrtc::PeerConnectionInterface::RTCConfiguration& config,
        WebRTCPeerConnection* observer) {
        if (!m_factory) return nullptr;
        return m_factory->CreatePeerConnection(config, nullptr, nullptr, observer);
    }

    webrtc::PeerConnectionFactoryInterface* get_factory() { return m_factory.get(); }
#endif
};

// #############################################################################
// WebRTCMultiplayerPeer - WebRTC multiplayer implementation
// #############################################################################
class WebRTCMultiplayerPeer : public MultiplayerPeer {
    XTU_GODOT_REGISTER_CLASS(WebRTCMultiplayerPeer, MultiplayerPeer)

private:
    std::unordered_map<int32_t, Ref<WebRTCPeerConnection>> m_connections;
    std::unordered_map<int32_t, Ref<WebRTCDataChannel>> m_data_channels;
    Ref<WebRTCPeerConnection> m_server_connection;
    std::vector<String> m_stun_servers;
    std::vector<String> m_turn_servers;
    bool m_compatible_mode = false;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("WebRTCMultiplayerPeer"); }

    void add_ice_server(const String& server) {
        if (server.begins_with("stun:")) {
            m_stun_servers.push_back(server);
        } else if (server.begins_with("turn:")) {
            m_turn_servers.push_back(server);
        }
    }

    void clear_ice_servers() {
        m_stun_servers.clear();
        m_turn_servers.clear();
    }

    void set_compatible_mode(bool enabled) { m_compatible_mode = enabled; }

    Error create_server(int32_t port, int32_t max_clients, const String& bind_address) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_unique_id = 1;
        m_connection_status = MultiplayerPeerConnectionStatus::CONNECTION_CONNECTED;
        return OK;
    }

    Error create_client(const String& address, int32_t port) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Create peer connection to signaling server
        return OK;
    }

    void create_data_channel(int32_t peer_id, const String& label) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_connections.find(peer_id);
        if (it != m_connections.end()) {
            auto channel = it->second->create_data_channel(label);
            m_data_channels[peer_id] = channel;
        }
    }

    void add_peer_connection(int32_t peer_id, const Ref<WebRTCPeerConnection>& connection) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_connections[peer_id] = connection;
    }

    void remove_peer_connection(int32_t peer_id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_connections.erase(peer_id);
        m_data_channels.erase(peer_id);
    }

    void poll() override {
        // Process incoming messages from data channels
    }

    Error put_packet(const uint8_t* buffer, int32_t size) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_target_peer == 0) {
            // Broadcast to all peers
            for (auto& kv : m_data_channels) {
                if (kv.second.is_valid() && kv.second->get_state() == WebRTCDataChannelState::STATE_OPEN) {
                    kv.second->send(std::vector<uint8_t>(buffer, buffer + size));
                }
            }
        } else {
            auto it = m_data_channels.find(m_target_peer);
            if (it != m_data_channels.end() && it->second.is_valid()) {
                it->second->send(std::vector<uint8_t>(buffer, buffer + size));
            }
        }
        return OK;
    }

    void close() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto& kv : m_connections) {
            kv.second->close();
        }
        m_connections.clear();
        m_data_channels.clear();
        m_connection_status = MultiplayerPeerConnectionStatus::CONNECTION_DISCONNECTED;
    }
};

} // namespace godot

// Bring into main namespace
using godot::WebRTCMultiplayerPeer;
using godot::WebRTCDataChannel;
using godot::WebRTCPeerConnection;
using godot::WebRTCLibrary;
using godot::WebRTCDataChannelState;
using godot::WebRTCPeerConnectionState;
using godot::WebRTCICEGatheringState;
using godot::WebRTCSDPType;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XWEBRTC_HPP