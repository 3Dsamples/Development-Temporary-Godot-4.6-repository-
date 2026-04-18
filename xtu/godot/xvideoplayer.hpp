// include/xtu/godot/xvideoplayer.hpp
// xtensor-unified - Video playback for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XVIDEOPLAYER_HPP
#define XTU_GODOT_XVIDEOPLAYER_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xaudioserver.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class VideoStream;
class VideoStreamPlayback;
class VideoStreamPlayer;
class VideoStreamTheora;
class VideoStreamWebM;

// #############################################################################
// Video loop mode
// #############################################################################
enum class VideoLoopMode : uint8_t {
    LOOP_MODE_NONE = 0,
    LOOP_MODE_LOOP = 1,
    LOOP_MODE_PING_PONG = 2
};

// #############################################################################
// Video buffering state
// #############################################################################
enum class VideoBufferingState : uint8_t {
    BUFFERING_IDLE = 0,
    BUFFERING_LOADING = 1,
    BUFFERING_READY = 2,
    BUFFERING_STALLED = 3
};

// #############################################################################
// Video playback state
// #############################################################################
enum class VideoPlaybackState : uint8_t {
    STATE_STOPPED = 0,
    STATE_PLAYING = 1,
    STATE_PAUSED = 2,
    STATE_SEEKING = 3
};

// #############################################################################
// Video format types
// #############################################################################
enum class VideoFormat : uint8_t {
    FORMAT_UNKNOWN = 0,
    FORMAT_THEORA = 1,
    FORMAT_WEBM_VP8 = 2,
    FORMAT_WEBM_VP9 = 3,
    FORMAT_OGV = 4,
    FORMAT_MP4_H264 = 5,
    FORMAT_MP4_H265 = 6
};

// #############################################################################
// Video frame data
// #############################################################################
struct VideoFrame {
    std::vector<uint8_t> data;
    size_t width = 0;
    size_t height = 0;
    double timestamp = 0.0;
    graphics::texture_format format = graphics::texture_format::rgba8;
    bool key_frame = false;
};

// #############################################################################
// VideoStream - Base class for video resources
// #############################################################################
class VideoStream : public Resource {
    XTU_GODOT_REGISTER_CLASS(VideoStream, Resource)

protected:
    String m_file_path;
    VideoFormat m_format = VideoFormat::FORMAT_UNKNOWN;
    bool m_audio_enabled = true;
    float m_buffering_ms = 500.0f;

public:
    static StringName get_class_static() { return StringName("VideoStream"); }

    void set_file(const String& path) { m_file_path = path; }
    String get_file() const { return m_file_path; }

    void set_audio_enabled(bool enabled) { m_audio_enabled = enabled; }
    bool is_audio_enabled() const { return m_audio_enabled; }

    void set_buffering_ms(float ms) { m_buffering_ms = ms; }
    float get_buffering_ms() const { return m_buffering_ms; }

    VideoFormat get_format() const { return m_format; }

    virtual Ref<VideoStreamPlayback> instantiate_playback() = 0;
    virtual double get_length() const { return 0.0; }
    virtual vec2i get_size() const { return vec2i(0, 0); }
    virtual float get_framerate() const { return 0.0f; }
};

// #############################################################################
// VideoStreamPlayback - Playback instance
// #############################################################################
class VideoStreamPlayback : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(VideoStreamPlayback, RefCounted)

public:
    static StringName get_class_static() { return StringName("VideoStreamPlayback"); }

    virtual void start() = 0;
    virtual void stop() = 0;
    virtual bool is_playing() const = 0;
    virtual void pause() = 0;
    virtual void resume() = 0;
    virtual void seek(double time) = 0;
    virtual double get_position() const = 0;
    virtual double get_length() const = 0;
    virtual VideoBufferingState get_buffering_state() const = 0;
    virtual VideoPlaybackState get_playback_state() const = 0;
    virtual bool has_next_frame() = 0;
    virtual VideoFrame get_next_frame() = 0;
    virtual void update(double delta) = 0;
    virtual int get_mix_rate() const { return 44100; }
    virtual int get_channels() const { return 2; }
    virtual int mix_audio(AudioFrame* buffer, int frames) { return 0; }
};

// #############################################################################
// VideoStreamTheora - Theora video decoder
// #############################################################################
class VideoStreamTheora : public VideoStream {
    XTU_GODOT_REGISTER_CLASS(VideoStreamTheora, VideoStream)

public:
    static StringName get_class_static() { return StringName("VideoStreamTheora"); }

    Ref<VideoStreamPlayback> instantiate_playback() override;
    double get_length() const override;
    vec2i get_size() const override;
};

// #############################################################################
// VideoStreamWebM - WebM video decoder
// #############################################################################
class VideoStreamWebM : public VideoStream {
    XTU_GODOT_REGISTER_CLASS(VideoStreamWebM, VideoStream)

public:
    static StringName get_class_static() { return StringName("VideoStreamWebM"); }

    Ref<VideoStreamPlayback> instantiate_playback() override;
    double get_length() const override;
    vec2i get_size() const override;
};

// #############################################################################
// Generic VideoStreamPlayback implementation
// #############################################################################
class GenericVideoPlayback : public VideoStreamPlayback {
    XTU_GODOT_REGISTER_CLASS(GenericVideoPlayback, VideoStreamPlayback)

private:
    VideoStream* m_stream = nullptr;
    VideoPlaybackState m_state = VideoPlaybackState::STATE_STOPPED;
    VideoBufferingState m_buffering = VideoBufferingState::BUFFERING_IDLE;
    double m_position = 0.0;
    double m_length = 0.0;
    std::queue<VideoFrame> m_frame_queue;
    std::mutex m_mutex;
    std::thread m_decode_thread;
    std::atomic<bool> m_decode_running{false};
    float m_playback_speed = 1.0f;

public:
    static StringName get_class_static() { return StringName("GenericVideoPlayback"); }

    GenericVideoPlayback() = default;
    ~GenericVideoPlayback() { stop(); }

    void set_stream(VideoStream* stream) { m_stream = stream; }

    void start() override {
        if (m_state == VideoPlaybackState::STATE_PLAYING) return;
        m_state = VideoPlaybackState::STATE_PLAYING;
        start_decoder();
    }

    void stop() override {
        m_state = VideoPlaybackState::STATE_STOPPED;
        stop_decoder();
        std::lock_guard<std::mutex> lock(m_mutex);
        while (!m_frame_queue.empty()) m_frame_queue.pop();
    }

    bool is_playing() const override { return m_state == VideoPlaybackState::STATE_PLAYING; }

    void pause() override {
        if (m_state == VideoPlaybackState::STATE_PLAYING) {
            m_state = VideoPlaybackState::STATE_PAUSED;
        }
    }

    void resume() override {
        if (m_state == VideoPlaybackState::STATE_PAUSED) {
            m_state = VideoPlaybackState::STATE_PLAYING;
        }
    }

    void seek(double time) override {
        m_state = VideoPlaybackState::STATE_SEEKING;
        m_position = std::clamp(time, 0.0, m_length);
        // Flush queue and restart decoder at new position
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            while (!m_frame_queue.empty()) m_frame_queue.pop();
        }
        m_state = VideoPlaybackState::STATE_PLAYING;
    }

    double get_position() const override { return m_position; }
    double get_length() const override { return m_length; }

    VideoBufferingState get_buffering_state() const override { return m_buffering; }
    VideoPlaybackState get_playback_state() const override { return m_state; }

    bool has_next_frame() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        return !m_frame_queue.empty();
    }

    VideoFrame get_next_frame() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_frame_queue.empty()) return VideoFrame{};
        VideoFrame frame = std::move(m_frame_queue.front());
        m_frame_queue.pop();
        m_position = frame.timestamp;
        return frame;
    }

    void update(double delta) override {
        if (m_state == VideoPlaybackState::STATE_PLAYING) {
            m_position += delta * m_playback_speed;
        }
    }

    void set_playback_speed(float speed) { m_playback_speed = speed; }
    float get_playback_speed() const { return m_playback_speed; }

private:
    void start_decoder() {
        if (m_decode_running) return;
        m_decode_running = true;
        m_decode_thread = std::thread([this]() { decode_loop(); });
    }

    void stop_decoder() {
        m_decode_running = false;
        if (m_decode_thread.joinable()) m_decode_thread.join();
    }

    void decode_loop() {
        while (m_decode_running) {
            // Decode next frame
            VideoFrame frame = decode_next_frame();
            if (frame.data.empty()) {
                m_buffering = VideoBufferingState::BUFFERING_STALLED;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_frame_queue.size() < 30) { // Buffer up to 30 frames
                    m_frame_queue.push(std::move(frame));
                    m_buffering = VideoBufferingState::BUFFERING_READY;
                }
            }
        }
    }

    VideoFrame decode_next_frame() {
        // Placeholder - actual decoding would use libtheora/libvpx
        VideoFrame frame;
        return frame;
    }
};

// #############################################################################
// VideoStreamPlayer - Node for video playback
// #############################################################################
class VideoStreamPlayer : public Control {
    XTU_GODOT_REGISTER_CLASS(VideoStreamPlayer, Control)

private:
    Ref<VideoStream> m_stream;
    Ref<VideoStreamPlayback> m_playback;
    VideoLoopMode m_loop_mode = VideoLoopMode::LOOP_MODE_NONE;
    float m_volume_db = 0.0f;
    bool m_autoplay = false;
    bool m_paused = false;
    bool m_expand = true;
    float m_buffering_ms = 500.0f;
    int m_audio_bus_index = 0;
    Ref<Texture2D> m_current_frame_texture;
    std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("VideoStreamPlayer"); }

    void set_stream(const Ref<VideoStream>& stream) {
        if (m_stream == stream) return;
        stop();
        m_stream = stream;
        if (m_stream.is_valid()) {
            m_playback = m_stream->instantiate_playback();
        }
        update_configuration_warnings();
    }

    Ref<VideoStream> get_stream() const { return m_stream; }

    void set_loop_mode(VideoLoopMode mode) { m_loop_mode = mode; }
    VideoLoopMode get_loop_mode() const { return m_loop_mode; }

    void set_volume_db(float db) { m_volume_db = db; }
    float get_volume_db() const { return m_volume_db; }

    void set_autoplay(bool enabled) { m_autoplay = enabled; }
    bool is_autoplay_enabled() const { return m_autoplay; }

    void set_paused(bool paused) {
        if (m_paused == paused) return;
        m_paused = paused;
        if (m_playback.is_valid()) {
            if (paused) m_playback->pause();
            else m_playback->resume();
        }
    }

    bool is_paused() const { return m_paused; }

    void set_expand(bool expand) { m_expand = expand; update(); }
    bool has_expand() const { return m_expand; }

    void set_buffering_ms(float ms) { m_buffering_ms = ms; }
    float get_buffering_ms() const { return m_buffering_ms; }

    void set_audio_bus(const StringName& bus) {
        m_audio_bus_index = AudioServer::get_singleton()->get_bus_index(bus);
    }

    StringName get_audio_bus() const {
        return AudioServer::get_singleton()->get_bus_name(m_audio_bus_index);
    }

    void play() {
        if (!m_playback.is_valid()) return;
        m_playback->start();
        m_paused = false;
    }

    void stop() {
        if (!m_playback.is_valid()) return;
        m_playback->stop();
        m_paused = false;
    }

    bool is_playing() const {
        return m_playback.is_valid() && m_playback->is_playing();
    }

    void seek(double time) {
        if (m_playback.is_valid()) m_playback->seek(time);
    }

    double get_position() const {
        return m_playback.is_valid() ? m_playback->get_position() : 0.0;
    }

    double get_length() const {
        return m_stream.is_valid() ? m_stream->get_length() : 0.0;
    }

    VideoBufferingState get_buffering_state() const {
        return m_playback.is_valid() ? m_playback->get_buffering_state() : VideoBufferingState::BUFFERING_IDLE;
    }

    Ref<Texture2D> get_video_texture() const {
        return m_current_frame_texture;
    }

    void _ready() override {
        if (m_autoplay && m_stream.is_valid()) {
            play();
        }
    }

    void _process(double delta) override {
        if (!m_playback.is_valid() || !m_playback->is_playing()) return;

        m_playback->update(delta);

        if (m_playback->has_next_frame()) {
            VideoFrame frame = m_playback->get_next_frame();
            update_texture(frame);
        }

        double pos = m_playback->get_position();
        double len = m_playback->get_length();
        if (len > 0 && pos >= len - 0.05) {
            if (m_loop_mode == VideoLoopMode::LOOP_MODE_LOOP) {
                m_playback->seek(0);
            } else if (m_loop_mode == VideoLoopMode::LOOP_MODE_PING_PONG) {
                // Ping-pong logic
            } else {
                stop();
                emit_signal("finished");
            }
        }
    }

    void _draw() override {
        if (m_current_frame_texture.is_valid()) {
            Rect2 rect = Rect2(vec2f(0, 0), get_size());
            if (m_expand) {
                vec2f tex_size = m_current_frame_texture->get_size();
                float scale = std::min(get_size().x() / tex_size.x(), get_size().y() / tex_size.y());
                vec2f draw_size = tex_size * scale;
                rect.position = (get_size() - draw_size) * 0.5f;
                rect.size = draw_size;
            }
            draw_texture_rect(m_current_frame_texture, rect, false);
        }
    }

private:
    void update_texture(const VideoFrame& frame) {
        if (frame.data.empty()) return;
        if (!m_current_frame_texture.is_valid() ||
            m_current_frame_texture->get_width() != frame.width ||
            m_current_frame_texture->get_height() != frame.height) {
            m_current_frame_texture.instance();
            m_current_frame_texture->create_from_data(frame.width, frame.height, false,
                Image::FORMAT_RGBA8, frame.data);
        } else {
            m_current_frame_texture->update(frame.data);
        }
        update();
    }
};

// #############################################################################
// VideoStreamPlayer implementation for Theora
// #############################################################################
class TheoraVideoPlayback : public GenericVideoPlayback {
    XTU_GODOT_REGISTER_CLASS(TheoraVideoPlayback, GenericVideoPlayback)

public:
    static StringName get_class_static() { return StringName("TheoraVideoPlayback"); }
};

Ref<VideoStreamPlayback> VideoStreamTheora::instantiate_playback() {
    Ref<TheoraVideoPlayback> pb;
    pb.instance();
    pb->set_stream(this);
    return pb;
}

// #############################################################################
// VideoStreamPlayer implementation for WebM
// #############################################################################
class WebMVideoPlayback : public GenericVideoPlayback {
    XTU_GODOT_REGISTER_CLASS(WebMVideoPlayback, GenericVideoPlayback)

public:
    static StringName get_class_static() { return StringName("WebMVideoPlayback"); }
};

Ref<VideoStreamPlayback> VideoStreamWebM::instantiate_playback() {
    Ref<WebMVideoPlayback> pb;
    pb.instance();
    pb->set_stream(this);
    return pb;
}

} // namespace godot

// Bring into main namespace
using godot::VideoStream;
using godot::VideoStreamPlayback;
using godot::VideoStreamPlayer;
using godot::VideoStreamTheora;
using godot::VideoStreamWebM;
using godot::VideoLoopMode;
using godot::VideoBufferingState;
using godot::VideoPlaybackState;
using godot::VideoFormat;
using godot::VideoFrame;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XVIDEOPLAYER_HPP