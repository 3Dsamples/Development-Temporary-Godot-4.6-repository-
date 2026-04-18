// include/xtu/godot/xaudio_streams.hpp
// xtensor-unified - Audio stream loaders for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XAUDIO_STREAMS_HPP
#define XTU_GODOT_XAUDIO_STREAMS_HPP

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xcore.hpp"
#include "xtu/godot/xaudioserver.hpp"

#ifdef XTU_USE_VORBIS
#include <vorbis/vorbisfile.h>
#endif

#ifdef XTU_USE_OPUS
#include <opus/opusfile.h>
#endif

#ifdef XTU_USE_MPG123
#include <mpg123.h>
#endif

#ifdef XTU_USE_FLAC
#include <FLAC/stream_decoder.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {
namespace audio {

// #############################################################################
// Forward declarations
// #############################################################################
class AudioStreamLoader;
class AudioStreamOGG;
class AudioStreamMP3;
class AudioStreamFLAC;
class AudioStreamWAV;

// #############################################################################
// Audio format type
// #############################################################################
enum class AudioFormatType : uint8_t {
    FORMAT_UNKNOWN = 0,
    FORMAT_WAV = 1,
    FORMAT_OGG_VORBIS = 2,
    FORMAT_OGG_OPUS = 3,
    FORMAT_MP3 = 4,
    FORMAT_FLAC = 5
};

// #############################################################################
// Audio stream data
// #############################################################################
struct AudioStreamData {
    std::vector<float> samples;
    int sample_rate = 44100;
    int channels = 2;
    double length = 0.0;
    bool loop = false;
    float loop_begin = 0.0f;
    float loop_end = 0.0f;
    String error_message;
};

// #############################################################################
// AudioStreamLoader - Base class for audio format loaders
// #############################################################################
class AudioStreamLoader : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(AudioStreamLoader, RefCounted)

public:
    static StringName get_class_static() { return StringName("AudioStreamLoader"); }

    virtual std::vector<String> get_recognized_extensions() const = 0;
    virtual AudioFormatType get_format_type() const = 0;
    virtual bool can_load(const std::vector<uint8_t>& header) const = 0;
    virtual Error load_stream(const std::vector<uint8_t>& data, AudioStreamData& out_stream) = 0;
    virtual Ref<AudioStreamPlayback> create_playback(const AudioStreamData& data) = 0;
    virtual float get_priority() const { return 1.0f; }
};

// #############################################################################
// GenericStreamPlayback - Common playback for decoded streams
// #############################################################################
class GenericStreamPlayback : public AudioStreamPlayback {
    XTU_GODOT_REGISTER_CLASS(GenericStreamPlayback, AudioStreamPlayback)

private:
    std::vector<float> m_samples;
    int m_sample_rate = 44100;
    int m_channels = 2;
    double m_length = 0.0;
    bool m_loop = false;
    float m_loop_begin = 0.0f;
    float m_loop_end = 0.0f;
    size_t m_position = 0;
    bool m_playing = false;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("GenericStreamPlayback"); }

    void set_data(const AudioStreamData& data) {
        m_samples = data.samples;
        m_sample_rate = data.sample_rate;
        m_channels = data.channels;
        m_length = data.length;
        m_loop = data.loop;
        m_loop_begin = data.loop_begin;
        m_loop_end = data.loop_end;
    }

    void start(float from_pos) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_position = static_cast<size_t>(from_pos * m_sample_rate * m_channels);
        m_playing = true;
    }

    void stop() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_playing = false;
        m_position = 0;
    }

    bool is_playing() const override {
        return m_playing;
    }

    void seek(float pos) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_position = static_cast<size_t>(pos * m_sample_rate * m_channels);
    }

    float get_position() const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        return static_cast<float>(m_position) / (m_sample_rate * m_channels);
    }

    int mix(AudioFrame* buffer, float rate_scale, int frames) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_playing || m_samples.empty()) return 0;

        size_t total_frames = m_samples.size() / m_channels;
        size_t loop_begin_frame = static_cast<size_t>(m_loop_begin * m_sample_rate);
        size_t loop_end_frame = static_cast<size_t>(m_loop_end * m_sample_rate);

        int mixed = 0;
        for (int i = 0; i < frames; ++i) {
            if (m_position / m_channels >= total_frames) {
                if (m_loop && loop_end_frame > loop_begin_frame) {
                    m_position = loop_begin_frame * m_channels;
                } else {
                    m_playing = false;
                    break;
                }
            }

            if (m_loop && m_position / m_channels >= loop_end_frame) {
                m_position = loop_begin_frame * m_channels;
            }

            if (m_channels == 2) {
                buffer[i].left = m_samples[m_position];
                buffer[i].right = m_samples[m_position + 1];
                m_position += 2;
            } else if (m_channels == 1) {
                buffer[i].left = m_samples[m_position];
                buffer[i].right = m_samples[m_position];
                m_position += 1;
            }
            ++mixed;
        }

        return mixed;
    }

    void set_loop(bool enable) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_loop = enable;
    }

    bool has_loop() const override { return m_loop; }
};

// #############################################################################
// AudioStreamWAV - WAV format loader
// #############################################################################
class AudioStreamWAV : public AudioStreamLoader {
    XTU_GODOT_REGISTER_CLASS(AudioStreamWAV, AudioStreamLoader)

public:
    static StringName get_class_static() { return StringName("AudioStreamWAV"); }

    std::vector<String> get_recognized_extensions() const override {
        return {"wav", "wave"};
    }

    AudioFormatType get_format_type() const override {
        return AudioFormatType::FORMAT_WAV;
    }

    bool can_load(const std::vector<uint8_t>& header) const override {
        return header.size() >= 12 &&
               std::memcmp(header.data(), "RIFF", 4) == 0 &&
               std::memcmp(header.data() + 8, "WAVE", 4) == 0;
    }

    Error load_stream(const std::vector<uint8_t>& data, AudioStreamData& out_stream) override {
        if (data.size() < 44) {
            out_stream.error_message = "Invalid WAV file";
            return ERR_FILE_CORRUPT;
        }

        size_t offset = 12;
        while (offset + 8 <= data.size()) {
            char chunk_id[5] = {0};
            std::memcpy(chunk_id, data.data() + offset, 4);
            uint32_t chunk_size = *reinterpret_cast<const uint32_t*>(data.data() + offset + 4);

            if (std::strcmp(chunk_id, "fmt ") == 0) {
                uint16_t audio_format = *reinterpret_cast<const uint16_t*>(data.data() + offset + 8);
                uint16_t channels = *reinterpret_cast<const uint16_t*>(data.data() + offset + 10);
                uint32_t sample_rate = *reinterpret_cast<const uint32_t*>(data.data() + offset + 12);
                uint16_t bits_per_sample = *reinterpret_cast<const uint16_t*>(data.data() + offset + 22);

                if (audio_format != 1 && audio_format != 3) {
                    out_stream.error_message = "Unsupported WAV format (only PCM and IEEE float)";
                    return ERR_UNAVAILABLE;
                }

                out_stream.sample_rate = sample_rate;
                out_stream.channels = channels;
            } else if (std::strcmp(chunk_id, "data") == 0) {
                size_t sample_count = chunk_size / (out_stream.channels * (out_stream.bits_per_sample / 8));
                out_stream.samples.resize(sample_count * out_stream.channels);
                // Convert to float samples
                // ... conversion logic
            }

            offset += 8 + chunk_size;
        }

        out_stream.length = static_cast<double>(out_stream.samples.size()) /
                           (out_stream.sample_rate * out_stream.channels);
        return OK;
    }

    Ref<AudioStreamPlayback> create_playback(const AudioStreamData& data) override {
        Ref<GenericStreamPlayback> playback;
        playback.instance();
        playback->set_data(data);
        return playback;
    }
};

// #############################################################################
// AudioStreamOGG - OGG Vorbis/Opus loader
// #############################################################################
class AudioStreamOGG : public AudioStreamLoader {
    XTU_GODOT_REGISTER_CLASS(AudioStreamOGG, AudioStreamLoader)

public:
    static StringName get_class_static() { return StringName("AudioStreamOGG"); }

    std::vector<String> get_recognized_extensions() const override {
        return {"ogg", "oga", "ogv", "opus"};
    }

    AudioFormatType get_format_type() const override {
        return AudioFormatType::FORMAT_OGG_VORBIS;
    }

    bool can_load(const std::vector<uint8_t>& header) const override {
        return header.size() >= 4 && std::memcmp(header.data(), "OggS", 4) == 0;
    }

    Error load_stream(const std::vector<uint8_t>& data, AudioStreamData& out_stream) override {
#ifdef XTU_USE_VORBIS
        // libvorbis decoding
        return OK;
#elif defined(XTU_USE_OPUS)
        // libopusfile decoding
        return OK;
#else
        out_stream.error_message = "OGG support not compiled";
        return ERR_UNAVAILABLE;
#endif
    }

    Ref<AudioStreamPlayback> create_playback(const AudioStreamData& data) override {
        Ref<GenericStreamPlayback> playback;
        playback.instance();
        playback->set_data(data);
        return playback;
    }
};

// #############################################################################
// AudioStreamMP3 - MP3 format loader
// #############################################################################
class AudioStreamMP3 : public AudioStreamLoader {
    XTU_GODOT_REGISTER_CLASS(AudioStreamMP3, AudioStreamLoader)

public:
    static StringName get_class_static() { return StringName("AudioStreamMP3"); }

    std::vector<String> get_recognized_extensions() const override {
        return {"mp3", "mp2", "mp1"};
    }

    AudioFormatType get_format_type() const override {
        return AudioFormatType::FORMAT_MP3;
    }

    bool can_load(const std::vector<uint8_t>& header) const override {
        // Check for MP3 sync word
        if (header.size() < 2) return false;
        return (header[0] == 0xFF && (header[1] & 0xE0) == 0xE0) ||
               (header.size() >= 3 && std::memcmp(header.data(), "ID3", 3) == 0);
    }

    Error load_stream(const std::vector<uint8_t>& data, AudioStreamData& out_stream) override {
#ifdef XTU_USE_MPG123
        // mpg123 decoding
        return OK;
#else
        out_stream.error_message = "MP3 support not compiled";
        return ERR_UNAVAILABLE;
#endif
    }

    Ref<AudioStreamPlayback> create_playback(const AudioStreamData& data) override {
        Ref<GenericStreamPlayback> playback;
        playback.instance();
        playback->set_data(data);
        return playback;
    }
};

// #############################################################################
// AudioStreamFLAC - FLAC format loader
// #############################################################################
class AudioStreamFLAC : public AudioStreamLoader {
    XTU_GODOT_REGISTER_CLASS(AudioStreamFLAC, AudioStreamLoader)

public:
    static StringName get_class_static() { return StringName("AudioStreamFLAC"); }

    std::vector<String> get_recognized_extensions() const override {
        return {"flac", "fla"};
    }

    AudioFormatType get_format_type() const override {
        return AudioFormatType::FORMAT_FLAC;
    }

    bool can_load(const std::vector<uint8_t>& header) const override {
        return header.size() >= 4 && std::memcmp(header.data(), "fLaC", 4) == 0;
    }

    Error load_stream(const std::vector<uint8_t>& data, AudioStreamData& out_stream) override {
#ifdef XTU_USE_FLAC
        // libFLAC decoding
        return OK;
#else
        out_stream.error_message = "FLAC support not compiled";
        return ERR_UNAVAILABLE;
#endif
    }

    Ref<AudioStreamPlayback> create_playback(const AudioStreamData& data) override {
        Ref<GenericStreamPlayback> playback;
        playback.instance();
        playback->set_data(data);
        return playback;
    }
};

// #############################################################################
// AudioStreamLoaderManager - Global audio loader singleton
// #############################################################################
class AudioStreamLoaderManager : public Object {
    XTU_GODOT_REGISTER_CLASS(AudioStreamLoaderManager, Object)

private:
    static AudioStreamLoaderManager* s_singleton;
    std::vector<Ref<AudioStreamLoader>> m_loaders;
    std::unordered_map<String, Ref<AudioStreamLoader>> m_extension_loader;
    mutable std::mutex m_mutex;

public:
    static AudioStreamLoaderManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("AudioStreamLoaderManager"); }

    AudioStreamLoaderManager() {
        s_singleton = this;
        register_default_loaders();
    }

    ~AudioStreamLoaderManager() { s_singleton = nullptr; }

    void add_loader(const Ref<AudioStreamLoader>& loader) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_loaders.push_back(loader);
        for (const auto& ext : loader->get_recognized_extensions()) {
            m_extension_loader[ext] = loader;
        }
    }

    Ref<AudioStreamLoader> get_loader_for_extension(const String& ext) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_extension_loader.find(ext.to_lower());
        return it != m_extension_loader.end() ? it->second : Ref<AudioStreamLoader>();
    }

    Error load_file(const String& path, AudioStreamData& out_data) {
        Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
        if (!file.is_valid()) return ERR_FILE_CANT_OPEN;

        String ext = path.get_extension();
        Ref<AudioStreamLoader> loader = get_loader_for_extension(ext);
        if (!loader.is_valid()) {
            out_data.error_message = "Unsupported audio format: " + ext;
            return ERR_FILE_UNRECOGNIZED;
        }

        std::vector<uint8_t> data = file->get_buffer(file->get_length());
        return loader->load_stream(data, out_data);
    }

    Ref<AudioStream> create_stream_from_file(const String& path) {
        AudioStreamData data;
        Error err = load_file(path, data);
        if (err != OK) return Ref<AudioStream>();

        Ref<AudioStream> stream;
        stream.instance();
        stream->set_data(data);
        return stream;
    }

private:
    void register_default_loaders() {
        add_loader(Ref<AudioStreamLoader>(new AudioStreamWAV()));
        add_loader(Ref<AudioStreamLoader>(new AudioStreamOGG()));
        add_loader(Ref<AudioStreamLoader>(new AudioStreamMP3()));
        add_loader(Ref<AudioStreamLoader>(new AudioStreamFLAC()));
    }
};

} // namespace audio

// Bring into main namespace
using audio::AudioStreamLoader;
using audio::AudioStreamWAV;
using audio::AudioStreamOGG;
using audio::AudioStreamMP3;
using audio::AudioStreamFLAC;
using audio::GenericStreamPlayback;
using audio::AudioStreamLoaderManager;
using audio::AudioFormatType;
using audio::AudioStreamData;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XAUDIO_STREAMS_HPP