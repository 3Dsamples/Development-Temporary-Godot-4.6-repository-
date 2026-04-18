// godot/xaudioserver.hpp

#ifndef XTENSOR_XAUDIOSERVER_HPP
#define XTENSOR_XAUDIOSERVER_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"
#include "../math/xtransform.hpp"
#include "../signal/lfilter.hpp"
#include "../signal/xwindows.hpp"
#include "../io/xaudio.hpp"
#include "xvariant.hpp"
#include "xclassdb.hpp"
#include "xnode.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <complex>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/audio_server.hpp>
    #include <godot_cpp/classes/audio_stream.hpp>
    #include <godot_cpp/classes/audio_stream_player.hpp>
    #include <godot_cpp/classes/audio_stream_player3d.hpp>
    #include <godot_cpp/classes/audio_effect.hpp>
    #include <godot_cpp/classes/audio_bus_layout.hpp>
    #include <godot_cpp/classes/audio_stream_generator.hpp>
    #include <godot_cpp/classes/audio_stream_microphone.hpp>
    #include <godot_cpp/classes/audio_effect_chorus.hpp>
    #include <godot_cpp/classes/audio_effect_compressor.hpp>
    #include <godot_cpp/classes/audio_effect_delay.hpp>
    #include <godot_cpp/classes/audio_effect_distortion.hpp>
    #include <godot_cpp/classes/audio_effect_eq.hpp>
    #include <godot_cpp/classes/audio_effect_filter.hpp>
    #include <godot_cpp/classes/audio_effect_limiter.hpp>
    #include <godot_cpp/classes/audio_effect_panner.hpp>
    #include <godot_cpp/classes/audio_effect_phaser.hpp>
    #include <godot_cpp/classes/audio_effect_pitch_shift.hpp>
    #include <godot_cpp/classes/audio_effect_reverb.hpp>
    #include <godot_cpp/classes/audio_effect_spectrum_analyzer.hpp>
    #include <godot_cpp/classes/audio_effect_stereo_enhance.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/vector3.hpp>
    #include <godot_cpp/variant/vector2.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // Audio buffer and format utilities
            // --------------------------------------------------------------------
            using AudioBuffer = xarray_container<float>; // N x C (samples x channels)

            struct AudioDeviceInfo
            {
                std::string name;
                int sample_rate = 48000;
                int channels = 2;
                int buffer_size = 1024;
                bool is_input = false;
                bool is_default = false;
            };

            // --------------------------------------------------------------------
            // Tensor-based Audio Bus Management
            // --------------------------------------------------------------------
            class XAudioBus
            {
            public:
                std::string name;
                float volume_db = 0.0f;
                bool mute = false;
                bool solo = false;
                bool bypass = false;
                AudioBuffer eq_gains;      // N x 1 (gain per band)
                AudioBuffer eq_frequencies; // N x 1 (center frequencies)
                std::vector<godot::Ref<godot::AudioEffect>> effects;
                std::vector<AudioBuffer> effect_buffers;
                int output_bus = 0; // Master by default
            };

            // --------------------------------------------------------------------
            // XAudioServer - Tensor-based audio server extension
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XAudioServer : public godot::Object
            {
                GDCLASS(XAudioServer, godot::Object)

            private:
                static XAudioServer* s_singleton;
                
                godot::AudioServer* m_as = nullptr;
                
                // Bus management
                std::map<int, XAudioBus> m_buses;
                std::map<std::string, int> m_bus_name_to_index;
                
                // Audio stream generators (for tensor-based synthesis)
                std::map<uint64_t, godot::Ref<godot::AudioStreamGenerator>> m_generators;
                std::map<uint64_t, godot::Ref<godot::AudioStreamGeneratorPlayback>> m_playbacks;
                std::map<uint64_t, AudioBuffer> m_generator_buffers;
                
                // Audio input capture
                std::map<uint64_t, AudioBuffer> m_capture_buffers;
                std::map<uint64_t, size_t> m_capture_positions;
                
                // Spectrum analysis
                std::map<int, std::vector<std::complex<float>>> m_spectrum_history;
                size_t m_spectrum_history_size = 10;
                
                // Mixing and processing
                std::mutex m_audio_mutex;
                bool m_processing_active = false;
                int m_mix_rate = 48000;
                int m_mix_buffer_size = 1024;

            protected:
                static void _bind_methods()
                {
                    // Singleton access
                    godot::ClassDB::bind_method(godot::D_METHOD("get_singleton"), &XAudioServer::get_singleton);
                    
                    // Device management
                    godot::ClassDB::bind_method(godot::D_METHOD("get_input_devices"), &XAudioServer::get_input_devices);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_output_devices"), &XAudioServer::get_output_devices);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_input_device", "device_name"), &XAudioServer::set_input_device);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_output_device", "device_name"), &XAudioServer::set_output_device);
                    
                    // Bus management
                    godot::ClassDB::bind_method(godot::D_METHOD("add_bus", "name"), &XAudioServer::add_bus);
                    godot::ClassDB::bind_method(godot::D_METHOD("remove_bus", "index"), &XAudioServer::remove_bus);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_bus_count"), &XAudioServer::get_bus_count);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_bus_name", "index"), &XAudioServer::get_bus_name);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_bus_index", "name"), &XAudioServer::get_bus_index);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_bus_volume", "index", "volume_db"), &XAudioServer::set_bus_volume);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_bus_volume", "index"), &XAudioServer::get_bus_volume);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_bus_mute", "index", "mute"), &XAudioServer::set_bus_mute);
                    godot::ClassDB::bind_method(godot::D_METHOD("is_bus_mute", "index"), &XAudioServer::is_bus_mute);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_bus_solo", "index", "solo"), &XAudioServer::set_bus_solo);
                    godot::ClassDB::bind_method(godot::D_METHOD("is_bus_solo", "index"), &XAudioServer::is_bus_solo);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_bus_send", "index", "send_bus"), &XAudioServer::set_bus_send);
                    
                    // Effect management
                    godot::ClassDB::bind_method(godot::D_METHOD("add_bus_effect", "bus_index", "effect_type", "params"), &XAudioServer::add_bus_effect);
                    godot::ClassDB::bind_method(godot::D_METHOD("remove_bus_effect", "bus_index", "effect_index"), &XAudioServer::remove_bus_effect);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_bus_effect_params", "bus_index", "effect_index", "params"), &XAudioServer::set_bus_effect_params);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_bus_effects", "bus_index"), &XAudioServer::get_bus_effects);
                    
                    // Tensor-based audio processing
                    godot::ClassDB::bind_method(godot::D_METHOD("process_bus_audio", "bus_index", "input_tensor"), &XAudioServer::process_bus_audio);
                    godot::ClassDB::bind_method(godot::D_METHOD("mix_buses_to_tensor", "bus_indices"), &XAudioServer::mix_buses_to_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_bus_peak_volume", "bus_index"), &XAudioServer::get_bus_peak_volume);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_bus_spectrum", "bus_index", "fft_size"), &XAudioServer::get_bus_spectrum, godot::DEFVAL(1024));
                    
                    // Audio synthesis (tensor to stream)
                    godot::ClassDB::bind_method(godot::D_METHOD("create_generator", "sample_rate", "channels"), &XAudioServer::create_generator);
                    godot::ClassDB::bind_method(godot::D_METHOD("push_audio_tensor", "generator_id", "tensor"), &XAudioServer::push_audio_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_generator_playback", "generator_id"), &XAudioServer::get_generator_playback);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear_generator", "generator_id"), &XAudioServer::clear_generator);
                    
                    // Audio capture (tensor from microphone)
                    godot::ClassDB::bind_method(godot::D_METHOD("start_capture", "device_name", "sample_rate", "channels"), &XAudioServer::start_capture);
                    godot::ClassDB::bind_method(godot::D_METHOD("stop_capture", "capture_id"), &XAudioServer::stop_capture);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_captured_tensor", "capture_id"), &XAudioServer::get_captured_tensor);
                    
                    // 3D Audio spatialization
                    godot::ClassDB::bind_method(godot::D_METHOD("spatialize_audio", "source_tensor", "source_positions", "listener_position", "listener_orientation"), &XAudioServer::spatialize_audio);
                    godot::ClassDB::bind_method(godot::D_METHOD("compute_3d_panning", "source_positions", "listener_position", "listener_orientation"), &XAudioServer::compute_3d_panning);
                    
                    // Real-time analysis
                    godot::ClassDB::bind_method(godot::D_METHOD("analyze_spectrum_tensor", "audio_tensor", "fft_size", "window_type"), &XAudioServer::analyze_spectrum_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("compute_mfcc_tensor", "audio_tensor", "sample_rate", "num_coeffs"), &XAudioServer::compute_mfcc_tensor);
                    
                    ADD_SIGNAL(godot::MethodInfo("bus_added", godot::PropertyInfo(godot::Variant::INT, "index"), godot::PropertyInfo(godot::Variant::STRING, "name")));
                    ADD_SIGNAL(godot::MethodInfo("bus_removed", godot::PropertyInfo(godot::Variant::INT, "index")));
                    ADD_SIGNAL(godot::MethodInfo("audio_processed", godot::PropertyInfo(godot::Variant::INT, "bus_index")));
                }

            public:
                XAudioServer()
                {
                    s_singleton = this;
                    m_as = godot::AudioServer::get_singleton();
                    // Initialize master bus
                    XAudioBus master;
                    master.name = "Master";
                    master.volume_db = 0.0f;
                    m_buses[0] = master;
                    m_bus_name_to_index["Master"] = 0;
                }
                
                ~XAudioServer()
                {
                    for (auto& p : m_generators)
                        if (p.second.is_valid())
                            p.second->end_stream();
                    s_singleton = nullptr;
                }
                
                static XAudioServer* get_singleton() { return s_singleton; }

                // --------------------------------------------------------------------
                // Device Management
                // --------------------------------------------------------------------
                godot::Array get_input_devices() const
                {
                    godot::Array devices;
                    if (m_as)
                    {
                        godot::PackedStringArray devs = m_as->get_input_device_list();
                        for (int i = 0; i < devs.size(); ++i)
                            devices.append(devs[i]);
                    }
                    return devices;
                }

                godot::Array get_output_devices() const
                {
                    godot::Array devices;
                    if (m_as)
                    {
                        godot::PackedStringArray devs = m_as->get_output_device_list();
                        for (int i = 0; i < devs.size(); ++i)
                            devices.append(devs[i]);
                    }
                    return devices;
                }

                void set_input_device(const godot::String& device_name)
                {
                    if (m_as)
                        m_as->set_input_device(device_name);
                }

                void set_output_device(const godot::String& device_name)
                {
                    if (m_as)
                        m_as->set_output_device(device_name);
                }

                // --------------------------------------------------------------------
                // Bus Management
                // --------------------------------------------------------------------
                int add_bus(const godot::String& name)
                {
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    std::string n = name.utf8().get_data();
                    if (m_bus_name_to_index.find(n) != m_bus_name_to_index.end())
                        return m_bus_name_to_index[n]; // Already exists
                    
                    int new_idx = static_cast<int>(m_buses.size());
                    while (m_buses.find(new_idx) != m_buses.end())
                        new_idx++;
                    
                    XAudioBus bus;
                    bus.name = n;
                    m_buses[new_idx] = bus;
                    m_bus_name_to_index[n] = new_idx;
                    
                    emit_signal("bus_added", new_idx, name);
                    return new_idx;
                }

                void remove_bus(int index)
                {
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    if (index == 0) return; // Cannot remove master
                    auto it = m_buses.find(index);
                    if (it != m_buses.end())
                    {
                        m_bus_name_to_index.erase(it->second.name);
                        m_buses.erase(it);
                        emit_signal("bus_removed", index);
                    }
                }

                int get_bus_count() const
                {
                    return static_cast<int>(m_buses.size());
                }

                godot::String get_bus_name(int index) const
                {
                    auto it = m_buses.find(index);
                    if (it != m_buses.end())
                        return godot::String(it->second.name.c_str());
                    return godot::String();
                }

                int get_bus_index(const godot::String& name) const
                {
                    std::string n = name.utf8().get_data();
                    auto it = m_bus_name_to_index.find(n);
                    if (it != m_bus_name_to_index.end())
                        return it->second;
                    return -1;
                }

                void set_bus_volume(int index, float volume_db)
                {
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    auto it = m_buses.find(index);
                    if (it != m_buses.end())
                    {
                        it->second.volume_db = volume_db;
                        if (m_as)
                            m_as->set_bus_volume_db(index, volume_db);
                    }
                }

                float get_bus_volume(int index) const
                {
                    auto it = m_buses.find(index);
                    if (it != m_buses.end())
                        return it->second.volume_db;
                    return 0.0f;
                }

                void set_bus_mute(int index, bool mute)
                {
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    auto it = m_buses.find(index);
                    if (it != m_buses.end())
                    {
                        it->second.mute = mute;
                        if (m_as)
                            m_as->set_bus_mute(index, mute);
                    }
                }

                bool is_bus_mute(int index) const
                {
                    auto it = m_buses.find(index);
                    if (it != m_buses.end())
                        return it->second.mute;
                    return false;
                }

                void set_bus_solo(int index, bool solo)
                {
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    auto it = m_buses.find(index);
                    if (it != m_buses.end())
                    {
                        it->second.solo = solo;
                        if (m_as)
                            m_as->set_bus_solo(index, solo);
                    }
                }

                bool is_bus_solo(int index) const
                {
                    auto it = m_buses.find(index);
                    if (it != m_buses.end())
                        return it->second.solo;
                    return false;
                }

                void set_bus_send(int index, int send_bus)
                {
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    auto it = m_buses.find(index);
                    if (it != m_buses.end())
                    {
                        it->second.output_bus = send_bus;
                        if (m_as)
                            m_as->set_bus_send(index, send_bus);
                    }
                }

                // --------------------------------------------------------------------
                // Effect Management
                // --------------------------------------------------------------------
                int add_bus_effect(int bus_index, const godot::String& effect_type, const godot::Dictionary& params)
                {
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    auto it = m_buses.find(bus_index);
                    if (it == m_buses.end()) return -1;
                    
                    godot::Ref<godot::AudioEffect> effect;
                    std::string type = effect_type.utf8().get_data();
                    
                    if (type == "Reverb")
                    {
                        godot::Ref<godot::AudioEffectReverb> rev;
                        rev.instantiate();
                        if (params.has("room_size")) rev->set_room_size(params["room_size"]);
                        if (params.has("damping")) rev->set_damping(params["damping"]);
                        if (params.has("wet")) rev->set_wet(params["wet"]);
                        if (params.has("dry")) rev->set_dry(params["dry"]);
                        effect = rev;
                    }
                    else if (type == "Delay")
                    {
                        godot::Ref<godot::AudioEffectDelay> delay;
                        delay.instantiate();
                        if (params.has("delay_msec")) delay->set_delay_msec(params["delay_msec"]);
                        if (params.has("feedback")) delay->set_feedback(params["feedback"]);
                        if (params.has("tap1_msec")) delay->set_tap1_msec(params["tap1_msec"]);
                        effect = delay;
                    }
                    else if (type == "Chorus")
                    {
                        godot::Ref<godot::AudioEffectChorus> chorus;
                        chorus.instantiate();
                        if (params.has("voice_count")) chorus->set_voice_count(params["voice_count"]);
                        if (params.has("depth")) chorus->set_depth(params["depth"]);
                        if (params.has("rate")) chorus->set_rate(params["rate"]);
                        effect = chorus;
                    }
                    else if (type == "EQ")
                    {
                        godot::Ref<godot::AudioEffectEQ> eq;
                        eq.instantiate();
                        if (params.has("bands"))
                        {
                            godot::Array bands = params["bands"];
                            for (int i = 0; i < bands.size(); ++i)
                            {
                                godot::Dictionary band = bands[i];
                                if (band.has("gain_db"))
                                    eq->set_band_gain_db(i, band["gain_db"]);
                            }
                        }
                        effect = eq;
                    }
                    else if (type == "Compressor")
                    {
                        godot::Ref<godot::AudioEffectCompressor> comp;
                        comp.instantiate();
                        if (params.has("threshold")) comp->set_threshold(params["threshold"]);
                        if (params.has("ratio")) comp->set_ratio(params["ratio"]);
                        if (params.has("attack")) comp->set_attack_us(params["attack"]);
                        if (params.has("release")) comp->set_release_ms(params["release"]);
                        effect = comp;
                    }
                    else if (type == "Panner")
                    {
                        godot::Ref<godot::AudioEffectPanner> panner;
                        panner.instantiate();
                        if (params.has("pan")) panner->set_pan(params["pan"]);
                        effect = panner;
                    }
                    else if (type == "PitchShift")
                    {
                        godot::Ref<godot::AudioEffectPitchShift> ps;
                        ps.instantiate();
                        if (params.has("pitch_scale")) ps->set_pitch_scale(params["pitch_scale"]);
                        effect = ps;
                    }
                    else if (type == "Phaser")
                    {
                        godot::Ref<godot::AudioEffectPhaser> phaser;
                        phaser.instantiate();
                        if (params.has("depth")) phaser->set_depth(params["depth"]);
                        if (params.has("feedback")) phaser->set_feedback(params["feedback"]);
                        effect = phaser;
                    }
                    else if (type == "Distortion")
                    {
                        godot::Ref<godot::AudioEffectDistortion> dist;
                        dist.instantiate();
                        if (params.has("drive")) dist->set_drive(params["drive"]);
                        if (params.has("keep_hf")) dist->set_keep_hf_hz(params["keep_hf"]);
                        effect = dist;
                    }
                    else if (type == "StereoEnhance")
                    {
                        godot::Ref<godot::AudioEffectStereoEnhance> se;
                        se.instantiate();
                        if (params.has("pan_pullout")) se->set_pan_pullout(params["pan_pullout"]);
                        effect = se;
                    }
                    else if (type == "SpectrumAnalyzer")
                    {
                        godot::Ref<godot::AudioEffectSpectrumAnalyzer> sa;
                        sa.instantiate();
                        if (params.has("buffer_length")) sa->set_buffer_length(params["buffer_length"]);
                        effect = sa;
                    }
                    else
                    {
                        return -1;
                    }
                    
                    if (effect.is_valid())
                    {
                        it->second.effects.push_back(effect);
                        if (m_as)
                            m_as->add_bus_effect(bus_index, effect);
                        return static_cast<int>(it->second.effects.size() - 1);
                    }
                    return -1;
                }

                void remove_bus_effect(int bus_index, int effect_index)
                {
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    auto it = m_buses.find(bus_index);
                    if (it != m_buses.end() && effect_index >= 0 && effect_index < static_cast<int>(it->second.effects.size()))
                    {
                        if (m_as)
                            m_as->remove_bus_effect(bus_index, effect_index);
                        it->second.effects.erase(it->second.effects.begin() + effect_index);
                    }
                }

                void set_bus_effect_params(int bus_index, int effect_index, const godot::Dictionary& params)
                {
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    auto it = m_buses.find(bus_index);
                    if (it == m_buses.end() || effect_index >= static_cast<int>(it->second.effects.size()))
                        return;
                    
                    godot::Ref<godot::AudioEffect> effect = it->second.effects[effect_index];
                    // Set parameters based on effect type (simplified)
                    // In a full implementation, we would cast to specific effect types
                    if (m_as)
                        m_as->set_bus_effect_enabled(bus_index, effect_index, params.has("enabled") ? static_cast<bool>(params["enabled"]) : true);
                }

                godot::Array get_bus_effects(int bus_index) const
                {
                    godot::Array result;
                    auto it = m_buses.find(bus_index);
                    if (it != m_buses.end())
                    {
                        for (const auto& e : it->second.effects)
                            result.append(e);
                    }
                    return result;
                }

                // --------------------------------------------------------------------
                // Tensor-based Audio Processing
                // --------------------------------------------------------------------
                godot::Ref<XTensorNode> process_bus_audio(int bus_index, const godot::Ref<XTensorNode>& input_tensor)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!input_tensor.is_valid()) return result;
                    
                    auto audio = input_tensor->get_tensor_resource()->m_data.to_float_array();
                    if (audio.dimension() != 2)
                    {
                        godot::UtilityFunctions::printerr("process_bus_audio: input must be N x C (samples x channels)");
                        return result;
                    }
                    
                    size_t num_samples = audio.shape()[0];
                    size_t num_channels = audio.shape()[1];
                    
                    // Apply bus volume
                    auto it = m_buses.find(bus_index);
                    if (it != m_buses.end())
                    {
                        float gain = std::pow(10.0f, it->second.volume_db / 20.0f);
                        if (it->second.mute) gain = 0.0f;
                        audio = audio * gain;
                    }
                    
                    // Apply effects (simplified)
                    for (const auto& effect : it->second.effects)
                    {
                        if (effect->is_class("AudioEffectPanner"))
                        {
                            godot::Ref<godot::AudioEffectPanner> panner = effect;
                            float pan = panner->get_pan();
                            for (size_t i = 0; i < num_samples; ++i)
                            {
                                float left = audio(i, 0);
                                float right = (num_channels > 1) ? audio(i, 1) : left;
                                if (pan < 0)
                                {
                                    audio(i, 0) = left * (1.0f + pan);
                                    if (num_channels > 1)
                                        audio(i, 1) = right;
                                }
                                else
                                {
                                    if (num_channels > 1)
                                        audio(i, 1) = right * (1.0f - pan);
                                }
                            }
                        }
                        // Additional effect processing would go here
                    }
                    
                    result->set_data(XVariant::from_xarray(audio.cast<double>()).variant());
                    emit_signal("audio_processed", bus_index);
                    return result;
                }

                godot::Ref<XTensorNode> mix_buses_to_tensor(const godot::PackedInt64Array& bus_indices)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    // Mix audio from multiple buses (simplified)
                    return result;
                }

                float get_bus_peak_volume(int bus_index)
                {
                    // Placeholder: return peak volume from recent audio
                    return 0.0f;
                }

                godot::Ref<XTensorNode> get_bus_spectrum(int bus_index, int fft_size)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    // Get spectrum from bus (requires capturing audio)
                    return result;
                }

                // --------------------------------------------------------------------
                // Audio Synthesis (Tensor to Stream)
                // --------------------------------------------------------------------
                uint64_t create_generator(int sample_rate, int channels)
                {
                    godot::Ref<godot::AudioStreamGenerator> gen;
                    gen.instantiate();
                    gen->set_mix_rate(sample_rate);
                    
                    godot::Ref<godot::AudioStreamGeneratorPlayback> pb = gen->get_playback();
                    
                    uint64_t id = gen->get_instance_id();
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    m_generators[id] = gen;
                    m_playbacks[id] = pb;
                    m_generator_buffers[id] = AudioBuffer({0, static_cast<size_t>(channels)});
                    return id;
                }

                void push_audio_tensor(uint64_t generator_id, const godot::Ref<XTensorNode>& tensor)
                {
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    auto it = m_playbacks.find(generator_id);
                    if (it == m_playbacks.end() || !it->second.is_valid() || !tensor.is_valid())
                        return;
                    
                    auto audio = tensor->get_tensor_resource()->m_data.to_float_array();
                    if (audio.dimension() == 1)
                    {
                        // Reshape to Nx1
                        audio = xt::view(audio, xt::all(), xt::newaxis());
                    }
                    
                    size_t num_frames = audio.shape()[0];
                    size_t num_channels = audio.shape()[1];
                    
                    // Push to AudioStreamGeneratorPlayback
                    for (size_t i = 0; i < num_frames; ++i)
                    {
                        godot::Vector2 frame;
                        frame.x = audio(i, 0);
                        frame.y = (num_channels > 1) ? audio(i, 1) : frame.x;
                        it->second->push_frame(frame);
                    }
                    
                    // Store buffer for later retrieval
                    m_generator_buffers[generator_id] = audio;
                }

                godot::Ref<godot::AudioStreamGeneratorPlayback> get_generator_playback(uint64_t generator_id) const
                {
                    auto it = m_playbacks.find(generator_id);
                    if (it != m_playbacks.end())
                        return it->second;
                    return godot::Ref<godot::AudioStreamGeneratorPlayback>();
                }

                void clear_generator(uint64_t generator_id)
                {
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    auto it = m_generators.find(generator_id);
                    if (it != m_generators.end())
                    {
                        it->second->end_stream();
                        m_generators.erase(it);
                        m_playbacks.erase(generator_id);
                        m_generator_buffers.erase(generator_id);
                    }
                }

                // --------------------------------------------------------------------
                // Audio Capture (Tensor from Microphone)
                // --------------------------------------------------------------------
                uint64_t start_capture(const godot::String& device_name, int sample_rate, int channels)
                {
                    godot::Ref<godot::AudioStreamMicrophone> mic;
                    mic.instantiate();
                    // Note: AudioStreamMicrophone requires enabling input in project settings
                    
                    uint64_t id = mic->get_instance_id();
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    m_capture_buffers[id] = AudioBuffer({0, static_cast<size_t>(channels)});
                    m_capture_positions[id] = 0;
                    return id;
                }

                void stop_capture(uint64_t capture_id)
                {
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    m_capture_buffers.erase(capture_id);
                    m_capture_positions.erase(capture_id);
                }

                godot::Ref<XTensorNode> get_captured_tensor(uint64_t capture_id)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    std::lock_guard<std::mutex> lock(m_audio_mutex);
                    auto it = m_capture_buffers.find(capture_id);
                    if (it != m_capture_buffers.end())
                    {
                        result->set_data(XVariant::from_xarray(it->second.cast<double>()).variant());
                    }
                    return result;
                }

                // --------------------------------------------------------------------
                // 3D Audio Spatialization
                // --------------------------------------------------------------------
                godot::Ref<XTensorNode> spatialize_audio(const godot::Ref<XTensorNode>& source_tensor,
                                                         const godot::Ref<XTensorNode>& source_positions,
                                                         const godot::Vector3& listener_position,
                                                         const godot::Basis& listener_orientation)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!source_tensor.is_valid() || !source_positions.is_valid())
                        return result;
                    
                    auto audio = source_tensor->get_tensor_resource()->m_data.to_float_array();
                    auto positions = source_positions->get_tensor_resource()->m_data.to_float_array();
                    
                    if (audio.shape()[0] != positions.shape()[0])
                    {
                        godot::UtilityFunctions::printerr("spatialize_audio: source count mismatch");
                        return result;
                    }
                    
                    size_t num_sources = audio.shape()[0];
                    size_t num_samples = audio.shape()[1];
                    
                    xarray_container<float> spatialized({num_samples, 2}); // stereo output
                    godot::Vector3 listener_pos = listener_position;
                    godot::Vector3 listener_forward = -listener_orientation.get_column(2);
                    godot::Vector3 listener_right = listener_orientation.get_column(0);
                    
                    for (size_t i = 0; i < num_sources; ++i)
                    {
                        godot::Vector3 source_pos(positions(i, 0), positions(i, 1), positions(i, 2));
                        godot::Vector3 to_source = source_pos - listener_pos;
                        float distance = to_source.length();
                        if (distance < 0.001f) distance = 0.001f;
                        
                        // Distance attenuation (inverse square)
                        float attenuation = 1.0f / (1.0f + distance * distance * 0.01f);
                        
                        // Panning based on angle to listener
                        godot::Vector3 dir = to_source / distance;
                        float dot_right = dir.dot(listener_right);
                        float dot_forward = dir.dot(listener_forward);
                        
                        // Simple stereo panning
                        float pan = std::clamp(dot_right, -1.0f, 1.0f);
                        float left_gain = std::sqrt(0.5f * (1.0f - pan));
                        float right_gain = std::sqrt(0.5f * (1.0f + pan));
                        
                        // Delay based on distance (speed of sound ~ 343 m/s)
                        // Not implemented in this simplified version
                        
                        // Mix into output
                        for (size_t s = 0; s < num_samples; ++s)
                        {
                            float sample_val = (audio.shape().size() > 2) ? audio(i, s) : audio.flat(i * num_samples + s);
                            spatialized(s, 0) += sample_val * attenuation * left_gain;
                            spatialized(s, 1) += sample_val * attenuation * right_gain;
                        }
                    }
                    
                    // Normalize
                    float max_val = xt::amax(xt::abs(spatialized))();
                    if (max_val > 1.0f)
                        spatialized = spatialized / max_val;
                    
                    result->set_data(XVariant::from_xarray(spatialized.cast<double>()).variant());
                    return result;
                }

                godot::Ref<XTensorNode> compute_3d_panning(const godot::Ref<XTensorNode>& source_positions,
                                                           const godot::Vector3& listener_position,
                                                           const godot::Basis& listener_orientation)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!source_positions.is_valid()) return result;
                    
                    auto positions = source_positions->get_tensor_resource()->m_data.to_float_array();
                    size_t num_sources = positions.shape()[0];
                    xarray_container<float> panning({num_sources, 4}); // left, right, distance, angle
                    
                    godot::Vector3 listener_pos = listener_position;
                    godot::Vector3 listener_forward = -listener_orientation.get_column(2);
                    godot::Vector3 listener_right = listener_orientation.get_column(0);
                    
                    for (size_t i = 0; i < num_sources; ++i)
                    {
                        godot::Vector3 source_pos(positions(i, 0), positions(i, 1), positions(i, 2));
                        godot::Vector3 to_source = source_pos - listener_pos;
                        float distance = to_source.length();
                        if (distance < 0.001f) distance = 0.001f;
                        
                        godot::Vector3 dir = to_source / distance;
                        float dot_right = dir.dot(listener_right);
                        float dot_forward = dir.dot(listener_forward);
                        
                        float attenuation = 1.0f / (1.0f + distance * distance * 0.01f);
                        float pan = std::clamp(dot_right, -1.0f, 1.0f);
                        float left_gain = std::sqrt(0.5f * (1.0f - pan));
                        float right_gain = std::sqrt(0.5f * (1.0f + pan));
                        float angle = std::atan2(dot_right, dot_forward);
                        
                        panning(i, 0) = left_gain * attenuation;
                        panning(i, 1) = right_gain * attenuation;
                        panning(i, 2) = distance;
                        panning(i, 3) = angle;
                    }
                    
                    result->set_data(XVariant::from_xarray(panning.cast<double>()).variant());
                    return result;
                }

                // --------------------------------------------------------------------
                // Real-time Analysis
                // --------------------------------------------------------------------
                godot::Ref<XTensorNode> analyze_spectrum_tensor(const godot::Ref<XTensorNode>& audio_tensor,
                                                                int fft_size, const godot::String& window_type)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!audio_tensor.is_valid()) return result;
                    
                    auto audio = audio_tensor->get_tensor_resource()->m_data.to_float_array();
                    if (audio.dimension() != 1)
                    {
                        if (audio.dimension() == 2)
                            audio = xt::view(audio, xt::all(), 0); // Use first channel
                        else
                            return result;
                    }
                    
                    // Apply window
                    auto window = signal::get_window(window_type.utf8().get_data(), audio.size(), false);
                    for (size_t i = 0; i < std::min(audio.size(), window.size()); ++i)
                        audio(i) = audio(i) * static_cast<float>(window(i));
                    
                    // Compute FFT
                    size_t n_fft = static_cast<size_t>(fft_size);
                    if (n_fft > audio.size())
                    {
                        // Zero pad
                        xarray_container<float> padded({n_fft}, 0.0f);
                        for (size_t i = 0; i < audio.size(); ++i)
                            padded(i) = audio(i);
                        audio = padded;
                    }
                    else if (n_fft < audio.size())
                    {
                        audio = xt::view(audio, xt::range(0, n_fft));
                    }
                    
                    auto spectrum = xt::fft::rfft(audio.cast<double>());
                    size_t num_bins = spectrum.size();
                    xarray_container<float> magnitude({num_bins});
                    for (size_t i = 0; i < num_bins; ++i)
                        magnitude(i) = std::abs(static_cast<float>(spectrum(i)));
                    
                    result->set_data(XVariant::from_xarray(magnitude.cast<double>()).variant());
                    return result;
                }

                godot::Ref<XTensorNode> compute_mfcc_tensor(const godot::Ref<XTensorNode>& audio_tensor,
                                                            int sample_rate, int num_coeffs)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!audio_tensor.is_valid()) return result;
                    
                    auto audio = audio_tensor->get_tensor_resource()->m_data.to_float_array();
                    if (audio.dimension() != 1)
                    {
                        if (audio.dimension() == 2)
                            audio = xt::view(audio, xt::all(), 0);
                        else
                            return result;
                    }
                    
                    // Compute MFCC using audio module functions
                    audio::AudioSignal sig(audio.cast<double>(), static_cast<size_t>(sample_rate));
                    auto mfcc_result = audio::mfcc(sig, static_cast<size_t>(num_coeffs), 40, 1024, 512);
                    result->set_data(XVariant::from_xarray(mfcc_result).variant());
                    return result;
                }
            };

            // Singleton instance
            XAudioServer* XAudioServer::s_singleton = nullptr;
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XAudioServerRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XAudioServer>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::XAudioServer;
        using godot_bridge::XAudioServerRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XAUDIOSERVER_HPP

// godot/xaudioserver.hpp