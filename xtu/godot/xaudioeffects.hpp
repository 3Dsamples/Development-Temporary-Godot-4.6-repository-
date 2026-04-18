// godot/xaudioeffects.hpp

#ifndef XTENSOR_XAUDIOEFFECTS_HPP
#define XTENSOR_XAUDIOEFFECTS_HPP

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
#include <complex>
#include <random>
#include <queue>
#include <mutex>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/audio_effect.hpp>
    #include <godot_cpp/classes/audio_effect_instance.hpp>
    #include <godot_cpp/classes/audio_server.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // Base Audio Effect Processor (tensor-based, batch processing)
            // --------------------------------------------------------------------
            class XAudioEffectProcessor
            {
            public:
                virtual ~XAudioEffectProcessor() = default;
                virtual AudioBuffer process(const AudioBuffer& input, float sample_rate) = 0;
                virtual void reset() = 0;
                virtual void set_parameter(const std::string& name, float value) = 0;
                virtual float get_parameter(const std::string& name) const = 0;
                virtual std::vector<std::string> get_parameter_names() const = 0;
            };

            // --------------------------------------------------------------------
            // Delay Effect (with feedback and modulation)
            // --------------------------------------------------------------------
            class XDelayEffect : public XAudioEffectProcessor
            {
            public:
                XDelayEffect(float max_delay_sec = 1.0f, float sample_rate = 44100.0f)
                    : m_max_delay_samples(static_cast<size_t>(max_delay_sec * sample_rate))
                {
                    m_buffer.resize(m_max_delay_samples, 0.0f);
                    m_delay_samples = static_cast<size_t>(0.25f * sample_rate); // 250ms default
                    m_feedback = 0.3f;
                    m_mix = 0.5f;
                    m_modulation_rate = 0.5f;
                    m_modulation_depth = 0.002f;
                }

                AudioBuffer process(const AudioBuffer& input, float sample_rate) override
                {
                    size_t num_samples = input.shape()[0];
                    size_t num_channels = input.shape()[1];
                    AudioBuffer output = input;
                    
                    // Update delay line length based on sample rate
                    float target_delay_sec = static_cast<float>(m_delay_samples) / sample_rate;
                    
                    for (size_t ch = 0; ch < num_channels; ++ch)
                    {
                        size_t write_pos = m_write_pos;
                        for (size_t i = 0; i < num_samples; ++i)
                        {
                            // Modulated delay length (flange/chorus effect)
                            float mod = 1.0f + m_modulation_depth * std::sin(2.0f * M_PI * m_modulation_rate * static_cast<float>(i + m_phase) / sample_rate);
                            size_t delay_len = static_cast<size_t>(std::clamp(static_cast<float>(m_delay_samples) * mod, 1.0f, static_cast<float>(m_max_delay_samples - 1)));
                            
                            // Read delayed sample (linear interpolation)
                            float read_pos_f = static_cast<float>(write_pos + m_max_delay_samples - delay_len);
                            if (read_pos_f >= static_cast<float>(m_max_delay_samples))
                                read_pos_f -= static_cast<float>(m_max_delay_samples);
                            
                            size_t idx0 = static_cast<size_t>(std::floor(read_pos_f));
                            size_t idx1 = (idx0 + 1) % m_max_delay_samples;
                            float frac = read_pos_f - static_cast<float>(idx0);
                            
                            float delayed = m_buffer[idx0] * (1.0f - frac) + m_buffer[idx1] * frac;
                            
                            // Mix with input
                            float dry = input(i, ch);
                            float wet = delayed * m_feedback;
                            output(i, ch) = dry * (1.0f - m_mix) + delayed * m_mix;
                            
                            // Write to delay buffer (with feedback)
                            m_buffer[write_pos] = dry + wet;
                            
                            write_pos = (write_pos + 1) % m_max_delay_samples;
                        }
                    }
                    m_write_pos = (m_write_pos + num_samples) % m_max_delay_samples;
                    m_phase += num_samples;
                    return output;
                }

                void reset() override
                {
                    std::fill(m_buffer.begin(), m_buffer.end(), 0.0f);
                    m_write_pos = 0;
                    m_phase = 0;
                }

                void set_parameter(const std::string& name, float value) override
                {
                    if (name == "delay_time_ms")
                        m_delay_samples = static_cast<size_t>(value * 0.001f * 44100.0f); // assumes 44.1k, will be adjusted in process
                    else if (name == "feedback")
                        m_feedback = std::clamp(value, 0.0f, 0.99f);
                    else if (name == "mix")
                        m_mix = std::clamp(value, 0.0f, 1.0f);
                    else if (name == "mod_rate")
                        m_modulation_rate = value;
                    else if (name == "mod_depth")
                        m_modulation_depth = value;
                }

                float get_parameter(const std::string& name) const override
                {
                    if (name == "delay_time_ms")
                        return static_cast<float>(m_delay_samples) * 1000.0f / 44100.0f;
                    else if (name == "feedback") return m_feedback;
                    else if (name == "mix") return m_mix;
                    else if (name == "mod_rate") return m_modulation_rate;
                    else if (name == "mod_depth") return m_modulation_depth;
                    return 0.0f;
                }

                std::vector<std::string> get_parameter_names() const override
                {
                    return {"delay_time_ms", "feedback", "mix", "mod_rate", "mod_depth"};
                }

            private:
                std::vector<float> m_buffer;
                size_t m_max_delay_samples;
                size_t m_delay_samples = 11025; // 250ms @ 44.1k
                size_t m_write_pos = 0;
                size_t m_phase = 0;
                float m_feedback = 0.3f;
                float m_mix = 0.5f;
                float m_modulation_rate = 0.5f;
                float m_modulation_depth = 0.002f;
            };

            // --------------------------------------------------------------------
            // Reverb Effect (Schroeder-Moorer model)
            // --------------------------------------------------------------------
            class XReverbEffect : public XAudioEffectProcessor
            {
            public:
                XReverbEffect(float sample_rate = 44100.0f)
                {
                    // Initialize comb filters
                    m_comb_delays = {0.0297f, 0.0371f, 0.0411f, 0.0437f};
                    m_comb_gains = {0.8f, 0.8f, 0.8f, 0.8f};
                    m_allpass_delays = {0.005f, 0.0017f};
                    m_allpass_gain = 0.5f;
                    
                    for (size_t i = 0; i < m_comb_delays.size(); ++i)
                    {
                        m_comb_buffers.emplace_back(static_cast<size_t>(m_comb_delays[i] * sample_rate), 0.0f);
                        m_comb_pos.push_back(0);
                    }
                    for (size_t i = 0; i < m_allpass_delays.size(); ++i)
                    {
                        m_allpass_buffers.emplace_back(static_cast<size_t>(m_allpass_delays[i] * sample_rate), 0.0f);
                        m_allpass_pos.push_back(0);
                    }
                    
                    m_room_size = 0.5f;
                    m_damping = 0.5f;
                    m_wet = 0.3f;
                    m_dry = 0.7f;
                    m_width = 1.0f;
                }

                AudioBuffer process(const AudioBuffer& input, float sample_rate) override
                {
                    size_t num_samples = input.shape()[0];
                    size_t num_channels = input.shape()[1];
                    AudioBuffer output = input;
                    
                    // Mix to mono for processing
                    AudioBuffer mono_input({num_samples, 1}, 0.0f);
                    for (size_t i = 0; i < num_samples; ++i)
                    {
                        float sum = 0.0f;
                        for (size_t ch = 0; ch < num_channels; ++ch)
                            sum += input(i, ch);
                        mono_input(i, 0) = sum / static_cast<float>(num_channels);
                    }
                    
                    // Process comb filters
                    AudioBuffer comb_out = mono_input;
                    for (size_t c = 0; c < m_comb_buffers.size(); ++c)
                    {
                        size_t delay_len = static_cast<size_t>(m_comb_delays[c] * sample_rate * m_room_size);
                        if (delay_len >= m_comb_buffers[c].size()) delay_len = m_comb_buffers[c].size() - 1;
                        float gain = m_comb_gains[c] * (1.0f - m_damping);
                        
                        size_t pos = m_comb_pos[c];
                        for (size_t i = 0; i < num_samples; ++i)
                        {
                            float delayed = m_comb_buffers[c][pos];
                            float in_val = mono_input(i, 0);
                            float out_val = in_val + gain * delayed;
                            m_comb_buffers[c][pos] = out_val;
                            comb_out(i, 0) += out_val;
                            pos = (pos + 1) % m_comb_buffers[c].size();
                        }
                        m_comb_pos[c] = pos;
                    }
                    
                    // Process all-pass filters
                    AudioBuffer ap_out = comb_out;
                    for (size_t a = 0; a < m_allpass_buffers.size(); ++a)
                    {
                        size_t delay_len = static_cast<size_t>(m_allpass_delays[a] * sample_rate * m_room_size);
                        if (delay_len >= m_allpass_buffers[a].size()) delay_len = m_allpass_buffers[a].size() - 1;
                        
                        size_t pos = m_allpass_pos[a];
                        for (size_t i = 0; i < num_samples; ++i)
                        {
                            float delayed = m_allpass_buffers[a][pos];
                            float in_val = ap_out(i, 0);
                            float out_val = -m_allpass_gain * in_val + in_val + m_allpass_gain * delayed;
                            m_allpass_buffers[a][pos] = out_val;
                            ap_out(i, 0) = out_val;
                            pos = (pos + 1) % m_allpass_buffers[a].size();
                        }
                        m_allpass_pos[a] = pos;
                    }
                    
                    // Mix with dry signal and expand to stereo
                    for (size_t i = 0; i < num_samples; ++i)
                    {
                        for (size_t ch = 0; ch < num_channels; ++ch)
                        {
                            float dry = input(i, ch);
                            float wet = ap_out(i, 0);
                            // Add stereo width by phase inversion on right channel
                            if (ch == 1 && num_channels > 1)
                                wet = -wet * m_width;
                            output(i, ch) = dry * m_dry + wet * m_wet;
                        }
                    }
                    
                    return output;
                }

                void reset() override
                {
                    for (auto& buf : m_comb_buffers) std::fill(buf.begin(), buf.end(), 0.0f);
                    for (auto& buf : m_allpass_buffers) std::fill(buf.begin(), buf.end(), 0.0f);
                    std::fill(m_comb_pos.begin(), m_comb_pos.end(), 0);
                    std::fill(m_allpass_pos.begin(), m_allpass_pos.end(), 0);
                }

                void set_parameter(const std::string& name, float value) override
                {
                    if (name == "room_size") m_room_size = std::clamp(value, 0.1f, 1.0f);
                    else if (name == "damping") m_damping = std::clamp(value, 0.0f, 1.0f);
                    else if (name == "wet") m_wet = std::clamp(value, 0.0f, 1.0f);
                    else if (name == "dry") m_dry = std::clamp(value, 0.0f, 1.0f);
                    else if (name == "width") m_width = std::clamp(value, 0.0f, 1.0f);
                }

                float get_parameter(const std::string& name) const override
                {
                    if (name == "room_size") return m_room_size;
                    else if (name == "damping") return m_damping;
                    else if (name == "wet") return m_wet;
                    else if (name == "dry") return m_dry;
                    else if (name == "width") return m_width;
                    return 0.0f;
                }

                std::vector<std::string> get_parameter_names() const override
                {
                    return {"room_size", "damping", "wet", "dry", "width"};
                }

            private:
                std::vector<std::vector<float>> m_comb_buffers;
                std::vector<std::vector<float>> m_allpass_buffers;
                std::vector<size_t> m_comb_pos;
                std::vector<size_t> m_allpass_pos;
                std::vector<float> m_comb_delays;
                std::vector<float> m_comb_gains;
                std::vector<float> m_allpass_delays;
                float m_allpass_gain;
                float m_room_size;
                float m_damping;
                float m_wet;
                float m_dry;
                float m_width;
            };

            // --------------------------------------------------------------------
            // Compressor / Limiter Effect
            // --------------------------------------------------------------------
            class XCompressorEffect : public XAudioEffectProcessor
            {
            public:
                XCompressorEffect()
                {
                    m_threshold = -12.0f;
                    m_ratio = 4.0f;
                    m_attack_ms = 10.0f;
                    m_release_ms = 100.0f;
                    m_knee_width = 3.0f;
                    m_makeup_gain = 0.0f;
                    m_envelope = 0.0f;
                }

                AudioBuffer process(const AudioBuffer& input, float sample_rate) override
                {
                    size_t num_samples = input.shape()[0];
                    size_t num_channels = input.shape()[1];
                    AudioBuffer output = input;
                    
                    float attack_coeff = std::exp(-1.0f / (m_attack_ms * 0.001f * sample_rate));
                    float release_coeff = std::exp(-1.0f / (m_release_ms * 0.001f * sample_rate));
                    
                    float threshold_linear = std::pow(10.0f, m_threshold / 20.0f);
                    float knee_half = m_knee_width * 0.5f;
                    float ratio_inv = 1.0f / m_ratio;
                    float makeup = std::pow(10.0f, m_makeup_gain / 20.0f);
                    
                    for (size_t ch = 0; ch < num_channels; ++ch)
                    {
                        float envelope = m_envelope;
                        for (size_t i = 0; i < num_samples; ++i)
                        {
                            float in_val = input(i, ch);
                            float in_abs = std::abs(in_val);
                            
                            // Envelope detection
                            if (in_abs > envelope)
                                envelope = attack_coeff * envelope + (1.0f - attack_coeff) * in_abs;
                            else
                                envelope = release_coeff * envelope + (1.0f - release_coeff) * in_abs;
                            
                            // Gain computer
                            float db_in = 20.0f * std::log10(envelope + 1e-10f);
                            float gain_db = 0.0f;
                            
                            if (db_in < m_threshold - knee_half)
                            {
                                gain_db = 0.0f;
                            }
                            else if (db_in > m_threshold + knee_half)
                            {
                                gain_db = m_threshold + (db_in - m_threshold) * ratio_inv - db_in;
                            }
                            else
                            {
                                // Knee interpolation
                                float knee_start = m_threshold - knee_half;
                                float knee_end = m_threshold + knee_half;
                                float overshoot = db_in - knee_start;
                                float knee_ratio = overshoot / (2.0f * knee_half);
                                float linear_gain_db = (knee_start + overshoot * ratio_inv) - db_in;
                                gain_db = linear_gain_db * knee_ratio;
                            }
                            
                            float gain_linear = std::pow(10.0f, gain_db / 20.0f);
                            output(i, ch) = in_val * gain_linear * makeup;
                        }
                        m_envelope = envelope;
                    }
                    
                    return output;
                }

                void reset() override
                {
                    m_envelope = 0.0f;
                }

                void set_parameter(const std::string& name, float value) override
                {
                    if (name == "threshold_db") m_threshold = value;
                    else if (name == "ratio") m_ratio = std::max(1.0f, value);
                    else if (name == "attack_ms") m_attack_ms = std::max(0.1f, value);
                    else if (name == "release_ms") m_release_ms = std::max(0.1f, value);
                    else if (name == "knee_db") m_knee_width = std::max(0.0f, value);
                    else if (name == "makeup_gain_db") m_makeup_gain = value;
                }

                float get_parameter(const std::string& name) const override
                {
                    if (name == "threshold_db") return m_threshold;
                    else if (name == "ratio") return m_ratio;
                    else if (name == "attack_ms") return m_attack_ms;
                    else if (name == "release_ms") return m_release_ms;
                    else if (name == "knee_db") return m_knee_width;
                    else if (name == "makeup_gain_db") return m_makeup_gain;
                    return 0.0f;
                }

                std::vector<std::string> get_parameter_names() const override
                {
                    return {"threshold_db", "ratio", "attack_ms", "release_ms", "knee_db", "makeup_gain_db"};
                }

            private:
                float m_threshold;
                float m_ratio;
                float m_attack_ms;
                float m_release_ms;
                float m_knee_width;
                float m_makeup_gain;
                float m_envelope;
            };

            // --------------------------------------------------------------------
            // Equalizer (Parametric EQ with multiple bands)
            // --------------------------------------------------------------------
            class XEqualizerEffect : public XAudioEffectProcessor
            {
            public:
                struct Band
                {
                    float frequency = 1000.0f;
                    float gain_db = 0.0f;
                    float q = 1.0f;
                    int type = 0; // 0=peaking, 1=lowshelf, 2=highshelf
                };

                XEqualizerEffect()
                {
                    // Default 3-band EQ
                    m_bands.resize(3);
                    m_bands[0] = {100.0f, 0.0f, 0.7f, 1}; // low shelf
                    m_bands[1] = {1000.0f, 0.0f, 1.0f, 0}; // peaking
                    m_bands[2] = {8000.0f, 0.0f, 0.7f, 2}; // high shelf
                }

                AudioBuffer process(const AudioBuffer& input, float sample_rate) override
                {
                    size_t num_samples = input.shape()[0];
                    size_t num_channels = input.shape()[1];
                    AudioBuffer output = input;
                    
                    if (m_need_coeff_update || m_cached_sample_rate != sample_rate)
                    {
                        update_coefficients(sample_rate);
                        m_cached_sample_rate = sample_rate;
                        m_need_coeff_update = false;
                    }
                    
                    for (size_t ch = 0; ch < num_channels; ++ch)
                    {
                        for (size_t band_idx = 0; band_idx < m_bands.size(); ++band_idx)
                        {
                            const auto& coeff = m_band_coeffs[band_idx];
                            // Biquad filter (Direct Form I)
                            float x1 = m_x1[ch * m_bands.size() + band_idx];
                            float x2 = m_x2[ch * m_bands.size() + band_idx];
                            float y1 = m_y1[ch * m_bands.size() + band_idx];
                            float y2 = m_y2[ch * m_bands.size() + band_idx];
                            
                            for (size_t i = 0; i < num_samples; ++i)
                            {
                                float x0 = output(i, ch);
                                float y0 = coeff.b0 * x0 + coeff.b1 * x1 + coeff.b2 * x2
                                         - coeff.a1 * y1 - coeff.a2 * y2;
                                output(i, ch) = y0;
                                x2 = x1; x1 = x0;
                                y2 = y1; y1 = y0;
                            }
                            
                            m_x1[ch * m_bands.size() + band_idx] = x1;
                            m_x2[ch * m_bands.size() + band_idx] = x2;
                            m_y1[ch * m_bands.size() + band_idx] = y1;
                            m_y2[ch * m_bands.size() + band_idx] = y2;
                        }
                    }
                    
                    return output;
                }

                void reset() override
                {
                    std::fill(m_x1.begin(), m_x1.end(), 0.0f);
                    std::fill(m_x2.begin(), m_x2.end(), 0.0f);
                    std::fill(m_y1.begin(), m_y1.end(), 0.0f);
                    std::fill(m_y2.begin(), m_y2.end(), 0.0f);
                }

                void set_parameter(const std::string& name, float value) override
                {
                    // Parse band index and parameter: e.g., "band0_freq", "band1_gain"
                    if (name.find("band") == 0)
                    {
                        size_t underscore = name.find('_');
                        if (underscore != std::string::npos)
                        {
                            int band_idx = std::stoi(name.substr(4, underscore - 4));
                            std::string param = name.substr(underscore + 1);
                            if (band_idx >= 0 && band_idx < static_cast<int>(m_bands.size()))
                            {
                                if (param == "freq")
                                    m_bands[band_idx].frequency = value;
                                else if (param == "gain")
                                    m_bands[band_idx].gain_db = value;
                                else if (param == "q")
                                    m_bands[band_idx].q = std::max(0.1f, value);
                                else if (param == "type")
                                    m_bands[band_idx].type = static_cast<int>(value);
                                m_need_coeff_update = true;
                            }
                        }
                    }
                }

                float get_parameter(const std::string& name) const override
                {
                    if (name.find("band") == 0)
                    {
                        size_t underscore = name.find('_');
                        if (underscore != std::string::npos)
                        {
                            int band_idx = std::stoi(name.substr(4, underscore - 4));
                            std::string param = name.substr(underscore + 1);
                            if (band_idx >= 0 && band_idx < static_cast<int>(m_bands.size()))
                            {
                                if (param == "freq") return m_bands[band_idx].frequency;
                                else if (param == "gain") return m_bands[band_idx].gain_db;
                                else if (param == "q") return m_bands[band_idx].q;
                                else if (param == "type") return static_cast<float>(m_bands[band_idx].type);
                            }
                        }
                    }
                    return 0.0f;
                }

                std::vector<std::string> get_parameter_names() const override
                {
                    std::vector<std::string> names;
                    for (size_t i = 0; i < m_bands.size(); ++i)
                    {
                        names.push_back("band" + std::to_string(i) + "_freq");
                        names.push_back("band" + std::to_string(i) + "_gain");
                        names.push_back("band" + std::to_string(i) + "_q");
                        names.push_back("band" + std::to_string(i) + "_type");
                    }
                    return names;
                }

                void set_bands(const std::vector<Band>& bands)
                {
                    m_bands = bands;
                    m_need_coeff_update = true;
                    size_t num_bands = m_bands.size();
                    m_x1.resize(num_bands * 2, 0.0f);
                    m_x2.resize(num_bands * 2, 0.0f);
                    m_y1.resize(num_bands * 2, 0.0f);
                    m_y2.resize(num_bands * 2, 0.0f);
                    m_band_coeffs.resize(num_bands);
                }

            private:
                struct BiquadCoeff
                {
                    float b0, b1, b2, a1, a2;
                };

                std::vector<Band> m_bands;
                std::vector<BiquadCoeff> m_band_coeffs;
                std::vector<float> m_x1, m_x2, m_y1, m_y2;
                float m_cached_sample_rate = 44100.0f;
                bool m_need_coeff_update = true;

                void update_coefficients(float sample_rate)
                {
                    m_band_coeffs.resize(m_bands.size());
                    for (size_t i = 0; i < m_bands.size(); ++i)
                    {
                        const Band& band = m_bands[i];
                        float freq = band.frequency;
                        float gain = std::pow(10.0f, band.gain_db / 20.0f);
                        float q = band.q;
                        float w0 = 2.0f * M_PI * freq / sample_rate;
                        float cos_w0 = std::cos(w0);
                        float sin_w0 = std::sin(w0);
                        float alpha = sin_w0 / (2.0f * q);
                        
                        BiquadCoeff& coeff = m_band_coeffs[i];
                        if (band.type == 0) // peaking
                        {
                            float A = std::sqrt(gain);
                            coeff.b0 = 1.0f + alpha * A;
                            coeff.b1 = -2.0f * cos_w0;
                            coeff.b2 = 1.0f - alpha * A;
                            coeff.a1 = -2.0f * cos_w0;
                            coeff.a2 = 1.0f - alpha / A;
                            float a0_inv = 1.0f / (1.0f + alpha / A);
                            coeff.b0 *= a0_inv;
                            coeff.b1 *= a0_inv;
                            coeff.b2 *= a0_inv;
                            coeff.a1 *= a0_inv;
                            coeff.a2 *= a0_inv;
                        }
                        else if (band.type == 1) // low shelf
                        {
                            float A = std::sqrt(gain);
                            coeff.b0 = A * ((A + 1.0f) - (A - 1.0f) * cos_w0 + 2.0f * std::sqrt(A) * alpha);
                            coeff.b1 = 2.0f * A * ((A - 1.0f) - (A + 1.0f) * cos_w0);
                            coeff.b2 = A * ((A + 1.0f) - (A - 1.0f) * cos_w0 - 2.0f * std::sqrt(A) * alpha);
                            coeff.a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cos_w0);
                            coeff.a2 = (A + 1.0f) + (A - 1.0f) * cos_w0 - 2.0f * std::sqrt(A) * alpha;
                            float a0_inv = 1.0f / ((A + 1.0f) + (A - 1.0f) * cos_w0 + 2.0f * std::sqrt(A) * alpha);
                            coeff.b0 *= a0_inv;
                            coeff.b1 *= a0_inv;
                            coeff.b2 *= a0_inv;
                            coeff.a1 *= a0_inv;
                            coeff.a2 *= a0_inv;
                        }
                        else if (band.type == 2) // high shelf
                        {
                            float A = std::sqrt(gain);
                            coeff.b0 = A * ((A + 1.0f) + (A - 1.0f) * cos_w0 + 2.0f * std::sqrt(A) * alpha);
                            coeff.b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cos_w0);
                            coeff.b2 = A * ((A + 1.0f) + (A - 1.0f) * cos_w0 - 2.0f * std::sqrt(A) * alpha);
                            coeff.a1 = 2.0f * ((A - 1.0f) - (A + 1.0f) * cos_w0);
                            coeff.a2 = (A + 1.0f) - (A - 1.0f) * cos_w0 - 2.0f * std::sqrt(A) * alpha;
                            float a0_inv = 1.0f / ((A + 1.0f) - (A - 1.0f) * cos_w0 + 2.0f * std::sqrt(A) * alpha);
                            coeff.b0 *= a0_inv;
                            coeff.b1 *= a0_inv;
                            coeff.b2 *= a0_inv;
                            coeff.a1 *= a0_inv;
                            coeff.a2 *= a0_inv;
                        }
                        else // default peaking
                        {
                            float A = std::sqrt(gain);
                            coeff.b0 = 1.0f + alpha * A;
                            coeff.b1 = -2.0f * cos_w0;
                            coeff.b2 = 1.0f - alpha * A;
                            coeff.a1 = -2.0f * cos_w0;
                            coeff.a2 = 1.0f - alpha / A;
                            float a0_inv = 1.0f / (1.0f + alpha / A);
                            coeff.b0 *= a0_inv;
                            coeff.b1 *= a0_inv;
                            coeff.b2 *= a0_inv;
                            coeff.a1 *= a0_inv;
                            coeff.a2 *= a0_inv;
                        }
                    }
                }
            };

            // --------------------------------------------------------------------
            // Chorus / Flanger Effect
            // --------------------------------------------------------------------
            class XChorusEffect : public XAudioEffectProcessor
            {
            public:
                XChorusEffect(float sample_rate = 44100.0f)
                {
                    m_max_delay = static_cast<size_t>(0.05f * sample_rate); // 50ms max
                    m_buffer.resize(m_max_delay, 0.0f);
                    m_rate = 0.5f;
                    m_depth = 0.002f;
                    m_feedback = 0.3f;
                    m_mix = 0.5f;
                    m_voices = 1;
                }

                AudioBuffer process(const AudioBuffer& input, float sample_rate) override
                {
                    size_t num_samples = input.shape()[0];
                    size_t num_channels = input.shape()[1];
                    AudioBuffer output = input;
                    
                    for (size_t ch = 0; ch < num_channels; ++ch)
                    {
                        size_t write_pos = m_write_pos;
                        for (size_t i = 0; i < num_samples; ++i)
                        {
                            float dry = input(i, ch);
                            float wet_sum = 0.0f;
                            
                            for (int v = 0; v < m_voices; ++v)
                            {
                                float phase_offset = 2.0f * M_PI * v / m_voices;
                                float mod = 1.0f + m_depth * std::sin(2.0f * M_PI * m_rate * static_cast<float>(i + m_phase) / sample_rate + phase_offset);
                                size_t delay_len = static_cast<size_t>(std::clamp(mod * static_cast<float>(m_base_delay), 1.0f, static_cast<float>(m_max_delay - 1)));
                                
                                float read_pos_f = static_cast<float>(write_pos + m_max_delay - delay_len);
                                if (read_pos_f >= static_cast<float>(m_max_delay))
                                    read_pos_f -= static_cast<float>(m_max_delay);
                                
                                size_t idx0 = static_cast<size_t>(std::floor(read_pos_f));
                                size_t idx1 = (idx0 + 1) % m_max_delay;
                                float frac = read_pos_f - static_cast<float>(idx0);
                                
                                float delayed = m_buffer[idx0] * (1.0f - frac) + m_buffer[idx1] * frac;
                                wet_sum += delayed;
                            }
                            
                            float wet = wet_sum / static_cast<float>(m_voices);
                            output(i, ch) = dry * (1.0f - m_mix) + wet * m_mix;
                            
                            // Write to buffer with feedback
                            m_buffer[write_pos] = dry + wet * m_feedback;
                            
                            write_pos = (write_pos + 1) % m_max_delay;
                        }
                    }
                    m_write_pos = (m_write_pos + num_samples) % m_max_delay;
                    m_phase += num_samples;
                    return output;
                }

                void reset() override
                {
                    std::fill(m_buffer.begin(), m_buffer.end(), 0.0f);
                    m_write_pos = 0;
                    m_phase = 0;
                }

                void set_parameter(const std::string& name, float value) override
                {
                    if (name == "rate") m_rate = value;
                    else if (name == "depth") m_depth = value;
                    else if (name == "feedback") m_feedback = std::clamp(value, 0.0f, 0.99f);
                    else if (name == "mix") m_mix = std::clamp(value, 0.0f, 1.0f);
                    else if (name == "voices") m_voices = std::max(1, static_cast<int>(value));
                    else if (name == "delay_ms") m_base_delay = static_cast<size_t>(value * 0.001f * 44100.0f);
                }

                float get_parameter(const std::string& name) const override
                {
                    if (name == "rate") return m_rate;
                    else if (name == "depth") return m_depth;
                    else if (name == "feedback") return m_feedback;
                    else if (name == "mix") return m_mix;
                    else if (name == "voices") return static_cast<float>(m_voices);
                    else if (name == "delay_ms") return static_cast<float>(m_base_delay) * 1000.0f / 44100.0f;
                    return 0.0f;
                }

                std::vector<std::string> get_parameter_names() const override
                {
                    return {"rate", "depth", "feedback", "mix", "voices", "delay_ms"};
                }

            private:
                std::vector<float> m_buffer;
                size_t m_max_delay;
                size_t m_base_delay = 882; // 20ms @ 44.1k
                size_t m_write_pos = 0;
                size_t m_phase = 0;
                float m_rate;
                float m_depth;
                float m_feedback;
                float m_mix;
                int m_voices;
            };

            // --------------------------------------------------------------------
            // Pitch Shifter Effect (time-domain PSOLA-like)
            // --------------------------------------------------------------------
            class XPitchShifterEffect : public XAudioEffectProcessor
            {
            public:
                XPitchShifterEffect()
                {
                    m_pitch_ratio = 1.0f;
                    m_window_size = 1024;
                    m_overlap = 4;
                }

                AudioBuffer process(const AudioBuffer& input, float sample_rate) override
                {
                    // Simplified: just a placeholder for pitch shifting
                    // Full implementation would use phase vocoder or PSOLA
                    return input;
                }

                void reset() override {}
                void set_parameter(const std::string& name, float value) override
                {
                    if (name == "pitch_ratio") m_pitch_ratio = std::clamp(value, 0.25f, 4.0f);
                    else if (name == "window_size") m_window_size = static_cast<size_t>(value);
                }

                float get_parameter(const std::string& name) const override
                {
                    if (name == "pitch_ratio") return m_pitch_ratio;
                    else if (name == "window_size") return static_cast<float>(m_window_size);
                    return 0.0f;
                }

                std::vector<std::string> get_parameter_names() const override
                {
                    return {"pitch_ratio", "window_size"};
                }

            private:
                float m_pitch_ratio;
                size_t m_window_size;
                size_t m_overlap;
            };

            // --------------------------------------------------------------------
            // Effect Chain (multiple effects in series)
            // --------------------------------------------------------------------
            class XEffectChain
            {
            public:
                void add_effect(std::unique_ptr<XAudioEffectProcessor> effect)
                {
                    m_effects.push_back(std::move(effect));
                }

                AudioBuffer process(const AudioBuffer& input, float sample_rate)
                {
                    AudioBuffer current = input;
                    for (auto& effect : m_effects)
                        current = effect->process(current, sample_rate);
                    return current;
                }

                void reset()
                {
                    for (auto& effect : m_effects)
                        effect->reset();
                }

                size_t size() const { return m_effects.size(); }
                XAudioEffectProcessor* get_effect(size_t idx) { return m_effects[idx].get(); }

            private:
                std::vector<std::unique_ptr<XAudioEffectProcessor>> m_effects;
            };

            // --------------------------------------------------------------------
            // XAudioEffectNode - Godot node for tensor-based audio effects
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XAudioEffectNode : public godot::Node
            {
                GDCLASS(XAudioEffectNode, godot::Node)

            public:
                enum EffectType
                {
                    EFFECT_DELAY = 0,
                    EFFECT_REVERB = 1,
                    EFFECT_COMPRESSOR = 2,
                    EFFECT_EQUALIZER = 3,
                    EFFECT_CHORUS = 4,
                    EFFECT_PITCH_SHIFTER = 5
                };

            private:
                godot::Ref<XTensorNode> m_input_tensor;
                godot::Ref<XTensorNode> m_output_tensor;
                std::unique_ptr<XEffectChain> m_chain;
                EffectType m_effect_type = EFFECT_REVERB;
                godot::Dictionary m_parameters;
                float m_sample_rate = 44100.0f;
                bool m_auto_process = false;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_input_tensor", "tensor"), &XAudioEffectNode::set_input_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_input_tensor"), &XAudioEffectNode::get_input_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_output_tensor", "tensor"), &XAudioEffectNode::set_output_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_output_tensor"), &XAudioEffectNode::get_output_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_effect_type", "type"), &XAudioEffectNode::set_effect_type);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_effect_type"), &XAudioEffectNode::get_effect_type);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_parameter", "name", "value"), &XAudioEffectNode::set_parameter);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_parameter", "name"), &XAudioEffectNode::get_parameter);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_sample_rate", "rate"), &XAudioEffectNode::set_sample_rate);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_sample_rate"), &XAudioEffectNode::get_sample_rate);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_auto_process", "enabled"), &XAudioEffectNode::set_auto_process);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_auto_process"), &XAudioEffectNode::get_auto_process);
                    godot::ClassDB::bind_method(godot::D_METHOD("process_audio"), &XAudioEffectNode::process_audio);
                    godot::ClassDB::bind_method(godot::D_METHOD("reset"), &XAudioEffectNode::reset);
                    godot::ClassDB::bind_method(godot::D_METHOD("add_effect_to_chain", "type", "params"), &XAudioEffectNode::add_effect_to_chain);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear_chain"), &XAudioEffectNode::clear_chain);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "input_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_input_tensor", "get_input_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "output_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_output_tensor", "get_output_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "effect_type", godot::PROPERTY_HINT_ENUM, "Delay,Reverb,Compressor,Equalizer,Chorus,PitchShifter"), "set_effect_type", "get_effect_type");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "sample_rate"), "set_sample_rate", "get_sample_rate");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "auto_process"), "set_auto_process", "get_auto_process");
                    
                    ADD_SIGNAL(godot::MethodInfo("audio_processed"));
                    
                    BIND_ENUM_CONSTANT(EFFECT_DELAY);
                    BIND_ENUM_CONSTANT(EFFECT_REVERB);
                    BIND_ENUM_CONSTANT(EFFECT_COMPRESSOR);
                    BIND_ENUM_CONSTANT(EFFECT_EQUALIZER);
                    BIND_ENUM_CONSTANT(EFFECT_CHORUS);
                    BIND_ENUM_CONSTANT(EFFECT_PITCH_SHIFTER);
                }

            public:
                XAudioEffectNode()
                {
                    m_chain = std::make_unique<XEffectChain>();
                }

                void _ready() override
                {
                    if (m_auto_process && m_input_tensor.is_valid())
                        process_audio();
                }

                void set_input_tensor(const godot::Ref<XTensorNode>& tensor) { m_input_tensor = tensor; }
                godot::Ref<XTensorNode> get_input_tensor() const { return m_input_tensor; }
                void set_output_tensor(const godot::Ref<XTensorNode>& tensor) { m_output_tensor = tensor; }
                godot::Ref<XTensorNode> get_output_tensor() const { return m_output_tensor; }
                void set_effect_type(EffectType type) { m_effect_type = type; }
                EffectType get_effect_type() const { return m_effect_type; }
                void set_sample_rate(float rate) { m_sample_rate = rate; }
                float get_sample_rate() const { return m_sample_rate; }
                void set_auto_process(bool enabled) { m_auto_process = enabled; }
                bool get_auto_process() const { return m_auto_process; }

                void set_parameter(const godot::String& name, float value)
                {
                    std::string n = name.utf8().get_data();
                    m_parameters[name] = value;
                    if (m_chain->size() == 0)
                    {
                        ensure_effect();
                    }
                    if (m_chain->size() > 0)
                    {
                        m_chain->get_effect(0)->set_parameter(n, value);
                    }
                }

                float get_parameter(const godot::String& name) const
                {
                    std::string n = name.utf8().get_data();
                    if (m_chain->size() > 0)
                        return m_chain->get_effect(0)->get_parameter(n);
                    return 0.0f;
                }

                void process_audio()
                {
                    if (!m_input_tensor.is_valid())
                    {
                        godot::UtilityFunctions::printerr("XAudioEffectNode: input tensor not set");
                        return;
                    }
                    
                    auto input = m_input_tensor->get_tensor_resource()->m_data.to_float_array();
                    if (input.dimension() != 2)
                    {
                        if (input.dimension() == 1)
                            input = xt::view(input, xt::all(), xt::newaxis());
                        else
                        {
                            godot::UtilityFunctions::printerr("XAudioEffectNode: input must be N x C");
                            return;
                        }
                    }
                    
                    ensure_effect();
                    
                    AudioBuffer output = m_chain->process(input, m_sample_rate);
                    
                    if (!m_output_tensor.is_valid())
                        m_output_tensor.instantiate();
                    m_output_tensor->set_data(XVariant::from_xarray(output.cast<double>()).variant());
                    
                    emit_signal("audio_processed");
                }

                void reset()
                {
                    m_chain->reset();
                }

                void add_effect_to_chain(EffectType type, const godot::Dictionary& params)
                {
                    std::unique_ptr<XAudioEffectProcessor> effect = create_effect(type);
                    if (effect)
                    {
                        godot::Array keys = params.keys();
                        for (int i = 0; i < keys.size(); ++i)
                        {
                            godot::String key = keys[i];
                            effect->set_parameter(key.utf8().get_data(), params[key]);
                        }
                        m_chain->add_effect(std::move(effect));
                    }
                }

                void clear_chain()
                {
                    m_chain = std::make_unique<XEffectChain>();
                }

            private:
                void ensure_effect()
                {
                    if (m_chain->size() == 0)
                    {
                        auto effect = create_effect(m_effect_type);
                        if (effect)
                        {
                            // Apply stored parameters
                            godot::Array keys = m_parameters.keys();
                            for (int i = 0; i < keys.size(); ++i)
                            {
                                godot::String key = keys[i];
                                effect->set_parameter(key.utf8().get_data(), m_parameters[key]);
                            }
                            m_chain->add_effect(std::move(effect));
                        }
                    }
                }

                std::unique_ptr<XAudioEffectProcessor> create_effect(EffectType type)
                {
                    switch (type)
                    {
                        case EFFECT_DELAY:
                            return std::make_unique<XDelayEffect>(1.0f, m_sample_rate);
                        case EFFECT_REVERB:
                            return std::make_unique<XReverbEffect>(m_sample_rate);
                        case EFFECT_COMPRESSOR:
                            return std::make_unique<XCompressorEffect>();
                        case EFFECT_EQUALIZER:
                            return std::make_unique<XEqualizerEffect>();
                        case EFFECT_CHORUS:
                            return std::make_unique<XChorusEffect>(m_sample_rate);
                        case EFFECT_PITCH_SHIFTER:
                            return std::make_unique<XPitchShifterEffect>();
                        default:
                            return nullptr;
                    }
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XAudioEffectsRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XAudioEffectNode>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::XAudioEffectProcessor;
        using godot_bridge::XDelayEffect;
        using godot_bridge::XReverbEffect;
        using godot_bridge::XCompressorEffect;
        using godot_bridge::XEqualizerEffect;
        using godot_bridge::XChorusEffect;
        using godot_bridge::XPitchShifterEffect;
        using godot_bridge::XEffectChain;
        using godot_bridge::XAudioEffectNode;
        using godot_bridge::XAudioEffectsRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XAUDIOEFFECTS_HPP

// godot/xaudioeffects.hpp