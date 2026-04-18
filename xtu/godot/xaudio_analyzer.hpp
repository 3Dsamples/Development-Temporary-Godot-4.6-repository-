// include/xtu/godot/xaudio_analyzer.hpp
// xtensor-unified - Audio spectrum analyzer for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XAUDIO_ANALYZER_HPP
#define XTU_GODOT_XAUDIO_ANALYZER_HPP

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xaudioserver.hpp"
#include "xtu/signal/fft.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace audio {

// #############################################################################
// Forward declarations
// #############################################################################
class AudioStreamAnalyzer;
class AudioEffectSpectrumAnalyzer;
class AudioEffectCapture;
class AudioStreamGeneratorAnalyzer;

// #############################################################################
// Magnitude mode for spectrum analysis
// #############################################################################
enum class MagnitudeMode : uint8_t {
    MAGNITUDE_LINEAR = 0,
    MAGNITUDE_DECIBEL = 1,
    MAGNITUDE_NORMALIZED = 2
};

// #############################################################################
// Frequency band scaling
// #############################################################################
enum class FrequencyScale : uint8_t {
    SCALE_LINEAR = 0,
    SCALE_LOGARITHMIC = 1,
    SCALE_MEL = 2,
    SCALE_BARK = 3
};

// #############################################################################
// Spectrum analyzer data
// #############################################################################
struct SpectrumData {
    std::vector<float> magnitudes;      // Magnitude per frequency bin
    std::vector<float> frequencies;     // Center frequency per bin
    std::vector<float> phases;          // Phase per bin (if available)
    float sample_rate = 44100.0f;
    size_t fft_size = 1024;
    float time_since_last_update = 0.0f;
    bool valid = false;
};

// #############################################################################
// Beat detection result
// #############################################################################
struct BeatInfo {
    bool beat_detected = false;
    float bpm = 0.0f;
    float confidence = 0.0f;
    float beat_time = 0.0f;
    int beat_count = 0;
};

// #############################################################################
// AudioStreamAnalyzer - Analyzes audio streams
// #############################################################################
class AudioStreamAnalyzer : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(AudioStreamAnalyzer, RefCounted)

private:
    Ref<AudioStream> m_stream;
    Ref<AudioStreamPlayback> m_playback;
    std::vector<float> m_audio_buffer;
    SpectrumData m_spectrum;
    MagnitudeMode m_magnitude_mode = MagnitudeMode::MAGNITUDE_DECIBEL;
    FrequencyScale m_frequency_scale = FrequencyScale::SCALE_LOGARITHMIC;
    size_t m_fft_size = 2048;
    float m_hop_size = 0.5f;  // Overlap factor
    float m_analysis_time = 0.0f;
    mutable std::mutex m_mutex;
    
    // Beat detection
    std::vector<float> m_energy_history;
    float m_beat_threshold = 1.3f;
    float m_beat_min_interval = 0.3f;
    float m_last_beat_time = -1.0f;
    BeatInfo m_beat_info;

public:
    static StringName get_class_static() { return StringName("AudioStreamAnalyzer"); }

    void set_stream(const Ref<AudioStream>& stream) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stream = stream;
        if (stream.is_valid()) {
            m_playback = stream->instantiate_playback();
        } else {
            m_playback = Ref<AudioStreamPlayback>();
        }
        m_analysis_time = 0.0f;
        m_spectrum.valid = false;
    }

    Ref<AudioStream> get_stream() const { return m_stream; }

    void set_fft_size(size_t size) {
        size_t valid = 64;
        while (valid < size) valid <<= 1;
        m_fft_size = valid;
    }

    size_t get_fft_size() const { return m_fft_size; }

    void set_magnitude_mode(MagnitudeMode mode) { m_magnitude_mode = mode; }
    MagnitudeMode get_magnitude_mode() const { return m_magnitude_mode; }

    void set_frequency_scale(FrequencyScale scale) { m_frequency_scale = scale; }
    FrequencyScale get_frequency_scale() const { return m_frequency_scale; }

    SpectrumData get_spectrum() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_spectrum;
    }

    std::vector<float> get_magnitudes() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_spectrum.magnitudes;
    }

    std::vector<float> get_frequencies() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_spectrum.frequencies;
    }

    float get_magnitude_at_freq(float freq) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_spectrum.valid || m_spectrum.frequencies.empty()) return 0.0f;
        
        size_t idx = 0;
        while (idx < m_spectrum.frequencies.size() && m_spectrum.frequencies[idx] < freq) ++idx;
        if (idx == 0) return m_spectrum.magnitudes[0];
        if (idx >= m_spectrum.frequencies.size()) return m_spectrum.magnitudes.back();
        
        float t = (freq - m_spectrum.frequencies[idx - 1]) / 
                  (m_spectrum.frequencies[idx] - m_spectrum.frequencies[idx - 1]);
        return m_spectrum.magnitudes[idx - 1] * (1.0f - t) + m_spectrum.magnitudes[idx] * t;
    }

    float get_rms() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_audio_buffer.empty()) return 0.0f;
        float sum = 0.0f;
        for (float v : m_audio_buffer) sum += v * v;
        return std::sqrt(sum / m_audio_buffer.size());
    }

    float get_peak() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_audio_buffer.empty()) return 0.0f;
        float peak = 0.0f;
        for (float v : m_audio_buffer) peak = std::max(peak, std::abs(v));
        return peak;
    }

    BeatInfo get_beat_info() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_beat_info;
    }

    void set_beat_detection_enabled(bool enabled) { m_beat_detection_enabled = enabled; }
    void set_beat_threshold(float threshold) { m_beat_threshold = threshold; }

    void update(float delta) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_playback.is_valid()) return;
        
        m_analysis_time += delta;
        
        // Collect audio samples
        int channels = m_playback->get_channels();
        int mix_rate = m_playback->get_mix_rate();
        size_t samples_needed = m_fft_size;
        
        if (m_audio_buffer.size() < samples_needed) {
            std::vector<AudioFrame> frames(samples_needed - m_audio_buffer.size());
            int mixed = m_playback->mix(frames.data(), 1.0f, static_cast<int>(frames.size()));
            
            for (int i = 0; i < mixed; ++i) {
                // Mix to mono
                float mono = (frames[i].left + frames[i].right) * 0.5f;
                m_audio_buffer.push_back(mono);
            }
        }
        
        // Perform FFT when enough samples
        if (m_audio_buffer.size() >= m_fft_size) {
            perform_fft();
            
            // Slide window
            size_t hop_samples = static_cast<size_t>(m_fft_size * (1.0f - m_hop_size));
            m_audio_buffer.erase(m_audio_buffer.begin(), 
                                 m_audio_buffer.begin() + hop_samples);
        }
        
        // Beat detection
        if (m_beat_detection_enabled && m_spectrum.valid) {
            detect_beat(delta);
        }
    }

private:
    bool m_beat_detection_enabled = false;

    void perform_fft() {
        if (m_audio_buffer.size() < m_fft_size) return;
        
        // Prepare input (apply window function)
        std::vector<std::complex<float>> input(m_fft_size);
        for (size_t i = 0; i < m_fft_size; ++i) {
            // Hann window
            float window = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (m_fft_size - 1)));
            input[i] = std::complex<float>(m_audio_buffer[i] * window, 0.0f);
        }
        
        // Perform FFT
        auto output = signal::fft(input);
        
        // Compute magnitudes
        size_t half_size = m_fft_size / 2;
        m_spectrum.magnitudes.resize(half_size);
        m_spectrum.frequencies.resize(half_size);
        m_spectrum.sample_rate = 44100.0f;
        m_spectrum.fft_size = m_fft_size;
        
        float freq_resolution = m_spectrum.sample_rate / m_fft_size;
        
        for (size_t i = 0; i < half_size; ++i) {
            float real = output[i].real();
            float imag = output[i].imag();
            float mag = std::sqrt(real * real + imag * imag);
            
            if (m_magnitude_mode == MagnitudeMode::MAGNITUDE_DECIBEL) {
                mag = 20.0f * std::log10(std::max(mag, 1e-10f));
            }
            
            m_spectrum.magnitudes[i] = mag;
            m_spectrum.frequencies[i] = i * freq_resolution;
        }
        
        // Apply frequency scaling if needed
        if (m_frequency_scale == FrequencyScale::SCALE_LOGARITHMIC) {
            apply_log_scale();
        }
        
        m_spectrum.valid = true;
    }

    void apply_log_scale() {
        // Resample magnitudes to logarithmic scale (octave bands)
        size_t num_bands = 32;
        std::vector<float> log_mags(num_bands, 0.0f);
        std::vector<float> log_freqs(num_bands);
        
        float min_freq = 20.0f;
        float max_freq = m_spectrum.sample_rate / 2.0f;
        
        for (size_t i = 0; i < num_bands; ++i) {
            float t = static_cast<float>(i) / (num_bands - 1);
            log_freqs[i] = min_freq * std::pow(max_freq / min_freq, t);
        }
        
        for (size_t i = 0; i < m_spectrum.magnitudes.size(); ++i) {
            float freq = m_spectrum.frequencies[i];
            float mag = m_spectrum.magnitudes[i];
            
            // Find which band this belongs to
            for (size_t b = 0; b < num_bands; ++b) {
                float band_center = log_freqs[b];
                float band_width = (b == 0) ? log_freqs[1] - log_freqs[0] : 
                                               log_freqs[b] - log_freqs[b-1];
                if (std::abs(freq - band_center) < band_width) {
                    log_mags[b] = std::max(log_mags[b], mag);
                    break;
                }
            }
        }
        
        m_spectrum.magnitudes = log_mags;
        m_spectrum.frequencies = log_freqs;
    }

    void detect_beat(float delta) {
        if (!m_spectrum.valid) return;
        
        // Compute energy in bass frequencies (20-200 Hz)
        float energy = 0.0f;
        size_t count = 0;
        for (size_t i = 0; i < m_spectrum.frequencies.size(); ++i) {
            if (m_spectrum.frequencies[i] >= 20.0f && m_spectrum.frequencies[i] <= 200.0f) {
                energy += m_spectrum.magnitudes[i];
                ++count;
            }
        }
        if (count > 0) energy /= count;
        
        // Add to history
        m_energy_history.push_back(energy);
        if (m_energy_history.size() > 43) {  // ~1 second at 43 FFTs/sec
            m_energy_history.erase(m_energy_history.begin());
        }
        
        // Compute average energy
        if (m_energy_history.size() < 10) return;
        
        float avg_energy = 0.0f;
        for (float e : m_energy_history) avg_energy += e;
        avg_energy /= m_energy_history.size();
        
        // Detect beat
        bool beat = (energy > avg_energy * m_beat_threshold);
        float current_time = m_analysis_time;
        
        if (beat && (m_last_beat_time < 0 || current_time - m_last_beat_time > m_beat_min_interval)) {
            m_beat_info.beat_detected = true;
            m_beat_info.beat_time = current_time;
            m_beat_info.beat_count++;
            m_last_beat_time = current_time;
            
            // Estimate BPM
            if (m_beat_info.beat_count >= 4) {
                float avg_interval = current_time / (m_beat_info.beat_count - 1);
                if (avg_interval > 0) {
                    m_beat_info.bpm = 60.0f / avg_interval;
                }
            }
            m_beat_info.confidence = (energy / avg_energy - 1.0f) / (m_beat_threshold - 1.0f);
        } else {
            m_beat_info.beat_detected = false;
        }
    }
};

// #############################################################################
// AudioEffectSpectrumAnalyzer - Audio effect for real-time analysis
// #############################################################################

class AudioEffectSpectrumAnalyzer : public AudioEffect {
    XTU_GODOT_REGISTER_CLASS(AudioEffectSpectrumAnalyzer, AudioEffect)

public:
    enum FFTSize {
        FFT_SIZE_256 = 256,
        FFT_SIZE_512 = 512,
        FFT_SIZE_1024 = 1024,
        FFT_SIZE_2048 = 2048,
        FFT_SIZE_4096 = 4096
    };

private:
    FFTSize m_fft_size = FFT_SIZE_1024;
    MagnitudeMode m_magnitude_mode = MagnitudeMode::MAGNITUDE_DECIBEL;
    float m_buffer_length = 0.1f;
    float m_tap_back_pos = 0.1f;
    std::vector<float> m_magnitudes;
    std::vector<float> m_frequencies;
    std::vector<AudioFrame> m_buffer;
    size_t m_buffer_pos = 0;
    bool m_dirty = false;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("AudioEffectSpectrumAnalyzer"); }

    void set_fft_size(FFTSize size) { m_fft_size = size; m_dirty = true; }
    FFTSize get_fft_size() const { return m_fft_size; }

    void set_magnitude_mode(MagnitudeMode mode) { m_magnitude_mode = mode; }
    MagnitudeMode get_magnitude_mode() const { return m_magnitude_mode; }

    void set_buffer_length(float seconds) { m_buffer_length = seconds; m_dirty = true; }
    float get_buffer_length() const { return m_buffer_length; }

    void set_tap_back_pos(float pos) { m_tap_back_pos = pos; }
    float get_tap_back_pos() const { return m_tap_back_pos; }

    std::vector<float> get_magnitudes() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_magnitudes;
    }

    std::vector<float> get_frequencies() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_frequencies;
    }

    float get_magnitude_at_freq(float freq) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_frequencies.empty()) return 0.0f;
        
        size_t idx = 0;
        while (idx < m_frequencies.size() && m_frequencies[idx] < freq) ++idx;
        if (idx == 0) return m_magnitudes[0];
        if (idx >= m_frequencies.size()) return m_magnitudes.back();
        
        float t = (freq - m_frequencies[idx - 1]) / (m_frequencies[idx] - m_frequencies[idx - 1]);
        return m_magnitudes[idx - 1] * (1.0f - t) + m_magnitudes[idx] * t;
    }

    void process(AudioFrame* buffer, int frames, float sample_rate) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // Resize buffer if needed
        size_t target_size = static_cast<size_t>(m_buffer_length * sample_rate);
        if (m_buffer.size() != target_size) {
            m_buffer.resize(target_size);
            m_buffer_pos = 0;
        }
        
        // Copy frames to circular buffer
        for (int i = 0; i < frames; ++i) {
            m_buffer[m_buffer_pos] = buffer[i];
            m_buffer_pos = (m_buffer_pos + 1) % m_buffer.size();
        }
        
        // Perform FFT if enough data
        size_t fft_samples = static_cast<size_t>(m_fft_size);
        if (m_buffer.size() >= fft_samples) {
            perform_fft(sample_rate);
        }
    }

    void reset() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_buffer.clear();
        m_buffer_pos = 0;
        m_magnitudes.clear();
        m_frequencies.clear();
    }

private:
    void perform_fft(float sample_rate) {
        size_t fft_size = static_cast<size_t>(m_fft_size);
        
        // Extract samples from circular buffer
        std::vector<std::complex<float>> input(fft_size);
        size_t start_pos = (m_buffer_pos + m_buffer.size() - 
                           static_cast<size_t>(m_tap_back_pos * sample_rate)) % m_buffer.size();
        
        for (size_t i = 0; i < fft_size; ++i) {
            size_t idx = (start_pos + i) % m_buffer.size();
            float mono = (m_buffer[idx].left + m_buffer[idx].right) * 0.5f;
            
            // Hann window
            float window = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (fft_size - 1)));
            input[i] = std::complex<float>(mono * window, 0.0f);
        }
        
        // FFT
        auto output = signal::fft(input);
        
        // Compute magnitudes
        size_t half_size = fft_size / 2;
        m_magnitudes.resize(half_size);
        m_frequencies.resize(half_size);
        
        float freq_resolution = sample_rate / fft_size;
        
        for (size_t i = 0; i < half_size; ++i) {
            float real = output[i].real();
            float imag = output[i].imag();
            float mag = std::sqrt(real * real + imag * imag);
            
            if (m_magnitude_mode == MagnitudeMode::MAGNITUDE_DECIBEL) {
                mag = 20.0f * std::log10(std::max(mag, 1e-10f));
            }
            
            m_magnitudes[i] = mag;
            m_frequencies[i] = i * freq_resolution;
        }
    }
};

// #############################################################################
// AudioEffectCapture - Captures audio for external processing
// #############################################################################

class AudioEffectCapture : public AudioEffect {
    XTU_GODOT_REGISTER_CLASS(AudioEffectCapture, AudioEffect)

private:
    std::vector<AudioFrame> m_buffer;
    size_t m_buffer_size = 4096;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("AudioEffectCapture"); }

    void set_buffer_size(size_t size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_buffer_size = size;
        if (m_buffer.size() > size) {
            m_buffer.resize(size);
        }
    }

    size_t get_buffer_size() const { return m_buffer_size; }

    void process(AudioFrame* buffer, int frames, float sample_rate) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // Add to circular buffer
        for (int i = 0; i < frames; ++i) {
            if (m_buffer.size() >= m_buffer_size) {
                m_buffer.erase(m_buffer.begin());
            }
            m_buffer.push_back(buffer[i]);
        }
    }

    void reset() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_buffer.clear();
    }

    std::vector<AudioFrame> get_buffer() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_buffer;
    }

    std::vector<float> get_buffer_mono() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<float> result;
        result.reserve(m_buffer.size());
        for (const auto& frame : m_buffer) {
            result.push_back((frame.left + frame.right) * 0.5f);
        }
        return result;
    }

    std::vector<float> get_buffer_left() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<float> result;
        result.reserve(m_buffer.size());
        for (const auto& frame : m_buffer) {
            result.push_back(frame.left);
        }
        return result;
    }

    std::vector<float> get_buffer_right() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<float> result;
        result.reserve(m_buffer.size());
        for (const auto& frame : m_buffer) {
            result.push_back(frame.right);
        }
        return result;
    }

    size_t get_available_frames() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_buffer.size();
    }

    void clear_buffer() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_buffer.clear();
    }
};

// #############################################################################
// AudioStreamGeneratorAnalyzer - Combined generator and analyzer
// #############################################################################

class AudioStreamGeneratorAnalyzer : public AudioStreamGenerator {
    XTU_GODOT_REGISTER_CLASS(AudioStreamGeneratorAnalyzer, AudioStreamGenerator)

private:
    Ref<AudioStreamAnalyzer> m_analyzer;
    std::vector<AudioFrame> m_generated_buffer;

public:
    static StringName get_class_static() { return StringName("AudioStreamGeneratorAnalyzer"); }

    AudioStreamGeneratorAnalyzer() {
        m_analyzer.instance();
    }

    Ref<AudioStreamAnalyzer> get_analyzer() const { return m_analyzer; }

    void analyze_frame(const AudioFrame& frame) {
        m_generated_buffer.push_back(frame);
        if (m_generated_buffer.size() >= 2048) {
            // Feed to analyzer
            m_generated_buffer.clear();
        }
    }

    SpectrumData get_spectrum() const { return m_analyzer->get_spectrum(); }
    float get_rms() const { return m_analyzer->get_rms(); }
    float get_peak() const { return m_analyzer->get_peak(); }
};

} // namespace audio

// Bring into main namespace
using audio::AudioStreamAnalyzer;
using audio::AudioEffectSpectrumAnalyzer;
using audio::AudioEffectCapture;
using audio::AudioStreamGeneratorAnalyzer;
using audio::SpectrumData;
using audio::BeatInfo;
using audio::MagnitudeMode;
using audio::FrequencyScale;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XAUDIO_ANALYZER_HPP