// io/xaudio.hpp
#ifndef XTENSOR_XAUDIO_HPP
#define XTENSOR_XAUDIO_HPP

// ----------------------------------------------------------------------------
// xaudio.hpp – Audio I/O and processing for xtensor
// ----------------------------------------------------------------------------
// This header provides audio loading, saving, playback, and processing:
//   - I/O: WAV, AIFF, FLAC, Ogg Vorbis (via stb_vorbis)
//   - Playback: platform‑independent audio output (PulseAudio, WASAPI, CoreAudio)
//   - Processing: resampling, mixing, normalization, fade, trim
//   - Effects: delay, reverb, chorus, flanger, EQ, compression
//   - Analysis: RMS, peak, spectrogram, MFCC (via FFT)
//   - Signal generation: sine, square, sawtooth, noise
//
// All sample types are supported, including bignumber::BigNumber for high
// precision. FFT is used for spectral analysis and convolution‑based effects.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <complex>

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "fft.hpp"
#include "io/xstb_vorbis.hpp"
#include "bignumber/bignumber.hpp"

namespace xt {
namespace audio {

// ========================================================================
// Audio buffer type (samples × channels)
// ========================================================================
template <class T>
using audio_buffer = xarray_container<T>;  // shape: (num_samples, num_channels)

// ========================================================================
// I/O: WAV format
// ========================================================================
template <class T>
audio_buffer<T> load_wav(const std::string& filename, int* sample_rate = nullptr);

template <class T>
void save_wav(const std::string& filename, const audio_buffer<T>& data,
              int sample_rate, int bits_per_sample = 16);

// ========================================================================
// I/O: FLAC, Ogg Vorbis (via stb_vorbis)
// ========================================================================
template <class T>
audio_buffer<T> load_vorbis(const std::string& filename, int* sample_rate = nullptr);

// Generic load (auto‑detect format)
template <class T>
audio_buffer<T> load_audio(const std::string& filename, int* sample_rate = nullptr);

template <class T>
void save_audio(const std::string& filename, const audio_buffer<T>& data,
                int sample_rate, const std::string& format = "wav");

// ========================================================================
// Playback
// ========================================================================
template <class T>
class audio_player {
public:
    audio_player(int sample_rate, int channels, int buffer_size = 4096);
    ~audio_player();

    void play(const audio_buffer<T>& data);
    void stop();
    void pause();
    void resume();
    bool is_playing() const;
    void set_volume(float volume);

private:
    // Platform‑specific implementation hidden
    void* m_impl;
};

// ========================================================================
// Processing: basic
// ========================================================================
template <class T>
audio_buffer<T> resample(const audio_buffer<T>& data, int src_rate, int dst_rate);

template <class T>
audio_buffer<T> mix_mono_to_stereo(const audio_buffer<T>& mono);

template <class T>
audio_buffer<T> mix_stereo_to_mono(const audio_buffer<T>& stereo);

template <class T>
audio_buffer<T> normalize(const audio_buffer<T>& data, T peak = T(1));

template <class T>
audio_buffer<T> fade_in(const audio_buffer<T>& data, double duration_sec, int sample_rate);

template <class T>
audio_buffer<T> fade_out(const audio_buffer<T>& data, double duration_sec, int sample_rate);

template <class T>
audio_buffer<T> trim(const audio_buffer<T>& data, double start_sec, double end_sec, int sample_rate);

// ========================================================================
// Effects
// ========================================================================
template <class T>
audio_buffer<T> delay(const audio_buffer<T>& data, double delay_sec, T feedback, T wet, int sample_rate);

template <class T>
audio_buffer<T> reverb(const audio_buffer<T>& data, double room_size, T damping, T wet, int sample_rate);

template <class T>
audio_buffer<T> chorus(const audio_buffer<T>& data, double rate, double depth, T wet, int sample_rate);

template <class T>
audio_buffer<T> flanger(const audio_buffer<T>& data, double rate, double depth, T feedback, T wet, int sample_rate);

template <class T>
audio_buffer<T> equalizer(const audio_buffer<T>& data, const std::vector<std::pair<double, T>>& bands, int sample_rate);

template <class T>
audio_buffer<T> compressor(const audio_buffer<T>& data, T threshold, T ratio, T attack, T release, int sample_rate);

// ========================================================================
// Convolution reverb (FFT‑accelerated)
// ========================================================================
template <class T>
audio_buffer<T> convolve_reverb(const audio_buffer<T>& data, const audio_buffer<T>& impulse_response);

// ========================================================================
// Analysis
// ========================================================================
template <class T>
T rms(const audio_buffer<T>& data);

template <class T>
T peak_amplitude(const audio_buffer<T>& data);

template <class T>
xarray_container<T> spectrogram(const audio_buffer<T>& data, int fft_size, int hop_size, int sample_rate);

template <class T>
xarray_container<T> mfcc(const audio_buffer<T>& data, int num_coeffs, int sample_rate,
                         int fft_size = 2048, int hop_size = 512);

// ========================================================================
// Signal generation
// ========================================================================
template <class T>
audio_buffer<T> generate_sine(double freq, double duration_sec, int sample_rate, T amplitude = T(1));

template <class T>
audio_buffer<T> generate_square(double freq, double duration_sec, int sample_rate, T amplitude = T(1));

template <class T>
audio_buffer<T> generate_sawtooth(double freq, double duration_sec, int sample_rate, T amplitude = T(1));

template <class T>
audio_buffer<T> generate_triangle(double freq, double duration_sec, int sample_rate, T amplitude = T(1));

template <class T>
audio_buffer<T> generate_noise(double duration_sec, int sample_rate, const std::string& color = "white");

template <class T>
audio_buffer<T> generate_silence(double duration_sec, int sample_rate);

} // namespace audio

using audio::audio_buffer;
using audio::load_wav;
using audio::save_wav;
using audio::load_vorbis;
using audio::load_audio;
using audio::save_audio;
using audio::audio_player;
using audio::resample;
using audio::normalize;
using audio::fade_in;
using audio::fade_out;
using audio::trim;
using audio::delay;
using audio::reverb;
using audio::chorus;
using audio::flanger;
using audio::equalizer;
using audio::compressor;
using audio::convolve_reverb;
using audio::rms;
using audio::peak_amplitude;
using audio::spectrogram;
using audio::mfcc;
using audio::generate_sine;
using audio::generate_square;
using audio::generate_sawtooth;
using audio::generate_triangle;
using audio::generate_noise;

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs
// ----------------------------------------------------------------------------
namespace xt {
namespace audio {

template <class T> audio_buffer<T> load_wav(const std::string& filename, int* sample_rate)
{ /* TODO: parse WAV header and PCM data */ return {}; }
template <class T> void save_wav(const std::string& filename, const audio_buffer<T>& data, int sample_rate, int bits_per_sample)
{ /* TODO: write WAV header and samples */ }

template <class T> audio_buffer<T> load_vorbis(const std::string& filename, int* sample_rate)
{ /* TODO: use stb_vorbis */ return {}; }
template <class T> audio_buffer<T> load_audio(const std::string& filename, int* sample_rate)
{ /* TODO: detect format by extension/magic */ return {}; }
template <class T> void save_audio(const std::string& filename, const audio_buffer<T>& data, int sample_rate, const std::string& format)
{ /* TODO: dispatch to appropriate saver */ }

template <class T> audio_player<T>::audio_player(int sample_rate, int channels, int buffer_size) { /* TODO: init platform audio */ }
template <class T> audio_player<T>::~audio_player() { /* TODO: cleanup */ }
template <class T> void audio_player<T>::play(const audio_buffer<T>& data) { /* TODO: start playback */ }
template <class T> void audio_player<T>::stop() { /* TODO: stop playback */ }
template <class T> void audio_player<T>::pause() { /* TODO: pause */ }
template <class T> void audio_player<T>::resume() { /* TODO: resume */ }
template <class T> bool audio_player<T>::is_playing() const { return false; }
template <class T> void audio_player<T>::set_volume(float volume) { /* TODO: adjust volume */ }

template <class T> audio_buffer<T> resample(const audio_buffer<T>& data, int src_rate, int dst_rate)
{ /* TODO: band‑limited interpolation */ return data; }
template <class T> audio_buffer<T> mix_mono_to_stereo(const audio_buffer<T>& mono)
{ /* TODO: duplicate channel */ return {}; }
template <class T> audio_buffer<T> mix_stereo_to_mono(const audio_buffer<T>& stereo)
{ /* TODO: average channels */ return {}; }
template <class T> audio_buffer<T> normalize(const audio_buffer<T>& data, T peak)
{ /* TODO: scale to peak */ return data; }
template <class T> audio_buffer<T> fade_in(const audio_buffer<T>& data, double duration_sec, int sample_rate)
{ /* TODO: apply linear/exponential fade */ return data; }
template <class T> audio_buffer<T> fade_out(const audio_buffer<T>& data, double duration_sec, int sample_rate)
{ /* TODO: apply linear/exponential fade */ return data; }
template <class T> audio_buffer<T> trim(const audio_buffer<T>& data, double start_sec, double end_sec, int sample_rate)
{ /* TODO: slice buffer */ return data; }

template <class T> audio_buffer<T> delay(const audio_buffer<T>& data, double delay_sec, T feedback, T wet, int sample_rate)
{ /* TODO: implement feedback delay line */ return data; }
template <class T> audio_buffer<T> reverb(const audio_buffer<T>& data, double room_size, T damping, T wet, int sample_rate)
{ /* TODO: Schroeder or FDN reverb */ return data; }
template <class T> audio_buffer<T> chorus(const audio_buffer<T>& data, double rate, double depth, T wet, int sample_rate)
{ /* TODO: modulated delay */ return data; }
template <class T> audio_buffer<T> flanger(const audio_buffer<T>& data, double rate, double depth, T feedback, T wet, int sample_rate)
{ /* TODO: flanger effect */ return data; }
template <class T> audio_buffer<T> equalizer(const audio_buffer<T>& data, const std::vector<std::pair<double, T>>& bands, int sample_rate)
{ /* TODO: cascade of biquad filters */ return data; }
template <class T> audio_buffer<T> compressor(const audio_buffer<T>& data, T threshold, T ratio, T attack, T release, int sample_rate)
{ /* TODO: dynamic range compression */ return data; }

template <class T> audio_buffer<T> convolve_reverb(const audio_buffer<T>& data, const audio_buffer<T>& impulse_response)
{ /* TODO: FFT‑based convolution */ return data; }

template <class T> T rms(const audio_buffer<T>& data)
{ /* TODO: sqrt(mean(sample^2)) */ return T(0); }
template <class T> T peak_amplitude(const audio_buffer<T>& data)
{ /* TODO: max absolute sample */ return T(0); }
template <class T> xarray_container<T> spectrogram(const audio_buffer<T>& data, int fft_size, int hop_size, int sample_rate)
{ /* TODO: STFT magnitude */ return {}; }
template <class T> xarray_container<T> mfcc(const audio_buffer<T>& data, int num_coeffs, int sample_rate, int fft_size, int hop_size)
{ /* TODO: mel‑filterbank + DCT */ return {}; }

template <class T> audio_buffer<T> generate_sine(double freq, double duration_sec, int sample_rate, T amplitude)
{ /* TODO: sin(2πf t) */ return {}; }
template <class T> audio_buffer<T> generate_square(double freq, double duration_sec, int sample_rate, T amplitude)
{ /* TODO: sign(sin(...)) */ return {}; }
template <class T> audio_buffer<T> generate_sawtooth(double freq, double duration_sec, int sample_rate, T amplitude)
{ /* TODO: 2*(t*f mod 1) - 1 */ return {}; }
template <class T> audio_buffer<T> generate_triangle(double freq, double duration_sec, int sample_rate, T amplitude)
{ /* TODO: 2*abs(saw) - 1 */ return {}; }
template <class T> audio_buffer<T> generate_noise(double duration_sec, int sample_rate, const std::string& color)
{ /* TODO: white/pink/brown noise */ return {}; }
template <class T> audio_buffer<T> generate_silence(double duration_sec, int sample_rate)
{ /* TODO: zero‑filled buffer */ return {}; }

} // namespace audio
} // namespace xt

#endif // XTENSOR_XAUDIO_HPPsilent)
                        {
                            start = (i >= min_silence_samples) ? i - min_silence_samples : 0;
                            break;
                        }
                    }
                    
                    // Find end (last sample above threshold)
                    size_type end = num_samples();
                    for (size_type i = num_samples(); i-- > 0; )
                    {
                        bool is_silent = true;
                        for (size_type ch = 0; ch < m_num_channels; ++ch)
                            if (std::abs(m_data(i, ch)) > threshold)
                                { is_silent = false; break; }
                        if (!is_silent)
                        {
                            end = std::min(i + min_silence_samples, num_samples());
                            break;
                        }
                    }
                    
                    if (start >= end)
                        return AudioSignal(1, m_num_channels, m_sample_rate); // empty
                    
                    AudioSignal result(end - start, m_num_channels, m_sample_rate);
                    xt::view(result.m_data, xt::all(), xt::all()) = xt::view(m_data, xt::range(start, end), xt::all());
                    return result;
                }

                // --------------------------------------------------------------------
                // Effects
                // --------------------------------------------------------------------
                
                // Delay effect
                AudioSignal delay(double delay_time, float feedback = 0.3f, float mix = 0.5f) const
                {
                    size_type delay_samples = static_cast<size_type>(delay_time * m_sample_rate);
                    AudioSignal result(*this);
                    
                    for (size_type i = delay_samples; i < num_samples(); ++i)
                    {
                        for (size_type ch = 0; ch < m_num_channels; ++ch)
                        {
                            float delayed = result.m_data(i - delay_samples, ch);
                            result.m_data(i, ch) = result.m_data(i, ch) * (1.0f - mix) + delayed * mix;
                            // Feedback
                            if (feedback > 0 && i + delay_samples < num_samples())
                                result.m_data(i + delay_samples, ch) += delayed * feedback;
                        }
                    }
                    return result;
                }

                // Basic reverb (convolution with exponential decay)
                AudioSignal reverb(double room_size = 0.5, double damping = 0.5, float mix = 0.3f) const
                {
                    // Simplified Schroeder reverb
                    AudioSignal result(*this);
                    std::vector<float> comb_delays = {0.0297f, 0.0371f, 0.0411f, 0.0437f};
                    std::vector<float> comb_gains = {0.8f, 0.8f, 0.8f, 0.8f};
                    std::vector<float> allpass_delays = {0.005f, 0.0017f};
                    float allpass_gain = 0.5f;
                    
                    // Scale delays by room size
                    for (auto& d : comb_delays) d *= static_cast<float>(room_size);
                    for (auto& d : allpass_delays) d *= static_cast<float>(room_size);
                    
                    // Apply comb filters (feedback delay)
                    for (size_type ch = 0; ch < m_num_channels; ++ch)
                    {
                        auto channel_data = xt::view(result.m_data, xt::all(), ch);
                        for (size_t c = 0; c < comb_delays.size(); ++c)
                        {
                            size_type delay = static_cast<size_type>(comb_delays[c] * m_sample_rate);
                            float gain = comb_gains[c] * (1.0f - static_cast<float>(damping));
                            xarray_container<float> delayed = xt::zeros_like(channel_data);
                            for (size_type i = delay; i < num_samples(); ++i)
                                delayed(i) = channel_data(i) + gain * delayed(i - delay);
                            channel_data = channel_data + delayed * mix;
                        }
                    }
                    return result;
                }

            private:
                xarray_container<float> m_data;
                size_type m_sample_rate = 44100;
                size_type m_num_channels = 1;
            };

            // --------------------------------------------------------------------
            // WAV file I/O
            // --------------------------------------------------------------------
            namespace detail
            {
                struct WavHeader
                {
                    // RIFF header
                    char chunk_id[4] = {'R', 'I', 'F', 'F'};
                    uint32_t chunk_size;
                    char format[4] = {'W', 'A', 'V', 'E'};
                    // fmt subchunk
                    char subchunk1_id[4] = {'f', 'm', 't', ' '};
                    uint32_t subchunk1_size = 16;
                    uint16_t audio_format = 1; // PCM
                    uint16_t num_channels;
                    uint32_t sample_rate;
                    uint32_t byte_rate;
                    uint16_t block_align;
                    uint16_t bits_per_sample;
                    // data subchunk
                    char subchunk2_id[4] = {'d', 'a', 't', 'a'};
                    uint32_t subchunk2_size;
                };

                template<typename T>
                void write_wav(std::ostream& out, const AudioSignal& signal, AudioFormat format)
                {
                    WavHeader hdr;
                    hdr.num_channels = static_cast<uint16_t>(signal.num_channels());
                    hdr.sample_rate = static_cast<uint32_t>(signal.sample_rate());
                    
                    size_t bytes_per_sample = 0;
                    bool is_float = false;
                    switch (format)
                    {
                        case AudioFormat::PCM_U8:
                            hdr.audio_format = 1;
                            hdr.bits_per_sample = 8;
                            bytes_per_sample = 1;
                            break;
                        case AudioFormat::PCM_S16:
                            hdr.audio_format = 1;
                            hdr.bits_per_sample = 16;
                            bytes_per_sample = 2;
                            break;
                        case AudioFormat::PCM_S24:
                            hdr.audio_format = 1;
                            hdr.bits_per_sample = 24;
                            bytes_per_sample = 3;
                            break;
                        case AudioFormat::PCM_S32:
                            hdr.audio_format = 1;
                            hdr.bits_per_sample = 32;
                            bytes_per_sample = 4;
                            break;
                        case AudioFormat::FLOAT32:
                            hdr.audio_format = 3; // IEEE_FLOAT
                            hdr.bits_per_sample = 32;
                            bytes_per_sample = 4;
                            is_float = true;
                            break;
                        case AudioFormat::FLOAT64:
                            hdr.audio_format = 3;
                            hdr.bits_per_sample = 64;
                            bytes_per_sample = 8;
                            is_float = true;
                            break;
                    }
                    
                    hdr.byte_rate = hdr.sample_rate * hdr.num_channels * static_cast<uint32_t>(bytes_per_sample);
                    hdr.block_align = hdr.num_channels * static_cast<uint16_t>(bytes_per_sample);
                    hdr.subchunk2_size = static_cast<uint32_t>(signal.num_samples() * hdr.block_align);
                    hdr.chunk_size = 36 + hdr.subchunk2_size;
                    
                    // Write header
                    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
                    
                    // Write data
                    const auto& data = signal.data();
                    for (size_t i = 0; i < signal.num_samples(); ++i)
                    {
                        for (size_t ch = 0; ch < signal.num_channels(); ++ch)
                        {
                            float val = data(i, ch);
                            if (!is_float)
                            {
                                // Clamp for integer formats
                                val = std::clamp(val, -1.0f, 1.0f);
                            }
                            
                            switch (format)
                            {
                                case AudioFormat::PCM_U8:
                                {
                                    uint8_t sample = static_cast<uint8_t>((val * 0.5f + 0.5f) * 255.0f);
                                    out.write(reinterpret_cast<const char*>(&sample), 1);
                                    break;
                                }
                                case AudioFormat::PCM_S16:
                                {
                                    int16_t sample = static_cast<int16_t>(val * 32767.0f);
                                    out.write(reinterpret_cast<const char*>(&sample), 2);
                                    break;
                                }
                                case AudioFormat::PCM_S24:
                                {
                                    int32_t sample = static_cast<int32_t>(val * 8388607.0f);
                                    out.write(reinterpret_cast<const char*>(&sample), 3);
                                    break;
                                }
                                case AudioFormat::PCM_S32:
                                {
                                    int32_t sample = static_cast<int32_t>(val * 2147483647.0f);
                                    out.write(reinterpret_cast<const char*>(&sample), 4);
                                    break;
                                }
                                case AudioFormat::FLOAT32:
                                {
                                    out.write(reinterpret_cast<const char*>(&val), 4);
                                    break;
                                }
                                case AudioFormat::FLOAT64:
                                {
                                    double dval = static_cast<double>(val);
                                    out.write(reinterpret_cast<const char*>(&dval), 8);
                                    break;
                                }
                            }
                        }
                    }
                }

                inline AudioSignal read_wav(std::istream& in)
                {
                    WavHeader hdr;
                    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
                    
                    if (std::memcmp(hdr.chunk_id, "RIFF", 4) != 0 ||
                        std::memcmp(hdr.format, "WAVE", 4) != 0)
                        XTENSOR_THROW(std::runtime_error, "Invalid WAV file");
                    
                    size_t num_channels = hdr.num_channels;
                    size_t sample_rate = hdr.sample_rate;
                    size_t bits = hdr.bits_per_sample;
                    bool is_float = (hdr.audio_format == 3);
                    
                    size_t bytes_per_sample = bits / 8;
                    size_t num_samples = hdr.subchunk2_size / (num_channels * bytes_per_sample);
                    
                    AudioSignal signal(num_samples, num_channels, sample_rate);
                    auto& data = signal.data();
                    
                    for (size_t i = 0; i < num_samples; ++i)
                    {
                        for (size_t ch = 0; ch < num_channels; ++ch)
                        {
                            float val = 0.0f;
                            if (is_float && bits == 32)
                            {
                                in.read(reinterpret_cast<char*>(&val), 4);
                            }
                            else if (is_float && bits == 64)
                            {
                                double dval;
                                in.read(reinterpret_cast<char*>(&dval), 8);
                                val = static_cast<float>(dval);
                            }
                            else if (bits == 8)
                            {
                                uint8_t sample;
                                in.read(reinterpret_cast<char*>(&sample), 1);
                                val = (sample / 255.0f) * 2.0f - 1.0f;
                            }
                            else if (bits == 16)
                            {
                                int16_t sample;
                                in.read(reinterpret_cast<char*>(&sample), 2);
                                val = sample / 32768.0f;
                            }
                            else if (bits == 24)
                            {
                                int32_t sample = 0;
                                in.read(reinterpret_cast<char*>(&sample), 3);
                                if (sample & 0x800000) sample |= 0xFF000000; // sign extend
                                val = sample / 8388608.0f;
                            }
                            else if (bits == 32)
                            {
                                int32_t sample;
                                in.read(reinterpret_cast<char*>(&sample), 4);
                                val = sample / 2147483648.0f;
                            }
                            data(i, ch) = val;
                        }
                    }
                    return signal;
                }
            }

            // Public I/O
            inline void write_wav(const std::string& filename, const AudioSignal& signal,
                                  AudioFormat format = AudioFormat::PCM_S16)
            {
                std::ofstream out(filename, std::ios::binary);
                if (!out)
                    XTENSOR_THROW(std::runtime_error, "Cannot open WAV file for writing: " + filename);
                detail::write_wav<float>(out, signal, format);
            }

            inline AudioSignal read_wav(const std::string& filename)
            {
                std::ifstream in(filename, std::ios::binary);
                if (!in)
                    XTENSOR_THROW(std::runtime_error, "Cannot open WAV file: " + filename);
                return detail::read_wav(in);
            }

            // --------------------------------------------------------------------
            // Spectrogram generation
            // --------------------------------------------------------------------
            inline xarray_container<float> spectrogram(const AudioSignal& signal,
                                                        size_t window_size = 1024,
                                                        size_t hop_size = 512,
                                                        const std::string& window_type = "hann")
            {
                size_t num_samples = signal.num_samples();
                size_t num_frames = (num_samples - window_size) / hop_size + 1;
                size_t num_freq_bins = window_size / 2 + 1;
                
                xarray_container<float> spec({num_frames, num_freq_bins}, 0.0f);
                auto window = signal::get_window(window_type, window_size, false);
                
                // Use first channel only for spectrogram
                auto channel_data = signal.channel(0);
                
                for (size_t frame = 0; frame < num_frames; ++frame)
                {
                    size_t start = frame * hop_size;
                    xarray_container<float> segment({window_size});
                    for (size_t i = 0; i < window_size; ++i)
                        segment(i) = channel_data(start + i) * window(i);
                    
                    auto fft_result = xt::fft::rfft(segment);
                    for (size_t f = 0; f < num_freq_bins; ++f)
                        spec(frame, f) = std::abs(fft_result(f));
                }
                return spec;
            }

            // Mel spectrogram
            inline xarray_container<float> mel_spectrogram(const AudioSignal& signal,
                                                            size_t n_mels = 128,
                                                            size_t window_size = 1024,
                                                            size_t hop_size = 512,
                                                            size_t sample_rate = 0,
                                                            float f_min = 0.0f,
                                                            float f_max = 0.0f)
            {
                if (sample_rate == 0) sample_rate = signal.sample_rate();
                if (f_max <= 0) f_max = sample_rate / 2.0f;
                
                auto spec = spectrogram(signal, window_size, hop_size);
                size_t n_frames = spec.shape()[0];
                size_t n_fft = (spec.shape()[1] - 1) * 2;
                
                // Create mel filterbank
                auto mel_filters = xarray_container<float>({n_mels, spec.shape()[1]}, 0.0f);
                
                auto hz_to_mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
                auto mel_to_hz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };
                
                float mel_min = hz_to_mel(f_min);
                float mel_max = hz_to_mel(f_max);
                std::vector<float> mel_points(n_mels + 2);
                for (size_t i = 0; i < n_mels + 2; ++i)
                    mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
                
                std::vector<size_t> fft_points(n_mels + 2);
                for (size_t i = 0; i < n_mels + 2; ++i)
                    fft_points[i] = static_cast<size_t>(std::floor((n_fft + 1) * mel_to_hz(mel_points[i]) / sample_rate));
                
                for (size_t m = 1; m <= n_mels; ++m)
                {
                    for (size_t k = fft_points[m-1]; k < fft_points[m]; ++k)
                        mel_filters(m-1, k) = (k - fft_points[m-1]) / static_cast<float>(fft_points[m] - fft_points[m-1]);
                    for (size_t k = fft_points[m]; k < fft_points[m+1]; ++k)
                        mel_filters(m-1, k) = (fft_points[m+1] - k) / static_cast<float>(fft_points[m+1] - fft_points[m]);
                }
                
                // Apply mel filterbank
                xarray_container<float> mel_spec({n_frames, n_mels}, 0.0f);
                for (size_t f = 0; f < n_frames; ++f)
                {
                    for (size_t m = 0; m < n_mels; ++m)
                    {
                        float sum = 0.0f;
                        for (size_t k = 0; k < spec.shape()[1]; ++k)
                            sum += spec(f, k) * mel_filters(m, k);
                        mel_spec(f, m) = sum;
                    }
                }
                return mel_spec;
            }

            // MFCC
            inline xarray_container<float> mfcc(const AudioSignal& signal,
                                                 size_t n_mfcc = 13,
                                                 size_t n_mels = 40,
                                                 size_t window_size = 1024,
                                                 size_t hop_size = 512)
            {
                auto mel_spec = mel_spectrogram(signal, n_mels, window_size, hop_size);
                // Log
                for (auto& v : mel_spec) v = std::log(v + 1e-6f);
                // DCT type II
                size_t n_frames = mel_spec.shape()[0];
                xarray_container<float> mfcc_result({n_frames, n_mfcc}, 0.0f);
                for (size_t f = 0; f < n_frames; ++f)
                {
                    for (size_t k = 0; k < n_mfcc; ++k)
                    {
                        float sum = 0.0f;
                        for (size_t n = 0; n < n_mels; ++n)
                            sum += mel_spec(f, n) * std::cos(M_PI * k * (n + 0.5f) / n_mels);
                        mfcc_result(f, k) = sum * std::sqrt(2.0f / n_mels);
                    }
                }
                return mfcc_result;
            }

            // --------------------------------------------------------------------
            // Audio processing utilities
            // --------------------------------------------------------------------
            
            // RMS energy
            inline float rms(const AudioSignal& signal)
            {
                float sum_sq = 0.0f;
                const auto& data = signal.data();
                for (size_t i = 0; i < data.size(); ++i)
                    sum_sq += data.flat(i) * data.flat(i);
                return std::sqrt(sum_sq / data.size());
            }

            // Peak amplitude
            inline float peak(const AudioSignal& signal)
            {
                return xt::amax(xt::abs(signal.data()))();
            }

            // Zero crossing rate
            inline float zero_crossing_rate(const AudioSignal& signal)
            {
                size_t crossings = 0;
                auto ch = signal.channel(0);
                for (size_t i = 1; i < signal.num_samples(); ++i)
                    if (ch(i-1) * ch(i) < 0)
                        crossings++;
                return static_cast<float>(crossings) / signal.num_samples();
            }

            // Generate sine wave
            inline AudioSignal sine_wave(double frequency, double duration_sec, size_t sample_rate = 44100, float amplitude = 0.5f)
            {
                size_t num_samples = static_cast<size_t>(duration_sec * sample_rate);
                AudioSignal result(num_samples, 1, sample_rate);
                auto& data = result.data();
                for (size_t i = 0; i < num_samples; ++i)
                {
                    double t = static_cast<double>(i) / sample_rate;
                    data(i, 0) = amplitude * static_cast<float>(std::sin(2.0 * M_PI * frequency * t));
                }
                return result;
            }

            // Generate white noise
            inline AudioSignal white_noise(double duration_sec, size_t sample_rate = 44100, float amplitude = 0.1f)
            {
                size_t num_samples = static_cast<size_t>(duration_sec * sample_rate);
                AudioSignal result(num_samples, 1, sample_rate);
                auto& data = result.data();
                std::mt19937 rng(std::random_device{}());
                std::uniform_real_distribution<float> dist(-amplitude, amplitude);
                for (size_t i = 0; i < num_samples; ++i)
                    data(i, 0) = dist(rng);
                return result;
            }

            // Generate pink noise (1/f)
            inline AudioSignal pink_noise(double duration_sec, size_t sample_rate = 44100, float amplitude = 0.1f)
            {
                size_t num_samples = static_cast<size_t>(duration_sec * sample_rate);
                AudioSignal result(num_samples, 1, sample_rate);
                auto& data = result.data();
                // Paul Kellet's method
                float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f, b4 = 0.0f, b5 = 0.0f, b6 = 0.0f;
                std::mt19937 rng(std::random_device{}());
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                for (size_t i = 0; i < num_samples; ++i)
                {
                    float white = dist(rng);
                    b0 = 0.99886f * b0 + white * 0.0555179f;
                    b1 = 0.99332f * b1 + white * 0.0750759f;
                    b2 = 0.96900f * b2 + white * 0.1538520f;
                    b3 = 0.86650f * b3 + white * 0.3104856f;
                    b4 = 0.55000f * b4 + white * 0.5329522f;
                    b5 = -0.7616f * b5 - white * 0.0168980f;
                    float pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362f;
                    b6 = white * 0.115926f;
                    data(i, 0) = pink * amplitude * 0.11f;
                }
                return result;
            }

        } // namespace audio

        // Bring audio classes into xt namespace
        using audio::AudioSignal;
        using audio::AudioFormat;
        using audio::read_wav;
        using audio::write_wav;
        using audio::spectrogram;
        using audio::mel_spectrogram;
        using audio::mfcc;
        using audio::rms;
        using audio::peak;
        using audio::zero_crossing_rate;
        using audio::sine_wave;
        using audio::white_noise;
        using audio::pink_noise;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XAUDIO_HPP

// io/xaudio.hpp