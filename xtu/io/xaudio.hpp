// io/xaudio.hpp

#ifndef XTENSOR_XAUDIO_HPP
#define XTENSOR_XAUDIO_HPP

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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <memory>
#include <map>
#include <functional>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace audio
        {
            // --------------------------------------------------------------------
            // Audio format enumeration
            // --------------------------------------------------------------------
            enum class AudioFormat
            {
                PCM_U8,          // Unsigned 8-bit
                PCM_S16,         // Signed 16-bit little-endian
                PCM_S24,         // Signed 24-bit little-endian (packed)
                PCM_S32,         // Signed 32-bit little-endian
                FLOAT32,         // 32-bit float
                FLOAT64          // 64-bit float
            };

            // --------------------------------------------------------------------
            // Audio signal class
            // --------------------------------------------------------------------
            class AudioSignal
            {
            public:
                using size_type = std::size_t;
                using value_type = float;  // Internal representation always float

                AudioSignal() = default;
                
                AudioSignal(size_type num_samples, size_type num_channels = 1, size_type sample_rate = 44100)
                    : m_sample_rate(sample_rate), m_num_channels(num_channels)
                {
                    m_data = xt::zeros<float>({num_samples, num_channels});
                }

                explicit AudioSignal(const xarray_container<float>& data, size_type sample_rate = 44100)
                    : m_data(data), m_sample_rate(sample_rate)
                {
                    if (data.dimension() == 1)
                    {
                        m_data = xt::view(data, xt::all(), xt::newaxis());
                        m_num_channels = 1;
                    }
                    else if (data.dimension() == 2)
                    {
                        m_num_channels = data.shape()[1];
                    }
                    else
                    {
                        XTENSOR_THROW(std::invalid_argument, "AudioSignal: data must be 1D or 2D");
                    }
                }

                // Accessors
                size_type num_samples() const { return m_data.shape()[0]; }
                size_type num_channels() const { return m_num_channels; }
                size_type sample_rate() const { return m_sample_rate; }
                double duration() const { return static_cast<double>(num_samples()) / m_sample_rate; }

                const xarray_container<float>& data() const { return m_data; }
                xarray_container<float>& data() { return m_data; }

                // Get single channel
                xarray_container<float> channel(size_type ch) const
                {
                    if (ch >= m_num_channels)
                        XTENSOR_THROW(std::out_of_range, "AudioSignal::channel: channel index out of range");
                    return xt::view(m_data, xt::all(), ch);
                }

                // Set sample rate
                void set_sample_rate(size_type sr) { m_sample_rate = sr; }

                // Normalize to [-1, 1] range
                void normalize()
                {
                    float max_val = xt::amax(xt::abs(m_data))();
                    if (max_val > 0)
                        m_data = m_data / max_val;
                }

                // Resample to new sample rate
                AudioSignal resample(size_type new_rate) const
                {
                    if (new_rate == m_sample_rate)
                        return *this;
                    
                    size_type num_out = static_cast<size_type>(std::ceil(num_samples() * static_cast<double>(new_rate) / m_sample_rate));
                    AudioSignal result(num_out, m_num_channels, new_rate);
                    
                    for (size_type ch = 0; ch < m_num_channels; ++ch)
                    {
                        auto in_ch = channel(ch);
                        // Use linear interpolation for resampling
                        double ratio = static_cast<double>(num_samples() - 1) / (num_out - 1);
                        for (size_type i = 0; i < num_out; ++i)
                        {
                            double pos = i * ratio;
                            size_type idx0 = static_cast<size_type>(std::floor(pos));
                            size_type idx1 = std::min(idx0 + 1, num_samples() - 1);
                            double frac = pos - idx0;
                            result.m_data(i, ch) = in_ch(idx0) * (1.0f - static_cast<float>(frac)) + 
                                                   in_ch(idx1) * static_cast<float>(frac);
                        }
                    }
                    return result;
                }

                // Change number of channels (mono <-> stereo)
                AudioSignal set_channels(size_type new_channels) const
                {
                    if (new_channels == m_num_channels)
                        return *this;
                    
                    AudioSignal result(num_samples(), new_channels, m_sample_rate);
                    if (new_channels == 1 && m_num_channels >= 1)
                    {
                        // Downmix to mono
                        for (size_type i = 0; i < num_samples(); ++i)
                        {
                            float sum = 0.0f;
                            for (size_type ch = 0; ch < m_num_channels; ++ch)
                                sum += m_data(i, ch);
                            result.m_data(i, 0) = sum / static_cast<float>(m_num_channels);
                        }
                    }
                    else if (m_num_channels == 1 && new_channels >= 2)
                    {
                        // Duplicate mono to all channels
                        for (size_type ch = 0; ch < new_channels; ++ch)
                            xt::view(result.m_data, xt::all(), ch) = xt::view(m_data, xt::all(), 0);
                    }
                    else
                    {
                        // Generic: copy up to min channels, zero others
                        size_type min_ch = std::min(m_num_channels, new_channels);
                        for (size_type ch = 0; ch < min_ch; ++ch)
                            xt::view(result.m_data, xt::all(), ch) = xt::view(m_data, xt::all(), ch);
                    }
                    return result;
                }

                // Concatenate with another audio signal
                AudioSignal concat(const AudioSignal& other) const
                {
                    if (m_sample_rate != other.m_sample_rate)
                        XTENSOR_THROW(std::invalid_argument, "AudioSignal::concat: sample rate mismatch");
                    
                    size_type new_channels = std::max(m_num_channels, other.m_num_channels);
                    AudioSignal a = this->set_channels(new_channels);
                    AudioSignal b = other.set_channels(new_channels);
                    
                    AudioSignal result(a.num_samples() + b.num_samples(), new_channels, m_sample_rate);
                    xt::view(result.m_data, xt::range(0, a.num_samples()), xt::all()) = a.m_data;
                    xt::view(result.m_data, xt::range(a.num_samples(), xt::placeholders::_), xt::all()) = b.m_data;
                    return result;
                }

                // Mix (add) another audio signal (aligned to start)
                AudioSignal mix(const AudioSignal& other, float gain = 1.0f) const
                {
                    if (m_sample_rate != other.m_sample_rate)
                        XTENSOR_THROW(std::invalid_argument, "AudioSignal::mix: sample rate mismatch");
                    
                    size_type new_channels = std::max(m_num_channels, other.m_num_channels);
                    AudioSignal a = this->set_channels(new_channels);
                    AudioSignal b = other.set_channels(new_channels);
                    
                    size_type max_len = std::max(a.num_samples(), b.num_samples());
                    AudioSignal result(max_len, new_channels, m_sample_rate);
                    
                    xt::view(result.m_data, xt::range(0, a.num_samples()), xt::all()) = a.m_data;
                    for (size_type i = 0; i < b.num_samples(); ++i)
                        for (size_type ch = 0; ch < new_channels; ++ch)
                            result.m_data(i, ch) += b.m_data(i, ch) * gain;
                    
                    return result;
                }

                // Apply gain
                AudioSignal operator*(float gain) const
                {
                    AudioSignal result(*this);
                    result.m_data = result.m_data * gain;
                    return result;
                }

                // Fade in/out
                AudioSignal fade_in(double duration_sec, const std::string& curve = "linear") const
                {
                    AudioSignal result(*this);
                    size_type fade_len = static_cast<size_type>(duration_sec * m_sample_rate);
                    fade_len = std::min(fade_len, num_samples());
                    
                    for (size_type i = 0; i < fade_len; ++i)
                    {
                        float gain;
                        double t = static_cast<double>(i) / fade_len;
                        if (curve == "linear")
                            gain = static_cast<float>(t);
                        else if (curve == "exponential")
                            gain = static_cast<float>(t * t);
                        else if (curve == "logarithmic")
                            gain = static_cast<float>(std::log10(1.0 + 9.0 * t));
                        else
                            gain = static_cast<float>(t);
                        
                        for (size_type ch = 0; ch < m_num_channels; ++ch)
                            result.m_data(i, ch) *= gain;
                    }
                    return result;
                }

                AudioSignal fade_out(double duration_sec, const std::string& curve = "linear") const
                {
                    AudioSignal result(*this);
                    size_type fade_len = static_cast<size_type>(duration_sec * m_sample_rate);
                    fade_len = std::min(fade_len, num_samples());
                    size_type start = num_samples() - fade_len;
                    
                    for (size_type i = 0; i < fade_len; ++i)
                    {
                        float gain;
                        double t = static_cast<double>(i) / fade_len;
                        if (curve == "linear")
                            gain = 1.0f - static_cast<float>(t);
                        else if (curve == "exponential")
                            gain = 1.0f - static_cast<float>(t * t);
                        else if (curve == "logarithmic")
                            gain = 1.0f - static_cast<float>(std::log10(1.0 + 9.0 * t));
                        else
                            gain = 1.0f - static_cast<float>(t);
                        
                        for (size_type ch = 0; ch < m_num_channels; ++ch)
                            result.m_data(start + i, ch) *= gain;
                    }
                    return result;
                }

                // Trim silence from beginning and end
                AudioSignal trim_silence(float threshold = 0.001f, double min_silence_duration = 0.1) const
                {
                    size_type min_silence_samples = static_cast<size_type>(min_silence_duration * m_sample_rate);
                    
                    // Find start (first sample above threshold)
                    size_type start = 0;
                    for (size_type i = 0; i < num_samples(); ++i)
                    {
                        bool is_silent = true;
                        for (size_type ch = 0; ch < m_num_channels; ++ch)
                            if (std::abs(m_data(i, ch)) > threshold)
                                { is_silent = false; break; }
                        if (!is_silent)
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