//File 0505 : xtensor-io/xaudio.hpp
//Audio file reader/writer via libsndfile with SIMD‑accelerated sample conversion, multi‑channel handling, and seamless xtensor integration for real‑time DSP.
#ifndef XTENSOR_IO_XAUDIO_HPP
#define XTENSOR_IO_XAUDIO_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sndfile.hh>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xio_config.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xtensor_simd.hpp"

namespace xt {
namespace io {

    /**
     * @enum audio_normalization
     * @brief Behavior for sample conversion.
     */
    enum class audio_normalization : uint8_t {
        none,          // Keep raw integer values (may be > 1.0)
        peak,          // Divide by max representable value (e.g. 32768 for int16)
        rms            // Normalize by RMS power
    };

    namespace detail {
        /**
         * Compute the scaling factor for a given integer sample format.
         */
        template <class T>
        inline double sample_scale_factor() noexcept {
            if constexpr (std::is_same_v<T, int16_t>) return 1.0 / 32768.0;
            else if constexpr (std::is_same_v<T, int32_t>) return 1.0 / 2147483648.0;
            else if constexpr (std::is_same_v<T, uint8_t>) return 1.0 / 128.0;
            else if constexpr (std::is_floating_point_v<T>) return 1.0;
            else return 1.0;
        }

        /**
         * Convert a buffer of raw audio samples to double with SIMD.
         */
        template <class T>
        inline void convert_samples_to_double(const T* src, double* dst, std::size_t count,
                                               double scale = 1.0) {
            if constexpr (std::is_same_v<T, double>) {
                if (scale == 1.0) std::copy(src, src + count, dst);
                else for (std::size_t i = 0; i < count; ++i) dst[i] = src[i] * scale;
            } else if constexpr (std::is_same_v<T, float>) {
                if (scale == 1.0) for (std::size_t i = 0; i < count; ++i) dst[i] = static_cast<double>(src[i]);
                else for (std::size_t i = 0; i < count; ++i) dst[i] = static_cast<double>(src[i]) * scale;
            } else if constexpr (simd_enabled_v<double> && sizeof(T) <= 4) {
                using simd_type = xsimd::batch<double, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = count / simd_size;
                simd_type vscale(scale);
                for (std::size_t i = 0; i < vec_count; ++i) {
                    alignas(64) double buf[simd_size];
                    for (std::size_t k = 0; k < simd_size; ++k)
                        buf[k] = static_cast<double>(src[i * simd_size + k]);
                    simd_type v = simd_type::load_aligned(buf);
                    (v * vscale).store_unaligned(dst + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < count; ++i)
                    dst[i] = static_cast<double>(src[i]) * scale;
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    dst[i] = static_cast<double>(src[i]) * scale;
            }
        }

        /**
         * Convert double samples back to integer with optional clipping.
         */
        template <class T>
        inline void convert_double_to_samples(const double* src, T* dst, std::size_t count,
                                               double scale = 1.0) {
            if constexpr (std::is_same_v<T, double>) {
                if (scale == 1.0) std::copy(src, src + count, dst);
                else for (std::size_t i = 0; i < count; ++i) dst[i] = static_cast<T>(src[i] * scale);
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    double val = src[i] * scale;
                    if (val > 1.0) val = 1.0;
                    if (val < -1.0) val = -1.0;
                    dst[i] = static_cast<T>(val * static_cast<double>(std::numeric_limits<T>::max()));
                }
            }
        }
    }

    /**
     * Load an audio file (WAV, OGG, FLAC, etc.) via libsndfile.
     * Returns a pair of (sample_rate, xarray<double>) with shape (frames, channels).
     * @param filename Path to audio file.
     * @param norm Normalization mode.
     * @return Tuple of (samplerate, audio_data).
     */
    template <class T = double>
    inline auto load_audio(const std::string& filename,
                           audio_normalization norm = audio_normalization::peak) {
        SndfileHandle file(filename);
        if (!file || file.rawHandle() == nullptr)
            throw std::runtime_error("load_audio: " + std::string(file.strError()));

        std::size_t frames = static_cast<std::size_t>(file.frames());
        std::size_t channels = static_cast<std::size_t>(file.channels());
        int format = file.format();
        int major_format = format & SF_FORMAT_TYPEMASK;
        int subtype = format & SF_FORMAT_SUBMASK;

        // Determine input sample type
        xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({frames, channels});

        // Read raw data based on input format
        if (subtype == SF_FORMAT_PCM_16 || subtype == SF_FORMAT_PCM_U8) {
            std::vector<int16_t> raw(frames * channels);
            file.read(raw.data(), static_cast<sf_count_t>(raw.size()));
            double scale = (norm == audio_normalization::peak) ? detail::sample_scale_factor<int16_t>() : 1.0;
            detail::convert_samples_to_double(raw.data(), reinterpret_cast<double*>(result.data()),
                                               raw.size(), scale);
        } else if (subtype == SF_FORMAT_PCM_32) {
            std::vector<int32_t> raw(frames * channels);
            file.read(raw.data(), static_cast<sf_count_t>(raw.size()));
            double scale = (norm == audio_normalization::peak) ? detail::sample_scale_factor<int32_t>() : 1.0;
            detail::convert_samples_to_double(raw.data(), reinterpret_cast<double*>(result.data()),
                                               raw.size(), scale);
        } else if (subtype == SF_FORMAT_FLOAT || subtype == SF_FORMAT_DOUBLE) {
            std::vector<double> raw(frames * channels);
            file.read(raw.data(), static_cast<sf_count_t>(raw.size()));
            std::copy(raw.begin(), raw.end(), result.data());
        } else {
            // Generic: use double read
            std::vector<double> raw(frames * channels);
            file.read(raw.data(), static_cast<sf_count_t>(raw.size()));
            std::copy(raw.begin(), raw.end(), result.data());
        }

        return std::make_tuple(static_cast<int>(file.samplerate()), std::move(result));
    }

    /**
     * Save an xtensor expression as an audio file.
     * @param filename Output path (extension determines format).
     * @param data Expression containing audio samples (frames × channels).
     * @param samplerate Sample rate in Hz.
     * @param format libsndfile format flags (default: 16-bit PCM WAV).
     */
    template <class E>
    inline void dump_audio(const std::string& filename,
                           const xexpression<E>& data,
                           int samplerate,
                           int format = SF_FORMAT_WAV | SF_FORMAT_PCM_16) {
        auto&& de = xt::eval(data.derived_cast());
        auto shape = de.shape();
        if (shape.size() != 2)
            throw std::runtime_error("dump_audio: expected 2D array (frames × channels).");

        std::size_t frames = shape[0];
        std::size_t channels = shape[1];

        SndfileHandle file(filename, SFM_WRITE, format,
                           static_cast<int>(channels), samplerate);
        if (!file || file.rawHandle() == nullptr)
            throw std::runtime_error("dump_audio: " + std::string(file.strError()));

        // Convert double to appropriate output format
        int subtype = format & SF_FORMAT_SUBMASK;
        if (subtype == SF_FORMAT_PCM_16) {
            std::vector<int16_t> raw(frames * channels);
            detail::convert_double_to_samples(de.data(), raw.data(), raw.size(),
                                              32767.0);
            file.write(raw.data(), static_cast<sf_count_t>(raw.size()));
        } else if (subtype == SF_FORMAT_PCM_32) {
            std::vector<int32_t> raw(frames * channels);
            detail::convert_double_to_samples(de.data(), raw.data(), raw.size(),
                                              2147483647.0);
            file.write(raw.data(), static_cast<sf_count_t>(raw.size()));
        } else if (subtype == SF_FORMAT_FLOAT) {
            std::vector<float> raw(frames * channels);
            for (std::size_t i = 0; i < raw.size(); ++i)
                raw[i] = static_cast<float>(de.data()[i]);
            file.write(raw.data(), static_cast<sf_count_t>(raw.size()));
        } else {
            file.write(de.data(), static_cast<sf_count_t>(de.size()));
        }
    }

    /**
     * Query audio metadata without loading full file.
     * @param filename Path to audio file.
     * @return Tuple of (frames, channels, samplerate, format).
     */
    inline auto audio_info(const std::string& filename) {
        SndfileHandle file(filename);
        if (!file || file.rawHandle() == nullptr)
            throw std::runtime_error("audio_info: " + std::string(file.strError()));
        return std::make_tuple(
            static_cast<std::size_t>(file.frames()),
            static_cast<std::size_t>(file.channels()),
            static_cast<int>(file.samplerate()),
            file.format()
        );
    }

} // namespace io
} // namespace xt

#endif // XTENSOR_IO_XAUDIO_HPP