//File 0609 : xtensor-signal/xtensor-signal.hpp
//Top‑level include for xtensor‑signal: aggregates all signal processing modules (convolution, correlation, filtering, windows, spectrogram, STFT, resampling) with full C++17 SIMD support.
#ifndef XTENSOR_SIGNAL_HPP
#define XTENSOR_SIGNAL_HPP

#include "xtensor_signal_config.hpp"
#include "xsignal_common.hpp"
#include "xconvolve.hpp"
#include "xcorrelate.hpp"
#include "xfilter.hpp"
#include "xwindows.hpp"
#include "xspectrogram.hpp"
#include "xstft.hpp"
#include "xresample.hpp"

namespace xt {
namespace signal {

    // Re‑export commonly used functions and types
    using signal::convolve1d;
    using signal::convolve2d;
    using signal::correlate1d;
    using signal::correlate2d;
    using signal::firfilter;
    using signal::iirfilter;
    using signal::filtfilt;
    using signal::moving_average;
    using signal::medfilt1d;
    using signal::medfilt2d;
    using signal::ema_filter;
    using signal::hamming;
    using signal::hanning;
    using signal::blackman;
    using signal::bartlett;
    using signal::kaiser;
    using signal::flattop;
    using signal::rectangular;
    using signal::window;
    using signal::spectrogram;
    using signal::stft;
    using signal::istft;
    using signal::resample;
    using signal::decimate;
    using signal::upsample;
    using signal::rms;
    using signal::envelope;
    using signal::to_db;
    using signal::from_db;
    using signal::frequency_axis;
    using signal::time_axis;
    using signal::magnitude_to_db;
    using signal::db_to_magnitude;

} // namespace signal
} // namespace xt

#endif // XTENSOR_SIGNAL_HPP