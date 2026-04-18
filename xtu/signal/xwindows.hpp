// signal/xwindows.hpp

#ifndef XTENSOR_XWINDOWS_HPP
#define XTENSOR_XWINDOWS_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../math/xstats.hpp"
#include "../math/xspecial.hpp"  // for bessel functions if needed

#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <functional>
#include <type_traits>
#include <complex>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace signal
        {
            // --------------------------------------------------------------------
            // Window function generator base
            // --------------------------------------------------------------------
            enum class WindowType
            {
                Boxcar,           // Rectangular
                Triang,           // Triangular
                Bartlett,         // Bartlett (triangular with zero endpoints)
                Hann,             // Hann (Hanning)
                Hamming,          // Hamming
                Blackman,         // Blackman
                BlackmanHarris,   // Blackman-Harris
                Nuttall,          // Nuttall
                BlackmanNuttall,  // Blackman-Nuttall
                FlatTop,          // Flat top
                Cosine,           // Cosine
                Gaussian,         // Gaussian
                Kaiser,           // Kaiser
                Chebyshev,        // Dolph-Chebyshev
                Exponential,      // Exponential
                Tukey,            // Tukey (tapered cosine)
                Bohman,           // Bohman
                Parzen,           // Parzen
                Lanczos,          // Lanczos
                Custom
            };

            namespace detail
            {
                // Modified Bessel function of first kind I0(x) - used for Kaiser window
                inline double bessel_i0(double x)
                {
                    double sum = 1.0;
                    double term = 1.0;
                    double x2 = x * x / 4.0;
                    for (int k = 1; k <= 50; ++k)
                    {
                        term *= x2 / (k * k);
                        sum += term;
                        if (term < 1e-15 * sum) break;
                    }
                    return sum;
                }

                // Chebyshev polynomial evaluation for Dolph-Chebyshev window
                inline double chebyshev_poly(int n, double x)
                {
                    if (std::abs(x) <= 1.0)
                        return std::cos(n * std::acos(x));
                    else
                        return std::cosh(n * std::acosh(x));
                }

                // Normalize window to have unit sum (for convolution) or unit energy
                template <class E>
                void normalize_window(xexpression<E>& win, const std::string& norm = "none")
                {
                    auto& w = win.derived_cast();
                    if (norm == "coherent_gain")
                    {
                        double mean_val = xt::mean(w)();
                        if (mean_val != 0)
                            w = w / mean_val;
                    }
                    else if (norm == "energy")
                    {
                        double rms = std::sqrt(xt::mean(w * w)());
                        if (rms != 0)
                            w = w / rms;
                    }
                    else if (norm == "peak")
                    {
                        double max_val = xt::amax(w)();
                        if (max_val != 0)
                            w = w / max_val;
                    }
                    // "none" does nothing
                }

                // Generate linear space vector
                inline xarray_container<double> linspace_vec(size_t n, bool symmetric)
                {
                    xarray_container<double> t({n});
                    if (symmetric)
                    {
                        for (size_t i = 0; i < n; ++i)
                            t(i) = static_cast<double>(i) - (static_cast<double>(n) - 1.0) / 2.0;
                        t = t * (2.0 / (static_cast<double>(n) - 1.0));
                    }
                    else
                    {
                        for (size_t i = 0; i < n; ++i)
                            t(i) = static_cast<double>(i) / static_cast<double>(n - 1);
                    }
                    return t;
                }
            }

            // --------------------------------------------------------------------
            // Window generation functions
            // --------------------------------------------------------------------

            // Rectangular (boxcar)
            inline xarray_container<double> boxcar(size_t n, bool sym = true)
            {
                return xt::ones<double>({n});
            }

            // Triangular
            inline xarray_container<double> triang(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                if (sym)
                {
                    double mid = (n - 1) / 2.0;
                    for (size_t i = 0; i < n; ++i)
                    {
                        w(i) = 1.0 - std::abs(static_cast<double>(i) - mid) / std::ceil(mid);
                    }
                }
                else
                {
                    double denom = (n % 2 == 0) ? n / 2.0 : (n + 1) / 2.0;
                    for (size_t i = 0; i < n; ++i)
                    {
                        if (i < n / 2)
                            w(i) = static_cast<double>(i + 1) / denom;
                        else
                            w(i) = static_cast<double>(n - i) / denom;
                    }
                }
                return w;
            }

            // Bartlett (triangular with zero endpoints)
            inline xarray_container<double> bartlett(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                if (sym)
                {
                    double L = static_cast<double>(n - 1);
                    for (size_t i = 0; i < n; ++i)
                    {
                        w(i) = 1.0 - std::abs(2.0 * i - L) / L;
                    }
                }
                else
                {
                    for (size_t i = 0; i < n; ++i)
                    {
                        w(i) = 1.0 - std::abs(2.0 * i - n + 1) / (n - 1);
                    }
                }
                return w;
            }

            // Hann (Hanning)
            inline xarray_container<double> hann(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                for (size_t i = 0; i < n; ++i)
                {
                    double a = 2.0 * M_PI * i / (sym ? (n - 1) : n);
                    w(i) = 0.5 - 0.5 * std::cos(a);
                }
                return w;
            }

            // Hamming
            inline xarray_container<double> hamming(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                for (size_t i = 0; i < n; ++i)
                {
                    double a = 2.0 * M_PI * i / (sym ? (n - 1) : n);
                    w(i) = 0.54 - 0.46 * std::cos(a);
                }
                return w;
            }

            // Blackman
            inline xarray_container<double> blackman(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                for (size_t i = 0; i < n; ++i)
                {
                    double a = 2.0 * M_PI * i / (sym ? (n - 1) : n);
                    w(i) = 0.42 - 0.5 * std::cos(a) + 0.08 * std::cos(2.0 * a);
                }
                return w;
            }

            // Blackman-Harris (4-term)
            inline xarray_container<double> blackmanharris(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                const double a0 = 0.35875;
                const double a1 = 0.48829;
                const double a2 = 0.14128;
                const double a3 = 0.01168;
                for (size_t i = 0; i < n; ++i)
                {
                    double a = 2.0 * M_PI * i / (sym ? (n - 1) : n);
                    w(i) = a0 - a1 * std::cos(a) + a2 * std::cos(2.0 * a) - a3 * std::cos(3.0 * a);
                }
                return w;
            }

            // Nuttall (4-term)
            inline xarray_container<double> nuttall(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                const double a0 = 0.3635819;
                const double a1 = 0.4891775;
                const double a2 = 0.1365995;
                const double a3 = 0.0106411;
                for (size_t i = 0; i < n; ++i)
                {
                    double a = 2.0 * M_PI * i / (sym ? (n - 1) : n);
                    w(i) = a0 - a1 * std::cos(a) + a2 * std::cos(2.0 * a) - a3 * std::cos(3.0 * a);
                }
                return w;
            }

            // Blackman-Nuttall
            inline xarray_container<double> blackmannuttall(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                const double a0 = 0.3635819;
                const double a1 = 0.4891775;
                const double a2 = 0.1365995;
                const double a3 = 0.0106411;
                for (size_t i = 0; i < n; ++i)
                {
                    double a = 2.0 * M_PI * i / (sym ? (n - 1) : n);
                    w(i) = a0 - a1 * std::cos(a) + a2 * std::cos(2.0 * a) - a3 * std::cos(3.0 * a);
                }
                return w;
            }

            // Flat top
            inline xarray_container<double> flattop(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                const double a0 = 0.21557895;
                const double a1 = 0.41663158;
                const double a2 = 0.277263158;
                const double a3 = 0.083578947;
                const double a4 = 0.006947368;
                for (size_t i = 0; i < n; ++i)
                {
                    double a = 2.0 * M_PI * i / (sym ? (n - 1) : n);
                    w(i) = a0 - a1 * std::cos(a) + a2 * std::cos(2.0 * a) - a3 * std::cos(3.0 * a) + a4 * std::cos(4.0 * a);
                }
                return w;
            }

            // Cosine
            inline xarray_container<double> cosine(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                for (size_t i = 0; i < n; ++i)
                {
                    double a = M_PI * i / (sym ? (n - 1) : n);
                    w(i) = std::sin(a);
                }
                return w;
            }

            // Gaussian
            inline xarray_container<double> gaussian(size_t n, double std_dev, bool sym = true)
            {
                xarray_container<double> w({n});
                double mid = sym ? (n - 1) / 2.0 : n / 2.0;
                for (size_t i = 0; i < n; ++i)
                {
                    double x = (static_cast<double>(i) - mid) / (std_dev * (sym ? (n - 1) / 2.0 : n / 2.0));
                    w(i) = std::exp(-0.5 * x * x);
                }
                return w;
            }

            // Kaiser
            inline xarray_container<double> kaiser(size_t n, double beta, bool sym = true)
            {
                xarray_container<double> w({n});
                double denom = detail::bessel_i0(beta);
                double mid = sym ? (n - 1) / 2.0 : n / 2.0;
                for (size_t i = 0; i < n; ++i)
                {
                    double x = 2.0 * static_cast<double>(i) / (n - 1) - 1.0;
                    double arg = beta * std::sqrt(1.0 - x * x);
                    w(i) = detail::bessel_i0(arg) / denom;
                }
                return w;
            }

            // Dolph-Chebyshev
            inline xarray_container<double> chebwin(size_t n, double at, bool sym = true)
            {
                // at = sidelobe attenuation in dB (e.g., 60)
                xarray_container<double> w({n});
                double alpha = std::cosh(std::acosh(std::pow(10.0, at / 20.0)) / (n - 1));
                double mid = sym ? (n - 1) / 2.0 : n / 2.0;
                for (size_t i = 0; i < n; ++i)
                {
                    double x = std::cos(M_PI * i / (n - 1));
                    w(i) = std::pow(-1.0, static_cast<double>(i)) * detail::chebyshev_poly(static_cast<int>(n - 1), alpha * x);
                }
                // Normalize to unit peak
                double max_val = std::abs(w(0));
                for (size_t i = 1; i < n; ++i)
                    max_val = std::max(max_val, std::abs(w(i)));
                w = w / max_val;
                return w;
            }

            // Exponential
            inline xarray_container<double> exponential(size_t n, double tau = 1.0, bool sym = true)
            {
                xarray_container<double> w({n});
                double mid = sym ? (n - 1) / 2.0 : n / 2.0;
                for (size_t i = 0; i < n; ++i)
                {
                    double x = std::abs(static_cast<double>(i) - mid) / mid;
                    w(i) = std::exp(-x / tau);
                }
                return w;
            }

            // Tukey (tapered cosine)
            inline xarray_container<double> tukey(size_t n, double alpha = 0.5, bool sym = true)
            {
                // alpha is the fraction of the window inside the cosine tapered region
                xarray_container<double> w({n});
                double L = static_cast<double>(n - 1);
                for (size_t i = 0; i < n; ++i)
                {
                    double x = static_cast<double>(i) / L;
                    if (x < alpha / 2.0)
                        w(i) = 0.5 * (1.0 + std::cos(2.0 * M_PI / alpha * (x - alpha / 2.0)));
                    else if (x > 1.0 - alpha / 2.0)
                        w(i) = 0.5 * (1.0 + std::cos(2.0 * M_PI / alpha * (x - 1.0 + alpha / 2.0)));
                    else
                        w(i) = 1.0;
                }
                return w;
            }

            // Bohman
            inline xarray_container<double> bohman(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                double L = static_cast<double>(n - 1);
                for (size_t i = 0; i < n; ++i)
                {
                    double x = std::abs(2.0 * i - L) / L;
                    w(i) = (1.0 - x) * std::cos(M_PI * x) + 1.0 / M_PI * std::sin(M_PI * x);
                }
                return w;
            }

            // Parzen
            inline xarray_container<double> parzen(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                double L = static_cast<double>(n - 1);
                for (size_t i = 0; i < n; ++i)
                {
                    double x = std::abs(2.0 * i - L) / L;
                    if (x <= 0.5)
                        w(i) = 1.0 - 6.0 * x * x * (1.0 - x);
                    else
                        w(i) = 2.0 * std::pow(1.0 - x, 3.0);
                }
                return w;
            }

            // Lanczos
            inline xarray_container<double> lanczos(size_t n, bool sym = true)
            {
                xarray_container<double> w({n});
                for (size_t i = 0; i < n; ++i)
                {
                    double x = 2.0 * i / (n - 1) - 1.0;
                    if (std::abs(x) < 1e-12)
                        w(i) = 1.0;
                    else
                        w(i) = std::sin(M_PI * x) / (M_PI * x);
                }
                return w;
            }

            // --------------------------------------------------------------------
            // Generic window function with name and parameters
            // --------------------------------------------------------------------
            inline xarray_container<double> get_window(const std::string& name, size_t n,
                                                       bool sym = true,
                                                       const std::vector<double>& params = {})
            {
                if (name == "boxcar" || name == "rectangular")
                    return boxcar(n, sym);
                else if (name == "triang" || name == "triangular")
                    return triang(n, sym);
                else if (name == "bartlett")
                    return bartlett(n, sym);
                else if (name == "hann" || name == "hanning")
                    return hann(n, sym);
                else if (name == "hamming")
                    return hamming(n, sym);
                else if (name == "blackman")
                    return blackman(n, sym);
                else if (name == "blackmanharris")
                    return blackmanharris(n, sym);
                else if (name == "nuttall")
                    return nuttall(n, sym);
                else if (name == "blackmannuttall")
                    return blackmannuttall(n, sym);
                else if (name == "flattop")
                    return flattop(n, sym);
                else if (name == "cosine")
                    return cosine(n, sym);
                else if (name == "gaussian")
                {
                    double std_dev = params.empty() ? 0.4 : params[0];
                    return gaussian(n, std_dev, sym);
                }
                else if (name == "kaiser")
                {
                    double beta = params.empty() ? 5.0 : params[0];
                    return kaiser(n, beta, sym);
                }
                else if (name == "chebwin" || name == "chebyshev")
                {
                    double at = params.empty() ? 60.0 : params[0];
                    return chebwin(n, at, sym);
                }
                else if (name == "exponential")
                {
                    double tau = params.empty() ? 1.0 : params[0];
                    return exponential(n, tau, sym);
                }
                else if (name == "tukey")
                {
                    double alpha = params.empty() ? 0.5 : params[0];
                    return tukey(n, alpha, sym);
                }
                else if (name == "bohman")
                    return bohman(n, sym);
                else if (name == "parzen")
                    return parzen(n, sym);
                else if (name == "lanczos")
                    return lanczos(n, sym);
                else
                {
                    XTENSOR_THROW(std::invalid_argument, "get_window: unknown window name '" + name + "'");
                    return xt::zeros<double>({n});
                }
            }

            // --------------------------------------------------------------------
            // Window utility functions
            // --------------------------------------------------------------------

            // Compute coherent gain of a window
            template <class E>
            inline double coherent_gain(const xexpression<E>& window)
            {
                const auto& w = window.derived_cast();
                return xt::mean(w)();
            }

            // Compute equivalent noise bandwidth (ENBW)
            template <class E>
            inline double enbw(const xexpression<E>& window)
            {
                const auto& w = window.derived_cast();
                double sum_w = xt::sum(w)();
                double sum_w2 = xt::sum(w * w)();
                return sum_w2 / (sum_w * sum_w) * w.size();
            }

            // Compute scalloping loss
            template <class E>
            inline double scalloping_loss(const xexpression<E>& window)
            {
                const auto& w = window.derived_cast();
                size_t n = w.size();
                // Evaluate frequency response at half-bin offset
                std::complex<double> sum(0,0);
                for (size_t i = 0; i < n; ++i)
                {
                    double phase = M_PI * i / n;
                    sum += std::complex<double>(w(i) * std::cos(phase), -w(i) * std::sin(phase));
                }
                double peak = std::abs(sum);
                // Compare to DC gain
                double dc_gain = xt::sum(w)();
                return -20.0 * std::log10(peak / dc_gain);
            }

            // Apply window to a signal (element-wise multiplication)
            template <class E1, class E2>
            inline auto apply_window(const xexpression<E1>& signal, const xexpression<E2>& window)
            {
                const auto& sig = signal.derived_cast();
                const auto& win = window.derived_cast();
                if (sig.dimension() != 1 || win.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "apply_window: both inputs must be 1-D");
                }
                if (sig.size() != win.size())
                {
                    XTENSOR_THROW(std::invalid_argument, "apply_window: signal and window must have same length");
                }
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                xarray_container<value_type> result({sig.size()});
                for (size_t i = 0; i < sig.size(); ++i)
                    result(i) = sig(i) * win(i);
                return result;
            }

            // Periodic window generation (for spectral analysis, symmetric=false equivalent to periodic)
            inline xarray_container<double> periodic_window(const std::string& name, size_t n,
                                                             const std::vector<double>& params = {})
            {
                return get_window(name, n, false, params);
            }

            // Symmetric window generation
            inline xarray_container<double> symmetric_window(const std::string& name, size_t n,
                                                              const std::vector<double>& params = {})
            {
                return get_window(name, n, true, params);
            }

            // --------------------------------------------------------------------
            // Windowed FFT utilities
            // --------------------------------------------------------------------
            template <class E>
            inline auto windowed_fft(const xexpression<E>& signal, const xexpression<E>& window)
            {
                auto windowed_signal = apply_window(signal, window);
                return fft(windowed_signal);
            }

            template <class E>
            inline auto periodogram(const xexpression<E>& signal, const std::string& window_name = "boxcar",
                                    size_t nfft = 0, const std::vector<double>& win_params = {})
            {
                const auto& sig = signal.derived_cast();
                size_t n = sig.size();
                if (nfft == 0) nfft = next_fast_len(n);
                auto win = get_window(window_name, n, false, win_params);
                auto win_norm = xt::sum(win * win)();
                if (win_norm > 0) win = win / std::sqrt(win_norm);
                auto win_sig = apply_window(sig, win);
                // Zero pad to nfft
                xarray_container<std::complex<double>> padded({nfft}, 0.0);
                for (size_t i = 0; i < n; ++i)
                    padded(i) = static_cast<double>(win_sig(i));
                auto spec = fft(padded);
                xarray_container<double> psd({spec.size()});
                for (size_t i = 0; i < spec.size(); ++i)
                    psd(i) = std::norm(spec(i));
                return psd;
            }

            // Welch's method for power spectral density
            template <class E>
            inline auto welch(const xexpression<E>& signal, size_t nperseg = 256,
                              size_t noverlap = 0, const std::string& window_name = "hann",
                              size_t nfft = 0, const std::vector<double>& win_params = {})
            {
                const auto& sig = signal.derived_cast();
                if (sig.dimension() != 1)
                {
                    XTENSOR_THROW(std::invalid_argument, "welch: signal must be 1-D");
                }
                size_t n = sig.size();
                if (noverlap >= nperseg)
                    XTENSOR_THROW(std::invalid_argument, "welch: noverlap must be less than nperseg");
                if (nfft == 0) nfft = next_fast_len(nperseg);

                size_t hop = nperseg - noverlap;
                size_t n_segments = (n - nperseg) / hop + 1;
                if (n_segments == 0) n_segments = 1;

                auto win = get_window(window_name, nperseg, false, win_params);
                double win_sum_sq = xt::sum(win * win)();
                double scale = 1.0 / (win_sum_sq * n_segments);

                xarray_container<double> psd({nfft / 2 + 1}, 0.0);
                for (size_t seg = 0; seg < n_segments; ++seg)
                {
                    size_t start = seg * hop;
                    xarray_container<double> segment({nperseg});
                    for (size_t i = 0; i < nperseg; ++i)
                        segment(i) = sig(start + i) * win(i);
                    // Zero pad
                    xarray_container<std::complex<double>> padded({nfft}, 0.0);
                    for (size_t i = 0; i < nperseg; ++i)
                        padded(i) = segment(i);
                    auto spec = fft(padded);
                    for (size_t i = 0; i < psd.size(); ++i)
                        psd(i) += std::norm(spec(i));
                }
                psd = psd * scale;
                return psd;
            }

            // --------------------------------------------------------------------
            // Window design by optimization (e.g., least squares)
            // --------------------------------------------------------------------
            inline xarray_container<double> firwin(size_t numtaps, double cutoff,
                                                   const std::string& window_name = "hamming",
                                                   const std::string& pass_zero = "lowpass",
                                                   double fs = 2.0, const std::vector<double>& win_params = {})
            {
                // Design FIR filter using window method
                if (numtaps % 2 == 0) numtaps++; // ensure odd for type I
                double nyq = fs / 2.0;
                double fc = cutoff / nyq;
                if (fc <= 0.0 || fc >= 1.0)
                    XTENSOR_THROW(std::invalid_argument, "firwin: cutoff must be between 0 and fs/2");

                size_t n = numtaps;
                int center = static_cast<int>((n - 1) / 2);
                xarray_container<double> h({n}, 0.0);
                if (pass_zero == "lowpass" || pass_zero == "highpass")
                {
                    for (size_t i = 0; i < n; ++i)
                    {
                        int m = static_cast<int>(i) - center;
                        if (m == 0)
                            h(i) = 2.0 * fc;
                        else
                            h(i) = std::sin(2.0 * M_PI * fc * m) / (M_PI * m);
                    }
                    if (pass_zero == "highpass")
                    {
                        for (size_t i = 0; i < n; ++i)
                            h(i) = -h(i);
                        h(center) += 1.0;
                    }
                }
                else if (pass_zero == "bandpass")
                {
                    double f1 = cutoff; // assuming cutoff is array, but simplified
                    double f2 = win_params.empty() ? 0.8 : win_params[0];
                    for (size_t i = 0; i < n; ++i)
                    {
                        int m = static_cast<int>(i) - center;
                        if (m == 0)
                            h(i) = 2.0 * (f2 - f1);
                        else
                            h(i) = (std::sin(2.0 * M_PI * f2 * m) - std::sin(2.0 * M_PI * f1 * m)) / (M_PI * m);
                    }
                }
                else
                {
                    XTENSOR_THROW(std::invalid_argument, "firwin: unsupported pass_zero type");
                }

                // Apply window
                auto win = get_window(window_name, n, true, win_params);
                h = h * win;
                return h;
            }

            // --------------------------------------------------------------------
            // Frequency response of a window
            // --------------------------------------------------------------------
            template <class E>
            inline auto freqz(const xexpression<E>& b, const xexpression<E>& a = {},
                              size_t worN = 512, bool whole = false, double fs = 2.0 * M_PI)
            {
                const auto& B = b.derived_cast();
                xarray_container<std::complex<double>> w({worN});
                xarray_container<std::complex<double>> h({worN});
                double step = (whole ? 2.0 * M_PI : M_PI) / (worN - 1);
                for (size_t i = 0; i < worN; ++i)
                {
                    double omega = i * step;
                    w(i) = omega * fs / (2.0 * M_PI);
                    std::complex<double> z = std::polar(1.0, -omega);
                    std::complex<double> num(0,0), den(0,0);
                    for (size_t j = 0; j < B.size(); ++j)
                        num += static_cast<double>(B(j)) * std::pow(z, static_cast<int>(j));
                    if (a.derived_cast().size() > 0)
                    {
                        const auto& A = a.derived_cast();
                        den = 1.0;
                        for (size_t j = 1; j < A.size(); ++j)
                            den += static_cast<double>(A(j)) * std::pow(z, static_cast<int>(j));
                    }
                    else
                    {
                        den = 1.0;
                    }
                    h(i) = num / den;
                }
                return std::make_pair(w, h);
            }

            // --------------------------------------------------------------------
            // Overlap-add method for applying window to long signals
            // --------------------------------------------------------------------
            template <class E>
            inline auto ola_filter(const xexpression<E>& signal, const xexpression<E>& fir_coeff,
                                   size_t block_size = 1024)
            {
                const auto& sig = signal.derived_cast();
                const auto& h = fir_coeff.derived_cast();
                if (sig.dimension() != 1 || h.dimension() != 1)
                    XTENSOR_THROW(std::invalid_argument, "ola_filter: inputs must be 1-D");

                size_t n_sig = sig.size();
                size_t n_h = h.size();
                size_t L = block_size;
                size_t N = next_fast_len(L + n_h - 1);
                size_t hop = L - n_h + 1;
                if (hop <= 0) hop = L / 2;

                xarray_container<double> y({n_sig + n_h - 1}, 0.0);
                std::vector<complex128> H(N, 0.0);
                for (size_t i = 0; i < n_h; ++i)
                    H[i] = static_cast<double>(h(i));
                detail::fft_core(H.data(), N, false);

                std::vector<complex128> X(N, 0.0);
                size_t pos = 0;
                while (pos < n_sig)
                {
                    size_t len = std::min(hop, n_sig - pos);
                    std::fill(X.begin(), X.end(), complex128(0,0));
                    for (size_t i = 0; i < len; ++i)
                        X[i] = sig(pos + i);
                    detail::fft_core(X.data(), N, false);
                    for (size_t i = 0; i < N; ++i)
                        X[i] *= H[i];
                    detail::fft_core(X.data(), N, true);
                    for (size_t i = 0; i < N; ++i)
                    {
                        if (pos + i < y.size())
                            y(pos + i) += std::real(X[i]);
                    }
                    pos += hop;
                }
                return y;
            }

        } // namespace signal

        // Bring window functions into xt namespace
        using signal::boxcar;
        using signal::triang;
        using signal::bartlett;
        using signal::hann;
        using signal::hamming;
        using signal::blackman;
        using signal::blackmanharris;
        using signal::nuttall;
        using signal::blackmannuttall;
        using signal::flattop;
        using signal::cosine;
        using signal::gaussian;
        using signal::kaiser;
        using signal::chebwin;
        using signal::exponential;
        using signal::tukey;
        using signal::bohman;
        using signal::parzen;
        using signal::lanczos;
        using signal::get_window;
        using signal::periodic_window;
        using signal::symmetric_window;
        using signal::coherent_gain;
        using signal::enbw;
        using signal::scalloping_loss;
        using signal::apply_window;
        using signal::windowed_fft;
        using signal::periodogram;
        using signal::welch;
        using signal::firwin;
        using signal::freqz;
        using signal::ola_filter;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XWINDOWS_HPP

// signal/xwindows.hpp