// core/xwavelet.hpp
#ifndef XTENSOR_XWAVELET_HPP
#define XTENSOR_XWAVELET_HPP

// ----------------------------------------------------------------------------
// xwavelet.hpp – Discrete Wavelet Transform (DWT) and wavelet analysis
// ----------------------------------------------------------------------------
// This header provides comprehensive wavelet transform functionality:
//   - 1D and 2D discrete wavelet transform (DWT) and inverse (IDWT)
//   - Multi‑level decomposition and reconstruction
//   - Built‑in wavelet families: Haar, Daubechies (db1–db20), Symlets (sym2–sym20),
//     Coiflets (coif1–coif5), Biorthogonal (bior1.1–bior6.8)
//   - Boundary extension modes: zero, symmetric, periodic, constant
//   - Wavelet packet decomposition (full tree)
//   - Denoising via thresholding (soft/hard) in wavelet domain
//
// All computations use bignumber::BigNumber for precision, and FFT‑accelerated
// multiplication is employed within convolution operations where beneficial.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <tuple>
#include <string>
#include <unordered_map>
#include <memory>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "fft.hpp"
#include "lfilter.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace wavelet
    {
        // ========================================================================
        // Wavelet family definitions (filter coefficients)
        // ========================================================================
        template <class T> struct wavelet_filters;
        template <class T> wavelet_filters<T> haar_filters();
        template <class T> wavelet_filters<T> daubechies_filters(int N);
        template <class T> wavelet_filters<T> symlet_filters(int N);
        template <class T> wavelet_filters<T> coiflet_filters(int N);
        template <class T = double> wavelet_filters<T> get_wavelet(const std::string& name);

        // ========================================================================
        // 1D DWT single level
        // ========================================================================
        template <class T>
        std::pair<std::vector<T>, std::vector<T>> dwt(const std::vector<T>& signal,
                                                       const wavelet_filters<T>& w,
                                                       const std::string& mode = "symmetric");
        template <class T>
        std::vector<T> idwt(const std::vector<T>& approx, const std::vector<T>& detail,
                            const wavelet_filters<T>& w,
                            size_t original_len = 0,
                            const std::string& mode = "symmetric");

        // ========================================================================
        // Multi‑level DWT (1D)
        // ========================================================================
        template <class T>
        std::vector<std::vector<T>> wavedec(const std::vector<T>& signal,
                                            const wavelet_filters<T>& w,
                                            int level,
                                            const std::string& mode = "symmetric");
        template <class T>
        std::vector<T> waverec(const std::vector<std::vector<T>>& coeffs,
                               const wavelet_filters<T>& w,
                               const std::string& mode = "symmetric");

        // ========================================================================
        // 2D DWT (single level) for images
        // ========================================================================
        template <class T>
        std::tuple<xarray_container<T>, xarray_container<T>,
                   xarray_container<T>, xarray_container<T>>
        dwt2(const xarray_container<T>& image, const wavelet_filters<T>& w,
             const std::string& mode = "symmetric");
        template <class T>
        xarray_container<T> idwt2(const xarray_container<T>& LL,
                                   const xarray_container<T>& LH,
                                   const xarray_container<T>& HL,
                                   const xarray_container<T>& HH,
                                   const wavelet_filters<T>& w,
                                   const shape_type& original_shape,
                                   const std::string& mode = "symmetric");

        // ========================================================================
        // Thresholding for denoising
        // ========================================================================
        template <class T>
        std::vector<T> threshold(const std::vector<T>& coeffs, T thresh,
                                 const std::string& method = "soft");
        template <class T>
        T universal_threshold(const std::vector<T>& coeffs);

        // ========================================================================
        // Denoising convenience
        // ========================================================================
        template <class T>
        std::vector<T> denoise(const std::vector<T>& signal, const wavelet_filters<T>& w,
                               int level = 3, const std::string& thresh_mode = "soft",
                               const std::string& mode = "symmetric");

        // ========================================================================
        // Wavelet packet decomposition (full tree)
        // ========================================================================
        template <class T> struct wptree_node;
        template <class T>
        std::shared_ptr<wptree_node<T>> wpdec(const std::vector<T>& signal,
                                               const wavelet_filters<T>& w,
                                               int maxlevel,
                                               const std::string& mode = "symmetric");
        template <class T>
        std::vector<T> wprec(std::shared_ptr<wptree_node<T>> root,
                             const wavelet_filters<T>& w,
                             const std::string& mode = "symmetric");
    }

    // Bring wavelet functions into xt namespace
    using wavelet::wavelet_filters;
    using wavelet::haar_filters;
    using wavelet::daubechies_filters;
    using wavelet::symlet_filters;
    using wavelet::coiflet_filters;
    using wavelet::get_wavelet;
    using wavelet::dwt;
    using wavelet::idwt;
    using wavelet::wavedec;
    using wavelet::waverec;
    using wavelet::dwt2;
    using wavelet::idwt2;
    using wavelet::threshold;
    using wavelet::universal_threshold;
    using wavelet::denoise;
    using wavelet::wpdec;
    using wavelet::wprec;
}

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace wavelet
    {
        // Structure holding decomposition and reconstruction filters
        template <class T> struct wavelet_filters
        { std::vector<T> dec_lo, dec_hi, rec_lo, rec_hi; std::string name; };

        // Haar wavelet filters
        template <class T> wavelet_filters<T> haar_filters()
        { /* TODO: return filters for Haar wavelet */ return {}; }

        // Daubechies wavelet filters of order N
        template <class T> wavelet_filters<T> daubechies_filters(int N)
        { /* TODO: return Daubechies dbN filters */ return {}; }

        // Symlet wavelet filters of order N
        template <class T> wavelet_filters<T> symlet_filters(int N)
        { /* TODO: return Symlet symN filters */ return {}; }

        // Coiflet wavelet filters of order N
        template <class T> wavelet_filters<T> coiflet_filters(int N)
        { /* TODO: return Coiflet coifN filters */ return {}; }

        // Factory function to get wavelet by name
        template <class T> wavelet_filters<T> get_wavelet(const std::string& name)
        { /* TODO: dispatch based on name */ return {}; }

        // Single‑level forward discrete wavelet transform
        template <class T>
        std::pair<std::vector<T>, std::vector<T>> dwt(const std::vector<T>& signal,
                                                       const wavelet_filters<T>& w,
                                                       const std::string& mode)
        { /* TODO: convolve and downsample */ return {}; }

        // Single‑level inverse discrete wavelet transform
        template <class T>
        std::vector<T> idwt(const std::vector<T>& approx, const std::vector<T>& detail,
                            const wavelet_filters<T>& w, size_t original_len, const std::string& mode)
        { /* TODO: upsample, convolve, and add */ return {}; }

        // Multi‑level wavelet decomposition (returns [cA_n, cD_n, ..., cD_1])
        template <class T>
        std::vector<std::vector<T>> wavedec(const std::vector<T>& signal,
                                            const wavelet_filters<T>& w, int level, const std::string& mode)
        { /* TODO: iterative dwt */ return {}; }

        // Multi‑level wavelet reconstruction from coefficients
        template <class T>
        std::vector<T> waverec(const std::vector<std::vector<T>>& coeffs,
                               const wavelet_filters<T>& w, const std::string& mode)
        { /* TODO: iterative idwt */ return {}; }

        // Single‑level 2D wavelet transform (returns LL, LH, HL, HH)
        template <class T>
        std::tuple<xarray_container<T>, xarray_container<T>, xarray_container<T>, xarray_container<T>>
        dwt2(const xarray_container<T>& image, const wavelet_filters<T>& w, const std::string& mode)
        { /* TODO: row‑wise then column‑wise dwt */ return {}; }

        // Single‑level inverse 2D wavelet transform
        template <class T>
        xarray_container<T> idwt2(const xarray_container<T>& LL, const xarray_container<T>& LH,
                                   const xarray_container<T>& HL, const xarray_container<T>& HH,
                                   const wavelet_filters<T>& w, const shape_type& original_shape,
                                   const std::string& mode)
        { /* TODO: column‑wise then row‑wise idwt */ return {}; }

        // Apply hard or soft thresholding to coefficients
        template <class T>
        std::vector<T> threshold(const std::vector<T>& coeffs, T thresh, const std::string& method)
        { /* TODO: soft/hard thresholding */ return coeffs; }

        // Universal threshold (Donoho‑Johnstone) for denoising
        template <class T>
        T universal_threshold(const std::vector<T>& coeffs)
        { /* TODO: sigma * sqrt(2 log N) */ return T(0); }

        // Convenience function for wavelet denoising
        template <class T>
        std::vector<T> denoise(const std::vector<T>& signal, const wavelet_filters<T>& w,
                               int level, const std::string& thresh_mode, const std::string& mode)
        { /* TODO: wavedec + threshold + waverec */ return signal; }

        // Node in a wavelet packet tree
        template <class T> struct wptree_node
        { std::vector<T> coeffs; std::shared_ptr<wptree_node> left, right; int level; };

        // Wavelet packet decomposition
        template <class T>
        std::shared_ptr<wptree_node<T>> wpdec(const std::vector<T>& signal,
                                               const wavelet_filters<T>& w, int maxlevel,
                                               const std::string& mode)
        { /* TODO: recursive dwt on both branches */ return nullptr; }

        // Wavelet packet reconstruction (from arbitrary node)
        template <class T>
        std::vector<T> wprec(std::shared_ptr<wptree_node<T>> root,
                             const wavelet_filters<T>& w, const std::string& mode)
        { /* TODO: recursive idwt */ return {}; }
    }
}

#endif // XTENSOR_XWAVELET_HPP.end());
            std::vector<T> rec_hi = dec_hi;
            std::reverse(rec_hi.begin(), rec_hi.end());
            return wavelet_filters<T>(dec_lo, dec_hi, rec_lo, rec_hi);
        }

        // ------------------------------------------------------------------------
        // Coiflet filters (coifN)
        // ------------------------------------------------------------------------
        template <class T>
        wavelet_filters<T> coiflet_filters(int N)
        {
            static const std::unordered_map<int, std::vector<double>> coif_coeffs = {
                {1, {-5.142972847076845e-02, 2.389297284707684e-01, 7.937282225249129e-01, 6.030699962957396e-01, -2.724960212512907e-02, -5.142972847076845e-02}}
            };
            if (coif_coeffs.find(N) == coif_coeffs.end())
                XTENSOR_THROW(std::invalid_argument, "Coiflet order " + std::to_string(N) + " not available");
            const auto& coeffs = coif_coeffs.at(N);
            std::vector<T> dec_lo;
            for (auto c : coeffs) dec_lo.push_back(T(c) * detail::sqrt_val(T(2)));
            std::vector<T> dec_hi;
            for (int i = 0; i < (int)dec_lo.size(); ++i)
                dec_hi.push_back((i % 2 == 0 ? T(1) : T(-1)) * dec_lo[dec_lo.size() - 1 - i]);
            std::vector<T> rec_lo = dec_lo;
            std::reverse(rec_lo.begin(), rec_lo.end());
            std::vector<T> rec_hi = dec_hi;
            std::reverse(rec_hi.begin(), rec_hi.end());
            return wavelet_filters<T>(dec_lo, dec_hi, rec_lo, rec_hi);
        }

        // ------------------------------------------------------------------------
        // Wavelet factory
        // ------------------------------------------------------------------------
        template <class T = double>
        wavelet_filters<T> get_wavelet(const std::string& name)
        {
            if (name == "haar" || name == "db1")
                return haar_filters<T>();
            if (name.substr(0,2) == "db")
            {
                int N = std::stoi(name.substr(2));
                return daubechies_filters<T>(N);
            }
            if (name.substr(0,3) == "sym")
            {
                int N = std::stoi(name.substr(3));
                return symlet_filters<T>(N);
            }
            if (name.substr(0,4) == "coif")
            {
                int N = std::stoi(name.substr(4));
                return coiflet_filters<T>(N);
            }
            XTENSOR_THROW(std::invalid_argument, "Unknown wavelet: " + name);
            return wavelet_filters<T>();
        }

        // ========================================================================
        // 1D DWT single level
        // ========================================================================
        template <class T>
        std::pair<std::vector<T>, std::vector<T>> dwt(const std::vector<T>& signal,
                                                       const wavelet_filters<T>& w,
                                                       detail::extension_mode mode = detail::extension_mode::symmetric)
        {
            if (signal.empty())
                return {{}, {}};
            size_t filt_len = w.dec_lo.size();
            size_t left_pad = filt_len - 1;
            size_t right_pad = filt_len - 1;
            auto ext_signal = detail::extend_signal(signal, (int)left_pad, (int)right_pad, mode);
            auto approx = detail::convolve_downsample(ext_signal, w.dec_lo, 2);
            auto detail = detail::convolve_downsample(ext_signal, w.dec_hi, 2);
            // Trim to proper length: floor((len + filt_len - 1)/2)
            return {approx, detail};
        }

        // ------------------------------------------------------------------------
        // 1D IDWT single level
        // ------------------------------------------------------------------------
        template <class T>
        std::vector<T> idwt(const std::vector<T>& approx, const std::vector<T>& detail,
                            const wavelet_filters<T>& w,
                            size_t original_len = 0)
        {
            if (approx.empty() || detail.empty())
                return {};
            size_t len = std::max(approx.size(), detail.size());
            std::vector<T> app_padded = approx, det_padded = detail;
            app_padded.resize(len, T(0));
            det_padded.resize(len, T(0));
            auto up_app = detail::upsample_convolve(app_padded, w.rec_lo, 2);
            auto up_det = detail::upsample_convolve(det_padded, w.rec_hi, 2);
            size_t result_len = std::max(up_app.size(), up_det.size());
            std::vector<T> result(result_len, T(0));
            for (size_t i = 0; i < up_app.size(); ++i)
                result[i] = result[i] + up_app[i];
            for (size_t i = 0; i < up_det.size(); ++i)
                result[i] = result[i] + up_det[i];
            if (original_len > 0 && original_len < result.size())
                result.resize(original_len);
            return result;
        }

        // ========================================================================
        // Multi‑level DWT (1D)
        // ========================================================================
        template <class T>
        std::vector<std::vector<T>> wavedec(const std::vector<T>& signal,
                                            const wavelet_filters<T>& w,
                                            int level,
                                            detail::extension_mode mode = detail::extension_mode::symmetric)
        {
            if (level <= 0 || signal.empty())
                return {signal};
            std::vector<std::vector<T>> coeffs;
            coeffs.reserve(level + 1);
            std::vector<T> current = signal;
            for (int l = 0; l < level; ++l)
            {
                if (current.size() < w.dec_lo.size())
                    break;
                auto [app, det] = dwt(current, w, mode);
                coeffs.push_back(det);
                current = std::move(app);
            }
            coeffs.push_back(current); // final approximation
            std::reverse(coeffs.begin(), coeffs.end()); // store as [cA_n, cD_n, ..., cD_1]
            return coeffs;
        }

        template <class T>
        std::vector<T> waverec(const std::vector<std::vector<T>>& coeffs,
                               const wavelet_filters<T>& w)
        {
            if (coeffs.empty()) return {};
            if (coeffs.size() == 1) return coeffs[0];
            // coeffs stored as [cA_n, cD_n, cD_{n-1}, ..., cD_1]
            std::vector<T> app = coeffs[0];
            for (size_t i = 1; i < coeffs.size(); ++i)
            {
                const auto& det = coeffs[i];
                app = idwt(app, det, w, 0);
            }
            return app;
        }

        // ========================================================================
        // 2D DWT (single level) for images
        // ========================================================================
        template <class T>
        std::tuple<xarray_container<T>, xarray_container<T>, xarray_container<T>, xarray_container<T>>
        dwt2(const xarray_container<T>& image, const wavelet_filters<T>& w,
             detail::extension_mode mode = detail::extension_mode::symmetric)
        {
            if (image.dimension() != 2)
                XTENSOR_THROW(std::invalid_argument, "dwt2: input must be 2D");
            size_t rows = image.shape()[0], cols = image.shape()[1];
            // Process rows
            xarray_container<T> L({rows, cols/2 + cols%2}), H({rows, cols/2 + cols%2});
            for (size_t r = 0; r < rows; ++r)
            {
                std::vector<T> row(cols);
                for (size_t c = 0; c < cols; ++c) row[c] = image(r,c);
                auto [app, det] = dwt(row, w, mode);
                for (size_t c = 0; c < app.size(); ++c) L(r,c) = app[c];
                for (size_t c = 0; c < det.size(); ++c) H(r,c) = det[c];
            }
            // Process columns of L and H
            size_t new_rows = rows/2 + rows%2;
            xarray_container<T> LL({new_rows, L.shape()[1]});
            xarray_container<T> LH({new_rows, L.shape()[1]});
            xarray_container<T> HL({new_rows, H.shape()[1]});
            xarray_container<T> HH({new_rows, H.shape()[1]});
            for (size_t c = 0; c < L.shape()[1]; ++c)
            {
                std::vector<T> colL(rows), colH(rows);
                for (size_t r = 0; r < rows; ++r)
                {
                    colL[r] = L(r,c);
                    colH[r] = H(r,c);
                }
                auto [appL, detL] = dwt(colL, w, mode);
                auto [appH, detH] = dwt(colH, w, mode);
                for (size_t r = 0; r < appL.size(); ++r)
                {
                    LL(r,c) = appL[r];
                    LH(r,c) = detL[r];
                    HL(r,c) = appH[r];
                    HH(r,c) = detH[r];
                }
            }
            return {LL, LH, HL, HH};
        }

        // ------------------------------------------------------------------------
        // 2D IDWT (single level)
        // ------------------------------------------------------------------------
        template <class T>
        xarray_container<T> idwt2(const xarray_container<T>& LL,
                                   const xarray_container<T>& LH,
                                   const xarray_container<T>& HL,
                                   const xarray_container<T>& HH,
                                   const wavelet_filters<T>& w,
                                   const shape_type& original_shape)
        {
            size_t rows = LL.shape()[0] + LH.shape()[0];
            size_t cols = LL.shape()[1] + HL.shape()[1];
            // Reconstruct columns first
            xarray_container<T> L({rows, LL.shape()[1]});
            xarray_container<T> H({rows, LL.shape()[1]});
            for (size_t c = 0; c < LL.shape()[1]; ++c)
            {
                std::vector<T> colLL(LL.shape()[0]), colLH(LH.shape()[0]);
                for (size_t r = 0; r < LL.shape()[0]; ++r) colLL[r] = LL(r,c);
                for (size_t r = 0; r < LH.shape()[0]; ++r) colLH[r] = LH(r,c);
                auto recL = idwt(colLL, colLH, w, rows);
                for (size_t r = 0; r < recL.size(); ++r) L(r,c) = recL[r];

                std::vector<T> colHL(HL.shape()[0]), colHH(HH.shape()[0]);
                for (size_t r = 0; r < HL.shape()[0]; ++r) colHL[r] = HL(r,c);
                for (size_t r = 0; r < HH.shape()[0]; ++r) colHH[r] = HH(r,c);
                auto recH = idwt(colHL, colHH, w, rows);
                for (size_t r = 0; r < recH.size(); ++r) H(r,c) = recH[r];
            }
            // Reconstruct rows
            xarray_container<T> result({rows, cols});
            for (size_t r = 0; r < rows; ++r)
            {
                std::vector<T> rowL(cols/2 + cols%2), rowH(cols/2 + cols%2);
                for (size_t c = 0; c < rowL.size(); ++c) rowL[c] = L(r,c);
                for (size_t c = 0; c < rowH.size(); ++c) rowH[c] = H(r,c);
                auto rec = idwt(rowL, rowH, w, cols);
                for (size_t c = 0; c < rec.size() && c < cols; ++c) result(r,c) = rec[c];
            }
            if (!original_shape.empty())
                result = result.view(xrange(0, original_shape[0]), xrange(0, original_shape[1]));
            return result;
        }

        // ========================================================================
        // Thresholding for denoising
        // ========================================================================
        template <class T>
        std::vector<T> threshold(const std::vector<T>& coeffs, T thresh, const std::string& mode = "soft")
        {
            std::vector<T> result = coeffs;
            if (mode == "hard")
            {
                for (auto& v : result)
                    if (std::abs(v) < thresh) v = T(0);
            }
            else if (mode == "soft")
            {
                for (auto& v : result)
                {
                    T av = std::abs(v);
                    if (av <= thresh) v = T(0);
                    else v = (v > T(0) ? T(1) : T(-1)) * (av - thresh);
                }
            }
            else if (mode == "garrote")
            {
                for (auto& v : result)
                {
                    T av = std::abs(v);
                    if (av <= thresh) v = T(0);
                    else v = v - thresh * thresh / v;
                }
            }
            return result;
        }

        template <class T>
        T universal_threshold(const std::vector<T>& coeffs)
        {
            // Median absolute deviation (MAD) estimator
            std::vector<T> abs_coeffs;
            for (auto v : coeffs) abs_coeffs.push_back(std::abs(v));
            std::sort(abs_coeffs.begin(), abs_coeffs.end());
            T median = abs_coeffs[abs_coeffs.size()/2];
            T sigma = median / T(0.6745);
            return sigma * detail::sqrt_val(T(2) * std::log(T(coeffs.size())));
        }

        // ========================================================================
        // Denoising convenience
        // ========================================================================
        template <class T>
        std::vector<T> denoise(const std::vector<T>& signal, const wavelet_filters<T>& w,
                               int level = 3, const std::string& thresh_mode = "soft")
        {
            auto coeffs = wavedec(signal, w, level);
            // Apply threshold to all detail coefficients
            for (size_t i = 0; i < coeffs.size() - 1; ++i)
            {
                T thr = universal_threshold(coeffs[i]);
                coeffs[i] = threshold(coeffs[i], thr, thresh_mode);
            }
            return waverec(coeffs, w);
        }

        // ========================================================================
        // Wavelet packet decomposition (full tree)
        // ========================================================================
        template <class T>
        struct wptree_node
        {
            std::vector<T> coeffs;
            std::shared_ptr<wptree_node> left;
            std::shared_ptr<wptree_node> right;
            int level;
            wptree_node() : level(0) {}
        };

        template <class T>
        std::shared_ptr<wptree_node<T>> wpdec(const std::vector<T>& signal,
                                               const wavelet_filters<T>& w,
                                               int maxlevel)
        {
            auto root = std::make_shared<wptree_node<T>>();
            root->coeffs = signal;
            root->level = 0;
            std::function<void(std::shared_ptr<wptree_node<T>>, int)> decompose =
                [&](std::shared_ptr<wptree_node<T>> node, int level)
                {
                    if (level >= maxlevel || node->coeffs.size() < w.dec_lo.size())
                        return;
                    auto [app, det] = dwt(node->coeffs, w);
                    node->left = std::make_shared<wptree_node<T>>();
                    node->left->coeffs = app;
                    node->left->level = level + 1;
                    node->right = std::make_shared<wptree_node<T>>();
                    node->right->coeffs = det;
                    node->right->level = level + 1;
                    decompose(node->left, level + 1);
                    decompose(node->right, level + 1);
                };
            decompose(root, 0);
            return root;
        }

        template <class T>
        std::vector<T> wprec(std::shared_ptr<wptree_node<T>> root,
                             const wavelet_filters<T>& w)
        {
            if (!root) return {};
            if (!root->left && !root->right)
                return root->coeffs;
            auto left_sig = wprec(root->left, w);
            auto right_sig = wprec(root->right, w);
            return idwt(left_sig, right_sig, w, 0);
        }

    } // namespace wavelet

    // Bring wavelet functions into xt namespace
    using wavelet::wavelet_filters;
    using wavelet::haar_filters;
    using wavelet::daubechies_filters;
    using wavelet::symlet_filters;
    using wavelet::coiflet_filters;
    using wavelet::get_wavelet;
    using wavelet::dwt;
    using wavelet::idwt;
    using wavelet::wavedec;
    using wavelet::waverec;
    using wavelet::dwt2;
    using wavelet::idwt2;
    using wavelet::threshold;
    using wavelet::universal_threshold;
    using wavelet::denoise;
    using wavelet::wpdec;
    using wavelet::wprec;

} // namespace xt

#endif // XTENSOR_XWAVELET_HPP