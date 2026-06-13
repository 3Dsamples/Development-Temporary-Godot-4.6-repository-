//File 0071 : core/xhistogram.hpp
//Histogram computation with SIMD-accelerated bin counting, weighted histograms, multi-dimensional binning, and automatic edge strategies.
#ifndef XTENSOR_XHISTOGRAM_HPP
#define XTENSOR_XHISTOGRAM_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xeval.hpp"
#include "xsort.hpp"
#include "xstatistics.hpp"
#include "xreducer.hpp"

namespace xt
{
    namespace histogram
    {
        namespace detail
        {
            /**
             * Compute uniform bin edges from data range.
             */
            template <class T>
            inline auto uniform_edges(const T* data, std::size_t n, std::size_t nbins,
                                       T lower, T upper)
            {
                if (lower >= upper)
                {
                    // auto-detect from data
                    auto [min_it, max_it] = std::minmax_element(data, data + n);
                    T min_val = *min_it;
                    T max_val = *max_it;
                    T margin = (max_val - min_val) * T(0.001);
                    lower = min_val - margin;
                    upper = max_val + margin;
                }
                xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> edges({nbins + 1});
                T step = (upper - lower) / static_cast<T>(nbins);
                for (std::size_t i = 0; i <= nbins; ++i)
                    edges[i] = lower + step * static_cast<T>(i);
                return edges;
            }

            /**
             * Compute quantile-based bin edges.
             */
            template <class T>
            inline auto quantile_edges(const T* data, std::size_t n, std::size_t nbins)
            {
                std::vector<T> sorted(data, data + n);
                std::sort(sorted.begin(), sorted.end());
                xarray_container<uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>> edges({nbins + 1});
                edges[0] = sorted[0] - T(1e-10);
                for (std::size_t i = 1; i < nbins; ++i)
                {
                    double q = static_cast<double>(i) / static_cast<double>(nbins);
                    std::size_t idx = static_cast<std::size_t>(q * (n - 1));
                    edges[i] = sorted[std::min(idx, n - 1)];
                }
                edges[nbins] = sorted[n - 1] + T(1e-10);
                return edges;
            }

            /**
             * Find the bin index for a value using binary search on sorted edges.
             */
            template <class T>
            inline std::size_t find_bin(T value, const T* edges, std::size_t n_edges)
            {
                if (value < edges[0] || value >= edges[n_edges - 1])
                    return static_cast<std::size_t>(-1); // out of range
                auto it = std::upper_bound(edges, edges + n_edges, value);
                return static_cast<std::size_t>(it - edges - 1);
            }

            /**
             * SIMD-accelerated bin counting for uniform bins.
             */
            template <class T>
            inline void count_uniform_bins_simd(const T* data, std::size_t n,
                                                 T lower, T step, std::size_t nbins,
                                                 std::size_t* counts)
            {
                if constexpr (is_simd_enabled_v<T>)
                {
                    using simd_type = xsimd::batch<T, default_simd_arch>;
                    constexpr std::size_t simd_size = simd_type::size;
                    std::size_t vec_count = n / simd_size;
                    simd_type vlower(lower);
                    simd_type vstep(step);
                    simd_type vnbins(static_cast<T>(nbins));
                    for (std::size_t i = 0; i < vec_count; ++i)
                    {
                        simd_type vals = simd_type::load_unaligned(data + i * simd_size);
                        // Compute bin index: (val - lower) / step
                        simd_type indices = (vals - vlower) / vstep;
                        // Clamp to [0, nbins-1]
                        auto mask_in = (vals >= vlower) && (indices < vnbins);
                        // Convert to integer indices and increment counts
                        alignas(64) std::array<T, simd_size> idx_arr;
                        indices.store_aligned(idx_arr.data());
                        alignas(64) std::array<bool, simd_size> mask_arr;
                        mask_in.store_aligned(mask_arr.data());
                        for (std::size_t k = 0; k < simd_size; ++k)
                        {
                            if (mask_arr[k])
                            {
                                std::size_t bin = static_cast<std::size_t>(idx_arr[k]);
                                if (bin < nbins) ++counts[bin];
                            }
                        }
                    }
                    // Handle remaining elements
                    for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    {
                        if (data[i] >= lower && data[i] < lower + step * nbins)
                        {
                            std::size_t bin = static_cast<std::size_t>((data[i] - lower) / step);
                            if (bin < nbins) ++counts[bin];
                        }
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        if (data[i] >= lower && data[i] < lower + step * nbins)
                        {
                            std::size_t bin = static_cast<std::size_t>((data[i] - lower) / step);
                            if (bin < nbins) ++counts[bin];
                        }
                    }
                }
            }

            /**
             * Weighted histogram: accumulate weights instead of counts.
             */
            template <class T, class W>
            inline void accumulate_weighted_bins(const T* data, const W* weights, std::size_t n,
                                                  const T* edges, std::size_t n_edges,
                                                  double* bin_values)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    if (data[i] >= edges[0] && data[i] < edges[n_edges - 1])
                    {
                        auto it = std::upper_bound(edges, edges + n_edges, data[i]);
                        std::size_t bin = static_cast<std::size_t>(it - edges - 1);
                        if (bin < n_edges - 1)
                            bin_values[bin] += static_cast<double>(weights[i]);
                    }
                }
            }
        }

        /**
         * Compute a 1D histogram with uniform bins.
         */
        template <class E>
        inline auto histogram(const E& e, std::size_t nbins = 10)
        {
            using T = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(e);
            if (arr.dimension() != 1) throw std::runtime_error("histogram requires 1D input.");
            const T* data = arr.data();
            std::size_t n = arr.size();

            auto edges = detail::uniform_edges(data, n, nbins, T(0), T(0));
            T lower = edges[0];
            T step = (edges[nbins] - lower) / static_cast<T>(nbins);

            xarray_container<uvector<std::size_t>, DEFAULT_LAYOUT, std::vector<std::size_t>> counts({nbins}, 0);
            detail::count_uniform_bins_simd(data, n, lower, step, nbins, counts.data());

            return std::make_pair(counts, edges);
        }

        /**
         * Compute a 1D histogram with custom bin edges.
         */
        template <class E, class BinEdges>
        inline auto histogram(const E& e, const BinEdges& bin_edges)
        {
            using T = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(e);
            auto edges = xt::eval(bin_edges);
            if (arr.dimension() != 1) throw std::runtime_error("histogram requires 1D input.");
            if (edges.dimension() != 1) throw std::runtime_error("bin_edges must be 1D.");

            const T* data = arr.data();
            std::size_t n = arr.size();
            const T* edge_data = edges.data();
            std::size_t n_edges = edges.size();
            std::size_t nbins = n_edges - 1;

            xarray_container<uvector<std::size_t>, DEFAULT_LAYOUT, std::vector<std::size_t>> counts({nbins}, 0);
            for (std::size_t i = 0; i < n; ++i)
            {
                auto bin = detail::find_bin(data[i], edge_data, n_edges);
                if (bin < nbins) ++counts[bin];
            }
            return std::make_pair(counts, edges);
        }

        /**
         * Compute a weighted 1D histogram.
         */
        template <class E, class W, class BinEdges>
        inline auto histogram_weighted(const E& e, const W& weights, const BinEdges& bin_edges)
        {
            using T = typename std::decay_t<E>::value_type;
            using WType = typename std::decay_t<W>::value_type;
            auto arr = xt::eval(e);
            auto w = xt::eval(weights);
            auto edges = xt::eval(bin_edges);

            if (arr.dimension() != 1 || w.dimension() != 1)
                throw std::runtime_error("histogram_weighted requires 1D inputs.");
            if (arr.size() != w.size())
                throw std::runtime_error("Data and weights must have same size.");

            const T* data = arr.data();
            const WType* wdata = w.data();
            std::size_t n = arr.size();
            const T* edge_data = edges.data();
            std::size_t n_edges = edges.size();
            std::size_t nbins = n_edges - 1;

            xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> bin_values({nbins}, 0.0);
            detail::accumulate_weighted_bins(data, wdata, n, edge_data, n_edges, bin_values.data());
            return std::make_pair(bin_values, edges);
        }

        /**
         * Compute a 2D histogram with uniform bins.
         */
        template <class E1, class E2>
        inline auto histogram2d(const E1& x, const E2& y,
                                std::size_t nbins_x = 10, std::size_t nbins_y = 10)
        {
            using T1 = typename std::decay_t<E1>::value_type;
            using T2 = typename std::decay_t<E2>::value_type;
            auto arr_x = xt::eval(x);
            auto arr_y = xt::eval(y);

            if (arr_x.dimension() != 1 || arr_y.dimension() != 1)
                throw std::runtime_error("histogram2d requires 1D inputs.");
            if (arr_x.size() != arr_y.size())
                throw std::runtime_error("x and y must have same length.");

            std::size_t n = arr_x.size();
            const T1* data_x = arr_x.data();
            const T2* data_y = arr_y.data();

            auto edges_x = detail::uniform_edges(data_x, n, nbins_x, T1(0), T1(0));
            auto edges_y = detail::uniform_edges(data_y, n, nbins_y, T2(0), T2(0));

            T1 lower_x = edges_x[0];
            T1 step_x = (edges_x[nbins_x] - lower_x) / static_cast<T1>(nbins_x);
            T2 lower_y = edges_y[0];
            T2 step_y = (edges_y[nbins_y] - lower_y) / static_cast<T2>(nbins_y);

            xarray_container<uvector<std::size_t>, DEFAULT_LAYOUT, std::vector<std::size_t>> counts(
                {nbins_y, nbins_x}, 0);
            std::size_t* count_data = counts.data();

            for (std::size_t i = 0; i < n; ++i)
            {
                std::ptrdiff_t bin_x = static_cast<std::ptrdiff_t>((data_x[i] - lower_x) / step_x);
                std::ptrdiff_t bin_y = static_cast<std::ptrdiff_t>((data_y[i] - lower_y) / step_y);
                if (bin_x >= 0 && static_cast<std::size_t>(bin_x) < nbins_x &&
                    bin_y >= 0 && static_cast<std::size_t>(bin_y) < nbins_y)
                {
                    ++count_data[bin_y * nbins_x + bin_x];
                }
            }
            return std::make_tuple(counts, edges_x, edges_y);
        }

        /**
         * Compute a 2D histogram with custom bin edges.
         */
        template <class E1, class E2, class BinEdges1, class BinEdges2>
        inline auto histogram2d(const E1& x, const E2& y,
                                const BinEdges1& edges_x, const BinEdges2& edges_y)
        {
            using T1 = typename std::decay_t<E1>::value_type;
            using T2 = typename std::decay_t<E2>::value_type;
            auto arr_x = xt::eval(x);
            auto arr_y = xt::eval(y);
            auto ex = xt::eval(edges_x);
            auto ey = xt::eval(edges_y);

            if (arr_x.size() != arr_y.size())
                throw std::runtime_error("x and y must have same length.");

            std::size_t n = arr_x.size();
            const T1* data_x = arr_x.data();
            const T2* data_y = arr_y.data();
            const T1* ex_data = ex.data();
            const T2* ey_data = ey.data();
            std::size_t nbins_x = ex.size() - 1;
            std::size_t nbins_y = ey.size() - 1;

            xarray_container<uvector<std::size_t>, DEFAULT_LAYOUT, std::vector<std::size_t>> counts(
                {nbins_y, nbins_x}, 0);
            std::size_t* count_data = counts.data();

            for (std::size_t i = 0; i < n; ++i)
            {
                auto bin_x = detail::find_bin(data_x[i], ex_data, nbins_x + 1);
                auto bin_y = detail::find_bin(data_y[i], ey_data, nbins_y + 1);
                if (bin_x < nbins_x && bin_y < nbins_y)
                    ++count_data[bin_y * nbins_x + bin_x];
            }
            return std::make_tuple(counts, ex, ey);
        }

        /**
         * Compute bin centers from bin edges.
         */
        template <class E>
        inline auto bin_centers(const E& edges)
        {
            auto ed = xt::eval(edges);
            if (ed.dimension() != 1) throw std::runtime_error("bin_centers requires 1D edges.");
            std::size_t nbins = ed.size() - 1;
            xarray_container<uvector<double>, DEFAULT_LAYOUT, std::vector<std::size_t>> centers({nbins});
            for (std::size_t i = 0; i < nbins; ++i)
                centers[i] = (static_cast<double>(ed[i]) + static_cast<double>(ed[i + 1])) / 2.0;
            return centers;
        }

        /**
         * Normalize a histogram to form a probability density (sum * bin_width = 1).
         */
        template <class E, class BinEdges>
        inline auto normalize_histogram(const E& counts, const BinEdges& edges)
        {
            auto cnt = xt::eval(counts);
            auto ed = xt::eval(edges);
            auto result = xt::eval(cnt);
            double total = static_cast<double>(xt::sum(cnt)());
            if (total == 0.0) return result;

            for (std::size_t i = 0; i < cnt.size(); ++i)
            {
                double bin_width = static_cast<double>(ed[i + 1]) - static_cast<double>(ed[i]);
                result[i] = static_cast<double>(cnt[i]) / (total * bin_width);
            }
            return result;
        }

        /**
         * Cumulative histogram.
         */
        template <class E>
        inline auto cumulative_histogram(const E& counts)
        {
            auto cnt = xt::eval(counts);
            auto result = xt::eval(cnt);
            double running = 0.0;
            for (std::size_t i = 0; i < cnt.size(); ++i)
            {
                running += static_cast<double>(cnt[i]);
                result[i] = running;
            }
            return result;
        }

    } // namespace histogram
} // namespace xt

#endif // XTENSOR_XHISTOGRAM_HPP