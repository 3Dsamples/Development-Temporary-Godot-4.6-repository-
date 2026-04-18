// include/xtu/metrics/xmetrics.hpp
// xtensor-unified - Distance metrics and pairwise distance computations
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_METRICS_XMETRICS_HPP
#define XTU_METRICS_XMETRICS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/containers/xarray.hpp"
#include "xtu/containers/xtensor.hpp"
#include "xtu/math/xreducer.hpp"
#include "xtu/math/xnorm.hpp"
#include "xtu/math/xsorting.hpp"

XTU_NAMESPACE_BEGIN
namespace metrics {

// #############################################################################
// Distance metric types
// #############################################################################
enum class distance_metric {
    euclidean,
    manhattan,
    chebyshev,
    minkowski,
    cosine,
    correlation,
    hamming,
    jaccard,
    canberra,
    braycurtis,
    mahalanobis,
    seuclidean,
    wminkowski,
    sqeuclidean
};

// #############################################################################
// Pairwise distance computation between two vectors
// #############################################################################

/// Euclidean distance (L2 norm)
template <class E1, class E2>
auto euclidean(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Euclidean distance requires 1D vectors");
    XTU_ASSERT_MSG(a.size() == b.size(), "Vectors must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    value_type sum_sq = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        value_type diff = static_cast<value_type>(a[i]) - static_cast<value_type>(b[i]);
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

/// Squared Euclidean distance
template <class E1, class E2>
auto sqeuclidean(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Squared Euclidean distance requires 1D vectors");
    XTU_ASSERT_MSG(a.size() == b.size(), "Vectors must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    value_type sum_sq = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        value_type diff = static_cast<value_type>(a[i]) - static_cast<value_type>(b[i]);
        sum_sq += diff * diff;
    }
    return sum_sq;
}

/// Manhattan distance (L1 norm)
template <class E1, class E2>
auto manhattan(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Manhattan distance requires 1D vectors");
    XTU_ASSERT_MSG(a.size() == b.size(), "Vectors must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    value_type sum_abs = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum_abs += std::abs(static_cast<value_type>(a[i]) - static_cast<value_type>(b[i]));
    }
    return sum_abs;
}

/// Chebyshev distance (L∞ norm)
template <class E1, class E2>
auto chebyshev(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Chebyshev distance requires 1D vectors");
    XTU_ASSERT_MSG(a.size() == b.size(), "Vectors must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    value_type max_abs = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        value_type abs_diff = std::abs(static_cast<value_type>(a[i]) - static_cast<value_type>(b[i]));
        if (abs_diff > max_abs) max_abs = abs_diff;
    }
    return max_abs;
}

/// Minkowski distance (generalized Lp norm)
template <class E1, class E2>
auto minkowski(const xexpression<E1>& x, const xexpression<E2>& y, double p) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Minkowski distance requires 1D vectors");
    XTU_ASSERT_MSG(a.size() == b.size(), "Vectors must have same size");
    XTU_ASSERT_MSG(p >= 1.0, "p must be >= 1");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    if (std::isinf(p)) {
        return chebyshev(x, y);
    }
    value_type sum_pow = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum_pow += std::pow(std::abs(static_cast<value_type>(a[i]) - static_cast<value_type>(b[i])), p);
    }
    return std::pow(sum_pow, 1.0 / p);
}

/// Cosine distance (1 - cosine similarity)
template <class E1, class E2>
auto cosine(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Cosine distance requires 1D vectors");
    XTU_ASSERT_MSG(a.size() == b.size(), "Vectors must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    value_type dot = 0, norm_a = 0, norm_b = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        value_type va = static_cast<value_type>(a[i]);
        value_type vb = static_cast<value_type>(b[i]);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    if (norm_a == 0 || norm_b == 0) {
        return std::numeric_limits<value_type>::quiet_NaN();
    }
    return value_type(1) - dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

/// Cosine similarity (1 - cosine distance)
template <class E1, class E2>
auto cosine_similarity(const xexpression<E1>& x, const xexpression<E2>& y) {
    return 1.0 - cosine(x, y);
}

/// Correlation distance (1 - Pearson correlation)
template <class E1, class E2>
auto correlation(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Correlation distance requires 1D vectors");
    XTU_ASSERT_MSG(a.size() == b.size(), "Vectors must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    size_t n = a.size();
    if (n < 2) return std::numeric_limits<value_type>::quiet_NaN();
    value_type mean_a = 0, mean_b = 0;
    for (size_t i = 0; i < n; ++i) {
        mean_a += static_cast<value_type>(a[i]);
        mean_b += static_cast<value_type>(b[i]);
    }
    mean_a /= static_cast<value_type>(n);
    mean_b /= static_cast<value_type>(n);
    value_type num = 0, den_a = 0, den_b = 0;
    for (size_t i = 0; i < n; ++i) {
        value_type da = static_cast<value_type>(a[i]) - mean_a;
        value_type db = static_cast<value_type>(b[i]) - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    if (den_a == 0 || den_b == 0) return value_type(1);
    value_type corr = num / std::sqrt(den_a * den_b);
    return value_type(1) - corr;
}

/// Hamming distance (proportion of differing elements)
template <class E1, class E2>
auto hamming(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Hamming distance requires 1D vectors");
    XTU_ASSERT_MSG(a.size() == b.size(), "Vectors must have same size");
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++diff;
    }
    return static_cast<double>(diff) / static_cast<double>(a.size());
}

/// Jaccard distance for binary vectors
template <class E1, class E2>
auto jaccard(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Jaccard distance requires 1D vectors");
    XTU_ASSERT_MSG(a.size() == b.size(), "Vectors must have same size");
    size_t intersection = 0, union_count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        bool a_nonzero = (a[i] != 0);
        bool b_nonzero = (b[i] != 0);
        if (a_nonzero && b_nonzero) ++intersection;
        if (a_nonzero || b_nonzero) ++union_count;
    }
    if (union_count == 0) return 0.0;
    return 1.0 - static_cast<double>(intersection) / static_cast<double>(union_count);
}

/// Canberra distance
template <class E1, class E2>
auto canberra(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Canberra distance requires 1D vectors");
    XTU_ASSERT_MSG(a.size() == b.size(), "Vectors must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    value_type sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        value_type va = std::abs(static_cast<value_type>(a[i]));
        value_type vb = std::abs(static_cast<value_type>(b[i]));
        value_type denom = va + vb;
        if (denom > 0) {
            sum += std::abs(static_cast<value_type>(a[i]) - static_cast<value_type>(b[i])) / denom;
        }
    }
    return sum;
}

/// Bray-Curtis dissimilarity
template <class E1, class E2>
auto braycurtis(const xexpression<E1>& x, const xexpression<E2>& y) {
    const auto& a = x.derived_cast();
    const auto& b = y.derived_cast();
    XTU_ASSERT_MSG(a.dimension() == 1 && b.dimension() == 1, "Bray-Curtis distance requires 1D vectors");
    XTU_ASSERT_MSG(a.size() == b.size(), "Vectors must have same size");
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    value_type sum_abs_diff = 0, sum_abs_sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum_abs_diff += std::abs(static_cast<value_type>(a[i]) - static_cast<value_type>(b[i]));
        sum_abs_sum += std::abs(static_cast<value_type>(a[i])) + std::abs(static_cast<value_type>(b[i]));
    }
    if (sum_abs_sum == 0) return value_type(0);
    return sum_abs_diff / sum_abs_sum;
}

// #############################################################################
// Generic distance function (dispatches to appropriate metric)
// #############################################################################
template <class E1, class E2>
auto distance(const xexpression<E1>& x, const xexpression<E2>& y, 
              distance_metric metric = distance_metric::euclidean, double p = 2.0) {
    switch (metric) {
        case distance_metric::euclidean:    return euclidean(x, y);
        case distance_metric::sqeuclidean:  return sqeuclidean(x, y);
        case distance_metric::manhattan:    return manhattan(x, y);
        case distance_metric::chebyshev:    return chebyshev(x, y);
        case distance_metric::minkowski:    return minkowski(x, y, p);
        case distance_metric::cosine:       return cosine(x, y);
        case distance_metric::correlation:  return correlation(x, y);
        case distance_metric::hamming:      return hamming(x, y);
        case distance_metric::jaccard:      return jaccard(x, y);
        case distance_metric::canberra:     return canberra(x, y);
        case distance_metric::braycurtis:   return braycurtis(x, y);
        default: XTU_THROW(std::invalid_argument, "Unsupported distance metric");
    }
}

// #############################################################################
// Pairwise distance matrix (all pairs)
// #############################################################################
template <class E>
auto pdist(const xexpression<E>& data, distance_metric metric = distance_metric::euclidean, double p = 2.0) {
    const auto& mat = data.derived_cast();
    XTU_ASSERT_MSG(mat.dimension() == 2, "pdist requires 2D array (observations x features)");
    using value_type = typename E::value_type;
    size_t n = mat.shape()[0];
    size_t m = mat.shape()[1];
    size_t n_pairs = n * (n - 1) / 2;
    xarray_container<value_type> result({n_pairs});
    
    size_t idx = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            // Extract vectors i and j
            xarray_container<value_type> vi({m}), vj({m});
            for (size_t k = 0; k < m; ++k) {
                vi[k] = mat(i, k);
                vj[k] = mat(j, k);
            }
            result[idx++] = distance(vi, vj, metric, p);
        }
    }
    return result;
}

/// Pairwise distances as square matrix
template <class E>
auto cdist(const xexpression<E>& X, const xexpression<E>& Y, 
           distance_metric metric = distance_metric::euclidean, double p = 2.0) {
    const auto& matX = X.derived_cast();
    const auto& matY = Y.derived_cast();
    XTU_ASSERT_MSG(matX.dimension() == 2 && matY.dimension() == 2, "cdist requires 2D arrays");
    XTU_ASSERT_MSG(matX.shape()[1] == matY.shape()[1], "Feature dimensions must match");
    using value_type = typename E::value_type;
    size_t nx = matX.shape()[0];
    size_t ny = matY.shape()[0];
    size_t m = matX.shape()[1];
    xarray_container<value_type> result({nx, ny});
    
    for (size_t i = 0; i < nx; ++i) {
        xarray_container<value_type> vi({m});
        for (size_t k = 0; k < m; ++k) vi[k] = matX(i, k);
        for (size_t j = 0; j < ny; ++j) {
            xarray_container<value_type> vj({m});
            for (size_t k = 0; k < m; ++k) vj[k] = matY(j, k);
            result(i, j) = distance(vi, vj, metric, p);
        }
    }
    return result;
}

// #############################################################################
// Silhouette score for clustering evaluation
// #############################################################################
template <class E, class Labels>
auto silhouette_score(const xexpression<E>& data, const xexpression<Labels>& labels,
                      distance_metric metric = distance_metric::euclidean) {
    const auto& mat = data.derived_cast();
    const auto& lbls = labels.derived_cast();
    XTU_ASSERT_MSG(mat.dimension() == 2, "Data must be 2D");
    XTU_ASSERT_MSG(lbls.dimension() == 1 && lbls.size() == mat.shape()[0], "Labels must be 1D with matching size");
    using value_type = typename E::value_type;
    size_t n = mat.shape()[0];
    size_t m = mat.shape()[1];
    
    // Find unique labels
    std::vector<size_t> unique_labels;
    for (size_t i = 0; i < n; ++i) {
        size_t lab = static_cast<size_t>(lbls[i]);
        if (std::find(unique_labels.begin(), unique_labels.end(), lab) == unique_labels.end()) {
            unique_labels.push_back(lab);
        }
    }
    if (unique_labels.size() < 2) return value_type(-1);
    
    // Precompute all pairwise distances (could be optimized)
    auto dist_mat = cdist(data, data, metric);
    
    value_type total_score = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t own_label = static_cast<size_t>(lbls[i]);
        // Compute a(i): mean distance to points in same cluster
        value_type a_i = 0;
        size_t own_count = 0;
        for (size_t j = 0; j < n; ++j) {
            if (i != j && static_cast<size_t>(lbls[j]) == own_label) {
                a_i += dist_mat(i, j);
                ++own_count;
            }
        }
        if (own_count > 0) a_i /= own_count;
        else a_i = 0;
        
        // Compute b(i): min mean distance to other clusters
        value_type b_i = std::numeric_limits<value_type>::max();
        for (size_t other_label : unique_labels) {
            if (other_label == own_label) continue;
            value_type mean_dist = 0;
            size_t other_count = 0;
            for (size_t j = 0; j < n; ++j) {
                if (static_cast<size_t>(lbls[j]) == other_label) {
                    mean_dist += dist_mat(i, j);
                    ++other_count;
                }
            }
            if (other_count > 0) {
                mean_dist /= other_count;
                if (mean_dist < b_i) b_i = mean_dist;
            }
        }
        if (own_count == 0) continue;
        value_type s_i = (b_i - a_i) / std::max(a_i, b_i);
        total_score += s_i;
    }
    return total_score / static_cast<value_type>(n);
}

// #############################################################################
// KNN (K-Nearest Neighbors) search
// #############################################################################
template <class E>
auto knn(const xexpression<E>& data, const xexpression<E>& query, size_t k,
         distance_metric metric = distance_metric::euclidean) {
    const auto& train = data.derived_cast();
    const auto& test = query.derived_cast();
    XTU_ASSERT_MSG(train.dimension() == 2 && test.dimension() == 2, "Data must be 2D");
    XTU_ASSERT_MSG(train.shape()[1] == test.shape()[1], "Feature dimensions must match");
    using value_type = typename E::value_type;
    size_t n_train = train.shape()[0];
    size_t n_query = test.shape()[0];
    size_t m = train.shape()[1];
    k = std::min(k, n_train);
    
    xarray_container<size_t> indices({n_query, k});
    xarray_container<value_type> distances({n_query, k});
    
    for (size_t q = 0; q < n_query; ++q) {
        // Compute distances from query point to all training points
        std::vector<std::pair<value_type, size_t>> dist_pairs;
        dist_pairs.reserve(n_train);
        xarray_container<value_type> vq({m});
        for (size_t f = 0; f < m; ++f) vq[f] = test(q, f);
        for (size_t t = 0; t < n_train; ++t) {
            xarray_container<value_type> vt({m});
            for (size_t f = 0; f < m; ++f) vt[f] = train(t, f);
            value_type d = distance(vq, vt, metric);
            dist_pairs.emplace_back(d, t);
        }
        // Find k nearest
        std::nth_element(dist_pairs.begin(), dist_pairs.begin() + k, dist_pairs.end());
        std::sort(dist_pairs.begin(), dist_pairs.begin() + k);
        for (size_t i = 0; i < k; ++i) {
            distances(q, i) = dist_pairs[i].first;
            indices(q, i) = dist_pairs[i].second;
        }
    }
    return std::make_pair(std::move(distances), std::move(indices));
}

// #############################################################################
// Utility: squareform to convert between condensed and square distance matrices
// #############################################################################
template <class E>
auto squareform(const xexpression<E>& dist, bool to_square = true) {
    const auto& d = dist.derived_cast();
    using value_type = typename E::value_type;
    if (to_square) {
        // Convert condensed vector to square matrix
        XTU_ASSERT_MSG(d.dimension() == 1, "Input must be 1D condensed distance vector");
        size_t n_pairs = d.size();
        size_t n = static_cast<size_t>((1 + std::sqrt(1 + 8 * n_pairs)) / 2);
        XTU_ASSERT_MSG(n * (n - 1) / 2 == n_pairs, "Invalid condensed distance vector size");
        xarray_container<value_type> result({n, n});
        size_t idx = 0;
        for (size_t i = 0; i < n; ++i) {
            result(i, i) = 0;
            for (size_t j = i + 1; j < n; ++j) {
                result(i, j) = d[idx];
                result(j, i) = d[idx];
                ++idx;
            }
        }
        return result;
    } else {
        // Convert square matrix to condensed vector
        XTU_ASSERT_MSG(d.dimension() == 2, "Input must be 2D square distance matrix");
        size_t n = d.shape()[0];
        XTU_ASSERT_MSG(d.shape()[1] == n, "Matrix must be square");
        size_t n_pairs = n * (n - 1) / 2;
        xarray_container<value_type> result({n_pairs});
        size_t idx = 0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                result[idx++] = d(i, j);
            }
        }
        return result;
    }
}

} // namespace metrics

// Bring into main namespace for convenience
using metrics::distance_metric;
using metrics::euclidean;
using metrics::sqeuclidean;
using metrics::manhattan;
using metrics::chebyshev;
using metrics::minkowski;
using metrics::cosine;
using metrics::cosine_similarity;
using metrics::correlation;
using metrics::hamming;
using metrics::jaccard;
using metrics::canberra;
using metrics::braycurtis;
using metrics::distance;
using metrics::pdist;
using metrics::cdist;
using metrics::silhouette_score;
using metrics::knn;
using metrics::squareform;

XTU_NAMESPACE_END

#endif // XTU_METRICS_XMETRICS_HPP