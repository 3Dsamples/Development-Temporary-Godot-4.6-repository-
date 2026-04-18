// include/xtu/cluster/xcluster.hpp
// xtensor-unified - Clustering algorithms: K-Means, DBSCAN, Hierarchical, Spectral
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_CLUSTER_XCLUSTER_HPP
#define XTU_CLUSTER_XCLUSTER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include <queue>
#include <tuple>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/containers/xarray.hpp"
#include "xtu/containers/xtensor.hpp"
#include "xtu/math/xrandom.hpp"
#include "xtu/math/xnorm.hpp"
#include "xtu/math/xsorting.hpp"
#include "xtu/math/xlinalg.hpp"
#include "xtu/metrics/xmetrics.hpp"
#include "xtu/parallel/xparallel.hpp"
#include "xtu/manipulation/xmanipulation.hpp"

XTU_NAMESPACE_BEGIN
namespace cluster {

// #############################################################################
// K-Means clustering (Lloyd algorithm)
// #############################################################################
template <class T = double>
class KMeans {
private:
    size_t m_n_clusters;
    size_t m_max_iter;
    T m_tol;
    size_t m_n_init;
    std::string m_init_method;
    math::random::random_engine m_rng;
    xarray_container<T> m_cluster_centers_;
    xarray_container<size_t> m_labels_;
    T m_inertia_;
    size_t m_n_iter_;

public:
    KMeans(size_t n_clusters = 8, size_t max_iter = 300, T tol = 1e-4,
           size_t n_init = 10, const std::string& init = "k-means++")
        : m_n_clusters(n_clusters), m_max_iter(max_iter), m_tol(tol),
          m_n_init(n_init), m_init_method(init), m_rng(std::random_device{}()) {}

    // Fit model
    template <class E>
    void fit(const xexpression<E>& X) {
        const auto& data = X.derived_cast();
        XTU_ASSERT_MSG(data.dimension() == 2, "Input must be 2D (n_samples, n_features)");
        size_t n_samples = data.shape()[0];
        size_t n_features = data.shape()[1];
        XTU_ASSERT_MSG(n_samples >= m_n_clusters, "n_samples must be >= n_clusters");

        T best_inertia = std::numeric_limits<T>::max();
        xarray_container<T> best_centers;
        xarray_container<size_t> best_labels;
        size_t best_iter = 0;

        for (size_t init_run = 0; init_run < m_n_init; ++init_run) {
            xarray_container<T> centers = initialize_centers(data);
            xarray_container<size_t> labels({n_samples});
            T inertia = 0;
            size_t iter;
            for (iter = 0; iter < m_max_iter; ++iter) {
                // Assign labels
                inertia = assign_labels(data, centers, labels);
                // Update centers
                xarray_container<T> new_centers = update_centers(data, labels);
                // Check convergence
                T center_shift = 0;
                for (size_t k = 0; k < m_n_clusters; ++k) {
                    for (size_t f = 0; f < n_features; ++f) {
                        T diff = centers(k, f) - new_centers(k, f);
                        center_shift += diff * diff;
                    }
                }
                centers = std::move(new_centers);
                if (std::sqrt(center_shift) < m_tol) break;
            }
            if (inertia < best_inertia) {
                best_inertia = inertia;
                best_centers = centers;
                best_labels = std::move(labels);
                best_iter = iter;
            }
        }
        m_cluster_centers_ = std::move(best_centers);
        m_labels_ = std::move(best_labels);
        m_inertia_ = best_inertia;
        m_n_iter_ = best_iter;
    }

    template <class E>
    auto predict(const xexpression<E>& X) const {
        const auto& data = X.derived_cast();
        XTU_ASSERT_MSG(data.dimension() == 2 && data.shape()[1] == m_cluster_centers_.shape()[1],
                       "Feature dimension mismatch");
        size_t n_samples = data.shape()[0];
        xarray_container<size_t> labels({n_samples});
        assign_labels(data, m_cluster_centers_, labels);
        return labels;
    }

    template <class E>
    auto fit_predict(const xexpression<E>& X) {
        fit(X);
        return m_labels_;
    }

    template <class E>
    auto transform(const xexpression<E>& X) const {
        const auto& data = X.derived_cast();
        size_t n_samples = data.shape()[0];
        size_t n_features = data.shape()[1];
        xarray_container<T> result({n_samples, m_n_clusters});
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t k = 0; k < m_n_clusters; ++k) {
                T dist = 0;
                for (size_t f = 0; f < n_features; ++f) {
                    T diff = data(i, f) - m_cluster_centers_(k, f);
                    dist += diff * diff;
                }
                result(i, k) = std::sqrt(dist);
            }
        }
        return result;
    }

    const xarray_container<T>& cluster_centers() const { return m_cluster_centers_; }
    const xarray_container<size_t>& labels() const { return m_labels_; }
    T inertia() const { return m_inertia_; }
    size_t n_iter() const { return m_n_iter_; }

private:
    template <class E>
    xarray_container<T> initialize_centers(const E& data) {
        size_t n_samples = data.shape()[0];
        size_t n_features = data.shape()[1];
        xarray_container<T> centers({m_n_clusters, n_features});
        if (m_init_method == "random") {
            std::uniform_int_distribution<size_t> dist(0, n_samples - 1);
            for (size_t k = 0; k < m_n_clusters; ++k) {
                size_t idx = dist(m_rng);
                for (size_t f = 0; f < n_features; ++f) {
                    centers(k, f) = data(idx, f);
                }
            }
        } else { // k-means++
            // Choose first center uniformly
            std::uniform_int_distribution<size_t> dist(0, n_samples - 1);
            size_t first = dist(m_rng);
            for (size_t f = 0; f < n_features; ++f) centers(0, f) = data(first, f);
            std::vector<T> min_dist_sq(n_samples, std::numeric_limits<T>::max());
            for (size_t k = 1; k < m_n_clusters; ++k) {
                // Update distances
                for (size_t i = 0; i < n_samples; ++i) {
                    T dist_sq = 0;
                    for (size_t f = 0; f < n_features; ++f) {
                        T diff = data(i, f) - centers(k-1, f);
                        dist_sq += diff * diff;
                    }
                    if (dist_sq < min_dist_sq[i]) min_dist_sq[i] = dist_sq;
                }
                // Sample with probability proportional to min_dist_sq
                T sum = std::accumulate(min_dist_sq.begin(), min_dist_sq.end(), T(0));
                std::uniform_real_distribution<T> udist(0, sum);
                T r = udist(m_rng);
                T accum = 0;
                size_t next_idx = 0;
                for (size_t i = 0; i < n_samples; ++i) {
                    accum += min_dist_sq[i];
                    if (accum >= r) {
                        next_idx = i;
                        break;
                    }
                }
                for (size_t f = 0; f < n_features; ++f) centers(k, f) = data(next_idx, f);
            }
        }
        return centers;
    }

    template <class E>
    T assign_labels(const E& data, const xarray_container<T>& centers, xarray_container<size_t>& labels) const {
        size_t n_samples = data.shape()[0];
        T inertia = 0;
        parallel::parallel_for(0, n_samples, [&](size_t i) {
            T min_dist_sq = std::numeric_limits<T>::max();
            size_t best_k = 0;
            for (size_t k = 0; k < m_n_clusters; ++k) {
                T dist_sq = 0;
                for (size_t f = 0; f < data.shape()[1]; ++f) {
                    T diff = data(i, f) - centers(k, f);
                    dist_sq += diff * diff;
                }
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_k = k;
                }
            }
            labels[i] = best_k;
            inertia += min_dist_sq;
        });
        return inertia;
    }

    template <class E>
    xarray_container<T> update_centers(const E& data, const xarray_container<size_t>& labels) const {
        size_t n_features = data.shape()[1];
        xarray_container<T> centers({m_n_clusters, n_features});
        std::fill(centers.begin(), centers.end(), T(0));
        std::vector<size_t> counts(m_n_clusters, 0);
        for (size_t i = 0; i < data.shape()[0]; ++i) {
            size_t k = labels[i];
            ++counts[k];
            for (size_t f = 0; f < n_features; ++f) {
                centers(k, f) += data(i, f);
            }
        }
        for (size_t k = 0; k < m_n_clusters; ++k) {
            if (counts[k] > 0) {
                for (size_t f = 0; f < n_features; ++f) {
                    centers(k, f) /= static_cast<T>(counts[k]);
                }
            }
        }
        return centers;
    }
};

// #############################################################################
// DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
// #############################################################################
template <class T = double>
class DBSCAN {
private:
    T m_eps;
    size_t m_min_samples;
    metrics::distance_metric m_metric;
    xarray_container<int32_t> m_labels_;
    size_t m_n_noise_;

public:
    DBSCAN(T eps = 0.5, size_t min_samples = 5, metrics::distance_metric metric = metrics::distance_metric::euclidean)
        : m_eps(eps), m_min_samples(min_samples), m_metric(metric), m_n_noise_(0) {}

    template <class E>
    void fit(const xexpression<E>& X) {
        const auto& data = X.derived_cast();
        XTU_ASSERT_MSG(data.dimension() == 2, "Input must be 2D (n_samples, n_features)");
        size_t n_samples = data.shape()[0];
        // Precompute neighborhood
        std::vector<std::vector<size_t>> neighbors(n_samples);
        parallel::parallel_for(0, n_samples, [&](size_t i) {
            for (size_t j = 0; j < n_samples; ++j) {
                if (i == j) continue;
                T dist = pairwise_distance(data, i, j);
                if (dist <= m_eps) neighbors[i].push_back(j);
            }
        });

        m_labels_ = xarray_container<int32_t>({n_samples});
        std::fill(m_labels_.begin(), m_labels_.end(), -1); // -1 = unvisited
        int32_t cluster_id = 0;
        for (size_t i = 0; i < n_samples; ++i) {
            if (m_labels_[i] != -1) continue;
            if (neighbors[i].size() < m_min_samples) {
                m_labels_[i] = -2; // noise
                ++m_n_noise_;
                continue;
            }
            // Expand cluster
            m_labels_[i] = cluster_id;
            std::vector<size_t> seeds = neighbors[i];
            size_t idx = 0;
            while (idx < seeds.size()) {
                size_t j = seeds[idx++];
                if (m_labels_[j] == -2) m_labels_[j] = cluster_id;
                if (m_labels_[j] != -1) continue;
                m_labels_[j] = cluster_id;
                if (neighbors[j].size() >= m_min_samples) {
                    seeds.insert(seeds.end(), neighbors[j].begin(), neighbors[j].end());
                }
            }
            ++cluster_id;
        }
    }

    template <class E>
    auto fit_predict(const xexpression<E>& X) {
        fit(X);
        return m_labels_;
    }

    const xarray_container<int32_t>& labels() const { return m_labels_; }
    size_t n_noise() const { return m_n_noise_; }

private:
    template <class E>
    T pairwise_distance(const E& data, size_t i, size_t j) const {
        T dist = 0;
        for (size_t f = 0; f < data.shape()[1]; ++f) {
            T diff = data(i, f) - data(j, f);
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }
};

// #############################################################################
// Agglomerative Hierarchical Clustering
// #############################################################################
template <class T = double>
class AgglomerativeClustering {
public:
    enum class linkage_type { ward, complete, average, single };

private:
    size_t m_n_clusters;
    linkage_type m_linkage;
    metrics::distance_metric m_metric;
    xarray_container<size_t> m_labels_;
    std::vector<std::vector<size_t>> m_children_;

public:
    AgglomerativeClustering(size_t n_clusters = 2, linkage_type linkage = linkage_type::ward,
                            metrics::distance_metric metric = metrics::distance_metric::euclidean)
        : m_n_clusters(n_clusters), m_linkage(linkage), m_metric(metric) {}

    template <class E>
    void fit(const xexpression<E>& X) {
        const auto& data = X.derived_cast();
        size_t n_samples = data.shape()[0];
        if (m_n_clusters > n_samples) m_n_clusters = n_samples;

        // Compute initial pairwise distances
        size_t n_pairs = n_samples * (n_samples - 1) / 2;
        std::vector<T> dist_condensed = metrics::pdist(data, m_metric).template as_array<T>();
        // Convert to square matrix for easier update
        xarray_container<T> dist_mat({n_samples, n_samples});
        size_t idx = 0;
        for (size_t i = 0; i < n_samples; ++i) {
            dist_mat(i, i) = 0;
            for (size_t j = i + 1; j < n_samples; ++j) {
                dist_mat(i, j) = dist_condensed[idx];
                dist_mat(j, i) = dist_condensed[idx];
                ++idx;
            }
        }

        // Each sample starts as its own cluster
        std::vector<std::vector<size_t>> clusters(n_samples);
        for (size_t i = 0; i < n_samples; ++i) clusters[i] = {i};
        std::vector<bool> active(n_samples, true);
        m_children_.clear();

        for (size_t iter = 0; iter < n_samples - m_n_clusters; ++iter) {
            // Find closest pair of active clusters
            T min_dist = std::numeric_limits<T>::max();
            std::pair<size_t, size_t> merge_pair;
            for (size_t i = 0; i < n_samples; ++i) {
                if (!active[i]) continue;
                for (size_t j = i + 1; j < n_samples; ++j) {
                    if (!active[j]) continue;
                    if (dist_mat(i, j) < min_dist) {
                        min_dist = dist_mat(i, j);
                        merge_pair = {i, j};
                    }
                }
            }
            size_t a = merge_pair.first, b = merge_pair.second;
            // Merge b into a
            clusters[a].insert(clusters[a].end(), clusters[b].begin(), clusters[b].end());
            active[b] = false;
            m_children_.push_back({a, b});

            // Update distances from new cluster to all others
            for (size_t k = 0; k < n_samples; ++k) {
                if (!active[k] || k == a) continue;
                T new_dist = compute_linkage_distance(clusters[a], clusters[k], dist_mat, clusters[a].size(), clusters[k].size());
                dist_mat(a, k) = new_dist;
                dist_mat(k, a) = new_dist;
            }
        }

        // Assign labels
        m_labels_ = xarray_container<size_t>({n_samples});
        int32_t label = 0;
        for (size_t i = 0; i < n_samples; ++i) {
            if (active[i]) {
                for (size_t sample : clusters[i]) m_labels_[sample] = label;
                ++label;
            }
        }
    }

    template <class E>
    auto fit_predict(const xexpression<E>& X) {
        fit(X);
        return m_labels_;
    }

    const xarray_container<size_t>& labels() const { return m_labels_; }
    const std::vector<std::vector<size_t>>& children() const { return m_children_; }

private:
    T compute_linkage_distance(const std::vector<size_t>& cluster_a,
                               const std::vector<size_t>& cluster_b,
                               const xarray_container<T>& dist_mat,
                               size_t size_a, size_t size_b) {
        switch (m_linkage) {
            case linkage_type::single: {
                T min_d = std::numeric_limits<T>::max();
                for (size_t i : cluster_a) for (size_t j : cluster_b) min_d = std::min(min_d, dist_mat(i, j));
                return min_d;
            }
            case linkage_type::complete: {
                T max_d = 0;
                for (size_t i : cluster_a) for (size_t j : cluster_b) max_d = std::max(max_d, dist_mat(i, j));
                return max_d;
            }
            case linkage_type::average: {
                T sum = 0;
                for (size_t i : cluster_a) for (size_t j : cluster_b) sum += dist_mat(i, j);
                return sum / static_cast<T>(size_a * size_b);
            }
            case linkage_type::ward: {
                // Lance-Williams formula for Ward
                // Not fully implemented here; fallback to average
                T sum = 0;
                for (size_t i : cluster_a) for (size_t j : cluster_b) sum += dist_mat(i, j) * dist_mat(i, j);
                return std::sqrt(sum / static_cast<T>(size_a * size_b));
            }
        }
        return 0;
    }
};

// #############################################################################
// Spectral Clustering
// #############################################################################
template <class T = double>
class SpectralClustering {
private:
    size_t m_n_clusters;
    T m_gamma; // for RBF kernel
    bool m_assign_labels;
    xarray_container<size_t> m_labels_;
    xarray_container<T> m_affinity_matrix_;

public:
    SpectralClustering(size_t n_clusters = 8, T gamma = 1.0, bool assign_labels = true)
        : m_n_clusters(n_clusters), m_gamma(gamma), m_assign_labels(assign_labels) {}

    template <class E>
    void fit(const xexpression<E>& X) {
        const auto& data = X.derived_cast();
        XTU_ASSERT_MSG(data.dimension() == 2, "Input must be 2D");
        size_t n_samples = data.shape()[0];
        // Compute affinity matrix (RBF kernel)
        m_affinity_matrix_ = xarray_container<T>({n_samples, n_samples});
        for (size_t i = 0; i < n_samples; ++i) {
            m_affinity_matrix_(i, i) = 0;
            for (size_t j = i + 1; j < n_samples; ++j) {
                T dist_sq = 0;
                for (size_t f = 0; f < data.shape()[1]; ++f) {
                    T diff = data(i, f) - data(j, f);
                    dist_sq += diff * diff;
                }
                T val = std::exp(-m_gamma * dist_sq);
                m_affinity_matrix_(i, j) = val;
                m_affinity_matrix_(j, i) = val;
            }
        }
        // Compute Laplacian: L = D^{-1/2} * A * D^{-1/2} (normalized)
        xarray_container<T> degree({n_samples});
        for (size_t i = 0; i < n_samples; ++i) {
            degree[i] = 0;
            for (size_t j = 0; j < n_samples; ++j) degree[i] += m_affinity_matrix_(i, j);
        }
        xarray_container<T> laplacian({n_samples, n_samples});
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < n_samples; ++j) {
                if (degree[i] > 0 && degree[j] > 0) {
                    laplacian(i, j) = -m_affinity_matrix_(i, j) / std::sqrt(degree[i] * degree[j]);
                } else {
                    laplacian(i, j) = 0;
                }
            }
            laplacian(i, i) = 1;
        }
        // Eigendecomposition
        auto [eigvals, eigvecs] = math::eig(laplacian);
        // Select eigenvectors for smallest m_n_clusters eigenvalues
        std::vector<std::pair<T, size_t>> sorted_vals;
        for (size_t i = 0; i < n_samples; ++i) sorted_vals.emplace_back(eigvals[i], i);
        std::sort(sorted_vals.begin(), sorted_vals.end());
        xarray_container<T> embedding({n_samples, m_n_clusters});
        for (size_t k = 0; k < m_n_clusters; ++k) {
            size_t vec_idx = sorted_vals[k].second;
            for (size_t i = 0; i < n_samples; ++i) {
                embedding(i, k) = eigvecs(i, vec_idx);
            }
        }
        // Normalize rows
        for (size_t i = 0; i < n_samples; ++i) {
            T norm = 0;
            for (size_t k = 0; k < m_n_clusters; ++k) norm += embedding(i, k) * embedding(i, k);
            if (norm > 0) {
                norm = std::sqrt(norm);
                for (size_t k = 0; k < m_n_clusters; ++k) embedding(i, k) /= norm;
            }
        }
        // Cluster embedding with KMeans
        KMeans<T> kmeans(m_n_clusters, 300, 1e-4, 10);
        m_labels_ = kmeans.fit_predict(embedding);
    }

    template <class E>
    auto fit_predict(const xexpression<E>& X) {
        fit(X);
        return m_labels_;
    }

    const xarray_container<size_t>& labels() const { return m_labels_; }
    const xarray_container<T>& affinity_matrix() const { return m_affinity_matrix_; }
};

} // namespace cluster

// Bring into main namespace
using cluster::KMeans;
using cluster::DBSCAN;
using cluster::AgglomerativeClustering;
using cluster::SpectralClustering;

XTU_NAMESPACE_END

#endif // XTU_CLUSTER_XCLUSTER_HPP