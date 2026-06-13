// system name : onetbb-warp
// File 0038 : core/math/clustering.h
// Description : Clustering algorithms: k‑means, DBSCAN, mean‑shift, hierarchical.

#ifndef __TBB_WARP_CORE_MATH_CLUSTERING_H
#define __TBB_WARP_CORE_MATH_CLUSTERING_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/random.h"
#include "core/math/statistics.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <queue>
#include <unordered_set>
#include <functional>
#include <cstdint>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Distance functions for clustering
// ============================================================

template<typename T>
T squared_distance(const std::vector<T>& a, const std::vector<T>& b) {
    T sum = T(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        T d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

template<typename T>
T euclidean_distance(const std::vector<T>& a, const std::vector<T>& b) {
    return std::sqrt(squared_distance(a, b));
}

// ============================================================
// k‑means clustering (Lloyd's algorithm)
// ============================================================

template<typename T>
std::vector<int> kmeans(const std::vector<std::vector<T>>& data, int k, int max_iter = 300, int seed = 42) {
    std::size_t n = data.size();
    if (n == 0 || k <= 0 || k > static_cast<int>(n)) return std::vector<int>(n, 0);
    std::size_t dim = data[0].size();
    xorshift64 rng(seed);
    std::vector<std::vector<T>> centroids(k, std::vector<T>(dim));
    std::vector<int> labels(n, 0);
    // Initialize centroids using k‑means++
    centroids[0] = data[uniform_uint(rng, 0, static_cast<uint32_t>(n-1))];
    for (int c = 1; c < k; ++c) {
        T total_dist = T(0);
        std::vector<T> min_dists(n, std::numeric_limits<T>::max());
        for (std::size_t i = 0; i < n; ++i) {
            for (int j = 0; j < c; ++j) {
                T d = squared_distance(data[i], centroids[j]);
                if (d < min_dists[i]) min_dists[i] = d;
            }
            total_dist += min_dists[i];
        }
        T r = uniform_float(rng) * total_dist;
        T accum = T(0);
        std::size_t chosen = 0;
        for (std::size_t i = 0; i < n; ++i) {
            accum += min_dists[i];
            if (accum >= r) { chosen = i; break; }
        }
        centroids[c] = data[chosen];
    }
    for (int iter = 0; iter < max_iter; ++iter) {
        bool changed = false;
        for (std::size_t i = 0; i < n; ++i) {
            T min_dist = std::numeric_limits<T>::max();
            int best = 0;
            for (int c = 0; c < k; ++c) {
                T d = squared_distance(data[i], centroids[c]);
                if (d < min_dist) { min_dist = d; best = c; }
            }
            if (labels[i] != best) { labels[i] = best; changed = true; }
        }
        if (!changed) break;
        std::vector<std::vector<T>> new_centroids(k, std::vector<T>(dim, T(0)));
        std::vector<int> counts(k, 0);
        for (std::size_t i = 0; i < n; ++i) {
            int c = labels[i];
            counts[c]++;
            for (std::size_t j = 0; j < dim; ++j) new_centroids[c][j] += data[i][j];
        }
        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                T inv = T(1) / counts[c];
                for (std::size_t j = 0; j < dim; ++j) centroids[c][j] = new_centroids[c][j] * inv;
            }
        }
    }
    return labels;
}

// ============================================================
// DBSCAN clustering
// ============================================================

template<typename T>
std::vector<int> dbscan(const std::vector<std::vector<T>>& data, T eps, int min_pts) {
    std::size_t n = data.size();
    std::vector<int> labels(n, -1); // -1 = noise, 0+ = cluster id
    int cluster_id = 0;
    auto region_query = [&](std::size_t p) {
        std::vector<std::size_t> neighbors;
        for (std::size_t i = 0; i < n; ++i) {
            if (squared_distance(data[p], data[i]) <= eps * eps)
                neighbors.push_back(i);
        }
        return neighbors;
    };
    std::function<void(std::size_t, std::vector<std::size_t>&)> expand_cluster =
        [&](std::size_t p, std::vector<std::size_t>& neighbors) {
            labels[p] = cluster_id;
            for (std::size_t i = 0; i < neighbors.size(); ++i) {
                std::size_t q = neighbors[i];
                if (labels[q] == -1) {
                    labels[q] = cluster_id;
                    auto q_neighbors = region_query(q);
                    if (q_neighbors.size() >= static_cast<std::size_t>(min_pts)) {
                        neighbors.insert(neighbors.end(), q_neighbors.begin(), q_neighbors.end());
                    }
                }
                if (labels[q] == -1) labels[q] = cluster_id;
            }
        };
    for (std::size_t p = 0; p < n; ++p) {
        if (labels[p] != -1) continue;
        auto neighbors = region_query(p);
        if (neighbors.size() < static_cast<std::size_t>(min_pts)) {
            labels[p] = -1; // noise
        } else {
            expand_cluster(p, neighbors);
            ++cluster_id;
        }
    }
    return labels;
}

// ============================================================
// Mean‑shift clustering
// ============================================================

template<typename T>
std::vector<int> mean_shift(const std::vector<std::vector<T>>& data, T bandwidth, T convergence_threshold = T(1e-6), int max_iter = 300) {
    std::size_t n = data.size();
    std::size_t dim = data[0].size();
    std::vector<std::vector<T>> points = data;
    for (int iter = 0; iter < max_iter; ++iter) {
        T max_shift = T(0);
        std::vector<std::vector<T>> new_points(n, std::vector<T>(dim, T(0)));
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<T> sum(dim, T(0));
            T weight_sum = T(0);
            for (std::size_t j = 0; j < n; ++j) {
                T d2 = squared_distance(points[i], data[j]);
                if (d2 > bandwidth * bandwidth) continue;
                T weight = std::exp(-d2 / (T(2) * bandwidth * bandwidth));
                weight_sum += weight;
                for (std::size_t d = 0; d < dim; ++d) sum[d] += data[j][d] * weight;
            }
            if (weight_sum > T(0)) {
                for (std::size_t d = 0; d < dim; ++d) new_points[i][d] = sum[d] / weight_sum;
            } else {
                new_points[i] = points[i];
            }
            T shift = squared_distance(new_points[i], points[i]);
            if (shift > max_shift) max_shift = shift;
        }
        points.swap(new_points);
        if (max_shift < convergence_threshold) break;
    }
    // Cluster converged points by merging close modes
    std::vector<int> labels(n, -1);
    int cluster_id = 0;
    T merge_threshold = bandwidth * T(0.5);
    for (std::size_t i = 0; i < n; ++i) {
        if (labels[i] != -1) continue;
        labels[i] = cluster_id;
        for (std::size_t j = i + 1; j < n; ++j) {
            if (labels[j] != -1) continue;
            if (squared_distance(points[i], points[j]) < merge_threshold * merge_threshold) {
                labels[j] = cluster_id;
            }
        }
        ++cluster_id;
    }
    return labels;
}

// ============================================================
// Agglomerative hierarchical clustering (single‑linkage)
// ============================================================

template<typename T>
struct agglomerative_node {
    int left;
    int right;
    T distance;
    int id;
};

template<typename T>
std::vector<agglomerative_node<T>> hierarchical_clustering_single_linkage(
    const std::vector<std::vector<T>>& data, int target_clusters) {
    std::size_t n = data.size();
    if (n <= 1 || target_clusters >= static_cast<int>(n)) return {};
    // Distance matrix (condensed)
    std::vector<std::vector<T>> dist(n, std::vector<T>(n, T(0)));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            dist[i][j] = dist[j][i] = euclidean_distance(data[i], data[j]);
    std::vector<int> active(n);
    for (std::size_t i = 0; i < n; ++i) active[i] = static_cast<int>(i);
    std::vector<agglomerative_node<T>> dendrogram;
    int next_id = static_cast<int>(n);
    std::vector<int> cluster_size(n, 1);
    for (int step = 0; step < static_cast<int>(n) - target_clusters; ++step) {
        T min_dist = std::numeric_limits<T>::max();
        int a = -1, b = -1;
        for (std::size_t i = 0; i < active.size(); ++i) {
            for (std::size_t j = i + 1; j < active.size(); ++j) {
                int ci = active[i], cj = active[j];
                T d = dist[ci][cj];
                if (d < min_dist) { min_dist = d; a = ci; b = cj; }
            }
        }
        if (a < 0 || b < 0) break;
        agglomerative_node<T> node;
        node.left = a; node.right = b;
        node.distance = min_dist;
        node.id = next_id++;
        dendrogram.push_back(node);
        // Update distances (single linkage: min)
        int new_c = node.id;
        // Remove a, b from active; add new_c
        active.erase(std::remove(active.begin(), active.end(), a), active.end());
        active.erase(std::remove(active.begin(), active.end(), b), active.end());
        active.push_back(new_c);
        // Compute distances from new_c to all other clusters
        for (int c : active) {
            if (c == new_c) continue;
            T d = std::min(dist[a][c], dist[b][c]);
            dist[new_c][c] = d;
            dist[c][new_c] = d;
        }
    }
    return dendrogram;
}

// ============================================================
// Elbow method for optimal k in k‑means
// ============================================================

template<typename T>
std::vector<T> kmeans_wcss(const std::vector<std::vector<T>>& data, int max_k) {
    std::vector<T> wcss(max_k + 1, T(0));
    for (int k = 1; k <= max_k; ++k) {
        auto labels = kmeans(data, k);
        std::vector<std::vector<T>> centroids(k, std::vector<T>(data[0].size(), T(0)));
        std::vector<int> counts(k, 0);
        for (std::size_t i = 0; i < data.size(); ++i) {
            int c = labels[i];
            counts[c]++;
            for (std::size_t d = 0; d < data[0].size(); ++d)
                centroids[c][d] += data[i][d];
        }
        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) continue;
            T inv = T(1) / counts[c];
            for (std::size_t d = 0; d < data[0].size(); ++d)
                centroids[c][d] *= inv;
        }
        T sse = T(0);
        for (std::size_t i = 0; i < data.size(); ++i) {
            sse += squared_distance(data[i], centroids[labels[i]]);
        }
        wcss[k] = sse;
    }
    return wcss;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_CLUSTERING_H