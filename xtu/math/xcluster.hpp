// math/xcluster.hpp

#ifndef XTENSOR_XCLUSTER_HPP
#define XTENSOR_XCLUSTER_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"
#include "../math/xrandom.hpp"
#include "../math/xsorting.hpp"
#include "../math/xmetrics.hpp"
#include "../math/xoptimize.hpp"

#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <tuple>
#include <memory>
#include <random>
#include <iostream>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace cluster
        {
            using Matrix = xarray_container<double>;
            using Vector = xarray_container<double>;
            using Labels = xarray_container<int>;

            // --------------------------------------------------------------------
            // Distance metrics
            // --------------------------------------------------------------------
            namespace distance
            {
                inline double euclidean(const Vector& a, const Vector& b)
                {
                    return xt::norm_l2(a - b)();
                }

                inline double squared_euclidean(const Vector& a, const Vector& b)
                {
                    return xt::sum(xt::square(a - b))();
                }

                inline double manhattan(const Vector& a, const Vector& b)
                {
                    return xt::norm_l1(a - b)();
                }

                inline double cosine(const Vector& a, const Vector& b)
                {
                    double dot = xt::sum(a * b)();
                    double norm_a = xt::norm_l2(a)();
                    double norm_b = xt::norm_l2(b)();
                    if (norm_a < 1e-12 || norm_b < 1e-12) return 1.0;
                    return 1.0 - dot / (norm_a * norm_b);
                }

                inline double correlation(const Vector& a, const Vector& b)
                {
                    double ma = xt::mean(a)();
                    double mb = xt::mean(b)();
                    Vector a_centered = a - ma;
                    Vector b_centered = b - mb;
                    double dot = xt::sum(a_centered * b_centered)();
                    double norm_a = xt::norm_l2(a_centered)();
                    double norm_b = xt::norm_l2(b_centered)();
                    if (norm_a < 1e-12 || norm_b < 1e-12) return 1.0;
                    return 1.0 - dot / (norm_a * norm_b);
                }
            }

            // --------------------------------------------------------------------
            // K-Means Clustering
            // --------------------------------------------------------------------
            class KMeans
            {
            public:
                KMeans(size_t n_clusters = 8, size_t max_iter = 300, double tol = 1e-4,
                       size_t n_init = 10, const std::string& init = "k-means++",
                       size_t random_state = 42)
                    : m_n_clusters(n_clusters), m_max_iter(max_iter), m_tol(tol),
                      m_n_init(n_init), m_init(init), m_rng(random_state)
                {
                }

                template <class E>
                Labels fit_predict(const xexpression<E>& X)
                {
                    fit(X);
                    return m_labels;
                }

                template <class E>
                void fit(const xexpression<E>& X)
                {
                    const auto& data = X.derived_cast();
                    if (data.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "KMeans: input must be 2D (samples x features)");
                    m_n_samples = data.shape()[0];
                    m_n_features = data.shape()[1];
                    if (m_n_clusters > m_n_samples)
                        XTENSOR_THROW(std::invalid_argument, "KMeans: n_clusters > n_samples");

                    m_data = eval(data);
                    double best_inertia = std::numeric_limits<double>::max();

                    for (size_t init_run = 0; init_run < m_n_init; ++init_run)
                    {
                        Matrix centroids = initialize_centroids();
                        Labels labels({m_n_samples});
                        double inertia = 0.0;
                        for (size_t iter = 0; iter < m_max_iter; ++iter)
                        {
                            // Assign labels
                            inertia = 0.0;
                            bool changed = false;
                            for (size_t i = 0; i < m_n_samples; ++i)
                            {
                                Vector x = xt::view(m_data, i, xt::all());
                                int best_cluster = 0;
                                double min_dist = std::numeric_limits<double>::max();
                                for (size_t c = 0; c < m_n_clusters; ++c)
                                {
                                    Vector centroid = xt::view(centroids, c, xt::all());
                                    double dist = distance::squared_euclidean(x, centroid);
                                    if (dist < min_dist)
                                    {
                                        min_dist = dist;
                                        best_cluster = static_cast<int>(c);
                                    }
                                }
                                if (labels(i) != best_cluster) changed = true;
                                labels(i) = best_cluster;
                                inertia += min_dist;
                            }

                            // Update centroids
                            Matrix new_centroids = xt::zeros<double>({m_n_clusters, m_n_features});
                            std::vector<size_t> counts(m_n_clusters, 0);
                            for (size_t i = 0; i < m_n_samples; ++i)
                            {
                                size_t c = static_cast<size_t>(labels(i));
                                counts[c]++;
                                for (size_t f = 0; f < m_n_features; ++f)
                                    new_centroids(c, f) += m_data(i, f);
                            }
                            for (size_t c = 0; c < m_n_clusters; ++c)
                            {
                                if (counts[c] > 0)
                                    for (size_t f = 0; f < m_n_features; ++f)
                                        new_centroids(c, f) /= static_cast<double>(counts[c]);
                                else
                                    for (size_t f = 0; f < m_n_features; ++f)
                                        new_centroids(c, f) = centroids(c, f);
                            }

                            // Check convergence
                            double shift = 0.0;
                            for (size_t c = 0; c < m_n_clusters; ++c)
                                shift += distance::squared_euclidean(
                                    xt::view(centroids, c, xt::all()),
                                    xt::view(new_centroids, c, xt::all()));
                            centroids = std::move(new_centroids);
                            if (!changed || shift < m_tol)
                                break;
                        }
                        if (inertia < best_inertia)
                        {
                            best_inertia = inertia;
                            m_centroids = centroids;
                            m_labels = labels;
                            m_inertia = inertia;
                        }
                    }
                }

                const Matrix& cluster_centers() const { return m_centroids; }
                const Labels& labels() const { return m_labels; }
                double inertia() const { return m_inertia; }

            private:
                size_t m_n_clusters;
                size_t m_max_iter;
                double m_tol;
                size_t m_n_init;
                std::string m_init;
                std::mt19937 m_rng;

                size_t m_n_samples = 0;
                size_t m_n_features = 0;
                Matrix m_data;
                Matrix m_centroids;
                Labels m_labels;
                double m_inertia = 0.0;

                Matrix initialize_centroids()
                {
                    if (m_init == "random")
                    {
                        std::uniform_int_distribution<size_t> dist(0, m_n_samples - 1);
                        Matrix centroids({m_n_clusters, m_n_features});
                        for (size_t c = 0; c < m_n_clusters; ++c)
                        {
                            size_t idx = dist(m_rng);
                            for (size_t f = 0; f < m_n_features; ++f)
                                centroids(c, f) = m_data(idx, f);
                        }
                        return centroids;
                    }
                    else // k-means++
                    {
                        Matrix centroids({m_n_clusters, m_n_features});
                        std::uniform_int_distribution<size_t> dist(0, m_n_samples - 1);
                        size_t first = dist(m_rng);
                        for (size_t f = 0; f < m_n_features; ++f)
                            centroids(0, f) = m_data(first, f);

                        std::vector<double> min_dist_sq(m_n_samples, std::numeric_limits<double>::max());
                        for (size_t c = 1; c < m_n_clusters; ++c)
                        {
                            // Update minimum distances to existing centroids
                            double sum_dist = 0.0;
                            for (size_t i = 0; i < m_n_samples; ++i)
                            {
                                Vector x = xt::view(m_data, i, xt::all());
                                double dist_sq = distance::squared_euclidean(x, xt::view(centroids, c-1, xt::all()));
                                if (dist_sq < min_dist_sq[i])
                                    min_dist_sq[i] = dist_sq;
                                sum_dist += min_dist_sq[i];
                            }
                            // Sample proportional to distance
                            std::uniform_real_distribution<double> prob_dist(0.0, sum_dist);
                            double threshold = prob_dist(m_rng);
                            double cum = 0.0;
                            size_t chosen = 0;
                            for (size_t i = 0; i < m_n_samples; ++i)
                            {
                                cum += min_dist_sq[i];
                                if (cum >= threshold)
                                {
                                    chosen = i;
                                    break;
                                }
                            }
                            for (size_t f = 0; f < m_n_features; ++f)
                                centroids(c, f) = m_data(chosen, f);
                        }
                        return centroids;
                    }
                }
            };

            // --------------------------------------------------------------------
            // DBSCAN
            // --------------------------------------------------------------------
            class DBSCAN
            {
            public:
                DBSCAN(double eps = 0.5, size_t min_samples = 5, const std::string& metric = "euclidean")
                    : m_eps(eps), m_min_samples(min_samples), m_metric(metric)
                {
                }

                template <class E>
                Labels fit_predict(const xexpression<E>& X)
                {
                    fit(X);
                    return m_labels;
                }

                template <class E>
                void fit(const xexpression<E>& X)
                {
                    const auto& data = X.derived_cast();
                    if (data.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "DBSCAN: input must be 2D");
                    m_n_samples = data.shape()[0];
                    m_n_features = data.shape()[1];
                    m_data = eval(data);
                    m_labels = Labels({m_n_samples}, -1);

                    // Compute neighbors within eps
                    std::vector<std::vector<size_t>> neighbors(m_n_samples);
                    for (size_t i = 0; i < m_n_samples; ++i)
                    {
                        Vector xi = xt::view(m_data, i, xt::all());
                        for (size_t j = 0; j < m_n_samples; ++j)
                        {
                            if (i == j) continue;
                            Vector xj = xt::view(m_data, j, xt::all());
                            double dist = compute_distance(xi, xj);
                            if (dist <= m_eps)
                                neighbors[i].push_back(j);
                        }
                    }

                    int cluster_id = 0;
                    for (size_t i = 0; i < m_n_samples; ++i)
                    {
                        if (m_labels(i) != -1) continue;
                        if (neighbors[i].size() + 1 < m_min_samples)
                        {
                            m_labels(i) = -1; // noise
                            continue;
                        }
                        // Expand cluster
                        m_labels(i) = cluster_id;
                        std::queue<size_t> seeds;
                        for (size_t nb : neighbors[i])
                            seeds.push(nb);
                        while (!seeds.empty())
                        {
                            size_t p = seeds.front(); seeds.pop();
                            if (m_labels(p) == -1)
                            {
                                m_labels(p) = cluster_id;
                                if (neighbors[p].size() + 1 >= m_min_samples)
                                {
                                    for (size_t nb : neighbors[p])
                                        if (m_labels(nb) == -1)
                                            seeds.push(nb);
                                }
                            }
                        }
                        cluster_id++;
                    }
                    m_n_clusters = cluster_id;
                }

                const Labels& labels() const { return m_labels; }
                size_t n_clusters() const { return m_n_clusters; }

            private:
                double m_eps;
                size_t m_min_samples;
                std::string m_metric;
                size_t m_n_samples = 0;
                size_t m_n_features = 0;
                size_t m_n_clusters = 0;
                Matrix m_data;
                Labels m_labels;

                double compute_distance(const Vector& a, const Vector& b) const
                {
                    if (m_metric == "euclidean")
                        return distance::euclidean(a, b);
                    else if (m_metric == "manhattan")
                        return distance::manhattan(a, b);
                    else if (m_metric == "cosine")
                        return distance::cosine(a, b);
                    else
                        return distance::euclidean(a, b);
                }
            };

            // --------------------------------------------------------------------
            // Agglomerative Hierarchical Clustering
            // --------------------------------------------------------------------
            class AgglomerativeClustering
            {
            public:
                enum Linkage { Ward, Complete, Average, Single };

                AgglomerativeClustering(size_t n_clusters = 2, Linkage linkage = Ward,
                                        const std::string& metric = "euclidean")
                    : m_n_clusters(n_clusters), m_linkage(linkage), m_metric(metric)
                {
                }

                template <class E>
                Labels fit_predict(const xexpression<E>& X)
                {
                    fit(X);
                    return m_labels;
                }

                template <class E>
                void fit(const xexpression<E>& X)
                {
                    const auto& data = X.derived_cast();
                    if (data.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "AgglomerativeClustering: input must be 2D");
                    m_n_samples = data.shape()[0];
                    m_data = eval(data);

                    // Initially each point is its own cluster
                    size_t n_clusters = m_n_samples;
                    std::vector<std::vector<size_t>> clusters(m_n_samples);
                    for (size_t i = 0; i < m_n_samples; ++i)
                        clusters[i].push_back(i);

                    // Compute pairwise distances
                    Matrix dist_matrix({m_n_samples, m_n_samples});
                    for (size_t i = 0; i < m_n_samples; ++i)
                    {
                        dist_matrix(i, i) = 0.0;
                        for (size_t j = i + 1; j < m_n_samples; ++j)
                        {
                            double d = compute_distance(xt::view(m_data, i, xt::all()),
                                                        xt::view(m_data, j, xt::all()));
                            dist_matrix(i, j) = d;
                            dist_matrix(j, i) = d;
                        }
                    }

                    // Active clusters set
                    std::set<size_t> active;
                    for (size_t i = 0; i < m_n_samples; ++i) active.insert(i);

                    while (n_clusters > m_n_clusters && active.size() > 1)
                    {
                        // Find closest pair of clusters
                        double min_dist = std::numeric_limits<double>::max();
                        size_t ci = 0, cj = 0;
                        for (size_t i : active)
                        {
                            for (size_t j : active)
                            {
                                if (i >= j) continue;
                                double d = compute_cluster_distance(clusters[i], clusters[j], dist_matrix);
                                if (d < min_dist)
                                {
                                    min_dist = d;
                                    ci = i;
                                    cj = j;
                                }
                            }
                        }
                        // Merge cj into ci
                        clusters[ci].insert(clusters[ci].end(), clusters[cj].begin(), clusters[cj].end());
                        clusters[cj].clear();
                        active.erase(cj);
                        n_clusters--;
                    }

                    // Assign labels
                    m_labels = Labels({m_n_samples}, -1);
                    int label = 0;
                    for (size_t i : active)
                    {
                        for (size_t idx : clusters[i])
                            m_labels(idx) = label;
                        label++;
                    }
                }

                const Labels& labels() const { return m_labels; }

            private:
                size_t m_n_clusters;
                Linkage m_linkage;
                std::string m_metric;
                size_t m_n_samples = 0;
                Matrix m_data;
                Labels m_labels;

                double compute_distance(const Vector& a, const Vector& b) const
                {
                    if (m_metric == "euclidean")
                        return distance::euclidean(a, b);
                    else
                        return distance::euclidean(a, b);
                }

                double compute_cluster_distance(const std::vector<size_t>& c1,
                                                const std::vector<size_t>& c2,
                                                const Matrix& dist) const
                {
                    if (m_linkage == Single)
                    {
                        double min_d = std::numeric_limits<double>::max();
                        for (size_t i : c1)
                            for (size_t j : c2)
                                min_d = std::min(min_d, dist(i, j));
                        return min_d;
                    }
                    else if (m_linkage == Complete)
                    {
                        double max_d = 0.0;
                        for (size_t i : c1)
                            for (size_t j : c2)
                                max_d = std::max(max_d, dist(i, j));
                        return max_d;
                    }
                    else if (m_linkage == Average)
                    {
                        double sum = 0.0;
                        for (size_t i : c1)
                            for (size_t j : c2)
                                sum += dist(i, j);
                        return sum / (c1.size() * c2.size());
                    }
                    else // Ward
                    {
                        // Ward distance = sqrt(2 * n1 * n2 / (n1+n2)) * ||mean1 - mean2||
                        Vector mean1 = xt::zeros<double>({m_data.shape()[1]});
                        for (size_t i : c1)
                            mean1 += xt::view(m_data, i, xt::all());
                        mean1 /= static_cast<double>(c1.size());
                        Vector mean2 = xt::zeros<double>({m_data.shape()[1]});
                        for (size_t i : c2)
                            mean2 += xt::view(m_data, i, xt::all());
                        mean2 /= static_cast<double>(c2.size());
                        double dist_mean = distance::squared_euclidean(mean1, mean2);
                        return std::sqrt(2.0 * c1.size() * c2.size() / (c1.size() + c2.size())) * std::sqrt(dist_mean);
                    }
                }
            };

            // --------------------------------------------------------------------
            // Mean Shift
            // --------------------------------------------------------------------
            class MeanShift
            {
            public:
                MeanShift(double bandwidth = 2.0, size_t max_iter = 300, double tol = 1e-3,
                          bool cluster_all = true, size_t bin_seeding = 0)
                    : m_bandwidth(bandwidth), m_max_iter(max_iter), m_tol(tol),
                      m_cluster_all(cluster_all), m_bin_seeding(bin_seeding)
                {
                }

                template <class E>
                Labels fit_predict(const xexpression<E>& X)
                {
                    fit(X);
                    return m_labels;
                }

                template <class E>
                void fit(const xexpression<E>& X)
                {
                    const auto& data = X.derived_cast();
                    if (data.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "MeanShift: input must be 2D");
                    m_n_samples = data.shape()[0];
                    m_n_features = data.shape()[1];
                    m_data = eval(data);

                    // Initialize seeds (use all points or bin seeding)
                    std::vector<Vector> seeds;
                    if (m_bin_seeding > 0)
                    {
                        // Simple grid seeding - not fully implemented
                        for (size_t i = 0; i < m_n_samples; ++i)
                            seeds.push_back(xt::view(m_data, i, xt::all()));
                    }
                    else
                    {
                        for (size_t i = 0; i < m_n_samples; ++i)
                            seeds.push_back(xt::view(m_data, i, xt::all()));
                    }

                    // Mean shift for each seed
                    std::vector<Vector> shifted_seeds;
                    for (const auto& seed : seeds)
                    {
                        Vector pt = seed;
                        for (size_t iter = 0; iter < m_max_iter; ++iter)
                        {
                            Vector new_pt = xt::zeros<double>({m_n_features});
                            double total_weight = 0.0;
                            for (size_t i = 0; i < m_n_samples; ++i)
                            {
                                Vector xi = xt::view(m_data, i, xt::all());
                                double dist = distance::squared_euclidean(pt, xi);
                                if (dist <= m_bandwidth * m_bandwidth)
                                {
                                    double weight = std::exp(-dist / (2.0 * m_bandwidth * m_bandwidth));
                                    new_pt += weight * xi;
                                    total_weight += weight;
                                }
                            }
                            if (total_weight > 0)
                                new_pt /= total_weight;
                            else
                                break;
                            double shift = distance::euclidean(pt, new_pt);
                            pt = new_pt;
                            if (shift < m_tol) break;
                        }
                        shifted_seeds.push_back(pt);
                    }

                    // Cluster the shifted seeds
                    std::vector<Vector> cluster_centers;
                    std::vector<int> seed_labels(shifted_seeds.size(), -1);
                    for (size_t i = 0; i < shifted_seeds.size(); ++i)
                    {
                        if (seed_labels[i] != -1) continue;
                        cluster_centers.push_back(shifted_seeds[i]);
                        int cluster_id = static_cast<int>(cluster_centers.size()) - 1;
                        for (size_t j = i; j < shifted_seeds.size(); ++j)
                        {
                            if (distance::euclidean(shifted_seeds[i], shifted_seeds[j]) <= m_bandwidth)
                                seed_labels[j] = cluster_id;
                        }
                    }

                    // Assign each original point to nearest cluster center
                    m_labels = Labels({m_n_samples}, -1);
                    for (size_t i = 0; i < m_n_samples; ++i)
                    {
                        Vector xi = xt::view(m_data, i, xt::all());
                        double min_dist = std::numeric_limits<double>::max();
                        int best = -1;
                        for (size_t c = 0; c < cluster_centers.size(); ++c)
                        {
                            double d = distance::euclidean(xi, cluster_centers[c]);
                            if (d < min_dist)
                            {
                                min_dist = d;
                                best = static_cast<int>(c);
                            }
                        }
                        if (best != -1)
                            m_labels(i) = best;
                        else if (m_cluster_all)
                            m_labels(i) = 0;
                    }
                }

                const Labels& labels() const { return m_labels; }

            private:
                double m_bandwidth;
                size_t m_max_iter;
                double m_tol;
                bool m_cluster_all;
                size_t m_bin_seeding;
                size_t m_n_samples = 0;
                size_t m_n_features = 0;
                Matrix m_data;
                Labels m_labels;
            };

            // --------------------------------------------------------------------
            // Gaussian Mixture Model (EM)
            // --------------------------------------------------------------------
            class GaussianMixture
            {
            public:
                GaussianMixture(size_t n_components = 1, const std::string& covariance_type = "full",
                                double tol = 1e-3, size_t max_iter = 100, size_t n_init = 1,
                                const std::string& init_params = "kmeans", size_t random_state = 42)
                    : m_n_components(n_components), m_covariance_type(covariance_type),
                      m_tol(tol), m_max_iter(max_iter), m_n_init(n_init),
                      m_init_params(init_params), m_rng(random_state)
                {
                }

                template <class E>
                Labels fit_predict(const xexpression<E>& X)
                {
                    fit(X);
                    return m_labels;
                }

                template <class E>
                void fit(const xexpression<E>& X)
                {
                    const auto& data = X.derived_cast();
                    if (data.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "GaussianMixture: input must be 2D");
                    m_n_samples = data.shape()[0];
                    m_n_features = data.shape()[1];
                    m_data = eval(data);

                    double best_log_likelihood = -std::numeric_limits<double>::infinity();
                    for (size_t init_run = 0; init_run < m_n_init; ++init_run)
                    {
                        initialize_parameters();
                        double log_likelihood = 0.0;
                        for (size_t iter = 0; iter < m_max_iter; ++iter)
                        {
                            // E-step: compute responsibilities
                            m_resp = Matrix({m_n_samples, m_n_components});
                            for (size_t i = 0; i < m_n_samples; ++i)
                            {
                                Vector x = xt::view(m_data, i, xt::all());
                                double sum = 0.0;
                                for (size_t k = 0; k < m_n_components; ++k)
                                {
                                    double pdf = gaussian_pdf(x, k);
                                    double resp = m_weights(k) * pdf;
                                    m_resp(i, k) = resp;
                                    sum += resp;
                                }
                                if (sum > 0)
                                    for (size_t k = 0; k < m_n_components; ++k)
                                        m_resp(i, k) /= sum;
                            }

                            // M-step: update parameters
                            Vector nk({m_n_components}, 0.0);
                            for (size_t i = 0; i < m_n_samples; ++i)
                                for (size_t k = 0; k < m_n_components; ++k)
                                    nk(k) += m_resp(i, k);

                            // Update weights
                            m_weights = nk / static_cast<double>(m_n_samples);

                            // Update means
                            m_means = Matrix({m_n_components, m_n_features}, 0.0);
                            for (size_t k = 0; k < m_n_components; ++k)
                            {
                                if (nk(k) < 1e-10) continue;
                                for (size_t i = 0; i < m_n_samples; ++i)
                                {
                                    for (size_t f = 0; f < m_n_features; ++f)
                                        m_means(k, f) += m_resp(i, k) * m_data(i, f);
                                }
                                for (size_t f = 0; f < m_n_features; ++f)
                                    m_means(k, f) /= nk(k);
                            }

                            // Update covariances
                            if (m_covariance_type == "full")
                            {
                                m_covariances = std::vector<Matrix>(m_n_components,
                                    Matrix({m_n_features, m_n_features}, 0.0));
                                for (size_t k = 0; k < m_n_components; ++k)
                                {
                                    if (nk(k) < 1e-10) continue;
                                    for (size_t i = 0; i < m_n_samples; ++i)
                                    {
                                        Vector x = xt::view(m_data, i, xt::all());
                                        Vector diff = x - xt::view(m_means, k, xt::all());
                                        double resp = m_resp(i, k);
                                        for (size_t f1 = 0; f1 < m_n_features; ++f1)
                                            for (size_t f2 = 0; f2 < m_n_features; ++f2)
                                                m_covariances[k](f1, f2) += resp * diff(f1) * diff(f2);
                                    }
                                    m_covariances[k] /= nk(k);
                                    // Regularization
                                    for (size_t d = 0; d < m_n_features; ++d)
                                        m_covariances[k](d, d) += 1e-6;
                                }
                            }
                            else if (m_covariance_type == "diag")
                            {
                                m_cov_diag = std::vector<Vector>(m_n_components, Vector({m_n_features}, 0.0));
                                for (size_t k = 0; k < m_n_components; ++k)
                                {
                                    if (nk(k) < 1e-10) continue;
                                    for (size_t i = 0; i < m_n_samples; ++i)
                                    {
                                        Vector x = xt::view(m_data, i, xt::all());
                                        Vector diff = x - xt::view(m_means, k, xt::all());
                                        for (size_t f = 0; f < m_n_features; ++f)
                                            m_cov_diag[k](f) += m_resp(i, k) * diff(f) * diff(f);
                                    }
                                    for (size_t f = 0; f < m_n_features; ++f)
                                    {
                                        m_cov_diag[k](f) = m_cov_diag[k](f) / nk(k) + 1e-6;
                                    }
                                }
                            }

                            // Compute log likelihood
                            double new_log_likelihood = 0.0;
                            for (size_t i = 0; i < m_n_samples; ++i)
                            {
                                Vector x = xt::view(m_data, i, xt::all());
                                double prob = 0.0;
                                for (size_t k = 0; k < m_n_components; ++k)
                                    prob += m_weights(k) * gaussian_pdf(x, k);
                                new_log_likelihood += std::log(prob + 1e-12);
                            }
                            if (std::abs(new_log_likelihood - log_likelihood) < m_tol * std::abs(new_log_likelihood))
                                break;
                            log_likelihood = new_log_likelihood;
                        }
                        if (log_likelihood > best_log_likelihood)
                        {
                            best_log_likelihood = log_likelihood;
                            m_best_weights = m_weights;
                            m_best_means = m_means;
                            m_best_covs = m_covariances;
                            m_best_cov_diag = m_cov_diag;
                            // Assign labels
                            m_labels = Labels({m_n_samples});
                            for (size_t i = 0; i < m_n_samples; ++i)
                            {
                                m_labels(i) = static_cast<int>(std::distance(m_resp.begin(),
                                    std::max_element(m_resp.begin() + i * m_n_components,
                                                     m_resp.begin() + (i+1) * m_n_components)) % m_n_components);
                            }
                        }
                    }
                }

                const Labels& labels() const { return m_labels; }

            private:
                size_t m_n_components;
                std::string m_covariance_type;
                double m_tol;
                size_t m_max_iter;
                size_t m_n_init;
                std::string m_init_params;
                std::mt19937 m_rng;

                size_t m_n_samples = 0;
                size_t m_n_features = 0;
                Matrix m_data;

                Vector m_weights;
                Matrix m_means;
                std::vector<Matrix> m_covariances;
                std::vector<Vector> m_cov_diag;
                Matrix m_resp;
                Labels m_labels;

                Vector m_best_weights;
                Matrix m_best_means;
                std::vector<Matrix> m_best_covs;
                std::vector<Vector> m_best_cov_diag;

                void initialize_parameters()
                {
                    m_weights = Vector({m_n_components}, 1.0 / m_n_components);
                    m_means = Matrix({m_n_components, m_n_features});
                    // Initialize means with random samples or k-means
                    if (m_init_params == "kmeans")
                    {
                        KMeans km(m_n_components, 10, 1e-4, 1, "k-means++", m_rng());
                        km.fit(m_data);
                        m_means = km.cluster_centers();
                    }
                    else
                    {
                        std::uniform_int_distribution<size_t> dist(0, m_n_samples - 1);
                        for (size_t k = 0; k < m_n_components; ++k)
                        {
                            size_t idx = dist(m_rng);
                            for (size_t f = 0; f < m_n_features; ++f)
                                m_means(k, f) = m_data(idx, f);
                        }
                    }
                    // Initialize covariances
                    if (m_covariance_type == "full")
                    {
                        m_covariances.resize(m_n_components);
                        Matrix init_cov = xt::eye<double>(m_n_features);
                        for (size_t k = 0; k < m_n_components; ++k)
                            m_covariances[k] = init_cov;
                    }
                    else if (m_covariance_type == "diag")
                    {
                        m_cov_diag.resize(m_n_components);
                        Vector init_diag({m_n_features}, 1.0);
                        for (size_t k = 0; k < m_n_components; ++k)
                            m_cov_diag[k] = init_diag;
                    }
                }

                double gaussian_pdf(const Vector& x, size_t comp_idx) const
                {
                    Vector mean = xt::view(m_means, comp_idx, xt::all());
                    Vector diff = x - mean;
                    if (m_covariance_type == "full")
                    {
                        const Matrix& cov = m_covariances[comp_idx];
                        // Compute log pdf: -0.5 * (log(det) + diff^T * inv(cov) * diff + d*log(2pi))
                        double sign;
                        double log_det = xt::linalg::slogdet(cov).second;
                        Matrix cov_inv = xt::linalg::inv(cov);
                        double quad = xt::linalg::dot(xt::linalg::dot(diff, cov_inv), diff)();
                        return std::exp(-0.5 * (log_det + quad + m_n_features * std::log(2.0 * M_PI)));
                    }
                    else if (m_covariance_type == "diag")
                    {
                        const Vector& cov_diag = m_cov_diag[comp_idx];
                        double log_det = xt::sum(xt::log(cov_diag))();
                        double quad = xt::sum(diff * diff / cov_diag)();
                        return std::exp(-0.5 * (log_det + quad + m_n_features * std::log(2.0 * M_PI)));
                    }
                    else // spherical
                    {
                        double var = m_cov_diag[comp_idx](0);
                        double quad = xt::sum(diff * diff)() / var;
                        return std::exp(-0.5 * (m_n_features * std::log(var) + quad + m_n_features * std::log(2.0 * M_PI)));
                    }
                }
            };

            // --------------------------------------------------------------------
            // Spectral Clustering
            // --------------------------------------------------------------------
            class SpectralClustering
            {
            public:
                SpectralClustering(size_t n_clusters = 8, double gamma = 1.0,
                                   const std::string& affinity = "rbf", size_t random_state = 42)
                    : m_n_clusters(n_clusters), m_gamma(gamma), m_affinity(affinity), m_rng(random_state)
                {
                }

                template <class E>
                Labels fit_predict(const xexpression<E>& X)
                {
                    fit(X);
                    return m_labels;
                }

                template <class E>
                void fit(const xexpression<E>& X)
                {
                    const auto& data = X.derived_cast();
                    if (data.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "SpectralClustering: input must be 2D");
                    m_n_samples = data.shape()[0];
                    m_data = eval(data);

                    // Compute affinity matrix
                    Matrix affinity_mat = compute_affinity();

                    // Compute Laplacian
                    Matrix degree = xt::zeros<double>({m_n_samples, m_n_samples});
                    for (size_t i = 0; i < m_n_samples; ++i)
                    {
                        double sum = xt::sum(xt::view(affinity_mat, i, xt::all()))();
                        degree(i, i) = 1.0 / std::sqrt(sum + 1e-10);
                    }
                    Matrix L = xt::linalg::dot(degree, xt::linalg::dot(affinity_mat, degree));

                    // Eigenvalue decomposition
                    auto [eigvals, eigvecs] = xt::linalg::eigh(L);
                    // Take eigenvectors corresponding to largest k eigenvalues
                    Matrix embedding({m_n_samples, m_n_clusters});
                    for (size_t i = 0; i < m_n_samples; ++i)
                        for (size_t j = 0; j < m_n_clusters; ++j)
                            embedding(i, j) = eigvecs(i, m_n_samples - 1 - j);

                    // Normalize rows
                    for (size_t i = 0; i < m_n_samples; ++i)
                    {
                        double norm = std::sqrt(xt::sum(xt::view(embedding, i, xt::all()) *
                                                        xt::view(embedding, i, xt::all()))());
                        if (norm > 0)
                            for (size_t j = 0; j < m_n_clusters; ++j)
                                embedding(i, j) /= norm;
                    }

                    // Cluster with KMeans
                    KMeans km(m_n_clusters, 100, 1e-4, 10, "k-means++", m_rng());
                    m_labels = km.fit_predict(embedding);
                }

                const Labels& labels() const { return m_labels; }

            private:
                size_t m_n_clusters;
                double m_gamma;
                std::string m_affinity;
                std::mt19937 m_rng;
                size_t m_n_samples = 0;
                Matrix m_data;
                Labels m_labels;

                Matrix compute_affinity()
                {
                    Matrix aff({m_n_samples, m_n_samples});
                    if (m_affinity == "rbf")
                    {
                        for (size_t i = 0; i < m_n_samples; ++i)
                        {
                            aff(i, i) = 1.0;
                            Vector xi = xt::view(m_data, i, xt::all());
                            for (size_t j = i + 1; j < m_n_samples; ++j)
                            {
                                Vector xj = xt::view(m_data, j, xt::all());
                                double dist_sq = distance::squared_euclidean(xi, xj);
                                double sim = std::exp(-m_gamma * dist_sq);
                                aff(i, j) = sim;
                                aff(j, i) = sim;
                            }
                        }
                    }
                    else // nearest neighbors
                    {
                        // Simplified: use RBF with small gamma or identity
                        aff = xt::eye<double>(m_n_samples);
                    }
                    return aff;
                }
            };

            // --------------------------------------------------------------------
            // Clustering evaluation metrics
            // --------------------------------------------------------------------
            namespace metrics
            {
                // Silhouette Score
                template <class E>
                inline double silhouette_score(const xexpression<E>& X, const Labels& labels,
                                               const std::string& metric = "euclidean")
                {
                    const auto& data = X.derived_cast();
                    size_t n_samples = data.shape()[0];
                    if (labels.size() != n_samples)
                        XTENSOR_THROW(std::invalid_argument, "silhouette_score: labels size mismatch");

                    // Get unique labels
                    std::set<int> unique_labels(labels.begin(), labels.end());
                    if (unique_labels.size() < 2) return -1.0;

                    double total_score = 0.0;
                    for (size_t i = 0; i < n_samples; ++i)
                    {
                        int label_i = labels(i);
                        Vector xi = xt::view(data, i, xt::all());

                        // Compute a: mean distance to same cluster
                        double a = 0.0;
                        size_t same_count = 0;
                        for (size_t j = 0; j < n_samples; ++j)
                        {
                            if (i == j) continue;
                            if (labels(j) == label_i)
                            {
                                a += distance::euclidean(xi, xt::view(data, j, xt::all()));
                                same_count++;
                            }
                        }
                        a = (same_count > 0) ? a / same_count : 0.0;

                        // Compute b: min mean distance to other clusters
                        double b = std::numeric_limits<double>::max();
                        for (int other_label : unique_labels)
                        {
                            if (other_label == label_i) continue;
                            double sum_dist = 0.0;
                            size_t other_count = 0;
                            for (size_t j = 0; j < n_samples; ++j)
                            {
                                if (labels(j) == other_label)
                                {
                                    sum_dist += distance::euclidean(xi, xt::view(data, j, xt::all()));
                                    other_count++;
                                }
                            }
                            if (other_count > 0)
                            {
                                double mean_dist = sum_dist / other_count;
                                if (mean_dist < b) b = mean_dist;
                            }
                        }
                        if (b > 1e-12)
                        {
                            double s = (b - a) / std::max(a, b);
                            total_score += s;
                        }
                        else
                        {
                            total_score += 0.0;
                        }
                    }
                    return total_score / n_samples;
                }

                // Davies-Bouldin Index
                template <class E>
                inline double davies_bouldin_score(const xexpression<E>& X, const Labels& labels)
                {
                    const auto& data = X.derived_cast();
                    size_t n_samples = data.shape()[0];
                    std::set<int> unique_labels(labels.begin(), labels.end());
                    size_t n_clusters = unique_labels.size();
                    if (n_clusters < 2) return 0.0;

                    // Compute cluster centers and average distances
                    std::vector<Vector> centers;
                    std::vector<double> avg_dist;
                    for (int lbl : unique_labels)
                    {
                        Vector center = xt::zeros<double>({data.shape()[1]});
                        size_t count = 0;
                        for (size_t i = 0; i < n_samples; ++i)
                        {
                            if (labels(i) == lbl)
                            {
                                center += xt::view(data, i, xt::all());
                                count++;
                            }
                        }
                        center /= static_cast<double>(count);
                        centers.push_back(center);

                        double sum_dist = 0.0;
                        for (size_t i = 0; i < n_samples; ++i)
                        {
                            if (labels(i) == lbl)
                                sum_dist += distance::euclidean(xt::view(data, i, xt::all()), center);
                        }
                        avg_dist.push_back(sum_dist / count);
                    }

                    double db = 0.0;
                    for (size_t i = 0; i < n_clusters; ++i)
                    {
                        double max_ratio = 0.0;
                        for (size_t j = 0; j < n_clusters; ++j)
                        {
                            if (i == j) continue;
                            double d_ij = distance::euclidean(centers[i], centers[j]);
                            if (d_ij > 0)
                            {
                                double ratio = (avg_dist[i] + avg_dist[j]) / d_ij;
                                if (ratio > max_ratio) max_ratio = ratio;
                            }
                        }
                        db += max_ratio;
                    }
                    return db / n_clusters;
                }

                // Calinski-Harabasz Index
                template <class E>
                inline double calinski_harabasz_score(const xexpression<E>& X, const Labels& labels)
                {
                    const auto& data = X.derived_cast();
                    size_t n_samples = data.shape()[0];
                    size_t n_features = data.shape()[1];
                    std::set<int> unique_labels(labels.begin(), labels.end());
                    size_t n_clusters = unique_labels.size();
                    if (n_clusters < 2) return 0.0;

                    Vector overall_mean = xt::mean(data, {0});
                    double ssb = 0.0; // between-cluster sum of squares
                    double ssw = 0.0; // within-cluster sum of squares

                    for (int lbl : unique_labels)
                    {
                        std::vector<size_t> indices;
                        for (size_t i = 0; i < n_samples; ++i)
                            if (labels(i) == lbl) indices.push_back(i);
                        size_t nk = indices.size();
                        if (nk == 0) continue;

                        Vector cluster_mean = xt::zeros<double>({n_features});
                        for (size_t i : indices)
                            cluster_mean += xt::view(data, i, xt::all());
                        cluster_mean /= static_cast<double>(nk);

                        Vector diff = cluster_mean - overall_mean;
                        ssb += nk * xt::sum(diff * diff)();

                        for (size_t i : indices)
                        {
                            Vector diff2 = xt::view(data, i, xt::all()) - cluster_mean;
                            ssw += xt::sum(diff2 * diff2)();
                        }
                    }
                    if (ssw == 0) return std::numeric_limits<double>::infinity();
                    return (ssb / (n_clusters - 1)) / (ssw / (n_samples - n_clusters));
                }

                // Adjusted Rand Index (using existing metrics implementation or own)
                template <class E>
                inline double adjusted_rand_score(const xexpression<E>& labels_true,
                                                  const xexpression<E>& labels_pred)
                {
                    return metrics::adjusted_rand_score(labels_true, labels_pred);
                }

                // Normalized Mutual Information
                template <class E1, class E2>
                inline double normalized_mutual_info_score(const xexpression<E1>& labels_true,
                                                           const xexpression<E2>& labels_pred,
                                                           const std::string& average_method = "arithmetic")
                {
                    return metrics::normalized_mutual_info_score(labels_true, labels_pred, average_method);
                }

                // Homogeneity, Completeness, V-measure
                template <class E1, class E2>
                inline auto homogeneity_completeness_v_measure(const xexpression<E1>& labels_true,
                                                               const xexpression<E2>& labels_pred)
                {
                    // Simplified: use mutual information
                    double mi = normalized_mutual_info_score(labels_true, labels_pred, "arithmetic");
                    // Approximate
                    return std::make_tuple(mi, mi, mi);
                }
            }

        } // namespace cluster

        // Bring clustering classes and functions into xt namespace
        using cluster::KMeans;
        using cluster::DBSCAN;
        using cluster::AgglomerativeClustering;
        using cluster::MeanShift;
        using cluster::GaussianMixture;
        using cluster::SpectralClustering;
        using cluster::metrics::silhouette_score;
        using cluster::metrics::davies_bouldin_score;
        using cluster::metrics::calinski_harabasz_score;
        using cluster::metrics::adjusted_rand_score;
        using cluster::metrics::normalized_mutual_info_score;
        using cluster::metrics::homogeneity_completeness_v_measure;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XCLUSTER_HPP

// math/xcluster.hpp