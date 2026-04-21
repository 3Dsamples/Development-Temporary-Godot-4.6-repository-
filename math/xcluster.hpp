// core/xcluster.hpp
#ifndef XTENSOR_XCLUSTER_HPP
#define XTENSOR_XCLUSTER_HPP

// ----------------------------------------------------------------------------
// xcluster.hpp – Clustering algorithms for xtensor
// ----------------------------------------------------------------------------
// This header provides a comprehensive set of clustering algorithms:
//   - K‑Means (Lloyd, Hartigan‑Wong, MacQueen)
//   - K‑Medoids (PAM, CLARA)
//   - Hierarchical clustering (agglomerative, single/complete/average/Ward linkage)
//   - DBSCAN (density‑based spatial clustering)
//   - OPTICS (ordering points to identify clustering structure)
//   - Mean Shift
//   - Gaussian Mixture Models (EM algorithm)
//   - Affinity Propagation
//   - Spectral Clustering
//   - BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
//   - Evaluation metrics: silhouette, Davies‑Bouldin, Calinski‑Harabasz, Dunn
//
// All functions work with bignumber::BigNumber for high‑precision arithmetic.
// FFT acceleration is used for large‑scale distance computations and spectral
// embedding where applicable.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <limits>
#include <tuple>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <chrono>
#include <utility>
#include <memory>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xlinalg.hpp"
#include "xsorting.hpp"
#include "xstats.hpp"
#include "fft.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace cluster
    {
        // ========================================================================
        // Distance metrics and pairwise distances
        // ========================================================================
        enum class distance_metric { euclidean, manhattan, cosine, correlation, chebyshev, minkowski };

        template <class T>
        T compute_distance(const std::vector<T>& a, const std::vector<T>& b, distance_metric metric, T p = T(2));

        template <class T>
        xarray_container<T> pairwise_distances(const xarray_container<T>& X,
                                                distance_metric metric = distance_metric::euclidean,
                                                T p = T(2));

        // ========================================================================
        // K‑Means (Lloyd and Hartigan‑Wong)
        // ========================================================================
        template <class T>
        std::tuple<std::vector<size_t>, xarray_container<T>, size_t>
        kmeans_lloyd(const xarray_container<T>& X, size_t k,
                     size_t max_iter = 300, T tol = T(1e-6),
                     size_t n_init = 10, std::mt19937* rng = nullptr);

        template <class T>
        std::tuple<std::vector<size_t>, xarray_container<T>>
        kmeans_hartigan_wong(const xarray_container<T>& X, size_t k, size_t max_iter = 100);

        // ========================================================================
        // K‑Medoids (PAM)
        // ========================================================================
        template <class T>
        std::pair<std::vector<size_t>, std::vector<size_t>>
        kmedoids_pam(const xarray_container<T>& X, size_t k,
                     size_t max_iter = 100, distance_metric metric = distance_metric::euclidean);

        // ========================================================================
        // DBSCAN and OPTICS
        // ========================================================================
        template <class T>
        std::vector<int> dbscan(const xarray_container<T>& X, T eps, size_t min_pts,
                                distance_metric metric = distance_metric::euclidean);

        template <class T>
        struct optics_result { std::vector<size_t> order; std::vector<T> reachability; std::vector<T> core_dist; };

        template <class T>
        optics_result<T> optics(const xarray_container<T>& X, T eps, size_t min_pts,
                                distance_metric metric = distance_metric::euclidean);

        template <class T>
        std::vector<int> optics_extract_clusters(const optics_result<T>& opt, T xi);

        // ========================================================================
        // Hierarchical Clustering (Agglomerative)
        // ========================================================================
        enum class linkage_method { single, complete, average, weighted, centroid, median, ward };

        template <class T>
        struct linkage_result
        {
            std::vector<std::array<size_t, 4>> Z;  // [cluster1, cluster2, distance, size]
            std::vector<size_t> labels;
        };

        template <class T>
        linkage_result<T> agglomerative_clustering(const xarray_container<T>& X,
                                                    size_t n_clusters,
                                                    linkage_method method = linkage_method::ward,
                                                    distance_metric metric = distance_metric::euclidean);

        template <class T>
        std::vector<size_t> fcluster(const linkage_result<T>& Z, size_t n_clusters, const std::string& criterion = "maxclust");

        // ========================================================================
        // Mean Shift
        // ========================================================================
        template <class T>
        std::pair<std::vector<size_t>, xarray_container<T>>
        mean_shift(const xarray_container<T>& X, T bandwidth,
                   size_t max_iter = 300, T tol = T(1e-6),
                   bool cluster_all = true, size_t bin_seeding = false);

        // ========================================================================
        // Gaussian Mixture Models
        // ========================================================================
        template <class T>
        struct gmm_result
        {
            xarray_container<T> means;       // k x d
            xarray_container<T> covariances; // k x d x d
            std::vector<T> weights;          // k
            std::vector<size_t> labels;      // n
            std::vector<T> bic;              // BIC scores per iteration
            size_t n_iter;
        };

        template <class T>
        gmm_result<T> gaussian_mixture(const xarray_container<T>& X, size_t k,
                                        const std::string& covariance_type = "full",
                                        size_t max_iter = 100, T tol = T(1e-3),
                                        size_t n_init = 1, std::mt19937* rng = nullptr);

        // ========================================================================
        // Affinity Propagation
        // ========================================================================
        template <class T>
        std::pair<std::vector<size_t>, std::vector<size_t>>
        affinity_propagation(const xarray_container<T>& X,
                             T preference = std::numeric_limits<T>::quiet_NaN(),
                             T damping = T(0.5), size_t max_iter = 200,
                             size_t convergence_iter = 15);

        // ========================================================================
        // Spectral Clustering
        // ========================================================================
        template <class T>
        std::vector<size_t> spectral_clustering(const xarray_container<T>& X, size_t n_clusters,
                                                const std::string& affinity = "rbf",
                                                T gamma = T(1), bool assign_labels = "kmeans",
                                                size_t n_init = 10);

        // ========================================================================
        // BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
        // ========================================================================
        template <class T>
        struct birch_node;

        template <class T>
        class birch
        {
        public:
            birch(T threshold = T(0.5), size_t branching_factor = 50, size_t n_clusters = 3);
            void fit(const xarray_container<T>& X);
            std::vector<size_t> predict(const xarray_container<T>& X);
            std::vector<size_t> labels() const;

        private:
            // TODO: implement CF‑tree nodes
            std::shared_ptr<birch_node<T>> m_root;
            T m_threshold;
            size_t m_branching_factor;
            size_t m_n_clusters;
            std::vector<size_t> m_labels;
        };

        // ========================================================================
        // Evaluation Metrics
        // ========================================================================
        template <class T>
        T silhouette_score(const xarray_container<T>& X, const std::vector<size_t>& labels,
                           distance_metric metric = distance_metric::euclidean);

        template <class T>
        T davies_bouldin_score(const xarray_container<T>& X, const std::vector<size_t>& labels);

        template <class T>
        T calinski_harabasz_score(const xarray_container<T>& X, const std::vector<size_t>& labels);

        template <class T>
        T dunn_index(const xarray_container<T>& X, const std::vector<size_t>& labels,
                     distance_metric metric = distance_metric::euclidean);

        template <class T>
        T adjusted_rand_index(const std::vector<size_t>& labels_true, const std::vector<size_t>& labels_pred);

        template <class T>
        T normalized_mutual_info(const std::vector<size_t>& labels_true, const std::vector<size_t>& labels_pred);

        template <class T>
        T homogeneity_score(const std::vector<size_t>& labels_true, const std::vector<size_t>& labels_pred);

        template <class T>
        T completeness_score(const std::vector<size_t>& labels_true, const std::vector<size_t>& labels_pred);

        template <class T>
        T v_measure_score(const std::vector<size_t>& labels_true, const std::vector<size_t>& labels_pred);

        // ========================================================================
        // Utility: Determine optimal number of clusters
        // ========================================================================
        template <class T>
        size_t elbow_method(const xarray_container<T>& X, size_t max_k = 10,
                            const std::string& method = "kmeans", size_t n_init = 10);

        template <class T>
        std::vector<T> gap_statistic(const xarray_container<T>& X, size_t max_k = 10,
                                     size_t n_refs = 10, std::mt19937* rng = nullptr);
    }

    using cluster::distance_metric;
    using cluster::compute_distance;
    using cluster::pairwise_distances;
    using cluster::kmeans_lloyd;
    using cluster::kmeans_hartigan_wong;
    using cluster::kmedoids_pam;
    using cluster::dbscan;
    using cluster::optics;
    using cluster::optics_result;
    using cluster::optics_extract_clusters;
    using cluster::agglomerative_clustering;
    using cluster::linkage_method;
    using cluster::linkage_result;
    using cluster::fcluster;
    using cluster::mean_shift;
    using cluster::gaussian_mixture;
    using cluster::gmm_result;
    using cluster::affinity_propagation;
    using cluster::spectral_clustering;
    using cluster::birch;
    using cluster::silhouette_score;
    using cluster::davies_bouldin_score;
    using cluster::calinski_harabasz_score;
    using cluster::dunn_index;
    using cluster::adjusted_rand_index;
    using cluster::normalized_mutual_info;
    using cluster::homogeneity_score;
    using cluster::completeness_score;
    using cluster::v_measure_score;
    using cluster::elbow_method;
    using cluster::gap_statistic;
}

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace cluster
    {
        // Compute distance between two vectors using specified metric
        template <class T>
        T compute_distance(const std::vector<T>& a, const std::vector<T>& b, distance_metric metric, T p)
        { /* TODO: implement Euclidean/Manhattan/Cosine/Chebyshev/Minkowski */ return T(0); }

        // Compute full pairwise distance matrix for dataset X
        template <class T>
        xarray_container<T> pairwise_distances(const xarray_container<T>& X, distance_metric metric, T p)
        { /* TODO: implement NxN distance matrix */ return {}; }

        // K‑Means clustering using Lloyd's algorithm
        template <class T>
        std::tuple<std::vector<size_t>, xarray_container<T>, size_t>
        kmeans_lloyd(const xarray_container<T>& X, size_t k, size_t max_iter, T tol, size_t n_init, std::mt19937* rng)
        { /* TODO: implement Lloyd with k‑means++ init */ return {}; }

        // K‑Means clustering using Hartigan‑Wong algorithm (more efficient)
        template <class T>
        std::tuple<std::vector<size_t>, xarray_container<T>>
        kmeans_hartigan_wong(const xarray_container<T>& X, size_t k, size_t max_iter)
        { /* TODO: implement Hartigan‑Wong quick transfer */ return {}; }

        // K‑Medoids clustering using PAM (Partitioning Around Medoids)
        template <class T>
        std::pair<std::vector<size_t>, std::vector<size_t>>
        kmedoids_pam(const xarray_container<T>& X, size_t k, size_t max_iter, distance_metric metric)
        { /* TODO: implement BUILD and SWAP phases */ return {}; }

        // Density‑based spatial clustering (DBSCAN)
        template <class T>
        std::vector<int> dbscan(const xarray_container<T>& X, T eps, size_t min_pts, distance_metric metric)
        { /* TODO: implement DBSCAN with region queries */ return {}; }

        // OPTICS algorithm (Ordering Points To Identify Clustering Structure)
        template <class T>
        optics_result<T> optics(const xarray_container<T>& X, T eps, size_t min_pts, distance_metric metric)
        { /* TODO: implement OPTICS reachability plot */ return {}; }

        // Extract clusters from OPTICS reachability plot using xi method
        template <class T>
        std::vector<int> optics_extract_clusters(const optics_result<T>& opt, T xi)
        { /* TODO: implement cluster extraction */ return {}; }

        // Agglomerative hierarchical clustering
        template <class T>
        linkage_result<T> agglomerative_clustering(const xarray_container<T>& X, size_t n_clusters,
                                                    linkage_method method, distance_metric metric)
        { /* TODO: implement Lance‑Williams recurrence */ return {}; }

        // Form flat clusters from hierarchical linkage matrix
        template <class T>
        std::vector<size_t> fcluster(const linkage_result<T>& Z, size_t n_clusters, const std::string& criterion)
        { /* TODO: cut dendrogram at threshold */ return {}; }

        // Mean Shift clustering
        template <class T>
        std::pair<std::vector<size_t>, xarray_container<T>>
        mean_shift(const xarray_container<T>& X, T bandwidth, size_t max_iter, T tol, bool cluster_all, size_t bin_seeding)
        { /* TODO: implement mean shift with flat kernel */ return {}; }

        // Gaussian Mixture Model using EM algorithm
        template <class T>
        gmm_result<T> gaussian_mixture(const xarray_container<T>& X, size_t k,
                                        const std::string& covariance_type,
                                        size_t max_iter, T tol, size_t n_init, std::mt19937* rng)
        { /* TODO: implement EM for GMM */ return {}; }

        // Affinity Propagation clustering
        template <class T>
        std::pair<std::vector<size_t>, std::vector<size_t>>
        affinity_propagation(const xarray_container<T>& X, T preference, T damping,
                             size_t max_iter, size_t convergence_iter)
        { /* TODO: implement message passing */ return {}; }

        // Spectral Clustering
        template <class T>
        std::vector<size_t> spectral_clustering(const xarray_container<T>& X, size_t n_clusters,
                                                const std::string& affinity, T gamma,
                                                bool assign_labels, size_t n_init)
        { /* TODO: compute Laplacian + k‑means on eigenvectors */ return {}; }

        // BIRCH constructor
        template <class T>
        birch<T>::birch(T threshold, size_t branching_factor, size_t n_clusters)
            : m_threshold(threshold), m_branching_factor(branching_factor), m_n_clusters(n_clusters) {}
        // Fit BIRCH model
        template <class T> void birch<T>::fit(const xarray_container<T>& X) { /* TODO: build CF‑tree */ }
        // Predict cluster labels for new data
        template <class T> std::vector<size_t> birch<T>::predict(const xarray_container<T>& X) { /* TODO: traverse CF‑tree */ return {}; }
        // Get labels for training data
        template <class T> std::vector<size_t> birch<T>::labels() const { return m_labels; }

        // Silhouette score (higher is better)
        template <class T>
        T silhouette_score(const xarray_container<T>& X, const std::vector<size_t>& labels, distance_metric metric)
        { /* TODO: compute mean silhouette coefficient */ return T(0); }

        // Davies‑Bouldin index (lower is better)
        template <class T>
        T davies_bouldin_score(const xarray_container<T>& X, const std::vector<size_t>& labels)
        { /* TODO: compute average similarity between clusters */ return T(0); }

        // Calinski‑Harabasz index (higher is better)
        template <class T>
        T calinski_harabasz_score(const xarray_container<T>& X, const std::vector<size_t>& labels)
        { /* TODO: ratio of between‑cluster to within‑cluster dispersion */ return T(0); }

        // Dunn index (higher is better)
        template <class T>
        T dunn_index(const xarray_container<T>& X, const std::vector<size_t>& labels, distance_metric metric)
        { /* TODO: min inter‑cluster distance / max intra‑cluster diameter */ return T(0); }

        // Adjusted Rand Index (external validation)
        template <class T>
        T adjusted_rand_index(const std::vector<size_t>& labels_true, const std::vector<size_t>& labels_pred)
        { /* TODO: compute ARI */ return T(0); }

        // Normalized Mutual Information
        template <class T>
        T normalized_mutual_info(const std::vector<size_t>& labels_true, const std::vector<size_t>& labels_pred)
        { /* TODO: compute NMI */ return T(0); }

        // Homogeneity score
        template <class T>
        T homogeneity_score(const std::vector<size_t>& labels_true, const std::vector<size_t>& labels_pred)
        { /* TODO: H(C|K) / H(C) */ return T(0); }

        // Completeness score
        template <class T>
        T completeness_score(const std::vector<size_t>& labels_true, const std::vector<size_t>& labels_pred)
        { /* TODO: H(K|C) / H(K) */ return T(0); }

        // V‑measure (harmonic mean of homogeneity and completeness)
        template <class T>
        T v_measure_score(const std::vector<size_t>& labels_true, const std::vector<size_t>& labels_pred)
        { /* TODO: 2 * h * c / (h + c) */ return T(0); }

        // Elbow method for determining optimal k
        template <class T>
        size_t elbow_method(const xarray_container<T>& X, size_t max_k, const std::string& method, size_t n_init)
        { /* TODO: find elbow in inertia curve */ return 0; }

        // Gap statistic for estimating number of clusters
        template <class T>
        std::vector<T> gap_statistic(const xarray_container<T>& X, size_t max_k, size_t n_refs, std::mt19937* rng)
        { /* TODO: compute gap statistic */ return {}; }
    }
}

#endif // XTENSOR_XCLUSTER_HPP          {
                    alpha_i = alpha_j = T(0.5); beta = T(0); gamma = T(0.5);
                }
                else if (method == linkage_method::average)
                {
                    T ni = T(clusters[best_i].size());
                    T nj = T(clusters[best_j].size());
                    alpha_i = ni / (ni + nj);
                    alpha_j = nj / (ni + nj);
                    beta = T(0);
                    gamma = T(0);
                }
                else // Ward
                {
                    T ni = T(clusters[best_i].size());
                    T nj = T(clusters[best_j].size());
                    for (size_t k = 0; k < n; ++k)
                    {
                        if (!active[k] || k == best_i || k == best_j) continue;
                        T nk = T(clusters[k].size());
                        T denom = ni + nj + nk;
                        alpha_i = (ni + nk) / denom;
                        alpha_j = (nj + nk) / denom;
                        beta = -nk / denom;
                        gamma = T(0);
                        D(best_i, k) = alpha_i * D(best_i, k) + alpha_j * D(best_j, k) + beta * D(best_i, best_j);
                        D(k, best_i) = D(best_i, k);
                    }
                    active[best_j] = false;
                    clusters[best_i].insert(clusters[best_i].end(),
                                            clusters[best_j].begin(), clusters[best_j].end());
                    n_active--;
                    continue;
                }

                for (size_t k = 0; k < n; ++k)
                {
                    if (!active[k] || k == best_i || k == best_j) continue;
                    D(best_i, k) = alpha_i * D(best_i, k) + alpha_j * D(best_j, k) +
                                   beta * D(best_i, best_j) + gamma * detail::abs_val(D(best_i, k) - D(best_j, k));
                    D(k, best_i) = D(best_i, k);
                }
                active[best_j] = false;
                clusters[best_i].insert(clusters[best_i].end(),
                                        clusters[best_j].begin(), clusters[best_j].end());
                n_active--;
            }

            // Assign flat labels
            result.labels.resize(n);
            int label = 0;
            for (size_t i = 0; i < n; ++i)
            {
                if (active[i])
                {
                    for (size_t idx : clusters[i])
                        result.labels[idx] = label;
                    label++;
                }
            }
            return result;
        }

        // ========================================================================
        // Gaussian Mixture Model (EM algorithm)
        // ========================================================================
        template <class T>
        struct gmm_result
        {
            xarray_container<T> means;       // k x d
            xarray_container<T> covariances; // k x d x d
            std::vector<T> weights;          // k
            std::vector<size_t> labels;      // n
            std::vector<T> bic;              // BIC scores
            size_t n_iter;
        };

        template <class T>
        gmm_result<T> gaussian_mixture(const xarray_container<T>& X, size_t k,
                                        const std::string& covariance_type = "full",
                                        size_t max_iter = 100, T tol = T(1e-3),
                                        size_t n_init = 1, std::mt19937* rng = nullptr)
        {
            size_t n = X.shape()[0];
            size_t d = X.shape()[1];
            std::mt19937 local_rng;
            if (!rng)
            {
                local_rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
                rng = &local_rng;
            }

            gmm_result<T> best_result;
            T best_log_likelihood = -std::numeric_limits<T>::max();

            for (size_t init = 0; init < n_init; ++init)
            {
                // Initialize with k‑means
                auto [labels_init, means_init, iter_init] = kmeans_lloyd(X, k, 100, tol, 1, rng);

                xarray_container<T> means = means_init;
                xarray_container<T> covs({k, d, d}, T(0));
                std::vector<T> weights(k, T(1) / T(k));

                // Initialize covariances as spherical with data variance
                T global_var = T(0);
                for (size_t i = 0; i < n; ++i)
                    for (size_t j = 0; j < d; ++j)
                        global_var = global_var + detail::multiply(X(i, j), X(i, j));
                global_var = global_var / T(n * d);
                for (size_t c = 0; c < k; ++c)
                    for (size_t j = 0; j < d; ++j)
                        covs(c, j, j) = global_var;

                // EM iterations
                xarray_container<T> resp({n, k}); // responsibilities
                T log_likelihood = -std::numeric_limits<T>::max();
                size_t iter_used = 0;

                for (size_t iter = 0; iter < max_iter; ++iter)
                {
                    // E‑step: compute responsibilities
                    T new_log_likelihood = T(0);
                    for (size_t i = 0; i < n; ++i)
                    {
                        T sum = T(0);
                        std::vector<T> row_resp(k);
                        for (size_t c = 0; c < k; ++c)
                        {
                            // Compute multivariate Gaussian log‑likelihood
                            T log_prob = T(0);
                            // Determinant and inverse of covariance
                            if (covariance_type == "spherical")
                            {
                                T var = covs(c, 0, 0);
                                T log_det = d * std::log(var);
                                T mahal = T(0);
                                for (size_t j = 0; j < d; ++j)
                                {
                                    T diff = X(i, j) - means(c, j);
                                    mahal = mahal + detail::multiply(diff, diff);
                                }
                                mahal = mahal / var;
                                log_prob = -T(0.5) * (T(d) * std::log(T(2) * T(3.1415926535)) + log_det + mahal);
                            }
                            else if (covariance_type == "diag")
                            {
                                T log_det = T(0);
                                T mahal = T(0);
                                for (size_t j = 0; j < d; ++j)
                                {
                                    T var = covs(c, j, j);
                                    log_det = log_det + std::log(var);
                                    T diff = X(i, j) - means(c, j);
                                    mahal = mahal + detail::multiply(diff, diff) / var;
                                }
                                log_prob = -T(0.5) * (T(d) * std::log(T(2) * T(3.1415926535)) + log_det + mahal);
                            }
                            else // full
                            {
                                // For full covariance, we need determinant and inverse
                                // We'll use Cholesky
                                xarray_container<T> cov({d, d});
                                for (size_t r = 0; r < d; ++r)
                                    for (size_t cc = 0; cc < d; ++cc)
                                        cov(r, cc) = covs(c, r, cc);
                                auto L = linalg::cholesky_ll(cov);
                                T log_det = T(0);
                                for (size_t j = 0; j < d; ++j) log_det = log_det + std::log(L(j, j));
                                log_det = log_det * T(2);
                                // Solve L * y = diff
                                std::vector<T> diff(d);
                                for (size_t j = 0; j < d; ++j) diff[j] = X(i, j) - means(c, j);
                                std::vector<T> y(d);
                                for (size_t j = 0; j < d; ++j)
                                {
                                    T sum_diff = diff[j];
                                    for (size_t p = 0; p < j; ++p)
                                        sum_diff = sum_diff - L(j, p) * y[p];
                                    y[j] = sum_diff / L(j, j);
                                }
                                T mahal = T(0);
                                for (size_t j = 0; j < d; ++j) mahal = mahal + y[j] * y[j];
                                log_prob = -T(0.5) * (T(d) * std::log(T(2) * T(3.1415926535)) + log_det + mahal);
                            }
                            row_resp[c] = std::exp(log_prob) * weights[c];
                            sum = sum + row_resp[c];
                        }
                        for (size_t c = 0; c < k; ++c)
                        {
                            resp(i, c) = row_resp[c] / sum;
                            new_log_likelihood = new_log_likelihood + std::log(sum);
                        }
                    }

                    // M‑step: update parameters
                    std::vector<T> nk(k, T(0));
                    for (size_t c = 0; c < k; ++c)
                    {
                        for (size_t i = 0; i < n; ++i)
                            nk[c] = nk[c] + resp(i, c);
                        weights[c] = nk[c] / T(n);
                        for (size_t j = 0; j < d; ++j)
                        {
                            means(c, j) = T(0);
                            for (size_t i = 0; i < n; ++i)
                                means(c, j) = means(c, j) + resp(i, c) * X(i, j);
                            means(c, j) = means(c, j) / nk[c];
                        }
                    }

                    // Update covariances
                    for (size_t c = 0; c < k; ++c)
                    {
                        if (covariance_type == "full")
                        {
                            for (size_t r = 0; r < d; ++r)
                            {
                                for (size_t cc = 0; cc < d; ++cc)
                                {
                                    T sum = T(0);
                                    for (size_t i = 0; i < n; ++i)
                                    {
                                        T diff_r = X(i, r) - means(c, r);
                                        T diff_c = X(i, cc) - means(c, cc);
                                        sum = sum + resp(i, c) * detail::multiply(diff_r, diff_c);
                                    }
                                    covs(c, r, cc) = sum / nk[c];
                                }
                            }
                        }
                        else if (covariance_type == "diag")
                        {
                            for (size_t j = 0; j < d; ++j)
                            {
                                T sum = T(0);
                                for (size_t i = 0; i < n; ++i)
                                {
                                    T diff = X(i, j) - means(c, j);
                                    sum = sum + resp(i, c) * detail::multiply(diff, diff);
                                }
                                covs(c, j, j) = sum / nk[c];
                            }
                        }
                        else // spherical
                        {
                            T sum = T(0);
                            for (size_t j = 0; j < d; ++j)
                            {
                                for (size_t i = 0; i < n; ++i)
                                {
                                    T diff = X(i, j) - means(c, j);
                                    sum = sum + resp(i, c) * detail::multiply(diff, diff);
                                }
                            }
                            T var = sum / (nk[c] * T(d));
                            for (size_t j = 0; j < d; ++j)
                                covs(c, j, j) = var;
                        }
                    }

                    iter_used = iter + 1;
                    if (detail::abs_val(new_log_likelihood - log_likelihood) < tol)
                    {
                        log_likelihood = new_log_likelihood;
                        break;
                    }
                    log_likelihood = new_log_likelihood;
                }

                // Compute BIC
                size_t n_params = k * d; // means
                if (covariance_type == "spherical") n_params += k;
                else if (covariance_type == "diag") n_params += k * d;
                else n_params += k * d * (d + 1) / 2;
                n_params += k - 1; // weights
                T bic = log_likelihood - T(0.5) * T(n_params) * std::log(T(n));

                if (log_likelihood > best_log_likelihood)
                {
                    best_log_likelihood = log_likelihood;
                    best_result.means = means;
                    best_result.covariances = covs;
                    best_result.weights = weights;
                    best_result.bic = {bic};
                    best_result.n_iter = iter_used;

                    // Assign labels
                    best_result.labels.resize(n);
                    for (size_t i = 0; i < n; ++i)
                    {
                        T max_resp = T(0);
                        size_t best_c = 0;
                        for (size_t c = 0; c < k; ++c)
                        {
                            if (resp(i, c) > max_resp)
                            {
                                max_resp = resp(i, c);
                                best_c = c;
                            }
                        }
                        best_result.labels[i] = best_c;
                    }
                }
            }
            return best_result;
        }

        // ========================================================================
        // Silhouette Score
        // ========================================================================
        template <class T>
        T silhouette_score(const xarray_container<T>& X, const std::vector<size_t>& labels,
                           detail::distance_metric metric = detail::distance_metric::euclidean)
        {
            size_t n = X.shape()[0];
            if (labels.size() != n)
                XTENSOR_THROW(std::invalid_argument, "silhouette_score: labels size mismatch");

            // Check if more than one cluster and not all same
            size_t n_clusters = *std::max_element(labels.begin(), labels.end()) + 1;
            if (n_clusters < 2) return T(0);

            std::vector<std::vector<size_t>> cluster_points(n_clusters);
            for (size_t i = 0; i < n; ++i)
                cluster_points[labels[i]].push_back(i);

            T total_score = T(0);
            size_t valid_points = 0;

            for (size_t i = 0; i < n; ++i)
            {
                size_t cluster_i = labels[i];
                if (cluster_points[cluster_i].size() <= 1) continue;

                // Compute a(i): mean distance to other points in same cluster
                T a_i = T(0);
                for (size_t j : cluster_points[cluster_i])
                {
                    if (j == i) continue;
                    std::vector<T> xi(d), xj(d);
                    for (size_t kk = 0; kk < X.shape()[1]; ++kk)
                    {
                        xi[kk] = X(i, kk);
                        xj[kk] = X(j, kk);
                    }
                    a_i = a_i + detail::compute_distance(xi, xj, metric);
                }
                a_i = a_i / T(cluster_points[cluster_i].size() - 1);

                // Compute b(i): min mean distance to points in other clusters
                T b_i = std::numeric_limits<T>::max();
                for (size_t c = 0; c < n_clusters; ++c)
                {
                    if (c == cluster_i || cluster_points[c].empty()) continue;
                    T mean_dist = T(0);
                    for (size_t j : cluster_points[c])
                    {
                        std::vector<T> xi(d), xj(d);
                        for (size_t kk = 0; kk < X.shape()[1]; ++kk)
                        {
                            xi[kk] = X(i, kk);
                            xj[kk] = X(j, kk);
                        }
                        mean_dist = mean_dist + detail::compute_distance(xi, xj, metric);
                    }
                    mean_dist = mean_dist / T(cluster_points[c].size());
                    if (mean_dist < b_i) b_i = mean_dist;
                }

                T s_i = (b_i - a_i) / std::max(a_i, b_i);
                total_score = total_score + s_i;
                valid_points++;
            }

            return (valid_points > 0) ? total_score / T(valid_points) : T(0);
        }

        // ========================================================================
        // Davies‑Bouldin Index
        // ========================================================================
        template <class T>
        T davies_bouldin_score(const xarray_container<T>& X, const std::vector<size_t>& labels)
        {
            size_t n = X.shape()[0];
            size_t d = X.shape()[1];
            size_t n_clusters = *std::max_element(labels.begin(), labels.end()) + 1;

            std::vector<std::vector<size_t>> cluster_points(n_clusters);
            for (size_t i = 0; i < n; ++i)
                cluster_points[labels[i]].push_back(i);

            // Compute cluster centers and average intra‑cluster distances
            std::vector<std::vector<T>> centers(n_clusters, std::vector<T>(d, T(0)));
            std::vector<T> avg_dist(n_clusters, T(0));

            for (size_t c = 0; c < n_clusters; ++c)
            {
                if (cluster_points[c].empty()) continue;
                for (size_t idx : cluster_points[c])
                    for (size_t j = 0; j < d; ++j)
                        centers[c][j] = centers[c][j] + X(idx, j);
                for (size_t j = 0; j < d; ++j)
                    centers[c][j] = centers[c][j] / T(cluster_points[c].size());

                for (size_t idx : cluster_points[c])
                {
                    std::vector<T> xi(d);
                    for (size_t j = 0; j < d; ++j) xi[j] = X(idx, j);
                    avg_dist[c] = avg_dist[c] + detail::euclidean_distance(xi, centers[c]);
                }
                avg_dist[c] = avg_dist[c] / T(cluster_points[c].size());
            }

            T db = T(0);
            for (size_t i = 0; i < n_clusters; ++i)
            {
                if (cluster_points[i].empty()) continue;
                T max_ratio = T(0);
                for (size_t j = 0; j < n_clusters; ++j)
                {
                    if (i == j || cluster_points[j].empty()) continue;
                    T dist_centers = detail::euclidean_distance(centers[i], centers[j]);
                    if (dist_centers == T(0)) continue;
                    T ratio = (avg_dist[i] + avg_dist[j]) / dist_centers;
                    if (ratio > max_ratio) max_ratio = ratio;
                }
                db = db + max_ratio;
            }
            return db / T(n_clusters);
        }

    } // namespace cluster

    using cluster::kmeans_lloyd;
    using cluster::kmeans_hartigan_wong;
    using cluster::dbscan;
    using cluster::agglomerative_clustering;
    using cluster::linkage_method;
    using cluster::linkage_result;
    using cluster::gaussian_mixture;
    using cluster::gmm_result;
    using cluster::silhouette_score;
    using cluster::davies_bouldin_score;
    using cluster::pairwise_distances;

} // namespace xt

#endif // XTENSOR_XCLUSTER_HPP