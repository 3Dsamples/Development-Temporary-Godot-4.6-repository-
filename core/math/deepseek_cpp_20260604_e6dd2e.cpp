// system name : onetbb-warp
// File 0039 : core/math/dimensionality_reduction.h
// Description : PCA, LDA, t‑SNE building blocks for dimensionality reduction.

#ifndef __TBB_WARP_CORE_MATH_DIMENSIONALITY_REDUCTION_H
#define __TBB_WARP_CORE_MATH_DIMENSIONALITY_REDUCTION_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/statistics.h"
#include "core/math/linear_algebra_ext.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <cstdint>
#include <functional>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Center data matrix (subtract mean)
// ============================================================

template<typename T>
void center_data(std::vector<std::vector<T>>& X) {
    std::size_t n = X.size();
    if (n == 0) return;
    std::size_t d = X[0].size();
    std::vector<T> mean(d, T(0));
    for (const auto& row : X) {
        for (std::size_t j = 0; j < d; ++j) mean[j] += row[j];
    }
    T inv_n = T(1) / n;
    for (auto& m : mean) m *= inv_n;
    for (auto& row : X) {
        for (std::size_t j = 0; j < d; ++j) row[j] -= mean[j];
    }
}

// ============================================================
// PCA via SVD on centered data
// ============================================================

template<typename T>
struct pca_result {
    std::vector<std::vector<T>> components;     // V (d×k) principal directions
    std::vector<T> eigenvalues;                 // sigma^2 (variance explained)
    std::vector<std::vector<T>> projected;      // reduced data (n×k)
    std::vector<T> mean;                        // original data mean
};

template<typename T>
pca_result<T> pca(const std::vector<std::vector<T>>& data, int k) {
    pca_result<T> result;
    if (data.empty()) return result;
    std::size_t n = data.size();
    std::size_t d = data[0].size();
    if (k < 1) k = static_cast<int>(d);
    k = std::min(k, static_cast<int>(d));
    // Copy and center
    std::vector<std::vector<T>> X = data;
    std::vector<T> mean(d, T(0));
    for (const auto& row : X)
        for (std::size_t j = 0; j < d; ++j) mean[j] += row[j];
    T inv_n = T(1) / n;
    for (auto& m : mean) m *= inv_n;
    result.mean = mean;
    for (auto& row : X)
        for (std::size_t j = 0; j < d; ++j) row[j] -= mean[j];
    // Transpose X to d×n for SVD convention? SVD expects rows=observations, but our svd works on m×n.
    // We'll use X as m×d (n×d). We need to compute SVD of X'X = V Sigma^2 V^T. So we can SVD X directly.
    std::vector<std::vector<T>> U;
    std::vector<T> sigma;
    std::vector<std::vector<T>> V;
    svd(X, U, sigma, V); // U = n×k, sigma = k, V = d×k
    // V columns are principal directions
    result.components = V; // V is d×k (actual size k)
    for (std::size_t i = 0; i < sigma.size(); ++i) {
        result.eigenvalues.push_back(sigma[i] * sigma[i] / static_cast<T>(n - 1));
    }
    // Project: Y = X * V (n×k)
    result.projected.resize(n, std::vector<T>(k, T(0)));
    for (std::size_t i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            T dot = T(0);
            for (std::size_t c = 0; c < d; ++c) {
                if (c < V.size() && j < static_cast<int>(V[c].size()))
                    dot += X[i][c] * V[c][j];
            }
            result.projected[i][j] = dot;
        }
    }
    return result;
}

// ============================================================
// PCA reconstruction from reduced data
// ============================================================

template<typename T>
std::vector<std::vector<T>> pca_reconstruct(const pca_result<T>& pca, int n_components = -1) {
    if (pca.projected.empty()) return {};
    if (n_components < 0) n_components = static_cast<int>(pca.components[0].size());
    std::size_t n = pca.projected.size();
    std::size_t d = pca.mean.size();
    std::vector<std::vector<T>> reconstructed(n, std::vector<T>(d, T(0)));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            T sum = T(0);
            for (int c = 0; c < n_components; ++c) {
                if (c < static_cast<int>(pca.projected[i].size()) && j < pca.components.size() &&
                    c < static_cast<int>(pca.components[j].size()))
                    sum += pca.projected[i][c] * pca.components[j][c];
            }
            reconstructed[i][j] = sum + pca.mean[j];
        }
    }
    return reconstructed;
}

// ============================================================
// Fisher LDA for two classes
// ============================================================

template<typename T>
std::vector<T> fisher_lda_two_class(const std::vector<std::vector<T>>& class0,
                                    const std::vector<std::vector<T>>& class1) {
    std::size_t d = class0[0].size();
    std::vector<T> mu0(d, T(0)), mu1(d, T(0));
    for (const auto& x : class0) for (std::size_t j = 0; j < d; ++j) mu0[j] += x[j];
    for (const auto& x : class1) for (std::size_t j = 0; j < d; ++j) mu1[j] += x[j];
    T inv_n0 = T(1) / class0.size(), inv_n1 = T(1) / class1.size();
    for (auto& m : mu0) m *= inv_n0;
    for (auto& m : mu1) m *= inv_n1;
    // Within-class scatter matrix Sw
    std::vector<std::vector<T>> Sw(d, std::vector<T>(d, T(0)));
    for (const auto& x : class0) {
        for (std::size_t i = 0; i < d; ++i) {
            T diff_i = x[i] - mu0[i];
            for (std::size_t j = 0; j < d; ++j) {
                Sw[i][j] += diff_i * (x[j] - mu0[j]);
            }
        }
    }
    for (const auto& x : class1) {
        for (std::size_t i = 0; i < d; ++i) {
            T diff_i = x[i] - mu1[i];
            for (std::size_t j = 0; j < d; ++j) {
                Sw[i][j] += diff_i * (x[j] - mu1[j]);
            }
        }
    }
    // Between-class difference vector
    std::vector<T> diff(d);
    for (std::size_t j = 0; j < d; ++j) diff[j] = mu0[j] - mu1[j];
    // Solve Sw * w = diff using pseudo‑inverse
    std::vector<std::vector<T>> Sw_inv = pseudo_inverse(Sw);
    std::vector<T> w(d, T(0));
    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            w[i] += Sw_inv[i][j] * diff[j];
        }
    }
    // Normalize
    T norm = T(0);
    for (auto v : w) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > T(1e-12)) {
        T inv = T(1) / norm;
        for (auto& v : w) v *= inv;
    }
    return w;
}

// ============================================================
// Multi‑class LDA (maximize between‑class / within‑class scatter)
// ============================================================

template<typename T>
struct lda_result {
    std::vector<std::vector<T>> projection; // eigenvectors of Sw^{-1}Sb
    std::vector<T> eigenvalues;
};

template<typename T>
lda_result<T> lda_multi_class(const std::vector<std::vector<T>>& data,
                              const std::vector<int>& labels, int n_components) {
    lda_result<T> result;
    std::size_t n = data.size();
    if (n == 0) return result;
    std::size_t d = data[0].size();
    // Compute overall mean and class means
    std::vector<T> overall_mean(d, T(0));
    std::unordered_map<int, std::vector<T>> class_means;
    std::unordered_map<int, std::size_t> class_counts;
    for (std::size_t i = 0; i < n; ++i) {
        int c = labels[i];
        class_counts[c]++;
        if (class_means.find(c) == class_means.end())
            class_means[c] = std::vector<T>(d, T(0));
        for (std::size_t j = 0; j < d; ++j) {
            overall_mean[j] += data[i][j];
            class_means[c][j] += data[i][j];
        }
    }
    T inv_n = T(1) / n;
    for (auto& m : overall_mean) m *= inv_n;
    for (auto& kv : class_means) {
        T inv_c = T(1) / class_counts[kv.first];
        for (auto& v : kv.second) v *= inv_c;
    }
    // Within‑class scatter Sw
    std::vector<std::vector<T>> Sw(d, std::vector<T>(d, T(0)));
    for (std::size_t i = 0; i < n; ++i) {
        int c = labels[i];
        const auto& mu_c = class_means[c];
        for (std::size_t p = 0; p < d; ++p) {
            T diff_p = data[i][p] - mu_c[p];
            for (std::size_t q = 0; q < d; ++q) {
                Sw[p][q] += diff_p * (data[i][q] - mu_c[q]);
            }
        }
    }
    // Between‑class scatter Sb
    std::vector<std::vector<T>> Sb(d, std::vector<T>(d, T(0)));
    for (const auto& kv : class_counts) {
        int c = kv.first;
        const auto& mu_c = class_means[c];
        std::vector<T> diff_mu(d);
        for (std::size_t j = 0; j < d; ++j) diff_mu[j] = mu_c[j] - overall_mean[j];
        T weight = static_cast<T>(kv.second);
        for (std::size_t p = 0; p < d; ++p) {
            for (std::size_t q = 0; q < d; ++q) {
                Sb[p][q] += weight * diff_mu[p] * diff_mu[q];
            }
        }
    }
    // Solve generalized eigenvalue problem: Sb w = lambda Sw w
    // Approximate via pseudo‑inverse: Sw^{-1}Sb (may be non‑symmetric)
    std::vector<std::vector<T>> Sw_inv = pseudo_inverse(Sw);
    std::vector<std::vector<T>> M(d, std::vector<T>(d, T(0)));
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j)
            for (std::size_t k = 0; k < d; ++k)
                M[i][j] += Sw_inv[i][k] * Sb[k][j];
    // Use real Schur decomposition to get eigenvalues
    std::vector<std::complex<T>> eigenvals;
    real_schur(M, eigenvals);
    // Sort by magnitude descending
    std::vector<std::pair<T, std::complex<T>>> sorted;
    for (std::size_t i = 0; i < eigenvals.size(); ++i) {
        sorted.emplace_back(std::abs(eigenvals[i]), eigenvals[i]);
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    result.eigenvalues.resize(n_components);
    for (int i = 0; i < n_components && i < static_cast<int>(sorted.size()); ++i) {
        result.eigenvalues[i] = sorted[i].first;
    }
    // For simplicity, we don't extract the actual eigenvectors here; a full implementation would use inverse iteration.
    return result;
}

// ============================================================
// t‑SNE: compute pairwise affinities (high‑dimensional)
// ============================================================

template<typename T>
std::vector<std::vector<T>> tsne_pairwise_affinities(const std::vector<std::vector<T>>& X,
                                                     T perplexity = T(30), T tol = T(1e-5)) {
    std::size_t n = X.size();
    if (n < 2) return {};
    // Compute squared distances
    std::vector<std::vector<T>> dist2(n, std::vector<T>(n, T(0)));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            T d = squared_distance(X[i], X[j]);
            dist2[i][j] = dist2[j][i] = d;
        }
    // Binary search for sigma_i such that perplexity matches
    std::vector<T> sigma(n, T(1));
    T log_perp = std::log(perplexity);
    for (std::size_t i = 0; i < n; ++i) {
        T lo = T(0), hi = T(1000);
        for (int iter = 0; iter < 50; ++iter) {
            T mid = (lo + hi) * T(0.5);
            T sum_P = T(0);
            T H = T(0);
            for (std::size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                T P_ij = std::exp(-dist2[i][j] / (T(2) * mid * mid));
                sum_P += P_ij;
                H += P_ij * std::log(std::max(P_ij, T(1e-20)));
            }
            T H_val = std::log(sum_P) + (-dist2[i][i] ? T(0) : T(0)) - std::log(T(2))? not needed.
            // Actually perplexity: 2^{H(P_i)} where H(P_i) = -sum p_j log2(p_j)
            // We'll compute using natural log: H_nat = -sum p_j ln(p_j), then perp = exp(H_nat)
            // We want perp = specified perplexity -> H_target = log(perplexity)
            // We'll adjust sigma until H matches.
            // We'll compute normalized P.
            T H_est = T(0);
            for (std::size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                T p = std::exp(-dist2[i][j] / (T(2) * mid * mid));
                if (p > T(1e-20)) H_est -= p * std::log(p);
            }
            if (std::abs(H_est - log_perp) < tol) {
                sigma[i] = mid;
                break;
            }
            if (H_est > log_perp) lo = mid; else hi = mid;
        }
    }
    // Compute symmetric P (joint probabilities)
    std::vector<std::vector<T>> P(n, std::vector<T>(n, T(0)));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            T num = std::exp(-dist2[i][j] / (T(2) * sigma[i] * sigma[i])) +
                    std::exp(-dist2[i][j] / (T(2) * sigma[j] * sigma[j]));
            P[i][j] = P[j][i] = num / (T(2) * n);
        }
    }
    // Normalize
    T sum_P = T(0);
    for (auto& row : P) for (auto val : row) sum_P += val;
    if (sum_P > T(1e-12)) {
        T inv = T(1) / sum_P;
        for (auto& row : P) for (auto& val : row) val *= inv;
    }
    return P;
}

// ============================================================
// t‑SNE: gradient of KL divergence w.r.t. low‑dim embeddings Y
// ============================================================

template<typename T>
std::vector<std::vector<T>> tsne_gradient(const std::vector<std::vector<T>>& Y,
                                          const std::vector<std::vector<T>>& P) {
    std::size_t n = Y.size();
    if (n < 2) return {};
    std::size_t k = Y[0].size();
    // Compute low‑dimensional affinities Q
    std::vector<std::vector<T>> dist2(n, std::vector<T>(n, T(0)));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            T d = squared_distance(Y[i], Y[j]);
            dist2[i][j] = dist2[j][i] = d;
        }
    T sum_Q = T(0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            sum_Q += T(1) / (T(1) + dist2[i][j]);
        }
    std::vector<std::vector<T>> grad(n, std::vector<T>(k, T(0)));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            T q_ij = (T(1) / (T(1) + dist2[i][j])) / sum_Q;
            T diff = (P[i][j] - q_ij) * q_ij * T(4);
            for (std::size_t d = 0; d < k; ++d) {
                grad[i][d] += diff * (Y[i][d] - Y[j][d]);
            }
        }
    }
    return grad;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_DIMENSIONALITY_REDUCTION_H