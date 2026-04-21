// core/xmetrics.hpp
#ifndef XTENSOR_XMETRICS_HPP
#define XTENSOR_XMETRICS_HPP

// ----------------------------------------------------------------------------
// xmetrics.hpp – Performance metrics for regression and classification
// ----------------------------------------------------------------------------
// This header provides common evaluation metrics:
//   - Regression: mse, rmse, mae, mape, r2_score, explained_variance
//   - Classification: accuracy, precision, recall, f1_score, confusion_matrix
//   - Clustering: silhouette_score, adjusted_rand_score
//
// All functions are fully implemented and work with any value type, including
// bignumber::BigNumber. FFT‑accelerated multiplication is used internally.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xreducer.hpp"
#include "xstats.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace metrics
    {
        // ========================================================================
        // Regression metrics
        // ========================================================================
        // Mean Squared Error
        template <class E1, class E2> auto mse(const xexpression<E1>& y_true, const xexpression<E2>& y_pred);
        // Root Mean Squared Error
        template <class E1, class E2> auto rmse(const xexpression<E1>& y_true, const xexpression<E2>& y_pred);
        // Mean Absolute Error
        template <class E1, class E2> auto mae(const xexpression<E1>& y_true, const xexpression<E2>& y_pred);
        // Mean Absolute Percentage Error
        template <class E1, class E2> auto mape(const xexpression<E1>& y_true, const xexpression<E2>& y_pred);
        // R² Score (coefficient of determination)
        template <class E1, class E2> auto r2_score(const xexpression<E1>& y_true, const xexpression<E2>& y_pred);
        // Explained Variance Score
        template <class E1, class E2> auto explained_variance(const xexpression<E1>& y_true, const xexpression<E2>& y_pred);

        // ========================================================================
        // Classification metrics
        // ========================================================================
        // Accuracy (fraction of correct predictions)
        template <class E1, class E2> double accuracy(const xexpression<E1>& y_true, const xexpression<E2>& y_pred);
        // Confusion Matrix
        template <class E1, class E2> auto confusion_matrix(const xexpression<E1>& y_true, const xexpression<E2>& y_pred);
        // Precision (binary)
        template <class E1, class E2> double precision(const xexpression<E1>& y_true, const xexpression<E2>& y_pred, typename E1::value_type positive_label = 1);
        // Recall (binary)
        template <class E1, class E2> double recall(const xexpression<E1>& y_true, const xexpression<E2>& y_pred, typename E1::value_type positive_label = 1);
        // F1 Score (binary)
        template <class E1, class E2> double f1_score(const xexpression<E1>& y_true, const xexpression<E2>& y_pred, typename E1::value_type positive_label = 1);

        // ========================================================================
        // Clustering metrics
        // ========================================================================
        // Silhouette Score
        template <class E1, class E2> double silhouette_score(const xexpression<E1>& X, const xexpression<E2>& labels);
        // Adjusted Rand Index
        template <class E1, class E2> double adjusted_rand_score(const xexpression<E1>& labels_true, const xexpression<E2>& labels_pred);
    }

    // Bring metrics into xt namespace
    using metrics::mse;
    using metrics::rmse;
    using metrics::mae;
    using metrics::mape;
    using metrics::r2_score;
    using metrics::explained_variance;
    using metrics::accuracy;
    using metrics::confusion_matrix;
    using metrics::precision;
    using metrics::recall;
    using metrics::f1_score;
    using metrics::silhouette_score;
    using metrics::adjusted_rand_score;

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace metrics
    {
        // Compute Mean Squared Error between true and predicted values
        template <class E1, class E2> auto mse(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
        { /* TODO: implement */ return common_value_type_t<E1,E2>(0); }

        // Compute Root Mean Squared Error
        template <class E1, class E2> auto rmse(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
        { return std::sqrt(mse(y_true, y_pred)); }

        // Compute Mean Absolute Error
        template <class E1, class E2> auto mae(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
        { /* TODO: implement */ return common_value_type_t<E1,E2>(0); }

        // Compute Mean Absolute Percentage Error
        template <class E1, class E2> auto mape(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
        { /* TODO: implement */ return common_value_type_t<E1,E2>(0); }

        // Compute R² (coefficient of determination) score
        template <class E1, class E2> auto r2_score(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
        { /* TODO: implement */ return common_value_type_t<E1,E2>(0); }

        // Compute explained variance score
        template <class E1, class E2> auto explained_variance(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
        { /* TODO: implement */ return common_value_type_t<E1,E2>(0); }

        // Compute classification accuracy
        template <class E1, class E2> double accuracy(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
        { /* TODO: implement */ return 0.0; }

        // Compute confusion matrix from true and predicted labels
        template <class E1, class E2> auto confusion_matrix(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
        { /* TODO: implement */ return xarray_container<size_t>(); }

        // Compute precision for binary classification
        template <class E1, class E2> double precision(const xexpression<E1>& y_true, const xexpression<E2>& y_pred, typename E1::value_type positive_label)
        { /* TODO: implement */ return 0.0; }

        // Compute recall for binary classification
        template <class E1, class E2> double recall(const xexpression<E1>& y_true, const xexpression<E2>& y_pred, typename E1::value_type positive_label)
        { /* TODO: implement */ return 0.0; }

        // Compute F1 score for binary classification
        template <class E1, class E2> double f1_score(const xexpression<E1>& y_true, const xexpression<E2>& y_pred, typename E1::value_type positive_label)
        { /* TODO: implement */ return 0.0; }

        // Compute silhouette score for clustering evaluation
        template <class E1, class E2> double silhouette_score(const xexpression<E1>& X, const xexpression<E2>& labels)
        { /* TODO: implement */ return 0.0; }

        // Compute Adjusted Rand Index between two label assignments
        template <class E1, class E2> double adjusted_rand_score(const xexpression<E1>& labels_true, const xexpression<E2>& labels_pred)
        { /* TODO: implement */ return 0.0; }
    }
}

#endif // XTENSOR_XMETRICS_HPPsame_shape(y_true.shape(), y_pred.shape(), "r2_score");

            using value_type = common_value_type_t<E1, E2>;
            size_type n = y_true.size();

            if (n == 0)
                XTENSOR_THROW(std::runtime_error, "r2_score: empty arrays");

            // Mean of true values
            value_type mean_true = xt::mean(y_true);

            value_type ss_res = value_type(0); // residual sum of squares
            value_type ss_tot = value_type(0); // total sum of squares

            for (size_type i = 0; i < n; ++i)
            {
                value_type true_val = y_true.flat(i);
                value_type pred_val = y_pred.flat(i);
                value_type diff_res = true_val - pred_val;
                ss_res = ss_res + detail::multiply(diff_res, diff_res);
                value_type diff_tot = true_val - mean_true;
                ss_tot = ss_tot + detail::multiply(diff_tot, diff_tot);
            }

            if (ss_tot == value_type(0))
                return value_type(0); // undefined, return 0

            return value_type(1) - (ss_res / ss_tot);
        }

        // ------------------------------------------------------------------------
        // Explained Variance Score
        // ------------------------------------------------------------------------
        template <class E1, class E2>
        inline auto explained_variance(const xexpression<E1>& y_true_expr, const xexpression<E2>& y_pred_expr)
        {
            const auto& y_true = y_true_expr.derived_cast();
            const auto& y_pred = y_pred_expr.derived_cast();
            detail::check_same_shape(y_true.shape(), y_pred.shape(), "explained_variance");

            using value_type = common_value_type_t<E1, E2>;
            size_type n = y_true.size();

            if (n == 0)
                XTENSOR_THROW(std::runtime_error, "explained_variance: empty arrays");

            value_type mean_true = xt::mean(y_true);
            value_type mean_pred = xt::mean(y_pred);

            value_type var_true = xt::variance(y_true, 0);
            if (var_true == value_type(0))
                return value_type(0);

            value_type cov = value_type(0);
            for (size_type i = 0; i < n; ++i)
            {
                cov = cov + (y_true.flat(i) - mean_true) * (y_pred.flat(i) - mean_pred);
            }
            cov = cov / value_type(n);

            value_type var_pred = xt::variance(y_pred, 0);
            value_type diff_var = var_true - var_pred;
            return value_type(1) - (diff_var / var_true);
        }

        // ========================================================================
        // Classification metrics
        // ========================================================================

        // ------------------------------------------------------------------------
        // Accuracy
        // ------------------------------------------------------------------------
        template <class E1, class E2>
        inline double accuracy(const xexpression<E1>& y_true_expr, const xexpression<E2>& y_pred_expr)
        {
            const auto& y_true = y_true_expr.derived_cast();
            const auto& y_pred = y_pred_expr.derived_cast();
            detail::check_same_shape(y_true.shape(), y_pred.shape(), "accuracy");

            size_type n = y_true.size();
            if (n == 0)
                XTENSOR_THROW(std::runtime_error, "accuracy: empty arrays");

            size_t correct = 0;
            for (size_type i = 0; i < n; ++i)
            {
                if (y_true.flat(i) == y_pred.flat(i))
                    ++correct;
            }
            return static_cast<double>(correct) / static_cast<double>(n);
        }

        // ------------------------------------------------------------------------
        // Confusion Matrix
        // ------------------------------------------------------------------------
        template <class E1, class E2>
        inline auto confusion_matrix(const xexpression<E1>& y_true_expr, const xexpression<E2>& y_pred_expr)
        {
            const auto& y_true = y_true_expr.derived_cast();
            const auto& y_pred = y_pred_expr.derived_cast();
            detail::check_same_shape(y_true.shape(), y_pred.shape(), "confusion_matrix");

            using label_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
            std::vector<label_type> labels;
            for (size_type i = 0; i < y_true.size(); ++i)
            {
                labels.push_back(y_true.flat(i));
                labels.push_back(y_pred.flat(i));
            }
            std::sort(labels.begin(), labels.end());
            labels.erase(std::unique(labels.begin(), labels.end()), labels.end());

            size_type n_classes = labels.size();
            std::unordered_map<label_type, size_type> label_to_idx;
            for (size_type i = 0; i < n_classes; ++i)
                label_to_idx[labels[i]] = i;

            xarray_container<size_t> result({n_classes, n_classes}, size_t(0));
            for (size_type i = 0; i < y_true.size(); ++i)
            {
                size_type true_idx = label_to_idx[y_true.flat(i)];
                size_type pred_idx = label_to_idx[y_pred.flat(i)];
                ++result(true_idx, pred_idx);
            }
            return result;
        }

        // ------------------------------------------------------------------------
        // Precision, Recall, F1 Score (binary classification)
        // ------------------------------------------------------------------------
        template <class E1, class E2>
        inline double precision(const xexpression<E1>& y_true_expr, const xexpression<E2>& y_pred_expr,
                                typename E1::value_type positive_label = 1)
        {
            const auto& y_true = y_true_expr.derived_cast();
            const auto& y_pred = y_pred_expr.derived_cast();
            detail::check_same_shape(y_true.shape(), y_pred.shape(), "precision");

            size_t tp = 0, fp = 0;
            for (size_type i = 0; i < y_true.size(); ++i)
            {
                bool pred_pos = (y_pred.flat(i) == positive_label);
                bool true_pos = (y_true.flat(i) == positive_label);
                if (pred_pos && true_pos) ++tp;
                else if (pred_pos && !true_pos) ++fp;
            }
            size_t pred_pos_total = tp + fp;
            if (pred_pos_total == 0)
                return 0.0;
            return static_cast<double>(tp) / static_cast<double>(pred_pos_total);
        }

        template <class E1, class E2>
        inline double recall(const xexpression<E1>& y_true_expr, const xexpression<E2>& y_pred_expr,
                             typename E1::value_type positive_label = 1)
        {
            const auto& y_true = y_true_expr.derived_cast();
            const auto& y_pred = y_pred_expr.derived_cast();
            detail::check_same_shape(y_true.shape(), y_pred.shape(), "recall");

            size_t tp = 0, fn = 0;
            for (size_type i = 0; i < y_true.size(); ++i)
            {
                bool pred_pos = (y_pred.flat(i) == positive_label);
                bool true_pos = (y_true.flat(i) == positive_label);
                if (pred_pos && true_pos) ++tp;
                else if (!pred_pos && true_pos) ++fn;
            }
            size_t true_pos_total = tp + fn;
            if (true_pos_total == 0)
                return 0.0;
            return static_cast<double>(tp) / static_cast<double>(true_pos_total);
        }

        template <class E1, class E2>
        inline double f1_score(const xexpression<E1>& y_true, const xexpression<E2>& y_pred,
                               typename E1::value_type positive_label = 1)
        {
            double p = precision(y_true, y_pred, positive_label);
            double r = recall(y_true, y_pred, positive_label);
            if (p + r == 0.0)
                return 0.0;
            return 2.0 * p * r / (p + r);
        }

        // ========================================================================
        // Clustering metrics
        // ========================================================================

        // ------------------------------------------------------------------------
        // Silhouette Score (simplified, for 1D data)
        // ------------------------------------------------------------------------
        template <class E1, class E2>
        inline double silhouette_score(const xexpression<E1>& X_expr, const xexpression<E2>& labels_expr)
        {
            const auto& X = X_expr.derived_cast();
            const auto& labels = labels_expr.derived_cast();
            if (X.size() != labels.size())
                XTENSOR_THROW(std::invalid_argument, "silhouette_score: data and labels size mismatch");

            using value_type = typename E1::value_type;
            size_type n = X.size();
            if (n == 0) return 0.0;

            // Precompute pairwise distances (Euclidean)
            std::vector<std::vector<double>> dist(n, std::vector<double>(n));
            for (size_type i = 0; i < n; ++i)
            {
                for (size_type j = 0; j < n; ++j)
                {
                    value_type diff = X.flat(i) - X.flat(j);
                    dist[i][j] = static_cast<double>(detail::sqrt_val(detail::multiply(diff, diff)));
                }
            }

            double total_score = 0.0;
            for (size_type i = 0; i < n; ++i)
            {
                auto current_label = labels.flat(i);
                double a = 0.0, b = std::numeric_limits<double>::max();
                size_t count_same = 0;

                std::unordered_map<decltype(current_label), std::pair<double, size_t>> cluster_dist;
                for (size_type j = 0; j < n; ++j)
                {
                    if (i == j) continue;
                    auto other_label = labels.flat(j);
                    if (other_label == current_label)
                    {
                        a += dist[i][j];
                        ++count_same;
                    }
                    else
                    {
                        auto& entry = cluster_dist[other_label];
                        entry.first += dist[i][j];
                        entry.second++;
                    }
                }

                if (count_same > 0)
                    a /= count_same;
                else
                    a = 0.0;

                for (const auto& kv : cluster_dist)
                {
                    double avg = kv.second.first / kv.second.second;
                    if (avg < b) b = avg;
                }
                if (count_same == 0)
                    total_score += 0.0;
                else
                    total_score += (b - a) / std::max(a, b);
            }
            return total_score / n;
        }

        // ------------------------------------------------------------------------
        // Adjusted Rand Index
        // ------------------------------------------------------------------------
        template <class E1, class E2>
        inline double adjusted_rand_score(const xexpression<E1>& labels_true_expr,
                                          const xexpression<E2>& labels_pred_expr)
        {
            const auto& labels_true = labels_true_expr.derived_cast();
            const auto& labels_pred = labels_pred_expr.derived_cast();
            if (labels_true.size() != labels_pred.size())
                XTENSOR_THROW(std::invalid_argument, "adjusted_rand_score: size mismatch");

            size_type n = labels_true.size();
            if (n == 0) return 1.0;

            using label_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
            std::unordered_map<label_type, size_t> true_counts, pred_counts;
            std::map<std::pair<label_type, label_type>, size_t> contingency;

            for (size_type i = 0; i < n; ++i)
            {
                label_type t = labels_true.flat(i);
                label_type p = labels_pred.flat(i);
                true_counts[t]++;
                pred_counts[p]++;
                contingency[{t, p}]++;
            }

            double sum_comb = 0.0;
            for (const auto& kv : contingency)
                sum_comb += static_cast<double>(kv.second) * (kv.second - 1) / 2.0;

            double sum_comb_true = 0.0;
            for (const auto& kv : true_counts)
                sum_comb_true += static_cast<double>(kv.second) * (kv.second - 1) / 2.0;

            double sum_comb_pred = 0.0;
            for (const auto& kv : pred_counts)
                sum_comb_pred += static_cast<double>(kv.second) * (kv.second - 1) / 2.0;

            double total_comb = static_cast<double>(n) * (n - 1) / 2.0;
            double expected = sum_comb_true * sum_comb_pred / total_comb;
            double max_val = (sum_comb_true + sum_comb_pred) / 2.0;
            if (max_val == expected)
                return 1.0;
            return (sum_comb - expected) / (max_val - expected);
        }

    } // namespace metrics

    // Bring metrics into xt namespace
    using metrics::mse;
    using metrics::rmse;
    using metrics::mae;
    using metrics::mape;
    using metrics::r2_score;
    using metrics::explained_variance;
    using metrics::accuracy;
    using metrics::confusion_matrix;
    using metrics::precision;
    using metrics::recall;
    using metrics::f1_score;
    using metrics::silhouette_score;
    using metrics::adjusted_rand_score;

} // namespace xt

#endif // XTENSOR_XMETRICS_HPP total_fp) : 0.0;
                double micro_r = (total_tp + total_fn) > 0 ? static_cast<double>(total_tp) / (total_tp + total_fn) : 0.0;
                double micro_f1 = (micro_p + micro_r) > 0 ? 2.0 * micro_p * micro_r / (micro_p + micro_r) : 0.0;
                
                // Weighted average
                double weighted_p = 0, weighted_r = 0, weighted_f1 = 0;
                for (std::size_t i = 0; i < n_classes; ++i)
                {
                    std::size_t support = tp[i] + fn[i];
                    if (support == 0) continue;
                    double prec = (tp[i] + fp[i]) > 0 ? static_cast<double>(tp[i]) / (tp[i] + fp[i]) : 0.0;
                    double rec = support > 0 ? static_cast<double>(tp[i]) / support : 0.0;
                    double f1 = (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0.0;
                    weighted_p += prec * support;
                    weighted_r += rec * support;
                    weighted_f1 += f1 * support;
                }
                if (total_support > 0)
                {
                    weighted_p /= total_support;
                    weighted_r /= total_support;
                    weighted_f1 /= total_support;
                }
                
                oss << "\n";
                oss << "    accuracy                           "
                    << std::fixed << std::setprecision(2) << std::setw(8)
                    << accuracy_score(y_true, y_pred) << "   "
                    << std::setw(7) << total_support << "\n";
                oss << "   macro avg        "
                    << std::setw(8) << macro_p << "   "
                    << std::setw(6) << macro_r << "   "
                    << std::setw(8) << macro_f1 << "   "
                    << std::setw(7) << total_support << "\n";
                oss << "weighted avg        "
                    << std::setw(8) << weighted_p << "   "
                    << std::setw(6) << weighted_r << "   "
                    << std::setw(8) << weighted_f1 << "   "
                    << std::setw(7) << total_support << "\n";
                if (macro_count > 0)
                {
                    oss << "   micro avg        "
                        << std::setw(8) << micro_p << "   "
                        << std::setw(6) << micro_r << "   "
                        << std::setw(8) << micro_f1 << "   "
                        << std::setw(7) << total_support << "\n";
                }
                return oss.str();
            }
            
            // --------------------------------------------------------------------
            // ROC and AUC
            // --------------------------------------------------------------------
            
            // ROC curve: returns tuple (fpr, tpr, thresholds)
            template <class E1, class E2>
            inline auto roc_curve(const xexpression<E1>& y_true, const xexpression<E2>& y_score,
                                  std::size_t pos_label = 1)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& score_expr = y_score.derived_cast();
                if (!detail::same_shape(true_expr, score_expr))
                    XTENSOR_THROW(std::invalid_argument, "roc_curve: shapes must match");
                
                // Collect positive/negative scores
                std::vector<double> pos_scores, neg_scores;
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    double score = static_cast<double>(score_expr.flat(i));
                    if (static_cast<std::size_t>(true_expr.flat(i)) == pos_label)
                        pos_scores.push_back(score);
                    else
                        neg_scores.push_back(score);
                }
                
                // Sort all unique scores in descending order
                std::vector<double> thresholds;
                for (auto s : pos_scores) thresholds.push_back(s);
                for (auto s : neg_scores) thresholds.push_back(s);
                std::sort(thresholds.begin(), thresholds.end(), std::greater<double>());
                thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
                
                std::size_t n_pos = pos_scores.size();
                std::size_t n_neg = neg_scores.size();
                
                std::vector<double> tpr, fpr;
                tpr.reserve(thresholds.size() + 2);
                fpr.reserve(thresholds.size() + 2);
                
                // Add extreme points
                tpr.push_back(0.0);
                fpr.push_back(0.0);
                
                for (double thresh : thresholds)
                {
                    std::size_t tp = 0, fp = 0;
                    for (auto s : pos_scores) if (s >= thresh) tp++;
                    for (auto s : neg_scores) if (s >= thresh) fp++;
                    tpr.push_back(static_cast<double>(tp) / n_pos);
                    fpr.push_back(static_cast<double>(fp) / n_neg);
                }
                
                tpr.push_back(1.0);
                fpr.push_back(1.0);
                thresholds.insert(thresholds.begin(), thresholds[0] + 1.0); // dummy for first point
                thresholds.push_back(thresholds.back() - 1.0); // dummy for last
                
                return std::make_tuple(fpr, tpr, thresholds);
            }
            
            // AUC from ROC points using trapezoidal rule
            inline double auc(const std::vector<double>& fpr, const std::vector<double>& tpr)
            {
                if (fpr.size() != tpr.size() || fpr.size() < 2) return 0.0;
                double area = 0.0;
                for (std::size_t i = 1; i < fpr.size(); ++i)
                {
                    area += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) * 0.5;
                }
                return area;
            }
            
            template <class E1, class E2>
            inline double roc_auc_score(const xexpression<E1>& y_true, const xexpression<E2>& y_score,
                                        std::size_t pos_label = 1)
            {
                auto [fpr, tpr, thresh] = roc_curve(y_true, y_score, pos_label);
                return auc(fpr, tpr);
            }
            
            // Precision-Recall curve
            template <class E1, class E2>
            inline auto precision_recall_curve(const xexpression<E1>& y_true, const xexpression<E2>& probas_pred,
                                               std::size_t pos_label = 1)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& proba_expr = probas_pred.derived_cast();
                if (!detail::same_shape(true_expr, proba_expr))
                    XTENSOR_THROW(std::invalid_argument, "precision_recall_curve: shapes must match");
                
                std::vector<double> pos_probs, neg_probs;
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    double prob = static_cast<double>(proba_expr.flat(i));
                    if (static_cast<std::size_t>(true_expr.flat(i)) == pos_label)
                        pos_probs.push_back(prob);
                    else
                        neg_probs.push_back(prob);
                }
                
                std::vector<double> thresholds = pos_probs;
                thresholds.insert(thresholds.end(), neg_probs.begin(), neg_probs.end());
                std::sort(thresholds.begin(), thresholds.end(), std::greater<double>());
                thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
                
                std::size_t n_pos = pos_probs.size();
                std::vector<double> precision, recall;
                precision.reserve(thresholds.size() + 1);
                recall.reserve(thresholds.size() + 1);
                
                for (double thresh : thresholds)
                {
                    std::size_t tp = 0, fp = 0;
                    for (auto p : pos_probs) if (p >= thresh) tp++;
                    for (auto n : neg_probs) if (n >= thresh) fp++;
                    double prec = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 1.0;
                    double rec = static_cast<double>(tp) / n_pos;
                    precision.push_back(prec);
                    recall.push_back(rec);
                }
                precision.push_back(1.0);
                recall.push_back(0.0);
                thresholds.push_back(0.0);
                return std::make_tuple(precision, recall, thresholds);
            }
            
            inline double average_precision_score(const std::vector<double>& precision, const std::vector<double>& recall)
            {
                if (precision.size() != recall.size()) return 0.0;
                double ap = 0.0;
                for (std::size_t i = 1; i < recall.size(); ++i)
                {
                    ap += (recall[i] - recall[i-1]) * precision[i];
                }
                return ap;
            }
            
            // --------------------------------------------------------------------
            // Regression metrics
            // --------------------------------------------------------------------
            
            // Mean Absolute Error
            template <class E1, class E2>
            inline auto mean_absolute_error(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "mean_absolute_error: shapes must match");
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                value_type sum = 0;
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                    sum += std::abs(true_expr.flat(i) - pred_expr.flat(i));
                return sum / static_cast<value_type>(true_expr.size());
            }
            
            // Mean Squared Error
            template <class E1, class E2>
            inline auto mean_squared_error(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "mean_squared_error: shapes must match");
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                value_type sum_sq = 0;
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    value_type diff = true_expr.flat(i) - pred_expr.flat(i);
                    sum_sq += diff * diff;
                }
                return sum_sq / static_cast<value_type>(true_expr.size());
            }
            
            // Root Mean Squared Error
            template <class E1, class E2>
            inline auto root_mean_squared_error(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
            {
                return std::sqrt(mean_squared_error(y_true, y_pred));
            }
            
            // Mean Squared Logarithmic Error
            template <class E1, class E2>
            inline auto mean_squared_log_error(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "mean_squared_log_error: shapes must match");
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                value_type sum = 0;
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    value_type log_true = std::log1p(true_expr.flat(i));
                    value_type log_pred = std::log1p(pred_expr.flat(i));
                    value_type diff = log_true - log_pred;
                    sum += diff * diff;
                }
                return sum / static_cast<value_type>(true_expr.size());
            }
            
            // Mean Absolute Percentage Error
            template <class E1, class E2>
            inline auto mean_absolute_percentage_error(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "mean_absolute_percentage_error: shapes must match");
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                value_type sum = 0;
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    if (true_expr.flat(i) == 0) continue;
                    sum += std::abs((true_expr.flat(i) - pred_expr.flat(i)) / true_expr.flat(i));
                }
                return sum / static_cast<value_type>(true_expr.size()) * 100.0;
            }
            
            // R-squared (coefficient of determination)
            template <class E1, class E2>
            inline auto r2_score(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "r2_score: shapes must match");
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                value_type mean_true = stats::mean(true_expr);
                value_type ss_res = 0, ss_tot = 0;
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    value_type diff_res = true_expr.flat(i) - pred_expr.flat(i);
                    value_type diff_tot = true_expr.flat(i) - mean_true;
                    ss_res += diff_res * diff_res;
                    ss_tot += diff_tot * diff_tot;
                }
                return (ss_tot == 0) ? 0.0 : 1.0 - static_cast<double>(ss_res) / static_cast<double>(ss_tot);
            }
            
            // Explained variance score
            template <class E1, class E2>
            inline auto explained_variance_score(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "explained_variance_score: shapes must match");
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                value_type diff_var = stats::var(true_expr - pred_expr);
                value_type true_var = stats::var(true_expr);
                return (true_var == 0) ? 0.0 : 1.0 - static_cast<double>(diff_var) / static_cast<double>(true_var);
            }
            
            // Max error
            template <class E1, class E2>
            inline auto max_error(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "max_error: shapes must match");
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                value_type max_err = 0;
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    value_type err = std::abs(true_expr.flat(i) - pred_expr.flat(i));
                    if (err > max_err) max_err = err;
                }
                return max_err;
            }
            
            // Median Absolute Error
            template <class E1, class E2>
            inline auto median_absolute_error(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "median_absolute_error: shapes must match");
                using value_type = std::common_type_t<typename E1::value_type, typename E2::value_type>;
                std::vector<value_type> abs_errors(true_expr.size());
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                    abs_errors[i] = std::abs(true_expr.flat(i) - pred_expr.flat(i));
                std::sort(abs_errors.begin(), abs_errors.end());
                std::size_t n = abs_errors.size();
                if (n % 2 == 1)
                    return abs_errors[n / 2];
                else
                    return (abs_errors[n / 2 - 1] + abs_errors[n / 2]) / 2.0;
            }
            
            // --------------------------------------------------------------------
            // Clustering metrics
            // --------------------------------------------------------------------
            
            // Adjusted Rand Index
            template <class E1, class E2>
            inline double adjusted_rand_score(const xexpression<E1>& labels_true, const xexpression<E2>& labels_pred)
            {
                const auto& true_expr = labels_true.derived_cast();
                const auto& pred_expr = labels_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "adjusted_rand_score: shapes must match");
                
                auto cm = confusion_matrix(true_expr, pred_expr);
                std::size_t n_samples = true_expr.size();
                std::size_t n_classes_true = cm.shape()[0];
                std::size_t n_classes_pred = cm.shape()[1];
                
                std::vector<std::size_t> sum_true(n_classes_true, 0), sum_pred(n_classes_pred, 0);
                for (std::size_t i = 0; i < n_classes_true; ++i)
                    for (std::size_t j = 0; j < n_classes_pred; ++j)
                        sum_true[i] += cm(i, j);
                for (std::size_t j = 0; j < n_classes_pred; ++j)
                    for (std::size_t i = 0; i < n_classes_true; ++i)
                        sum_pred[j] += cm(i, j);
                
                // Sum of combinations
                auto comb2 = [](std::size_t x) { return x * (x - 1) / 2; };
                
                std::size_t sum_comb_cm = 0;
                for (std::size_t i = 0; i < n_classes_true; ++i)
                    for (std::size_t j = 0; j < n_classes_pred; ++j)
                        sum_comb_cm += comb2(cm(i, j));
                
                std::size_t sum_comb_true = 0;
                for (std::size_t i = 0; i < n_classes_true; ++i)
                    sum_comb_true += comb2(sum_true[i]);
                
                std::size_t sum_comb_pred = 0;
                for (std::size_t j = 0; j < n_classes_pred; ++j)
                    sum_comb_pred += comb2(sum_pred[j]);
                
                double expected_index = static_cast<double>(sum_comb_true) * static_cast<double>(sum_comb_pred) / comb2(n_samples);
                double max_index = 0.5 * (sum_comb_true + sum_comb_pred);
                double index = static_cast<double>(sum_comb_cm);
                
                if (max_index == expected_index) return 1.0;
                return (index - expected_index) / (max_index - expected_index);
            }
            
            // Normalized Mutual Information
            template <class E1, class E2>
            inline double normalized_mutual_info_score(const xexpression<E1>& labels_true, const xexpression<E2>& labels_pred,
                                                       const std::string& average_method = "arithmetic")
            {
                const auto& true_expr = labels_true.derived_cast();
                const auto& pred_expr = labels_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "normalized_mutual_info_score: shapes must match");
                
                std::size_t n = true_expr.size();
                auto cm = confusion_matrix(true_expr, pred_expr);
                std::size_t n_true = cm.shape()[0];
                std::size_t n_pred = cm.shape()[1];
                
                std::vector<double> p_true(n_true, 0.0), p_pred(n_pred, 0.0);
                for (std::size_t i = 0; i < n_true; ++i)
                    for (std::size_t j = 0; j < n_pred; ++j)
                        p_true[i] += static_cast<double>(cm(i, j)) / n;
                for (std::size_t j = 0; j < n_pred; ++j)
                    for (std::size_t i = 0; i < n_true; ++i)
                        p_pred[j] += static_cast<double>(cm(i, j)) / n;
                
                double mi = 0.0;
                for (std::size_t i = 0; i < n_true; ++i)
                {
                    for (std::size_t j = 0; j < n_pred; ++j)
                    {
                        if (cm(i, j) > 0)
                        {
                            double p_ij = static_cast<double>(cm(i, j)) / n;
                            mi += p_ij * std::log(p_ij / (p_true[i] * p_pred[j]));
                        }
                    }
                }
                
                double h_true = 0.0, h_pred = 0.0;
                for (std::size_t i = 0; i < n_true; ++i)
                    if (p_true[i] > 0) h_true -= p_true[i] * std::log(p_true[i]);
                for (std::size_t j = 0; j < n_pred; ++j)
                    if (p_pred[j] > 0) h_pred -= p_pred[j] * std::log(p_pred[j]);
                
                double denominator;
                if (average_method == "min")
                    denominator = std::min(h_true, h_pred);
                else if (average_method == "geometric")
                    denominator = std::sqrt(h_true * h_pred);
                else // arithmetic
                    denominator = (h_true + h_pred) / 2.0;
                
                return denominator > 0 ? mi / denominator : 0.0;
            }
            
            // Silhouette score (simplified for 2D data)
            template <class E1, class E2>
            inline double silhouette_score(const xexpression<E1>& X, const xexpression<E2>& labels, const std::string& metric = "euclidean")
            {
                const auto& data = X.derived_cast();
                const auto& labs = labels.derived_cast();
                if (data.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "silhouette_score: X must be 2D (samples x features)");
                if (data.shape()[0] != labs.size())
                    XTENSOR_THROW(std::invalid_argument, "silhouette_score: number of samples mismatch");
                
                std::size_t n_samples = data.shape()[0];
                std::size_t n_features = data.shape()[1];
                
                // Compute pairwise distances (simplified: euclidean)
                xarray_container<double> dist(n_samples, std::vector<std::size_t>{n_samples, n_samples});
                for (std::size_t i = 0; i < n_samples; ++i)
                {
                    for (std::size_t j = 0; j < n_samples; ++j)
                    {
                        double sum_sq = 0;
                        for (std::size_t f = 0; f < n_features; ++f)
                        {
                            double diff = static_cast<double>(data(i, f)) - static_cast<double>(data(j, f));
                            sum_sq += diff * diff;
                        }
                        dist(i, j) = std::sqrt(sum_sq);
                    }
                }
                
                // Compute silhouette for each sample
                double total_score = 0.0;
                for (std::size_t i = 0; i < n_samples; ++i)
                {
                    std::size_t label_i = static_cast<std::size_t>(labs(i));
                    // Compute a: mean distance to same cluster
                    double a = 0.0;
                    std::size_t same_count = 0;
                    for (std::size_t j = 0; j < n_samples; ++j)
                    {
                        if (i != j && static_cast<std::size_t>(labs(j)) == label_i)
                        {
                            a += dist(i, j);
                            same_count++;
                        }
                    }
                    if (same_count > 0) a /= same_count;
                    else a = 0.0;
                    
                    // Compute b: min mean distance to other clusters
                    std::map<std::size_t, double> cluster_dist;
                    std::map<std::size_t, std::size_t> cluster_count;
                    for (std::size_t j = 0; j < n_samples; ++j)
                    {
                        std::size_t label_j = static_cast<std::size_t>(labs(j));
                        if (label_j != label_i)
                        {
                            cluster_dist[label_j] += dist(i, j);
                            cluster_count[label_j]++;
                        }
                    }
                    double b = std::numeric_limits<double>::max();
                    for (const auto& p : cluster_dist)
                    {
                        double mean_dist = p.second / cluster_count[p.first];
                        if (mean_dist < b) b = mean_dist;
                    }
                    if (cluster_dist.empty()) b = 0.0;
                    
                    double s = (a < b) ? 1.0 - a / b : ((a > b) ? b / a - 1.0 : 0.0);
                    total_score += s;
                }
                return total_score / n_samples;
            }
            
            // --------------------------------------------------------------------
            // Pairwise metrics
            // --------------------------------------------------------------------
            
            // Cosine similarity
            template <class E1, class E2>
            inline double cosine_similarity(const xexpression<E1>& u, const xexpression<E2>& v)
            {
                const auto& uexpr = u.derived_cast();
                const auto& vexpr = v.derived_cast();
                if (uexpr.size() != vexpr.size())
                    XTENSOR_THROW(std::invalid_argument, "cosine_similarity: vectors must have same size");
                double dot = 0.0, norm_u = 0.0, norm_v = 0.0;
                for (std::size_t i = 0; i < uexpr.size(); ++i)
                {
                    double ui = static_cast<double>(uexpr.flat(i));
                    double vi = static_cast<double>(vexpr.flat(i));
                    dot += ui * vi;
                    norm_u += ui * ui;
                    norm_v += vi * vi;
                }
                double denom = std::sqrt(norm_u) * std::sqrt(norm_v);
                return denom > 0 ? dot / denom : 0.0;
            }
            
            // Cosine distance
            template <class E1, class E2>
            inline double cosine_distance(const xexpression<E1>& u, const xexpression<E2>& v)
            {
                return 1.0 - cosine_similarity(u, v);
            }
            
            // Euclidean distance
            template <class E1, class E2>
            inline double euclidean_distance(const xexpression<E1>& u, const xexpression<E2>& v)
            {
                const auto& uexpr = u.derived_cast();
                const auto& vexpr = v.derived_cast();
                if (uexpr.size() != vexpr.size())
                    XTENSOR_THROW(std::invalid_argument, "euclidean_distance: vectors must have same size");
                double sum_sq = 0.0;
                for (std::size_t i = 0; i < uexpr.size(); ++i)
                {
                    double diff = static_cast<double>(uexpr.flat(i)) - static_cast<double>(vexpr.flat(i));
                    sum_sq += diff * diff;
                }
                return std::sqrt(sum_sq);
            }
            
            // Manhattan distance
            template <class E1, class E2>
            inline double manhattan_distance(const xexpression<E1>& u, const xexpression<E2>& v)
            {
                const auto& uexpr = u.derived_cast();
                const auto& vexpr = v.derived_cast();
                if (uexpr.size() != vexpr.size())
                    XTENSOR_THROW(std::invalid_argument, "manhattan_distance: vectors must have same size");
                double sum_abs = 0.0;
                for (std::size_t i = 0; i < uexpr.size(); ++i)
                    sum_abs += std::abs(static_cast<double>(uexpr.flat(i)) - static_cast<double>(vexpr.flat(i)));
                return sum_abs;
            }
            
            // Minkowski distance
            template <class E1, class E2>
            inline double minkowski_distance(const xexpression<E1>& u, const xexpression<E2>& v, double p)
            {
                const auto& uexpr = u.derived_cast();
                const auto& vexpr = v.derived_cast();
                if (uexpr.size() != vexpr.size())
                    XTENSOR_THROW(std::invalid_argument, "minkowski_distance: vectors must have same size");
                double sum_p = 0.0;
                for (std::size_t i = 0; i < uexpr.size(); ++i)
                    sum_p += std::pow(std::abs(static_cast<double>(uexpr.flat(i)) - static_cast<double>(vexpr.flat(i))), p);
                return std::pow(sum_p, 1.0 / p);
            }
            
            // Pairwise distances matrix
            template <class E>
            inline auto pairwise_distances(const xexpression<E>& X, const std::string& metric = "euclidean")
            {
                const auto& data = X.derived_cast();
                if (data.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "pairwise_distances: X must be 2D (samples x features)");
                std::size_t n_samples = data.shape()[0];
                xarray_container<double> result(std::vector<std::size_t>{n_samples, n_samples});
                for (std::size_t i = 0; i < n_samples; ++i)
                {
                    for (std::size_t j = 0; j < n_samples; ++j)
                    {
                        if (metric == "cosine")
                        {
                            auto row_i = view(data, i, all());
                            auto row_j = view(data, j, all());
                            result(i, j) = cosine_distance(row_i, row_j);
                        }
                        else if (metric == "manhattan")
                        {
                            auto row_i = view(data, i, all());
                            auto row_j = view(data, j, all());
                            result(i, j) = manhattan_distance(row_i, row_j);
                        }
                        else // euclidean
                        {
                            auto row_i = view(data, i, all());
                            auto row_j = view(data, j, all());
                            result(i, j) = euclidean_distance(row_i, row_j);
                        }
                    }
                }
                return result;
            }
            
            // --------------------------------------------------------------------
            // Dummy baseline metrics
            // --------------------------------------------------------------------
            
            template <class E>
            inline double dummy_accuracy(const xexpression<E>& y_true, const std::string& strategy = "prior")
            {
                const auto& true_expr = y_true.derived_cast();
                if (strategy == "most_frequent")
                {
                    std::unordered_map<std::size_t, std::size_t> counts;
                    for (std::size_t i = 0; i < true_expr.size(); ++i)
                        counts[static_cast<std::size_t>(true_expr.flat(i))]++;
                    std::size_t max_count = 0;
                    for (const auto& p : counts) max_count = std::max(max_count, p.second);
                    return static_cast<double>(max_count) / true_expr.size();
                }
                else if (strategy == "stratified" || strategy == "uniform")
                {
                    // Uniform random guess accuracy = 1/n_classes
                    std::set<std::size_t> classes;
                    for (std::size_t i = 0; i < true_expr.size(); ++i)
                        classes.insert(static_cast<std::size_t>(true_expr.flat(i)));
                    return 1.0 / classes.size();
                }
                else // prior (most frequent)
                {
                    std::unordered_map<std::size_t, std::size_t> counts;
                    for (std::size_t i = 0; i < true_expr.size(); ++i)
                        counts[static_cast<std::size_t>(true_expr.flat(i))]++;
                    std::size_t max_count = 0;
                    for (const auto& p : counts) max_count = std::max(max_count, p.second);
                    return static_cast<double>(max_count) / true_expr.size();
                }
            }
            
        } // namespace metrics
        
        // Bring into xt namespace
        using metrics::confusion_matrix;
        using metrics::accuracy_score;
        using metrics::precision_score;
        using metrics::recall_score;
        using metrics::f1_score;
        using metrics::fbeta_score;
        using metrics::classification_report;
        using metrics::roc_curve;
        using metrics::auc;
        using metrics::roc_auc_score;
        using metrics::precision_recall_curve;
        using metrics::average_precision_score;
        using metrics::mean_absolute_error;
        using metrics::mean_squared_error;
        using metrics::root_mean_squared_error;
        using metrics::mean_squared_log_error;
        using metrics::mean_absolute_percentage_error;
        using metrics::r2_score;
        using metrics::explained_variance_score;
        using metrics::max_error;
        using metrics::median_absolute_error;
        using metrics::adjusted_rand_score;
        using metrics::normalized_mutual_info_score;
        using metrics::silhouette_score;
        using metrics::cosine_similarity;
        using metrics::cosine_distance;
        using metrics::euclidean_distance;
        using metrics::manhattan_distance;
        using metrics::minkowski_distance;
        using metrics::pairwise_distances;
        using metrics::dummy_accuracy;
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XMETRICS_HPP

// math/xmetrics.hpp