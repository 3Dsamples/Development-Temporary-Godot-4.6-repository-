// math/xmetrics.hpp

#ifndef XTENSOR_XMETRICS_HPP
#define XTENSOR_XMETRICS_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xreducer.hpp"
#include "../core/xview.hpp"
#include "xsorting.hpp"
#include "xstats.hpp"
#include "xmissing.hpp"
#include "xlinalg.hpp"

#include <cmath>
#include <complex>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <vector>
#include <functional>
#include <stdexcept>
#include <limits>
#include <tuple>
#include <map>
#include <unordered_map>
#include <set>
#include <string>
#include <sstream>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace metrics
        {
            // --------------------------------------------------------------------
            // Utility functions for classification metrics
            // --------------------------------------------------------------------
            
            namespace detail
            {
                // Check if two arrays have same shape
                template <class E1, class E2>
                inline bool same_shape(const xexpression<E1>& e1, const xexpression<E2>& e2)
                {
                    const auto& a = e1.derived_cast();
                    const auto& b = e2.derived_cast();
                    if (a.dimension() != b.dimension()) return false;
                    for (std::size_t d = 0; d < a.dimension(); ++d)
                        if (a.shape()[d] != b.shape()[d]) return false;
                    return true;
                }
                
                // Convert probabilities to class predictions (argmax along axis)
                template <class E>
                inline auto argmax_axis(const xexpression<E>& proba, std::size_t axis = 1)
                {
                    const auto& expr = proba.derived_cast();
                    std::size_t ax = normalize_axis(static_cast<std::ptrdiff_t>(axis), expr.dimension());
                    auto result_shape = expr.shape();
                    result_shape[ax] = 1;
                    using size_type = typename E::size_type;
                    xarray_container<size_type> result(result_shape);
                    
                    std::size_t num_slices = expr.size() / expr.shape()[ax];
                    std::size_t axis_len = expr.shape()[ax];
                    
                    for (std::size_t slice = 0; slice < num_slices; ++slice)
                    {
                        std::vector<std::size_t> coords(expr.dimension(), 0);
                        std::size_t temp = slice;
                        for (std::size_t d = 0; d < expr.dimension(); ++d)
                        {
                            if (d == ax) continue;
                            std::size_t stride_after = 1;
                            for (std::size_t k = d + 1; k < expr.dimension(); ++k)
                                if (k != ax) stride_after *= expr.shape()[k];
                            coords[d] = temp / stride_after;
                            temp %= stride_after;
                        }
                        
                        size_type best_idx = 0;
                        auto best_val = expr.element(coords);
                        for (size_type i = 1; i < axis_len; ++i)
                        {
                            coords[ax] = i;
                            auto val = expr.element(coords);
                            if (val > best_val)
                            {
                                best_val = val;
                                best_idx = i;
                            }
                        }
                        coords[ax] = 0;
                        result.element(coords) = best_idx;
                    }
                    return squeeze(result, ax);
                }
                
                // Binarize predictions based on threshold
                template <class E>
                inline auto binarize(const xexpression<E>& e, double threshold = 0.5)
                {
                    const auto& expr = e.derived_cast();
                    using value_type = typename E::value_type;
                    xarray_container<int> result(expr.shape());
                    for (std::size_t i = 0; i < expr.size(); ++i)
                    {
                        result.flat(i) = (expr.flat(i) > threshold) ? 1 : 0;
                    }
                    return result;
                }
            }
            
            // --------------------------------------------------------------------
            // Classification metrics
            // --------------------------------------------------------------------
            
            // Confusion matrix
            template <class E1, class E2>
            inline auto confusion_matrix(const xexpression<E1>& y_true, const xexpression<E2>& y_pred,
                                         std::size_t n_classes = 0)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                
                if (!detail::same_shape(true_expr, pred_expr))
                {
                    XTENSOR_THROW(std::invalid_argument, "confusion_matrix: shapes must match");
                }
                
                // Determine number of classes if not provided
                if (n_classes == 0)
                {
                    int max_true = *std::max_element(true_expr.begin(), true_expr.end());
                    int max_pred = *std::max_element(pred_expr.begin(), pred_expr.end());
                    n_classes = static_cast<std::size_t>(std::max(max_true, max_pred) + 1);
                }
                
                xarray_container<std::size_t> cm(std::vector<std::size_t>{n_classes, n_classes}, 0);
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    std::size_t t = static_cast<std::size_t>(true_expr.flat(i));
                    std::size_t p = static_cast<std::size_t>(pred_expr.flat(i));
                    if (t < n_classes && p < n_classes)
                        cm(t, p)++;
                }
                return cm;
            }
            
            // Accuracy score
            template <class E1, class E2>
            inline double accuracy_score(const xexpression<E1>& y_true, const xexpression<E2>& y_pred)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "accuracy_score: shapes must match");
                std::size_t correct = 0;
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                    if (true_expr.flat(i) == pred_expr.flat(i)) ++correct;
                return static_cast<double>(correct) / static_cast<double>(true_expr.size());
            }
            
            // Precision score (macro, micro, weighted, binary)
            template <class E1, class E2>
            inline double precision_score(const xexpression<E1>& y_true, const xexpression<E2>& y_pred,
                                          const std::string& average = "binary", std::size_t pos_label = 1)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "precision_score: shapes must match");
                
                if (average == "binary")
                {
                    std::size_t tp = 0, fp = 0;
                    for (std::size_t i = 0; i < true_expr.size(); ++i)
                    {
                        bool t = (static_cast<std::size_t>(true_expr.flat(i)) == pos_label);
                        bool p = (static_cast<std::size_t>(pred_expr.flat(i)) == pos_label);
                        if (p && t) tp++;
                        else if (p && !t) fp++;
                    }
                    return (tp + fp) > 0 ? static_cast<double>(tp) / static_cast<double>(tp + fp) : 0.0;
                }
                
                // Determine classes
                std::set<std::size_t> classes;
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    classes.insert(static_cast<std::size_t>(true_expr.flat(i)));
                    classes.insert(static_cast<std::size_t>(pred_expr.flat(i)));
                }
                std::vector<std::size_t> class_list(classes.begin(), classes.end());
                std::size_t n_classes = class_list.size();
                
                std::vector<std::size_t> tp(n_classes, 0), fp(n_classes, 0), support(n_classes, 0);
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    std::size_t t = static_cast<std::size_t>(true_expr.flat(i));
                    std::size_t p = static_cast<std::size_t>(pred_expr.flat(i));
                    auto it_t = std::find(class_list.begin(), class_list.end(), t);
                    auto it_p = std::find(class_list.begin(), class_list.end(), p);
                    if (it_t != class_list.end())
                    {
                        std::size_t idx_t = static_cast<std::size_t>(std::distance(class_list.begin(), it_t));
                        support[idx_t]++;
                        if (t == p)
                            tp[idx_t]++;
                    }
                    if (it_p != class_list.end())
                    {
                        std::size_t idx_p = static_cast<std::size_t>(std::distance(class_list.begin(), it_p));
                        if (t != p)
                            fp[idx_p]++;
                    }
                }
                
                if (average == "micro")
                {
                    std::size_t total_tp = std::accumulate(tp.begin(), tp.end(), std::size_t(0));
                    std::size_t total_fp = std::accumulate(fp.begin(), fp.end(), std::size_t(0));
                    return (total_tp + total_fp) > 0 ? static_cast<double>(total_tp) / static_cast<double>(total_tp + total_fp) : 0.0;
                }
                else if (average == "macro")
                {
                    double sum = 0.0;
                    std::size_t count = 0;
                    for (std::size_t i = 0; i < n_classes; ++i)
                    {
                        if (support[i] > 0)
                        {
                            double prec = (tp[i] + fp[i]) > 0 ? static_cast<double>(tp[i]) / static_cast<double>(tp[i] + fp[i]) : 0.0;
                            sum += prec;
                            count++;
                        }
                    }
                    return count > 0 ? sum / static_cast<double>(count) : 0.0;
                }
                else if (average == "weighted")
                {
                    double weighted_sum = 0.0;
                    std::size_t total_support = std::accumulate(support.begin(), support.end(), std::size_t(0));
                    if (total_support == 0) return 0.0;
                    for (std::size_t i = 0; i < n_classes; ++i)
                    {
                        if (support[i] > 0)
                        {
                            double prec = (tp[i] + fp[i]) > 0 ? static_cast<double>(tp[i]) / static_cast<double>(tp[i] + fp[i]) : 0.0;
                            weighted_sum += prec * static_cast<double>(support[i]);
                        }
                    }
                    return weighted_sum / static_cast<double>(total_support);
                }
                return 0.0;
            }
            
            // Recall score
            template <class E1, class E2>
            inline double recall_score(const xexpression<E1>& y_true, const xexpression<E2>& y_pred,
                                       const std::string& average = "binary", std::size_t pos_label = 1)
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                if (!detail::same_shape(true_expr, pred_expr))
                    XTENSOR_THROW(std::invalid_argument, "recall_score: shapes must match");
                
                if (average == "binary")
                {
                    std::size_t tp = 0, fn = 0;
                    for (std::size_t i = 0; i < true_expr.size(); ++i)
                    {
                        bool t = (static_cast<std::size_t>(true_expr.flat(i)) == pos_label);
                        bool p = (static_cast<std::size_t>(pred_expr.flat(i)) == pos_label);
                        if (p && t) tp++;
                        else if (!p && t) fn++;
                    }
                    return (tp + fn) > 0 ? static_cast<double>(tp) / static_cast<double>(tp + fn) : 0.0;
                }
                
                std::set<std::size_t> classes;
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    classes.insert(static_cast<std::size_t>(true_expr.flat(i)));
                    classes.insert(static_cast<std::size_t>(pred_expr.flat(i)));
                }
                std::vector<std::size_t> class_list(classes.begin(), classes.end());
                std::size_t n_classes = class_list.size();
                
                std::vector<std::size_t> tp(n_classes, 0), fn(n_classes, 0), support(n_classes, 0);
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    std::size_t t = static_cast<std::size_t>(true_expr.flat(i));
                    std::size_t p = static_cast<std::size_t>(pred_expr.flat(i));
                    auto it_t = std::find(class_list.begin(), class_list.end(), t);
                    if (it_t != class_list.end())
                    {
                        std::size_t idx_t = static_cast<std::size_t>(std::distance(class_list.begin(), it_t));
                        support[idx_t]++;
                        if (t == p)
                            tp[idx_t]++;
                        else
                            fn[idx_t]++;
                    }
                }
                
                if (average == "micro")
                {
                    std::size_t total_tp = std::accumulate(tp.begin(), tp.end(), std::size_t(0));
                    std::size_t total_fn = std::accumulate(fn.begin(), fn.end(), std::size_t(0));
                    return (total_tp + total_fn) > 0 ? static_cast<double>(total_tp) / static_cast<double>(total_tp + total_fn) : 0.0;
                }
                else if (average == "macro")
                {
                    double sum = 0.0;
                    std::size_t count = 0;
                    for (std::size_t i = 0; i < n_classes; ++i)
                    {
                        if (support[i] > 0)
                        {
                            double rec = (tp[i] + fn[i]) > 0 ? static_cast<double>(tp[i]) / static_cast<double>(tp[i] + fn[i]) : 0.0;
                            sum += rec;
                            count++;
                        }
                    }
                    return count > 0 ? sum / static_cast<double>(count) : 0.0;
                }
                else if (average == "weighted")
                {
                    double weighted_sum = 0.0;
                    std::size_t total_support = std::accumulate(support.begin(), support.end(), std::size_t(0));
                    if (total_support == 0) return 0.0;
                    for (std::size_t i = 0; i < n_classes; ++i)
                    {
                        if (support[i] > 0)
                        {
                            double rec = (tp[i] + fn[i]) > 0 ? static_cast<double>(tp[i]) / static_cast<double>(tp[i] + fn[i]) : 0.0;
                            weighted_sum += rec * static_cast<double>(support[i]);
                        }
                    }
                    return weighted_sum / static_cast<double>(total_support);
                }
                return 0.0;
            }
            
            // F1 score
            template <class E1, class E2>
            inline double f1_score(const xexpression<E1>& y_true, const xexpression<E2>& y_pred,
                                   const std::string& average = "binary", std::size_t pos_label = 1)
            {
                double p = precision_score(y_true, y_pred, average, pos_label);
                double r = recall_score(y_true, y_pred, average, pos_label);
                return (p + r) > 0 ? 2.0 * p * r / (p + r) : 0.0;
            }
            
            // F-beta score
            template <class E1, class E2>
            inline double fbeta_score(const xexpression<E1>& y_true, const xexpression<E2>& y_pred,
                                      double beta, const std::string& average = "binary", std::size_t pos_label = 1)
            {
                double p = precision_score(y_true, y_pred, average, pos_label);
                double r = recall_score(y_true, y_pred, average, pos_label);
                double beta2 = beta * beta;
                return (beta2 * p + r) > 0 ? (1.0 + beta2) * p * r / (beta2 * p + r) : 0.0;
            }
            
            // Classification report (returns a string)
            template <class E1, class E2>
            inline std::string classification_report(const xexpression<E1>& y_true, const xexpression<E2>& y_pred,
                                                     const std::vector<std::string>& target_names = {})
            {
                const auto& true_expr = y_true.derived_cast();
                const auto& pred_expr = y_pred.derived_cast();
                
                std::set<std::size_t> classes_set;
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                    classes_set.insert(static_cast<std::size_t>(true_expr.flat(i)));
                std::vector<std::size_t> classes(classes_set.begin(), classes_set.end());
                std::size_t n_classes = classes.size();
                
                std::vector<std::size_t> tp(n_classes, 0), fp(n_classes, 0), fn(n_classes, 0);
                for (std::size_t i = 0; i < true_expr.size(); ++i)
                {
                    std::size_t t = static_cast<std::size_t>(true_expr.flat(i));
                    std::size_t p = static_cast<std::size_t>(pred_expr.flat(i));
                    std::size_t idx_t = static_cast<std::size_t>(std::distance(classes.begin(), std::find(classes.begin(), classes.end(), t)));
                    if (t == p)
                        tp[idx_t]++;
                    else
                    {
                        fn[idx_t]++;
                        auto it_p = std::find(classes.begin(), classes.end(), p);
                        if (it_p != classes.end())
                        {
                            std::size_t idx_p = static_cast<std::size_t>(std::distance(classes.begin(), it_p));
                            fp[idx_p]++;
                        }
                    }
                }
                
                std::ostringstream oss;
                oss << "              precision    recall  f1-score   support\n\n";
                std::size_t total_support = 0;
                double macro_p = 0, macro_r = 0, macro_f1 = 0;
                std::size_t macro_count = 0;
                
                for (std::size_t i = 0; i < n_classes; ++i)
                {
                    std::size_t support = tp[i] + fn[i];
                    if (support == 0) continue;
                    total_support += support;
                    double prec = (tp[i] + fp[i]) > 0 ? static_cast<double>(tp[i]) / (tp[i] + fp[i]) : 0.0;
                    double rec = support > 0 ? static_cast<double>(tp[i]) / support : 0.0;
                    double f1 = (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0.0;
                    std::string name = (i < target_names.size()) ? target_names[i] : std::to_string(classes[i]);
                    oss << std::setw(12) << name << "   "
                        << std::fixed << std::setprecision(2) << std::setw(8) << prec << "   "
                        << std::setw(6) << rec << "   "
                        << std::setw(8) << f1 << "   "
                        << std::setw(7) << support << "\n";
                    macro_p += prec;
                    macro_r += rec;
                    macro_f1 += f1;
                    macro_count++;
                }
                
                if (macro_count > 0)
                {
                    macro_p /= macro_count;
                    macro_r /= macro_count;
                    macro_f1 /= macro_count;
                }
                
                // Micro average
                std::size_t total_tp = std::accumulate(tp.begin(), tp.end(), std::size_t(0));
                std::size_t total_fp = std::accumulate(fp.begin(), fp.end(), std::size_t(0));
                std::size_t total_fn = std::accumulate(fn.begin(), fn.end(), std::size_t(0));
                double micro_p = (total_tp + total_fp) > 0 ? static_cast<double>(total_tp) / (total_tp + total_fp) : 0.0;
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