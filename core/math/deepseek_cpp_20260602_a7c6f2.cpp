//File 0109 : core/parallel/parallel_reduce.h
//Parallel reduce over a range with associative binary operation; deterministic tree reduction using the work‑stealing scheduler.
#ifndef CORE_PARALLEL_PARALLEL_REDUCE_H
#define CORE_PARALLEL_PARALLEL_REDUCE_H

#include "task_scheduler.h"
#include <atomic>
#include <thread>
#include <cstdint>
#include <functional>
#include <vector>
#include <algorithm>

namespace SimulationMath {
namespace parallel {

// -----------------------------------------------------------------------------
// 1. Parallel reduce over [0, n) with a value extraction function and binary operation.
//    The binary operation must be associative (but not necessarily commutative).
//    The order of combination may not be deterministic unless use_tree_reduce = true.
// -----------------------------------------------------------------------------
template <typename Index, typename T, typename ExtractFunc, typename BinaryOp>
T parallel_reduce(Index n, T identity, ExtractFunc&& extract, BinaryOp&& combine,
                  bool use_tree_reduce = false) noexcept {
    if (n <= 0) return identity;
    TaskScheduler& scheduler = TaskScheduler::instance();
    unsigned num_workers = scheduler.num_workers();
    if (num_workers == 0) {
        // Serial fallback
        T result = identity;
        for (Index i = 0; i < n; ++i) result = combine(result, extract(i));
        return result;
    }

    if (use_tree_reduce) {
        return parallel_reduce_tree(n, identity, std::forward<ExtractFunc>(extract),
                                    std::forward<BinaryOp>(combine), num_workers);
    } else {
        return parallel_reduce_flat(n, identity, std::forward<ExtractFunc>(extract),
                                    std::forward<BinaryOp>(combine), num_workers);
    }
}

// -----------------------------------------------------------------------------
// 2. Flat parallel reduce: split range into chunks, compute partial reductions,
//    then combine sequentially.
// -----------------------------------------------------------------------------
template <typename Index, typename T, typename ExtractFunc, typename BinaryOp>
T parallel_reduce_flat(Index n, T identity, ExtractFunc&& extract, BinaryOp&& combine,
                       unsigned num_workers) noexcept {
    Index chunk_size = (n + num_workers - 1) / num_workers;
    std::vector<T> partial_results(num_workers, identity);
    std::atomic<unsigned> remaining(num_workers);

    TaskScheduler& scheduler = TaskScheduler::instance();

    for (unsigned w = 0; w < num_workers; ++w) {
        Index start = w * chunk_size;
        Index end = (start + chunk_size) < n ? (start + chunk_size) : n;
        if (start >= end) {
            remaining.fetch_sub(1, std::memory_order_acq_rel);
            continue;
        }
        scheduler.submit([&, w, start, end]() {
            T local = identity;
            for (Index i = start; i < end; ++i) local = combine(local, extract(i));
            partial_results[w] = local;
            remaining.fetch_sub(1, std::memory_order_acq_rel);
        });
    }

    // Wait for all chunks to finish
    while (remaining.load(std::memory_order_acquire) > 0) {
        std::this_thread::yield();
    }

    // Combine partial results
    T result = identity;
    for (unsigned w = 0; w < num_workers; ++w) result = combine(result, partial_results[w]);
    return result;
}

// -----------------------------------------------------------------------------
// 3. Tree parallel reduce: builds a binary tree of tasks, each combining two partials.
//    Ensures deterministic order independent of number of workers.
// -----------------------------------------------------------------------------
template <typename Index, typename T, typename ExtractFunc, typename BinaryOp>
T parallel_reduce_tree(Index n, T identity, ExtractFunc&& extract, BinaryOp&& combine,
                       unsigned num_workers) noexcept {
    if (n <= 0) return identity;

    // We'll implement a two‑level approach:
    // 1) Split the range into leaves (at the granularity of chunk_size).
    // 2) Use a parallel tree reduction: combine pairs of leaves recursively.
    // Because we cannot easily spawn subtasks that return values via the current scheduler,
    // we will use a shared array and atomic counters.

    Index chunk_size = 256;   // fixed leaf size for tree reduction
    Index num_leaves = (n + chunk_size - 1) / chunk_size;
    if (num_leaves < 2) {
        // just one leaf, compute directly
        T result = identity;
        for (Index i = 0; i < n; ++i) result = combine(result, extract(i));
        return result;
    }

    std::vector<T> values(num_leaves, identity);
    std::atomic<Index> leaf_completed(0);

    TaskScheduler& scheduler = TaskScheduler::instance();

    // Launch leaf computations as tasks
    for (Index leaf = 0; leaf < num_leaves; ++leaf) {
        Index start = leaf * chunk_size;
        Index end = std::min(start + chunk_size, n);
        scheduler.submit([&, leaf, start, end]() {
            T local = identity;
            for (Index i = start; i < end; ++i) local = combine(local, extract(i));
            values[leaf] = local;
            leaf_completed.fetch_add(1, std::memory_order_acq_rel);
        });
    }

    // Wait for all leaves to be computed
    while (leaf_completed.load(std::memory_order_acquire) < num_leaves) {
        std::this_thread::yield();
    }

    // Now reduce the values array using a binary tree on the calling thread (sequential combiner)
    Index active = num_leaves;
    while (active > 1) {
        Index new_active = 0;
        for (Index i = 0; i < active; i += 2) {
            if (i + 1 < active) {
                values[new_active++] = combine(values[i], values[i + 1]);
            } else {
                values[new_active++] = values[i];
            }
        }
        active = new_active;
    }
    return values[0];
}

} // namespace parallel
} // namespace SimulationMath

#endif // CORE_PARALLEL_PARALLEL_REDUCE_H