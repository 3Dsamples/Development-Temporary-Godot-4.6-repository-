//File 0110 : core/parallel/parallel_sort.h
//Parallel sample sort using the global task scheduler: random splitters, atomic bucket sizes, parallel sorting per bucket, and efficient raw array version with a single temporary buffer.
#ifndef CORE_PARALLEL_PARALLEL_SORT_H
#define CORE_PARALLEL_PARALLEL_SORT_H

#include "task_scheduler.h"
#include "parallel_for.h"
#include <atomic>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

namespace SimulationMath {
namespace parallel {

// -----------------------------------------------------------------------------
// 1. Parallel sample sort for std::vector<T> using the global scheduler.
//    The comparator Comp must be a strict weak ordering.
// -----------------------------------------------------------------------------
template <typename T, typename Comp = std::less<T>>
void parallel_sort(std::vector<T>& data, Comp comp = Comp()) noexcept {
    size_t n = data.size();
    if (n <= 1024) {
        std::sort(data.begin(), data.end(), comp);
        return;
    }

    TaskScheduler& scheduler = TaskScheduler::instance();
    unsigned p = scheduler.num_workers();
    if (p <= 1) {
        std::sort(data.begin(), data.end(), comp);
        return;
    }

    // --- Step 1: Select splitters ---
    const unsigned sample_size = 3 * p;
    std::vector<T> sample(sample_size);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    for (unsigned i = 0; i < sample_size; ++i)
        sample[i] = data[dist(rng)];

    std::sort(sample.begin(), sample.end(), comp);
    std::vector<T> splitters(p - 1);
    for (unsigned i = 0; i < p - 1; ++i)
        splitters[i] = sample[(i + 1) * sample_size / p];

    // --- Step 2: Compute bucket sizes with atomic counters ---
    std::vector<std::atomic<size_t>> bucket_counts(p);
    for (auto& c : bucket_counts) c.store(0, std::memory_order_relaxed);

    parallel_for(n, [&](size_t i) {
        const T& val = data[i];
        auto it = std::upper_bound(splitters.begin(), splitters.end(), val, comp);
        unsigned bucket = static_cast<unsigned>(it - splitters.begin());
        bucket_counts[bucket].fetch_add(1, std::memory_order_relaxed);
    });

    // --- Step 3: Compute bucket offsets ---
    std::vector<size_t> bucket_offsets(p, 0);
    size_t total = 0;
    for (unsigned i = 0; i < p; ++i) {
        bucket_offsets[i] = total;
        total += bucket_counts[i].load(std::memory_order_acquire);
    }

    // --- Step 4: Copy elements into temporary storage ---
    std::vector<T> temp(total);
    for (auto& c : bucket_counts) c.store(0, std::memory_order_relaxed);

    parallel_for(n, [&](size_t i) {
        const T& val = data[i];
        auto it = std::upper_bound(splitters.begin(), splitters.end(), val, comp);
        unsigned bucket = static_cast<unsigned>(it - splitters.begin());
        size_t pos = bucket_offsets[bucket] + bucket_counts[bucket].fetch_add(1, std::memory_order_relaxed);
        temp[pos] = val;
    });

    // --- Step 5: Sort each bucket in parallel ---
    std::vector<std::pair<size_t, size_t>> bucket_ranges(p);
    for (unsigned i = 0; i < p; ++i) {
        bucket_ranges[i] = {bucket_offsets[i], bucket_counts[i].load(std::memory_order_acquire)};
    }

    parallel_for(p, [&](unsigned i) {
        size_t start = bucket_ranges[i].first;
        size_t len   = bucket_ranges[i].second;
        if (len > 1)
            std::sort(temp.begin() + start, temp.begin() + start + len, comp);
    });

    // --- Step 6: Swap to finalize ---
    data.swap(temp);
}

// -----------------------------------------------------------------------------
// 2. Parallel sample sort for a raw pointer array (in‑place with temporary buffer)
// -----------------------------------------------------------------------------
template <typename T, typename Comp = std::less<T>>
void parallel_sort(T* data, size_t n, Comp comp = Comp()) noexcept {
    if (n <= 1024) {
        std::sort(data, data + n, comp);
        return;
    }

    TaskScheduler& scheduler = TaskScheduler::instance();
    unsigned p = scheduler.num_workers();
    if (p <= 1) {
        std::sort(data, data + n, comp);
        return;
    }

    // --- Step 1: Select splitters ---
    const unsigned sample_size = 3 * p;
    std::vector<T> sample(sample_size);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    for (unsigned i = 0; i < sample_size; ++i)
        sample[i] = data[dist(rng)];

    std::sort(sample.begin(), sample.end(), comp);
    std::vector<T> splitters(p - 1);
    for (unsigned i = 0; i < p - 1; ++i)
        splitters[i] = sample[(i + 1) * sample_size / p];

    // --- Step 2: Compute bucket sizes ---
    std::vector<std::atomic<size_t>> bucket_counts(p);
    for (auto& c : bucket_counts) c.store(0, std::memory_order_relaxed);

    parallel_for(n, [&](size_t i) {
        const T& val = data[i];
        auto it = std::upper_bound(splitters.begin(), splitters.end(), val, comp);
        unsigned bucket = static_cast<unsigned>(it - splitters.begin());
        bucket_counts[bucket].fetch_add(1, std::memory_order_relaxed);
    });

    // --- Step 3: Compute offsets ---
    std::vector<size_t> bucket_offsets(p, 0);
    size_t total = 0;
    for (unsigned i = 0; i < p; ++i) {
        bucket_offsets[i] = total;
        total += bucket_counts[i].load(std::memory_order_acquire);
    }

    // --- Step 4: Copy into temporary buffer ---
    // Allocate a single buffer of size n
    std::unique_ptr<T[]> temp(new T[total]);
    for (auto& c : bucket_counts) c.store(0, std::memory_order_relaxed);

    parallel_for(n, [&](size_t i) {
        const T& val = data[i];
        auto it = std::upper_bound(splitters.begin(), splitters.end(), val, comp);
        unsigned bucket = static_cast<unsigned>(it - splitters.begin());
        size_t pos = bucket_offsets[bucket] + bucket_counts[bucket].fetch_add(1, std::memory_order_relaxed);
        temp[pos] = std::move(val);
    });

    // --- Step 5: Sort each bucket in parallel ---
    parallel_for(p, [&](unsigned i) {
        size_t start = bucket_offsets[i];
        size_t len   = bucket_counts[i].load(std::memory_order_acquire);
        if (len > 1)
            std::sort(temp.get() + start, temp.get() + start + len, comp);
    });

    // --- Step 6: Move back to original array ---
    for (size_t i = 0; i < n; ++i) {
        data[i] = std::move(temp[i]);
    }
}

} // namespace parallel
} // namespace SimulationMath

#endif // CORE_PARALLEL_PARALLEL_SORT_H