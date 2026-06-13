//File 0111 : core/parallel/parallel_scan.h
//Parallel prefix scan (inclusive/exclusive) and parallel transform using the global work‑stealing scheduler; two‑pass algorithm with partial sums.
#ifndef CORE_PARALLEL_PARALLEL_SCAN_H
#define CORE_PARALLEL_PARALLEL_SCAN_H

#include "task_scheduler.h"
#include "parallel_for.h"
#include <vector>
#include <algorithm>
#include <cstdint>
#include <atomic>

namespace SimulationMath {
namespace parallel {

// -----------------------------------------------------------------------------
// 1. Inclusive parallel scan (prefix sum) for a range [first, last) into output.
//    The binary operation must be associative (e.g., addition, multiplication).
//    The output range must already have the same number of elements as input.
// -----------------------------------------------------------------------------
template <typename InputIterator, typename OutputIterator, typename T, typename BinaryOp>
void inclusive_scan(InputIterator first, InputIterator last,
                    OutputIterator output, T identity, BinaryOp&& op) noexcept {
    size_t n = std::distance(first, last);
    if (n == 0) return;

    TaskScheduler& scheduler = TaskScheduler::instance();
    unsigned num_workers = scheduler.num_workers();
    if (num_workers <= 1 || n <= 1024) {
        // Serial fallback
        T sum = identity;
        for (size_t i = 0; i < n; ++i) {
            sum = op(sum, *first++);
            *output++ = sum;
        }
        return;
    }

    // Chunk size
    size_t chunk_size = (n + num_workers - 1) / num_workers;
    size_t num_chunks = (n + chunk_size - 1) / chunk_size;

    // Per‑chunk partial sums
    std::vector<T> partial_sums(num_chunks, identity);

    // Phase 1: compute partial sums for each chunk (in parallel)
    parallel_for(num_chunks, [&](size_t c) {
        size_t start = c * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        if (start >= end) return;
        T local = identity;
        for (size_t i = start; i < end; ++i) {
            local = op(local, first[i]);
        }
        partial_sums[c] = local;
    });

    // Phase 2: compute exclusive prefix of partial sums (serial)
    std::vector<T> exclusive_prefix(num_chunks, identity);
    T running = identity;
    for (size_t c = 0; c < num_chunks; ++c) {
        exclusive_prefix[c] = running;
        running = op(running, partial_sums[c]);
    }

    // Phase 3: in parallel, produce final output using the exclusive prefix
    parallel_for(num_chunks, [&](size_t c) {
        size_t start = c * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        if (start >= end) return;
        T sum = exclusive_prefix[c];
        for (size_t i = start; i < end; ++i) {
            sum = op(sum, first[i]);
            output[i] = sum;
        }
    });
}

// -----------------------------------------------------------------------------
// 2. Exclusive parallel scan (prefix sum) for a range [first, last) into output.
//    The binary operation must be associative.
//    The first element of output is identity.
// -----------------------------------------------------------------------------
template <typename InputIterator, typename OutputIterator, typename T, typename BinaryOp>
void exclusive_scan(InputIterator first, InputIterator last,
                    OutputIterator output, T identity, BinaryOp&& op) noexcept {
    size_t n = std::distance(first, last);
    if (n == 0) return;

    TaskScheduler& scheduler = TaskScheduler::instance();
    unsigned num_workers = scheduler.num_workers();
    if (num_workers <= 1 || n <= 1024) {
        T sum = identity;
        for (size_t i = 0; i < n; ++i) {
            *output++ = sum;
            sum = op(sum, *first++);
        }
        return;
    }

    size_t chunk_size = (n + num_workers - 1) / num_workers;
    size_t num_chunks = (n + chunk_size - 1) / chunk_size;

    std::vector<T> partial_sums(num_chunks, identity);
    parallel_for(num_chunks, [&](size_t c) {
        size_t start = c * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        if (start >= end) return;
        T local = identity;
        for (size_t i = start; i < end; ++i) {
            local = op(local, first[i]);
        }
        partial_sums[c] = local;
    });

    std::vector<T> exclusive_prefix(num_chunks, identity);
    T running = identity;
    for (size_t c = 0; c < num_chunks; ++c) {
        exclusive_prefix[c] = running;
        running = op(running, partial_sums[c]);
    }

    parallel_for(num_chunks, [&](size_t c) {
        size_t start = c * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        if (start >= end) return;
        T sum = exclusive_prefix[c];
        for (size_t i = start; i < end; ++i) {
            output[i] = sum;
            sum = op(sum, first[i]);
        }
    });
}

// -----------------------------------------------------------------------------
// 3. Convenience parallel inclusive scan over std::vector (in‑place)
// -----------------------------------------------------------------------------
template <typename T, typename BinaryOp = std::plus<T>>
void inclusive_scan_inplace(std::vector<T>& data, T identity = T{}, BinaryOp op = BinaryOp()) noexcept {
    inclusive_scan(data.begin(), data.end(), data.begin(), identity, std::move(op));
}

// -----------------------------------------------------------------------------
// 4. Convenience parallel exclusive scan over std::vector (in‑place)
// -----------------------------------------------------------------------------
template <typename T, typename BinaryOp = std::plus<T>>
void exclusive_scan_inplace(std::vector<T>& data, T identity = T{}, BinaryOp op = BinaryOp()) noexcept {
    // Need a temporary output; we'll create a copy.
    std::vector<T> temp(data.size());
    exclusive_scan(data.begin(), data.end(), temp.begin(), identity, std::move(op));
    data.swap(temp);
}

// -----------------------------------------------------------------------------
// 5. Parallel transform over a range [first, last) into output, applying a unary function.
// -----------------------------------------------------------------------------
template <typename InputIterator, typename OutputIterator, typename UnaryFunc>
void parallel_transform(InputIterator first, InputIterator last,
                        OutputIterator output, UnaryFunc&& func) noexcept {
    size_t n = std::distance(first, last);
    if (n == 0) return;
    parallel_for(n, [&](size_t i) {
        output[i] = func(first[i]);
    });
}

} // namespace parallel
} // namespace SimulationMath

#endif // CORE_PARALLEL_PARALLEL_SCAN_H