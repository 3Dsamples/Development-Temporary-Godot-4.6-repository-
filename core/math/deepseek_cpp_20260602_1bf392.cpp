//File 0108 : core/parallel/parallel_for.h
//Parallel for and for_each in 1D/2D/3D using the global work‑stealing scheduler with automatic chunking and a local busy‑wait completion.
#ifndef CORE_PARALLEL_PARALLEL_FOR_H
#define CORE_PARALLEL_PARALLEL_FOR_H

#include "task_scheduler.h"
#include <atomic>
#include <thread>
#include <cstdint>
#include <algorithm>

namespace SimulationMath {
namespace parallel {

// -----------------------------------------------------------------------------
// 1. Parallel for over a 1D index range [0, n)
// -----------------------------------------------------------------------------
template <typename Index, typename Function>
void parallel_for(Index n, Function&& func) noexcept {
    if (n <= 0) return;
    TaskScheduler& scheduler = TaskScheduler::instance();
    unsigned num_workers = scheduler.num_workers();
    if (num_workers == 0) {
        // Fallback to serial if no workers (e.g., not initialised)
        for (Index i = 0; i < n; ++i) func(i);
        return;
    }

    Index chunk_size = (n + num_workers - 1) / num_workers;
    std::atomic<Index> remaining_tasks(num_workers);

    for (unsigned w = 0; w < num_workers; ++w) {
        Index start = w * chunk_size;
        Index end = (start + chunk_size) < n ? (start + chunk_size) : n;
        if (start >= end) {
            remaining_tasks.fetch_sub(1, std::memory_order_acq_rel);
            continue;
        }
        scheduler.submit([start, end, &func, &remaining_tasks]() {
            for (Index i = start; i < end; ++i) func(i);
            remaining_tasks.fetch_sub(1, std::memory_order_acq_rel);
        });
    }

    // Wait until all chunks are processed (busy‑wait with yield)
    while (remaining_tasks.load(std::memory_order_acquire) > 0) {
        std::this_thread::yield();
    }
}

// -----------------------------------------------------------------------------
// 2. Parallel for_each over a container (using iterators or indices)
// -----------------------------------------------------------------------------
template <typename Container, typename Function>
void parallel_for_each(Container& container, Function&& func) noexcept {
    size_t n = container.size();
    parallel_for(n, [&](size_t i) { func(container[i]); });
}

// -----------------------------------------------------------------------------
// 3. Parallel 2D for over a rectangular grid [0, nx) x [0, ny)
// -----------------------------------------------------------------------------
template <typename Index, typename Function>
void parallel_for_2d(Index nx, Index ny, Function&& func) noexcept {
    parallel_for(nx * ny, [&](Index linear) {
        Index y = linear / nx;
        Index x = linear % nx;
        func(x, y);
    });
}

// -----------------------------------------------------------------------------
// 4. Parallel 3D for over a volumetric grid [0, nx) x [0, ny) x [0, nz)
// -----------------------------------------------------------------------------
template <typename Index, typename Function>
void parallel_for_3d(Index nx, Index ny, Index nz, Function&& func) noexcept {
    parallel_for(nx * ny * nz, [&](Index linear) {
        Index z = linear / (nx * ny);
        Index remainder = linear % (nx * ny);
        Index y = remainder / nx;
        Index x = remainder % nx;
        func(x, y, z);
    });
}

} // namespace parallel
} // namespace SimulationMath

#endif // CORE_PARALLEL_PARALLEL_FOR_H