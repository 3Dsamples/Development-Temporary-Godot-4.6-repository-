// include/xtu/parallel/xparallel.hpp
// xtensor-unified - Parallel execution policies and thread pool
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_PARALLEL_XPARALLEL_HPP
#define XTU_PARALLEL_XPARALLEL_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"

#ifdef XTU_USE_OPENMP
#include <omp.h>
#endif

#ifdef XTU_USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>
#endif

XTU_NAMESPACE_BEGIN
namespace parallel {

// #############################################################################
// Execution policies
// #############################################################################
enum class execution_policy {
    sequential,
    parallel,
    parallel_unseq,
    unseq
};

// #############################################################################
// Thread pool configuration and management
// #############################################################################
class thread_config {
private:
    static size_t s_max_threads;
    static thread_local size_t s_thread_id;
    static bool s_initialized;

public:
    static void initialize(size_t num_threads = 0) {
        if (s_initialized) return;
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
        }
        s_max_threads = num_threads;
#ifdef XTU_USE_OPENMP
        omp_set_num_threads(static_cast<int>(num_threads));
#endif
#ifdef XTU_USE_TBB
        tbb::task_arena::initialize(tbb::task_arena::automatic);
#endif
        s_initialized = true;
    }

    static size_t max_threads() {
        if (!s_initialized) initialize();
        return s_max_threads;
    }

    static size_t thread_id() {
#ifdef XTU_USE_OPENMP
        return static_cast<size_t>(omp_get_thread_num());
#else
        return s_thread_id;
#endif
    }

    static void set_thread_id(size_t id) {
        s_thread_id = id;
    }

    static size_t num_threads() {
#ifdef XTU_USE_OPENMP
        return static_cast<size_t>(omp_get_max_threads());
#else
        return s_max_threads;
#endif
    }
};

// Static members initialization
inline size_t thread_config::s_max_threads = 0;
inline thread_local size_t thread_config::s_thread_id = 0;
inline bool thread_config::s_initialized = false;

// #############################################################################
// Parallel for loop (static scheduling)
// #############################################################################
template <class Index, class Func>
void parallel_for(Index start, Index end, Func&& func, execution_policy policy = execution_policy::parallel) {
    if (policy == execution_policy::sequential) {
        for (Index i = start; i < end; ++i) {
            func(i);
        }
        return;
    }
#ifdef XTU_USE_OPENMP
    if (policy == execution_policy::parallel) {
        #pragma omp parallel for schedule(static)
        for (Index i = start; i < end; ++i) {
            func(i);
        }
        return;
    }
#endif
#ifdef XTU_USE_TBB
    if (policy == execution_policy::parallel) {
        tbb::parallel_for(tbb::blocked_range<Index>(start, end),
            [&func](const tbb::blocked_range<Index>& r) {
                for (Index i = r.begin(); i != r.end(); ++i) {
                    func(i);
                }
            });
        return;
    }
#endif
    // Fallback sequential
    for (Index i = start; i < end; ++i) {
        func(i);
    }
}

// #############################################################################
// Parallel for with chunk size (dynamic/guided scheduling)
// #############################################################################
template <class Index, class Func>
void parallel_for_dynamic(Index start, Index end, Index chunk_size, Func&& func) {
#ifdef XTU_USE_OPENMP
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (Index i = start; i < end; ++i) {
        func(i);
    }
#else
    // Fallback: static partitioning
    size_t n = static_cast<size_t>(end - start);
    size_t num_threads = thread_config::max_threads();
    size_t chunk = (n + num_threads - 1) / num_threads;
    if (chunk < static_cast<size_t>(chunk_size)) chunk = static_cast<size_t>(chunk_size);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        Index t_start = start + static_cast<Index>(t * chunk);
        Index t_end = std::min(t_start + static_cast<Index>(chunk), end);
        if (t_start >= end) break;
        threads.emplace_back([=, &func]() {
            thread_config::set_thread_id(t);
            for (Index i = t_start; i < t_end; ++i) {
                func(i);
            }
        });
    }
    for (auto& th : threads) th.join();
#endif
}

// #############################################################################
// Parallel reduce (sum, min, max, etc.)
// #############################################################################
template <class Index, class T, class Func, class Combine>
T parallel_reduce(Index start, Index end, T identity, Func&& func, Combine&& combine,
                  execution_policy policy = execution_policy::parallel) {
    if (policy == execution_policy::sequential) {
        T result = identity;
        for (Index i = start; i < end; ++i) {
            result = combine(result, func(i));
        }
        return result;
    }
#ifdef XTU_USE_OPENMP
    if (policy == execution_policy::parallel) {
        T result = identity;
        #pragma omp parallel for reduction(combine:result)
        for (Index i = start; i < end; ++i) {
            result = combine(result, func(i));
        }
        return result;
    }
#endif
#ifdef XTU_USE_TBB
    if (policy == execution_policy::parallel) {
        return tbb::parallel_reduce(
            tbb::blocked_range<Index>(start, end),
            identity,
            [&func](const tbb::blocked_range<Index>& r, T init) -> T {
                T local = init;
                for (Index i = r.begin(); i != r.end(); ++i) {
                    local = combine(local, func(i));
                }
                return local;
            },
            combine
        );
    }
#endif
    // Fallback: manual parallel reduction
    size_t n = static_cast<size_t>(end - start);
    size_t num_threads = thread_config::max_threads();
    size_t chunk = (n + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    std::vector<T> partial(num_threads, identity);
    for (size_t t = 0; t < num_threads; ++t) {
        Index t_start = start + static_cast<Index>(t * chunk);
        Index t_end = std::min(t_start + static_cast<Index>(chunk), end);
        if (t_start >= end) break;
        threads.emplace_back([=, &func, &combine, &partial]() {
            T local = identity;
            for (Index i = t_start; i < t_end; ++i) {
                local = combine(local, func(i));
            }
            partial[t] = local;
        });
    }
    for (auto& th : threads) th.join();
    T result = identity;
    for (const auto& p : partial) result = combine(result, p);
    return result;
}

// #############################################################################
// Parallel transform (apply function to range, store in output)
// #############################################################################
template <class InputIt, class OutputIt, class Func>
void parallel_transform(InputIt first, InputIt last, OutputIt out, Func&& func,
                        execution_policy policy = execution_policy::parallel) {
    size_t n = static_cast<size_t>(std::distance(first, last));
    if (policy == execution_policy::sequential) {
        std::transform(first, last, out, std::forward<Func>(func));
        return;
    }
#ifdef XTU_USE_OPENMP
    if (policy == execution_policy::parallel) {
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            out[i] = func(first[i]);
        }
        return;
    }
#endif
#ifdef XTU_USE_TBB
    if (policy == execution_policy::parallel) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    out[i] = func(first[i]);
                }
            });
        return;
    }
#endif
    // Fallback: manual threading
    size_t num_threads = thread_config::max_threads();
    size_t chunk = (n + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t t_start = t * chunk;
        size_t t_end = std::min(t_start + chunk, n);
        if (t_start >= n) break;
        threads.emplace_back([=, &func, first, out]() {
            for (size_t i = t_start; i < t_end; ++i) {
                out[i] = func(first[i]);
            }
        });
    }
    for (auto& th : threads) th.join();
}

// #############################################################################
// Parallel evaluation of expression assignment
// #############################################################################
template <class Container, class Expr>
void parallel_assign(Container& dest, const Expr& expr, execution_policy policy = execution_policy::parallel) {
    size_t size = dest.size();
    if (policy == execution_policy::sequential) {
        for (size_t i = 0; i < size; ++i) {
            dest.flat(i) = expr.flat(i);
        }
        return;
    }
#ifdef XTU_USE_OPENMP
    if (policy == execution_policy::parallel) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            dest.flat(i) = expr.flat(i);
        }
        return;
    }
#endif
#ifdef XTU_USE_TBB
    if (policy == execution_policy::parallel) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, size),
            [&dest, &expr](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    dest.flat(i) = expr.flat(i);
                }
            });
        return;
    }
#endif
    // Fallback: manual threading
    size_t num_threads = thread_config::max_threads();
    size_t chunk = (size + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t t_start = t * chunk;
        size_t t_end = std::min(t_start + chunk, size);
        if (t_start >= size) break;
        threads.emplace_back([=, &dest, &expr]() {
            for (size_t i = t_start; i < t_end; ++i) {
                dest.flat(i) = expr.flat(i);
            }
        });
    }
    for (auto& th : threads) th.join();
}

// #############################################################################
// Parallel sort (multi-threaded)
// #############################################################################
template <class RandomIt>
void parallel_sort(RandomIt first, RandomIt last, execution_policy policy = execution_policy::parallel) {
    if (policy == execution_policy::sequential) {
        std::sort(first, last);
        return;
    }
#ifdef XTU_USE_TBB
    if (policy == execution_policy::parallel) {
        tbb::parallel_sort(first, last);
        return;
    }
#endif
#ifdef XTU_USE_OPENMP
    // OpenMP doesn't have built-in parallel sort; fallback to manual
#endif
    // Manual parallel sort using std::sort on chunks then merge
    size_t n = static_cast<size_t>(std::distance(first, last));
    if (n < 10000) {
        std::sort(first, last);
        return;
    }
    size_t num_threads = thread_config::max_threads();
    size_t chunk = (n + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t t_start = t * chunk;
        size_t t_end = std::min(t_start + chunk, n);
        if (t_start >= n) break;
        threads.emplace_back([=]() {
            std::sort(first + t_start, first + t_end);
        });
    }
    for (auto& th : threads) th.join();
    // Merge sorted chunks (simple pairwise merge)
    std::vector<typename std::iterator_traits<RandomIt>::value_type> temp(n);
    size_t chunk_count = (n + chunk - 1) / chunk;
    std::vector<size_t> offsets(chunk_count + 1);
    for (size_t i = 0; i <= chunk_count; ++i) {
        offsets[i] = std::min(i * chunk, n);
    }
    while (chunk_count > 1) {
        size_t new_count = 0;
        for (size_t i = 0; i < chunk_count; i += 2) {
            if (i + 1 < chunk_count) {
                std::merge(first + offsets[i], first + offsets[i+1],
                           first + offsets[i+1], first + offsets[i+2],
                           temp.begin() + offsets[i]);
                offsets[new_count++] = offsets[i];
            } else {
                std::copy(first + offsets[i], first + offsets[i+1],
                          temp.begin() + offsets[i]);
                offsets[new_count++] = offsets[i];
            }
        }
        offsets[new_count] = n;
        chunk_count = new_count;
        std::copy(temp.begin(), temp.begin() + n, first);
    }
}

// #############################################################################
// Thread pool for task-based parallelism
// #############################################################################
class thread_pool {
private:
    std::vector<std::thread> m_workers;
    std::vector<std::function<void()>> m_tasks;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_stop;
    size_t m_active;

public:
    explicit thread_pool(size_t num_threads = 0) : m_stop(false), m_active(0) {
        if (num_threads == 0) num_threads = thread_config::max_threads();
        for (size_t i = 0; i < num_threads; ++i) {
            m_workers.emplace_back([this, i]() {
                thread_config::set_thread_id(i);
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_cv.wait(lock, [this]() { return m_stop || !m_tasks.empty(); });
                        if (m_stop && m_tasks.empty()) return;
                        task = std::move(m_tasks.back());
                        m_tasks.pop_back();
                        ++m_active;
                    }
                    task();
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        --m_active;
                    }
                    m_cv.notify_all();
                }
            });
        }
    }

    ~thread_pool() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        for (auto& w : m_workers) w.join();
    }

    template <class F>
    void enqueue(F&& f) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_tasks.emplace_back(std::forward<F>(f));
        }
        m_cv.notify_one();
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this]() { return m_tasks.empty() && m_active == 0; });
    }

    size_t num_threads() const { return m_workers.size(); }
};

// Global thread pool instance
inline thread_pool& global_pool() {
    static thread_pool pool(thread_config::max_threads());
    return pool;
}

} // namespace parallel

// Bring into main namespace for convenience
using parallel::execution_policy;
using parallel::parallel_for;
using parallel::parallel_for_dynamic;
using parallel::parallel_reduce;
using parallel::parallel_transform;
using parallel::parallel_assign;
using parallel::parallel_sort;
using parallel::thread_pool;
using parallel::global_pool;
using parallel::thread_config;

XTU_NAMESPACE_END

#endif // XTU_PARALLEL_XPARALLEL_HPP