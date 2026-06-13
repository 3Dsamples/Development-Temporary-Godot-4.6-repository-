//File 0114 : core/parallel/flow_graph.h
//Lightweight flow‑graph executor: directed acyclic task graph with join counters; tasks are submitted to the global work‑stealing scheduler for dependency‑driven parallelism.
#ifndef CORE_PARALLEL_FLOW_GRAPH_H
#define CORE_PARALLEL_FLOW_GRAPH_H

#include "task_scheduler.h"
#include <vector>
#include <functional>
#include <atomic>
#include <cstdint>
#include <memory>

namespace SimulationMath {
namespace parallel {

// -----------------------------------------------------------------------------
// 1. FlowGraph – holds a set of nodes and edges, executes via global scheduler.
// -----------------------------------------------------------------------------
class FlowGraph {
public:
    using NodeFunc = std::function<void()>;

    // Add a node with a function to execute. Returns the node index.
    uint32_t add_node(NodeFunc func) {
        nodes_.push_back({
            std::move(func),
            {},  // successors
            0    // join_counter (initialised at run time)
        });
        return static_cast<uint32_t>(nodes_.size() - 1);
    }

    // Create a directed edge: node `from` must complete before node `to` starts.
    void make_edge(uint32_t from, uint32_t to) {
        if (from >= nodes_.size() || to >= nodes_.size()) return;
        nodes_[from].successors.push_back(to);
        // We'll count the number of predecessors at run time.
    }

    // Execute the graph. Returns when all nodes have finished.
    void run() {
        size_t n = nodes_.size();
        if (n == 0) return;

        // Count predecessors for each node
        for (auto& node : nodes_) {
            node.pred_count.store(0, std::memory_order_relaxed);
        }
        for (const auto& node : nodes_) {
            for (uint32_t succ : node.successors) {
                nodes_[succ].pred_count.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Total number of nodes that must execute (used to wait for completion)
        std::atomic<size_t> remaining(n);
        TaskScheduler& scheduler = TaskScheduler::instance();

        // Launch nodes with zero predecessors
        for (uint32_t i = 0; i < static_cast<uint32_t>(n); ++i) {
            if (nodes_[i].pred_count.load(std::memory_order_relaxed) == 0) {
                scheduler.submit([this, i, &remaining]() {
                    execute_node(i, remaining);
                });
            }
        }

        // Wait until all nodes are processed
        while (remaining.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
    }

private:
    struct Node {
        NodeFunc func;
        std::vector<uint32_t> successors;
        std::atomic<int> pred_count;  // number of predecessors not yet executed
    };

    std::vector<Node> nodes_;

    // Execute a node and signal successors
    void execute_node(uint32_t idx, std::atomic<size_t>& remaining) {
        // Run the node's function
        if (nodes_[idx].func) nodes_[idx].func();

        // Notify successors
        for (uint32_t succ : nodes_[idx].successors) {
            int prev = nodes_[succ].pred_count.fetch_sub(1, std::memory_order_acq_rel);
            if (prev == 1) {
                // All predecessors completed, enqueue this successor
                TaskScheduler::instance().submit([this, succ, &remaining]() {
                    execute_node(succ, remaining);
                });
            }
        }

        // Decrement global remaining counter
        remaining.fetch_sub(1, std::memory_order_acq_rel);
    }
};

} // namespace parallel
} // namespace SimulationMath

#endif // CORE_PARALLEL_FLOW_GRAPH_H