// system name : onetbb-warp
// File 0045 : core/math/graph_algorithms.h
// Description : Graph data structures, Dijkstra, A*, Kruskal, MST, topological sort, network flows.

#ifndef __TBB_WARP_CORE_MATH_GRAPH_ALGORITHMS_H
#define __TBB_WARP_CORE_MATH_GRAPH_ALGORITHMS_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include <cmath>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>
#include <limits>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tbb {
namespace core {
namespace math {
namespace graph {

// ============================================================
// Graph data structures (adjacency list)
// ============================================================

template<typename T>
struct weighted_edge {
    std::size_t from;
    std::size_t to;
    T weight;
};

template<typename T>
class graph {
public:
    using edge_type = weighted_edge<T>;

    graph(std::size_t num_vertices) : m_adj(num_vertices) {}

    void add_directed_edge(std::size_t from, std::size_t to, T weight) noexcept {
        m_edges.push_back({from, to, weight});
        m_adj[from].push_back(m_edges.size() - 1);
    }

    void add_undirected_edge(std::size_t u, std::size_t v, T weight) noexcept {
        add_directed_edge(u, v, weight);
        add_directed_edge(v, u, weight);
    }

    std::size_t num_vertices() const noexcept { return m_adj.size(); }
    std::size_t num_edges() const noexcept { return m_edges.size(); }

    const std::vector<std::size_t>& adjacent_edges(std::size_t v) const noexcept { return m_adj[v]; }
    const edge_type& edge(std::size_t e) const noexcept { return m_edges[e]; }

    // Access for sparse graph modifications
    void clear() noexcept { m_adj.clear(); m_edges.clear(); }

private:
    std::vector<std::vector<std::size_t>> m_adj;  // indices into m_edges
    std::vector<edge_type> m_edges;
};

// ============================================================
// Dijkstra's shortest path (non‑negative weights)
// ============================================================

template<typename T>
std::pair<std::vector<T>, std::vector<std::size_t>> dijkstra(
    const graph<T>& g, std::size_t source) noexcept
{
    std::size_t n = g.num_vertices();
    std::vector<T> dist(n, std::numeric_limits<T>::max());
    std::vector<std::size_t> prev(n, static_cast<std::size_t>(-1));
    dist[source] = T(0);
    using state = std::pair<T, std::size_t>;
    std::priority_queue<state, std::vector<state>, std::greater<state>> pq;
    pq.push({T(0), source});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto ei : g.adjacent_edges(u)) {
            const auto& e = g.edge(ei);
            std::size_t v = e.to;
            T nd = d + e.weight;
            if (nd < dist[v]) {
                dist[v] = nd;
                prev[v] = u;
                pq.push({nd, v});
            }
        }
    }
    return {dist, prev};
}

// ============================================================
// Reconstruct path from predecessor array
// ============================================================

template<typename T>
std::vector<std::size_t> reconstruct_path(const std::vector<std::size_t>& prev, std::size_t target) {
    std::vector<std::size_t> path;
    for (std::size_t at = target; at != static_cast<std::size_t>(-1); at = prev[at])
        path.push_back(at);
    std::reverse(path.begin(), path.end());
    return path;
}

// ============================================================
// A* search with heuristic function
// ============================================================

template<typename T>
std::pair<std::vector<T>, std::vector<std::size_t>> a_star(
    const graph<T>& g, std::size_t source, std::size_t target,
    const std::function<T(std::size_t)>& heuristic) noexcept
{
    std::size_t n = g.num_vertices();
    std::vector<T> g_score(n, std::numeric_limits<T>::max());
    std::vector<std::size_t> prev(n, static_cast<std::size_t>(-1));
    g_score[source] = T(0);
    using state = std::pair<T, std::size_t>;
    std::priority_queue<state, std::vector<state>, std::greater<state>> open;
    open.push({heuristic(source), source});
    std::vector<bool> closed(n, false);
    while (!open.empty()) {
        auto [f, u] = open.top(); open.pop();
        if (u == target) break;
        if (closed[u]) continue;
        closed[u] = true;
        for (auto ei : g.adjacent_edges(u)) {
            const auto& e = g.edge(ei);
            std::size_t v = e.to;
            T tentative_g = g_score[u] + e.weight;
            if (tentative_g < g_score[v]) {
                g_score[v] = tentative_g;
                prev[v] = u;
                open.push({tentative_g + heuristic(v), v});
            }
        }
    }
    return {g_score, prev};
}

// ============================================================
// Bellman‑Ford (allows negative weights, detects negative cycles)
// ============================================================

template<typename T>
std::pair<std::vector<T>, std::vector<std::size_t>> bellman_ford(
    const graph<T>& g, std::size_t source) noexcept
{
    std::size_t n = g.num_vertices();
    std::vector<T> dist(n, std::numeric_limits<T>::max());
    std::vector<std::size_t> prev(n, static_cast<std::size_t>(-1));
    dist[source] = T(0);
    for (std::size_t i = 0; i < n - 1; ++i) {
        bool relaxed = false;
        for (std::size_t e = 0; e < g.num_edges(); ++e) {
            const auto& edge = g.edge(e);
            if (dist[edge.from] < std::numeric_limits<T>::max() &&
                dist[edge.from] + edge.weight < dist[edge.to]) {
                dist[edge.to] = dist[edge.from] + edge.weight;
                prev[edge.to] = edge.from;
                relaxed = true;
            }
        }
        if (!relaxed) break;
    }
    // Check negative cycles
    for (std::size_t e = 0; e < g.num_edges(); ++e) {
        const auto& edge = g.edge(e);
        if (dist[edge.from] < std::numeric_limits<T>::max() &&
            dist[edge.from] + edge.weight < dist[edge.to]) {
            // Negative cycle detected; return empty with flag? We'll set prev[target] to -2 for cycle.
            // We'll handle by marking.
            dist[edge.to] = -std::numeric_limits<T>::max(); // indicate cycle
        }
    }
    return {dist, prev};
}

// ============================================================
// Floyd‑Warshall all‑pairs shortest path (dense)
// ============================================================

template<typename T>
std::vector<std::vector<T>> floyd_warshall(const graph<T>& g) noexcept {
    std::size_t n = g.num_vertices();
    std::vector<std::vector<T>> dist(n, std::vector<T>(n, std::numeric_limits<T>::max()));
    for (std::size_t i = 0; i < n; ++i) dist[i][i] = T(0);
    for (std::size_t e = 0; e < g.num_edges(); ++e) {
        const auto& edge = g.edge(e);
        dist[edge.from][edge.to] = std::min(dist[edge.from][edge.to], edge.weight);
    }
    for (std::size_t k = 0; k < n; ++k)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (dist[i][k] < std::numeric_limits<T>::max() &&
                    dist[k][j] < std::numeric_limits<T>::max())
                    dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
    return dist;
}

// ============================================================
// Minimum Spanning Tree – Kruskal's algorithm
// ============================================================

template<typename T>
std::vector<weighted_edge<T>> kruskal_mst(const graph<T>& g) noexcept {
    std::size_t n = g.num_vertices();
    std::vector<weighted_edge<T>> all_edges(g.num_edges());
    for (std::size_t i = 0; i < g.num_edges(); ++i) all_edges[i] = g.edge(i);
    std::sort(all_edges.begin(), all_edges.end(),
              [](const weighted_edge<T>& a, const weighted_edge<T>& b) { return a.weight < b.weight; });
    std::vector<std::size_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<std::size_t> rank(n, 0);
    auto find = [&](std::size_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](std::size_t x, std::size_t y) {
        x = find(x); y = find(y);
        if (x == y) return false;
        if (rank[x] < rank[y]) std::swap(x, y);
        parent[y] = x;
        if (rank[x] == rank[y]) ++rank[x];
        return true;
    };
    std::vector<weighted_edge<T>> mst;
    for (const auto& e : all_edges) {
        if (unite(e.from, e.to)) mst.push_back(e);
        if (mst.size() == n - 1) break;
    }
    return mst;
}

// ============================================================
// Prim's algorithm (dense) for MST
// ============================================================

template<typename T>
std::vector<weighted_edge<T>> prim_mst(const graph<T>& g, std::size_t start = 0) noexcept {
    std::size_t n = g.num_vertices();
    std::vector<T> min_edge(n, std::numeric_limits<T>::max());
    std::vector<std::size_t> prev(n, static_cast<std::size_t>(-1));
    std::vector<bool> in_mst(n, false);
    min_edge[start] = T(0);
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t v = static_cast<std::size_t>(-1);
        T best = std::numeric_limits<T>::max();
        for (std::size_t j = 0; j < n; ++j) {
            if (!in_mst[j] && min_edge[j] < best) { best = min_edge[j]; v = j; }
        }
        if (v == static_cast<std::size_t>(-1)) break;
        in_mst[v] = true;
        for (auto ei : g.adjacent_edges(v)) {
            const auto& e = g.edge(ei);
            if (!in_mst[e.to] && e.weight < min_edge[e.to]) {
                min_edge[e.to] = e.weight;
                prev[e.to] = v;
            }
        }
    }
    std::vector<weighted_edge<T>> mst;
    for (std::size_t i = 0; i < n; ++i) {
        if (prev[i] != static_cast<std::size_t>(-1)) {
            mst.push_back({prev[i], i, min_edge[i]});
        }
    }
    return mst;
}

// ============================================================
// Connected components (undirected)
// ============================================================

template<typename T>
std::vector<int> connected_components(const graph<T>& g) noexcept {
    std::size_t n = g.num_vertices();
    std::vector<int> comp(n, -1);
    int comp_id = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (comp[i] != -1) continue;
        std::stack<std::size_t> st;
        st.push(i);
        comp[i] = comp_id;
        while (!st.empty()) {
            std::size_t u = st.top(); st.pop();
            for (auto ei : g.adjacent_edges(u)) {
                const auto& e = g.edge(ei);
                if (comp[e.to] == -1) { comp[e.to] = comp_id; st.push(e.to); }
            }
        }
        ++comp_id;
    }
    return comp;
}

// ============================================================
// Topological sort (Kahn's algorithm, directed acyclic)
// ============================================================

template<typename T>
std::vector<std::size_t> topological_sort(const graph<T>& g) {
    std::size_t n = g.num_vertices();
    std::vector<int> in_degree(n, 0);
    for (std::size_t e = 0; e < g.num_edges(); ++e) {
        ++in_degree[g.edge(e).to];
    }
    std::queue<std::size_t> q;
    for (std::size_t i = 0; i < n; ++i)
        if (in_degree[i] == 0) q.push(i);
    std::vector<std::size_t> order;
    while (!q.empty()) {
        std::size_t u = q.front(); q.pop();
        order.push_back(u);
        for (auto ei : g.adjacent_edges(u)) {
            std::size_t v = g.edge(ei).to;
            if (--in_degree[v] == 0) q.push(v);
        }
    }
    if (order.size() != n) throw std::runtime_error("Graph has a cycle");
    return order;
}

// ============================================================
// Strongly Connected Components (Kosaraju)
// ============================================================

template<typename T>
std::vector<int> strongly_connected_components(const graph<T>& g) noexcept {
    std::size_t n = g.num_vertices();
    std::vector<int> comp(n, -1);
    std::vector<bool> visited(n, false);
    std::vector<std::size_t> order;
    std::function<void(std::size_t)> dfs1 = [&](std::size_t u) {
        visited[u] = true;
        for (auto ei : g.adjacent_edges(u)) {
            std::size_t v = g.edge(ei).to;
            if (!visited[v]) dfs1(v);
        }
        order.push_back(u);
    };
    for (std::size_t i = 0; i < n; ++i) if (!visited[i]) dfs1(i);
    // Build reverse graph
    graph<T> rev(n);
    for (std::size_t e = 0; e < g.num_edges(); ++e) {
        const auto& edge = g.edge(e);
        rev.add_directed_edge(edge.to, edge.from, edge.weight);
    }
    int comp_id = 0;
    std::function<void(std::size_t)> dfs2 = [&](std::size_t u) {
        comp[u] = comp_id;
        for (auto ei : rev.adjacent_edges(u)) {
            std::size_t v = rev.edge(ei).to;
            if (comp[v] == -1) dfs2(v);
        }
    };
    for (std::size_t i = order.size(); i-- > 0;) {
        std::size_t u = order[i];
        if (comp[u] == -1) { dfs2(u); ++comp_id; }
    }
    return comp;
}

// ============================================================
// Edmonds‑Karp max flow (Ford‑Fulkerson with BFS)
// ============================================================

template<typename T>
T max_flow_edmonds_karp(const graph<T>& capacity, std::size_t source, std::size_t sink) {
    std::size_t n = capacity.num_vertices();
    // Build residual graph as a mutable copy of edges (bidirectional)
    struct residual_edge {
        std::size_t to;
        T capacity;
        std::size_t rev_idx;
    };
    std::vector<std::vector<residual_edge>> adj(n);
    for (std::size_t e = 0; e < capacity.num_edges(); ++e) {
        const auto& edge = capacity.edge(e);
        adj[edge.from].push_back({edge.to, edge.weight, adj[edge.to].size()});
        adj[edge.to].push_back({edge.from, T(0), adj[edge.from].size() - 1});
    }
    T flow = T(0);
    while (true) {
        std::vector<std::size_t> parent(n, static_cast<std::size_t>(-1));
        std::vector<std::size_t> parent_edge(n);
        std::queue<std::size_t> q;
        q.push(source);
        parent[source] = source;
        while (!q.empty() && parent[sink] == static_cast<std::size_t>(-1)) {
            std::size_t u = q.front(); q.pop();
            for (std::size_t i = 0; i < adj[u].size(); ++i) {
                const auto& re = adj[u][i];
                if (re.capacity > T(0) && parent[re.to] == static_cast<std::size_t>(-1)) {
                    parent[re.to] = u;
                    parent_edge[re.to] = i;
                    q.push(re.to);
                }
            }
        }
        if (parent[sink] == static_cast<std::size_t>(-1)) break;
        T add_flow = std::numeric_limits<T>::max();
        for (std::size_t v = sink; v != source; v = parent[v]) {
            std::size_t u = parent[v];
            const auto& re = adj[u][parent_edge[v]];
            add_flow = std::min(add_flow, re.capacity);
        }
        for (std::size_t v = sink; v != source; v = parent[v]) {
            std::size_t u = parent[v];
            auto& re = adj[u][parent_edge[v]];
            re.capacity -= add_flow;
            adj[v][re.rev_idx].capacity += add_flow;
        }
        flow += add_flow;
    }
    return flow;
}

// ============================================================
// Min‑cost max‑flow using successive shortest augmenting paths
// ============================================================

template<typename T>
struct cost_capacity_edge {
    std::size_t from, to;
    T capacity;
    T cost;
};

template<typename T>
std::pair<T, T> min_cost_max_flow(const std::vector<cost_capacity_edge<T>>& edges, std::size_t source, std::size_t sink, T max_flow_limit = std::numeric_limits<T>::max()) {
    std::size_t n = 0;
    for (const auto& e : edges) n = std::max(n, std::max(e.from, e.to) + 1);
    struct edge { std::size_t to; T cap; T cost; std::size_t rev; };
    std::vector<std::vector<edge>> g(n);
    for (const auto& e : edges) {
        g[e.from].push_back({e.to, e.capacity, e.cost, g[e.to].size()});
        g[e.to].push_back({e.from, T(0), -e.cost, g[e.from].size() - 1});
    }
    T flow = T(0), cost = T(0);
    std::vector<T> dist(n), potential(n, T(0));
    std::vector<std::size_t> prevv(n), preve(n);
    while (flow < max_flow_limit) {
        std::fill(dist.begin(), dist.end(), std::numeric_limits<T>::max());
        dist[source] = T(0);
        using state = std::pair<T, std::size_t>;
        std::priority_queue<state, std::vector<state>, std::greater<state>> pq;
        pq.push({T(0), source});
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;
            for (std::size_t i = 0; i < g[u].size(); ++i) {
                const auto& e = g[u][i];
                if (e.cap > T(0) && dist[e.to] > dist[u] + e.cost + potential[u] - potential[e.to]) {
                    dist[e.to] = dist[u] + e.cost + potential[u] - potential[e.to];
                    prevv[e.to] = u; preve[e.to] = i;
                    pq.push({dist[e.to], e.to});
                }
            }
        }
        if (dist[sink] == std::numeric_limits<T>::max()) break;
        for (std::size_t v = 0; v < n; ++v) potential[v] += dist[v];
        T add = max_flow_limit - flow;
        for (std::size_t v = sink; v != source; v = prevv[v]) {
            add = std::min(add, g[prevv[v]][preve[v]].cap);
        }
        flow += add;
        cost += add * potential[sink];
        for (std::size_t v = sink; v != source; v = prevv[v]) {
            auto& e = g[prevv[v]][preve[v]];
            e.cap -= add;
            g[v][e.rev].cap += add;
        }
    }
    return {flow, cost};
}

} // namespace graph
} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_GRAPH_ALGORITHMS_H