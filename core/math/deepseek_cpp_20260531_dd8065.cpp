//File 0057 : core/math/graph_algorithms.h
//Graph algorithms for 3D grid and general graphs: Dijkstra, A*, BFS, topological sort, connectivity, minimum spanning tree, with SIMD‑accelerated distance heuristics.
#ifndef CORE_MATH_GRAPH_ALGORITHMS_H
#define CORE_MATH_GRAPH_ALGORITHMS_H

#include "vector_math.h"
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <cstdint>
#include <functional>
#include <algorithm>

namespace SimulationMath {
namespace graph {

using SimdVec = DirectX::XMVECTOR;

// -----------------------------------------------------------------------------
// 1. A generic directed graph using adjacency lists
// -----------------------------------------------------------------------------
template <typename NodeData, typename EdgeWeight = float>
class Graph {
public:
    struct Edge {
        uint32_t to;
        EdgeWeight weight;
    };

    Graph() = default;

    uint32_t add_node(const NodeData& data = NodeData{}) {
        nodes_.push_back(data);
        adj_.emplace_back();
        return static_cast<uint32_t>(nodes_.size() - 1);
    }

    void add_edge(uint32_t from, uint32_t to, EdgeWeight weight = EdgeWeight{1}) {
        adj_[from].push_back({to, weight});
    }

    size_t node_count() const noexcept { return nodes_.size(); }
    const NodeData& node_data(uint32_t id) const noexcept { return nodes_[id]; }
    const std::vector<Edge>& neighbors(uint32_t id) const noexcept { return adj_[id]; }

private:
    std::vector<NodeData> nodes_;
    std::vector<std::vector<Edge>> adj_;
};

// -----------------------------------------------------------------------------
// 2. BFS (Breadth‑First Search) – returns distances from source
// -----------------------------------------------------------------------------
template <typename GraphType>
std::vector<uint32_t> bfs(const GraphType& graph, uint32_t source) {
    std::vector<uint32_t> dist(graph.node_count(), 0xFFFFFFFFu);
    std::queue<uint32_t> q;
    dist[source] = 0;
    q.push(source);
    while (!q.empty()) {
        uint32_t u = q.front(); q.pop();
        for (const auto& edge : graph.neighbors(u)) {
            if (dist[edge.to] == 0xFFFFFFFFu) {
                dist[edge.to] = dist[u] + 1;
                q.push(edge.to);
            }
        }
    }
    return dist;
}

// -----------------------------------------------------------------------------
// 3. Dijkstra (shortest paths with non‑negative weights)
// -----------------------------------------------------------------------------
template <typename GraphType, typename EdgeWeight = float>
std::vector<EdgeWeight> dijkstra(const GraphType& graph, uint32_t source,
                                 std::vector<uint32_t>* prev = nullptr) {
    size_t n = graph.node_count();
    std::vector<EdgeWeight> dist(n, std::numeric_limits<EdgeWeight>::max());
    if (prev) prev->assign(n, 0xFFFFFFFFu);

    using Pair = std::pair<EdgeWeight, uint32_t>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pq;

    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue; // outdated entry
        for (const auto& edge : graph.neighbors(u)) {
            EdgeWeight new_dist = d + edge.weight;
            if (new_dist < dist[edge.to]) {
                dist[edge.to] = new_dist;
                if (prev) (*prev)[edge.to] = u;
                pq.push({new_dist, edge.to});
            }
        }
    }
    return dist;
}

// -----------------------------------------------------------------------------
// 4. A* (heuristic search) for graphs with 3D positions as heuristic
//    Heuristic: Euclidean distance between node positions.
// -----------------------------------------------------------------------------
template <typename GraphType, typename PositionFunc>
std::vector<float> a_star(const GraphType& graph, uint32_t source, uint32_t target,
                          PositionFunc pos_func,
                          std::vector<uint32_t>* prev = nullptr) {
    size_t n = graph.node_count();
    std::vector<float> g_score(n, std::numeric_limits<float>::max());
    std::vector<float> f_score(n, std::numeric_limits<float>::max());
    if (prev) prev->assign(n, 0xFFFFFFFFu);

    using Pair = std::pair<float, uint32_t>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> open;

    g_score[source] = 0;
    f_score[source] = vector_math::length3_scalar(
        DirectX::XMVectorSubtract(pos_func(target), pos_func(source)));
    open.push({f_score[source], source});

    while (!open.empty()) {
        uint32_t u = open.top().second;
        open.pop();
        if (u == target) break;

        for (const auto& edge : graph.neighbors(u)) {
            float tentative_g = g_score[u] + edge.weight;
            if (tentative_g < g_score[edge.to]) {
                g_score[edge.to] = tentative_g;
                f_score[edge.to] = tentative_g +
                    vector_math::length3_scalar(DirectX::XMVectorSubtract(pos_func(target), pos_func(edge.to)));
                if (prev) (*prev)[edge.to] = u;
                open.push({f_score[edge.to], edge.to});
            }
        }
    }
    return g_score;
}

// -----------------------------------------------------------------------------
// 5. Topological sort (for DAGs) – returns list of vertices in order
// -----------------------------------------------------------------------------
template <typename GraphType>
std::vector<uint32_t> topological_sort(const GraphType& graph) {
    size_t n = graph.node_count();
    std::vector<uint32_t> indegree(n, 0);
    for (uint32_t u = 0; u < n; ++u)
        for (const auto& edge : graph.neighbors(u))
            ++indegree[edge.to];

    std::queue<uint32_t> q;
    for (uint32_t u = 0; u < n; ++u)
        if (indegree[u] == 0) q.push(u);

    std::vector<uint32_t> order;
    while (!q.empty()) {
        uint32_t u = q.front(); q.pop();
        order.push_back(u);
        for (const auto& edge : graph.neighbors(u)) {
            if (--indegree[edge.to] == 0)
                q.push(edge.to);
        }
    }
    if (order.size() != n) return {}; // cycle detected
    return order;
}

// -----------------------------------------------------------------------------
// 6. Kruskal's Minimum Spanning Tree (undirected, returns edge list)
// -----------------------------------------------------------------------------
struct UndirectedEdge {
    uint32_t u, v;
    float weight;
};

inline std::vector<UndirectedEdge> kruskal_mst(const std::vector<UndirectedEdge>& edges, size_t node_count) {
    // Union‑Find
    std::vector<uint32_t> parent(node_count);
    for (uint32_t i = 0; i < node_count; ++i) parent[i] = i;
    std::function<uint32_t(uint32_t)> find = [&](uint32_t x) -> uint32_t {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    };
    auto unite = [&](uint32_t a, uint32_t b) {
        a = find(a); b = find(b);
        if (a != b) { parent[b] = a; return true; }
        return false;
    };

    // Sort edges by weight
    std::vector<UndirectedEdge> sorted = edges;
    std::sort(sorted.begin(), sorted.end(), [](const UndirectedEdge& a, const UndirectedEdge& b) {
        return a.weight < b.weight;
    });

    std::vector<UndirectedEdge> mst;
    for (const auto& e : sorted) {
        if (unite(e.u, e.v)) {
            mst.push_back(e);
            if (mst.size() == node_count - 1) break;
        }
    }
    return mst;
}

// -----------------------------------------------------------------------------
// 7. Grid‑based pathfinding on 3D occupancy grid (A* using direct neighbor offsets)
//    grid: 3D array of bool (true = walkable), dimensions (Nx,Ny,Nz), cell size.
// -----------------------------------------------------------------------------
struct GridPathResult {
    std::vector<SimdVec> positions;
    float total_cost;
};

inline GridPathResult a_star_grid_3d(
    const std::vector<std::vector<std::vector<bool>>>& walkable,
    SimdVec start, SimdVec goal, float cell_size,
    int max_expand = 100000)
{
    // 3D grid dimensions from the nested vector (assuming all rows same size)
    size_t Nz = walkable.size();
    size_t Ny = walkable[0].size();
    size_t Nx = walkable[0][0].size();

    auto cell_index = [&](SimdVec p) -> std::tuple<int,int,int> {
        int ix = static_cast<int>(std::floor(vector_math::get_x(p) / cell_size));
        int iy = static_cast<int>(std::floor(vector_math::get_y(p) / cell_size));
        int iz = static_cast<int>(std::floor(vector_math::get_z(p) / cell_size));
        return {ix, iy, iz};
    };

    auto idx = [&](int ix, int iy, int iz) -> uint64_t {
        if (ix<0||iy<0||iz<0||ix>=(int)Nx||iy>=(int)Ny||iz>=(int)Nz) return UINT64_MAX;
        return static_cast<uint64_t>(ix) + Nx*(iy + Ny*iz);
    };

    using Node = std::pair<float, uint64_t>;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;
    std::unordered_map<uint64_t, float> g_score;
    std::unordered_map<uint64_t, uint64_t> parent;

    SimdVec start_pos = start;
    uint64_t start_id = idx(std::get<0>(cell_index(start)), std::get<1>(cell_index(start)), std::get<2>(cell_index(start)));
    uint64_t goal_id  = idx(std::get<0>(cell_index(goal)), std::get<1>(cell_index(goal)), std::get<2>(cell_index(goal)));

    if (start_id == UINT64_MAX || goal_id == UINT64_MAX) return {{}, 0.0f};

    g_score[start_id] = 0;
    open.push({vector_math::length3_scalar(DirectX::XMVectorSubtract(start, goal)), start_id});

    const int neighbors[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};

    while (!open.empty()) {
        auto [f, id] = open.top(); open.pop();
        if (id == goal_id) break;

        // decode id to ix,iy,iz
        int ix = id % Nx;
        int iy = (id / Nx) % Ny;
        int iz = id / (Nx*Ny);
        float current_g = g_score[id];

        for (int d = 0; d < 6; ++d) {
            int nx = ix + neighbors[d][0];
            int ny = iy + neighbors[d][1];
            int nz = iz + neighbors[d][2];
            if (nx<0||ny<0||nz<0||nx>=(int)Nx||ny>=(int)Ny||nz>=(int)Nz) continue;
            if (!walkable[nz][ny][nx]) continue;
            uint64_t nid = idx(nx, ny, nz);
            float new_g = current_g + cell_size; // unit distance between cells
            auto it = g_score.find(nid);
            if (it == g_score.end() || new_g < it->second) {
                g_score[nid] = new_g;
                float h = vector_math::length3_scalar(
                    DirectX::XMVectorSubtract(
                        DirectX::XMVectorSet((nx+0.5f)*cell_size, (ny+0.5f)*cell_size, (nz+0.5f)*cell_size,0),
                        goal));
                open.push({new_g + h, nid});
                parent[nid] = id;
            }
        }
    }

    // Reconstruct path if goal reached
    std::vector<SimdVec> path;
    if (parent.find(goal_id) == parent.end() && start_id != goal_id) return {{}, 0.0f};
    uint64_t cur = goal_id;
    while (cur != start_id) {
        int ix = cur % Nx, iy = (cur / Nx) % Ny, iz = cur / (Nx*Ny);
        path.push_back(DirectX::XMVectorSet((ix+0.5f)*cell_size, (iy+0.5f)*cell_size, (iz+0.5f)*cell_size, 0));
        cur = parent[cur];
    }
    path.push_back(start);
    std::reverse(path.begin(), path.end());
    return {path, g_score[goal_id]};
}

// -----------------------------------------------------------------------------
// 8. Utility: reconstruct path from predecessor array
// -----------------------------------------------------------------------------
inline std::vector<uint32_t> reconstruct_path(const std::vector<uint32_t>& prev, uint32_t target) {
    std::vector<uint32_t> path;
    for (uint32_t at = target; at != 0xFFFFFFFFu; at = prev[at])
        path.push_back(at);
    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace graph
} // namespace SimulationMath

#endif // CORE_MATH_GRAPH_ALGORITHMS_H