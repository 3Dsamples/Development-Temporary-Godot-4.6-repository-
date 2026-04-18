// godot/xpathfinding.hpp

#ifndef XTENSOR_XPATHFINDING_HPP
#define XTENSOR_XPATHFINDING_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xintersection.hpp"
#include "../math/xgraph.hpp"
#include "../math/xoptimize.hpp"
#include "xvariant.hpp"
#include "xclassdb.hpp"
#include "xnode.hpp"
#include "xresource.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <queue>
#include <mutex>
#include <unordered_set>
#include <stack>
#include <chrono>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/node.hpp>
    #include <godot_cpp/classes/node2d.hpp>
    #include <godot_cpp/classes/node3d.hpp>
    #include <godot_cpp/classes/navigation_server2d.hpp>
    #include <godot_cpp/classes/navigation_server3d.hpp>
    #include <godot_cpp/classes/navigation_agent2d.hpp>
    #include <godot_cpp/classes/navigation_agent3d.hpp>
    #include <godot_cpp/classes/navigation_obstacle2d.hpp>
    #include <godot_cpp/classes/navigation_obstacle3d.hpp>
    #include <godot_cpp/classes/navigation_mesh.hpp>
    #include <godot_cpp/classes/navigation_polygon.hpp>
    #include <godot_cpp/classes/resource.hpp>
    #include <godot_cpp/classes/ref_counted.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/vector2.hpp>
    #include <godot_cpp/variant/vector3.hpp>
    #include <godot_cpp/variant/packed_vector2_array.hpp>
    #include <godot_cpp/variant/packed_vector3_array.hpp>
    #include <godot_cpp/variant/packed_int32_array.hpp>
    #include <godot_cpp/variant/transform3d.hpp>
    #include <godot_cpp/variant/aabb.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // Graph representation for pathfinding
            // --------------------------------------------------------------------
            class PathfindingGraph
            {
            public:
                // Adjacency list: nodes x max_edges (with -1 for no edge)
                xarray_container<int64_t> edges;         // N x max_degree
                xarray_container<float> edge_weights;    // N x max_degree
                xarray_container<float> node_positions;  // N x 2 (2D) or N x 3 (3D)
                xarray_container<float> node_costs;      // N (additional per-node cost)
                bool is_directed = false;
                int64_t max_degree = 8;

                PathfindingGraph() = default;

                // Build from grid (2D)
                static PathfindingGraph from_grid(const xarray_container<float>& grid,
                                                  float walkable_threshold = 0.5f,
                                                  bool allow_diagonal = true)
                {
                    PathfindingGraph graph;
                    if (grid.dimension() != 2) return graph;

                    size_t h = grid.shape()[0];
                    size_t w = grid.shape()[1];
                    size_t n = h * w;
                    
                    graph.max_degree = allow_diagonal ? 8 : 4;
                    graph.edges = xarray_container<int64_t>({n, static_cast<size_t>(graph.max_degree)}, -1);
                    graph.edge_weights = xarray_container<float>({n, static_cast<size_t>(graph.max_degree)}, 0.0f);
                    graph.node_positions = xarray_container<float>({n, 2});
                    graph.node_costs = xarray_container<float>({n}, 0.0f);

                    // Fill positions and costs
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            size_t idx = y * w + x;
                            graph.node_positions(idx, 0) = static_cast<float>(x);
                            graph.node_positions(idx, 1) = static_cast<float>(y);
                            graph.node_costs(idx) = grid(y, x) < walkable_threshold ? 1.0f : std::numeric_limits<float>::infinity();
                        }
                    }

                    // Build edges (4 or 8 connectivity)
                    const int dx_4[4] = {1, 0, -1, 0};
                    const int dy_4[4] = {0, 1, 0, -1};
                    const int dx_8[8] = {1, 1, 0, -1, -1, -1, 0, 1};
                    const int dy_8[8] = {0, 1, 1, 1, 0, -1, -1, -1};
                    const int* dx = allow_diagonal ? dx_8 : dx_4;
                    const int* dy = allow_diagonal ? dy_8 : dy_4;
                    int degree = allow_diagonal ? 8 : 4;

                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            size_t idx = y * w + x;
                            if (std::isinf(graph.node_costs(idx))) continue;

                            int edge_idx = 0;
                            for (int d = 0; d < degree; ++d)
                            {
                                int nx = static_cast<int>(x) + dx[d];
                                int ny = static_cast<int>(y) + dy[d];
                                if (nx < 0 || nx >= static_cast<int>(w) || ny < 0 || ny >= static_cast<int>(h))
                                    continue;
                                size_t nidx = ny * w + nx;
                                if (std::isinf(graph.node_costs(nidx))) continue;

                                graph.edges(idx, edge_idx) = static_cast<int64_t>(nidx);
                                float weight = (allow_diagonal && (dx[d] != 0 && dy[d] != 0)) ? 1.414f : 1.0f;
                                graph.edge_weights(idx, edge_idx) = weight;
                                ++edge_idx;
                            }
                        }
                    }
                    return graph;
                }

                // Build from navigation mesh (simplified)
                static PathfindingGraph from_navmesh(const godot::Ref<godot::NavigationMesh>& navmesh)
                {
                    PathfindingGraph graph;
                    // Convert navigation mesh polygons to graph
                    // Placeholder implementation
                    return graph;
                }
            };

            // --------------------------------------------------------------------
            // Batch Pathfinding (A*)
            // --------------------------------------------------------------------
            class BatchAStar
            {
            public:
                struct PathResult
                {
                    std::vector<int64_t> path;       // node indices
                    float cost = 0.0f;
                    bool reached = false;
                    size_t nodes_explored = 0;
                };

                // Single query
                static PathResult find_path(const PathfindingGraph& graph,
                                            int64_t start,
                                            int64_t goal,
                                            const std::string& heuristic = "euclidean")
                {
                    PathResult result;
                    size_t n = graph.node_positions.shape()[0];
                    if (start < 0 || start >= static_cast<int64_t>(n) ||
                        goal < 0 || goal >= static_cast<int64_t>(n))
                        return result;

                    // Heuristic function
                    auto h = [&](int64_t node) -> float {
                        if (heuristic == "manhattan")
                        {
                            return std::abs(graph.node_positions(node, 0) - graph.node_positions(goal, 0)) +
                                   std::abs(graph.node_positions(node, 1) - graph.node_positions(goal, 1));
                        }
                        else // euclidean
                        {
                            float dx = graph.node_positions(node, 0) - graph.node_positions(goal, 0);
                            float dy = graph.node_positions(node, 1) - graph.node_positions(goal, 1);
                            return std::sqrt(dx*dx + dy*dy);
                        }
                    };

                    // Priority queue: (f, node)
                    using PQElement = std::pair<float, int64_t>;
                    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> open_set;
                    std::unordered_map<int64_t, float> g_score;
                    std::unordered_map<int64_t, int64_t> came_from;
                    std::unordered_set<int64_t> closed_set;

                    g_score[start] = 0.0f;
                    open_set.push({h(start), start});

                    while (!open_set.empty())
                    {
                        int64_t current = open_set.top().second;
                        open_set.pop();
                        result.nodes_explored++;

                        if (current == goal)
                        {
                            result.reached = true;
                            result.cost = g_score[current];
                            // Reconstruct path
                            int64_t node = goal;
                            while (node != start)
                            {
                                result.path.push_back(node);
                                node = came_from[node];
                            }
                            result.path.push_back(start);
                            std::reverse(result.path.begin(), result.path.end());
                            break;
                        }

                        if (closed_set.count(current)) continue;
                        closed_set.insert(current);

                        // Explore neighbors
                        for (int e = 0; e < graph.max_degree; ++e)
                        {
                            int64_t neighbor = graph.edges(current, e);
                            if (neighbor < 0) break;
                            if (closed_set.count(neighbor)) continue;

                            float tentative_g = g_score[current] + graph.edge_weights(current, e) + graph.node_costs(neighbor);
                            if (!g_score.count(neighbor) || tentative_g < g_score[neighbor])
                            {
                                came_from[neighbor] = current;
                                g_score[neighbor] = tentative_g;
                                float f = tentative_g + h(neighbor);
                                open_set.push({f, neighbor});
                            }
                        }
                    }
                    return result;
                }

                // Batch query: multiple start-goal pairs
                static std::vector<PathResult> find_paths_batch(const PathfindingGraph& graph,
                                                                const xarray_container<int64_t>& starts,
                                                                const xarray_container<int64_t>& goals,
                                                                const std::string& heuristic = "euclidean")
                {
                    size_t n_queries = starts.shape()[0];
                    std::vector<PathResult> results(n_queries);
                    for (size_t i = 0; i < n_queries; ++i)
                    {
                        results[i] = find_path(graph, starts(i), goals(i), heuristic);
                    }
                    return results;
                }

                // Multi-agent path finding with conflict avoidance (CBS - Conflict-Based Search)
                static std::vector<PathResult> find_paths_cbs(const PathfindingGraph& graph,
                                                              const xarray_container<int64_t>& starts,
                                                              const xarray_container<int64_t>& goals,
                                                              const std::string& heuristic = "euclidean")
                {
                    // Simplified: just independent paths
                    return find_paths_batch(graph, starts, goals, heuristic);
                }
            };

            // --------------------------------------------------------------------
            // Flow Field (for large crowds)
            // --------------------------------------------------------------------
            class FlowField
            {
            public:
                xarray_container<float> cost_map;       // H x W
                xarray_container<float> integration;    // H x W (distance to goal)
                xarray_container<float> flow_x;         // H x W (direction x)
                xarray_container<float> flow_y;         // H x W (direction y)
                int64_t goal_x = 0, goal_y = 0;

                void build(const xarray_container<float>& cost_map,
                           const std::vector<std::pair<int,int>>& goals)
                {
                    this->cost_map = cost_map;
                    size_t h = cost_map.shape()[0];
                    size_t w = cost_map.shape()[1];
                    
                    integration = xarray_container<float>({h, w}, std::numeric_limits<float>::max());
                    flow_x = xarray_container<float>({h, w}, 0.0f);
                    flow_y = xarray_container<float>({h, w}, 0.0f);
                    
                    // Use first goal as target
                    if (!goals.empty())
                    {
                        goal_x = goals[0].first;
                        goal_y = goals[0].second;
                    }
                    
                    // Dijkstra / BFS from goal to compute integration field
                    std::queue<std::pair<int,int>> q;
                    integration(goal_y, goal_x) = 0.0f;
                    q.push({goal_x, goal_y});
                    
                    const int dx[4] = {1, 0, -1, 0};
                    const int dy[4] = {0, 1, 0, -1};
                    
                    while (!q.empty())
                    {
                        auto [x, y] = q.front(); q.pop();
                        float dist = integration(y, x);
                        
                        for (int d = 0; d < 4; ++d)
                        {
                            int nx = x + dx[d];
                            int ny = y + dy[d];
                            if (nx < 0 || nx >= static_cast<int>(w) || ny < 0 || ny >= static_cast<int>(h))
                                continue;
                            if (cost_map(ny, nx) > 0.5f) continue; // obstacle
                            
                            float new_dist = dist + 1.0f;
                            if (new_dist < integration(ny, nx))
                            {
                                integration(ny, nx) = new_dist;
                                q.push({nx, ny});
                            }
                        }
                    }
                    
                    // Compute flow directions (gradient of integration field)
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            if (std::isinf(integration(y, x))) continue;
                            
                            float min_val = integration(y, x);
                            int best_dx = 0, best_dy = 0;
                            for (int d = 0; d < 4; ++d)
                            {
                                int nx = static_cast<int>(x) + dx[d];
                                int ny = static_cast<int>(y) + dy[d];
                                if (nx < 0 || nx >= static_cast<int>(w) || ny < 0 || ny >= static_cast<int>(h))
                                    continue;
                                if (integration(ny, nx) < min_val)
                                {
                                    min_val = integration(ny, nx);
                                    best_dx = dx[d];
                                    best_dy = dy[d];
                                }
                            }
                            flow_x(y, x) = static_cast<float>(best_dx);
                            flow_y(y, x) = static_cast<float>(best_dy);
                        }
                    }
                }

                godot::Vector2 get_direction(float x, float y) const
                {
                    int ix = static_cast<int>(x);
                    int iy = static_cast<int>(y);
                    size_t h = flow_x.shape()[0];
                    size_t w = flow_x.shape()[1];
                    if (ix < 0 || ix >= static_cast<int>(w) || iy < 0 || iy >= static_cast<int>(h))
                        return godot::Vector2();
                    return godot::Vector2(flow_x(iy, ix), flow_y(iy, ix));
                }

                // Sample flow direction with bilinear interpolation
                godot::Vector2 sample_direction(float x, float y) const
                {
                    size_t h = flow_x.shape()[0];
                    size_t w = flow_x.shape()[1];
                    if (x < 0 || x >= w-1 || y < 0 || y >= h-1)
                        return godot::Vector2();
                    
                    int x0 = static_cast<int>(std::floor(x));
                    int y0 = static_cast<int>(std::floor(y));
                    int x1 = std::min(x0 + 1, static_cast<int>(w)-1);
                    int y1 = std::min(y0 + 1, static_cast<int>(h)-1);
                    float fx = x - x0;
                    float fy = y - y0;
                    
                    float v00_x = flow_x(y0, x0), v00_y = flow_y(y0, x0);
                    float v10_x = flow_x(y0, x1), v10_y = flow_y(y0, x1);
                    float v01_x = flow_x(y1, x0), v01_y = flow_y(y1, x0);
                    float v11_x = flow_x(y1, x1), v11_y = flow_y(y1, x1);
                    
                    float v0_x = v00_x * (1-fx) + v10_x * fx;
                    float v0_y = v00_y * (1-fx) + v10_y * fx;
                    float v1_x = v01_x * (1-fx) + v11_x * fx;
                    float v1_y = v01_y * (1-fx) + v11_y * fx;
                    
                    float vx = v0_x * (1-fy) + v1_x * fy;
                    float vy = v0_y * (1-fy) + v1_y * fy;
                    return godot::Vector2(vx, vy);
                }
            };

            // --------------------------------------------------------------------
            // XPathfindingNode - Godot node for tensor-based pathfinding
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XPathfindingNode : public godot::Node
            {
                GDCLASS(XPathfindingNode, godot::Node)

            private:
                PathfindingGraph m_graph;
                godot::Ref<XTensorNode> m_cost_map;
                godot::Ref<XTensorNode> m_starts;
                godot::Ref<XTensorNode> m_goals;
                godot::Ref<XTensorNode> m_paths_output;
                FlowField m_flow_field;
                bool m_use_flow_field = false;
                bool m_allow_diagonal = true;
                float m_agent_radius = 0.5f;
                godot::String m_heuristic = "euclidean";

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_cost_map", "tensor"), &XPathfindingNode::set_cost_map);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_cost_map"), &XPathfindingNode::get_cost_map);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_starts", "tensor"), &XPathfindingNode::set_starts);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_starts"), &XPathfindingNode::get_starts);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_goals", "tensor"), &XPathfindingNode::set_goals);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_goals"), &XPathfindingNode::get_goals);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_paths_output", "tensor"), &XPathfindingNode::set_paths_output);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_paths_output"), &XPathfindingNode::get_paths_output);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("set_allow_diagonal", "enabled"), &XPathfindingNode::set_allow_diagonal);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_allow_diagonal"), &XPathfindingNode::get_allow_diagonal);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_use_flow_field", "enabled"), &XPathfindingNode::set_use_flow_field);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_use_flow_field"), &XPathfindingNode::get_use_flow_field);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_heuristic", "name"), &XPathfindingNode::set_heuristic);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_heuristic"), &XPathfindingNode::get_heuristic);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("build_graph"), &XPathfindingNode::build_graph);
                    godot::ClassDB::bind_method(godot::D_METHOD("find_paths"), &XPathfindingNode::find_paths);
                    godot::ClassDB::bind_method(godot::D_METHOD("build_flow_field", "goals"), &XPathfindingNode::build_flow_field);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_flow_direction", "position"), &XPathfindingNode::get_flow_direction);
                    godot::ClassDB::bind_method(godot::D_METHOD("sample_flow_direction", "position"), &XPathfindingNode::sample_flow_direction);
                    godot::ClassDB::bind_method(godot::D_METHOD("smooth_path", "path", "iterations"), &XPathfindingNode::smooth_path, godot::DEFVAL(5));
                    godot::ClassDB::bind_method(godot::D_METHOD("simplify_path", "path", "epsilon"), &XPathfindingNode::simplify_path, godot::DEFVAL(0.1f));
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "cost_map", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_cost_map", "get_cost_map");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "starts", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_starts", "get_starts");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "goals", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_goals", "get_goals");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "paths_output", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_paths_output", "get_paths_output");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "allow_diagonal"), "set_allow_diagonal", "get_allow_diagonal");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "use_flow_field"), "set_use_flow_field", "get_use_flow_field");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "heuristic"), "set_heuristic", "get_heuristic");
                    
                    ADD_SIGNAL(godot::MethodInfo("paths_found", godot::PropertyInfo(godot::Variant::INT, "count")));
                }

            public:
                void set_cost_map(const godot::Ref<XTensorNode>& tensor) { m_cost_map = tensor; }
                godot::Ref<XTensorNode> get_cost_map() const { return m_cost_map; }
                void set_starts(const godot::Ref<XTensorNode>& tensor) { m_starts = tensor; }
                godot::Ref<XTensorNode> get_starts() const { return m_starts; }
                void set_goals(const godot::Ref<XTensorNode>& tensor) { m_goals = tensor; }
                godot::Ref<XTensorNode> get_goals() const { return m_goals; }
                void set_paths_output(const godot::Ref<XTensorNode>& tensor) { m_paths_output = tensor; }
                godot::Ref<XTensorNode> get_paths_output() const { return m_paths_output; }
                
                void set_allow_diagonal(bool enable) { m_allow_diagonal = enable; }
                bool get_allow_diagonal() const { return m_allow_diagonal; }
                void set_use_flow_field(bool enable) { m_use_flow_field = enable; }
                bool get_use_flow_field() const { return m_use_flow_field; }
                void set_heuristic(const godot::String& name) { m_heuristic = name; }
                godot::String get_heuristic() const { return m_heuristic; }

                void build_graph()
                {
                    if (!m_cost_map.is_valid())
                    {
                        godot::UtilityFunctions::printerr("XPathfindingNode: cost_map not set");
                        return;
                    }
                    auto grid = m_cost_map->get_tensor_resource()->m_data.to_float_array();
                    m_graph = PathfindingGraph::from_grid(grid, 0.5f, m_allow_diagonal);
                }

                void find_paths()
                {
                    if (!m_starts.is_valid() || !m_goals.is_valid())
                    {
                        godot::UtilityFunctions::printerr("XPathfindingNode: starts or goals not set");
                        return;
                    }
                    if (m_graph.edges.size() == 0)
                    {
                        build_graph();
                    }
                    
                    auto starts = m_starts->get_tensor_resource()->m_data.to_int_array();
                    auto goals = m_goals->get_tensor_resource()->m_data.to_int_array();
                    size_t n = std::min(starts.size(), goals.size());
                    
                    std::string heuristic = m_heuristic.utf8().get_data();
                    auto results = BatchAStar::find_paths_batch(m_graph, starts, goals, heuristic);
                    
                    // Pack results into tensor
                    // Format: [path_count, max_path_length, 3] where 3 = (x, y, cost)?
                    // We'll output a variable-length path array using a ragged tensor approach,
                    // but for simplicity, we pad to max length.
                    size_t max_len = 0;
                    for (const auto& r : results)
                        max_len = std::max(max_len, r.path.size());
                    
                    xarray_container<float> paths({n, max_len, 2}, -1.0f);
                    for (size_t i = 0; i < n; ++i)
                    {
                        const auto& path = results[i].path;
                        for (size_t j = 0; j < path.size(); ++j)
                        {
                            int64_t node = path[j];
                            if (node >= 0 && static_cast<size_t>(node) < m_graph.node_positions.shape()[0])
                            {
                                paths(i, j, 0) = m_graph.node_positions(node, 0);
                                paths(i, j, 1) = m_graph.node_positions(node, 1);
                            }
                        }
                    }
                    
                    if (!m_paths_output.is_valid())
                        m_paths_output.instantiate();
                    m_paths_output->set_data(XVariant::from_xarray(paths.cast<double>()).variant());
                    emit_signal("paths_found", static_cast<int64_t>(n));
                }

                void build_flow_field(const godot::PackedVector2Array& goals)
                {
                    if (!m_cost_map.is_valid()) return;
                    auto grid = m_cost_map->get_tensor_resource()->m_data.to_float_array();
                    
                    std::vector<std::pair<int,int>> goal_cells;
                    for (int i = 0; i < goals.size(); ++i)
                    {
                        godot::Vector2 g = goals[i];
                        goal_cells.emplace_back(static_cast<int>(g.x), static_cast<int>(g.y));
                    }
                    m_flow_field.build(grid, goal_cells);
                }

                godot::Vector2 get_flow_direction(const godot::Vector2& position) const
                {
                    return m_flow_field.get_direction(position.x, position.y);
                }

                godot::Vector2 sample_flow_direction(const godot::Vector2& position) const
                {
                    return m_flow_field.sample_direction(position.x, position.y);
                }

                godot::PackedVector2Array smooth_path(const godot::PackedVector2Array& path, int iterations)
                {
                    godot::PackedVector2Array result = path;
                    for (int iter = 0; iter < iterations; ++iter)
                    {
                        godot::PackedVector2Array smoothed = result;
                        for (int i = 1; i < result.size() - 1; ++i)
                        {
                            godot::Vector2 prev = result[i-1];
                            godot::Vector2 curr = result[i];
                            godot::Vector2 next = result[i+1];
                            smoothed.set(i, (prev + curr + next) / 3.0f);
                        }
                        result = smoothed;
                    }
                    return result;
                }

                godot::PackedVector2Array simplify_path(const godot::PackedVector2Array& path, float epsilon)
                {
                    // Ramer-Douglas-Peucker algorithm
                    if (path.size() < 3) return path;
                    
                    std::vector<bool> keep(path.size(), true);
                    rdp_simplify(path, 0, path.size() - 1, epsilon, keep);
                    
                    godot::PackedVector2Array result;
                    for (int i = 0; i < path.size(); ++i)
                        if (keep[i])
                            result.append(path[i]);
                    return result;
                }

            private:
                void rdp_simplify(const godot::PackedVector2Array& points,
                                  int start, int end, float epsilon,
                                  std::vector<bool>& keep)
                {
                    if (end - start < 2) return;
                    
                    godot::Vector2 line_start = points[start];
                    godot::Vector2 line_end = points[end];
                    godot::Vector2 line_vec = line_end - line_start;
                    float line_len = line_vec.length();
                    
                    float max_dist = 0.0f;
                    int max_idx = start;
                    for (int i = start + 1; i < end; ++i)
                    {
                        godot::Vector2 pt = points[i];
                        godot::Vector2 pt_vec = pt - line_start;
                        float dist;
                        if (line_len < 1e-6f)
                            dist = pt_vec.length();
                        else
                            dist = std::abs(pt_vec.cross(line_vec)) / line_len;
                        if (dist > max_dist)
                        {
                            max_dist = dist;
                            max_idx = i;
                        }
                    }
                    
                    if (max_dist > epsilon)
                    {
                        rdp_simplify(points, start, max_idx, epsilon, keep);
                        rdp_simplify(points, max_idx, end, epsilon, keep);
                    }
                    else
                    {
                        for (int i = start + 1; i < end; ++i)
                            keep[i] = false;
                    }
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XPathfindingRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XPathfindingNode>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::PathfindingGraph;
        using godot_bridge::BatchAStar;
        using godot_bridge::FlowField;
        using godot_bridge::XPathfindingNode;
        using godot_bridge::XPathfindingRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XPATHFINDING_HPP

// godot/xpathfinding.hpp