//File 0104 : core/math/mesh_topology_analysis.h
//Complete topology analysis of triangle meshes: Euler characteristic, genus, boundary loops, manifoldness, orientability, and connected components using half‑edge traversal.
#ifndef CORE_MATH_MESH_TOPOLOGY_ANALYSIS_H
#define CORE_MATH_MESH_TOPOLOGY_ANALYSIS_H

#include "mesh_data.h"               // HalfEdgeMesh
#include "vector_math.h"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <limits>

namespace SimulationMath {
namespace mesh_topology {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. Count vertices, undirected edges, faces, and boundary edges
// -----------------------------------------------------------------------------
struct TopologyCounts {
    size_t num_vertices;
    size_t num_edges;          // undirected edges (each counted once)
    size_t num_faces;
    size_t num_boundary_edges; // edges with only one incident face
};

inline TopologyCounts count_elements(const HalfEdgeMesh& mesh) noexcept {
    TopologyCounts counts;
    counts.num_vertices = mesh.vertex_count();
    counts.num_faces    = mesh.faces().size();

    const auto& hedges = mesh.half_edges();
    std::unordered_set<uint64_t> undirected_edges;
    size_t boundary_edges = 0;
    for (size_t i = 0; i < hedges.size(); ++i) {
        const HalfEdge& he = hedges[i];
        uint32_t v0 = he.vertex_index;
        uint32_t v1 = hedges[he.next_edge].vertex_index;
        uint64_t key = (static_cast<uint64_t>(std::min(v0, v1)) << 32) | std::max(v0, v1);
        if (undirected_edges.find(key) == undirected_edges.end()) {
            undirected_edges.insert(key);
            if (he.twin_edge == 0xFFFFFFFFu) boundary_edges++;
        }
    }
    counts.num_edges          = undirected_edges.size();
    counts.num_boundary_edges = boundary_edges;
    return counts;
}

// -----------------------------------------------------------------------------
// 2. Euler characteristic χ = V - E + F
// -----------------------------------------------------------------------------
inline int euler_characteristic(const HalfEdgeMesh& mesh) noexcept {
    TopologyCounts counts = count_elements(mesh);
    return static_cast<int>(counts.num_vertices - counts.num_edges + counts.num_faces);
}

// -----------------------------------------------------------------------------
// 3. Count boundary loops (connected components of boundary edges)
// -----------------------------------------------------------------------------
inline size_t boundary_loop_count(const HalfEdgeMesh& mesh) noexcept {
    const auto& hedges = mesh.half_edges();
    std::vector<bool> visited_he(hedges.size(), false);
    size_t loops = 0;
    for (size_t i = 0; i < hedges.size(); ++i) {
        if (hedges[i].twin_edge != 0xFFFFFFFFu) continue; // not boundary
        if (visited_he[i]) continue;
        // Traverse the boundary loop
        loops++;
        uint32_t he = static_cast<uint32_t>(i);
        do {
            visited_he[he] = true;
            // move to next boundary half‑edge: it is the next edge of the current face (since face's edges are all boundary)
            he = hedges[he].next_edge;
            if (he == 0xFFFFFFFFu) break;
        } while (he != i);
    }
    return loops;
}

// -----------------------------------------------------------------------------
// 4. Genus for an orientable surface: g = (2 - χ - b) / 2
//    where b = number of boundary loops.
// -----------------------------------------------------------------------------
inline int genus(const HalfEdgeMesh& mesh) noexcept {
    int chi = euler_characteristic(mesh);
    size_t b = boundary_loop_count(mesh);
    // Formula only holds for orientable surfaces; for non‑orientable, genus concept differs.
    // We assume orientable closed surface (or with boundaries). For boundaries, genus formula is χ = 2 - 2g - b.
    int g = (2 - chi - static_cast<int>(b)) / 2;
    return std::max(0, g);
}

// -----------------------------------------------------------------------------
// 5. Edge manifoldness: each edge must have exactly 1 (boundary) or 2 (interior) incident faces.
// -----------------------------------------------------------------------------
inline bool is_edge_manifold(const HalfEdgeMesh& mesh) noexcept {
    const auto& hedges = mesh.half_edges();
    // Count incident faces per undirected edge
    std::unordered_map<uint64_t, int> edge_count;
    for (size_t i = 0; i < hedges.size(); ++i) {
        const HalfEdge& he = hedges[i];
        uint32_t v0 = he.vertex_index;
        uint32_t v1 = hedges[he.next_edge].vertex_index;
        uint64_t key = (static_cast<uint64_t>(std::min(v0, v1)) << 32) | std::max(v0, v1);
        edge_count[key]++; // each half‑edge increments count (two per interior edge, one per boundary)
    }
    for (const auto& ec : edge_count) {
        if (ec.second != 2 && ec.second != 1) return false; // non‑manifold edge
    }
    return true;
}

// -----------------------------------------------------------------------------
// 6. Vertex manifoldness: the set of incident faces around a vertex must form a single cycle
//    (topological disk or half‑disk for boundary).
// -----------------------------------------------------------------------------
inline bool is_vertex_manifold(const HalfEdgeMesh& mesh) noexcept {
    const auto& hedges = mesh.half_edges();
    size_t nv = mesh.vertex_count();
    for (size_t v = 0; v < nv; ++v) {
        uint32_t start_he = mesh.vertices()[v].first_edge;
        if (start_he == 0xFFFFFFFFu) continue;

        // Count total incident faces (number of half‑edges around vertex)
        uint32_t he = start_he;
        size_t incident_count = 0;
        do {
            incident_count++;
            he = hedges[hedges[he].prev_edge].twin_edge;
            if (he == 0xFFFFFFFFu) break; // boundary
        } while (he != start_he);

        // Now traverse to check that we visit all incident half‑edges without encountering multiple boundary segments.
        // The link of a manifold vertex is either a single cycle (interior) or a single chain (boundary).
        // If boundary exists, the vertex is on boundary. We should see exactly 2 boundary half‑edges (the two ends of the chain).
        size_t boundary_incident = 0;
        he = start_he;
        do {
            if (hedges[he].twin_edge == 0xFFFFFFFFu)
                boundary_incident++;
            he = hedges[hedges[he].prev_edge].twin_edge;
            if (he == 0xFFFFFFFFu) break;
        } while (he != start_he);

        if (boundary_incident > 0 && boundary_incident != 2) return false; // non‑manifold boundary vertex
        // For interior vertex, boundary_incident must be 0.
        // The traversal should have visited exactly `incident_count` half‑edges. If not, the vertex is non‑manifold.
    }
    return true;
}

inline bool is_manifold(const HalfEdgeMesh& mesh) noexcept {
    return is_edge_manifold(mesh) && is_vertex_manifold(mesh);
}

// -----------------------------------------------------------------------------
// 7. Orientability: can we assign consistent normal directions to all faces?
// -----------------------------------------------------------------------------
inline bool is_orientable(const HalfEdgeMesh& mesh) noexcept {
    size_t nf = mesh.faces().size();
    if (nf == 0) return true;
    std::vector<int> orientation(nf, 0); // 0 = unvisited, 1 / -1
    std::queue<uint32_t> q;
    q.push(0);
    orientation[0] = 1;
    const auto& hedges = mesh.half_edges();

    while (!q.empty()) {
        uint32_t f = q.front(); q.pop();
        const MeshFace& face = mesh.faces()[f];
        uint32_t he0 = face.first_edge;
        if (he0 == 0xFFFFFFFFu) continue;
        uint32_t he = he0;
        do {
            uint32_t twin = hedges[he].twin_edge;
            if (twin != 0xFFFFFFFFu) {
                uint32_t adj_f = hedges[twin].face_index;
                // Determine consistency: current edge direction (v0 -> v1), twin direction (v0' -> v1')
                uint32_t cur_v0 = hedges[he].vertex_index;
                uint32_t cur_v1 = hedges[hedges[he].next_edge].vertex_index;
                uint32_t twin_v0 = hedges[twin].vertex_index;
                // For consistent orientation, the twin edge must go from cur_v1 to cur_v0
                bool consistent = (twin_v0 == cur_v1);
                int expected = consistent ? orientation[f] : -orientation[f];
                if (orientation[adj_f] == 0) {
                    orientation[adj_f] = expected;
                    q.push(adj_f);
                } else if (orientation[adj_f] != expected) {
                    return false; // conflict => non‑orientable
                }
            }
            he = hedges[he].next_edge;
        } while (he != he0);
    }
    return true;
}

// -----------------------------------------------------------------------------
// 8. Number of connected components (via face adjacency)
// -----------------------------------------------------------------------------
inline size_t connected_components(const HalfEdgeMesh& mesh) noexcept {
    size_t nf = mesh.faces().size();
    if (nf == 0) return 0;
    std::vector<bool> visited(nf, false);
    size_t components = 0;
    const auto& hedges = mesh.half_edges();
    for (size_t f = 0; f < nf; ++f) {
        if (visited[f]) continue;
        components++;
        std::queue<uint32_t> q;
        q.push(static_cast<uint32_t>(f));
        visited[f] = true;
        while (!q.empty()) {
            uint32_t cur = q.front(); q.pop();
            const MeshFace& face = mesh.faces()[cur];
            uint32_t he0 = face.first_edge;
            if (he0 == 0xFFFFFFFFu) continue;
            uint32_t he = he0;
            do {
                uint32_t twin = hedges[he].twin_edge;
                if (twin != 0xFFFFFFFFu) {
                    uint32_t adj = hedges[twin].face_index;
                    if (!visited[adj]) {
                        visited[adj] = true;
                        q.push(adj);
                    }
                }
                he = hedges[he].next_edge;
            } while (he != he0);
        }
    }
    return components;
}

} // namespace mesh_topology
} // namespace SimulationMath

#endif // CORE_MATH_MESH_TOPOLOGY_ANALYSIS_H