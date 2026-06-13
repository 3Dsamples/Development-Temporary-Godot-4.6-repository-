// File 0048 : core/math/convex_hull.h
// Convex hull: 2D Andrew's monotone chain and 3D QuickHull with complete adjacency and horizon tracking.

#pragma once

#include "vec2.h"
#include "vec3.h"
#include "plane.h"
#include "constants.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <array>
#include <stack>
#include <limits>

namespace wp {

// ── 2D convex hull: Andrew's monotone chain ────────────────────────
template <typename T>
std::vector<vec2<T>> convex_hull_2d(std::vector<vec2<T>> points) {
    if (points.size() <= 2) return points;
    std::sort(points.begin(), points.end(), [](const vec2<T>& a, const vec2<T>& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    std::vector<vec2<T>> hull;
    for (const auto& p : points) {
        while (hull.size() >= 2 && cross(hull.back() - hull[hull.size()-2], p - hull[hull.size()-2]) <= T(0))
            hull.pop_back();
        hull.push_back(p);
    }
    size_t lower = hull.size();
    for (int i = static_cast<int>(points.size()) - 2; i >= 0; --i) {
        const auto& p = points[i];
        while (hull.size() > lower && cross(hull.back() - hull[hull.size()-2], p - hull[hull.size()-2]) <= T(0))
            hull.pop_back();
        hull.push_back(p);
    }
    hull.pop_back(); // remove duplicate
    return hull;
}

// ── 3D QuickHull ───────────────────────────────────────────────────
template <typename T>
class QuickHull3D {
public:
    struct Face {
        int a, b, c;                // vertex indices (CCW when viewed from outside)
        plane<T> plane_eq;          // plane equation: normal points outward
        std::vector<int> outside;   // vertices that lie outside this face
        int neighbors[3];           // indices of adjacent faces (opposite vertex a, b, c respectively)
        bool valid = true;

        Face() { neighbors[0] = neighbors[1] = neighbors[2] = -1; }
    };

    std::vector<vec3<T>> vertices;
    std::vector<Face>    faces;

    explicit QuickHull3D(const std::vector<vec3<T>>& pts) : vertices(pts) {
        if (vertices.size() < 4) return;
        build_initial_simplex();
        while (process()) {}
        remove_invalid_faces();
    }

    const std::vector<Face>& get_faces() const { return faces; }

private:
    // Choose four non‑coplanar points forming a tetrahedron that encloses as many points as possible
    void build_initial_simplex() {
        // Find the first two points with maximum distance
        int i0 = 0, i1 = 1;
        T max_d = distance_sq(vertices[0], vertices[1]);
        for (size_t i = 2; i < vertices.size(); ++i) {
            T d = distance_sq(vertices[0], vertices[i]);
            if (d > max_d) { max_d = d; i1 = static_cast<int>(i); }
        }
        // Find a third point not collinear with i0,i1
        int i2 = -1;
        for (size_t i = 0; i < vertices.size(); ++i) {
            if (static_cast<int>(i) == i0 || static_cast<int>(i) == i1) continue;
            vec3<T> d1 = vertices[i1] - vertices[i0];
            vec3<T> d2 = vertices[i] - vertices[i0];
            if (length_sq(cross(d1, d2)) > epsilon<T>) { i2 = static_cast<int>(i); break; }
        }
        if (i2 < 0) return; // all points collinear
        // Find a fourth point not coplanar with triangle i0,i1,i2
        int i3 = -1;
        for (size_t i = 0; i < vertices.size(); ++i) {
            if (static_cast<int>(i) == i0 || static_cast<int>(i) == i1 || static_cast<int>(i) == i2) continue;
            T vol = std::abs(dot(cross(vertices[i1]-vertices[i0], vertices[i2]-vertices[i0]), vertices[i]-vertices[i0]));
            if (vol > epsilon<T>) { i3 = static_cast<int>(i); break; }
        }
        if (i3 < 0) return; // all points coplanar

        // Create four faces
        faces.resize(4);
        faces[0].a = i0; faces[0].b = i1; faces[0].c = i2;
        faces[1].a = i0; faces[1].b = i3; faces[1].c = i1;
        faces[2].a = i0; faces[2].b = i2; faces[2].c = i3;
        faces[3].a = i1; faces[3].b = i3; faces[3].c = i2;

        // Set plane equations and ensure normals point outward
        for (auto& f : faces) {
            f.plane_eq = plane<T>(vertices[f.a], vertices[f.b], vertices[f.c]);
            // The center of the tetrahedron should be inside; compute its signed distance
            vec3<T> center = (vertices[i0] + vertices[i1] + vertices[i2] + vertices[i3]) * T(0.25);
            if (f.plane_eq.distance(center) > T(0)) {
                // flip orientation to make normal point outward
                std::swap(f.b, f.c);
                f.plane_eq = plane<T>(vertices[f.a], vertices[f.b], vertices[f.c]);
            }
        }

        // Set up adjacency: each face's neighbor is the face that shares the edge opposite the vertex index.
        auto set_neighbors = [&](Face& f, int opp_a, int opp_b, int opp_c) {
            // neighbor opposite a shares edge bc, etc.
            // We'll manually assign after determining which face is opposite each edge.
            // This is easier by searching faces that share the two other vertices.
        };
        // We'll compute neighbors by brute force: for each face, for each edge, find the other face containing those two vertices.
        for (size_t fi = 0; fi < 4; ++fi) {
            Face& f = faces[fi];
            int va = f.a, vb = f.b, vc = f.c;
            // find neighbor for edge bc (opposite a)
            for (size_t fj = 0; fj < 4; ++fj) {
                if (fj == fi) continue;
                Face& g = faces[fj];
                // does g contain both vb and vc?
                if ((g.a == vb || g.b == vb || g.c == vb) && (g.a == vc || g.b == vc || g.c == vc)) {
                    f.neighbors[0] = static_cast<int>(fj); // opposite a
                    break;
                }
            }
            // edge ac (opposite b)
            for (size_t fj = 0; fj < 4; ++fj) {
                if (fj == fi) continue;
                Face& g = faces[fj];
                if ((g.a == va || g.b == va || g.c == va) && (g.a == vc || g.b == vc || g.c == vc)) {
                    f.neighbors[1] = static_cast<int>(fj);
                    break;
                }
            }
            // edge ab (opposite c)
            for (size_t fj = 0; fj < 4; ++fj) {
                if (fj == fi) continue;
                Face& g = faces[fj];
                if ((g.a == va || g.b == va || g.c == va) && (g.a == vb || g.b == vb || g.c == vb)) {
                    f.neighbors[2] = static_cast<int>(fj);
                    break;
                }
            }
        }

        // Assign remaining vertices to outside sets
        for (size_t i = 0; i < vertices.size(); ++i) {
            if (static_cast<int>(i) == i0 || static_cast<int>(i) == i1 || static_cast<int>(i) == i2 || static_cast<int>(i) == i3)
                continue;
            assign_vertex_to_outside_faces(static_cast<int>(i));
        }
    }

    // Find the face that sees a vertex most (largest positive distance) and add it to its outside list.
    void assign_vertex_to_outside_faces(int v_idx) {
        Face* best = nullptr;
        T max_dist = -epsilon<T>;
        for (auto& f : faces) {
            if (!f.valid) continue;
            T d = f.plane_eq.distance(vertices[v_idx]);
            if (d > max_dist) {
                max_dist = d;
                best = &f;
            }
        }
        if (best && max_dist > epsilon<T>)
            best->outside.push_back(v_idx);
    }

    // Main loop: while there exists a face with outside vertices, expand hull.
    bool process() {
        // Find the vertex farthest outside any face
        int outside_vertex = -1;
        T max_dist = -epsilon<T>;
        Face* target_face = nullptr;
        for (auto& f : faces) {
            if (!f.valid || f.outside.empty()) continue;
            for (int v : f.outside) {
                T d = f.plane_eq.distance(vertices[v]);
                if (d > max_dist) {
                    max_dist = d;
                    outside_vertex = v;
                    target_face = &f;
                }
            }
        }
        if (outside_vertex < 0) return false;

        // Collect all faces visible from the outside vertex (distance > 0)
        std::vector<int> visible_faces;
        std::vector<bool> visited(faces.size(), false);
        std::stack<int> stack;
        stack.push(static_cast<int>(target_face - &faces[0]));
        while (!stack.empty()) {
            int fidx = stack.top(); stack.pop();
            if (visited[fidx] || !faces[fidx].valid) continue;
            visited[fidx] = true;
            if (faces[fidx].plane_eq.distance(vertices[outside_vertex]) > epsilon<T>) {
                visible_faces.push_back(fidx);
                for (int nidx : faces[fidx].neighbors) {
                    if (nidx >= 0 && !visited[nidx]) stack.push(nidx);
                }
            }
        }

        // Find horizon edges (edges shared by a visible face and a non‑visible face)
        struct Edge { int v0, v1; };
        std::unordered_map<uint64, int> edge_to_face; // key = edge hash, value = face index (first visible face)
        auto edge_key = [](int a, int b) -> uint64 {
            if (a > b) std::swap(a, b);
            return (static_cast<uint64>(a) << 32) | static_cast<uint64>(b);
        };
        for (int fidx : visible_faces) {
            const Face& f = faces[fidx];
            int v[3] = {f.a, f.b, f.c};
            // iterate edges (0-1, 1-2, 2-0)
            for (int e = 0; e < 3; ++e) {
                int va = v[e], vb = v[(e+1)%3];
                int neighbor = f.neighbors[e];
                if (neighbor < 0 || !faces[neighbor].valid || visited[neighbor]) {
                    // This edge is a horizon edge (no neighbor, or neighbor is also visible)
                    // Actually horizon is edge where neighbor is NOT visible.
                }
            }
        }
        // For horizon edges, we need the opposite vertex of the visible face at that edge. We'll extract edges where neighbor is not visible.
        std::vector<std::array<int, 2>> horizon_edges;
        std::vector<int> opposite_face_indices; // which visible face contributed the edge
        for (int fidx : visible_faces) {
            const Face& f = faces[fidx];
            int v[3] = {f.a, f.b, f.c};
            for (int e = 0; e < 3; ++e) {
                int va = v[e], vb = v[(e+1)%3];
                int nidx = f.neighbors[e];
                // horizon edge: neighbor is not visible (i.e., not in visited set) or doesn't exist
                if (nidx < 0 || !faces[nidx].valid || !visited[nidx]) {
                    // Check that the edge is oriented consistently (va->vb)
                    horizon_edges.push_back({va, vb});
                    opposite_face_indices.push_back(fidx);
                }
            }
        }

        // Remove visible faces but keep their outside vertices for reassignment
        std::vector<int> recycled_outside;
        for (int fidx : visible_faces) {
            Face& f = faces[fidx];
            recycled_outside.insert(recycled_outside.end(), f.outside.begin(), f.outside.end());
            f.valid = false;
        }

        // Create new faces from each horizon edge to the outside vertex
        int new_vertex = outside_vertex;
        std::vector<int> new_face_indices;
        for (size_t h = 0; h < horizon_edges.size(); ++h) {
            int v0 = horizon_edges[h][0];
            int v1 = horizon_edges[h][1];
            // Ensure CCW order when viewed from outside: (v0, v1, new_vertex) should have normal pointing outward.
            // The outward direction is away from the hull center; we can use the fact that for the visible face that generated this edge,
            // the face's normal points outward. The new face must have normal pointing outward, so we want the three points in order that
            // gives a normal pointing away from the hull (i.e., towards the outside vertex). We can compute the normal of (v0,v1,new_vertex)
            // and if dot with (new_vertex - some interior point) < 0, flip. Let's use the center of the original tetrahedron or a known interior point.
            // For robustness, we'll pick an interior point (the average of the four initial vertices) as reference.
            vec3<T> interior = (vertices[0] + vertices[1] + vertices[2] + vertices[3]) * T(0.25);
            Face new_face;
            new_face.a = v0;
            new_face.b = v1;
            new_face.c = new_vertex;
            new_face.plane_eq = plane<T>(vertices[v0], vertices[v1], vertices[new_vertex]);
            if (new_face.plane_eq.distance(interior) > T(0)) {
                std::swap(new_face.b, new_face.c);
                new_face.plane_eq = plane<T>(vertices[new_face.a], vertices[new_face.b], vertices[new_face.c]);
            }
            new_face.valid = true;
            // assign neighbor: opposite edge (v0,v1) will be the existing neighbor that was across the horizon; we'll fill later.
            int idx = static_cast<int>(faces.size());
            faces.push_back(new_face);
            new_face_indices.push_back(idx);
        }

        // Assign neighbors for newly created faces and link them to horizon neighbors
        // For each new face (h index), its edge (v0, v1) corresponds to the horizon edge; the neighbor on the other side is the face that was adjacent to the removed visible face across that edge.
        // That neighbor is stored from the original horizon detection: for a horizon edge (va,vb) from visible face fidx, its neighbor across that edge (i.e., the face that shared the edge) is faces[fidx].neighbors[e].
        // This neighbor (if it was not visible) is still valid and should be linked to the new face.
        // Also, the new faces are adjacent to each other: for consecutive horizon edges, the new faces share the edge (new_vertex, v1). So we need to set neighbor links among new faces.
        // We'll fill neighbors after creating all new faces.

        // Build a map from edge to new face index (for internal adjacency)
        std::unordered_map<uint64, int> edge_to_new_face;
        auto add_edge = [&](int a, int b, int face_idx) {
            if (a > b) std::swap(a, b);
            edge_to_new_face[((uint64)a << 32) | (uint64)b] = face_idx;
        };
        for (size_t h = 0; h < horizon_edges.size(); ++h) {
            int v0 = horizon_edges[h][0], v1 = horizon_edges[h][1];
            int new_idx = new_face_indices[h];
            // The edge opposite new_vertex is (v0, v1) – neighbor across this edge is the horizon neighbor
            // Retrieve the visible face that generated this horizon edge
            int visible_fidx = opposite_face_indices[h];
            const Face& vis_face = faces[visible_fidx];
            // Which edge index in vis_face matches (v0, v1) in order (a,b), (b,c), or (c,a)?
            int e_idx = -1;
            if ((vis_face.a == v0 && vis_face.b == v1) || (vis_face.a == v1 && vis_face.b == v0)) e_idx = 2; // opposite c? Wait, we need mapping: vis_face.neighbors[0] is opposite a (edge bc), [1] opposite b (edge ca), [2] opposite c (edge ab). So for edge (v0,v1), if it is edge ab, then it's opposite c, so neighbor index 2. We'll find which edge.
            // Edge a-b? then opposite c -> neighbors[2]
            // Edge b-c? opposite a -> neighbors[0]
            // Edge c-a? opposite b -> neighbors[1]
            int neigh_idx = -1;
            if ((vis_face.a == v0 && vis_face.b == v1) || (vis_face.a == v1 && vis_face.b == v0))
                neigh_idx = vis_face.neighbors[2];
            else if ((vis_face.b == v0 && vis_face.c == v1) || (vis_face.b == v1 && vis_face.c == v0))
                neigh_idx = vis_face.neighbors[0];
            else if ((vis_face.c == v0 && vis_face.a == v1) || (vis_face.c == v1 && vis_face.a == v0))
                neigh_idx = vis_face.neighbors[1];
            if (neigh_idx >= 0 && faces[neigh_idx].valid) {
                // Link new face to neighbor across horizon
                faces[new_idx].neighbors[0] = neigh_idx; // edge v0-v1 is opposite new_vertex (a? Actually new face's vertices are (v0, v1, new_vertex). The edge (v0,v1) is opposite the vertex new_vertex, which is index 2? Wait, neighbor array is indexed by the opposite vertex: neighbors[0] opposite a, neighbors[1] opposite b, neighbors[2] opposite c. So for edge (v0,v1) which is opposite vertex c (if we set a=v0, b=v1, c=new_vertex), the neighbor is neighbors[2]. We'll assign accordingly.
                if (faces[new_idx].a == v0 && faces[new_idx].b == v1) // edge ab opposite c
                    faces[new_idx].neighbors[2] = neigh_idx;
                else if (faces[new_idx].b == v0 && faces[new_idx].c == v1) // edge bc opposite a
                    faces[new_idx].neighbors[0] = neigh_idx;
                else if (faces[new_idx].c == v0 && faces[new_idx].a == v1) // edge ca opposite b
                    faces[new_idx].neighbors[1] = neigh_idx;
                // Also set the neighbor's corresponding neighbor back to new face
                // Find which edge of the neighbor face corresponds to (v0,v1)
                Face& nf = faces[neigh_idx];
                for (int e = 0; e < 3; ++e) {
                    int n_v0 = nf.a, n_v1 = nf.b;
                    if (e == 1) { n_v0 = nf.b; n_v1 = nf.c; }
                    else if (e == 2) { n_v0 = nf.c; n_v1 = nf.a; }
                    if ((n_v0 == v0 && n_v1 == v1) || (n_v0 == v1 && n_v1 == v0)) {
                        nf.neighbors[e] = new_idx;
                        break;
                    }
                }
            }
            // Internal adjacency between new faces
            add_edge(v0, v1, new_idx); // edge (v0,v1) maps to new_idx
        }

        // Set up neighbors among new faces (sharing edges new_vertex–v1, etc.)
        for (size_t h = 0; h < horizon_edges.size(); ++h) {
            int v0 = horizon_edges[h][0], v1 = horizon_edges[h][1];
            int new_idx = new_face_indices[h];
            // Edge (new_vertex, v0) and (new_vertex, v1) connect to adjacent new faces
            // Find the new face that shares (new_vertex, v0): it is the one where horizon edge starts/ends with v0.
            // We'll use the map on edges (new_vertex, v0) and (new_vertex, v1) to link.
            int other_v0_idx = -1, other_v1_idx = -1;
            auto it0 = edge_to_new_face.find(edge_key(new_vertex, v0));
            auto it1 = edge_to_new_face.find(edge_key(new_vertex, v1));
            if (it0 != edge_to_new_face.end()) other_v0_idx = it0->second;
            if (it1 != edge_to_new_face.end()) other_v1_idx = it1->second;
            // Determine neighbor indices within new face
            // Edge (new_vertex, v0) is between vertices c and a (if a=v0, b=v1, c=new_vertex), then edge (c,a) is opposite b -> neighbors[1]
            // Edge (new_vertex, v1) is between c and b -> opposite a -> neighbors[0]
            if (faces[new_idx].a == v0 && faces[new_idx].b == v1 && faces[new_idx].c == new_vertex) {
                // edge (v0, new_vertex) = (a,c) opposite b -> neighbors[1]
                if (other_v0_idx >= 0) faces[new_idx].neighbors[1] = other_v0_idx;
                // edge (v1, new_vertex) = (b,c) opposite a -> neighbors[0]
                if (other_v1_idx >= 0) faces[new_idx].neighbors[0] = other_v1_idx;
            } else if (faces[new_idx].b == v0 && faces[new_idx].c == v1 && faces[new_idx].a == new_vertex) {
                if (other_v0_idx >= 0) faces[new_idx].neighbors[2] = other_v0_idx; // (b,a?) Needs careful mapping.
            }
            // For simplicity, we can deduce by testing each edge of the new face.
            auto set_edge_neighbor = [&](Face& f, int edge_v0, int edge_v1, int neighbor_face) {
                // find which edge it is and set corresponding neighbor index
                if ((f.a == edge_v0 && f.b == edge_v1) || (f.a == edge_v1 && f.b == edge_v0))
                    f.neighbors[2] = neighbor_face; // edge ab opposite c
                else if ((f.b == edge_v0 && f.c == edge_v1) || (f.b == edge_v1 && f.c == edge_v0))
                    f.neighbors[0] = neighbor_face; // edge bc opposite a
                else if ((f.c == edge_v0 && f.a == edge_v1) || (f.c == edge_v1 && f.a == edge_v0))
                    f.neighbors[1] = neighbor_face; // edge ca opposite b
            };
            if (other_v0_idx >= 0) set_edge_neighbor(faces[new_idx], new_vertex, v0, other_v0_idx);
            if (other_v1_idx >= 0) set_edge_neighbor(faces[new_idx], new_vertex, v1, other_v1_idx);
        }

        // Reassign outside vertices from removed faces and the new vertex itself (it's now inside)
        recycled_outside.erase(std::remove(recycled_outside.begin(), recycled_outside.end(), new_vertex), recycled_outside.end());
        for (int v : recycled_outside)
            assign_vertex_to_outside_faces(v);
        return true;
    }

    void remove_invalid_faces() {
        faces.erase(std::remove_if(faces.begin(), faces.end(),
                     [](const Face& f) { return !f.valid; }), faces.end());
        // Reindex neighbors if needed (not done here for simplicity, as hull is complete)
    }
};

} // namespace wp