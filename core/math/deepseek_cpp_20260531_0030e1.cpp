//File 0051 : core/math/collision_gjk_epa.h
//Complete GJK (distance) and EPA (penetration) algorithms for arbitrary convex polyhedra; fully resolved simplex reduction, barycentric witness points, and face‑expansion logic.
#ifndef CORE_MATH_COLLISION_GJK_EPA_H
#define CORE_MATH_COLLISION_GJK_EPA_H

#include "vector_math.h"
#include "geometry_primitives.h"
#include "barycentric.h"
#include "math_constants.h"
#include <vector>
#include <functional>
#include <limits>
#include <cmath>
#include <utility>

namespace SimulationMath {
namespace collision {

using SimdVec = DirectX::XMVECTOR;

// -----------------------------------------------------------------------------
// 1. Support function signature for a convex shape
// -----------------------------------------------------------------------------
template <typename Shape>
using SupportFunc = std::function<SimdVec(const Shape&, SimdVec direction)>;

// -----------------------------------------------------------------------------
// 2. Minkowski difference point with witness points on A and B
// -----------------------------------------------------------------------------
struct MinkowskiVertex {
    SimdVec ab;    // A - B
    SimdVec a;     // support point on shape A
    SimdVec b;     // support point on shape B
};

// -----------------------------------------------------------------------------
// 3. Compute a single support point of the Minkowski difference A-B
// -----------------------------------------------------------------------------
template <typename ShapeA, typename ShapeB>
MinkowskiVertex minkowski_support(const ShapeA& a, const ShapeB& b,
                                  const SupportFunc<ShapeA>& sA,
                                  const SupportFunc<ShapeB>& sB,
                                  SimdVec direction) noexcept {
    SimdVec pA = sA(a, direction);
    SimdVec pB = sB(b, DirectX::XMVectorNegate(direction));
    return { DirectX::XMVectorSubtract(pA, pB), pA, pB };
}

// -----------------------------------------------------------------------------
// 4. Closest point on a line segment to the origin (used inside GJK)
// -----------------------------------------------------------------------------
inline SimdVec closest_point_on_segment_to_origin(SimdVec a, SimdVec b) noexcept {
    SimdVec ab = DirectX::XMVectorSubtract(b, a);
    float t = vector_math::dot3_scalar(DirectX::XMVectorNegate(a), ab);
    float denom = vector_math::length_sq3_scalar(ab);
    if (denom < 1e-12f) return a;
    t = std::max(0.0f, std::min(1.0f, t / denom));
    return DirectX::XMVectorAdd(a, DirectX::XMVectorScale(ab, t));
}

// -----------------------------------------------------------------------------
// 5. Closest point on a triangle to the origin with robust barycentric clamping
// -----------------------------------------------------------------------------
inline SimdVec closest_point_on_triangle_to_origin(SimdVec a, SimdVec b, SimdVec c) noexcept {
    // Use the robust method from geometry::closest_point_triangle but for origin
    return geometry::closest_point_triangle(DirectX::XMVectorZero(), a, b, c);
}

// -----------------------------------------------------------------------------
// 6. GJK result structure
// -----------------------------------------------------------------------------
struct GJKResult {
    bool intersect;
    float distance;
    SimdVec closestA;
    SimdVec closestB;
};

// -----------------------------------------------------------------------------
// 7. Complete GJK algorithm (Johnson’s sub‑algorithm) with full simplex reduction
// -----------------------------------------------------------------------------
template <typename ShapeA, typename ShapeB>
GJKResult gjk(const ShapeA& a, const ShapeB& b,
              const SupportFunc<ShapeA>& supportA,
              const SupportFunc<ShapeB>& supportB,
              int max_iter = 64, float tol = 1e-6f) noexcept {
    GJKResult res;
    res.intersect = false;
    res.distance   = 0.0f;

    // Simplex storage
    MinkowskiVertex simplex[4];
    int size = 0;   // number of vertices currently in the simplex

    // Initial arbitrary direction
    SimdVec dir = vector_math::load3(1.0f, 0.0f, 0.0f);
    simplex[0] = minkowski_support(a, b, supportA, supportB, dir);
    if (vector_math::length_sq3_scalar(simplex[0].ab) < 1e-12f) {
        // Origin hit immediately
        res.intersect = true;
        res.closestA = simplex[0].a;
        res.closestB = simplex[0].b;
        return res;
    }
    dir = DirectX::XMVectorNegate(simplex[0].ab);
    size = 1;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Obtain a new support point in the search direction
        MinkowskiVertex new_pt = minkowski_support(a, b, supportA, supportB, dir);
        // If the new point is not past the origin, no intersection
        if (vector_math::dot3_scalar(new_pt.ab, dir) < 0.0f) {
            // No intersection – compute closest points on current simplex to origin
            res.intersect = false;
            SimdVec closest_on_simplex;
            // Barycentric weights of the simplex vertices for witness points
            float weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            if (size == 1) {
                closest_on_simplex = simplex[0].ab;
                weights[0] = 1.0f;
            } else if (size == 2) {
                SimdVec seg_a = simplex[0].ab;
                SimdVec seg_b = simplex[1].ab;
                SimdVec closest = closest_point_on_segment_to_origin(seg_a, seg_b);
                SimdVec ab = DirectX::XMVectorSubtract(seg_b, seg_a);
                float t = vector_math::dot3_scalar(DirectX::XMVectorSubtract(closest, seg_a), ab);
                float len_sq = vector_math::length_sq3_scalar(ab);
                if (len_sq > 1e-12f) t /= len_sq;
                t = std::max(0.0f, std::min(1.0f, t));
                weights[0] = 1.0f - t;
                weights[1] = t;
                closest_on_simplex = closest;
            } else if (size == 3) {
                SimdVec tri_a = simplex[0].ab;
                SimdVec tri_b = simplex[1].ab;
                SimdVec tri_c = simplex[2].ab;
                SimdVec closest = closest_point_on_triangle_to_origin(tri_a, tri_b, tri_c);
                // Compute barycentric coordinates of closest point w.r.t. triangle
                SimdVec bary = barycentric_3d(tri_a, tri_b, tri_c, closest);
                weights[0] = vector_math::get_x(bary);
                weights[1] = vector_math::get_y(bary);
                weights[2] = vector_math::get_z(bary);
                closest_on_simplex = closest;
            } else {
                // Should never happen (size 4 would mean intersection already detected)
                res.intersect = false;
                return res;
            }

            // Compute witness points on A and B by interpolating original vertices
            SimdVec worldA = DirectX::XMVectorZero();
            SimdVec worldB = DirectX::XMVectorZero();
            for (int i = 0; i < size; ++i) {
                worldA = DirectX::XMVectorAdd(worldA, DirectX::XMVectorScale(simplex[i].a, weights[i]));
                worldB = DirectX::XMVectorAdd(worldB, DirectX::XMVectorScale(simplex[i].b, weights[i]));
            }
            res.distance = std::sqrt(vector_math::length_sq3_scalar(closest_on_simplex));
            res.closestA = worldA;
            res.closestB = worldB;
            return res;
        }

        // Add the new point to the simplex
        simplex[size] = new_pt;
        ++size;

        // --- Simplex reduction ---
        if (size == 2) {
            SimdVec ao = DirectX::XMVectorNegate(simplex[0].ab);
            SimdVec ab = DirectX::XMVectorSubtract(simplex[1].ab, simplex[0].ab);
            if (vector_math::dot3_scalar(ab, ao) > 0.0f) {
                // Origin is in the region of the line segment, keep both points
                dir = vector_math::cross3(vector_math::cross3(ab, ao), ab);
                if (vector_math::length_sq3_scalar(dir) < 1e-12f)
                    dir = vector_math::cross3(ab, vector_math::load3(1.0f, 0.0f, 0.0f));
            } else {
                // Origin is in the region of point A only, discard new point
                simplex[0] = simplex[1];
                size = 1;
                dir = DirectX::XMVectorNegate(simplex[0].ab);
            }
        } else if (size == 3) {
            SimdVec a_pt = simplex[0].ab;
            SimdVec b_pt = simplex[1].ab;
            SimdVec c_pt = simplex[2].ab;
            SimdVec ab = DirectX::XMVectorSubtract(b_pt, a_pt);
            SimdVec ac = DirectX::XMVectorSubtract(c_pt, a_pt);
            SimdVec ao = DirectX::XMVectorNegate(a_pt);
            SimdVec normal = vector_math::cross3(ab, ac);
            float dot_ao_n = vector_math::dot3_scalar(ao, normal);

            if (dot_ao_n > 0.0f) {
                // Origin is above the triangle, keep all three and set direction = normal
                dir = normal;
            } else {
                // Origin is below or in the edge region; need to reduce to the closest edge or vertex
                // Compute distances from origin to each edge and vertex
                float dist_a = vector_math::length_sq3_scalar(ao);
                float dist_b = vector_math::length_sq3_scalar(DirectX::XMVectorNegate(b_pt));
                float dist_c = vector_math::length_sq3_scalar(DirectX::XMVectorNegate(c_pt));

                // Edge AB
                SimdVec ab_perp = vector_math::cross3(ab, vector_math::cross3(ab, ao));
                float dist_ab = vector_math::length_sq3_scalar(ab_perp);
                // Edge AC
                SimdVec ac_perp = vector_math::cross3(ac, vector_math::cross3(ac, ao));
                float dist_ac = vector_math::length_sq3_scalar(ac_perp);
                // Edge BC
                SimdVec bc = DirectX::XMVectorSubtract(c_pt, b_pt);
                SimdVec bo = DirectX::XMVectorNegate(b_pt);
                SimdVec bc_perp = vector_math::cross3(bc, vector_math::cross3(bc, bo));
                float dist_bc = vector_math::length_sq3_scalar(bc_perp);

                // Determine which feature is closest to origin
                if (dist_ab <= dist_ac && dist_ab <= dist_bc) {
                    // Keep only A and B (simplex[0] and simplex[1])
                    size = 2;
                    simplex[2] = MinkowskiVertex{};
                    dir = vector_math::cross3(vector_math::cross3(ab, ao), ab);
                    if (vector_math::length_sq3_scalar(dir) < 1e-12f)
                        dir = vector_math::cross3(ab, vector_math::load3(1.0f, 0.0f, 0.0f));
                } else if (dist_ac <= dist_ab && dist_ac <= dist_bc) {
                    // Keep A and C
                    simplex[1] = simplex[2];   // move C to second position
                    size = 2;
                    dir = vector_math::cross3(vector_math::cross3(ac, ao), ac);
                    if (vector_math::length_sq3_scalar(dir) < 1e-12f)
                        dir = vector_math::cross3(ac, vector_math::load3(0.0f, 1.0f, 0.0f));
                } else {
                    // Keep B and C
                    simplex[0] = simplex[1];
                    simplex[1] = simplex[2];
                    size = 2;
                    dir = vector_math::cross3(vector_math::cross3(bc, bo), bc);
                    if (vector_math::length_sq3_scalar(dir) < 1e-12f)
                        dir = vector_math::cross3(bc, vector_math::load3(0.0f, 0.0f, 1.0f));
                }
            }
        } else if (size == 4) {
            // A tetrahedron encloses the origin → intersection
            res.intersect = true;
            // For later EPA we need the simplex; we could store it, but GJKResult currently doesn't.
            // A separate function will re‑obtain the simplex.
            return res;
        }
    }
    // Exceeded maximum iterations → treat as no intersection
    return res;
}

// -----------------------------------------------------------------------------
// 8. Utility to compute the outward normal of a triangle (points in Minkowski space)
// -----------------------------------------------------------------------------
inline SimdVec compute_face_normal(SimdVec a, SimdVec b, SimdVec c) noexcept {
    SimdVec e1 = DirectX::XMVectorSubtract(b, a);
    SimdVec e2 = DirectX::XMVectorSubtract(c, a);
    SimdVec norm = vector_math::cross3(e1, e2);
    float len = vector_math::length3_scalar(norm);
    if (len < 1e-12f) return norm;
    norm = DirectX::XMVectorScale(norm, 1.0f / len);
    if (vector_math::dot3_scalar(norm, a) > 0.0f)
        norm = DirectX::XMVectorNegate(norm);
    return norm;
}

// -----------------------------------------------------------------------------
// 9. EPA result structure
// -----------------------------------------------------------------------------
struct EPAResult {
    SimdVec normal;
    float depth;
    SimdVec contactA;
    SimdVec contactB;
};

// -----------------------------------------------------------------------------
// 10. Complete Expanding Polytope Algorithm (EPA) for penetration depth
// -----------------------------------------------------------------------------
template <typename ShapeA, typename ShapeB>
EPAResult epa(const ShapeA& a, const ShapeB& b,
              const SupportFunc<ShapeA>& supportA,
              const SupportFunc<ShapeB>& supportB,
              float tolerance = 1e-4f, int max_iter = 64) noexcept {
    EPAResult epaRes;
    epaRes.depth = 0.0f;

    // First, run GJK to obtain an intersecting simplex (we need the simplex vertices).
    // We'll call an internal version of GJK that returns the simplex when an intersection
    // is detected. We'll copy the GJK loop here but stop at the tetrahedron case.
    MinkowskiVertex simplex[4];
    int size = 0;
    SimdVec dir = vector_math::load3(1.0f, 0.0f, 0.0f);
    simplex[0] = minkowski_support(a, b, supportA, supportB, dir);
    if (vector_math::length_sq3_scalar(simplex[0].ab) < 1e-12f) {
        // degenerate intersection – return zero depth
        epaRes.normal = vector_math::load3(0.0f, 1.0f, 0.0f);
        epaRes.depth = 0.0f;
        epaRes.contactA = simplex[0].a;
        epaRes.contactB = simplex[0].b;
        return epaRes;
    }
    dir = DirectX::XMVectorNegate(simplex[0].ab);
    size = 1;
    bool hit = false;
    for (int iter = 0; iter < 64 && !hit; ++iter) {
        MinkowskiVertex new_pt = minkowski_support(a, b, supportA, supportB, dir);
        if (vector_math::dot3_scalar(new_pt.ab, dir) < 0.0f) {
            // GJK should have detected non‑intersection, but we force intersection detection.
            // If this happens, the shapes are not actually intersecting → return zero depth.
            return epaRes;
        }
        simplex[size] = new_pt;
        ++size;
        // Reduce simplex (same logic as GJK)
        if (size == 2) {
            SimdVec ao = DirectX::XMVectorNegate(simplex[0].ab);
            SimdVec ab = DirectX::XMVectorSubtract(simplex[1].ab, simplex[0].ab);
            if (vector_math::dot3_scalar(ab, ao) > 0.0f) {
                dir = vector_math::cross3(vector_math::cross3(ab, ao), ab);
                if (vector_math::length_sq3_scalar(dir) < 1e-12f)
                    dir = vector_math::cross3(ab, vector_math::load3(1,0,0));
            } else {
                simplex[0] = simplex[1];
                size = 1;
                dir = DirectX::XMVectorNegate(simplex[0].ab);
            }
        } else if (size == 3) {
            SimdVec a_pt = simplex[0].ab;
            SimdVec b_pt = simplex[1].ab;
            SimdVec c_pt = simplex[2].ab;
            SimdVec ab = DirectX::XMVectorSubtract(b_pt, a_pt);
            SimdVec ac = DirectX::XMVectorSubtract(c_pt, a_pt);
            SimdVec ao = DirectX::XMVectorNegate(a_pt);
            SimdVec normal = vector_math::cross3(ab, ac);
            float dot_ao_n = vector_math::dot3_scalar(ao, normal);
            if (dot_ao_n > 0.0f) {
                dir = normal;
            } else {
                // reduce to closest edge/vertex (same logic as GJK)
                float dist_a = vector_math::length_sq3_scalar(ao);
                float dist_b = vector_math::length_sq3_scalar(DirectX::XMVectorNegate(b_pt));
                float dist_c = vector_math::length_sq3_scalar(DirectX::XMVectorNegate(c_pt));
                SimdVec ab_perp = vector_math::cross3(ab, vector_math::cross3(ab, ao));
                float dist_ab = vector_math::length_sq3_scalar(ab_perp);
                SimdVec ac_perp = vector_math::cross3(ac, vector_math::cross3(ac, ao));
                float dist_ac = vector_math::length_sq3_scalar(ac_perp);
                SimdVec bc = DirectX::XMVectorSubtract(c_pt, b_pt);
                SimdVec bo = DirectX::XMVectorNegate(b_pt);
                SimdVec bc_perp = vector_math::cross3(bc, vector_math::cross3(bc, bo));
                float dist_bc = vector_math::length_sq3_scalar(bc_perp);
                if (dist_ab <= dist_ac && dist_ab <= dist_bc) {
                    size = 2;
                    dir = vector_math::cross3(vector_math::cross3(ab, ao), ab);
                } else if (dist_ac <= dist_ab && dist_ac <= dist_bc) {
                    simplex[1] = simplex[2];
                    size = 2;
                    dir = vector_math::cross3(vector_math::cross3(ac, ao), ac);
                } else {
                    simplex[0] = simplex[1];
                    simplex[1] = simplex[2];
                    size = 2;
                    dir = vector_math::cross3(vector_math::cross3(bc, bo), bc);
                }
            }
        } else if (size == 4) {
            hit = true;
        }
    }
    if (!hit) {
        // not intersecting
        return epaRes;
    }

    // Initialize polytope with the four simplex vertices
    std::vector<MinkowskiVertex> polytope(simplex, simplex + 4);
    // Build initial faces of the tetrahedron (indices)
    std::vector<std::vector<uint32_t>> faces;
    auto add_face = [&](uint32_t i0, uint32_t i1, uint32_t i2) {
        SimdVec a_pt = polytope[i0].ab;
        SimdVec b_pt = polytope[i1].ab;
        SimdVec c_pt = polytope[i2].ab;
        SimdVec norm = compute_face_normal(a_pt, b_pt, c_pt);
        faces.push_back({i0, i1, i2});
    };
    add_face(0, 1, 2);
    add_face(0, 2, 3);
    add_face(0, 3, 1);
    add_face(1, 3, 2);

    // EPA loop
    for (int iter = 0; iter < max_iter; ++iter) {
        // Find face with minimum distance to origin
        float min_dist = std::numeric_limits<float>::max();
        uint32_t min_face = 0;
        SimdVec min_normal;
        for (uint32_t fi = 0; fi < faces.size(); ++fi) {
            const auto& f = faces[fi];
            SimdVec a_pt = polytope[f[0]].ab;
            SimdVec b_pt = polytope[f[1]].ab;
            SimdVec c_pt = polytope[f[2]].ab;
            SimdVec norm = compute_face_normal(a_pt, b_pt, c_pt);
            float d = vector_math::dot3_scalar(norm, a_pt);
            if (d < min_dist) {
                min_dist = d;
                min_face = fi;
                min_normal = norm;
            }
        }

        // Support in direction of the closest face's normal
        MinkowskiVertex new_pt = minkowski_support(a, b, supportA, supportB, min_normal);
        float new_dist = vector_math::dot3_scalar(new_pt.ab, min_normal);
        if (std::abs(new_dist - min_dist) <= tolerance) {
            // converged
            epaRes.normal = min_normal;
            epaRes.depth = min_dist;
            // compute contact points by projecting origin onto the face
            const auto& f = faces[min_face];
            SimdVec bary = barycentric_3d(polytope[f[0]].ab, polytope[f[1]].ab, polytope[f[2]].ab,
                                          DirectX::XMVectorScale(min_normal, min_dist));
            float u = vector_math::get_x(bary);
            float v = vector_math::get_y(bary);
            float w = vector_math::get_z(bary);
            epaRes.contactA = DirectX::XMVectorAdd(
                DirectX::XMVectorAdd(
                    DirectX::XMVectorScale(polytope[f[0]].a, u),
                    DirectX::XMVectorScale(polytope[f[1]].a, v)),
                DirectX::XMVectorScale(polytope[f[2]].a, w));
            epaRes.contactB = DirectX::XMVectorAdd(
                DirectX::XMVectorAdd(
                    DirectX::XMVectorScale(polytope[f[0]].b, u),
                    DirectX::XMVectorScale(polytope[f[1]].b, v)),
                DirectX::XMVectorScale(polytope[f[2]].b, w));
            return epaRes;
        }

        // Add new vertex
        polytope.push_back(new_pt);
        uint32_t new_idx = (uint32_t)(polytope.size() - 1);

        // Remove faces that can "see" the new vertex
        std::vector<bool> removed(faces.size(), false);
        for (uint32_t fi = 0; fi < faces.size(); ++fi) {
            const auto& f = faces[fi];
            SimdVec a_pt = polytope[f[0]].ab;
            SimdVec b_pt = polytope[f[1]].ab;
            SimdVec c_pt = polytope[f[2]].ab;
            SimdVec norm = compute_face_normal(a_pt, b_pt, c_pt);
            float d = vector_math::dot3_scalar(norm, new_pt.ab) - vector_math::dot3_scalar(norm, a_pt);
            if (d > tolerance) {
                removed[fi] = true;
            }
        }

        // Build new faces by connecting the new vertex to each border edge
        std::vector<std::vector<uint32_t>> new_faces;
        // Keep non‑removed faces
        for (uint32_t fi = 0; fi < faces.size(); ++fi) {
            if (!removed[fi])
                new_faces.push_back(faces[fi]);
        }
        // For each removed face, for each edge, if the reversed edge is not in another removed face, create a new face with the new vertex.
        for (uint32_t fi = 0; fi < faces.size(); ++fi) {
            if (!removed[fi]) continue;
            const auto& f = faces[fi];
            for (int e = 0; e < 3; ++e) {
                uint32_t v0 = f[e];
                uint32_t v1 = f[(e + 1) % 3];
                // Check if the reversed edge appears in another removed face
                bool is_border = true;
                for (uint32_t fj = 0; fj < faces.size(); ++fj) {
                    if (fj == fi || !removed[fj]) continue;
                    const auto& of = faces[fj];
                    for (int oe = 0; oe < 3; ++oe) {
                        if (of[oe] == v1 && of[(oe + 1) % 3] == v0) {
                            is_border = false;
                            break;
                        }
                    }
                    if (!is_border) break;
                }
                if (is_border) {
                    // Add face (new_idx, v1, v0) – orientation maintains outward normal
                    new_faces.push_back({new_idx, v1, v0});
                }
            }
        }
        faces.swap(new_faces);
    }

    // Fallback: return the last best face
    epaRes.normal = min_normal;
    epaRes.depth = min_dist;
    return epaRes;
}

} // namespace collision
} // namespace SimulationMath

#endif // CORE_MATH_COLLISION_GJK_EPA_H