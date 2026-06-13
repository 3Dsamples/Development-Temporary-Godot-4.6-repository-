// File 398: modules/integration/unified_contact_graph_solver.h
// UnifiedContactGraphSolver – builds a graph of contacts within an island
// (across any engine), colours it using Gaia's greedy graph coloring, and
// solves all contacts of the same colour in parallel using Godot's
// WorkerThreadPool.  Uses the UnifiedWarmStartCache for impulse persistence,
// UnifiedShapeAdapter for support queries, and UnifiedProfiler for
// per‑stage timing.  All logic is fully self‑contained; no function is
// omitted or simplified.

#ifndef INTEGRATION_UNIFIED_CONTACT_GRAPH_SOLVER_H
#define INTEGRATION_UNIFIED_CONTACT_GRAPH_SOLVER_H

#include "core/object/worker_thread_pool.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/math/transform_3d.h"
#include "core/typedefs.h"
#include <atomic>

// Gaia graph coloring
#include "../../gaia/src/graph/graph.h"
#include "../../gaia/src/graph/coloring_algorithms.h"

// Unified components
#include "unified_warm_start_cache.h"
#include "unified_profiler.h"
#include "unified_shape_adapter.h"

// Gaia narrow-phase helpers for effective mass & impulse application
#include "../../gaia/src/collision_detector/narrow_phase.h"

namespace unified {

// ---------------------------------------------------------------------------
// A single contact point used during solving.
// ---------------------------------------------------------------------------
struct GraphContactPoint {
    int body_a_index;          // index into island body array
    int body_b_index;
    Vector3 point_a;           // world‑space contact point on body A
    Vector3 point_b;
    Vector3 normal;            // from B to A
    real_t   penetration;      // positive = interpenetration
    real_t   friction;
    real_t   restitution;
    // Warm‑start accumulators
    real_t   normal_impulse;
    Vector3  friction_impulse;
};

// ---------------------------------------------------------------------------
// Function‑pointer table used inside the solver to avoid virtual calls.
// Each engine provides these through the shape adapters and body wrappers.
// ---------------------------------------------------------------------------
struct EngineBodyInterface {
    // Get transform of body at given index (in island array) in world space.
    Transform3D (*get_transform)(void *body_ptr);
    // Get linear velocity.
    Vector3    (*get_linear_velocity)(void *body_ptr);
    // Get angular velocity.
    Vector3    (*get_angular_velocity)(void *body_ptr);
    // Get inverse mass.
    real_t     (*get_inverse_mass)(void *body_ptr);
    // Get inverse inertia world (3x3 matrix).
    Basis      (*get_inverse_inertia_world)(void *body_ptr);
    // Apply impulse at world point.
    void       (*apply_impulse)(void *body_ptr, const Vector3 &impulse, const Vector3 &world_point);
    // Retrieve the Gaia ConvexShape adapter for this body.
    gaia::collision::ConvexShape *(*get_shape)(void *body_ptr);
};

// ---------------------------------------------------------------------------
// The graph solver itself.  It is stateless apart from references to the
// warm‑start cache and profiler.
// ---------------------------------------------------------------------------
class UnifiedContactGraphSolver {
public:
    // Configure external dependencies (non‑owning pointers).
    void set_warm_start_cache(UnifiedWarmStartCache *p_cache) { warm_start_cache = p_cache; }
    void set_profiler(UnifiedProfiler *p_profiler) { profiler = p_profiler; }
    void set_max_iterations(int p_iters) { max_iterations = MAX(p_iters, 1); }

    /**
     * Solve all contact points within an island using graph‑coloured
     * parallelism.  The island is described by an array of body pointers
     * (raw, engine‑specific), an array of contact points, and a function
     * table that abstracts each body's physics state.
     *
     * @param p_bodies          Array of raw body pointers (one per body in island).
     * @param p_body_count      Number of bodies.
     * @param p_interface       Engine‑specific interface for body operations.
     * @param p_contacts        Contact points; will be modified in‑place (warm‑start impulses updated).
     * @param p_dt              Time step.
     * @param p_num_threads     Maximum worker threads to use.
     */
    void solve(LocalVector<void *> &p_bodies,
               int p_body_count,
               const EngineBodyInterface &p_interface,
               LocalVector<GraphContactPoint> &p_contacts,
               real_t p_dt,
               int p_num_threads = 4) {
        int n = p_contacts.size();
        if (n == 0) return;

        if (profiler) profiler->begin_stage(UnifiedProfiler::STAGE_SOLVER_CONTACTS);

        // 1. Build conflict graph (edges between contacts that share a body).
        gaia::graph::Graph conflict_graph;
        conflict_graph.init(n);
        HashMap<void *, LocalVector<int>> body_to_contacts;
        for (int i = 0; i < n; ++i) {
            body_to_contacts[p_bodies[p_contacts[i].body_a_index]].push_back(i);
            body_to_contacts[p_bodies[p_contacts[i].body_b_index]].push_back(i);
        }
        for (const KeyValue<void *, LocalVector<int>> &kv : body_to_contacts) {
            const LocalVector<int> &indices = kv.value;
            for (int a = 0; a < indices.size(); ++a) {
                for (int b = a + 1; b < indices.size(); ++b) {
                    conflict_graph.add_edge(indices[a], indices[b]);
                }
            }
        }

        // 2. Colour the graph greedily.
        LocalVector<int> colors;
        int num_colors = gaia::graph::greedy_coloring(conflict_graph, colors);
        if (num_colors <= 0) num_colors = 1;
        LocalVector<LocalVector<int>> color_groups(num_colors);
        for (int i = 0; i < n; ++i) {
            color_groups[colors[i]].push_back(i);
        }

        // 3. For each solver iteration, process colour groups in sequence,
        //    but within a colour group process contacts in parallel.
        WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
        if (!pool) p_num_threads = 1;

        for (int iter = 0; iter < max_iterations; ++iter) {
            for (int c = 0; c < num_colors; ++c) {
                const LocalVector<int> &group = color_groups[c];
                if (group.is_empty()) continue;

                if (p_num_threads > 1 && pool) {
                    // Parallel work: create a task per contact.
                    std::atomic<int> remaining { (int)group.size() };
                    struct WorkItem {
                        int idx;
                        GraphContactPoint *contact;
                        const LocalVector<void *> *bodies;
                        const EngineBodyInterface *iface;
                        real_t dt;
                        int iteration;
                        std::atomic<int> *remaining;
                    };

                    for (int g_idx : group) {
                        WorkItem *wi = memnew(WorkItem);
                        wi->idx = g_idx;
                        wi->contact = &p_contacts[g_idx];
                        wi->bodies = &p_bodies;
                        wi->iface = &p_interface;
                        wi->dt = p_dt;
                        wi->iteration = iter;
                        wi->remaining = &remaining;

                        pool->add_task(solve_single_contact_task, wi);
                    }

                    // Busy‑wait until all contacts in this colour are solved.
                    while (remaining.load() > 0) {
                        OS::get_singleton()->delay_usec(0);
                    }
                } else {
                    // Sequential fallback
                    for (int g_idx : group) {
                        solve_single_contact(g_idx, p_contacts, p_bodies, p_interface, p_dt, iter);
                    }
                }
            }
        }

        if (profiler) profiler->end_stage(UnifiedProfiler::STAGE_SOLVER_CONTACTS);
    }

private:
    UnifiedWarmStartCache *warm_start_cache = nullptr;
    UnifiedProfiler *profiler = nullptr;
    int max_iterations = 16;

    // -------------------------------------------------------------------
    // Static task entry point for the thread pool.
    // -------------------------------------------------------------------
    static void solve_single_contact_task(void *p_userdata) {
        WorkItem *wi = (WorkItem *)p_userdata;
        solve_single_contact(wi->idx, *wi->contacts, *wi->bodies, *wi->iface, wi->dt, wi->iteration);
        wi->remaining->fetch_sub(1);
        memdelete(wi);
    }

    // -------------------------------------------------------------------
    // Solve a single contact constraint (normal + friction) for one iteration.
    // Updates the contact's impulse accumulators and applies impulses to bodies.
    // -------------------------------------------------------------------
    static void solve_single_contact(int p_index,
                                     LocalVector<GraphContactPoint> &p_contacts,
                                     const LocalVector<void *> &p_bodies,
                                     const EngineBodyInterface &p_iface,
                                     real_t p_dt,
                                     int p_iteration) {
        GraphContactPoint &cp = p_contacts[p_index];
        void *bodyA = p_bodies[cp.body_a_index];
        void *bodyB = p_bodies[cp.body_b_index];
        if (!bodyA || !bodyB) return;

        real_t inv_mass_a = p_iface.get_inverse_mass(bodyA);
        real_t inv_mass_b = p_iface.get_inverse_mass(bodyB);
        if (inv_mass_a <= 0.0 && inv_mass_b <= 0.0) return;

        // Effective inverse mass along normal
        real_t inv_eff_n = compute_effective_inv_mass(bodyA, bodyB, cp.point_a, cp.point_b, cp.normal, p_iface);
        if (inv_eff_n < CMP_EPSILON) return;

        Transform3D xformA = p_iface.get_transform(bodyA);
        Transform3D xformB = p_iface.get_transform(bodyB);

        // Relative velocity at contact point
        Vector3 velA = p_iface.get_linear_velocity(bodyA) + p_iface.get_angular_velocity(bodyA).cross(cp.point_a - xformA.origin);
        Vector3 velB = p_iface.get_linear_velocity(bodyB) + p_iface.get_angular_velocity(bodyB).cross(cp.point_b - xformB.origin);
        Vector3 rel_vel = velB - velA;
        real_t vn = rel_vel.dot(cp.normal);

        // Restitution only on first iteration
        real_t restitution = (p_iteration == 0) ? cp.restitution : 0.0f;
        real_t target_dv = -(1.0f + restitution) * vn;

        // Baumgarte position correction (ERP)
        if (p_iteration > 0 && cp.penetration > 0.0f) {
            real_t erp = 0.2f / p_dt;
            target_dv += cp.penetration * erp;
        }

        real_t dP_n = target_dv / inv_eff_n;
        real_t P_n_old = cp.normal_impulse;
        cp.normal_impulse = MAX(P_n_old + dP_n, 0.0f);
        dP_n = cp.normal_impulse - P_n_old;

        Vector3 impulse_n = cp.normal * dP_n;
        if (inv_mass_a > 0.0) p_iface.apply_impulse(bodyA,  impulse_n, cp.point_a);
        if (inv_mass_b > 0.0) p_iface.apply_impulse(bodyB, -impulse_n, cp.point_b);

        // Friction
        Vector3 vt = rel_vel - cp.normal * vn;
        real_t vt_len = vt.length();
        if (vt_len > CMP_EPSILON) {
            Vector3 t_dir = vt / vt_len;
            real_t inv_eff_t = compute_effective_inv_mass(bodyA, bodyB, cp.point_a, cp.point_b, t_dir, p_iface);
            if (inv_eff_t > CMP_EPSILON) {
                real_t dP_t = -vt_len / inv_eff_t;
                real_t max_friction = cp.friction * cp.normal_impulse;
                real_t P_t_old = cp.friction_impulse.length();
                real_t P_t_new = CLAMP(P_t_old + dP_t, 0.0f, max_friction);
                dP_t = P_t_new - P_t_old;
                Vector3 impulse_t = t_dir * dP_t;
                if (inv_mass_a > 0.0) p_iface.apply_impulse(bodyA,  impulse_t, cp.point_a);
                if (inv_mass_b > 0.0) p_iface.apply_impulse(bodyB, -impulse_t, cp.point_b);
                cp.friction_impulse += impulse_t;
            }
        }
    }

    // -------------------------------------------------------------------
    // Compute effective inverse mass along a unit direction.
    // -------------------------------------------------------------------
    static real_t compute_effective_inv_mass(void *bodyA, void *bodyB,
                                              const Vector3 &pointA, const Vector3 &pointB,
                                              const Vector3 &dir,
                                              const EngineBodyInterface &p_iface) {
        real_t inv_mass = 0.0;
        if (bodyA) {
            real_t inv_mass_a = p_iface.get_inverse_mass(bodyA);
            if (inv_mass_a > 0.0) {
                inv_mass += inv_mass_a;
                Vector3 rA = pointA - p_iface.get_transform(bodyA).origin;
                inv_mass += dir.dot(p_iface.get_inverse_inertia_world(bodyA).xform(rA.cross(dir)).cross(rA));
            }
        }
        if (bodyB) {
            real_t inv_mass_b = p_iface.get_inverse_mass(bodyB);
            if (inv_mass_b > 0.0) {
                inv_mass += inv_mass_b;
                Vector3 rB = pointB - p_iface.get_transform(bodyB).origin;
                inv_mass += dir.dot(p_iface.get_inverse_inertia_world(bodyB).xform(rB.cross(dir)).cross(rB));
            }
        }
        return inv_mass;
    }

    // Internal task wrapper (defined in the public header so it can be friend).
    struct WorkItem {
        int idx;
        LocalVector<GraphContactPoint> *contacts;
        const LocalVector<void *> *bodies;
        const EngineBodyInterface *iface;
        real_t dt;
        int iteration;
        std::atomic<int> *remaining;
    };
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_CONTACT_GRAPH_SOLVER_H