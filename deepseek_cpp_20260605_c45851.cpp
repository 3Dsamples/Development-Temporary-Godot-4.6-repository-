//File 0229 : sparse/xsparse_contact.hpp
//Constraint-based contact resolution with Lagrange multipliers, sparse KKT system assembly, and projected CG solver for non-penetration constraints.
#ifndef XTENSOR_XSPARSE_CONTACT_HPP
#define XTENSOR_XSPARSE_CONTACT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xarray.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xeval.hpp"
#include "../core/xnorm.hpp"
#include "../core/xbuilder.hpp"
#include "../sparse/xcsr.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xsparse_config.hpp"
#include "../sparse/xsparse_reducer.hpp"
#include "../sparse/xsparse_solver.hpp"
#include "../sparse/xsparse_lcp.hpp"

namespace xt {
namespace sparse {

    /**
     * @enum contact_detection_mode
     * @brief Types of contact detection supported.
     */
    enum class contact_detection_mode
    {
        node_to_plane,      // Nodes against a fixed plane
        node_to_sphere,     // Nodes against a fixed sphere
        node_to_node,       // Pairs of nodes with given radius
        edge_to_edge        // Edge-edge proximity (3D)
    };

    /**
     * @struct contact_constraint
     * @brief Describes a single active contact constraint.
     */
    template <class T>
    struct contact_constraint
    {
        std::size_t node_a;       // Primary node index
        std::size_t node_b;       // Secondary node index (or -1 for ground)
        T gap;                    // Signed gap distance (negative = penetration)
        T normal[3];              // Contact normal direction (from B toward A)
        T mu;                     // Friction coefficient (Coulomb)
        T compliance;             // Constraint compliance (inverse stiffness)
        bool active;              // Whether this constraint is currently active
    };

    namespace detail
    {
        /**
         * Compute the closest point on a triangle (v0,v1,v2) to a point p.
         * Returns barycentric coordinates (u, v, w) and the closest point.
         */
        template <class T>
        inline auto closest_point_triangle(
            const T* v0, const T* v1, const T* v2,
            const T* p)
        {
            T diff[3] = { v0[0] - p[0], v0[1] - p[1], v0[2] - p[2] };
            T edge0[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
            T edge1[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };
            T a00 = edge0[0]*edge0[0] + edge0[1]*edge0[1] + edge0[2]*edge0[2];
            T a01 = edge0[0]*edge1[0] + edge0[1]*edge1[1] + edge0[2]*edge1[2];
            T a11 = edge1[0]*edge1[0] + edge1[1]*edge1[1] + edge1[2]*edge1[2];
            T b0 = edge0[0]*diff[0] + edge0[1]*diff[1] + edge0[2]*diff[2];
            T b1 = edge1[0]*diff[0] + edge1[1]*diff[1] + edge1[2]*diff[2];
            T det = a00 * a11 - a01 * a01;
            T u = (a11 * b0 - a01 * b1) / std::max(det, T(1e-15));
            T v = (a00 * b1 - a01 * b0) / std::max(det, T(1e-15));
            u = std::max(T(0), std::min(T(1), u));
            v = std::max(T(0), std::min(T(1), v));
            T w = T(1) - u - v;
            T closest[3] = {
                v0[0] + u*edge0[0] + v*edge1[0],
                v0[1] + u*edge0[1] + v*edge1[1],
                v0[2] + u*edge0[2] + v*edge1[2]
            };
            T dist = std::sqrt(
                (closest[0]-p[0])*(closest[0]-p[0]) +
                (closest[1]-p[1])*(closest[1]-p[1]) +
                (closest[2]-p[2])*(closest[2]-p[2])
            );
            return std::make_tuple(u, v, w, closest[0], closest[1], closest[2], dist);
        }

        /**
         * Assemble the sparse KKT system for constrained dynamics:
         * [ M + h²K    J^T ] [ Δv ]   [ f ]
         * [    J      -C   ] [ λ  ] = [ c ]
         * where J is the constraint Jacobian and C is the compliance matrix.
         */
        template <class T>
        inline auto assemble_kkt_system(
            const xcsr_matrix<T>& A_free,
            const std::vector<contact_constraint<T>>& constraints,
            T dt)
        {
            std::size_t n_free = A_free.rows();
            std::size_t n_constraints = 0;
            for (auto& c : constraints) if (c.active) n_constraints++;
            std::size_t total = n_free + n_constraints;
            xcoo_matrix<T> KKT(total, total);
            // Copy free system into upper-left block
            for (std::size_t r = 0; r < A_free.rows(); ++r)
                for (std::size_t j = A_free.row_ptr()[r]; j < A_free.row_ptr()[r + 1]; ++j)
                    KKT.append(r, A_free.col_idx()[j], A_free.values()[j]);
            // Add constraint blocks
            std::size_t con_idx = 0;
            for (const auto& c : constraints)
            {
                if (!c.active) continue;
                std::size_t row_con = n_free + con_idx;
                // J block: row = constraint index, columns = dofs of involved nodes
                std::size_t dofs[6] = { c.node_a*3, c.node_a*3+1, c.node_a*3+2,
                                        c.node_b*3, c.node_b*3+1, c.node_b*3+2 };
                T sign = T(-1); // constraint: n·(v_a - v_b) >= 0, so J row: n for a, -n for b
                for (int d = 0; d < 3; ++d)
                {
                    KKT.append(row_con, dofs[d], sign * c.normal[d]);
                    if (c.node_b != static_cast<std::size_t>(-1))
                        KKT.append(row_con, dofs[3+d], -sign * c.normal[d]);
                }
                // J^T block: transpose of J
                for (int d = 0; d < 3; ++d)
                {
                    KKT.append(dofs[d], row_con, sign * c.normal[d]);
                    if (c.node_b != static_cast<std::size_t>(-1))
                        KKT.append(dofs[3+d], row_con, -sign * c.normal[d]);
                }
                // Compliance diagonal
                KKT.append(row_con, row_con, -c.compliance);
                con_idx++;
            }
            return xcsr_matrix<T>::from_coo(KKT);
        }
    }

    /**
     * Detect node-to-plane contacts and populate the constraint list.
     * @param positions Node positions (N x 3).
     * @param radii Node radii for contact detection.
     * @param plane_normal Plane normal (unit vector).
     * @param plane_offset Signed distance of plane from origin along normal.
     * @param constraints Output list, appended with detected contacts.
     */
    template <class T>
    inline void detect_node_plane_contacts(
        const xarray_container<uvector<T>>& positions,
        const xarray_container<uvector<T>>& radii,
        const T* plane_normal,
        T plane_offset,
        std::vector<contact_constraint<T>>& constraints)
    {
        std::size_t N = positions.shape()[0];
        for (std::size_t i = 0; i < N; ++i)
        {
            T d = positions(i,0)*plane_normal[0] + positions(i,1)*plane_normal[1] +
                  positions(i,2)*plane_normal[2] - plane_offset;
            T gap = d - radii[i];
            if (gap < T(0))
            {
                contact_constraint<T> c;
                c.node_a = i;
                c.node_b = static_cast<std::size_t>(-1);
                c.gap = gap;
                c.normal[0] = plane_normal[0];
                c.normal[1] = plane_normal[1];
                c.normal[2] = plane_normal[2];
                c.mu = T(0.5);
                c.compliance = T(1e-6);
                c.active = true;
                constraints.push_back(c);
            }
        }
    }

    /**
     * Detect node-to-node contacts based on proximity.
     * Uses O(N²) brute force; for large N use spatial hashing (not implemented here).
     */
    template <class T>
    inline void detect_node_node_contacts(
        const xarray_container<uvector<T>>& positions,
        const xarray_container<uvector<T>>& radii,
        std::vector<contact_constraint<T>>& constraints)
    {
        std::size_t N = positions.shape()[0];
        for (std::size_t i = 0; i < N; ++i)
        {
            for (std::size_t j = i + 1; j < N; ++j)
            {
                T dx = positions(j,0) - positions(i,0);
                T dy = positions(j,1) - positions(i,1);
                T dz = positions(j,2) - positions(i,2);
                T dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                T gap = dist - (radii[i] + radii[j]);
                if (gap < T(0) && dist > T(1e-15))
                {
                    contact_constraint<T> c;
                    c.node_a = i;
                    c.node_b = j;
                    c.gap = gap;
                    T inv_dist = T(1) / dist;
                    c.normal[0] = dx * inv_dist;
                    c.normal[1] = dy * inv_dist;
                    c.normal[2] = dz * inv_dist;
                    c.mu = T(0.3);
                    c.compliance = T(1e-6);
                    c.active = true;
                    constraints.push_back(c);
                }
            }
        }
    }

    /**
     * Solve the constrained dynamics step using the KKT approach.
     * @param A_free The unconstrained system matrix (mass + stiffness).
     * @param rhs_free The unconstrained right-hand side.
     * @param constraints List of active contact constraints.
     * @param dt Time step size.
     * @return Pair (velocity increment Δv, Lagrange multipliers λ).
     */
    template <class T>
    inline auto solve_contact_kkt(
        const xcsr_matrix<T>& A_free,
        const xarray_container<uvector<T>>& rhs_free,
        std::vector<contact_constraint<T>>& constraints,
        T dt)
    {
        std::size_t n_free = A_free.rows();
        std::size_t n_active = 0;
        for (auto& c : constraints) if (c.active) n_active++;
        std::size_t total = n_free + n_active;
        // Assemble KKT system
        auto KKT = detail::assemble_kkt_system(A_free, constraints, dt);
        // Assemble right-hand side
        xarray_container<uvector<T>> rhs_total({total}, T(0));
        // Copy free RHS
        for (std::size_t i = 0; i < n_free; ++i)
            rhs_total[i] = rhs_free[i];
        // Constraint RHS: Baumgarte stabilization term
        std::size_t con_idx = 0;
        T baumgarte_factor = T(0.1) / dt;
        for (const auto& c : constraints)
        {
            if (!c.active) continue;
            rhs_total[n_free + con_idx] = baumgarte_factor * c.gap;
            con_idx++;
        }
        // Solve using sparse CG with diagonal preconditioner
        auto solution = cg_solve(KKT, rhs_total, T(1e-6), 1000, preconditioner_type::diagonal);
        // Split solution into Δv and λ
        xarray_container<uvector<T>> dv({n_free});
        xarray_container<uvector<T>> lambda({n_active});
        for (std::size_t i = 0; i < n_free; ++i)
            dv[i] = solution[i];
        for (std::size_t i = 0; i < n_active; ++i)
            lambda[i] = solution[n_free + i];
        // Enforce complementarity: λ >= 0, if negative, remove constraint and re-solve? For simplicity, clamp.
        for (std::size_t i = 0; i < n_active; ++i)
            if (lambda[i] < T(0))
                lambda[i] = T(0);
        return std::make_pair(dv, lambda);
    }

    /**
     * Apply Coulomb friction forces based on contact constraints and Lagrange multipliers.
     * @param velocity Current velocity vector.
     * @param constraints List of active contacts.
     * @param lambda Lagrange multipliers (normal forces).
     * @param dt Time step.
     * @return Friction force vector.
     */
    template <class T>
    inline auto apply_coulomb_friction(
        const xarray_container<uvector<T>>& velocity,
        const std::vector<contact_constraint<T>>& constraints,
        const xarray_container<uvector<T>>& lambda,
        T dt)
    {
        std::size_t n_dofs = velocity.size();
        xarray_container<uvector<T>> f_friction({n_dofs}, T(0));
        T* fric_data = f_friction.data();
        const T* vel_data = velocity.data();
        std::size_t con_idx = 0;
        for (const auto& c : constraints)
        {
            if (!c.active) continue;
            T lambda_n = lambda[con_idx];
            if (lambda_n <= T(0)) { con_idx++; continue; }
            // Compute relative tangential velocity at contact
            T vel_rel[3];
            for (int d = 0; d < 3; ++d)
                vel_rel[d] = vel_data[c.node_a*3 + d];
            if (c.node_b != static_cast<std::size_t>(-1))
                for (int d = 0; d < 3; ++d)
                    vel_rel[d] -= vel_data[c.node_b*3 + d];
            // Tangential component
            T vn = vel_rel[0]*c.normal[0] + vel_rel[1]*c.normal[1] + vel_rel[2]*c.normal[2];
            T v_tang[3];
            for (int d = 0; d < 3; ++d) v_tang[d] = vel_rel[d] - vn * c.normal[d];
            T vt_norm = std::sqrt(v_tang[0]*v_tang[0] + v_tang[1]*v_tang[1] + v_tang[2]*v_tang[2]);
            T max_friction = c.mu * lambda_n;
            if (vt_norm > T(1e-15))
            {
                T friction_mag = std::min(max_friction, vt_norm / dt);
                for (int d = 0; d < 3; ++d)
                {
                    T f_dir = -v_tang[d] / vt_norm * friction_mag;
                    fric_data[c.node_a*3 + d] += f_dir;
                    if (c.node_b != static_cast<std::size_t>(-1))
                        fric_data[c.node_b*3 + d] -= f_dir;
                }
            }
            con_idx++;
        }
        return f_friction;
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_CONTACT_HPP