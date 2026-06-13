//File 0230 : sparse/xsparse_topology.hpp
//Topology optimization using sparse FEM, SIMD sensitivity filtering, and optimality criteria for minimum compliance with volume constraints.
#ifndef XTENSOR_XSPARSE_TOPOLOGY_HPP
#define XTENSOR_XSPARSE_TOPOLOGY_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
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
#include "../core/xmanipulation.hpp"
#include "../core/xsort.hpp"
#include "../sparse/xcsr.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xsparse_config.hpp"
#include "../sparse/xsparse_reducer.hpp"
#include "../sparse/xsparse_solver.hpp"
#include "../sparse/xsparse_mesh.hpp"

namespace xt {
namespace sparse {

    /**
     * @enum topology_filter_type
     * @brief Available sensitivity filtering strategies.
     */
    enum class topology_filter_type
    {
        none,              // No filtering (prone to checkerboard)
        density_filter,    // Filter on design variables directly
        sensitivity_filter // Filter on sensitivities before update
    };

    /**
     * @enum topology_penalty_type
     * @brief Material interpolation schemes for SIMP.
     */
    enum class topology_penalty_type
    {
        simp,              // Solid Isotropic Material with Penalization: E(x) = x^p
        ramp               // Rational Approximation of Material Properties: E(x) = x/(1+q*(1-x))
    };

    namespace detail
    {
        /**
         * Compute the element stiffness matrix scaled by SIMP material model.
         * E_e = E_min + x_e^penalty * (E_0 - E_min)
         * Returns the scaled local stiffness matrix.
         */
        template <class T>
        inline auto simp_element_stiffness(
            T x_e, T penalty, T young, T poisson,
            T x0, T y0, T x1, T y1, T x2, T y2)
        {
            T E_eff = T(1e-9) + std::pow(x_e, penalty) * (young - T(1e-9));
            return triangle_stiffness(x0, y0, x1, y1, x2, y2, E_eff, poisson);
        }

        /**
         * Assemble the global stiffness matrix with SIMP interpolation.
         * Each element stiffness is scaled by x_e^penalty.
         */
        template <class T>
        inline auto assemble_simp_stiffness(
            const xarray_container<uvector<T>>& nodes,
            const xarray_container<uvector<std::size_t>>& elements,
            const xarray_container<uvector<T>>& x,
            T penalty, T young, T poisson)
        {
            std::size_t N = nodes.shape()[0];
            std::size_t M = elements.shape()[0];
            xcoo_matrix<T> coo(N, N);
            for (std::size_t e = 0; e < M; ++e)
            {
                std::size_t n0 = elements(e, 0), n1 = elements(e, 1), n2 = elements(e, 2);
                T xe = x[e];
                T x0 = nodes(n0, 0), y0 = nodes(n0, 1);
                T x1 = nodes(n1, 0), y1 = nodes(n1, 1);
                T x2 = nodes(n2, 0), y2 = nodes(n2, 1);
                auto Ke = simp_element_stiffness(xe, penalty, young, poisson, x0, y0, x1, y1, x2, y2);
                std::size_t dofs[3] = { n0, n1, n2 };
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        coo.append(dofs[i], dofs[j], Ke(i, j));
            }
            return xcsr_matrix<T>::from_coo(coo);
        }

        /**
         * Build the density filter matrix H where H(i,j) = max(0, r_min - dist(i,j)).
         * Normalized such that each row sums to 1.
         */
        template <class T>
        inline auto build_density_filter(
            const xarray_container<uvector<T>>& element_centroids,
            T r_min)
        {
            std::size_t M = element_centroids.shape()[0];
            xcoo_matrix<T> H_coo(M, M);
            for (std::size_t i = 0; i < M; ++i)
            {
                T sum = T(0);
                std::vector<T> weights(M, T(0));
                for (std::size_t j = 0; j < M; ++j)
                {
                    T dx = element_centroids(i, 0) - element_centroids(j, 0);
                    T dy = element_centroids(i, 1) - element_centroids(j, 1);
                    T dist = std::sqrt(dx*dx + dy*dy);
                    T w = std::max(T(0), r_min - dist);
                    weights[j] = w;
                    sum += w;
                }
                if (sum > T(0))
                {
                    for (std::size_t j = 0; j < M; ++j)
                        if (weights[j] > T(0))
                            H_coo.append(i, j, weights[j] / sum);
                }
                else
                {
                    H_coo.append(i, i, T(1)); // self
                }
            }
            return xcsr_matrix<T>::from_coo(H_coo);
        }

        /**
         * Compute element centroids from node positions and element connectivity.
         */
        template <class T>
        inline auto compute_element_centroids(
            const xarray_container<uvector<T>>& nodes,
            const xarray_container<uvector<std::size_t>>& elements)
        {
            std::size_t M = elements.shape()[0];
            xarray_container<uvector<T>> centroids({M, 2});
            for (std::size_t e = 0; e < M; ++e)
            {
                T cx = T(0), cy = T(0);
                for (std::size_t k = 0; k < elements.shape()[1]; ++k)
                {
                    cx += nodes(elements(e, k), 0);
                    cy += nodes(elements(e, k), 1);
                }
                centroids(e, 0) = cx / T(elements.shape()[1]);
                centroids(e, 1) = cy / T(elements.shape()[1]);
            }
            return centroids;
        }
    }

    /**
     * @class topology_optimizer
     * @brief Minimum compliance topology optimization using SIMP and optimality criteria.
     *
     * Solves: min c(x) = U^T * K * U  s.t. V(x)/V_0 ≤ f, 0 < x_min ≤ x_e ≤ 1
     * where K depends on x via SIMP (K = ∑ x_e^p * K_e^0).
     */
    template <class T>
    class topology_optimizer
    {
    public:
        using vector_type = xarray_container<uvector<T>>;
        using matrix_type = xcsr_matrix<T>;
        using index_array = xarray_container<uvector<std::size_t>>;

        /**
         * Initialize the optimizer with a mesh and boundary conditions.
         * @param nodes Node coordinates (N x 2).
         * @param elements Triangle connectivity (M x 3).
         * @param fixed_dofs Indices of constrained DOFs (Dirichlet BC).
         * @param force_dofs Indices where external forces are applied.
         * @param force_values Values of the applied forces.
         * @param young Young's modulus of solid material.
         * @param poisson Poisson's ratio.
         */
        topology_optimizer(const vector_type& nodes,
                           const index_array& elements,
                           const std::vector<std::size_t>& fixed_dofs,
                           const std::vector<std::size_t>& force_dofs,
                           const vector_type& force_values,
                           T young, T poisson)
            : m_nodes(nodes)
            , m_elements(elements)
            , m_fixed_dofs(fixed_dofs)
            , m_force_dofs(force_dofs)
            , m_force_values(force_values)
            , m_young(young)
            , m_poisson(poisson)
            , m_penalty(T(3.0))
            , m_volume_fraction(T(0.4))
            , m_filter_radius(T(1.5))
            , m_filter_type(topology_filter_type::sensitivity_filter)
            , m_move_limit(T(0.2))
            , m_x_min(T(0.001))
            , m_max_iter(100)
            , m_tol(T(0.01))
        {
            std::size_t N = nodes.shape()[0];
            std::size_t M = elements.shape()[0];
            m_density = vector_type({M}, m_volume_fraction);
            m_ndof = N * 2;
            // Precompute element centroids for filtering
            m_centroids = detail::compute_element_centroids(nodes, elements);
            if (m_filter_type != topology_filter_type::none)
                m_H = detail::build_density_filter(m_centroids, m_filter_radius);
            // Assemble reference element stiffness for unit Young's modulus (will be scaled by SIMP)
            m_K0 = assemble_stiffness_tri2d(nodes, elements, T(1), poisson);
        }

        /**
         * Set the SIMP penalty exponent (typical range 1‑5).
         */
        void set_penalty(T p) noexcept { m_penalty = p; }

        /**
         * Set the target volume fraction.
         */
        void set_volume_fraction(T f) noexcept { m_volume_fraction = f; }

        /**
         * Set the filter radius (relative to element size).
         */
        void set_filter_radius(T r) noexcept { m_filter_radius = r; }

        /**
         * Set the maximum number of iterations.
         */
        void set_max_iterations(std::size_t n) noexcept { m_max_iter = n; }

        /**
         * Set the convergence tolerance (relative change in design variables).
         */
        void set_tolerance(T tol) noexcept { m_tol = tol; }

        /**
         * Set the move limit for the OC updater.
         */
        void set_move_limit(T m) noexcept { m_move_limit = m; }

        /**
         * Run the topology optimization.
         * @return The optimized density field (element‑wise).
         */
        vector_type optimize()
        {
            std::size_t M = m_elements.shape()[0];
            vector_type x = m_density;
            vector_type x_old = x;
            T change = std::numeric_limits<T>::max();

            for (std::size_t iter = 0; iter < m_max_iter && change > m_tol; ++iter)
            {
                // Assemble global stiffness with current densities
                auto K = detail::assemble_simp_stiffness(m_nodes, m_elements, x, m_penalty, m_young, m_poisson);

                // Solve K * U = F with Dirichlet BC
                auto U = solve_fem(K);

                // Compute compliance and sensitivities
                vector_type compliance_sens({M}, T(0));
                compute_sensitivities(K, U, x, compliance_sens);

                // Apply sensitivity filter
                if (m_filter_type == topology_filter_type::sensitivity_filter)
                {
                    compliance_sens = apply_sensitivity_filter(compliance_sens, x);
                }
                else if (m_filter_type == topology_filter_type::density_filter)
                {
                    compliance_sens = apply_density_filter(compliance_sens, x);
                }

                // Update design variables using optimality criteria
                x_old = x;
                oc_update(compliance_sens, x);
                // Apply density filter to the updated design
                if (m_filter_type == topology_filter_type::density_filter)
                {
                    x = m_H.dot(x);
                }
                // Clamp to bounds
                for (std::size_t e = 0; e < M; ++e)
                    x[e] = std::max(m_x_min, std::min(T(1), x[e]));

                // Compute change
                change = T(0);
                for (std::size_t e = 0; e < M; ++e)
                    change = std::max(change, std::abs(x[e] - x_old[e]));

                m_compliance = compute_compliance(K, U);
            }
            m_density = x;
            return m_density;
        }

        /**
         * Get the final compliance value.
         */
        T compliance() const noexcept { return m_compliance; }

        /**
         * Get the current density field.
         */
        const vector_type& density() const noexcept { return m_density; }

    private:
        vector_type m_nodes;
        index_array m_elements;
        std::vector<std::size_t> m_fixed_dofs;
        std::vector<std::size_t> m_force_dofs;
        vector_type m_force_values;
        T m_young, m_poisson;
        T m_penalty, m_volume_fraction, m_filter_radius;
        topology_filter_type m_filter_type;
        T m_move_limit, m_x_min;
        std::size_t m_max_iter;
        T m_tol;
        std::size_t m_ndof;
        vector_type m_density;
        vector_type m_centroids;
        matrix_type m_K0;
        matrix_type m_H;
        T m_compliance = T(0);

        /**
         * Solve the linear system K*U = F applying Dirichlet boundary conditions.
         * Fixed DOFs are eliminated (set to 0 in solution, modify RHS).
         */
        vector_type solve_fem(const matrix_type& K)
        {
            vector_type U({m_ndof}, T(0));
            vector_type F({m_ndof}, T(0));
            for (std::size_t i = 0; i < m_force_dofs.size(); ++i)
                F[m_force_dofs[i]] = m_force_values[i];
            // Eliminate fixed DOFs: set corresponding rows to zero and diagonal to 1, RHS to 0
            matrix_type K_mod = K;
            vector_type F_mod = F;
            for (auto dof : m_fixed_dofs)
            {
                F_mod[dof] = T(0);
                // Set row to identity: this is expensive in CSR; we'll simply use penalty method.
                // For proper sparse handling, we rebuild K_mod with modified rows.
                // Here we use a dense approach for small problems; for large, use proper sparse BC application.
            }
            // Since modifying CSR rows is complex, we convert to dense for BC application (only for small problems).
            auto K_dense = to_dense(K_mod);
            // Apply BC: set rows/cols of fixed DOFs to identity
            for (auto dof : m_fixed_dofs)
            {
                for (std::size_t j = 0; j < m_ndof; ++j)
                {
                    K_dense(dof, j) = T(0);
                    K_dense(j, dof) = T(0);
                }
                K_dense(dof, dof) = T(1);
                F_mod[dof] = T(0);
            }
            // Solve with conjugate gradient
            U = xt::linalg::solve(K_dense, F_mod);
            return U;
        }

        /**
         * Compute element sensitivities: ∂c/∂x_e = -p * x_e^{p-1} * U_e^T * K_e^0 * U_e.
         */
        void compute_sensitivities(const matrix_type& K, const vector_type& U,
                                    const vector_type& x, vector_type& sens)
        {
            std::size_t M = m_elements.shape()[0];
            for (std::size_t e = 0; e < M; ++e)
            {
                std::size_t n0 = m_elements(e, 0), n1 = m_elements(e, 1), n2 = m_elements(e, 2);
                T xe = x[e];
                T factor = -m_penalty * std::pow(xe, m_penalty - T(1));
                T u_e[6] = { U[2*n0], U[2*n0+1], U[2*n1], U[2*n1+1], U[2*n2], U[2*n2+1] };
                T uKu = T(0);
                for (std::size_t i = m_K0.row_ptr()[2*n0]; i < m_K0.row_ptr()[2*n0+1]; ++i)
                {
                    std::size_t col = m_K0.col_idx()[i];
                    T kval = m_K0.values()[i];
                    uKu += u_e[0] * kval * U[col];
                }
                sens[e] = -factor * uKu;
            }
        }

        /**
         * Apply sensitivity filtering: modified sensitivities using weighted average.
         */
        vector_type apply_sensitivity_filter(const vector_type& sens, const vector_type& x)
        {
            std::size_t M = sens.size();
            vector_type filtered({M}, T(0));
            for (std::size_t e = 0; e < M; ++e)
            {
                T numerator = T(0);
                T denominator = T(0);
                for (std::size_t j = 0; j < M; ++j)
                {
                    T w = std::max(T(0), m_filter_radius - std::sqrt(
                        (m_centroids(e,0)-m_centroids(j,0))*(m_centroids(e,0)-m_centroids(j,0)) +
                        (m_centroids(e,1)-m_centroids(j,1))*(m_centroids(e,1)-m_centroids(j,1))
                    ));
                    numerator += w * x[j] * sens[j];
                    denominator += w;
                }
                filtered[e] = numerator / std::max(denominator, T(1e-15)) / std::max(x[e], T(1e-15));
            }
            return filtered;
        }

        /**
         * Apply density filtering: filter design variables directly.
         */
        vector_type apply_density_filter(const vector_type& sens, const vector_type& x)
        {
            return m_H.dot(sens);
        }

        /**
         * Optimality criteria update of design variables.
         * x_new = max(x_min, min(1, max(x - move, min(x + move, x * (-sens/lambda)^eta))))
         */
        void oc_update(const vector_type& sens, vector_type& x)
        {
            std::size_t M = x.size();
            T l1 = T(0), l2 = T(1e6);
            T eta = T(0.5); // damping
            // Bisection for Lagrange multiplier lambda
            while (l2 - l1 > T(1e-10))
            {
                T lambda = (l1 + l2) / T(2);
                T sum = T(0);
                for (std::size_t e = 0; e < M; ++e)
                {
                    T x_new = x[e] * std::pow(-sens[e] / lambda, eta);
                    x_new = std::max(m_x_min, std::min(T(1), x_new));
                    x_new = std::max(x[e] - m_move_limit, std::min(x[e] + m_move_limit, x_new));
                    sum += x_new;
                }
                if (sum > m_volume_fraction * M)
                    l1 = lambda;
                else
                    l2 = lambda;
            }
            T lambda = (l1 + l2) / T(2);
            for (std::size_t e = 0; e < M; ++e)
            {
                T x_new = x[e] * std::pow(-sens[e] / lambda, eta);
                x_new = std::max(m_x_min, std::min(T(1), x_new));
                x[e] = std::max(x[e] - m_move_limit, std::min(x[e] + m_move_limit, x_new));
            }
        }

        /**
         * Compute compliance: c = U^T * K * U.
         */
        T compute_compliance(const matrix_type& K, const vector_type& U)
        {
            auto KU = K.dot(U);
            T c = T(0);
            for (std::size_t i = 0; i < m_ndof; ++i)
                c += U[i] * KU[i];
            return T(0.5) * c;
        }
    };

    /**
     * Convenience function: run topology optimization and return final density.
     */
    template <class T>
    inline auto topology_optimize(
        const xarray_container<uvector<T>>& nodes,
        const xarray_container<uvector<std::size_t>>& elements,
        const std::vector<std::size_t>& fixed_dofs,
        const std::vector<std::size_t>& force_dofs,
        const xarray_container<uvector<T>>& force_values,
        T young, T poisson,
        T volume_fraction = T(0.4),
        T filter_radius = T(1.5),
        std::size_t max_iter = 100)
    {
        topology_optimizer<T> opt(nodes, elements, fixed_dofs, force_dofs, force_values, young, poisson);
        opt.set_volume_fraction(volume_fraction);
        opt.set_filter_radius(filter_radius);
        opt.set_max_iterations(max_iter);
        return opt.optimize();
    }

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_TOPOLOGY_HPP