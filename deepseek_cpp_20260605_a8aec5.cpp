//File 0228 : sparse/xsparse_simulation.hpp
//Sparse implicit time integration for deformable bodies with finite‑element assembly, Rayleigh damping, and penalty‑based contact resolution using CG.
#ifndef XTENSOR_XSPARSE_SIMULATION_HPP
#define XTENSOR_XSPARSE_SIMULATION_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
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
#include "../sparse/xcsr.hpp"
#include "../sparse/xcoo.hpp"
#include "../sparse/xcsc.hpp"
#include "../sparse/xsparse_config.hpp"
#include "../sparse/xsparse_reducer.hpp"
#include "../sparse/xsparse_solver.hpp"
#include "../sparse/xsparse_symmetry.hpp"
#include "../sparse/xsparse_mesh.hpp"
#include "../sparse/xsparse_lcp.hpp"

namespace xt {
namespace sparse {

    /**
     * @enum time_integration_method
     * @brief Available implicit time stepping schemes.
     */
    enum class time_integration_method
    {
        backward_euler,       // first‑order unconditionally stable
        newmark,              // second‑order Newmark‑β (β=0.25, γ=0.5 for average acceleration)
        generalized_alpha     // generalized‑α method for controlled numerical dissipation
    };

    /**
     * @class deformable_solid_simulator
     * @brief Implicit time stepper for deformable bodies using sparse FEM matrices.
     *
     * Assembles mass and stiffness matrices from a 2D/3D mesh, applies Rayleigh damping,
     * and solves the resulting linear system via sparse CG. Contact is handled by a
     * penalty method that adds forces and modifies the system matrix.
     */
    template <class T>
    class deformable_solid_simulator
    {
    public:
        using matrix_type = xcsr_matrix<T>;
        using vector_type = xarray_container<uvector<T>>;
        using index_array = xarray_container<uvector<std::size_t>>;

        /**
         * Construct the simulator for a mesh with given nodes and elements.
         * @param nodes Current node positions (N x dim).
         * @param elements Element connectivity (M x nodes_per_elem).
         * @param young Young's modulus.
         * @param poisson Poisson's ratio.
         * @param density Mass density.
         * @param dt Initial time step.
         */
        deformable_solid_simulator(const vector_type& nodes,
                                   const index_array& elements,
                                   T young, T poisson, T density, T dt)
            : m_nodes(nodes)
            , m_elements(elements)
            , m_young(young)
            , m_poisson(poisson)
            , m_density(density)
            , m_dt(dt)
            , m_time(T(0))
            , m_integration(time_integration_method::backward_euler)
        {
            std::size_t N = nodes.shape()[0];
            std::size_t dim = nodes.shape()[1];
            m_dim = dim;
            m_dofs = N * dim;
            // Allocate state vectors
            m_velocity = vector_type({m_dofs}, T(0));
            m_external_force = vector_type({m_dofs}, T(0));
            m_displacement = vector_type({m_dofs}, T(0));
            // Assemble initial mass matrix (constant for small deformations)
            assemble_mass();
            // Assemble initial stiffness matrix
            assemble_stiffness();
            // Store initial positions for strain computation
            m_rest_nodes = nodes;
        }

        /**
         * Set the time integration method.
         */
        void set_integration_method(time_integration_method method) noexcept
        {
            m_integration = method;
        }

        /**
         * Set Rayleigh damping coefficients. C = α*M + β*K.
         */
        void set_rayleigh_damping(T alpha, T beta) noexcept
        {
            m_alpha = alpha;
            m_beta = beta;
        }

        /**
         * Apply an external force to a node.
         */
        void apply_force(std::size_t node_idx, T fx, T fy, T fz = T(0))
        {
            m_external_force[node_idx * m_dim + 0] += fx;
            m_external_force[node_idx * m_dim + 1] += fy;
            if (m_dim > 2) m_external_force[node_idx * m_dim + 2] += fz;
        }

        /**
         * Apply gravity to all nodes.
         */
        void apply_gravity(T gx, T gy, T gz = T(0))
        {
            for (std::size_t i = 0; i < m_nodes.shape()[0]; ++i)
            {
                T mass = m_lumped_mass[i];
                m_external_force[i * m_dim + 0] += mass * gx;
                m_external_force[i * m_dim + 1] += mass * gy;
                if (m_dim > 2) m_external_force[i * m_dim + 2] += mass * gz;
            }
        }

        /**
         * Add a penalty contact force: a spring with stiffness k_contact acting between
         * node and a fixed ground plane at y = ground_y when node penetrates.
         */
        void add_ground_contact(T ground_y, T k_contact)
        {
            m_contact_ground_y = ground_y;
            m_contact_k = k_contact;
            m_has_ground_contact = true;
        }

        /**
         * Advance simulation by one time step.
         */
        void step()
        {
            // Assemble current stiffness matrix (may be nonlinear with large deformations)
            assemble_stiffness();
            // Compute internal forces: f_int = -K * x   (x is displacement from rest)
            auto f_int = m_stiffness.dot(m_displacement);
            for (std::size_t i = 0; i < m_dofs; ++i) f_int[i] = -f_int[i];

            // Apply penalty contact forces
            apply_contact_forces();

            // Compute right‑hand side and system matrix depending on method
            vector_type rhs({m_dofs}, T(0));
            matrix_type A(0, 0);
            assemble_system(rhs, A, f_int);

            // Solve A * dx = rhs using sparse CG with diagonal preconditioner
            auto dx = cg_solve(A, rhs, T(1e-6), 500, preconditioner_type::diagonal);

            // Update state
            if (m_integration == time_integration_method::backward_euler)
            {
                // dx = Δv (velocity increment)
                for (std::size_t i = 0; i < m_dofs; ++i)
                {
                    m_velocity[i] += dx[i];
                    m_displacement[i] += m_dt * m_velocity[i];
                }
            }
            else if (m_integration == time_integration_method::newmark)
            {
                // Newmark with β=0.25, γ=0.5 (average acceleration, unconditionally stable)
                T beta_nm = T(0.25);
                T gamma_nm = T(0.5);
                // dx = acceleration increment → a_new = a_old + dx
                // v_new = v_old + dt * ((1-γ)*a_old + γ*a_new)
                // u_new = u_old + dt*v_old + dt²/2 * ((1-2β)*a_old + 2β*a_new)
                for (std::size_t i = 0; i < m_dofs; ++i)
                {
                    T a_old = m_acceleration[i];
                    T a_new = a_old + dx[i];
                    T v_new = m_velocity[i] + m_dt * ((T(1)-gamma_nm)*a_old + gamma_nm*a_new);
                    T u_new = m_displacement[i] + m_dt * m_velocity[i]
                            + T(0.5) * m_dt * m_dt * ((T(1) - T(2)*beta_nm)*a_old + T(2)*beta_nm*a_new);
                    m_velocity[i] = v_new;
                    m_displacement[i] = u_new;
                    m_acceleration[i] = a_new;
                }
            }
            // Update node positions
            for (std::size_t i = 0; i < m_nodes.shape()[0]; ++i)
                for (std::size_t d = 0; d < m_dim; ++d)
                    m_nodes(i, d) = m_rest_nodes(i, d) + m_displacement[i * m_dim + d];

            // Reset external forces for next step
            std::fill(m_external_force.data(), m_external_force.data() + m_dofs, T(0));
            m_time += m_dt;
        }

        /**
         * Run simulation for a given duration.
         */
        void run(T duration)
        {
            T end_time = m_time + duration;
            while (m_time < end_time)
                step();
        }

        // Accessors
        const vector_type& nodes() const noexcept { return m_nodes; }
        const vector_type& velocity() const noexcept { return m_velocity; }
        const vector_type& displacement() const noexcept { return m_displacement; }
        const matrix_type& stiffness_matrix() const noexcept { return m_stiffness; }
        const matrix_type& mass_matrix() const noexcept { return m_mass; }
        T time() const noexcept { return m_time; }
        void set_time_step(T dt) noexcept { m_dt = dt; }

    private:
        vector_type m_nodes;
        vector_type m_rest_nodes;
        index_array m_elements;
        T m_young, m_poisson, m_density;
        T m_dt, m_time;
        T m_alpha = T(0), m_beta = T(0);
        std::size_t m_dim, m_dofs;
        time_integration_method m_integration;

        vector_type m_velocity;
        vector_type m_acceleration; // for Newmark
        vector_type m_displacement;
        vector_type m_external_force;
        vector_type m_lumped_mass; // lumped mass per node
        matrix_type m_mass;
        matrix_type m_stiffness;

        // Contact
        bool m_has_ground_contact = false;
        T m_contact_ground_y = T(0);
        T m_contact_k = T(0);

        /**
         * Assemble lumped mass matrix (diagonal: mass of each node = density * area/volume per node).
         * For triangular mesh, mass per node = sum of (element_area * density / 3) for elements sharing the node.
         */
        void assemble_mass()
        {
            std::size_t N = m_nodes.shape()[0];
            m_lumped_mass = vector_type({N}, T(0));
            for (std::size_t e = 0; e < m_elements.shape()[0]; ++e)
            {
                T area = compute_element_area(e);
                T mass_per_node = area * m_density / T(m_elements.shape()[1]); // equally distributed
                for (std::size_t k = 0; k < m_elements.shape()[1]; ++k)
                    m_lumped_mass[m_elements(e, k)] += mass_per_node;
            }
            // Build sparse diagonal mass matrix
            xcoo_matrix<T> mass_coo(m_dofs, m_dofs);
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t d = 0; d < m_dim; ++d)
                    mass_coo.append(i * m_dim + d, i * m_dim + d, m_lumped_mass[i]);
            m_mass = xcsr_matrix<T>::from_coo(mass_coo);
        }

        /**
         * Assemble stiffness matrix from current node positions.
         * Delegates to the mesh assembly functions.
         */
        void assemble_stiffness()
        {
            if (m_elements.shape()[1] == 3 && m_dim == 2)
            {
                m_stiffness = assemble_stiffness_tri2d(m_nodes, m_elements, m_young, m_poisson);
            }
            else if (m_elements.shape()[1] == 4 && m_dim == 2)
            {
                m_stiffness = assemble_stiffness_quad2d(m_nodes, m_elements, m_young, m_poisson);
            }
            else
            {
                throw std::runtime_error("Unsupported element type for stiffness assembly.");
            }
        }

        /**
         * Compute the area of a 2D triangular element.
         */
        T compute_element_area(std::size_t e) const
        {
            std::size_t n0 = m_elements(e, 0), n1 = m_elements(e, 1), n2 = m_elements(e, 2);
            T x0 = m_nodes(n0, 0), y0 = m_nodes(n0, 1);
            T x1 = m_nodes(n1, 0), y1 = m_nodes(n1, 1);
            T x2 = m_nodes(n2, 0), y2 = m_nodes(n2, 1);
            return T(0.5) * std::abs((x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0));
        }

        /**
         * Apply penalty contact forces from ground plane.
         */
        void apply_contact_forces()
        {
            if (!m_has_ground_contact) return;
            for (std::size_t i = 0; i < m_nodes.shape()[0]; ++i)
            {
                T y = m_nodes(i, 1);
                if (y < m_contact_ground_y)
                {
                    T penetration = m_contact_ground_y - y;
                    // Penalty force upward
                    m_external_force[i * m_dim + 1] += m_contact_k * penetration;
                }
            }
        }

        /**
         * Assemble the linear system A * dx = rhs for the current time step.
         */
        void assemble_system(vector_type& rhs, matrix_type& A, const vector_type& f_int)
        {
            T dt = m_dt;
            if (m_integration == time_integration_method::backward_euler)
            {
                // (M + dt*C + dt²*K) * Δv = dt * (f_ext + f_int)
                // C = α*M + β*K, so system = M + dt*(αM+βK) + dt²*K = (1+α*dt)*M + (β*dt+dt²)*K
                T coeff_m = T(1) + m_alpha * dt;
                T coeff_k = m_beta * dt + dt * dt;
                A = combine_mass_stiffness(coeff_m, coeff_k);
                // rhs = dt * (f_ext + f_int)
                for (std::size_t i = 0; i < m_dofs; ++i)
                    rhs[i] = dt * (m_external_force[i] + f_int[i]);
            }
            else if (m_integration == time_integration_method::newmark)
            {
                T beta_nm = T(0.25);
                T gamma_nm = T(0.5);
                T coeff_m = T(1) + m_alpha * gamma_nm * dt;
                T coeff_k = beta_nm * dt * dt + m_beta * gamma_nm * dt;
                A = combine_mass_stiffness(coeff_m, coeff_k);
                // rhs = dt * (f_ext + f_int) - (dt*C + dt²*K) * v_old - dt*K * u_old  (simplified)
                for (std::size_t i = 0; i < m_dofs; ++i)
                    rhs[i] = dt * (m_external_force[i] + f_int[i]);
            }
        }

        /**
         * Build A = coeff_m * M + coeff_k * K by merging two CSR matrices.
         */
        matrix_type combine_mass_stiffness(T coeff_m, T coeff_k)
        {
            // Scale M by coeff_m, K by coeff_k, then add (they have the same sparsity pattern for FEM).
            // If patterns differ, use general sparse addition; here we assume same pattern.
            auto M_scaled = m_mass;
            M_scaled *= coeff_m;
            auto K_scaled = m_stiffness;
            K_scaled *= coeff_k;
            return xcsr_matrix<T>::add(M_scaled, K_scaled);
        }
    };

} // namespace sparse
} // namespace xt

#endif // XTENSOR_XSPARSE_SIMULATION_HPP