// physics/xspd_solver.hpp
#ifndef XTENSOR_XSPD_SOLVER_HPP
#define XTENSOR_XSPD_SOLVER_HPP

// ----------------------------------------------------------------------------
// xspd_solver.hpp – Stable Principal Dynamics for real‑time soft bodies
// ----------------------------------------------------------------------------
// Provides a highly stable, GPU‑friendly solver for deformable solids:
//   - Subspace deformation with nonlinear corrections (StVK, Neo‑Hookean)
//   - Implicit integration using Chebyshev semi‑iterative method
//   - Support for tetrahedral, hexahedral, and voxel meshes
//   - Collision handling with signed distance fields (SDF)
//   - Fracture and tearing via modal analysis
//   - Target‑driven actuation (muscle‑like contraction)
//   - Integration with BigNumber for extreme precision
//   - FFT‑accelerated modal projection
//
// Designed for 120 fps performance on CPU and GPU.
// ----------------------------------------------------------------------------

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "xmesh.hpp"
#include "xlinalg.hpp"
#include "xdecomposition.hpp"
#include "fft.hpp"
#include "bignumber/bignumber.hpp"

namespace xt {
namespace physics {
namespace spd {

// ========================================================================
// Subspace basis (precomputed modes)
// ========================================================================
template <class T>
class subspace_basis {
public:
    subspace_basis();
    subspace_basis(const xarray_container<T>& U, const std::vector<T>& eigenvalues);

    size_t num_modes() const;
    size_t full_dimension() const;

    // Project full vector to reduced coordinates: q = U^T * (x - x0)
    xarray_container<T> project(const xarray_container<T>& displacement) const;

    // Lift reduced coordinates to full displacement: u = U * q
    xarray_container<T> lift(const xarray_container<T>& q) const;

    // Reduced stiffness and mass matrices (K_red = U^T * K * U)
    const xarray_container<T>& K_red() const;
    const xarray_container<T>& M_red() const;

    // Compute basis from mesh and material properties
    void compute_from_mesh(const mesh::mesh<T>& mesh, T youngs_modulus,
                           T poisson_ratio, T density, size_t num_modes);

    // Load precomputed modes from file
    void load(const std::string& filename);
    void save(const std::string& filename) const;

private:
    xarray_container<T> m_U;           // full_dim × num_modes
    std::vector<T> m_eigenvalues;
    xarray_container<T> m_K_red;       // num_modes × num_modes
    xarray_container<T> m_M_red;       // num_modes × num_modes
    xarray_container<T> m_rest_pos;    // full_dim (3×nodes)
};

// ========================================================================
// SPD Material Model
// ========================================================================
template <class T>
class spd_material {
public:
    virtual ~spd_material() = default;
    virtual T strain_energy(const xarray_container<T>& F) const = 0;
    virtual xarray_container<T> first_piola_kirchhoff(const xarray_container<T>& F) const = 0;
    virtual xarray_container<T> elasticity_tensor(const xarray_container<T>& F) const = 0;
    virtual std::string name() const = 0;
};

// Stable Neo‑Hookean (Smith et al. 2018)
template <class T>
class stable_neo_hookean : public spd_material<T> {
public:
    stable_neo_hookean(T youngs_modulus, T poisson_ratio);
    T strain_energy(const xarray_container<T>& F) const override;
    xarray_container<T> first_piola_kirchhoff(const xarray_container<T>& F) const override;
    xarray_container<T> elasticity_tensor(const xarray_container<T>& F) const override;
    std::string name() const override { return "StableNeoHookean"; }
private:
    T m_mu, m_lambda;
};

// St. Venant‑Kirchhoff
template <class T>
class stvk_material : public spd_material<T> {
public:
    stvk_material(T youngs_modulus, T poisson_ratio);
    T strain_energy(const xarray_container<T>& F) const override;
    xarray_container<T> first_piola_kirchhoff(const xarray_container<T>& F) const override;
    xarray_container<T> elasticity_tensor(const xarray_container<T>& F) const override;
    std::string name() const override { return "StVK"; }
private:
    T m_mu, m_lambda;
};

// ========================================================================
// SPD Solver
// ========================================================================
template <class T>
class spd_solver {
public:
    spd_solver();

    void set_mesh(const mesh::mesh<T>& mesh);
    void set_material(std::shared_ptr<spd_material<T>> material);
    void set_subspace(const subspace_basis<T>& basis);
    void set_subspace_auto(size_t num_modes);  // compute basis internally

    // Integration parameters
    void set_timestep(T dt);
    void set_damping(T rayleigh_alpha, T rayleigh_beta);
    void set_chebyshev_iterations(size_t iterations);

    // Boundary conditions
    void add_fixed_node(size_t node_index);
    void add_prescribed_displacement(size_t node_index, const xarray_container<T>& u);
    void add_force(size_t node_index, const xarray_container<T>& force);
    void clear_boundary_conditions();

    // Step forward
    void step();

    // Results
    const xarray_container<T>& full_displacement() const;
    const xarray_container<T>& reduced_coordinates() const;
    mesh::mesh<T> deformed_mesh() const;

    // Nonlinear correction (cubic spline lookup or direct evaluation)
    void enable_nonlinear_correction(bool enable);
    void set_nonlinear_samples(size_t num_samples);

    // Performance
    void set_use_gpu(bool use_gpu);
    void set_use_fft_acceleration(bool use_fft);

private:
    mesh::mesh<T> m_rest_mesh;
    std::shared_ptr<spd_material<T>> m_material;
    subspace_basis<T> m_basis;

    T m_dt;
    T m_alpha, m_beta;        // Rayleigh damping
    size_t m_cheb_iters;

    // Boundary conditions
    std::vector<size_t> m_fixed_nodes;
    std::unordered_map<size_t, xarray_container<T>> m_prescribed_u;
    std::unordered_map<size_t, xarray_container<T>> m_forces;

    // State
    xarray_container<T> m_q;        // reduced coordinates
    xarray_container<T> m_qdot;     // reduced velocity
    xarray_container<T> m_u_full;   // full displacement (cached)

    bool m_nonlinear_correction;
    size_t m_nonlinear_samples;
    bool m_use_gpu;
    bool m_use_fft;

    // Internal helpers
    xarray_container<T> compute_reduced_force() const;
    xarray_container<T> compute_nonlinear_force_correction(const xarray_container<T>& q) const;
    void solve_linear_system(const xarray_container<T>& A, xarray_container<T>& x,
                             const xarray_container<T>& b);
};

// ========================================================================
// Modal Fracture
// ========================================================================
template <class T>
class modal_fracture {
public:
    modal_fracture(T tensile_strength, T shear_strength);

    // Analyze stress in full displacement field and return new fracture surfaces
    std::vector<size_t> detect_fracture(const spd_solver<T>& solver,
                                        const xarray_container<T>& u);

    // Update subspace basis after fracture (remesh and recompute modes)
    subspace_basis<T> update_basis(const spd_solver<T>& solver,
                                   const std::vector<size_t>& fractured_elements);

private:
    T m_tensile, m_shear;
};

// ========================================================================
// Actuation (muscle‑like contraction)
// ========================================================================
template <class T>
class muscle_actuator {
public:
    muscle_actuator(const std::vector<size_t>& fiber_elements,
                    const xarray_container<T>& fiber_direction);

    void set_activation(T level);  // 0 = relaxed, 1 = fully contracted
    void apply_to_solver(spd_solver<T>& solver);

private:
    std::vector<size_t> m_elements;
    xarray_container<T> m_fiber_dir;
    T m_activation;
};

// ========================================================================
// GPU Offload
// ========================================================================
template <class T>
class gpu_spd_solver {
public:
    gpu_spd_solver();
    ~gpu_spd_solver();

    void upload(const spd_solver<T>& solver);
    void step();
    void download(spd_solver<T>& solver);

    bool is_available() const;

private:
    void* m_context;
    bool m_available;
};

} // namespace spd

using spd::subspace_basis;
using spd::spd_solver;
using spd::stable_neo_hookean;
using spd::stvk_material;
using spd::modal_fracture;
using spd::muscle_actuator;
using spd::gpu_spd_solver;

} // namespace physics
} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs
// ----------------------------------------------------------------------------
namespace xt {
namespace physics {
namespace spd {

// subspace_basis
template <class T> subspace_basis<T>::subspace_basis() = default;
template <class T> subspace_basis<T>::subspace_basis(const xarray_container<T>& U, const std::vector<T>& eig) : m_U(U), m_eigenvalues(eig) {}
template <class T> size_t subspace_basis<T>::num_modes() const { return m_U.shape()[1]; }
template <class T> size_t subspace_basis<T>::full_dimension() const { return m_U.shape()[0]; }
template <class T> xarray_container<T> subspace_basis<T>::project(const xarray_container<T>& displacement) const { return {}; }
template <class T> xarray_container<T> subspace_basis<T>::lift(const xarray_container<T>& q) const { return {}; }
template <class T> const xarray_container<T>& subspace_basis<T>::K_red() const { return m_K_red; }
template <class T> const xarray_container<T>& subspace_basis<T>::M_red() const { return m_M_red; }
template <class T> void subspace_basis<T>::compute_from_mesh(const mesh::mesh<T>& mesh, T E, T nu, T rho, size_t num_modes) {}
template <class T> void subspace_basis<T>::load(const std::string& filename) {}
template <class T> void subspace_basis<T>::save(const std::string& filename) const {}

// materials
template <class T> stable_neo_hookean<T>::stable_neo_hookean(T E, T nu) { m_mu = E/(T(2)*(T(1)+nu)); m_lambda = E*nu/((T(1)+nu)*(T(1)-T(2)*nu)); }
template <class T> T stable_neo_hookean<T>::strain_energy(const xarray_container<T>& F) const { return T(0); }
template <class T> xarray_container<T> stable_neo_hookean<T>::first_piola_kirchhoff(const xarray_container<T>& F) const { return {}; }
template <class T> xarray_container<T> stable_neo_hookean<T>::elasticity_tensor(const xarray_container<T>& F) const { return {}; }

template <class T> stvk_material<T>::stvk_material(T E, T nu) { m_mu = E/(T(2)*(T(1)+nu)); m_lambda = E*nu/((T(1)+nu)*(T(1)-T(2)*nu)); }
template <class T> T stvk_material<T>::strain_energy(const xarray_container<T>& F) const { return T(0); }
template <class T> xarray_container<T> stvk_material<T>::first_piola_kirchhoff(const xarray_container<T>& F) const { return {}; }
template <class T> xarray_container<T> stvk_material<T>::elasticity_tensor(const xarray_container<T>& F) const { return {}; }

// solver
template <class T> spd_solver<T>::spd_solver() : m_dt(0.01), m_alpha(0), m_beta(0), m_cheb_iters(10), m_nonlinear_correction(true), m_nonlinear_samples(32), m_use_gpu(false), m_use_fft(false) {}
template <class T> void spd_solver<T>::set_mesh(const mesh::mesh<T>& mesh) { m_rest_mesh = mesh; }
template <class T> void spd_solver<T>::set_material(std::shared_ptr<spd_material<T>> material) { m_material = material; }
template <class T> void spd_solver<T>::set_subspace(const subspace_basis<T>& basis) { m_basis = basis; m_q = xarray_container<T>({basis.num_modes()}, T(0)); m_qdot = xarray_container<T>({basis.num_modes()}, T(0)); }
template <class T> void spd_solver<T>::set_subspace_auto(size_t num_modes) { m_basis.compute_from_mesh(m_rest_mesh, 1e6, 0.3, 1000, num_modes); set_subspace(m_basis); }
template <class T> void spd_solver<T>::set_timestep(T dt) { m_dt = dt; }
template <class T> void spd_solver<T>::set_damping(T alpha, T beta) { m_alpha = alpha; m_beta = beta; }
template <class T> void spd_solver<T>::set_chebyshev_iterations(size_t iter) { m_cheb_iters = iter; }
template <class T> void spd_solver<T>::add_fixed_node(size_t node) { m_fixed_nodes.push_back(node); }
template <class T> void spd_solver<T>::add_prescribed_displacement(size_t node, const xarray_container<T>& u) { m_prescribed_u[node] = u; }
template <class T> void spd_solver<T>::add_force(size_t node, const xarray_container<T>& force) { m_forces[node] = force; }
template <class T> void spd_solver<T>::clear_boundary_conditions() { m_fixed_nodes.clear(); m_prescribed_u.clear(); m_forces.clear(); }
template <class T> void spd_solver<T>::step() {}
template <class T> const xarray_container<T>& spd_solver<T>::full_displacement() const { return m_u_full; }
template <class T> const xarray_container<T>& spd_solver<T>::reduced_coordinates() const { return m_q; }
template <class T> mesh::mesh<T> spd_solver<T>::deformed_mesh() const { return m_rest_mesh; }
template <class T> void spd_solver<T>::enable_nonlinear_correction(bool enable) { m_nonlinear_correction = enable; }
template <class T> void spd_solver<T>::set_nonlinear_samples(size_t n) { m_nonlinear_samples = n; }
template <class T> void spd_solver<T>::set_use_gpu(bool use) { m_use_gpu = use; }
template <class T> void spd_solver<T>::set_use_fft_acceleration(bool use) { m_use_fft = use; }
template <class T> xarray_container<T> spd_solver<T>::compute_reduced_force() const { return {}; }
template <class T> xarray_container<T> spd_solver<T>::compute_nonlinear_force_correction(const xarray_container<T>& q) const { return {}; }
template <class T> void spd_solver<T>::solve_linear_system(const xarray_container<T>& A, xarray_container<T>& x, const xarray_container<T>& b) {}

// fracture
template <class T> modal_fracture<T>::modal_fracture(T tensile, T shear) : m_tensile(tensile), m_shear(shear) {}
template <class T> std::vector<size_t> modal_fracture<T>::detect_fracture(const spd_solver<T>& solver, const xarray_container<T>& u) { return {}; }
template <class T> subspace_basis<T> modal_fracture<T>::update_basis(const spd_solver<T>& solver, const std::vector<size_t>& fractured_elements) { return {}; }

// actuator
template <class T> muscle_actuator<T>::muscle_actuator(const std::vector<size_t>& fiber_elements, const xarray_container<T>& fiber_dir) : m_elements(fiber_elements), m_fiber_dir(fiber_dir), m_activation(0) {}
template <class T> void muscle_actuator<T>::set_activation(T level) { m_activation = level; }
template <class T> void muscle_actuator<T>::apply_to_solver(spd_solver<T>& solver) {}

// GPU
template <class T> gpu_spd_solver<T>::gpu_spd_solver() : m_context(nullptr), m_available(false) {}
template <class T> gpu_spd_solver<T>::~gpu_spd_solver() {}
template <class T> void gpu_spd_solver<T>::upload(const spd_solver<T>& solver) {}
template <class T> void gpu_spd_solver<T>::step() {}
template <class T> void gpu_spd_solver<T>::download(spd_solver<T>& solver) {}
template <class T> bool gpu_spd_solver<T>::is_available() const { return m_available; }

} // namespace spd
} // namespace physics
} // namespace xt

#endif // XTENSOR_XSPD_SOLVER_HPP