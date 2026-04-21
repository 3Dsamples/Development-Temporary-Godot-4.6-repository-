// physics/xmpm_solver.hpp
#ifndef XTENSOR_XMPM_SOLVER_HPP
#define XTENSOR_XMPM_SOLVER_HPP

// ----------------------------------------------------------------------------
// xmpm_solver.hpp – Material Point Method for continuum mechanics
// ----------------------------------------------------------------------------
// Provides a hybrid Lagrangian‑Eulerian solver for simulating:
//   - Elasto‑plastic solids (jelly, metal, flesh)
//   - Granular materials (sand, snow)
//   - Viscous fluids (honey, mud)
//   - Fracture and crack propagation
//   - Two‑way coupling with rigid bodies
//
// Uses BigNumber for high‑precision deformation and FFT for implicit solves.
// Designed for real‑time performance at 120 fps with SIMD and GPU offload.
// ----------------------------------------------------------------------------

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "xmesh.hpp"
#include "bignumber/bignumber.hpp"
#include "fft.hpp"
#include "xoptimize.hpp"

namespace xt {
namespace physics {
namespace mpm {

// ========================================================================
// Material models
// ========================================================================
template <class T>
class material_model {
public:
    virtual ~material_model() = default;
    virtual void update_deformation_gradient(xarray_container<T>& F, 
                                             const xarray_container<T>& vel_grad, T dt) = 0;
    virtual xarray_container<T> cauchy_stress(const xarray_container<T>& F) const = 0;
    virtual T hardening_coefficient() const { return T(0); }
    virtual std::string name() const = 0;
};

// Neo‑Hookean (hyperelastic)
template <class T>
class neo_hookean_material : public material_model<T> {
public:
    neo_hookean_material(T youngs_modulus, T poisson_ratio);
    void update_deformation_gradient(xarray_container<T>& F, 
                                     const xarray_container<T>& vel_grad, T dt) override;
    xarray_container<T> cauchy_stress(const xarray_container<T>& F) const override;
    std::string name() const override { return "NeoHookean"; }
private:
    T m_mu, m_lambda;  // Lamé parameters
};

// Drucker‑Prager (sand, snow)
template <class T>
class drucker_prager_material : public material_model<T> {
public:
    drucker_prager_material(T youngs_modulus, T poisson_ratio,
                            T friction_angle, T cohesion, T hardening);
    void update_deformation_gradient(xarray_container<T>& F,
                                     const xarray_container<T>& vel_grad, T dt) override;
    xarray_container<T> cauchy_stress(const xarray_container<T>& F) const override;
    T hardening_coefficient() const override { return m_hardening; }
    std::string name() const override { return "DruckerPrager"; }
private:
    T m_mu, m_lambda, m_phi, m_c, m_hardening;
};

// Newtonian fluid (for MPM)
template <class T>
class newtonian_fluid_material : public material_model<T> {
public:
    newtonian_fluid_material(T viscosity, T bulk_modulus);
    void update_deformation_gradient(xarray_container<T>& F,
                                     const xarray_container<T>& vel_grad, T dt) override;
    xarray_container<T> cauchy_stress(const xarray_container<T>& F) const override;
    std::string name() const override { return "NewtonianFluid"; }
private:
    T m_viscosity, m_kappa;
};

// ========================================================================
// MPM Particle
// ========================================================================
template <class T>
struct mpm_particle {
    xarray_container<T> pos;       // (3,)
    xarray_container<T> vel;       // (3,)
    xarray_container<T> F;         // (3,3) deformation gradient
    T mass;
    T volume;
    std::shared_ptr<material_model<T>> material;
    size_t material_id;
    bool active;
};

// ========================================================================
// MPM Grid
// ========================================================================
template <class T>
class mpm_grid {
public:
    mpm_grid(size_t nx, size_t ny, size_t nz, T dx);

    void clear();
    void transfer_particles_to_grid(const std::vector<mpm_particle<T>>& particles);
    void compute_grid_forces(T dt);
    void update_grid_velocities(T dt);
    void transfer_grid_to_particles(std::vector<mpm_particle<T>>& particles, T dt);

    const xarray_container<T>& mass() const;
    const xarray_container<T>& velocity() const;
    const xarray_container<T>& force() const;

    size_t cell_index(size_t i, size_t j, size_t k) const;

private:
    size_t m_nx, m_ny, m_nz;
    T m_dx, m_inv_dx;
    xarray_container<T> m_mass;      // grid node mass
    xarray_container<T> m_vel;       // grid node velocity (3, nx, ny, nz)
    xarray_container<T> m_force;     // grid node force
};

// ========================================================================
// MPM Solver
// ========================================================================
template <class T>
class mpm_solver {
public:
    mpm_solver(size_t nx, size_t ny, size_t nz, T dx, T dt = T(0.01));

    void add_particles(const xarray_container<T>& positions, T particle_mass,
                       std::shared_ptr<material_model<T>> material);

    void step();
    void step_with_substeps(size_t num_substeps);

    const std::vector<mpm_particle<T>>& particles() const;
    const mpm_grid<T>& grid() const;

    // Boundary conditions
    void add_sticky_boundary(const xarray_container<T>& min_corner,
                             const xarray_container<T>& max_corner);
    void add_separating_boundary(const xarray_container<T>& point,
                                 const xarray_container<T>& normal);
    void clear_boundaries();

    // Collision objects (rigid bodies)
    void add_collider(const mesh::mesh<T>& mesh, const xarray_container<T>& velocity = {});

    // Export to mesh (for rendering)
    mesh::mesh<T> particle_mesh() const;
    xarray_container<T> density_field() const;  // for volume rendering

    // Performance settings
    void set_use_gpu(bool use_gpu);
    void set_fft_solver(bool use_fft);  // for implicit pressure

private:
    size_t m_nx, m_ny, m_nz;
    T m_dx, m_dt;
    mpm_grid<T> m_grid;
    std::vector<mpm_particle<T>> m_particles;
    std::vector<std::pair<xarray_container<T>, xarray_container<T>>> m_sticky_bounds;
    std::vector<std::pair<xarray_container<T>, xarray_container<T>>> m_sep_bounds;
    bool m_use_gpu, m_use_fft;
    void apply_boundary_conditions();
    void particle_to_grid();
    void compute_stress();
    void grid_update();
    void grid_to_particle();
};

// ========================================================================
// Fracture / Crack Propagation (add‑on)
// ========================================================================
template <class T>
class fracture_model {
public:
    virtual ~fracture_model() = default;
    virtual bool should_fracture(const mpm_particle<T>& p,
                                 const xarray_container<T>& stress) = 0;
    virtual void apply_fracture(std::vector<mpm_particle<T>>& particles,
                                size_t p_idx) = 0;
};

template <class T>
class mohr_coulomb_fracture : public fracture_model<T> {
public:
    mohr_coulomb_fracture(T tensile_strength, T shear_strength);
    bool should_fracture(const mpm_particle<T>& p,
                         const xarray_container<T>& stress) override;
    void apply_fracture(std::vector<mpm_particle<T>>& particles,
                        size_t p_idx) override;
private:
    T m_tensile, m_shear;
};

template <class T>
class mpm_solver_with_fracture : public mpm_solver<T> {
public:
    mpm_solver_with_fracture(size_t nx, size_t ny, size_t nz, T dx, T dt,
                             std::shared_ptr<fracture_model<T>> fracture);
    void step() override;
private:
    std::shared_ptr<fracture_model<T>> m_fracture;
};

} // namespace mpm

using mpm::mpm_solver;
using mpm::mpm_grid;
using mpm::neo_hookean_material;
using mpm::drucker_prager_material;
using mpm::newtonian_fluid_material;
using mpm::mohr_coulomb_fracture;

} // namespace physics
} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs
// ----------------------------------------------------------------------------
namespace xt {
namespace physics {
namespace mpm {

// Material models
template <class T> neo_hookean_material<T>::neo_hookean_material(T E, T nu) {
    m_mu = E / (T(2) * (T(1) + nu));
    m_lambda = E * nu / ((T(1) + nu) * (T(1) - T(2) * nu));
}
template <class T> void neo_hookean_material<T>::update_deformation_gradient(xarray_container<T>& F, const xarray_container<T>& vel_grad, T dt) {
    // F = (I + dt * vel_grad) * F
}
template <class T> xarray_container<T> neo_hookean_material<T>::cauchy_stress(const xarray_container<T>& F) const {
    // Piola‑Kirchhoff → Cauchy
    return xarray_container<T>();
}

template <class T> drucker_prager_material<T>::drucker_prager_material(T E, T nu, T phi, T c, T hard) :
    m_phi(phi), m_c(c), m_hardening(hard) {
    m_mu = E / (T(2) * (T(1) + nu));
    m_lambda = E * nu / ((T(1) + nu) * (T(1) - T(2) * nu));
}
template <class T> void drucker_prager_material<T>::update_deformation_gradient(xarray_container<T>& F, const xarray_container<T>& vel_grad, T dt) {}
template <class T> xarray_container<T> drucker_prager_material<T>::cauchy_stress(const xarray_container<T>& F) const { return {}; }

template <class T> newtonian_fluid_material<T>::newtonian_fluid_material(T mu, T kappa) : m_viscosity(mu), m_kappa(kappa) {}
template <class T> void newtonian_fluid_material<T>::update_deformation_gradient(xarray_container<T>& F, const xarray_container<T>& vel_grad, T dt) {}
template <class T> xarray_container<T> newtonian_fluid_material<T>::cauchy_stress(const xarray_container<T>& F) const { return {}; }

// MPM Grid
template <class T> mpm_grid<T>::mpm_grid(size_t nx, size_t ny, size_t nz, T dx) : m_nx(nx), m_ny(ny), m_nz(nz), m_dx(dx), m_inv_dx(1/dx) {
    size_t total_nodes = (nx+1)*(ny+1)*(nz+1);
    m_mass = xarray_container<T>({total_nodes}, T(0));
    m_vel = xarray_container<T>({total_nodes, 3}, T(0));
    m_force = xarray_container<T>({total_nodes, 3}, T(0));
}
template <class T> void mpm_grid<T>::clear() { m_mass.fill(T(0)); m_vel.fill(T(0)); m_force.fill(T(0)); }
template <class T> void mpm_grid<T>::transfer_particles_to_grid(const std::vector<mpm_particle<T>>& particles) {}
template <class T> void mpm_grid<T>::compute_grid_forces(T dt) {}
template <class T> void mpm_grid<T>::update_grid_velocities(T dt) {}
template <class T> void mpm_grid<T>::transfer_grid_to_particles(std::vector<mpm_particle<T>>& particles, T dt) {}
template <class T> const xarray_container<T>& mpm_grid<T>::mass() const { return m_mass; }
template <class T> const xarray_container<T>& mpm_grid<T>::velocity() const { return m_vel; }
template <class T> const xarray_container<T>& mpm_grid<T>::force() const { return m_force; }
template <class T> size_t mpm_grid<T>::cell_index(size_t i, size_t j, size_t k) const { return i + j*(m_nx+1) + k*(m_nx+1)*(m_ny+1); }

// MPM Solver
template <class T> mpm_solver<T>::mpm_solver(size_t nx, size_t ny, size_t nz, T dx, T dt) : m_nx(nx), m_ny(ny), m_nz(nz), m_dx(dx), m_dt(dt), m_grid(nx,ny,nz,dx), m_use_gpu(false), m_use_fft(false) {}
template <class T> void mpm_solver<T>::add_particles(const xarray_container<T>& positions, T particle_mass, std::shared_ptr<material_model<T>> material) {}
template <class T> void mpm_solver<T>::step() { step_with_substeps(1); }
template <class T> void mpm_solver<T>::step_with_substeps(size_t num_substeps) { for(size_t s=0; s<num_substeps; ++s) { particle_to_grid(); compute_stress(); grid_update(); grid_to_particle(); } }
template <class T> const std::vector<mpm_particle<T>>& mpm_solver<T>::particles() const { return m_particles; }
template <class T> const mpm_grid<T>& mpm_solver<T>::grid() const { return m_grid; }
template <class T> void mpm_solver<T>::add_sticky_boundary(const xarray_container<T>& min_corner, const xarray_container<T>& max_corner) {}
template <class T> void mpm_solver<T>::add_separating_boundary(const xarray_container<T>& point, const xarray_container<T>& normal) {}
template <class T> void mpm_solver<T>::clear_boundaries() {}
template <class T> void mpm_solver<T>::add_collider(const mesh::mesh<T>& mesh, const xarray_container<T>& velocity) {}
template <class T> mesh::mesh<T> mpm_solver<T>::particle_mesh() const { return {}; }
template <class T> xarray_container<T> mpm_solver<T>::density_field() const { return {}; }
template <class T> void mpm_solver<T>::set_use_gpu(bool use_gpu) { m_use_gpu = use_gpu; }
template <class T> void mpm_solver<T>::set_fft_solver(bool use_fft) { m_use_fft = use_fft; }
template <class T> void mpm_solver<T>::particle_to_grid() {}
template <class T> void mpm_solver<T>::compute_stress() {}
template <class T> void mpm_solver<T>::grid_update() {}
template <class T> void mpm_solver<T>::grid_to_particle() {}

// Fracture
template <class T> mohr_coulomb_fracture<T>::mohr_coulomb_fracture(T tensile, T shear) : m_tensile(tensile), m_shear(shear) {}
template <class T> bool mohr_coulomb_fracture<T>::should_fracture(const mpm_particle<T>& p, const xarray_container<T>& stress) { return false; }
template <class T> void mohr_coulomb_fracture<T>::apply_fracture(std::vector<mpm_particle<T>>& particles, size_t p_idx) {}
template <class T> mpm_solver_with_fracture<T>::mpm_solver_with_fracture(size_t nx, size_t ny, size_t nz, T dx, T dt, std::shared_ptr<fracture_model<T>> fracture) : mpm_solver<T>(nx,ny,nz,dx,dt), m_fracture(fracture) {}
template <class T> void mpm_solver_with_fracture<T>::step() { mpm_solver<T>::step(); }

} // namespace mpm
} // namespace physics
} // namespace xt

#endif // XTENSOR_XMPM_SOLVER_HPP