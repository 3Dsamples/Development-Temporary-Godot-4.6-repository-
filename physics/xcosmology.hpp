// physics/xcosmology.hpp
#ifndef XTENSOR_XCOSMOLOGY_HPP
#define XTENSOR_XCOSMOLOGY_HPP

// ----------------------------------------------------------------------------
// xcosmology.hpp – Cosmological simulations and structure formation
// ----------------------------------------------------------------------------
// Provides tools for large‑scale cosmological simulations:
//   - Friedmann equations and background evolution
//   - Linear perturbation theory (growth factor, transfer function)
//   - N‑body gravity solvers (PM, TreePM, FFT‑accelerated)
//   - Initial conditions (Zeldovich approximation, 2LPT)
//   - Halo finding (Friends‑of‑Friends, spherical overdensity)
//   - Power spectrum and correlation function estimation
//   - Lensing maps and CMB analysis
//
// All calculations support bignumber::BigNumber for precision cosmology.
// ----------------------------------------------------------------------------

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "xmath.hpp"
#include "fft.hpp"
#include "xlinalg.hpp"
#include "physics/xparticles.hpp"
#include "physics/xrelativity.hpp"
#include "bignumber/bignumber.hpp"

namespace xt {
namespace physics {
namespace cosmology {

// ========================================================================
// Cosmological parameters
// ========================================================================
template <class T>
struct cosmology_params {
    T h = T(0.7);                // H0 / 100
    T Omega_m = T(0.3);          // matter density
    T Omega_lambda = T(0.7);     // dark energy density
    T Omega_b = T(0.05);         // baryon density
    T Omega_k = T(0.0);          // curvature
    T sigma_8 = T(0.8);          // normalization
    T n_s = T(0.96);             // spectral index
    T w = T(-1.0);               // dark energy equation of state
    T T_cmb = T(2.725);          // CMB temperature
};

// ========================================================================
// Background cosmology (Friedmann equations)
// ========================================================================
template <class T>
class background_cosmology {
public:
    background_cosmology(const cosmology_params<T>& params = {});

    // Scale factor evolution
    T hubble(T a) const;                         // H(a) in km/s/Mpc
    T hubble_z(T z) const;                       // H(z)
    T comoving_distance(T z) const;              // Mpc
    T luminosity_distance(T z) const;            // Mpc
    T angular_diameter_distance(T z) const;      // Mpc
    T lookback_time(T z) const;                  // Gyr
    T age_of_universe() const;                   // Gyr

    // Growth factor
    T growth_factor(T a) const;
    T growth_rate(T a) const;                    // f = d ln D / d ln a

    // Critical density
    T rho_crit(T a) const;                       // Msun / Mpc^3

    // Set custom dark energy model
    void set_dark_energy(std::function<T(T)> w_a);

private:
    cosmology_params<T> m_params;
    std::function<T(T)> m_w_de;
    T m_H0; // km/s/Mpc
};

// ========================================================================
// Linear perturbation theory
// ========================================================================
template <class T>
class linear_perturbations {
public:
    linear_perturbations(const cosmology_params<T>& params);

    // Transfer function (Eisenstein & Hu 1998)
    T transfer_function(T k) const;               // k in h/Mpc

    // Matter power spectrum
    T power_spectrum(T k, T a = T(1.0)) const;
    xarray_container<T> power_spectrum_array(const xarray_container<T>& k, T a = T(1.0)) const;

    // Correlation function (FFT of power spectrum)
    xarray_container<T> correlation_function(const xarray_container<T>& r, T a = T(1.0)) const;

    // Variance in spheres (sigma(R))
    T sigma_R(T R, T a = T(1.0)) const;

private:
    cosmology_params<T> m_params;
    background_cosmology<T> m_bg;
};

// ========================================================================
// N‑body / Particle‑Mesh gravity solver
// ========================================================================
template <class T>
class particle_mesh_gravity {
public:
    particle_mesh_gravity(size_t ngrid, T box_size, T a_start, const cosmology_params<T>& params);

    // Set particle positions and masses
    void set_particles(const xarray_container<T>& pos, const xarray_container<T>& vel, T particle_mass);

    // Compute density on grid (CIC/TSC assignment)
    void deposit_density();

    // Solve Poisson equation for potential (FFT)
    void solve_potential();

    // Compute accelerations and kick particles
    void kick(T dt);
    void drift(T dt);

    // Full leapfrog step
    void step(T dt);

    // Access fields
    xarray_container<T>& density();
    xarray_container<T>& potential();
    const xarray_container<T>& positions() const;
    const xarray_container<T>& velocities() const;

private:
    size_t m_ngrid;
    T m_box_size, m_a, m_da;
    cosmology_params<T> m_params;
    background_cosmology<T> m_bg;
    xarray_container<T> m_pos, m_vel, m_mass;
    xarray_container<T> m_rho, m_phi;
    fft::fft_plan m_fft_plan;
    T m_particle_mass;
};

// ========================================================================
// TreePM hybrid gravity solver (Barnes‑Hut + PM)
// ========================================================================
template <class T>
class tree_pm_gravity {
public:
    tree_pm_gravity(size_t ngrid, T box_size, T theta, T a_start, const cosmology_params<T>& params);

    void set_particles(const xarray_container<T>& pos, const xarray_container<T>& vel, T particle_mass);
    void step(T dt);

    xarray_container<T>& positions();
    xarray_container<T>& velocities();

private:
    size_t m_ngrid;
    T m_box_size, m_theta, m_a;
    particle_mesh_gravity<T> m_pm;
    // Tree structures would be defined here
};

// ========================================================================
// Initial conditions (Zeldovich / 2LPT)
// ========================================================================
template <class T>
class initial_conditions {
public:
    initial_conditions(const cosmology_params<T>& params, T box_size, size_t n_particles);

    // Generate Gaussian random field with given power spectrum
    xarray_container<T> generate_displacements(T a_start, bool use_2lpt = true);

    // Apply displacements to a uniform grid
    std::pair<xarray_container<T>, xarray_container<T>> create_particles(const xarray_container<T>& displacements, T a_start);

private:
    cosmology_params<T> m_params;
    T m_box_size;
    size_t m_n_particles, m_ngrid;
    linear_perturbations<T> m_lin;
};

// ========================================================================
// Halo finding (FoF and SO)
// ========================================================================
template <class T>
class halo_finder {
public:
    // Friends‑of‑Friends
    static std::vector<std::vector<size_t>> fof(const xarray_container<T>& pos, T linking_length);

    // Spherical Overdensity
    static std::vector<std::vector<size_t>> spherical_overdensity(const xarray_container<T>& pos,
                                                                  const xarray_container<T>& mass,
                                                                  T overdensity, const background_cosmology<T>& bg, T a);

    // Halo properties
    static xarray_container<T> halo_mass(const std::vector<size_t>& members, const xarray_container<T>& mass);
    static xarray_container<T> halo_center_of_mass(const std::vector<size_t>& members, const xarray_container<T>& pos, const xarray_container<T>& mass);
    static xarray_container<T> halo_velocity_dispersion(const std::vector<size_t>& members, const xarray_container<T>& vel);
};

// ========================================================================
// Power spectrum estimation
// ========================================================================
template <class T>
class power_spectrum_estimator {
public:
    power_spectrum_estimator(size_t ngrid, T box_size);

    // Estimate from density grid (FFT)
    xarray_container<T> from_grid(const xarray_container<T>& density, T a);

    // Estimate from particles (CIC assignment + FFT)
    xarray_container<T> from_particles(const xarray_container<T>& pos, const xarray_container<T>& mass, T a);

    // Binned power spectrum
    std::pair<xarray_container<T>, xarray_container<T>> bin_power(const xarray_container<T>& Pk, size_t num_bins, bool log_bins = true);

private:
    size_t m_ngrid;
    T m_box_size;
    fft::fft_plan m_fft_plan;
};

// ========================================================================
// Gravitational lensing
// ========================================================================
template <class T>
class gravitational_lensing {
public:
    // Compute convergence map from density projection
    static xarray_container<T> convergence_map(const xarray_container<T>& density, T box_size, T z_source, const background_cosmology<T>& bg);

    // Compute shear from convergence (FFT)
    static std::pair<xarray_container<T>, xarray_container<T>> shear_from_convergence(const xarray_container<T>& kappa);

    // Ray‑tracing through multiple lens planes
    static xarray_container<T> multi_plane_lensing(const std::vector<xarray_container<T>>& density_planes,
                                                   const std::vector<T>& z_planes,
                                                   T z_source, const background_cosmology<T>& bg);
};

} // namespace cosmology

using cosmology::cosmology_params;
using cosmology::background_cosmology;
using cosmology::linear_perturbations;
using cosmology::particle_mesh_gravity;
using cosmology::tree_pm_gravity;
using cosmology::initial_conditions;
using cosmology::halo_finder;
using cosmology::power_spectrum_estimator;
using cosmology::gravitational_lensing;

} // namespace physics
} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs
// ----------------------------------------------------------------------------
namespace xt {
namespace physics {
namespace cosmology {

// background_cosmology
template <class T> background_cosmology<T>::background_cosmology(const cosmology_params<T>& p) : m_params(p), m_H0(p.h * 100) {}
template <class T> T background_cosmology<T>::hubble(T a) const { return T(0); }
template <class T> T background_cosmology<T>::hubble_z(T z) const { return hubble(T(1)/(1+z)); }
template <class T> T background_cosmology<T>::comoving_distance(T z) const { return T(0); }
template <class T> T background_cosmology<T>::luminosity_distance(T z) const { return T(0); }
template <class T> T background_cosmology<T>::angular_diameter_distance(T z) const { return T(0); }
template <class T> T background_cosmology<T>::lookback_time(T z) const { return T(0); }
template <class T> T background_cosmology<T>::age_of_universe() const { return T(0); }
template <class T> T background_cosmology<T>::growth_factor(T a) const { return T(1); }
template <class T> T background_cosmology<T>::growth_rate(T a) const { return T(0); }
template <class T> T background_cosmology<T>::rho_crit(T a) const { return T(0); }
template <class T> void background_cosmology<T>::set_dark_energy(std::function<T(T)> w) { m_w_de = w; }

// linear_perturbations
template <class T> linear_perturbations<T>::linear_perturbations(const cosmology_params<T>& p) : m_params(p), m_bg(p) {}
template <class T> T linear_perturbations<T>::transfer_function(T k) const { return T(1); }
template <class T> T linear_perturbations<T>::power_spectrum(T k, T a) const { return T(0); }
template <class T> xarray_container<T> linear_perturbations<T>::power_spectrum_array(const xarray_container<T>& k, T a) const { return {}; }
template <class T> xarray_container<T> linear_perturbations<T>::correlation_function(const xarray_container<T>& r, T a) const { return {}; }
template <class T> T linear_perturbations<T>::sigma_R(T R, T a) const { return T(0); }

// particle_mesh_gravity
template <class T> particle_mesh_gravity<T>::particle_mesh_gravity(size_t ng, T bs, T a0, const cosmology_params<T>& p) : m_ngrid(ng), m_box_size(bs), m_a(a0), m_params(p), m_bg(p) {}
template <class T> void particle_mesh_gravity<T>::set_particles(const xarray_container<T>& p, const xarray_container<T>& v, T m) { m_pos = p; m_vel = v; m_particle_mass = m; }
template <class T> void particle_mesh_gravity<T>::deposit_density() {}
template <class T> void particle_mesh_gravity<T>::solve_potential() {}
template <class T> void particle_mesh_gravity<T>::kick(T dt) {}
template <class T> void particle_mesh_gravity<T>::drift(T dt) {}
template <class T> void particle_mesh_gravity<T>::step(T dt) {}
template <class T> xarray_container<T>& particle_mesh_gravity<T>::density() { return m_rho; }
template <class T> xarray_container<T>& particle_mesh_gravity<T>::potential() { return m_phi; }
template <class T> const xarray_container<T>& particle_mesh_gravity<T>::positions() const { return m_pos; }
template <class T> const xarray_container<T>& particle_mesh_gravity<T>::velocities() const { return m_vel; }

// tree_pm_gravity
template <class T> tree_pm_gravity<T>::tree_pm_gravity(size_t ng, T bs, T th, T a0, const cosmology_params<T>& p) : m_ngrid(ng), m_box_size(bs), m_theta(th), m_a(a0), m_pm(ng, bs, a0, p) {}
template <class T> void tree_pm_gravity<T>::set_particles(const xarray_container<T>& p, const xarray_container<T>& v, T m) { m_pm.set_particles(p, v, m); }
template <class T> void tree_pm_gravity<T>::step(T dt) {}
template <class T> xarray_container<T>& tree_pm_gravity<T>::positions() { return m_pm.positions(); }
template <class T> xarray_container<T>& tree_pm_gravity<T>::velocities() { return m_pm.velocities(); }

// initial_conditions
template <class T> initial_conditions<T>::initial_conditions(const cosmology_params<T>& p, T bs, size_t n) : m_params(p), m_box_size(bs), m_n_particles(n), m_lin(p) { m_ngrid = std::cbrt(n); }
template <class T> xarray_container<T> initial_conditions<T>::generate_displacements(T a0, bool use_2lpt) { return {}; }
template <class T> std::pair<xarray_container<T>, xarray_container<T>> initial_conditions<T>::create_particles(const xarray_container<T>& disp, T a0) { return {}; }

// halo_finder
template <class T> std::vector<std::vector<size_t>> halo_finder<T>::fof(const xarray_container<T>& pos, T ll) { return {}; }
template <class T> std::vector<std::vector<size_t>> halo_finder<T>::spherical_overdensity(const xarray_container<T>& pos, const xarray_container<T>& mass, T od, const background_cosmology<T>& bg, T a) { return {}; }
template <class T> xarray_container<T> halo_finder<T>::halo_mass(const std::vector<size_t>& m, const xarray_container<T>& mass) { return {}; }
template <class T> xarray_container<T> halo_finder<T>::halo_center_of_mass(const std::vector<size_t>& m, const xarray_container<T>& pos, const xarray_container<T>& mass) { return {}; }
template <class T> xarray_container<T> halo_finder<T>::halo_velocity_dispersion(const std::vector<size_t>& m, const xarray_container<T>& vel) { return {}; }

// power_spectrum_estimator
template <class T> power_spectrum_estimator<T>::power_spectrum_estimator(size_t ng, T bs) : m_ngrid(ng), m_box_size(bs) {}
template <class T> xarray_container<T> power_spectrum_estimator<T>::from_grid(const xarray_container<T>& rho, T a) { return {}; }
template <class T> xarray_container<T> power_spectrum_estimator<T>::from_particles(const xarray_container<T>& pos, const xarray_container<T>& mass, T a) { return {}; }
template <class T> std::pair<xarray_container<T>, xarray_container<T>> power_spectrum_estimator<T>::bin_power(const xarray_container<T>& Pk, size_t nb, bool log) { return {}; }

// gravitational_lensing
template <class T> xarray_container<T> gravitational_lensing<T>::convergence_map(const xarray_container<T>& rho, T bs, T zs, const background_cosmology<T>& bg) { return {}; }
template <class T> std::pair<xarray_container<T>, xarray_container<T>> gravitational_lensing<T>::shear_from_convergence(const xarray_container<T>& kappa) { return {}; }
template <class T> xarray_container<T> gravitational_lensing<T>::multi_plane_lensing(const std::vector<xarray_container<T>>& planes, const std::vector<T>& zs, T zsrc, const background_cosmology<T>& bg) { return {}; }

} // namespace cosmology
} // namespace physics
} // namespace xt

#endif // XTENSOR_XCOSMOLOGY_HPP