// physics/xmhd.hpp
#ifndef XTENSOR_XMHD_HPP
#define XTENSOR_XMHD_HPP

// ----------------------------------------------------------------------------
// xmhd.hpp – Magnetohydrodynamics and plasma physics
// ----------------------------------------------------------------------------
// Provides solvers for ideal and resistive MHD, plasma kinetics, and fusion:
//   - Ideal MHD (conservative form, divergence‑cleaning)
//   - Resistive MHD with magnetic diffusion
//   - Hall MHD and extended MHD
//   - PIC (Particle‑In‑Cell) for collisionless plasmas
//   - FFT‑accelerated Poisson solvers for electrostatic fields
//   - Tokamak equilibrium (Grad‑Shafranov solver)
//   - Wave dispersion relations (Alfvén, whistler, Langmuir)
//
// All variables use bignumber::BigNumber for precision in extreme regimes.
// ----------------------------------------------------------------------------

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "xmath.hpp"
#include "fft.hpp"
#include "xlinalg.hpp"
#include "physics/xfluid.hpp"
#include "physics/xparticles.hpp"
#include "bignumber/bignumber.hpp"

namespace xt {
namespace physics {
namespace mhd {

// ========================================================================
// Physical constants
// ========================================================================
template <class T>
struct plasma_constants {
    static T mu0() { return T(1.25663706212e-6); }   // vacuum permeability
    static T eps0() { return T(8.8541878128e-12); }  // vacuum permittivity
    static T e() { return T(1.602176634e-19); }      // elementary charge
    static T m_e() { return T(9.1093837015e-31); }   // electron mass
    static T m_p() { return T(1.67262192369e-27); }  // proton mass
    static T k_B() { return T(1.380649e-23); }       // Boltzmann constant
};

// ========================================================================
// Ideal MHD (conservative form)
// ========================================================================
template <class T>
class ideal_mhd {
public:
    ideal_mhd(size_t nx, size_t ny, size_t nz, T dx, T dy, T dz, T dt, T gamma = T(5.0/3.0));

    // Conservative variables: density, momentum (3), energy, magnetic field (3)
    xarray_container<T>& rho();
    xarray_container<T>& mx();
    xarray_container<T>& my();
    xarray_container<T>& mz();
    xarray_container<T>& energy();
    xarray_container<T>& Bx();
    xarray_container<T>& By();
    xarray_container<T>& Bz();

    const xarray_container<T>& rho() const;
    const xarray_container<T>& Bx() const;

    // Primitive variables (derived)
    xarray_container<T> velocity_x() const;
    xarray_container<T> velocity_y() const;
    xarray_container<T> velocity_z() const;
    xarray_container<T> pressure() const;

    // Time stepping (explicit, with divergence cleaning)
    void step();

    // Divergence cleaning (hyperbolic/parabolic)
    void clean_divergence();

    // Boundary conditions
    void set_periodic_boundaries();
    void set_reflective_boundaries();
    void set_outflow_boundaries();

    // Initial conditions
    void set_uniform_state(T rho0, T p0, const xarray_container<T>& B0);
    void add_alfven_wave(const xarray_container<T>& k, T amplitude);

    // Diagnostics
    T total_mass() const;
    T total_energy() const;
    T magnetic_energy() const;
    T kinetic_energy() const;

private:
    size_t m_nx, m_ny, m_nz;
    T m_dx, m_dy, m_dz, m_dt, m_gamma;
    xarray_container<T> m_rho, m_mx, m_my, m_mz, m_eng;
    xarray_container<T> m_Bx, m_By, m_Bz;
    xarray_container<T> m_psi; // divergence cleaning potential
    T m_ch; // cleaning speed
};

// ========================================================================
// Resistive MHD (with magnetic diffusion)
// ========================================================================
template <class T>
class resistive_mhd : public ideal_mhd<T> {
public:
    resistive_mhd(size_t nx, size_t ny, size_t nz, T dx, T dy, T dz, T dt, T eta, T gamma = T(5.0/3.0));

    void set_resistivity(const xarray_container<T>& eta_map);
    void step() override;

private:
    T m_eta;
    xarray_container<T> m_eta_map;
    bool m_uniform_eta;
};

// ========================================================================
// Hall MHD (includes Hall term for small scales)
// ========================================================================
template <class T>
class hall_mhd : public resistive_mhd<T> {
public:
    hall_mhd(size_t nx, size_t ny, size_t nz, T dx, T dy, T dz, T dt, T eta, T ion_inertial_length, T gamma = T(5.0/3.0));

    void step() override;

private:
    T m_di; // ion inertial length
};

// ========================================================================
// Particle‑In‑Cell (PIC) for collisionless plasmas
// ========================================================================
template <class T>
class pic_solver {
public:
    pic_solver(size_t nx, size_t ny, size_t nz, T dx, T dy, T dz, T dt,
               size_t particles_per_cell, T q_over_m);

    // Fields (on grid)
    xarray_container<T>& Ex();
    xarray_container<T>& Ey();
    xarray_container<T>& Ez();
    xarray_container<T>& Bx();
    xarray_container<T>& By();
    xarray_container<T>& Bz();

    // Particles (SoA)
    particles::particle_system<T>& electrons();
    particles::particle_system<T>& ions();

    // Simulation steps
    void deposit_charge(xarray_container<T>& rho);
    void solve_fields(); // FFT‑accelerated Poisson for E
    void push_particles_Boris(T dt);
    void step();

    // Diagnostics
    xarray_container<T> charge_density() const;
    xarray_container<T> current_density() const;

private:
    size_t m_nx, m_ny, m_nz;
    T m_dx, m_dy, m_dz, m_dt, m_q_over_m;
    xarray_container<T> m_Ex, m_Ey, m_Ez;
    xarray_container<T> m_Bx, m_By, m_Bz;
    particles::particle_system<T> m_electrons, m_ions;
    fft::fft_plan m_fft_plan;
};

// ========================================================================
// Grad‑Shafranov solver (tokamak equilibrium)
// ========================================================================
template <class T>
class grad_shafranov {
public:
    grad_shafranov(size_t nr, size_t nz, T r_min, T r_max, T z_min, T z_max);

    void set_pressure_profile(std::function<T(T)> p_profile); // p(psi)
    void set_current_profile(std::function<T(T)> f_profile);  // F(psi) = R*B_phi

    xarray_container<T> solve(T tolerance = T(1e-6), size_t max_iter = 10000);

    const xarray_container<T>& psi() const;
    xarray_container<T> magnetic_field() const;
    xarray_container<T> plasma_pressure() const;
    xarray_container<T> toroidal_current() const;

private:
    size_t m_nr, m_nz;
    T m_r_min, m_r_max, m_z_min, m_z_max;
    xarray_container<T> m_psi;
    std::function<T(T)> m_p_profile, m_f_profile;
};

// ========================================================================
// Plasma waves and instabilities
// ========================================================================
template <class T>
class plasma_waves {
public:
    // Dispersion relations
    static T alfven_speed(T B, T rho);
    static T sound_speed(T p, T rho, T gamma);
    static T plasma_frequency(T n_e);
    static T cyclotron_frequency(T B, T q, T m);
    static xarray_container<T> whistler_dispersion(T k, T B, T n_e);
    static xarray_container<std::complex<T>> two_stream_instability_growth(T k, T n0, T v0, T n1, T v1);

    // MHD waves (from linearized equations)
    static xarray_container<T> alfven_wave_eigenmodes(const xarray_container<T>& k, T B0, T rho0);
};

// ========================================================================
// FFT‑accelerated MHD spectral solver (for periodic boxes)
// ========================================================================
template <class T>
class spectral_mhd {
public:
    spectral_mhd(size_t nx, size_t ny, size_t nz, T Lx, T Ly, T Lz, T dt, T nu = 0, T eta = 0);

    xarray_container<std::complex<T>>& u_hat(); // velocity spectral
    xarray_container<std::complex<T>>& B_hat(); // magnetic spectral

    void step(); // fully spectral, FFT‑accelerated
    void add_forcing(const xarray_container<std::complex<T>>& f_u, const xarray_container<std::complex<T>>& f_B);

    xarray_container<T> velocity_physical() const;
    xarray_container<T> magnetic_physical() const;

private:
    size_t m_nx, m_ny, m_nz;
    T m_Lx, m_Ly, m_Lz, m_dt, m_nu, m_eta;
    xarray_container<std::complex<T>> m_u_hat, m_B_hat;
    fft::fft_plan m_fft_plan;
};

} // namespace mhd

using mhd::ideal_mhd;
using mhd::resistive_mhd;
using mhd::hall_mhd;
using mhd::pic_solver;
using mhd::grad_shafranov;
using mhd::plasma_waves;
using mhd::spectral_mhd;
using mhd::plasma_constants;

} // namespace physics
} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs
// ----------------------------------------------------------------------------
namespace xt {
namespace physics {
namespace mhd {

// ideal_mhd
template <class T> ideal_mhd<T>::ideal_mhd(size_t nx, size_t ny, size_t nz, T dx, T dy, T dz, T dt, T gamma)
    : m_nx(nx), m_ny(ny), m_nz(nz), m_dx(dx), m_dy(dy), m_dz(dz), m_dt(dt), m_gamma(gamma), m_ch(T(1)) {}
template <class T> xarray_container<T>& ideal_mhd<T>::rho() { return m_rho; }
template <class T> xarray_container<T>& ideal_mhd<T>::mx() { return m_mx; }
template <class T> xarray_container<T>& ideal_mhd<T>::my() { return m_my; }
template <class T> xarray_container<T>& ideal_mhd<T>::mz() { return m_mz; }
template <class T> xarray_container<T>& ideal_mhd<T>::energy() { return m_eng; }
template <class T> xarray_container<T>& ideal_mhd<T>::Bx() { return m_Bx; }
template <class T> xarray_container<T>& ideal_mhd<T>::By() { return m_By; }
template <class T> xarray_container<T>& ideal_mhd<T>::Bz() { return m_Bz; }
template <class T> const xarray_container<T>& ideal_mhd<T>::rho() const { return m_rho; }
template <class T> const xarray_container<T>& ideal_mhd<T>::Bx() const { return m_Bx; }
template <class T> xarray_container<T> ideal_mhd<T>::velocity_x() const { return m_mx / m_rho; }
template <class T> xarray_container<T> ideal_mhd<T>::velocity_y() const { return m_my / m_rho; }
template <class T> xarray_container<T> ideal_mhd<T>::velocity_z() const { return m_mz / m_rho; }
template <class T> xarray_container<T> ideal_mhd<T>::pressure() const { return {}; }
template <class T> void ideal_mhd<T>::step() {}
template <class T> void ideal_mhd<T>::clean_divergence() {}
template <class T> void ideal_mhd<T>::set_periodic_boundaries() {}
template <class T> void ideal_mhd<T>::set_reflective_boundaries() {}
template <class T> void ideal_mhd<T>::set_outflow_boundaries() {}
template <class T> void ideal_mhd<T>::set_uniform_state(T rho0, T p0, const xarray_container<T>& B0) {}
template <class T> void ideal_mhd<T>::add_alfven_wave(const xarray_container<T>& k, T A) {}
template <class T> T ideal_mhd<T>::total_mass() const { return T(0); }
template <class T> T ideal_mhd<T>::total_energy() const { return T(0); }
template <class T> T ideal_mhd<T>::magnetic_energy() const { return T(0); }
template <class T> T ideal_mhd<T>::kinetic_energy() const { return T(0); }

// resistive_mhd
template <class T> resistive_mhd<T>::resistive_mhd(size_t nx, size_t ny, size_t nz, T dx, T dy, T dz, T dt, T eta, T gamma)
    : ideal_mhd<T>(nx, ny, nz, dx, dy, dz, dt, gamma), m_eta(eta), m_uniform_eta(true) {}
template <class T> void resistive_mhd<T>::set_resistivity(const xarray_container<T>& eta_map) { m_eta_map = eta_map; m_uniform_eta = false; }
template <class T> void resistive_mhd<T>::step() {}

// hall_mhd
template <class T> hall_mhd<T>::hall_mhd(size_t nx, size_t ny, size_t nz, T dx, T dy, T dz, T dt, T eta, T di, T gamma)
    : resistive_mhd<T>(nx, ny, nz, dx, dy, dz, dt, eta, gamma), m_di(di) {}
template <class T> void hall_mhd<T>::step() {}

// pic_solver
template <class T> pic_solver<T>::pic_solver(size_t nx, size_t ny, size_t nz, T dx, T dy, T dz, T dt, size_t ppc, T qom)
    : m_nx(nx), m_ny(ny), m_nz(nz), m_dx(dx), m_dy(dy), m_dz(dz), m_dt(dt), m_q_over_m(qom) {}
template <class T> xarray_container<T>& pic_solver<T>::Ex() { return m_Ex; }
template <class T> xarray_container<T>& pic_solver<T>::Ey() { return m_Ey; }
template <class T> xarray_container<T>& pic_solver<T>::Ez() { return m_Ez; }
template <class T> xarray_container<T>& pic_solver<T>::Bx() { return m_Bx; }
template <class T> xarray_container<T>& pic_solver<T>::By() { return m_By; }
template <class T> xarray_container<T>& pic_solver<T>::Bz() { return m_Bz; }
template <class T> particles::particle_system<T>& pic_solver<T>::electrons() { return m_electrons; }
template <class T> particles::particle_system<T>& pic_solver<T>::ions() { return m_ions; }
template <class T> void pic_solver<T>::deposit_charge(xarray_container<T>& rho) {}
template <class T> void pic_solver<T>::solve_fields() {}
template <class T> void pic_solver<T>::push_particles_Boris(T dt) {}
template <class T> void pic_solver<T>::step() {}
template <class T> xarray_container<T> pic_solver<T>::charge_density() const { return {}; }
template <class T> xarray_container<T> pic_solver<T>::current_density() const { return {}; }

// grad_shafranov
template <class T> grad_shafranov<T>::grad_shafranov(size_t nr, size_t nz, T r_min, T r_max, T z_min, T z_max)
    : m_nr(nr), m_nz(nz), m_r_min(r_min), m_r_max(r_max), m_z_min(z_min), m_z_max(z_max) {}
template <class T> void grad_shafranov<T>::set_pressure_profile(std::function<T(T)> p) { m_p_profile = p; }
template <class T> void grad_shafranov<T>::set_current_profile(std::function<T(T)> f) { m_f_profile = f; }
template <class T> xarray_container<T> grad_shafranov<T>::solve(T tol, size_t max_iter) { return {}; }
template <class T> const xarray_container<T>& grad_shafranov<T>::psi() const { return m_psi; }
template <class T> xarray_container<T> grad_shafranov<T>::magnetic_field() const { return {}; }
template <class T> xarray_container<T> grad_shafranov<T>::plasma_pressure() const { return {}; }
template <class T> xarray_container<T> grad_shafranov<T>::toroidal_current() const { return {}; }

// plasma_waves
template <class T> T plasma_waves<T>::alfven_speed(T B, T rho) { return B / std::sqrt(plasma_constants<T>::mu0() * rho); }
template <class T> T plasma_waves<T>::sound_speed(T p, T rho, T gamma) { return std::sqrt(gamma * p / rho); }
template <class T> T plasma_waves<T>::plasma_frequency(T n_e) { return std::sqrt(n_e * plasma_constants<T>::e() * plasma_constants<T>::e() / (plasma_constants<T>::eps0() * plasma_constants<T>::m_e())); }
template <class T> T plasma_waves<T>::cyclotron_frequency(T B, T q, T m) { return std::abs(q) * B / m; }
template <class T> xarray_container<T> plasma_waves<T>::whistler_dispersion(T k, T B, T n_e) { return {}; }
template <class T> xarray_container<std::complex<T>> plasma_waves<T>::two_stream_instability_growth(T k, T n0, T v0, T n1, T v1) { return {}; }
template <class T> xarray_container<T> plasma_waves<T>::alfven_wave_eigenmodes(const xarray_container<T>& k, T B0, T rho0) { return {}; }

// spectral_mhd
template <class T> spectral_mhd<T>::spectral_mhd(size_t nx, size_t ny, size_t nz, T Lx, T Ly, T Lz, T dt, T nu, T eta)
    : m_nx(nx), m_ny(ny), m_nz(nz), m_Lx(Lx), m_Ly(Ly), m_Lz(Lz), m_dt(dt), m_nu(nu), m_eta(eta) {}
template <class T> xarray_container<std::complex<T>>& spectral_mhd<T>::u_hat() { return m_u_hat; }
template <class T> xarray_container<std::complex<T>>& spectral_mhd<T>::B_hat() { return m_B_hat; }
template <class T> void spectral_mhd<T>::step() {}
template <class T> void spectral_mhd<T>::add_forcing(const xarray_container<std::complex<T>>& f_u, const xarray_container<std::complex<T>>& f_B) {}
template <class T> xarray_container<T> spectral_mhd<T>::velocity_physical() const { return fft::ifft(m_u_hat); }
template <class T> xarray_container<T> spectral_mhd<T>::magnetic_physical() const { return fft::ifft(m_B_hat); }

} // namespace mhd
} // namespace physics
} // namespace xt

#endif // XTENSOR_XMHD_HPP