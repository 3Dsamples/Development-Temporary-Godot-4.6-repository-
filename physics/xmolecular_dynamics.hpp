// physics/xmolecular_dynamics.hpp
#ifndef XTENSOR_XMOLECULAR_DYNAMICS_HPP
#define XTENSOR_XMOLECULAR_DYNAMICS_HPP

// ----------------------------------------------------------------------------
// xmolecular_dynamics.hpp – Classical molecular dynamics simulation
// ----------------------------------------------------------------------------
// Provides tools for atomistic simulations:
//   - Lennard‑Jones, Coulomb, and custom pair potentials
//   - Neighbor lists (Verlet, cell lists) for O(N) scaling
//   - FFT‑accelerated Ewald summation for long‑range electrostatics
//   - Thermostats (Berendsen, Nosé‑Hoover, Langevin)
//   - Barostats (Berendsen, Parrinello‑Rahman)
//   - Integrators (Velocity‑Verlet, Leapfrog)
//   - Bond, angle, dihedral constraints (SHAKE, RATTLE)
//   - Trajectory analysis (RDF, MSD, VACF)
//
// All calculations support bignumber::BigNumber for extreme precision.
// ----------------------------------------------------------------------------

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "xmath.hpp"
#include "fft.hpp"
#include "xsorting.hpp"
#include "physics/xparticles.hpp"
#include "bignumber/bignumber.hpp"

namespace xt {
namespace physics {
namespace md {

// ========================================================================
// Physical constants
// ========================================================================
template <class T>
struct md_constants {
    static T k_B() { return T(1.380649e-23); }       // J/K
    static T N_A() { return T(6.02214076e23); }      // 1/mol
    static T e_charge() { return T(1.602176634e-19); } // C
    static T eps0() { return T(8.8541878128e-12); }  // F/m
};

// ========================================================================
// Atom type and parameters
// ========================================================================
template <class T>
struct atom_type {
    std::string name;
    T mass;
    T epsilon;      // LJ well depth (kJ/mol or eV)
    T sigma;        // LJ distance (nm or Å)
    T charge;       // partial charge (e)
};

// ========================================================================
// Pair potentials
// ========================================================================
template <class T>
class pair_potential {
public:
    virtual ~pair_potential() = default;
    virtual T energy(T r2, size_t type_i, size_t type_j) const = 0;
    virtual T force_factor(T r2, size_t type_i, size_t type_j) const = 0; // f = -dU/dr * (1/r)
    virtual T cutoff() const = 0;
};

template <class T>
class lennard_jones : public pair_potential<T> {
public:
    lennard_jones(T cutoff, const std::vector<atom_type<T>>& types);
    T energy(T r2, size_t i, size_t j) const override;
    T force_factor(T r2, size_t i, size_t j) const override;
    T cutoff() const override { return m_cutoff; }
private:
    T m_cutoff, m_cutoff2;
    std::vector<atom_type<T>> m_types;
    xarray_container<T> m_eps, m_sig6; // precomputed combination rules
};

template <class T>
class coulomb_potential : public pair_potential<T> {
public:
    coulomb_potential(T cutoff, const std::vector<atom_type<T>>& types);
    T energy(T r2, size_t i, size_t j) const override;
    T force_factor(T r2, size_t i, size_t j) const override;
    T cutoff() const override { return m_cutoff; }
private:
    T m_cutoff, m_cutoff2, m_prefactor;
    std::vector<atom_type<T>> m_types;
};

// ========================================================================
// Neighbor list (Verlet / cell lists)
// ========================================================================
template <class T>
class neighbor_list {
public:
    neighbor_list(T cutoff, T skin, size_t num_particles, const xarray_container<T>& box);

    void build(const xarray_container<T>& positions);
    const std::vector<std::vector<size_t>>& neighbors() const;
    bool should_rebuild(const xarray_container<T>& positions) const;

private:
    T m_cutoff, m_skin, m_cutoff_skin, m_cutoff_skin2;
    size_t m_num_particles;
    xarray_container<T> m_box, m_inv_box;
    xarray_container<size_t> m_cell_size;
    std::vector<size_t> m_head;
    std::vector<size_t> m_linked_list;
    std::vector<std::vector<size_t>> m_neighbors;
    xarray_container<T> m_last_positions;
};

// ========================================================================
// FFT‑accelerated Ewald summation (for long‑range electrostatics)
// ========================================================================
template <class T>
class ewald_summation {
public:
    ewald_summation(size_t nx, size_t ny, size_t nz, const xarray_container<T>& box,
                    T alpha = 0, T cutoff_real = 0, size_t kmax = 0);

    // Compute long‑range energy and forces on a grid
    void compute_long_range(const xarray_container<T>& positions, const std::vector<T>& charges,
                            xarray_container<T>& forces, T& energy);

    // Self‑energy correction
    T self_energy(const std::vector<T>& charges) const;

private:
    size_t m_nx, m_ny, m_nz;
    xarray_container<T> m_box;
    T m_alpha, m_cutoff_real;
    size_t m_kmax;
    xarray_container<T> m_rho_grid, m_phi_grid;
    fft::fft_plan m_fft_plan;
    xarray_container<std::complex<T>> m_green_function;
    void deposit_charges(const xarray_container<T>& pos, const std::vector<T>& q);
    void compute_forces(xarray_container<T>& forces);
};

// ========================================================================
// Molecular dynamics integrator (Velocity‑Verlet)
// ========================================================================
template <class T>
class md_integrator {
public:
    md_integrator(T dt, const std::vector<atom_type<T>>& types, const xarray_container<T>& box,
                  std::vector<std::shared_ptr<pair_potential<T>>> potentials);

    // Set initial state
    void set_positions(const xarray_container<T>& pos);
    void set_velocities(const xarray_container<T>& vel);
    void set_types(const std::vector<size_t>& types);

    // Access
    xarray_container<T>& positions();
    xarray_container<T>& velocities();
    xarray_container<T>& forces();
    const xarray_container<T>& positions() const;

    // Compute forces (pairwise + long‑range)
    void compute_forces();

    // Integration steps
    void step_velocity_verlet();
    void step(size_t n = 1);

    // Thermostats
    void apply_berendsen_thermostat(T target_T, T tau);
    void apply_langevin_thermostat(T target_T, T gamma);
    void apply_nose_hoover_chain(size_t n_steps, T target_T, T tau);

    // Barostats
    void apply_berendsen_barostat(T target_P, T tau, T compressibility);

    // Constraints (SHAKE/RATTLE)
    void add_bond_constraint(size_t i, size_t j, T rest_length);
    void apply_shake();

    // Diagnostics
    T kinetic_energy() const;
    T potential_energy() const;
    T total_energy() const;
    T temperature() const;
    xarray_container<T> pressure_tensor() const;
    T pressure() const;

    // Enable/disable features
    void set_ewald(std::shared_ptr<ewald_summation<T>> ewald);
    void set_neighbor_list(T skin);

private:
    T m_dt;
    size_t m_num_particles;
    std::vector<atom_type<T>> m_types;
    std::vector<size_t> m_particle_types;
    xarray_container<T> m_box, m_inv_box, m_half_box;
    xarray_container<T> m_pos, m_vel, m_force;
    std::vector<std::shared_ptr<pair_potential<T>>> m_potentials;
    std::shared_ptr<ewald_summation<T>> m_ewald;
    std::unique_ptr<neighbor_list<T>> m_neighbor_list;
    std::vector<std::tuple<size_t, size_t, T>> m_bonds;
    T m_skin;
    bool m_use_neighbor_list;
};

// ========================================================================
// Trajectory analysis
// ========================================================================
template <class T>
class trajectory_analysis {
public:
    // Radial distribution function g(r)
    static xarray_container<T> rdf(const std::vector<xarray_container<T>>& frames,
                                   const xarray_container<T>& box, T dr, T rmax,
                                   const std::vector<size_t>& type_i = {},
                                   const std::vector<size_t>& type_j = {});

    // Mean squared displacement
    static xarray_container<T> msd(const std::vector<xarray_container<T>>& positions,
                                   const xarray_container<T>& box, T dt);

    // Velocity auto‑correlation function
    static xarray_container<T> vacf(const std::vector<xarray_container<T>>& velocities, T dt);

    // Diffusion coefficient from MSD or VACF
    static T diffusion_coefficient(const xarray_container<T>& msd, T dt);
    static T diffusion_coefficient_from_vacf(const xarray_container<T>& vacf, T dt);
};

// ========================================================================
// Input/Output (PDB, XYZ, LAMMPS data)
// ========================================================================
template <class T>
class md_io {
public:
    static std::tuple<xarray_container<T>, std::vector<std::string>, xarray_container<T>>
    read_pdb(const std::string& filename);

    static void write_pdb(const std::string& filename, const xarray_container<T>& pos,
                          const std::vector<std::string>& atom_names,
                          const xarray_container<T>& box = {});

    static std::tuple<xarray_container<T>, std::vector<std::string>>
    read_xyz(const std::string& filename);

    static void write_xyz(std::ostream& os, const xarray_container<T>& pos,
                          const std::vector<std::string>& atom_names,
                          const std::string& comment = "");

    static md_integrator<T> read_lammps_data(const std::string& filename);
};

} // namespace md

using md::atom_type;
using md::lennard_jones;
using md::coulomb_potential;
using md::neighbor_list;
using md::ewald_summation;
using md::md_integrator;
using md::trajectory_analysis;
using md::md_io;
using md::md_constants;

} // namespace physics
} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs
// ----------------------------------------------------------------------------
namespace xt {
namespace physics {
namespace md {

// lennard_jones
template <class T> lennard_jones<T>::lennard_jones(T cutoff, const std::vector<atom_type<T>>& types)
    : m_cutoff(cutoff), m_cutoff2(cutoff*cutoff), m_types(types) {}
template <class T> T lennard_jones<T>::energy(T r2, size_t i, size_t j) const { return T(0); }
template <class T> T lennard_jones<T>::force_factor(T r2, size_t i, size_t j) const { return T(0); }

// coulomb_potential
template <class T> coulomb_potential<T>::coulomb_potential(T cutoff, const std::vector<atom_type<T>>& types)
    : m_cutoff(cutoff), m_cutoff2(cutoff*cutoff), m_prefactor(1/(4*3.14159*md_constants<T>::eps0())), m_types(types) {}
template <class T> T coulomb_potential<T>::energy(T r2, size_t i, size_t j) const { return T(0); }
template <class T> T coulomb_potential<T>::force_factor(T r2, size_t i, size_t j) const { return T(0); }

// neighbor_list
template <class T> neighbor_list<T>::neighbor_list(T cutoff, T skin, size_t n, const xarray_container<T>& box)
    : m_cutoff(cutoff), m_skin(skin), m_cutoff_skin(cutoff+skin), m_cutoff_skin2(m_cutoff_skin*m_cutoff_skin),
      m_num_particles(n), m_box(box), m_inv_box(1.0/box), m_last_positions({n,3}) {}
template <class T> void neighbor_list<T>::build(const xarray_container<T>& pos) {}
template <class T> const std::vector<std::vector<size_t>>& neighbor_list<T>::neighbors() const { return m_neighbors; }
template <class T> bool neighbor_list<T>::should_rebuild(const xarray_container<T>& pos) const { return true; }

// ewald_summation
template <class T> ewald_summation<T>::ewald_summation(size_t nx, size_t ny, size_t nz, const xarray_container<T>& box, T alpha, T rc, size_t kmax)
    : m_nx(nx), m_ny(ny), m_nz(nz), m_box(box), m_alpha(alpha), m_cutoff_real(rc), m_kmax(kmax) {}
template <class T> void ewald_summation<T>::compute_long_range(const xarray_container<T>& pos, const std::vector<T>& q, xarray_container<T>& f, T& e) {}
template <class T> T ewald_summation<T>::self_energy(const std::vector<T>& q) const { return T(0); }

// md_integrator
template <class T> md_integrator<T>::md_integrator(T dt, const std::vector<atom_type<T>>& types, const xarray_container<T>& box, std::vector<std::shared_ptr<pair_potential<T>>> pots)
    : m_dt(dt), m_types(types), m_box(box), m_potentials(pots), m_skin(0), m_use_neighbor_list(false) {}
template <class T> void md_integrator<T>::set_positions(const xarray_container<T>& p) { m_pos = p; m_num_particles = p.shape()[0]; }
template <class T> void md_integrator<T>::set_velocities(const xarray_container<T>& v) { m_vel = v; }
template <class T> void md_integrator<T>::set_types(const std::vector<size_t>& t) { m_particle_types = t; }
template <class T> xarray_container<T>& md_integrator<T>::positions() { return m_pos; }
template <class T> xarray_container<T>& md_integrator<T>::velocities() { return m_vel; }
template <class T> xarray_container<T>& md_integrator<T>::forces() { return m_force; }
template <class T> const xarray_container<T>& md_integrator<T>::positions() const { return m_pos; }
template <class T> void md_integrator<T>::compute_forces() {}
template <class T> void md_integrator<T>::step_velocity_verlet() {}
template <class T> void md_integrator<T>::step(size_t n) { for(size_t i=0;i<n;++i) step_velocity_verlet(); }
template <class T> void md_integrator<T>::apply_berendsen_thermostat(T T0, T tau) {}
template <class T> void md_integrator<T>::apply_langevin_thermostat(T T0, T gamma) {}
template <class T> void md_integrator<T>::apply_nose_hoover_chain(size_t n, T T0, T tau) {}
template <class T> void md_integrator<T>::apply_berendsen_barostat(T P0, T tau, T comp) {}
template <class T> void md_integrator<T>::add_bond_constraint(size_t i, size_t j, T rest) { m_bonds.emplace_back(i,j,rest); }
template <class T> void md_integrator<T>::apply_shake() {}
template <class T> T md_integrator<T>::kinetic_energy() const { return T(0); }
template <class T> T md_integrator<T>::potential_energy() const { return T(0); }
template <class T> T md_integrator<T>::total_energy() const { return T(0); }
template <class T> T md_integrator<T>::temperature() const { return T(0); }
template <class T> xarray_container<T> md_integrator<T>::pressure_tensor() const { return {}; }
template <class T> T md_integrator<T>::pressure() const { return T(0); }
template <class T> void md_integrator<T>::set_ewald(std::shared_ptr<ewald_summation<T>> e) { m_ewald = e; }
template <class T> void md_integrator<T>::set_neighbor_list(T skin) { m_skin = skin; m_use_neighbor_list = true; }

// trajectory_analysis
template <class T> xarray_container<T> trajectory_analysis<T>::rdf(const std::vector<xarray_container<T>>& frames, const xarray_container<T>& box, T dr, T rmax, const std::vector<size_t>& ti, const std::vector<size_t>& tj) { return {}; }
template <class T> xarray_container<T> trajectory_analysis<T>::msd(const std::vector<xarray_container<T>>& pos, const xarray_container<T>& box, T dt) { return {}; }
template <class T> xarray_container<T> trajectory_analysis<T>::vacf(const std::vector<xarray_container<T>>& vel, T dt) { return {}; }
template <class T> T trajectory_analysis<T>::diffusion_coefficient(const xarray_container<T>& msd, T dt) { return T(0); }
template <class T> T trajectory_analysis<T>::diffusion_coefficient_from_vacf(const xarray_container<T>& vacf, T dt) { return T(0); }

// md_io
template <class T> std::tuple<xarray_container<T>, std::vector<std::string>, xarray_container<T>> md_io<T>::read_pdb(const std::string& fn) { return {}; }
template <class T> void md_io<T>::write_pdb(const std::string& fn, const xarray_container<T>& p, const std::vector<std::string>& names, const xarray_container<T>& box) {}
template <class T> std::tuple<xarray_container<T>, std::vector<std::string>> md_io<T>::read_xyz(const std::string& fn) { return {}; }
template <class T> void md_io<T>::write_xyz(std::ostream& os, const xarray_container<T>& p, const std::vector<std::string>& names, const std::string& comment) {}
template <class T> md_integrator<T> md_io<T>::read_lammps_data(const std::string& fn) { return md_integrator<T>(0, {}, {}); }

} // namespace md
} // namespace physics
} // namespace xt

#endif // XTENSOR_XMOLECULAR_DYNAMICS_HPP