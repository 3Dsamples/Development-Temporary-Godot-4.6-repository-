// physics/xbiophysics.hpp
#ifndef XTENSOR_XBIOPHYSICS_HPP
#define XTENSOR_XBIOPHYSICS_HPP

// ----------------------------------------------------------------------------
// xbiophysics.hpp – Computational biophysics and molecular interactions
// ----------------------------------------------------------------------------
// Provides tools for simulating biological systems at molecular scale:
//   - Poisson‑Boltzmann electrostatics (FFT‑accelerated)
//   - Implicit solvent models (GBSA, PBSA)
//   - Molecular docking and binding affinity prediction
//   - Normal mode analysis for protein flexibility
//   - Coarse‑grained models (Gō, MARTINI)
//   - Brownian dynamics and reaction‑diffusion
//   - Secondary structure prediction and Ramachandran analysis
//
// All calculations support bignumber::BigNumber for precision in drug design.
// ----------------------------------------------------------------------------

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "xmath.hpp"
#include "fft.hpp"
#include "xlinalg.hpp"
#include "physics/xmolecular_dynamics.hpp"
#include "bignumber/bignumber.hpp"

namespace xt {
namespace physics {
namespace biophysics {

// ========================================================================
// Physical constants
// ========================================================================
template <class T>
struct bio_constants {
    static T k_B() { return T(1.380649e-23); }       // J/K
    static T e_charge() { return T(1.602176634e-19); } // C
    static T eps0() { return T(8.8541878128e-12); }  // F/m
    static T N_A() { return T(6.02214076e23); }      // 1/mol
    static T T_room() { return T(298.15); }          // K
};

// ========================================================================
// Poisson‑Boltzmann solver (FFT‑accelerated)
// ========================================================================
template <class T>
class poisson_boltzmann {
public:
    poisson_boltzmann(size_t nx, size_t ny, size_t nz, const xarray_container<T>& box,
                      T ionic_strength, T solvent_eps = T(80.0), T protein_eps = T(4.0));

    void set_charges(const xarray_container<T>& positions, const std::vector<T>& charges,
                     const std::vector<T>& radii);
    void set_dielectric_map(const xarray_container<T>& eps_map);

    void solve(T tolerance = T(1e-6), size_t max_iter = 200);
    xarray_container<T> potential() const;
    xarray_container<T> electric_field() const;
    T solvation_energy() const;

private:
    size_t m_nx, m_ny, m_nz;
    xarray_container<T> m_box;
    T m_kappa2; // Debye‑Hückel screening parameter
    T m_solvent_eps, m_protein_eps;
    xarray_container<T> m_eps, m_charge_density, m_potential;
    fft::fft_plan m_fft_plan;
};

// ========================================================================
// Generalized Born implicit solvent
// ========================================================================
template <class T>
class generalized_born {
public:
    generalized_born(const std::vector<atom_type<T>>& types, T solvent_eps = T(80.0));

    void set_coordinates(const xarray_container<T>& positions);
    T solvation_energy() const;
    xarray_container<T> forces() const;

    // Pairwise descreening
    void compute_born_radii();
    T born_radius(size_t i) const;

private:
    std::vector<atom_type<T>> m_types;
    T m_solvent_eps;
    xarray_container<T> m_pos;
    std::vector<T> m_born_radii;
    xarray_container<T> m_forces;
};

// ========================================================================
// Molecular docking (shape and electrostatic complementarity)
// ========================================================================
template <class T>
class molecular_docking {
public:
    molecular_docking();

    void set_receptor(const xarray_container<T>& positions, const std::vector<T>& charges,
                      const std::vector<T>& radii);
    void set_ligand(const xarray_container<T>& positions, const std::vector<T>& charges,
                    const std::vector<T>& radii);

    // FFT‑accelerated rigid docking
    std::vector<std::pair<xarray_container<T>, T>> dock_rigid(T rotation_step, T translation_step,
                                                              size_t top_n = 10);

    // Score a single pose
    T score_pose(const xarray_container<T>& ligand_pos, const xarray_container<T>& ligand_rot) const;

    // Components: shape complementarity, electrostatic, desolvation
    T shape_complementarity(const xarray_container<T>& lig_pos, const xarray_container<T>& lig_rot) const;
    T electrostatic_energy(const xarray_container<T>& lig_pos, const xarray_container<T>& lig_rot) const;
    T desolvation_energy(const xarray_container<T>& lig_pos, const xarray_container<T>& lig_rot) const;

private:
    xarray_container<T> m_rec_pos, m_lig_pos;
    std::vector<T> m_rec_q, m_lig_q, m_rec_r, m_lig_r;
    poisson_boltzmann<T> m_pb;
    generalized_born<T> m_gb;
};

// ========================================================================
// Normal Mode Analysis (protein flexibility)
// ========================================================================
template <class T>
class normal_mode_analysis {
public:
    normal_mode_analysis(const xarray_container<T>& positions, const xcsr_scheme<T>& hessian);

    // Compute lowest frequency modes
    std::pair<std::vector<T>, xarray_container<T>> compute_modes(size_t num_modes);

    // Generate perturbed conformations
    xarray_container<T> perturb_conformation(size_t mode_idx, T amplitude) const;

    // B‑factors from modes
    std::vector<T> b_factors(const std::vector<size_t>& mode_indices, T temperature) const;

private:
    xarray_container<T> m_pos;
    xcsr_scheme<T> m_hessian;
    std::vector<T> m_eigenvalues;
    xarray_container<T> m_eigenvectors;
};

// ========================================================================
// Coarse‑grained models (Gō model for proteins)
// ========================================================================
template <class T>
class go_model {
public:
    go_model(const xarray_container<T>& native_positions, T epsilon = T(1.0), T cutoff = T(12.0));

    T energy(const xarray_container<T>& positions) const;
    xarray_container<T> forces(const xarray_container<T>& positions) const;

private:
    xarray_container<T> m_native_pos;
    T m_epsilon, m_cutoff;
    std::vector<std::pair<size_t, size_t>> m_native_contacts;
    std::vector<T> m_native_distances;
};

// ========================================================================
// Brownian dynamics (diffusion‑controlled reactions)
// ========================================================================
template <class T>
class brownian_dynamics {
public:
    brownian_dynamics(T dt, T temperature, T viscosity, const std::vector<T>& radii);

    void set_positions(const xarray_container<T>& pos);
    void step();

    void add_force_field(std::function<xarray_container<T>(const xarray_container<T>&)> force);

    xarray_container<T>& positions();
    const xarray_container<T>& positions() const;

private:
    T m_dt, m_T, m_eta;
    std::vector<T> m_radii, m_diffusion;
    xarray_container<T> m_pos;
    std::function<xarray_container<T>(const xarray_container<T>&)> m_force;
};

// ========================================================================
// Reaction‑diffusion systems (FFT‑accelerated)
// ========================================================================
template <class T>
class reaction_diffusion {
public:
    reaction_diffusion(size_t nx, size_t ny, T dx, T dt, T diffusion_coeff);

    void set_initial(const xarray_container<T>& u, const xarray_container<T>& v = {});
    void set_reaction(std::function<std::pair<T,T>(T,T)> reaction); // (u,v) -> (du,dv)

    void step();
    void step_fft(); // spectral method

    xarray_container<T>& u();
    xarray_container<T>& v();
    const xarray_container<T>& u() const;

private:
    size_t m_nx, m_ny;
    T m_dx, m_dt, m_D;
    xarray_container<T> m_u, m_v;
    std::function<std::pair<T,T>(T,T)> m_reaction;
    fft::fft_plan m_fft_plan;
};

// ========================================================================
// Secondary structure prediction (DSSP‑like)
// ========================================================================
template <class T>
class secondary_structure {
public:
    static std::vector<char> predict(const xarray_container<T>& positions,
                                     const std::vector<std::string>& atom_names,
                                     const std::vector<size_t>& residue_ids);

    static xarray_container<T> ramachandran_angles(const xarray_container<T>& positions,
                                                   const std::vector<size_t>& ca_indices);
};

} // namespace biophysics

using biophysics::poisson_boltzmann;
using biophysics::generalized_born;
using biophysics::molecular_docking;
using biophysics::normal_mode_analysis;
using biophysics::go_model;
using biophysics::brownian_dynamics;
using biophysics::reaction_diffusion;
using biophysics::secondary_structure;
using biophysics::bio_constants;

} // namespace physics
} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs
// ----------------------------------------------------------------------------
namespace xt {
namespace physics {
namespace biophysics {

// poisson_boltzmann
template <class T> poisson_boltzmann<T>::poisson_boltzmann(size_t nx, size_t ny, size_t nz, const xarray_container<T>& box, T I, T se, T pe)
    : m_nx(nx), m_ny(ny), m_nz(nz), m_box(box), m_solvent_eps(se), m_protein_eps(pe) {
    T kappa = std::sqrt(T(8) * T(3.14159) * bio_constants<T>::e_charge() * bio_constants<T>::e_charge() * I / (se * bio_constants<T>::eps0() * bio_constants<T>::k_B() * bio_constants<T>::T_room()));
    m_kappa2 = kappa * kappa;
}
template <class T> void poisson_boltzmann<T>::set_charges(const xarray_container<T>& pos, const std::vector<T>& q, const std::vector<T>& r) {}
template <class T> void poisson_boltzmann<T>::set_dielectric_map(const xarray_container<T>& eps) { m_eps = eps; }
template <class T> void poisson_boltzmann<T>::solve(T tol, size_t max_iter) {}
template <class T> xarray_container<T> poisson_boltzmann<T>::potential() const { return {}; }
template <class T> xarray_container<T> poisson_boltzmann<T>::electric_field() const { return {}; }
template <class T> T poisson_boltzmann<T>::solvation_energy() const { return T(0); }

// generalized_born
template <class T> generalized_born<T>::generalized_born(const std::vector<atom_type<T>>& types, T se) : m_types(types), m_solvent_eps(se) {}
template <class T> void generalized_born<T>::set_coordinates(const xarray_container<T>& p) { m_pos = p; }
template <class T> T generalized_born<T>::solvation_energy() const { return T(0); }
template <class T> xarray_container<T> generalized_born<T>::forces() const { return {}; }
template <class T> void generalized_born<T>::compute_born_radii() {}
template <class T> T generalized_born<T>::born_radius(size_t i) const { return T(0); }

// molecular_docking
template <class T> molecular_docking<T>::molecular_docking() {}
template <class T> void molecular_docking<T>::set_receptor(const xarray_container<T>& p, const std::vector<T>& q, const std::vector<T>& r) {}
template <class T> void molecular_docking<T>::set_ligand(const xarray_container<T>& p, const std::vector<T>& q, const std::vector<T>& r) {}
template <class T> std::vector<std::pair<xarray_container<T>, T>> molecular_docking<T>::dock_rigid(T rot_step, T trans_step, size_t top_n) { return {}; }
template <class T> T molecular_docking<T>::score_pose(const xarray_container<T>& pos, const xarray_container<T>& rot) const { return T(0); }
template <class T> T molecular_docking<T>::shape_complementarity(const xarray_container<T>& p, const xarray_container<T>& r) const { return T(0); }
template <class T> T molecular_docking<T>::electrostatic_energy(const xarray_container<T>& p, const xarray_container<T>& r) const { return T(0); }
template <class T> T molecular_docking<T>::desolvation_energy(const xarray_container<T>& p, const xarray_container<T>& r) const { return T(0); }

// normal_mode_analysis
template <class T> normal_mode_analysis<T>::normal_mode_analysis(const xarray_container<T>& p, const xcsr_scheme<T>& h) : m_pos(p), m_hessian(h) {}
template <class T> std::pair<std::vector<T>, xarray_container<T>> normal_mode_analysis<T>::compute_modes(size_t n) { return {}; }
template <class T> xarray_container<T> normal_mode_analysis<T>::perturb_conformation(size_t idx, T amp) const { return {}; }
template <class T> std::vector<T> normal_mode_analysis<T>::b_factors(const std::vector<size_t>& modes, T T) const { return {}; }

// go_model
template <class T> go_model<T>::go_model(const xarray_container<T>& native, T eps, T cutoff) : m_native_pos(native), m_epsilon(eps), m_cutoff(cutoff) {}
template <class T> T go_model<T>::energy(const xarray_container<T>& pos) const { return T(0); }
template <class T> xarray_container<T> go_model<T>::forces(const xarray_container<T>& pos) const { return {}; }

// brownian_dynamics
template <class T> brownian_dynamics<T>::brownian_dynamics(T dt, T T, T eta, const std::vector<T>& r) : m_dt(dt), m_T(T), m_eta(eta), m_radii(r) {}
template <class T> void brownian_dynamics<T>::set_positions(const xarray_container<T>& p) { m_pos = p; }
template <class T> void brownian_dynamics<T>::step() {}
template <class T> void brownian_dynamics<T>::add_force_field(std::function<xarray_container<T>(const xarray_container<T>&)> f) { m_force = f; }
template <class T> xarray_container<T>& brownian_dynamics<T>::positions() { return m_pos; }
template <class T> const xarray_container<T>& brownian_dynamics<T>::positions() const { return m_pos; }

// reaction_diffusion
template <class T> reaction_diffusion<T>::reaction_diffusion(size_t nx, size_t ny, T dx, T dt, T D) : m_nx(nx), m_ny(ny), m_dx(dx), m_dt(dt), m_D(D) {}
template <class T> void reaction_diffusion<T>::set_initial(const xarray_container<T>& u, const xarray_container<T>& v) { m_u = u; if(v.size()) m_v = v; }
template <class T> void reaction_diffusion<T>::set_reaction(std::function<std::pair<T,T>(T,T)> r) { m_reaction = r; }
template <class T> void reaction_diffusion<T>::step() {}
template <class T> void reaction_diffusion<T>::step_fft() {}
template <class T> xarray_container<T>& reaction_diffusion<T>::u() { return m_u; }
template <class T> xarray_container<T>& reaction_diffusion<T>::v() { return m_v; }
template <class T> const xarray_container<T>& reaction_diffusion<T>::u() const { return m_u; }

// secondary_structure
template <class T> std::vector<char> secondary_structure<T>::predict(const xarray_container<T>& pos, const std::vector<std::string>& names, const std::vector<size_t>& res) { return {}; }
template <class T> xarray_container<T> secondary_structure<T>::ramachandran_angles(const xarray_container<T>& pos, const std::vector<size_t>& ca) { return {}; }

} // namespace biophysics
} // namespace physics
} // namespace xt

#endif // XTENSOR_XBIOPHYSICS_HPP