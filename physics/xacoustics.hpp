// physics/xacoustics.hpp
#ifndef XTENSOR_XACOUSTICS_HPP
#define XTENSOR_XACOUSTICS_HPP

// ----------------------------------------------------------------------------
// xacoustics.hpp – Acoustic wave simulation and signal processing
// ----------------------------------------------------------------------------
// Provides tools for simulating acoustic wave propagation:
//   - Linear wave equation solvers (FDTD, pseudospectral, k‑space)
//   - Frequency‑domain Helmholtz solver (FFT‑accelerated)
//   - Ray‑tracing acoustics for room impulse response
//   - Boundary conditions (absorbing, reflecting, PML)
//   - Material properties (impedance, absorption coefficients)
//   - Sound source models (monopole, dipole, plane wave)
//   - Microphone array processing and beamforming
//
// All calculations support BigNumber for precision and FFT for speed.
// ----------------------------------------------------------------------------

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "xmath.hpp"
#include "fft.hpp"
#include "bignumber/bignumber.hpp"

namespace xt {
namespace physics {
namespace acoustics {

// ========================================================================
// Physical constants
// ========================================================================
template <class T>
struct acoustic_constants {
    static T c_air() { return T(343.0); }           // speed of sound in air at 20°C (m/s)
    static T c_water() { return T(1484.0); }        // speed of sound in water (m/s)
    static T rho_air() { return T(1.204); }         // air density (kg/m³)
    static T rho_water() { return T(1000.0); }      // water density (kg/m³)
    static T reference_pressure() { return T(20e-6); } // dB SPL reference (Pa)
};

// ========================================================================
// Wave equation solver (FDTD)
// ========================================================================
template <class T>
class fdtd_acoustic {
public:
    // 1D, 2D, or 3D grid
    fdtd_acoustic(const shape_type& grid_shape, T dx, T dt, T c0 = acoustic_constants<T>::c_air());

    // Pressure field (scalar)
    xarray_container<T>& pressure();
    const xarray_container<T>& pressure() const;

    // Particle velocity fields (vector)
    std::vector<xarray_container<T>>& velocity();
    const std::vector<xarray_container<T>>& velocity() const;

    // Update step (explicit)
    void step();

    // Source injection
    void add_source(const std::vector<size_t>& location, T amplitude, T frequency, T phase = 0);
    void clear_sources();

    // Boundary conditions
    void set_reflective_boundaries();
    void set_absorbing_boundaries(T absorption_coefficient = 1.0);
    void set_pml_boundaries(size_t num_layers);

    // Material heterogeneity
    void set_sound_speed(const xarray_container<T>& c_map);
    void set_density(const xarray_container<T>& rho_map);

    // Probes
    T probe_pressure(const std::vector<size_t>& location) const;
    std::vector<T> probe_pressure_history(const std::vector<size_t>& location) const;
    void record_probe(const std::vector<size_t>& location);
    void stop_recording();

private:
    shape_type m_shape;
    size_t m_dim;
    T m_dx, m_dt, m_c0;
    xarray_container<T> m_p, m_p_prev;
    std::vector<xarray_container<T>> m_v;
    xarray_container<T> m_c_map, m_rho_map;
    std::vector<std::vector<T>> m_probe_history;
    std::vector<std::vector<size_t>> m_probe_locs;
};

// ========================================================================
// Pseudospectral k‑space method (highly accurate for large timesteps)
// ========================================================================
template <class T>
class kspace_solver {
public:
    kspace_solver(const shape_type& grid_shape, T dx, T dt, T c0 = acoustic_constants<T>::c_air());

    xarray_container<T>& pressure();
    const xarray_container<T>& pressure() const;

    void step();
    void add_source(const std::vector<size_t>& location, T amplitude);
    void set_sound_speed(const xarray_container<T>& c_map);

private:
    shape_type m_shape;
    T m_dx, m_dt, m_c0;
    xarray_container<T> m_p, m_p_prev;
    xarray_container<std::complex<T>> m_kappa;
    fft::fft_plan m_fft_plan;
};

// ========================================================================
// Frequency‑domain Helmholtz solver (FFT‑accelerated)
// ========================================================================
template <class T>
class helmholtz_acoustic {
public:
    helmholtz_acoustic(const shape_type& grid_shape, T dx);

    void set_frequency(T f, T c0 = acoustic_constants<T>::c_air());
    void set_source(const xarray_container<T>& source_distribution);
    void set_sound_speed(const xarray_container<T>& c_map);

    xarray_container<std::complex<T>> solve();

private:
    shape_type m_shape;
    T m_dx, m_k;
    xarray_container<T> m_c_map;
    xarray_container<T> m_source;
    fft::fft_plan m_fft_plan;
};

// ========================================================================
// Ray‑tracing acoustics (room impulse response)
// ========================================================================
template <class T>
class acoustic_ray_tracer {
public:
    struct ray {
        xarray_container<T> origin;
        xarray_container<T> direction;
        T energy;
        size_t bounce_count;
    };

    acoustic_ray_tracer();

    void set_room_dimensions(const xarray_container<T>& min_corner,
                             const xarray_container<T>& max_corner);
    void set_absorption(const xarray_container<T>& wall_absorption); // 6 walls

    void add_source(const xarray_container<T>& position, T power);
    void add_receiver(const xarray_container<T>& position);

    void trace_rays(size_t num_rays, size_t max_bounces);
    xarray_container<T> impulse_response(T time_resolution, T max_time) const;

private:
    xarray_container<T> m_min, m_max;
    xarray_container<T> m_absorption;
    xarray_container<T> m_source_pos, m_receiver_pos;
    std::vector<std::pair<T, T>> m_impulse_hits; // (time, energy)
};

// ========================================================================
// Beamforming and array processing
// ========================================================================
template <class T>
class beamformer {
public:
    beamformer(const xarray_container<T>& array_positions, T sound_speed);

    // Delay‑and‑sum beamforming
    xarray_container<std::complex<T>> steered_response(const xarray_container<T>& steering_angles,
                                                       const std::vector<xarray_container<T>>& sensor_signals,
                                                       T frequency);

    // MVDR (Capon) adaptive beamforming
    xarray_container<std::complex<T>> mvdr_response(const xarray_container<T>& steering_angles,
                                                    const std::vector<xarray_container<T>>& sensor_signals,
                                                    T frequency, T diagonal_loading = T(0.01));

    // Beam pattern for given frequency
    xarray_container<T> beam_pattern(T frequency, const xarray_container<T>& angles) const;

private:
    xarray_container<T> m_array_pos; // (M, 3)
    T m_c;
    size_t m_num_sensors;
};

} // namespace acoustics

using acoustics::fdtd_acoustic;
using acoustics::kspace_solver;
using acoustics::helmholtz_acoustic;
using acoustics::acoustic_ray_tracer;
using acoustics::beamformer;
using acoustics::acoustic_constants;

} // namespace physics
} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs
// ----------------------------------------------------------------------------
namespace xt {
namespace physics {
namespace acoustics {

// fdtd_acoustic
template <class T> fdtd_acoustic<T>::fdtd_acoustic(const shape_type& s, T dx, T dt, T c0) : m_shape(s), m_dim(s.size()), m_dx(dx), m_dt(dt), m_c0(c0) {}
template <class T> xarray_container<T>& fdtd_acoustic<T>::pressure() { return m_p; }
template <class T> const xarray_container<T>& fdtd_acoustic<T>::pressure() const { return m_p; }
template <class T> std::vector<xarray_container<T>>& fdtd_acoustic<T>::velocity() { return m_v; }
template <class T> const std::vector<xarray_container<T>>& fdtd_acoustic<T>::velocity() const { return m_v; }
template <class T> void fdtd_acoustic<T>::step() {}
template <class T> void fdtd_acoustic<T>::add_source(const std::vector<size_t>& loc, T amp, T f, T ph) {}
template <class T> void fdtd_acoustic<T>::clear_sources() {}
template <class T> void fdtd_acoustic<T>::set_reflective_boundaries() {}
template <class T> void fdtd_acoustic<T>::set_absorbing_boundaries(T alpha) {}
template <class T> void fdtd_acoustic<T>::set_pml_boundaries(size_t n) {}
template <class T> void fdtd_acoustic<T>::set_sound_speed(const xarray_container<T>& c) { m_c_map = c; }
template <class T> void fdtd_acoustic<T>::set_density(const xarray_container<T>& r) { m_rho_map = r; }
template <class T> T fdtd_acoustic<T>::probe_pressure(const std::vector<size_t>& loc) const { return T(0); }
template <class T> std::vector<T> fdtd_acoustic<T>::probe_pressure_history(const std::vector<size_t>& loc) const { return {}; }
template <class T> void fdtd_acoustic<T>::record_probe(const std::vector<size_t>& loc) {}
template <class T> void fdtd_acoustic<T>::stop_recording() {}

// kspace_solver
template <class T> kspace_solver<T>::kspace_solver(const shape_type& s, T dx, T dt, T c0) : m_shape(s), m_dx(dx), m_dt(dt), m_c0(c0) {}
template <class T> xarray_container<T>& kspace_solver<T>::pressure() { return m_p; }
template <class T> const xarray_container<T>& kspace_solver<T>::pressure() const { return m_p; }
template <class T> void kspace_solver<T>::step() {}
template <class T> void kspace_solver<T>::add_source(const std::vector<size_t>& loc, T amp) {}
template <class T> void kspace_solver<T>::set_sound_speed(const xarray_container<T>& c) { m_c_map = c; }

// helmholtz_acoustic
template <class T> helmholtz_acoustic<T>::helmholtz_acoustic(const shape_type& s, T dx) : m_shape(s), m_dx(dx), m_k(0) {}
template <class T> void helmholtz_acoustic<T>::set_frequency(T f, T c0) { m_k = T(2) * T(3.1415926535) * f / c0; }
template <class T> void helmholtz_acoustic<T>::set_source(const xarray_container<T>& s) { m_source = s; }
template <class T> void helmholtz_acoustic<T>::set_sound_speed(const xarray_container<T>& c) { m_c_map = c; }
template <class T> xarray_container<std::complex<T>> helmholtz_acoustic<T>::solve() { return {}; }

// acoustic_ray_tracer
template <class T> acoustic_ray_tracer<T>::acoustic_ray_tracer() {}
template <class T> void acoustic_ray_tracer<T>::set_room_dimensions(const xarray_container<T>& min, const xarray_container<T>& max) { m_min = min; m_max = max; }
template <class T> void acoustic_ray_tracer<T>::set_absorption(const xarray_container<T>& a) { m_absorption = a; }
template <class T> void acoustic_ray_tracer<T>::add_source(const xarray_container<T>& pos, T power) { m_source_pos = pos; }
template <class T> void acoustic_ray_tracer<T>::add_receiver(const xarray_container<T>& pos) { m_receiver_pos = pos; }
template <class T> void acoustic_ray_tracer<T>::trace_rays(size_t num_rays, size_t max_bounces) {}
template <class T> xarray_container<T> acoustic_ray_tracer<T>::impulse_response(T dt, T tmax) const { return {}; }

// beamformer
template <class T> beamformer<T>::beamformer(const xarray_container<T>& pos, T c) : m_array_pos(pos), m_c(c), m_num_sensors(pos.shape()[0]) {}
template <class T> xarray_container<std::complex<T>> beamformer<T>::steered_response(const xarray_container<T>& angles, const std::vector<xarray_container<T>>& signals, T f) { return {}; }
template <class T> xarray_container<std::complex<T>> beamformer<T>::mvdr_response(const xarray_container<T>& angles, const std::vector<xarray_container<T>>& signals, T f, T dl) { return {}; }
template <class T> xarray_container<T> beamformer<T>::beam_pattern(T f, const xarray_container<T>& angles) const { return {}; }

} // namespace acoustics
} // namespace physics
} // namespace xt

#endif // XTENSOR_XACOUSTICS_HPP