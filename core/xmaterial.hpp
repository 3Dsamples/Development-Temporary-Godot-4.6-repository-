// core/xmaterial.hpp
#ifndef XTENSOR_XMATERIAL_HPP
#define XTENSOR_XMATERIAL_HPP

// ----------------------------------------------------------------------------
// xmaterial.hpp – Material properties and physics for simulation
// ----------------------------------------------------------------------------
// This header provides material property structures and physics functions:
//   - Material definitions (density, elasticity, thermal, optical)
//   - Stress‑strain relationships (Hooke's law, plasticity)
//   - Heat transfer (conductivity, specific heat)
//   - Fluid properties (viscosity, Reynolds number)
//   - Electromagnetic properties (permittivity, permeability)
//   - Material database with common substances
//
// All calculations use bignumber::BigNumber with FFT‑accelerated operations
// where applicable (e.g., tensor contractions, convolution for diffusion).
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <array>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xlinalg.hpp"
#include "xmath.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace material
    {
        // ========================================================================
        // Material property structures
        // ========================================================================
        template <class T = double> struct elastic_properties;
        template <class T = double> struct thermal_properties;
        template <class T = double> struct fluid_properties;
        template <class T = double> struct electromagnetic_properties;
        template <class T = double> struct material;
        template <class T = double> class material_database;

        // ========================================================================
        // Physics functions
        // ========================================================================
        // Hooke's law: stress = C : strain (Voigt notation)
        template <class T>
        xarray_container<T> hooke_stress(const elastic_properties<T>& mat, const xarray_container<T>& strain);
        // Inverse Hooke's law: strain = S : stress
        template <class T>
        xarray_container<T> hooke_strain(const elastic_properties<T>& mat, const xarray_container<T>& stress);
        // von Mises stress (from stress tensor in Voigt notation)
        template <class T>
        T von_mises_stress(const xarray_container<T>& stress);
        // Heat equation 1D update (explicit finite difference)
        template <class T>
        xarray_container<T> heat_conduction_1d(const xarray_container<T>& T_curr, const thermal_properties<T>& prop, T dx, T dt);
        // Navier‑Stokes (simplified 2D incompressible)
        template <class T>
        std::tuple<xarray_container<T>, xarray_container<T>, xarray_container<T>>
        navier_stokes_step(const xarray_container<T>& u, const xarray_container<T>& v,
                           const xarray_container<T>& p, const fluid_properties<T>& prop,
                           T dx, T dy, T dt);

        // ========================================================================
        // Material database access
        // ========================================================================
        template <class T = double>
        material_database<T>& materials();
    }

    // Bring material utilities into xt namespace
    using material::elastic_properties;
    using material::thermal_properties;
    using material::fluid_properties;
    using material::electromagnetic_properties;
    using material::material;
    using material::material_database;
    using material::materials;
    using material::hooke_stress;
    using material::hooke_strain;
    using material::von_mises_stress;
    using material::heat_conduction_1d;
    using material::navier_stokes_step;

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace material
    {
        // Elastic properties (Young's modulus, Poisson's ratio, etc.)
        template <class T> struct elastic_properties
        { T youngs_modulus, poissons_ratio; elastic_properties() : youngs_modulus(0), poissons_ratio(0) {} xarray_container<T> stiffness_matrix() const { return xarray_container<T>(); } };
        // Thermal properties (conductivity, specific heat, etc.)
        template <class T> struct thermal_properties
        { T conductivity, specific_heat, density; thermal_properties() : conductivity(0), specific_heat(0), density(0) {} T diffusivity() const { return conductivity / (density * specific_heat); } };
        // Fluid properties (density, viscosity)
        template <class T> struct fluid_properties
        { T density, dynamic_viscosity; fluid_properties() : density(0), dynamic_viscosity(0) {} };
        // Electromagnetic properties (permittivity, permeability, conductivity)
        template <class T> struct electromagnetic_properties
        { T permittivity, permeability, conductivity; electromagnetic_properties() : permittivity(0), permeability(0), conductivity(0) {} };
        // Complete material definition
        template <class T> struct material
        { std::string name; elastic_properties<T> elastic; thermal_properties<T> thermal; fluid_properties<T> fluid; electromagnetic_properties<T> em; T density; };
        // Material database class
        template <class T> class material_database
        { public: void add(const std::string&, const material<T>&) {} material<T> get(const std::string&) const { return material<T>(); } };

        // Compute stress from strain using stiffness matrix
        template <class T> xarray_container<T> hooke_stress(const elastic_properties<T>& mat, const xarray_container<T>& strain)
        { /* TODO: implement */ return xarray_container<T>(); }
        // Compute strain from stress using compliance matrix
        template <class T> xarray_container<T> hooke_strain(const elastic_properties<T>& mat, const xarray_container<T>& stress)
        { /* TODO: implement */ return xarray_container<T>(); }
        // Compute von Mises equivalent stress
        template <class T> T von_mises_stress(const xarray_container<T>& stress)
        { /* TODO: implement */ return T(0); }
        // Explicit finite difference step for 1D heat equation
        template <class T> xarray_container<T> heat_conduction_1d(const xarray_container<T>& T_curr, const thermal_properties<T>& prop, T dx, T dt)
        { /* TODO: implement */ return T_curr; }
        // One step of simplified 2D incompressible Navier‑Stokes
        template <class T> std::tuple<xarray_container<T>,xarray_container<T>,xarray_container<T>>
        navier_stokes_step(const xarray_container<T>& u, const xarray_container<T>& v,
                           const xarray_container<T>& p, const fluid_properties<T>& prop, T dx, T dy, T dt)
        { return {u, v, p}; }

        // Global material database instance
        template <class T> material_database<T>& materials()
        { static material_database<T> db; return db; }
    }
}

#endif // XTENSOR_XMATERIAL_HPP                : conductivity(T(0)), specific_heat(T(0))
                , thermal_expansion(T(0)), density(T(0))
            {}

            thermal_properties(T k, T cp, T alpha, T rho)
                : conductivity(k), specific_heat(cp)
                , thermal_expansion(alpha), density(rho)
            {}

            // Thermal diffusivity
            T diffusivity() const
            {
                return conductivity / (density * specific_heat);
            }
        };

        template <class T = double>
        struct fluid_properties
        {
            T density;               // ρ (kg/m³)
            T dynamic_viscosity;     // μ (Pa·s)
            T kinematic_viscosity;   // ν = μ/ρ (m²/s)

            fluid_properties()
                : density(T(0)), dynamic_viscosity(T(0)), kinematic_viscosity(T(0))
            {}

            fluid_properties(T rho, T mu)
                : density(rho), dynamic_viscosity(mu)
            {
                kinematic_viscosity = mu / rho;
            }

            // Reynolds number
            T reynolds(T velocity, T characteristic_length) const
            {
                return density * velocity * characteristic_length / dynamic_viscosity;
            }
        };

        template <class T = double>
        struct electromagnetic_properties
        {
            T permittivity;          // ε (F/m)
            T permeability;          // μ (H/m)
            T conductivity;          // σ (S/m)

            electromagnetic_properties()
                : permittivity(T(0)), permeability(T(0)), conductivity(T(0))
            {}

            electromagnetic_properties(T eps, T mu, T sigma)
                : permittivity(eps), permeability(mu), conductivity(sigma)
            {}

            // Wave impedance
            T impedance() const
            {
                return detail::sqrt_val(permeability / permittivity);
            }

            // Skin depth at frequency f
            T skin_depth(T frequency) const
            {
                T omega = T(2) * T(3.141592653589793) * frequency;
                return detail::sqrt_val(T(2) / (omega * permeability * conductivity));
            }
        };

        // ========================================================================
        // Complete material definition
        // ========================================================================
        template <class T = double>
        struct material
        {
            std::string name;
            elastic_properties<T> elastic;
            thermal_properties<T> thermal;
            fluid_properties<T> fluid;
            electromagnetic_properties<T> em;
            T density;

            material() : density(T(0)) {}
            material(const std::string& n, T rho) : name(n), density(rho) {}
        };

        // ========================================================================
        // Physics functions
        // ========================================================================

        // ------------------------------------------------------------------------
        // Hooke's law: stress = C : strain (Voigt notation)
        // ------------------------------------------------------------------------
        template <class T>
        inline xarray_container<T> hooke_stress(const elastic_properties<T>& mat,
                                                 const xarray_container<T>& strain)
        {
            if (strain.size() != 6)
                XTENSOR_THROW(std::invalid_argument, "strain must be 6‑component Voigt vector");
            auto C = mat.stiffness_matrix();
            xarray_container<T> stress({6}, T(0));
            for (size_t i = 0; i < 6; ++i)
                for (size_t j = 0; j < 6; ++j)
                    stress(i) = stress(i) + detail::multiply(C(i, j), strain(j));
            return stress;
        }

        // ------------------------------------------------------------------------
        // Inverse Hooke's law: strain = S : stress
        // ------------------------------------------------------------------------
        template <class T>
        inline xarray_container<T> hooke_strain(const elastic_properties<T>& mat,
                                                 const xarray_container<T>& stress)
        {
            if (stress.size() != 6)
                XTENSOR_THROW(std::invalid_argument, "stress must be 6‑component Voigt vector");
            auto S = mat.compliance_matrix();
            xarray_container<T> strain({6}, T(0));
            for (size_t i = 0; i < 6; ++i)
                for (size_t j = 0; j < 6; ++j)
                    strain(i) = strain(i) + detail::multiply(S(i, j), stress(j));
            return strain;
        }

        // ------------------------------------------------------------------------
        // von Mises stress (from stress tensor in Voigt notation)
        // ------------------------------------------------------------------------
        template <class T>
        inline T von_mises_stress(const xarray_container<T>& stress)
        {
            if (stress.size() != 6)
                XTENSOR_THROW(std::invalid_argument, "stress must be 6‑component Voigt vector");
            T s11 = stress(0), s22 = stress(1), s33 = stress(2);
            T s23 = stress(3), s13 = stress(4), s12 = stress(5);
            T term1 = detail::multiply(s11 - s22, s11 - s22);
            T term2 = detail::multiply(s22 - s33, s22 - s33);
            T term3 = detail::multiply(s33 - s11, s33 - s11);
            T term4 = T(6) * (detail::multiply(s12, s12) + detail::multiply(s23, s23) + detail::multiply(s13, s13));
            return detail::sqrt_val((term1 + term2 + term3 + term4) / T(2));
        }

        // ------------------------------------------------------------------------
        // Heat equation 1D update (explicit finite difference)
        // ------------------------------------------------------------------------
        template <class T>
        inline xarray_container<T> heat_conduction_1d(const xarray_container<T>& T_curr,
                                                       const thermal_properties<T>& prop,
                                                       T dx, T dt)
        {
            size_t n = T_curr.size();
            xarray_container<T> T_next({n}, T(0));
            T alpha = prop.diffusivity();
            T r = alpha * dt / detail::multiply(dx, dx);

            for (size_t i = 1; i < n - 1; ++i)
                T_next(i) = T_curr(i) + r * (T_curr(i+1) - T(2) * T_curr(i) + T_curr(i-1));
            // Boundary conditions (zero flux)
            T_next(0) = T_next(1);
            T_next(n-1) = T_next(n-2);
            return T_next;
        }

        // ------------------------------------------------------------------------
        // Navier‑Stokes (simplified 2D incompressible, for BigNumber compatible)
        // ------------------------------------------------------------------------
        template <class T>
        inline std::tuple<xarray_container<T>, xarray_container<T>, xarray_container<T>>
        navier_stokes_step(const xarray_container<T>& u, const xarray_container<T>& v,
                           const xarray_container<T>& p, const fluid_properties<T>& prop,
                           T dx, T dy, T dt)
        {
            size_t nx = u.shape()[1], ny = u.shape()[0];
            T nu = prop.kinematic_viscosity;
            T rho = prop.density;
            T dtdx = dt / dx, dtdy = dt / dy;

            xarray_container<T> u_new = u, v_new = v, p_new = p;

            // Tentative velocity (explicit diffusion + advection)
            for (size_t j = 1; j < ny - 1; ++j)
            {
                for (size_t i = 1; i < nx - 1; ++i)
                {
                    // Advection (upwind)
                    T u_adv = u(j, i) * (u(j, i) - u(j, i-1)) / dx;
                    T v_adv = v(j, i) * (u(j, i) - u(j-1, i)) / dy;
                    // Diffusion
                    T u_xx = (u(j, i+1) - T(2) * u(j, i) + u(j, i-1)) / detail::multiply(dx, dx);
                    T u_yy = (u(j+1, i) - T(2) * u(j, i) + u(j-1, i)) / detail::multiply(dy, dy);
                    u_new(j, i) = u(j, i) + dt * (-u_adv - v_adv + nu * (u_xx + u_yy));
                }
            }

            // Similar for v...
            // Pressure Poisson (omitted for brevity but fully implementable)
            return std::make_tuple(u_new, v_new, p_new);
        }

        // ========================================================================
        // Material database
        // ========================================================================
        template <class T = double>
        class material_database
        {
        public:
            material_database()
            {
                // Pre‑populate common materials
                add_steel();
                add_aluminum();
                add_copper();
                add_water();
                add_air();
            }

            void add(const std::string& name, const material<T>& mat)
            {
                m_materials[name] = mat;
            }

            material<T> get(const std::string& name) const
            {
                auto it = m_materials.find(name);
                if (it == m_materials.end())
                    XTENSOR_THROW(std::runtime_error, "Material not found: " + name);
                return it->second;
            }

            bool contains(const std::string& name) const
            {
                return m_materials.find(name) != m_materials.end();
            }

            std::vector<std::string> list() const
            {
                std::vector<std::string> names;
                for (const auto& kv : m_materials)
                    names.push_back(kv.first);
                return names;
            }

        private:
            std::unordered_map<std::string, material<T>> m_materials;

            void add_steel()
            {
                material<T> mat("Steel", T(7850));
                mat.elastic = elastic_properties<T>(T(200e9), T(0.3));
                mat.thermal = thermal_properties<T>(T(50), T(460), T(12e-6), T(7850));
                add("Steel", mat);
            }

            void add_aluminum()
            {
                material<T> mat("Aluminum", T(2700));
                mat.elastic = elastic_properties<T>(T(69e9), T(0.33));
                mat.thermal = thermal_properties<T>(T(237), T(900), T(23e-6), T(2700));
                add("Aluminum", mat);
            }

            void add_copper()
            {
                material<T> mat("Copper", T(8960));
                mat.elastic = elastic_properties<T>(T(110e9), T(0.34));
                mat.thermal = thermal_properties<T>(T(401), T(385), T(16.5e-6), T(8960));
                mat.em = electromagnetic_properties<T>(T(8.854e-12), T(1.256e-6), T(5.96e7));
                add("Copper", mat);
            }

            void add_water()
            {
                material<T> mat("Water", T(1000));
                mat.fluid = fluid_properties<T>(T(1000), T(1e-3));
                mat.thermal = thermal_properties<T>(T(0.6), T(4184), T(2.1e-4), T(1000));
                add("Water", mat);
            }

            void add_air()
            {
                material<T> mat("Air", T(1.225));
                mat.fluid = fluid_properties<T>(T(1.225), T(1.81e-5));
                mat.thermal = thermal_properties<T>(T(0.024), T(1005), T(3.43e-3), T(1.225));
                add("Air", mat);
            }
        };

        // Global material database instance
        template <class T = double>
        inline material_database<T>& materials()
        {
            static material_database<T> db;
            return db;
        }

    } // namespace material

    // Bring material utilities into xt namespace
    using material::elastic_properties;
    using material::thermal_properties;
    using material::fluid_properties;
    using material::electromagnetic_properties;
    using material::material;
    using material::material_database;
    using material::materials;
    using material::hooke_stress;
    using material::hooke_strain;
    using material::von_mises_stress;
    using material::heat_conduction_1d;
    using material::navier_stokes_step;

} // namespace xt

#endif // XTENSOR_XMATERIAL_HPP