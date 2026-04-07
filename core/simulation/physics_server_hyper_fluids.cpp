--- START OF FILE core/simulation/physics_server_hyper_fluids.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/kernel_registry.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * SPH Smoothing Kernels (Deterministic)
 * Optimized for Warp-style parallel sweeps.
 */
static _FORCE_INLINE_ FixedMathCore sph_kernel_poly6(FixedMathCore p_r2, FixedMathCore p_h) {
    FixedMathCore h2 = p_h * p_h;
    if (p_r2 >= h2) return MathConstants<FixedMathCore>::zero();
    
    FixedMathCore h9 = p_h.power(9);
    FixedMathCore factor = FixedMathCore(315LL, false) / (FixedMathCore(64LL, false) * Math::pi() * h9);
    FixedMathCore diff = h2 - p_r2;
    return factor * diff * diff * diff;
}

static _FORCE_INLINE_ Vector3f sph_kernel_spiky_gradient(const Vector3f &p_r_vec, FixedMathCore p_r, FixedMathCore p_h) {
    if (p_r >= p_h || p_r.get_raw() == 0) return Vector3f();
    
    FixedMathCore h6 = p_h.power(6);
    FixedMathCore factor = -FixedMathCore(45LL, false) / (Math::pi() * h6);
    FixedMathCore diff = p_h - p_r;
    FixedMathCore magnitude = factor * diff * diff;
    return p_r_vec.normalized() * magnitude;
}

/**
 * Warp Kernel: FluidDensityKernel
 * Calculates the local density of each fluid particle based on its neighbors.
 */
void fluid_density_kernel(
        const BigIntCore &p_index,
        const Vector3f &p_pos,
        const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
        const Vector3f *p_all_positions,
        const BigIntCore *p_all_sx, const BigIntCore *p_all_sy, const BigIntCore *p_all_sz,
        FixedMathCore &r_density,
        uint64_t p_count,
        FixedMathCore p_h,
        FixedMathCore p_mass) {

    FixedMathCore density_accum = MathConstants<FixedMathCore>::zero();
    FixedMathCore h2 = p_h * p_h;

    for (uint64_t j = 0; j < p_count; j++) {
        // Resolve galactic distance between particles i and j
        BigIntCore dsx = p_all_sx[j] - p_sx;
        BigIntCore dsy = p_all_sy[j] - p_sy;
        BigIntCore dsz = p_all_sz[j] - p_sz;

        // Sector-aware distance calculation
        FixedMathCore threshold(10000LL, false); 
        Vector3f rel_pos = (p_all_positions[j] + Vector3f(threshold * FixedMathCore(static_cast<int64_t>(std::stoll(dsx.to_string()))),
                                                         threshold * FixedMathCore(static_cast<int64_t>(std::stoll(dsy.to_string()))),
                                                         threshold * FixedMathCore(static_cast<int64_t>(std::stoll(dsz.to_string()))))) - p_pos;

        FixedMathCore r2 = rel_pos.length_squared();
        if (r2 < h2) {
            density_accum += p_mass * sph_kernel_poly6(r2, p_h);
        }
    }
    r_density = density_accum;
}

/**
 * Warp Kernel: FluidForceKernel
 * Resolves pressure and viscosity forces between particles.
 */
void fluid_force_kernel(
        const BigIntCore &p_index,
        const Vector3f &p_pos,
        const Vector3f &p_vel,
        const FixedMathCore &p_density,
        const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
        const Vector3f *p_all_positions,
        const Vector3f *p_all_velocities,
        const FixedMathCore *p_all_densities,
        const BigIntCore *p_all_sx, const BigIntCore *p_all_sy, const BigIntCore *p_all_sz,
        Vector3f &r_acceleration,
        uint64_t p_count,
        FixedMathCore p_h,
        FixedMathCore p_mass,
        FixedMathCore p_gas_constant,
        FixedMathCore p_rest_density,
        FixedMathCore p_viscosity) {

    Vector3f f_pressure;
    Vector3f f_viscosity;

    FixedMathCore p_i = p_gas_constant * (p_density - p_rest_density);

    for (uint64_t j = 0; j < p_count; j++) {
        uint64_t i_val = static_cast<uint64_t>(std::stoll(p_index.to_string()));
        if (i_val == j) continue;

        BigIntCore dsx = p_all_sx[j] - p_sx;
        BigIntCore dsy = p_all_sy[j] - p_sy;
        BigIntCore dsz = p_all_sz[j] - p_sz;
        FixedMathCore threshold(10000LL, false); 

        Vector3f rel_pos = (p_all_positions[j] + Vector3f(threshold * FixedMathCore(static_cast<int64_t>(std::stoll(dsx.to_string()))),
                                                         threshold * FixedMathCore(static_cast<int64_t>(std::stoll(dsy.to_string()))),
                                                         threshold * FixedMathCore(static_cast<int64_t>(std::stoll(dsz.to_string()))))) - p_pos;

        FixedMathCore r = rel_pos.length();
        if (r < p_h) {
            FixedMathCore p_j = p_gas_constant * (p_all_densities[j] - p_rest_density);
            
            // Pressure Force
            FixedMathCore pressure_avg = (p_i + p_j) * MathConstants<FixedMathCore>::half();
            f_pressure -= sph_kernel_spiky_gradient(rel_pos, r, p_h) * (p_mass * pressure_avg / p_all_densities[j]);

            // Viscosity Force
            FixedMathCore h2 = p_h * p_h;
            FixedMathCore h3 = h2 * p_h;
            FixedMathCore laplacian = (FixedMathCore(45LL, false) / (Math::pi() * h3 * h3)) * (p_h - r);
            f_viscosity += (p_all_velocities[j] - p_vel) * (p_viscosity * p_mass * laplacian / p_all_densities[j]);
        }
    }

    r_acceleration = (f_pressure + f_viscosity) / p_density;
}

/**
 * update_fluid_simulation()
 * Master orchestrator for parallelized 120 FPS fluid dynamics.
 */
void PhysicsServerHyper::update_fluid_simulation(const FixedMathCore &p_delta) {
    auto &registry = get_kernel_registry();
    uint64_t count = registry.get_stream<Vector3f>().size();
    if (count == 0) return;

    // Phase 1: Update Densities
    SimulationThreadPool::get_singleton()->enqueue_task([&]() {
        for (uint64_t i = 0; i < count; i++) {
            BigIntCore idx(static_cast<int64_t>(i));
            fluid_density_kernel(
                idx,
                registry.get_stream<Vector3f>()[i], // pos
                registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X)[i],
                registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y)[i],
                registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z)[i],
                registry.get_stream<Vector3f>().get_base_ptr(),
                registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X).get_base_ptr(),
                registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y).get_base_ptr(),
                registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z).get_base_ptr(),
                registry.get_stream<FixedMathCore>(COMPONENT_DENSITY)[i],
                count,
                FixedMathCore(1LL, false), // h
                FixedMathCore(1LL, false)  // mass
            );
        }
    }, SimulationThreadPool::PRIORITY_CRITICAL);

    SimulationThreadPool::get_singleton()->wait_for_all();

    // Phase 2: Resolve Forces and Integrate
    SimulationThreadPool::get_singleton()->enqueue_task([&]() {
        for (uint64_t i = 0; i < count; i++) {
            Vector3f acceleration;
            BigIntCore idx(static_cast<int64_t>(i));
            fluid_force_kernel(
                idx,
                registry.get_stream<Vector3f>(COMPONENT_POSITION)[i],
                registry.get_stream<Vector3f>(COMPONENT_VELOCITY)[i],
                registry.get_stream<FixedMathCore>(COMPONENT_DENSITY)[i],
                registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X)[i],
                registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y)[i],
                registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z)[i],
                registry.get_stream<Vector3f>(COMPONENT_POSITION).get_base_ptr(),
                registry.get_stream<Vector3f>(COMPONENT_VELOCITY).get_base_ptr(),
                registry.get_stream<FixedMathCore>(COMPONENT_DENSITY).get_base_ptr(),
                registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X).get_base_ptr(),
                registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y).get_base_ptr(),
                registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z).get_base_ptr(),
                acceleration,
                count,
                FixedMathCore(1LL, false), // h
                FixedMathCore(1LL, false), // mass
                FixedMathCore(2000LL, false), // gas constant
                FixedMathCore(1000LL, false), // rest density
                FixedMathCore(10LL, false)    // viscosity
            );
            
            // Integration: v = v + a*dt; p = p + v*dt
            registry.get_stream<Vector3f>(COMPONENT_VELOCITY)[i] += acceleration * p_delta;
            registry.get_stream<Vector3f>(COMPONENT_POSITION)[i] += registry.get_stream<Vector3f>(COMPONENT_VELOCITY)[i] * p_delta;
        }
    }, SimulationThreadPool::PRIORITY_CRITICAL);

    SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_fluids.cpp ---
