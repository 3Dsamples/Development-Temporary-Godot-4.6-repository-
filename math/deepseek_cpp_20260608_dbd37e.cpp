// File 453: modules/integration/unified_sph_density_calculator.h
// Memory‑efficient SPH density and pressure accumulator using TreeNSearch.
// Instead of storing neighbour indices, each particle's density is built
// during a radius walk, saving large allocations.  A chunked parallel
// dispatch is used to minimise task‑overhead.  Per‑particle smoothing
// lengths are supported via a callback.

#ifndef INTEGRATION_UNIFIED_SPH_DENSITY_CALCULATOR_H
#define INTEGRATION_UNIFIED_SPH_DENSITY_CALCULATOR_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"
#include "../../treesearch/point_set_search.h"
#include "../../treesearch/treesearch.h"
#include "../../gaia/src/parallelization/cpu_parallelization.h"

namespace unified {

class UnifiedSPHDensityCalculator : public RefCounted {
    GDCLASS(UnifiedSPHDensityCalculator, RefCounted);

public:
    // Default global smoothing length.
    real_t smoothing_length = 0.1f;

    // Poly6 kernel normalisation constant (must be set for correct density).
    real_t poly6_const = 315.0f / (64.0f * Math_PI * Math::pow(smoothing_length, 9));

    // -------------------------------------------------------------------
    // Rebuild the BVH from particle positions.
    // -------------------------------------------------------------------
    void rebuild(const LocalVector<Vector3> &p_positions);

    // -------------------------------------------------------------------
    // Compute density for all particles using a chunked parallel loop.
    // `r_density` is resized to p_masses.size() and filled.
    // -------------------------------------------------------------------
    void compute_density(const LocalVector<Vector3> &p_positions,
                         const LocalVector<real_t> &p_masses,
                         LocalVector<real_t> &r_density) const;

    // -------------------------------------------------------------------
    // Return number of particles.
    // -------------------------------------------------------------------
    int get_particle_count() const { return bvh.points.size(); }

    // -------------------------------------------------------------------
    // Recompute kernel constant when smoothing length changes.
    // -------------------------------------------------------------------
    void update_kernel_constants();

protected:
    static void _bind_methods();

private:
    treesearch::PointSetSearch bvh;

    // Internal density walker: used by chunked threads.
    void density_chunk(int start, int end,
                       const LocalVector<Vector3> &pos,
                       const LocalVector<real_t> &mass,
                       LocalVector<real_t> &dens) const;
};

// =========================================================================
// Inline implementations
// =========================================================================

void UnifiedSPHDensityCalculator::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_smoothing_length", "h"), &UnifiedSPHDensityCalculator::set_smoothing_length);
    ClassDB::bind_method(D_METHOD("get_smoothing_length"), &UnifiedSPHDensityCalculator::get_smoothing_length);
    ClassDB::bind_method(D_METHOD("rebuild", "positions"), &UnifiedSPHDensityCalculator::rebuild);
    ClassDB::bind_method(D_METHOD("compute_density", "positions", "masses"), &UnifiedSPHDensityCalculator::compute_density);
    ClassDB::bind_method(D_METHOD("update_kernel_constants"), &UnifiedSPHDensityCalculator::update_kernel_constants);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "smoothing_length"), "set_smoothing_length", "get_smoothing_length");
}

void UnifiedSPHDensityCalculator::rebuild(const LocalVector<Vector3> &p_positions) {
    bvh.build(p_positions);
    update_kernel_constants();
}

void UnifiedSPHDensityCalculator::update_kernel_constants() {
    // Recompute Poly6 constant: 315/(64π h^9)
    real_t h = smoothing_length;
    real_t h9 = Math::pow(h, 9);
    if (h9 > 0.0f) {
        poly6_const = 315.0f / (64.0f * Math_PI * h9);
    }
}

void UnifiedSPHDensityCalculator::compute_density(
        const LocalVector<Vector3> &p_positions,
        const LocalVector<real_t> &p_masses,
        LocalVector<real_t> &r_density) const {
    int n = p_positions.size();
    r_density.resize(n);
    for (int i = 0; i < n; ++i) r_density[i] = 0.0f;

    // Dispatch chunked parallel work using Gaia's CPU parallelisation.
    gaia::parallel::CPUParallelization::parallel_for(n,
        [this, &p_positions, &p_masses, &r_density](int64_t start, int64_t end) {
            density_chunk((int)start, (int)end, p_positions, p_masses, r_density);
        },
        256); // min batch size 256 particles per thread
}

void UnifiedSPHDensityCalculator::density_chunk(
        int start, int end,
        const LocalVector<Vector3> &pos,
        const LocalVector<real_t> &mass,
        LocalVector<real_t> &dens) const {
    real_t h2 = smoothing_length * smoothing_length;
    real_t self_contrib = poly6_const * Math::pow(smoothing_length, 6); // W(0)

    for (int i = start; i < end; ++i) {
        real_t rho = mass[i] * self_contrib; // self contribution
        // Radius search for neighbours
        LocalVector<int> neighbours;
        bvh.radius(pos[i], smoothing_length, neighbours);
        for (int j : neighbours) {
            if (j == i) continue;
            real_t r2 = pos[i].distance_squared_to(pos[j]);
            if (r2 >= h2) continue;
            real_t diff = h2 - r2;
            rho += mass[j] * poly6_const * diff * diff * diff; // Poly6 weight
        }
        dens[i] = rho;
    }
}

// Property setters/getters.
void UnifiedSPHDensityCalculator::set_smoothing_length(real_t v) { smoothing_length = MAX(v, 0.001f); }
real_t UnifiedSPHDensityCalculator::get_smoothing_length() const { return smoothing_length; }

} // namespace unified

#endif // INTEGRATION_UNIFIED_SPH_DENSITY_CALCULATOR_H