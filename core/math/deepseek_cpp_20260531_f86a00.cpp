//File 0075 : core/math/probabilistic_filters.h
//Particle filter (sequential Monte Carlo) for generic state models: prediction, update with weights, systematic resampling, effective sample size, all with SIMD vector operations.
#ifndef CORE_MATH_PROBABILISTIC_FILTERS_H
#define CORE_MATH_PROBABILISTIC_FILTERS_H

#include "vector_math.h"
#include "random.h"               // for PCG32 and sampling
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace SimulationMath {
namespace filters {

// -----------------------------------------------------------------------------
// 1. A single particle with state and weight
// -----------------------------------------------------------------------------
struct Particle {
    DirectX::XMVECTOR state;   // generic state vector (e.g., position, velocity)
    float weight;

    Particle() noexcept : state(DirectX::XMVectorZero()), weight(0.0f) {}
    Particle(DirectX::FXMVECTOR s, float w) noexcept : state(s), weight(w) {}
};

// -----------------------------------------------------------------------------
// 2. Particle filter class
// -----------------------------------------------------------------------------
class ParticleFilter {
public:
    using State = DirectX::XMVECTOR;  // can be extended to vectors

    // motion_model: given previous state and random noise vector, returns new state
    // measurement_likelihood: given state and measurement, returns probability weight
    ParticleFilter(size_t num_particles)
        : num_particles_(num_particles), particles_(num_particles),
          rng_(std::random_device{}()) {}

    // Initialize particles by sampling from prior distribution
    void initialize(const std::function<State(PCG32&)>& prior_sampler) noexcept {
        for (size_t i = 0; i < num_particles_; ++i) {
            particles_[i].state = prior_sampler(rng_);
            particles_[i].weight = 1.0f / num_particles_;
        }
    }

    // Predict step: apply motion model to each particle
    // motion_model(state, rng) -> new state
    void predict(const std::function<State(const State&, PCG32&)>& motion_model) noexcept {
        for (auto& p : particles_) {
            p.state = motion_model(p.state, rng_);
        }
    }

    // Update step: reweight particles based on measurement likelihood
    // measurement: current observation
    // likelihood: (state, measurement) -> weight (un-normalized)
    void update(const DirectX::XMVECTOR& measurement,
                const std::function<float(const State&, const DirectX::XMVECTOR&)>& likelihood) noexcept {
        float sum_w = 0.0f;
        for (auto& p : particles_) {
            p.weight *= likelihood(p.state, measurement); // multiply by likelihood
            sum_w += p.weight;
        }
        // Normalize weights
        if (sum_w > 1e-12f) {
            float inv_sum = 1.0f / sum_w;
            for (auto& p : particles_) p.weight *= inv_sum;
        } else {
            // Degenerate case: reset to uniform
            float w = 1.0f / num_particles_;
            for (auto& p : particles_) p.weight = w;
        }
    }

    // Effective sample size (ESS): 1 / sum(w_i^2)
    float effective_sample_size() const noexcept {
        float sum_sq = 0.0f;
        for (const auto& p : particles_) sum_sq += p.weight * p.weight;
        if (sum_sq < 1e-12f) return 0.0f;
        return 1.0f / sum_sq;
    }

    // Systematic resampling (if ESS < threshold, e.g., N/2, call this)
    // Returns new resampled particles with equal weights.
    void systematic_resampling() noexcept {
        std::vector<Particle> new_particles(num_particles_);
        float inv_N = 1.0f / num_particles_;
        float u0 = uniform_float(rng_) * inv_N;
        float cumulative = particles_[0].weight;
        size_t idx = 0;
        for (size_t i = 0; i < num_particles_; ++i) {
            float u = u0 + i * inv_N;
            while (u > cumulative && idx + 1 < num_particles_) {
                ++idx;
                cumulative += particles_[idx].weight;
            }
            new_particles[i].state = particles_[idx].state;
            new_particles[i].weight = inv_N;
        }
        particles_ = std::move(new_particles);
    }

    // Estimate current state as weighted mean of particles
    DirectX::XMVECTOR estimate_state() const noexcept {
        DirectX::XMVECTOR sum = DirectX::XMVectorZero();
        for (const auto& p : particles_) {
            sum = DirectX::XMVectorAdd(sum, DirectX::XMVectorScale(p.state, p.weight));
        }
        return sum;
    }

    // Get particles for external visualization
    const std::vector<Particle>& particles() const noexcept { return particles_; }

private:
    size_t num_particles_;
    std::vector<Particle> particles_;
    PCG32 rng_;
};

} // namespace filters
} // namespace SimulationMath

#endif // CORE_MATH_PROBABILISTIC_FILTERS_H