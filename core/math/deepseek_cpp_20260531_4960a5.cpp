// SPDX-FileCopyrightText: Copyright (c) 2025 – C++17 Math/Physics Library
// SPDX-License-Identifier: MIT
#pragma once

#include "spatial.hpp"
#include "hashgrid.hpp"
#include <vector>
#include <algorithm>
#include <execution>

namespace wp {

// ── Particle state (position + velocity) ──
template <typename T>
struct Particle {
    vec_t<3,T> pos{0,0,0};
    vec_t<3,T> vel{0,0,0};
    vec_t<3,T> force{0,0,0};
    T mass{1};
    T radius{0.05f};
};

// ── Rigid‑body state (pose + spatial velocity) ──
template <typename T>
struct RigidBody {
    vec_t<3,T>  pos{0,0,0};
    quat_t<T>   rot{0,0,0,1};
    spatial_vector_t<T> vel{0,0,0,0,0,0};   // angular (top) + linear (bottom)
    spatial_inertia_t<T> inertia;
};

// ── Simulation parameters ──
template <typename T>
struct SimParams {
    T dt{0.01};
    T gravity{9.81};
    vec_t<3,T> gravity_dir{0,-1,0};
    int substeps{4};
    T damping{0.995f};
};

// ── Unified simulation kernel ──
template <typename T>
class SimulationKernel {
public:
    // Particle system
    std::vector<Particle<T>> particles;
    // Rigid bodies
    std::vector<RigidBody<T>> bodies;
    // Spatial acceleration structure
    HashGrid<T> grid{T(0.1)};

    SimParams<T> params;

    // ── Advance one time step ──
    void step() {
        T sub_dt = params.dt / T(params.substeps);
        for (int s = 0; s < params.substeps; ++s) {
            integrate_particles(sub_dt);
            integrate_rigid_bodies(sub_dt);
            resolve_collisions();
        }
    }

private:
    // ── Semi‑implicit Euler for particles (with gravity) ──
    void integrate_particles(T dt) {
        std::for_each(std::execution::par, particles.begin(), particles.end(), [&](Particle<T>& p) {
            // Gravity + external forces
            vec_t<3,T> acc = params.gravity_dir * params.gravity + p.force / p.mass;
            p.vel += acc * dt;
            p.vel *= params.damping;
            p.pos += p.vel * dt;
            p.force = vec_t<3,T>(0);
        });
    }

    // ── Newton–Euler integration for rigid bodies ──
    void integrate_rigid_bodies(T dt) {
        for (auto& body : bodies) {
            // Spatial acceleration: I⁻¹ * F
            spatial_vector_t<T> f_ext{0,0,0, 0, params.gravity*body.inertia.mass, 0};  // gravity on y
            spatial_vector_t<T> acc = body.inertia.mul(f_ext);

            // Velocity update
            body.vel = body.vel + acc * dt;
            body.vel = body.vel * params.damping;

            // Position update: linear
            body.pos = body.pos + v_vec(body.vel) * dt;

            // Orientation update: exponential map
            vec_t<3,T> omega = w_vec(body.vel);
            T angle = length(omega) * dt;
            if (angle > Constants<T>::epsilon) {
                vec_t<3,T> axis = normalize(omega);
                quat_t<T> delta_rot(std::sin(angle/T(2))*axis[0],
                                    std::sin(angle/T(2))*axis[1],
                                    std::sin(angle/T(2))*axis[2],
                                    std::cos(angle/T(2)));
                body.rot = normalize(mul(body.rot, delta_rot));
            }
        }
    }

    // ── Particle–particle collision resolution (hard‑sphere DEM) ──
    void resolve_collisions() {
        // Build hash grid
        std::vector<vec_t<3,T>> positions;
        positions.reserve(particles.size());
        for (auto& p : particles) positions.push_back(p.pos);
        grid.build(positions);

        for (size_t i = 0; i < particles.size(); ++i) {
            auto neighbors = grid.query(particles[i].pos, T(2)*particles[i].radius);
            for (int j_idx : neighbors) {
                if (j_idx <= static_cast<int>(i)) continue;
                auto& pi = particles[i];
                auto& pj = particles[j_idx];
                vec_t<3,T> delta = pi.pos - pj.pos;
                T dist = length(delta);
                T min_dist = pi.radius + pj.radius;
                if (dist < min_dist && dist > Constants<T>::epsilon) {
                    vec_t<3,T> n = delta / dist;
                    T overlap = min_dist - dist;
                    // Position correction
                    pi.pos += n * (overlap * T(0.5));
                    pj.pos -= n * (overlap * T(0.5));
                    // Velocity correction (simplified elastic collision)
                    T vrel = dot(pi.vel - pj.vel, n);
                    if (vrel > 0) {
                        T impulse = T(2) * vrel / (T(1)/pi.mass + T(1)/pj.mass);
                        pi.vel -= n * (impulse / pi.mass);
                        pj.vel += n * (impulse / pj.mass);
                    }
                }
            }
        }
    }
};

// ── 2D specialisation (template, uses XY plane) ──
template <typename T>
struct Particle2D {
    vec_t<2,T> pos{0,0};
    vec_t<2,T> vel{0,0};
    T mass{1};
    T radius{0.05f};
};

} // namespace wp