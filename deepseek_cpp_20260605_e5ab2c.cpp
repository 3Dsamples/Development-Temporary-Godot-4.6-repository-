//File 0035 : core/xsimulation.hpp
//Simulation framework: particle systems, rigid body dynamics, collision detection, and time integration loops for 2D/3D real-time physics with SIMD force evaluation.
#ifndef XTENSOR_XSIMULATION_HPP
#define XTENSOR_XSIMULATION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xarray.hpp"
#include "xeval.hpp"
#include "xreducer.hpp"
#include "xlinalg.hpp"
#include "xrandom.hpp"
#include "xsort.hpp"
#include "xode.hpp"
#include "xgeometry.hpp"
#include "xsparse.hpp"

namespace xt {
namespace simulation {

    using real = double;

    /*********************************************
     * Particle and Particle System
     *********************************************/
    struct Particle {
        xarray_container<uvector<real>> position;   // shape (3,)
        xarray_container<uvector<real>> velocity;   // shape (3,)
        xarray_container<uvector<real>> force;      // shape (3,)
        real mass;
        real radius;
        bool active;

        Particle() : position({3}, 0.0), velocity({3}, 0.0), force({3}, 0.0), mass(1.0), radius(0.1), active(true) {}
        Particle(const xarray_container<uvector<real>>& pos, real m = 1.0, real r = 0.1)
            : position(pos), velocity({3}, 0.0), force({3}, 0.0), mass(m), radius(r), active(true) {}
    };

    class ParticleSystem {
    public:
        ParticleSystem() = default;

        void addParticle(const Particle& p) { particles_.push_back(p); }
        void addParticle(Particle&& p) { particles_.push_back(std::move(p)); }
        std::size_t size() const { return particles_.size(); }
        Particle& particle(std::size_t i) { return particles_.at(i); }
        const Particle& particle(std::size_t i) const { return particles_.at(i); }

        // Access positions as a 2D array (n x 3) for SIMD operations
        xarray_container<uvector<real>> positions() const {
            auto n = particles_.size();
            if (n == 0) return xarray_container<uvector<real>>({0,3});
            xarray_container<uvector<real>> pos({n, 3});
            for (std::size_t i = 0; i < n; ++i) {
                for (int d = 0; d < 3; ++d) pos(i, d) = particles_[i].position[d];
            }
            return pos;
        }

        void setPositions(const xarray_container<uvector<real>>& pos) {
            std::size_t n = particles_.size();
            for (std::size_t i = 0; i < n; ++i) {
                for (int d = 0; d < 3; ++d) particles_[i].position[d] = pos(i, d);
            }
        }

        // Apply gravity force to all particles
        void applyGravity(const xarray_container<uvector<real>>& gravity) {
            for (auto& p : particles_) {
                if (!p.active) continue;
                p.force = p.force + gravity * p.mass;
            }
        }

        // Apply viscous drag: force -= drag_coeff * velocity
        void applyDrag(real drag_coeff) {
            for (auto& p : particles_) {
                if (!p.active) continue;
                p.force = p.force - drag_coeff * p.velocity;
            }
        }

        // Apply spring forces between all pairs? Usually between connected pairs. We'll provide pair-wise spring between specified indices.
        void applySpringForces(const std::vector<std::pair<std::size_t, std::size_t>>& connections,
                               real stiffness, real rest_length) {
            for (auto& [i, j] : connections) {
                if (i >= particles_.size() || j >= particles_.size()) continue;
                auto& a = particles_[i];
                auto& b = particles_[j];
                auto dir = b.position - a.position;
                real dist = xt::norm::norm_l2(dir)();
                if (dist == 0.0) continue;
                real force_mag = stiffness * (dist - rest_length);
                auto force_dir = dir / dist;
                auto force_vec = force_dir * force_mag;
                a.force = a.force + force_vec;
                b.force = b.force - force_vec;
            }
        }

        // Reset forces to zero
        void clearForces() {
            for (auto& p : particles_) {
                p.force = p.position * 0.0; // set to zero vector
            }
        }

        // Integration: semi-implicit Euler (symplectic)
        void integrateEuler(real dt) {
            for (auto& p : particles_) {
                if (!p.active) continue;
                p.velocity = p.velocity + p.force / p.mass * dt;
                p.position = p.position + p.velocity * dt;
            }
        }

        // Integration: velocity Verlet (need previous force? Actually usual Verlet stores position and previous position; we use acceleration).
        void integrateVerlet(real dt, const std::function<void(ParticleSystem&)>& forceCallback) {
            // This version uses current forces as a(t), then updates position, recompute forces, update velocity.
            for (auto& p : particles_) {
                if (!p.active) continue;
                p.position = p.position + p.velocity * dt + 0.5 * (p.force / p.mass) * dt * dt;
                p.velocity = p.velocity + 0.5 * (p.force / p.mass) * dt;
            }
            forceCallback(*this);
            for (auto& p : particles_) {
                if (!p.active) continue;
                p.velocity = p.velocity + 0.5 * (p.force / p.mass) * dt;
            }
        }

        // Build spatial grid (uniform grid) for neighbor search
        void buildNeighborGrid(real cell_size) {
            neighbor_grid_cell_size_ = cell_size;
            grid_.clear();
            for (std::size_t i = 0; i < particles_.size(); ++i) {
                if (!particles_[i].active) continue;
                int cx = static_cast<int>(std::floor(particles_[i].position[0] / cell_size));
                int cy = static_cast<int>(std::floor(particles_[i].position[1] / cell_size));
                int cz = static_cast<int>(std::floor(particles_[i].position[2] / cell_size));
                grid_[{cx, cy, cz}].push_back(i);
            }
        }

        // Get neighbors within a given radius using the grid (returns indices)
        std::vector<std::size_t> getNeighbors(std::size_t idx, real radius) const {
            std::vector<std::size_t> result;
            if (idx >= particles_.size()) return result;
            const auto& pos = particles_[idx].position;
            int cx = static_cast<int>(std::floor(pos[0] / neighbor_grid_cell_size_));
            int cy = static_cast<int>(std::floor(pos[1] / neighbor_grid_cell_size_));
            int cz = static_cast<int>(std::floor(pos[2] / neighbor_grid_cell_size_));
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        auto key = std::make_tuple(cx+dx, cy+dy, cz+dz);
                        auto it = grid_.find(key);
                        if (it != grid_.end()) {
                            for (auto j : it->second) {
                                if (j == idx) continue;
                                real dist = xt::norm::norm_l2(particles_[j].position - pos)();
                                if (dist < radius) result.push_back(j);
                            }
                        }
                    }
            return result;
        }

    private:
        std::vector<Particle> particles_;
        real neighbor_grid_cell_size_;
        std::map<std::tuple<int,int,int>, std::vector<std::size_t>> grid_;
    };

    /*********************************************
     * Rigid Body
     *********************************************/
    struct RigidBody {
        xarray_container<uvector<real>> position;        // center of mass, shape (3,)
        xarray_container<uvector<real>> orientation;     // quaternion (w,x,y,z), shape (4,)
        xarray_container<uvector<real>> linear_velocity; // shape (3,)
        xarray_container<uvector<real>> angular_velocity;// shape (3,)
        real mass;
        xarray_container<uvector<real>> inertia_tensor;  // diagonal inertia (Ixx, Iyy, Izz), shape (3,)
        xarray_container<uvector<real>> force;           // accumulated force
        xarray_container<uvector<real>> torque;          // accumulated torque

        RigidBody() : position({3}, 0.0), orientation({1,0,0,0}), linear_velocity({3},0.0),
                      angular_velocity({3},0.0), mass(1.0), inertia_tensor({1,1,1}),
                      force({3},0.0), torque({3},0.0) {}

        // Normalize quaternion
        void normalizeOrientation() {
            real norm = std::sqrt(orientation[0]*orientation[0] + orientation[1]*orientation[1] +
                                  orientation[2]*orientation[2] + orientation[3]*orientation[3]);
            orientation = orientation / norm;
        }

        // Apply force at a world-space point (adds force and torque)
        void addForceAtPoint(const xarray_container<uvector<real>>& world_force,
                             const xarray_container<uvector<real>>& world_point) {
            force = force + world_force;
            auto r = world_point - position;
            // torque += r x force
            torque[0] += r[1]*world_force[2] - r[2]*world_force[1];
            torque[1] += r[2]*world_force[0] - r[0]*world_force[2];
            torque[2] += r[0]*world_force[1] - r[1]*world_force[0];
        }

        // Integrate using symplectic Euler (with quaternion update)
        void integrate(real dt) {
            if (mass <= 0.0) return;
            // linear
            linear_velocity = linear_velocity + force / mass * dt;
            position = position + linear_velocity * dt;
            // angular
            // I_inv * torque (assuming diagonal inertia)
            auto ang_acc = xarray_container<uvector<real>>({3}, 0.0);
            for (int i = 0; i < 3; ++i)
                ang_acc[i] = torque[i] / inertia_tensor[i];
            angular_velocity = angular_velocity + ang_acc * dt;
            // quaternion derivative: dq = 0.5 * omega_quat * q, where omega_quat = (0, w)
            auto omega_quat = xarray_container<uvector<real>>({0, angular_velocity[0], angular_velocity[1], angular_velocity[2]});
            // quaternion multiplication
            auto dq = quatMultiply(omega_quat, orientation) * 0.5;
            orientation = orientation + dq * dt;
            normalizeOrientation();
            // reset forces
            force = force * 0.0;
            torque = torque * 0.0;
        }

        static xarray_container<uvector<real>> quatMultiply(const xarray_container<uvector<real>>& a,
                                                            const xarray_container<uvector<real>>& b) {
            real w1=a[0], x1=a[1], y1=a[2], z1=a[3];
            real w2=b[0], x2=b[1], y2=b[2], z2=b[3];
            xarray_container<uvector<real>> res({4});
            res[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
            res[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
            res[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
            res[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
            return res;
        }
    };

    /*********************************************
     * Simulation loop manager
     *********************************************/
    class Simulation {
    public:
        using ForceCallback = std::function<void(ParticleSystem&, const std::vector<RigidBody*>&)>;

        Simulation() : time_(0.0), dt_(0.01) {}

        void setTimeStep(real dt) { dt_ = dt; }
        real time() const { return time_; }

        void addParticleSystem(ParticleSystem& psys) { particle_systems_.push_back(&psys); }
        void addRigidBody(RigidBody& rb) { rigid_bodies_.push_back(&rb); }

        void setForceCallback(ForceCallback cb) { force_callback_ = std::move(cb); }

        void step() {
            // Call external force computation
            if (force_callback_) {
                for (auto* ps : particle_systems_)
                    force_callback_(*ps, rigid_bodies_);
            }
            // Integrate particles
            for (auto* ps : particle_systems_) {
                ps->integrateEuler(dt_);
                ps->clearForces();
            }
            // Integrate rigid bodies
            for (auto* rb : rigid_bodies_) {
                rb->integrate(dt_);
            }
            time_ += dt_;
        }

        void run(real duration) {
            real end_time = time_ + duration;
            while (time_ < end_time) {
                step();
            }
        }

    private:
        real time_;
        real dt_;
        std::vector<ParticleSystem*> particle_systems_;
        std::vector<RigidBody*> rigid_bodies_;
        ForceCallback force_callback_;
    };

    /*********************************************
     * Collision detection utilities (sphere-sphere)
     *********************************************/
    inline bool sphereSphereCollision(const Particle& a, const Particle& b, real& penetration, xarray_container<uvector<real>>& normal) {
        auto dir = b.position - a.position;
        real dist = xt::norm::norm_l2(dir)();
        real min_dist = a.radius + b.radius;
        if (dist >= min_dist || dist == 0.0) return false;
        penetration = min_dist - dist;
        normal = dir / dist;
        return true;
    }

    inline void resolveCollision(Particle& a, Particle& b, real restitution = 0.8) {
        real penetration;
        xarray_container<uvector<real>> normal;
        if (!sphereSphereCollision(a, b, penetration, normal)) return;
        // position correction
        real inv_mass_a = 1.0 / a.mass;
        real inv_mass_b = 1.0 / b.mass;
        real total_inv_mass = inv_mass_a + inv_mass_b;
        a.position = a.position - normal * (penetration * inv_mass_a / total_inv_mass);
        b.position = b.position + normal * (penetration * inv_mass_b / total_inv_mass);
        // velocity response
        auto relative_vel = a.velocity - b.velocity;
        real vel_along_normal = relative_vel[0]*normal[0] + relative_vel[1]*normal[1] + relative_vel[2]*normal[2];
        if (vel_along_normal > 0) return;
        real j = -(1.0 + restitution) * vel_along_normal / total_inv_mass;
        a.velocity = a.velocity + normal * (j * inv_mass_a);
        b.velocity = b.velocity - normal * (j * inv_mass_b);
    }

} // namespace simulation
} // namespace xt

#endif // XTENSOR_XSIMULATION_HPP