// genesis/engine/force_fields.cpp

#include "genesis/engine/force_fields.h"
#include <algorithm>
#include <random>
#include <array>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// GravityForce implementation
//------------------------------------------------------------------------------
void GravityForce::apply(const std::vector<datatypes::Vector3>& positions,
                         const std::vector<datatypes::Vector3>& velocities,
                         const std::vector<datatypes::real>& masses,
                         std::vector<datatypes::Vector3>& forces,
                         datatypes::real dt) const {
    if (!enabled_) return;
    datatypes::Vector3 acc = acceleration_ * strength_;
    for (size_t i = 0; i < positions.size(); ++i) {
        forces[i] += masses[i] * acc;
    }
}

//------------------------------------------------------------------------------
// DragForce implementation
//------------------------------------------------------------------------------
void DragForce::apply(const std::vector<datatypes::Vector3>& positions,
                      const std::vector<datatypes::Vector3>& velocities,
                      const std::vector<datatypes::real>& masses,
                      std::vector<datatypes::Vector3>& forces,
                      datatypes::real dt) const {
    if (!enabled_) return;
    for (size_t i = 0; i < velocities.size(); ++i) {
        datatypes::Vector3 v = velocities[i];
        datatypes::real speed = v.norm();
        if (speed < 1e-12) continue;
        datatypes::Vector3 dir = v / speed;
        datatypes::real drag_force = 0;
        if (mode_ == Mode::LINEAR || mode_ == Mode::BOTH) {
            drag_force += linear_coef_ * speed;
        }
        if (mode_ == Mode::QUADRATIC || mode_ == Mode::BOTH) {
            drag_force += quadratic_coef_ * speed * speed;
        }
        forces[i] -= dir * (drag_force * strength_);
    }
}

//------------------------------------------------------------------------------
// PointForce implementation
//------------------------------------------------------------------------------
void PointForce::apply(const std::vector<datatypes::Vector3>& positions,
                       const std::vector<datatypes::Vector3>& velocities,
                       const std::vector<datatypes::real>& masses,
                       std::vector<datatypes::Vector3>& forces,
                       datatypes::real dt) const {
    if (!enabled_) return;
    for (size_t i = 0; i < positions.size(); ++i) {
        datatypes::Vector3 dir = positions[i] - position_;
        datatypes::real dist = dir.norm();
        if (dist < 1e-12) continue;
        if (dist > max_distance_) continue;
        dir /= dist;
        datatypes::real factor = strength_ * magnitude_ / std::pow(dist, exponent_);
        if (type_ == Type::REPULSE) {
            forces[i] += dir * factor * masses[i];
        } else {
            forces[i] -= dir * factor * masses[i];
        }
    }
}

//------------------------------------------------------------------------------
// TurbulenceForce implementation
//------------------------------------------------------------------------------
namespace {
    // Permutation table for Perlin noise
    const std::array<int, 256> perm = {
        151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,
        142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,
        203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
        74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,
        220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,
        132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,
        186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,
        59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,
        70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,
        178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,
        241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,
        176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,
        128,195,78,66,215,61,156,180
    };

    // Double permutation to avoid overflow
    std::array<int, 512> p;
    bool perm_initialized = false;
    std::mutex perm_mutex;

    void init_perm() {
        std::lock_guard<std::mutex> lock(perm_mutex);
        if (perm_initialized) return;
        for (int i = 0; i < 256; ++i) {
            p[i] = p[i+256] = perm[i];
        }
        perm_initialized = true;
    }

    inline datatypes::real fade(datatypes::real t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    inline datatypes::real lerp(datatypes::real t, datatypes::real a, datatypes::real b) {
        return a + t * (b - a);
    }

    inline datatypes::real grad(int hash, datatypes::real x, datatypes::real y, datatypes::real z) {
        int h = hash & 15;
        datatypes::real u = h < 8 ? x : y;
        datatypes::real v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

    datatypes::real perlin3d(datatypes::real x, datatypes::real y, datatypes::real z) {
        if (!perm_initialized) init_perm();

        int X = static_cast<int>(std::floor(x)) & 255;
        int Y = static_cast<int>(std::floor(y)) & 255;
        int Z = static_cast<int>(std::floor(z)) & 255;
        x -= std::floor(x);
        y -= std::floor(y);
        z -= std::floor(z);
        datatypes::real u = fade(x);
        datatypes::real v = fade(y);
        datatypes::real w = fade(z);

        int A = p[X] + Y;
        int AA = p[A] + Z;
        int AB = p[A+1] + Z;
        int B = p[X+1] + Y;
        int BA = p[B] + Z;
        int BB = p[B+1] + Z;

        return lerp(w,
                lerp(v,
                    lerp(u, grad(p[AA], x, y, z), grad(p[BA], x-1, y, z)),
                    lerp(u, grad(p[AB], x, y-1, z), grad(p[BB], x-1, y-1, z))
                ),
                lerp(v,
                    lerp(u, grad(p[AA+1], x, y, z-1), grad(p[BA+1], x-1, y, z-1)),
                    lerp(u, grad(p[AB+1], x, y-1, z-1), grad(p[BB+1], x-1, y-1, z-1))
                )
            );
    }
}

void TurbulenceForce::update(datatypes::real time, datatypes::real dt) {
    time_ = time;
}

datatypes::Vector3 TurbulenceForce::noise3(const datatypes::Vector3& p) const {
    datatypes::real nx = perlin3d(p[0], p[1], p[2]);
    datatypes::real ny = perlin3d(p[0] + 100.0f, p[1] + 200.0f, p[2] + 300.0f);
    datatypes::real nz = perlin3d(p[0] + 400.0f, p[1] + 500.0f, p[2] + 600.0f);
    return datatypes::Vector3(nx, ny, nz);
}

void TurbulenceForce::apply(const std::vector<datatypes::Vector3>& positions,
                            const std::vector<datatypes::Vector3>& velocities,
                            const std::vector<datatypes::real>& masses,
                            std::vector<datatypes::Vector3>& forces,
                            datatypes::real dt) const {
    if (!enabled_) return;
    for (size_t i = 0; i < positions.size(); ++i) {
        datatypes::Vector3 p = positions[i] * noise_scale_ + datatypes::Vector3(time_ * frequency_);
        datatypes::Vector3 noise = noise3(p);
        forces[i] += noise * (noise_strength_ * strength_ * masses[i]);
    }
}

//------------------------------------------------------------------------------
// WindForce implementation
//------------------------------------------------------------------------------
void WindForce::update(datatypes::real time, datatypes::real dt) {
    time_ = time;
    if (gustiness_ > 0) {
        // Simple gust variation: sin wave + noise-like variation
        datatypes::real phase = time_ * 0.5;
        current_gust_factor_ = 1.0 + gustiness_ * (0.5 * std::sin(phase) + 0.5 * std::sin(phase * 1.7 + 1.2));
        current_gust_factor_ = std::max(0.0f, current_gust_factor_);
    } else {
        current_gust_factor_ = 1.0;
    }
}

void WindForce::apply(const std::vector<datatypes::Vector3>& positions,
                      const std::vector<datatypes::Vector3>& velocities,
                      const std::vector<datatypes::real>& masses,
                      std::vector<datatypes::Vector3>& forces,
                      datatypes::real dt) const {
    if (!enabled_) return;
    datatypes::Vector3 wind_vec = direction_ * (base_speed_ * current_gust_factor_ * strength_);
    // Apply as force proportional to velocity difference (simplified drag)
    for (size_t i = 0; i < positions.size(); ++i) {
        datatypes::Vector3 rel_vel = wind_vec - velocities[i];
        datatypes::real speed = rel_vel.norm();
        if (speed < 1e-6) continue;
        // Simple quadratic air resistance approximation
        datatypes::real area = 1.0; // assume unit area; could be per-particle
        datatypes::real air_density = 1.225;
        datatypes::real drag_coef = 0.5 * air_density * area;
        forces[i] += rel_vel.normalized() * (drag_coef * speed * speed * dt);
    }
}

//------------------------------------------------------------------------------
// ForceFieldManager implementation
//------------------------------------------------------------------------------
void ForceFieldManager::add(std::shared_ptr<ForceField> field) {
    if (field) fields_.push_back(field);
}

void ForceFieldManager::remove(const std::string& name) {
    // Not directly supported by base class; could be implemented with custom naming.
    // For simplicity, we rely on external indexing or other methods.
}

void ForceFieldManager::clear() {
    fields_.clear();
}

void ForceFieldManager::apply_all(const std::vector<datatypes::Vector3>& positions,
                                  const std::vector<datatypes::Vector3>& velocities,
                                  const std::vector<datatypes::real>& masses,
                                  std::vector<datatypes::Vector3>& forces,
                                  datatypes::real dt) const {
    for (const auto& field : fields_) {
        field->apply(positions, velocities, masses, forces, dt);
    }
}

void ForceFieldManager::update_all(datatypes::real time, datatypes::real dt) {
    for (auto& field : fields_) {
        field->update(time, dt);
    }
}

std::shared_ptr<ForceField> ForceFieldManager::find_by_type(const std::string& type) const {
    for (const auto& field : fields_) {
        if (field->type_name() == type) return field;
    }
    return nullptr;
}

//------------------------------------------------------------------------------
// Force presets
//------------------------------------------------------------------------------
namespace force_presets {
    std::shared_ptr<GravityForce> earth_gravity() {
        return std::make_shared<GravityForce>(datatypes::Vector3(0, 0, -9.80665));
    }

    std::shared_ptr<GravityForce> moon_gravity() {
        return std::make_shared<GravityForce>(datatypes::Vector3(0, 0, -1.625));
    }

    std::shared_ptr<DragForce> air_drag_standard() {
        return std::make_shared<DragForce>(0.1, 0.01, DragForce::Mode::BOTH);
    }

    std::shared_ptr<DragForce> water_drag() {
        return std::make_shared<DragForce>(0.5, 0.1, DragForce::Mode::BOTH);
    }

    std::shared_ptr<TurbulenceForce> light_turbulence() {
        return std::make_shared<TurbulenceForce>(0.5, 1.0, 2.0);
    }

    std::shared_ptr<WindForce> gentle_breeze(const datatypes::Vector3& direction) {
        return std::make_shared<WindForce>(direction, 2.0, 0.2);
    }
}

} // namespace engine
} // namespace genesis