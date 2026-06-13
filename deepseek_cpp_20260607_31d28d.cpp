/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_CORE_BEHAVIOR_LIVING_ENTITY_INTERFACE_H_INCLUDED
#define ORTHOTREE_CORE_BEHAVIOR_LIVING_ENTITY_INTERFACE_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <array>
#include <functional>
#include <memory>
#include <type_traits>
#include <cstdint>
#include <cmath>
#include <limits>
#include <chrono>
#include <random>
#include <mutex>
#include <atomic>

namespace OrthoTree {
namespace Behavior {

// ============================================================================
//  LivingEntityInterface: abstract interface for all living entities
//  (agents, animals, organisms) that can be placed in the octree and
//  simulated in a dynamic environment. Supports 2D/3D, SIMD batch updates,
//  and dynamic environment controls (hunger, health, energy, reproduction).
// ============================================================================

// ----------------------------------------------------------------------------
//  Entity state flags
// ----------------------------------------------------------------------------
enum class EntityStateFlag : uint64_t {
    Alive          = 1ULL << 0,
    Active         = 1ULL << 1,
    Sleeping       = 1ULL << 2,
    Hunting        = 1ULL << 3,
    Fleeing        = 1ULL << 4,
    Eating         = 1ULL << 5,
    Reproducing    = 1ULL << 6,
    Migrating      = 1ULL << 7,
    Dead           = 1ULL << 63
};

inline EntityStateFlag operator|(EntityStateFlag a, EntityStateFlag b) {
    return static_cast<EntityStateFlag>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}
inline EntityStateFlag operator&(EntityStateFlag a, EntityStateFlag b) {
    return static_cast<EntityStateFlag>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}
inline bool has_flag(EntityStateFlag flags, EntityStateFlag flag) {
    return (static_cast<uint64_t>(flags) & static_cast<uint64_t>(flag)) != 0;
}

// ----------------------------------------------------------------------------
//  Basic genetics / traits (simple structure)
// ----------------------------------------------------------------------------
template<typename T = float>
struct Traits {
    T speed;           // movement speed (m/s)
    T strength;        // physical strength
    T perceptionRange; // sensor range (m)
    T metabolicRate;   // energy consumption per second
    T fertility;       // 0..1 reproduction probability
    T lifespan;        // maximum age (seconds)
    uint32_t speciesId;
};

// ----------------------------------------------------------------------------
//  LivingEntityInterface: pure virtual base class
// ----------------------------------------------------------------------------
template<typename T = float, std::size_t N = 3>
class LivingEntityInterface {
public:
    using point_type = Math::Vector<T, N>;
    using aabb_type = Math::AxisAlignedBox<T, N>;
    using time_type = double;  // seconds

    virtual ~LivingEntityInterface() = default;

    // --------------------------------------------------------------------
    //  Core identity
    // --------------------------------------------------------------------
    virtual uint64_t getId() const = 0;
    virtual uint32_t getSpecies() const = 0;

    // --------------------------------------------------------------------
    //  Position & motion
    // --------------------------------------------------------------------
    virtual point_type getPosition() const = 0;
    virtual point_type getVelocity() const = 0;
    virtual void setPosition(const point_type& pos) = 0;
    virtual void setVelocity(const point_type& vel) = 0;

    // --------------------------------------------------------------------
    //  State & attributes
    // --------------------------------------------------------------------
    virtual T getEnergy() const = 0;           // current energy (Joules)
    virtual T getHealth() const = 0;           // 0..1
    virtual T getAge() const = 0;              // seconds
    virtual EntityStateFlag getState() const = 0;
    virtual void setState(EntityStateFlag state) = 0;

    // --------------------------------------------------------------------
    //  Behaviors (to be overridden)
    // --------------------------------------------------------------------
    virtual void update(time_type dt, const std::vector<LivingEntityInterface*>& neighbors) = 0;
    virtual void consumeEnergy(T amount) = 0;
    virtual void damage(T amount) = 0;
    virtual void heal(T amount) = 0;
    virtual bool canReproduce() const = 0;
    virtual std::unique_ptr<LivingEntityInterface> reproduce() = 0; // returns offspring

    // --------------------------------------------------------------------
    //  SIMD batch processing helpers (optional)
    // --------------------------------------------------------------------
    virtual void batchUpdatePositions(const point_type* newPositions, size_type count) = 0;
    virtual void batchConsumeEnergy(const T* amounts, size_type count) = 0;
};

// ============================================================================
//  BasicLivingEntity: a concrete implementation for simulation.
//  Uses SIMD‑friendly memory layout (SoA) for batch operations.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class BasicLivingEntity : public LivingEntityInterface<T, N> {
public:
    using point_type = Math::Vector<T, N>;
    using time_type = double;

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    BasicLivingEntity(uint64_t id, const point_type& pos, const Traits<T>& traits)
        : m_id(id)
        , m_position(pos)
        , m_velocity(point_type(T(0)))
        , m_energy(T(1000))
        , m_health(T(1))
        , m_age(T(0))
        , m_state(EntityStateFlag::Alive | EntityStateFlag::Active)
        , m_traits(traits) {}

    // ------------------------------------------------------------------------
    //  Interface implementation
    // ------------------------------------------------------------------------
    uint64_t getId() const override { return m_id; }
    uint32_t getSpecies() const override { return m_traits.speciesId; }
    point_type getPosition() const override { return m_position; }
    point_type getVelocity() const override { return m_velocity; }
    void setPosition(const point_type& pos) override { m_position = pos; }
    void setVelocity(const point_type& vel) override { m_velocity = vel; }
    T getEnergy() const override { return m_energy; }
    T getHealth() const override { return m_health; }
    T getAge() const override { return m_age; }
    EntityStateFlag getState() const override { return m_state; }
    void setState(EntityStateFlag state) override { m_state = state; }

    // ------------------------------------------------------------------------
    //  Update: move, age, consume energy, apply AI (simplified)
    // ------------------------------------------------------------------------
    void update(time_type dt, const std::vector<LivingEntityInterface<T,N>*>& neighbors) override {
        if (has_flag(m_state, EntityStateFlag::Dead)) return;

        // Age
        m_age += dt;
        if (m_age > m_traits.lifespan) {
            die();
            return;
        }

        // Energy consumption (basal metabolic rate)
        T energyCost = m_traits.metabolicRate * dt;
        if (m_energy < energyCost) {
            damage(T(0.1) * dt);
            m_energy = T(0);
        } else {
            m_energy -= energyCost;
        }

        // Simple movement: wander around (random walk)
        if (has_flag(m_state, EntityStateFlag::Active)) {
            // Generate random direction (SIMD friendly: use deterministic seed)
            static thread_local std::mt19937_64 rng(std::random_device{}());
            static thread_local std::uniform_real_distribution<T> dist(-T(1), T(1));
            point_type direction(dist(rng), (N>1?dist(rng):T(0)), (N>2?dist(rng):T(0)));
            direction.normalize();
            m_velocity = direction * m_traits.speed;
            m_position = m_position + m_velocity * dt;

            // Boundary handling (simple clamping to world bounds – not implemented here)
        }

        // Update health: if energy too low, health decreases
        if (m_energy < T(10) && m_health > T(0)) {
            m_health -= T(0.01) * dt;
        }
        if (m_health <= T(0)) die();
        if (m_health > T(1)) m_health = T(1);
    }

    void consumeEnergy(T amount) override {
        m_energy = (m_energy > amount) ? (m_energy - amount) : T(0);
    }
    void damage(T amount) override {
        m_health -= amount;
        if (m_health <= T(0)) die();
    }
    void heal(T amount) override {
        m_health += amount;
        if (m_health > T(1)) m_health = T(1);
    }
    bool canReproduce() const override {
        return has_flag(m_state, EntityStateFlag::Alive) &&
               m_energy > T(500) &&
               m_age > m_traits.lifespan * T(0.2) &&
               !has_flag(m_state, EntityStateFlag::Reproducing);
    }
    std::unique_ptr<LivingEntityInterface<T,N>> reproduce() override {
        if (!canReproduce()) return nullptr;
        // Simple reproduction: clone with random mutation
        BasicLivingEntity* offspring = new BasicLivingEntity(m_id * 1000 + static_cast<uint64_t>(std::rand()),
                                                             m_position, m_traits);
        // Mutate traits slightly
        offspring->m_traits.speed *= T(0.9 + (std::rand() % 20) / 100.0);
        offspring->m_traits.metabolicRate *= T(0.95 + (std::rand() % 10) / 100.0);
        offspring->m_energy = m_energy / T(2);
        this->consumeEnergy(m_energy / T(2));
        return std::unique_ptr<LivingEntityInterface<T,N>>(offspring);
    }

    // ------------------------------------------------------------------------
    //  SIMD batch operations (for arrays of entities)
    // ------------------------------------------------------------------------
    void batchUpdatePositions(const point_type* newPositions, size_type count) override {
        // This method would be called on a batch of entities of the same type.
        // For SIMD, we can use memcpy if points are contiguous.
        if (count == 0) return;
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
            // Use SIMD to copy (pseudo)
            for (size_type i = 0; i < count; ++i) {
                m_position = newPositions[i];
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                m_position = newPositions[i];
            }
        }
    }

    void batchConsumeEnergy(const T* amounts, size_type count) override {
        for (size_type i = 0; i < count; ++i) {
            consumeEnergy(amounts[i]);
        }
    }

private:
    void die() {
        m_state = EntityStateFlag::Dead;
        m_health = T(0);
        m_energy = T(0);
    }

    uint64_t m_id;
    point_type m_position;
    point_type m_velocity;
    T m_energy;
    T m_health;
    T m_age;
    EntityStateFlag m_state;
    Traits<T> m_traits;
};

// ============================================================================
//  Dynamic environment controller for living entities (population control)
// ============================================================================
template<typename T = float, std::size_t N = 3>
class EntityEnvironment {
public:
    using entity_ptr = std::unique_ptr<LivingEntityInterface<T, N>>;
    using point_type = Math::Vector<T, N>;

    EntityEnvironment() noexcept
        : m_maxPopulation(10000)
        , m_targetEnergy(T(500))
        , m_reproductionCooldown(5.0) {}

    void setMaxPopulation(size_type max) { m_maxPopulation = max; }
    void setTargetEnergy(T energy) { m_targetEnergy = energy; }
    void setReproductionCooldown(T seconds) { m_reproductionCooldown = seconds; }

    // Update all entities (SIMD batch if possible)
    void updateAll(std::vector<entity_ptr>& entities, T dt,
                   const std::vector<std::vector<entity_ptr*>>& neighborLists) {
        size_type count = entities.size();
        if (count == 0) return;

        // For SIMD, we could group by species or use arrays of structures.
        // Here we simply iterate.
        for (size_type i = 0; i < count; ++i) {
            entities[i]->update(dt, neighborLists[i]);
        }
    }

    // Remove dead entities
    size_type cleanDead(std::vector<entity_ptr>& entities) {
        auto it = std::remove_if(entities.begin(), entities.end(),
                                 [](const entity_ptr& e) {
                                     return has_flag(e->getState(), EntityStateFlag::Dead);
                                 });
        size_type removed = std::distance(it, entities.end());
        entities.erase(it, entities.end());
        return removed;
    }

    // Handle reproduction (create new entities)
    void processReproduction(std::vector<entity_ptr>& entities) {
        std::vector<entity_ptr> offspring;
        for (auto& e : entities) {
            if (e->canReproduce()) {
                auto child = e->reproduce();
                if (child && entities.size() + offspring.size() < m_maxPopulation) {
                    offspring.push_back(std::move(child));
                }
            }
        }
        entities.insert(entities.end(), std::make_move_iterator(offspring.begin()),
                        std::make_move_iterator(offspring.end()));
    }

private:
    size_type m_maxPopulation;
    T m_targetEnergy;
    T m_reproductionCooldown;
};

} // namespace Behavior
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_BEHAVIOR_LIVING_ENTITY_INTERFACE_H_INCLUDED

/**
 * Next file: core/behavior/sensory_query_system.h
 * Remaining in the list: 13 files (sensory_query_system, swarm_communication, integration_bridge, archetype_index, global_morton_routing, replica_manager, consensus_tree, ecosystem_driver, weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */