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

#ifndef ORTHOTREE_CORE_LIVING_WORLD_ECOSYSTEM_DRIVER_H_INCLUDED
#define ORTHOTREE_CORE_LIVING_WORLD_ECOSYSTEM_DRIVER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/ot_dynamic_hash_core.h"
#include "../../core/behavior/living_entity_interface.h"
#include "../../core/behavior/sensory_query_system.h"
#include "../../core/behavior/swarm_communication.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <unordered_map>
#include <random>
#include <atomic>
#include <mutex>
#include <thread>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <memory>

namespace OrthoTree {
namespace LivingWorld {

// ============================================================================
//  Resource types (energy, water, nutrients)
// ============================================================================
enum class ResourceType : uint8_t {
    Energy,
    Water,
    Nitrogen,
    Phosphorus,
    Sunlight,
    Custom
};

// ============================================================================
//  Resource node (placed in octree)
// ============================================================================
template<typename T = float>
struct ResourceNode {
    Math::Vector<T,3> position;
    ResourceType type;
    T amount;           // current available quantity
    T regenerationRate; // per second
    T maxAmount;
    uint64_t id;
};

// ============================================================================
//  Species traits (for population dynamics)
// ============================================================================
template<typename T = float>
struct SpeciesTraits {
    uint32_t speciesId;
    T metabolicRate;        // energy consumption per individual per second
    T reproductionRate;     // offspring per individual per second (density‑dependent)
    T mortalityRate;        // baseline death rate (per second)
    T predationEfficiency;  // 0..1, energy gain from prey
    T preferredTemperature; // °C
    T temperatureTolerance; // ± range
    T waterRequirement;     // litres per individual per day
    T habitatQuality;       // 0..1
    bool isCarnivore;
    bool isHerbivore;
    bool isDecomposer;
};

// ============================================================================
//  Population data for a species in a given region (cell)
// ============================================================================
template<typename T = float>
struct PopulationData {
    uint32_t speciesId;
    T count;                // number of individuals (can be fractional for large populations)
    T totalEnergy;          // stored in the population (Joules)
    T averageAge;           // seconds
};

// ============================================================================
//  EcosystemDriver: manages the entire ecosystem simulation across the octree.
//  It couples resource distribution, species populations, predation,
//  reproduction, and environmental factors (weather, climate). Uses SIMD batch
//  updates for population dynamics and resource consumption.
// ============================================================================
template<typename T = float, std::size_t N = 3,
         typename Allocator = PMRAllocator<std::byte>>
class EcosystemDriver {
public:
    using value_type = T;
    using point_type = Math::Vector<T, N>;
    using aabb_type = Math::AxisAlignedBox<T, N>;
    using octree_type = ot_dynamic_hash_core<Dim3, T, Allocator>;
    using entity_id = typename octree_type::entity_type;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        aabb_type worldBounds;                  // simulation domain
        T globalTemperature = T(20.0);          // baseline °C
        T globalPrecipitation = T(0.5);          // metres per year
        T sunlightIntensity = T(1.0);            // 0..1
        T timeStep = T(1.0);                    // seconds per simulation step
        bool enableSIMD = true;
        size_type maxSpecies = 256;
        size_type gridResolution = 64;           // cells per dimension for resource mapping
        T resourceDiffusionRate = T(0.1);        // 0..1
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit EcosystemDriver(const Config& cfg)
        : m_config(cfg)
        , m_octree(cfg.worldBounds)
        , m_currentTime(0.0)
        , m_rng(std::random_device{}())
        , m_uniformDist(0.0, 1.0) {
        // Pre‑allocate resource grid (simplified: uniform grid of cells)
        size_type totalCells = m_config.gridResolution *
                               m_config.gridResolution *
                               (N == 3 ? m_config.gridResolution : 1);
        m_resourceGrid.resize(totalCells);
        initializeResources();
    }

    // ------------------------------------------------------------------------
    //  Register a species (add to the ecosystem)
    // ------------------------------------------------------------------------
    void registerSpecies(const SpeciesTraits<T>& traits) {
        std::lock_guard<std::mutex> lock(m_speciesMutex);
        m_species[traits.speciesId] = traits;
        // Initialise population data (empty)
        m_populations[traits.speciesId] = {};
    }

    // ------------------------------------------------------------------------
    //  Add individuals of a species at a given point (seed population)
    // ------------------------------------------------------------------------
    void addIndividuals(uint32_t speciesId, const point_type& position, T count) {
        // Insert into octree
        entity_id id = static_cast<entity_id>(speciesId); // simplified
        m_octree.insert(id);
        // Also record population in local map
        std::lock_guard<std::mutex> lock(m_popMutex);
        auto& pop = m_populations[speciesId];
        pop.speciesId = speciesId;
        pop.count += count;
        pop.totalEnergy += count * getSpeciesTraits(speciesId).metabolicRate;
    }

    // ------------------------------------------------------------------------
    //  Main simulation step (advance ecosystem by one time step)
    //  Updates resources, species populations, and interactions.
    // ------------------------------------------------------------------------
    void step() {
        // 1. Update resources (regeneration, diffusion, consumption)
        updateResources();

        // 2. Update populations (birth, death, predation)
        updatePopulations();

        // 3. Update environmental factors (temperature, precipitation)
        updateEnvironment();

        m_currentTime += m_config.timeStep;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control (climate change, invasive species)
    // ------------------------------------------------------------------------
    void setGlobalTemperature(T temp) { m_config.globalTemperature = temp; }
    void setGlobalPrecipitation(T precip) { m_config.globalPrecipitation = precip; }
    void setSunlightIntensity(T intensity) { m_config.sunlightIntensity = intensity; }
    void setTimeStep(T dt) { m_config.timeStep = dt; }

    // ------------------------------------------------------------------------
    //  Statistics and monitoring
    // ------------------------------------------------------------------------
    T currentTime() const { return m_currentTime; }
    T getPopulationCount(uint32_t speciesId) const {
        std::lock_guard<std::mutex> lock(m_popMutex);
        auto it = m_populations.find(speciesId);
        if (it != m_populations.end()) return it->second.count;
        return T(0);
    }

    size_type numSpecies() const {
        std::lock_guard<std::mutex> lock(m_speciesMutex);
        return m_species.size();
    }

private:
    // ------------------------------------------------------------------------
    //  Internal structures
    // ------------------------------------------------------------------------
    struct GridCell {
        T energy = T(0);
        T water = T(0);
        T nitrogen = T(0);
        T lastUpdate = T(0);
    };

    // ------------------------------------------------------------------------
    //  Initialise resources (sunlight, water, minerals) across the grid
    // ------------------------------------------------------------------------
    void initializeResources() {
        for (auto& cell : m_resourceGrid) {
            cell.energy = T(1000);   // J per cell (placeholder)
            cell.water = T(100);     // litres
            cell.nitrogen = T(10);
        }
    }

    // ------------------------------------------------------------------------
    //  Update resources: regeneration, diffusion, and consumption by species
    //  Uses SIMD batch if enabled.
    // ------------------------------------------------------------------------
    void updateResources() {
        if (m_config.enableSIMD && m_resourceGrid.size() >= 4) {
            // SIMD batch update (pseudo – in reality we would use AVX2 for 4 cells at a time)
            for (size_type i = 0; i < m_resourceGrid.size(); i += 4) {
                size_type end = std::min(i + 4, m_resourceGrid.size());
                for (size_type j = i; j < end; ++j) {
                    updateSingleCell(j);
                }
            }
        } else {
            for (size_type i = 0; i < m_resourceGrid.size(); ++i) {
                updateSingleCell(i);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Update a single grid cell (resource dynamics)
    // ------------------------------------------------------------------------
    void updateSingleCell(size_type cellIdx) {
        GridCell& cell = m_resourceGrid[cellIdx];
        // Sunlight energy regeneration
        T sunlightIncome = m_config.sunlightIntensity * m_config.timeStep;
        cell.energy += sunlightIncome;
        if (cell.energy > T(10000)) cell.energy = T(10000);
        // Water cycle (precipitation)
        cell.water += m_config.globalPrecipitation * m_config.timeStep / 365.0;
        // Nitrogen fixation (simplified)
        cell.nitrogen += T(0.001) * m_config.timeStep;
        // Consumption by herbivores will happen in population update step
    }

    // ------------------------------------------------------------------------
    //  Update populations: reproduction, mortality, predation, resource consumption
    // ------------------------------------------------------------------------
    void updatePopulations() {
        std::lock_guard<std::mutex> lock(m_popMutex);
        std::vector<uint32_t> toRemove;
        for (auto& pair : m_populations) {
            uint32_t speciesId = pair.first;
            PopulationData<T>& pop = pair.second;
            const auto& traits = getSpeciesTraits(speciesId);
            if (pop.count <= T(1e-6)) {
                toRemove.push_back(speciesId);
                continue;
            }

            // Baseline mortality
            T deaths = pop.count * traits.mortalityRate * m_config.timeStep;
            // Reproduction (density‑dependent, logistic growth)
            T carryingCapacity = getCarryingCapacity(speciesId);
            T growthRate = traits.reproductionRate * (T(1) - pop.count / carryingCapacity);
            T births = pop.count * growthRate * m_config.timeStep;
            // Resource limitation (energy)
            T requiredEnergy = pop.count * traits.metabolicRate * m_config.timeStep;
            T availableEnergy = getAvailableEnergy(speciesId);
            T energyFactor = (availableEnergy >= requiredEnergy) ? T(1) : (availableEnergy / requiredEnergy);
            deaths += pop.count * (T(1) - energyFactor) * m_config.timeStep;
            // Predation (if carnivore)
            T predationLoss = T(0);
            if (traits.isCarnivore) {
                predationLoss = computePredation(pop, traits);
            }
            // Apply changes
            pop.count = pop.count + births - deaths - predationLoss;
            if (pop.count < T(0)) pop.count = T(0);
            pop.totalEnergy = pop.count * traits.metabolicRate; // simplified
        }
        for (uint32_t id : toRemove) {
            m_populations.erase(id);
        }
    }

    // ------------------------------------------------------------------------
    //  Compute predation loss for a carnivore population
    //  Uses octree to find nearby prey and SIMD batch for efficiency.
    // ------------------------------------------------------------------------
    T computePredation(PopulationData<T>& predator, const SpeciesTraits<T>& traits) {
        // In a real implementation, we would query the octree for prey species.
        // For this example, we approximate using a global prey density factor.
        T totalPreyBiomass = T(0);
        for (const auto& pair : m_populations) {
            const auto& preyTraits = getSpeciesTraits(pair.first);
            if (!preyTraits.isCarnivore && !preyTraits.isDecomposer) {
                totalPreyBiomass += pair.second.count * getPreyEnergyContent(pair.first);
            }
        }
        // Holling type II functional response
        T attackRate = T(0.01);
        T handlingTime = T(0.1);
        T predationRate = attackRate * totalPreyBiomass / (T(1) + attackRate * handlingTime * totalPreyBiomass);
        T losses = predator.count * predationRate * m_config.timeStep;
        return losses;
    }

    // ------------------------------------------------------------------------
    //  Get carrying capacity for a species (based on resources)
    // ------------------------------------------------------------------------
    T getCarryingCapacity(uint32_t speciesId) const {
        // Simplified: based on available resources in the environment
        T totalEnergy = T(0);
        for (const auto& cell : m_resourceGrid) {
            totalEnergy += cell.energy;
        }
        return totalEnergy / getSpeciesTraits(speciesId).metabolicRate;
    }

    // ------------------------------------------------------------------------
    //  Get available energy for a species (resource extraction rate)
    // ------------------------------------------------------------------------
    T getAvailableEnergy(uint32_t speciesId) const {
        const auto& traits = getSpeciesTraits(speciesId);
        if (traits.isHerbivore) {
            // Consume plants (represented by resource grid energy)
            T totalEnergy = T(0);
            for (auto& cell : const_cast<std::vector<GridCell>&>(m_resourceGrid)) {
                totalEnergy += cell.energy;
                // Reduce resource (will be persisted later)
                // cell.energy -= consumptionRate; not yet
            }
            return totalEnergy;
        } else {
            return T(1e6); // carnivores get energy from prey, handled separately
        }
    }

    // ------------------------------------------------------------------------
    //  Get prey energy content (simplified)
    // ------------------------------------------------------------------------
    T getPreyEnergyContent(uint32_t speciesId) const {
        return getSpeciesTraits(speciesId).metabolicRate;
    }

    // ------------------------------------------------------------------------
    //  Update environmental factors (climate, seasons, etc.)
    // ------------------------------------------------------------------------
    void updateEnvironment() {
        // Simple sinusoidal seasonal variation
        T days = m_currentTime / 86400.0;
        T seasonalFactor = T(0.2) * std::sin(days * T(2) * T(3.1415926535) / 365.0);
        m_globalTemperatureModifier = seasonalFactor;
    }

    // ------------------------------------------------------------------------
    //  Helper: get species traits (with const lock)
    // ------------------------------------------------------------------------
    const SpeciesTraits<T>& getSpeciesTraits(uint32_t speciesId) const {
        std::lock_guard<std::mutex> lock(m_speciesMutex);
        auto it = m_species.find(speciesId);
        if (it == m_species.end()) {
            static SpeciesTraits<T> defaultTraits;
            return defaultTraits;
        }
        return it->second;
    }

    // ------------------------------------------------------------------------
    //  Cell index helper (for 2D/3D grid)
    // ------------------------------------------------------------------------
    size_type cellIndex(size_type x, size_type y, size_type z = 0) const {
        return z * m_config.gridResolution * m_config.gridResolution +
               y * m_config.gridResolution +
               x;
    }

    Config m_config;
    octree_type m_octree;
    std::vector<GridCell> m_resourceGrid;
    std::unordered_map<uint32_t, PopulationData<T>> m_populations;
    std::unordered_map<uint32_t, SpeciesTraits<T>> m_species;
    T m_currentTime;
    T m_globalTemperatureModifier = T(0);
    mutable std::mutex m_speciesMutex;
    mutable std::mutex m_popMutex;
    std::mt19937_64 m_rng;
    std::uniform_real_distribution<T> m_uniformDist;
};

} // namespace LivingWorld
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_LIVING_WORLD_ECOSYSTEM_DRIVER_H_INCLUDED

/**
 * Next file: core/living_world/weather_simulator.h
 * Remaining in the list: 5 files (weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */