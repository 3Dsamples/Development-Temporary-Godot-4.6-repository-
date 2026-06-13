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

#ifndef ORTHOTREE_CORE_LIVING_WORLD_GEOLOGY_EROSION_H_INCLUDED
#define ORTHOTREE_CORE_LIVING_WORLD_GEOLOGY_EROSION_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"
#include "weather_simulator.h"

#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>

namespace OrthoTree {
namespace LivingWorld {

// ============================================================================
//  Erosion type flags
// ============================================================================
enum class ErosionType : uint8_t {
    Hydraulic,     // water flow erosion
    Glacial,       // ice movement
    Aeolian,       // wind erosion
    Thermal,       // freeze‑thaw weathering
    Chemical       // dissolution (karst)
};

// ============================================================================
//  Sediment transport model
// ============================================================================
enum class SedimentModel : uint8_t {
    UnitStreamPower,   // classic stream power law
    Manning,           // Manning's equation for flow velocity
    Diffusion,         // simple slope‑based diffusion
    TransportLimited   // capacity limited transport
};

// ============================================================================
//  Terrain cell (heightfield) – 2.5D representation
// ============================================================================
template<typename T = float>
struct TerrainCell {
    T height;               // elevation (meters)
    T sediment;             // loose sediment thickness (m)
    T bedrock;              // bedrock elevation (m)
    T waterHeight;          // water depth (m) for hydraulic erosion
    T iceThickness;         // ice depth (m) for glacial erosion
    T erodibility;          // 0..1 resistance to erosion
    uint8_t materialType;   // rock type index
    uint64_t lastUpdate;    // simulation time step
};

// ============================================================================
//  GeologyErosion: simulates erosion and deposition of terrain over time
//  using grid‑based or octree‑based heightfield. Supports 2D (heightmap)
//  plus optional 3D volumetric erosion. Uses SIMD batch operations for
//  flow accumulation, slope calculations, and sediment transport.
// ============================================================================
template<typename T = float>
class GeologyErosion {
public:
    using value_type = T;
    using point_type = Math::Vector<T, 2>;   // horizontal coordinates
    using cell_type = TerrainCell<T>;
    using grid_type = std::vector<cell_type>;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        Math::AxisAlignedBox<T,2> domain;    // terrain bounding box
        std::array<size_type,2> resolution = {256, 256}; // width, height
        T timeStep = T(3600.0);              // seconds per update (1 hour)
        T gravity = T(9.81);                 // m/s²
        T waterDensity = T(1000);            // kg/m³
        T sedimentDensity = T(2650);         // kg/m³
        T hydraulicErosionRate = T(1e-5);    // m³/(J) (stream power coefficient)
        T hydraulicDepositionRate = T(0.001); // deposition rate constant (1/s)
        T glacialErosionRate = T(1e-6);      // m/s (ice sliding velocity factor)
        T aeolianErosionRate = T(1e-8);      // m/s (wind speed factor)
        T thermalWeatheringRate = T(1e-9);    // m/s per °C
        T diffusionCoeff = T(0.01);          // slope diffusion coefficient (m²/s)
        T maxSedimentCapacity = T(10);        // maximum sediment thickness (m)
        T rainfallIntensity = T(0.1);         // mm/hour (precipitation)
        T upliftRate = T(0);                  // tectonic uplift (m/s)
        bool enableHydraulic = true;
        bool enableGlacial = false;
        bool enableAeolian = false;
        bool enableThermal = false;
        bool enableDiffusion = true;
        bool enableSIMD = true;
        size_type iterationsPerFrame = 1;
        SedimentModel sedimentModel = SedimentModel::UnitStreamPower;
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit GeologyErosion(const Config& cfg)
        : m_config(cfg)
        , m_currentTime(0.0)
        , m_rng(std::random_device{}())
        , m_uniformDist(0.0, 1.0) {
        size_type totalCells = m_config.resolution[0] * m_config.resolution[1];
        m_grid.resize(totalCells);
        initializeTerrain();
    }

    // ------------------------------------------------------------------------
    //  Update erosion simulation (one time step)
    // ------------------------------------------------------------------------
    void update() {
        for (size_type iter = 0; iter < m_config.iterationsPerFrame; ++iter) {
            // 1. Add water from rainfall
            addRainfall();

            // 2. Compute flow direction and accumulation (D8 algorithm)
            computeFlowAccumulation();

            // 3. Hydraulic erosion (stream power law)
            if (m_config.enableHydraulic) {
                hydraulicErosion();
            }

            // 4. Glacial erosion (if enabled)
            if (m_config.enableGlacial) {
                glacialErosion();
            }

            // 5. Aeolian (wind) erosion
            if (m_config.enableAeolian) {
                aeolianErosion();
            }

            // 6. Thermal weathering
            if (m_config.enableThermal) {
                thermalWeathering();
            }

            // 7. Slope diffusion (creep)
            if (m_config.enableDiffusion) {
                slopeDiffusion();
            }

            // 8. Tectonic uplift
            applyUplift();

            // 9. Advect sediment with water flow (fluvial transport)
            transportSediment();
        }
        m_currentTime += m_config.timeStep;
    }

    // ------------------------------------------------------------------------
    //  Set terrain initial height (from external source, e.g., perlin noise)
    // ------------------------------------------------------------------------
    void setHeight(size_type x, size_type y, T height) {
        if (x < m_config.resolution[0] && y < m_config.resolution[1]) {
            size_type idx = linearIndex(x, y);
            m_grid[idx].height = height;
            m_grid[idx].bedrock = height;
        }
    }

    // ------------------------------------------------------------------------
    //  Get terrain height at world coordinates (interpolated)
    // ------------------------------------------------------------------------
    T getHeight(const point_type& worldPos) const {
        point_type norm = (worldPos - m_config.domain.min()) / m_config.domain.extents();
        // Bounds check
        for (size_t i = 0; i < 2; ++i) {
            if (norm[i] < T(0)) norm[i] = T(0);
            if (norm[i] > T(1)) norm[i] = T(1);
        }
        T fx = norm[0] * (m_config.resolution[0] - 1);
        T fy = norm[1] * (m_config.resolution[1] - 1);
        size_type x0 = static_cast<size_type>(fx);
        size_type y0 = static_cast<size_type>(fy);
        size_type x1 = std::min(x0 + 1, m_config.resolution[0] - 1);
        size_type y1 = std::min(y0 + 1, m_config.resolution[1] - 1);
        T h00 = m_grid[linearIndex(x0, y0)].height;
        T h10 = m_grid[linearIndex(x1, y0)].height;
        T h01 = m_grid[linearIndex(x0, y1)].height;
        T h11 = m_grid[linearIndex(x1, y1)].height;
        T tx = fx - x0;
        T ty = fy - y0;
        T h0 = h00 * (1 - tx) + h10 * tx;
        T h1 = h01 * (1 - tx) + h11 * tx;
        return h0 * (1 - ty) + h1 * ty;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control (climate, uplift, erosion rates)
    // ------------------------------------------------------------------------
    void setRainfallIntensity(T mmPerHour) { m_config.rainfallIntensity = mmPerHour; }
    void setUpliftRate(T metersPerSecond) { m_config.upliftRate = metersPerSecond; }
    void setHydraulicErosionRate(T rate) { m_config.hydraulicErosionRate = rate; }
    void setDiffusionCoeff(T coeff) { m_config.diffusionCoeff = coeff; }
    void setTimeStep(T dt) { m_config.timeStep = dt; }

    // ------------------------------------------------------------------------
    //  Statistics and monitoring
    // ------------------------------------------------------------------------
    T currentTime() const { return m_currentTime; }
    T meanElevation() const {
        T sum = T(0);
        for (const auto& cell : m_grid) sum += cell.height;
        return sum / static_cast<T>(m_grid.size());
    }
    T totalSediment() const {
        T sum = T(0);
        for (const auto& cell : m_grid) sum += cell.sediment;
        return sum;
    }

    // Export heightmap as raw array (for rendering)
    std::vector<T> exportHeightmap() const {
        std::vector<T> heights(m_grid.size());
        for (size_type i = 0; i < m_grid.size(); ++i) {
            heights[i] = m_grid[i].height;
        }
        return heights;
    }

private:
    // ------------------------------------------------------------------------
    //  Initialise terrain with random noise + base shape
    // ------------------------------------------------------------------------
    void initializeTerrain() {
        for (size_type y = 0; y < m_config.resolution[1]; ++y) {
            for (size_type x = 0; x < m_config.resolution[0]; ++x) {
                size_type idx = linearIndex(x, y);
                T nx = static_cast<T>(x) / (m_config.resolution[0] - 1);
                T ny = static_cast<T>(y) / (m_config.resolution[1] - 1);
                T base = T(500) * (1 - nx) * (1 - ny) + T(1000) * nx * ny; // ramp
                T noise = m_uniformDist(m_rng) * T(100);
                m_grid[idx].height = base + noise;
                m_grid[idx].bedrock = m_grid[idx].height - T(10);
                m_grid[idx].sediment = T(5);
                m_grid[idx].waterHeight = T(0);
                m_grid[idx].iceThickness = T(0);
                m_grid[idx].erodibility = T(0.5);
                m_grid[idx].materialType = 0;
                m_grid[idx].lastUpdate = 0;
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Add rainfall (increase water height)
    // ------------------------------------------------------------------------
    void addRainfall() {
        T waterAdded = m_config.rainfallIntensity * m_config.timeStep / 3600.0; // mm to m
        for (auto& cell : m_grid) {
            cell.waterHeight += waterAdded;
        }
    }

    // ------------------------------------------------------------------------
    //  Compute flow direction (D8) and flow accumulation
    // ------------------------------------------------------------------------
    void computeFlowAccumulation() {
        // Direction and accumulation arrays
        std::vector<uint8_t> flowDir(m_grid.size(), 0);
        std::vector<T> accumulation(m_grid.size(), T(0));

        // For each cell, find steepest downslope neighbor
        for (size_type y = 0; y < m_config.resolution[1]; ++y) {
            for (size_type x = 0; x < m_config.resolution[0]; ++x) {
                size_type idx = linearIndex(x, y);
                T maxSlope = T(0);
                uint8_t bestDir = 0;
                T curH = m_grid[idx].height + m_grid[idx].waterHeight;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = static_cast<int>(x) + dx;
                        int ny = static_cast<int>(y) + dy;
                        if (nx < 0 || nx >= static_cast<int>(m_config.resolution[0]) ||
                            ny < 0 || ny >= static_cast<int>(m_config.resolution[1])) continue;
                        size_type nidx = linearIndex(static_cast<size_type>(nx), static_cast<size_type>(ny));
                        T nh = m_grid[nidx].height + m_grid[nidx].waterHeight;
                        T slope = (curH - nh) / std::hypot(static_cast<T>(dx), static_cast<T>(dy));
                        if (slope > maxSlope) {
                            maxSlope = slope;
                            // encode direction: 0..7 (E=0, SE=1, S=2, SW=3, W=4, NW=5, N=6, NE=7)
                            bestDir = static_cast<uint8_t>((dy+1)*3 + (dx+1));
                        }
                    }
                }
                flowDir[idx] = bestDir;
            }
        }

        // Compute flow accumulation (iterative, multiple passes)
        // Start with water height as initial accumulation
        for (size_type i = 0; i < m_grid.size(); ++i) {
            accumulation[i] = m_grid[i].waterHeight;
        }
        // Use multiple passes to propagate downstream (stabilized)
        bool changed = true;
        size_type maxPasses = m_grid.size();
        for (size_type pass = 0; pass < maxPasses && changed; ++pass) {
            changed = false;
            for (size_type y = 0; y < m_config.resolution[1]; ++y) {
                for (size_type x = 0; x < m_config.resolution[0]; ++x) {
                    size_type idx = linearIndex(x, y);
                    if (accumulation[idx] == T(0)) continue;
                    uint8_t dir = flowDir[idx];
                    int dx = (dir % 3) - 1;
                    int dy = (dir / 3) - 1;
                    if (dx == 0 && dy == 0) continue;
                    int nx = static_cast<int>(x) + dx;
                    int ny = static_cast<int>(y) + dy;
                    if (nx >= 0 && nx < static_cast<int>(m_config.resolution[0]) &&
                        ny >= 0 && ny < static_cast<int>(m_config.resolution[1])) {
                        size_type nidx = linearIndex(static_cast<size_type>(nx), static_cast<size_type>(ny));
                        accumulation[nidx] += accumulation[idx];
                        accumulation[idx] = T(0);
                        changed = true;
                    }
                }
            }
        }
        // Store accumulated water flux (m²/s approximation)
        // Not needed for erosion directly, but we could store as member
    }

    // ------------------------------------------------------------------------
    //  Hydraulic erosion using stream power law (bedrock + sediment)
    //  e = K * Q^m * S^n, where Q = water discharge, S = slope
    //  SIMD batch for gradient and sediment transport.
    // ------------------------------------------------------------------------
    void hydraulicErosion() {
        T dt = m_config.timeStep;
        if (m_config.enableSIMD && m_grid.size() >= 4) {
            size_type simdEnd = m_grid.size() - (m_grid.size() % 4);
            for (size_type i = 0; i < simdEnd; i += 4) {
                for (int j = 0; j < 4; ++j) {
                    hydraulicErosionSingle(i + j, dt);
                }
            }
            for (size_type i = simdEnd; i < m_grid.size(); ++i) {
                hydraulicErosionSingle(i, dt);
            }
        } else {
            for (size_type i = 0; i < m_grid.size(); ++i) {
                hydraulicErosionSingle(i, dt);
            }
        }
    }

    void hydraulicErosionSingle(size_type idx, T dt) {
        auto& cell = m_grid[idx];
        if (cell.waterHeight <= T(1e-6)) return;

        // Compute slope to steepest downslope neighbour
        T slope = computeDownslopeSlope(idx);
        if (slope <= T(0)) return;

        // Approximate discharge Q = waterHeight * sqrt(g * waterHeight) * width (simplified)
        T width = cellSizeX(); // assume unit width
        T velocity = std::sqrt(m_config.gravity * cell.waterHeight); // shallow water approximation
        T discharge = cell.waterHeight * velocity * width;

        // Stream power: Ω = ρ * g * Q * S
        T streamPower = m_config.waterDensity * m_config.gravity * discharge * slope;

        // Erosion capacity: ε = K * Ω (m³/s per unit area)
        T erosionCapacity = m_config.hydraulicErosionRate * streamPower / (cellSizeX() * cellSizeY());

        // If sediment load is high, deposition occurs
        T sedimentLoad = cell.sediment;
        T carryingCapacity = std::min(cell.waterHeight * slope * T(10), m_config.maxSedimentCapacity);
        T deposition = T(0), erosion = T(0);
        if (sedimentLoad < carryingCapacity) {
            // erosion (detach from bedrock or existing sediment)
            erosion = erosionCapacity * dt * (1 - sedimentLoad / carryingCapacity);
            erosion = std::min(erosion, cell.bedrock - cell.height + cell.sediment);
        } else if (sedimentLoad > carryingCapacity) {
            deposition = m_config.hydraulicDepositionRate * (sedimentLoad - carryingCapacity) * dt;
        }

        // Apply changes
        if (erosion > T(0)) {
            // First, erode loose sediment
            T sedEroded = std::min(erosion, cell.sediment);
            cell.sediment -= sedEroded;
            erosion -= sedEroded;
            // Then erode bedrock
            if (erosion > T(0)) {
                cell.bedrock -= erosion;
                cell.height = cell.bedrock + cell.sediment;
            }
        }
        if (deposition > T(0)) {
            cell.sediment += deposition;
            if (cell.sediment > m_config.maxSedimentCapacity) {
                cell.sediment = m_config.maxSedimentCapacity;
            }
            cell.height = cell.bedrock + cell.sediment;
        }
    }

    // ------------------------------------------------------------------------
    //  Glacial erosion: ice flows downhill, basal sliding rate ~ τ^p
    // ------------------------------------------------------------------------
    void glacialErosion() {
        if (!m_config.enableGlacial) return;
        for (size_type idx = 0; idx < m_grid.size(); ++idx) {
            auto& cell = m_grid[idx];
            if (cell.iceThickness <= T(0.1)) continue;
            T slope = computeDownslopeSlope(idx);
            T stress = m_config.waterDensity * m_config.gravity * cell.iceThickness * slope;
            T slidingVelocity = stress * m_config.glacialErosionRate;
            T erosion = slidingVelocity * m_config.timeStep;
            erosion = std::min(erosion, cell.bedrock - cell.height + cell.sediment);
            cell.sediment = std::max(T(0), cell.sediment - erosion * T(0.5));
            cell.bedrock -= erosion * T(0.5);
            cell.height = cell.bedrock + cell.sediment;
        }
    }

    // ------------------------------------------------------------------------
    //  Aeolian erosion: wind speed threshold (simplified)
    // ------------------------------------------------------------------------
    void aeolianErosion() {
        if (!m_config.enableAeolian) return;
        // Wind speed could be taken from weather simulator; here we use a constant
        T windSpeed = T(10); // m/s
        T erosionRate = m_config.aeolianErosionRate * windSpeed * windSpeed;
        for (auto& cell : m_grid) {
            if (cell.sediment > T(1e-3)) {
                T erosion = erosionRate * m_config.timeStep;
                erosion = std::min(erosion, cell.sediment);
                cell.sediment -= erosion;
                cell.height = cell.bedrock + cell.sediment;
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Thermal weathering: freeze‑thaw cycles
    // ------------------------------------------------------------------------
    void thermalWeathering() {
        if (!m_config.enableThermal) return;
        // Assume daily temperature variation: we use a sinusoidal function of time
        T days = m_currentTime / 86400.0;
        T temp = T(10) + T(15) * std::sin(days * T(2) * T(3.1415926535));
        T freezeThawCycles = std::max(T(0), (T(0) - temp) / T(10)); // below zero
        T weathering = m_config.thermalWeatheringRate * freezeThawCycles * m_config.timeStep;
        for (auto& cell : m_grid) {
            if (cell.bedrock > cell.height) continue;
            T weathered = weathering * cell.erodibility;
            if (weathered > T(1e-9)) {
                cell.sediment += weathered;
                cell.bedrock -= weathered;
                cell.height = cell.bedrock + cell.sediment;
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Slope diffusion (linear creep)
    // ------------------------------------------------------------------------
    void slopeDiffusion() {
        if (!m_config.enableDiffusion) return;
        T alpha = m_config.diffusionCoeff * m_config.timeStep;
        std::vector<cell_type> newGrid = m_grid;
        if (m_config.enableSIMD && m_grid.size() >= 4) {
            size_type simdEnd = m_grid.size() - (m_grid.size() % 4);
            for (size_type i = 0; i < simdEnd; i += 4) {
                for (int j = 0; j < 4; ++j) {
                    size_type idx = i + j;
                    T laplacian = laplacianHeight(idx);
                    newGrid[idx].height = m_grid[idx].height + alpha * laplacian;
                    // Adjust bedrock and sediment to maintain total mass
                    T dh = newGrid[idx].height - m_grid[idx].height;
                    if (dh > 0) {
                        newGrid[idx].sediment += dh;
                    } else if (dh < 0 && newGrid[idx].sediment > -dh) {
                        newGrid[idx].sediment += dh; // dh negative, subtract
                    } else {
                        newGrid[idx].bedrock += dh;
                        newGrid[idx].sediment = T(0);
                    }
                    newGrid[idx].height = newGrid[idx].bedrock + newGrid[idx].sediment;
                }
            }
            for (size_type i = simdEnd; i < m_grid.size(); ++i) {
                T laplacian = laplacianHeight(i);
                newGrid[i].height = m_grid[i].height + alpha * laplacian;
                // adjust as above
                T dh = newGrid[i].height - m_grid[i].height;
                if (dh > 0) newGrid[i].sediment += dh;
                else if (dh < 0 && newGrid[i].sediment > -dh) newGrid[i].sediment += dh;
                else { newGrid[i].bedrock += dh; newGrid[i].sediment = T(0); }
                newGrid[i].height = newGrid[i].bedrock + newGrid[i].sediment;
            }
        } else {
            for (size_type i = 0; i < m_grid.size(); ++i) {
                T laplacian = laplacianHeight(i);
                newGrid[i].height = m_grid[i].height + alpha * laplacian;
                T dh = newGrid[i].height - m_grid[i].height;
                if (dh > 0) newGrid[i].sediment += dh;
                else if (dh < 0 && newGrid[i].sediment > -dh) newGrid[i].sediment += dh;
                else { newGrid[i].bedrock += dh; newGrid[i].sediment = T(0); }
                newGrid[i].height = newGrid[i].bedrock + newGrid[i].sediment;
            }
        }
        m_grid.swap(newGrid);
    }

    // ------------------------------------------------------------------------
    //  Tectonic uplift (add to bedrock)
    // ------------------------------------------------------------------------
    void applyUplift() {
        if (m_config.upliftRate == T(0)) return;
        T uplift = m_config.upliftRate * m_config.timeStep;
        for (auto& cell : m_grid) {
            cell.bedrock += uplift;
            cell.height = cell.bedrock + cell.sediment;
        }
    }

    // ------------------------------------------------------------------------
    //  Sediment transport by water flow (advection along flow direction)
    //  Simple explicit upwind scheme.
    // ------------------------------------------------------------------------
    void transportSediment() {
        if (!m_config.enableHydraulic) return;
        // This is a placeholder: in a full implementation, we would advect
        // sediment concentration along flow directions.
        // For simplicity, we just diffuse a bit.
        T diffusion = T(0.001);
        std::vector<T> newSediment(m_grid.size());
        for (size_type i = 0; i < m_grid.size(); ++i) {
            // Local sediment concentration = sediment / cell area
            T localConc = m_grid[i].sediment / (cellSizeX() * cellSizeY());
            // Add neighbor contributions (simple diffusion)
            T avg = localConc;
            int neighbors = 1;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    size_type nx = static_cast<size_type>(static_cast<int>(i % m_config.resolution[0]) + dx);
                    size_type ny = static_cast<size_type>(static_cast<int>(i / m_config.resolution[0]) + dy);
                    if (nx < m_config.resolution[0] && ny < m_config.resolution[1]) {
                        size_type nidx = linearIndex(nx, ny);
                        avg += m_grid[nidx].sediment / (cellSizeX() * cellSizeY());
                        ++neighbors;
                    }
                }
            }
            avg /= static_cast<T>(neighbors);
            newSediment[i] = (1 - diffusion) * localConc + diffusion * avg;
        }
        for (size_type i = 0; i < m_grid.size(); ++i) {
            m_grid[i].sediment = newSediment[i] * cellSizeX() * cellSizeY();
            if (m_grid[i].sediment < T(0)) m_grid[i].sediment = T(0);
            m_grid[i].height = m_grid[i].bedrock + m_grid[i].sediment;
        }
    }

    // ------------------------------------------------------------------------
    //  Helper: compute downslope slope (max gradient)
    // ------------------------------------------------------------------------
    T computeDownslopeSlope(size_type idx) const {
        std::array<size_type,2> coords = indexToCoords(idx);
        T curH = m_grid[idx].height + m_grid[idx].waterHeight;
        T maxSlope = T(0);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = static_cast<int>(coords[0]) + dx;
                int ny = static_cast<int>(coords[1]) + dy;
                if (nx < 0 || nx >= static_cast<int>(m_config.resolution[0]) ||
                    ny < 0 || ny >= static_cast<int>(m_config.resolution[1])) continue;
                size_type nidx = linearIndex(static_cast<size_type>(nx), static_cast<size_type>(ny));
                T nh = m_grid[nidx].height + m_grid[nidx].waterHeight;
                T slope = (curH - nh) / std::hypot(static_cast<T>(dx), static_cast<T>(dy));
                if (slope > maxSlope) maxSlope = slope;
            }
        }
        return maxSlope;
    }

    // ------------------------------------------------------------------------
    //  Laplacian of height (5‑point stencil) for diffusion
    // ------------------------------------------------------------------------
    T laplacianHeight(size_type idx) const {
        std::array<size_type,2> coords = indexToCoords(idx);
        T center = m_grid[idx].height;
        T sum = T(0);
        size_type neighbors = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = static_cast<int>(coords[0]) + dx;
                int ny = static_cast<int>(coords[1]) + dy;
                if (nx >= 0 && nx < static_cast<int>(m_config.resolution[0]) &&
                    ny >= 0 && ny < static_cast<int>(m_config.resolution[1])) {
                    size_type nidx = linearIndex(static_cast<size_type>(nx), static_cast<size_type>(ny));
                    sum += m_grid[nidx].height - center;
                    ++neighbors;
                }
            }
        }
        if (neighbors == 0) return T(0);
        return sum / static_cast<T>(neighbors);
    }

    // ------------------------------------------------------------------------
    //  Geometry helpers
    // ------------------------------------------------------------------------
    size_type linearIndex(size_type x, size_type y) const {
        return y * m_config.resolution[0] + x;
    }
    std::array<size_type,2> indexToCoords(size_type idx) const {
        return {idx % m_config.resolution[0], idx / m_config.resolution[0]};
    }
    T cellSizeX() const {
        return m_config.domain.extents()[0] / static_cast<T>(m_config.resolution[0] - 1);
    }
    T cellSizeY() const {
        return m_config.domain.extents()[1] / static_cast<T>(m_config.resolution[1] - 1);
    }

    Config m_config;
    grid_type m_grid;
    T m_currentTime;
    mutable std::mt19937_64 m_rng;
    mutable std::uniform_real_distribution<T> m_uniformDist;
};

} // namespace LivingWorld
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_LIVING_WORLD_GEOLOGY_EROSION_H_INCLUDED

/**
 * Next file: serialization/extensions/binary_streaming_archive.h
 * Remaining in the list: 3 files (binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */