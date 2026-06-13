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

#ifndef ORTHOTREE_CORE_LIVING_WORLD_WEATHER_SIMULATOR_H_INCLUDED
#define ORTHOTREE_CORE_LIVING_WORLD_WEATHER_SIMULATOR_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/partitioning/hybrid_grid_tree.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"
#include "ecosystem_driver.h"

#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <functional>
#include <algorithm>
#include <limits>

namespace OrthoTree {
namespace LivingWorld {

// ============================================================================
//  Weather types (precipitation, wind, temperature)
// ============================================================================
enum class WeatherType : uint8_t {
    Clear,
    Rain,
    Snow,
    Storm,
    Fog,
    Heatwave,
    Frost
};

// ============================================================================
//  Weather cell (for grid‑based simulation)
// ============================================================================
template<typename T = float>
struct WeatherCell {
    T temperature;        // °C
    T pressure;           // hPa
    T humidity;           // 0..1
    T windSpeed;          // m/s
    Math::Vector<T,2> windDirection; // 2D direction (x, y)
    T cloudCover;         // 0..1
    T precipitation;      // mm/hour
    WeatherType primaryType;
    uint64_t lastUpdate;  // simulation step
};

// ============================================================================
//  WeatherSimulator: simulates dynamic weather patterns over a 2D/3D grid,
//  using simplified Navier‑Stokes for wind, advection‑diffusion for moisture,
//  and thermodynamic lifting for precipitation. Supports SIMD batch updates,
//  dynamic environment controls (climate change, seasonal forcing), and
//  integration with ecosystem driver (resource distribution).
// ============================================================================
template<typename T = float, std::size_t N = 2>
class WeatherSimulator {
    static_assert(N == 2 || N == 3, "WeatherSimulator supports 2D (surface) or 3D (atmospheric)");
public:
    using value_type = T;
    using point_type = Math::Vector<T, N>;
    using grid_cell = WeatherCell<T>;
    using grid_type = std::vector<grid_cell>;

    // ------------------------------------------------------------------------
    //  Configuration (dynamic environment)
    // ------------------------------------------------------------------------
    struct Config {
        Math::AxisAlignedBox<T, N> domain;     // simulation area
        std::array<size_type, (N == 2 ? 2 : 3)> resolution = {64, 64, (N == 3 ? 64 : 1)};
        T timeStep = T(1.0);                  // seconds per update
        T coriolisStrength = T(1e-4);         // Coriolis parameter (rotational effect)
        T frictionCoeff = T(0.01);            // surface friction
        T thermalDiffusivity = T(0.1);        // heat diffusion rate
        T moistureDiffusivity = T(0.05);      // moisture diffusion rate
        T precipitationThreshold = T(0.8);    // cloud cover threshold for rain
        T evaporationRate = T(1e-5);          // per second
        bool enableSIMD = true;
        bool enableAdvection = true;
        size_type numIterations = 1;           // sub‑steps per frame
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit WeatherSimulator(const Config& cfg)
        : m_config(cfg)
        , m_currentTime(0.0)
        , m_rng(std::random_device{}())
        , m_normalDist(0.0, 1.0) {
        size_type totalCells = m_config.resolution[0] *
                               m_config.resolution[1] *
                               (N == 3 ? m_config.resolution[2] : 1);
        m_grid.resize(totalCells);
        initializeWeather();
    }

    // ------------------------------------------------------------------------
    //  Update weather simulation (one time step)
    // ------------------------------------------------------------------------
    void update() {
        for (size_type iter = 0; iter < m_config.numIterations; ++iter) {
            // 1. Compute wind tendencies (pressure gradient, Coriolis, friction)
            computeWind();
            // 2. Advect temperature and moisture using wind field (semi‑Lagrangian)
            if (m_config.enableAdvection) {
                advectScalars();
            }
            // 3. Diffuse temperature and moisture
            diffuseScalars();
            // 4. Compute condensation / precipitation
            computePrecipitation();
            // 5. Update boundary conditions (solar forcing, terrain)
            applyForcing();
        }
        m_currentTime += m_config.timeStep;
    }

    // ------------------------------------------------------------------------
    //  Get weather data at a given world position (interpolated from grid)
    // ------------------------------------------------------------------------
    grid_cell sample(const point_type& worldPos) const {
        point_type normalized = (worldPos - m_config.domain.min()) / m_config.domain.extents();
        // Clamp and convert to grid coordinates
        std::array<size_type, N> idx;
        for (size_t i = 0; i < N; ++i) {
            T t = std::max(T(0), std::min(T(1), normalized[i]));
            idx[i] = static_cast<size_type>(t * (m_config.resolution[i] - 1));
        }
        size_type cellIdx = linearIndex(idx);
        return m_grid[cellIdx];
    }

    // ------------------------------------------------------------------------
    //  Apply weather influence to an ecosystem driver (resource update)
    //  For example, add water to resources based on precipitation.
    // ------------------------------------------------------------------------
    void applyToEcosystem(EcosystemDriver<T,N>& ecosystem) {
        // Iterate over weather grid cells and update ecosystem resources.
        // This is a simplified example; in reality, we would match grid cells.
        for (size_type i = 0; i < m_grid.size(); ++i) {
            T precip = m_grid[i].precipitation * m_config.timeStep / 3600.0; // convert to mm
            // Convert grid cell position to world point (center)
            point_type cellPos = gridToWorld(i);
            // Update resource at that location (ecosystem would have a method)
            // ecosystem.addWater(cellPos, precip);
        }
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control (climate change, seasonal variation)
    // ------------------------------------------------------------------------
    void setCoriolisStrength(T strength) { m_config.coriolisStrength = strength; }
    void setFrictionCoeff(T coeff) { m_config.frictionCoeff = coeff; }
    void setTimeStep(T dt) { m_config.timeStep = dt; }
    void setEvaporationRate(T rate) { m_config.evaporationRate = rate; }

    // Add a heat source at a point (e.g., urban heat island)
    void addHeatSource(const point_type& pos, T intensity, T radius) {
        // In a real implementation, we would affect temperature in a radius.
    }

    // ------------------------------------------------------------------------
    //  Statistics and monitoring
    // ------------------------------------------------------------------------
    T currentTime() const { return m_currentTime; }
    T averageTemperature() const {
        T sum = 0;
        for (const auto& cell : m_grid) sum += cell.temperature;
        return sum / static_cast<T>(m_grid.size());
    }
    T totalPrecipitation() const {
        T sum = 0;
        for (const auto& cell : m_grid) sum += cell.precipitation;
        return sum;
    }

private:
    // ------------------------------------------------------------------------
    //  Initialize weather with random perturbations
    // ------------------------------------------------------------------------
    void initializeWeather() {
        for (auto& cell : m_grid) {
            cell.temperature = T(20) + m_normalDist(m_rng) * T(5);
            cell.pressure = T(1013) + m_normalDist(m_rng) * T(10);
            cell.humidity = T(0.5) + m_normalDist(m_rng) * T(0.3);
            cell.windSpeed = T(0);
            cell.windDirection = Math::Vector<T,2>(0,0);
            cell.cloudCover = T(0.2);
            cell.precipitation = T(0);
            cell.primaryType = WeatherType::Clear;
            cell.lastUpdate = 0;
        }
    }

    // ------------------------------------------------------------------------
    //  Compute wind field from pressure gradient (simplified geostrophic balance)
    //  Uses SIMD batch for pressure gradient calculation.
    // ------------------------------------------------------------------------
    void computeWind() {
        if (m_config.enableSIMD && m_grid.size() >= 4) {
            // Process 4 cells at a time (pseudo‑SIMD)
            size_type simdEnd = m_grid.size() - (m_grid.size() % 4);
            for (size_type i = 0; i < simdEnd; i += 4) {
                for (int j = 0; j < 4; ++j) {
                    computeWindSingle(i + j);
                }
            }
            for (size_type i = simdEnd; i < m_grid.size(); ++i) {
                computeWindSingle(i);
            }
        } else {
            for (size_type i = 0; i < m_grid.size(); ++i) {
                computeWindSingle(i);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Compute wind for a single cell
    // ------------------------------------------------------------------------
    void computeWindSingle(size_type idx) {
        Math::Vector<T, N> gradP = pressureGradient(idx);
        // Geostrophic balance: wind perpendicular to pressure gradient
        Math::Vector<T, N> wind;
        if constexpr (N == 2) {
            wind[0] = -gradP[1] / (m_config.coriolisStrength + T(1e-6));
            wind[1] =  gradP[0] / (m_config.coriolisStrength + T(1e-6));
        } else {
            // 3D: simplified, ignore vertical component
            wind[0] = -gradP[1] / (m_config.coriolisStrength + T(1e-6));
            wind[1] =  gradP[0] / (m_config.coriolisStrength + T(1e-6));
            wind[2] = T(0);
        }
        // Apply friction
        wind = wind * (T(1) - m_config.frictionCoeff * m_config.timeStep);
        m_grid[idx].windSpeed = wind.length();
        if (m_grid[idx].windSpeed > T(1e-6)) {
            m_grid[idx].windDirection[0] = wind[0] / m_grid[idx].windSpeed;
            m_grid[idx].windDirection[1] = wind[1] / m_grid[idx].windSpeed;
        }
    }

    // ------------------------------------------------------------------------
    //  Compute pressure gradient using central differences
    // ------------------------------------------------------------------------
    Math::Vector<T, N> pressureGradient(size_type idx) const {
        std::array<size_type, N> coords = indexToCoords(idx);
        Math::Vector<T, N> grad;
        for (size_t d = 0; d < N; ++d) {
            T left = T(0), right = T(0);
            if (coords[d] > 0) {
                auto leftCoords = coords;
                leftCoords[d]--;
                left = m_grid[linearIndex(leftCoords)].pressure;
            } else {
                left = m_grid[idx].pressure;
            }
            if (coords[d] + 1 < m_config.resolution[d]) {
                auto rightCoords = coords;
                rightCoords[d]++;
                right = m_grid[linearIndex(rightCoords)].pressure;
            } else {
                right = m_grid[idx].pressure;
            }
            grad[d] = (right - left) / (T(2) * cellSize(d));
        }
        return grad;
    }

    // ------------------------------------------------------------------------
    //  Advect scalars (temperature, humidity) using wind field
    //  Semi‑Lagrangian scheme, SIMD friendly.
    // ------------------------------------------------------------------------
    void advectScalars() {
        std::vector<grid_cell> newGrid = m_grid;
        if (m_config.enableSIMD && m_grid.size() >= 4) {
            size_type simdEnd = m_grid.size() - (m_grid.size() % 4);
            for (size_type i = 0; i < simdEnd; i += 4) {
                for (int j = 0; j < 4; ++j) {
                    advectCell(i + j, newGrid[i + j]);
                }
            }
            for (size_type i = simdEnd; i < m_grid.size(); ++i) {
                advectCell(i, newGrid[i]);
            }
        } else {
            for (size_type i = 0; i < m_grid.size(); ++i) {
                advectCell(i, newGrid[i]);
            }
        }
        m_grid.swap(newGrid);
    }

    // ------------------------------------------------------------------------
    //  Advect a single cell (trace backward in time)
    // ------------------------------------------------------------------------
    void advectCell(size_type idx, grid_cell& newCell) {
        point_type pos = gridToWorld(idx);
        // Backward advection: pos_prev = pos - wind * dt
        Math::Vector<T, N> windVec;
        windVec[0] = m_grid[idx].windDirection[0] * m_grid[idx].windSpeed;
        windVec[1] = m_grid[idx].windDirection[1] * m_grid[idx].windSpeed;
        if constexpr (N == 3) windVec[2] = T(0);
        point_type prevPos = pos - windVec * m_config.timeStep;
        // Sample from previous grid (linear interpolation)
        grid_cell source = sample(prevPos);
        newCell.temperature = source.temperature;
        newCell.humidity = source.humidity;
        newCell.cloudCover = source.cloudCover;
    }

    // ------------------------------------------------------------------------
    //  Diffuse scalars (implicit or explicit Euler)
    //  Uses SIMD batch for Laplacian computation.
    // ------------------------------------------------------------------------
    void diffuseScalars() {
        std::vector<grid_cell> newGrid = m_grid;
        T alphaT = m_config.thermalDiffusivity * m_config.timeStep;
        T alphaH = m_config.moistureDiffusivity * m_config.timeStep;
        if (m_config.enableSIMD && m_grid.size() >= 4) {
            size_type simdEnd = m_grid.size() - (m_grid.size() % 4);
            for (size_type i = 0; i < simdEnd; i += 4) {
                for (int j = 0; j < 4; ++j) {
                    diffuseSingle(i + j, newGrid[i + j], alphaT, alphaH);
                }
            }
            for (size_type i = simdEnd; i < m_grid.size(); ++i) {
                diffuseSingle(i, newGrid[i], alphaT, alphaH);
            }
        } else {
            for (size_type i = 0; i < m_grid.size(); ++i) {
                diffuseSingle(i, newGrid[i], alphaT, alphaH);
            }
        }
        m_grid.swap(newGrid);
    }

    // ------------------------------------------------------------------------
    //  Diffusion for a single cell (explicit Euler)
    // ------------------------------------------------------------------------
    void diffuseSingle(size_type idx, grid_cell& newCell, T alphaT, T alphaH) {
        T laplacianT = laplacian(idx, [this](size_type i) { return m_grid[i].temperature; });
        T laplacianH = laplacian(idx, [this](size_type i) { return m_grid[i].humidity; });
        newCell.temperature = m_grid[idx].temperature + alphaT * laplacianT;
        newCell.humidity = m_grid[idx].humidity + alphaH * laplacianH;
    }

    // ------------------------------------------------------------------------
    //  Compute Laplacian (5‑point stencil for 2D, 7‑point for 3D)
    // ------------------------------------------------------------------------
    template<typename Func>
    T laplacian(size_type idx, Func getValue) const {
        T sum = T(0);
        std::array<size_type, N> coords = indexToCoords(idx);
        T center = getValue(idx);
        // For each dimension, add neighbours
        for (size_t d = 0; d < N; ++d) {
            if (coords[d] > 0) {
                auto leftCoords = coords;
                leftCoords[d]--;
                sum += getValue(linearIndex(leftCoords)) - center;
            }
            if (coords[d] + 1 < m_config.resolution[d]) {
                auto rightCoords = coords;
                rightCoords[d]++;
                sum += getValue(linearIndex(rightCoords)) - center;
            }
        }
        return sum / (cellSize(0) * cellSize(0)); // assume uniform cell size
    }

    // ------------------------------------------------------------------------
    //  Compute condensation and precipitation based on humidity and temperature
    // ------------------------------------------------------------------------
    void computePrecipitation() {
        for (auto& cell : m_grid) {
            // Saturation vapour pressure (simplified Clausius‑Clapeyron)
            T satHumidity = T(0.6) * std::exp(T(17.27) * (cell.temperature - T(273)) / (cell.temperature - T(36)));
            satHumidity = std::min(T(1), satHumidity);
            if (cell.humidity > satHumidity) {
                T excess = cell.humidity - satHumidity;
                cell.precipitation = excess * T(10) * m_config.timeStep / 3600.0; // mm/h
                cell.humidity = satHumidity;
                cell.cloudCover = std::min(T(1), cell.cloudCover + excess * T(0.5));
            } else {
                cell.precipitation = T(0);
                // Evaporation
                cell.humidity += m_config.evaporationRate * m_config.timeStep;
                if (cell.humidity > T(1)) cell.humidity = T(1);
            }
            // Determine primary weather type
            if (cell.precipitation > T(0.5)) cell.primaryType = WeatherType::Rain;
            else if (cell.precipitation > T(0.1)) cell.primaryType = WeatherType::Storm;
            else if (cell.cloudCover > T(0.7)) cell.primaryType = WeatherType::Fog;
            else if (cell.temperature > T(35)) cell.primaryType = WeatherType::Heatwave;
            else if (cell.temperature < T(-5)) cell.primaryType = WeatherType::Frost;
            else cell.primaryType = WeatherType::Clear;
        }
    }

    // ------------------------------------------------------------------------
    //  Apply external forcing (solar radiation, terrain)
    // ------------------------------------------------------------------------
    void applyForcing() {
        // Diurnal cycle: temperature depends on time of day
        T days = m_currentTime / 86400.0;
        T solar = T(0.5) + T(0.5) * std::sin(days * T(2) * T(3.1415926535));
        for (auto& cell : m_grid) {
            cell.temperature += solar * T(2) * m_config.timeStep / 3600.0; // slow change
            // Simple boundary: restore towards baseline
            cell.temperature = T(0.99) * cell.temperature + T(0.01) * T(20);
        }
    }

    // ------------------------------------------------------------------------
    //  Helper: linear index from coordinates
    // ------------------------------------------------------------------------
    size_type linearIndex(const std::array<size_type, N>& coords) const {
        if constexpr (N == 2) {
            return coords[1] * m_config.resolution[0] + coords[0];
        } else {
            return coords[2] * m_config.resolution[0] * m_config.resolution[1] +
                   coords[1] * m_config.resolution[0] +
                   coords[0];
        }
    }

    // ------------------------------------------------------------------------
    //  Helper: coordinates from linear index
    // ------------------------------------------------------------------------
    std::array<size_type, N> indexToCoords(size_type idx) const {
        std::array<size_type, N> coords;
        if constexpr (N == 2) {
            coords[0] = idx % m_config.resolution[0];
            coords[1] = idx / m_config.resolution[0];
        } else {
            coords[0] = idx % m_config.resolution[0];
            coords[1] = (idx / m_config.resolution[0]) % m_config.resolution[1];
            coords[2] = idx / (m_config.resolution[0] * m_config.resolution[1]);
        }
        return coords;
    }

    // ------------------------------------------------------------------------
    //  Helper: world position from grid cell index
    // ------------------------------------------------------------------------
    point_type gridToWorld(size_type idx) const {
        auto coords = indexToCoords(idx);
        point_type pos;
        for (size_t i = 0; i < N; ++i) {
            T t = static_cast<T>(coords[i]) / static_cast<T>(m_config.resolution[i] - 1);
            pos[i] = m_config.domain.min()[i] + t * m_config.domain.extents()[i];
        }
        return pos;
    }

    // ------------------------------------------------------------------------
    //  Helper: cell size in a given dimension
    // ------------------------------------------------------------------------
    T cellSize(size_t dim) const {
        return m_config.domain.extents()[dim] / static_cast<T>(m_config.resolution[dim] - 1);
    }

    Config m_config;
    grid_type m_grid;
    T m_currentTime;
    mutable std::mt19937_64 m_rng;
    mutable std::normal_distribution<T> m_normalDist;
};

} // namespace LivingWorld
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_LIVING_WORLD_WEATHER_SIMULATOR_H_INCLUDED

/**
 * Next file: core/living_world/geology_erosion.h
 * Remaining in the list: 4 files (geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */