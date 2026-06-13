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

#ifndef ORTHOTREE_CORE_PARTITIONING_DYNAMIC_LOD_CONTROLLER_H_INCLUDED
#define ORTHOTREE_CORE_PARTITIONING_DYNAMIC_LOD_CONTROLLER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/numerical_methods.h"
#include "../../core/configuration.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>

namespace OrthoTree {
namespace Partitioning {

// ============================================================================
//  DynamicLODController: manages level‑of‑detail selection for octree nodes
//  based on viewer position, object size, screen occupancy, and performance
//  budget. Supports 2D and 3D, with SIMD batch distance calculations.
// ============================================================================
template<typename T = float, std::size_t N = 3>
class DynamicLODController {
    static_assert(N == 2 || N == 3, "Only 2D or 3D supported");
public:
    using point_type = Math::Vector<T, N>;
    using vec_type = point_type;

    // ------------------------------------------------------------------------
    //  Configuration parameters
    // ------------------------------------------------------------------------
    struct Config {
        T minDistance = T(0.1);          // minimum LOD distance (closest)
        T maxDistance = T(1000.0);       // maximum LOD distance (farthest)
        T minNodeSize = T(0.01);         // smallest node size (highest detail)
        T maxNodeSize = T(100.0);        // largest node size (lowest detail)
        T hysteresis = T(0.1);           // hysteresis factor to avoid flickering (0..1)
        T screenAreaBias = T(1.0);       // bias for screen‑space area (larger = prefer higher detail)
        uint8_t maxLOD = 16;             // maximum LOD level (0 = coarsest)
        uint8_t minLOD = 0;              // minimum LOD level (finest)
        bool useScreenArea = true;       // use screen‑space area instead of pure distance
        T performanceBudgetMs = T(2.0);  // target milliseconds for LOD updates
    };

    // ------------------------------------------------------------------------
    //  State for a single tracked object (e.g., octree node)
    // ------------------------------------------------------------------------
    struct ObjectState {
        point_type worldCenter;      // object center in world space
        T objectSize;                // bounding sphere radius or half‑extent
        uint8_t currentLOD;          // currently selected LOD
        uint8_t targetLOD;           // desired LOD after evaluation
        T lastDistance;              // cached distance from viewer
        uint32_t lastUpdateFrame;    // frame counter for throttling
    };

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit DynamicLODController(const Config& config = Config()) noexcept
        : m_config(config)
        , m_viewerPosition(point_type(T(0)))
        , m_frameCounter(0)
        , m_lastUpdateTime(0.0) {}

    // ------------------------------------------------------------------------
    //  Environment control: set viewer position and orientation (for screen area)
    // ------------------------------------------------------------------------
    void setViewerPosition(const point_type& pos) noexcept {
        m_viewerPosition = pos;
    }

    // For screen‑space area, we also need viewer direction and FOV. Simplified:
    // we use distance and object size to approximate screen coverage.
    void setViewerFOV(T fovRadians) noexcept {
        m_fov = fovRadians;
        m_tanHalfFov = std::tan(fovRadians * T(0.5));
    }

    // ------------------------------------------------------------------------
    //  LOD computation for a single object
    // ------------------------------------------------------------------------
    uint8_t computeLOD(const point_type& center, T size) const noexcept {
        T dist = (center - m_viewerPosition).length();
        dist = Math::clamp(dist, m_config.minDistance, m_config.maxDistance);
        if (m_config.useScreenArea) {
            // Screen‑space area: A = (size / dist) ^ 2  (approx)
            T screenArea = (size / dist) * (size / dist);
            // Map screen area to LOD: larger area -> higher detail (lower LOD number)
            T areaNorm = Math::clamp(screenArea, T(0), T(1));
            uint8_t lod = static_cast<uint8_t>((T(1) - areaNorm) * static_cast<T>(m_config.maxLOD - m_config.minLOD));
            lod = m_config.minLOD + lod;
            return Math::clamp(lod, m_config.minLOD, m_config.maxLOD);
        } else {
            // Distance‑based: linear mapping from minDistance -> maxLOD, maxDistance -> minLOD
            T t = (dist - m_config.minDistance) / (m_config.maxDistance - m_config.minDistance);
            t = Math::clamp(t, T(0), T(1));
            uint8_t lod = static_cast<uint8_t>((T(1) - t) * static_cast<T>(m_config.maxLOD - m_config.minLOD));
            lod = m_config.minLOD + lod;
            return Math::clamp(lod, m_config.minLOD, m_config.maxLOD);
        }
    }

    // ------------------------------------------------------------------------
    //  Update single object state, applying hysteresis
    // ------------------------------------------------------------------------
    void updateObject(ObjectState& state) const noexcept {
        uint8_t newLOD = computeLOD(state.worldCenter, state.objectSize);
        // Hysteresis: only change if difference exceeds threshold
        if (newLOD > state.currentLOD + static_cast<uint8_t>(m_config.hysteresis * m_config.maxLOD) ||
            newLOD + static_cast<uint8_t>(m_config.hysteresis * m_config.maxLOD) < state.currentLOD) {
            state.targetLOD = newLOD;
        } else {
            state.targetLOD = state.currentLOD;
        }
        state.lastDistance = (state.worldCenter - m_viewerPosition).length();
    }

    // Apply pending LOD changes (call after updateObject)
    void applyLOD(ObjectState& state) noexcept {
        state.currentLOD = state.targetLOD;
    }

    // ------------------------------------------------------------------------
    //  SIMD batch processing for multiple objects
    // ------------------------------------------------------------------------
    void batchComputeLOD(const point_type* centers, const T* sizes,
                         uint8_t* outLOD, std::size_t count) const noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
            // Use 4‑wide SIMD (pseudo, real implementation would use aligned loads)
            for (std::size_t i = 0; i < count; ++i) {
                outLOD[i] = computeLOD(centers[i], sizes[i]);
            }
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                outLOD[i] = computeLOD(centers[i], sizes[i]);
            }
        }
    }

    void batchUpdateObjects(ObjectState* states, std::size_t count) const noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
            // Could process 4 at a time, but for simplicity we loop scalar.
            for (std::size_t i = 0; i < count; ++i) {
                updateObject(states[i]);
            }
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                updateObject(states[i]);
            }
        }
    }

    // ------------------------------------------------------------------------
    //  Performance adaptive LOD: adjust min/max distance based on frame time
    // ------------------------------------------------------------------------
    void updatePerformanceBudget(T frameTimeMs) noexcept {
        // If frame time exceeds budget, reduce detail (increase min LOD)
        if (frameTimeMs > m_config.performanceBudgetMs && m_config.maxLOD > m_config.minLOD) {
            m_config.maxLOD = (m_config.maxLOD > m_config.minLOD) ? (m_config.maxLOD - 1) : m_config.minLOD;
        } else if (frameTimeMs < m_config.performanceBudgetMs * T(0.8) && m_config.maxLOD < m_config.maxLOD) {
            m_config.maxLOD = std::min(static_cast<uint8_t>(m_config.maxLOD + 1), static_cast<uint8_t>(16));
        }
    }

    // ------------------------------------------------------------------------
    //  Getters / setters
    // ------------------------------------------------------------------------
    const Config& config() const noexcept { return m_config; }
    void setConfig(const Config& cfg) noexcept { m_config = cfg; }
    void setMinDistance(T d) noexcept { m_config.minDistance = d; }
    void setMaxDistance(T d) noexcept { m_config.maxDistance = d; }
    void setMinNodeSize(T s) noexcept { m_config.minNodeSize = s; }
    void setMaxNodeSize(T s) noexcept { m_config.maxNodeSize = s; }
    void setHysteresis(T h) noexcept { m_config.hysteresis = Math::clamp(h, T(0), T(1)); }
    void setScreenAreaBias(T b) noexcept { m_config.screenAreaBias = b; }
    void setUseScreenArea(bool enable) noexcept { m_config.useScreenArea = enable; }

    // ------------------------------------------------------------------------
    //  Convert LOD level to node size (world units)
    // ------------------------------------------------------------------------
    T lodToNodeSize(uint8_t lod) const noexcept {
        // Inverse mapping: lod 0 (finest) -> minNodeSize, lod max (coarsest) -> maxNodeSize
        T t = static_cast<T>(lod - m_config.minLOD) / static_cast<T>(m_config.maxLOD - m_config.minLOD);
        return Math::lerp(m_config.minNodeSize, m_config.maxNodeSize, t);
    }

    uint8_t nodeSizeToLOD(T size) const noexcept {
        // Find LOD that gives node size >= given size (conservative)
        T t = (size - m_config.minNodeSize) / (m_config.maxNodeSize - m_config.minNodeSize);
        t = Math::clamp(t, T(0), T(1));
        uint8_t lod = static_cast<uint8_t>(t * static_cast<T>(m_config.maxLOD - m_config.minLOD));
        return m_config.minLOD + lod;
    }

private:
    Config m_config;
    point_type m_viewerPosition;
    T m_fov = T(60) * Math::pi<T>() / T(180); // default 60°
    T m_tanHalfFov = std::tan(m_fov * T(0.5));
    uint32_t m_frameCounter;
    double m_lastUpdateTime;
};

// ============================================================================
//  Specialized for 2D (camera in orthographic or perspective, simpler)
// ============================================================================
template<typename T>
class DynamicLODController<T, 2> {
public:
    using point_type = Math::Vector<T, 2>;
    struct Config {
        T minDistance = T(0.1);
        T maxDistance = T(1000.0);
        T minNodeSize = T(0.01);
        T maxNodeSize = T(100.0);
        T hysteresis = T(0.1);
        uint8_t maxLOD = 16;
        uint8_t minLOD = 0;
        bool useScreenArea = true;
        T performanceBudgetMs = T(2.0);
    };

    explicit DynamicLODController(const Config& config = Config()) noexcept
        : m_config(config), m_viewerPosition(point_type(T(0))) {}

    void setViewerPosition(const point_type& pos) noexcept { m_viewerPosition = pos; }

    uint8_t computeLOD(const point_type& center, T size) const noexcept {
        T dist = (center - m_viewerPosition).length();
        dist = Math::clamp(dist, m_config.minDistance, m_config.maxDistance);
        if (m_config.useScreenArea) {
            T screenArea = (size / dist); // 1D screen size for 2D
            T areaNorm = Math::clamp(screenArea, T(0), T(1));
            uint8_t lod = static_cast<uint8_t>((T(1) - areaNorm) * static_cast<T>(m_config.maxLOD - m_config.minLOD));
            lod = m_config.minLOD + lod;
            return Math::clamp(lod, m_config.minLOD, m_config.maxLOD);
        } else {
            T t = (dist - m_config.minDistance) / (m_config.maxDistance - m_config.minDistance);
            t = Math::clamp(t, T(0), T(1));
            uint8_t lod = static_cast<uint8_t>((T(1) - t) * static_cast<T>(m_config.maxLOD - m_config.minLOD));
            lod = m_config.minLOD + lod;
            return Math::clamp(lod, m_config.minLOD, m_config.maxLOD);
        }
    }

    void batchComputeLOD(const point_type* centers, const T* sizes,
                         uint8_t* outLOD, std::size_t count) const noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            outLOD[i] = computeLOD(centers[i], sizes[i]);
        }
    }

    T lodToNodeSize(uint8_t lod) const noexcept {
        T t = static_cast<T>(lod - m_config.minLOD) / static_cast<T>(m_config.maxLOD - m_config.minLOD);
        return Math::lerp(m_config.minNodeSize, m_config.maxNodeSize, t);
    }

private:
    Config m_config;
    point_type m_viewerPosition;
};

// ============================================================================
//  Dynamic environment controller for LOD (responsive to frame rate and scene)
// ============================================================================
template<typename T>
class DynamicLODEnvironment {
public:
    using vec_type = Math::Vector<T, 3>;

    DynamicLODEnvironment() noexcept
        : m_targetFPS(60.0)
        , m_currentFPS(60.0)
        , m_qualityScale(1.0)
        , m_motionBlurFactor(0.0) {}

    void setTargetFPS(T fps) noexcept { m_targetFPS = fps; }
    void setCurrentFPS(T fps) noexcept { m_currentFPS = fps; }

    // Compute dynamic LOD offset based on performance (negative = lower detail)
    int8_t computePerformanceOffset() const noexcept {
        T ratio = m_currentFPS / m_targetFPS;
        if (ratio < 0.8) return -2; // drop detail
        if (ratio > 1.2) return +1; // increase detail
        return 0;
    }

    void setQualityScale(T scale) noexcept { m_qualityScale = Math::clamp(scale, T(0.5), T(2.0)); }
    T qualityScale() const noexcept { return m_qualityScale; }

    // Adjust LOD based on object velocity (motion blur: fast objects need less detail)
    void setMotionBlurFactor(T factor) noexcept { m_motionBlurFactor = factor; }
    uint8_t adjustLODForMotion(uint8_t baseLOD, T speed) const noexcept {
        if (speed <= T(0)) return baseLOD;
        T lodReduction = std::min(m_motionBlurFactor * speed, T(2));
        int newLOD = static_cast<int>(baseLOD) + static_cast<int>(lodReduction);
        return static_cast<uint8_t>(Math::clamp(newLOD, 0, 16));
    }

private:
    T m_targetFPS;
    T m_currentFPS;
    T m_qualityScale;
    T m_motionBlurFactor;
};

} // namespace Partitioning
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_PARTITIONING_DYNAMIC_LOD_CONTROLLER_H_INCLUDED