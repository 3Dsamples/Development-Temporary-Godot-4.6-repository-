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

#ifndef ORTHOTREE_CORE_OT_ALIASES_H_INCLUDED
#define ORTHOTREE_CORE_OT_ALIASES_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/configuration.h"
#include "../core/ot_dynamic_hash_core.h"
#include "../core/ot_static_linear_core.h"
#include "../core/ot_managed.h"
#include "../core/ot_query.h"
#include "../core/entity_adapter.h"
#include "../detail/memory_resource.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <span>

namespace OrthoTree {

// ============================================================================
//  Dynamic octree aliases (contiguous container, e.g., std::vector)
// ============================================================================

/// 2D quadtree for points (dynamic, contiguous)
template<typename T = float>
using QuadtreePoint = ot_dynamic_hash_core<Dim2, T>;

/// 2D quadtree for bounding boxes (dynamic, contiguous)
template<typename T = float>
using QuadtreeBox = ot_dynamic_hash_core<Dim2, T>;

/// 3D octree for points (dynamic, contiguous)
template<typename T = float>
using OctreePoint = ot_dynamic_hash_core<Dim3, T>;

/// 3D octree for bounding boxes (dynamic, contiguous)
template<typename T = float>
using OctreeBox = ot_dynamic_hash_core<Dim3, T>;

// ----------------------------------------------------------------------------
//  Managed (owning) versions
// ----------------------------------------------------------------------------

/// Managed 2D quadtree for points (dynamic, contiguous)
template<typename T = float>
using QuadtreePointM = ot_managed<Dim2, T>;

/// Managed 2D quadtree for bounding boxes (dynamic, contiguous)
template<typename T = float>
using QuadtreeBoxM = ot_managed<Dim2, T>;

/// Managed 3D octree for points (dynamic, contiguous)
template<typename T = float>
using OctreePointM = ot_managed<Dim3, T>;

/// Managed 3D octree for bounding boxes (dynamic, contiguous)
template<typename T = float>
using OctreeBoxM = ot_managed<Dim3, T>;

// ============================================================================
//  Dynamic octree aliases (std::unordered_map based, custom keys)
// ============================================================================

/// 2D quadtree for points using unordered_map (key: entity ID)
template<typename T = float, typename Key = index_t>
using QuadtreePointMap = ot_dynamic_hash_core<Dim2, T, PMRAllocator<std::byte>>; // with map adapter

/// 2D quadtree for boxes using unordered_map
template<typename T = float, typename Key = index_t>
using QuadtreeBoxMap = ot_dynamic_hash_core<Dim2, T, PMRAllocator<std::byte>>;

/// 3D octree for points using unordered_map
template<typename T = float, typename Key = index_t>
using OctreePointMap = ot_dynamic_hash_core<Dim3, T, PMRAllocator<std::byte>>;

/// 3D octree for boxes using unordered_map
template<typename T = float, typename Key = index_t>
using OctreeBoxMap = ot_dynamic_hash_core<Dim3, T, PMRAllocator<std::byte>>;

// ----------------------------------------------------------------------------
//  Managed + map versions
// ----------------------------------------------------------------------------

template<typename T = float, typename Key = index_t>
using QuadtreePointMapM = ot_managed<Dim2, T>;

template<typename T = float, typename Key = index_t>
using QuadtreeBoxMapM = ot_managed<Dim2, T>;

template<typename T = float, typename Key = index_t>
using OctreePointMapM = ot_managed<Dim3, T>;

template<typename T = float, typename Key = index_t>
using OctreeBoxMapM = ot_managed<Dim3, T>;

// ============================================================================
//  Static linear BVH aliases (contiguous)
// ============================================================================

/// Static 2D BVH for points
template<typename T = float>
using StaticQuadtreePoint = ot_static_linear_core<Dim2, T>;

/// Static 2D BVH for bounding boxes
template<typename T = float>
using StaticQuadtreeBox = ot_static_linear_core<Dim2, T>;

/// Static 3D BVH for points
template<typename T = float>
using StaticOctreePoint = ot_static_linear_core<Dim3, T>;

/// Static 3D BVH for bounding boxes
template<typename T = float>
using StaticOctreeBox = ot_static_linear_core<Dim3, T>;

// ----------------------------------------------------------------------------
//  Managed static BVH (owning)
// ----------------------------------------------------------------------------

template<typename T = float>
using StaticQuadtreePointM = ot_managed<Dim2, T>;

template<typename T = float>
using StaticQuadtreeBoxM = ot_managed<Dim2, T>;

template<typename T = float>
using StaticOctreePointM = ot_managed<Dim3, T>;

template<typename T = float>
using StaticOctreeBoxM = ot_managed<Dim3, T>;

// ============================================================================
//  Static BVH with unordered_map (keyed)
// ============================================================================

template<typename T = float, typename Key = index_t>
using StaticQuadtreePointMap = ot_static_linear_core<Dim2, T>;

template<typename T = float, typename Key = index_t>
using StaticQuadtreeBoxMap = ot_static_linear_core<Dim2, T>;

template<typename T = float, typename Key = index_t>
using StaticOctreePointMap = ot_static_linear_core<Dim3, T>;

template<typename T = float, typename Key = index_t>
using StaticOctreeBoxMap = ot_static_linear_core<Dim3, T>;

// ----------------------------------------------------------------------------
//  Managed + map for static BVH
// ----------------------------------------------------------------------------

template<typename T = float, typename Key = index_t>
using StaticQuadtreePointMapM = ot_managed<Dim2, T>;

template<typename T = float, typename Key = index_t>
using StaticQuadtreeBoxMapM = ot_managed<Dim2, T>;

template<typename T = float, typename Key = index_t>
using StaticOctreePointMapM = ot_managed<Dim3, T>;

template<typename T = float, typename Key = index_t>
using StaticOctreeBoxMapM = ot_managed<Dim3, T>;

// ============================================================================
//  N‑dimensional generic aliases (using runtime dimension)
// ============================================================================

/// Dynamic N‑dimensional point octree (contiguous)
template<Dimension D, typename T = float>
using OrthoTreePointND = ot_dynamic_hash_core<D, T>;

/// Dynamic N‑dimensional box octree (contiguous)
template<Dimension D, typename T = float>
using OrthoTreeBoxND = ot_dynamic_hash_core<D, T>;

/// Managed N‑dimensional point octree
template<Dimension D, typename T = float>
using OrthoTreePointManagedND = ot_managed<D, T>;

/// Managed N‑dimensional box octree
template<Dimension D, typename T = float>
using OrthoTreeBoxManagedND = ot_managed<D, T>;

/// Static N‑dimensional BVH for points
template<Dimension D, typename T = float>
using StaticOrthoTreePointND = ot_static_linear_core<D, T>;

/// Static N‑dimensional BVH for boxes
template<Dimension D, typename T = float>
using StaticOrthoTreeBoxND = ot_static_linear_core<D, T>;

// ============================================================================
//  Dynamic environment controller for aliases (runtime selection)
// ============================================================================

class AliasEnvironment {
public:
    using CoreType = std::variant<
        std::monostate,
        ot_dynamic_hash_core<Dim2, float>*,
        ot_dynamic_hash_core<Dim3, float>*,
        ot_static_linear_core<Dim2, float>*,
        ot_static_linear_core<Dim3, float>*
    >;

    static AliasEnvironment& instance() {
        static AliasEnvironment env;
        return env;
    }

    // Set the active core for generic operations
    void setActiveCore(CoreType core) { m_activeCore = core; }

    // Execute a generic query (e.g., point query) on the active core
    template<typename OutputIt>
    size_t genericQueryPoint(const Math::Vector<float,3>& point, OutputIt out) {
        return std::visit([&](auto* core) -> size_t {
            using CorePtr = std::decay_t<decltype(core)>;
            if constexpr (std::is_pointer_v<CorePtr>) {
                if constexpr (std::is_same_v<std::remove_pointer_t<CorePtr>,
                              ot_dynamic_hash_core<Dim3, float>> ||
                              std::is_same_v<std::remove_pointer_t<CorePtr>,
                              ot_static_linear_core<Dim3, float>>) {
                    return core->queryPoint(point, out);
                } else if constexpr (std::is_same_v<std::remove_pointer_t<CorePtr>,
                              ot_dynamic_hash_core<Dim2, float>> ||
                              std::is_same_v<std::remove_pointer_t<CorePtr>,
                              ot_static_linear_core<Dim2, float>>) {
                    // Promote 2D point to 3D? Not supported; return 0.
                    return 0;
                }
            }
            return 0;
        }, m_activeCore);
    }

private:
    AliasEnvironment() = default;
    CoreType m_activeCore;
};

// ============================================================================
//  Helper to create a default octree with optimal configuration for the scale
// ============================================================================

enum class SimulationScale {
    Microscopic,
    Macroscopic,
    Planetary,
    Galactic
};

template<Dimension Dim, typename T = float>
auto createOctreeForScale(SimulationScale scale,
                          const Math::AxisAlignedBox<T, Dim>& bounds) {
    RuntimeConfiguration cfg;
    switch (scale) {
        case SimulationScale::Microscopic:
            cfg.maxDepth = 16;
            cfg.bucketSize = 4;
            cfg.growthPolicy = GrowthPolicy::ExpandRoot;
            cfg.adaptiveRefinement = true;
            break;
        case SimulationScale::Macroscopic:
            cfg.maxDepth = 12;
            cfg.bucketSize = 8;
            cfg.growthPolicy = GrowthPolicy::ExpandRoot;
            break;
        case SimulationScale::Planetary:
            cfg.maxDepth = 20;
            cfg.bucketSize = 16;
            cfg.growthPolicy = GrowthPolicy::RebuildOnExpand;
            break;
        case SimulationScale::Galactic:
            cfg.maxDepth = 24;
            cfg.bucketSize = 32;
            cfg.growthPolicy = GrowthPolicy::DynamicGrid;
            cfg.adaptiveRefinement = true;
            break;
    }
    // Apply configuration to octree (simplified: pass to constructor)
    // In a real implementation, we would use a factory.
    return ot_dynamic_hash_core<Dim, T>(bounds, cfg.maxDepth, cfg.bucketSize);
}

} // namespace OrthoTree

#endif // ORTHOTREE_CORE_OT_ALIASES_H_INCLUDED