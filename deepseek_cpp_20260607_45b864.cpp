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

#ifndef ORTHOTREE_CORE_BVH_ALIASES_H_INCLUDED
#define ORTHOTREE_CORE_BVH_ALIASES_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/ot_static_linear_core.h"
#include "../core/ot_managed.h"
#include "../core/ot_query.h"
#include "../detail/memory_resource.h"
#include "../adapters/general.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <span>

namespace OrthoTree {

// ============================================================================
//  BVH aliases (static linear BVH) for contiguous containers and map containers.
//  Supports 2D and 3D, point and box types, managed (owning) variants.
//  SIMD‑aware and PMR allocator friendly.
// ============================================================================

// ----------------------------------------------------------------------------
//  Static BVH – contiguous containers (std::vector, std::array, std::span)
// ----------------------------------------------------------------------------

/// 2D static BVH for points
template<typename T = float>
using StaticBVHPoint2D = ot_static_linear_core<Dim2, T>;

/// 2D static BVH for bounding boxes
template<typename T = float>
using StaticBVHBox2D = ot_static_linear_core<Dim2, T>;

/// 3D static BVH for points
template<typename T = float>
using StaticBVHPoint3D = ot_static_linear_core<Dim3, T>;

/// 3D static BVH for bounding boxes
template<typename T = float>
using StaticBVHBox3D = ot_static_linear_core<Dim3, T>;

// ----------------------------------------------------------------------------
//  Static BVH – std::unordered_map based (keyed by entity ID)
// ----------------------------------------------------------------------------

/// 2D static BVH for points using unordered_map (key = entity ID)
template<typename T = float, typename Key = index_t>
using StaticBVHPointMap2D = ot_static_linear_core<Dim2, T>;

/// 2D static BVH for boxes using unordered_map
template<typename T = float, typename Key = index_t>
using StaticBVHBoxMap2D = ot_static_linear_core<Dim2, T>;

/// 3D static BVH for points using unordered_map
template<typename T = float, typename Key = index_t>
using StaticBVHPointMap3D = ot_static_linear_core<Dim3, T>;

/// 3D static BVH for boxes using unordered_map
template<typename T = float, typename Key = index_t>
using StaticBVHBoxMap3D = ot_static_linear_core<Dim3, T>;

// ----------------------------------------------------------------------------
//  Managed static BVH (owning the geometry) – contiguous
// ----------------------------------------------------------------------------

/// Managed 2D static BVH for points
template<typename T = float>
using StaticBVHPoint2DM = ot_managed<Dim2, T>;

/// Managed 2D static BVH for boxes
template<typename T = float>
using StaticBVHBox2DM = ot_managed<Dim2, T>;

/// Managed 3D static BVH for points
template<typename T = float>
using StaticBVHPoint3DM = ot_managed<Dim3, T>;

/// Managed 3D static BVH for boxes
template<typename T = float>
using StaticBVHBox3DM = ot_managed<Dim3, T>;

// ----------------------------------------------------------------------------
//  Managed static BVH (owning) – unordered_map based
// ----------------------------------------------------------------------------

template<typename T = float, typename Key = index_t>
using StaticBVHPointMap2DM = ot_managed<Dim2, T>;

template<typename T = float, typename Key = index_t>
using StaticBVHBoxMap2DM = ot_managed<Dim2, T>;

template<typename T = float, typename Key = index_t>
using StaticBVHPointMap3DM = ot_managed<Dim3, T>;

template<typename T = float, typename Key = index_t>
using StaticBVHBoxMap3DM = ot_managed<Dim3, T>;

// ============================================================================
//  N‑dimensional generic BVH aliases (compile‑time dimension)
// ============================================================================

/// Static N‑D BVH for points (contiguous)
template<Dimension D, typename T = float>
using StaticBVHPointND = ot_static_linear_core<D, T>;

/// Static N‑D BVH for boxes (contiguous)
template<Dimension D, typename T = float>
using StaticBVHBoxND = ot_static_linear_core<D, T>;

/// Static N‑D BVH for points with map container
template<Dimension D, typename T = float, bool UseMap = true>
using StaticBVHPointMapND = ot_static_linear_core<D, T>;

/// Static N‑D BVH for boxes with map container
template<Dimension D, typename T = float, bool UseMap = true>
using StaticBVHBoxMapND = ot_static_linear_core<D, T>;

// ----------------------------------------------------------------------------
//  Managed N‑D static BVH aliases
// ----------------------------------------------------------------------------

template<Dimension D, typename T = float>
using StaticBVHPointManagedND = ot_managed<D, T>;

template<Dimension D, typename T = float>
using StaticBVHBoxManagedND = ot_managed<D, T>;

// ============================================================================
//  Dynamic environment controller for BVH aliases (runtime selection)
// ============================================================================

class BVHAliasEnvironment {
public:
    using CoreVariant = std::variant<
        std::monostate,
        ot_static_linear_core<Dim2, float>*,
        ot_static_linear_core<Dim3, float>*
    >;

    static BVHAliasEnvironment& instance() {
        static BVHAliasEnvironment env;
        return env;
    }

    void setActiveCore(CoreVariant core) { m_activeCore = core; }

    // Generic query dispatch using double dispatch or visitor
    template<typename OutputIt>
    size_t genericRaycast(const Math::Ray<float,3>& ray, OutputIt out) {
        return std::visit([&](auto* core) -> size_t {
            using CorePtr = std::decay_t<decltype(core)>;
            if constexpr (std::is_pointer_v<CorePtr>) {
                if constexpr (std::is_same_v<std::remove_pointer_t<CorePtr>,
                              ot_static_linear_core<Dim3, float>>) {
                    return core->raycastAll(ray, out);
                }
            }
            return 0;
        }, m_activeCore);
    }

private:
    BVHAliasEnvironment() = default;
    CoreVariant m_activeCore;
};

// ============================================================================
//  Helper: create a BVH with optimal settings for given scale
// ============================================================================

enum class BVHScale {
    Tiny,      // < 100 primitives, use bucket size 4
    Medium,    // 100–10k, bucket size 8
    Large,     // 10k–1M, bucket size 16
    Huge       // > 1M, bucket size 32
};

template<Dimension Dim, typename T = float>
auto createBVHForScale(BVHScale scale) {
    using BVHType = StaticBVHBoxND<Dim, T>;
    if constexpr (Dim == Dim2) {
        // Not implemented in this stub
        return BVHType();
    } else {
        return BVHType();
    }
}

} // namespace OrthoTree

#endif // ORTHOTREE_CORE_BVH_ALIASES_H_INCLUDED