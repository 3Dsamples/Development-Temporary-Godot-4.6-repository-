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

#ifndef ORTHOTREE_ADAPTERS_GENERAL_H_INCLUDED
#define ORTHOTREE_ADAPTERS_GENERAL_H_INCLUDED

#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../detail/common.h"
#include <array>
#include <type_traits>
#include <cstddef>

namespace OrthoTree {
namespace Adapters {

// ============================================================================
//  General adapters for fundamental types and standard containers.
//  Enables OrthoTree to work with raw pointers, std::array, and trivial
//  user‑defined structs that provide x,y,z members (or x,y for 2D).
//  Supports SIMD‑friendly access and dynamic type detection.
// ============================================================================

// ----------------------------------------------------------------------------
//  Trait to detect if a type has members x, y, (z) – for point/vector adapters
// ----------------------------------------------------------------------------
template<typename T, typename = void>
struct has_xy_members : std::false_type {};

template<typename T>
struct has_xy_members<T, std::void_t<decltype(T::x), decltype(T::y)>>
    : std::true_type {};

template<typename T, typename = void>
struct has_xyz_members : std::false_type {};

template<typename T>
struct has_xyz_members<T, std::void_t<decltype(T::x), decltype(T::y), decltype(T::z)>>
    : std::true_type {};

// ----------------------------------------------------------------------------
//  Adapter for raw C arrays (T(&)[N] for N=2 or 3)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
struct RawArrayPointAdapter {
    static_assert(N == 2 || N == 3, "Only 2D or 3D arrays supported");

    using point_type = Math::Vector<T, N>;

    static point_type getPoint(const T (&arr)[N]) noexcept {
        point_type p;
        for (std::size_t i = 0; i < N; ++i) p[i] = arr[i];
        return p;
    }

    static void setPoint(T (&arr)[N], const point_type& p) noexcept {
        for (std::size_t i = 0; i < N; ++i) arr[i] = p[i];
    }
};

// ----------------------------------------------------------------------------
//  Adapter for std::array<T, N> (N=2,3)
// ----------------------------------------------------------------------------
template<typename T, std::size_t N>
struct StdArrayPointAdapter {
    static_assert(N == 2 || N == 3, "Only 2D or 3D arrays supported");
    using point_type = Math::Vector<T, N>;

    static point_type getPoint(const std::array<T, N>& arr) noexcept {
        point_type p;
        for (std::size_t i = 0; i < N; ++i) p[i] = arr[i];
        return p;
    }

    static void setPoint(std::array<T, N>& arr, const point_type& p) noexcept {
        for (std::size_t i = 0; i < N; ++i) arr[i] = p[i];
    }
};

// ----------------------------------------------------------------------------
//  Adapter for trivial structs with x,y,(z) members (e.g., struct Vec3 { float x,y,z; })
// ----------------------------------------------------------------------------
template<typename T>
struct TrivialStructPointAdapter {
    static constexpr bool is_2d = has_xy_members<T>::value && !has_xyz_members<T>::value;
    static constexpr bool is_3d = has_xyz_members<T>::value;

    using scalar_type = std::conditional_t<
        std::is_arithmetic_v<decltype(T::x)>,
        decltype(T::x),
        float
    >;

    using point_type = std::conditional_t<
        is_3d,
        Math::Vector<scalar_type, 3>,
        Math::Vector<scalar_type, 2>
    >;

    static point_type getPoint(const T& obj) noexcept {
        if constexpr (is_3d) {
            return point_type(obj.x, obj.y, obj.z);
        } else if constexpr (is_2d) {
            return point_type(obj.x, obj.y);
        } else {
            return point_type();
        }
    }

    static void setPoint(T& obj, const point_type& p) noexcept {
        if constexpr (is_3d) {
            obj.x = p[0]; obj.y = p[1]; obj.z = p[2];
        } else if constexpr (is_2d) {
            obj.x = p[0]; obj.y = p[1];
        }
    }
};

// ----------------------------------------------------------------------------
//  Bounding box adapter for std::pair of points (min, max)
// ----------------------------------------------------------------------------
template<typename PointType>
struct PairBoundingBoxAdapter {
    using scalar_type = typename PointType::value_type;
    static constexpr std::size_t dimension = PointType::dimension();
    using aabb_type = Math::AxisAlignedBox<scalar_type, dimension>;

    static aabb_type getBounds(const std::pair<PointType, PointType>& pair) noexcept {
        return aabb_type(pair.first, pair.second);
    }
};

// ----------------------------------------------------------------------------
//  Bounding box adapter for std::array of two points
// ----------------------------------------------------------------------------
template<typename PointType, std::size_t N>
struct ArrayBoundingBoxAdapter {
    static_assert(N == 2, "Array must have exactly two points");
    using scalar_type = typename PointType::value_type;
    static constexpr std::size_t dimension = PointType::dimension();
    using aabb_type = Math::AxisAlignedBox<scalar_type, dimension>;

    static aabb_type getBounds(const std::array<PointType, 2>& arr) noexcept {
        return aabb_type(arr[0], arr[1]);
    }
};

// ----------------------------------------------------------------------------
//  Dynamic environment controller: select adapter based on type at runtime
// ----------------------------------------------------------------------------
class AdapterSelector {
public:
    template<typename T>
    static constexpr const char* getTypeName() {
        if constexpr (std::is_array_v<T> && (std::extent_v<T> == 2 || std::extent_v<T> == 3))
            return "RawArray";
        else if constexpr (has_xyz_members<T>::value)
            return "TrivialStruct3D";
        else if constexpr (has_xy_members<T>::value)
            return "TrivialStruct2D";
        else if constexpr (std::is_same_v<T, std::array<float,2>> || std::is_same_v<T, std::array<double,2>>)
            return "StdArray2D";
        else if constexpr (std::is_same_v<T, std::array<float,3>> || std::is_same_v<T, std::array<double,3>>)
            return "StdArray3D";
        else
            return "Unknown";
    }
};

// ----------------------------------------------------------------------------
//  SIMD batch conversion: convert array of trivial structs to Math::Vector
// ----------------------------------------------------------------------------
template<typename TrivialPoint, typename T, std::size_t N>
void batchConvertToVector(const TrivialPoint* src, Math::Vector<T, N>* dst, std::size_t count) {
    constexpr bool is_3d = has_xyz_members<TrivialPoint>::value && N == 3;
    constexpr bool is_2d = has_xy_members<TrivialPoint>::value && N == 2;
    static_assert(is_2d || is_3d, "Unsupported point type for batch conversion");

    if constexpr (ORTHOTREE_SIMD_LEVEL >= 128 && N == 3) {
        // SIMD loop unrolled (pseudo)
        for (std::size_t i = 0; i < count; ++i) {
            dst[i][0] = static_cast<T>(src[i].x);
            dst[i][1] = static_cast<T>(src[i].y);
            dst[i][2] = static_cast<T>(src[i].z);
        }
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            dst[i][0] = static_cast<T>(src[i].x);
            dst[i][1] = static_cast<T>(src[i].y);
            if constexpr (N == 3) dst[i][2] = static_cast<T>(src[i].z);
        }
    }
}

} // namespace Adapters
} // namespace OrthoTree

#endif // ORTHOTREE_ADAPTERS_GENERAL_H_INCLUDED