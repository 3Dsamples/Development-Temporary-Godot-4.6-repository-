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

#ifndef ORTHOTREE_ADAPTERS_CONCEPTS_H_INCLUDED
#define ORTHOTREE_ADAPTERS_CONCEPTS_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/geometry_queries.h"
#include "../detail/common.h"
#include <type_traits>
#include <cstddef>

namespace OrthoTree {
namespace Adapters {

// ============================================================================
//  C++17‑compatible concepts (via SFINAE traits) for geometric adapters.
//  Provides compile‑time checks for point, vector, bounding box, ray, and
//  plane types used in orthotree. Allows custom geometric types to be
//  detected and adapted. Includes dynamic environment controls for enabling
//  strict or loose checking.
// ============================================================================

// ----------------------------------------------------------------------------
//  Primary traits (to be specialised by user or detected automatically)
// ----------------------------------------------------------------------------
template<typename T, typename = void>
struct is_point : std::false_type {};

template<typename T, typename = void>
struct is_vector : std::false_type {};

template<typename T, typename = void>
struct is_bounding_box : std::false_type {};

template<typename T, typename = void>
struct is_ray : std::false_type {};

template<typename T, typename = void>
struct is_plane : std::false_type {};

// ----------------------------------------------------------------------------
//  Detection for simple structs with x, y, (z) members
// ----------------------------------------------------------------------------
template<typename T>
struct has_x_y_z {
private:
    template<typename U>
    static auto test(int) -> decltype(
        std::declval<U>().x,
        std::declval<U>().y,
        std::declval<U>().z,
        std::true_type{}
    );
    template<typename> static std::false_type test(...);
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename T>
struct has_x_y {
private:
    template<typename U>
    static auto test(int) -> decltype(
        std::declval<U>().x,
        std::declval<U>().y,
        std::true_type{}
    );
    template<typename> static std::false_type test(...);
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename T>
struct is_point<T, std::enable_if_t<has_x_y_z<T>::value || has_x_y<T>::value>>
    : std::true_type {};

template<typename T>
struct is_vector<T, std::enable_if_t<has_x_y_z<T>::value || has_x_y<T>::value>>
    : std::true_type {};

// ----------------------------------------------------------------------------
//  Detection for std::array (N=2 or 3)
// ----------------------------------------------------------------------------
template<typename T>
struct is_array_point : std::false_type {};

template<typename T, std::size_t N>
struct is_array_point<std::array<T, N>> : std::integral_constant<bool, N == 2 || N == 3> {};

template<typename T>
struct is_point<T, std::enable_if_t<is_array_point<T>::value>> : std::true_type {};

template<typename T>
struct is_vector<T, std::enable_if_t<is_array_point<T>::value>> : std::true_type {};

// ----------------------------------------------------------------------------
//  Detection for bounding box (min / max members or methods)
// ----------------------------------------------------------------------------
template<typename T>
struct has_min_max {
private:
    template<typename U>
    static auto test(int) -> decltype(
        std::declval<U>().min,
        std::declval<U>().max,
        std::true_type{}
    );
    template<typename> static std::false_type test(...);
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename T>
struct is_bounding_box<T, std::enable_if_t<has_min_max<T>::value>> : std::true_type {};

// ----------------------------------------------------------------------------
//  Detection for ray (origin, direction members or methods)
// ----------------------------------------------------------------------------
template<typename T>
struct has_origin_direction {
private:
    template<typename U>
    static auto test(int) -> decltype(
        std::declval<U>().origin,
        std::declval<U>().direction,
        std::true_type{}
    );
    template<typename> static std::false_type test(...);
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename T>
struct is_ray<T, std::enable_if_t<has_origin_direction<T>::value>> : std::true_type {};

// ----------------------------------------------------------------------------
//  Detection for plane (normal, d members)
// ----------------------------------------------------------------------------
template<typename T>
struct has_normal_d {
private:
    template<typename U>
    static auto test(int) -> decltype(
        std::declval<U>().normal,
        std::declval<U>().d,
        std::true_type{}
    );
    template<typename> static std::false_type test(...);
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename T>
struct is_plane<T, std::enable_if_t<has_normal_d<T>::value>> : std::true_type {};

// ----------------------------------------------------------------------------
//  Coordinate type and dimension extractors
// ----------------------------------------------------------------------------
template<typename T>
struct point_traits {
    using scalar_type = float;
    static constexpr std::size_t dimension = 3;
};

template<typename T>
struct point_traits<T, std::enable_if_t<is_point<T>::value>> {
    using scalar_type = std::conditional_t<
        has_x_y_z<T>::value,
        decltype(T::x),
        float
    >;
    static constexpr std::size_t dimension = (has_x_y_z<T>::value) ? 3 : 2;
};

// ----------------------------------------------------------------------------
//  Dynamic environment controller for concepts (enable/disable strict checking)
// ----------------------------------------------------------------------------
class ConceptsEnvironment {
public:
    static ConceptsEnvironment& instance() {
        static ConceptsEnvironment env;
        return env;
    }

    void setStrictChecking(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_strictChecking = enable;
    }
    bool strictChecking() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_strictChecking;
    }

    void setEnableAutoDeduction(bool enable) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_autoDeduction = enable;
    }
    bool autoDeduction() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_autoDeduction;
    }

private:
    ConceptsEnvironment() : m_strictChecking(true), m_autoDeduction(true) {}
    mutable std::mutex m_mutex;
    bool m_strictChecking;
    bool m_autoDeduction;
};

// ----------------------------------------------------------------------------
//  Helper: check if a type satisfies a concept at runtime (for debugging)
// ----------------------------------------------------------------------------
template<typename T>
constexpr const char* point_concept_name() {
    return is_point<T>::value ? "point" : "not a point";
}

} // namespace Adapters
} // namespace OrthoTree

#endif // ORTHOTREE_ADAPTERS_CONCEPTS_H_INCLUDED