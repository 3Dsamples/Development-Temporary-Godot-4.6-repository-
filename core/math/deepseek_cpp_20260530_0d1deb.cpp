// File 0026 : core/math/rect2.h
// 2D axis‑aligned rectangle (min, max or position + size) with area, perimeter, intersection, union, and point containment.

#pragma once

#include "vec2.h"
#include <algorithm>

namespace wp {

template <typename T>
struct rect2 {
    vec2<T> min;
    vec2<T> max;

    constexpr rect2() noexcept : min(T(0)), max(T(0)) {}
    constexpr rect2(const vec2<T>& mn, const vec2<T>& mx) noexcept : min(mn), max(mx) {}
    constexpr rect2(const vec2<T>& position, const vec2<T>& size) noexcept : min(position), max(position + size) {}
    template <typename U> constexpr explicit rect2(const rect2<U>& o) noexcept : min(o.min), max(o.max) {}

    constexpr vec2<T> position() const noexcept { return min; }
    constexpr vec2<T> size() const noexcept { return max - min; }
    constexpr T width() const noexcept { return max.x - min.x; }
    constexpr T height() const noexcept { return max.y - min.y; }
    constexpr vec2<T> center() const noexcept { return (min + max) * T(0.5); }
    constexpr T area() const noexcept { vec2<T> s = size(); return s.x * s.y; }
    constexpr T perimeter() const noexcept { vec2<T> s = size(); return T(2) * (s.x + s.y); }

    constexpr bool is_empty() const noexcept { return min.x > max.x || min.y > max.y; }
    constexpr bool has_point(const vec2<T>& p) const noexcept { return p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y; }

    constexpr bool intersects(const rect2& other) const noexcept {
        return min.x <= other.max.x && max.x >= other.min.x && min.y <= other.max.y && max.y >= other.min.y;
    }

    constexpr rect2 intersection(const rect2& other) const noexcept {
        return rect2(wp::max(min, other.min), wp::min(max, other.max));
    }
    constexpr rect2 merge(const rect2& other) const noexcept {
        return rect2(wp::min(min, other.min), wp::max(max, other.max));
    }

    constexpr rect2& expand(const vec2<T>& p) noexcept {
        min = wp::min(min, p);
        max = wp::max(max, p);
        return *this;
    }
    constexpr rect2 expanded(const vec2<T>& p) const noexcept { rect2 r = *this; r.expand(p); return r; }

    constexpr rect2& grow(T amount) noexcept { min -= vec2<T>(amount); max += vec2<T>(amount); return *this; }
    constexpr rect2 grown(T amount) const noexcept { rect2 r = *this; r.grow(amount); return r; }

    constexpr bool operator==(const rect2& o) const noexcept { return min == o.min && max == o.max; }
    constexpr bool operator!=(const rect2& o) const noexcept { return !(*this == o); }
};

template <typename T> constexpr rect2<T> rect2_from_center_size(const vec2<T>& center, const vec2<T>& size) noexcept {
    vec2<T> half = size * T(0.5);
    return rect2<T>(center - half, center + half);
}

using rect2f = rect2<float>;
using rect2d = rect2<double>;

} // namespace wp