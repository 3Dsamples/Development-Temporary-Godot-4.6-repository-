// File 0024 : core/math/grid.h
// Uniform 2D/3D spatial grids for point-based neighbor searches, radius queries, and nearest neighbor lookups.

#pragma once

#include "constants.h"
#include "vec2.h"
#include "vec3.h"
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <tuple>

namespace wp {

// ---------- 3D Grid -------------------------------------------------
template <typename T>
class Grid3D {
public:
    using Point = vec3<T>;

    Grid3D(T cell_size = T(1)) : m_cs(cell_size), m_ics(T(1)/cell_size) {}

    void build(const std::vector<Point>& points) {
        m_pts = points;
        m_cells.clear();
        for (int32 i = 0; i < static_cast<int32>(m_pts.size()); ++i) {
            uint64 key = cell_key(m_pts[i]);
            m_cells[key].push_back(i);
        }
    }

    std::vector<int32> query(const Point& query_pt, T radius) const {
        std::vector<int32> result;
        T r2 = radius * radius;
        vec3<int32> center = cell_coord(query_pt);
        int32 sr = static_cast<int32>(std::ceil(radius * m_ics)) + 1;
        for (int32 dx = -sr; dx <= sr; ++dx)
            for (int32 dy = -sr; dy <= sr; ++dy)
                for (int32 dz = -sr; dz <= sr; ++dz) {
                    uint64 key = make_key(center.x + dx, center.y + dy, center.z + dz);
                    auto it = m_cells.find(key);
                    if (it == m_cells.end()) continue;
                    for (int32 idx : it->second)
                        if (length_sq(query_pt - m_pts[idx]) <= r2)
                            result.push_back(idx);
                }
        return result;
    }

    int32 nearest(const Point& query_pt, T max_radius = MathConst<T>::infinity) const {
        int32 best = -1;
        T best_d2 = max_radius * max_radius;
        vec3<int32> center = cell_coord(query_pt);
        int32 sr = static_cast<int32>(std::ceil(max_radius * m_ics)) + 1;
        for (int32 dx = -sr; dx <= sr; ++dx)
            for (int32 dy = -sr; dy <= sr; ++dy)
                for (int32 dz = -sr; dz <= sr; ++dz) {
                    uint64 key = make_key(center.x + dx, center.y + dy, center.z + dz);
                    auto it = m_cells.find(key);
                    if (it == m_cells.end()) continue;
                    for (int32 idx : it->second) {
                        T d2 = length_sq(query_pt - m_pts[idx]);
                        if (d2 < best_d2) { best_d2 = d2; best = idx; }
                    }
                }
        return best;
    }

    void clear() { m_pts.clear(); m_cells.clear(); }

private:
    T m_cs, m_ics;
    std::vector<Point> m_pts;
    std::unordered_map<uint64, std::vector<int32>> m_cells;

    static constexpr int32 bits_per_axis = 21;
    static constexpr int32 mask = (1 << bits_per_axis) - 1;

    static uint64 make_key(int32 x, int32 y, int32 z) noexcept {
        return (uint64(x & mask) << (bits_per_axis*2)) |
               (uint64(y & mask) <<  bits_per_axis)     |
               (uint64(z & mask));
    }

    vec3<int32> cell_coord(const Point& p) const noexcept {
        return {static_cast<int32>(std::floor(p.x * m_ics)),
                static_cast<int32>(std::floor(p.y * m_ics)),
                static_cast<int32>(std::floor(p.z * m_ics))};
    }

    uint64 cell_key(const Point& p) const noexcept {
        auto c = cell_coord(p);
        return make_key(c.x, c.y, c.z);
    }
};

// ---------- 2D Grid -------------------------------------------------
template <typename T>
class Grid2D {
public:
    using Point = vec2<T>;

    Grid2D(T cell_size = T(1)) : m_cs(cell_size), m_ics(T(1)/cell_size) {}

    void build(const std::vector<Point>& points) {
        m_pts = points;
        m_cells.clear();
        for (int32 i = 0; i < static_cast<int32>(m_pts.size()); ++i) {
            uint64 key = cell_key(m_pts[i]);
            m_cells[key].push_back(i);
        }
    }

    std::vector<int32> query(const Point& query_pt, T radius) const {
        std::vector<int32> result;
        T r2 = radius * radius;
        vec2<int32> center = cell_coord(query_pt);
        int32 sr = static_cast<int32>(std::ceil(radius * m_ics)) + 1;
        for (int32 dx = -sr; dx <= sr; ++dx)
            for (int32 dy = -sr; dy <= sr; ++dy) {
                uint64 key = make_key(center.x + dx, center.y + dy);
                auto it = m_cells.find(key);
                if (it == m_cells.end()) continue;
                for (int32 idx : it->second)
                    if (length_sq(query_pt - m_pts[idx]) <= r2)
                        result.push_back(idx);
            }
        return result;
    }

    int32 nearest(const Point& query_pt, T max_radius = MathConst<T>::infinity) const {
        int32 best = -1;
        T best_d2 = max_radius * max_radius;
        vec2<int32> center = cell_coord(query_pt);
        int32 sr = static_cast<int32>(std::ceil(max_radius * m_ics)) + 1;
        for (int32 dx = -sr; dx <= sr; ++dx)
            for (int32 dy = -sr; dy <= sr; ++dy) {
                uint64 key = make_key(center.x + dx, center.y + dy);
                auto it = m_cells.find(key);
                if (it == m_cells.end()) continue;
                for (int32 idx : it->second) {
                    T d2 = length_sq(query_pt - m_pts[idx]);
                    if (d2 < best_d2) { best_d2 = d2; best = idx; }
                }
            }
        return best;
    }

    void clear() { m_pts.clear(); m_cells.clear(); }

private:
    T m_cs, m_ics;
    std::vector<Point> m_pts;
    std::unordered_map<uint64, std::vector<int32>> m_cells;

    static constexpr int32 bits_per_axis = 31; // 31 bits fits in 64-bit for 2 axes

    static uint64 make_key(int32 x, int32 y) noexcept {
        return (uint64(uint32(x)) << 32) | uint64(uint32(y));
    }

    vec2<int32> cell_coord(const Point& p) const noexcept {
        return {static_cast<int32>(std::floor(p.x * m_ics)),
                static_cast<int32>(std::floor(p.y * m_ics))};
    }

    uint64 cell_key(const Point& p) const noexcept {
        auto c = cell_coord(p);
        return make_key(c.x, c.y);
    }
};

} // namespace wp