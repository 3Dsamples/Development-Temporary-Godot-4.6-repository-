// SPDX-FileCopyrightText: Copyright (c) 2025 – C++17 Math/Physics Library
// SPDX-License-Identifier: MIT
#pragma once

#include "vec.hpp"
#include <vector>
#include <unordered_map>
#include <tuple>

namespace wp {

// ── Hash grid for fast neighbour searches (matching Warp's wp.HashGrid) ──
template <typename T>
class HashGrid {
public:
    using point_t = vec_t<3,T>;
    using index_t = int;

    HashGrid(T cell_size = T(1)) : m_cell_size(cell_size), m_inv_cell_size(T(1)/cell_size) {}

    // Build grid from point array (copy of positions for CPU reference)
    void build(const std::vector<point_t>& points) {
        m_cells.clear();
        m_points = points;
        m_num_points = static_cast<int>(points.size());
        for (int i = 0; i < m_num_points; ++i) {
            auto key = cell_key(m_points[i]);
            m_cells[key].push_back(i);
        }
    }

    // Radius query: returns indices of all points within `radius` of `query_point`
    std::vector<index_t> query(const point_t& query_point, T radius) const {
        std::vector<index_t> result;
        T r2 = radius * radius;
        vec_t<3,int> center = cell_coord(query_point);
        int sr = static_cast<int>(std::ceil(radius * m_inv_cell_size)) + 1;
        for (int dx = -sr; dx <= sr; ++dx)
            for (int dy = -sr; dy <= sr; ++dy)
                for (int dz = -sr; dz <= sr; ++dz) {
                    auto key = std::make_tuple(center[0]+dx, center[1]+dy, center[2]+dz);
                    auto it = m_cells.find(key);
                    if (it == m_cells.end()) continue;
                    for (int idx : it->second) {
                        if (distance_sq(query_point, m_points[idx]) <= r2)
                            result.push_back(idx);
                    }
                }
        return result;
    }

    // Nearest‑neighbour query (single)
    index_t nearest(const point_t& query_point, T max_radius = Constants<T>::infinity) const {
        index_t best = -1;
        T best_dist2 = max_radius * max_radius;
        vec_t<3,int> center = cell_coord(query_point);
        int sr = static_cast<int>(std::ceil(max_radius * m_inv_cell_size)) + 1;
        for (int dx = -sr; dx <= sr; ++dx)
            for (int dy = -sr; dy <= sr; ++dy)
                for (int dz = -sr; dz <= sr; ++dz) {
                    auto key = std::make_tuple(center[0]+dx, center[1]+dy, center[2]+dz);
                    auto it = m_cells.find(key);
                    if (it == m_cells.end()) continue;
                    for (int idx : it->second) {
                        T d2 = distance_sq(query_point, m_points[idx]);
                        if (d2 < best_dist2) { best_dist2 = d2; best = idx; }
                    }
                }
        return best;
    }

private:
    using cell_key_t = std::tuple<int,int,int>;
    T m_cell_size, m_inv_cell_size;
    std::vector<point_t> m_points;
    int m_num_points = 0;
    std::unordered_map<cell_key_t, std::vector<index_t>> m_cells;

    vec_t<3,int> cell_coord(const point_t& p) const noexcept {
        return { static_cast<int>(std::floor(p[0] * m_inv_cell_size)),
                 static_cast<int>(std::floor(p[1] * m_inv_cell_size)),
                 static_cast<int>(std::floor(p[2] * m_inv_cell_size)) };
    }

    cell_key_t cell_key(const point_t& p) const noexcept {
        auto c = cell_coord(p);
        return std::make_tuple(c[0], c[1], c[2]);
    }
};

} // namespace wp