// system name : onetbb-warp
// File 0040 : core/math/sampling.h
// Description : Advanced Monte Carlo samplers: Halton, Sobol, Hammersley, stratified, Poisson disk, importance sampling.

#ifndef __TBB_WARP_CORE_MATH_SAMPLING_H
#define __TBB_WARP_CORE_MATH_SAMPLING_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/random.h"
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <functional>
#include <cstdint>
#include <numeric>
#include <limits>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Halton sequence (radical inverse)
// ============================================================

inline float halton_sequence(std::uint32_t index, std::uint32_t base) noexcept {
    float result = 0.0f;
    float inv_base = 1.0f / static_cast<float>(base);
    float factor = inv_base;
    while (index > 0) {
        result += static_cast<float>(index % base) * factor;
        index /= base;
        factor *= inv_base;
    }
    return result;
}

template<std::size_t Dim>
std::array<float, Dim> halton_point(std::uint32_t index, const std::array<std::uint32_t, Dim>& bases) noexcept {
    std::array<float, Dim> point;
    for (std::size_t d = 0; d < Dim; ++d) {
        point[d] = halton_sequence(index, bases[d]);
    }
    return point;
}

inline std::vector<std::array<float, 2>> halton_2d_sequence(std::uint32_t count,
                                                              std::uint32_t base_x = 2,
                                                              std::uint32_t base_y = 3) noexcept {
    std::vector<std::array<float, 2>> points(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        points[i][0] = halton_sequence(i, base_x);
        points[i][1] = halton_sequence(i, base_y);
    }
    return points;
}

inline std::vector<std::array<float, 3>> halton_3d_sequence(std::uint32_t count,
                                                              std::uint32_t base_x = 2,
                                                              std::uint32_t base_y = 3,
                                                              std::uint32_t base_z = 5) noexcept {
    std::vector<std::array<float, 3>> points(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        points[i][0] = halton_sequence(i, base_x);
        points[i][1] = halton_sequence(i, base_y);
        points[i][2] = halton_sequence(i, base_z);
    }
    return points;
}

// ============================================================
// Sobol sequence (Gray code, direction numbers)
// ============================================================

class sobol_sequence {
public:
    explicit sobol_sequence(std::uint32_t dimensions) : m_dim(dimensions), m_index(0) {
        // Precompute direction numbers for up to 32 dimensions (max 21201)
        m_max_dim = std::min(dimensions, 32u);
        m_direction_numbers.resize(m_max_dim, std::vector<std::uint32_t>(32, 0));
        generate_direction_numbers();
    }

    std::vector<float> next() {
        std::vector<float> point(m_dim, 0.0f);
        if (m_index == 0) {
            m_index++;
            return point;
        }
        // Find rightmost zero bit
        std::uint32_t c = 0;
        std::uint32_t idx = m_index;
        while (idx & 1) { idx >>= 1; c++; }
        // XOR direction numbers for each dimension
        for (std::uint32_t d = 0; d < m_max_dim; ++d) {
            m_x[d] ^= m_direction_numbers[d][c];
            point[d] = static_cast<float>(m_x[d]) / static_cast<float>(0x100000000ULL);
        }
        // Remaining dimensions (if any) use Halton
        for (std::uint32_t d = m_max_dim; d < m_dim; ++d) {
            point[d] = halton_sequence(m_index, 2 + d - m_max_dim);
        }
        m_index++;
        return point;
    }

    void skip(std::uint32_t n) {
        for (std::uint32_t i = 0; i < n; ++i) next();
    }

    std::uint32_t index() const noexcept { return m_index; }

private:
    std::uint32_t m_dim;
    std::uint32_t m_max_dim;
    std::uint32_t m_index;
    std::vector<std::uint32_t> m_x;  // current state per dimension
    std::vector<std::vector<std::uint32_t>> m_direction_numbers;

    void generate_direction_numbers() {
        m_x.assign(m_max_dim, 0);
        // Polynomials for first 32 dimensions (irreducible primitive polynomials in GF(2))
        static const std::uint32_t polynomials[] = {
            0, 3, 7, 11, 13, 19, 25, 37, 59, 47, 61, 55, 41, 67, 97, 91,
            109, 103, 115, 131, 193, 137, 145, 143, 241, 157, 185, 167, 229, 171, 213, 191
        };
        for (std::uint32_t d = 0; d < m_max_dim; ++d) {
            std::uint32_t poly = polynomials[d];
            // Compute m_i values using recurrence
            std::uint32_t m[32];
            // Initialize first bits from d
            for (int i = 0; i < 32; ++i) m[i] = 1 << (31 - i);
            // For i >= degree, m_i = m_{i-d} xor (m_{i-d} >> d)
            int degree = 0;
            std::uint32_t temp = poly;
            while (temp) { degree++; temp >>= 1; }
            degree--;
            for (int i = degree; i < 32; ++i) {
                m[i] = m[i - degree];
                for (int j = 1; j < degree; ++j) {
                    if (poly & (1u << (degree - j))) {
                        m[i] ^= (m[i - j] >> (degree - j));
                    }
                }
            }
            // Direction numbers: v_{d,i} = m_i / 2^i (represented as integer: m_i * 2^(31-i) shifted)
            for (int i = 0; i < 32; ++i) {
                m_direction_numbers[d][i] = m[i] << i;
            }
        }
    }
};

// ============================================================
// Hammersley point set (fixed N, dimensions)
// ============================================================

inline std::vector<std::array<float, 2>> hammersley_2d(std::uint32_t N) {
    std::vector<std::array<float, 2>> points(N);
    for (std::uint32_t i = 0; i < N; ++i) {
        points[i][0] = static_cast<float>(i) / static_cast<float>(N);
        points[i][1] = halton_sequence(i, 2);
    }
    return points;
}

inline std::vector<std::array<float, 3>> hammersley_3d(std::uint32_t N) {
    std::vector<std::array<float, 3>> points(N);
    for (std::uint32_t i = 0; i < N; ++i) {
        points[i][0] = static_cast<float>(i) / static_cast<float>(N);
        points[i][1] = halton_sequence(i, 2);
        points[i][2] = halton_sequence(i, 3);
    }
    return points;
}

// ============================================================
// Stratified sampling (jittered grid)
// ============================================================

inline std::vector<std::array<float, 2>> stratified_2d(std::uint32_t nx, std::uint32_t ny, xorshift64& rng) {
    std::vector<std::array<float, 2>> points(nx * ny);
    float dx = 1.0f / nx;
    float dy = 1.0f / ny;
    for (std::uint32_t iy = 0; iy < ny; ++iy) {
        for (std::uint32_t ix = 0; ix < nx; ++ix) {
            float u = (ix + uniform_float(rng)) * dx;
            float v = (iy + uniform_float(rng)) * dy;
            points[iy * nx + ix] = {u, v};
        }
    }
    return points;
}

inline std::vector<std::array<float, 3>> stratified_3d(std::uint32_t nx, std::uint32_t ny, std::uint32_t nz, xorshift64& rng) {
    std::vector<std::array<float, 3>> points(nx * ny * nz);
    float dx = 1.0f / nx, dy = 1.0f / ny, dz = 1.0f / nz;
    for (std::uint32_t iz = 0; iz < nz; ++iz) {
        for (std::uint32_t iy = 0; iy < ny; ++iy) {
            for (std::uint32_t ix = 0; ix < nx; ++ix) {
                float u = (ix + uniform_float(rng)) * dx;
                float v = (iy + uniform_float(rng)) * dy;
                float w = (iz + uniform_float(rng)) * dz;
                points[(iz * ny + iy) * nx + ix] = {u, v, w};
            }
        }
    }
    return points;
}

// ============================================================
// Poisson disk sampling (2D) – Bridson's algorithm
// ============================================================

inline std::vector<std::array<float, 2>> poisson_disk_2d(float radius, std::uint32_t max_attempts = 30, xorshift64& rng = xorshift64(12345)) {
    float cell_size = radius / std::sqrt(2.0f);
    int grid_w = static_cast<int>(std::ceil(1.0f / cell_size)) + 1;
    int grid_h = static_cast<int>(std::ceil(1.0f / cell_size)) + 1;
    std::vector<int> grid(grid_w * grid_h, -1);
    std::vector<std::array<float, 2>> points;
    std::vector<int> active_list;

    auto grid_insert = [&](const std::array<float, 2>& p, int idx) {
        int gx = static_cast<int>(p[0] / cell_size);
        int gy = static_cast<int>(p[1] / cell_size);
        grid[gy * grid_w + gx] = idx;
    };

    auto is_valid = [&](const std::array<float, 2>& p) {
        if (p[0] < 0.0f || p[0] >= 1.0f || p[1] < 0.0f || p[1] >= 1.0f) return false;
        int gx = static_cast<int>(p[0] / cell_size);
        int gy = static_cast<int>(p[1] / cell_size);
        int gx_min = std::max(0, gx - 2), gx_max = std::min(grid_w - 1, gx + 2);
        int gy_min = std::max(0, gy - 2), gy_max = std::min(grid_h - 1, gy + 2);
        float r2 = radius * radius;
        for (int iy = gy_min; iy <= gy_max; ++iy) {
            for (int ix = gx_min; ix <= gx_max; ++ix) {
                int idx = grid[iy * grid_w + ix];
                if (idx >= 0) {
                    float dx = p[0] - points[idx][0];
                    float dy = p[1] - points[idx][1];
                    if (dx*dx + dy*dy < r2) return false;
                }
            }
        }
        return true;
    };

    std::array<float, 2> first = {uniform_float(rng), uniform_float(rng)};
    points.push_back(first);
    grid_insert(first, 0);
    active_list.push_back(0);

    while (!active_list.empty()) {
        int rand_idx = static_cast<int>(uniform_uint(rng, 0, static_cast<uint32_t>(active_list.size() - 1)));
        int current_idx = active_list[rand_idx];
        const auto& current = points[current_idx];
        bool found = false;
        for (std::uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
            float angle = uniform_range(rng, 0.0f, TAU_F);
            float dist = uniform_range(rng, radius, 2.0f * radius);
            std::array<float, 2> candidate = {
                current[0] + dist * std::cos(angle),
                current[1] + dist * std::sin(angle)
            };
            if (is_valid(candidate)) {
                int new_idx = static_cast<int>(points.size());
                points.push_back(candidate);
                grid_insert(candidate, new_idx);
                active_list.push_back(new_idx);
                found = true;
                break;
            }
        }
        if (!found) {
            active_list[rand_idx] = active_list.back();
            active_list.pop_back();
        }
    }
    return points;
}

// ============================================================
// Poisson disk sampling (3D) – Bridson's algorithm
// ============================================================

inline std::vector<std::array<float, 3>> poisson_disk_3d(float radius, std::uint32_t max_attempts = 30, xorshift64& rng = xorshift64(54321)) {
    float cell_size = radius / std::sqrt(3.0f);
    int grid_w = static_cast<int>(std::ceil(1.0f / cell_size)) + 1;
    int grid_h = static_cast<int>(std::ceil(1.0f / cell_size)) + 1;
    int grid_d = static_cast<int>(std::ceil(1.0f / cell_size)) + 1;
    std::vector<int> grid(grid_w * grid_h * grid_d, -1);
    std::vector<std::array<float, 3>> points;
    std::vector<int> active_list;

    auto grid_insert = [&](const std::array<float, 3>& p, int idx) {
        int gx = static_cast<int>(p[0] / cell_size);
        int gy = static_cast<int>(p[1] / cell_size);
        int gz = static_cast<int>(p[2] / cell_size);
        grid[(gz * grid_h + gy) * grid_w + gx] = idx;
    };

    auto is_valid = [&](const std::array<float, 3>& p) {
        if (p[0] < 0.0f || p[0] >= 1.0f || p[1] < 0.0f || p[1] >= 1.0f || p[2] < 0.0f || p[2] >= 1.0f) return false;
        int gx = static_cast<int>(p[0] / cell_size);
        int gy = static_cast<int>(p[1] / cell_size);
        int gz = static_cast<int>(p[2] / cell_size);
        float r2 = radius * radius;
        for (int iz = std::max(0, gz-2); iz <= std::min(grid_d-1, gz+2); ++iz) {
            for (int iy = std::max(0, gy-2); iy <= std::min(grid_h-1, gy+2); ++iy) {
                for (int ix = std::max(0, gx-2); ix <= std::min(grid_w-1, gx+2); ++ix) {
                    int idx = grid[(iz * grid_h + iy) * grid_w + ix];
                    if (idx >= 0) {
                        float dx = p[0] - points[idx][0];
                        float dy = p[1] - points[idx][1];
                        float dz = p[2] - points[idx][2];
                        if (dx*dx + dy*dy + dz*dz < r2) return false;
                    }
                }
            }
        }
        return true;
    };

    std::array<float, 3> first = {uniform_float(rng), uniform_float(rng), uniform_float(rng)};
    points.push_back(first);
    grid_insert(first, 0);
    active_list.push_back(0);

    while (!active_list.empty()) {
        int rand_idx = static_cast<int>(uniform_uint(rng, 0, static_cast<uint32_t>(active_list.size() - 1)));
        int current_idx = active_list[rand_idx];
        const auto& current = points[current_idx];
        bool found = false;
        for (std::uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
            float theta = uniform_range(rng, 0.0f, TAU_F);
            float phi = std::acos(uniform_range(rng, -1.0f, 1.0f));
            float dist = uniform_range(rng, radius, 2.0f * radius);
            std::array<float, 3> candidate = {
                current[0] + dist * std::sin(phi) * std::cos(theta),
                current[1] + dist * std::sin(phi) * std::sin(theta),
                current[2] + dist * std::cos(phi)
            };
            if (is_valid(candidate)) {
                int new_idx = static_cast<int>(points.size());
                points.push_back(candidate);
                grid_insert(candidate, new_idx);
                active_list.push_back(new_idx);
                found = true;
                break;
            }
        }
        if (!found) {
            active_list[rand_idx] = active_list.back();
            active_list.pop_back();
        }
    }
    return points;
}

// ============================================================
// Blue noise (Poisson disk) with variable density
// ============================================================

template<typename DensityFunc>
std::vector<std::array<float, 2>> variable_poisson_disk_2d(DensityFunc density_func,
                                                             float max_radius, float min_radius,
                                                             std::uint32_t max_attempts = 30,
                                                             xorshift64& rng = xorshift64(42)) {
    float cell_size = min_radius / std::sqrt(2.0f);
    int grid_w = static_cast<int>(std::ceil(1.0f / cell_size)) + 1;
    int grid_h = static_cast<int>(std::ceil(1.0f / cell_size)) + 1;
    std::vector<int> grid(grid_w * grid_h, -1);
    std::vector<std::array<float, 2>> points;
    std::vector<int> active_list;

    auto grid_insert = [&](const std::array<float, 2>& p, int idx) {
        int gx = static_cast<int>(p[0] / cell_size);
        int gy = static_cast<int>(p[1] / cell_size);
        grid[gy * grid_w + gx] = idx;
    };

    auto is_valid = [&](const std::array<float, 2>& p, float r) {
        if (p[0] < 0.0f || p[0] >= 1.0f || p[1] < 0.0f || p[1] >= 1.0f) return false;
        int gx = static_cast<int>(p[0] / cell_size);
        int gy = static_cast<int>(p[1] / cell_size);
        float r2 = r * r;
        for (int iy = std::max(0, gy-2); iy <= std::min(grid_h-1, gy+2); ++iy) {
            for (int ix = std::max(0, gx-2); ix <= std::min(grid_w-1, gx+2); ++ix) {
                int idx = grid[iy * grid_w + ix];
                if (idx >= 0) {
                    float dx = p[0] - points[idx][0];
                    float dy = p[1] - points[idx][1];
                    if (dx*dx + dy*dy < r2) return false;
                }
            }
        }
        return true;
    };

    std::array<float, 2> first = {uniform_float(rng), uniform_float(rng)};
    points.push_back(first);
    grid_insert(first, 0);
    active_list.push_back(0);

    while (!active_list.empty()) {
        int rand_idx = static_cast<int>(uniform_uint(rng, 0, static_cast<uint32_t>(active_list.size() - 1)));
        int current_idx = active_list[rand_idx];
        const auto& current = points[current_idx];
        float local_density = density_func(current[0], current[1]);
        float local_radius = min_radius + (max_radius - min_radius) * local_density;
        bool found = false;
        for (std::uint32_t attempt = 0; attempt < max_attempts; ++attempt) {
            float angle = uniform_range(rng, 0.0f, TAU_F);
            float dist = uniform_range(rng, local_radius, 2.0f * local_radius);
            std::array<float, 2> candidate = {
                current[0] + dist * std::cos(angle),
                current[1] + dist * std::sin(angle)
            };
            if (is_valid(candidate, local_radius)) {
                int new_idx = static_cast<int>(points.size());
                points.push_back(candidate);
                grid_insert(candidate, new_idx);
                active_list.push_back(new_idx);
                found = true;
                break;
            }
        }
        if (!found) {
            active_list[rand_idx] = active_list.back();
            active_list.pop_back();
        }
    }
    return points;
}

// ============================================================
// Importance sampling framework (generic)
// ============================================================

template<typename T, std::size_t Dim>
class importance_sampler {
public:
    using point_type = std::array<T, Dim>;
    using cdf_type = std::function<T(const point_type&)>;  // cumulative distribution function value (0..1)
    using sample_func = std::function<point_type(xorshift64&)>;

    importance_sampler() = default;

    // Construct with a CDF (integrates to 1) and a method to sample from it.
    importance_sampler(cdf_type cdf, sample_func sampler)
        : m_cdf(std::move(cdf)), m_sampler(std::move(sampler)) {}

    // Draw a sample, returning (point, pdf)
    std::pair<point_type, T> sample(xorshift64& rng) const {
        point_type p = m_sampler(rng);
        T pdf_value = m_cdf(p);
        return {p, pdf_value};
    }

    // Integrate a function using importance sampling
    template<typename Func>
    T integrate(Func f, std::uint32_t N, xorshift64& rng) const {
        T sum = T(0);
        for (std::uint32_t i = 0; i < N; ++i) {
            auto [point, pdf] = sample(rng);
            if (pdf > T(1e-12)) sum += f(point) / pdf;
        }
        return sum / static_cast<T>(N);
    }

private:
    cdf_type m_cdf;
    sample_func m_sampler;
};

// ============================================================
// Predefined importance samplers
// ============================================================

// Cosine‑weighted hemisphere
inline importance_sampler<float, 3> cosine_hemisphere_sampler() {
    auto cdf = [](const std::array<float, 3>& dir) -> float {
        // PDF = cos(theta) / pi, where cos(theta) = dir.y (assumes normal = (0,1,0))
        return std::max(0.0f, dir[1]) / PI_F;
    };
    auto sampler = [](xorshift64& rng) -> std::array<float, 3> {
        float u1 = uniform_float(rng);
        float u2 = uniform_float(rng);
        float r = std::sqrt(u1);
        float theta = TAU_F * u2;
        float x = r * std::cos(theta);
        float z = r * std::sin(theta);
        float y = std::sqrt(std::max(0.0f, 1.0f - u1));
        return {x, y, z};
    };
    return importance_sampler<float, 3>(cdf, sampler);
}

// Uniform sphere
inline importance_sampler<float, 3> uniform_sphere_sampler() {
    auto cdf = [](const std::array<float, 3>&) -> float {
        return 1.0f / (4.0f * PI_F);
    };
    auto sampler = [](xorshift64& rng) -> std::array<float, 3> {
        float z = uniform_range(rng, -1.0f, 1.0f);
        float r = std::sqrt(1.0f - z*z);
        float theta = TAU_F * uniform_float(rng);
        return {r * std::cos(theta), r * std::sin(theta), z};
    };
    return importance_sampler<float, 3>(cdf, sampler);
}

// GGX / Trowbridge‑Reitz (microfacet distribution)
inline importance_sampler<float, 3> ggx_vndf_sampler(float roughness) {
    auto sampler = [roughness](xorshift64& rng) -> std::array<float, 3> {
        float u1 = uniform_float(rng);
        float u2 = uniform_float(rng);
        float a = roughness * roughness;
        float phi = TAU_F * u1;
        float cos_theta = std::sqrt((1.0f - u2) / (1.0f + (a*a - 1.0f) * u2));
        float sin_theta = std::sqrt(1.0f - cos_theta*cos_theta);
        float x = sin_theta * std::cos(phi);
        float y = sin_theta * std::sin(phi);
        float z = cos_theta;
        return {x, y, z}; // sampled half‑vector in local frame (normal = 0,0,1)
    };
    auto cdf = [roughness](const std::array<float, 3>& h) -> float {
        float cos_theta = std::max(0.0f, h[2]);
        float a = roughness * roughness;
        float denom = cos_theta*cos_theta * (a*a - 1.0f) + 1.0f;
        return a*a / (PI_F * denom*denom);
    };
    return importance_sampler<float, 3>(cdf, sampler);
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_SAMPLING_H