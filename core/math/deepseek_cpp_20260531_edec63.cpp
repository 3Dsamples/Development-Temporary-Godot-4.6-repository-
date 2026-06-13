//File 0069 : core/math/grid_interpolators.h
//Higher‑order 3D grid interpolation (tricubic, Catmull‑Rom, monotone cubic Hermite) for scalar and vector fields using SIMD vector_math.
#ifndef CORE_MATH_GRID_INTERPOLATORS_H
#define CORE_MATH_GRID_INTERPOLATORS_H

#include "vector_math.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace SimulationMath {
namespace grid_interp {

// -----------------------------------------------------------------------------
// 1. Cubic convolution kernel (used in tricubic interpolation)
// -----------------------------------------------------------------------------
inline float cubic_kernel(float x) noexcept {
    x = std::fabs(x);
    if (x <= 1.0f) {
        return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    } else if (x < 2.0f) {
        float t = 2.0f - x;
        return -0.5f * t * t * t + t * t + 0.5f * t;
    }
    return 0.0f;
}

// -----------------------------------------------------------------------------
// 2. Tricubic scalar interpolation (3D grid of floats)
// -----------------------------------------------------------------------------
inline float tricubic_scalar(const std::vector<std::vector<std::vector<float>>>& data,
                              float x, float y, float z, size_t Nx, size_t Ny, size_t Nz) noexcept {
    int ix = static_cast<int>(std::floor(x));
    int iy = static_cast<int>(std::floor(y));
    int iz = static_cast<int>(std::floor(z));
    float fx = x - ix, fy = y - iy, fz = z - iz;

    auto clamp_idx = [](int idx, int max) { return std::max(0, std::min(idx, max)); };
    auto sample = [&](int i, int j, int k) -> float {
        return data[clamp_idx(k, (int)Nz-1)][clamp_idx(j, (int)Ny-1)][clamp_idx(i, (int)Nx-1)];
    };

    float result = 0.0f;
    for (int k = iz-1; k <= iz+2; ++k) {
        float wk = cubic_kernel(fz - (k - iz));
        if (wk == 0.0f) continue;
        for (int j = iy-1; j <= iy+2; ++j) {
            float wj = cubic_kernel(fy - (j - iy));
            if (wj == 0.0f) continue;
            float wjk = wk * wj;
            for (int i = ix-1; i <= ix+2; ++i) {
                float wi = cubic_kernel(fx - (i - ix));
                if (wi == 0.0f) continue;
                result += wjk * wi * sample(i, j, k);
            }
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
// 3. Tricubic vector interpolation (grid of XMVECTORs)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR tricubic_vector(const std::vector<std::vector<std::vector<DirectX::XMVECTOR>>>& data,
                                          float x, float y, float z, size_t Nx, size_t Ny, size_t Nz) noexcept {
    int ix = static_cast<int>(std::floor(x));
    int iy = static_cast<int>(std::floor(y));
    int iz = static_cast<int>(std::floor(z));
    float fx = x - ix, fy = y - iy, fz = z - iz;

    auto clamp_idx = [](int idx, int max) { return std::max(0, std::min(idx, max)); };
    auto sample = [&](int i, int j, int k) -> DirectX::XMVECTOR {
        return data[clamp_idx(k, (int)Nz-1)][clamp_idx(j, (int)Ny-1)][clamp_idx(i, (int)Nx-1)];
    };

    DirectX::XMVECTOR result = DirectX::XMVectorZero();
    for (int k = iz-1; k <= iz+2; ++k) {
        float wk = cubic_kernel(fz - (k - iz));
        if (wk == 0.0f) continue;
        for (int j = iy-1; j <= iy+2; ++j) {
            float wj = cubic_kernel(fy - (j - iy));
            if (wj == 0.0f) continue;
            float wjk = wk * wj;
            for (int i = ix-1; i <= ix+2; ++i) {
                float wi = cubic_kernel(fx - (i - ix));
                if (wi == 0.0f) continue;
                result = DirectX::XMVectorMultiplyAdd(sample(i, j, k), DirectX::XMVectorReplicate(wjk * wi), result);
            }
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
// 4. Monotone cubic Hermite interpolation (1D, Fritsch‑Carlson slopes)
// -----------------------------------------------------------------------------
inline float monotone_hermite_1d(const float* x, const float* y, size_t n, float xi) noexcept {
    if (n <= 1) return y[0];
    if (xi <= x[0]) return y[0];
    if (xi >= x[n-1]) return y[n-1];

    size_t k = 0;
    while (k < n-1 && x[k+1] < xi) ++k;
    float h = x[k+1] - x[k];
    if (h == 0.0f) return y[k];

    float t = (xi - x[k]) / h;

    // Compute secant slopes
    std::vector<float> delta(n-1);
    for (size_t i = 0; i < n-1; ++i) {
        float hi = x[i+1] - x[i];
        if (hi == 0.0f) delta[i] = 0.0f;
        else delta[i] = (y[i+1] - y[i]) / hi;
    }

    // Compute slopes dk and dk+1 at endpoints of the interval
    float dk = 0.0f, dk1 = 0.0f;
    // For interior intervals
    if (k > 0 && k < n-1) {
        float h0 = x[k] - x[k-1];
        float h1 = x[k+1] - x[k];
        float s0 = delta[k-1];
        float s1 = delta[k];
        if (s0 * s1 <= 0.0f) dk = 0.0f;
        else {
            float w1 = 2.0f * h1 + h0;
            float w2 = h1 + 2.0f * h0;
            dk = (w1 / s0 + w2 / s1) != 0.0f ? (3.0f * (s0 * w1 + s1 * w2) / (w1 / s0 + w2 / s1)) * 0.5f : 0.0f;
        }
    }
    if (k+1 > 0 && k+1 < n-1) {
        float h0 = x[k+1] - x[k];
        float h1 = x[k+2] - x[k+1];
        float s0 = delta[k];
        float s1 = delta[k+1];
        if (s0 * s1 <= 0.0f) dk1 = 0.0f;
        else {
            float w1 = 2.0f * h1 + h0;
            float w2 = h1 + 2.0f * h0;
            dk1 = (3.0f * (s0 * w1 + s1 * w2) / (w1 / s0 + w2 / s1)) * 0.5f;
        }
    }

    // Hermite interpolation
    float h00 = (1.0f + 2.0f * t) * (1.0f - t) * (1.0f - t);
    float h10 = t * (1.0f - t) * (1.0f - t);
    float h01 = t * t * (3.0f - 2.0f * t);
    float h11 = t * t * (t - 1.0f);

    return y[k] * h00 + dk * h * h10 + y[k+1] * h01 + dk1 * h * h11;
}

// -----------------------------------------------------------------------------
// 5. Monotone cubic Hermite 3D (separable, using 3 axes)
// -----------------------------------------------------------------------------
inline float monotone_hermite_3d(const std::vector<std::vector<std::vector<float>>>& data,
                                  const std::vector<float>& x_axis,
                                  const std::vector<float>& y_axis,
                                  const std::vector<float>& z_axis,
                                  size_t Nx, size_t Ny, size_t Nz,
                                  float x, float y, float z) noexcept {
    // Interpolate along x for each y,z
    std::vector<std::vector<float>> temp_y(Ny, std::vector<float>(Nz));
    for (size_t k = 0; k < Nz; ++k) {
        for (size_t j = 0; j < Ny; ++j) {
            std::vector<float> column(Nx);
            for (size_t i = 0; i < Nx; ++i) column[i] = data[k][j][i];
            temp_y[j][k] = monotone_hermite_1d(x_axis.data(), column.data(), Nx, x);
        }
    }
    // Interpolate along y
    std::vector<float> temp_z(Nz);
    for (size_t k = 0; k < Nz; ++k) {
        std::vector<float> row(Ny);
        for (size_t j = 0; j < Ny; ++j) row[j] = temp_y[j][k];
        temp_z[k] = monotone_hermite_1d(y_axis.data(), row.data(), Ny, y);
    }
    // Interpolate along z
    return monotone_hermite_1d(z_axis.data(), temp_z.data(), Nz, z);
}

} // namespace grid_interp
} // namespace SimulationMath

#endif // CORE_MATH_GRID_INTERPOLATORS_H