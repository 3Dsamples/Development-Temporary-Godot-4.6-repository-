//File 0073 : core/math/finite_difference.h
//High‑order finite difference approximations (orders 2,4,6,8) for first and second derivatives on uniform 1D/2D/3D grids, with SIMD‑accelerated stencils.
#ifndef CORE_MATH_FINITE_DIFFERENCE_H
#define CORE_MATH_FINITE_DIFFERENCE_H

#include "vector_math.h"
#include <vector>
#include <cstdint>
#include <cmath>

namespace SimulationMath {
namespace fd {

// -----------------------------------------------------------------------------
// 1. First derivative stencils (central) of order 2/4/6/8 on uniform grid
// -----------------------------------------------------------------------------

// O(h²) 3‑point stencil: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
inline float df1_o2(const std::vector<float>& f, size_t idx, float h) noexcept {
    size_t n = f.size();
    if (idx == 0)       return (-3.0f*f[0] + 4.0f*f[1] - f[2]) / (2.0f*h);
    if (idx == n-1)     return (3.0f*f[n-1] - 4.0f*f[n-2] + f[n-3]) / (2.0f*h);
    return (f[idx+1] - f[idx-1]) / (2.0f * h);
}

// O(h⁴) 5‑point stencil: f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
inline float df1_o4(const std::vector<float>& f, size_t idx, float h) noexcept {
    size_t n = f.size();
    if (idx < 2) { // forward 5‑point stencil
        float a0 = -25.0f/12.0f, a1 = 4.0f, a2 = -3.0f, a3 = 4.0f/3.0f, a4 = -1.0f/4.0f;
        return (a0*f[idx] + a1*f[idx+1] + a2*f[idx+2] + a3*f[idx+3] + a4*f[idx+4]) / h;
    }
    if (idx > n-3) { // backward 5‑point stencil (mirrored)
        float a0 = 25.0f/12.0f, a1 = -4.0f, a2 = 3.0f, a3 = -4.0f/3.0f, a4 = 1.0f/4.0f;
        size_t i = idx;
        return (a0*f[i] + a1*f[i-1] + a2*f[i-2] + a3*f[i-3] + a4*f[i-4]) / h;
    }
    return (-f[idx+2] + 8.0f*f[idx+1] - 8.0f*f[idx-1] + f[idx-2]) / (12.0f * h);
}

// O(h⁶) 7‑point stencil
inline float df1_o6(const std::vector<float>& f, size_t idx, float h) noexcept {
    size_t n = f.size();
    if (idx < 3) { // forward 7‑point stencil (formulas derived from Taylor)
        // coefficients:  -49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6
        float coeff[7] = { -49.0f/20.0f, 6.0f, -7.5f, 20.0f/3.0f, -3.75f, 6.0f/5.0f, -1.0f/6.0f };
        float val = 0.0f;
        for (int k = 0; k < 7; ++k) val += coeff[k] * f[idx+k];
        return val / h;
    }
    if (idx > n-4) { // backward mirror
        float coeff[7] = { 49.0f/20.0f, -6.0f, 7.5f, -20.0f/3.0f, 3.75f, -1.2f, 1.0f/6.0f };
        float val = 0.0f;
        for (int k = 0; k < 7; ++k) val += coeff[k] * f[idx-k];
        return val / h;
    }
    // central 7‑point: ( -f3 + 9f2 - 45f1 + 0f0 + 45f-1 - 9f-2 + f-3 ) / (60h)
    return (-f[idx+3] + 9.0f*f[idx+2] - 45.0f*f[idx+1] + 45.0f*f[idx-1] - 9.0f*f[idx-2] + f[idx-3]) / (60.0f * h);
}

// O(h⁸) 9‑point stencil
inline float df1_o8(const std::vector<float>& f, size_t idx, float h) noexcept {
    size_t n = f.size();
    if (idx < 4) {
        // forward 9‑point (coefficients from Fornberg algorithm)
        float coeff[9] = { -761.0f/280.0f, 8.0f, -14.0f, 56.0f/3.0f, -35.0f/2.0f, 56.0f/5.0f, -14.0f/3.0f, 8.0f/7.0f, -1.0f/8.0f };
        float val = 0.0f;
        for (int k = 0; k < 9; ++k) val += coeff[k] * f[idx+k];
        return val / h;
    }
    if (idx > n-5) {
        float coeff[9] = { 761.0f/280.0f, -8.0f, 14.0f, -56.0f/3.0f, 35.0f/2.0f, -11.2f, 14.0f/3.0f, -8.0f/7.0f, 1.0f/8.0f };
        float val = 0.0f;
        for (int k = 0; k < 9; ++k) val += coeff[k] * f[idx-k];
        return val / h;
    }
    // central 9‑point: ( 3f4 - 32f3 + 168f2 - 672f1 + 0 + 672f-1 - 168f-2 + 32f-3 - 3f-4 ) / (840h)
    return (3.0f*f[idx+4] - 32.0f*f[idx+3] + 168.0f*f[idx+2] - 672.0f*f[idx+1] + 672.0f*f[idx-1] - 168.0f*f[idx-2] + 32.0f*f[idx-3] - 3.0f*f[idx-4]) / (840.0f * h);
}

// -----------------------------------------------------------------------------
// 2. Second derivative stencils (central) order 2/4/6
// -----------------------------------------------------------------------------
inline float df2_o2(const std::vector<float>& f, size_t idx, float h) noexcept {
    size_t n = f.size();
    if (idx == 0)       return (2.0f*f[0] - 5.0f*f[1] + 4.0f*f[2] - f[3]) / (h*h);
    if (idx == n-1)     return (2.0f*f[n-1] - 5.0f*f[n-2] + 4.0f*f[n-3] - f[n-4]) / (h*h);
    return (f[idx+1] - 2.0f*f[idx] + f[idx-1]) / (h * h);
}

inline float df2_o4(const std::vector<float>& f, size_t idx, float h) noexcept {
    size_t n = f.size();
    if (idx < 2) {
        // forward stencil (coefficients from Fornberg)
        float a0 = 15.0f/4.0f, a1 = -77.0f/6.0f, a2 = 107.0f/6.0f, a3 = -13.0f, a4 = 61.0f/12.0f, a5 = -5.0f/6.0f;
        return (a0*f[idx] + a1*f[idx+1] + a2*f[idx+2] + a3*f[idx+3] + a4*f[idx+4] + a5*f[idx+5]) / (h*h);
    }
    if (idx > n-3) {
        // backward (mirror)
        size_t i = idx;
        float a0 = 15.0f/4.0f, a1 = -77.0f/6.0f, a2 = 107.0f/6.0f, a3 = -13.0f, a4 = 61.0f/12.0f, a5 = -5.0f/6.0f;
        return (a0*f[i] + a1*f[i-1] + a2*f[i-2] + a3*f[i-3] + a4*f[i-4] + a5*f[i-5]) / (h*h);
    }
    return (-f[idx+2] + 16.0f*f[idx+1] - 30.0f*f[idx] + 16.0f*f[idx-1] - f[idx-2]) / (12.0f * h * h);
}

inline float df2_o6(const std::vector<float>& f, size_t idx, float h) noexcept {
    size_t n = f.size();
    if (idx < 3) {
        // forward 7‑point
        float coeff[7] = { 49.0f/20.0f, -54.0f/5.0f, 15.0f, -80.0f/3.0f, 45.0f/4.0f, -12.0f/5.0f, 1.0f/3.0f };
        float val = 0.0f;
        for (int k = 0; k < 7; ++k) val += coeff[k] * f[idx+k];
        return val / (h*h);
    }
    if (idx > n-4) {
        float coeff[7] = { 49.0f/20.0f, -54.0f/5.0f, 15.0f, -80.0f/3.0f, 45.0f/4.0f, -12.0f/5.0f, 1.0f/3.0f };
        float val = 0.0f;
        for (int k = 0; k < 7; ++k) val += coeff[k] * f[idx-k];
        return val / (h*h);
    }
    // central 7‑point: ( 2f3 - 27f2 + 270f1 - 490f0 + 270f-1 - 27f-2 + 2f-3 ) / (180h²)
    return (2.0f*f[idx+3] - 27.0f*f[idx+2] + 270.0f*f[idx+1] - 490.0f*f[idx] + 270.0f*f[idx-1] - 27.0f*f[idx-2] + 2.0f*f[idx-3]) / (180.0f * h * h);
}

// -----------------------------------------------------------------------------
// 3. SIMD versions for vector fields (array of DirectX::XMVECTOR)
// -----------------------------------------------------------------------------
inline DirectX::XMVECTOR df1_vec_o2(const std::vector<DirectX::XMVECTOR>& f, size_t idx, float h) noexcept {
    size_t n = f.size();
    DirectX::XMVECTOR result;
    if (idx == 0)
        result = DirectX::XMVectorScale(
            DirectX::XMVectorAdd(
                DirectX::XMVectorAdd(DirectX::XMVectorScale(f[0], -3.0f), DirectX::XMVectorScale(f[1], 4.0f)),
                DirectX::XMVectorNegate(f[2])), 1.0f/(2.0f*h));
    else if (idx == n-1)
        result = DirectX::XMVectorScale(
            DirectX::XMVectorAdd(
                DirectX::XMVectorAdd(DirectX::XMVectorScale(f[n-1], 3.0f), DirectX::XMVectorScale(f[n-2], -4.0f)),
                f[n-3]), 1.0f/(2.0f*h));
    else
        result = DirectX::XMVectorScale(
            DirectX::XMVectorSubtract(f[idx+1], f[idx-1]), 1.0f/(2.0f*h));
    return result;
}

inline DirectX::XMVECTOR df1_vec_o4(const std::vector<DirectX::XMVECTOR>& f, size_t idx, float h) noexcept {
    size_t n = f.size();
    if (idx < 2) {
        // forward (same coefficients as scalar)
        float a[5] = { -25.0f/12.0f, 4.0f, -3.0f, 4.0f/3.0f, -1.0f/4.0f };
        DirectX::XMVECTOR sum = DirectX::XMVectorZero();
        for (int k = 0; k < 5; ++k)
            sum = DirectX::XMVectorMultiplyAdd(f[idx+k], DirectX::XMVectorReplicate(a[k]), sum);
        return DirectX::XMVectorScale(sum, 1.0f/h);
    }
    if (idx > n-3) {
        float a[5] = { 25.0f/12.0f, -4.0f, 3.0f, -4.0f/3.0f, 1.0f/4.0f };
        DirectX::XMVECTOR sum = DirectX::XMVectorZero();
        for (int k = 0; k < 5; ++k)
            sum = DirectX::XMVectorMultiplyAdd(f[idx-k], DirectX::XMVectorReplicate(a[k]), sum);
        return DirectX::XMVectorScale(sum, 1.0f/h);
    }
    return DirectX::XMVectorScale(
        DirectX::XMVectorAdd(
            DirectX::XMVectorAdd(
                DirectX::XMVectorScale(f[idx+2], -1.0f),
                DirectX::XMVectorScale(f[idx+1], 8.0f)),
            DirectX::XMVectorAdd(
                DirectX::XMVectorScale(f[idx-1], -8.0f),
                f[idx-2])), 1.0f/(12.0f*h));
}

inline DirectX::XMVECTOR df2_vec_o2(const std::vector<DirectX::XMVECTOR>& f, size_t idx, float h) noexcept {
    size_t n = f.size();
    if (idx == 0)
        return DirectX::XMVectorScale(
            DirectX::XMVectorAdd(
                DirectX::XMVectorAdd(DirectX::XMVectorScale(f[0], 2.0f), DirectX::XMVectorScale(f[1], -5.0f)),
                DirectX::XMVectorAdd(DirectX::XMVectorScale(f[2], 4.0f), DirectX::XMVectorNegate(f[3]))), 1.0f/(h*h));
    if (idx == n-1)
        return DirectX::XMVectorScale(
            DirectX::XMVectorAdd(
                DirectX::XMVectorAdd(DirectX::XMVectorScale(f[n-1], 2.0f), DirectX::XMVectorScale(f[n-2], -5.0f)),
                DirectX::XMVectorAdd(DirectX::XMVectorScale(f[n-3], 4.0f), DirectX::XMVectorNegate(f[n-4]))), 1.0f/(h*h));
    return DirectX::XMVectorScale(
        DirectX::XMVectorAdd(
            DirectX::XMVectorAdd(f[idx+1], f[idx-1]),
            DirectX::XMVectorScale(f[idx], -2.0f)), 1.0f/(h*h));
}

inline DirectX::XMVECTOR df2_vec_o4(const std::vector<DirectX::XMVECTOR>& f, size_t idx, float h) noexcept {
    size_t n = f.size();
    if (idx < 2) {
        float a[6] = { 15.0f/4.0f, -77.0f/6.0f, 107.0f/6.0f, -13.0f, 61.0f/12.0f, -5.0f/6.0f };
        DirectX::XMVECTOR sum = DirectX::XMVectorZero();
        for (int k = 0; k < 6; ++k)
            sum = DirectX::XMVectorMultiplyAdd(f[idx+k], DirectX::XMVectorReplicate(a[k]), sum);
        return DirectX::XMVectorScale(sum, 1.0f/(h*h));
    }
    if (idx > n-3) {
        float a[6] = { 15.0f/4.0f, -77.0f/6.0f, 107.0f/6.0f, -13.0f, 61.0f/12.0f, -5.0f/6.0f };
        DirectX::XMVECTOR sum = DirectX::XMVectorZero();
        for (int k = 0; k < 6; ++k)
            sum = DirectX::XMVectorMultiplyAdd(f[idx-k], DirectX::XMVectorReplicate(a[k]), sum);
        return DirectX::XMVectorScale(sum, 1.0f/(h*h));
    }
    return DirectX::XMVectorScale(
        DirectX::XMVectorAdd(
            DirectX::XMVectorAdd(
                DirectX::XMVectorScale(f[idx+2], -1.0f),
                DirectX::XMVectorScale(f[idx+1], 16.0f)),
            DirectX::XMVectorAdd(
                DirectX::XMVectorScale(f[idx], -30.0f),
                DirectX::XMVectorAdd(
                    DirectX::XMVectorScale(f[idx-1], 16.0f),
                    DirectX::XMVectorNegate(f[idx-2])))), 1.0f/(12.0f*h*h));
}

// -----------------------------------------------------------------------------
// 4. 2D/3D Laplacian (central second order) on uniform grid
// -----------------------------------------------------------------------------
inline float laplacian_2d(const std::vector<std::vector<float>>& grid, size_t x, size_t y, float h) noexcept {
    float val = -4.0f * grid[y][x];
    val += grid[y][x+1];
    val += grid[y][x-1];
    val += grid[y+1][x];
    val += grid[y-1][x];
    return val / (h * h);
}

inline float laplacian_3d(const std::vector<std::vector<std::vector<float>>>& grid,
                          size_t x, size_t y, size_t z, float h) noexcept {
    float val = -6.0f * grid[z][y][x];
    val += grid[z][y][x+1];
    val += grid[z][y][x-1];
    val += grid[z][y+1][x];
    val += grid[z][y-1][x];
    val += grid[z+1][y][x];
    val += grid[z-1][y][x];
    return val / (h * h);
}

} // namespace fd
} // namespace SimulationMath

#endif // CORE_MATH_FINITE_DIFFERENCE_H