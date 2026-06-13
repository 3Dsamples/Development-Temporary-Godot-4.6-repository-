//File 0063 : core/math/polar_decomposition.h
//Polar decomposition of 3×3 matrix (A = R·S) using Higham iteration; R orthogonal, S symmetric positive‑semidefinite; uses DirectXMath for SIMD.
#ifndef CORE_MATH_POLAR_DECOMPOSITION_H
#define CORE_MATH_POLAR_DECOMPOSITION_H

#include "vector_math.h"
#include "matrix_math.h"
#include <cmath>

namespace SimulationMath {
namespace polar {

// -----------------------------------------------------------------------------
// 1. Polar decomposition A = R * S, A is a 3×3 matrix stored in a 4×4 XMMATRIX (upper‑left 3×3, rest identity)
// -----------------------------------------------------------------------------
inline void decompose(DirectX::FXMMATRIX A, DirectX::XMMATRIX& out_R, DirectX::XMMATRIX& out_S,
                      int max_iters = 20, float tolerance = 1e-6f) noexcept {
    DirectX::XMMATRIX X = A;
    DirectX::XMMATRIX X_prev;
    for (int iter = 0; iter < max_iters; ++iter) {
        X_prev = X;
        DirectX::XMMATRIX X_inv_trans;
        DirectX::XMVECTOR det;
        DirectX::XMMATRIX X_inv = DirectX::XMMatrixInverse(&det, X);
        X_inv_trans = DirectX::XMMatrixTranspose(X_inv);
        X = DirectX::XMMatrixMultiply(
            DirectX::XMMatrixMultiply(
                DirectX::XMMatrixAdd(X, X_inv_trans),
                DirectX::XMMatrixReplicate(0.5f)),
            DirectX::XMMatrixIdentity()); // essentially scale by 0.5
        // Check convergence: compute norm of difference
        DirectX::XMMATRIX diff = DirectX::XMMatrixSubtract(X, X_prev);
        float diff_norm = std::sqrt(
            vector_math::get_x(diff.r[0]) * vector_math::get_x(diff.r[0]) +
            vector_math::get_y(diff.r[0]) * vector_math::get_y(diff.r[0]) +
            vector_math::get_z(diff.r[0]) * vector_math::get_z(diff.r[0]) +
            vector_math::get_x(diff.r[1]) * vector_math::get_x(diff.r[1]) +
            vector_math::get_y(diff.r[1]) * vector_math::get_y(diff.r[1]) +
            vector_math::get_z(diff.r[1]) * vector_math::get_z(diff.r[1]) +
            vector_math::get_x(diff.r[2]) * vector_math::get_x(diff.r[2]) +
            vector_math::get_y(diff.r[2]) * vector_math::get_y(diff.r[2]) +
            vector_math::get_z(diff.r[2]) * vector_math::get_z(diff.r[2]));
        if (diff_norm < tolerance) break;
    }
    out_R = X;
    // S = R^T * A
    DirectX::XMMATRIX R_T = DirectX::XMMatrixTranspose(out_R);
    out_S = DirectX::XMMatrixMultiply(R_T, A);
}

} // namespace polar
} // namespace SimulationMath

#endif // CORE_MATH_POLAR_DECOMPOSITION_H