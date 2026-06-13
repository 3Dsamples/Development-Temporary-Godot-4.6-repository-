//File 0068 : core/math/tensor_mechanics.h
//Continuum mechanics tensor operations: deformation gradient, Green‑Lagrange strain, Cauchy/1st PK/2nd PK stress conversions, principal stresses, invariants, von Mises stress; SIMD‑accelerated.
#ifndef CORE_MATH_TENSOR_MECHANICS_H
#define CORE_MATH_TENSOR_MECHANICS_H

#include "vector_math.h"
#include "matrix_math.h"
#include "linear_algebra.h"
#include <Eigen/Dense>
#include <cmath>

namespace SimulationMath {
namespace tensor_mech {

using EigenMatrix3f = Eigen::Matrix3f;
using EigenVector3f = Eigen::Vector3f;

// -----------------------------------------------------------------------------
// 1. Deformation gradient from reference (X) and current (x) tetrahedron edges
// -----------------------------------------------------------------------------
inline EigenMatrix3f deformation_gradient(const EigenVector3f& dX1, const EigenVector3f& dX2, const EigenVector3f& dX3,
                                          const EigenVector3f& dx1, const EigenVector3f& dx2, const EigenVector3f& dx3) noexcept {
    EigenMatrix3f Dm;
    Dm.col(0) = dX1; Dm.col(1) = dX2; Dm.col(2) = dX3;
    EigenMatrix3f Ds;
    Ds.col(0) = dx1; Ds.col(1) = dx2; Ds.col(2) = dx3;
    return Ds * Dm.inverse();
}

// -----------------------------------------------------------------------------
// 2. Green‑Lagrange strain: E = 0.5 * (F^T * F - I)
// -----------------------------------------------------------------------------
inline EigenMatrix3f green_lagrange_strain(const EigenMatrix3f& F) noexcept {
    return 0.5f * (F.transpose() * F - EigenMatrix3f::Identity());
}

// -----------------------------------------------------------------------------
// 3. Cauchy stress (σ) to 1st Piola‑Kirchhoff (P): P = J σ F^{-T}
// -----------------------------------------------------------------------------
inline EigenMatrix3f cauchy_to_first_piola(const EigenMatrix3f& sigma, const EigenMatrix3f& F) noexcept {
    float J = F.determinant();
    if (J <= 0.0f) J = 1e-8f;
    return J * sigma * F.inverse().transpose();
}

// -----------------------------------------------------------------------------
// 4. 1st Piola‑Kirchhoff to 2nd Piola‑Kirchhoff (S): S = F^{-1} * P
// -----------------------------------------------------------------------------
inline EigenMatrix3f first_piola_to_second_piola(const EigenMatrix3f& P, const EigenMatrix3f& F) noexcept {
    return F.inverse() * P;
}

// -----------------------------------------------------------------------------
// 5. 2nd PK stress to Cauchy: σ = (1/J) * F * S * F^T
// -----------------------------------------------------------------------------
inline EigenMatrix3f second_piola_to_cauchy(const EigenMatrix3f& S, const EigenMatrix3f& F) noexcept {
    float J = F.determinant();
    if (J <= 0.0f) J = 1e-8f;
    return (1.0f / J) * F * S * F.transpose();
}

// -----------------------------------------------------------------------------
// 6. Isotropic elasticity: compute 2nd PK from linear strain and Lamé parameters
// -----------------------------------------------------------------------------
inline EigenMatrix3f linear_elastic_second_piola(const EigenMatrix3f& E, float mu, float lambda) noexcept {
    float trE = E.trace();
    return 2.0f * mu * E + lambda * trE * EigenMatrix3f::Identity();
}

// -----------------------------------------------------------------------------
// 7. Principal stresses (eigenvalues of Cauchy stress)
// -----------------------------------------------------------------------------
inline EigenVector3f principal_stresses(const EigenMatrix3f& sigma) noexcept {
    Eigen::SelfAdjointEigenSolver<EigenMatrix3f> solver(sigma);
    return solver.eigenvalues();
}

// -----------------------------------------------------------------------------
// 8. Invariants of a 3×3 tensor (I1, I2, I3)
// -----------------------------------------------------------------------------
inline float invariant_I1(const EigenMatrix3f& T) noexcept { return T.trace(); }
inline float invariant_I2(const EigenMatrix3f& T) noexcept {
    return 0.5f * (T.trace() * T.trace() - (T * T).trace());
}
inline float invariant_I3(const EigenMatrix3f& T) noexcept { return T.determinant(); }

// -----------------------------------------------------------------------------
// 9. Deviatoric part: dev(T) = T - (1/3) * tr(T) * I
// -----------------------------------------------------------------------------
inline EigenMatrix3f deviatoric(const EigenMatrix3f& T) noexcept {
    return T - (T.trace() / 3.0f) * EigenMatrix3f::Identity();
}

// -----------------------------------------------------------------------------
// 10. Von Mises stress: sqrt(3/2 * dev(sigma) : dev(sigma))
// -----------------------------------------------------------------------------
inline float von_mises_stress(const EigenMatrix3f& sigma) noexcept {
    EigenMatrix3f dev = deviatoric(sigma);
    double sum = 0.0;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) sum += dev(i,j) * dev(i,j);
    return std::sqrt(1.5f * static_cast<float>(sum));
}

// -----------------------------------------------------------------------------
// 11. Material models: St. Venant‑Kirchhoff (hyperelastic) stress S = λ tr(E)I + 2μ E
// -----------------------------------------------------------------------------
inline EigenMatrix3f st_venant_kirchhoff_stress(const EigenMatrix3f& E, float lambda, float mu) noexcept {
    return lambda * E.trace() * EigenMatrix3f::Identity() + 2.0f * mu * E;
}

// -----------------------------------------------------------------------------
// 12. Neo‑Hookean 1st PK stress: P = μ (F - F^{-T}) + λ ln(J) F^{-T}
// -----------------------------------------------------------------------------
inline EigenMatrix3f neo_hookean_first_piola(const EigenMatrix3f& F, float mu, float lambda) noexcept {
    float J = F.determinant();
    if (J <= 0.0f) J = 1e-8f;
    EigenMatrix3f FinvT = F.inverse().transpose();
    return mu * (F - FinvT) + lambda * std::log(J) * FinvT;
}

// -----------------------------------------------------------------------------
// 13. Rotation extraction via polar decomposition (right stretch U)
// -----------------------------------------------------------------------------
inline void right_stretch(const EigenMatrix3f& F, EigenMatrix3f& R, EigenMatrix3f& U) noexcept {
    Eigen::JacobiSVD<EigenMatrix3f> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0.0f) {
        EigenMatrix3f Umat = svd.matrixU();
        Umat.col(2) *= -1.0f;
        R = Umat * svd.matrixV().transpose();
    }
    U = R.transpose() * F;
}

} // namespace tensor_mech
} // namespace SimulationMath

#endif // CORE_MATH_TENSOR_MECHANICS_H