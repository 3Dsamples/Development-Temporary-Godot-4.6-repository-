//File 0106 : core/math/mesh_spherical_harmonics.h
//Spherical harmonic decomposition of scalar functions on a mesh mapped to sphere: Legendre recurrences, real SH basis, coefficient fitting, and reconstruction.
#ifndef CORE_MATH_MESH_SPHERICAL_HARMONICS_H
#define CORE_MATH_MESH_SPHERICAL_HARMONICS_H

#include "mesh_data.h"               // HalfEdgeMesh
#include "vector_math.h"
#include "math_constants.h"
#include <vector>
#include <complex>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

namespace SimulationMath {
namespace sh {

using namespace mesh;

// -----------------------------------------------------------------------------
// 1. Normalised associated Legendre polynomial P_l^m(x) using recurrence.
//    Uses the standard normalisation such that the spherical harmonic Y_l^m
//    integrates to 1 over the unit sphere.
// -----------------------------------------------------------------------------
inline double associated_legendre(int l, int m, double x) noexcept {
    // x in [-1,1]
    if (m < 0 || m > l) return 0.0;
    double pmm = 1.0;
    if (m > 0) {
        double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if (l == m) return pmm;

    double pmmp1 = x * (2.0 * m + 1.0) * pmm;
    if (l == m + 1) return pmmp1;

    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (double)(ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

// -----------------------------------------------------------------------------
// 2. Real spherical harmonic basis function Y_l^m(theta, phi)
//    theta: polar angle (0 to PI), phi: azimuth (0 to 2PI).
//    l >= 0, -l <= m <= l.
//    Convention: Y_l^m(θ,φ) = K_l^m * P_l^|m|(cosθ) * f(m,φ)
//    where K_l^m = sqrt( (2l+1) / (4π) * (l-|m|)! / (l+|m|)! )
//    and f(m,φ) = cos(|m|φ) for m>=0, sin(|m|φ) for m<0.
// -----------------------------------------------------------------------------
inline double spherical_harmonic_real(int l, int m, double theta, double phi) noexcept {
    double abs_m = (double)std::abs(m);
    double P = associated_legendre(l, (int)abs_m, std::cos(theta));
    // Normalisation factor
    double fact = 1.0;
    for (int i = l - (int)abs_m + 1; i <= l + (int)abs_m; ++i) fact *= i; // (l+|m|)! / (l-|m|)!
    // Actually K = sqrt( (2l+1)/(4π) * (l-|m|)! / (l+|m|)! )
    // We'll compute factorial ratio iteratively to avoid overflow.
    double ratio = 1.0;
    for (int i = l - (int)abs_m + 1; i <= l + (int)abs_m; ++i) ratio /= i;
    double K = std::sqrt((2.0 * l + 1.0) / (4.0 * constants::PI) * ratio);
    // Multiply by appropriate trigonometric function
    double val = K * P;
    if (m >= 0)
        return val * std::cos(abs_m * phi);
    else
        return val * std::sin(abs_m * phi);
}

// -----------------------------------------------------------------------------
// 3. Map mesh vertices to unit sphere by normalising positions (assumes mesh centered at origin)
//    Returns the spherical coordinates (theta, phi) and the radius values.
// -----------------------------------------------------------------------------
inline void map_mesh_to_sphere(const HalfEdgeMesh& mesh,
                                std::vector<double>& radii,
                                std::vector<double>& thetas,
                                std::vector<double>& phis) noexcept {
    size_t nv = mesh.vertex_count();
    radii.resize(nv);
    thetas.resize(nv);
    phis.resize(nv);
    const auto& verts = mesh.vertices();
    for (size_t i = 0; i < nv; ++i) {
        DirectX::XMVECTOR p = verts[i].position;
        double x = vector_math::get_x(p);
        double y = vector_math::get_y(p);
        double z = vector_math::get_z(p);
        double r = std::sqrt(x*x + y*y + z*z);
        if (r < 1e-12) r = 1e-12;
        radii[i] = r;
        thetas[i] = std::acos(z / r);       // polar angle [0,PI]
        phis[i]   = std::atan2(y, x);       // azimuth [-PI,PI]
    }
}

// -----------------------------------------------------------------------------
// 4. Estimate area weight for each vertex using one‑third of incident face areas.
//    Returns a vector of weights proportional to area.
// -----------------------------------------------------------------------------
inline std::vector<double> compute_vertex_area_weights(const HalfEdgeMesh& mesh) noexcept {
    size_t nv = mesh.vertex_count();
    std::vector<double> weights(nv, 0.0);
    const auto& hedges = mesh.half_edges();
    const auto& verts  = mesh.vertices();
    for (size_t f = 0; f < mesh.faces().size(); ++f) {
        const MeshFace& face = mesh.faces()[f];
        uint32_t he0 = face.first_edge;
        if (he0 == 0xFFFFFFFFu) continue;
        uint32_t v0 = hedges[he0].vertex_index;
        uint32_t v1 = hedges[hedges[he0].next_edge].vertex_index;
        uint32_t v2 = hedges[hedges[he0].next_edge].next_edge;
        v2 = hedges[v2].vertex_index;
        DirectX::XMVECTOR p0 = verts[v0].position;
        DirectX::XMVECTOR p1 = verts[v1].position;
        DirectX::XMVECTOR p2 = verts[v2].position;
        DirectX::XMVECTOR cross = vector_math::cross3(
            DirectX::XMVectorSubtract(p1, p0), DirectX::XMVectorSubtract(p2, p0));
        double area = 0.5 * vector_math::length3_scalar(cross);
        double contrib = area / 3.0;
        weights[v0] += contrib;
        weights[v1] += contrib;
        weights[v2] += contrib;
    }
    return weights;
}

// -----------------------------------------------------------------------------
// 5. Fit spherical harmonic coefficients up to a given maximum degree L_max
//    to a scalar function defined on the mesh vertices.
//    The function values are given by `scalar_field`.
//    Uses the real spherical harmonics Y_l^m evaluated at each vertex's
//    spherical coordinates (theta, phi). The weights are used for integration.
//    Returns the coefficients stored as a vector of size (L_max+1)^2,
//    indexed linearly by l,m: index = l*(l+1) + m.
// -----------------------------------------------------------------------------
inline std::vector<double> fit_spherical_harmonics(
    const std::vector<double>& scalar_field,
    const std::vector<double>& thetas,
    const std::vector<double>& phis,
    const std::vector<double>& weights,
    int L_max) noexcept {
    size_t nv = scalar_field.size();
    int num_coeff = (L_max + 1) * (L_max + 1);
    std::vector<double> coeff(num_coeff, 0.0);

    // Compute the sum of weights for normalisation (should be total area ~ 4π)
    double total_weight = 0.0;
    for (size_t i = 0; i < nv; ++i) total_weight += weights[i];
    if (total_weight <= 0.0) return coeff;

    for (size_t i = 0; i < nv; ++i) {
        double f = scalar_field[i];
        double w = weights[i];
        double theta = thetas[i];
        double phi   = phis[i];
        for (int l = 0; l <= L_max; ++l) {
            for (int m = -l; m <= l; ++m) {
                double y = spherical_harmonic_real(l, m, theta, phi);
                int idx = l * (l + 1) + m;
                coeff[idx] += f * y * w / total_weight * (4.0 * constants::PI); // normalise to sphere area 4π
            }
        }
    }
    return coeff;
}

// -----------------------------------------------------------------------------
// 6. Reconstruct scalar value from SH coefficients at a given direction (theta, phi)
// -----------------------------------------------------------------------------
inline double reconstruct_from_sh(const std::vector<double>& coeff, double theta, double phi) noexcept {
    int L_max = (int)std::sqrt((double)coeff.size()) - 1;
    double val = 0.0;
    for (int l = 0; l <= L_max; ++l) {
        for (int m = -l; m <= l; ++m) {
            int idx = l * (l + 1) + m;
            val += coeff[idx] * spherical_harmonic_real(l, m, theta, phi);
        }
    }
    return val;
}

} // namespace sh
} // namespace SimulationMath

#endif // CORE_MATH_MESH_SPHERICAL_HARMONICS_H