// system name : Octree Spatial Master
//File 0041 : core/math/fixed_tensor_visualization.h
//Tensor glyph generation (ellipsoids, superquadrics), perceptual colour mapping, streamline integration, SIMD batch
#pragma once
#include "core/math/fixed_scalar.h"
#include "core/math/fixed_vec3.h"
#include "core/math/fixed_mat.h"
#include "core/math/fixed_tensor.h"
#include "core/math/fixed_spectral_decomposition.h"
#include "core/math/fixed_interpolation.h"
#include "sim_math_unified_conversions.h"
#include "perceptual_color.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>

namespace fixed_math {

// Tensor ellipsoid from symmetric 3x3 matrix (e.g., stress, strain, diffusion tensor)
// Axes are eigenvectors scaled by eigenvalues; returns three axis vectors.
inline void tensor_ellipsoid_axes(const fmat3& T, fvec3 axes[3]) noexcept {
    fmat3 V;
    fixed64_t lambda[3];
    symmetric_eigen_decomposition(T, V, lambda);
    sort_eigen_descending(V, lambda);
    for (int i=0; i<3; ++i) {
        if (lambda[i] < 0) lambda[i] = 0;
        axes[i] = fvec3_scale({V.rows[0].x+i, V.rows[1].x+i, V.rows[2].x+i}, fixed_sqrt(lambda[i]));
    }
}

// Superquadric exponent parameters from tensor anisotropy (Cl = 2 for isotropic)
inline void tensor_superquadric_params(const fmat3& T, fixed64_t& alpha, fixed64_t& beta, fvec3 axes[3]) noexcept {
    tensor_ellipsoid_axes(T, axes);
    fixed64_t l1 = fvec3_length(axes[0]), l2 = fvec3_length(axes[1]), l3 = fvec3_length(axes[2]);
    fixed64_t maxL = std::max({l1,l2,l3});
    if (maxL == 0) maxL = 1;
    l1 = fixed_div(l1, maxL); l2 = fixed_div(l2, maxL); l3 = fixed_div(l3, maxL);
    // Cl = 2 - 2*(c_s / c_l)^2   approximate mapping
    fixed64_t cl = l1; // dominant
    fixed64_t cs = l3; // minor
    fixed64_t shape = FIXED64_ONE - fixed_mul(cs, cs) / (cl*cl + 1); // avoid division by zero
    shape = fixed_clamp(shape, 0, FIXED64_ONE);
    alpha = FIXED64_ONE + shape;  // between 1 and 2
    beta = FIXED64_ONE + shape;
}

// Evaluate superquadric surface radius in direction (theta, phi) with exponents e1, e2
inline fixed64_t superquadric_radius(fixed64_t theta, fixed64_t phi, fixed64_t e1, fixed64_t e2,
                                      fixed64_t a1, fixed64_t a2, fixed64_t a3) noexcept {
    fixed64_t ct = fixed_cos(theta), st = fixed_sin(theta);
    fixed64_t cp = fixed_cos(phi), sp = fixed_sin(phi);
    fixed64_t t1 = fixed_pow(fixed_abs(st), e1);
    fixed64_t t2 = fixed_pow(fixed_abs(ct), e1);
    fixed64_t t3 = fixed_pow(fixed_abs(sp), e2);
    fixed64_t t4 = fixed_pow(fixed_abs(cp), e2);
    fixed64_t term1 = a1 * t1 * t3;
    fixed64_t term2 = a2 * t1 * t4;
    fixed64_t term3 = a3 * t2;
    fixed64_t denom = fixed_pow(fixed_pow(term1, e2) + fixed_pow(term2, e2) + fixed_pow(term3, e2), fixed_rcp(e2));
    if (denom == 0) return 0;
    return fixed_div(FIXED64_ONE, denom);
}

// Map tensor to a perceptual colour via principal stress direction and magnitude
inline fvec3 tensor_color(const fmat3& T) noexcept {
    fmat3 V;
    fixed64_t lambda[3];
    symmetric_eigen_decomposition(T, V, lambda);
    // Normalize first principal direction and map to RGB
    fvec3 dir = {V.rows[0].x+0, V.rows[1].x+0, V.rows[2].x+0}; // first eigenvector column
    dir = fvec3_normalize(dir);
    fixed64_t magnitude = fixed_sqrt(lambda[0]*lambda[0] + lambda[1]*lambda[1] + lambda[2]*lambda[2]);
    fixed64_t max_mag = (magnitude == 0) ? FIXED64_ONE : magnitude;
    fixed64_t r = fixed_div(fixed_abs(dir.x) * max_mag, max_mag);
    fixed64_t g = fixed_div(fixed_abs(dir.y) * max_mag, max_mag);
    fixed64_t b = fixed_div(fixed_abs(dir.z) * max_mag, max_mag);
    r = fixed_clamp(r, 0, FIXED64_ONE); g = fixed_clamp(g, 0, FIXED64_ONE); b = fixed_clamp(b, 0, FIXED64_ONE);
    fvec3 linear = {r, g, b};
    return perceptual_color::linear_srgb_to_oklab(linear);
}

// Streamline integration for a vector field (using Euler method)
// field: function taking a point and returning a vector; start point, step size, steps
inline std::vector<fvec3> integrate_streamline(const std::function<fvec3(const fvec3&)>& field,
                                                const fvec3& start, fixed64_t step, int max_steps) noexcept {
    std::vector<fvec3> pts;
    pts.push_back(start);
    fvec3 pos = start;
    for (int i=0; i<max_steps; ++i) {
        fvec3 vel = field(pos);
        pos = fvec3_add(pos, fvec3_scale(vel, step));
        pts.push_back(pos);
    }
    return pts;
}

// Streamline for a tensor field (major eigenvector direction)
inline std::vector<fvec3> tensor_streamline(const std::function<fmat3(const fvec3&)>& tensor_field,
                                             const fvec3& start, fixed64_t step, int max_steps) noexcept {
    auto vec_field = [&](const fvec3& p) -> fvec3 {
        fmat3 T = tensor_field(p);
        fmat3 V; fixed64_t lambda[3];
        symmetric_eigen_decomposition(T, V, lambda);
        // major eigenvector
        return fvec3_normalize({V.rows[0].x+0, V.rows[1].x+0, V.rows[2].x+0});
    };
    return integrate_streamline(vec_field, start, step, max_steps);
}

// Colour a streamline segment based on local tensor magnitude
inline void streamline_colors(const std::vector<fvec3>& points,
                              const std::function<fmat3(const fvec3&)>& tensor_field,
                              std::vector<fvec3>& colors) noexcept {
    colors.resize(points.size());
    for (size_t i=0; i<points.size(); ++i) {
        fmat3 T = tensor_field(points[i]);
        colors[i] = tensor_color(T);
    }
}

// SIMD batch: compute ellipsoid axes for 4 tensors
inline void tensor_ellipsoid_batch(const fmat3 T[4], fvec3 axes[4][3]) noexcept {
    for (int i=0; i<4; ++i) tensor_ellipsoid_axes(T[i], axes[i]);
}

// Perceptual glyph colour for multiple tensors (array output)
inline void tensor_color_batch(const fmat3 T[4], fvec3 colors[4]) noexcept {
    for (int i=0; i<4; ++i) colors[i] = tensor_color(T[i]);
}

} // namespace fixed_math