// system name : onetbb-warp
// File 0052 : core/math/radiosity.h
// Description : Form‑factor computation, progressive radiosity solver, spherical harmonic radiosity maps.

#ifndef __TBB_WARP_CORE_MATH_RADIOSITY_H
#define __TBB_WARP_CORE_MATH_RADIOSITY_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include "core/math/vector3.h"
#include "core/math/matrix3.h"
#include "core/math/matrix4.h"
#include "core/math/geometry.h"
#include "core/math/quaternion.h"
#include "core/math/color.h"
#include "core/math/spherical_harmonics.h"
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <cstring>

namespace tbb {
namespace core {
namespace math {
namespace radiosity {

// ============================================================
// 1. Radiosity patch descriptor
// ============================================================

template<typename T>
struct radiosity_patch {
    vector3<T> centroid;
    vector3<T> normal;
    T area;
    T emissivity;           // emitted radiosity (W/m² or arbitrary)
    T reflectance;          // diffuse reflectance (0..1)
    T radiosity;            // current total radiosity
    T unshot_radiosity;     // energy not yet distributed
    vector3<T> color;       // base colour (modulated by radiosity)
    std::uint32_t id;
    std::vector<vector3<T>> vertices; // for projection
};

// ============================================================
// 2. Hemicube face descriptor
// ============================================================

template<typename T>
struct hemicube_face {
    vector3<T> center;       // world position of face center
    vector3<T> u_axis;       // horizontal axis
    vector3<T> v_axis;       // vertical axis
    vector3<T> normal;
    T half_size;
};

// ============================================================
// 3. Hemicube construction from source patch
// ============================================================

template<typename T>
std::array<hemicube_face<T>, 5> build_hemicube(const radiosity_patch<T>& src, T half_size = T(1)) {
    std::array<hemicube_face<T>, 5> faces;
    vector3<T> N = src.normal;
    vector3<T> tangent, bitangent;
    orthonormal_basis(N, tangent, bitangent);
    vector3<T> pos = src.centroid + N * half_size; // hemicube bottom at source, top faces at +half_size
    // Top face (+Y in hemicube space = +N)
    faces[0] = { pos + N * half_size, tangent, bitangent, N, half_size };
    // Four side faces: -X, +X, -Y, +Y? Actually standard hemicube: top, front, back, left, right.
    // We'll map: +tangent = right, +bitangent = front, -tangent = left, -bitangent = back.
    faces[1] = { pos + tangent * half_size, bitangent, N, tangent, half_size };  // right
    faces[2] = { pos - tangent * half_size, bitangent, -N, -tangent, half_size }; // left
    faces[3] = { pos + bitangent * half_size, -tangent, N, bitangent, half_size }; // front
    faces[4] = { pos - bitangent * half_size, tangent, N, -bitangent, half_size }; // back
    return faces;
}

// ============================================================
// 4. Project patch onto hemicube face and accumulate delta form factors
// ============================================================

template<typename T>
T delta_form_factor_pixel(const vector3<T>& face_center, const vector3<T>& normal,
                          const vector3<T>& world_point, T pixel_area) {
    vector3<T> to_point = world_point - face_center;
    T dist2 = length_sq(to_point);
    if (dist2 < T(1e-12)) return T(0);
    T cos_src = std::abs(dot(normal, to_point / std::sqrt(dist2)));
    return (pixel_area * cos_src) / (T(PI_D) * dist2);
}

template<typename T>
void rasterize_patch_on_face(const radiosity_patch<T>& patch,
                             const hemicube_face<T>& face,
                             std::uint32_t resolution,
                             std::vector<T>& face_buffer,
                             T total_pixels) {
    // Project patch vertices to 2D pixel coordinates on this face
    std::vector<std::array<T,2>> projected(patch.vertices.size());
    bool all_behind = true;
    T half = face.half_size;
    for (std::size_t i = 0; i < patch.vertices.size(); ++i) {
        vector3<T> rel = patch.vertices[i] - face.center;
        T u = dot(rel, face.u_axis) / half;
        T v = dot(rel, face.v_axis) / half;
        if (std::abs(u) < T(1.001) && std::abs(v) < T(1.001) && dot(rel, face.normal) > T(0)) {
            all_behind = false;
            projected[i] = { (u + T(1)) * T(0.5), (T(1) - v) * T(0.5) }; // 0..1 range
        } else {
            projected[i] = { -T(1), -T(1) }; // outside
        }
    }
    if (all_behind) return;
    // Rasterize convex polygon using simple scanline (triangulate? Use bounding box and inside test)
    // We'll use a simple bounding box fill
    T min_u = T(1), max_u = T(0), min_v = T(1), max_v = T(0);
    for (auto& p : projected) {
        if (p[0] < T(0)) continue;
        min_u = std::min(min_u, p[0]); max_u = std::max(max_u, p[0]);
        min_v = std::min(min_v, p[1]); max_v = std::max(max_v, p[1]);
    }
    int x0 = std::max(0, static_cast<int>(min_u * resolution));
    int x1 = std::min(static_cast<int>(resolution)-1, static_cast<int>(max_u * resolution + 1));
    int y0 = std::max(0, static_cast<int>(min_v * resolution));
    int y1 = std::min(static_cast<int>(resolution)-1, static_cast<int>(max_v * resolution + 1));
    // Inside test using cross products
    auto is_inside = [&](T u, T v) -> bool {
        int n = static_cast<int>(projected.size());
        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;
            if (projected[i][0] < T(0) || projected[j][0] < T(0)) continue;
            T edge = (u - projected[i][0]) * (projected[j][1] - projected[i][1]) -
                     (v - projected[i][1]) * (projected[j][0] - projected[i][0]);
            if (edge < T(0)) return false;
        }
        return true;
    };
    T pixel_area = T(4) / (total_pixels); // hemicube pixel area (4 = face area * 1? Actually area per pixel = (2*half)^2 / res^2)
    for (int iy = y0; iy <= y1; ++iy) {
        for (int ix = x0; ix <= x1; ++ix) {
            T u = (ix + T(0.5)) / resolution;
            T v = (iy + T(0.5)) / resolution;
            if (is_inside(u, v)) {
                T df = delta_form_factor_pixel(face.center, face.normal, patch.centroid, pixel_area);
                std::size_t idx = iy * resolution + ix;
                if (idx < face_buffer.size()) face_buffer[idx] += df;
            }
        }
    }
}

// ============================================================
// 5. Compute form factors from one source patch to all patches
// ============================================================

template<typename T>
std::vector<T> compute_form_factors_hemicube(const radiosity_patch<T>& src,
                                             const std::vector<radiosity_patch<T>>& all_patches,
                                             std::uint32_t resolution = 64) {
    auto faces = build_hemicube(src, T(1));
    std::vector<std::vector<T>> face_buffers(5);
    T total_pixels = T(resolution * resolution) * T(5);
    for (int f = 0; f < 5; ++f) {
        face_buffers[f].assign(resolution * resolution, T(0));
    }
    // For each other patch, rasterize onto faces
    for (std::size_t pid = 0; pid < all_patches.size(); ++pid) {
        if (pid == src.id) continue;
        for (int f = 0; f < 5; ++f) {
            rasterize_patch_on_face(all_patches[pid], faces[f], resolution, face_buffers[f], total_pixels);
        }
    }
    // Accumulate per-patch form factors from the face buffers
    std::vector<T> form_factors(all_patches.size(), T(0));
    // For each face pixel, we need to know which patch contributed. We stored only the delta form factor sum.
    // So we need to re-project patches and identify patch per pixel. Simpler: we re-loop over patches.
    // We'll use item buffer approach: for each pixel, store patch id and form factor delta.
    // We'll reconstruct by re-evaluating each patch's bounding box and accumulating directly.
    // We'll re-call the same projection but now with a buffer that records patch id per pixel.
    // Let's create a new function that does both.
    // For brevity, we'll approximate by summing the form factors from the face buffers and then distributing
    // based on the fraction of visible area? Not accurate. We'll implement the proper item buffer.
    // We'll modify the above: we need an item buffer of size resolution*resolution*5 storing patch ID.
    // Let's create a new function `compute_form_factors_item_buffer`.
    // We'll rewrite inside this function.
    // Actually we can reuse the rasterize_patch_on_face but with an additional output array for patch ID.
    // We'll create a new rasterize that writes to two buffers: one for form factor delta, one for patch ID.
    // Since we cannot change the signature now, we'll inline a solution.
    // We'll just compute by re-running a simpler loop: for each face pixel, cast a ray from source? Too expensive.
    // The correct way: use item buffer. We'll implement it now.
    // We'll create a new class or function inside this one.
    // I'll implement the item buffer approach here.

    // Item buffer: for each hemicube pixel, store the patch ID that is visible (closest).
    std::vector<std::int32_t> item_buffer(resolution * resolution * 5, -1);
    std::vector<T> depth_buffer(resolution * resolution * 5, std::numeric_limits<T>::max());
    T pixel_area = T(4) / (T(resolution * resolution) * T(5)); // each pixel area in hemicube

    for (std::size_t pid = 0; pid < all_patches.size(); ++pid) {
        if (pid == src.id) continue;
        const auto& patch = all_patches[pid];
        for (int f = 0; f < 5; ++f) {
            const auto& face = faces[f];
            // project vertices
            std::vector<std::array<T,2>> proj(patch.vertices.size());
            T avg_depth = T(0);
            bool inside = true;
            for (std::size_t k = 0; k < patch.vertices.size(); ++k) {
                vector3<T> rel = patch.vertices[k] - face.center;
                T d = dot(rel, face.normal);
                if (d < T(0)) { inside = false; break; }
                T u = dot(rel, face.u_axis) / face.half_size;
                T v = dot(rel, face.v_axis) / face.half_size;
                proj[k] = { (u + T(1)) * T(0.5), (T(1) - v) * T(0.5) };
                avg_depth += d;
            }
            if (!inside) continue;
            // bounding box
            T min_u=T(1), max_u=T(0), min_v=T(1), max_v=T(0);
            for (auto& p : proj) { min_u=std::min(min_u,p[0]); max_u=std::max(max_u,p[0]); min_v=std::min(min_v,p[1]); max_v=std::max(max_v,p[1]); }
            int x0 = std::max(0, (int)(min_u*resolution)); int x1 = std::min((int)resolution-1, (int)(max_u*resolution+1));
            int y0 = std::max(0, (int)(min_v*resolution)); int y1 = std::min((int)resolution-1, (int)(max_v*resolution+1));
            T depth_avg = avg_depth / T(patch.vertices.size());
            auto is_inside_poly = [&](T uu, T vv) {
                int n = (int)proj.size();
                for (int i=0; i<n; ++i) {
                    int j=(i+1)%n;
                    T edge = (uu - proj[i][0]) * (proj[j][1] - proj[i][1]) - (vv - proj[i][1]) * (proj[j][0] - proj[i][0]);
                    if (edge < T(0)) return false;
                }
                return true;
            };
            for (int iy=y0; iy<=y1; ++iy) {
                for (int ix=x0; ix<=x1; ++ix) {
                    T uu = (ix+T(0.5))/resolution; T vv = (iy+T(0.5))/resolution;
                    if (is_inside_poly(uu, vv)) {
                        std::size_t pixel_idx = (f * resolution + iy) * resolution + ix;
                        if (depth_avg < depth_buffer[pixel_idx]) {
                            depth_buffer[pixel_idx] = depth_avg;
                            item_buffer[pixel_idx] = static_cast<std::int32_t>(pid);
                        }
                    }
                }
            }
        }
    }
    // Now compute form factors from item buffer
    for (std::size_t p = 0; p < item_buffer.size(); ++p) {
        if (item_buffer[p] >= 0) {
            std::uint32_t pid = static_cast<std::uint32_t>(item_buffer[p]);
            T df = delta_form_factor_pixel(
                faces[p / (resolution*resolution)].center,
                faces[p / (resolution*resolution)].normal,
                all_patches[pid].centroid, pixel_area);
            form_factors[pid] += df;
        }
    }
    // The source patch should have zero form factor to itself (we skipped it)
    return form_factors;
}

// ============================================================
// 6. Progressive radiosity solver
// ============================================================

template<typename T>
class progressive_radiosity_solver {
public:
    std::vector<radiosity_patch<T>> patches;
    T total_energy = T(0);
    std::uint32_t hemicube_res = 64;

    progressive_radiosity_solver() = default;

    void add_patch(const radiosity_patch<T>& p) {
        patches.push_back(p);
        patches.back().id = static_cast<std::uint32_t>(patches.size() - 1);
    }

    // Perform one shooting iteration
    bool shoot_iteration() {
        if (patches.empty()) return false;
        // Find patch with highest unshot radiosity
        std::size_t best = 0;
        T max_unshot = T(0);
        for (std::size_t i = 0; i < patches.size(); ++i) {
            if (patches[i].unshot_radiosity > max_unshot) {
                max_unshot = patches[i].unshot_radiosity;
                best = i;
            }
        }
        if (max_unshot < T(1e-8)) return false;
        // Compute form factors from this patch
        auto ff = compute_form_factors_hemicube(patches[best], patches, hemicube_res);
        T F_sum = T(0);
        for (std::size_t i = 0; i < ff.size(); ++i) F_sum += ff[i];
        T unshot = patches[best].unshot_radiosity;
        // Distribute unshot radiosity
        for (std::size_t i = 0; i < patches.size(); ++i) {
            if (i == best) continue;
            T delta = unshot * ff[i] * patches[i].reflectance;
            patches[i].radiosity += delta;
            patches[i].unshot_radiosity += delta;
        }
        // Reset unshot of shooter
        patches[best].unshot_radiosity = T(0);
        return true;
    }

    // Run solver
    void solve(int max_iterations = 500) {
        for (int iter = 0; iter < max_iterations; ++iter) {
            if (!shoot_iteration()) break;
        }
    }

    // Initialize unshot radiosity from emissivity
    void initialize_radiosity() {
        for (auto& p : patches) {
            p.radiosity = p.emissivity;
            p.unshot_radiosity = p.emissivity;
        }
    }

    // Get total radiosity per patch as a vector of scalars (for display)
    std::vector<T> get_radiosity_values() const {
        std::vector<T> vals(patches.size());
        for (std::size_t i = 0; i < patches.size(); ++i) vals[i] = patches[i].radiosity;
        return vals;
    }
};

// ============================================================
// 7. Spherical harmonic radiosity map (for directional emission)
// ============================================================

template<typename T>
struct sh_radiosity_map {
    std::vector<std::vector<T>> coefficients; // per patch, up to order L
    int max_l;
};

template<typename T>
sh_radiosity_map<T> compute_sh_radiosity(const std::vector<radiosity_patch<T>>& patches,
                                          const std::vector<std::vector<T>>& form_factors,
                                          int max_l = 2) {
    sh_radiosity_map<T> map;
    map.max_l = max_l;
    map.coefficients.resize(patches.size());
    std::size_t n = patches.size();
    for (std::size_t i = 0; i < n; ++i) {
        // For each patch, project the incoming radiosity distribution onto SH
        std::vector<std::vector<T>> coeffs(max_l + 1);
        for (int l = 0; l <= max_l; ++l) coeffs[l].assign(2 * l + 1, T(0));
        // Accumulate contributions from all other patches
        // For each neighbour j, we know form factor F_ij and radiosity B_j.
        // The incident direction from j to i is dir_ij = normalize(centroid_j - centroid_i).
        // We project a delta function at that direction onto SH.
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            if (form_factors[i][j] < T(1e-12)) continue;
            vector3<T> dir = normalize(patches[j].centroid - patches[i].centroid);
            T theta = std::acos(dir.y); // assuming Y up? need mapping. We'll use standard SH convention where Y=up.
            T phi = std::atan2(dir.z, dir.x);
            T weight = form_factors[i][j] * patches[j].radiosity;
            for (int l = 0; l <= max_l; ++l) {
                for (int m = -l; m <= l; ++m) {
                    T Ylm = sh::real_sh(l, m, theta, phi);
                    coeffs[l][m + l] += weight * Ylm;
                }
            }
        }
        // Normalize? The SH coefficients represent the directional radiosity distribution.
        map.coefficients[i] = coeffs;
    }
    return map;
}

// ============================================================
// 8. Progressive refinement using ambient term (classic radiosity)
// ============================================================

template<typename T>
void progressive_radiosity_with_ambient(std::vector<radiosity_patch<T>>& patches,
                                        int max_iterations = 500) {
    progressive_radiosity_solver<T> solver;
    for (const auto& p : patches) solver.add_patch(p);
    solver.initialize_radiosity();
    solver.hemicube_res = 64;
    solver.solve(max_iterations);
    // Copy back results
    for (std::size_t i = 0; i < patches.size(); ++i) {
        patches[i].radiosity = solver.patches[i].radiosity;
        patches[i].unshot_radiosity = solver.patches[i].unshot_radiosity;
    }
}

} // namespace radiosity
} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_RADIOSITY_H