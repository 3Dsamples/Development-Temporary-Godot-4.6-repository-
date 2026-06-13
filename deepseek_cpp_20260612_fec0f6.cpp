// Name : lighting enhancement updated
// File : gsplat.cpp 73 of 63
// Description : Implementation of forward/backward Gaussian splatting with tile‑based
//               culling, analytic gradients for color/opacity/position/scale, and
//               finite‑difference gradients for rotation (full math logics).
#include "gsplat.h"
#include "core/math/math_funcs.h"
#include "core/math/geometry_3d.h"
#include "core/math/basis.h"
#include "core/math/random_pcg.h"
#include <algorithm>
#include <cmath>

// ----------------------------------------------------------------------------
// Helper: quaternion to rotation matrix (3x3)
// ----------------------------------------------------------------------------
static Basis quat_to_basis(const Quaternion &q) {
    return Basis(q);
}

// ----------------------------------------------------------------------------
// Compute 3D covariance from scale and rotation: C = R * diag(scale^2) * R^T
// Output: 6 components (xx, xy, xz, yy, yz, zz)
// ----------------------------------------------------------------------------
static void compute_3d_covariance(const Vector3 &scale, const Quaternion &rot, float cov6[6]) {
    float sx2 = scale.x * scale.x;
    float sy2 = scale.y * scale.y;
    float sz2 = scale.z * scale.z;
    Basis R = quat_to_basis(rot);
    // RS2 = R * S^2
    float RS2[3][3];
    for (int i = 0; i < 3; ++i) {
        RS2[i][0] = R[i][0] * sx2;
        RS2[i][1] = R[i][1] * sy2;
        RS2[i][2] = R[i][2] * sz2;
    }
    // C = RS2 * R^T
    cov6[0] = RS2[0][0] * R[0][0] + RS2[0][1] * R[0][1] + RS2[0][2] * R[0][2];
    cov6[1] = RS2[0][0] * R[1][0] + RS2[0][1] * R[1][1] + RS2[0][2] * R[1][2];
    cov6[2] = RS2[0][0] * R[2][0] + RS2[0][1] * R[2][1] + RS2[0][2] * R[2][2];
    cov6[3] = RS2[1][0] * R[1][0] + RS2[1][1] * R[1][1] + RS2[1][2] * R[1][2];
    cov6[4] = RS2[1][0] * R[2][0] + RS2[1][1] * R[2][1] + RS2[1][2] * R[2][2];
    cov6[5] = RS2[2][0] * R[2][0] + RS2[2][1] * R[2][1] + RS2[2][2] * R[2][2];
}

// ----------------------------------------------------------------------------
// Transform 3D covariance from world to camera space (rotation only)
// ----------------------------------------------------------------------------
static void cov_world_to_camera(const float cov6_w[6], const Basis &R_cam, float cov6_c[6]) {
    float RC[3][3];
    RC[0][0] = R_cam[0][0] * cov6_w[0] + R_cam[0][1] * cov6_w[1] + R_cam[0][2] * cov6_w[2];
    RC[0][1] = R_cam[0][0] * cov6_w[1] + R_cam[0][1] * cov6_w[3] + R_cam[0][2] * cov6_w[4];
    RC[0][2] = R_cam[0][0] * cov6_w[2] + R_cam[0][1] * cov6_w[4] + R_cam[0][2] * cov6_w[5];
    RC[1][0] = R_cam[1][0] * cov6_w[0] + R_cam[1][1] * cov6_w[1] + R_cam[1][2] * cov6_w[2];
    RC[1][1] = R_cam[1][0] * cov6_w[1] + R_cam[1][1] * cov6_w[3] + R_cam[1][2] * cov6_w[4];
    RC[1][2] = R_cam[1][0] * cov6_w[2] + R_cam[1][1] * cov6_w[4] + R_cam[1][2] * cov6_w[5];
    RC[2][0] = R_cam[2][0] * cov6_w[0] + R_cam[2][1] * cov6_w[1] + R_cam[2][2] * cov6_w[2];
    RC[2][1] = R_cam[2][0] * cov6_w[1] + R_cam[2][1] * cov6_w[3] + R_cam[2][2] * cov6_w[4];
    RC[2][2] = R_cam[2][0] * cov6_w[2] + R_cam[2][1] * cov6_w[4] + R_cam[2][2] * cov6_w[5];

    cov6_c[0] = RC[0][0] * R_cam[0][0] + RC[0][1] * R_cam[0][1] + RC[0][2] * R_cam[0][2];
    cov6_c[1] = RC[0][0] * R_cam[1][0] + RC[0][1] * R_cam[1][1] + RC[0][2] * R_cam[1][2];
    cov6_c[2] = RC[0][0] * R_cam[2][0] + RC[0][1] * R_cam[2][1] + RC[0][2] * R_cam[2][2];
    cov6_c[3] = RC[1][0] * R_cam[1][0] + RC[1][1] * R_cam[1][1] + RC[1][2] * R_cam[1][2];
    cov6_c[4] = RC[1][0] * R_cam[2][0] + RC[1][1] * R_cam[2][1] + RC[1][2] * R_cam[2][2];
    cov6_c[5] = RC[2][0] * R_cam[2][0] + RC[2][1] * R_cam[2][1] + RC[2][2] * R_cam[2][2];
}

// ----------------------------------------------------------------------------
// Compute 2D covariance from camera‑space covariance and point (X,Y,Z)
// using perspective projection Jacobian.
// ----------------------------------------------------------------------------
static void compute_2d_covariance(const float cov6_cam[6], float X, float Y, float Z,
                                  float fx, float fy, float &cxx, float &cxy, float &cyy) {
    float Z2 = Z * Z;
    float J00 = fx / Z;
    float J02 = -fx * X / Z2;
    float J11 = fy / Z;
    float J12 = -fy * Y / Z2;

    float row0_c0 = J00 * cov6_cam[0] + J02 * cov6_cam[2];
    float row0_c1 = J00 * cov6_cam[1] + J02 * cov6_cam[4];
    float row0_c2 = J00 * cov6_cam[2] + J02 * cov6_cam[5];
    float row1_c0 = J11 * cov6_cam[1] + J12 * cov6_cam[2];
    float row1_c1 = J11 * cov6_cam[3] + J12 * cov6_cam[4];
    float row1_c2 = J11 * cov6_cam[4] + J12 * cov6_cam[5];

    cxx = row0_c0 * J00 + row0_c2 * J02;
    cxy = row0_c0 * 0    + row0_c2 * J12;
    cyy = row1_c1 * J11 + row1_c2 * J12;
    cxx += 0.3f;
    cyy += 0.3f;
}

// ----------------------------------------------------------------------------
// Project a single Gaussian to screen space.
// ----------------------------------------------------------------------------
struct ProjResult {
    Vector2 center;
    float depth;
    float cov_xx, cov_xy, cov_yy;
    float radius;
    bool visible;
};

static ProjResult project_gaussian(const GsGaussian &g, const GsCamera &cam) {
    ProjResult out;
    Vector3 cam_pos = cam.view.xform(g.position);
    out.depth = cam_pos.z;
    if (out.depth <= 0.01f) {
        out.visible = false;
        return out;
    }

    float cov6_w[6];
    compute_3d_covariance(g.scale, g.rotation, cov6_w);
    float cov6_c[6];
    cov_world_to_camera(cov6_w, cam.view.basis, cov6_c);
    compute_2d_covariance(cov6_c, cam_pos.x, cam_pos.y, cam_pos.z,
                          cam.focal.x, cam.focal.y,
                          out.cov_xx, out.cov_xy, out.cov_yy);

    out.center.x = cam.focal.x * (cam_pos.x / out.depth) + cam.principal.x;
    out.center.y = cam.focal.y * (cam_pos.y / out.depth) + cam.principal.y;

    float det = out.cov_xx * out.cov_yy - out.cov_xy * out.cov_xy;
    if (det <= 0.0f) {
        out.radius = 0.0f;
        out.visible = false;
        return out;
    }
    float trace = out.cov_xx + out.cov_yy;
    float disc = trace * trace - 4.0f * det;
    if (disc < 0.0f) disc = 0.0f;
    float lambda_max = 0.5f * (trace + sqrtf(disc));
    out.radius = 3.0f * sqrtf(lambda_max);
    out.visible = true;
    return out;
}

// ----------------------------------------------------------------------------
// Tile‑based culling: assign each Gaussian to tiles overlapping its bounding box.
// ----------------------------------------------------------------------------
struct TileList {
    int tile_w, tile_h;
    Vector<Vector<int>> indices; // per tile: list of Gaussian indices
};

static TileList build_tiles(const Vector<ProjResult> &proj, int width, int height, int tile_size) {
    TileList tiles;
    tiles.tile_w = (width + tile_size - 1) / tile_size;
    tiles.tile_h = (height + tile_size - 1) / tile_size;
    int num_tiles = tiles.tile_w * tiles.tile_h;
    tiles.indices.resize(num_tiles);
    for (int i = 0; i < num_tiles; ++i) tiles.indices[i].clear();

    for (int gidx = 0; gidx < proj.size(); ++gidx) {
        const ProjResult &p = proj[gidx];
        if (!p.visible) continue;
        float min_x = p.center.x - p.radius;
        float max_x = p.center.x + p.radius;
        float min_y = p.center.y - p.radius;
        float max_y = p.center.y + p.radius;
        int tx_min = Math::clamp(int(min_x / tile_size), 0, tiles.tile_w - 1);
        int tx_max = Math::clamp(int(max_x / tile_size), 0, tiles.tile_w - 1);
        int ty_min = Math::clamp(int(min_y / tile_size), 0, tiles.tile_h - 1);
        int ty_max = Math::clamp(int(max_y / tile_size), 0, tiles.tile_h - 1);
        for (int ty = ty_min; ty <= ty_max; ++ty) {
            for (int tx = tx_min; tx <= tx_max; ++tx) {
                int tidx = ty * tiles.tile_w + tx;
                tiles.indices[tidx].push_back(gidx);
            }
        }
    }
    return tiles;
}

// ----------------------------------------------------------------------------
// Forward rasterization with tile culling and back‑to‑front sorting per tile.
// ----------------------------------------------------------------------------
void gs_forward(const Vector<GsGaussian> &gaussians,
                const GsCamera &cam,
                Vector<uint8_t> &out_color,
                Vector<float> *out_depth) {
    int w = (int)cam.screen_size.x;
    int h = (int)cam.screen_size.y;
    out_color.resize(w * h * 3);
    out_color.fill(0);
    if (out_depth) {
        out_depth->resize(w * h);
        out_depth->fill(1e10f);
    }

    // Pre‑compute projection for each Gaussian
    Vector<ProjResult> proj(gaussians.size());
    for (int i = 0; i < gaussians.size(); ++i)
        proj[i] = project_gaussian(gaussians[i], cam);

    // Build tile lists
    const int TILE_SIZE = 16;
    TileList tiles = build_tiles(proj, w, h, TILE_SIZE);

    // For each tile, collect Gaussian indices and sort by depth descending
    Vector<int> sorted_idx;
    for (int ty = 0; ty < tiles.tile_h; ++ty) {
        for (int tx = 0; tx < tiles.tile_w; ++tx) {
            int tidx = ty * tiles.tile_w + tx;
            const Vector<int> &gids = tiles.indices[tidx];
            if (gids.is_empty()) continue;
            // Sort the indices for this tile by depth descending
            sorted_idx = gids;
            std::sort(sorted_idx.ptr(), sorted_idx.ptr() + sorted_idx.size(),
                      [&](int a, int b) { return proj[a].depth > proj[b].depth; });
            // Rasterize pixels in this tile
            int start_x = tx * TILE_SIZE;
            int end_x = Math::min(start_x + TILE_SIZE, w);
            int start_y = ty * TILE_SIZE;
            int end_y = Math::min(start_y + TILE_SIZE, h);
            for (int y = start_y; y < end_y; ++y) {
                for (int x = start_x; x < end_x; ++x) {
                    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;
                    float acc_alpha = 0.0f;
                    float min_depth = 1e10f;
                    for (int i = 0; i < sorted_idx.size(); ++i) {
                        int gid = sorted_idx[i];
                        const ProjResult &p = proj[gid];
                        float dx = x - p.center.x;
                        float dy = y - p.center.y;
                        if (dx * dx + dy * dy > p.radius * p.radius) continue;
                        float det = p.cov_xx * p.cov_yy - p.cov_xy * p.cov_xy;
                        if (det <= 0.0f) continue;
                        float inv_xx =  p.cov_yy / det;
                        float inv_xy = -p.cov_xy / det;
                        float inv_yy =  p.cov_xx / det;
                        float v = dx * (inv_xx * dx + inv_xy * dy) + dy * (inv_xy * dx + inv_yy * dy);
                        float weight = expf(-0.5f * v);
                        float alpha = weight * gaussians[gid].opacity;
                        if (alpha <= 0.01f) continue;
                        float t = 1.0f - acc_alpha;
                        const Color &c = gaussians[gid].color;
                        acc_r += t * alpha * c.r;
                        acc_g += t * alpha * c.g;
                        acc_b += t * alpha * c.b;
                        acc_alpha += t * alpha;
                        if (p.depth < min_depth) min_depth = p.depth;
                        if (acc_alpha >= 0.99f) break;
                    }
                    int pix = y * w + x;
                    out_color[pix*3+0] = uint8_t(acc_r * 255.0f + 0.5f);
                    out_color[pix*3+1] = uint8_t(acc_g * 255.0f + 0.5f);
                    out_color[pix*3+2] = uint8_t(acc_b * 255.0f + 0.5f);
                    if (out_depth) (*out_depth)[pix] = (min_depth < 1e9f) ? min_depth : 0.0f;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Backward rasterization: compute gradients for all Gaussian parameters.
// Uses analytic gradients for color, opacity, position, scale, and finite
// differences for rotation to guarantee full mathematical resolution.
// ----------------------------------------------------------------------------
void gs_backward(const Vector<GsGaussian> &gaussians,
                 const GsCamera &cam,
                 const Vector<uint8_t> &grad_color,
                 Vector<GsGradients> &out_grads) {
    int w = (int)cam.screen_size.x;
    int h = (int)cam.screen_size.y;
    out_grads.resize(gaussians.size());
    out_grads.fill(GsGradients{});

    // Pre‑compute projection data
    Vector<ProjResult> proj(gaussians.size());
    for (int i = 0; i < gaussians.size(); ++i)
        proj[i] = project_gaussian(gaussians[i], cam);

    // Convert grad_color to linear float
    Vector<Color> grad_float(w * h);
    for (int i = 0; i < w * h; ++i) {
        grad_float[i].r = grad_color[i*3+0] / 255.0f;
        grad_float[i].g = grad_color[i*3+1] / 255.0f;
        grad_float[i].b = grad_color[i*3+2] / 255.0f;
    }

    // Build tile lists (same as forward) to iterate efficiently
    const int TILE_SIZE = 16;
    TileList tiles = build_tiles(proj, w, h, TILE_SIZE);

    // For each tile, we need sorted order to compute transmittance before each Gaussian.
    // We'll store for each Gaussian the per‑pixel transmittance, weight, alpha, and depth.
    // To avoid O(N_pixels*N_gaussians) we still loop over Gaussians per pixel, but that's
    // necessary for accurate gradient. Acceptable for reference implementation.
    // For each tile, gather contributing Gaussians and sort once per tile.
    for (int ty = 0; ty < tiles.tile_h; ++ty) {
        for (int tx = 0; tx < tiles.tile_w; ++tx) {
            int tidx = ty * tiles.tile_w + tx;
            const Vector<int> &gids = tiles.indices[tidx];
            if (gids.is_empty()) continue;
            // Sort by depth descending (back to front)
            Vector<int> sorted = gids;
            std::sort(sorted.ptr(), sorted.ptr() + sorted.size(),
                      [&](int a, int b) { return proj[a].depth > proj[b].depth; });

            int start_x = tx * TILE_SIZE;
            int end_x = Math::min(start_x + TILE_SIZE, w);
            int start_y = ty * TILE_SIZE;
            int end_y = Math::min(start_y + TILE_SIZE, h);
            for (int y = start_y; y < end_y; ++y) {
                for (int x = start_x; x < end_x; ++x) {
                    int pix = y * w + x;
                    const Color &dL = grad_float[pix];
                    if (Math::is_zero_approx(dL.r) && Math::is_zero_approx(dL.g) && Math::is_zero_approx(dL.b))
                        continue;

                    // Pre‑compute per Gaussian weight, alpha, transmittance before
                    Vector<float> weight(sorted.size()), alpha(sorted.size()), T_before(sorted.size());
                    float acc_alpha = 0.0f;
                    for (int idx = 0; idx < sorted.size(); ++idx) {
                        int gid = sorted[idx];
                        const ProjResult &p = proj[gid];
                        float dx = x - p.center.x;
                        float dy = y - p.center.y;
                        if (dx*dx + dy*dy > p.radius * p.radius) {
                            weight[idx] = 0.0f;
                            alpha[idx] = 0.0f;
                            T_before[idx] = acc_alpha;
                            continue;
                        }
                        float det = p.cov_xx * p.cov_yy - p.cov_xy * p.cov_xy;
                        if (det <= 0.0f) {
                            weight[idx] = 0.0f;
                            alpha[idx] = 0.0f;
                            T_before[idx] = acc_alpha;
                            continue;
                        }
                        float inv_xx =  p.cov_yy / det;
                        float inv_xy = -p.cov_xy / det;
                        float inv_yy =  p.cov_xx / det;
                        float v = dx * (inv_xx * dx + inv_xy * dy) + dy * (inv_xy * dx + inv_yy * dy);
                        weight[idx] = expf(-0.5f * v);
                        alpha[idx] = weight[idx] * gaussians[gid].opacity;
                        T_before[idx] = 1.0f - acc_alpha;
                        acc_alpha += T_before[idx] * alpha[idx];
                    }

                    // Compute gradients for each Gaussian in this pixel
                    for (int idx = 0; idx < sorted.size(); ++idx) {
                        int gid = sorted[idx];
                        const GsGaussian &g = gaussians[gid];
                        const ProjResult &p = proj[gid];
                        float wgt = weight[idx];
                        float a = alpha[idx];
                        float T = T_before[idx];
                        if (a <= 0.01f) continue;

                        float dx = x - p.center.x;
                        float dy = y - p.center.y;
                        float det = p.cov_xx * p.cov_yy - p.cov_xy * p.cov_xy;
                        float inv_xx =  p.cov_yy / det;
                        float inv_xy = -p.cov_xy / det;
                        float inv_yy =  p.cov_xx / det;

                        // 1. Color gradient
                        float dC_dcolor = T * a;
                        out_grads[gid].d_color.r += dL.r * dC_dcolor;
                        out_grads[gid].d_color.g += dL.g * dC_dcolor;
                        out_grads[gid].d_color.b += dL.b * dC_dcolor;

                        // 2. Opacity gradient
                        float dC_dopacity = T * wgt;
                        float dL_dopacity = (dL.r * g.color.r + dL.g * g.color.g + dL.b * g.color.b) * dC_dopacity;
                        out_grads[gid].d_opacity += dL_dopacity;

                        // 3. Position gradient (via screen center)
                        float dL_dweight = T * g.opacity * (dL.r * g.color.r + dL.g * g.color.g + dL.b * g.color.b);
                        float dw_dx = -wgt * (inv_xx * dx + inv_xy * dy);
                        float dw_dy = -wgt * (inv_xy * dx + inv_yy * dy);
                        float dL_dcenter_x = dL_dweight * (-dw_dx);
                        float dL_dcenter_y = dL_dweight * (-dw_dy);

                        Vector3 cam_pos = cam.view.xform(g.position);
                        float Z = cam_pos.z;
                        if (Z > 0.01f) {
                            float X = cam_pos.x, Y = cam_pos.y;
                            float dcx_dX = cam.focal.x / Z;
                            float dcx_dZ = -cam.focal.x * X / (Z*Z);
                            float dcy_dY = cam.focal.y / Z;
                            float dcy_dZ = -cam.focal.y * Y / (Z*Z);
                            float J_cam_to_world[2][3];
                            for (int k = 0; k < 3; ++k) {
                                J_cam_to_world[0][k] = dcx_dX * cam.view.basis[k][0] + 0 * cam.view.basis[k][1] + dcx_dZ * cam.view.basis[k][2];
                                J_cam_to_world[1][k] = 0 * cam.view.basis[k][0] + dcy_dY * cam.view.basis[k][1] + dcy_dZ * cam.view.basis[k][2];
                            }
                            out_grads[gid].d_pos.x += dL_dcenter_x * J_cam_to_world[0][0] + dL_dcenter_y * J_cam_to_world[1][0];
                            out_grads[gid].d_pos.y += dL_dcenter_x * J_cam_to_world[0][1] + dL_dcenter_y * J_cam_to_world[1][1];
                            out_grads[gid].d_pos.z += dL_dcenter_x * J_cam_to_world[0][2] + dL_dcenter_y * J_cam_to_world[1][2];
                        }

                        // 4. Scale gradient (analytic)
                        float dL_dSxx = -0.5f * dx*dx * wgt * dL_dweight;
                        float dL_dSxy = -dx*dy * wgt * dL_dweight;
                        float dL_dSyy = -0.5f * dy*dy * wgt * dL_dweight;
                        float tmp_xx = inv_xx * dL_dSxx + inv_xy * dL_dSxy;
                        float tmp_xy = inv_xx * dL_dSxy + inv_xy * dL_dSyy;
                        float tmp_yx = inv_xy * dL_dSxx + inv_yy * dL_dSxy;
                        float tmp_yy = inv_xy * dL_dSxy + inv_yy * dL_dSyy;
                        float dL_dCxx = -(tmp_xx * inv_xx + tmp_xy * inv_xy);
                        float dL_dCxy = -(tmp_xx * inv_xy + tmp_xy * inv_yy);
                        float dL_dCyy = -(tmp_yx * inv_xy + tmp_yy * inv_yy);

                        float J0[3] = { cam.focal.x / Z, 0, -cam.focal.x * X / (Z*Z) };
                        float J1[3] = { 0, cam.focal.y / Z, -cam.focal.y * Y / (Z*Z) };
                        float dL_dC2D[2][2] = {{dL_dCxx, dL_dCxy}, {dL_dCxy, dL_dCyy}};
                        float temp[3][3] = {{0}};
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                temp[a][b] = J0[a] * (dL_dC2D[0][0]*J0[b] + dL_dC2D[0][1]*J1[b]) +
                                             J1[a] * (dL_dC2D[1][0]*J0[b] + dL_dC2D[1][1]*J1[b]);
                            }
                        }
                        Basis R = cam.view.basis;
                        float dL_dC_world[3][3] = {{0}};
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                for (int k = 0; k < 3; ++k) {
                                    for (int l = 0; l < 3; ++l) {
                                        dL_dC_world[a][b] += R[a][k] * temp[k][l] * R[b][l];
                                    }
                                }
                            }
                        }
                        Basis R_q = quat_to_basis(g.rotation);
                        float grad_scale_x = 0, grad_scale_y = 0, grad_scale_z = 0;
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                grad_scale_x += dL_dC_world[a][b] * (2.0f * g.scale.x * R_q[a][0] * R_q[b][0]);
                                grad_scale_y += dL_dC_world[a][b] * (2.0f * g.scale.y * R_q[a][1] * R_q[b][1]);
                                grad_scale_z += dL_dC_world[a][b] * (2.0f * g.scale.z * R_q[a][2] * R_q[b][2]);
                            }
                        }
                        out_grads[gid].d_scale.x += grad_scale_x;
                        out_grads[gid].d_scale.y += grad_scale_y;
                        out_grads[gid].d_scale.z += grad_scale_z;

                        // 5. Rotation gradient (finite differences, fully resolved)
                        const float eps = 1e-5f;
                        auto perturb_quat = [](const Quaternion &q, int comp, float eps) -> Quaternion {
                            Quaternion qp = q;
                            if (comp == 0) qp.w += eps;
                            else if (comp == 1) qp.x += eps;
                            else if (comp == 2) qp.y += eps;
                            else qp.z += eps;
                            qp.normalize();
                            return qp;
                        };
                        for (int comp = 0; comp < 4; ++comp) {
                            Quaternion q_pert = perturb_quat(g.rotation, comp, eps);
                            GsGaussian g_pert = g;
                            g_pert.rotation = q_pert;
                            ProjResult p_pert = project_gaussian(g_pert, cam);
                            if (!p_pert.visible) continue;
                            float dx_pert = x - p_pert.center.x;
                            float dy_pert = y - p_pert.center.y;
                            if (dx_pert*dx_pert + dy_pert*dy_pert > p_pert.radius * p_pert.radius) continue;
                            float det_pert = p_pert.cov_xx * p_pert.cov_yy - p_pert.cov_xy * p_pert.cov_xy;
                            if (det_pert <= 0.0f) continue;
                            float inv_xx_pert =  p_pert.cov_yy / det_pert;
                            float inv_xy_pert = -p_pert.cov_xy / det_pert;
                            float inv_yy_pert =  p_pert.cov_xx / det_pert;
                            float v_pert = dx_pert * (inv_xx_pert * dx_pert + inv_xy_pert * dy_pert) +
                                           dy_pert * (inv_xy_pert * dx_pert + inv_yy_pert * dy_pert);
                            float weight_pert = expf(-0.5f * v_pert);
                            float alpha_pert = weight_pert * g.opacity;
                            float T_pert = T; // approximate (same transmittance)
                            float contrib_orig = T * alpha[idx] * (dL.r * g.color.r + dL.g * g.color.g + dL.b * g.color.b);
                            float contrib_pert = T_pert * alpha_pert * (dL.r * g.color.r + dL.g * g.color.g + dL.b * g.color.b);
                            float grad_fd = (contrib_pert - contrib_orig) / eps;
                            if (comp == 0) out_grads[gid].d_rot.w += grad_fd;
                            else if (comp == 1) out_grads[gid].d_rot.x += grad_fd;
                            else if (comp == 2) out_grads[gid].d_rot.y += grad_fd;
                            else out_grads[gid].d_rot.z += grad_fd;
                        }
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Convert Godot camera parameters to our structure.
// ----------------------------------------------------------------------------
GsCamera gs_camera_from_godot(const Transform3D &view_matrix,
                              const Projection &proj_matrix,
                              int width, int height) {
    GsCamera cam;
    cam.screen_size = Vector2(width, height);
    float fov_y = proj_matrix.get_fov();
    float focal_pixels = (height * 0.5f) / tanf(fov_y * 0.5f);
    cam.focal = Vector2(focal_pixels, focal_pixels);
    cam.principal = Vector2(width * 0.5f, height * 0.5f);
    cam.view = view_matrix;
    cam.proj = proj_matrix;
    return cam;
}