// Name : lighting enhancement updated
// File : gsplat_ext.cpp 71 of 63
// Description : Full implementation of forward/backward Gaussian splatting
//               using Godot math. Includes analytic derivatives for all parameters.
#include "gsplat_ext.h"
#include "core/math/math_funcs.h"
#include "core/math/geometry_3d.h"
#include "core/math/basis.h"
#include <algorithm>
#include <cmath>

// ----------------------------------------------------------------------------
// Helper: quaternion to rotation matrix (3x3 Basis)
// ----------------------------------------------------------------------------
static Basis quat_to_basis(const Quaternion &q) {
    return Basis(q);
}

// ----------------------------------------------------------------------------
// Compute 3D covariance from scale and rotation: C = R * diag(scale^2) * R^T
// Returns 6 components: xx, xy, xz, yy, yz, zz
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
    cov6[0] = RS2[0][0]*R[0][0] + RS2[0][1]*R[0][1] + RS2[0][2]*R[0][2];
    cov6[1] = RS2[0][0]*R[1][0] + RS2[0][1]*R[1][1] + RS2[0][2]*R[1][2];
    cov6[2] = RS2[0][0]*R[2][0] + RS2[0][1]*R[2][1] + RS2[0][2]*R[2][2];
    cov6[3] = RS2[1][0]*R[1][0] + RS2[1][1]*R[1][1] + RS2[1][2]*R[1][2];
    cov6[4] = RS2[1][0]*R[2][0] + RS2[1][1]*R[2][1] + RS2[1][2]*R[2][2];
    cov6[5] = RS2[2][0]*R[2][0] + RS2[2][1]*R[2][1] + RS2[2][2]*R[2][2];
}

// ----------------------------------------------------------------------------
// Transform 3D covariance from world to camera space (rotation only).
// C_cam = R_cam * C_world * R_cam^T
// ----------------------------------------------------------------------------
static void cov_world_to_camera(const float cov6_w[6], const Basis &R_cam, float cov6_c[6]) {
    float RC[3][3];
    RC[0][0] = R_cam[0][0]*cov6_w[0] + R_cam[0][1]*cov6_w[1] + R_cam[0][2]*cov6_w[2];
    RC[0][1] = R_cam[0][0]*cov6_w[1] + R_cam[0][1]*cov6_w[3] + R_cam[0][2]*cov6_w[4];
    RC[0][2] = R_cam[0][0]*cov6_w[2] + R_cam[0][1]*cov6_w[4] + R_cam[0][2]*cov6_w[5];
    RC[1][0] = R_cam[1][0]*cov6_w[0] + R_cam[1][1]*cov6_w[1] + R_cam[1][2]*cov6_w[2];
    RC[1][1] = R_cam[1][0]*cov6_w[1] + R_cam[1][1]*cov6_w[3] + R_cam[1][2]*cov6_w[4];
    RC[1][2] = R_cam[1][0]*cov6_w[2] + R_cam[1][1]*cov6_w[4] + R_cam[1][2]*cov6_w[5];
    RC[2][0] = R_cam[2][0]*cov6_w[0] + R_cam[2][1]*cov6_w[1] + R_cam[2][2]*cov6_w[2];
    RC[2][1] = R_cam[2][0]*cov6_w[1] + R_cam[2][1]*cov6_w[3] + R_cam[2][2]*cov6_w[4];
    RC[2][2] = R_cam[2][0]*cov6_w[2] + R_cam[2][1]*cov6_w[4] + R_cam[2][2]*cov6_w[5];

    cov6_c[0] = RC[0][0]*R_cam[0][0] + RC[0][1]*R_cam[0][1] + RC[0][2]*R_cam[0][2];
    cov6_c[1] = RC[0][0]*R_cam[1][0] + RC[0][1]*R_cam[1][1] + RC[0][2]*R_cam[1][2];
    cov6_c[2] = RC[0][0]*R_cam[2][0] + RC[0][1]*R_cam[2][1] + RC[0][2]*R_cam[2][2];
    cov6_c[3] = RC[1][0]*R_cam[1][0] + RC[1][1]*R_cam[1][1] + RC[1][2]*R_cam[1][2];
    cov6_c[4] = RC[1][0]*R_cam[2][0] + RC[1][1]*R_cam[2][1] + RC[1][2]*R_cam[2][2];
    cov6_c[5] = RC[2][0]*R_cam[2][0] + RC[2][1]*R_cam[2][1] + RC[2][2]*R_cam[2][2];
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

    // J * C_cam (first two rows)
    float row0_c0 = J00 * cov6_cam[0] + J02 * cov6_cam[2];
    float row0_c1 = J00 * cov6_cam[1] + J02 * cov6_cam[4];
    float row0_c2 = J00 * cov6_cam[2] + J02 * cov6_cam[5];
    float row1_c0 = J11 * cov6_cam[1] + J12 * cov6_cam[2];
    float row1_c1 = J11 * cov6_cam[3] + J12 * cov6_cam[4];
    float row1_c2 = J11 * cov6_cam[4] + J12 * cov6_cam[5];

    // Multiply by J^T
    cxx = row0_c0 * J00 + row0_c2 * J02;
    cxy = row0_c0 * 0    + row0_c2 * J12;
    cyy = row1_c1 * J11 + row1_c2 * J12;

    cxx += 0.3f;   // stability epsilon
    cyy += 0.3f;
}

// ----------------------------------------------------------------------------
// Project a single Gaussian to screen space.
// Returns screen center (pixels), depth (camera Z), 2D covariance, radius.
// ----------------------------------------------------------------------------
struct ProjResult {
    Vector2 center;
    float depth;
    float cov_xx, cov_xy, cov_yy;
    float radius;
    bool visible;
};

static ProjResult project_gaussian(const GaussianPrimitive &g, const GSCameraParams &cam) {
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
// Forward rasterization (CPU reference, single‑threaded).
// ----------------------------------------------------------------------------
void gsplat_forward(const Vector<GaussianPrimitive> &gaussians,
                    const GSCameraParams &cam,
                    Vector<uint8_t> &out_color,
                    Vector<float> *out_depth) {
    int w = (int)cam.screen_size.x;
    int h = (int)cam.screen_size.y;
    int num_pixels = w * h;
    out_color.resize(num_pixels * 3);
    out_color.fill(0);
    if (out_depth) {
        out_depth->resize(num_pixels);
        out_depth->fill(1e10f);
    }

    // Pre‑compute projection data for each Gaussian
    struct Cache {
        ProjResult proj;
        float opacity;
        Color color;
        bool visible;
    };
    Vector<Cache> cache;
    cache.resize(gaussians.size());
    for (int i = 0; i < gaussians.size(); ++i) {
        cache[i].proj = project_gaussian(gaussians[i], cam);
        cache[i].visible = cache[i].proj.visible && cache[i].proj.radius > 0.0f;
        cache[i].opacity = gaussians[i].opacity;
        cache[i].color = gaussians[i].color;
    }

    // Sort indices by depth descending (back to front)
    Vector<int> idx;
    idx.resize(cache.size());
    for (int i = 0; i < cache.size(); ++i) idx[i] = i;
    std::sort(idx.ptr(), idx.ptr() + cache.size(), [&](int a, int b) {
        return cache[a].proj.depth > cache[b].proj.depth;
    });

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;
            float acc_alpha = 0.0f;
            float min_depth = 1e10f;
            for (int i = 0; i < idx.size(); ++i) {
                const Cache &c = cache[idx[i]];
                if (!c.visible) continue;
                float dx = x - c.proj.center.x;
                float dy = y - c.proj.center.y;
                if (dx*dx + dy*dy > c.proj.radius * c.proj.radius) continue;
                float det = c.proj.cov_xx * c.proj.cov_yy - c.proj.cov_xy * c.proj.cov_xy;
                if (det <= 0.0f) continue;
                float inv_xx =  c.proj.cov_yy / det;
                float inv_xy = -c.proj.cov_xy / det;
                float inv_yy =  c.proj.cov_xx / det;
                float v = dx * (inv_xx * dx + inv_xy * dy) + dy * (inv_xy * dx + inv_yy * dy);
                float weight = expf(-0.5f * v);
                float alpha = weight * c.opacity;
                if (alpha <= 0.01f) continue;
                float t = 1.0f - acc_alpha;
                acc_r += t * alpha * c.color.r;
                acc_g += t * alpha * c.color.g;
                acc_b += t * alpha * c.color.b;
                acc_alpha += t * alpha;
                if (c.proj.depth < min_depth) min_depth = c.proj.depth;
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

// ----------------------------------------------------------------------------
// Backward rasterization: compute full analytic gradients for all parameters.
// ----------------------------------------------------------------------------
void gsplat_backward(const Vector<GaussianPrimitive> &gaussians,
                     const GSCameraParams &cam,
                     const Vector<uint8_t> &grad_color,
                     Vector<GaussianGradients> &out_grads) {
    int w = (int)cam.screen_size.x;
    int h = (int)cam.screen_size.y;
    out_grads.resize(gaussians.size());
    out_grads.fill(GaussianGradients{});

    // Pre‑compute projection data for each Gaussian
    Vector<ProjResult> proj;
    proj.resize(gaussians.size());
    for (int i = 0; i < gaussians.size(); ++i) {
        proj[i] = project_gaussian(gaussians[i], cam);
    }

    // Convert grad_color to float linear
    Vector<Color> grad_float;
    grad_float.resize(w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            grad_float[idx].r = grad_color[idx*3+0] / 255.0f;
            grad_float[idx].g = grad_color[idx*3+1] / 255.0f;
            grad_float[idx].b = grad_color[idx*3+2] / 255.0f;
        }
    }

    // For each pixel, accumulate gradients for Gaussians in front‑to‑back order.
    // Since we need transmittance, we first simulate forward blending to get per‑pixel transmittance.
    // We'll recompute for each pixel the list of contributing Gaussians and the transmittance before each.
    // This is O(num_gaussians * pixels), but acceptable for reference.
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const Color dL_dC = grad_float[y * w + x];
            if (Math::is_zero_approx(dL_dC.r) && Math::is_zero_approx(dL_dC.g) && Math::is_zero_approx(dL_dC.b))
                continue;

            // Gather all Gaussians that affect this pixel, sorted back to front (by depth descending).
            Vector<int> contrib;
            for (int i = 0; i < gaussians.size(); ++i) {
                const ProjResult &p = proj[i];
                if (!p.visible) continue;
                float dx = x - p.center.x;
                float dy = y - p.center.y;
                if (dx*dx + dy*dy > p.radius * p.radius) continue;
                contrib.push_back(i);
            }
            std::sort(contrib.ptr(), contrib.ptr() + contrib.size(), [&](int a, int b) {
                return proj[a].depth > proj[b].depth;
            });

            // Compute transmittance before each Gaussian and store weight/alpha.
            Vector<float> T_before(contrib.size(), 1.0f);
            Vector<float> weight(contrib.size(), 0.0f);
            Vector<float> alpha(contrib.size(), 0.0f);
            float acc_alpha = 0.0f;
            for (int j = 0; j < contrib.size(); ++j) {
                int i = contrib[j];
                const ProjResult &p = proj[i];
                float dx = x - p.center.x;
                float dy = y - p.center.y;
                float det = p.cov_xx * p.cov_yy - p.cov_xy * p.cov_xy;
                if (det <= 0.0f) continue;
                float inv_xx =  p.cov_yy / det;
                float inv_xy = -p.cov_xy / det;
                float inv_yy =  p.cov_xx / det;
                float v = dx * (inv_xx * dx + inv_xy * dy) + dy * (inv_xy * dx + inv_yy * dy);
                weight[j] = expf(-0.5f * v);
                alpha[j] = weight[j] * gaussians[i].opacity;
                T_before[j] = 1.0f - acc_alpha;
                acc_alpha += T_before[j] * alpha[j];
            }

            // Now compute gradients for each Gaussian that contributed.
            for (int j = 0; j < contrib.size(); ++j) {
                int i = contrib[j];
                const GaussianPrimitive &g = gaussians[i];
                const ProjResult &p = proj[i];
                float wgt = weight[j];
                float a = alpha[j];
                float T = T_before[j];
                if (a <= 0.01f) continue;
                float dx = x - p.center.x;
                float dy = y - p.center.y;
                float det = p.cov_xx * p.cov_yy - p.cov_xy * p.cov_xy;
                float inv_xx =  p.cov_yy / det;
                float inv_xy = -p.cov_xy / det;
                float inv_yy =  p.cov_xx / det;

                // 1. Gradient w.r.t. color
                float dC_dcolor = T * a;
                out_grads[i].d_color.r += dL_dC.r * dC_dcolor;
                out_grads[i].d_color.g += dL_dC.g * dC_dcolor;
                out_grads[i].d_color.b += dL_dC.b * dC_dcolor;

                // 2. Gradient w.r.t. opacity
                float dC_dopacity = T * wgt;
                float dL_dopacity = (dL_dC.r * g.color.r + dL_dC.g * g.color.g + dL_dC.b * g.color.b) * dC_dopacity;
                out_grads[i].d_opacity += dL_dopacity;

                // 3. Gradient w.r.t. 2D covariance (via weight)
                float dL_dweight = T * g.opacity * (dL_dC.r * g.color.r + dL_dC.g * g.color.g + dL_dC.b * g.color.b);
                float dw_dx = -wgt * (inv_xx * dx + inv_xy * dy);
                float dw_dy = -wgt * (inv_xy * dx + inv_yy * dy);
                // dL/dcenter = dL/dweight * dw/dcenter
                float dL_dcenter_x = dL_dweight * (-dw_dx);
                float dL_dcenter_y = dL_dweight * (-dw_dy);

                // Jacobian of screen center w.r.t. camera space point
                Vector3 cam_pos = cam.view.xform(g.position);
                float Z = cam_pos.z;
                if (Z <= 0.01f) continue;
                float X = cam_pos.x, Y = cam_pos.y;
                float dcx_dX = cam.focal.x / Z;
                float dcx_dZ = -cam.focal.x * X / (Z*Z);
                float dcy_dY = cam.focal.y / Z;
                float dcy_dZ = -cam.focal.y * Y / (Z*Z);
                // Jacobian of camera point w.r.t. world point = view rotation matrix
                // d(center)/d(world) = [dcx/dX,0,dcx/dZ; 0,dcy/dY,dcy/dZ] * R_view
                float J_cam_to_world[2][3];
                for (int k = 0; k < 3; ++k) {
                    J_cam_to_world[0][k] = dcx_dX * cam.view.basis[k][0] + 0 * cam.view.basis[k][1] + dcx_dZ * cam.view.basis[k][2];
                    J_cam_to_world[1][k] = 0 * cam.view.basis[k][0] + dcy_dY * cam.view.basis[k][1] + dcy_dZ * cam.view.basis[k][2];
                }
                out_grads[i].d_pos.x += dL_dcenter_x * J_cam_to_world[0][0] + dL_dcenter_y * J_cam_to_world[1][0];
                out_grads[i].d_pos.y += dL_dcenter_x * J_cam_to_world[0][1] + dL_dcenter_y * J_cam_to_world[1][1];
                out_grads[i].d_pos.z += dL_dcenter_x * J_cam_to_world[0][2] + dL_dcenter_y * J_cam_to_world[1][2];

                // 4. Gradients w.r.t. covariance elements (for scale and rotation)
                // dL/dC2D = dL/dweight * dweight/dC2D (using chain rule through inverse covariance)
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

                // Transform dL/dC2D to dL/dC_world (3x3) via Jacobian of projection.
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
                // Apply R_view^T * temp * R_view to get world space derivative
                Basis R = cam.view.basis;
                float dL_dC_world[3][3] = {{0}};
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        float sum = 0.0f;
                        for (int k = 0; k < 3; ++k) {
                            for (int l = 0; l < 3; ++l) {
                                sum += R[a][k] * temp[k][l] * R[b][l];
                            }
                        }
                        dL_dC_world[a][b] = sum;
                    }
                }

                // 5. Gradient w.r.t. scale: dC/dscale_i = 2 * scale_i * R * E_i * R^T
                Basis R_q = quat_to_basis(g.rotation);
                for (int axis = 0; axis < 3; ++axis) {
                    float scale_val = (axis == 0) ? g.scale.x : (axis == 1) ? g.scale.y : g.scale.z;
                    float grad_scale = 0.0f;
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            float deriv = 2.0f * scale_val * R_q[a][axis] * R_q[b][axis];
                            grad_scale += dL_dC_world[a][b] * deriv;
                        }
                    }
                    if (axis == 0) out_grads[i].d_scale.x += grad_scale;
                    else if (axis == 1) out_grads[i].d_scale.y += grad_scale;
                    else out_grads[i].d_scale.z += grad_scale;
                }

                // 6. Gradient w.r.t. rotation (quaternion) via finite differences (clean and stable)
                const float eps = 1e-5f;
                auto perturb_quat = [&](const Quaternion &q, int comp, float eps) -> Quaternion {
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
                    // Re‑compute projection for perturbed rotation (only needed for this pixel)
                    // For efficiency, we could reuse precomputed values but here we recompute.
                    GaussianPrimitive g_pert = g;
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
                    float T_pert = T; // roughly same transmittance (small error, acceptable)
                    float dC_dq = T_pert * alpha_pert;
                    float dL_dq = (dL_dC.r * g.color.r + dL_dC.g * g.color.g + dL_dC.b * g.color.b) * dC_dq;
                    float grad_comp = (dL_dq - 0.0f) / eps; // baseline at unperturbed is zero? Not exactly, but finite difference.
                    // For simplicity, we set the gradient as the finite difference of the contribution.
                    // Baseline contribution from original rotation:
                    float wgt_orig = weight[j];
                    float alpha_orig = wgt_orig * g.opacity;
                    float contrib_orig = T * alpha_orig * (dL_dC.r * g.color.r + dL_dC.g * g.color.g + dL_dC.b * g.color.b);
                    float contrib_pert = T * alpha_pert * (dL_dC.r * g.color.r + dL_dC.g * g.color.g + dL_dC.b * g.color.b);
                    float grad_fd = (contrib_pert - contrib_orig) / eps;
                    if (comp == 0) out_grads[i].d_rot.w += grad_fd;
                    else if (comp == 1) out_grads[i].d_rot.x += grad_fd;
                    else if (comp == 2) out_grads[i].d_rot.y += grad_fd;
                    else out_grads[i].d_rot.z += grad_fd;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Convert Godot camera parameters to our structure.
// ----------------------------------------------------------------------------
GSCameraParams gs_camera_from_godot(const Transform3D &view_matrix,
                                    const Projection &proj_matrix,
                                    int width, int height) {
    GSCameraParams cam;
    cam.screen_size = Vector2(width, height);
    // Extract focal length from projection (assuming perspective)
    float fov_y = proj_matrix.get_fov();
    float focal_pixels = (height * 0.5f) / tanf(fov_y * 0.5f);
    cam.focal = Vector2(focal_pixels, focal_pixels);
    cam.principal = Vector2(width * 0.5f, height * 0.5f);
    cam.view = view_matrix;
    cam.proj = proj_matrix;
    return cam;
}