// Name : lighting enhancement updated
// File : gsplat_rasterizer.cpp 64 of 63
// Description : Full CPU implementation of tile‑based Gaussian splatting rasterization.
//               Computes 2D covariance, sorts Gaussians back‑to‑front, and blends.
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

// ----------------------------------------------------------------------------
// Basic vector and matrix types (matching Godot's math, but self‑contained)
// ----------------------------------------------------------------------------
struct Vec3 { float x, y, z; };
struct Vec4 { float x, y, z, w; };
struct Mat3 {
    float m[9];
    Vec3 operator*(const Vec3 &v) const {
        return Vec3{
            m[0]*v.x + m[1]*v.y + m[2]*v.z,
            m[3]*v.x + m[4]*v.y + m[5]*v.z,
            m[6]*v.x + m[7]*v.y + m[8]*v.z
        };
    }
};
struct Vec2 { float x, y; };
struct Quat { float w, x, y, z; };

// ----------------------------------------------------------------------------
// Gaussian primitive data (CPU)
// ----------------------------------------------------------------------------
struct Gaussian {
    Vec3 pos;
    Vec3 scale;      // ellipsoid radii (sx, sy, sz)
    Quat rot;        // rotation quaternion (w, x, y, z), normalized
    Vec3 color;      // RGB base color (ignoring SH for simplicity)
    float opacity;
    // Spherical harmonics (9 per channel) – not used here, would expand color.
};

struct Camera {
    Mat3 view_rot;   // rotation part of view matrix (world → camera)
    Vec3 view_trans; // translation part
    float focal_x, focal_y;
    float width, height;
    Mat3 proj;       // perspective projection matrix (4x4 stored as 3x4? we'll use separate)
    float near, far;
};

// ----------------------------------------------------------------------------
// Quaternion to rotation matrix (3x3)
// ----------------------------------------------------------------------------
static Mat3 quat_to_mat3(const Quat &q) {
    float w = q.w, x = q.x, y = q.y, z = q.z;
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;
    Mat3 m;
    m.m[0] = 1 - 2*(yy + zz); m.m[1] = 2*(xy - wz);   m.m[2] = 2*(xz + wy);
    m.m[3] = 2*(xy + wz);    m.m[4] = 1 - 2*(xx + zz); m.m[5] = 2*(yz - wx);
    m.m[6] = 2*(xz - wy);    m.m[7] = 2*(yz + wx);    m.m[8] = 1 - 2*(xx + yy);
    return m;
}

// ----------------------------------------------------------------------------
// Transform point by 3x3 matrix
// ----------------------------------------------------------------------------
static Vec3 mat3_mul_vec3(const Mat3 &m, const Vec3 &v) {
    return Vec3{
        m.m[0]*v.x + m.m[1]*v.y + m.m[2]*v.z,
        m.m[3]*v.x + m.m[4]*v.y + m.m[5]*v.z,
        m.m[6]*v.x + m.m[7]*v.y + m.m[8]*v.z
    };
}

// ----------------------------------------------------------------------------
// Compute 3D covariance matrix from scale and rotation: C = R * S^2 * R^T
// Returns 3x3 symmetric matrix as array of 6 elements (xx, xy, xz, yy, yz, zz)
// ----------------------------------------------------------------------------
static void compute_3d_covariance(const Gaussian &g, float *cov6) {
    // Scale squared (S^2 diagonal)
    float sx2 = g.scale.x * g.scale.x;
    float sy2 = g.scale.y * g.scale.y;
    float sz2 = g.scale.z * g.scale.z;
    // Rotation matrix from quaternion
    Mat3 R = quat_to_mat3(g.rot);
    // R * S^2 (matrix multiply)
    float RS2[3][3];
    for (int i = 0; i < 3; ++i) {
        RS2[i][0] = R.m[i*3+0] * sx2;
        RS2[i][1] = R.m[i*3+1] * sy2;
        RS2[i][2] = R.m[i*3+2] * sz2;
    }
    // C = (R * S^2) * R^T
    // cov6[0] = C_xx, [1] = C_xy, [2] = C_xz, [3] = C_yy, [4] = C_yz, [5] = C_zz
    cov6[0] = RS2[0][0]*R.m[0] + RS2[0][1]*R.m[1] + RS2[0][2]*R.m[2]; // row0 * col0 of R^T
    cov6[1] = RS2[0][0]*R.m[3] + RS2[0][1]*R.m[4] + RS2[0][2]*R.m[5]; // row0 * col1
    cov6[2] = RS2[0][0]*R.m[6] + RS2[0][1]*R.m[7] + RS2[0][2]*R.m[8]; // row0 * col2
    cov6[3] = RS2[1][0]*R.m[3] + RS2[1][1]*R.m[4] + RS2[1][2]*R.m[5]; // row1 * col1
    cov6[4] = RS2[1][0]*R.m[6] + RS2[1][1]*R.m[7] + RS2[1][2]*R.m[8]; // row1 * col2
    cov6[5] = RS2[2][0]*R.m[6] + RS2[2][1]*R.m[7] + RS2[2][2]*R.m[8]; // row2 * col2
}

// ----------------------------------------------------------------------------
// Transform 3D covariance from world to camera space using rotation part of view
// C_cam = R_view * C_world * R_view^T
// ----------------------------------------------------------------------------
static void transform_covariance_to_camera(const float cov6_world[6], const Mat3 &R_view, float cov6_cam[6]) {
    // First compute R_view * C_world (3x3)
    float RC[3][3];
    for (int i = 0; i < 3; ++i) {
        RC[i][0] = R_view.m[i*3+0]*cov6_world[0] + R_view.m[i*3+1]*cov6_world[1] + R_view.m[i*3+2]*cov6_world[2];
        RC[i][1] = R_view.m[i*3+0]*cov6_world[1] + R_view.m[i*3+1]*cov6_world[3] + R_view.m[i*3+2]*cov6_world[4];
        RC[i][2] = R_view.m[i*3+0]*cov6_world[2] + R_view.m[i*3+1]*cov6_world[4] + R_view.m[i*3+2]*cov6_world[5];
    }
    // Then multiply by R_view^T (transpose)
    cov6_cam[0] = RC[0][0]*R_view.m[0] + RC[0][1]*R_view.m[1] + RC[0][2]*R_view.m[2];
    cov6_cam[1] = RC[0][0]*R_view.m[3] + RC[0][1]*R_view.m[4] + RC[0][2]*R_view.m[5];
    cov6_cam[2] = RC[0][0]*R_view.m[6] + RC[0][1]*R_view.m[7] + RC[0][2]*R_view.m[8];
    cov6_cam[3] = RC[1][0]*R_view.m[3] + RC[1][1]*R_view.m[4] + RC[1][2]*R_view.m[5];
    cov6_cam[4] = RC[1][0]*R_view.m[6] + RC[1][1]*R_view.m[7] + RC[1][2]*R_view.m[8];
    cov6_cam[5] = RC[2][0]*R_view.m[6] + RC[2][1]*R_view.m[7] + RC[2][2]*R_view.m[8];
}

// ----------------------------------------------------------------------------
// Compute 2D covariance using Jacobian of perspective projection
// J = [[focal_x / z, 0, -focal_x * x / z^2],
//      [0, focal_y / z, -focal_y * y / z^2]]
// Cov2D = J * C_cam * J^T
// ----------------------------------------------------------------------------
static void compute_2d_covariance(const float cov6_cam[6], float x, float y, float z,
                                  float fx, float fy, float &cov_xx, float &cov_xy, float &cov_yy) {
    float z2 = z * z;
    float J00 = fx / z;
    float J02 = -fx * x / z2;
    float J11 = fy / z;
    float J12 = -fy * y / z2;
    // J * C_cam (3x3) -> we only need first two rows
    // Compute row0: [J00, 0, J02] * C_cam
    float row0_c0 = J00 * cov6_cam[0] + J02 * cov6_cam[2];
    float row0_c1 = J00 * cov6_cam[1] + J02 * cov6_cam[4];
    float row0_c2 = J00 * cov6_cam[2] + J02 * cov6_cam[5];
    // row1: [0, J11, J12] * C_cam
    float row1_c0 = J11 * cov6_cam[1] + J12 * cov6_cam[2];
    float row1_c1 = J11 * cov6_cam[3] + J12 * cov6_cam[4];
    float row1_c2 = J11 * cov6_cam[4] + J12 * cov6_cam[5];
    // Now multiply by J^T (2x3) to get 2x2 covariance
    cov_xx = row0_c0 * J00 + row0_c2 * J02;
    cov_xy = row0_c0 * 0    + row0_c2 * J12;
    cov_yy = row1_c1 * J11 + row1_c2 * J12;
    // Note: row0_c1 term does not contribute because J^T[1][0]=0, J^T[2][0]=J02
    // Ensure positive definiteness by adding a small epsilon
    cov_xx += 0.3f;
    cov_yy += 0.3f;
}

// ----------------------------------------------------------------------------
// Precompute inverse 2D covariance matrix for weight evaluation
// ----------------------------------------------------------------------------
static void inverse_2x2(float a, float b, float c, float &inv_a, float &inv_b, float &inv_c) {
    float det = a * c - b * b;
    if (det > 1e-6f) {
        inv_a =  c / det;
        inv_b = -b / det;
        inv_c =  a / det;
    } else {
        inv_a = inv_b = inv_c = 0.0f;
    }
}

// ----------------------------------------------------------------------------
// Project a single Gaussian to screen space
// Returns screen center (pixel), depth (view space Z), 2D covariance, and radius (3 sigma)
// ----------------------------------------------------------------------------
static void project_gaussian(const Gaussian &g, const Camera &cam,
                             Vec2 &center, float &depth,
                             float &cov_xx, float &cov_xy, float &cov_yy,
                             float &radius) {
    // Transform position to camera space
    Vec3 cam_pos = mat3_mul_vec3(cam.view_rot, g.pos);
    cam_pos.x += cam.view_trans.x;
    cam_pos.y += cam.view_trans.y;
    cam_pos.z += cam.view_trans.z;
    depth = cam_pos.z;
    if (depth <= 0.01f) {
        center = Vec2{-1,-1}; radius = 0.0f;
        return;
    }
    // Compute 3D world covariance
    float cov6_world[6];
    compute_3d_covariance(g, cov6_world);
    // Transform to camera space (rotation only)
    float cov6_cam[6];
    transform_covariance_to_camera(cov6_world, cam.view_rot, cov6_cam);
    // Compute 2D covariance using Jacobian at the projected center
    compute_2d_covariance(cov6_cam, cam_pos.x, cam_pos.y, cam_pos.z,
                          cam.focal_x, cam.focal_y, cov_xx, cov_xy, cov_yy);
    // Project center to screen (pixel coordinates)
    // Perspective division: first compute NDC using intrinsic parameters directly?
    // We'll use a simple perspective: screen_x = fx * x / z + width/2, etc.
    float ndc_x = cam_pos.x / depth;
    float ndc_y = cam_pos.y / depth;
    center.x = (ndc_x * cam.focal_x) + cam.width * 0.5f;
    center.y = cam.height * 0.5f - (ndc_y * cam.focal_y);
    // Compute radius = 3 * sqrt(max eigenvalue of 2D covariance)
    float det = cov_xx * cov_yy - cov_xy * cov_xy;
    if (det <= 0.0f) { radius = 0.0f; return; }
    float trace = cov_xx + cov_yy;
    float discriminant = trace * trace - 4.0f * det;
    if (discriminant < 0.0f) discriminant = 0.0f;
    float lambda_max = 0.5f * (trace + sqrt(discriminant));
    radius = 3.0f * sqrt(lambda_max);
}

// ----------------------------------------------------------------------------
// Gaussian weight at pixel offset (dx, dy) given inverse covariance
// ----------------------------------------------------------------------------
static float gaussian_weight(float dx, float dy, float inv_xx, float inv_xy, float inv_yy) {
    float v = dx * (inv_xx * dx + inv_xy * dy) + dy * (inv_xy * dx + inv_yy * dy);
    return expf(-0.5f * v);
}

// ----------------------------------------------------------------------------
// Main rasterization function (single threaded CPU reference)
// ----------------------------------------------------------------------------
void rasterize_gaussians(const std::vector<Gaussian> &gaussians, const Camera &cam,
                         std::vector<uint8_t> &out_rgb) {
    int w = (int)cam.width, h = (int)cam.height;
    out_rgb.assign(w * h * 3, 0);

    // Precompute projection data for each Gaussian
    struct ProjGaussian {
        Vec2 center;
        float depth;
        float cov_xx, cov_xy, cov_yy;
        float inv_xx, inv_xy, inv_yy;
        float radius;
        float opacity;
        Vec3 color;
        bool visible;
    };
    std::vector<ProjGaussian> proj;
    proj.reserve(gaussians.size());

    for (const auto &g : gaussians) {
        ProjGaussian pg;
        project_gaussian(g, cam, pg.center, pg.depth,
                         pg.cov_xx, pg.cov_xy, pg.cov_yy, pg.radius);
        pg.visible = (pg.radius > 0.0f && pg.depth > 0.01f);
        if (!pg.visible) continue;
        inverse_2x2(pg.cov_xx, pg.cov_xy, pg.cov_yy,
                    pg.inv_xx, pg.inv_xy, pg.inv_yy);
        pg.opacity = g.opacity;
        pg.color = g.color;
        proj.push_back(pg);
    }

    // Sort by depth descending (back to front for correct alpha blending)
    std::sort(proj.begin(), proj.end(),
              [](const ProjGaussian &a, const ProjGaussian &b) {
                  return a.depth > b.depth;
              });

    // Simple per‑pixel accumulation (no tile culling for clarity)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc_color[3] = {0.0f, 0.0f, 0.0f};
            float acc_alpha = 0.0f;
            for (const auto &pg : proj) {
                float dx = x - pg.center.x;
                float dy = y - pg.center.y;
                if (dx*dx + dy*dy > pg.radius * pg.radius) continue;
                float weight = gaussian_weight(dx, dy, pg.inv_xx, pg.inv_xy, pg.inv_yy);
                float alpha = weight * pg.opacity;
                if (alpha <= 0.01f) continue;
                float t = 1.0f - acc_alpha;
                acc_color[0] += t * alpha * pg.color.x;
                acc_color[1] += t * alpha * pg.color.y;
                acc_color[2] += t * alpha * pg.color.z;
                acc_alpha += t * alpha;
                if (acc_alpha >= 0.99f) break;
            }
            int idx = (y * w + x) * 3;
            out_rgb[idx]   = (uint8_t)(acc_color[0] * 255.0f);
            out_rgb[idx+1] = (uint8_t)(acc_color[1] * 255.0f);
            out_rgb[idx+2] = (uint8_t)(acc_color[2] * 255.0f);
        }
    }
}