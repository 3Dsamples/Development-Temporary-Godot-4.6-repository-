// Name : lighting enhancement updated
// File : gsplat_projection.cpp 65 of 63
// Description : Projection of 3D Gaussians to 2D screen space, covariance matrix math,
//               tile culling data structures, and depth sorting helpers.
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

// ----------------------------------------------------------------------------
// Math types (reusing previous definitions)
// ----------------------------------------------------------------------------
struct Vec3 { float x, y, z; };
struct Vec4 { float x, y, z, w; };
struct Mat3 { float m[9]; };
struct Quat { float w, x, y, z; };
struct Mat4 { float m[16]; };

// ----------------------------------------------------------------------------
// Camera parameters for projection
// ----------------------------------------------------------------------------
struct CameraParams {
    Mat4 view;               // world → camera (row‑major)
    Mat4 proj;               // camera → clip
    float width, height;     // image dimensions
    float fx, fy;            // focal lengths (pixels)
    float cx, cy;            // principal point (pixels)
};

// ----------------------------------------------------------------------------
// Quaternion to rotation matrix (3x3)
// ----------------------------------------------------------------------------
static Mat3 quat_to_mat3(const Quat &q) {
    float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    Mat3 m;
    m.m[0] = 1 - 2*(yy + zz); m.m[1] = 2*(xy - wz);    m.m[2] = 2*(xz + wy);
    m.m[3] = 2*(xy + wz);    m.m[4] = 1 - 2*(xx + zz); m.m[5] = 2*(yz - wx);
    m.m[6] = 2*(xz - wy);    m.m[7] = 2*(yz + wx);    m.m[8] = 1 - 2*(xx + yy);
    return m;
}

// ----------------------------------------------------------------------------
// 3x3 matrix multiplied by vector
// ----------------------------------------------------------------------------
static Vec3 mat3_mul_vec3(const Mat3 &m, const Vec3 &v) {
    return Vec3{
        m.m[0]*v.x + m.m[1]*v.y + m.m[2]*v.z,
        m.m[3]*v.x + m.m[4]*v.y + m.m[5]*v.z,
        m.m[6]*v.x + m.m[7]*v.y + m.m[8]*v.z
    };
}

// ----------------------------------------------------------------------------
// Transform point by 4x4 matrix (row‑major)
// ----------------------------------------------------------------------------
static Vec3 mat4_mul_vec3(const Mat4 &m, const Vec3 &v) {
    float w = m.m[12]*v.x + m.m[13]*v.y + m.m[14]*v.z + m.m[15];
    if (fabs(w) < 1e-6f) w = 1.0f;
    return Vec3{
        (m.m[0]*v.x + m.m[1]*v.y + m.m[2]*v.z + m.m[3]) / w,
        (m.m[4]*v.x + m.m[5]*v.y + m.m[6]*v.z + m.m[7]) / w,
        (m.m[8]*v.x + m.m[9]*v.y + m.m[10]*v.z + m.m[11]) / w
    };
}

// ----------------------------------------------------------------------------
// 3D covariance from scale and rotation: C = R * S^2 * R^T (6‑component)
// Returns array of 6 floats: xx, xy, xz, yy, yz, zz
// ----------------------------------------------------------------------------
static void compute_3d_covariance(const Vec3 &scale, const Quat &rot, float *cov6) {
    float sx2 = scale.x * scale.x;
    float sy2 = scale.y * scale.y;
    float sz2 = scale.z * scale.z;
    Mat3 R = quat_to_mat3(rot);
    // Compute R * S^2 (3x3)
    float RS2[3][3];
    for (int i = 0; i < 3; ++i) {
        RS2[i][0] = R.m[i*3+0] * sx2;
        RS2[i][1] = R.m[i*3+1] * sy2;
        RS2[i][2] = R.m[i*3+2] * sz2;
    }
    // Multiply by R^T
    cov6[0] = RS2[0][0]*R.m[0] + RS2[0][1]*R.m[1] + RS2[0][2]*R.m[2];
    cov6[1] = RS2[0][0]*R.m[3] + RS2[0][1]*R.m[4] + RS2[0][2]*R.m[5];
    cov6[2] = RS2[0][0]*R.m[6] + RS2[0][1]*R.m[7] + RS2[0][2]*R.m[8];
    cov6[3] = RS2[1][0]*R.m[3] + RS2[1][1]*R.m[4] + RS2[1][2]*R.m[5];
    cov6[4] = RS2[1][0]*R.m[6] + RS2[1][1]*R.m[7] + RS2[1][2]*R.m[8];
    cov6[5] = RS2[2][0]*R.m[6] + RS2[2][1]*R.m[7] + RS2[2][2]*R.m[8];
}

// ----------------------------------------------------------------------------
// Transform covariance from world to camera space (only rotation part)
// ----------------------------------------------------------------------------
static void covariance_world_to_camera(const float cov6_world[6], const Mat3 &R_cam,
                                       float cov6_cam[6]) {
    float RC[3][3];
    RC[0][0] = R_cam.m[0]*cov6_world[0] + R_cam.m[1]*cov6_world[1] + R_cam.m[2]*cov6_world[2];
    RC[0][1] = R_cam.m[0]*cov6_world[1] + R_cam.m[1]*cov6_world[3] + R_cam.m[2]*cov6_world[4];
    RC[0][2] = R_cam.m[0]*cov6_world[2] + R_cam.m[1]*cov6_world[4] + R_cam.m[2]*cov6_world[5];
    RC[1][0] = R_cam.m[3]*cov6_world[0] + R_cam.m[4]*cov6_world[1] + R_cam.m[5]*cov6_world[2];
    RC[1][1] = R_cam.m[3]*cov6_world[1] + R_cam.m[4]*cov6_world[3] + R_cam.m[5]*cov6_world[4];
    RC[1][2] = R_cam.m[3]*cov6_world[2] + R_cam.m[4]*cov6_world[4] + R_cam.m[5]*cov6_world[5];
    RC[2][0] = R_cam.m[6]*cov6_world[0] + R_cam.m[7]*cov6_world[1] + R_cam.m[8]*cov6_world[2];
    RC[2][1] = R_cam.m[6]*cov6_world[1] + R_cam.m[7]*cov6_world[3] + R_cam.m[8]*cov6_world[4];
    RC[2][2] = R_cam.m[6]*cov6_world[2] + R_cam.m[7]*cov6_world[4] + R_cam.m[8]*cov6_world[5];

    // Multiply by R_cam^T (transpose)
    cov6_cam[0] = RC[0][0]*R_cam.m[0] + RC[0][1]*R_cam.m[1] + RC[0][2]*R_cam.m[2];
    cov6_cam[1] = RC[0][0]*R_cam.m[3] + RC[0][1]*R_cam.m[4] + RC[0][2]*R_cam.m[5];
    cov6_cam[2] = RC[0][0]*R_cam.m[6] + RC[0][1]*R_cam.m[7] + RC[0][2]*R_cam.m[8];
    cov6_cam[3] = RC[1][0]*R_cam.m[3] + RC[1][1]*R_cam.m[4] + RC[1][2]*R_cam.m[5];
    cov6_cam[4] = RC[1][0]*R_cam.m[6] + RC[1][1]*R_cam.m[7] + RC[1][2]*R_cam.m[8];
    cov6_cam[5] = RC[2][0]*R_cam.m[6] + RC[2][1]*R_cam.m[7] + RC[2][2]*R_cam.m[8];
}

// ----------------------------------------------------------------------------
// Compute 2D covariance given camera‑space covariance and 3D point (x,y,z)
// using perspective projection Jacobian
// ----------------------------------------------------------------------------
static void compute_2d_covariance(const float cov6_cam[6], float x, float y, float z,
                                  float fx, float fy, float &cxx, float &cxy, float &cyy) {
    float z2 = z * z;
    float J00 = fx / z;
    float J02 = -fx * x / z2;
    float J11 = fy / z;
    float J12 = -fy * y / z2;

    // J * C_cam (first two rows)
    float row0_c0 = J00 * cov6_cam[0] + J02 * cov6_cam[2];
    float row0_c1 = J00 * cov6_cam[1] + J02 * cov6_cam[4];
    float row0_c2 = J00 * cov6_cam[2] + J02 * cov6_cam[5];
    float row1_c0 = J11 * cov6_cam[1] + J12 * cov6_cam[2];
    float row1_c1 = J11 * cov6_cam[3] + J12 * cov6_cam[4];
    float row1_c2 = J11 * cov6_cam[4] + J12 * cov6_cam[5];

    // Multiply by J^T (2x3)
    cxx = row0_c0 * J00 + row0_c2 * J02;
    cxy = row0_c0 * 0    + row0_c2 * J12;
    cyy = row1_c1 * J11 + row1_c2 * J12;
    // Stabilization
    cxx += 0.3f;
    cyy += 0.3f;
}

// ----------------------------------------------------------------------------
// Project a single Gaussian to screen space (pixel coordinates and covariance)
// ----------------------------------------------------------------------------
struct ProjectedGaussian {
    float depth;               // camera Z (positive)
    Vec2 screen_center;        // pixel coordinates
    float cov_xx, cov_xy, cov_yy;
    float radius;              // 3 sigma bounding box radius (pixels)
};

static ProjectedGaussian project_gaussian(const Vec3 &pos, const Vec3 &scale, const Quat &rot,
                                          const CameraParams &cam) {
    ProjectedGaussian out = {};

    // Transform position to camera space (using view matrix)
    Vec3 cam_pos = mat4_mul_vec3(cam.view, pos);
    out.depth = cam_pos.z;
    if (out.depth <= 0.01f) {
        out.radius = 0.0f;
        return out;
    }

    // Compute world covariance and transform to camera space
    float cov6_world[6];
    compute_3d_covariance(scale, rot, cov6_world);
    // Extract rotation part of view matrix (3x3)
    Mat3 R_cam;
    for (int i = 0; i < 3; ++i) {
        R_cam.m[i*3+0] = cam.view.m[i*4+0];
        R_cam.m[i*3+1] = cam.view.m[i*4+1];
        R_cam.m[i*3+2] = cam.view.m[i*4+2];
    }
    float cov6_cam[6];
    covariance_world_to_camera(cov6_world, R_cam, cov6_cam);

    // Compute 2D covariance
    compute_2d_covariance(cov6_cam, cam_pos.x, cam_pos.y, cam_pos.z,
                          cam.fx, cam.fy, out.cov_xx, out.cov_xy, out.cov_yy);

    // Project center to screen pixel coordinates
    // Using intrinsic parameters
    float ndc_x = cam_pos.x / out.depth;
    float ndc_y = cam_pos.y / out.depth;
    out.screen_center.x = ndc_x * cam.fx + cam.cx;
    out.screen_center.y = ndc_y * cam.fy + cam.cy;

    // Compute bounding box radius (3 sigma)
    float det = out.cov_xx * out.cov_yy - out.cov_xy * out.cov_xy;
    if (det <= 0.0f) {
        out.radius = 0.0f;
    } else {
        float trace = out.cov_xx + out.cov_yy;
        float disc = trace * trace - 4.0f * det;
        if (disc < 0.0f) disc = 0.0f;
        float lambda_max = 0.5f * (trace + sqrtf(disc));
        out.radius = 3.0f * sqrtf(lambda_max);
    }
    return out;
}

// ----------------------------------------------------------------------------
// Tile culling: assign each Gaussian to tiles overlapping its bounding box
// ----------------------------------------------------------------------------
struct TileInfo {
    int tile_w, tile_h;            // number of tiles in X and Y
    int tile_size;                 // pixels per tile (e.g., 16)
    std::vector<std::vector<int>> tile_lists;  // for each tile, list of Gaussian indices
};

static TileInfo build_tile_culling(const std::vector<ProjectedGaussian> &projected,
                                   int width, int height, int tile_size = 16) {
    TileInfo info;
    info.tile_size = tile_size;
    info.tile_w = (width + tile_size - 1) / tile_size;
    info.tile_h = (height + tile_size - 1) / tile_size;
    info.tile_lists.resize(info.tile_w * info.tile_h);

    for (int gidx = 0; gidx < (int)projected.size(); ++gidx) {
        const ProjectedGaussian &pg = projected[gidx];
        if (pg.radius <= 0.0f) continue;
        int min_x = (int)(pg.screen_center.x - pg.radius);
        int max_x = (int)(pg.screen_center.x + pg.radius);
        int min_y = (int)(pg.screen_center.y - pg.radius);
        int max_y = (int)(pg.screen_center.y + pg.radius);
        min_x = std::max(0, min_x);
        max_x = std::min(width - 1, max_x);
        min_y = std::max(0, min_y);
        max_y = std::min(height - 1, max_y);
        int first_tile_x = min_x / tile_size;
        int last_tile_x  = max_x / tile_size;
        int first_tile_y = min_y / tile_size;
        int last_tile_y  = max_y / tile_size;
        for (int ty = first_tile_y; ty <= last_tile_y; ++ty) {
            for (int tx = first_tile_x; tx <= last_tile_x; ++tx) {
                int tile_idx = ty * info.tile_w + tx;
                info.tile_lists[tile_idx].push_back(gidx);
            }
        }
    }
    return info;
}

// ----------------------------------------------------------------------------
// Helper: inverse 2x2 matrix (for weight evaluation)
// ----------------------------------------------------------------------------
static void inv_2x2(float a, float b, float c, float &ia, float &ib, float &ic) {
    float det = a * c - b * b;
    if (det > 1e-6f) {
        ia =  c / det;
        ib = -b / det;
        ic =  a / det;
    } else {
        ia = ib = ic = 0.0f;
    }
}

// ----------------------------------------------------------------------------
// Gaussian weight function
// ----------------------------------------------------------------------------
static float gaussian_weight(float dx, float dy, float inv_xx, float inv_xy, float inv_yy) {
    float v = dx * (inv_xx * dx + inv_xy * dy) + dy * (inv_xy * dx + inv_yy * dy);
    return expf(-0.5f * v);
}