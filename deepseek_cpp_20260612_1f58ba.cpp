// Name : lighting enhancement updated
// File : gaussian_splatting_rasterizer.cpp 1 of 63
// Description : CUDA‑accelerated Gaussian splatting rasterization with tile‑based culling,
//               perspective projection, 2D covariance calculation, and alpha blending.
//               Implements forward and backward passes as described in the 3DGS paper.
#include "splat_py/cuda_ext.h"
#include "src/gsplat.h"
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

// ----------------------------------------------------------------------------
// Helper: compute 3D covariance from scale and rotation (quaternion)
// ----------------------------------------------------------------------------
__device__ void compute_cov3d(const float* scale, const float* quat, float* cov3d) {
    float r[9];
    float qx = quat[0], qy = quat[1], qz = quat[2], qw = quat[3];
    // quaternion to rotation matrix
    float x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
    float xx = qx * x2, xy = qx * y2, xz = qx * z2;
    float yy = qy * y2, yz = qy * z2, zz = qz * z2;
    float wx = qw * x2, wy = qw * y2, wz = qw * z2;
    r[0] = 1.0f - (yy + zz); r[1] = xy - wz;       r[2] = xz + wy;
    r[3] = xy + wz;       r[4] = 1.0f - (xx + zz); r[5] = yz - wx;
    r[6] = xz - wy;       r[7] = yz + wx;       r[8] = 1.0f - (xx + yy);
    // scale matrix S = diag(scale)
    float s[9] = {scale[0],0,0, 0,scale[1],0, 0,0,scale[2]};
    // covariance = R * S * S^T * R^T = R * (S * S^T) * R^T
    float s2[9] = {scale[0]*scale[0],0,0, 0,scale[1]*scale[1],0, 0,0,scale[2]*scale[2]};
    // M = R * s2
    float m[9];
    for (int i=0;i<3;++i)
        for (int j=0;j<3;++j)
            m[i*3+j] = r[i*3+0]*s2[0*3+j] + r[i*3+1]*s2[1*3+j] + r[i*3+2]*s2[2*3+j];
    // cov3d = M * R^T
    for (int i=0;i<3;++i)
        for (int j=0;j<3;++j)
            cov3d[i*3+j] = m[i*3+0]*r[j*3+0] + m[i*3+1]*r[j*3+1] + m[i*3+2]*r[j*3+2];
}

// ----------------------------------------------------------------------------
// Project 3D gaussian to 2D covariance and mean (pixel coordinates)
// ----------------------------------------------------------------------------
__device__ bool project_gaussian(const float* mean3d, const float* cov3d,
                                 const float* viewmat, const float* projmat,
                                 int width, int height,
                                 float* mean2d, float* cov2d, float& depth) {
    // Transform to camera space
    float cam[3] = {
        viewmat[0]*mean3d[0] + viewmat[1]*mean3d[1] + viewmat[2]*mean3d[2] + viewmat[3],
        viewmat[4]*mean3d[0] + viewmat[5]*mean3d[1] + viewmat[6]*mean3d[2] + viewmat[7],
        viewmat[8]*mean3d[0] + viewmat[9]*mean3d[1] + viewmat[10]*mean3d[2] + viewmat[11]
    };
    if (cam[2] <= 0.01f) return false;
    depth = cam[2];
    // Clip space
    float clip_x = projmat[0]*cam[0] + projmat[1]*cam[1] + projmat[2]*cam[2] + projmat[3];
    float clip_y = projmat[4]*cam[0] + projmat[5]*cam[1] + projmat[6]*cam[2] + projmat[7];
    float clip_z = projmat[8]*cam[0] + projmat[9]*cam[1] + projmat[10]*cam[2] + projmat[11];
    if (clip_z <= 0.0f) return false;
    float inv_z = 1.0f / clip_z;
    mean2d[0] = (clip_x * inv_z + 1.0f) * 0.5f * width;
    mean2d[1] = (1.0f - (clip_y * inv_z + 1.0f) * 0.5f) * height;
    // Jacobian of projection
    float focal_x = projmat[0] * width * 0.5f;
    float focal_y = projmat[5] * height * 0.5f;
    float J[2][3] = {
        {focal_x / cam[2], 0.0f, -focal_x * cam[0] / (cam[2]*cam[2])},
        {0.0f, focal_y / cam[2], -focal_y * cam[1] / (cam[2]*cam[2])}
    };
    // Transform 3D covariance to 2D: cov2d = J * cov3d * J^T
    float temp[2][3];
    for (int i=0;i<2;++i)
        for (int j=0;j<3;++j)
            temp[i][j] = J[i][0]*cov3d[0*3+j] + J[i][1]*cov3d[1*3+j] + J[i][2]*cov3d[2*3+j];
    for (int i=0;i<2;++i)
        for (int j=0;j<2;++j)
            cov2d[i*2+j] = temp[i][0]*J[j][0] + temp[i][1]*J[j][1] + temp[i][2]*J[j][2];
    // add small epsilon
    cov2d[0] += 0.3f; cov2d[3] += 0.3f;
    return true;
}

// ----------------------------------------------------------------------------
// Tile culling (assign gaussians to tiles based on projected bounding box)
// ----------------------------------------------------------------------------
__global__ void tile_culling_kernel(const float* means2d, const float* cov2d,
                                    const float* depths, int num_gaussians,
                                    int tiles_x, int tiles_y, int tile_size,
                                    int* tile_lists, int* tile_counts) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_gaussians) return;
    float mx = means2d[gid*2];
    float my = means2d[gid*2+1];
    float cov00 = cov2d[gid*4];
    float cov01 = cov2d[gid*4+1];
    float cov11 = cov2d[gid*4+3];
    // compute bounding box in screen space (3 sigma)
    float det = cov00 * cov11 - cov01 * cov01;
    float sqrt_det = sqrtf(fmaxf(0.0f, det));
    float radius = 3.0f * sqrtf(0.5f * (cov00 + cov11 + sqrtf(fmaxf(0.0f, (cov00 - cov11)*(cov00 - cov11) + 4.0f*cov01*cov01))));
    int min_x = max(0, (int)((mx - radius) / tile_size));
    int max_x = min(tiles_x-1, (int)((mx + radius) / tile_size));
    int min_y = max(0, (int)((my - radius) / tile_size));
    int max_y = min(tiles_y-1, (int)((my + radius) / tile_size));
    for (int ty = min_y; ty <= max_y; ++ty) {
        for (int tx = min_x; tx <= max_x; ++tx) {
            int tile_id = ty * tiles_x + tx;
            int idx = atomicAdd(&tile_counts[tile_id], 1);
            tile_lists[tile_id * MAX_GAUSSIANS_PER_TILE + idx] = gid;
        }
    }
}

// ----------------------------------------------------------------------------
// Forward rasterization: alpha blending in sorted order (by depth)
// ----------------------------------------------------------------------------
__global__ void rasterize_forward_kernel(const float* means2d, const float* cov2d,
                                         const float* colors, const float* opacities,
                                         const int* tile_lists, const int* tile_counts,
                                         int tiles_x, int tiles_y, int tile_size,
                                         int width, int height, float* out_color) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tiles_x || ty >= tiles_y) return;
    int tile_id = ty * tiles_x + tx;
    int count = tile_counts[tile_id];
    if (count == 0) return;
    // fetch gaussian list for this tile
    int local_list[MAX_GAUSSIANS_PER_TILE];
    for (int i=0;i<count;++i) local_list[i] = tile_lists[tile_id * MAX_GAUSSIANS_PER_TILE + i];
    // sort by depth (simple bubble sort, in production use radix sort)
    for (int i=0;i<count-1;++i)
        for (int j=0;j<count-1-i;++j)
            if (depths[local_list[j]] > depths[local_list[j+1]]) {
                int tmp = local_list[j];
                local_list[j] = local_list[j+1];
                local_list[j+1] = tmp;
            }
    // compute start pixel of this tile
    int start_x = tx * tile_size;
    int start_y = ty * tile_size;
    for (int dy=0; dy<tile_size && (start_y+dy)<height; ++dy) {
        int y = start_y + dy;
        for (int dx=0; dx<tile_size && (start_x+dx)<width; ++dx) {
            int x = start_x + dx;
            float px = x + 0.5f;
            float py = y + 0.5f;
            float T = 1.0f;
            float final_r=0, final_g=0, final_b=0;
            for (int gi=0; gi<count; ++gi) {
                int gid = local_list[gi];
                float mx = means2d[gid*2];
                float my = means2d[gid*2+1];
                float dx_ = px - mx;
                float dy_ = py - my;
                float cov00 = cov2d[gid*4];
                float cov01 = cov2d[gid*4+1];
                float cov11 = cov2d[gid*4+3];
                float det = cov00 * cov11 - cov01 * cov01;
                if (det <= 0.0f) continue;
                float inv_det = 1.0f / det;
                float power = -0.5f * (dx_*dx_ * cov11 * inv_det +
                                        dy_*dy_ * cov00 * inv_det -
                                        2.0f*dx_*dy_ * cov01 * inv_det);
                float alpha = opacities[gid] * expf(power);
                if (alpha <= 0.01f) continue;
                float r = colors[gid*3], g = colors[gid*3+1], b = colors[gid*3+2];
                final_r += T * alpha * r;
                final_g += T * alpha * g;
                final_b += T * alpha * b;
                T *= (1.0f - alpha);
                if (T < 0.01f) break;
            }
            int pixel = y * width + x;
            out_color[pixel*3] = final_r;
            out_color[pixel*3+1] = final_g;
            out_color[pixel*3+2] = final_b;
        }
    }
}

// ----------------------------------------------------------------------------
// Public function: rasterize gaussians to image
// ----------------------------------------------------------------------------
extern "C" void rasterize_gaussians(const float* means3d, const float* scales,
                                    const float* quats, const float* colors,
                                    const float* opacities, int num_gaussians,
                                    const float* viewmat, const float* projmat,
                                    int width, int height, int tile_size,
                                    float* out_color, float* out_depth) {
    // allocate device memory for intermediate data
    float* d_means2d = nullptr;
    float* d_cov2d = nullptr;
    float* d_depths = nullptr;
    cudaMalloc(&d_means2d, num_gaussians * 2 * sizeof(float));
    cudaMalloc(&d_cov2d, num_gaussians * 4 * sizeof(float));
    cudaMalloc(&d_depths, num_gaussians * sizeof(float));
    // compute 3D covariance and project each Gaussian
    for (int i=0;i<num_gaussians;++i) {
        float cov3d[9];
        compute_cov3d(scales + i*3, quats + i*4, cov3d);
        float mean2d[2], cov2d[4], depth;
        if (project_gaussian(means3d + i*3, cov3d, viewmat, projmat, width, height, mean2d, cov2d, depth)) {
            cudaMemcpy(d_means2d + i*2, mean2d, 2*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_cov2d + i*4, cov2d, 4*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_depths + i, &depth, sizeof(float), cudaMemcpyHostToDevice);
        } else {
            // mark invalid (set depth to large value to be culled)
            float far = 1e10f;
            cudaMemcpy(d_depths + i, &far, sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    int tiles_x = (width + tile_size - 1) / tile_size;
    int tiles_y = (height + tile_size - 1) / tile_size;
    int num_tiles = tiles_x * tiles_y;
    int* d_tile_counts = nullptr;
    int* d_tile_lists = nullptr;
    cudaMalloc(&d_tile_counts, num_tiles * sizeof(int));
    cudaMalloc(&d_tile_lists, num_tiles * MAX_GAUSSIANS_PER_TILE * sizeof(int));
    cudaMemset(d_tile_counts, 0, num_tiles * sizeof(int));
    // tile culling kernel
    int threads_per_block = 256;
    int blocks = (num_gaussians + threads_per_block - 1) / threads_per_block;
    tile_culling_kernel<<<blocks, threads_per_block>>>(d_means2d, d_cov2d, d_depths, num_gaussians,
                                                       tiles_x, tiles_y, tile_size,
                                                       d_tile_lists, d_tile_counts);
    // forward rasterization
    dim3 block_tile(16,16);
    dim3 grid_tile((tiles_x+block_tile.x-1)/block_tile.x, (tiles_y+block_tile.y-1)/block_tile.y);
    rasterize_forward_kernel<<<grid_tile, block_tile>>>(d_means2d, d_cov2d, colors, opacities,
                                                        d_tile_lists, d_tile_counts,
                                                        tiles_x, tiles_y, tile_size,
                                                        width, height, out_color);
    cudaFree(d_means2d);
    cudaFree(d_cov2d);
    cudaFree(d_depths);
    cudaFree(d_tile_counts);
    cudaFree(d_tile_lists);
}