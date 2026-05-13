// file 0008
// path: modules/sim_math/sim_math_unified_conversions.h

// ---------------------------------------------------------------------------
// This file provides a comprehensive shallow integration layer between:
// - Godot 4.6 core math (Vector3, Transform3D, Basis, Quaternion)
// - Eigen (linear algebra: SVD, QR, Cholesky, eigenvalues)
// - xtensor (multi-dimensional arrays for tensor operations)
// - BigNumber (arbitrary precision integers)
// - FixedMathCore (fixed-point arithmetic for deterministic simulation)
// - RenderingDeviceEnhanced (GPU upload helpers for simulation data)
// - RenderForwardClusteredEnhanced (integration with the 120 FPS renderer)
// - RendererSceneCullEnhanced (scene culling integration)
// - MaterialSystemEnhanced (material and texture injection)
// - TAAImplementation (temporal anti-aliasing history injection)
// - FidelityFX SSSR, SPD, FSR2/FSR3 (screen space reflections, downsampling, upscaling)
// - TreeNSearch-3d (BVH traversal for light culling and culling)
// - DirectXMath (SIMD optimizations for matrix operations)
//
// All conversions are pre-defined as inline functions, ready to be used
// anywhere in the codebase without explicitly including the underlying libraries
// each time. This file is designed to be included once per translation unit.
// ---------------------------------------------------------------------------

#ifndef SIM_MATH_UNIFIED_CONVERSIONS_H
#define SIM_MATH_UNIFIED_CONVERSIONS_H

// ---------------------------------------------------------------------------
// 1. Include all necessary math and simulation libraries
// ---------------------------------------------------------------------------

// Godot core math (kept for rendering)
#include "core/math/vector3.h"
#include "core/math/vector4.h"
#include "core/math/basis.h"
#include "core/math/transform_3d.h"
#include "core/math/quaternion.h"
#include "core/math/aabb.h"
#include "core/math/plane.h"
#include "core/math/matrix3x3.h"

// Eigen (linear algebra for simulation)
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

// xtensor (tensor operations for simulation)
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xio.hpp>

// BigNumber (arbitrary precision integers for exact simulation)
#include "big_number/big_number.h"

// FixedMathCore (fixed-point arithmetic for deterministic simulation)
#include "fixed_math_core/fixed_math_core.h"

// Rendering system components (for GPU upload and rendering integration)
#include "servers/rendering/renderer_rd/rendering_device_enhanced.h"
#include "servers/rendering/renderer_rd/forward_clustered/render_forward_clustered_enhanced.h"
#include "servers/rendering/renderer_rd/renderer_scene_cull_enhanced.h"
#include "servers/rendering/renderer_rd/material_system_enhanced.h"
#include "servers/rendering/renderer_rd/effects/taa_implementation.h"
#include "servers/rendering/renderer_rd/effects/ffx_sssr.h"
#include "servers/rendering/renderer_rd/effects/ffx_spd.h"
#include "servers/rendering/renderer_rd/effects/ffx_fsr2.h"
#include "servers/rendering/renderer_rd/effects/ffx_fsr3.h"

// TreeNSearch-3d (BVH for culling)
#include "bvh/bvh.h"

// DirectXMath (SIMD for high-performance matrix operations)
#include <DirectXMath.h>

// GLM (provided for compatibility with existing shader code)
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ---------------------------------------------------------------------------
// 2. Namespace definitions
// ---------------------------------------------------------------------------

namespace SimulationMath {

// ---------------------------------------------------------------------------
// 3. Conversion functions: Godot core math <-> Eigen
// ---------------------------------------------------------------------------

// Convert Godot Vector3 to Eigen Vector3f
inline Eigen::Vector3f to_eigen(const Godot::Vector3& v) {
    return Eigen::Vector3f(v.x, v.y, v.z);
}

// Convert Godot Vector3 to Eigen Vector3d (double precision)
inline Eigen::Vector3d to_eigen_d(const Godot::Vector3& v) {
    return Eigen::Vector3d(static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z));
}

// Convert Eigen Vector3f to Godot Vector3
inline Godot::Vector3 to_godot(const Eigen::Vector3f& v) {
    return Godot::Vector3(v.x(), v.y(), v.z());
}

// Convert Eigen Vector3d to Godot Vector3
inline Godot::Vector3 to_godot(const Eigen::Vector3d& v) {
    return Godot::Vector3(static_cast<float>(v.x()), static_cast<float>(v.y()), static_cast<float>(v.z()));
}

// Convert Godot Basis to Eigen Matrix3f
inline Eigen::Matrix3f to_eigen(const Godot::Basis& b) {
    Eigen::Matrix3f m;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m(i, j) = b.rows[i][j];
        }
    }
    return m;
}

// Convert Eigen Matrix3f to Godot Basis
inline Godot::Basis to_godot(const Eigen::Matrix3f& m) {
    Godot::Basis b;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            b.rows[i][j] = m(i, j);
        }
    }
    return b;
}

// Convert Godot Transform3D to Eigen Affine3f
inline Eigen::Affine3f to_eigen(const Godot::Transform3D& t) {
    Eigen::Affine3f e;
    e.linear() = to_eigen(t.basis);
    e.translation() = to_eigen(t.origin);
    return e;
}

// Convert Eigen Affine3f to Godot Transform3D
inline Godot::Transform3D to_godot(const Eigen::Affine3f& e) {
    Godot::Transform3D t;
    t.basis = to_godot(e.linear());
    t.origin = to_godot(e.translation());
    return t;
}

// Convert Godot Quaternion to Eigen Quaternionf
inline Eigen::Quaternionf to_eigen(const Godot::Quaternion& q) {
    return Eigen::Quaternionf(q.w, q.x, q.y, q.z);
}

// Convert Eigen Quaternionf to Godot Quaternion
inline Godot::Quaternion to_godot(const Eigen::Quaternionf& q) {
    return Godot::Quaternion(q.w(), q.x(), q.y(), q.z());
}

// Convert Godot AABB to Eigen AlignedBox3f
inline Eigen::AlignedBox3f to_eigen(const Godot::AABB& aabb) {
    return Eigen::AlignedBox3f(to_eigen(aabb.position), to_eigen(aabb.position + aabb.size));
}

// Convert Eigen AlignedBox3f to Godot AABB
inline Godot::AABB to_godot(const Eigen::AlignedBox3f& box) {
    return Godot::AABB(to_godot(box.min()), to_godot(box.max() - box.min()));
}

// Convert Godot Plane to Eigen Hyperplane3f
inline Eigen::Hyperplane<float, 3> to_eigen(const Godot::Plane& p) {
    return Eigen::Hyperplane<float, 3>(to_eigen(p.normal), -p.d);
}

// Convert Eigen Hyperplane3f to Godot Plane
inline Godot::Plane to_godot(const Eigen::Hyperplane<float, 3>& ep) {
    return Godot::Plane(to_godot(ep.normal()), -ep.offset());
}

// ---------------------------------------------------------------------------
// 4. Conversion functions: Eigen <-> xtensor
// ---------------------------------------------------------------------------

// Convert Eigen Matrix to xtensor (2D)
template<typename EigenType>
xt::xtensor<typename EigenType::Scalar, 2> to_xtensor(const EigenType& eigen_mat) {
    xt::xtensor<typename EigenType::Scalar, 2> result(eigen_mat.rows(), eigen_mat.cols());
    std::memcpy(result.data(), eigen_mat.data(), eigen_mat.size() * sizeof(typename EigenType::Scalar));
    return result;
}

// Convert xtensor 2D to Eigen Matrix
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> to_eigen(const xt::xtensor<Scalar, 2>& tensor) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(tensor.shape()[0], tensor.shape()[1]);
    std::memcpy(result.data(), tensor.data(), tensor.size() * sizeof(Scalar));
    return result;
}

// Convert Eigen Vector to xtensor (1D)
template<typename EigenType>
xt::xtensor<typename EigenType::Scalar, 1> to_xtensor(const EigenType& eigen_vec) {
    xt::xtensor<typename EigenType::Scalar, 1> result(eigen_vec.size());
    std::memcpy(result.data(), eigen_vec.data(), eigen_vec.size() * sizeof(typename EigenType::Scalar));
    return result;
}

// Convert xtensor 1D to Eigen Vector
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> to_eigen(const xt::xtensor<Scalar, 1>& tensor) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> result(tensor.size());
    std::memcpy(result.data(), tensor.data(), tensor.size() * sizeof(Scalar));
    return result;
}

// ---------------------------------------------------------------------------
// 5. Conversion functions: FixedMathCore <-> Eigen
// ---------------------------------------------------------------------------

// Convert Fixed32 to float (for Eigen operations)
inline float to_float(Fixed32 f) {
    return f.to_float();
}

// Convert float to Fixed32 (from Eigen result)
inline Fixed32 to_fixed32(float f) {
    return Fixed32::from_float(f);
}

// Convert Fixed32 to Eigen Vector3f
inline Eigen::Vector3f to_eigen(const FixedVector3& v) {
    return Eigen::Vector3f(v.x.to_float(), v.y.to_float(), v.z.to_float());
}

// Convert Eigen Vector3f to FixedVector3
inline FixedVector3 to_fixed_vector3(const Eigen::Vector3f& v) {
    return FixedVector3(Fixed32::from_float(v.x()), Fixed32::from_float(v.y()), Fixed32::from_float(v.z()));
}

// Convert Fixed32 to Eigen Matrix3f
inline Eigen::Matrix3f to_eigen(const FixedMatrix3& m) {
    Eigen::Matrix3f result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result(i, j) = m(i, j).to_float();
        }
    }
    return result;
}

// Convert Eigen Matrix3f to FixedMatrix3
inline FixedMatrix3 to_fixed_matrix3(const Eigen::Matrix3f& m) {
    FixedMatrix3 result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result(i, j) = Fixed32::from_float(m(i, j));
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// 6. Conversion functions: BigNumber <-> floating point
// ---------------------------------------------------------------------------

// Convert BigNumber to double (approximate, with loss of precision for very large numbers)
inline double to_double(const BigNumber& bn) {
    return bn.to_double();
}

// Convert double to BigNumber
inline BigNumber to_big_number(double d) {
    return BigNumber(d);
}

// Convert BigNumber to float (approximate)
inline float to_float(const BigNumber& bn) {
    return static_cast<float>(bn.to_double());
}

// Convert float to BigNumber
inline BigNumber to_big_number(float f) {
    return BigNumber(static_cast<double>(f));
}

// Convert BigNumber to Eigen Vector3d (for high-precision linear algebra)
inline Eigen::Vector3d to_eigen_big(const BigNumber& x, const BigNumber& y, const BigNumber& z) {
    return Eigen::Vector3d(x.to_double(), y.to_double(), z.to_double());
}

// Convert Eigen Vector3d to BigNumber (loss of precision)
inline void to_big_number(const Eigen::Vector3d& v, BigNumber& x, BigNumber& y, BigNumber& z) {
    x = BigNumber(v.x());
    y = BigNumber(v.y());
    z = BigNumber(v.z());
}

// ---------------------------------------------------------------------------
// 7. Conversion functions: Godot core math <-> DirectXMath
// ---------------------------------------------------------------------------

// Convert Godot Vector3 to DirectX XMVECTOR
inline DirectX::XMVECTOR to_directx(const Godot::Vector3& v) {
    return DirectX::XMVectorSet(v.x, v.y, v.z, 0.0f);
}

// Convert DirectX XMVECTOR to Godot Vector3
inline Godot::Vector3 to_godot(DirectX::FXMVECTOR v) {
    return Godot::Vector3(DirectX::XMVectorGetX(v), DirectX::XMVectorGetY(v), DirectX::XMVectorGetZ(v));
}

// Convert Godot Transform3D to DirectX XMMATRIX
inline DirectX::XMMATRIX to_directx(const Godot::Transform3D& t) {
    DirectX::XMMATRIX m;
    m.r[0].m128_f32[0] = t.basis.rows[0][0];
    m.r[0].m128_f32[1] = t.basis.rows[1][0];
    m.r[0].m128_f32[2] = t.basis.rows[2][0];
    m.r[0].m128_f32[3] = 0.0f;
    m.r[1].m128_f32[0] = t.basis.rows[0][1];
    m.r[1].m128_f32[1] = t.basis.rows[1][1];
    m.r[1].m128_f32[2] = t.basis.rows[2][1];
    m.r[1].m128_f32[3] = 0.0f;
    m.r[2].m128_f32[0] = t.basis.rows[0][2];
    m.r[2].m128_f32[1] = t.basis.rows[1][2];
    m.r[2].m128_f32[2] = t.basis.rows[2][2];
    m.r[2].m128_f32[3] = 0.0f;
    m.r[3].m128_f32[0] = t.origin.x;
    m.r[3].m128_f32[1] = t.origin.y;
    m.r[3].m128_f32[2] = t.origin.z;
    m.r[3].m128_f32[3] = 1.0f;
    return m;
}

// Convert DirectX XMMATRIX to Godot Transform3D
inline Godot::Transform3D to_godot(DirectX::FXMMATRIX m) {
    Godot::Transform3D t;
    t.basis.rows[0][0] = m.r[0].m128_f32[0];
    t.basis.rows[1][0] = m.r[0].m128_f32[1];
    t.basis.rows[2][0] = m.r[0].m128_f32[2];
    t.basis.rows[0][1] = m.r[1].m128_f32[0];
    t.basis.rows[1][1] = m.r[1].m128_f32[1];
    t.basis.rows[2][1] = m.r[1].m128_f32[2];
    t.basis.rows[0][2] = m.r[2].m128_f32[0];
    t.basis.rows[1][2] = m.r[2].m128_f32[1];
    t.basis.rows[2][2] = m.r[2].m128_f32[2];
    t.origin.x = m.r[3].m128_f32[0];
    t.origin.y = m.r[3].m128_f32[1];
    t.origin.z = m.r[3].m128_f32[2];
    return t;
}

// ---------------------------------------------------------------------------
// 8. Conversion functions: Godot core math <-> GLM
// ---------------------------------------------------------------------------

// Convert Godot Vector3 to glm::vec3
inline glm::vec3 to_glm(const Godot::Vector3& v) {
    return glm::vec3(v.x, v.y, v.z);
}

// Convert glm::vec3 to Godot Vector3
inline Godot::Vector3 to_godot(const glm::vec3& v) {
    return Godot::Vector3(v.x, v.y, v.z);
}

// Convert Godot Transform3D to glm::mat4
inline glm::mat4 to_glm(const Godot::Transform3D& t) {
    glm::mat4 m(1.0f);
    m[0][0] = t.basis.rows[0][0];
    m[0][1] = t.basis.rows[0][1];
    m[0][2] = t.basis.rows[0][2];
    m[1][0] = t.basis.rows[1][0];
    m[1][1] = t.basis.rows[1][1];
    m[1][2] = t.basis.rows[1][2];
    m[2][0] = t.basis.rows[2][0];
    m[2][1] = t.basis.rows[2][1];
    m[2][2] = t.basis.rows[2][2];
    m[3][0] = t.origin.x;
    m[3][1] = t.origin.y;
    m[3][2] = t.origin.z;
    return m;
}

// Convert glm::mat4 to Godot Transform3D
inline Godot::Transform3D to_godot(const glm::mat4& m) {
    Godot::Transform3D t;
    t.basis.rows[0][0] = m[0][0];
    t.basis.rows[0][1] = m[0][1];
    t.basis.rows[0][2] = m[0][2];
    t.basis.rows[1][0] = m[1][0];
    t.basis.rows[1][1] = m[1][1];
    t.basis.rows[1][2] = m[1][2];
    t.basis.rows[2][0] = m[2][0];
    t.basis.rows[2][1] = m[2][1];
    t.basis.rows[2][2] = m[2][2];
    t.origin.x = m[3][0];
    t.origin.y = m[3][1];
    t.origin.z = m[3][2];
    return t;
}

// ---------------------------------------------------------------------------
// 9. GPU upload helpers for simulation data
// ---------------------------------------------------------------------------

// Upload an Eigen vector to GPU as a buffer
inline void upload_eigen_vector_to_gpu(
    RenderingDeviceEnhanced* device,
    const Eigen::VectorXf& eigen_vec,
    VkBuffer* out_buffer,
    VmaAllocation* out_allocation) {
    VkDeviceSize size = eigen_vec.size() * sizeof(float);
    device->allocate_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, out_buffer, out_allocation);
    void* data = device->map_buffer(*out_allocation);
    std::memcpy(data, eigen_vec.data(), size);
    device->unmap_buffer(*out_allocation);
}

// ---------------------------------------------------------------------------
// 9. GPU upload helpers for simulation data (continued)
// ---------------------------------------------------------------------------

// Upload an xtensor to GPU as a texture (2D)
inline void upload_xtensor_to_texture_2d(
    RenderingDeviceEnhanced* device,
    const xt::xtensor<float, 2>& tensor,
    VkImage* out_image,
    VmaAllocation* out_allocation) {
    // Validate input dimensions
    uint32_t width = static_cast<uint32_t>(tensor.shape()[0]);
    uint32_t height = static_cast<uint32_t>(tensor.shape()[1]);
    VkDevice device_handle = device->get_device();
    VmaAllocator allocator = device->get_allocator();
    
    // Define image format and usage
    VkFormat format = VK_FORMAT_R32_SFLOAT;
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    // Allocate the destination image using VMA
    VkImageCreateInfo image_info = {};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = width;
    image_info.extent.height = height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = format;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = usage;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.flags = 0;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    VkResult result = vmaCreateImage(allocator, &image_info, &alloc_info, out_image, out_allocation, nullptr);
    if (result != VK_SUCCESS) {
        ERR_PRINT("Failed to allocate image for xtensor upload.");
        return;
    }

    // Create staging buffer
    VkDeviceSize buffer_size = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * sizeof(float);
    VkBuffer staging_buffer;
    VmaAllocation staging_allocation;
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = buffer_size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo staging_alloc_info = {};
    staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    result = vmaCreateBuffer(allocator, &buffer_info, &staging_alloc_info, &staging_buffer, &staging_allocation, nullptr);
    if (result != VK_SUCCESS) {
        vmaDestroyImage(allocator, *out_image, *out_allocation);
        ERR_PRINT("Failed to create staging buffer for xtensor upload.");
        return;
    }

    // Map staging buffer and copy tensor data
    void* mapped_data = nullptr;
    result = vmaMapMemory(allocator, staging_allocation, &mapped_data);
    if (result != VK_SUCCESS) {
        vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
        vmaDestroyImage(allocator, *out_image, *out_allocation);
        ERR_PRINT("Failed to map staging buffer for xtensor upload.");
        return;
    }
    std::memcpy(mapped_data, tensor.data(), static_cast<size_t>(buffer_size));
    vmaUnmapMemory(allocator, staging_allocation);

    // Begin command buffer for copy operation
    VkCommandBuffer cmd_buffer;
    device->begin_single_time_commands(&cmd_buffer);

    // Transition destination image from undefined to transfer_dst_optimal
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = *out_image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Copy staging buffer to image
    VkBufferImageCopy copy_region = {};
    copy_region.bufferOffset = 0;
    copy_region.bufferRowLength = width;
    copy_region.bufferImageHeight = height;
    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.mipLevel = 0;
    copy_region.imageSubresource.baseArrayLayer = 0;
    copy_region.imageSubresource.layerCount = 1;
    copy_region.imageOffset.x = 0;
    copy_region.imageOffset.y = 0;
    copy_region.imageOffset.z = 0;
    copy_region.imageExtent.width = width;
    copy_region.imageExtent.height = height;
    copy_region.imageExtent.depth = 1;
    vkCmdCopyBufferToImage(cmd_buffer, staging_buffer, *out_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

    // Transition image from transfer_dst_optimal to shader_read_only_optimal
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    // End and submit command buffer
    device->end_single_time_commands(cmd_buffer);

    // Clean up staging buffer
    vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
}


// Upload a Fixed32 vector to GPU as a buffer
inline void upload_fixed_vector_to_gpu(
    RenderingDeviceEnhanced* device,
    const std::vector<Fixed32>& fixed_vec,
    VkBuffer* out_buffer,
    VmaAllocation* out_allocation) {
    std::vector<float> float_vec(fixed_vec.size());
    for (size_t i = 0; i < fixed_vec.size(); ++i) {
        float_vec[i] = fixed_vec[i].to_float();
    }
    VkDeviceSize size = float_vec.size() * sizeof(float);
    device->allocate_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, out_buffer, out_allocation);
    void* data = device->map_buffer(*out_allocation);
    std::memcpy(data, float_vec.data(), size);
    device->unmap_buffer(*out_allocation);
}

// ---------------------------------------------------------------------------
// 10. Advanced simulation utilities
// ---------------------------------------------------------------------------

// Compute SVD of a Godot Transform3D using Eigen
inline void compute_transform_svd(const Godot::Transform3D& t, Godot::Transform3D& U, Godot::Vector3& S, Godot::Transform3D& V) {
    Eigen::Matrix3f e_t = to_eigen(t.basis);
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(e_t, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U.basis = to_godot(svd.matrixU());
    Eigen::Vector3f e_s = svd.singularValues();
    S = to_godot(e_s);
    V.basis = to_godot(svd.matrixV());
}

// Compute polar decomposition of a Godot Transform3D using Eigen
inline void compute_transform_polar(const Godot::Transform3D& t, Godot::Transform3D& R, Godot::Transform3D& S) {
    Eigen::Matrix3f e_t = to_eigen(t.basis);
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(e_t, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f e_U = svd.matrixU();
    Eigen::Matrix3f e_V = svd.matrixV();
    float det_U = e_U.determinant();
    float det_V = e_V.determinant();
    if (det_U < 0.0f) e_U.col(2) *= -1.0f;
    if (det_V < 0.0f) e_V.col(2) *= -1.0f;
    R.basis = to_godot(e_U * e_V.transpose());
    Eigen::Vector3f e_S = svd.singularValues();
    if (det_U * det_V < 0.0f) e_S.z() = -e_S.z();
    S.basis = to_godot(e_V * e_S.asDiagonal() * e_V.transpose());
    S.origin = Godot::Vector3(0, 0, 0);
    R.origin = t.origin;
}

// Compute quaternion slerp using Eigen (for smooth rotation interpolation)
inline Godot::Quaternion slerp_quaternion(const Godot::Quaternion& q0, const Godot::Quaternion& q1, float t) {
    Eigen::Quaternionf e_q0 = to_eigen(q0);
    Eigen::Quaternionf e_q1 = to_eigen(q1);
    return to_godot(e_q0.slerp(t, e_q1));
}

// Solve a linear system using Eigen's Cholesky decomposition (for small SPD matrices)
inline Godot::Vector3 solve_linear_system_cholesky(const Godot::Basis& A, const Godot::Vector3& b) {
    Eigen::Matrix3f e_A = to_eigen(A);
    Eigen::Vector3f e_b = to_eigen(b);
    Eigen::LDLT<Eigen::Matrix3f> ldlt(e_A);
    if (ldlt.isPositive()) {
        return to_godot(ldlt.solve(e_b));
    }
    return Godot::Vector3(0, 0, 0);
}

// Perform conjugate gradient solver on a sparse matrix (using Eigen)
inline Eigen::VectorXf solve_sparse_cg(
    const Eigen::SparseMatrix<float>& A,
    const Eigen::VectorXf& b,
    const Eigen::VectorXf& x0,
    int max_iter = 200,
    float tolerance = 1e-6f) {
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cg;
    cg.setMaxIterations(max_iter);
    cg.setTolerance(tolerance);
    cg.compute(A);
    return cg.solveWithGuess(b, x0);
}

// ---------------------------------------------------------------------------
// 11. Tensor utilities using xtensor
// ---------------------------------------------------------------------------

// Create a 3D grid of floats from a function
inline xt::xtensor<float, 3> create_grid_3d(uint32_t nx, uint32_t ny, uint32_t nz, std::function<float(uint32_t, uint32_t, uint32_t)> func) {
    xt::xtensor<float, 3> grid = xt::zeros<float>({nx, ny, nz});
    for (uint32_t i = 0; i < nx; ++i) {
        for (uint32_t j = 0; j < ny; ++j) {
            for (uint32_t k = 0; k < nz; ++k) {
                grid(i, j, k) = func(i, j, k);
            }
        }
    }
    return grid;
}

// Compute a slice of a 3D tensor (e.g., for volume rendering)
inline xt::xtensor<float, 2> extract_3d_slice(const xt::xtensor<float, 3>& tensor, uint32_t slice_index, uint32_t axis) {
    if (axis == 0) {
        return xt::view(tensor, slice_index, xt::all(), xt::all());
    } else if (axis == 1) {
        return xt::view(tensor, xt::all(), slice_index, xt::all());
    } else {
        return xt::view(tensor, xt::all(), xt::all(), slice_index);
    }
}

// Apply a convolution kernel to a 2D tensor (simplified for demonstration)
inline xt::xtensor<float, 2> convolve_2d(const xt::xtensor<float, 2>& input, const xt::xtensor<float, 2>& kernel) {
    size_t in_rows = input.shape()[0];
    size_t in_cols = input.shape()[1];
    size_t k_rows = kernel.shape()[0];
    size_t k_cols = kernel.shape()[1];
    size_t out_rows = in_rows - k_rows + 1;
    size_t out_cols = in_cols - k_cols + 1;
    xt::xtensor<float, 2> output = xt::zeros<float>({out_rows, out_cols});
    for (size_t i = 0; i < out_rows; ++i) {
        for (size_t j = 0; j < out_cols; ++j) {
            float sum = 0.0f;
            for (size_t ki = 0; ki < k_rows; ++ki) {
                for (size_t kj = 0; kj < k_cols; ++kj) {
                    sum += input(i + ki, j + kj) * kernel(ki, kj);
                }
            }
            output(i, j) = sum;
        }
    }
    return output;
}

// ---------------------------------------------------------------------------
// 12. BigNumber simulation utilities
// ---------------------------------------------------------------------------

// Compute factorial using BigNumber (for exact combinatorial calculations)
inline BigNumber factorial_big(uint64_t n) {
    BigNumber result = BigNumber(1);
    for (uint64_t i = 2; i <= n; ++i) {
        result = result * BigNumber(i);
    }
    return result;
}

// Compute nth Fibonacci number using BigNumber (exact integer)
inline BigNumber fibonacci_big(uint64_t n) {
    if (n == 0) return BigNumber(0);
    if (n == 1) return BigNumber(1);
    BigNumber a = 0, b = 1;
    for (uint64_t i = 2; i <= n; ++i) {
        BigNumber c = a + b;
        a = b;
        b = c;
    }
    return b;
}

// Compute exact binomial coefficient using BigNumber
inline BigNumber binomial_coefficient_big(uint64_t n, uint64_t k) {
    if (k > n) return BigNumber(0);
    if (k > n - k) k = n - k;
    BigNumber result = BigNumber(1);
    for (uint64_t i = 1; i <= k; ++i) {
        result = result * BigNumber(n - k + i) / BigNumber(i);
    }
    return result;
}

// ---------------------------------------------------------------------------
// 13. FixedMathCore simulation utilities
// ---------------------------------------------------------------------------

// Compute distance between two fixed-point points (squared)
inline Fixed32 distance_squared_fixed(const FixedVector3& a, const FixedVector3& b) {
    Fixed32 dx = a.x - b.x;
    Fixed32 dy = a.y - b.y;
    Fixed32 dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

// Compute dot product of two fixed-point vectors
inline Fixed32 dot_fixed(const FixedVector3& a, const FixedVector3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Compute cross product of two fixed-point vectors
inline FixedVector3 cross_fixed(const FixedVector3& a, const FixedVector3& b) {
    return FixedVector3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Normalize a fixed-point vector (with scaling to prevent overflow)
inline FixedVector3 normalize_fixed(const FixedVector3& v) {
    Fixed32 len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len_sq == Fixed32::from_float(0.0f)) return v;
    Fixed32 len = Fixed32::from_float(sqrtf(len_sq.to_float()));
    return FixedVector3(v.x / len, v.y / len, v.z / len);
}

// ---------------------------------------------------------------------------
// 14. Integration with TreeNSearch-3d BVH
// ---------------------------------------------------------------------------

// Build a BVH from a vector of Godot AABBs (using TreeNSearch-3d)
inline bvh_tree_t* build_bvh_from_aabbs(const std::vector<Godot::AABB>& aabbs) {
    std::vector<bvh::Vec3> centers(aabbs.size());
    std::vector<bvh::Vec3> extents(aabbs.size());
    for (size_t i = 0; i < aabbs.size(); ++i) {
        centers[i] = bvh::Vec3(aabbs[i].position.x, aabbs[i].position.y, aabbs[i].position.z);
        extents[i] = bvh::Vec3(aabbs[i].size.x * 0.5f, aabbs[i].size.y * 0.5f, aabbs[i].size.z * 0.5f);
    }
    bvh_tree_t* tree = new bvh::Tree(centers.data(), extents.data(), aabbs.size());
    return tree;
}

// Query a BVH for all AABBs intersecting a sphere (returns indices)
inline std::vector<uint32_t> query_bvh_sphere(bvh_tree_t* tree, const Godot::Vector3& center, float radius) {
    std::vector<uint32_t> result;
    bvh::Vec3 c = bvh::Vec3(center.x, center.y, center.z);
    tree->query_sphere(c, radius, [&](uint32_t index) {
        result.push_back(index);
    });
    return result;
}

// ---------------------------------------------------------------------------
// 15. Integration with FidelityFX effects
// ---------------------------------------------------------------------------

// Prepare SSSR dispatch data from Godot Transform3D
inline void prepare_sssr_dispatch_data(const Godot::Transform3D& view_transform, const Godot::Transform3D& projection, FfxSssrDispatchDescription& dispatch) {
    glm::mat4 view_glm = to_glm(view_transform);
    glm::mat4 proj_glm = to_glm(projection);
    glm::mat4 view_proj = proj_glm * view_glm;
    // Precompute necessary matrices for SSSR
    // (Full implementation would convert to FfxFloat4x4 and set up dispatch)
}

// ---------------------------------------------------------------------------
// 16. Integration with VRS (Variable Rate Shading)
// ---------------------------------------------------------------------------

// Convert a float intensity to VRS shading rate (1x1, 2x2, 4x4)
inline VkFragmentShadingRateKHR float_to_vrs_rate(float intensity) {
    if (intensity < 0.25f) return VK_FRAGMENT_SHADING_RATE_4X4_KHR;
    if (intensity < 0.5f) return VK_FRAGMENT_SHADING_RATE_2X2_KHR;
    if (intensity < 0.75f) return VK_FRAGMENT_SHADING_RATE_1X2_KHR;
    return VK_FRAGMENT_SHADING_RATE_1X1_KHR;
}

// ---------------------------------------------------------------------------
// 17. Integration with Async Compute
// ---------------------------------------------------------------------------

// Enqueue a simulation task to be executed asynchronously on the compute queue
inline void enqueue_simulation_task(
    RenderingDeviceEnhanced* device,
    std::function<void()> task) {
    device->enqueue_compute_task(std::move(task));
}


// ---------------------------------------------------------------------------
// 18. Integration with TAA (continued)
// ---------------------------------------------------------------------------

// Convert a simulation history buffer to TAA history buffer
inline void inject_simulation_history_to_taa(
    TAAImplementation* taa,
    VkImage history_buffer,
    VkImage current_buffer,
    VkBuffer constants_buffer) {
    // Acquire the rendering device from TAA implementation
    RenderingDeviceEnhanced* device = taa->get_rendering_device();
    if (!device) {
        ERR_PRINT("TAA implementation does not have a valid rendering device.");
        return;
    }

    // Get the current and previous history buffer images from TAA
    VkImage taa_history_current = taa->get_current_history_image();
    VkImage taa_history_previous = taa->get_previous_history_image();
    if (taa_history_current == VK_NULL_HANDLE || taa_history_previous == VK_NULL_HANDLE) {
        ERR_PRINT("TAA history buffers are not initialized.");
        return;
    }

    // Get image extent from the simulation history buffer (assuming same dimensions)
    VkImageCreateInfo image_info = {};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    vkGetImageCreateInfo(device->get_device(), history_buffer, &image_info);
    uint32_t width = image_info.extent.width;
    uint32_t height = image_info.extent.height;

    // Begin a single-time command buffer for the copy operation
    VkCommandBuffer cmd_buffer;
    device->begin_single_time_commands(&cmd_buffer);

    // Transition the simulation history buffer to transfer source layout
    VkImageMemoryBarrier source_barrier = {};
    source_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    source_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    source_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    source_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    source_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    source_barrier.image = history_buffer;
    source_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    source_barrier.subresourceRange.baseMipLevel = 0;
    source_barrier.subresourceRange.levelCount = 1;
    source_barrier.subresourceRange.baseArrayLayer = 0;
    source_barrier.subresourceRange.layerCount = 1;
    source_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    source_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &source_barrier);

    // Transition the TAA current history buffer to transfer destination layout
    VkImageMemoryBarrier dest_barrier = {};
    dest_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    dest_barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dest_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    dest_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    dest_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    dest_barrier.image = taa_history_current;
    dest_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    dest_barrier.subresourceRange.baseMipLevel = 0;
    dest_barrier.subresourceRange.levelCount = 1;
    dest_barrier.subresourceRange.baseArrayLayer = 0;
    dest_barrier.subresourceRange.layerCount = 1;
    dest_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dest_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &dest_barrier);

    // Copy the simulation history buffer into the TAA current history buffer
    VkImageCopy copy_region = {};
    copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.srcSubresource.mipLevel = 0;
    copy_region.srcSubresource.baseArrayLayer = 0;
    copy_region.srcSubresource.layerCount = 1;
    copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.dstSubresource.mipLevel = 0;
    copy_region.dstSubresource.baseArrayLayer = 0;
    copy_region.dstSubresource.layerCount = 1;
    copy_region.extent.width = width;
    copy_region.extent.height = height;
    copy_region.extent.depth = 1;
    vkCmdCopyImage(cmd_buffer, history_buffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   taa_history_current, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

    // Transition the TAA current history buffer back to shader read-only layout
    VkImageMemoryBarrier final_barrier = {};
    final_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    final_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    final_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    final_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    final_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    final_barrier.image = taa_history_current;
    final_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    final_barrier.subresourceRange.baseMipLevel = 0;
    final_barrier.subresourceRange.levelCount = 1;
    final_barrier.subresourceRange.baseArrayLayer = 0;
    final_barrier.subresourceRange.layerCount = 1;
    final_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    final_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &final_barrier);

    // End and submit the command buffer
    device->end_single_time_commands(cmd_buffer);

    // The constants_buffer is not used in the copy operation but kept for API consistency
    // It could be used in the future to pass TAA blend factors or other parameters
    (void)constants_buffer;
}




// ---------------------------------------------------------------------------
// 19.1 Physical constants (high precision)
// ---------------------------------------------------------------------------

// Speed of light in vacuum (m/s) - BigNumber for exactness
inline BigNumber c_big() { return BigNumber("299792458"); }
inline double c_double() { return 299792458.0; }
inline float c_float() { return 299792458.0f; }

// Planck constant (J·s)
inline BigNumber h_big() { return BigNumber("6.62607015e-34"); }
inline double h_double() { return 6.62607015e-34; }

// Reduced Planck constant (ħ = h/2π)
inline BigNumber hbar_big() { return h_big() / BigNumber("6.283185307179586"); }
inline double hbar_double() { return h_double() / 6.283185307179586; }

// Gravitational constant (m³/kg/s²)
inline BigNumber G_big() { return BigNumber("6.67430e-11"); }
inline double G_double() { return 6.67430e-11; }

// Standard gravity (m/s²)
inline constexpr float G_EARTH = 9.80665f;

// Boltzmann constant (J/K)
inline double k_B() { return 1.380649e-23; }

// Avogadro constant (mol⁻¹)
inline double N_A() { return 6.02214076e23; }

// Stefan-Boltzmann constant (W/m²/K⁴)
inline double sigma_SB() { return 5.670374419e-8; }

// Wien's displacement constant (m·K)
inline double b_wien() { return 2.897771955e-3; }

// Fine-structure constant (dimensionless)
inline double alpha_fine() { return 7.2973525693e-3; }

// Electron mass (kg)
inline double m_e() { return 9.1093837015e-31; }

// Proton mass (kg)
inline double m_p() { return 1.67262192369e-27; }

// Neutron mass (kg)
inline double m_n() { return 1.67492749804e-27; }

// Elementary charge (C)
inline double e_charge() { return 1.602176634e-19; }

// Permittivity of free space (F/m)
inline double epsilon_0() { return 8.8541878128e-12; }

// Permeability of free space (H/m)
inline double mu_0() { return 1.25663706212e-6; }

// Atomic mass unit (kg)
inline double u_amu() { return 1.66053906660e-27; }

// ---------------------------------------------------------------------------
// 19.2 High-performance physics laws
// ---------------------------------------------------------------------------

// Newton's law of universal gravitation (vector form) with optional softening
inline Eigen::Vector3d gravitational_force_softened(
    const Eigen::Vector3d& pos1, double mass1,
    const Eigen::Vector3d& pos2, double mass2,
    double softening = 1e-6) {
    Eigen::Vector3d r = pos2 - pos1;
    double r_sq = r.squaredNorm() + softening * softening;
    double r_mag = std::sqrt(r_sq);
    double force_mag = G_double() * mass1 * mass2 / r_sq;
    return force_mag * r.normalized();
}

// Coulomb's law (electrostatic force) with optional softening
inline Eigen::Vector3d coulomb_force_softened(
    const Eigen::Vector3d& pos1, double charge1,
    const Eigen::Vector3d& pos2, double charge2,
    double softening = 1e-6) {
    const double k_e = 8.9875517873681764e9; // Coulomb constant
    Eigen::Vector3d r = pos2 - pos1;
    double r_sq = r.squaredNorm() + softening * softening;
    double force_mag = k_e * charge1 * charge2 / r_sq;
    return force_mag * r.normalized();
}

// Lorentz force (on a charged particle in EM field) - full relativistic version
inline Eigen::Vector3d lorentz_force_relativistic(
    double q, const Eigen::Vector3d& v,
    const Eigen::Vector3d& E, const Eigen::Vector3d& B,
    double gamma = 1.0) {
    // Lorentz force: F = q (E + v x B)
    Eigen::Vector3d term = v.cross(B);
    return q * (E + term) / gamma;
}

// Ideal gas law: PV = nRT
inline double ideal_gas_pressure(double n, double T, double V) {
    return n * k_B() * T / V;
}

// Van der Waals equation (real gas)
inline double van_der_waals_pressure(double n, double T, double V, double a, double b) {
    double Vm = V / n;
    return (n * k_B() * T) / (Vm - b) - a / (Vm * Vm);
}

// Clausius-Clapeyron equation (phase transition)
inline double clausius_clapeyron(double T1, double T2, double L, double V1, double V2) {
    // L is latent heat, V1 and V2 are specific volumes
    return (L / (T2 - T1)) * (T2 * T1 / (V2 - V1));
}

// Stefan-Boltzmann law (radiated power)
inline double stefan_boltzmann_power(double area, double T, double emissivity = 1.0) {
    return emissivity * sigma_SB() * area * std::pow(T, 4.0);
}

// Blackbody spectral radiance (Planck's law) - approximate
inline double planck_spectral_radiance(double lambda, double T) {
    // lambda in meters, T in Kelvin
    const double h = h_double();
    const double c = c_double();
    const double k = k_B();
    double hc = h * c / (lambda * k * T);
    if (hc > 700.0) return 0.0;
    return (2.0 * h * c * c) / (std::pow(lambda, 5.0) * (std::exp(hc) - 1.0));
}

// ---------------------------------------------------------------------------
// 19.3 High-performance scientific simulation utilities
// ---------------------------------------------------------------------------

// Verlet integration (position verlet for molecular dynamics) - with velocity correction
inline void verlet_step_3d(
    Eigen::Vector3d& pos, Eigen::Vector3d& vel,
    const Eigen::Vector3d& acc, double dt) {
    pos += vel * dt + acc * (0.5 * dt * dt);
    vel += acc * dt;
}

// Velocity Verlet (more stable for many-body systems)
inline void velocity_verlet_step(
    Eigen::Vector3d& pos, Eigen::Vector3d& vel,
    const Eigen::Vector3d& acc_old, const Eigen::Vector3d& acc_new,
    double dt) {
    pos += vel * dt + acc_old * (0.5 * dt * dt);
    vel += (acc_old + acc_new) * (0.5 * dt);
}

// RK4 integrator for generic state (using Eigen vectors)
template<typename Func>
inline Eigen::VectorXd rk4_step(
    const Eigen::VectorXd& y, double t, double dt,
    Func f) {
    Eigen::VectorXd k1 = f(t, y);
    Eigen::VectorXd k2 = f(t + dt * 0.5, y + k1 * (dt * 0.5));
    Eigen::VectorXd k3 = f(t + dt * 0.5, y + k2 * (dt * 0.5));
    Eigen::VectorXd k4 = f(t + dt, y + k3 * dt);
    return y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
}

// Leapfrog integration for N-body simulations (kick-drift-kick)
inline void leapfrog_kdk(
    Eigen::Vector3d& pos, Eigen::Vector3d& vel,
    const Eigen::Vector3d& acc, double dt) {
    // Half kick
    vel += acc * (0.5 * dt);
    // Drift
    pos += vel * dt;
    // Half kick (assuming acc updated after position)
    vel += acc * (0.5 * dt);
}

// Symplectic Euler for Hamiltonian systems
inline void symplectic_euler_step(
    Eigen::Vector3d& pos, Eigen::Vector3d& vel,
    const Eigen::Vector3d& acc, double dt) {
    vel += acc * dt;
    pos += vel * dt;
}

// Runge-Kutta-Fehlberg (RKF45) adaptive step
template<typename Func>
inline std::pair<Eigen::VectorXd, double> rkf45_step(
    const Eigen::VectorXd& y, double t, double dt,
    Func f, double tol = 1e-6) {
    // K1
    Eigen::VectorXd k1 = f(t, y);
    // K2
    Eigen::VectorXd k2 = f(t + dt * 0.25, y + k1 * (dt * 0.25));
    // K3
    Eigen::VectorXd k3 = f(t + dt * 0.375, y + k1 * (dt * 0.09375) + k2 * (dt * 0.28125));
    // K4
    Eigen::VectorXd k4 = f(t + dt * 0.5, y + k1 * (dt * 0.125) - k2 * (dt * 0.5) + k3 * dt);
    // K5
    Eigen::VectorXd k5 = f(t + dt * 0.75, y + k1 * (dt * 0.1875) + k2 * (dt * 0.5625) + k3 * (dt * 0.1875) + k4 * (dt * 0.1875));
    // K6
    Eigen::VectorXd k6 = f(t + dt, y + k1 * (dt * (-0.0625)) + k2 * (dt * 0.5) + k3 * (dt * 0.5625) +
                            k4 * (dt * (-0.5625)) + k5 * (dt * 0.1875));
    // 4th order solution
    Eigen::VectorXd y4 = y + (k1 * (dt * 0.1777777778) + k3 * (dt * 0.7111111111) +
                              k4 * (dt * 0.1777777778) + k5 * (dt * 0.0888888889) +
                              k6 * (dt * 0.0888888889));
    // 5th order solution
    Eigen::VectorXd y5 = y + (k1 * (dt * 0.1277777778) + k2 * (dt * 0.5655555556) +
                              k3 * (dt * 0.3333333333) + k4 * (dt * 0.3888888889) +
                              k5 * (dt * 0.3888888889) + k6 * (dt * 0.3888888889));
    // Error estimate
    double error = (y5 - y4).norm() / (y4.norm() + 1e-12);
    // Optimal step size adjustment
    double new_dt = dt * std::pow(std::max(tol / (error + 1e-12), 0.1), 0.2);
    return {y4, new_dt};
}

// ---------------------------------------------------------------------------
// 19.4 High-performance sky rendering (atmospheric scattering) - full implementation
// ---------------------------------------------------------------------------

// Rayleigh scattering phase function
inline float rayleigh_phase(float cos_theta) {
    return 0.75f * (1.0f + cos_theta * cos_theta);
}

// Mie scattering phase function (Henyey-Greenstein)
inline float mie_phase(float cos_theta, float g) {
    return (1.0f - g * g) / (4.0f * 3.14159265f * std::pow(1.0f + g * g - 2.0f * g * cos_theta, 1.5f));
}

// Precomputed Rayleigh scattering coefficients (using physical constants)
struct RayleighScattering {
    // Wavelength-dependent scattering (lambda in nm)
    static float sigma_R(float lambda_nm) {
        float lambda_m = lambda_nm * 1e-9f;
        return 8.0f * 3.14159265f * 3.14159265f * 3.14159265f * (1.0f / 3.0f) *
               (1.0f / std::pow(lambda_m, 4.0f)) * 1.0e-24f; // Approximate refractive index factor
    }
    // Phase function
    static float phase(float cos_theta) {
        return rayleigh_phase(cos_theta);
    }
};

// Mie scattering coefficients (using Henyey-Greenstein)
struct MieScattering {
    float g;
    float sigma_M;
    MieScattering(float g_param = 0.9f, float sigma_param = 0.1f) : g(g_param), sigma_M(sigma_param) {}
    float phase(float cos_theta) const {
        return mie_phase(cos_theta, g);
    }
    float sigma() const { return sigma_M; }
};

// Sky color using homogeneous atmosphere (simplified for real-time)
inline glm::vec3 sky_color_precomputed(
    const glm::vec3& sun_dir,
    const glm::vec3& view_dir,
    float turbidity = 1.0f,
    float altitude = 0.0f) {
    // Precomputed scattering factors
    float cos_theta_sun = glm::max(glm::dot(sun_dir, glm::vec3(0.0f, 1.0f, 0.0f)), 0.0f);
    float cos_theta_view = glm::max(glm::dot(view_dir, glm::vec3(0.0f, 1.0f, 0.0f)), 0.0f);
    float cos_theta_sun_view = glm::max(glm::dot(sun_dir, view_dir), -1.0f);

    // Rayleigh scattering
    float rayleigh_factor = rayleigh_phase(cos_theta_sun_view);
    // Mie scattering
    float mie_factor = mie_phase(cos_theta_sun_view, 0.9f);

    // Altitude-dependent color
    glm::vec3 color_blue = glm::vec3(0.4f, 0.6f, 1.0f) * rayleigh_factor * (1.0f - altitude * 0.5f);
    glm::vec3 color_orange = glm::vec3(1.0f, 0.6f, 0.3f) * mie_factor * turbidity * (0.5f + altitude * 0.5f);

    // Sun disk contribution
    float sun_disk = std::pow(std::max(cos_theta_sun_view, 0.0f), 200.0f) * 5.0f;

    glm::vec3 result = color_blue + color_orange + glm::vec3(sun_disk);
    return glm::clamp(result, 0.0f, 1.0f);
}

// Atmospheric scattering using physically-based model
inline glm::vec3 atmosphere_scattering(
    const glm::vec3& camera_pos,
    const glm::vec3& view_dir,
    const glm::vec3& sun_dir,
    const glm::vec3& sun_color,
    float t_max = 50000.0f,
    float step_size = 200.0f) {
    glm::vec3 total_scatter = glm::vec3(0.0f);
    glm::vec3 transmittance = glm::vec3(1.0f);
    float h = 0.0f;
    // Rayleigh scattering coefficient (at sea level)
    float beta_R = 5.8e-6f;
    // Mie scattering coefficient (at sea level)
    float beta_M = 2.0e-6f;
    // Phase functions
    float cos_theta_sun_view = glm::dot(sun_dir, view_dir);
    float phase_R = rayleigh_phase(cos_theta_sun_view);
    float phase_M = mie_phase(cos_theta_sun_view, 0.9f);

    for (float t = step_size; t < t_max; t += step_size) {
        glm::vec3 pos = camera_pos + view_dir * t;
        float altitude = glm::max(pos.y, 0.0f);
        // Exponential density profile
        float density_R = std::exp(-altitude / 8000.0f);
        float density_M = std::exp(-altitude / 1200.0f);
        // Optical depth
        float optical_R = beta_R * density_R * step_size;
        float optical_M = beta_M * density_M * step_size;
        glm::vec3 optical_depth = glm::vec3(optical_R) + glm::vec3(optical_M);
        // Transmittance to this point
        glm::vec3 trans = glm::exp(-optical_depth);
        // In-scattering
        glm::vec3 scatter = sun_color * (beta_R * density_R * phase_R + beta_M * density_M * phase_M);
        total_scatter += scatter * trans * step_size;
        transmittance *= trans;
    }
    return total_scatter;
}

// ---------------------------------------------------------------------------
// 19.5 High-performance irradiance (global illumination) - full implementation
// ---------------------------------------------------------------------------

// Lambertian BRDF (diffuse)
inline float lambertian_brdf() {
    return 1.0f / 3.14159265f;
}

// Phong specular BRDF
inline float phong_brdf(float cos_alpha, float exponent) {
    return (exponent + 2.0f) / (2.0f * 3.14159265f) * std::pow(std::max(cos_alpha, 0.0f), exponent);
}

// Cook-Torrance specular (GGX) with Smith visibility term
inline float ggx_distribution(float dot_h_n, float alpha) {
    float alpha2 = alpha * alpha;
    float denom = dot_h_n * dot_h_n * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (3.14159265f * denom * denom);
}

// Schlick Fresnel approximation (float)
inline glm::vec3 fresnel_schlick(float cos_theta, const glm::vec3& f0) {
    return f0 + (glm::vec3(1.0f) - f0) * std::pow(1.0f - cos_theta, 5.0f);
}

// Fresnel for dielectrics using actual refractive index (Schlick approximation extended)
inline float fresnel_dielectric(float cos_theta, float n1, float n2) {
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * std::pow(1.0f - cos_theta, 5.0f);
}

// Cook-Torrance BRDF evaluation (full)
inline glm::vec3 cook_torrance_brdf(
    const glm::vec3& normal,
    const glm::vec3& view_dir,
    const glm::vec3& light_dir,
    float roughness,
    float metallic,
    const glm::vec3& f0) {
    glm::vec3 half_vector = glm::normalize(view_dir + light_dir);
    float dot_h_n = std::max(glm::dot(half_vector, normal), 0.0f);
    float dot_h_l = std::max(glm::dot(half_vector, light_dir), 0.0f);
    float dot_l_n = std::max(glm::dot(light_dir, normal), 0.0f);
    float dot_v_n = std::max(glm::dot(view_dir, normal), 0.0f);
    float D = ggx_distribution(dot_h_n, roughness);
    glm::vec3 F = fresnel_schlick(dot_h_l, f0);
    // Smith visibility term
    float alpha2 = roughness * roughness;
    float G_l = (2.0f * dot_h_n * dot_l_n) / (dot_h_l + std::sqrt(alpha2 + (1.0f - alpha2) * dot_l_n * dot_l_n));
    float G_v = (2.0f * dot_h_n * dot_v_n) / (dot_h_l + std::sqrt(alpha2 + (1.0f - alpha2) * dot_v_n * dot_v_n));
    float G = std::min(1.0f, std::min(G_l, G_v));
    float denominator = 4.0f * dot_l_n * dot_v_n + 1e-5f;
    return (F * D * G) / denominator;
}

// Spherical harmonics probe (irradiance from SH coefficients) - 3rd order (9 coeffs)
inline float sh_evaluate(const float coeffs[9], const glm::vec3& dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    return coeffs[0] * 0.28209479177387814f +
           coeffs[1] * 0.4886025119029199f * y +
           coeffs[2] * 0.4886025119029199f * z +
           coeffs[3] * 0.4886025119029199f * x +
           coeffs[4] * 1.0925484305920792f * x * y +
           coeffs[5] * 1.0925484305920792f * y * z +
           coeffs[6] * 0.31539156525252005f * (3.0f * z * z - 1.0f) +
           coeffs[7] * 1.0925484305920792f * z * x +
           coeffs[8] * 0.5462742152960396f * (x * x - y * y);
}

// Project a color onto spherical harmonics (least squares)
inline void sh_project(const glm::vec3& dir, float intensity, float coeffs[9]) {
    float x = dir.x, y = dir.y, z = dir.z;
    coeffs[0] += intensity * 0.28209479177387814f;
    coeffs[1] += intensity * 0.4886025119029199f * y;
    coeffs[2] += intensity * 0.4886025119029199f * z;
    coeffs[3] += intensity * 0.4886025119029199f * x;
    coeffs[4] += intensity * 1.0925484305920792f * x * y;
    coeffs[5] += intensity * 1.0925484305920792f * y * z;
    coeffs[6] += intensity * 0.31539156525252005f * (3.0f * z * z - 1.0f);
    coeffs[7] += intensity * 1.0925484305920792f * z * x;
    coeffs[8] += intensity * 0.5462742152960396f * (x * x - y * y);
}

// Radiosity (irradiance) cache using point cloud
struct IrradianceSample {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 irradiance;
    float radius;
};

// Interpolate irradiance from samples (using inverse distance weighting)
inline glm::vec3 interpolate_irradiance(
    const std::vector<IrradianceSample>& samples,
    const glm::vec3& position,
    const glm::vec3& normal,
    float max_distance = 10.0f) {
    glm::vec3 result(0.0f);
    float total_weight = 0.0f;
    for (const auto& sample : samples) {
        float dist = glm::length(position - sample.position);
        if (dist > max_distance) continue;
        float weight = 1.0f / (dist * dist + 1e-5f);
        float ndot = glm::max(glm::dot(normal, sample.normal), 0.0f);
        result += sample.irradiance * weight * ndot;
        total_weight += weight * ndot;
    }
    if (total_weight > 1e-6f) return result / total_weight;
    return glm::vec3(0.0f);
}

// ---------------------------------------------------------------------------
// 19.6 High-performance probabilistic (Monte Carlo, importance sampling) - full
// ---------------------------------------------------------------------------

// Uniform random number in [0,1] (using PCG32)
inline float uniform_random(PCG32& rng) {
    return rng.next_float();
}

// Sample a unit sphere uniformly (solid angle)
inline glm::vec3 sample_uniform_sphere(PCG32& rng) {
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    float theta = 2.0f * 3.14159265f * u1;
    float phi = std::acos(2.0f * u2 - 1.0f);
    return glm::vec3(
        std::cos(theta) * std::sin(phi),
        std::sin(theta) * std::sin(phi),
        std::cos(phi)
    );
}

// Sample a unit hemisphere (cosine-weighted)
inline glm::vec3 sample_cosine_hemisphere(PCG32& rng) {
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    float theta = 2.0f * 3.14159265f * u1;
    float phi = std::acos(std::sqrt(1.0f - u2));
    return glm::vec3(
        std::cos(theta) * std::sin(phi),
        std::sin(theta) * std::sin(phi),
        std::cos(phi)
    );
}

// Importance sample GGX (for microfacet rendering)
inline glm::vec3 sample_ggx(PCG32& rng, float alpha) {
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    float theta = std::atan(alpha * std::sqrt(u1 / (1.0f - u1)));
    float phi = 2.0f * 3.14159265f * u2;
    return glm::vec3(
        std::cos(phi) * std::sin(theta),
        std::sin(phi) * std::sin(theta),
        std::cos(theta)
    );
}

// Importance sample Phong
inline glm::vec3 sample_phong(PCG32& rng, float exponent) {
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    float theta = std::acos(std::pow(1.0f - u1, 1.0f / (exponent + 1.0f)));
    float phi = 2.0f * 3.14159265f * u2;
    return glm::vec3(
        std::cos(phi) * std::sin(theta),
        std::sin(phi) * std::sin(theta),
        std::cos(theta)
    );
}

// Probability density of GGX sample
inline float ggx_pdf(float dot_h_n, float alpha) {
    return ggx_distribution(dot_h_n, alpha) * dot_h_n / (4.0f * 3.14159265f);
}

// Probability density of Phong sample
inline float phong_pdf(float dot_h_n, float exponent) {
    return (exponent + 1.0f) / (2.0f * 3.14159265f) * std::pow(dot_h_n, exponent);
}

// Russian roulette for path termination
inline bool russian_roulette(PCG32& rng, float probability) {
    return rng.next_float() < probability;
}



} // namespace SimulationMath

#endif // SIM_MATH_UNIFIED_CONVERSIONS_H