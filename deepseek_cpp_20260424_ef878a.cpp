// genesis/grad/tensor.cpp

#include "genesis/grad/tensor.h"                    // Include corresponding header
#include "genesis/grad/__init__.h"                  // For device management
#include <stdexcept>                                // std::runtime_error, std::invalid_argument
#include <cstring>                                  // std::memcpy, std::memset
#include <algorithm>                                // std::copy, std::fill
#include <numeric>                                  // std::accumulate
#include <cmath>                                    // std::isfinite, etc.
#include <sstream>                                  // std::ostringstream for error messages

#ifdef GENESIS_USE_CUDA
#include <cuda_runtime.h>                           // CUDA runtime API
#endif

#ifdef GENESIS_USE_METAL
#include <Metal/Metal.h>                            // Metal framework (Objective-C++)
#endif

#ifdef GENESIS_USE_HIP
#include <hip/hip_runtime.h>                        // HIP runtime API
#endif

namespace genesis {
namespace grad {

//------------------------------------------------------------------------------
// Static member initialization
//------------------------------------------------------------------------------
std::atomic<uint64_t> Tensor::next_id_{1};           // Start IDs from 1

//------------------------------------------------------------------------------
// dtype_size implementation
//------------------------------------------------------------------------------
size_t dtype_size(DType dtype) {
    // Return size in bytes for given data type
    switch (dtype) {
        case DType::FLOAT32: return sizeof(float);   // 4 bytes
        case DType::FLOAT64: return sizeof(double);  // 8 bytes
        case DType::INT32:   return sizeof(int32_t); // 4 bytes
        case DType::INT64:   return sizeof(int64_t); // 8 bytes
        case DType::BOOL:    return sizeof(bool);    // 1 byte
        case DType::UINT8:   return sizeof(uint8_t); // 1 byte
        default:             return 0;               // Unknown type
    }
}

//------------------------------------------------------------------------------
// Helper: allocate memory on specified device
//------------------------------------------------------------------------------
static void* allocate_on_device(size_t size, Device device) {
    // Allocate memory on the specified device
    void* ptr = nullptr;
    switch (device.type) {
        case DeviceType::CPU:
            ptr = std::malloc(size);                 // Standard malloc for CPU
            if (!ptr) throw std::bad_alloc();        // Allocation failed
            break;
#ifdef GENESIS_USE_CUDA
        case DeviceType::CUDA: {
            cudaError_t err = cudaSetDevice(device.index); // Set current CUDA device
            if (err != cudaSuccess) throw std::runtime_error("cudaSetDevice failed");
            err = cudaMalloc(&ptr, size);            // Allocate GPU memory
            if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
            break;
        }
#endif
#ifdef GENESIS_USE_METAL
        case DeviceType::METAL: {
            // Metal allocation using MTLBuffer
            id<MTLDevice> mtl_device = get_metal_device(device.index); // Helper to get device
            if (!mtl_device) throw std::runtime_error("Metal device not available");
            // Use shared storage mode for CPU access (can be changed to private for performance)
            MTLResourceOptions options = MTLResourceStorageModeShared;
            id<MTLBuffer> buffer = [mtl_device newBufferWithLength:size options:options];
            if (!buffer) throw std::bad_alloc();
            ptr = (__bridge_retained void*)buffer;   // Retain and return as void*
            break;
        }
#endif
#ifdef GENESIS_USE_HIP
        case DeviceType::AMD: {
            hipError_t err = hipSetDevice(device.index);
            if (err != hipSuccess) throw std::runtime_error("hipSetDevice failed");
            err = hipMalloc(&ptr, size);
            if (err != hipSuccess) throw std::runtime_error("hipMalloc failed");
            break;
        }
#endif
        default:
            throw std::runtime_error("Unsupported device type for allocation");
    }
    return ptr;                                      // Return allocated pointer
}

//------------------------------------------------------------------------------
// Helper: free memory on specified device
//------------------------------------------------------------------------------
static void free_on_device(void* ptr, size_t size, Device device) {
    // Free memory allocated on the specified device
    if (!ptr) return;
    switch (device.type) {
        case DeviceType::CPU:
            std::free(ptr);                          // Standard free
            break;
#ifdef GENESIS_USE_CUDA
        case DeviceType::CUDA:
            cudaFree(ptr);                           // CUDA free
            break;
#endif
#ifdef GENESIS_USE_METAL
        case DeviceType::METAL: {
            id<MTLBuffer> buffer = (__bridge_transfer id)ptr; // Transfer ownership for release
            [buffer release];                        // Release Metal buffer
            break;
        }
#endif
#ifdef GENESIS_USE_HIP
        case DeviceType::AMD:
            hipFree(ptr);                            // HIP free
            break;
#endif
        default:
            // Unknown device type, no-op or warning
            break;
    }
    (void)size;                                      // Size not used for some backends
}

//------------------------------------------------------------------------------
// Helper: copy data between devices (simplified: always via host)
//------------------------------------------------------------------------------
static void copy_between_devices(void* dst, const void* src, size_t size,
                                 Device dst_device, Device src_device) {
    // Copy data from src_device to dst_device (via host staging buffer)
    if (dst_device.type == DeviceType::CPU && src_device.type == DeviceType::CPU) {
        std::memcpy(dst, src, size);                 // Direct CPU copy
        return;
    }
    // Allocate host staging buffer
    void* host_buffer = std::malloc(size);
    if (!host_buffer) throw std::bad_alloc();
    // Copy from source to host
    if (src_device.type == DeviceType::CPU) {
        std::memcpy(host_buffer, src, size);
    } else {
        // Device to host copy (backend-specific)
#ifdef GENESIS_USE_CUDA
        if (src_device.type == DeviceType::CUDA) {
            cudaMemcpy(host_buffer, src, size, cudaMemcpyDeviceToHost);
        } else
#endif
#ifdef GENESIS_USE_METAL
        if (src_device.type == DeviceType::METAL) {
            id<MTLBuffer> buffer = (__bridge id)src;
            std::memcpy(host_buffer, [buffer contents], size);
        } else
#endif
#ifdef GENESIS_USE_HIP
        if (src_device.type == DeviceType::AMD) {
            hipMemcpy(host_buffer, src, size, hipMemcpyDeviceToHost);
        } else
#endif
        {
            std::free(host_buffer);
            throw std::runtime_error("Unsupported device-to-host copy");
        }
    }
    // Copy from host to destination
    if (dst_device.type == DeviceType::CPU) {
        std::memcpy(dst, host_buffer, size);
    } else {
#ifdef GENESIS_USE_CUDA
        if (dst_device.type == DeviceType::CUDA) {
            cudaMemcpy(dst, host_buffer, size, cudaMemcpyHostToDevice);
        } else
#endif
#ifdef GENESIS_USE_METAL
        if (dst_device.type == DeviceType::METAL) {
            id<MTLBuffer> buffer = (__bridge id)dst;
            std::memcpy([buffer contents], host_buffer, size);
        } else
#endif
#ifdef GENESIS_USE_HIP
        if (dst_device.type == DeviceType::AMD) {
            hipMemcpy(dst, host_buffer, size, hipMemcpyHostToDevice);
        } else
#endif
        {
            std::free(host_buffer);
            throw std::runtime_error("Unsupported host-to-device copy");
        }
    }
    std::free(host_buffer);                          // Free staging buffer
}

//------------------------------------------------------------------------------
// Tensor: Construction
//------------------------------------------------------------------------------
Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype, Device device)
    : id_(next_id_++)                                // Assign unique ID
    , shape_(shape)                                  // Copy shape
    , dtype_(dtype)                                  // Set data type
    , device_(device)                                // Set device
    , data_(nullptr)                                 // Will allocate
    , owns_data_(true)                               // We own the data
    , requires_grad_(false)                          // Default no grad tracking
{
    // Validate shape
    numel_ = 1;
    for (auto d : shape_) {
        if (d <= 0) {
            throw std::invalid_argument("Tensor dimensions must be positive");
        }
        numel_ *= d;                                 // Compute total elements
    }
    compute_strides();                               // Compute strides (row-major)
    allocate_data();                                 // Allocate memory on device
    // Initialize to zero for safety
    zero_();
}

Tensor::Tensor(const void* data, const std::vector<int64_t>& shape, DType dtype, Device device)
    : id_(next_id_++)
    , shape_(shape)
    , dtype_(dtype)
    , device_(device)
    , data_(nullptr)
    , owns_data_(true)
    , requires_grad_(false)
{
    numel_ = 1;
    for (auto d : shape_) {
        if (d <= 0) throw std::invalid_argument("Tensor dimensions must be positive");
        numel_ *= d;
    }
    compute_strides();
    allocate_data();
    // Copy provided data to device
    copy_from_host(data);
}

Tensor::Tensor(const Tensor& other)
    : id_(next_id_++)
    , shape_(other.shape_)
    , strides_(other.strides_)
    , numel_(other.numel_)
    , dtype_(other.dtype_)
    , device_(other.device_)
    , data_(nullptr)
    , owns_data_(true)
    , requires_grad_(other.requires_grad_)
    , grad_fn_(other.grad_fn_)                       // Shallow copy grad_fn (shared)
{
    // Copy inputs (weak pointers)
    for (const auto& weak_in : other.inputs_) {
        inputs_.push_back(weak_in);                  // Copy weak reference
    }
    allocate_data();
    // Copy data from other tensor
    copy_between_devices(data_, other.data_, nbytes(), device_, other.device_);
    // Copy gradient if present
    if (other.grad_) {
        grad_ = std::make_shared<Tensor>(*other.grad_); // Deep copy gradient
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : id_(other.id_)
    , shape_(std::move(other.shape_))
    , strides_(std::move(other.strides_))
    , numel_(other.numel_)
    , dtype_(other.dtype_)
    , device_(other.device_)
    , data_(other.data_)
    , owns_data_(other.owns_data_)
    , data_owner_(std::move(other.data_owner_))
    , requires_grad_(other.requires_grad_)
    , grad_(std::move(other.grad_))
    , grad_fn_(std::move(other.grad_fn_))
    , inputs_(std::move(other.inputs_))
{
    // Invalidate source
    other.data_ = nullptr;
    other.owns_data_ = false;
    other.numel_ = 0;
}

Tensor::~Tensor() {
    // Free allocated memory if we own it
    deallocate_data();
}

Tensor& Tensor::operator=(const Tensor& other) {
    // Copy assignment
    if (this != &other) {
        deallocate_data();                           // Free current data
        shape_ = other.shape_;
        strides_ = other.strides_;
        numel_ = other.numel_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        requires_grad_ = other.requires_grad_;
        grad_fn_ = other.grad_fn_;
        inputs_.clear();
        for (const auto& weak_in : other.inputs_) {
            inputs_.push_back(weak_in);
        }
        allocate_data();
        copy_between_devices(data_, other.data_, nbytes(), device_, other.device_);
        if (other.grad_) {
            grad_ = std::make_shared<Tensor>(*other.grad_);
        } else {
            grad_.reset();
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    // Move assignment
    if (this != &other) {
        deallocate_data();
        id_ = other.id_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        numel_ = other.numel_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        data_owner_ = std::move(other.data_owner_);
        requires_grad_ = other.requires_grad_;
        grad_ = std::move(other.grad_);
        grad_fn_ = std::move(other.grad_fn_);
        inputs_ = std::move(other.inputs_);
        // Invalidate source
        other.data_ = nullptr;
        other.owns_data_ = false;
        other.numel_ = 0;
    }
    return *this;
}

//------------------------------------------------------------------------------
// Private memory management
//------------------------------------------------------------------------------
void Tensor::allocate_data() {
    // Allocate memory on the specified device
    size_t size = nbytes();
    if (size == 0) return;                           // Nothing to allocate
    data_ = allocate_on_device(size, device_);       // Use helper
    owns_data_ = true;
}

void Tensor::deallocate_data() {
    // Free memory if we own it
    if (owns_data_ && data_) {
        free_on_device(data_, nbytes(), device_);
        data_ = nullptr;
        owns_data_ = false;
    }
}

void Tensor::compute_strides() {
    // Compute row-major (C-style) strides
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    strides_.back() = 1;
    for (int64_t i = static_cast<int64_t>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

//------------------------------------------------------------------------------
// Gradient tracking
//------------------------------------------------------------------------------
void Tensor::set_requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;                  // Set flag
    if (!requires_grad) {
        grad_.reset();                               // Clear gradient if tracking disabled
    }
}

void Tensor::set_grad(std::shared_ptr<Tensor> grad) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!requires_grad_) {
        throw std::runtime_error("Cannot set grad on tensor that does not require grad");
    }
    grad_ = grad;                                    // Store gradient
}

void Tensor::zero_grad() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (grad_) {
        grad_->zero_();                              // Fill gradient with zeros
    }
}

void Tensor::set_inputs(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    inputs_.clear();
    for (const auto& in : inputs) {
        inputs_.push_back(in);                       // Store weak references
    }
}

std::shared_ptr<Tensor> Tensor::detach() const {
    // Create a new tensor with same data but no gradient tracking
    auto detached = std::make_shared<Tensor>(shape_, dtype_, device_);
    copy_between_devices(detached->data_, data_, nbytes(), device_, device_);
    detached->set_requires_grad(false);
    return detached;
}

//------------------------------------------------------------------------------
// Data access
//------------------------------------------------------------------------------
void* Tensor::data() {
    // Return mutable data pointer (only valid for CPU)
    if (device_.type != DeviceType::CPU) {
        throw std::runtime_error("Direct data access only supported for CPU tensors");
    }
    return data_;
}

const void* Tensor::data() const {
    // Return const data pointer (only valid for CPU)
    if (device_.type != DeviceType::CPU) {
        throw std::runtime_error("Direct data access only supported for CPU tensors");
    }
    return data_;
}

void Tensor::copy_to_host(void* host_ptr) const {
    // Copy tensor data to host memory
    copy_between_devices(host_ptr, data_, nbytes(), Device{DeviceType::CPU, 0}, device_);
}

void Tensor::copy_from_host(const void* host_ptr) {
    // Copy host data to tensor
    copy_between_devices(data_, host_ptr, nbytes(), device_, Device{DeviceType::CPU, 0});
}

//------------------------------------------------------------------------------
// Fill operations
//------------------------------------------------------------------------------
void Tensor::fill_(double value) {
    // Fill tensor with a constant value (CPU implementation)
    if (device_.type != DeviceType::CPU) {
        // For GPU, we would launch a kernel. For simplicity, copy via host.
        std::vector<uint8_t> host_buffer(nbytes());
        // Fill host buffer with value based on dtype
        if (dtype_ == DType::FLOAT32) {
            float fval = static_cast<float>(value);
            float* ptr = reinterpret_cast<float*>(host_buffer.data());
            std::fill(ptr, ptr + numel_, fval);
        } else if (dtype_ == DType::FLOAT64) {
            double dval = value;
            double* ptr = reinterpret_cast<double*>(host_buffer.data());
            std::fill(ptr, ptr + numel_, dval);
        } else if (dtype_ == DType::INT32) {
            int32_t ival = static_cast<int32_t>(value);
            int32_t* ptr = reinterpret_cast<int32_t*>(host_buffer.data());
            std::fill(ptr, ptr + numel_, ival);
        } else if (dtype_ == DType::INT64) {
            int64_t ival = static_cast<int64_t>(value);
            int64_t* ptr = reinterpret_cast<int64_t*>(host_buffer.data());
            std::fill(ptr, ptr + numel_, ival);
        } else {
            throw std::runtime_error("Unsupported dtype for fill_");
        }
        copy_from_host(host_buffer.data());
        return;
    }
    // CPU path: direct fill
    switch (dtype_) {
        case DType::FLOAT32:
            std::fill_n(static_cast<float*>(data_), numel_, static_cast<float>(value));
            break;
        case DType::FLOAT64:
            std::fill_n(static_cast<double*>(data_), numel_, value);
            break;
        case DType::INT32:
            std::fill_n(static_cast<int32_t*>(data_), numel_, static_cast<int32_t>(value));
            break;
        case DType::INT64:
            std::fill_n(static_cast<int64_t*>(data_), numel_, static_cast<int64_t>(value));
            break;
        default:
            throw std::runtime_error("Unsupported dtype for fill_");
    }
}

void Tensor::zero_() {
    // Fill tensor with zeros
    fill_(0.0);
}

//------------------------------------------------------------------------------
// View operations
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> Tensor::view(const std::vector<int64_t>& new_shape) const {
    // Create a view that shares data
    auto view_tensor = std::make_shared<Tensor>();   // Use private empty constructor? We'll manually build.
    // Copy metadata
    view_tensor->shape_ = new_shape;
    view_tensor->dtype_ = dtype_;
    view_tensor->device_ = device_;
    view_tensor->numel_ = 1;
    for (auto d : new_shape) view_tensor->numel_ *= d;
    if (view_tensor->numel_ != numel_) {
        throw std::invalid_argument("view: total elements must match");
    }
    view_tensor->compute_strides();                  // New strides
    view_tensor->data_ = data_;                      // Share data pointer
    view_tensor->owns_data_ = false;                 // View does not own data
    view_tensor->data_owner_ = data_owner_.expired() ? shared_from_this() : data_owner_.lock();
    view_tensor->requires_grad_ = requires_grad_;
    view_tensor->grad_ = grad_;
    return view_tensor;
}

std::shared_ptr<Tensor> Tensor::contiguous() const {
    // If already contiguous, return self; else copy
    // Check contiguity (row-major strides)
    std::vector<int64_t> contig_strides(shape_.size());
    if (!shape_.empty()) {
        contig_strides.back() = 1;
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
            contig_strides[i] = contig_strides[i+1] * shape_[i+1];
        }
    }
    bool is_contig = (strides_ == contig_strides);
    if (is_contig) {
        return std::const_pointer_cast<Tensor>(shared_from_this()); // Already contiguous
    }
    // Create a new contiguous copy
    auto copy = std::make_shared<Tensor>(shape_, dtype_, device_);
    // Copy data (this is simplified; proper handling would copy with stride iteration)
    copy_between_devices(copy->data_, data_, nbytes(), device_, device_);
    return copy;
}

//------------------------------------------------------------------------------
// Static factory methods
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> Tensor::zeros(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto t = std::make_shared<Tensor>(shape, dtype, device);
    t->zero_();
    return t;
}

std::shared_ptr<Tensor> Tensor::ones(const std::vector<int64_t>& shape, DType dtype, Device device) {
    auto t = std::make_shared<Tensor>(shape, dtype, device);
    t->fill_(1.0);
    return t;
}

std::shared_ptr<Tensor> Tensor::zeros_like(const Tensor& other) {
    return zeros(other.shape_, other.dtype_, other.device_);
}

std::shared_ptr<Tensor> Tensor::ones_like(const Tensor& other) {
    return ones(other.shape_, other.dtype_, other.device_);
}

void Tensor::initialize_factory() {
    // Placeholder for any static initialization needed
}

void Tensor::clear_global_cache() {
    // Placeholder for clearing caches
}

//------------------------------------------------------------------------------
// Copy helpers for concatenation/stacking (simplified)
//------------------------------------------------------------------------------
void Tensor::copy_from_tensor(const Tensor& src, int64_t offset, int64_t dim) {
    // Copy src into this tensor at given offset along dimension dim
    // Assumes contiguous tensors and matching dtypes/devices
    if (dtype_ != src.dtype_ || device_.type != src.device_.type) {
        throw std::invalid_argument("copy_from_tensor: dtype/device mismatch");
    }
    // Compute number of elements to copy = src.numel()
    size_t copy_bytes = src.nbytes();
    size_t offset_bytes = offset * strides_[dim] * dtype_size(dtype_);
    // Direct memory copy (simplified; assumes contiguous layout)
    if (device_.type == DeviceType::CPU) {
        char* dst_ptr = static_cast<char*>(data_) + offset_bytes;
        const char* src_ptr = static_cast<const char*>(src.data_);
        std::memcpy(dst_ptr, src_ptr, copy_bytes);
    } else {
        // GPU: use device-to-device copy via staging (or proper kernel)
        void* dst_ptr = static_cast<char*>(data_) + offset_bytes;
        copy_between_devices(dst_ptr, src.data_, copy_bytes, device_, src.device_);
    }
}

void Tensor::copy_from_tensor_at_index(const Tensor& src, int64_t index, int64_t dim) {
    // Similar to copy_from_tensor but uses index rather than element offset
    int64_t offset = index * src.shape()[dim];       // Offset in elements along dim
    copy_from_tensor(src, offset, dim);
}

//------------------------------------------------------------------------------
// In-place operations (CPU implementations for simplicity; GPU would use kernels)
//------------------------------------------------------------------------------
void Tensor::add_(const Tensor& other) {
    check_same_shape(other, "add_");
    check_same_dtype(other, "add_");
    check_same_device(other, "add_");
    if (device_.type == DeviceType::CPU) {
        if (dtype_ == DType::FLOAT32) {
            float* a = static_cast<float*>(data_);
            const float* b = static_cast<const float*>(other.data_);
            for (int64_t i = 0; i < numel_; ++i) a[i] += b[i];
        } else if (dtype_ == DType::FLOAT64) {
            double* a = static_cast<double*>(data_);
            const double* b = static_cast<const double*>(other.data_);
            for (int64_t i = 0; i < numel_; ++i) a[i] += b[i];
        } else {
            throw std::runtime_error("Unsupported dtype for add_");
        }
    } else {
        // GPU: copy to host, compute, copy back (inefficient but functional)
        std::vector<uint8_t> host_a(nbytes()), host_b(nbytes());
        copy_to_host(host_a.data());
        other.copy_to_host(host_b.data());
        if (dtype_ == DType::FLOAT32) {
            float* a = reinterpret_cast<float*>(host_a.data());
            const float* b = reinterpret_cast<const float*>(host_b.data());
            for (int64_t i = 0; i < numel_; ++i) a[i] += b[i];
        } else if (dtype_ == DType::FLOAT64) {
            double* a = reinterpret_cast<double*>(host_a.data());
            const double* b = reinterpret_cast<const double*>(host_b.data());
            for (int64_t i = 0; i < numel_; ++i) a[i] += b[i];
        }
        copy_from_host(host_a.data());
    }
}

void Tensor::sub_(const Tensor& other) {
    check_same_shape(other, "sub_");
    if (device_.type == DeviceType::CPU) {
        if (dtype_ == DType::FLOAT32) {
            float* a = static_cast<float*>(data_);
            const float* b = static_cast<const float*>(other.data_);
            for (int64_t i = 0; i < numel_; ++i) a[i] -= b[i];
        } else if (dtype_ == DType::FLOAT64) {
            double* a = static_cast<double*>(data_);
            const double* b = static_cast<const double*>(other.data_);
            for (int64_t i = 0; i < numel_; ++i) a[i] -= b[i];
        }
    } else {
        // GPU: host fallback
        std::vector<uint8_t> host_a(nbytes()), host_b(nbytes());
        copy_to_host(host_a.data());
        other.copy_to_host(host_b.data());
        if (dtype_ == DType::FLOAT32) {
            float* a = reinterpret_cast<float*>(host_a.data());
            const float* b = reinterpret_cast<const float*>(host_b.data());
            for (int64_t i = 0; i < numel_; ++i) a[i] -= b[i];
        } else if (dtype_ == DType::FLOAT64) {
            double* a = reinterpret_cast<double*>(host_a.data());
            const double* b = reinterpret_cast<const double*>(host_b.data());
            for (int64_t i = 0; i < numel_; ++i) a[i] -= b[i];
        }
        copy_from_host(host_a.data());
    }
}

void Tensor::mul_(const Tensor& other) {
    check_same_shape(other, "mul_");
    if (device_.type == DeviceType::CPU) {
        if (dtype_ == DType::FLOAT32) {
            float* a = static_cast<float*>(data_);
            const float* b = static_cast<const float*>(other.data_);
            for (int64_t i = 0; i < numel_; ++i) a[i] *= b[i];
        } else if (dtype_ == DType::FLOAT64) {
            double* a = static_cast<double*>(data_);
            const double* b = static_cast<const double*>(other.data_);
            for (int64_t i = 0; i < numel_; ++i) a[i] *= b[i];
        }
    } else {
        // GPU host fallback
        std::vector<uint8_t> host_a(nbytes()), host_b(nbytes());
        copy_to_host(host_a.data());
        other.copy_to_host(host_b.data());
        if (dtype_ == DType::FLOAT32) {
            float* a = reinterpret_cast<float*>(host_a.data());
            const float* b = reinterpret_cast<const float*>(host_b.data());
            for (int64_t i = 0; i < numel_; ++i) a[i] *= b[i];
        } else if (dtype_ == DType::FLOAT64) {
            double* a = reinterpret_cast<double*>(host_a.data());
            const double* b = reinterpret_cast<const double*>(host_b.data());
            for (int64_t i = 0; i < numel_; ++i) a[i] *= b[i];
        }
        copy_from_host(host_a.data());
    }
}

void Tensor::div_(const Tensor& other) {
    check_same_shape(other, "div_");
    if (device_.type == DeviceType::CPU) {
        if (dtype_ == DType::FLOAT32) {
            float* a = static_cast<float*>(data_);
            const float* b = static_cast<const float*>(other.data_);
            for (int64_t i = 0; i < numel_; ++i) a[i] /= b[i];
        } else if (dtype_ == DType::FLOAT64) {
            double* a = static_cast<double*>(data_);
            const double* b = static_cast<const double*>(other.data_);
            for (int64_t i = 0; i < numel_; ++i) a[i] /= b[i];
        }
    } else {
        // GPU host fallback
        std::vector<uint8_t> host_a(nbytes()), host_b(nbytes());
        copy_to_host(host_a.data());
        other.copy_to_host(host_b.data());
        if (dtype_ == DType::FLOAT32) {
            float* a = reinterpret_cast<float*>(host_a.data());
            const float* b = reinterpret_cast<const float*>(host_b.data());
            for (int64_t i = 0; i < numel_; ++i) a[i] /= b[i];
        } else if (dtype_ == DType::FLOAT64) {
            double* a = reinterpret_cast<double*>(host_a.data());
            const double* b = reinterpret_cast<const double*>(host_b.data());
            for (int64_t i = 0; i < numel_; ++i) a[i] /= b[i];
        }
        copy_from_host(host_a.data());
    }
}

void Tensor::add_scalar_(double scalar) {
    if (device_.type == DeviceType::CPU) {
        if (dtype_ == DType::FLOAT32) {
            float* a = static_cast<float*>(data_);
            float s = static_cast<float>(scalar);
            for (int64_t i = 0; i < numel_; ++i) a[i] += s;
        } else if (dtype_ == DType::FLOAT64) {
            double* a = static_cast<double*>(data_);
            for (int64_t i = 0; i < numel_; ++i) a[i] += scalar;
        }
    } else {
        // GPU host fallback
        std::vector<uint8_t> host_a(nbytes());
        copy_to_host(host_a.data());
        if (dtype_ == DType::FLOAT32) {
            float* a = reinterpret_cast<float*>(host_a.data());
            float s = static_cast<float>(scalar);
            for (int64_t i = 0; i < numel_; ++i) a[i] += s;
        } else if (dtype_ == DType::FLOAT64) {
            double* a = reinterpret_cast<double*>(host_a.data());
            for (int64_t i = 0; i < numel_; ++i) a[i] += scalar;
        }
        copy_from_host(host_a.data());
    }
}

void Tensor::mul_scalar_(double scalar) {
    if (device_.type == DeviceType::CPU) {
        if (dtype_ == DType::FLOAT32) {
            float* a = static_cast<float*>(data_);
            float s = static_cast<float>(scalar);
            for (int64_t i = 0; i < numel_; ++i) a[i] *= s;
        } else if (dtype_ == DType::FLOAT64) {
            double* a = static_cast<double*>(data_);
            for (int64_t i = 0; i < numel_; ++i) a[i] *= scalar;
        }
    } else {
        // GPU host fallback
        std::vector<uint8_t> host_a(nbytes());
        copy_to_host(host_a.data());
        if (dtype_ == DType::FLOAT32) {
            float* a = reinterpret_cast<float*>(host_a.data());
            float s = static_cast<float>(scalar);
            for (int64_t i = 0; i < numel_; ++i) a[i] *= s;
        } else if (dtype_ == DType::FLOAT64) {
            double* a = reinterpret_cast<double*>(host_a.data());
            for (int64_t i = 0; i < numel_; ++i) a[i] *= scalar;
        }
        copy_from_host(host_a.data());
    }
}

//------------------------------------------------------------------------------
// Validation helpers
//------------------------------------------------------------------------------
void Tensor::check_same_shape(const Tensor& other, const char* op) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument(std::string(op) + ": shape mismatch");
    }
}

void Tensor::check_same_dtype(const Tensor& other, const char* op) const {
    if (dtype_ != other.dtype_) {
        throw std::invalid_argument(std::string(op) + ": dtype mismatch");
    }
}

void Tensor::check_same_device(const Tensor& other, const char* op) const {
    if (device_.type != other.device_.type || device_.index != other.device_.index) {
        throw std::invalid_argument(std::string(op) + ": device mismatch");
    }
}

} // namespace grad
} // namespace genesis