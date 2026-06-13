// genesis/grad/tensor.h

#pragma once

//------------------------------------------------------------------------------
// Tensor class - core data structure for automatic differentiation.
// Represents a multi-dimensional array with gradient tracking.
//------------------------------------------------------------------------------

#include "genesis/datatypes.h"                      // For real type
#include <vector>                                   // std::vector for shape and strides
#include <memory>                                   // std::shared_ptr, std::weak_ptr
#include <string>                                   // std::string
#include <functional>                               // std::function for grad_fn
#include <atomic>                                   // std::atomic for ID generation
#include <mutex>                                    // std::mutex for thread safety
#include <unordered_map>                            // std::unordered_map for caching

namespace genesis {
namespace grad {

//------------------------------------------------------------------------------
// Forward declarations
//------------------------------------------------------------------------------
class Tensor;
class Function;
class GradContext;

//------------------------------------------------------------------------------
// Device structure (moved from __init__.h for completeness, redefined here)
//------------------------------------------------------------------------------
enum class DeviceType : uint8_t {
    CPU = 0,
    CUDA = 1,
    METAL = 2,
    AMD = 3
};

struct Device {
    DeviceType type = DeviceType::CPU;              // Type of device
    int index = 0;                                  // Device index (e.g., GPU number)
    
    bool operator==(const Device& other) const {
        return type == other.type && index == other.index; // Equality comparison
    }
    bool operator!=(const Device& other) const {
        return !(*this == other);                   // Inequality
    }
};

//------------------------------------------------------------------------------
// Data type enumeration
//------------------------------------------------------------------------------
enum class DType : uint8_t {
    FLOAT32 = 0,                                    // 32-bit floating point
    FLOAT64 = 1,                                    // 64-bit floating point
    INT32 = 2,                                      // 32-bit signed integer
    INT64 = 3,                                      // 64-bit signed integer
    BOOL = 4,                                       // Boolean
    UINT8 = 5                                       // 8-bit unsigned integer
};

// Get size in bytes for a given dtype
size_t dtype_size(DType dtype);                     // Returns sizeof for each type

//------------------------------------------------------------------------------
// Tensor class definition
//------------------------------------------------------------------------------
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    //----------------------------------------------------------------------
    // Construction and destruction
    //----------------------------------------------------------------------
    // Create an uninitialized tensor with given shape, dtype, and device
    Tensor(const std::vector<int64_t>& shape,
           DType dtype = DType::FLOAT32,
           Device device = Device{DeviceType::CPU, 0});
    
    // Create a tensor from existing data buffer (takes ownership or copies)
    Tensor(const void* data,
           const std::vector<int64_t>& shape,
           DType dtype = DType::FLOAT32,
           Device device = Device{DeviceType::CPU, 0});
    
    // Copy constructor (deep copy)
    Tensor(const Tensor& other);
    
    // Move constructor
    Tensor(Tensor&& other) noexcept;
    
    // Destructor
    ~Tensor();

    // Assignment operators
    Tensor& operator=(const Tensor& other);         // Deep copy
    Tensor& operator=(Tensor&& other) noexcept;     // Move assignment

    //----------------------------------------------------------------------
    // Basic properties
    //----------------------------------------------------------------------
    const std::vector<int64_t>& shape() const { return shape_; } // Tensor dimensions
    int64_t dim() const { return static_cast<int64_t>(shape_.size()); } // Number of dimensions
    int64_t numel() const { return numel_; }        // Total number of elements
    DType dtype() const { return dtype_; }          // Data type
    Device device() const { return device_; }       // Device where data resides
    size_t nbytes() const { return numel_ * dtype_size(dtype_); } // Size in bytes
    
    // Strides (for viewing)
    const std::vector<int64_t>& strides() const { return strides_; }
    
    // Unique identifier (for debugging)
    uint64_t id() const { return id_; }

    //----------------------------------------------------------------------
    // Gradient tracking
    //----------------------------------------------------------------------
    bool requires_grad() const { return requires_grad_; } // Whether gradient is tracked
    void set_requires_grad(bool requires_grad);     // Enable/disable gradient tracking
    
    std::shared_ptr<Tensor> grad() const { return grad_; } // Accumulated gradient
    void set_grad(std::shared_ptr<Tensor> grad);    // Set gradient tensor
    
    void zero_grad();                               // Reset gradient to zero
    
    // Gradient function (the operation that produced this tensor)
    std::shared_ptr<Function> grad_fn() const { return grad_fn_; }
    void set_grad_fn(std::shared_ptr<Function> fn) { grad_fn_ = fn; }
    
    // Input tensors to the operation that created this tensor
    const std::vector<std::weak_ptr<Tensor>>& inputs() const { return inputs_; }
    void set_inputs(const std::vector<std::shared_ptr<Tensor>>& inputs);
    
    // Detach from computation graph (returns a new tensor without grad tracking)
    std::shared_ptr<Tensor> detach() const;

    //----------------------------------------------------------------------
    // Data access (host only; for device access use specialized methods)
    //----------------------------------------------------------------------
    // Get raw pointer to data (asserts device is CPU)
    void* data();
    const void* data() const;
    
    // Typed accessors
    template<typename T> T* data_ptr();              // Mutable typed pointer
    template<typename T> const T* data_ptr() const;  // Const typed pointer
    
    // Copy data to/from host
    void copy_to_host(void* host_ptr) const;         // Copy tensor to host memory
    void copy_from_host(const void* host_ptr);       // Copy host memory to tensor
    
    // Fill tensor with a scalar value
    void fill_(double value);                        // In-place fill
    
    // Zero the tensor
    void zero_();                                    // In-place zero
    
    //----------------------------------------------------------------------
    // Views and reshaping
    //----------------------------------------------------------------------
    // Create a view with new shape (shares data)
    std::shared_ptr<Tensor> view(const std::vector<int64_t>& new_shape) const;
    
    // Create a contiguous copy (ensures row-major memory layout)
    std::shared_ptr<Tensor> contiguous() const;
    
    // Transpose dimensions (returns view)
    std::shared_ptr<Tensor> transpose(int64_t dim0, int64_t dim1) const;
    
    // Permute dimensions (returns view)
    std::shared_ptr<Tensor> permute(const std::vector<int64_t>& dims) const;
    
    // Slice along a dimension
    std::shared_ptr<Tensor> slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const;
    
    // Index tensor (returns a view)
    std::shared_ptr<Tensor> operator[](int64_t index) const;

    //----------------------------------------------------------------------
    // In-place operations (modify this tensor)
    //----------------------------------------------------------------------
    void add_(const Tensor& other);                  // this += other
    void sub_(const Tensor& other);                  // this -= other
    void mul_(const Tensor& other);                  // this *= other
    void div_(const Tensor& other);                  // this /= other
    void add_scalar_(double scalar);                 // this += scalar
    void mul_scalar_(double scalar);                 // this *= scalar
    
    //----------------------------------------------------------------------
    // Static factory methods
    //----------------------------------------------------------------------
    static std::shared_ptr<Tensor> zeros(const std::vector<int64_t>& shape,
                                         DType dtype = DType::FLOAT32,
                                         Device device = Device{DeviceType::CPU, 0});
    static std::shared_ptr<Tensor> ones(const std::vector<int64_t>& shape,
                                        DType dtype = DType::FLOAT32,
                                        Device device = Device{DeviceType::CPU, 0});
    static std::shared_ptr<Tensor> zeros_like(const Tensor& other);
    static std::shared_ptr<Tensor> ones_like(const Tensor& other);
    
    // Initialize tensor factory (called by grad module init)
    static void initialize_factory();
    static void clear_global_cache();

    //----------------------------------------------------------------------
    // Internal: copy from another tensor with offset (for concatenation)
    //----------------------------------------------------------------------
    void copy_from_tensor(const Tensor& src, int64_t offset, int64_t dim);
    void copy_from_tensor_at_index(const Tensor& src, int64_t index, int64_t dim);

private:
    //----------------------------------------------------------------------
    // Private member variables
    //----------------------------------------------------------------------
    static std::atomic<uint64_t> next_id_;           // Static ID counter
    
    uint64_t id_;                                    // Unique identifier
    std::vector<int64_t> shape_;                     // Tensor dimensions
    std::vector<int64_t> strides_;                   // Strides in elements
    int64_t numel_;                                  // Total number of elements
    DType dtype_;                                    // Data type
    Device device_;                                  // Device location
    
    void* data_;                                     // Raw data pointer (device-specific)
    bool owns_data_;                                 // Whether we own the data memory
    std::shared_ptr<void> data_owner_;               // Shared ownership of data buffer
    
    bool requires_grad_;                             // Gradient tracking flag
    std::shared_ptr<Tensor> grad_;                   // Accumulated gradient
    std::shared_ptr<Function> grad_fn_;              // Function that produced this tensor
    std::vector<std::weak_ptr<Tensor>> inputs_;      // Input tensors (weak refs)
    
    mutable std::mutex mutex_;                       // Mutex for thread-safe operations
    
    //----------------------------------------------------------------------
    // Private helper methods
    //----------------------------------------------------------------------
    void allocate_data();                            // Allocate memory on device
    void deallocate_data();                          // Free allocated memory
    void compute_strides();                          // Compute strides from shape (row-major)
    void copy_data_from(const void* src, size_t size); // Raw copy from host to device
    void copy_data_to(void* dst, size_t size) const; // Raw copy from device to host
    
    // Validate arguments for operations
    void check_same_shape(const Tensor& other, const char* op) const;
    void check_same_dtype(const Tensor& other, const char* op) const;
    void check_same_device(const Tensor& other, const char* op) const;
    
    // Element-wise operation helpers (CPU implementation)
    template<typename Func>
    void apply_unary_op(Func func);                  // this = func(this)
    template<typename Func>
    void apply_binary_op(const Tensor& other, Func func); // this = func(this, other)
    void apply_binary_op_cpu(const Tensor& other,
                             std::function<void(void*, const void*, const void*, size_t)> op_func);
};

//------------------------------------------------------------------------------
// Template method implementations (inline)
//------------------------------------------------------------------------------
template<typename T>
T* Tensor::data_ptr() {
    // Return typed mutable pointer (only valid for CPU)
    if (device_.type != DeviceType::CPU) {
        throw std::runtime_error("data_ptr only available for CPU tensors");
    }
    if (sizeof(T) != dtype_size(dtype_)) {
        throw std::runtime_error("data_ptr type size mismatch with tensor dtype");
    }
    return static_cast<T*>(data_);                   // Cast to requested type
}

template<typename T>
const T* Tensor::data_ptr() const {
    // Return typed const pointer (only valid for CPU)
    if (device_.type != DeviceType::CPU) {
        throw std::runtime_error("data_ptr only available for CPU tensors");
    }
    if (sizeof(T) != dtype_size(dtype_)) {
        throw std::runtime_error("data_ptr type size mismatch with tensor dtype");
    }
    return static_cast<const T*>(data_);             // Cast to requested type
}

} // namespace grad
} // namespace genesis