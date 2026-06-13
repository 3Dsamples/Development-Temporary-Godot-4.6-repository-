// genesis/grad/creation_ops.cpp

#include "genesis/grad/creation_ops.h"               // Include corresponding header
#include "genesis/grad/__init__.h"                   // For device management and grad enabled
#include <algorithm>                                 // std::fill, std::generate, std::copy
#include <numeric>                                   // std::iota
#include <random>                                    // std::mt19937, std::uniform_real_distribution, etc.
#include <chrono>                                    // std::chrono for seed generation
#include <cmath>                                     // std::log10, std::pow, std::floor
#include <cstring>                                   // std::memcpy
#include <stdexcept>                                 // std::invalid_argument, std::runtime_error

namespace genesis {
namespace grad {

//------------------------------------------------------------------------------
// Global random number generator state
//------------------------------------------------------------------------------
static std::mt19937 g_rng;                           // Mersenne Twister engine instance
static bool g_rng_seeded = false;                    // Flag to track if user set seed
static std::mutex g_rng_mutex;                       // Mutex for thread-safe RNG access

//------------------------------------------------------------------------------
// Helper: Get RNG engine (thread-safe)
//------------------------------------------------------------------------------
static std::mt19937& get_rng() {
    // Return reference to RNG, seeding if not already seeded by user
    std::lock_guard<std::mutex> lock(g_rng_mutex);   // Lock for thread safety
    if (!g_rng_seeded) {
        // Auto-seed with random device and time
        std::random_device rd;                       // Hardware random device
        auto seed = rd() ^ std::chrono::steady_clock::now().time_since_epoch().count();
        g_rng.seed(static_cast<unsigned int>(seed)); // Seed the generator
        g_rng_seeded = true;                         // Mark as seeded
    }
    return g_rng;                                    // Return reference to RNG
}

//------------------------------------------------------------------------------
// manual_seed implementation
//------------------------------------------------------------------------------
void manual_seed(uint64_t seed) {
    // Set user-defined seed for reproducibility
    std::lock_guard<std::mutex> lock(g_rng_mutex);   // Lock mutex
    g_rng.seed(static_cast<unsigned int>(seed));     // Seed the generator
    g_rng_seeded = true;                             // Mark as manually seeded
}

//------------------------------------------------------------------------------
// zeros implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> zeros(const std::vector<int64_t>& shape,
                              DType dtype,
                              Device device) {
    // Create a tensor filled with zeros
    auto t = std::make_shared<Tensor>(shape, dtype, device); // Allocate tensor
    t->zero_();                                      // Fill with zeros (device-agnostic)
    return t;                                        // Return the tensor
}

//------------------------------------------------------------------------------
// ones implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> ones(const std::vector<int64_t>& shape,
                             DType dtype,
                             Device device) {
    // Create a tensor filled with ones
    auto t = std::make_shared<Tensor>(shape, dtype, device); // Allocate tensor
    t->fill_(1.0);                                   // Fill with ones
    return t;                                        // Return the tensor
}

//------------------------------------------------------------------------------
// full implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> full(const std::vector<int64_t>& shape,
                             double value,
                             DType dtype,
                             Device device) {
    // Create a tensor filled with a constant value
    auto t = std::make_shared<Tensor>(shape, dtype, device); // Allocate tensor
    t->fill_(value);                                 // Fill with the given value
    return t;                                        // Return the tensor
}

//------------------------------------------------------------------------------
// zeros_like implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> zeros_like(const std::shared_ptr<Tensor>& other) {
    // Create a tensor of zeros with same shape, dtype, and device as 'other'
    if (!other) {
        throw std::invalid_argument("zeros_like: input tensor is null");
    }
    return zeros(other->shape(), other->dtype(), other->device()); // Delegate to zeros
}

//------------------------------------------------------------------------------
// ones_like implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> ones_like(const std::shared_ptr<Tensor>& other) {
    // Create a tensor of ones with same shape, dtype, and device as 'other'
    if (!other) {
        throw std::invalid_argument("ones_like: input tensor is null");
    }
    return ones(other->shape(), other->dtype(), other->device()); // Delegate to ones
}

//------------------------------------------------------------------------------
// full_like implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> full_like(const std::shared_ptr<Tensor>& other, double value) {
    // Create a tensor filled with constant with same properties as 'other'
    if (!other) {
        throw std::invalid_argument("full_like: input tensor is null");
    }
    return full(other->shape(), value, other->dtype(), other->device()); // Delegate to full
}

//------------------------------------------------------------------------------
// arange implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> arange(double start, double end, double step,
                               DType dtype,
                               Device device) {
    // Create a 1D tensor with values from start to end (exclusive) with step
    if (step == 0.0) {
        throw std::invalid_argument("arange: step cannot be zero");
    }
    // Compute number of elements
    double range = end - start;                      // Total span
    int64_t num = static_cast<int64_t>(std::ceil(range / step)); // Number of steps
    if (num <= 0) {
        // Return empty tensor if range is zero or sign mismatch
        return zeros({0}, dtype, device);
    }
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{num}, dtype, device); // Allocate
    // Fill with sequential values
    std::vector<double> host_data(num);              // Host buffer
    for (int64_t i = 0; i < num; ++i) {
        host_data[i] = start + i * step;             // Compute value
    }
    t->copy_from_host(host_data.data());             // Copy to device
    return t;                                        // Return tensor
}

//------------------------------------------------------------------------------
// linspace implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> linspace(double start, double end, int64_t num_steps,
                                 DType dtype,
                                 Device device) {
    // Create a 1D tensor with values evenly spaced between start and end (inclusive)
    if (num_steps <= 0) {
        throw std::invalid_argument("linspace: num_steps must be positive");
    }
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{num_steps}, dtype, device); // Allocate
    std::vector<double> host_data(num_steps);        // Host buffer
    if (num_steps == 1) {
        host_data[0] = start;                        // Single point: use start
    } else {
        double step = (end - start) / (num_steps - 1); // Compute step size
        for (int64_t i = 0; i < num_steps; ++i) {
            host_data[i] = start + i * step;         // Compute value
        }
    }
    t->copy_from_host(host_data.data());             // Copy to device
    return t;                                        // Return tensor
}

//------------------------------------------------------------------------------
// logspace implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> logspace(double start, double end, int64_t num_steps,
                                 double base,
                                 DType dtype,
                                 Device device) {
    // Create a 1D tensor with values logarithmically spaced
    if (num_steps <= 0) {
        throw std::invalid_argument("logspace: num_steps must be positive");
    }
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{num_steps}, dtype, device); // Allocate
    std::vector<double> host_data(num_steps);        // Host buffer
    if (num_steps == 1) {
        host_data[0] = std::pow(base, start);        // Single point
    } else {
        double step = (end - start) / (num_steps - 1); // Linear step in exponent space
        for (int64_t i = 0; i < num_steps; ++i) {
            host_data[i] = std::pow(base, start + i * step); // Compute base^(exponent)
        }
    }
    t->copy_from_host(host_data.data());             // Copy to device
    return t;                                        // Return tensor
}

//------------------------------------------------------------------------------
// eye implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> eye(int64_t n,
                            DType dtype,
                            Device device) {
    // Create an identity matrix of size n x n
    if (n <= 0) {
        throw std::invalid_argument("eye: n must be positive");
    }
    auto t = zeros({n, n}, dtype, device);           // Allocate zero matrix
    // Set diagonal elements to 1
    std::vector<double> diag(n, 1.0);                // Host diagonal data
    // Create a view or set individually (simplified: use fill diagonal on CPU)
    // For performance, we'd use a custom kernel; here we use a simple loop
    if (device.type == DeviceType::CPU) {
        double* ptr = static_cast<double*>(t->mutable_data()); // Get CPU pointer
        for (int64_t i = 0; i < n; ++i) {
            ptr[i * n + i] = 1.0;                    // Set diagonal element
        }
    } else {
        // For GPU, we would launch a kernel; fallback to host copy
        std::vector<double> host_data(n * n, 0.0);   // Host zero matrix
        for (int64_t i = 0; i < n; ++i) {
            host_data[i * n + i] = 1.0;              // Set diagonal
        }
        t->copy_from_host(host_data.data());         // Copy to device
    }
    return t;                                        // Return identity tensor
}

//------------------------------------------------------------------------------
// diag implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> diag(const std::shared_ptr<Tensor>& input, int64_t k) {
    // Extract diagonal or create diagonal matrix
    if (!input) {
        throw std::invalid_argument("diag: input tensor is null");
    }
    const auto& shape = input->shape();
    if (shape.size() == 1) {
        // Input is 1D: create a diagonal matrix
        int64_t n = shape[0] + std::abs(k);          // Size of square matrix
        auto result = zeros({n, n}, input->dtype(), input->device()); // Zero matrix
        // Copy input to diagonal
        std::vector<double> host_data_input(input->numel());
        input->copy_to_host(host_data_input.data()); // Get input data
        if (input->device().type == DeviceType::CPU) {
            double* ptr = static_cast<double*>(result->mutable_data());
            for (int64_t i = 0; i < shape[0]; ++i) {
                int64_t row = (k >= 0) ? i : i - k;   // Row index
                int64_t col = (k >= 0) ? i + k : i;   // Column index
                if (row >= 0 && row < n && col >= 0 && col < n) {
                    ptr[row * n + col] = host_data_input[i]; // Set diagonal value
                }
            }
        } else {
            // GPU: use host buffer
            std::vector<double> host_data_result(n * n, 0.0);
            for (int64_t i = 0; i < shape[0]; ++i) {
                int64_t row = (k >= 0) ? i : i - k;
                int64_t col = (k >= 0) ? i + k : i;
                if (row >= 0 && row < n && col >= 0 && col < n) {
                    host_data_result[row * n + col] = host_data_input[i];
                }
            }
            result->copy_from_host(host_data_result.data());
        }
        return result;
    } else if (shape.size() == 2) {
        // Input is 2D: extract k-th diagonal
        int64_t n = std::min(shape[0], shape[1]) - std::abs(k);
        if (n <= 0) {
            return zeros({0}, input->dtype(), input->device()); // Empty tensor
        }
        auto result = std::make_shared<Tensor>(std::vector<int64_t>{n}, input->dtype(), input->device());
        std::vector<double> host_data_input(input->numel());
        input->copy_to_host(host_data_input.data());
        std::vector<double> host_data_result(n);
        for (int64_t i = 0; i < n; ++i) {
            int64_t row = (k >= 0) ? i : i - k;
            int64_t col = (k >= 0) ? i + k : i;
            host_data_result[i] = host_data_input[row * shape[1] + col]; // Extract
        }
        result->copy_from_host(host_data_result.data());
        return result;
    } else {
        throw std::invalid_argument("diag: input must be 1D or 2D tensor");
    }
}

//------------------------------------------------------------------------------
// rand implementation (uniform [0,1))
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> rand(const std::vector<int64_t>& shape,
                             DType dtype,
                             Device device) {
    // Create tensor with uniformly distributed random values in [0,1)
    auto t = std::make_shared<Tensor>(shape, dtype, device); // Allocate tensor
    size_t num_elements = t->numel();                // Number of elements
    std::vector<double> host_data(num_elements);     // Host buffer
    std::uniform_real_distribution<double> dist(0.0, 1.0); // Distribution
    auto& rng = get_rng();                           // Get thread-safe RNG
    for (size_t i = 0; i < num_elements; ++i) {
        host_data[i] = dist(rng);                    // Generate random value
    }
    t->copy_from_host(host_data.data());             // Copy to device
    return t;                                        // Return tensor
}

//------------------------------------------------------------------------------
// randn implementation (standard normal)
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> randn(const std::vector<int64_t>& shape,
                              DType dtype,
                              Device device) {
    // Create tensor with standard normal distribution (mean=0, std=1)
    auto t = std::make_shared<Tensor>(shape, dtype, device); // Allocate tensor
    size_t num_elements = t->numel();                // Number of elements
    std::vector<double> host_data(num_elements);     // Host buffer
    std::normal_distribution<double> dist(0.0, 1.0); // Normal distribution
    auto& rng = get_rng();                           // Get thread-safe RNG
    for (size_t i = 0; i < num_elements; ++i) {
        host_data[i] = dist(rng);                    // Generate random value
    }
    t->copy_from_host(host_data.data());             // Copy to device
    return t;                                        // Return tensor
}

//------------------------------------------------------------------------------
// randint implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> randint(int64_t low, int64_t high,
                                const std::vector<int64_t>& shape,
                                Device device) {
    // Create tensor with uniformly distributed integers in [low, high)
    if (low >= high) {
        throw std::invalid_argument("randint: low must be less than high");
    }
    auto t = std::make_shared<Tensor>(shape, DType::INT64, device); // Allocate int64 tensor
    size_t num_elements = t->numel();                // Number of elements
    std::vector<int64_t> host_data(num_elements);    // Host buffer
    std::uniform_int_distribution<int64_t> dist(low, high - 1); // Integer distribution
    auto& rng = get_rng();                           // Get thread-safe RNG
    for (size_t i = 0; i < num_elements; ++i) {
        host_data[i] = dist(rng);                    // Generate random integer
    }
    t->copy_from_host(host_data.data());             // Copy to device
    return t;                                        // Return tensor
}

//------------------------------------------------------------------------------
// randperm implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> randperm(int64_t n,
                                 Device device) {
    // Create a random permutation of integers from 0 to n-1
    if (n < 0) {
        throw std::invalid_argument("randperm: n must be non-negative");
    }
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{n}, DType::INT64, device); // Allocate
    std::vector<int64_t> host_data(n);               // Host buffer
    std::iota(host_data.begin(), host_data.end(), 0); // Fill with 0..n-1
    auto& rng = get_rng();                           // Get thread-safe RNG
    std::shuffle(host_data.begin(), host_data.end(), rng); // Random shuffle
    t->copy_from_host(host_data.data());             // Copy to device
    return t;                                        // Return permuted tensor
}

//------------------------------------------------------------------------------
// rand_like / randn_like implementations
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> rand_like(const std::shared_ptr<Tensor>& other) {
    // Create tensor with uniform random values with same shape/dtype/device as other
    if (!other) {
        throw std::invalid_argument("rand_like: input tensor is null");
    }
    return rand(other->shape(), other->dtype(), other->device());
}

std::shared_ptr<Tensor> randn_like(const std::shared_ptr<Tensor>& other) {
    // Create tensor with normal random values with same shape/dtype/device as other
    if (!other) {
        throw std::invalid_argument("randn_like: input tensor is null");
    }
    return randn(other->shape(), other->dtype(), other->device());
}

//------------------------------------------------------------------------------
// from_blob implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> from_blob(const void* data,
                                  const std::vector<int64_t>& shape,
                                  DType dtype,
                                  Device device) {
    // Create tensor from raw memory blob
    auto t = std::make_shared<Tensor>(shape, dtype, device); // Allocate tensor
    t->copy_from_host(data);                         // Copy blob to tensor
    return t;                                        // Return tensor
}

//------------------------------------------------------------------------------
// from_vector3 implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> from_vector3(const datatypes::Vector3& vec,
                                     Device device) {
    // Convert Genesis Vector3 to 1D tensor of size 3
    double data[3] = {vec[0], vec[1], vec[2]};       // Extract components
    return from_blob(data, {3}, DType::FLOAT64, device); // Create tensor
}

//------------------------------------------------------------------------------
// from_matrix3 implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> from_matrix3(const datatypes::Matrix3r& mat,
                                     Device device) {
    // Convert Genesis Matrix3r to 2D tensor of shape (3,3)
    double data[9];                                  // Flattened matrix (column-major as stored)
    for (int i = 0; i < 9; ++i) {
        data[i] = mat.data[i];                       // Copy data
    }
    return from_blob(data, {3, 3}, DType::FLOAT64, device); // Create tensor
}

//------------------------------------------------------------------------------
// reshape implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& tensor,
                                const std::vector<int64_t>& new_shape) {
    // Return a view (or copy) with new shape; same underlying data
    if (!tensor) {
        throw std::invalid_argument("reshape: input tensor is null");
    }
    // Validate total elements match
    int64_t new_numel = 1;
    for (auto d : new_shape) new_numel *= d;         // Compute product of new shape
    if (new_numel != tensor->numel()) {
        throw std::invalid_argument("reshape: total elements must remain the same");
    }
    // Create a new tensor that shares the same data buffer
    return tensor->view(new_shape);                  // Returns a view (or copy if not possible)
}

//------------------------------------------------------------------------------
// flatten implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> flatten(const std::shared_ptr<Tensor>& tensor) {
    // Flatten tensor to 1D
    if (!tensor) {
        throw std::invalid_argument("flatten: input tensor is null");
    }
    return reshape(tensor, {tensor->numel()});       // Reshape to 1D
}

//------------------------------------------------------------------------------
// squeeze implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> squeeze(const std::shared_ptr<Tensor>& tensor, int64_t dim) {
    // Remove dimensions of size 1
    if (!tensor) {
        throw std::invalid_argument("squeeze: input tensor is null");
    }
    const auto& shape = tensor->shape();
    std::vector<int64_t> new_shape;
    if (dim == -1) {
        // Remove all size-1 dimensions
        for (auto d : shape) {
            if (d != 1) new_shape.push_back(d);      // Keep non-1 dimensions
        }
    } else {
        // Remove specified dimension if it's size 1
        if (dim < 0) dim += static_cast<int64_t>(shape.size()); // Handle negative index
        if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
            throw std::invalid_argument("squeeze: dimension out of range");
        }
        if (shape[dim] != 1) {
            throw std::invalid_argument("squeeze: dimension to squeeze must be of size 1");
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            if (static_cast<int64_t>(i) != dim) new_shape.push_back(shape[i]); // Skip the dim
        }
    }
    return reshape(tensor, new_shape);               // Return reshaped tensor
}

//------------------------------------------------------------------------------
// unsqueeze implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> unsqueeze(const std::shared_ptr<Tensor>& tensor, int64_t dim) {
    // Add a dimension of size 1 at specified position
    if (!tensor) {
        throw std::invalid_argument("unsqueeze: input tensor is null");
    }
    const auto& shape = tensor->shape();
    if (dim < 0) dim += static_cast<int64_t>(shape.size()) + 1; // Handle negative
    if (dim < 0 || dim > static_cast<int64_t>(shape.size())) {
        throw std::invalid_argument("unsqueeze: dimension out of range");
    }
    std::vector<int64_t> new_shape = shape;
    new_shape.insert(new_shape.begin() + dim, 1);    // Insert size-1 dimension
    return reshape(tensor, new_shape);               // Return reshaped tensor
}

//------------------------------------------------------------------------------
// cat implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> cat(const std::vector<std::shared_ptr<Tensor>>& tensors, int64_t dim) {
    // Concatenate tensors along a dimension
    if (tensors.empty()) {
        throw std::invalid_argument("cat: tensor list is empty");
    }
    auto first = tensors[0];                         // Reference tensor for dtype/device/shape
    if (!first) {
        throw std::invalid_argument("cat: first tensor is null");
    }
    // Validate all tensors have same dtype and device
    for (const auto& t : tensors) {
        if (t->dtype() != first->dtype() || t->device().type != first->device().type) {
            throw std::invalid_argument("cat: all tensors must have same dtype and device");
        }
    }
    const auto& shape0 = first->shape();
    if (dim < 0) dim += static_cast<int64_t>(shape0.size()); // Handle negative dim
    if (dim < 0 || dim >= static_cast<int64_t>(shape0.size())) {
        throw std::invalid_argument("cat: dimension out of range");
    }
    // Compute new shape
    std::vector<int64_t> new_shape = shape0;
    new_shape[dim] = 0;
    for (const auto& t : tensors) {
        const auto& s = t->shape();
        if (s.size() != shape0.size()) {
            throw std::invalid_argument("cat: all tensors must have same number of dimensions");
        }
        for (size_t d = 0; d < s.size(); ++d) {
            if (static_cast<int64_t>(d) != dim && s[d] != shape0[d]) {
                throw std::invalid_argument("cat: tensor shapes mismatch on non-concatenated dimensions");
            }
        }
        new_shape[dim] += s[dim];                    // Accumulate concat dimension size
    }
    // Allocate result tensor
    auto result = std::make_shared<Tensor>(new_shape, first->dtype(), first->device());
    // Copy data from each tensor into the result
    int64_t offset = 0;                              // Offset along concat dimension
    for (const auto& t : tensors) {
        // Copy tensor t into result at appropriate slice
        result->copy_from_tensor(*t, offset, dim);   // Custom method to copy slice
        offset += t->shape()[dim];                   // Advance offset
    }
    return result;                                   // Return concatenated tensor
}

//------------------------------------------------------------------------------
// stack implementation
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> stack(const std::vector<std::shared_ptr<Tensor>>& tensors, int64_t dim) {
    // Stack tensors along a new dimension
    if (tensors.empty()) {
        throw std::invalid_argument("stack: tensor list is empty");
    }
    auto first = tensors[0];
    if (!first) {
        throw std::invalid_argument("stack: first tensor is null");
    }
    const auto& shape0 = first->shape();
    // All tensors must have identical shape
    for (const auto& t : tensors) {
        if (t->shape() != shape0) {
            throw std::invalid_argument("stack: all tensors must have the same shape");
        }
        if (t->dtype() != first->dtype() || t->device().type != first->device().type) {
            throw std::invalid_argument("stack: all tensors must have same dtype and device");
        }
    }
    if (dim < 0) dim += static_cast<int64_t>(shape0.size()) + 1;
    if (dim < 0 || dim > static_cast<int64_t>(shape0.size())) {
        throw std::invalid_argument("stack: dimension out of range");
    }
    // New shape: insert dimension of size N at 'dim'
    std::vector<int64_t> new_shape = shape0;
    new_shape.insert(new_shape.begin() + dim, static_cast<int64_t>(tensors.size()));
    auto result = std::make_shared<Tensor>(new_shape, first->dtype(), first->device());
    // Copy each tensor into result
    for (size_t i = 0; i < tensors.size(); ++i) {
        result->copy_from_tensor_at_index(*tensors[i], static_cast<int64_t>(i), dim);
    }
    return result;
}

} // namespace grad
} // namespace genesis