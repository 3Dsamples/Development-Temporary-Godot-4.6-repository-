// genesis/grad/creation_ops.h

#pragma once

//------------------------------------------------------------------------------
// Tensor creation operations - factory functions for building tensors.
// Includes zeros, ones, full, arange, linspace, random, and conversions.
//------------------------------------------------------------------------------

#include "genesis/grad/tensor.h"                    // Core Tensor class
#include "genesis/datatypes.h"                      // real, Vector3, etc.
#include <vector>                                   // std::vector for shape
#include <memory>                                   // std::shared_ptr
#include <random>                                   // std::mt19937 for random generation

namespace genesis {
namespace grad {

//------------------------------------------------------------------------------
// Basic creation functions
//------------------------------------------------------------------------------

// Create a tensor filled with zeros
std::shared_ptr<Tensor> zeros(const std::vector<int64_t>& shape,
                              DType dtype = DType::FLOAT32,
                              Device device = get_default_device());

// Create a tensor filled with ones
std::shared_ptr<Tensor> ones(const std::vector<int64_t>& shape,
                             DType dtype = DType::FLOAT32,
                             Device device = get_default_device());

// Create a tensor filled with a constant value
std::shared_ptr<Tensor> full(const std::vector<int64_t>& shape,
                             double value,
                             DType dtype = DType::FLOAT32,
                             Device device = get_default_device());

// Create a tensor with the same shape and device as another tensor, filled with zeros
std::shared_ptr<Tensor> zeros_like(const std::shared_ptr<Tensor>& other);

// Create a tensor with the same shape and device as another tensor, filled with ones
std::shared_ptr<Tensor> ones_like(const std::shared_ptr<Tensor>& other);

// Create a tensor with the same shape and device as another tensor, filled with a constant
std::shared_ptr<Tensor> full_like(const std::shared_ptr<Tensor>& other, double value);

//------------------------------------------------------------------------------
// Sequence generation
//------------------------------------------------------------------------------

// Create a 1D tensor with values from start to end (exclusive) with step
std::shared_ptr<Tensor> arange(double start, double end, double step = 1.0,
                               DType dtype = DType::FLOAT32,
                               Device device = get_default_device());

// Create a 1D tensor with values evenly spaced between start and end (inclusive)
std::shared_ptr<Tensor> linspace(double start, double end, int64_t num_steps,
                                 DType dtype = DType::FLOAT32,
                                 Device device = get_default_device());

// Create a 1D tensor with values logarithmically spaced between base^start and base^end
std::shared_ptr<Tensor> logspace(double start, double end, int64_t num_steps,
                                 double base = 10.0,
                                 DType dtype = DType::FLOAT32,
                                 Device device = get_default_device());

//------------------------------------------------------------------------------
// Identity and diagonal
//------------------------------------------------------------------------------

// Create a 2D identity matrix of size n x n
std::shared_ptr<Tensor> eye(int64_t n,
                            DType dtype = DType::FLOAT32,
                            Device device = get_default_device());

// Create a diagonal matrix from a 1D tensor, or extract diagonal from 2D tensor
std::shared_ptr<Tensor> diag(const std::shared_ptr<Tensor>& input, int64_t k = 0);

//------------------------------------------------------------------------------
// Random generation
//------------------------------------------------------------------------------

// Set global random seed for reproducibility
void manual_seed(uint64_t seed);

// Uniform distribution between 0 and 1
std::shared_ptr<Tensor> rand(const std::vector<int64_t>& shape,
                             DType dtype = DType::FLOAT32,
                             Device device = get_default_device());

// Standard normal distribution (mean=0, std=1)
std::shared_ptr<Tensor> randn(const std::vector<int64_t>& shape,
                              DType dtype = DType::FLOAT32,
                              Device device = get_default_device());

// Uniform integer distribution between low and high (exclusive)
std::shared_ptr<Tensor> randint(int64_t low, int64_t high,
                                const std::vector<int64_t>& shape,
                                Device device = get_default_device());

// Random permutation of integers from 0 to n-1
std::shared_ptr<Tensor> randperm(int64_t n,
                                 Device device = get_default_device());

// Create tensor with same shape as another, filled with uniform random values
std::shared_ptr<Tensor> rand_like(const std::shared_ptr<Tensor>& other);
std::shared_ptr<Tensor> randn_like(const std::shared_ptr<Tensor>& other);

//------------------------------------------------------------------------------
// Conversion from existing data
//------------------------------------------------------------------------------

// Create tensor from std::vector of data (1D)
template<typename T>
std::shared_ptr<Tensor> tensor(const std::vector<T>& data,
                               DType dtype = DType::FLOAT32,
                               Device device = get_default_device());

// Create tensor from multi-dimensional nested initializer list (recursive template)
template<typename T>
std::shared_ptr<Tensor> tensor(const std::initializer_list<T>& data,
                               DType dtype = DType::FLOAT32,
                               Device device = get_default_device());

// Create tensor from raw pointer and shape
std::shared_ptr<Tensor> from_blob(const void* data,
                                  const std::vector<int64_t>& shape,
                                  DType dtype,
                                  Device device = get_default_device());

// Convert a Genesis datatypes::Vector3 to a 1D tensor of size 3
std::shared_ptr<Tensor> from_vector3(const datatypes::Vector3& vec,
                                     Device device = get_default_device());

// Convert a Genesis datatypes::Matrix3r to a 2D tensor of shape (3,3)
std::shared_ptr<Tensor> from_matrix3(const datatypes::Matrix3r& mat,
                                     Device device = get_default_device());

//------------------------------------------------------------------------------
// Shape manipulation utilities
//------------------------------------------------------------------------------

// Create a tensor with the same data but new shape (returns view if possible)
std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& tensor,
                                const std::vector<int64_t>& new_shape);

// Flatten a tensor to 1D
std::shared_ptr<Tensor> flatten(const std::shared_ptr<Tensor>& tensor);

// Squeeze dimensions of size 1
std::shared_ptr<Tensor> squeeze(const std::shared_ptr<Tensor>& tensor, int64_t dim = -1);

// Unsqueeze: add a dimension of size 1 at specified position
std::shared_ptr<Tensor> unsqueeze(const std::shared_ptr<Tensor>& tensor, int64_t dim);

// Concatenate tensors along a dimension
std::shared_ptr<Tensor> cat(const std::vector<std::shared_ptr<Tensor>>& tensors, int64_t dim = 0);

// Stack tensors along a new dimension
std::shared_ptr<Tensor> stack(const std::vector<std::shared_ptr<Tensor>>& tensors, int64_t dim = 0);

//------------------------------------------------------------------------------
// Template implementation for tensor creation from vectors
//------------------------------------------------------------------------------
template<typename T>
std::shared_ptr<Tensor> tensor(const std::vector<T>& data,
                               DType dtype,
                               Device device) {
    // Determine DType from template parameter if not specified
    DType actual_dtype = dtype;
    if (dtype == DType::FLOAT32) {
        // Keep as float32 default
    }
    // Create tensor with shape {data.size()}
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(data.size())}, actual_dtype, device);
    // Copy data into tensor's buffer
    t->copy_from_host(data.data());                 // Copy from host memory
    return t;
}

// Specialization for nested initializer list (determine shape and flatten)
template<typename T>
std::shared_ptr<Tensor> tensor(const std::initializer_list<T>& data,
                               DType dtype,
                               Device device) {
    // Convert initializer list to vector and delegate
    std::vector<T> flat_data(data.begin(), data.end());
    return tensor(flat_data, dtype, device);
}

} // namespace grad
} // namespace genesis