// genesis/grad/__init__.h

#pragma once

//------------------------------------------------------------------------------
// Genesis Grad Module - Automatic differentiation and gradient computation.
// Provides tensor operations with automatic differentiation support.
// This header serves as the main entry point for the grad module.
//------------------------------------------------------------------------------

#include "genesis/grad/tensor.h"                    // Core Tensor class
#include "genesis/grad/creation_ops.h"              // Factory functions for tensors
#include <memory>                                   // std::shared_ptr
#include <vector>                                   // std::vector

namespace genesis {
namespace grad {

//------------------------------------------------------------------------------
// Forward declarations of all tensor operations
//------------------------------------------------------------------------------
class Tensor;
class GradContext;

//------------------------------------------------------------------------------
// Module initialization
//------------------------------------------------------------------------------
void initialize_grad();                             // Initialize the autograd system
void shutdown_grad();                               // Clean up resources
bool is_grad_initialized();                         // Check if module is initialized

//------------------------------------------------------------------------------
// Global gradient mode control
//------------------------------------------------------------------------------
void set_grad_enabled(bool enabled);                // Enable/disable gradient tracking globally
bool is_grad_enabled();                             // Check if gradient tracking is enabled

//------------------------------------------------------------------------------
// Device management
//------------------------------------------------------------------------------
enum class DeviceType : uint8_t {
    CPU = 0,
    CUDA = 1,
    METAL = 2,
    AMD = 3
};

struct Device {
    DeviceType type = DeviceType::CPU;
    int index = 0;
    
    bool operator==(const Device& other) const {
        return type == other.type && index == other.index; // Compare device
    }
};

Device get_default_device();                        // Get current default device
void set_default_device(Device device);             // Set default device for new tensors
int device_count(DeviceType type);                  // Number of available devices of a type

//------------------------------------------------------------------------------
// Gradient context management (for custom autograd functions)
//------------------------------------------------------------------------------
class GradContext {
public:
    virtual ~GradContext() = default;
    virtual void save_for_backward(const std::vector<std::shared_ptr<Tensor>>& tensors) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> saved_tensors() const = 0;
    virtual void mark_dirty(const std::shared_ptr<Tensor>& tensor) = 0;
};

//------------------------------------------------------------------------------
// Function for custom autograd operations
//------------------------------------------------------------------------------
class Function {
public:
    virtual ~Function() = default;
    virtual std::vector<std::shared_ptr<Tensor>> forward(
        GradContext& ctx,
        const std::vector<std::shared_ptr<Tensor>>& inputs) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> backward(
        GradContext& ctx,
        const std::vector<std::shared_ptr<Tensor>>& grad_outputs) = 0;
};

//------------------------------------------------------------------------------
// High-level gradient computation
//------------------------------------------------------------------------------
// Compute gradients of outputs with respect to inputs
void backward(const std::vector<std::shared_ptr<Tensor>>& outputs,
              const std::vector<std::shared_ptr<Tensor>>& inputs,
              const std::vector<std::shared_ptr<Tensor>>& grad_outputs = {});

// Compute gradient of a scalar output with respect to all tensors that require grad
void backward(std::shared_ptr<Tensor> output);

//------------------------------------------------------------------------------
// No-gradient context manager (RAII)
//------------------------------------------------------------------------------
class NoGradGuard {
public:
    NoGradGuard();                                   // Disable gradient tracking in scope
    ~NoGradGuard();                                  // Restore previous state
private:
    bool prev_enabled_;                              // Previous gradient tracking state
};

//------------------------------------------------------------------------------
// Utility: detach tensor from computation graph
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> detach(std::shared_ptr<Tensor> tensor);

//------------------------------------------------------------------------------
// Version information
//------------------------------------------------------------------------------
const char* grad_version();

} // namespace grad
} // namespace genesis