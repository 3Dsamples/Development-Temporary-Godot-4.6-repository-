// genesis/grad/__init__.cpp

#include "genesis/grad/__init__.h"                  // Include corresponding header
#include "genesis/grad/tensor.h"                    // Tensor class for autograd
#include <atomic>                                   // std::atomic for thread-safe flags
#include <mutex>                                    // std::mutex for device management
#include <vector>                                   // std::vector
#include <unordered_map>                            // std::unordered_map for device cache
#include <stdexcept>                                // std::runtime_error
#include <stack>                                    // std::stack for NoGradGuard state

namespace genesis {
namespace grad {

//------------------------------------------------------------------------------
// Global module state
//------------------------------------------------------------------------------
static std::atomic<bool> g_grad_initialized{false};  // Initialization flag
static std::atomic<bool> g_grad_enabled{true};       // Global gradient tracking enabled

//------------------------------------------------------------------------------
// Device management state
//------------------------------------------------------------------------------
static Device g_default_device{DeviceType::CPU, 0};  // Current default device
static std::mutex g_device_mutex;                    // Protects device state
static std::unordered_map<DeviceType, int> g_device_counts; // Cached device counts

//------------------------------------------------------------------------------
// NoGradGuard state stack (for nested guards)
//------------------------------------------------------------------------------
static std::stack<bool> g_grad_state_stack;          // Stack of previous enabled states
static std::mutex g_grad_state_mutex;                // Mutex for stack access

//------------------------------------------------------------------------------
// Module initialization
//------------------------------------------------------------------------------
void initialize_grad() {
    // Check if already initialized
    if (g_grad_initialized.load()) {
        return;                                      // Already initialized, nothing to do
    }
    
    // Initialize device counts by querying available hardware
    std::lock_guard<std::mutex> lock(g_device_mutex);
    
    // CPU is always available
    g_device_counts[DeviceType::CPU] = 1;            // At least one CPU device
    
    // Query CUDA devices if available
#ifdef GENESIS_USE_CUDA
    int cuda_count = 0;
    cudaError_t err = cudaGetDeviceCount(&cuda_count);
    if (err == cudaSuccess) {
        g_device_counts[DeviceType::CUDA] = cuda_count; // Store CUDA device count
    } else {
        g_device_counts[DeviceType::CUDA] = 0;       // No CUDA devices
    }
#else
    g_device_counts[DeviceType::CUDA] = 0;           // CUDA not compiled in
#endif

    // Query Metal devices if available
#ifdef GENESIS_USE_METAL
    if (@available(macOS 10.11, *)) {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        g_device_counts[DeviceType::METAL] = static_cast<int>([devices count]);
        [devices release];
    } else {
        g_device_counts[DeviceType::METAL] = 0;
    }
#else
    g_device_counts[DeviceType::METAL] = 0;          // Metal not compiled in
#endif

    // Query AMD (HIP) devices if available
#ifdef GENESIS_USE_HIP
    int hip_count = 0;
    hipError_t err = hipGetDeviceCount(&hip_count);
    if (err == hipSuccess) {
        g_device_counts[DeviceType::AMD] = hip_count;
    } else {
        g_device_counts[DeviceType::AMD] = 0;
    }
#else
    g_device_counts[DeviceType::AMD] = 0;            // HIP not compiled in
#endif

    // Set default device to first GPU if available, else CPU
    if (g_device_counts[DeviceType::CUDA] > 0) {
        g_default_device = Device{DeviceType::CUDA, 0};
    } else if (g_device_counts[DeviceType::METAL] > 0) {
        g_default_device = Device{DeviceType::METAL, 0};
    } else if (g_device_counts[DeviceType::AMD] > 0) {
        g_default_device = Device{DeviceType::AMD, 0};
    } else {
        g_default_device = Device{DeviceType::CPU, 0};
    }
    
    // Initialize tensor factory (if any static initialization needed)
    Tensor::initialize_factory();                    // Setup tensor creation backend
    
    g_grad_initialized.store(true);                  // Mark as initialized
}

void shutdown_grad() {
    // Clean up autograd resources
    if (!g_grad_initialized.load()) {
        return;                                      // Not initialized, nothing to shut down
    }
    
    // Clear any cached tensors or contexts
    Tensor::clear_global_cache();                    // Free tensor caches
    
    // Clear the NoGradGuard state stack
    {
        std::lock_guard<std::mutex> lock(g_grad_state_mutex);
        while (!g_grad_state_stack.empty()) {
            g_grad_state_stack.pop();                // Empty the stack
        }
    }
    
    g_grad_initialized.store(false);                 // Mark as uninitialized
}

bool is_grad_initialized() {
    // Return current initialization status
    return g_grad_initialized.load();                // Atomic read
}

//------------------------------------------------------------------------------
// Global gradient mode control
//------------------------------------------------------------------------------
void set_grad_enabled(bool enabled) {
    // Set global gradient tracking flag
    g_grad_enabled.store(enabled);                   // Atomic store
}

bool is_grad_enabled() {
    // Return current gradient tracking status
    return g_grad_enabled.load();                    // Atomic read
}

//------------------------------------------------------------------------------
// Device management
//------------------------------------------------------------------------------
Device get_default_device() {
    // Return current default device
    if (!g_grad_initialized.load()) {
        initialize_grad();                           // Auto-initialize if needed
    }
    std::lock_guard<std::mutex> lock(g_device_mutex);
    return g_default_device;                         // Return copy of current default
}

void set_default_device(Device device) {
    // Set new default device for tensor creation
    if (!g_grad_initialized.load()) {
        initialize_grad();                           // Ensure device counts are known
    }
    std::lock_guard<std::mutex> lock(g_device_mutex);
    // Validate that the requested device exists
    auto it = g_device_counts.find(device.type);
    if (it == g_device_counts.end() || device.index >= it->second) {
        throw std::runtime_error("Requested device not available");
    }
    g_default_device = device;                       // Update default
}

int device_count(DeviceType type) {
    // Return number of available devices of given type
    if (!g_grad_initialized.load()) {
        initialize_grad();                           // Auto-initialize
    }
    std::lock_guard<std::mutex> lock(g_device_mutex);
    auto it = g_device_counts.find(type);
    if (it != g_device_counts.end()) {
        return it->second;                           // Return cached count
    }
    return 0;                                        // Unknown type
}

//------------------------------------------------------------------------------
// NoGradGuard implementation
//------------------------------------------------------------------------------
NoGradGuard::NoGradGuard() {
    // Disable gradient tracking and save previous state
    std::lock_guard<std::mutex> lock(g_grad_state_mutex);
    prev_enabled_ = g_grad_enabled.load();           // Read current state
    g_grad_state_stack.push(prev_enabled_);          // Push to stack
    g_grad_enabled.store(false);                     // Disable tracking
}

NoGradGuard::~NoGradGuard() {
    // Restore gradient tracking to previous state
    std::lock_guard<std::mutex> lock(g_grad_state_mutex);
    if (!g_grad_state_stack.empty()) {
        bool restore_state = g_grad_state_stack.top(); // Get previous state
        g_grad_state_stack.pop();                    // Remove from stack
        // Only restore if stack is empty or top is different? Actually we restore to the popped value.
        g_grad_enabled.store(restore_state);         // Restore
    }
}

//------------------------------------------------------------------------------
// Utility: detach tensor from computation graph
//------------------------------------------------------------------------------
std::shared_ptr<Tensor> detach(std::shared_ptr<Tensor> tensor) {
    // Create a new tensor with same data but no gradient tracking
    if (!tensor) return nullptr;
    auto detached = std::make_shared<Tensor>(tensor->data(), tensor->shape(), tensor->device());
    detached->set_requires_grad(false);              // Disable gradient tracking
    return detached;
}

//------------------------------------------------------------------------------
// High-level gradient computation
//------------------------------------------------------------------------------
void backward(const std::vector<std::shared_ptr<Tensor>>& outputs,
              const std::vector<std::shared_ptr<Tensor>>& inputs,
              const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
    // Compute gradients of outputs with respect to inputs
    if (!g_grad_initialized.load()) {
        initialize_grad();                           // Ensure module is ready
    }
    if (outputs.empty()) return;                     // Nothing to differentiate
    
    // Create a gradient context to hold intermediate values
    auto ctx = std::make_shared<GradContextImpl>();  // Implementation of GradContext
    
    // Prepare initial gradients for outputs
    std::vector<std::shared_ptr<Tensor>> output_grads;
    output_grads.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (i < grad_outputs.size() && grad_outputs[i]) {
            output_grads.push_back(grad_outputs[i]); // Use provided gradient
        } else {
            // Default: ones_like(output)
            auto ones = Tensor::ones_like(outputs[i]);
            output_grads.push_back(ones);            // Default gradient is 1
        }
    }
    
    // Build the computation graph backwards from outputs
    std::unordered_map<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> grad_map;
    for (size_t i = 0; i < outputs.size(); ++i) {
        grad_map[outputs[i]] = output_grads[i];      // Seed gradient map
    }
    
    // Topological sort of nodes in reverse order
    auto sorted_nodes = topological_sort(outputs);   // Nodes in forward order
    std::reverse(sorted_nodes.begin(), sorted_nodes.end()); // Reverse for backward pass
    
    for (auto& node : sorted_nodes) {
        if (grad_map.find(node) == grad_map.end()) continue; // No gradient to propagate
        
        auto grad = grad_map[node];                  // Incoming gradient
        // If node has a grad_fn (the operation that produced it), call backward
        auto grad_fn = node->grad_fn();
        if (grad_fn) {
            std::vector<std::shared_ptr<Tensor>> node_inputs = node->inputs();
            auto input_grads = grad_fn->backward(ctx, {grad}); // Compute gradients w.r.t inputs
            for (size_t j = 0; j < node_inputs.size(); ++j) {
                if (node_inputs[j]->requires_grad()) {
                    if (grad_map.find(node_inputs[j]) == grad_map.end()) {
                        grad_map[node_inputs[j]] = input_grads[j];
                    } else {
                        // Accumulate gradient
                        grad_map[node_inputs[j]] = grad_map[node_inputs[j]] + input_grads[j];
                    }
                }
            }
        }
    }
    
    // Assign gradients to input tensors
    for (auto& input : inputs) {
        auto it = grad_map.find(input);
        if (it != grad_map.end()) {
            input->set_grad(it->second);             // Store computed gradient
        }
    }
}

void backward(std::shared_ptr<Tensor> output) {
    // Convenience overload: compute gradients for all tensors that require grad
    if (!output) return;
    if (output->shape().size() != 0 || output->shape()[0] != 1) {
        throw std::runtime_error("grad can be implicitly created only for scalar outputs");
    }
    std::vector<std::shared_ptr<Tensor>> inputs;
    // Collect all tensors in the graph that require grad
    auto nodes = topological_sort({output});
    for (auto& node : nodes) {
        if (node->requires_grad()) {
            inputs.push_back(node);                  // Add to inputs list
        }
    }
    backward({output}, inputs, {});                  // Call full backward
}

//------------------------------------------------------------------------------
// Version information
//------------------------------------------------------------------------------
const char* grad_version() {
    // Return the version string of the grad module
    return "0.4.6";                                  // Match Genesis version
}

} // namespace grad
} // namespace genesis