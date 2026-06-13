// genesis/engine/__init__.cpp

#include "genesis/engine/__init__.h"
#include "genesis/__init__.h"
#include "genesis/version.h"

#ifdef GENESIS_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef GENESIS_USE_HIP
#include <hip/hip_runtime.h>
#endif

#ifdef GENESIS_USE_METAL
#include <Metal/Metal.h>
#endif

#include <thread>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <fstream>
#include <iostream>
#include <cstring>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Global engine state
//------------------------------------------------------------------------------
namespace {
    std::atomic<bool> g_initialized{false};
    EngineConfig g_config;
    std::array<double, 3> g_gravity = {0.0, 0.0, -9.80665};
    double g_time_step = 0.01;
    std::atomic<bool> g_profiling_enabled{false};
    std::mutex g_engine_mutex;
    
    // Performance counters
    struct ProfilingData {
        std::chrono::steady_clock::time_point frame_start;
        double frame_time_ms = 0.0;
        double physics_time_ms = 0.0;
        double collision_time_ms = 0.0;
        double solver_time_ms = 0.0;
        double render_time_ms = 0.0;
        size_t particle_count = 0;
        size_t constraint_count = 0;
        size_t collision_pairs = 0;
    };
    ProfilingData g_profiling_data;
    
    void detect_hardware_info() {
        // CPU cores already handled by std::thread::hardware_concurrency()
    }
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------
void initialize(const EngineConfig& config) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    
    if (g_initialized.load()) {
        if (g_config.enable_logging) {
            std::cerr << "[Genesis Engine] Already initialized." << std::endl;
        }
        return;
    }
    
    g_config = config;
    
    // Validate configuration
    if (g_config.num_threads == 0) {
        g_config.num_threads = std::thread::hardware_concurrency();
        if (g_config.num_threads == 0) g_config.num_threads = 1;
    }
    
    // Initialize Genesis core if not already done
    if (!genesis::g_initialized.load()) {
        // Determine backend based on features
        Backend backend = Backend::CPU;
        if (has_feature(g_config.features, EngineFeature::GPU_ACCELERATION)) {
            backend = Backend::GPU;
        }
        genesis::init(backend, "32", Logger::Level::INFO, false, 0, 1e-15, 
                      genesis::g_theme, false, false);
    }
    
    // Initialize GPU if requested
    if (has_feature(g_config.features, EngineFeature::GPU_ACCELERATION)) {
#ifdef GENESIS_USE_CUDA
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err == cudaSuccess && device_count > 0) {
            cudaSetDevice(static_cast<int>(g_config.gpu_device_id));
            if (g_config.enable_logging) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, g_config.gpu_device_id);
                std::cout << "[Genesis Engine] Using CUDA device: " << prop.name << std::endl;
            }
        } else {
            if (g_config.enable_logging) {
                std::cerr << "[Genesis Engine] GPU acceleration requested but no CUDA device found." << std::endl;
            }
        }
#elif defined(GENESIS_USE_HIP)
        int device_count;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err == hipSuccess && device_count > 0) {
            hipSetDevice(static_cast<int>(g_config.gpu_device_id));
            if (g_config.enable_logging) {
                hipDeviceProp_t prop;
                hipGetDeviceProperties(&prop, g_config.gpu_device_id);
                std::cout << "[Genesis Engine] Using HIP device: " << prop.name << std::endl;
            }
        }
#elif defined(GENESIS_USE_METAL)
        if (@available(macOS 10.11, *)) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device) {
                if (g_config.enable_logging) {
                    std::cout << "[Genesis Engine] Using Metal device: " << [device.name UTF8String] << std::endl;
                }
                [device release];
            }
        }
#endif
    }
    
    // Initialize threading
    if (g_config.num_threads > 1 && g_config.enable_logging) {
        std::cout << "[Genesis Engine] Using " << g_config.num_threads << " threads." << std::endl;
    }
    
    g_profiling_enabled = g_config.enable_profiling;
    g_initialized = true;
    
    if (g_config.enable_logging && !g_config.log_file_path.empty()) {
        // Open log file (we'll just note that logging to file is enabled)
        std::cout << "[Genesis Engine] Logging to file: " << g_config.log_file_path << std::endl;
    }
}

void shutdown() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    
    if (!g_initialized.load()) return;
    
    if (g_config.enable_logging) {
        std::cout << "[Genesis Engine] Shutting down." << std::endl;
    }
    
    // Clear any remaining scenes or resources
    // (Scenes should be destroyed before engine shutdown)
    
    // Shutdown Genesis core
    genesis::destroy();
    
    g_initialized = false;
}

bool is_initialized() {
    return g_initialized.load();
}

const EngineConfig& get_config() {
    return g_config;
}

//------------------------------------------------------------------------------
// Global physics parameters
//------------------------------------------------------------------------------
void set_gravity(double gx, double gy, double gz) {
    g_gravity = {gx, gy, gz};
}

void set_gravity(const std::array<double, 3>& g) {
    g_gravity = g;
}

std::array<double, 3> get_gravity() {
    return g_gravity;
}

void set_time_step(double dt) {
    g_time_step = dt;
}

double get_time_step() {
    return g_time_step;
}

//------------------------------------------------------------------------------
// Profiling
//------------------------------------------------------------------------------
void enable_profiling(bool enable) {
    g_profiling_enabled = enable;
    g_config.enable_profiling = enable;
}

bool is_profiling_enabled() {
    return g_profiling_enabled.load();
}

//------------------------------------------------------------------------------
// Version
//------------------------------------------------------------------------------
std::string get_version_string() {
    return genesis::full_version_string();
}

//------------------------------------------------------------------------------
// Hardware info utilities
//------------------------------------------------------------------------------
int get_num_cpu_cores() {
    return static_cast<int>(std::thread::hardware_concurrency());
}

std::vector<GPUDeviceInfo> get_available_gpus() {
    std::vector<GPUDeviceInfo> devices;
    
#ifdef GENESIS_USE_CUDA
    int count;
    if (cudaGetDeviceCount(&count) == cudaSuccess) {
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                GPUDeviceInfo info;
                info.name = prop.name;
                info.total_memory_bytes = prop.totalGlobalMem;
                info.compute_capability_major = prop.major;
                info.compute_capability_minor = prop.minor;
                devices.push_back(info);
            }
        }
    }
#elif defined(GENESIS_USE_HIP)
    int count;
    if (hipGetDeviceCount(&count) == hipSuccess) {
        for (int i = 0; i < count; ++i) {
            hipDeviceProp_t prop;
            if (hipGetDeviceProperties(&prop, i) == hipSuccess) {
                GPUDeviceInfo info;
                info.name = prop.name;
                info.total_memory_bytes = prop.totalGlobalMem;
                info.compute_capability_major = prop.major;
                info.compute_capability_minor = prop.minor;
                devices.push_back(info);
            }
        }
    }
#elif defined(GENESIS_USE_METAL)
    if (@available(macOS 10.11, *)) {
        NSArray<id<MTLDevice>>* mtlDevices = MTLCopyAllDevices();
        for (id<MTLDevice> device in mtlDevices) {
            GPUDeviceInfo info;
            info.name = [device.name UTF8String];
            info.total_memory_bytes = device.recommendedMaxWorkingSetSize;
            // Metal doesn't have compute capability like CUDA; set to 0
            info.compute_capability_major = 0;
            info.compute_capability_minor = 0;
            devices.push_back(info);
        }
        [mtlDevices release];
    }
#endif
    
    return devices;
}

} // namespace engine
} // namespace genesis