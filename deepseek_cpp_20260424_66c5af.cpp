// genesis/__init__.cpp

#include "genesis/__init__.h"
#include <iostream>
#include <chrono>
#include <random>
#include <cstdlib>
#include <cstring>
#include <sstream>

#ifdef GENESIS_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef GENESIS_USE_HIP
#include <hip/hip_runtime.h>
#endif

#ifdef GENESIS_USE_METAL
// Metal headers would be included here in a real implementation
#include <mach/mach.h>
#endif

namespace genesis {

//------------------------------------------------------------------------------
// Logger implementation
//------------------------------------------------------------------------------
void Logger::log(const std::string& level, const std::string& msg) {
    // Get current time for logging
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    struct tm timeinfo;
#ifdef _WIN32
    localtime_s(&timeinfo, &time_t);
#else
    localtime_r(&time_t, &timeinfo);
#endif
    
    char time_buf[32];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &timeinfo);
    
    // Color output based on level (if terminal supports it)
    const char* color = "";
    const char* reset = "";
    if (g_theme != "dumb") {
        if (level == "DEBUG") color = "\033[36m";   // Cyan
        else if (level == "INFO")  color = "\033[32m"; // Green
        else if (level == "WARN")  color = "\033[33m"; // Yellow
        else if (level == "ERROR") color = "\033[31m"; // Red
        reset = "\033[0m";
    }
    
    std::ostringstream oss;
    oss << "[" << time_buf << "." << std::setfill('0') << std::setw(3) << ms.count() << "] "
        << color << "[" << level << "]" << reset << " " << msg << std::endl;
    
    if (level == "ERROR") {
        std::cerr << oss.str();
    } else {
        std::cout << oss.str();
    }
}

//------------------------------------------------------------------------------
// Device query implementation (real logic)
//------------------------------------------------------------------------------
DeviceInfo get_device(Backend preferred) {
    DeviceInfo info;
    info.backend = Backend::CPU;  // fallback default
    info.name = "CPU (fallback)";
    info.total_memory_bytes = 0;
    
    // Helper lambda to try CUDA
    auto try_cuda = [&]() -> bool {
#ifdef GENESIS_USE_CUDA
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err == cudaSuccess && device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            info.backend = Backend::CUDA;
            info.name = std::string(prop.name) + " (CUDA)";
            info.total_memory_bytes = prop.totalGlobalMem;
            return true;
        }
#endif
        return false;
    };
    
    auto try_hip = [&]() -> bool {
#ifdef GENESIS_USE_HIP
        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err == hipSuccess && device_count > 0) {
            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, 0);
            info.backend = Backend::AMDGPU;
            info.name = std::string(prop.name) + " (AMDGPU)";
            info.total_memory_bytes = prop.totalGlobalMem;
            return true;
        }
#endif
        return false;
    };
    
    auto try_metal = [&]() -> bool {
#ifdef GENESIS_USE_METAL
        // Check if Metal is available (simplified)
        if (MTLCreateSystemDefaultDevice() != nullptr) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            info.backend = Backend::METAL;
            info.name = std::string([device.name UTF8String]) + " (Metal)";
            info.total_memory_bytes = [device recommendedMaxWorkingSetSize];
            [device release];
            return true;
        }
#endif
        return false;
    };
    
    auto try_cpu = [&]() -> bool {
        // CPU is always available
        info.backend = Backend::CPU;
#ifdef __x86_64__
        info.name = "x86_64 CPU";
#elif defined(__aarch64__)
        info.name = "ARM64 CPU";
#else
        info.name = "Generic CPU";
#endif
        // Approximate system memory
#ifdef _WIN32
        MEMORYSTATUSEX memStatus;
        memStatus.dwLength = sizeof(memStatus);
        if (GlobalMemoryStatusEx(&memStatus)) {
            info.total_memory_bytes = memStatus.ullTotalPhys;
        }
#elif defined(__APPLE__)
        int mib[2] = { CTL_HW, HW_MEMSIZE };
        size_t len = sizeof(info.total_memory_bytes);
        sysctl(mib, 2, &info.total_memory_bytes, &len, nullptr, 0);
#else
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        info.total_memory_bytes = static_cast<size_t>(pages) * static_cast<size_t>(page_size);
#endif
        return true;
    };
    
    // Try in order of preference or fallback
    if (preferred == Backend::CUDA) {
        if (try_cuda()) return info;
        if (try_hip()) return info;
        if (try_metal()) return info;
        try_cpu();
    } else if (preferred == Backend::AMDGPU) {
        if (try_hip()) return info;
        if (try_cuda()) return info;
        if (try_metal()) return info;
        try_cpu();
    } else if (preferred == Backend::METAL) {
        if (try_metal()) return info;
        if (try_cuda()) return info;
        if (try_hip()) return info;
        try_cpu();
    } else if (preferred == Backend::GPU) {
        // Try all GPU backends in order of likelihood
        if (try_cuda()) return info;
        if (try_hip()) return info;
        if (try_metal()) return info;
        try_cpu();
    } else { // CPU explicitly requested
        try_cpu();
    }
    
    return info;
}

//------------------------------------------------------------------------------
// Random seed setting
//------------------------------------------------------------------------------
void set_random_seed(uint64_t seed) {
    // If seed is 0, use a random device to generate a seed
    if (seed == 0) {
        std::random_device rd;
        seed = static_cast<uint64_t>(rd()) << 32 | rd();
    }
    // Set seed for various RNGs (std::mt19937, etc.)
    std::srand(static_cast<unsigned int>(seed));
    // In a real implementation, we would also set seeds for CUDA, etc.
#ifdef GENESIS_USE_CUDA
    // Example: set CUDA random seed
    // curandSetPseudoRandomGeneratorSeed(...)
#endif
}

//------------------------------------------------------------------------------
// Explicit template instantiations (if any needed)
//------------------------------------------------------------------------------

} // namespace genesis