// main/genesis/__init__.h

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <atomic>
#include <unordered_set>
#include <map>

namespace genesis {

// Forward declarations
class Logger;
class Scene;
enum class Backend : uint8_t;

//-----------------------------------------------------------------------------
// Global state (mirrors Python module globals)
//-----------------------------------------------------------------------------
inline std::atomic<bool> g_initialized{false};
inline std::vector<std::weak_ptr<Scene>> g_scene_registry;
inline std::unordered_set<std::pair<std::function<void()>, std::function<void()>>> g_module_registry;
inline std::string g_theme = "dark";
inline std::unique_ptr<Logger> g_logger;
inline Backend g_backend;
inline bool g_use_ndarray = false;
inline bool g_use_fastcache = false;
inline bool g_use_zerocopy = false;
inline double g_eps = 1e-15;

// Version string (from version.py)
inline const std::string VERSION = "0.4.6";

//-----------------------------------------------------------------------------
// Backend enumeration (mirrors constants.backend)
//-----------------------------------------------------------------------------
enum class Backend : uint8_t {
    CPU = 0,
    GPU = 1,
    CUDA = 2,
    AMDGPU = 3,
    METAL = 4
};

//-----------------------------------------------------------------------------
// Exception class
//-----------------------------------------------------------------------------
class GenesisException : public std::runtime_error {
public:
    explicit GenesisException(const std::string& msg) : std::runtime_error(msg) {}
};

//-----------------------------------------------------------------------------
// Logger (simplified placeholder)
//-----------------------------------------------------------------------------
class Logger {
public:
    enum class Level { DEBUG, INFO, WARNING, ERROR };
    
    explicit Logger(Level level = Level::INFO) : m_level(level) {}
    void setLevel(Level level) { m_level = level; }
    Level getLevel() const { return m_level; }
    
    void debug(const std::string& msg) { if (m_level <= Level::DEBUG) log("DEBUG", msg); }
    void info(const std::string& msg)  { if (m_level <= Level::INFO)  log("INFO", msg); }
    void warn(const std::string& msg)  { if (m_level <= Level::WARNING) log("WARN", msg); }
    void error(const std::string& msg) { if (m_level <= Level::ERROR) log("ERROR", msg); }
    
private:
    Level m_level;
    void log(const std::string& level, const std::string& msg);
};

//-----------------------------------------------------------------------------
// Device information
//-----------------------------------------------------------------------------
struct DeviceInfo {
    Backend backend;
    std::string name;
    size_t total_memory_bytes;
};

//-----------------------------------------------------------------------------
// Core initialization function
//-----------------------------------------------------------------------------
inline void init(
    Backend backend = Backend::GPU,
    const std::string& precision = "32",
    Logger::Level logging_level = Logger::Level::INFO,
    bool debug = false,
    uint64_t seed = 0,
    double eps = 1e-15,
    const std::string& theme = "dark",
    bool logger_verbose_time = false,
    bool performance_mode = false
) {
    if (g_initialized.load()) {
        throw GenesisException("Genesis already initialized.");
    }
    
    // Ensure clean state
    destroy();
    
    // Validate theme
    if (theme != "dark" && theme != "light" && theme != "dumb") {
        throw GenesisException("Unsupported theme: " + theme);
    }
    g_theme = theme;
    
    // Validate precision
    if (precision != "32" && precision != "64") {
        throw GenesisException("Unsupported precision type: " + precision);
    }
    
    // Determine backend candidates
    std::vector<Backend> backend_candidates;
    if (debug) {
        backend_candidates = {Backend::CPU};
    } else if (backend == Backend::GPU) {
        backend_candidates = {Backend::CUDA, Backend::AMDGPU, Backend::METAL, Backend::CPU};
    } else {
        backend_candidates = {backend};
    }
    
    // Try each candidate until one works
    DeviceInfo device_info;
    bool found = false;
    for (auto cand : backend_candidates) {
        try {
            device_info = get_device(cand);
            g_backend = device_info.backend;
            found = true;
            break;
        } catch (const GenesisException&) {
            // Try next candidate
        }
    }
    
    if (!found) {
        throw GenesisException("No suitable backend found.");
    }
    
    // Set global precision epsilon
    g_eps = eps;
    
    // Initialize logger
    g_logger = std::make_unique<Logger>(logging_level);
    
    // Set random seed if provided
    if (seed != 0) {
        set_random_seed(seed);
    }
    
    g_initialized.store(true);
}

//-----------------------------------------------------------------------------
// Cleanup / destroy
//-----------------------------------------------------------------------------
inline void destroy() {
    // Clear scene registry
    g_scene_registry.clear();
    g_module_registry.clear();
    g_logger.reset();
    g_initialized.store(false);
}

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------
inline DeviceInfo get_device(Backend preferred) {
    // Stub implementation - actual implementation would query hardware
    DeviceInfo info;
    info.backend = preferred;
    info.name = "default";
    info.total_memory_bytes = 0;
    return info;
}

inline void set_random_seed(uint64_t seed) {
    // Stub - would set global random seed
}

//-----------------------------------------------------------------------------
// Version getter
//-----------------------------------------------------------------------------
inline const std::string& version() {
    return VERSION;
}

} // namespace genesis