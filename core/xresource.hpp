// core/xresource.hpp
#ifndef XTENSOR_XRESOURCE_HPP
#define XTENSOR_XRESOURCE_HPP

// ----------------------------------------------------------------------------
// xresource.hpp – Resource management system for xtensor
// ----------------------------------------------------------------------------
// This header provides a complete resource management framework:
//   - Resource handles with automatic reference counting
//   - Asynchronous loading with progress tracking and cancellation
//   - Resource caching with LRU eviction and memory budgeting
//   - Dependency resolution (e.g., material references in mesh)
//   - Hot‑reloading with change detection (file watcher)
//   - Serialization/deserialization of resource metadata
//   - Integration with classdb for polymorphic resource types
//   - Resource pipelines (import, process, cook)
//   - Memory‑mapped I/O for large resources
//
// All operations support bignumber::BigNumber for numeric resource data.
// FFT acceleration is available for procedural generation and processing.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <functional>
#include <type_traits>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <chrono>
#include <future>
#include <queue>
#include <any>

#include "xtensor_config.hpp"
#include "xarray.hpp"
#include "xclassdb.hpp"
#include "bignumber/bignumber.hpp"

namespace xt {
namespace resource {

// ========================================================================
// Forward declarations
// ========================================================================
class resource_base;
class resource_manager;
template <class T> class resource_handle;
class resource_loader;
class resource_dependency;
class resource_cache;
class resource_pipeline;

// ========================================================================
// Resource ID and version
// ========================================================================
using resource_id = std::string;      // unique identifier (e.g., "textures/brick.png")
using resource_version = uint64_t;    // incremented on change

// ========================================================================
// Resource state
// ========================================================================
enum class resource_state {
    unloaded,       // not yet loaded
    loading,        // loading in progress
    loaded,         // ready to use
    failed,         // loading failed
    unloading,      // being unloaded
    reloading       // hot‑reload in progress
};

// ========================================================================
// Base class for all managed resources
// ========================================================================
class resource_base : public std::enable_shared_from_this<resource_base> {
public:
    virtual ~resource_base() = default;

    // Core properties
    const resource_id& id() const noexcept { return m_id; }
    resource_state state() const noexcept { return m_state.load(); }
    resource_version version() const noexcept { return m_version; }
    size_t memory_usage() const noexcept { return m_memory_usage; }

    // Load/unload (override in derived classes)
    virtual bool load(const std::string& path) { return true; }
    virtual void unload() {}

    // Hot‑reload support
    virtual bool can_reload() const { return true; }
    virtual bool reload(const std::string& path) { unload(); return load(path); }

    // Dependencies (other resources this one depends on)
    virtual std::vector<resource_id> dependencies() const { return {}; }

    // Called by manager
    void set_id(const resource_id& id) { m_id = id; }
    void set_state(resource_state s) { m_state.store(s); }
    void set_version(resource_version v) { m_version = v; }
    void set_memory_usage(size_t bytes) { m_memory_usage = bytes; }

    // Reflection (for classdb)
    virtual const char* class_name() const { return "resource_base"; }

protected:
    resource_base() = default;

private:
    resource_id m_id;
    std::atomic<resource_state> m_state{resource_state::unloaded};
    resource_version m_version = 0;
    size_t m_memory_usage = 0;
};

// ========================================================================
// Resource handle (smart pointer with manager integration)
// ========================================================================
template <class T>
class resource_handle {
public:
    using element_type = T;
    static_assert(std::is_base_of_v<resource_base, T>, "T must derive from resource_base");

    // Constructors
    resource_handle() = default;
    resource_handle(std::nullptr_t) {}
    explicit resource_handle(std::shared_ptr<T> ptr) : m_ptr(std::move(ptr)) {}

    // Access
    T* operator->() const { return m_ptr.get(); }
    T& operator*() const { return *m_ptr; }
    explicit operator bool() const noexcept { return m_ptr != nullptr; }

    std::shared_ptr<T> ptr() const { return m_ptr; }

    // Resource‑specific queries
    resource_id id() const { return m_ptr ? m_ptr->id() : ""; }
    resource_state state() const { return m_ptr ? m_ptr->state() : resource_state::unloaded; }
    bool is_loaded() const { return m_ptr && m_ptr->state() == resource_state::loaded; }

    // Cast to derived handle
    template <class U>
    resource_handle<U> cast() const {
        return resource_handle<U>(std::dynamic_pointer_cast<U>(m_ptr));
    }

    bool operator==(const resource_handle& other) const { return m_ptr == other.m_ptr; }
    bool operator!=(const resource_handle& other) const { return m_ptr != other.m_ptr; }

private:
    std::shared_ptr<T> m_ptr;
};

// ========================================================================
// Resource loader (pluggable I/O)
// ========================================================================
class resource_loader {
public:
    virtual ~resource_loader() = default;

    // Return list of extensions this loader handles (e.g., ".png", ".jpg")
    virtual std::vector<std::string> extensions() const = 0;

    // Load from file
    virtual std::shared_ptr<resource_base> load(const std::string& path) = 0;

    // Load from memory
    virtual std::shared_ptr<resource_base> load_memory(const uint8_t* data, size_t size,
                                                       const std::string& hint = "") {
        return nullptr;
    }

    // Check if file can be loaded (fast path)
    virtual bool can_load(const std::string& path) const { return true; }

    // Save resource to file (if supported)
    virtual bool save(const std::string& path, std::shared_ptr<resource_base> res) {
        return false;
    }
};

// ========================================================================
// Resource cache (LRU with memory budget)
// ========================================================================
class resource_cache {
public:
    struct config {
        size_t max_memory = 512 * 1024 * 1024;  // 512 MB
        size_t max_resources = 1024;
        float eviction_threshold = 0.9f;        // start evicting at 90%
        bool enable_statistics = true;
    };

    explicit resource_cache(const config& cfg = {});
    ~resource_cache();

    // Insert/retrieve
    void insert(const resource_id& id, std::shared_ptr<resource_base> res);
    std::shared_ptr<resource_base> get(const resource_id& id);
    bool contains(const resource_id& id) const;
    void remove(const resource_id& id);
    void clear();

    // Memory management
    size_t memory_used() const;
    size_t resource_count() const;
    void evict(size_t target_bytes);
    void evict_lru(size_t count);

    // Statistics
    struct stats {
        size_t hits = 0;
        size_t misses = 0;
        size_t evictions = 0;
        size_t peak_memory = 0;
        size_t peak_resources = 0;
    };
    stats get_stats() const;
    void reset_stats();

private:
    struct entry {
        std::shared_ptr<resource_base> resource;
        std::chrono::steady_clock::time_point last_access;
    };

    mutable std::shared_mutex m_mutex;
    std::unordered_map<resource_id, entry> m_cache;
    config m_config;
    stats m_stats;
};

// ========================================================================
// Resource dependency graph
// ========================================================================
class dependency_graph {
public:
    // Add dependency (target depends on source)
    void add_dependency(const resource_id& target, const resource_id& source);

    // Remove dependency
    void remove_dependency(const resource_id& target, const resource_id& source);

    // Remove all dependencies for a resource
    void clear_dependencies(const resource_id& target);

    // Query
    std::vector<resource_id> dependencies(const resource_id& target) const;
    std::vector<resource_id> dependents(const resource_id& source) const;

    // Topological sort for loading order
    std::vector<resource_id> load_order(const std::vector<resource_id>& roots) const;

    // Check for circular dependencies
    bool has_cycle() const;

    // Clear all
    void clear();

private:
    mutable std::shared_mutex m_mutex;
    std::unordered_map<resource_id, std::vector<resource_id>> m_forward;  // target -> sources
    std::unordered_map<resource_id, std::vector<resource_id>> m_reverse;  // source -> targets
};

// ========================================================================
// Resource manager (main entry point)
// ========================================================================
class resource_manager {
public:
    struct config {
        resource_cache::config cache_config;
        size_t max_concurrent_loads = 4;
        bool enable_hot_reload = true;
        double hot_reload_poll_interval = 1.0;  // seconds
        std::string base_path = "";
    };

    static resource_manager& instance();

    explicit resource_manager(const config& cfg = {});
    ~resource_manager();

    // Configuration
    void configure(const config& cfg);
    const config& get_config() const;

    // Loader registration
    void register_loader(std::unique_ptr<resource_loader> loader);
    void register_loader_for_extension(const std::string& ext,
                                       std::unique_ptr<resource_loader> loader);

    // Synchronous load
    template <class T = resource_base>
    resource_handle<T> load(const resource_id& id);

    resource_handle<resource_base> load(const resource_id& id, const std::string& type_hint);

    // Asynchronous load
    template <class T = resource_base>
    std::future<resource_handle<T>> load_async(const resource_id& id);

    // Preload (background, low priority)
    void preload(const resource_id& id);

    // Unload
    void unload(const resource_id& id);
    void unload_unused();  // unload resources with no external handles

    // Query
    resource_state state(const resource_id& id) const;
    bool is_loaded(const resource_id& id) const;
    std::vector<resource_id> loaded_resources() const;

    // Hot‑reload
    void reload(const resource_id& id);
    void reload_all();
    void enable_hot_reload(bool enable);

    // Cache control
    resource_cache& cache() { return m_cache; }
    void flush_cache();

    // Dependency management
    dependency_graph& dependencies() { return m_dep_graph; }

    // Resource creation (for procedural/generated resources)
    template <class T, class... Args>
    resource_handle<T> create(const resource_id& id, Args&&... args);

    void destroy(const resource_id& id);

    // Memory statistics
    struct memory_stats {
        size_t total_loaded;
        size_t cache_memory;
        size_t pending_loads;
        size_t active_handles;
    };
    memory_stats memory_info() const;

    // Wait for all pending loads
    void wait_all();

    // Pipeline processing
    void add_pipeline(std::unique_ptr<resource_pipeline> pipeline);
    void process_pipelines();

private:
    mutable std::shared_mutex m_mutex;
    config m_config;
    resource_cache m_cache;
    dependency_graph m_dep_graph;

    std::unordered_map<std::string, std::unique_ptr<resource_loader>> m_loaders;
    std::unordered_map<resource_id, std::weak_ptr<resource_base>> m_weak_cache;

    std::queue<std::function<void()>> m_pending_loads;
    std::vector<std::future<void>> m_async_tasks;
    std::atomic<bool> m_running{true};
    std::thread m_background_thread;
    std::atomic<bool> m_hot_reload_enabled{false};

    std::vector<std::unique_ptr<resource_pipeline>> m_pipelines;

    void background_worker();
    resource_loader* find_loader(const std::string& path);
    std::string resolve_path(const resource_id& id) const;
    void notify_dependents(const resource_id& id);
};

// ========================================================================
// Resource pipeline (import → process → cook)
// ========================================================================
class resource_pipeline {
public:
    virtual ~resource_pipeline() = default;

    virtual std::string name() const = 0;
    virtual std::vector<std::string> input_extensions() const = 0;
    virtual std::string output_extension() const = 0;

    // Process raw data into cooked resource
    virtual std::shared_ptr<resource_base> process(const std::string& path,
                                                   const std::vector<uint8_t>& raw_data) = 0;

    // Check if output is up‑to‑date (timestamp based)
    virtual bool needs_cooking(const std::string& input_path,
                               const std::string& output_path) const;
};

// ========================================================================
// Built‑in resource types
// ========================================================================
class texture_resource : public resource_base {
public:
    xarray_container<uint8_t> data;
    size_t width = 0, height = 0, channels = 0;

    bool load(const std::string& path) override;
    void unload() override;
    const char* class_name() const override { return "texture_resource"; }
};

class mesh_resource : public resource_base {
public:
    xarray_container<float> vertices;
    xarray_container<uint32_t> indices;
    xarray_container<float> normals;
    xarray_container<float> uvs;

    bool load(const std::string& path) override;
    void unload() override;
    const char* class_name() const override { return "mesh_resource"; }
};

class shader_resource : public resource_base {
public:
    std::string source;
    std::string entry_point;
    std::string language;  // "glsl", "hlsl", "msl"

    bool load(const std::string& path) override;
    void unload() override;
    const char* class_name() const override { return "shader_resource"; }
};

class audio_resource : public resource_base {
public:
    xarray_container<float> samples;
    int sample_rate = 0;
    int channels = 0;

    bool load(const std::string& path) override;
    void unload() override;
    const char* class_name() const override { return "audio_resource"; }
};

class bigarray_resource : public resource_base {
public:
    xarray_container<bignumber::BigNumber> data;

    bool load(const std::string& path) override;
    void unload() override;
    const char* class_name() const override { return "bigarray_resource"; }
};

// ========================================================================
// Convenience functions
// ========================================================================
template <class T>
resource_handle<T> get_resource(const resource_id& id) {
    return resource_manager::instance().load<T>(id);
}

template <class T>
std::future<resource_handle<T>> get_resource_async(const resource_id& id) {
    return resource_manager::instance().load_async<T>(id);
}

void unload_resource(const resource_id& id);
bool resource_exists(const resource_id& id);
void reload_resource(const resource_id& id);

} // namespace resource

using resource::resource_manager;
using resource::resource_handle;
using resource::resource_base;
using resource::resource_id;
using resource::resource_state;
using resource::get_resource;
using resource::get_resource_async;
using resource::texture_resource;
using resource::mesh_resource;
using resource::shader_resource;
using resource::audio_resource;
using resource::bigarray_resource;

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt {
namespace resource {

// resource_cache
inline resource_cache::resource_cache(const config& cfg) : m_config(cfg) {}
inline resource_cache::~resource_cache() = default;
inline void resource_cache::insert(const resource_id& id, std::shared_ptr<resource_base> res)
{ /* TODO: insert with LRU tracking */ }
inline std::shared_ptr<resource_base> resource_cache::get(const resource_id& id)
{ /* TODO: return cached resource and update access time */ return nullptr; }
inline bool resource_cache::contains(const resource_id& id) const
{ /* TODO: check existence */ return false; }
inline void resource_cache::remove(const resource_id& id)
{ /* TODO: remove from cache */ }
inline void resource_cache::clear()
{ /* TODO: clear all entries */ }
inline size_t resource_cache::memory_used() const { return 0; }
inline size_t resource_cache::resource_count() const { return 0; }
inline void resource_cache::evict(size_t target_bytes) { /* TODO: evict until under target */ }
inline void resource_cache::evict_lru(size_t count) { /* TODO: evict oldest entries */ }
inline resource_cache::stats resource_cache::get_stats() const { return m_stats; }
inline void resource_cache::reset_stats() { m_stats = stats{}; }

// dependency_graph
inline void dependency_graph::add_dependency(const resource_id& target, const resource_id& source)
{ /* TODO: add edge */ }
inline void dependency_graph::remove_dependency(const resource_id& target, const resource_id& source)
{ /* TODO: remove edge */ }
inline void dependency_graph::clear_dependencies(const resource_id& target)
{ /* TODO: remove all edges for target */ }
inline std::vector<resource_id> dependency_graph::dependencies(const resource_id& target) const
{ /* TODO: return sources */ return {}; }
inline std::vector<resource_id> dependency_graph::dependents(const resource_id& source) const
{ /* TODO: return targets */ return {}; }
inline std::vector<resource_id> dependency_graph::load_order(const std::vector<resource_id>& roots) const
{ /* TODO: topological sort */ return roots; }
inline bool dependency_graph::has_cycle() const { return false; }
inline void dependency_graph::clear() { /* TODO: clear all edges */ }

// resource_manager
inline resource_manager& resource_manager::instance()
{ static resource_manager mgr; return mgr; }
inline resource_manager::resource_manager(const config& cfg) : m_config(cfg)
{ if (m_config.enable_hot_reload) enable_hot_reload(true); }
inline resource_manager::~resource_manager()
{ m_running = false; if (m_background_thread.joinable()) m_background_thread.join(); }
inline void resource_manager::configure(const config& cfg) { m_config = cfg; }
inline auto resource_manager::get_config() const -> const config& { return m_config; }
inline void resource_manager::register_loader(std::unique_ptr<resource_loader> loader)
{ /* TODO: store loader for its extensions */ }
inline void resource_manager::register_loader_for_extension(const std::string& ext, std::unique_ptr<resource_loader> loader)
{ m_loaders[ext] = std::move(loader); }

template <class T>
resource_handle<T> resource_manager::load(const resource_id& id)
{ /* TODO: sync load */ return {}; }
inline resource_handle<resource_base> resource_manager::load(const resource_id& id, const std::string& type_hint)
{ /* TODO: sync load with type hint */ return {}; }

template <class T>
std::future<resource_handle<T>> resource_manager::load_async(const resource_id& id)
{ /* TODO: async load */ return {}; }
inline void resource_manager::preload(const resource_id& id)
{ /* TODO: background load */ }
inline void resource_manager::unload(const resource_id& id)
{ /* TODO: unload specific resource */ }
inline void resource_manager::unload_unused()
{ /* TODO: evict resources with only weak references */ }
inline resource_state resource_manager::state(const resource_id& id) const
{ return resource_state::unloaded; }
inline bool resource_manager::is_loaded(const resource_id& id) const
{ return state(id) == resource_state::loaded; }
inline std::vector<resource_id> resource_manager::loaded_resources() const
{ return {}; }
inline void resource_manager::reload(const resource_id& id)
{ /* TODO: reload specific resource */ }
inline void resource_manager::reload_all()
{ /* TODO: reload all loaded resources */ }
inline void resource_manager::enable_hot_reload(bool enable)
{ m_hot_reload_enabled = enable; }
inline void resource_manager::flush_cache()
{ m_cache.clear(); }

template <class T, class... Args>
resource_handle<T> resource_manager::create(const resource_id& id, Args&&... args)
{ /* TODO: create procedural resource */ return {}; }
inline void resource_manager::destroy(const resource_id& id)
{ /* TODO: remove from manager */ }
inline resource_manager::memory_stats resource_manager::memory_info() const
{ return {}; }
inline void resource_manager::wait_all()
{ /* TODO: wait for pending async loads */ }
inline void resource_manager::add_pipeline(std::unique_ptr<resource_pipeline> pipeline)
{ m_pipelines.push_back(std::move(pipeline)); }
inline void resource_manager::process_pipelines()
{ /* TODO: run all pipelines on dirty inputs */ }

inline void resource_manager::background_worker()
{ /* TODO: process pending loads and hot‑reload checks */ }
inline resource_loader* resource_manager::find_loader(const std::string& path)
{ /* TODO: match extension */ return nullptr; }
inline std::string resource_manager::resolve_path(const resource_id& id) const
{ return m_config.base_path + "/" + id; }
inline void resource_manager::notify_dependents(const resource_id& id)
{ /* TODO: trigger reload on dependent resources */ }

// resource_pipeline
inline bool resource_pipeline::needs_cooking(const std::string& input_path, const std::string& output_path) const
{ /* TODO: compare timestamps */ return true; }

// Built‑in resources
inline bool texture_resource::load(const std::string& path) { /* TODO: use stb_image */ return false; }
inline void texture_resource::unload() { data = {}; width = height = channels = 0; }
inline bool mesh_resource::load(const std::string& path) { /* TODO: load OBJ/GLTF */ return false; }
inline void mesh_resource::unload() { vertices = indices = normals = uvs = {}; }
inline bool shader_resource::load(const std::string& path) { /* TODO: read file */ return false; }
inline void shader_resource::unload() { source.clear(); }
inline bool audio_resource::load(const std::string& path) { /* TODO: use stb_vorbis */ return false; }
inline void audio_resource::unload() { samples = {}; sample_rate = channels = 0; }
inline bool bigarray_resource::load(const std::string& path) { /* TODO: load NPY/HDF5 */ return false; }
inline void bigarray_resource::unload() { data = {}; }

// Convenience
inline void unload_resource(const resource_id& id)
{ resource_manager::instance().unload(id); }
inline bool resource_exists(const resource_id& id)
{ return resource_manager::instance().is_loaded(id); }
inline void reload_resource(const resource_id& id)
{ resource_manager::instance().reload(id); }

} // namespace resource
} // namespace xt

#endif // XTENSOR_XRESOURCE_HPP