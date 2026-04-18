// include/xtu/godot/xrenderer_rd.hpp
// xtensor-unified - RenderingDevice Backend for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XRENDERER_RD_HPP
#define XTU_GODOT_XRENDERER_RD_HPP

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace rendering {

// #############################################################################
// Forward declarations
// #############################################################################
class RendererStorageRD;
class UniformSetCacheRD;
class PipelineCacheRD;
class RenderingDeviceRD;

// #############################################################################
// Shader stage flags
// #############################################################################
enum class ShaderStageFlags : uint32_t {
    STAGE_NONE = 0,
    STAGE_VERTEX = 1 << 0,
    STAGE_FRAGMENT = 1 << 1,
    STAGE_COMPUTE = 1 << 2,
    STAGE_TESSELLATION_CONTROL = 1 << 3,
    STAGE_TESSELLATION_EVALUATION = 1 << 4,
    STAGE_ALL_GRAPHICS = STAGE_VERTEX | STAGE_FRAGMENT
};

inline ShaderStageFlags operator|(ShaderStageFlags a, ShaderStageFlags b) {
    return static_cast<ShaderStageFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

// #############################################################################
// Uniform set layout descriptor
// #############################################################################
struct UniformSetLayout {
    struct Binding {
        uint32_t binding = 0;
        RenderingDevice::UniformType type = RenderingDevice::UNIFORM_TYPE_BUFFER;
        ShaderStageFlags stages = ShaderStageFlags::STAGE_ALL_GRAPHICS;
        uint32_t count = 1;
    };
    std::vector<Binding> bindings;
    uint64_t hash = 0;
};

// #############################################################################
// Pipeline layout descriptor
// #############################################################################
struct PipelineLayout {
    std::vector<UniformSetLayout> set_layouts;
    std::vector<RenderingDevice::PushConstant> push_constants;
    uint64_t hash = 0;
};

// #############################################################################
// Graphics pipeline descriptor
// #############################################################################
struct GraphicsPipelineDescriptor {
    PipelineLayout layout;
    RID vertex_shader;
    RID fragment_shader;
    RID tess_control_shader;
    RID tess_evaluation_shader;
    RenderingDevice::PrimitiveTopology topology = RenderingDevice::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    RenderingDevice::PolygonCullMode cull_mode = RenderingDevice::POLYGON_CULL_BACK;
    RenderingDevice::PolygonFrontFace front_face = RenderingDevice::POLYGON_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depth_test = true;
    bool depth_write = true;
    RenderingDevice::CompareOp depth_compare = RenderingDevice::COMPARE_OP_LESS;
    bool stencil_test = false;
    uint32_t stencil_read_mask = 0xFF;
    uint32_t stencil_write_mask = 0xFF;
    uint32_t sample_count = 1;
    std::vector<RenderingDevice::BlendState> blend_states;
    std::vector<RID> color_formats;
    RID depth_format;
    uint64_t hash = 0;
};

// #############################################################################
// Compute pipeline descriptor
// #############################################################################
struct ComputePipelineDescriptor {
    PipelineLayout layout;
    RID compute_shader;
    uint64_t hash = 0;
};

// #############################################################################
// UniformSetCacheRD - Caches uniform set layouts and instances
// #############################################################################
class UniformSetCacheRD : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(UniformSetCacheRD, RefCounted)

private:
    struct CachedSetLayout {
        UniformSetLayout layout;
        RID rd_layout;
        uint32_t ref_count = 0;
    };

    struct CachedSet {
        RID layout;
        std::vector<RID> bindings; // Buffers/textures bound
        RID rd_set;
        uint64_t last_used_frame = 0;
    };

    RenderingDevice* m_device = nullptr;
    std::unordered_map<uint64_t, CachedSetLayout> m_layout_cache;
    std::unordered_map<uint64_t, CachedSet> m_set_cache;
    uint64_t m_current_frame = 0;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("UniformSetCacheRD"); }

    void set_device(RenderingDevice* device) { m_device = device; }

    RID get_or_create_layout(const UniformSetLayout& layout) {
        uint64_t hash = compute_layout_hash(layout);
        
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_layout_cache.find(hash);
        if (it != m_layout_cache.end()) {
            ++it->second.ref_count;
            return it->second.rd_layout;
        }

        // Create new layout via RenderingDevice
        CachedSetLayout cached;
        cached.layout = layout;
        cached.layout.hash = hash;
        cached.ref_count = 1;
        // cached.rd_layout = m_device->uniform_set_layout_create(...);
        
        m_layout_cache[hash] = cached;
        return cached.rd_layout;
    }

    void release_layout(uint64_t hash) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_layout_cache.find(hash);
        if (it != m_layout_cache.end()) {
            if (--it->second.ref_count == 0) {
                // m_device->free_rid(it->second.rd_layout);
                m_layout_cache.erase(it);
            }
        }
    }

    RID get_or_create_set(uint64_t layout_hash, const std::vector<RID>& bindings) {
        uint64_t set_hash = layout_hash;
        for (RID rid : bindings) {
            set_hash = hash_combine(set_hash, rid);
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_set_cache.find(set_hash);
        if (it != m_set_cache.end()) {
            it->second.last_used_frame = m_current_frame;
            return it->second.rd_set;
        }

        auto layout_it = m_layout_cache.find(layout_hash);
        if (layout_it == m_layout_cache.end()) return RID();

        CachedSet cached;
        cached.layout = layout_it->second.rd_layout;
        cached.bindings = bindings;
        cached.last_used_frame = m_current_frame;
        // cached.rd_set = m_device->uniform_set_create(...);
        
        m_set_cache[set_hash] = cached;
        return cached.rd_set;
    }

    void begin_frame() {
        ++m_current_frame;
    }

    void garbage_collect(uint64_t max_age = 60) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_set_cache.begin();
        while (it != m_set_cache.end()) {
            if (m_current_frame - it->second.last_used_frame > max_age) {
                // m_device->free_rid(it->second.rd_set);
                it = m_set_cache.erase(it);
            } else {
                ++it;
            }
        }
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        // Free all RIDs
        m_layout_cache.clear();
        m_set_cache.clear();
    }

private:
    uint64_t compute_layout_hash(const UniformSetLayout& layout) const {
        uint64_t h = 0;
        for (const auto& b : layout.bindings) {
            h = hash_combine(h, b.binding);
            h = hash_combine(h, static_cast<uint32_t>(b.type));
            h = hash_combine(h, static_cast<uint32_t>(b.stages));
            h = hash_combine(h, b.count);
        }
        return h;
    }

    uint64_t hash_combine(uint64_t seed, uint64_t v) const {
        return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }
};

// #############################################################################
// PipelineCacheRD - Caches graphics and compute pipelines
// #############################################################################
class PipelineCacheRD : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(PipelineCacheRD, RefCounted)

private:
    struct CachedGraphicsPipeline {
        GraphicsPipelineDescriptor desc;
        RID pipeline;
        uint64_t last_used_frame = 0;
    };

    struct CachedComputePipeline {
        ComputePipelineDescriptor desc;
        RID pipeline;
        uint64_t last_used_frame = 0;
    };

    RenderingDevice* m_device = nullptr;
    std::unordered_map<uint64_t, CachedGraphicsPipeline> m_graphics_cache;
    std::unordered_map<uint64_t, CachedComputePipeline> m_compute_cache;
    RID m_pipeline_cache_rid; // For driver-level pipeline cache
    uint64_t m_current_frame = 0;
    mutable std::mutex m_mutex;
    String m_cache_file_path;

public:
    static StringName get_class_static() { return StringName("PipelineCacheRD"); }

    PipelineCacheRD() {
        // m_pipeline_cache_rid = RenderingDevice::get_singleton()->pipeline_cache_create();
    }

    ~PipelineCacheRD() {
        // RenderingDevice::get_singleton()->free_rid(m_pipeline_cache_rid);
    }

    void set_device(RenderingDevice* device) { m_device = device; }
    void set_cache_file_path(const String& path) { m_cache_file_path = path; }

    RID get_or_create_graphics_pipeline(const GraphicsPipelineDescriptor& desc) {
        uint64_t hash = compute_graphics_pipeline_hash(desc);

        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_graphics_cache.find(hash);
        if (it != m_graphics_cache.end()) {
            it->second.last_used_frame = m_current_frame;
            return it->second.pipeline;
        }

        CachedGraphicsPipeline cached;
        cached.desc = desc;
        cached.desc.hash = hash;
        cached.last_used_frame = m_current_frame;
        // cached.pipeline = m_device->graphics_pipeline_create(desc, m_pipeline_cache_rid);
        
        m_graphics_cache[hash] = cached;
        return cached.pipeline;
    }

    RID get_or_create_compute_pipeline(const ComputePipelineDescriptor& desc) {
        uint64_t hash = compute_compute_pipeline_hash(desc);

        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_compute_cache.find(hash);
        if (it != m_compute_cache.end()) {
            it->second.last_used_frame = m_current_frame;
            return it->second.pipeline;
        }

        CachedComputePipeline cached;
        cached.desc = desc;
        cached.desc.hash = hash;
        cached.last_used_frame = m_current_frame;
        // cached.pipeline = m_device->compute_pipeline_create(desc, m_pipeline_cache_rid);
        
        m_compute_cache[hash] = cached;
        return cached.pipeline;
    }

    void begin_frame() {
        ++m_current_frame;
    }

    void garbage_collect(uint64_t max_age = 300) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        auto it = m_graphics_cache.begin();
        while (it != m_graphics_cache.end()) {
            if (m_current_frame - it->second.last_used_frame > max_age) {
                // m_device->free_rid(it->second.pipeline);
                it = m_graphics_cache.erase(it);
            } else {
                ++it;
            }
        }

        auto cit = m_compute_cache.begin();
        while (cit != m_compute_cache.end()) {
            if (m_current_frame - cit->second.last_used_frame > max_age) {
                // m_device->free_rid(cit->second.pipeline);
                cit = m_compute_cache.erase(cit);
            } else {
                ++cit;
            }
        }
    }

    void save_to_file() {
        if (m_cache_file_path.empty()) return;
        // std::vector<uint8_t> data = m_device->pipeline_cache_get_data(m_pipeline_cache_rid);
        // FileAccess::open(m_cache_file_path, FileAccess::WRITE)->store_buffer(data);
    }

    void load_from_file() {
        if (m_cache_file_path.empty() || !FileAccess::file_exists(m_cache_file_path)) return;
        // std::vector<uint8_t> data = FileAccess::open(m_cache_file_path, FileAccess::READ)->get_buffer();
        // m_device->pipeline_cache_set_data(m_pipeline_cache_rid, data);
    }

private:
    uint64_t compute_graphics_pipeline_hash(const GraphicsPipelineDescriptor& desc) const {
        uint64_t h = desc.layout.hash;
        h = hash_combine(h, desc.vertex_shader);
        h = hash_combine(h, desc.fragment_shader);
        h = hash_combine(h, static_cast<uint32_t>(desc.topology));
        h = hash_combine(h, static_cast<uint32_t>(desc.cull_mode));
        h = hash_combine(h, desc.depth_test ? 1 : 0);
        h = hash_combine(h, desc.depth_write ? 1 : 0);
        h = hash_combine(h, static_cast<uint32_t>(desc.depth_compare));
        h = hash_combine(h, desc.sample_count);
        for (RID fmt : desc.color_formats) {
            h = hash_combine(h, fmt);
        }
        h = hash_combine(h, desc.depth_format);
        return h;
    }

    uint64_t compute_compute_pipeline_hash(const ComputePipelineDescriptor& desc) const {
        uint64_t h = desc.layout.hash;
        h = hash_combine(h, desc.compute_shader);
        return h;
    }

    uint64_t hash_combine(uint64_t seed, uint64_t v) const {
        return seed ^ (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }
};

// #############################################################################
// RendererStorageRD - Storage for RD-based renderer
// #############################################################################
class RendererStorageRD : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(RendererStorageRD, RefCounted)

private:
    RenderingDevice* m_device = nullptr;
    Ref<UniformSetCacheRD> m_uniform_set_cache;
    Ref<PipelineCacheRD> m_pipeline_cache;

    // Texture storage
    struct TextureData {
        RID rd_texture;
        RID rd_texture_view;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t depth = 0;
        uint32_t mipmaps = 1;
        uint32_t layers = 1;
        RenderingDevice::TextureFormat format = RenderingDevice::TEXTURE_FORMAT_RGBA8_UNORM;
        RenderingDevice::TextureType type = RenderingDevice::TEXTURE_TYPE_2D;
        uint64_t last_used_frame = 0;
    };
    std::unordered_map<RID, TextureData> m_textures;

    // Buffer storage
    struct BufferData {
        RID rd_buffer;
        size_t size = 0;
        uint32_t usage = 0;
        uint64_t last_used_frame = 0;
    };
    std::unordered_map<RID, BufferData> m_buffers;

    // Sampler storage
    std::unordered_map<RID, RID> m_samplers; // Godot RID -> RD RID

    uint64_t m_current_frame = 0;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("RendererStorageRD"); }

    RendererStorageRD() {
        m_uniform_set_cache.instance();
        m_pipeline_cache.instance();
    }

    void set_device(RenderingDevice* device) {
        m_device = device;
        m_uniform_set_cache->set_device(device);
        m_pipeline_cache->set_device(device);
    }

    void begin_frame() {
        ++m_current_frame;
        m_uniform_set_cache->begin_frame();
        m_pipeline_cache->begin_frame();
    }

    void end_frame() {
        // Submit command buffers
    }

    // #########################################################################
    // Texture management
    // #########################################################################
    RID texture_create() {
        std::lock_guard<std::mutex> lock(m_mutex);
        RID rid = ++m_next_rid;
        m_textures[rid] = TextureData();
        return rid;
    }

    void texture_allocate(RID rid, uint32_t width, uint32_t height, uint32_t depth,
                          uint32_t mipmaps, RenderingDevice::TextureFormat format,
                          RenderingDevice::TextureType type, uint32_t layers = 1) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_textures.find(rid);
        if (it == m_textures.end()) return;

        TextureData& tex = it->second;
        tex.width = width;
        tex.height = height;
        tex.depth = depth;
        tex.mipmaps = mipmaps;
        tex.format = format;
        tex.type = type;
        tex.layers = layers;
        tex.last_used_frame = m_current_frame;

        // tex.rd_texture = m_device->texture_create(...);
        // tex.rd_texture_view = m_device->texture_create_view(tex.rd_texture, ...);
    }

    void texture_set_data(RID rid, const std::vector<uint8_t>& data, int layer = 0, int mipmap = 0) {
        // m_device->texture_update(rd_texture, layer, mipmap, data);
    }

    RID texture_get_rd_texture(RID rid) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_textures.find(rid);
        return it != m_textures.end() ? it->second.rd_texture : RID();
    }

    RID texture_get_rd_view(RID rid) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_textures.find(rid);
        return it != m_textures.end() ? it->second.rd_texture_view : RID();
    }

    // #########################################################################
    // Buffer management
    // #########################################################################
    RID buffer_create(size_t size, uint32_t usage) {
        std::lock_guard<std::mutex> lock(m_mutex);
        RID rid = ++m_next_rid;
        BufferData buf;
        buf.size = size;
        buf.usage = usage;
        buf.last_used_frame = m_current_frame;
        // buf.rd_buffer = m_device->buffer_create(size, usage);
        m_buffers[rid] = buf;
        return rid;
    }

    void buffer_update(RID rid, size_t offset, const std::vector<uint8_t>& data) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_buffers.find(rid);
        if (it != m_buffers.end()) {
            it->second.last_used_frame = m_current_frame;
            // m_device->buffer_update(it->second.rd_buffer, offset, data);
        }
    }

    RID buffer_get_rd_buffer(RID rid) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_buffers.find(rid);
        return it != m_buffers.end() ? it->second.rd_buffer : RID();
    }

    // #########################################################################
    // Sampler management
    // #########################################################################
    RID sampler_create(RenderingDevice::SamplerFilter min_filter,
                       RenderingDevice::SamplerFilter mag_filter,
                       RenderingDevice::SamplerMipmapMode mipmap_mode,
                       RenderingDevice::SamplerAddressMode address_u,
                       RenderingDevice::SamplerAddressMode address_v,
                       RenderingDevice::SamplerAddressMode address_w) {
        std::lock_guard<std::mutex> lock(m_mutex);
        RID rid = ++m_next_rid;
        // RID rd_sampler = m_device->sampler_create(...);
        // m_samplers[rid] = rd_sampler;
        return rid;
    }

    RID sampler_get_rd_sampler(RID rid) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_samplers.find(rid);
        return it != m_samplers.end() ? it->second : RID();
    }

    // #########################################################################
    // Garbage collection
    // #########################################################################
    void garbage_collect(uint64_t max_age = 300) {
        m_uniform_set_cache->garbage_collect(max_age);
        m_pipeline_cache->garbage_collect(max_age);

        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_textures.begin();
        while (it != m_textures.end()) {
            if (m_current_frame - it->second.last_used_frame > max_age) {
                // m_device->free_rid(it->second.rd_texture);
                // m_device->free_rid(it->second.rd_texture_view);
                it = m_textures.erase(it);
            } else {
                ++it;
            }
        }

        auto bit = m_buffers.begin();
        while (bit != m_buffers.end()) {
            if (m_current_frame - bit->second.last_used_frame > max_age) {
                // m_device->free_rid(bit->second.rd_buffer);
                bit = m_buffers.erase(bit);
            } else {
                ++bit;
            }
        }
    }

    void free_rid(RID rid) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_textures.erase(rid)) return;
        if (m_buffers.erase(rid)) return;
        if (m_samplers.erase(rid)) return;
    }

private:
    std::atomic<uint64_t> m_next_rid{1};
};

} // namespace rendering

// Bring into main namespace
using rendering::RendererStorageRD;
using rendering::UniformSetCacheRD;
using rendering::PipelineCacheRD;
using rendering::ShaderStageFlags;
using rendering::UniformSetLayout;
using rendering::PipelineLayout;
using rendering::GraphicsPipelineDescriptor;
using rendering::ComputePipelineDescriptor;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XRENDERER_RD_HPP