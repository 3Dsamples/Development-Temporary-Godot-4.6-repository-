// include/xtu/godot/xdecals.hpp
// xtensor-unified - Decal projection system for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XDECALS_HPP
#define XTU_GODOT_XDECALS_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
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
#include "xtu/godot/xnode.hpp"
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xrenderingserver.hpp"
#include "xtu/graphics/xgraphics.hpp"
#include "xtu/graphics/xtransform.hpp"
#include "xtu/graphics/xintersection.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// Forward declarations
// #############################################################################
class Decal;
class DecalTexture;
class DecalAtlas;

// #############################################################################
// Decal texture channels
// #############################################################################
enum class DecalTextureChannel : uint8_t {
    TEXTURE_CHANNEL_ALBEDO = 0,
    TEXTURE_CHANNEL_NORMAL = 1,
    TEXTURE_CHANNEL_ORM = 2,
    TEXTURE_CHANNEL_EMISSION = 3,
    TEXTURE_CHANNEL_MAX = 4
};

// #############################################################################
// Decal projection mode
// #############################################################################
enum class DecalMode : uint8_t {
    DECAL_MODE_PROJECT = 0,
    DECAL_MODE_MESH = 1
};

// #############################################################################
// Decal cull mask flags
// #############################################################################
enum class DecalCullMask : uint32_t {
    CULL_MASK_DEFAULT = 1 << 0,
    CULL_MASK_STATIC = 1 << 1,
    CULL_MASK_DYNAMIC = 1 << 2,
    CULL_MASK_CHARACTER = 1 << 3,
    CULL_MASK_TERRAIN = 1 << 4,
    CULL_MASK_ALL = 0xFFFFFFFF
};

// #############################################################################
// Decal size flags
// #############################################################################
enum class DecalSizeFlags : uint32_t {
    SIZE_FLAG_NONE = 0,
    SIZE_FLAG_KEEP_ASPECT = 1 << 0,
    SIZE_FLAG_USE_ALBEDO_TEXTURE_SIZE = 1 << 1,
    SIZE_FLAG_USE_NORMAL_TEXTURE_SIZE = 1 << 2,
    SIZE_FLAG_USE_ORM_TEXTURE_SIZE = 1 << 3
};

// #############################################################################
// DecalTexture - Texture resource for decals
// #############################################################################
class DecalTexture : public Resource {
    XTU_GODOT_REGISTER_CLASS(DecalTexture, Resource)

public:
    enum TextureChannel : uint8_t {
        CHANNEL_ALBEDO = 0,
        CHANNEL_NORMAL = 1,
        CHANNEL_ORM = 2,
        CHANNEL_EMISSION = 3
    };

private:
    Ref<Texture2D> m_albedo;
    Ref<Texture2D> m_normal;
    Ref<Texture2D> m_orm;
    Ref<Texture2D> m_emission;
    Color m_modulate = {1, 1, 1, 1};
    float m_normal_strength = 1.0f;
    float m_emission_energy = 1.0f;
    float m_albedo_mix = 1.0f;

public:
    static StringName get_class_static() { return StringName("DecalTexture"); }

    void set_albedo_texture(const Ref<Texture2D>& tex) { m_albedo = tex; }
    Ref<Texture2D> get_albedo_texture() const { return m_albedo; }

    void set_normal_texture(const Ref<Texture2D>& tex) { m_normal = tex; }
    Ref<Texture2D> get_normal_texture() const { return m_normal; }

    void set_orm_texture(const Ref<Texture2D>& tex) { m_orm = tex; }
    Ref<Texture2D> get_orm_texture() const { return m_orm; }

    void set_emission_texture(const Ref<Texture2D>& tex) { m_emission = tex; }
    Ref<Texture2D> get_emission_texture() const { return m_emission; }

    void set_modulate(const Color& color) { m_modulate = color; }
    Color get_modulate() const { return m_modulate; }

    void set_normal_strength(float strength) { m_normal_strength = strength; }
    float get_normal_strength() const { return m_normal_strength; }

    void set_emission_energy(float energy) { m_emission_energy = energy; }
    float get_emission_energy() const { return m_emission_energy; }

    void set_albedo_mix(float mix) { m_albedo_mix = mix; }
    float get_albedo_mix() const { return m_albedo_mix; }

    bool has_texture(TextureChannel channel) const {
        switch (channel) {
            case CHANNEL_ALBEDO: return m_albedo.is_valid();
            case CHANNEL_NORMAL: return m_normal.is_valid();
            case CHANNEL_ORM: return m_orm.is_valid();
            case CHANNEL_EMISSION: return m_emission.is_valid();
            default: return false;
        }
    }
};

// #############################################################################
// Decal - Projected texture decal
// #############################################################################
class Decal : public Node3D {
    XTU_GODOT_REGISTER_CLASS(Decal, Node3D)

public:
    enum DecalMode : uint8_t {
        MODE_PROJECT = 0,
        MODE_MESH = 1
    };

    enum CullMask : uint32_t {
        CULL_MASK_DEFAULT = 1 << 0,
        CULL_MASK_STATIC = 1 << 1,
        CULL_MASK_DYNAMIC = 1 << 2,
        CULL_MASK_CHARACTER = 1 << 3,
        CULL_MASK_TERRAIN = 1 << 4,
        CULL_MASK_ALL = 0xFFFFFFFF
    };

    enum SizeFlags : uint32_t {
        SIZE_FLAG_NONE = 0,
        SIZE_FLAG_KEEP_ASPECT = 1 << 0,
        SIZE_FLAG_USE_ALBEDO_TEXTURE_SIZE = 1 << 1,
        SIZE_FLAG_USE_NORMAL_TEXTURE_SIZE = 1 << 2,
        SIZE_FLAG_USE_ORM_TEXTURE_SIZE = 1 << 3
    };

private:
    Ref<DecalTexture> m_texture;
    vec3f m_size = {2, 2, 2};
    DecalMode m_mode = DecalMode::MODE_PROJECT;
    uint32_t m_cull_mask = static_cast<uint32_t>(CullMask::CULL_MASK_ALL);
    uint32_t m_size_flags = static_cast<uint32_t>(SizeFlags::SIZE_FLAG_NONE);
    float m_upper_fade = 0.3f;
    float m_lower_fade = 0.3f;
    float m_normal_fade = 0.0f;
    float m_emission_energy = 1.0f;
    Color m_modulate = {1, 1, 1, 1};
    float m_albedo_mix = 1.0f;
    bool m_enabled = true;
    mat4f m_inv_transform;
    aabb m_local_aabb;
    bool m_transform_dirty = true;

public:
    static StringName get_class_static() { return StringName("Decal"); }

    Decal() { update_transform(); }

    void set_texture(const Ref<DecalTexture>& texture) {
        m_texture = texture;
        update_size_from_texture();
    }
    Ref<DecalTexture> get_texture() const { return m_texture; }

    void set_size(const vec3f& size) { m_size = size; m_transform_dirty = true; }
    vec3f get_size() const { return m_size; }

    void set_mode(DecalMode mode) { m_mode = mode; }
    DecalMode get_mode() const { return m_mode; }

    void set_cull_mask(uint32_t mask) { m_cull_mask = mask; }
    uint32_t get_cull_mask() const { return m_cull_mask; }

    void set_upper_fade(float fade) { m_upper_fade = fade; }
    float get_upper_fade() const { return m_upper_fade; }

    void set_lower_fade(float fade) { m_lower_fade = fade; }
    float get_lower_fade() const { return m_lower_fade; }

    void set_normal_fade(float fade) { m_normal_fade = fade; }
    float get_normal_fade() const { return m_normal_fade; }

    void set_emission_energy(float energy) { m_emission_energy = energy; }
    float get_emission_energy() const { return m_emission_energy; }

    void set_modulate(const Color& color) { m_modulate = color; }
    Color get_modulate() const { return m_modulate; }

    void set_albedo_mix(float mix) { m_albedo_mix = mix; }
    float get_albedo_mix() const { return m_albedo_mix; }

    void set_enabled(bool enabled) { m_enabled = enabled; }
    bool is_enabled() const { return m_enabled; }

    const mat4f& get_inv_transform() const { return m_inv_transform; }
    aabb get_local_aabb() const { return m_local_aabb; }

    void update_transform() {
        mat4f transform = get_global_transform();
        m_inv_transform = inverse(transform);
        vec3f half = m_size * 0.5f;
        m_local_aabb = aabb(-half, half);
        m_transform_dirty = false;
    }

    bool is_affecting_bounds(const aabb& world_bounds) const {
        if (!m_enabled) return false;
        aabb local_bounds = world_bounds.transform(m_inv_transform);
        return m_local_aabb.intersects(local_bounds);
    }

    bool project_point(const vec3f& world_pos, vec3f& out_local_pos, vec2f& out_uv) const {
        vec4f local = m_inv_transform * vec4f(world_pos.x(), world_pos.y(), world_pos.z(), 1.0f);
        out_local_pos = vec3f(local.x(), local.y(), local.z());
        vec3f half = m_size * 0.5f;
        if (std::abs(out_local_pos.x()) > half.x() ||
            std::abs(out_local_pos.y()) > half.y() ||
            std::abs(out_local_pos.z()) > half.z()) {
            return false;
        }
        out_uv.x() = (out_local_pos.x() / m_size.x()) + 0.5f;
        out_uv.y() = (out_local_pos.y() / m_size.y()) + 0.5f;
        return true;
    }

    float get_alpha_fade(const vec3f& local_pos) const {
        float half_z = m_size.z() * 0.5f;
        float z = local_pos.z();
        if (z < -half_z + m_lower_fade) {
            return (z + half_z) / m_lower_fade;
        } else if (z > half_z - m_upper_fade) {
            return (half_z - z) / m_upper_fade;
        }
        return 1.0f;
    }

    void _notification(int p_what) override {
        if (p_what == NOTIFICATION_TRANSFORM_CHANGED) {
            m_transform_dirty = true;
        }
    }

    void _process(double delta) override {
        if (m_transform_dirty) update_transform();
    }

private:
    void update_size_from_texture() {
        if (!m_texture.is_valid()) return;
        if ((m_size_flags & static_cast<uint32_t>(SizeFlags::SIZE_FLAG_USE_ALBEDO_TEXTURE_SIZE)) &&
            m_texture->get_albedo_texture().is_valid()) {
            auto tex = m_texture->get_albedo_texture();
            vec2 tex_size = tex->get_size();
            float aspect = tex_size.x() / tex_size.y();
            m_size.x() = m_size.y() * aspect;
        }
    }
};

// #############################################################################
// DecalAtlas - Multiple decals packed into one atlas
// #############################################################################
class DecalAtlas : public Resource {
    XTU_GODOT_REGISTER_CLASS(DecalAtlas, Resource)

private:
    std::vector<Ref<DecalTexture>> m_decals;
    vec2i m_atlas_size = {2048, 2048};
    int m_padding = 2;

public:
    static StringName get_class_static() { return StringName("DecalAtlas"); }

    void add_decal(const Ref<DecalTexture>& decal) {
        if (std::find(m_decals.begin(), m_decals.end(), decal) == m_decals.end()) {
            m_decals.push_back(decal);
        }
    }

    void remove_decal(int idx) {
        if (idx >= 0 && idx < static_cast<int>(m_decals.size())) {
            m_decals.erase(m_decals.begin() + idx);
        }
    }

    int get_decal_count() const { return static_cast<int>(m_decals.size()); }
    Ref<DecalTexture> get_decal(int idx) const {
        return idx >= 0 && idx < static_cast<int>(m_decals.size()) ? m_decals[idx] : Ref<DecalTexture>();
    }

    void set_atlas_size(const vec2i& size) { m_atlas_size = size; }
    vec2i get_atlas_size() const { return m_atlas_size; }

    void bake() {
        // Pack textures into atlas
    }
};

// #############################################################################
// DecalManager - Global decal manager
// #############################################################################
class DecalManager : public Object {
    XTU_GODOT_REGISTER_CLASS(DecalManager, Object)

private:
    static DecalManager* s_singleton;
    std::vector<Decal*> m_active_decals;
    std::mutex m_mutex;

public:
    static DecalManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("DecalManager"); }

    DecalManager() { s_singleton = this; }
    ~DecalManager() { s_singleton = nullptr; }

    void register_decal(Decal* decal) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (std::find(m_active_decals.begin(), m_active_decals.end(), decal) == m_active_decals.end()) {
            m_active_decals.push_back(decal);
        }
    }

    void unregister_decal(Decal* decal) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = std::find(m_active_decals.begin(), m_active_decals.end(), decal);
        if (it != m_active_decals.end()) m_active_decals.erase(it);
    }

    std::vector<Decal*> get_decals_affecting(const aabb& world_bounds, uint32_t cull_mask) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<Decal*> result;
        for (Decal* decal : m_active_decals) {
            if (!decal->is_enabled()) continue;
            if ((decal->get_cull_mask() & cull_mask) == 0) continue;
            if (decal->is_affecting_bounds(world_bounds)) {
                result.push_back(decal);
            }
        }
        return result;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_active_decals.clear();
    }
};

} // namespace godot

// Bring into main namespace
using godot::Decal;
using godot::DecalTexture;
using godot::DecalAtlas;
using godot::DecalManager;
using godot::DecalTextureChannel;
using godot::DecalMode;
using godot::DecalCullMask;
using godot::DecalSizeFlags;

XTU_NAMESPACE_END

#endif // XTU_GODOT_XDECALS_HPP