// godot/xparticles2d.hpp

#ifndef XTENSOR_XPARTICLES2D_HPP
#define XTENSOR_XPARTICLES2D_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"
#include "../math/xrandom.hpp"
#include "../math/xinterp.hpp"
#include "../math/xintersection.hpp"
#include "../image/ximage_processing.hpp"
#include "xvariant.hpp"
#include "xclassdb.hpp"
#include "xnode.hpp"
#include "xresource.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <random>
#include <chrono>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/node2d.hpp>
    #include <godot_cpp/classes/particles2d.hpp>
    #include <godot_cpp/classes/particle_process_material.hpp>
    #include <godot_cpp/classes/gpu_particles2d.hpp>
    #include <godot_cpp/classes/atlas_texture.hpp>
    #include <godot_cpp/classes/curve.hpp>
    #include <godot_cpp/classes/curve_texture.hpp>
    #include <godot_cpp/classes/gradient.hpp>
    #include <godot_cpp/classes/gradient_texture1d.hpp>
    #include <godot_cpp/classes/viewport.hpp>
    #include <godot_cpp/classes/world_2d.hpp>
    #include <godot_cpp/classes/physics_direct_space_state2d.hpp>
    #include <godot_cpp/classes/rectangle_shape2d.hpp>
    #include <godot_cpp/classes/circle_shape2d.hpp>
    #include <godot_cpp/classes/collision_shape2d.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/vector2.hpp>
    #include <godot_cpp/variant/color.hpp>
    #include <godot_cpp/variant/rect2.hpp>
    #include <godot_cpp/variant/transform2d.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // Particle State Tensor Representation
            // --------------------------------------------------------------------
            // Batch particle system stored as Structure of Arrays (SoA) tensors
            struct ParticleSystemState2D
            {
                size_t max_particles = 10000;
                size_t active_count = 0;

                // Primary attributes: N x 1 or N x 2
                xarray_container<float> positions;       // N x 2 (x, y)
                xarray_container<float> velocities;      // N x 2 (vx, vy)
                xarray_container<float> accelerations;   // N x 2 (ax, ay)
                xarray_container<float> colors;          // N x 4 (r, g, b, a)
                xarray_container<float> sizes;           // N x 1 (or N x 2 for non-uniform)
                xarray_container<float> rotations;       // N x 1 (radians)
                xarray_container<float> lifetimes;       // N x 1 (current life)
                xarray_container<float> max_lifetimes;   // N x 1 (total life)
                xarray_container<float> custom_data;     // N x 4 (for shaders)

                // Flags
                xarray_container<uint8_t> active;        // N x 1 (1 = alive, 0 = dead)

                // Emission properties (global)
                float emission_rate = 50.0f;             // particles per second
                float emission_accumulator = 0.0f;
                godot::Vector2 emitter_position = godot::Vector2(0, 0);
                godot::Vector2 emitter_extents = godot::Vector2(10, 10);
                bool emitter_shape_box = true;
                float emitter_angle = 0.0f;
                float emitter_spread = 30.0f * M_PI / 180.0f;

                // Dynamics
                godot::Vector2 gravity = godot::Vector2(0, 98.0f);
                float damping = 0.0f;
                float initial_speed_min = 100.0f;
                float initial_speed_max = 200.0f;
                float initial_angle_offset = 0.0f;
                float lifetime_min = 1.0f;
                float lifetime_max = 3.0f;

                // Appearance
                godot::Color color_start = godot::Color(1, 1, 1, 1);
                godot::Color color_end = godot::Color(1, 1, 1, 0);
                float size_start = 16.0f;
                float size_end = 1.0f;
                float rotation_speed_min = -180.0f * M_PI / 180.0f;
                float rotation_speed_max = 180.0f * M_PI / 180.0f;
                xarray_container<float> rotation_speeds; // N x 1

                // Collision
                bool enable_collision = false;
                float bounce_factor = 0.5f;
                float collision_radius = 8.0f;

                // RNG
                std::mt19937 rng;
                std::uniform_real_distribution<float> dist01;

                ParticleSystemState2D(size_t capacity = 10000) : max_particles(capacity)
                {
                    auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
                    rng.seed(static_cast<unsigned int>(seed));
                    dist01 = std::uniform_real_distribution<float>(0.0f, 1.0f);
                    allocate(capacity);
                }

                void allocate(size_t capacity)
                {
                    max_particles = capacity;
                    positions = xarray_container<float>({capacity, 2}, 0.0f);
                    velocities = xarray_container<float>({capacity, 2}, 0.0f);
                    accelerations = xarray_container<float>({capacity, 2}, 0.0f);
                    colors = xarray_container<float>({capacity, 4}, 1.0f);
                    sizes = xarray_container<float>({capacity, 1}, 16.0f);
                    rotations = xarray_container<float>({capacity, 1}, 0.0f);
                    lifetimes = xarray_container<float>({capacity, 1}, 0.0f);
                    max_lifetimes = xarray_container<float>({capacity, 1}, 1.0f);
                    custom_data = xarray_container<float>({capacity, 4}, 0.0f);
                    active = xarray_container<uint8_t>({capacity, 1}, 0);
                    rotation_speeds = xarray_container<float>({capacity, 1}, 0.0f);
                }

                // Reset all particles to inactive
                void reset()
                {
                    active_count = 0;
                    emission_accumulator = 0.0f;
                    for (size_t i = 0; i < max_particles; ++i)
                        active(i, 0) = 0;
                }

                // Emit a single particle
                void emit_particle(float dt)
                {
                    size_t idx = find_free_slot();
                    if (idx >= max_particles) return;

                    // Position
                    float px, py;
                    if (emitter_shape_box)
                    {
                        px = emitter_position.x + (dist01(rng) - 0.5f) * emitter_extents.x;
                        py = emitter_position.y + (dist01(rng) - 0.5f) * emitter_extents.y;
                    }
                    else
                    {
                        // Circle
                        float angle = dist01(rng) * 2.0f * M_PI;
                        float radius = emitter_extents.x * std::sqrt(dist01(rng));
                        px = emitter_position.x + std::cos(angle) * radius;
                        py = emitter_position.y + std::sin(angle) * radius;
                    }
                    positions(idx, 0) = px;
                    positions(idx, 1) = py;

                    // Velocity
                    float speed = initial_speed_min + dist01(rng) * (initial_speed_max - initial_speed_min);
                    float angle = emitter_angle + (dist01(rng) - 0.5f) * emitter_spread;
                    velocities(idx, 0) = std::cos(angle) * speed;
                    velocities(idx, 1) = std::sin(angle) * speed;

                    // Lifetime
                    float life = lifetime_min + dist01(rng) * (lifetime_max - lifetime_min);
                    max_lifetimes(idx, 0) = life;
                    lifetimes(idx, 0) = 0.0f;

                    // Appearance
                    colors(idx, 0) = color_start.r;
                    colors(idx, 1) = color_start.g;
                    colors(idx, 2) = color_start.b;
                    colors(idx, 3) = color_start.a;

                    sizes(idx, 0) = size_start;

                    rotations(idx, 0) = initial_angle_offset + (dist01(rng) - 0.5f) * M_PI;

                    rotation_speeds(idx, 0) = rotation_speed_min + dist01(rng) * (rotation_speed_max - rotation_speed_min);

                    accelerations(idx, 0) = 0.0f;
                    accelerations(idx, 1) = 0.0f;

                    custom_data(idx, 0) = 0.0f;
                    custom_data(idx, 1) = 0.0f;
                    custom_data(idx, 2) = 0.0f;
                    custom_data(idx, 3) = 0.0f;

                    active(idx, 0) = 1;
                    active_count++;
                }

                // Emit particles based on emission rate
                void emit_batch(float delta)
                {
                    emission_accumulator += emission_rate * delta;
                    int to_emit = static_cast<int>(emission_accumulator);
                    emission_accumulator -= static_cast<float>(to_emit);

                    for (int i = 0; i < to_emit; ++i)
                        emit_particle(delta);
                }

                // Update all active particles
                void update(float delta)
                {
                    // Integration
                    for (size_t i = 0; i < max_particles; ++i)
                    {
                        if (!active(i, 0)) continue;

                        // Age
                        lifetimes(i, 0) += delta;
                        if (lifetimes(i, 0) >= max_lifetimes(i, 0))
                        {
                            active(i, 0) = 0;
                            active_count--;
                            continue;
                        }

                        // Physics
                        velocities(i, 0) += (gravity.x + accelerations(i, 0)) * delta;
                        velocities(i, 1) += (gravity.y + accelerations(i, 1)) * delta;
                        velocities(i, 0) *= (1.0f - damping * delta);
                        velocities(i, 1) *= (1.0f - damping * delta);

                        positions(i, 0) += velocities(i, 0) * delta;
                        positions(i, 1) += velocities(i, 1) * delta;

                        // Rotation
                        rotations(i, 0) += rotation_speeds(i, 0) * delta;

                        // Interpolate appearance
                        float t = lifetimes(i, 0) / max_lifetimes(i, 0);
                        colors(i, 0) = color_start.r * (1.0f - t) + color_end.r * t;
                        colors(i, 1) = color_start.g * (1.0f - t) + color_end.g * t;
                        colors(i, 2) = color_start.b * (1.0f - t) + color_end.b * t;
                        colors(i, 3) = color_start.a * (1.0f - t) + color_end.a * t;

                        sizes(i, 0) = size_start * (1.0f - t) + size_end * t;
                    }

                    // Simple collision with boundaries (if enabled)
                    if (enable_collision)
                    {
                        // Assume screen bounds for now
                        float left = 0, right = 1920, top = 0, bottom = 1080;
                        for (size_t i = 0; i < max_particles; ++i)
                        {
                            if (!active(i, 0)) continue;
                            float x = positions(i, 0);
                            float y = positions(i, 1);
                            if (x < left + collision_radius)
                            {
                                positions(i, 0) = left + collision_radius;
                                velocities(i, 0) = -velocities(i, 0) * bounce_factor;
                            }
                            else if (x > right - collision_radius)
                            {
                                positions(i, 0) = right - collision_radius;
                                velocities(i, 0) = -velocities(i, 0) * bounce_factor;
                            }
                            if (y < top + collision_radius)
                            {
                                positions(i, 1) = top + collision_radius;
                                velocities(i, 1) = -velocities(i, 1) * bounce_factor;
                            }
                            else if (y > bottom - collision_radius)
                            {
                                positions(i, 1) = bottom - collision_radius;
                                velocities(i, 1) = -velocities(i, 1) * bounce_factor;
                            }
                        }
                    }
                }

                // Find a free slot (returns max_particles if full)
                size_t find_free_slot()
                {
                    for (size_t i = 0; i < max_particles; ++i)
                        if (!active(i, 0)) return i;
                    return max_particles;
                }

                // Pack active particles into a compact buffer for GPU upload
                void pack_active(xarray_container<float>& packed_positions,
                                 xarray_container<float>& packed_colors,
                                 xarray_container<float>& packed_sizes,
                                 xarray_container<float>& packed_rotations,
                                 xarray_container<float>& packed_custom) const
                {
                    packed_positions = xarray_container<float>({active_count, 2});
                    packed_colors = xarray_container<float>({active_count, 4});
                    packed_sizes = xarray_container<float>({active_count, 1});
                    packed_rotations = xarray_container<float>({active_count, 1});
                    packed_custom = xarray_container<float>({active_count, 4});

                    size_t out_idx = 0;
                    for (size_t i = 0; i < max_particles; ++i)
                    {
                        if (!active(i, 0)) continue;
                        packed_positions(out_idx, 0) = positions(i, 0);
                        packed_positions(out_idx, 1) = positions(i, 1);
                        packed_colors(out_idx, 0) = colors(i, 0);
                        packed_colors(out_idx, 1) = colors(i, 1);
                        packed_colors(out_idx, 2) = colors(i, 2);
                        packed_colors(out_idx, 3) = colors(i, 3);
                        packed_sizes(out_idx, 0) = sizes(i, 0);
                        packed_rotations(out_idx, 0) = rotations(i, 0);
                        packed_custom(out_idx, 0) = custom_data(i, 0);
                        packed_custom(out_idx, 1) = custom_data(i, 1);
                        packed_custom(out_idx, 2) = custom_data(i, 2);
                        packed_custom(out_idx, 3) = custom_data(i, 3);
                        ++out_idx;
                    }
                }
            };

            // --------------------------------------------------------------------
            // XParticles2D - Godot node for tensor-based particles
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XParticles2D : public godot::Node2D
            {
                GDCLASS(XParticles2D, godot::Node2D)

            private:
                ParticleSystemState2D m_system;
                godot::Ref<XTensorNode> m_positions_tensor;
                godot::Ref<XTensorNode> m_velocities_tensor;
                godot::Ref<XTensorNode> m_colors_tensor;
                godot::Ref<XTensorNode> m_sizes_tensor;
                godot::Ref<XTensorNode> m_rotations_tensor;
                godot::Ref<XTensorNode> m_custom_tensor;

                godot::Ref<godot::Texture2D> m_texture;
                godot::Ref<godot::Curve> m_scale_curve;
                godot::Ref<godot::Curve> m_color_curve;
                godot::Ref<godot::Gradient> m_color_gradient;

                bool m_emitting = true;
                bool m_one_shot = false;
                bool m_auto_update = true;
                bool m_local_space = true;
                float m_speed_scale = 1.0f;
                float m_time = 0.0f;
                float m_lifetime = 0.0f;
                bool m_use_gpu = true;
                godot::RID m_multimesh_rid;
                godot::RID m_instance_rid;

                // Internal rendering
                std::vector<godot::Vector2> m_render_positions;
                std::vector<godot::Color> m_render_colors;
                std::vector<float> m_render_sizes;
                std::vector<float> m_render_rotations;

            protected:
                static void _bind_methods()
                {
                    // Tensor access
                    godot::ClassDB::bind_method(godot::D_METHOD("set_positions_tensor", "tensor"), &XParticles2D::set_positions_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_positions_tensor"), &XParticles2D::get_positions_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_velocities_tensor", "tensor"), &XParticles2D::set_velocities_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_velocities_tensor"), &XParticles2D::get_velocities_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_colors_tensor", "tensor"), &XParticles2D::set_colors_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_colors_tensor"), &XParticles2D::get_colors_tensor);

                    // Properties
                    godot::ClassDB::bind_method(godot::D_METHOD("set_emitting", "emitting"), &XParticles2D::set_emitting);
                    godot::ClassDB::bind_method(godot::D_METHOD("is_emitting"), &XParticles2D::is_emitting);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_one_shot", "one_shot"), &XParticles2D::set_one_shot);
                    godot::ClassDB::bind_method(godot::D_METHOD("is_one_shot"), &XParticles2D::is_one_shot);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_amount", "amount"), &XParticles2D::set_amount);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_amount"), &XParticles2D::get_amount);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_lifetime", "lifetime"), &XParticles2D::set_lifetime);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_lifetime"), &XParticles2D::get_lifetime);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_texture", "texture"), &XParticles2D::set_texture);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_texture"), &XParticles2D::get_texture);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_speed_scale", "scale"), &XParticles2D::set_speed_scale);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_speed_scale"), &XParticles2D::get_speed_scale);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_gravity", "gravity"), &XParticles2D::set_gravity);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_gravity"), &XParticles2D::get_gravity);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_initial_velocity", "min", "max"), &XParticles2D::set_initial_velocity);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_spread", "spread"), &XParticles2D::set_spread);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_spread"), &XParticles2D::get_spread);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_color", "start", "end"), &XParticles2D::set_color);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_size", "start", "end"), &XParticles2D::set_size);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_emitter_shape", "shape", "extents"), &XParticles2D::set_emitter_shape);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_collision", "enabled", "radius", "bounce"), &XParticles2D::set_collision);

                    // Control
                    godot::ClassDB::bind_method(godot::D_METHOD("restart"), &XParticles2D::restart);
                    godot::ClassDB::bind_method(godot::D_METHOD("emit_particle", "position", "velocity"), &XParticles2D::emit_particle);
                    godot::ClassDB::bind_method(godot::D_METHOD("emit_batch", "count"), &XParticles2D::emit_batch);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear"), &XParticles2D::clear);
                    godot::ClassDB::bind_method(godot::D_METHOD("capture_state"), &XParticles2D::capture_state);
                    godot::ClassDB::bind_method(godot::D_METHOD("restore_state"), &XParticles2D::restore_state);

                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "emitting"), "set_emitting", "is_emitting");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "one_shot"), "set_one_shot", "is_one_shot");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "amount"), "set_amount", "get_amount");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "lifetime"), "set_lifetime", "get_lifetime");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "texture", godot::PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "speed_scale"), "set_speed_scale", "get_speed_scale");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::VECTOR2, "gravity"), "set_gravity", "get_gravity");

                    ADD_SIGNAL(godot::MethodInfo("finished"));
                    ADD_SIGNAL(godot::MethodInfo("particle_emitted", godot::PropertyInfo(godot::Variant::INT, "count")));
                }

            public:
                XParticles2D() : m_system(10000)
                {
                    m_system.allocate(10000);
                }

                void _ready() override
                {
                    set_process(true);
                    if (m_use_gpu)
                    {
                        _create_multimesh();
                    }
                }

                void _process(double delta) override
                {
                    float dt = static_cast<float>(delta) * m_speed_scale;
                    if (m_emitting)
                    {
                        if (m_one_shot)
                        {
                            emit_batch(static_cast<int>(m_system.emission_rate));
                            m_emitting = false;
                        }
                        else
                        {
                            m_system.emit_batch(dt);
                        }
                    }

                    m_system.update(dt);

                    m_time += dt;
                    if (m_lifetime > 0 && m_time >= m_lifetime)
                    {
                        m_emitting = false;
                    }

                    if (m_system.active_count == 0 && m_time >= m_lifetime && m_lifetime > 0)
                    {
                        emit_signal("finished");
                    }

                    if (m_auto_update)
                    {
                        update_tensors();
                        queue_redraw();
                    }

                    if (m_use_gpu)
                    {
                        _update_multimesh();
                    }
                }

                void _draw() override
                {
                    if (m_use_gpu || !m_texture.is_valid()) return;

                    m_render_positions.clear();
                    m_render_colors.clear();
                    m_render_sizes.clear();
                    m_render_rotations.clear();

                    m_system.pack_active(m_positions_tensor->get_tensor_resource()->m_data.to_float_array(),
                                        m_colors_tensor->get_tensor_resource()->m_data.to_float_array(),
                                        m_sizes_tensor->get_tensor_resource()->m_data.to_float_array(),
                                        m_rotations_tensor->get_tensor_resource()->m_data.to_float_array(),
                                        m_custom_tensor->get_tensor_resource()->m_data.to_float_array());

                    auto pos = m_positions_tensor->get_tensor_resource()->m_data.to_float_array();
                    auto col = m_colors_tensor->get_tensor_resource()->m_data.to_float_array();
                    auto sz = m_sizes_tensor->get_tensor_resource()->m_data.to_float_array();
                    auto rot = m_rotations_tensor->get_tensor_resource()->m_data.to_float_array();

                    godot::Transform2D transform = m_local_space ? godot::Transform2D() : get_global_transform();
                    godot::Vector2 texture_size = m_texture->get_size();
                    godot::Rect2 src_rect(godot::Vector2(), texture_size);

                    for (size_t i = 0; i < m_system.active_count; ++i)
                    {
                        godot::Vector2 p(pos(i, 0), pos(i, 1));
                        godot::Color c(col(i, 0), col(i, 1), col(i, 2), col(i, 3));
                        float s = sz(i, 0);
                        float r = rot(i, 0);

                        p = transform.xform(p);
                        godot::Rect2 dst_rect(p - godot::Vector2(s * 0.5f, s * 0.5f), godot::Vector2(s, s));
                        draw_texture_rect_region(m_texture, dst_rect, src_rect, c, false, r);
                    }
                }

                // Tensor access
                void set_positions_tensor(const godot::Ref<XTensorNode>& tensor) { m_positions_tensor = tensor; }
                godot::Ref<XTensorNode> get_positions_tensor() const { return m_positions_tensor; }
                void set_velocities_tensor(const godot::Ref<XTensorNode>& tensor) { m_velocities_tensor = tensor; }
                godot::Ref<XTensorNode> get_velocities_tensor() const { return m_velocities_tensor; }
                void set_colors_tensor(const godot::Ref<XTensorNode>& tensor) { m_colors_tensor = tensor; }
                godot::Ref<XTensorNode> get_colors_tensor() const { return m_colors_tensor; }

                // Properties
                void set_emitting(bool emitting) { m_emitting = emitting; }
                bool is_emitting() const { return m_emitting; }
                void set_one_shot(bool one_shot) { m_one_shot = one_shot; }
                bool is_one_shot() const { return m_one_shot; }
                void set_amount(int amount) { m_system.max_particles = static_cast<size_t>(amount); m_system.allocate(m_system.max_particles); }
                int get_amount() const { return static_cast<int>(m_system.max_particles); }
                void set_lifetime(float lifetime) { m_lifetime = lifetime; }
                float get_lifetime() const { return m_lifetime; }
                void set_texture(const godot::Ref<godot::Texture2D>& texture) { m_texture = texture; }
                godot::Ref<godot::Texture2D> get_texture() const { return m_texture; }
                void set_speed_scale(float scale) { m_speed_scale = scale; }
                float get_speed_scale() const { return m_speed_scale; }
                void set_gravity(const godot::Vector2& gravity) { m_system.gravity = gravity; }
                godot::Vector2 get_gravity() const { return m_system.gravity; }
                void set_initial_velocity(float min, float max) { m_system.initial_speed_min = min; m_system.initial_speed_max = max; }
                void set_spread(float spread) { m_system.emitter_spread = spread * M_PI / 180.0f; }
                float get_spread() const { return m_system.emitter_spread * 180.0f / M_PI; }
                void set_color(const godot::Color& start, const godot::Color& end) { m_system.color_start = start; m_system.color_end = end; }
                void set_size(float start, float end) { m_system.size_start = start; m_system.size_end = end; }
                void set_emitter_shape(const godot::String& shape, const godot::Vector2& extents)
                {
                    m_system.emitter_shape_box = (shape == "box");
                    m_system.emitter_extents = extents;
                }
                void set_collision(bool enabled, float radius, float bounce)
                {
                    m_system.enable_collision = enabled;
                    m_system.collision_radius = radius;
                    m_system.bounce_factor = bounce;
                }

                // Control
                void restart()
                {
                    m_system.reset();
                    m_time = 0.0f;
                    m_emitting = true;
                }

                void emit_particle(const godot::Vector2& position, const godot::Vector2& velocity)
                {
                    size_t idx = m_system.find_free_slot();
                    if (idx >= m_system.max_particles) return;
                    m_system.positions(idx, 0) = position.x;
                    m_system.positions(idx, 1) = position.y;
                    m_system.velocities(idx, 0) = velocity.x;
                    m_system.velocities(idx, 1) = velocity.y;
                    m_system.active(idx, 0) = 1;
                    m_system.active_count++;
                    emit_signal("particle_emitted", 1);
                }

                void emit_batch(int count)
                {
                    for (int i = 0; i < count; ++i)
                        m_system.emit_particle(0.0f);
                    emit_signal("particle_emitted", count);
                }

                void clear()
                {
                    m_system.reset();
                }

                void capture_state()
                {
                    if (!m_positions_tensor.is_valid()) m_positions_tensor.instantiate();
                    if (!m_velocities_tensor.is_valid()) m_velocities_tensor.instantiate();
                    if (!m_colors_tensor.is_valid()) m_colors_tensor.instantiate();
                    if (!m_sizes_tensor.is_valid()) m_sizes_tensor.instantiate();
                    if (!m_rotations_tensor.is_valid()) m_rotations_tensor.instantiate();
                    if (!m_custom_tensor.is_valid()) m_custom_tensor.instantiate();

                    update_tensors();
                }

                void restore_state()
                {
                    if (m_positions_tensor.is_valid())
                    {
                        auto pos = m_positions_tensor->get_tensor_resource()->m_data.to_float_array();
                        for (size_t i = 0; i < std::min(pos.shape()[0], m_system.max_particles); ++i)
                        {
                            m_system.positions(i, 0) = pos(i, 0);
                            m_system.positions(i, 1) = pos(i, 1);
                            m_system.active(i, 0) = 1;
                        }
                        m_system.active_count = pos.shape()[0];
                    }
                    // Similarly for velocities, colors, etc.
                }

            private:
                void update_tensors()
                {
                    xarray_container<float> pos, col, sz, rot, cust;
                    m_system.pack_active(pos, col, sz, rot, cust);

                    if (m_positions_tensor.is_valid())
                        m_positions_tensor->set_data(XVariant::from_xarray(pos.cast<double>()).variant());
                    if (m_colors_tensor.is_valid())
                        m_colors_tensor->set_data(XVariant::from_xarray(col.cast<double>()).variant());
                    if (m_sizes_tensor.is_valid())
                        m_sizes_tensor->set_data(XVariant::from_xarray(sz.cast<double>()).variant());
                    if (m_rotations_tensor.is_valid())
                        m_rotations_tensor->set_data(XVariant::from_xarray(rot.cast<double>()).variant());
                    if (m_custom_tensor.is_valid())
                        m_custom_tensor->set_data(XVariant::from_xarray(cust.cast<double>()).variant());
                }

                void _create_multimesh()
                {
                    godot::RenderingServer* rs = godot::RenderingServer::get_singleton();
                    if (!rs) return;

                    godot::RID mesh_rid = _create_quad_mesh();
                    m_multimesh_rid = rs->multimesh_create();
                    rs->multimesh_allocate_data(m_multimesh_rid, static_cast<int>(m_system.max_particles),
                                                godot::RenderingServer::MULTIMESH_TRANSFORM_2D, true);
                    rs->multimesh_set_mesh(m_multimesh_rid, mesh_rid);

                    m_instance_rid = rs->instance_create2(m_multimesh_rid, get_world_2d()->get_canvas());
                    rs->instance_set_visible(m_instance_rid, true);
                }

                godot::RID _create_quad_mesh()
                {
                    godot::RenderingServer* rs = godot::RenderingServer::get_singleton();
                    godot::RID mesh_rid = rs->mesh_create();

                    godot::PackedVector2Array vertices;
                    vertices.push_back(godot::Vector2(-0.5, -0.5));
                    vertices.push_back(godot::Vector2( 0.5, -0.5));
                    vertices.push_back(godot::Vector2( 0.5,  0.5));
                    vertices.push_back(godot::Vector2(-0.5,  0.5));

                    godot::PackedVector2Array uvs;
                    uvs.push_back(godot::Vector2(0, 0));
                    uvs.push_back(godot::Vector2(1, 0));
                    uvs.push_back(godot::Vector2(1, 1));
                    uvs.push_back(godot::Vector2(0, 1));

                    godot::PackedInt32Array indices = {0, 1, 2, 0, 2, 3};

                    godot::Array arrays;
                    arrays.resize(godot::Mesh::ARRAY_MAX);
                    arrays[godot::Mesh::ARRAY_VERTEX] = vertices;
                    arrays[godot::Mesh::ARRAY_TEX_UV] = uvs;
                    arrays[godot::Mesh::ARRAY_INDEX] = indices;

                    rs->mesh_add_surface_from_arrays(mesh_rid, godot::Mesh::PRIMITIVE_TRIANGLES, arrays);
                    return mesh_rid;
                }

                void _update_multimesh()
                {
                    if (!m_multimesh_rid.is_valid()) return;
                    godot::RenderingServer* rs = godot::RenderingServer::get_singleton();

                    auto pos = m_positions_tensor.is_valid() ? m_positions_tensor->get_tensor_resource()->m_data.to_float_array() : xarray_container<float>();
                    auto col = m_colors_tensor.is_valid() ? m_colors_tensor->get_tensor_resource()->m_data.to_float_array() : xarray_container<float>();
                    auto sz = m_sizes_tensor.is_valid() ? m_sizes_tensor->get_tensor_resource()->m_data.to_float_array() : xarray_container<float>();
                    auto rot = m_rotations_tensor.is_valid() ? m_rotations_tensor->get_tensor_resource()->m_data.to_float_array() : xarray_container<float>();

                    size_t count = std::min(pos.shape()[0], m_system.max_particles);
                    rs->multimesh_set_visible_instances(m_multimesh_rid, static_cast<int>(count));

                    for (size_t i = 0; i < count; ++i)
                    {
                        godot::Transform2D t;
                        t.set_origin(godot::Vector2(pos(i, 0), pos(i, 1)));
                        t.set_rotation(rot.size() > 0 ? rot(i, 0) : 0.0f);
                        float scale = sz.size() > 0 ? sz(i, 0) : 16.0f;
                        t.set_scale(godot::Vector2(scale, scale));
                        rs->multimesh_instance_set_transform_2d(m_multimesh_rid, static_cast<int>(i), t);

                        if (col.size() > 0)
                        {
                            godot::Color c(col(i, 0), col(i, 1), col(i, 2), col(i, 3));
                            rs->multimesh_instance_set_color(m_multimesh_rid, static_cast<int>(i), c);
                        }
                    }
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XParticles2DRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XParticles2D>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::ParticleSystemState2D;
        using godot_bridge::XParticles2D;
        using godot_bridge::XParticles2DRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XPARTICLES2D_HPP

// godot/xparticles2d.hpp