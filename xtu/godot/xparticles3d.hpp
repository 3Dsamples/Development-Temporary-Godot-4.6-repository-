// godot/xparticles3d.hpp

#ifndef XTENSOR_XPARTICLES3D_HPP
#define XTENSOR_XPARTICLES3D_HPP

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
#include "../math/xquaternion.hpp"
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
    #include <godot_cpp/classes/node3d.hpp>
    #include <godot_cpp/classes/particles3d.hpp>
    #include <godot_cpp/classes/gpu_particles3d.hpp>
    #include <godot_cpp/classes/particle_process_material.hpp>
    #include <godot_cpp/classes/atlas_texture.hpp>
    #include <godot_cpp/classes/curve.hpp>
    #include <godot_cpp/classes/curve_texture.hpp>
    #include <godot_cpp/classes/gradient.hpp>
    #include <godot_cpp/classes/gradient_texture1d.hpp>
    #include <godot_cpp/classes/viewport.hpp>
    #include <godot_cpp/classes/world_3d.hpp>
    #include <godot_cpp/classes/physics_direct_space_state3d.hpp>
    #include <godot_cpp/classes/box_shape3d.hpp>
    #include <godot_cpp/classes/sphere_shape3d.hpp>
    #include <godot_cpp/classes/collision_shape3d.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/vector3.hpp>
    #include <godot_cpp/variant/color.hpp>
    #include <godot_cpp/variant/aabb.hpp>
    #include <godot_cpp/variant/transform3d.hpp>
    #include <godot_cpp/variant/basis.hpp>
    #include <godot_cpp/variant/quaternion.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // 3D Particle State Tensor Representation
            // --------------------------------------------------------------------
            struct ParticleSystemState3D
            {
                size_t max_particles = 10000;
                size_t active_count = 0;

                // Primary attributes: N x 1, N x 3, or N x 4
                xarray_container<float> positions;       // N x 3 (x, y, z)
                xarray_container<float> velocities;      // N x 3 (vx, vy, vz)
                xarray_container<float> accelerations;   // N x 3 (ax, ay, az)
                xarray_container<float> colors;          // N x 4 (r, g, b, a)
                xarray_container<float> sizes;           // N x 1 (uniform scale)
                xarray_container<float> rotations;       // N x 4 (quaternion w,x,y,z)
                xarray_container<float> lifetimes;       // N x 1 (current life)
                xarray_container<float> max_lifetimes;   // N x 1 (total life)
                xarray_container<float> custom_data;     // N x 4 (for shaders)
                xarray_container<float> angular_velocities; // N x 3 (rotation speed as axis-angle)

                xarray_container<uint8_t> active;        // N x 1 (1 = alive, 0 = dead)

                // Emission properties
                float emission_rate = 50.0f;
                float emission_accumulator = 0.0f;
                godot::Vector3 emitter_position = godot::Vector3(0, 0, 0);
                godot::Vector3 emitter_extents = godot::Vector3(5, 5, 5);
                bool emitter_shape_box = true;
                float emitter_angle_min = 0.0f;
                float emitter_angle_max = 360.0f * M_PI / 180.0f;
                float emitter_spread = 30.0f * M_PI / 180.0f;
                godot::Vector3 emitter_direction = godot::Vector3(0, 1, 0);

                // Dynamics
                godot::Vector3 gravity = godot::Vector3(0, -9.8f, 0);
                float damping = 0.0f;
                float initial_speed_min = 100.0f;
                float initial_speed_max = 200.0f;
                float lifetime_min = 1.0f;
                float lifetime_max = 3.0f;

                // Appearance
                godot::Color color_start = godot::Color(1, 1, 1, 1);
                godot::Color color_end = godot::Color(1, 1, 1, 0);
                float size_start = 1.0f;
                float size_end = 0.1f;
                float rotation_speed_min = -180.0f * M_PI / 180.0f;
                float rotation_speed_max = 180.0f * M_PI / 180.0f;
                xarray_container<float> rotation_axes;   // N x 3 (axis of rotation)

                // Collision
                bool enable_collision = false;
                float bounce_factor = 0.5f;
                float collision_radius = 0.5f;

                // RNG
                std::mt19937 rng;
                std::uniform_real_distribution<float> dist01;

                ParticleSystemState3D(size_t capacity = 10000) : max_particles(capacity)
                {
                    auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
                    rng.seed(static_cast<unsigned int>(seed));
                    dist01 = std::uniform_real_distribution<float>(0.0f, 1.0f);
                    allocate(capacity);
                }

                void allocate(size_t capacity)
                {
                    max_particles = capacity;
                    positions = xarray_container<float>({capacity, 3}, 0.0f);
                    velocities = xarray_container<float>({capacity, 3}, 0.0f);
                    accelerations = xarray_container<float>({capacity, 3}, 0.0f);
                    colors = xarray_container<float>({capacity, 4}, 1.0f);
                    sizes = xarray_container<float>({capacity, 1}, 1.0f);
                    rotations = xarray_container<float>({capacity, 4}, 0.0f);
                    lifetimes = xarray_container<float>({capacity, 1}, 0.0f);
                    max_lifetimes = xarray_container<float>({capacity, 1}, 1.0f);
                    custom_data = xarray_container<float>({capacity, 4}, 0.0f);
                    angular_velocities = xarray_container<float>({capacity, 3}, 0.0f);
                    rotation_axes = xarray_container<float>({capacity, 3}, 0.0f);
                    active = xarray_container<uint8_t>({capacity, 1}, 0);
                    // Initialize rotation quaternions to identity
                    for (size_t i = 0; i < capacity; ++i)
                        rotations(i, 0) = 1.0f;
                }

                void reset()
                {
                    active_count = 0;
                    emission_accumulator = 0.0f;
                    for (size_t i = 0; i < max_particles; ++i)
                        active(i, 0) = 0;
                }

                void emit_particle(float dt)
                {
                    size_t idx = find_free_slot();
                    if (idx >= max_particles) return;

                    // Position within emitter shape
                    float px, py, pz;
                    if (emitter_shape_box)
                    {
                        px = emitter_position.x + (dist01(rng) - 0.5f) * emitter_extents.x;
                        py = emitter_position.y + (dist01(rng) - 0.5f) * emitter_extents.y;
                        pz = emitter_position.z + (dist01(rng) - 0.5f) * emitter_extents.z;
                    }
                    else
                    {
                        // Sphere
                        float theta = dist01(rng) * 2.0f * M_PI;
                        float phi = std::acos(2.0f * dist01(rng) - 1.0f);
                        float radius = emitter_extents.x * std::cbrt(dist01(rng));
                        px = emitter_position.x + radius * std::sin(phi) * std::cos(theta);
                        py = emitter_position.y + radius * std::sin(phi) * std::sin(theta);
                        pz = emitter_position.z + radius * std::cos(phi);
                    }
                    positions(idx, 0) = px;
                    positions(idx, 1) = py;
                    positions(idx, 2) = pz;

                    // Velocity direction
                    float speed = initial_speed_min + dist01(rng) * (initial_speed_max - initial_speed_min);
                    godot::Vector3 base_dir = emitter_direction.normalized();
                    godot::Vector3 up = (std::abs(base_dir.y) < 0.99f) ? godot::Vector3(0, 1, 0) : godot::Vector3(1, 0, 0);
                    godot::Vector3 right = base_dir.cross(up).normalized();
                    godot::Vector3 forward = right.cross(base_dir).normalized();

                    float spread_angle = (dist01(rng) - 0.5f) * emitter_spread;
                    float rot_angle = emitter_angle_min + dist01(rng) * (emitter_angle_max - emitter_angle_min);
                    godot::Quaternion q1(base_dir, spread_angle);
                    godot::Quaternion q2(base_dir, rot_angle);
                    godot::Vector3 dir = q2.xform(q1.xform(base_dir));

                    velocities(idx, 0) = dir.x * speed;
                    velocities(idx, 1) = dir.y * speed;
                    velocities(idx, 2) = dir.z * speed;

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
                    rotations(idx, 0) = 1.0f; rotations(idx, 1) = 0.0f; rotations(idx, 2) = 0.0f; rotations(idx, 3) = 0.0f;

                    // Random rotation axis
                    float ax = dist01(rng) * 2.0f - 1.0f;
                    float ay = dist01(rng) * 2.0f - 1.0f;
                    float az = dist01(rng) * 2.0f - 1.0f;
                    float len = std::sqrt(ax*ax + ay*ay + az*az);
                    if (len > 0.001f) { ax /= len; ay /= len; az /= len; }
                    rotation_axes(idx, 0) = ax;
                    rotation_axes(idx, 1) = ay;
                    rotation_axes(idx, 2) = az;

                    float rot_speed = rotation_speed_min + dist01(rng) * (rotation_speed_max - rotation_speed_min);
                    angular_velocities(idx, 0) = ax * rot_speed;
                    angular_velocities(idx, 1) = ay * rot_speed;
                    angular_velocities(idx, 2) = az * rot_speed;

                    accelerations(idx, 0) = 0.0f;
                    accelerations(idx, 1) = 0.0f;
                    accelerations(idx, 2) = 0.0f;
                    custom_data(idx, 0) = 0.0f; custom_data(idx, 1) = 0.0f; custom_data(idx, 2) = 0.0f; custom_data(idx, 3) = 0.0f;

                    active(idx, 0) = 1;
                    active_count++;
                }

                void emit_batch(float delta)
                {
                    emission_accumulator += emission_rate * delta;
                    int to_emit = static_cast<int>(emission_accumulator);
                    emission_accumulator -= static_cast<float>(to_emit);
                    for (int i = 0; i < to_emit; ++i) emit_particle(delta);
                }

                void update(float delta)
                {
                    for (size_t i = 0; i < max_particles; ++i)
                    {
                        if (!active(i, 0)) continue;

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
                        velocities(i, 2) += (gravity.z + accelerations(i, 2)) * delta;
                        velocities(i, 0) *= (1.0f - damping * delta);
                        velocities(i, 1) *= (1.0f - damping * delta);
                        velocities(i, 2) *= (1.0f - damping * delta);

                        positions(i, 0) += velocities(i, 0) * delta;
                        positions(i, 1) += velocities(i, 1) * delta;
                        positions(i, 2) += velocities(i, 2) * delta;

                        // Rotation (quaternion integration)
                        float wx = angular_velocities(i, 0) * delta * 0.5f;
                        float wy = angular_velocities(i, 1) * delta * 0.5f;
                        float wz = angular_velocities(i, 2) * delta * 0.5f;
                        float qw = rotations(i, 0), qx = rotations(i, 1), qy = rotations(i, 2), qz = rotations(i, 3);
                        float nw = qw - wx*qx - wy*qy - wz*qz;
                        float nx = qx + wx*qw + wy*qz - wz*qy;
                        float ny = qy - wx*qz + wy*qw + wz*qx;
                        float nz = qz + wx*qy - wy*qx + wz*qw;
                        float len = std::sqrt(nw*nw + nx*nx + ny*ny + nz*nz);
                        if (len > 0) { nw/=len; nx/=len; ny/=len; nz/=len; }
                        rotations(i, 0) = nw; rotations(i, 1) = nx; rotations(i, 2) = ny; rotations(i, 3) = nz;

                        // Interpolate appearance
                        float t = lifetimes(i, 0) / max_lifetimes(i, 0);
                        colors(i, 0) = color_start.r * (1.0f - t) + color_end.r * t;
                        colors(i, 1) = color_start.g * (1.0f - t) + color_end.g * t;
                        colors(i, 2) = color_start.b * (1.0f - t) + color_end.b * t;
                        colors(i, 3) = color_start.a * (1.0f - t) + color_end.a * t;
                        sizes(i, 0) = size_start * (1.0f - t) + size_end * t;
                    }

                    // Collision with ground plane or bounds (simplified)
                    if (enable_collision)
                    {
                        for (size_t i = 0; i < max_particles; ++i)
                        {
                            if (!active(i, 0)) continue;
                            float y = positions(i, 1);
                            if (y < collision_radius)
                            {
                                positions(i, 1) = collision_radius;
                                velocities(i, 1) = -velocities(i, 1) * bounce_factor;
                            }
                        }
                    }
                }

                size_t find_free_slot()
                {
                    for (size_t i = 0; i < max_particles; ++i)
                        if (!active(i, 0)) return i;
                    return max_particles;
                }

                void pack_active(xarray_container<float>& packed_positions,
                                 xarray_container<float>& packed_colors,
                                 xarray_container<float>& packed_sizes,
                                 xarray_container<float>& packed_rotations,
                                 xarray_container<float>& packed_custom) const
                {
                    packed_positions = xarray_container<float>({active_count, 3});
                    packed_colors = xarray_container<float>({active_count, 4});
                    packed_sizes = xarray_container<float>({active_count, 1});
                    packed_rotations = xarray_container<float>({active_count, 4});
                    packed_custom = xarray_container<float>({active_count, 4});

                    size_t out_idx = 0;
                    for (size_t i = 0; i < max_particles; ++i)
                    {
                        if (!active(i, 0)) continue;
                        packed_positions(out_idx, 0) = positions(i, 0);
                        packed_positions(out_idx, 1) = positions(i, 1);
                        packed_positions(out_idx, 2) = positions(i, 2);
                        packed_colors(out_idx, 0) = colors(i, 0);
                        packed_colors(out_idx, 1) = colors(i, 1);
                        packed_colors(out_idx, 2) = colors(i, 2);
                        packed_colors(out_idx, 3) = colors(i, 3);
                        packed_sizes(out_idx, 0) = sizes(i, 0);
                        packed_rotations(out_idx, 0) = rotations(i, 0);
                        packed_rotations(out_idx, 1) = rotations(i, 1);
                        packed_rotations(out_idx, 2) = rotations(i, 2);
                        packed_rotations(out_idx, 3) = rotations(i, 3);
                        packed_custom(out_idx, 0) = custom_data(i, 0);
                        packed_custom(out_idx, 1) = custom_data(i, 1);
                        packed_custom(out_idx, 2) = custom_data(i, 2);
                        packed_custom(out_idx, 3) = custom_data(i, 3);
                        ++out_idx;
                    }
                }
            };

            // --------------------------------------------------------------------
            // XParticles3D - Godot node for tensor-based 3D particles
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XParticles3D : public godot::Node3D
            {
                GDCLASS(XParticles3D, godot::Node3D)

            private:
                ParticleSystemState3D m_system;
                godot::Ref<XTensorNode> m_positions_tensor;
                godot::Ref<XTensorNode> m_velocities_tensor;
                godot::Ref<XTensorNode> m_colors_tensor;
                godot::Ref<XTensorNode> m_sizes_tensor;
                godot::Ref<XTensorNode> m_rotations_tensor;
                godot::Ref<XTensorNode> m_custom_tensor;

                godot::Ref<godot::Texture2D> m_texture;
                godot::Ref<godot::Curve> m_scale_curve;
                godot::Ref<godot::Gradient> m_color_gradient;
                godot::Ref<godot::Material> m_material;

                bool m_emitting = true;
                bool m_one_shot = false;
                bool m_auto_update = true;
                bool m_local_space = true;
                bool m_use_gpu = true;
                float m_speed_scale = 1.0f;
                float m_time = 0.0f;
                float m_lifetime = 0.0f;

                godot::RID m_multimesh_rid;
                godot::RID m_instance_rid;

            protected:
                static void _bind_methods()
                {
                    // Tensor access
                    godot::ClassDB::bind_method(godot::D_METHOD("set_positions_tensor", "tensor"), &XParticles3D::set_positions_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_positions_tensor"), &XParticles3D::get_positions_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_velocities_tensor", "tensor"), &XParticles3D::set_velocities_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_velocities_tensor"), &XParticles3D::get_velocities_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_colors_tensor", "tensor"), &XParticles3D::set_colors_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_colors_tensor"), &XParticles3D::get_colors_tensor);

                    // Properties
                    godot::ClassDB::bind_method(godot::D_METHOD("set_emitting", "emitting"), &XParticles3D::set_emitting);
                    godot::ClassDB::bind_method(godot::D_METHOD("is_emitting"), &XParticles3D::is_emitting);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_one_shot", "one_shot"), &XParticles3D::set_one_shot);
                    godot::ClassDB::bind_method(godot::D_METHOD("is_one_shot"), &XParticles3D::is_one_shot);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_amount", "amount"), &XParticles3D::set_amount);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_amount"), &XParticles3D::get_amount);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_lifetime", "lifetime"), &XParticles3D::set_lifetime);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_lifetime"), &XParticles3D::get_lifetime);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_texture", "texture"), &XParticles3D::set_texture);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_texture"), &XParticles3D::get_texture);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_material", "material"), &XParticles3D::set_material);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_material"), &XParticles3D::get_material);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_speed_scale", "scale"), &XParticles3D::set_speed_scale);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_speed_scale"), &XParticles3D::get_speed_scale);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_gravity", "gravity"), &XParticles3D::set_gravity);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_gravity"), &XParticles3D::get_gravity);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_initial_velocity", "min", "max"), &XParticles3D::set_initial_velocity);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_spread", "spread"), &XParticles3D::set_spread);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_spread"), &XParticles3D::get_spread);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_color", "start", "end"), &XParticles3D::set_color);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_size", "start", "end"), &XParticles3D::set_size);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_emitter_shape", "shape", "extents", "direction"), &XParticles3D::set_emitter_shape);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_collision", "enabled", "radius", "bounce"), &XParticles3D::set_collision);

                    // Control
                    godot::ClassDB::bind_method(godot::D_METHOD("restart"), &XParticles3D::restart);
                    godot::ClassDB::bind_method(godot::D_METHOD("emit_particle", "position", "velocity"), &XParticles3D::emit_particle);
                    godot::ClassDB::bind_method(godot::D_METHOD("emit_batch", "count"), &XParticles3D::emit_batch);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear"), &XParticles3D::clear);
                    godot::ClassDB::bind_method(godot::D_METHOD("capture_state"), &XParticles3D::capture_state);
                    godot::ClassDB::bind_method(godot::D_METHOD("restore_state"), &XParticles3D::restore_state);

                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "emitting"), "set_emitting", "is_emitting");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "one_shot"), "set_one_shot", "is_one_shot");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "amount"), "set_amount", "get_amount");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "lifetime"), "set_lifetime", "get_lifetime");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "texture", godot::PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_texture", "get_texture");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "material", godot::PROPERTY_HINT_RESOURCE_TYPE, "Material"), "set_material", "get_material");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "speed_scale"), "set_speed_scale", "get_speed_scale");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");

                    ADD_SIGNAL(godot::MethodInfo("finished"));
                    ADD_SIGNAL(godot::MethodInfo("particle_emitted", godot::PropertyInfo(godot::Variant::INT, "count")));
                }

            public:
                XParticles3D() : m_system(10000) {}

                void _ready() override
                {
                    set_process(true);
                    if (m_use_gpu) _create_multimesh();
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
                        m_emitting = false;

                    if (m_system.active_count == 0 && m_time >= m_lifetime && m_lifetime > 0)
                        emit_signal("finished");

                    if (m_auto_update)
                    {
                        update_tensors();
                        if (m_use_gpu) _update_multimesh();
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
                void set_material(const godot::Ref<godot::Material>& material) { m_material = material; }
                godot::Ref<godot::Material> get_material() const { return m_material; }
                void set_speed_scale(float scale) { m_speed_scale = scale; }
                float get_speed_scale() const { return m_speed_scale; }
                void set_gravity(const godot::Vector3& gravity) { m_system.gravity = gravity; }
                godot::Vector3 get_gravity() const { return m_system.gravity; }
                void set_initial_velocity(float min, float max) { m_system.initial_speed_min = min; m_system.initial_speed_max = max; }
                void set_spread(float spread) { m_system.emitter_spread = spread * M_PI / 180.0f; }
                float get_spread() const { return m_system.emitter_spread * 180.0f / M_PI; }
                void set_color(const godot::Color& start, const godot::Color& end) { m_system.color_start = start; m_system.color_end = end; }
                void set_size(float start, float end) { m_system.size_start = start; m_system.size_end = end; }
                void set_emitter_shape(const godot::String& shape, const godot::Vector3& extents, const godot::Vector3& direction)
                {
                    m_system.emitter_shape_box = (shape == "box");
                    m_system.emitter_extents = extents;
                    m_system.emitter_direction = direction;
                }
                void set_collision(bool enabled, float radius, float bounce)
                {
                    m_system.enable_collision = enabled;
                    m_system.collision_radius = radius;
                    m_system.bounce_factor = bounce;
                }

                void restart() { m_system.reset(); m_time = 0.0f; m_emitting = true; }
                void emit_particle(const godot::Vector3& position, const godot::Vector3& velocity)
                {
                    size_t idx = m_system.find_free_slot();
                    if (idx >= m_system.max_particles) return;
                    m_system.positions(idx,0)=position.x; m_system.positions(idx,1)=position.y; m_system.positions(idx,2)=position.z;
                    m_system.velocities(idx,0)=velocity.x; m_system.velocities(idx,1)=velocity.y; m_system.velocities(idx,2)=velocity.z;
                    m_system.active(idx,0)=1;
                    m_system.active_count++;
                    emit_signal("particle_emitted", 1);
                }
                void emit_batch(int count) { for(int i=0;i<count;++i) m_system.emit_particle(0.0f); emit_signal("particle_emitted", count); }
                void clear() { m_system.reset(); }

                void capture_state() { update_tensors(); }
                void restore_state()
                {
                    if (m_positions_tensor.is_valid())
                    {
                        auto pos = m_positions_tensor->get_tensor_resource()->m_data.to_float_array();
                        for(size_t i=0;i<std::min(pos.shape()[0], m_system.max_particles);++i)
                        {
                            m_system.positions(i,0)=pos(i,0); m_system.positions(i,1)=pos(i,1); m_system.positions(i,2)=pos(i,2);
                            m_system.active(i,0)=1;
                        }
                        m_system.active_count = pos.shape()[0];
                    }
                }

            private:
                void update_tensors()
                {
                    xarray_container<float> pos, col, sz, rot, cust;
                    m_system.pack_active(pos, col, sz, rot, cust);
                    if (m_positions_tensor.is_valid()) m_positions_tensor->set_data(XVariant::from_xarray(pos.cast<double>()).variant());
                    if (m_colors_tensor.is_valid()) m_colors_tensor->set_data(XVariant::from_xarray(col.cast<double>()).variant());
                    if (m_sizes_tensor.is_valid()) m_sizes_tensor->set_data(XVariant::from_xarray(sz.cast<double>()).variant());
                    if (m_rotations_tensor.is_valid()) m_rotations_tensor->set_data(XVariant::from_xarray(rot.cast<double>()).variant());
                    if (m_custom_tensor.is_valid()) m_custom_tensor->set_data(XVariant::from_xarray(cust.cast<double>()).variant());
                }

                void _create_multimesh()
                {
                    godot::RenderingServer* rs = godot::RenderingServer::get_singleton();
                    if (!rs) return;
                    godot::RID mesh_rid = _create_quad_mesh();
                    m_multimesh_rid = rs->multimesh_create();
                    rs->multimesh_allocate_data(m_multimesh_rid, static_cast<int>(m_system.max_particles), godot::RenderingServer::MULTIMESH_TRANSFORM_3D, true);
                    rs->multimesh_set_mesh(m_multimesh_rid, mesh_rid);
                    m_instance_rid = rs->instance_create2(m_multimesh_rid, get_world_3d()->get_scenario());
                    rs->instance_set_visible(m_instance_rid, true);
                    if (m_material.is_valid()) rs->instance_geometry_set_material_override(m_instance_rid, m_material->get_rid());
                }

                godot::RID _create_quad_mesh()
                {
                    godot::RenderingServer* rs = godot::RenderingServer::get_singleton();
                    godot::RID mesh_rid = rs->mesh_create();
                    godot::PackedVector3Array vertices;
                    vertices.push_back(godot::Vector3(-0.5, -0.5, 0));
                    vertices.push_back(godot::Vector3( 0.5, -0.5, 0));
                    vertices.push_back(godot::Vector3( 0.5,  0.5, 0));
                    vertices.push_back(godot::Vector3(-0.5,  0.5, 0));
                    godot::PackedVector2Array uvs;
                    uvs.push_back(godot::Vector2(0,0)); uvs.push_back(godot::Vector2(1,0));
                    uvs.push_back(godot::Vector2(1,1)); uvs.push_back(godot::Vector2(0,1));
                    godot::PackedInt32Array indices = {0,1,2,0,2,3};
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
                        godot::Transform3D t;
                        t.origin = godot::Vector3(pos(i,0), pos(i,1), pos(i,2));
                        if (rot.size() > 0)
                            t.basis = godot::Basis(godot::Quaternion(rot(i,0), rot(i,1), rot(i,2), rot(i,3)));
                        else
                            t.basis = godot::Basis();
                        float scale = sz.size() > 0 ? sz(i,0) : 1.0f;
                        t.basis.scale(godot::Vector3(scale, scale, scale));
                        rs->multimesh_instance_set_transform(m_multimesh_rid, static_cast<int>(i), t);
                        if (col.size() > 0)
                            rs->multimesh_instance_set_color(m_multimesh_rid, static_cast<int>(i), godot::Color(col(i,0), col(i,1), col(i,2), col(i,3)));
                    }
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            class XParticles3DRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XParticles3D>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::ParticleSystemState3D;
        using godot_bridge::XParticles3D;
        using godot_bridge::XParticles3DRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XPARTICLES3D_HPP

// godot/xparticles3d.hpp