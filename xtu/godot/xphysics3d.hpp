// godot/xphysics3d.hpp

#ifndef XTENSOR_XPHYSICS3D_HPP
#define XTENSOR_XPHYSICS3D_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xintersection.hpp"
#include "../math/xquaternion.hpp"
#include "xvariant.hpp"
#include "xclassdb.hpp"
#include "xnode.hpp"

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

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/node3d.hpp>
    #include <godot_cpp/classes/physics_body3d.hpp>
    #include <godot_cpp/classes/rigid_body3d.hpp>
    #include <godot_cpp/classes/static_body3d.hpp>
    #include <godot_cpp/classes/character_body3d.hpp>
    #include <godot_cpp/classes/area3d.hpp>
    #include <godot_cpp/classes/collision_shape3d.hpp>
    #include <godot_cpp/classes/physics_direct_space_state3d.hpp>
    #include <godot_cpp/classes/physics_ray_query_parameters3d.hpp>
    #include <godot_cpp/classes/physics_shape_query_parameters3d.hpp>
    #include <godot_cpp/classes/world3d.hpp>
    #include <godot_cpp/classes/physics_material.hpp>
    #include <godot_cpp/classes/box_shape3d.hpp>
    #include <godot_cpp/classes/sphere_shape3d.hpp>
    #include <godot_cpp/classes/capsule_shape3d.hpp>
    #include <godot_cpp/classes/cylinder_shape3d.hpp>
    #include <godot_cpp/classes/convex_polygon_shape3d.hpp>
    #include <godot_cpp/classes/concave_polygon_shape3d.hpp>
    #include <godot_cpp/classes/height_map_shape3d.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/transform3d.hpp>
    #include <godot_cpp/variant/vector3.hpp>
    #include <godot_cpp/variant/basis.hpp>
    #include <godot_cpp/variant/quaternion.hpp>
    #include <godot_cpp/variant/aabb.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // Quaternion utilities for tensor operations
            // --------------------------------------------------------------------
            namespace quat_utils
            {
                inline xarray_container<double> quat_multiply(const xarray_container<double>& q1,
                                                               const xarray_container<double>& q2)
                {
                    // q = (w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    //      w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    //      w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    //      w1*z2 + x1*y2 - y1*x2 + z1*w2)
                    size_t n = q1.shape()[0];
                    xarray_container<double> result({n, 4});
                    for (size_t i = 0; i < n; ++i)
                    {
                        double w1 = q1(i, 0), x1 = q1(i, 1), y1 = q1(i, 2), z1 = q1(i, 3);
                        double w2 = q2(i, 0), x2 = q2(i, 1), y2 = q2(i, 2), z2 = q2(i, 3);
                        result(i, 0) = w1*w2 - x1*x2 - y1*y2 - z1*z2;
                        result(i, 1) = w1*x2 + x1*w2 + y1*z2 - z1*y2;
                        result(i, 2) = w1*y2 - x1*z2 + y1*w2 + z1*x2;
                        result(i, 3) = w1*z2 + x1*y2 - y1*x2 + z1*w2;
                    }
                    return result;
                }

                inline xarray_container<double> quat_rotate(const xarray_container<double>& q,
                                                            const xarray_container<double>& v)
                {
                    // Rotate vector v by quaternion q
                    size_t n = q.shape()[0];
                    xarray_container<double> result({n, 3});
                    for (size_t i = 0; i < n; ++i)
                    {
                        double qw = q(i, 0), qx = q(i, 1), qy = q(i, 2), qz = q(i, 3);
                        double vx = v(i, 0), vy = v(i, 1), vz = v(i, 2);
                        double tx = 2.0 * (qy*vz - qz*vy);
                        double ty = 2.0 * (qz*vx - qx*vz);
                        double tz = 2.0 * (qx*vy - qy*vx);
                        result(i, 0) = vx + qw*tx + (qy*tz - qz*ty);
                        result(i, 1) = vy + qw*ty + (qz*tx - qx*tz);
                        result(i, 2) = vz + qw*tz + (qx*ty - qy*tx);
                    }
                    return result;
                }

                inline xarray_container<double> quat_from_axis_angle(const xarray_container<double>& axis,
                                                                     const xarray_container<double>& angle)
                {
                    size_t n = axis.shape()[0];
                    xarray_container<double> result({n, 4});
                    for (size_t i = 0; i < n; ++i)
                    {
                        double ax = axis(i, 0), ay = axis(i, 1), az = axis(i, 2);
                        double len = std::sqrt(ax*ax + ay*ay + az*az);
                        if (len < 1e-10)
                        {
                            result(i, 0) = 1.0; result(i, 1) = 0.0; result(i, 2) = 0.0; result(i, 3) = 0.0;
                            continue;
                        }
                        ax /= len; ay /= len; az /= len;
                        double half_angle = angle(i) * 0.5;
                        double s = std::sin(half_angle);
                        result(i, 0) = std::cos(half_angle);
                        result(i, 1) = ax * s;
                        result(i, 2) = ay * s;
                        result(i, 3) = az * s;
                    }
                    return result;
                }

                inline xarray_container<double> quat_slerp(const xarray_container<double>& q1,
                                                           const xarray_container<double>& q2,
                                                           double t)
                {
                    size_t n = q1.shape()[0];
                    xarray_container<double> result({n, 4});
                    for (size_t i = 0; i < n; ++i)
                    {
                        double w1 = q1(i, 0), x1 = q1(i, 1), y1 = q1(i, 2), z1 = q1(i, 3);
                        double w2 = q2(i, 0), x2 = q2(i, 1), y2 = q2(i, 2), z2 = q2(i, 3);
                        double dot = w1*w2 + x1*x2 + y1*y2 + z1*z2;
                        if (dot < 0.0) { w2 = -w2; x2 = -x2; y2 = -y2; z2 = -z2; dot = -dot; }
                        if (dot > 0.9995)
                        {
                            result(i, 0) = w1 + t*(w2 - w1);
                            result(i, 1) = x1 + t*(x2 - x1);
                            result(i, 2) = y1 + t*(y2 - y1);
                            result(i, 3) = z1 + t*(z2 - z1);
                            double len = std::sqrt(result(i,0)*result(i,0) + result(i,1)*result(i,1) + result(i,2)*result(i,2) + result(i,3)*result(i,3));
                            result(i,0) /= len; result(i,1) /= len; result(i,2) /= len; result(i,3) /= len;
                        }
                        else
                        {
                            double theta_0 = std::acos(dot);
                            double theta = theta_0 * t;
                            double sin_theta = std::sin(theta);
                            double sin_theta_0 = std::sin(theta_0);
                            double s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
                            double s1 = sin_theta / sin_theta_0;
                            result(i, 0) = s0 * w1 + s1 * w2;
                            result(i, 1) = s0 * x1 + s1 * x2;
                            result(i, 2) = s0 * y1 + s1 * y2;
                            result(i, 3) = s0 * z1 + s1 * z2;
                        }
                    }
                    return result;
                }
            }

            // --------------------------------------------------------------------
            // XPhysicsTensor3D - Batch 3D physics operations using tensors
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XPhysicsTensor3D : public godot::Node3D
            {
                GDCLASS(XPhysicsTensor3D, godot::Node3D)

            private:
                godot::Ref<XTensorNode> m_positions_tensor;      // Nx3
                godot::Ref<XTensorNode> m_velocities_tensor;     // Nx3
                godot::Ref<XTensorNode> m_rotations_tensor;      // Nx4 (quaternions)
                godot::Ref<XTensorNode> m_angular_velocities_tensor; // Nx3
                godot::Ref<XTensorNode> m_masses_tensor;         // N
                godot::Ref<XTensorNode> m_forces_tensor;         // Nx3
                godot::Ref<XTensorNode> m_torques_tensor;        // Nx3
                godot::Ref<XTensorNode> m_inertia_tensor;        // Nx3 (diagonal inertia)
                
                std::vector<godot::RID> m_body_rids;
                std::vector<godot::Ref<godot::Shape3D>> m_shapes;
                
                godot::Ref<godot::PhysicsMaterial> m_physics_material;
                bool m_bodies_created = false;
                bool m_auto_sync = true;
                float m_time_step = 1.0f / 60.0f;
                godot::Vector3 m_gravity = godot::Vector3(0, -9.8f, 0);
                float m_damping_linear = 0.1f;
                float m_damping_angular = 0.1f;

            protected:
                static void _bind_methods()
                {
                    // Tensor access
                    godot::ClassDB::bind_method(godot::D_METHOD("set_positions_tensor", "tensor"), &XPhysicsTensor3D::set_positions_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_positions_tensor"), &XPhysicsTensor3D::get_positions_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_velocities_tensor", "tensor"), &XPhysicsTensor3D::set_velocities_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_velocities_tensor"), &XPhysicsTensor3D::get_velocities_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_rotations_tensor", "tensor"), &XPhysicsTensor3D::set_rotations_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_rotations_tensor"), &XPhysicsTensor3D::get_rotations_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_masses_tensor", "tensor"), &XPhysicsTensor3D::set_masses_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_masses_tensor"), &XPhysicsTensor3D::get_masses_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_forces_tensor", "tensor"), &XPhysicsTensor3D::set_forces_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_forces_tensor"), &XPhysicsTensor3D::get_forces_tensor);
                    
                    // Body management
                    godot::ClassDB::bind_method(godot::D_METHOD("create_bodies", "shape_type", "shape_params"), &XPhysicsTensor3D::create_bodies);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear_bodies"), &XPhysicsTensor3D::clear_bodies);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_body_count"), &XPhysicsTensor3D::get_body_count);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_body_positions", "positions"), &XPhysicsTensor3D::set_body_positions);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_body_positions"), &XPhysicsTensor3D::get_body_positions);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_body_velocities", "velocities"), &XPhysicsTensor3D::set_body_velocities);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_body_velocities"), &XPhysicsTensor3D::get_body_velocities);
                    godot::ClassDB::bind_method(godot::D_METHOD("apply_forces", "forces"), &XPhysicsTensor3D::apply_forces);
                    godot::ClassDB::bind_method(godot::D_METHOD("apply_impulses", "impulses", "positions"), &XPhysicsTensor3D::apply_impulses);
                    
                    // Simulation
                    godot::ClassDB::bind_method(godot::D_METHOD("step", "delta"), &XPhysicsTensor3D::step);
                    godot::ClassDB::bind_method(godot::D_METHOD("simulate", "duration", "sub_steps"), &XPhysicsTensor3D::simulate);
                    godot::ClassDB::bind_method(godot::D_METHOD("sync_from_bodies"), &XPhysicsTensor3D::sync_from_bodies);
                    godot::ClassDB::bind_method(godot::D_METHOD("sync_to_bodies"), &XPhysicsTensor3D::sync_to_bodies);
                    
                    // Queries
                    godot::ClassDB::bind_method(godot::D_METHOD("ray_cast_batch", "origins", "directions"), &XPhysicsTensor3D::ray_cast_batch);
                    godot::ClassDB::bind_method(godot::D_METHOD("overlap_sphere_batch", "centers", "radii"), &XPhysicsTensor3D::overlap_sphere_batch);
                    godot::ClassDB::bind_method(godot::D_METHOD("collision_pairs"), &XPhysicsTensor3D::collision_pairs);
                    
                    // Collision detection (tensorized)
                    godot::ClassDB::bind_method(godot::D_METHOD("compute_contacts"), &XPhysicsTensor3D::compute_contacts);
                    godot::ClassDB::bind_method(godot::D_METHOD("resolve_collisions", "restitution"), &XPhysicsTensor3D::resolve_collisions, godot::DEFVAL(0.5));
                    
                    // Properties
                    godot::ClassDB::bind_method(godot::D_METHOD("set_auto_sync", "enabled"), &XPhysicsTensor3D::set_auto_sync);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_auto_sync"), &XPhysicsTensor3D::get_auto_sync);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_gravity", "gravity"), &XPhysicsTensor3D::set_gravity);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_gravity"), &XPhysicsTensor3D::get_gravity);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_damping", "linear", "angular"), &XPhysicsTensor3D::set_damping);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "positions_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_positions_tensor", "get_positions_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "velocities_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_velocities_tensor", "get_velocities_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "rotations_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_rotations_tensor", "get_rotations_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "masses_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_masses_tensor", "get_masses_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "auto_sync"), "set_auto_sync", "get_auto_sync");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::VECTOR3, "gravity"), "set_gravity", "get_gravity");
                    
                    // Signals
                    ADD_SIGNAL(godot::MethodInfo("bodies_created", godot::PropertyInfo(godot::Variant::INT, "count")));
                    ADD_SIGNAL(godot::MethodInfo("collision_detected", godot::PropertyInfo(godot::Variant::INT, "body_a"), godot::PropertyInfo(godot::Variant::INT, "body_b")));
                }

            public:
                XPhysicsTensor3D()
                {
                    ensure_tensors();
                }

                void _ready() override
                {
                    if (m_bodies_created && m_auto_sync)
                        sync_to_bodies();
                }

                void _physics_process(double delta) override
                {
                    if (m_bodies_created && m_auto_sync)
                        step(static_cast<float>(delta));
                }

                // Tensor access
                void set_positions_tensor(const godot::Ref<XTensorNode>& tensor) { m_positions_tensor = tensor; }
                godot::Ref<XTensorNode> get_positions_tensor() const { return m_positions_tensor; }
                void set_velocities_tensor(const godot::Ref<XTensorNode>& tensor) { m_velocities_tensor = tensor; }
                godot::Ref<XTensorNode> get_velocities_tensor() const { return m_velocities_tensor; }
                void set_rotations_tensor(const godot::Ref<XTensorNode>& tensor) { m_rotations_tensor = tensor; }
                godot::Ref<XTensorNode> get_rotations_tensor() const { return m_rotations_tensor; }
                void set_masses_tensor(const godot::Ref<XTensorNode>& tensor) { m_masses_tensor = tensor; }
                godot::Ref<XTensorNode> get_masses_tensor() const { return m_masses_tensor; }
                void set_forces_tensor(const godot::Ref<XTensorNode>& tensor) { m_forces_tensor = tensor; }
                godot::Ref<XTensorNode> get_forces_tensor() const { return m_forces_tensor; }

                // Body management
                void create_bodies(const godot::String& shape_type, const godot::Variant& shape_params)
                {
                    clear_bodies();
                    size_t n = get_body_count_from_tensors();
                    if (n == 0) return;
                    
                    godot::PhysicsServer3D* ps = godot::PhysicsServer3D::get_singleton();
                    godot::RID space = get_world_3d()->get_space();
                    
                    godot::Ref<godot::Shape3D> base_shape;
                    std::string type = shape_type.utf8().get_data();
                    if (type == "sphere")
                    {
                        godot::Ref<godot::SphereShape3D> sphere;
                        sphere.instantiate();
                        float radius = shape_params.get_type() == godot::Variant::FLOAT ? 
                            static_cast<float>(shape_params) : 0.5f;
                        sphere->set_radius(radius);
                        base_shape = sphere;
                    }
                    else if (type == "box")
                    {
                        godot::Ref<godot::BoxShape3D> box;
                        box.instantiate();
                        if (shape_params.get_type() == godot::Variant::VECTOR3)
                            box->set_size(godot::Vector3(shape_params));
                        else
                            box->set_size(godot::Vector3(1, 1, 1));
                        base_shape = box;
                    }
                    else if (type == "capsule")
                    {
                        godot::Ref<godot::CapsuleShape3D> capsule;
                        capsule.instantiate();
                        if (shape_params.get_type() == godot::Variant::ARRAY)
                        {
                            godot::Array arr = shape_params;
                            if (arr.size() >= 2)
                            {
                                capsule->set_radius(arr[0]);
                                capsule->set_height(arr[1]);
                            }
                        }
                        base_shape = capsule;
                    }
                    else if (type == "cylinder")
                    {
                        godot::Ref<godot::CylinderShape3D> cylinder;
                        cylinder.instantiate();
                        if (shape_params.get_type() == godot::Variant::ARRAY)
                        {
                            godot::Array arr = shape_params;
                            if (arr.size() >= 2)
                            {
                                cylinder->set_radius(arr[0]);
                                cylinder->set_height(arr[1]);
                            }
                        }
                        base_shape = cylinder;
                    }
                    else
                    {
                        godot::Ref<godot::SphereShape3D> sphere;
                        sphere.instantiate();
                        sphere->set_radius(0.5f);
                        base_shape = sphere;
                    }
                    
                    auto positions = get_positions_array();
                    auto masses = get_masses_array();
                    
                    m_body_rids.resize(n);
                    m_shapes.resize(n, base_shape);
                    
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::RID body = ps->body_create();
                        ps->body_set_mode(body, godot::PhysicsServer3D::BODY_MODE_RIGID);
                        ps->body_set_space(body, space);
                        
                        godot::Vector3 pos(positions(i, 0), positions(i, 1), positions(i, 2));
                        godot::Transform3D transform(godot::Basis(), pos);
                        ps->body_set_state(body, godot::PhysicsServer3D::BODY_STATE_TRANSFORM, transform);
                        
                        float mass = (i < masses.size()) ? masses(i) : 1.0f;
                        ps->body_set_param(body, godot::PhysicsServer3D::BODY_PARAM_MASS, mass);
                        
                        godot::RID shape_rid = base_shape->get_rid();
                        ps->body_add_shape(body, shape_rid, godot::Transform3D());
                        
                        m_body_rids[i] = body;
                    }
                    
                    m_bodies_created = true;
                    emit_signal("bodies_created", static_cast<int64_t>(n));
                }

                void clear_bodies()
                {
                    godot::PhysicsServer3D* ps = godot::PhysicsServer3D::get_singleton();
                    for (const auto& rid : m_body_rids)
                    {
                        if (rid != godot::RID())
                            ps->free_rid(rid);
                    }
                    m_body_rids.clear();
                    m_bodies_created = false;
                }

                int64_t get_body_count() const { return static_cast<int64_t>(m_body_rids.size()); }

                void set_body_positions(const godot::Ref<XTensorNode>& positions)
                {
                    m_positions_tensor = positions;
                    if (m_auto_sync && m_bodies_created) sync_to_bodies();
                }

                godot::Ref<XTensorNode> get_body_positions() const { return m_positions_tensor; }

                void set_body_velocities(const godot::Ref<XTensorNode>& velocities)
                {
                    m_velocities_tensor = velocities;
                    if (m_auto_sync && m_bodies_created) sync_to_bodies();
                }

                godot::Ref<XTensorNode> get_body_velocities() const { return m_velocities_tensor; }

                void apply_forces(const godot::Ref<XTensorNode>& forces)
                {
                    if (!m_bodies_created) return;
                    godot::PhysicsServer3D* ps = godot::PhysicsServer3D::get_singleton();
                    auto forces_arr = forces->get_tensor_resource()->m_data.to_double_array();
                    for (size_t i = 0; i < m_body_rids.size() && i < forces_arr.shape()[0]; ++i)
                    {
                        godot::Vector3 force(forces_arr(i, 0), forces_arr(i, 1), forces_arr(i, 2));
                        ps->body_apply_central_force(m_body_rids[i], force);
                    }
                }

                void apply_impulses(const godot::Ref<XTensorNode>& impulses, const godot::Ref<XTensorNode>& positions)
                {
                    if (!m_bodies_created) return;
                    godot::PhysicsServer3D* ps = godot::PhysicsServer3D::get_singleton();
                    auto imp_arr = impulses->get_tensor_resource()->m_data.to_double_array();
                    auto pos_arr = positions.is_valid() ? positions->get_tensor_resource()->m_data.to_double_array() : xarray_container<double>();
                    for (size_t i = 0; i < m_body_rids.size() && i < imp_arr.shape()[0]; ++i)
                    {
                        godot::Vector3 impulse(imp_arr(i, 0), imp_arr(i, 1), imp_arr(i, 2));
                        if (pos_arr.size() > 0 && i < pos_arr.shape()[0])
                        {
                            godot::Vector3 pos(pos_arr(i, 0), pos_arr(i, 1), pos_arr(i, 2));
                            ps->body_apply_impulse(m_body_rids[i], impulse, pos);
                        }
                        else
                        {
                            ps->body_apply_central_impulse(m_body_rids[i], impulse);
                        }
                    }
                }

                // Simulation
                void step(float delta)
                {
                    if (!m_bodies_created) return;
                    godot::PhysicsServer3D* ps = godot::PhysicsServer3D::get_singleton();
                    
                    if (!m_auto_sync)
                    {
                        // Tensorized custom physics (simplified semi-implicit Euler)
                        auto pos = get_positions_array();
                        auto vel = get_velocities_array();
                        auto rot = get_rotations_array();
                        auto ang_vel = get_angular_velocities_array();
                        auto mass = get_masses_array();
                        auto force = get_forces_array();
                        auto torque = get_torques_array();
                        size_t n = pos.shape()[0];
                        
                        for (size_t i = 0; i < n; ++i)
                        {
                            float m = (i < mass.size()) ? mass(i) : 1.0f;
                            godot::Vector3 f = (i < force.shape()[0]) ? godot::Vector3(force(i,0), force(i,1), force(i,2)) : godot::Vector3();
                            godot::Vector3 a = f / m + m_gravity;
                            vel(i,0) += a.x * delta; vel(i,1) += a.y * delta; vel(i,2) += a.z * delta;
                            vel(i,0) *= (1.0f - m_damping_linear * delta);
                            vel(i,1) *= (1.0f - m_damping_linear * delta);
                            vel(i,2) *= (1.0f - m_damping_linear * delta);
                            pos(i,0) += vel(i,0) * delta;
                            pos(i,1) += vel(i,1) * delta;
                            pos(i,2) += vel(i,2) * delta;
                            
                            if (i < torque.shape()[0] && i < ang_vel.shape()[0])
                            {
                                godot::Vector3 t(torque(i,0), torque(i,1), torque(i,2));
                                // Simple angular integration
                                ang_vel(i,0) += t.x * delta;
                                ang_vel(i,1) += t.y * delta;
                                ang_vel(i,2) += t.z * delta;
                                // Integrate quaternion
                                xarray_container<double> q = xt::view(rot, i, xt::all());
                                xarray_container<double> omega_vec = xt::view(ang_vel, i, xt::all());
                                // q += 0.5 * dt * omega * q  (quaternion derivative)
                                double qw = q(0), qx = q(1), qy = q(2), qz = q(3);
                                double wx = omega_vec(0), wy = omega_vec(1), wz = omega_vec(2);
                                qw += 0.5 * delta * (-wx*qx - wy*qy - wz*qz);
                                qx += 0.5 * delta * ( wx*qw + wz*qy - wy*qz);
                                qy += 0.5 * delta * ( wy*qw - wz*qx + wx*qz);
                                qz += 0.5 * delta * ( wz*qw + wy*qx - wx*qy);
                                double len = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
                                rot(i,0) = qw/len; rot(i,1) = qx/len; rot(i,2) = qy/len; rot(i,3) = qz/len;
                            }
                        }
                        m_positions_tensor->set_data(XVariant::from_xarray(pos).variant());
                        m_velocities_tensor->set_data(XVariant::from_xarray(vel).variant());
                        m_rotations_tensor->set_data(XVariant::from_xarray(rot).variant());
                        m_angular_velocities_tensor->set_data(XVariant::from_xarray(ang_vel).variant());
                    }
                    else
                    {
                        sync_from_bodies();
                    }
                }

                void simulate(float duration, int sub_steps)
                {
                    float dt = duration / static_cast<float>(sub_steps);
                    for (int i = 0; i < sub_steps; ++i) step(dt);
                }

                void sync_from_bodies()
                {
                    if (!m_bodies_created) return;
                    godot::PhysicsServer3D* ps = godot::PhysicsServer3D::get_singleton();
                    size_t n = m_body_rids.size();
                    xarray_container<double> pos({n, 3}), vel({n, 3}), rot({n, 4}), ang_vel({n, 3});
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::Transform3D t = ps->body_get_state(m_body_rids[i], godot::PhysicsServer3D::BODY_STATE_TRANSFORM);
                        godot::Vector3 v = ps->body_get_state(m_body_rids[i], godot::PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY);
                        godot::Vector3 av = ps->body_get_state(m_body_rids[i], godot::PhysicsServer3D::BODY_STATE_ANGULAR_VELOCITY);
                        pos(i,0)=t.origin.x; pos(i,1)=t.origin.y; pos(i,2)=t.origin.z;
                        vel(i,0)=v.x; vel(i,1)=v.y; vel(i,2)=v.z;
                        ang_vel(i,0)=av.x; ang_vel(i,1)=av.y; ang_vel(i,2)=av.z;
                        godot::Quaternion q = t.basis.get_quaternion();
                        rot(i,0)=q.w; rot(i,1)=q.x; rot(i,2)=q.y; rot(i,3)=q.z;
                    }
                    m_positions_tensor->set_data(XVariant::from_xarray(pos).variant());
                    m_velocities_tensor->set_data(XVariant::from_xarray(vel).variant());
                    m_rotations_tensor->set_data(XVariant::from_xarray(rot).variant());
                    m_angular_velocities_tensor->set_data(XVariant::from_xarray(ang_vel).variant());
                }

                void sync_to_bodies()
                {
                    if (!m_bodies_created) return;
                    godot::PhysicsServer3D* ps = godot::PhysicsServer3D::get_singleton();
                    auto pos = get_positions_array();
                    auto rot = get_rotations_array();
                    auto vel = get_velocities_array();
                    auto ang_vel = get_angular_velocities_array();
                    for (size_t i = 0; i < m_body_rids.size(); ++i)
                    {
                        godot::Transform3D t;
                        t.origin = godot::Vector3(pos(i,0), pos(i,1), pos(i,2));
                        if (i < rot.shape()[0])
                            t.basis = godot::Basis(godot::Quaternion(rot(i,0), rot(i,1), rot(i,2), rot(i,3)));
                        else
                            t.basis = godot::Basis();
                        ps->body_set_state(m_body_rids[i], godot::PhysicsServer3D::BODY_STATE_TRANSFORM, t);
                        if (i < vel.shape()[0])
                            ps->body_set_state(m_body_rids[i], godot::PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY,
                                               godot::Vector3(vel(i,0), vel(i,1), vel(i,2)));
                        if (i < ang_vel.shape()[0])
                            ps->body_set_state(m_body_rids[i], godot::PhysicsServer3D::BODY_STATE_ANGULAR_VELOCITY,
                                               godot::Vector3(ang_vel(i,0), ang_vel(i,1), ang_vel(i,2)));
                    }
                }

                // Queries
                godot::Ref<XTensorNode> ray_cast_batch(const godot::Ref<XTensorNode>& origins,
                                                       const godot::Ref<XTensorNode>& directions)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!is_inside_tree()) return result;
                    auto world = get_world_3d();
                    if (!world.is_valid()) return result;
                    auto space_state = world->get_direct_space_state();
                    auto orig_arr = origins->get_tensor_resource()->m_data.to_double_array();
                    auto dir_arr = directions->get_tensor_resource()->m_data.to_double_array();
                    size_t n = std::min(orig_arr.shape()[0], dir_arr.shape()[0]);
                    xarray_container<double> hits({n, 5}); // distance, pos.x, pos.y, pos.z, normal packed as angles?
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::Vector3 from(orig_arr(i,0), orig_arr(i,1), orig_arr(i,2));
                        godot::Vector3 dir(dir_arr(i,0), dir_arr(i,1), dir_arr(i,2));
                        godot::PhysicsRayQueryParameters3D params;
                        params.set_from(from);
                        params.set_to(from + dir * 1000.0f);
                        godot::Dictionary hit = space_state->intersect_ray(params);
                        if (hit.is_empty())
                        {
                            hits(i,0) = -1.0; hits(i,1)=0; hits(i,2)=0; hits(i,3)=0; hits(i,4)=0;
                        }
                        else
                        {
                            godot::Vector3 pos = hit["position"];
                            godot::Vector3 norm = hit["normal"];
                            hits(i,0) = hit["distance"];
                            hits(i,1) = pos.x; hits(i,2) = pos.y; hits(i,3) = pos.z;
                            hits(i,4) = std::atan2(norm.y, norm.x);
                        }
                    }
                    result->set_data(XVariant::from_xarray(hits).variant());
                    return result;
                }

                godot::Ref<XTensorNode> overlap_sphere_batch(const godot::Ref<XTensorNode>& centers,
                                                             const godot::Ref<XTensorNode>& radii)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!is_inside_tree()) return result;
                    auto world = get_world_3d();
                    if (!world.is_valid()) return result;
                    auto space_state = world->get_direct_space_state();
                    auto center_arr = centers->get_tensor_resource()->m_data.to_double_array();
                    auto radii_arr = radii->get_tensor_resource()->m_data.to_double_array();
                    size_t n = std::min(center_arr.shape()[0], radii_arr.shape()[0]);
                    xarray_container<double> counts({n});
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::PhysicsShapeQueryParameters3D params;
                        godot::Ref<godot::SphereShape3D> sphere;
                        sphere.instantiate();
                        sphere->set_radius(static_cast<float>(radii_arr(i)));
                        params.set_shape(sphere);
                        params.set_transform(godot::Transform3D(godot::Basis(), godot::Vector3(center_arr(i,0), center_arr(i,1), center_arr(i,2))));
                        godot::Array hits = space_state->intersect_shape(params);
                        counts(i) = hits.size();
                    }
                    result->set_data(XVariant::from_xarray(counts).variant());
                    return result;
                }

                godot::Ref<XTensorNode> collision_pairs()
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!m_bodies_created) return result;
                    size_t n = m_body_rids.size();
                    auto pos = get_positions_array();
                    std::vector<std::pair<int,int>> pairs;
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::Vector3 pi(pos(i,0), pos(i,1), pos(i,2));
                        for (size_t j = i+1; j < n; ++j)
                        {
                            godot::Vector3 pj(pos(j,0), pos(j,1), pos(j,2));
                            if (pi.distance_squared_to(pj) < 4.0f) // hardcoded threshold
                                pairs.emplace_back(i, j);
                        }
                    }
                    if (!pairs.empty())
                    {
                        xarray_container<double> pair_arr({pairs.size(), 2});
                        for (size_t k = 0; k < pairs.size(); ++k)
                        {
                            pair_arr(k,0) = pairs[k].first;
                            pair_arr(k,1) = pairs[k].second;
                        }
                        result->set_data(XVariant::from_xarray(pair_arr).variant());
                    }
                    return result;
                }

                godot::Array compute_contacts()
                {
                    godot::Array contacts;
                    if (!m_bodies_created) return contacts;
                    auto pos = get_positions_array();
                    size_t n = pos.shape()[0];
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::Vector3 pi(pos(i,0), pos(i,1), pos(i,2));
                        for (size_t j = i+1; j < n; ++j)
                        {
                            godot::Vector3 pj(pos(j,0), pos(j,1), pos(j,2));
                            godot::Vector3 delta = pj - pi;
                            float dist = delta.length();
                            float min_dist = 1.0f; // radius sum
                            if (dist < min_dist)
                            {
                                godot::Dictionary contact;
                                contact["body_a"] = static_cast<int64_t>(i);
                                contact["body_b"] = static_cast<int64_t>(j);
                                contact["normal"] = dist > 0.001f ? delta / dist : godot::Vector3(1,0,0);
                                contact["penetration"] = min_dist - dist;
                                contact["position"] = pi + delta * 0.5f;
                                contacts.append(contact);
                                emit_signal("collision_detected", static_cast<int64_t>(i), static_cast<int64_t>(j));
                            }
                        }
                    }
                    return contacts;
                }

                void resolve_collisions(float restitution)
                {
                    auto contacts = compute_contacts();
                    if (contacts.is_empty()) return;
                    auto pos = get_positions_array();
                    auto vel = get_velocities_array();
                    auto mass = get_masses_array();
                    for (int c = 0; c < contacts.size(); ++c)
                    {
                        godot::Dictionary contact = contacts[c];
                        int i = contact["body_a"];
                        int j = contact["body_b"];
                        godot::Vector3 normal = contact["normal"];
                        float penetration = contact["penetration"];
                        float mi = (i < static_cast<int>(mass.size())) ? mass(i) : 1.0f;
                        float mj = (j < static_cast<int>(mass.size())) ? mass(j) : 1.0f;
                        float inv_mi = 1.0f / mi;
                        float inv_mj = 1.0f / mj;
                        godot::Vector3 vi(vel(i,0), vel(i,1), vel(i,2));
                        godot::Vector3 vj(vel(j,0), vel(j,1), vel(j,2));
                        godot::Vector3 pi(pos(i,0), pos(i,1), pos(i,2));
                        godot::Vector3 pj(pos(j,0), pos(j,1), pos(j,2));
                        godot::Vector3 correction = normal * penetration * 0.5f;
                        pos(i,0) = pi.x - correction.x; pos(i,1) = pi.y - correction.y; pos(i,2) = pi.z - correction.z;
                        pos(j,0) = pj.x + correction.x; pos(j,1) = pj.y + correction.y; pos(j,2) = pj.z + correction.z;
                        godot::Vector3 relative_vel = vj - vi;
                        float vel_along_normal = relative_vel.dot(normal);
                        if (vel_along_normal < 0)
                        {
                            float impulse = -(1.0f + restitution) * vel_along_normal / (inv_mi + inv_mj);
                            godot::Vector3 impulse_vec = normal * impulse;
                            vel(i,0) -= impulse_vec.x * inv_mi; vel(i,1) -= impulse_vec.y * inv_mi; vel(i,2) -= impulse_vec.z * inv_mi;
                            vel(j,0) += impulse_vec.x * inv_mj; vel(j,1) += impulse_vec.y * inv_mj; vel(j,2) += impulse_vec.z * inv_mj;
                        }
                    }
                    m_positions_tensor->set_data(XVariant::from_xarray(pos).variant());
                    m_velocities_tensor->set_data(XVariant::from_xarray(vel).variant());
                }

                // Properties
                void set_auto_sync(bool enabled) { m_auto_sync = enabled; }
                bool get_auto_sync() const { return m_auto_sync; }
                void set_gravity(const godot::Vector3& g) { m_gravity = g; }
                godot::Vector3 get_gravity() const { return m_gravity; }
                void set_damping(float linear, float angular) { m_damping_linear = linear; m_damping_angular = angular; }

            private:
                void ensure_tensors()
                {
                    if (!m_positions_tensor.is_valid()) m_positions_tensor.instantiate();
                    if (!m_velocities_tensor.is_valid()) m_velocities_tensor.instantiate();
                    if (!m_rotations_tensor.is_valid()) m_rotations_tensor.instantiate();
                    if (!m_angular_velocities_tensor.is_valid()) m_angular_velocities_tensor.instantiate();
                    if (!m_masses_tensor.is_valid()) m_masses_tensor.instantiate();
                    if (!m_forces_tensor.is_valid()) m_forces_tensor.instantiate();
                    if (!m_torques_tensor.is_valid()) m_torques_tensor.instantiate();
                }

                size_t get_body_count_from_tensors() const
                {
                    if (m_positions_tensor.is_valid())
                        return m_positions_tensor->get_tensor_resource()->m_data.to_double_array().shape()[0];
                    return 0;
                }

                xarray_container<double> get_positions_array() const
                {
                    if (m_positions_tensor.is_valid())
                        return m_positions_tensor->get_tensor_resource()->m_data.to_double_array();
                    return xt::zeros<double>({0, 3});
                }

                xarray_container<double> get_velocities_array() const
                {
                    if (m_velocities_tensor.is_valid())
                        return m_velocities_tensor->get_tensor_resource()->m_data.to_double_array();
                    return xt::zeros<double>({0, 3});
                }

                xarray_container<double> get_rotations_array() const
                {
                    if (m_rotations_tensor.is_valid())
                        return m_rotations_tensor->get_tensor_resource()->m_data.to_double_array();
                    return xt::zeros<double>({0, 4});
                }

                xarray_container<double> get_angular_velocities_array() const
                {
                    if (m_angular_velocities_tensor.is_valid())
                        return m_angular_velocities_tensor->get_tensor_resource()->m_data.to_double_array();
                    return xt::zeros<double>({0, 3});
                }

                xarray_container<double> get_masses_array() const
                {
                    if (m_masses_tensor.is_valid())
                        return m_masses_tensor->get_tensor_resource()->m_data.to_double_array();
                    return xt::zeros<double>({0});
                }

                xarray_container<double> get_forces_array() const
                {
                    if (m_forces_tensor.is_valid())
                        return m_forces_tensor->get_tensor_resource()->m_data.to_double_array();
                    return xt::zeros<double>({0, 3});
                }

                xarray_container<double> get_torques_array() const
                {
                    if (m_torques_tensor.is_valid())
                        return m_torques_tensor->get_tensor_resource()->m_data.to_double_array();
                    return xt::zeros<double>({0, 3});
                }
            };

            // --------------------------------------------------------------------
            // XCollisionWorld3D - Tensor-based collision world
            // --------------------------------------------------------------------
            class XCollisionWorld3D : public godot::Node3D
            {
                GDCLASS(XCollisionWorld3D, godot::Node3D)

            private:
                godot::Ref<XTensorNode> m_vertices_tensor;
                godot::Ref<XTensorNode> m_triangles_tensor;
                godot::Ref<XTensorNode> m_bboxes_tensor;
                bool m_dirty = true;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("build_from_meshes", "meshes"), &XCollisionWorld3D::build_from_meshes);
                    godot::ClassDB::bind_method(godot::D_METHOD("build_from_arrays", "vertices", "triangles"), &XCollisionWorld3D::build_from_arrays);
                    godot::ClassDB::bind_method(godot::D_METHOD("query_ray", "origin", "direction"), &XCollisionWorld3D::query_ray);
                    godot::ClassDB::bind_method(godot::D_METHOD("query_sphere", "center", "radius"), &XCollisionWorld3D::query_sphere);
                    godot::ClassDB::bind_method(godot::D_METHOD("batch_query_rays", "origins", "directions"), &XCollisionWorld3D::batch_query_rays);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_closest_point", "point"), &XCollisionWorld3D::get_closest_point);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_sdf", "point"), &XCollisionWorld3D::get_sdf);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear"), &XCollisionWorld3D::clear);
                }

            public:
                void build_from_meshes(const godot::Array& meshes)
                {
                    // Collect all vertices and triangles
                    std::vector<double> all_verts;
                    std::vector<int64_t> all_tris;
                    int64_t vert_offset = 0;
                    for (int i = 0; i < meshes.size(); ++i)
                    {
                        godot::Ref<godot::Mesh> mesh = meshes[i];
                        if (mesh.is_valid())
                        {
                            godot::Array arrays = mesh->surface_get_arrays(0);
                            if (arrays.size() == 0) continue;
                            godot::PackedVector3Array verts = arrays[godot::Mesh::ARRAY_VERTEX];
                            godot::PackedInt32Array indices = arrays[godot::Mesh::ARRAY_INDEX];
                            if (indices.size() == 0)
                            {
                                for (int j = 0; j < verts.size(); ++j)
                                    indices.append(j);
                            }
                            for (int j = 0; j < verts.size(); ++j)
                            {
                                all_verts.push_back(verts[j].x);
                                all_verts.push_back(verts[j].y);
                                all_verts.push_back(verts[j].z);
                            }
                            for (int j = 0; j < indices.size(); ++j)
                                all_tris.push_back(vert_offset + indices[j]);
                            vert_offset += verts.size();
                        }
                    }
                    if (all_verts.empty()) return;
                    size_t num_verts = all_verts.size() / 3;
                    size_t num_tris = all_tris.size() / 3;
                    xarray_container<double> verts({num_verts, 3});
                    xarray_container<double> tris({num_tris, 3});
                    for (size_t i = 0; i < num_verts; ++i)
                    {
                        verts(i,0) = all_verts[i*3];
                        verts(i,1) = all_verts[i*3+1];
                        verts(i,2) = all_verts[i*3+2];
                    }
                    for (size_t i = 0; i < num_tris; ++i)
                    {
                        tris(i,0) = all_tris[i*3];
                        tris(i,1) = all_tris[i*3+1];
                        tris(i,2) = all_tris[i*3+2];
                    }
                    m_vertices_tensor->set_data(XVariant::from_xarray(verts).variant());
                    m_triangles_tensor->set_data(XVariant::from_xarray(tris).variant());
                    compute_bboxes();
                    m_dirty = false;
                }

                void build_from_arrays(const godot::Ref<XTensorNode>& vertices, const godot::Ref<XTensorNode>& triangles)
                {
                    m_vertices_tensor = vertices;
                    m_triangles_tensor = triangles;
                    compute_bboxes();
                    m_dirty = false;
                }

                godot::Dictionary query_ray(const godot::Vector3& origin, const godot::Vector3& direction) const
                {
                    godot::Dictionary best_hit;
                    if (!m_vertices_tensor.is_valid() || !m_triangles_tensor.is_valid() || !m_bboxes_tensor.is_valid())
                        return best_hit;
                    auto verts = m_vertices_tensor->get_tensor_resource()->m_data.to_double_array();
                    auto tris = m_triangles_tensor->get_tensor_resource()->m_data.to_double_array();
                    auto bboxes = m_bboxes_tensor->get_tensor_resource()->m_data.to_double_array();
                    float best_t = 1e30f;
                    for (size_t i = 0; i < tris.shape()[0]; ++i)
                    {
                        // Ray-AABB test first
                        float tmin, tmax;
                        if (intersect_ray_aabb(origin, direction, i, bboxes, tmin, tmax) && tmin < best_t)
                        {
                            size_t i0 = static_cast<size_t>(tris(i,0));
                            size_t i1 = static_cast<size_t>(tris(i,1));
                            size_t i2 = static_cast<size_t>(tris(i,2));
                            godot::Vector3 v0(verts(i0,0), verts(i0,1), verts(i0,2));
                            godot::Vector3 v1(verts(i1,0), verts(i1,1), verts(i1,2));
                            godot::Vector3 v2(verts(i2,0), verts(i2,1), verts(i2,2));
                            float t, u, v;
                            if (intersect_ray_triangle(origin, direction, v0, v1, v2, t, u, v))
                            {
                                if (t < best_t)
                                {
                                    best_t = t;
                                    best_hit["index"] = static_cast<int64_t>(i);
                                    best_hit["distance"] = t;
                                    best_hit["position"] = origin + direction * t;
                                    best_hit["normal"] = (v1 - v0).cross(v2 - v0).normalized();
                                }
                            }
                        }
                    }
                    return best_hit;
                }

                godot::PackedInt64Array query_sphere(const godot::Vector3& center, float radius) const
                {
                    godot::PackedInt64Array result;
                    if (!m_bboxes_tensor.is_valid()) return result;
                    auto bboxes = m_bboxes_tensor->get_tensor_resource()->m_data.to_double_array();
                    float r2 = radius * radius;
                    for (size_t i = 0; i < bboxes.shape()[0]; ++i)
                    {
                        float dx = std::max(bboxes(i,0) - center.x, std::max(0.0f, center.x - bboxes(i,3)));
                        float dy = std::max(bboxes(i,1) - center.y, std::max(0.0f, center.y - bboxes(i,4)));
                        float dz = std::max(bboxes(i,2) - center.z, std::max(0.0f, center.z - bboxes(i,5)));
                        if (dx*dx + dy*dy + dz*dz <= r2)
                            result.append(static_cast<int64_t>(i));
                    }
                    return result;
                }

                godot::Ref<XTensorNode> batch_query_rays(const godot::Ref<XTensorNode>& origins,
                                                         const godot::Ref<XTensorNode>& directions) const
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!m_vertices_tensor.is_valid()) return result;
                    auto orig_arr = origins->get_tensor_resource()->m_data.to_double_array();
                    auto dir_arr = directions->get_tensor_resource()->m_data.to_double_array();
                    size_t n = std::min(orig_arr.shape()[0], dir_arr.shape()[0]);
                    xarray_container<double> hits({n, 5});
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::Vector3 orig(orig_arr(i,0), orig_arr(i,1), orig_arr(i,2));
                        godot::Vector3 dir(dir_arr(i,0), dir_arr(i,1), dir_arr(i,2));
                        auto hit = query_ray(orig, dir);
                        if (hit.is_empty())
                        {
                            hits(i,0) = -1; hits(i,1)=0; hits(i,2)=0; hits(i,3)=0; hits(i,4)=0;
                        }
                        else
                        {
                            hits(i,0) = hit["distance"];
                            godot::Vector3 pos = hit["position"];
                            godot::Vector3 norm = hit["normal"];
                            hits(i,1) = pos.x; hits(i,2) = pos.y; hits(i,3) = pos.z;
                            hits(i,4) = std::atan2(norm.y, norm.x);
                        }
                    }
                    result->set_data(XVariant::from_xarray(hits).variant());
                    return result;
                }

                godot::Vector3 get_closest_point(const godot::Vector3& point) const
                {
                    if (!m_vertices_tensor.is_valid()) return point;
                    auto verts = m_vertices_tensor->get_tensor_resource()->m_data.to_double_array();
                    auto tris = m_triangles_tensor->get_tensor_resource()->m_data.to_double_array();
                    godot::Vector3 closest = point;
                    float min_dist2 = 1e30f;
                    for (size_t i = 0; i < tris.shape()[0]; ++i)
                    {
                        size_t i0 = static_cast<size_t>(tris(i,0));
                        size_t i1 = static_cast<size_t>(tris(i,1));
                        size_t i2 = static_cast<size_t>(tris(i,2));
                        godot::Vector3 v0(verts(i0,0), verts(i0,1), verts(i0,2));
                        godot::Vector3 v1(verts(i1,0), verts(i1,1), verts(i1,2));
                        godot::Vector3 v2(verts(i2,0), verts(i2,1), verts(i2,2));
                        godot::Vector3 cp = closest_point_on_triangle(point, v0, v1, v2);
                        float d2 = point.distance_squared_to(cp);
                        if (d2 < min_dist2) { min_dist2 = d2; closest = cp; }
                    }
                    return closest;
                }

                float get_sdf(const godot::Vector3& point) const
                {
                    godot::Vector3 cp = get_closest_point(point);
                    float dist = point.distance_to(cp);
                    // Simplified: always return positive (outside)
                    return dist;
                }

                void clear()
                {
                    m_vertices_tensor.unref();
                    m_triangles_tensor.unref();
                    m_bboxes_tensor.unref();
                    m_dirty = true;
                }

            private:
                void compute_bboxes()
                {
                    if (!m_vertices_tensor.is_valid() || !m_triangles_tensor.is_valid()) return;
                    auto verts = m_vertices_tensor->get_tensor_resource()->m_data.to_double_array();
                    auto tris = m_triangles_tensor->get_tensor_resource()->m_data.to_double_array();
                    size_t num_tris = tris.shape()[0];
                    xarray_container<double> bboxes({num_tris, 6});
                    for (size_t i = 0; i < num_tris; ++i)
                    {
                        size_t i0 = static_cast<size_t>(tris(i,0));
                        size_t i1 = static_cast<size_t>(tris(i,1));
                        size_t i2 = static_cast<size_t>(tris(i,2));
                        float min_x = std::min({verts(i0,0), verts(i1,0), verts(i2,0)});
                        float min_y = std::min({verts(i0,1), verts(i1,1), verts(i2,1)});
                        float min_z = std::min({verts(i0,2), verts(i1,2), verts(i2,2)});
                        float max_x = std::max({verts(i0,0), verts(i1,0), verts(i2,0)});
                        float max_y = std::max({verts(i0,1), verts(i1,1), verts(i2,1)});
                        float max_z = std::max({verts(i0,2), verts(i1,2), verts(i2,2)});
                        bboxes(i,0) = min_x; bboxes(i,1) = min_y; bboxes(i,2) = min_z;
                        bboxes(i,3) = max_x; bboxes(i,4) = max_y; bboxes(i,5) = max_z;
                    }
                    if (!m_bboxes_tensor.is_valid()) m_bboxes_tensor.instantiate();
                    m_bboxes_tensor->set_data(XVariant::from_xarray(bboxes).variant());
                }

                bool intersect_ray_aabb(const godot::Vector3& origin, const godot::Vector3& dir,
                                        size_t tri_idx, const xarray_container<double>& bboxes,
                                        float& tmin, float& tmax) const
                {
                    float min_x = bboxes(tri_idx,0), min_y = bboxes(tri_idx,1), min_z = bboxes(tri_idx,2);
                    float max_x = bboxes(tri_idx,3), max_y = bboxes(tri_idx,4), max_z = bboxes(tri_idx,5);
                    float t1 = (min_x - origin.x) / dir.x;
                    float t2 = (max_x - origin.x) / dir.x;
                    float t3 = (min_y - origin.y) / dir.y;
                    float t4 = (max_y - origin.y) / dir.y;
                    float t5 = (min_z - origin.z) / dir.z;
                    float t6 = (max_z - origin.z) / dir.z;
                    float tmin_x = std::min(t1, t2), tmax_x = std::max(t1, t2);
                    float tmin_y = std::min(t3, t4), tmax_y = std::max(t3, t4);
                    float tmin_z = std::min(t5, t6), tmax_z = std::max(t5, t6);
                    tmin = std::max({tmin_x, tmin_y, tmin_z});
                    tmax = std::min({tmax_x, tmax_y, tmax_z});
                    return tmax >= 0 && tmin <= tmax;
                }

                bool intersect_ray_triangle(const godot::Vector3& orig, const godot::Vector3& dir,
                                            const godot::Vector3& v0, const godot::Vector3& v1, const godot::Vector3& v2,
                                            float& t, float& u, float& v) const
                {
                    godot::Vector3 e1 = v1 - v0;
                    godot::Vector3 e2 = v2 - v0;
                    godot::Vector3 h = dir.cross(e2);
                    float a = e1.dot(h);
                    if (std::abs(a) < 1e-6f) return false;
                    float f = 1.0f / a;
                    godot::Vector3 s = orig - v0;
                    u = f * s.dot(h);
                    if (u < 0.0f || u > 1.0f) return false;
                    godot::Vector3 q = s.cross(e1);
                    v = f * dir.dot(q);
                    if (v < 0.0f || u + v > 1.0f) return false;
                    t = f * e2.dot(q);
                    return t >= 0.0f;
                }

                godot::Vector3 closest_point_on_triangle(const godot::Vector3& p,
                                                         const godot::Vector3& a,
                                                         const godot::Vector3& b,
                                                         const godot::Vector3& c) const
                {
                    godot::Vector3 ab = b - a, ac = c - a, ap = p - a;
                    float d1 = ab.dot(ap), d2 = ac.dot(ap);
                    if (d1 <= 0 && d2 <= 0) return a;
                    godot::Vector3 bp = p - b;
                    float d3 = ab.dot(bp), d4 = ac.dot(bp);
                    if (d3 >= 0 && d4 <= d3) return b;
                    float vc = d1*d4 - d3*d2;
                    if (vc <= 0 && d1 >= 0 && d3 <= 0)
                    {
                        float v_frac = d1 / (d1 - d3);
                        return a + ab * v_frac;
                    }
                    godot::Vector3 cp = p - c;
                    float d5 = ab.dot(cp), d6 = ac.dot(cp);
                    if (d6 >= 0 && d5 <= d6) return c;
                    float vb = d5*d2 - d1*d6;
                    if (vb <= 0 && d2 >= 0 && d6 <= 0)
                    {
                        float w = d2 / (d2 - d6);
                        return a + ac * w;
                    }
                    float va = d3*d6 - d5*d4;
                    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
                    {
                        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                        return b + (c - b) * w;
                    }
                    float denom = 1.0f / (va + vb + vc);
                    float v_frac = vb * denom;
                    float w = vc * denom;
                    return a + ab * v_frac + ac * w;
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XPhysics3DRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XPhysicsTensor3D>();
                    godot::ClassDB::register_class<XCollisionWorld3D>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::XPhysicsTensor3D;
        using godot_bridge::XCollisionWorld3D;
        using godot_bridge::XPhysics3DRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XPHYSICS3D_HPP

// godot/xphysics3d.hpp