// godot/xphysics2d.hpp

#ifndef XTENSOR_XPHYSICS2D_HPP
#define XTENSOR_XPHYSICS2D_HPP

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
    #include <godot_cpp/classes/node2d.hpp>
    #include <godot_cpp/classes/physics_body2d.hpp>
    #include <godot_cpp/classes/rigid_body2d.hpp>
    #include <godot_cpp/classes/static_body2d.hpp>
    #include <godot_cpp/classes/character_body2d.hpp>
    #include <godot_cpp/classes/area2d.hpp>
    #include <godot_cpp/classes/collision_shape2d.hpp>
    #include <godot_cpp/classes/collision_polygon2d.hpp>
    #include <godot_cpp/classes/physics_direct_space_state2d.hpp>
    #include <godot_cpp/classes/physics_ray_query_parameters2d.hpp>
    #include <godot_cpp/classes/physics_shape_query_parameters2d.hpp>
    #include <godot_cpp/classes/physics_point_query_parameters2d.hpp>
    #include <godot_cpp/classes/world2d.hpp>
    #include <godot_cpp/classes/physics_material.hpp>
    #include <godot_cpp/classes/rectangle_shape2d.hpp>
    #include <godot_cpp/classes/circle_shape2d.hpp>
    #include <godot_cpp/classes/capsule_shape2d.hpp>
    #include <godot_cpp/classes/concave_polygon_shape2d.hpp>
    #include <godot_cpp/classes/convex_polygon_shape2d.hpp>
    #include <godot_cpp/classes/segment_shape2d.hpp>
    #include <godot_cpp/classes/separation_ray_shape2d.hpp>
    #include <godot_cpp/classes/world_boundary_shape2d.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/transform2d.hpp>
    #include <godot_cpp/variant/vector2.hpp>
    #include <godot_cpp/variant/rect2.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // XPhysicsTensor2D - Batch physics operations using tensors
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XPhysicsTensor2D : public godot::Node2D
            {
                GDCLASS(XPhysicsTensor2D, godot::Node2D)

            private:
                godot::Ref<XTensorNode> m_positions_tensor;      // Nx2: body positions
                godot::Ref<XTensorNode> m_velocities_tensor;     // Nx2: linear velocities
                godot::Ref<XTensorNode> m_rotations_tensor;      // N: rotations (radians)
                godot::Ref<XTensorNode> m_angular_velocities_tensor; // N: angular velocities
                godot::Ref<XTensorNode> m_masses_tensor;         // N: masses
                godot::Ref<XTensorNode> m_forces_tensor;         // Nx2: accumulated forces
                godot::Ref<XTensorNode> m_torques_tensor;        // N: accumulated torques
                
                std::vector<godot::RID> m_body_rids;
                std::vector<godot::Ref<godot::Shape2D>> m_shapes;
                std::vector<godot::Transform2D> m_shape_transforms;
                
                godot::Ref<godot::PhysicsMaterial> m_physics_material;
                bool m_bodies_created = false;
                bool m_auto_sync = true;
                float m_time_step = 1.0f / 60.0f;
                int m_iterations = 8;

            protected:
                static void _bind_methods()
                {
                    // Tensor access
                    godot::ClassDB::bind_method(godot::D_METHOD("set_positions_tensor", "tensor"), &XPhysicsTensor2D::set_positions_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_positions_tensor"), &XPhysicsTensor2D::get_positions_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_velocities_tensor", "tensor"), &XPhysicsTensor2D::set_velocities_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_velocities_tensor"), &XPhysicsTensor2D::get_velocities_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_masses_tensor", "tensor"), &XPhysicsTensor2D::set_masses_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_masses_tensor"), &XPhysicsTensor2D::get_masses_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_forces_tensor", "tensor"), &XPhysicsTensor2D::set_forces_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_forces_tensor"), &XPhysicsTensor2D::get_forces_tensor);
                    
                    // Body management
                    godot::ClassDB::bind_method(godot::D_METHOD("create_bodies", "shape_type", "shape_params"), &XPhysicsTensor2D::create_bodies);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear_bodies"), &XPhysicsTensor2D::clear_bodies);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_body_count"), &XPhysicsTensor2D::get_body_count);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_body_positions", "positions"), &XPhysicsTensor2D::set_body_positions);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_body_positions"), &XPhysicsTensor2D::get_body_positions);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_body_velocities", "velocities"), &XPhysicsTensor2D::set_body_velocities);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_body_velocities"), &XPhysicsTensor2D::get_body_velocities);
                    godot::ClassDB::bind_method(godot::D_METHOD("apply_forces", "forces"), &XPhysicsTensor2D::apply_forces);
                    godot::ClassDB::bind_method(godot::D_METHOD("apply_impulses", "impulses"), &XPhysicsTensor2D::apply_impulses);
                    
                    // Simulation
                    godot::ClassDB::bind_method(godot::D_METHOD("step", "delta"), &XPhysicsTensor2D::step);
                    godot::ClassDB::bind_method(godot::D_METHOD("simulate", "duration", "sub_steps"), &XPhysicsTensor2D::simulate);
                    godot::ClassDB::bind_method(godot::D_METHOD("sync_from_bodies"), &XPhysicsTensor2D::sync_from_bodies);
                    godot::ClassDB::bind_method(godot::D_METHOD("sync_to_bodies"), &XPhysicsTensor2D::sync_to_bodies);
                    
                    // Queries
                    godot::ClassDB::bind_method(godot::D_METHOD("ray_cast_batch", "origins", "directions"), &XPhysicsTensor2D::ray_cast_batch);
                    godot::ClassDB::bind_method(godot::D_METHOD("overlap_circle_batch", "centers", "radii"), &XPhysicsTensor2D::overlap_circle_batch);
                    godot::ClassDB::bind_method(godot::D_METHOD("collision_pairs"), &XPhysicsTensor2D::collision_pairs);
                    
                    // Collision detection (tensorized)
                    godot::ClassDB::bind_method(godot::D_METHOD("compute_contacts"), &XPhysicsTensor2D::compute_contacts);
                    godot::ClassDB::bind_method(godot::D_METHOD("resolve_collisions", "restitution"), &XPhysicsTensor2D::resolve_collisions, godot::DEFVAL(0.5));
                    
                    // Properties
                    godot::ClassDB::bind_method(godot::D_METHOD("set_auto_sync", "enabled"), &XPhysicsTensor2D::set_auto_sync);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_auto_sync"), &XPhysicsTensor2D::get_auto_sync);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_physics_material", "material"), &XPhysicsTensor2D::set_physics_material);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_physics_material"), &XPhysicsTensor2D::get_physics_material);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "positions_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_positions_tensor", "get_positions_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "velocities_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_velocities_tensor", "get_velocities_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "masses_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_masses_tensor", "get_masses_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "auto_sync"), "set_auto_sync", "get_auto_sync");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "physics_material", godot::PROPERTY_HINT_RESOURCE_TYPE, "PhysicsMaterial"), "set_physics_material", "get_physics_material");
                    
                    // Signals
                    ADD_SIGNAL(godot::MethodInfo("bodies_created", godot::PropertyInfo(godot::Variant::INT, "count")));
                    ADD_SIGNAL(godot::MethodInfo("collision_detected", godot::PropertyInfo(godot::Variant::INT, "body_a"), godot::PropertyInfo(godot::Variant::INT, "body_b")));
                }

            public:
                XPhysicsTensor2D()
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
                    {
                        step(static_cast<float>(delta));
                    }
                }

                // Tensor access
                void set_positions_tensor(const godot::Ref<XTensorNode>& tensor) { m_positions_tensor = tensor; }
                godot::Ref<XTensorNode> get_positions_tensor() const { return m_positions_tensor; }
                void set_velocities_tensor(const godot::Ref<XTensorNode>& tensor) { m_velocities_tensor = tensor; }
                godot::Ref<XTensorNode> get_velocities_tensor() const { return m_velocities_tensor; }
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
                    
                    godot::PhysicsServer2D* ps = godot::PhysicsServer2D::get_singleton();
                    godot::RID space = get_world_2d()->get_space();
                    
                    // Create shape
                    godot::Ref<godot::Shape2D> base_shape;
                    std::string type = shape_type.utf8().get_data();
                    if (type == "circle")
                    {
                        godot::Ref<godot::CircleShape2D> circle;
                        circle.instantiate();
                        float radius = shape_params.get_type() == godot::Variant::FLOAT ? 
                            static_cast<float>(shape_params) : 10.0f;
                        circle->set_radius(radius);
                        base_shape = circle;
                    }
                    else if (type == "rectangle")
                    {
                        godot::Ref<godot::RectangleShape2D> rect;
                        rect.instantiate();
                        if (shape_params.get_type() == godot::Variant::VECTOR2)
                            rect->set_size(godot::Vector2(shape_params));
                        else
                            rect->set_size(godot::Vector2(10, 10));
                        base_shape = rect;
                    }
                    else if (type == "capsule")
                    {
                        godot::Ref<godot::CapsuleShape2D> capsule;
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
                    else
                    {
                        godot::Ref<godot::CircleShape2D> circle;
                        circle.instantiate();
                        circle->set_radius(10.0f);
                        base_shape = circle;
                    }
                    
                    auto positions = get_positions_array();
                    auto masses = get_masses_array();
                    
                    m_body_rids.resize(n);
                    m_shapes.resize(n, base_shape);
                    m_shape_transforms.resize(n, godot::Transform2D());
                    
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::RID body = ps->body_create();
                        ps->body_set_mode(body, godot::PhysicsServer2D::BODY_MODE_RIGID);
                        ps->body_set_space(body, space);
                        
                        godot::Vector2 pos(positions(i, 0), positions(i, 1));
                        godot::Transform2D transform(0.0f, pos);
                        ps->body_set_state(body, godot::PhysicsServer2D::BODY_STATE_TRANSFORM, transform);
                        
                        float mass = (i < masses.size()) ? masses(i) : 1.0f;
                        ps->body_set_param(body, godot::PhysicsServer2D::BODY_PARAM_MASS, mass);
                        
                        // Add shape
                        godot::RID shape_rid = base_shape->get_rid();
                        ps->body_add_shape(body, shape_rid, godot::Transform2D());
                        
                        m_body_rids[i] = body;
                    }
                    
                    m_bodies_created = true;
                    emit_signal("bodies_created", static_cast<int64_t>(n));
                }

                void clear_bodies()
                {
                    godot::PhysicsServer2D* ps = godot::PhysicsServer2D::get_singleton();
                    for (const auto& rid : m_body_rids)
                    {
                        if (rid != godot::RID())
                            ps->free_rid(rid);
                    }
                    m_body_rids.clear();
                    m_bodies_created = false;
                }

                int64_t get_body_count() const
                {
                    return static_cast<int64_t>(m_body_rids.size());
                }

                void set_body_positions(const godot::Ref<XTensorNode>& positions)
                {
                    m_positions_tensor = positions;
                    if (m_auto_sync && m_bodies_created)
                        sync_to_bodies();
                }

                godot::Ref<XTensorNode> get_body_positions() const
                {
                    return m_positions_tensor;
                }

                void set_body_velocities(const godot::Ref<XTensorNode>& velocities)
                {
                    m_velocities_tensor = velocities;
                    if (m_auto_sync && m_bodies_created)
                        sync_to_bodies();
                }

                godot::Ref<XTensorNode> get_body_velocities() const
                {
                    return m_velocities_tensor;
                }

                void apply_forces(const godot::Ref<XTensorNode>& forces)
                {
                    if (!m_bodies_created) return;
                    godot::PhysicsServer2D* ps = godot::PhysicsServer2D::get_singleton();
                    auto forces_arr = forces->get_tensor_resource()->m_data.to_double_array();
                    for (size_t i = 0; i < m_body_rids.size() && i < forces_arr.shape()[0]; ++i)
                    {
                        godot::Vector2 force(forces_arr(i, 0), forces_arr(i, 1));
                        ps->body_apply_central_force(m_body_rids[i], force);
                    }
                }

                void apply_impulses(const godot::Ref<XTensorNode>& impulses)
                {
                    if (!m_bodies_created) return;
                    godot::PhysicsServer2D* ps = godot::PhysicsServer2D::get_singleton();
                    auto imp_arr = impulses->get_tensor_resource()->m_data.to_double_array();
                    for (size_t i = 0; i < m_body_rids.size() && i < imp_arr.shape()[0]; ++i)
                    {
                        godot::Vector2 impulse(imp_arr(i, 0), imp_arr(i, 1));
                        ps->body_apply_central_impulse(m_body_rids[i], impulse);
                    }
                }

                // Simulation
                void step(float delta)
                {
                    if (!m_bodies_created) return;
                    godot::PhysicsServer2D* ps = godot::PhysicsServer2D::get_singleton();
                    
                    // Sync forces and velocities to physics server
                    auto forces = get_forces_array();
                    auto velocities = get_velocities_array();
                    for (size_t i = 0; i < m_body_rids.size(); ++i)
                    {
                        if (i < forces.shape()[0])
                            ps->body_apply_central_force(m_body_rids[i], godot::Vector2(forces(i, 0), forces(i, 1)));
                        if (i < velocities.shape()[0])
                            ps->body_set_state(m_body_rids[i], godot::PhysicsServer2D::BODY_STATE_LINEAR_VELOCITY,
                                               godot::Vector2(velocities(i, 0), velocities(i, 1)));
                    }
                    
                    // Custom tensorized physics step (simplified Verlet)
                    if (!m_auto_sync)
                    {
                        // Run our own simplified physics
                        auto pos = get_positions_array();
                        auto vel = get_velocities_array();
                        auto mass = get_masses_array();
                        auto f = get_forces_array();
                        
                        for (size_t i = 0; i < pos.shape()[0]; ++i)
                        {
                            float m = (i < mass.size()) ? mass(i) : 1.0f;
                            godot::Vector2 acceleration = (i < f.shape()[0]) ? 
                                godot::Vector2(f(i, 0) / m, f(i, 1) / m) : godot::Vector2();
                            
                            vel(i, 0) += acceleration.x * delta;
                            vel(i, 1) += acceleration.y * delta;
                            pos(i, 0) += vel(i, 0) * delta;
                            pos(i, 1) += vel(i, 1) * delta;
                        }
                        
                        m_positions_tensor->set_data(XVariant::from_xarray(pos).variant());
                        m_velocities_tensor->set_data(XVariant::from_xarray(vel).variant());
                    }
                    else
                    {
                        // Let Godot physics handle it
                        for (const auto& rid : m_body_rids)
                        {
                            // Physics server processes automatically
                        }
                        sync_from_bodies();
                    }
                }

                void simulate(float duration, int sub_steps)
                {
                    float dt = duration / static_cast<float>(sub_steps);
                    for (int i = 0; i < sub_steps; ++i)
                        step(dt);
                }

                void sync_from_bodies()
                {
                    if (!m_bodies_created) return;
                    godot::PhysicsServer2D* ps = godot::PhysicsServer2D::get_singleton();
                    size_t n = m_body_rids.size();
                    
                    xarray_container<double> pos({n, 2});
                    xarray_container<double> vel({n, 2});
                    
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::Transform2D t = ps->body_get_state(m_body_rids[i], godot::PhysicsServer2D::BODY_STATE_TRANSFORM);
                        godot::Vector2 v = ps->body_get_state(m_body_rids[i], godot::PhysicsServer2D::BODY_STATE_LINEAR_VELOCITY);
                        pos(i, 0) = t.get_origin().x;
                        pos(i, 1) = t.get_origin().y;
                        vel(i, 0) = v.x;
                        vel(i, 1) = v.y;
                    }
                    
                    m_positions_tensor->set_data(XVariant::from_xarray(pos).variant());
                    m_velocities_tensor->set_data(XVariant::from_xarray(vel).variant());
                }

                void sync_to_bodies()
                {
                    if (!m_bodies_created) return;
                    godot::PhysicsServer2D* ps = godot::PhysicsServer2D::get_singleton();
                    auto pos = get_positions_array();
                    auto vel = get_velocities_array();
                    
                    for (size_t i = 0; i < m_body_rids.size(); ++i)
                    {
                        godot::Transform2D t = ps->body_get_state(m_body_rids[i], godot::PhysicsServer2D::BODY_STATE_TRANSFORM);
                        t.set_origin(godot::Vector2(pos(i, 0), pos(i, 1)));
                        ps->body_set_state(m_body_rids[i], godot::PhysicsServer2D::BODY_STATE_TRANSFORM, t);
                        if (i < vel.shape()[0])
                            ps->body_set_state(m_body_rids[i], godot::PhysicsServer2D::BODY_STATE_LINEAR_VELOCITY,
                                               godot::Vector2(vel(i, 0), vel(i, 1)));
                    }
                }

                // Queries
                godot::Ref<XTensorNode> ray_cast_batch(const godot::Ref<XTensorNode>& origins,
                                                       const godot::Ref<XTensorNode>& directions)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!is_inside_tree()) return result;
                    
                    auto world = get_world_2d();
                    if (!world.is_valid()) return result;
                    
                    auto space_state = world->get_direct_space_state();
                    auto orig_arr = origins->get_tensor_resource()->m_data.to_double_array();
                    auto dir_arr = directions->get_tensor_resource()->m_data.to_double_array();
                    size_t n = std::min(orig_arr.shape()[0], dir_arr.shape()[0]);
                    
                    xarray_container<double> hits({n, 4}); // hit distance, position.x, position.y, normal.x, normal.y? actually 4 values
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::Vector2 from(orig_arr(i, 0), orig_arr(i, 1));
                        godot::Vector2 to = from + godot::Vector2(dir_arr(i, 0), dir_arr(i, 1)) * 1000.0f;
                        
                        godot::PhysicsRayQueryParameters2D params;
                        params.set_from(from);
                        params.set_to(to);
                        godot::Dictionary hit = space_state->intersect_ray(params);
                        
                        if (hit.is_empty())
                        {
                            hits(i, 0) = -1.0;
                            hits(i, 1) = 0.0;
                            hits(i, 2) = 0.0;
                            hits(i, 3) = 0.0;
                        }
                        else
                        {
                            godot::Vector2 pos = hit["position"];
                            godot::Vector2 norm = hit["normal"];
                            hits(i, 0) = static_cast<double>(hit["distance"]);
                            hits(i, 1) = pos.x;
                            hits(i, 2) = pos.y;
                            hits(i, 3) = std::atan2(norm.y, norm.x);
                        }
                    }
                    
                    result->set_data(XVariant::from_xarray(hits).variant());
                    return result;
                }

                godot::Ref<XTensorNode> overlap_circle_batch(const godot::Ref<XTensorNode>& centers,
                                                             const godot::Ref<XTensorNode>& radii)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!is_inside_tree()) return result;
                    
                    auto world = get_world_2d();
                    if (!world.is_valid()) return result;
                    
                    auto space_state = world->get_direct_space_state();
                    auto center_arr = centers->get_tensor_resource()->m_data.to_double_array();
                    auto radii_arr = radii->get_tensor_resource()->m_data.to_double_array();
                    size_t n = std::min(center_arr.shape()[0], radii_arr.shape()[0]);
                    
                    xarray_container<double> counts({n});
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::PhysicsShapeQueryParameters2D params;
                        godot::Ref<godot::CircleShape2D> circle;
                        circle.instantiate();
                        circle->set_radius(static_cast<float>(radii_arr(i)));
                        params.set_shape(circle);
                        params.set_transform(godot::Transform2D(0.0f, godot::Vector2(center_arr(i, 0), center_arr(i, 1))));
                        godot::Array hits = space_state->intersect_shape(params);
                        counts(i) = static_cast<double>(hits.size());
                    }
                    
                    result->set_data(XVariant::from_xarray(counts).variant());
                    return result;
                }

                godot::Ref<XTensorNode> collision_pairs()
                {
                    // Return pairs of bodies that are colliding (Nx2 indices)
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!m_bodies_created) return result;
                    
                    godot::PhysicsServer2D* ps = godot::PhysicsServer2D::get_singleton();
                    size_t n = m_body_rids.size();
                    std::vector<std::pair<int, int>> pairs;
                    
                    // Brute-force broad phase
                    auto pos = get_positions_array();
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::Vector2 pi(pos(i, 0), pos(i, 1));
                        for (size_t j = i + 1; j < n; ++j)
                        {
                            godot::Vector2 pj(pos(j, 0), pos(j, 1));
                            float dist2 = pi.distance_squared_to(pj);
                            // Assume some radius for broad phase (could get from shapes)
                            if (dist2 < 100.0f) // hardcoded threshold
                                pairs.emplace_back(static_cast<int>(i), static_cast<int>(j));
                        }
                    }
                    
                    if (!pairs.empty())
                    {
                        xarray_container<double> pair_arr({pairs.size(), 2});
                        for (size_t k = 0; k < pairs.size(); ++k)
                        {
                            pair_arr(k, 0) = pairs[k].first;
                            pair_arr(k, 1) = pairs[k].second;
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
                    auto vel = get_velocities_array();
                    size_t n = pos.shape()[0];
                    
                    // Simple circle-circle collision detection (assuming all bodies are circles of radius 10)
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::Vector2 pi(pos(i, 0), pos(i, 1));
                        for (size_t j = i + 1; j < n; ++j)
                        {
                            godot::Vector2 pj(pos(j, 0), pos(j, 1));
                            godot::Vector2 delta = pj - pi;
                            float dist = delta.length();
                            float min_dist = 20.0f; // radius sum (10+10)
                            if (dist < min_dist)
                            {
                                godot::Dictionary contact;
                                contact["body_a"] = static_cast<int64_t>(i);
                                contact["body_b"] = static_cast<int64_t>(j);
                                contact["normal"] = dist > 0.001f ? delta / dist : godot::Vector2(1, 0);
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
                        int i = static_cast<int>(contact["body_a"]);
                        int j = static_cast<int>(contact["body_b"]);
                        godot::Vector2 normal = contact["normal"];
                        float penetration = contact["penetration"];
                        
                        float mi = (i < static_cast<int>(mass.size())) ? mass(i) : 1.0f;
                        float mj = (j < static_cast<int>(mass.size())) ? mass(j) : 1.0f;
                        float inv_mi = 1.0f / mi;
                        float inv_mj = 1.0f / mj;
                        
                        godot::Vector2 vi(vel(i, 0), vel(i, 1));
                        godot::Vector2 vj(vel(j, 0), vel(j, 1));
                        godot::Vector2 pi(pos(i, 0), pos(i, 1));
                        godot::Vector2 pj(pos(j, 0), pos(j, 1));
                        
                        // Positional correction
                        godot::Vector2 correction = normal * penetration * 0.5f;
                        pos(i, 0) = pi.x - correction.x;
                        pos(i, 1) = pi.y - correction.y;
                        pos(j, 0) = pj.x + correction.x;
                        pos(j, 1) = pj.y + correction.y;
                        
                        // Velocity resolution
                        godot::Vector2 relative_vel = vj - vi;
                        float vel_along_normal = relative_vel.dot(normal);
                        if (vel_along_normal < 0)
                        {
                            float e = restitution;
                            float impulse = -(1.0f + e) * vel_along_normal / (inv_mi + inv_mj);
                            godot::Vector2 impulse_vec = normal * impulse;
                            vel(i, 0) -= impulse_vec.x * inv_mi;
                            vel(i, 1) -= impulse_vec.y * inv_mi;
                            vel(j, 0) += impulse_vec.x * inv_mj;
                            vel(j, 1) += impulse_vec.y * inv_mj;
                        }
                    }
                    
                    m_positions_tensor->set_data(XVariant::from_xarray(pos).variant());
                    m_velocities_tensor->set_data(XVariant::from_xarray(vel).variant());
                }

                // Properties
                void set_auto_sync(bool enabled) { m_auto_sync = enabled; }
                bool get_auto_sync() const { return m_auto_sync; }
                void set_physics_material(const godot::Ref<godot::PhysicsMaterial>& mat) { m_physics_material = mat; }
                godot::Ref<godot::PhysicsMaterial> get_physics_material() const { return m_physics_material; }

            private:
                void ensure_tensors()
                {
                    if (!m_positions_tensor.is_valid())
                        m_positions_tensor.instantiate();
                    if (!m_velocities_tensor.is_valid())
                        m_velocities_tensor.instantiate();
                    if (!m_masses_tensor.is_valid())
                        m_masses_tensor.instantiate();
                    if (!m_forces_tensor.is_valid())
                        m_forces_tensor.instantiate();
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
                    return xt::zeros<double>({0, 2});
                }

                xarray_container<double> get_velocities_array() const
                {
                    if (m_velocities_tensor.is_valid())
                        return m_velocities_tensor->get_tensor_resource()->m_data.to_double_array();
                    return xt::zeros<double>({0, 2});
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
                    return xt::zeros<double>({0, 2});
                }
            };

            // --------------------------------------------------------------------
            // XCollisionWorld2D - Tensor-based collision world
            // --------------------------------------------------------------------
            class XCollisionWorld2D : public godot::Node2D
            {
                GDCLASS(XCollisionWorld2D, godot::Node2D)

            private:
                xarray_container<double> m_vertices;      // Vx2
                xarray_container<double> m_bboxes;        // Vx4 (min_x, min_y, max_x, max_y)
                std::vector<std::vector<size_t>> m_bvh;   // simple BVH
                bool m_dirty = true;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("build_from_shapes", "shapes"), &XCollisionWorld2D::build_from_shapes);
                    godot::ClassDB::bind_method(godot::D_METHOD("build_from_polygons", "polygons"), &XCollisionWorld2D::build_from_polygons);
                    godot::ClassDB::bind_method(godot::D_METHOD("query_point", "point"), &XCollisionWorld2D::query_point);
                    godot::ClassDB::bind_method(godot::D_METHOD("query_ray", "origin", "direction"), &XCollisionWorld2D::query_ray);
                    godot::ClassDB::bind_method(godot::D_METHOD("query_circle", "center", "radius"), &XCollisionWorld2D::query_circle);
                    godot::ClassDB::bind_method(godot::D_METHOD("batch_query_points", "points"), &XCollisionWorld2D::batch_query_points);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_closest_point", "point"), &XCollisionWorld2D::get_closest_point);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_sdf", "point"), &XCollisionWorld2D::get_sdf);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear"), &XCollisionWorld2D::clear);
                }

            public:
                void build_from_shapes(const godot::Array& shapes)
                {
                    clear();
                    std::vector<godot::PackedVector2Array> polygons;
                    for (int i = 0; i < shapes.size(); ++i)
                    {
                        godot::Ref<godot::Shape2D> shape = shapes[i];
                        if (shape.is_valid())
                        {
                            // Convert shape to polygon (approximate circles)
                            if (shape->is_class("CircleShape2D"))
                            {
                                godot::Ref<godot::CircleShape2D> circle = shape;
                                float r = circle->get_radius();
                                const int segs = 16;
                                godot::PackedVector2Array poly;
                                for (int j = 0; j < segs; ++j)
                                {
                                    float angle = 2.0f * Math_PI * j / segs;
                                    poly.append(godot::Vector2(std::cos(angle) * r, std::sin(angle) * r));
                                }
                                polygons.push_back(poly);
                            }
                            else if (shape->is_class("RectangleShape2D"))
                            {
                                godot::Ref<godot::RectangleShape2D> rect = shape;
                                godot::Vector2 ext = rect->get_size() * 0.5f;
                                godot::PackedVector2Array poly;
                                poly.append(godot::Vector2(-ext.x, -ext.y));
                                poly.append(godot::Vector2( ext.x, -ext.y));
                                poly.append(godot::Vector2( ext.x,  ext.y));
                                poly.append(godot::Vector2(-ext.x,  ext.y));
                                polygons.push_back(poly);
                            }
                            else if (shape->is_class("ConvexPolygonShape2D"))
                            {
                                godot::Ref<godot::ConvexPolygonShape2D> convex = shape;
                                polygons.push_back(convex->get_points());
                            }
                        }
                    }
                    build_from_polygons(polygons);
                }

                void build_from_polygons(const godot::Array& polygons_array)
                {
                    clear();
                    size_t total_verts = 0;
                    for (int i = 0; i < polygons_array.size(); ++i)
                        total_verts += static_cast<size_t>(godot::PackedVector2Array(polygons_array[i]).size());
                    
                    m_vertices = xt::zeros<double>({total_verts, 2});
                    m_bboxes = xt::zeros<double>({static_cast<size_t>(polygons_array.size()), 4});
                    
                    size_t vert_offset = 0;
                    for (int i = 0; i < polygons_array.size(); ++i)
                    {
                        godot::PackedVector2Array poly = polygons_array[i];
                        float min_x = 1e30f, min_y = 1e30f, max_x = -1e30f, max_y = -1e30f;
                        for (int j = 0; j < poly.size(); ++j)
                        {
                            godot::Vector2 v = poly[j];
                            m_vertices(vert_offset + j, 0) = v.x;
                            m_vertices(vert_offset + j, 1) = v.y;
                            min_x = std::min(min_x, v.x);
                            min_y = std::min(min_y, v.y);
                            max_x = std::max(max_x, v.x);
                            max_y = std::max(max_y, v.y);
                        }
                        m_bboxes(i, 0) = min_x;
                        m_bboxes(i, 1) = min_y;
                        m_bboxes(i, 2) = max_x;
                        m_bboxes(i, 3) = max_y;
                        vert_offset += static_cast<size_t>(poly.size());
                    }
                    m_dirty = false;
                }

                godot::PackedInt64Array query_point(const godot::Vector2& point) const
                {
                    godot::PackedInt64Array result;
                    for (size_t i = 0; i < m_bboxes.shape()[0]; ++i)
                    {
                        if (point.x >= m_bboxes(i, 0) && point.x <= m_bboxes(i, 2) &&
                            point.y >= m_bboxes(i, 1) && point.y <= m_bboxes(i, 3))
                        {
                            if (point_in_polygon(point, i))
                                result.append(static_cast<int64_t>(i));
                        }
                    }
                    return result;
                }

                godot::Dictionary query_ray(const godot::Vector2& origin, const godot::Vector2& direction) const
                {
                    godot::Dictionary best_hit;
                    float best_t = 1e30f;
                    
                    for (size_t i = 0; i < m_bboxes.shape()[0]; ++i)
                    {
                        // Ray-AABB test
                        float tmin, tmax;
                        if (intersect_ray_aabb(origin, direction, i, tmin, tmax))
                        {
                            if (tmin < best_t)
                            {
                                // Precise polygon test
                                float t_poly;
                                godot::Vector2 normal;
                                if (intersect_ray_polygon(origin, direction, i, t_poly, normal))
                                {
                                    if (t_poly < best_t)
                                    {
                                        best_t = t_poly;
                                        best_hit["index"] = static_cast<int64_t>(i);
                                        best_hit["distance"] = t_poly;
                                        best_hit["position"] = origin + direction * t_poly;
                                        best_hit["normal"] = normal;
                                    }
                                }
                            }
                        }
                    }
                    return best_hit;
                }

                godot::PackedInt64Array query_circle(const godot::Vector2& center, float radius) const
                {
                    godot::PackedInt64Array result;
                    float r2 = radius * radius;
                    for (size_t i = 0; i < m_bboxes.shape()[0]; ++i)
                    {
                        // Circle-AABB test
                        float dx = std::max(m_bboxes(i, 0) - center.x, std::max(0.0f, center.x - m_bboxes(i, 2)));
                        float dy = std::max(m_bboxes(i, 1) - center.y, std::max(0.0f, center.y - m_bboxes(i, 3)));
                        if (dx*dx + dy*dy <= r2)
                            result.append(static_cast<int64_t>(i));
                    }
                    return result;
                }

                godot::Ref<XTensorNode> batch_query_points(const godot::Ref<XTensorNode>& points) const
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    auto pts = points->get_tensor_resource()->m_data.to_double_array();
                    size_t n = pts.shape()[0];
                    xarray_container<double> hits({n});
                    for (size_t i = 0; i < n; ++i)
                    {
                        auto indices = query_point(godot::Vector2(pts(i, 0), pts(i, 1)));
                        hits(i) = indices.is_empty() ? -1.0 : static_cast<double>(indices[0]);
                    }
                    result->set_data(XVariant::from_xarray(hits).variant());
                    return result;
                }

                godot::Vector2 get_closest_point(const godot::Vector2& point) const
                {
                    godot::Vector2 closest = point;
                    float min_dist2 = 1e30f;
                    for (size_t i = 0; i < m_bboxes.shape()[0]; ++i)
                    {
                        godot::Vector2 cp = closest_point_on_polygon(point, i);
                        float d2 = point.distance_squared_to(cp);
                        if (d2 < min_dist2)
                        {
                            min_dist2 = d2;
                            closest = cp;
                        }
                    }
                    return closest;
                }

                float get_sdf(const godot::Vector2& point) const
                {
                    godot::Vector2 cp = get_closest_point(point);
                    float dist = point.distance_to(cp);
                    // Determine sign (inside/outside) using winding number
                    bool inside = false;
                    for (size_t i = 0; i < m_bboxes.shape()[0]; ++i)
                    {
                        if (point_in_polygon(point, i))
                        {
                            inside = true;
                            break;
                        }
                    }
                    return inside ? -dist : dist;
                }

                void clear()
                {
                    m_vertices = xt::zeros<double>({0, 2});
                    m_bboxes = xt::zeros<double>({0, 4});
                    m_bvh.clear();
                    m_dirty = true;
                }

            private:
                bool point_in_polygon(const godot::Vector2& p, size_t poly_idx) const
                {
                    // Find vertices for this polygon
                    size_t start = 0;
                    for (size_t i = 0; i < poly_idx; ++i)
                    {
                        // This requires storing polygon vertex ranges; simplified for now
                    }
                    // Placeholder
                    return false;
                }

                bool intersect_ray_aabb(const godot::Vector2& origin, const godot::Vector2& dir,
                                        size_t poly_idx, float& tmin, float& tmax) const
                {
                    float min_x = m_bboxes(poly_idx, 0);
                    float min_y = m_bboxes(poly_idx, 1);
                    float max_x = m_bboxes(poly_idx, 2);
                    float max_y = m_bboxes(poly_idx, 3);
                    float t1 = (min_x - origin.x) / dir.x;
                    float t2 = (max_x - origin.x) / dir.x;
                    float t3 = (min_y - origin.y) / dir.y;
                    float t4 = (max_y - origin.y) / dir.y;
                    tmin = std::max(std::min(t1, t2), std::min(t3, t4));
                    tmax = std::min(std::max(t1, t2), std::max(t3, t4));
                    return tmax >= 0 && tmin <= tmax;
                }

                bool intersect_ray_polygon(const godot::Vector2& origin, const godot::Vector2& dir,
                                           size_t poly_idx, float& t, godot::Vector2& normal) const
                {
                    // Simplified: return false
                    return false;
                }

                godot::Vector2 closest_point_on_polygon(const godot::Vector2& p, size_t poly_idx) const
                {
                    return p;
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XPhysics2DRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XPhysicsTensor2D>();
                    godot::ClassDB::register_class<XCollisionWorld2D>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::XPhysicsTensor2D;
        using godot_bridge::XCollisionWorld2D;
        using godot_bridge::XPhysics2DRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XPHYSICS2D_HPP

// godot/xphysics2d.hpp