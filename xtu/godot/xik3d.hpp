// godot/xik3d.hpp

#ifndef XTENSOR_XIK3D_HPP
#define XTENSOR_XIK3D_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xnorm.hpp"
#include "../math/xquaternion.hpp"
#include "../math/xoptimize.hpp"
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
#include <queue>
#include <tuple>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/node3d.hpp>
    #include <godot_cpp/classes/skeleton3d.hpp>
    #include <godot_cpp/classes/bone_attachment3d.hpp>
    #include <godot_cpp/classes/mesh_instance3d.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/transform3d.hpp>
    #include <godot_cpp/variant/vector3.hpp>
    #include <godot_cpp/variant/quaternion.hpp>
    #include <godot_cpp/variant/basis.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // Quaternion utilities for IK
            // --------------------------------------------------------------------
            namespace ik_quat_utils
            {
                inline xarray_container<double> quat_from_to(const xarray_container<double>& from,
                                                             const xarray_container<double>& to)
                {
                    // from and to are Nx3 direction vectors
                    size_t n = from.shape()[0];
                    xarray_container<double> result({n, 4});
                    for (size_t i = 0; i < n; ++i)
                    {
                        double fx = from(i,0), fy = from(i,1), fz = from(i,2);
                        double tx = to(i,0), ty = to(i,1), tz = to(i,2);
                        double flen = std::sqrt(fx*fx + fy*fy + fz*fz);
                        double tlen = std::sqrt(tx*tx + ty*ty + tz*tz);
                        if (flen < 1e-10 || tlen < 1e-10)
                        {
                            result(i,0) = 1.0; result(i,1) = 0.0; result(i,2) = 0.0; result(i,3) = 0.0;
                            continue;
                        }
                        fx /= flen; fy /= flen; fz /= flen;
                        tx /= tlen; ty /= tlen; tz /= tlen;
                        double dot = fx*tx + fy*ty + fz*tz;
                        if (dot > 0.999999)
                        {
                            result(i,0) = 1.0; result(i,1) = 0.0; result(i,2) = 0.0; result(i,3) = 0.0;
                        }
                        else if (dot < -0.999999)
                        {
                            result(i,0) = 0.0; result(i,1) = 1.0; result(i,2) = 0.0; result(i,3) = 0.0;
                        }
                        else
                        {
                            double w = 1.0 + dot;
                            double x = fy*tz - fz*ty;
                            double y = fz*tx - fx*tz;
                            double z = fx*ty - fy*tx;
                            double len = std::sqrt(w*w + x*x + y*y + z*z);
                            result(i,0) = w/len; result(i,1) = x/len; result(i,2) = y/len; result(i,3) = z/len;
                        }
                    }
                    return result;
                }

                inline xarray_container<double> quat_rotate_vector_batch(const xarray_container<double>& q,
                                                                         const xarray_container<double>& v)
                {
                    size_t n = q.shape()[0];
                    xarray_container<double> result({n, 3});
                    for (size_t i = 0; i < n; ++i)
                    {
                        double qw = q(i,0), qx = q(i,1), qy = q(i,2), qz = q(i,3);
                        double vx = v(i,0), vy = v(i,1), vz = v(i,2);
                        double tx = 2.0 * (qy*vz - qz*vy);
                        double ty = 2.0 * (qz*vx - qx*vz);
                        double tz = 2.0 * (qx*vy - qy*vx);
                        result(i,0) = vx + qw*tx + (qy*tz - qz*ty);
                        result(i,1) = vy + qw*ty + (qz*tx - qx*tz);
                        result(i,2) = vz + qw*tz + (qx*ty - qy*tx);
                    }
                    return result;
                }

                inline xarray_container<double> quat_mul_batch(const xarray_container<double>& q1,
                                                               const xarray_container<double>& q2)
                {
                    size_t n = q1.shape()[0];
                    xarray_container<double> result({n, 4});
                    for (size_t i = 0; i < n; ++i)
                    {
                        double a1 = q1(i,0), b1 = q1(i,1), c1 = q1(i,2), d1 = q1(i,3);
                        double a2 = q2(i,0), b2 = q2(i,1), c2 = q2(i,2), d2 = q2(i,3);
                        result(i,0) = a1*a2 - b1*b2 - c1*c2 - d1*d2;
                        result(i,1) = a1*b2 + b1*a2 + c1*d2 - d1*c2;
                        result(i,2) = a1*c2 - b1*d2 + c1*a2 + d1*b2;
                        result(i,3) = a1*d2 + b1*c2 - c1*b2 + d1*a2;
                    }
                    return result;
                }
            }

            // --------------------------------------------------------------------
            // XIKChain3D - Batch IK solver for multiple chains
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XIKChain3D : public godot::Node3D
            {
                GDCLASS(XIKChain3D, godot::Node3D)

            public:
                enum SolverType
                {
                    SOLVER_CCD = 0,          // Cyclic Coordinate Descent
                    SOLVER_FABRIK = 1,       // Forward And Backward Reaching IK
                    SOLVER_JACOBIAN = 2,     // Jacobian pseudo-inverse
                    SOLVER_OPTIMIZATION = 3   // Numerical optimization
                };

            private:
                // Batch data: N chains, each with M joints
                godot::Ref<XTensorNode> m_joint_positions;     // N x M x 3
                godot::Ref<XTensorNode> m_joint_rotations;     // N x M x 4 (quaternions)
                godot::Ref<XTensorNode> m_bone_lengths;        // N x (M-1)
                godot::Ref<XTensorNode> m_target_positions;    // N x 3
                godot::Ref<XTensorNode> m_effector_indices;    // N (which joint is end effector, default M-1)
                godot::Ref<XTensorNode> m_joint_weights;       // N x M (weight for CCD/FABRIK)
                godot::Ref<XTensorNode> m_pole_targets;        // N x 3 (for pole vector constraint)
                
                SolverType m_solver_type = SOLVER_CCD;
                int m_max_iterations = 20;
                float m_tolerance = 0.001f;
                float m_damping = 0.1f;           // for Jacobian
                bool m_use_pole_constraint = false;
                bool m_maintain_original_rotation = false;
                bool m_auto_solve = true;
                bool m_initialized = false;
                
                // Internal working arrays
                size_t m_num_chains = 0;
                size_t m_num_joints = 0;
                xarray_container<double> m_local_positions;
                xarray_container<double> m_local_rotations;
                xarray_container<double> m_world_positions;
                xarray_container<double> m_world_rotations;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_joint_positions", "tensor"), &XIKChain3D::set_joint_positions);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_joint_positions"), &XIKChain3D::get_joint_positions);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_joint_rotations", "tensor"), &XIKChain3D::set_joint_rotations);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_joint_rotations"), &XIKChain3D::get_joint_rotations);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_bone_lengths", "tensor"), &XIKChain3D::set_bone_lengths);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_bone_lengths"), &XIKChain3D::get_bone_lengths);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_target_positions", "tensor"), &XIKChain3D::set_target_positions);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_target_positions"), &XIKChain3D::get_target_positions);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_effector_indices", "tensor"), &XIKChain3D::set_effector_indices);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_effector_indices"), &XIKChain3D::get_effector_indices);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_joint_weights", "tensor"), &XIKChain3D::set_joint_weights);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_joint_weights"), &XIKChain3D::get_joint_weights);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_pole_targets", "tensor"), &XIKChain3D::set_pole_targets);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_pole_targets"), &XIKChain3D::get_pole_targets);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("set_solver_type", "type"), &XIKChain3D::set_solver_type);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_solver_type"), &XIKChain3D::get_solver_type);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_max_iterations", "iterations"), &XIKChain3D::set_max_iterations);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_max_iterations"), &XIKChain3D::get_max_iterations);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_tolerance", "tolerance"), &XIKChain3D::set_tolerance);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_tolerance"), &XIKChain3D::get_tolerance);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_damping", "damping"), &XIKChain3D::set_damping);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_damping"), &XIKChain3D::get_damping);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_use_pole_constraint", "enabled"), &XIKChain3D::set_use_pole_constraint);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_use_pole_constraint"), &XIKChain3D::get_use_pole_constraint);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_auto_solve", "enabled"), &XIKChain3D::set_auto_solve);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_auto_solve"), &XIKChain3D::get_auto_solve);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("initialize"), &XIKChain3D::initialize);
                    godot::ClassDB::bind_method(godot::D_METHOD("solve"), &XIKChain3D::solve);
                    godot::ClassDB::bind_method(godot::D_METHOD("solve_ccd"), &XIKChain3D::solve_ccd);
                    godot::ClassDB::bind_method(godot::D_METHOD("solve_fabrik"), &XIKChain3D::solve_fabrik);
                    godot::ClassDB::bind_method(godot::D_METHOD("solve_jacobian"), &XIKChain3D::solve_jacobian);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_current_error"), &XIKChain3D::get_current_error);
                    godot::ClassDB::bind_method(godot::D_METHOD("forward_kinematics"), &XIKChain3D::forward_kinematics);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "joint_positions", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_joint_positions", "get_joint_positions");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "joint_rotations", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_joint_rotations", "get_joint_rotations");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "bone_lengths", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_bone_lengths", "get_bone_lengths");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "target_positions", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_target_positions", "get_target_positions");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "solver_type", godot::PROPERTY_HINT_ENUM, "CCD,FABRIK,Jacobian,Optimization"), "set_solver_type", "get_solver_type");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "max_iterations"), "set_max_iterations", "get_max_iterations");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "tolerance"), "set_tolerance", "get_tolerance");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "auto_solve"), "set_auto_solve", "get_auto_solve");
                    
                    BIND_ENUM_CONSTANT(SOLVER_CCD);
                    BIND_ENUM_CONSTANT(SOLVER_FABRIK);
                    BIND_ENUM_CONSTANT(SOLVER_JACOBIAN);
                    BIND_ENUM_CONSTANT(SOLVER_OPTIMIZATION);
                    
                    ADD_SIGNAL(godot::MethodInfo("solution_updated"));
                    ADD_SIGNAL(godot::MethodInfo("target_reached", godot::PropertyInfo(godot::Variant::INT, "chain_index")));
                }

            public:
                XIKChain3D() {}
                
                void _ready() override
                {
                    if (m_auto_solve)
                    {
                        initialize();
                        solve();
                    }
                }
                
                void _process(double delta) override
                {
                    if (m_auto_solve && m_initialized)
                        solve();
                }

                // Setters/Getters
                void set_joint_positions(const godot::Ref<XTensorNode>& tensor) { m_joint_positions = tensor; m_initialized = false; }
                godot::Ref<XTensorNode> get_joint_positions() const { return m_joint_positions; }
                void set_joint_rotations(const godot::Ref<XTensorNode>& tensor) { m_joint_rotations = tensor; m_initialized = false; }
                godot::Ref<XTensorNode> get_joint_rotations() const { return m_joint_rotations; }
                void set_bone_lengths(const godot::Ref<XTensorNode>& tensor) { m_bone_lengths = tensor; m_initialized = false; }
                godot::Ref<XTensorNode> get_bone_lengths() const { return m_bone_lengths; }
                void set_target_positions(const godot::Ref<XTensorNode>& tensor) { m_target_positions = tensor; }
                godot::Ref<XTensorNode> get_target_positions() const { return m_target_positions; }
                void set_effector_indices(const godot::Ref<XTensorNode>& tensor) { m_effector_indices = tensor; }
                godot::Ref<XTensorNode> get_effector_indices() const { return m_effector_indices; }
                void set_joint_weights(const godot::Ref<XTensorNode>& tensor) { m_joint_weights = tensor; }
                godot::Ref<XTensorNode> get_joint_weights() const { return m_joint_weights; }
                void set_pole_targets(const godot::Ref<XTensorNode>& tensor) { m_pole_targets = tensor; }
                godot::Ref<XTensorNode> get_pole_targets() const { return m_pole_targets; }
                
                void set_solver_type(SolverType type) { m_solver_type = type; }
                SolverType get_solver_type() const { return m_solver_type; }
                void set_max_iterations(int iter) { m_max_iterations = iter; }
                int get_max_iterations() const { return m_max_iterations; }
                void set_tolerance(float tol) { m_tolerance = tol; }
                float get_tolerance() const { return m_tolerance; }
                void set_damping(float d) { m_damping = d; }
                float get_damping() const { return m_damping; }
                void set_use_pole_constraint(bool enable) { m_use_pole_constraint = enable; }
                bool get_use_pole_constraint() const { return m_use_pole_constraint; }
                void set_auto_solve(bool enable) { m_auto_solve = enable; }
                bool get_auto_solve() const { return m_auto_solve; }

                // Initialization
                void initialize()
                {
                    if (!m_joint_positions.is_valid())
                    {
                        godot::UtilityFunctions::printerr("XIKChain3D: joint_positions tensor not set");
                        return;
                    }
                    auto pos_arr = m_joint_positions->get_tensor_resource()->m_data.to_double_array();
                    if (pos_arr.dimension() != 3)
                    {
                        godot::UtilityFunctions::printerr("XIKChain3D: joint_positions must be 3D (chains x joints x 3)");
                        return;
                    }
                    m_num_chains = pos_arr.shape()[0];
                    m_num_joints = pos_arr.shape()[1];
                    
                    // Ensure other tensors have correct shapes or create defaults
                    if (!m_joint_rotations.is_valid())
                    {
                        xarray_container<double> rots({m_num_chains, m_num_joints, 4});
                        for (size_t i = 0; i < m_num_chains; ++i)
                            for (size_t j = 0; j < m_num_joints; ++j)
                            {
                                rots(i, j, 0) = 1.0; rots(i, j, 1) = 0.0; rots(i, j, 2) = 0.0; rots(i, j, 3) = 0.0;
                            }
                        m_joint_rotations.instantiate();
                        m_joint_rotations->set_data(XVariant::from_xarray(rots).variant());
                    }
                    if (!m_bone_lengths.is_valid() && m_num_joints > 1)
                    {
                        // Compute from initial positions
                        xarray_container<double> lengths({m_num_chains, m_num_joints - 1});
                        for (size_t c = 0; c < m_num_chains; ++c)
                        {
                            for (size_t j = 0; j < m_num_joints - 1; ++j)
                            {
                                double dx = pos_arr(c, j+1, 0) - pos_arr(c, j, 0);
                                double dy = pos_arr(c, j+1, 1) - pos_arr(c, j, 1);
                                double dz = pos_arr(c, j+1, 2) - pos_arr(c, j, 2);
                                lengths(c, j) = std::sqrt(dx*dx + dy*dy + dz*dz);
                            }
                        }
                        m_bone_lengths.instantiate();
                        m_bone_lengths->set_data(XVariant::from_xarray(lengths).variant());
                    }
                    if (!m_target_positions.is_valid())
                    {
                        xarray_container<double> targets({m_num_chains, 3});
                        m_target_positions.instantiate();
                        m_target_positions->set_data(XVariant::from_xarray(targets).variant());
                    }
                    if (!m_joint_weights.is_valid())
                    {
                        xarray_container<double> weights({m_num_chains, m_num_joints}, 1.0);
                        m_joint_weights.instantiate();
                        m_joint_weights->set_data(XVariant::from_xarray(weights).variant());
                    }
                    
                    // Copy to working arrays
                    m_local_positions = m_joint_positions->get_tensor_resource()->m_data.to_double_array();
                    m_local_rotations = m_joint_rotations->get_tensor_resource()->m_data.to_double_array();
                    m_world_positions = m_local_positions;
                    m_world_rotations = m_local_rotations;
                    
                    m_initialized = true;
                }

                // Main solve dispatcher
                void solve()
                {
                    if (!m_initialized) initialize();
                    if (!m_initialized) return;
                    
                    // Update targets from tensor
                    if (m_target_positions.is_valid())
                    {
                        auto targets = m_target_positions->get_tensor_resource()->m_data.to_double_array();
                        // Use targets in solver
                    }
                    
                    switch (m_solver_type)
                    {
                        case SOLVER_CCD:
                            solve_ccd();
                            break;
                        case SOLVER_FABRIK:
                            solve_fabrik();
                            break;
                        case SOLVER_JACOBIAN:
                            solve_jacobian();
                            break;
                        case SOLVER_OPTIMIZATION:
                            // Placeholder
                            break;
                    }
                    
                    // Write back results
                    m_joint_positions->set_data(XVariant::from_xarray(m_world_positions).variant());
                    m_joint_rotations->set_data(XVariant::from_xarray(m_world_rotations).variant());
                    emit_signal("solution_updated");
                }

                // Cyclic Coordinate Descent (batch)
                void solve_ccd()
                {
                    if (!m_initialized) return;
                    auto pos = m_world_positions;
                    auto rots = m_world_rotations;
                    auto targets = m_target_positions->get_tensor_resource()->m_data.to_double_array();
                    auto lengths = m_bone_lengths.is_valid() ? m_bone_lengths->get_tensor_resource()->m_data.to_double_array() : xarray_container<double>();
                    auto weights = m_joint_weights.is_valid() ? m_joint_weights->get_tensor_resource()->m_data.to_double_array() : xarray_container<double>();
                    
                    for (size_t c = 0; c < m_num_chains; ++c)
                    {
                        size_t effector_idx = m_num_joints - 1;
                        if (m_effector_indices.is_valid())
                        {
                            auto eff = m_effector_indices->get_tensor_resource()->m_data.to_double_array();
                            if (c < eff.size()) effector_idx = static_cast<size_t>(eff(c));
                        }
                        
                        godot::Vector3 target(targets(c,0), targets(c,1), targets(c,2));
                        
                        for (int iter = 0; iter < m_max_iterations; ++iter)
                        {
                            godot::Vector3 effector(pos(c, effector_idx, 0), pos(c, effector_idx, 1), pos(c, effector_idx, 2));
                            float error = effector.distance_to(target);
                            if (error < m_tolerance)
                            {
                                emit_signal("target_reached", static_cast<int64_t>(c));
                                break;
                            }
                            
                            // Iterate backwards from effector-1 to root
                            for (int j = static_cast<int>(effector_idx) - 1; j >= 0; --j)
                            {
                                float w = (j < static_cast<int>(weights.shape()[1])) ? weights(c, j) : 1.0f;
                                if (w < 0.001f) continue;
                                
                                godot::Vector3 joint_pos(pos(c, j, 0), pos(c, j, 1), pos(c, j, 2));
                                godot::Vector3 to_effector = effector - joint_pos;
                                godot::Vector3 to_target = target - joint_pos;
                                
                                float len_e = to_effector.length();
                                float len_t = to_target.length();
                                if (len_e < 1e-6f || len_t < 1e-6f) continue;
                                
                                to_effector /= len_e;
                                to_target /= len_t;
                                
                                // Compute rotation axis and angle
                                godot::Vector3 axis = to_effector.cross(to_target);
                                float axis_len = axis.length();
                                if (axis_len < 1e-6f) continue;
                                axis /= axis_len;
                                float angle = std::acos(std::clamp(to_effector.dot(to_target), -1.0f, 1.0f)) * w;
                                
                                // Create quaternion and rotate all descendant joints
                                float half_angle = angle * 0.5f;
                                float s = std::sin(half_angle);
                                float qw = std::cos(half_angle);
                                float qx = axis.x * s, qy = axis.y * s, qz = axis.z * s;
                                
                                // Apply to this joint's rotation
                                double rw = rots(c, j, 0), rx = rots(c, j, 1), ry = rots(c, j, 2), rz = rots(c, j, 3);
                                double new_rw = qw*rw - qx*rx - qy*ry - qz*rz;
                                double new_rx = qw*rx + qx*rw + qy*rz - qz*ry;
                                double new_ry = qw*ry - qx*rz + qy*rw + qz*rx;
                                double new_rz = qw*rz + qx*ry - qy*rx + qz*rw;
                                double rlen = std::sqrt(new_rw*new_rw + new_rx*new_rx + new_ry*new_ry + new_rz*new_rz);
                                rots(c, j, 0) = new_rw/rlen; rots(c, j, 1) = new_rx/rlen;
                                rots(c, j, 2) = new_ry/rlen; rots(c, j, 3) = new_rz/rlen;
                                
                                // Rotate all descendant joints (including effector)
                                for (size_t k = j + 1; k <= effector_idx; ++k)
                                {
                                    godot::Vector3 rel = godot::Vector3(pos(c, k, 0), pos(c, k, 1), pos(c, k, 2)) - joint_pos;
                                    godot::Vector3 rotated_rel = rotate_vector(rel, qw, qx, qy, qz);
                                    pos(c, k, 0) = joint_pos.x + rotated_rel.x;
                                    pos(c, k, 1) = joint_pos.y + rotated_rel.y;
                                    pos(c, k, 2) = joint_pos.z + rotated_rel.z;
                                    
                                    // Also rotate joint rotation
                                    if (k < m_num_joints)
                                    {
                                        double krw = rots(c, k, 0), krx = rots(c, k, 1), kry = rots(c, k, 2), krz = rots(c, k, 3);
                                        double nkrw = qw*krw - qx*krx - qy*kry - qz*krz;
                                        double nkrx = qw*krx + qx*krw + qy*krz - qz*kry;
                                        double nkry = qw*kry - qx*krz + qy*krw + qz*krx;
                                        double nkrz = qw*krz + qx*kry - qy*krx + qz*krw;
                                        double klen = std::sqrt(nkrw*nkrw + nkrx*nkrx + nkry*nkry + nkrz*nkrz);
                                        rots(c, k, 0) = nkrw/klen; rots(c, k, 1) = nkrx/klen;
                                        rots(c, k, 2) = nkry/klen; rots(c, k, 3) = nkrz/klen;
                                    }
                                }
                                effector = godot::Vector3(pos(c, effector_idx, 0), pos(c, effector_idx, 1), pos(c, effector_idx, 2));
                            }
                        }
                    }
                    m_world_positions = pos;
                    m_world_rotations = rots;
                }

                // FABRIK solver (batch)
                void solve_fabrik()
                {
                    if (!m_initialized) return;
                    auto pos = m_world_positions;
                    auto targets = m_target_positions->get_tensor_resource()->m_data.to_double_array();
                    auto lengths = m_bone_lengths.is_valid() ? m_bone_lengths->get_tensor_resource()->m_data.to_double_array() : xarray_container<double>();
                    auto weights = m_joint_weights.is_valid() ? m_joint_weights->get_tensor_resource()->m_data.to_double_array() : xarray_container<double>();
                    
                    for (size_t c = 0; c < m_num_chains; ++c)
                    {
                        size_t effector_idx = m_num_joints - 1;
                        godot::Vector3 target(targets(c,0), targets(c,1), targets(c,2));
                        godot::Vector3 root(pos(c, 0, 0), pos(c, 0, 1), pos(c, 0, 2));
                        
                        // Check if target is reachable
                        float total_length = 0.0f;
                        for (size_t j = 0; j < effector_idx; ++j)
                            total_length += lengths(c, j);
                        float dist_to_target = root.distance_to(target);
                        
                        if (dist_to_target > total_length)
                        {
                            // Unreachable: stretch towards target
                            godot::Vector3 dir = (target - root).normalized();
                            for (size_t j = 0; j < effector_idx; ++j)
                            {
                                pos(c, j+1, 0) = pos(c, j, 0) + dir.x * lengths(c, j);
                                pos(c, j+1, 1) = pos(c, j, 1) + dir.y * lengths(c, j);
                                pos(c, j+1, 2) = pos(c, j, 2) + dir.z * lengths(c, j);
                            }
                        }
                        else
                        {
                            // FABRIK iterations
                            for (int iter = 0; iter < m_max_iterations; ++iter)
                            {
                                // Forward reaching: from effector to root
                                pos(c, effector_idx, 0) = target.x;
                                pos(c, effector_idx, 1) = target.y;
                                pos(c, effector_idx, 2) = target.z;
                                
                                for (int j = static_cast<int>(effector_idx) - 1; j >= 0; --j)
                                {
                                    godot::Vector3 curr(pos(c, j, 0), pos(c, j, 1), pos(c, j, 2));
                                    godot::Vector3 next(pos(c, j+1, 0), pos(c, j+1, 1), pos(c, j+1, 2));
                                    godot::Vector3 dir = (curr - next).normalized();
                                    float len = lengths(c, j);
                                    float w = (j < static_cast<int>(weights.shape()[1])) ? weights(c, j) : 1.0f;
                                    pos(c, j, 0) = next.x + dir.x * len * w;
                                    pos(c, j, 1) = next.y + dir.y * len * w;
                                    pos(c, j, 2) = next.z + dir.z * len * w;
                                }
                                
                                // Backward reaching: from root to effector
                                for (size_t j = 0; j < effector_idx; ++j)
                                {
                                    godot::Vector3 curr(pos(c, j, 0), pos(c, j, 1), pos(c, j, 2));
                                    godot::Vector3 next(pos(c, j+1, 0), pos(c, j+1, 1), pos(c, j+1, 2));
                                    godot::Vector3 dir = (next - curr).normalized();
                                    float len = lengths(c, j);
                                    float w = (j < static_cast<int>(weights.shape()[1])) ? weights(c, j) : 1.0f;
                                    pos(c, j+1, 0) = curr.x + dir.x * len * w;
                                    pos(c, j+1, 1) = curr.y + dir.y * len * w;
                                    pos(c, j+1, 2) = curr.z + dir.z * len * w;
                                }
                                
                                godot::Vector3 effector(pos(c, effector_idx, 0), pos(c, effector_idx, 1), pos(c, effector_idx, 2));
                                if (effector.distance_to(target) < m_tolerance)
                                {
                                    emit_signal("target_reached", static_cast<int64_t>(c));
                                    break;
                                }
                            }
                        }
                        
                        // Update rotations to match positions (simple look-at)
                        for (size_t j = 0; j < effector_idx; ++j)
                        {
                            godot::Vector3 dir(pos(c, j+1, 0) - pos(c, j, 0),
                                              pos(c, j+1, 1) - pos(c, j, 1),
                                              pos(c, j+1, 2) - pos(c, j, 2));
                            float len = dir.length();
                            if (len > 1e-6f)
                            {
                                dir /= len;
                                // Default up vector (adjust based on pole if available)
                                godot::Vector3 up(0, 1, 0);
                                if (m_use_pole_constraint && m_pole_targets.is_valid())
                                {
                                    auto pole = m_pole_targets->get_tensor_resource()->m_data.to_double_array();
                                    if (c < pole.shape()[0])
                                        up = godot::Vector3(pole(c,0), pole(c,1), pole(c,2)) - godot::Vector3(pos(c, j, 0), pos(c, j, 1), pos(c, j, 2));
                                }
                                godot::Vector3 right = up.cross(dir).normalized();
                                godot::Vector3 new_up = dir.cross(right);
                                godot::Basis basis(right, new_up, dir);
                                godot::Quaternion q = basis.get_quaternion().normalized();
                                m_world_rotations(c, j, 0) = q.w;
                                m_world_rotations(c, j, 1) = q.x;
                                m_world_rotations(c, j, 2) = q.y;
                                m_world_rotations(c, j, 3) = q.z;
                            }
                        }
                    }
                    m_world_positions = pos;
                }

                // Jacobian pseudo-inverse solver
                void solve_jacobian()
                {
                    if (!m_initialized) return;
                    auto pos = m_world_positions;
                    auto rots = m_world_rotations;
                    auto targets = m_target_positions->get_tensor_resource()->m_data.to_double_array();
                    
                    for (size_t c = 0; c < m_num_chains; ++c)
                    {
                        size_t effector_idx = m_num_joints - 1;
                        godot::Vector3 target(targets(c,0), targets(c,1), targets(c,2));
                        
                        for (int iter = 0; iter < m_max_iterations; ++iter)
                        {
                            godot::Vector3 effector(pos(c, effector_idx, 0), pos(c, effector_idx, 1), pos(c, effector_idx, 2));
                            godot::Vector3 error_vec = target - effector;
                            float error = error_vec.length();
                            if (error < m_tolerance)
                            {
                                emit_signal("target_reached", static_cast<int64_t>(c));
                                break;
                            }
                            
                            // Build Jacobian (3 x (3*effector_idx))
                            size_t n_joints = effector_idx;
                            xarray_container<double> J({3, n_joints * 3}, 0.0);
                            for (size_t j = 0; j < n_joints; ++j)
                            {
                                godot::Vector3 joint_pos(pos(c, j, 0), pos(c, j, 1), pos(c, j, 2));
                                godot::Vector3 to_effector = effector - joint_pos;
                                
                                // Rotational part: axis x to_effector
                                J(0, j*3 + 0) = 0; J(0, j*3 + 1) = to_effector.z; J(0, j*3 + 2) = -to_effector.y;
                                J(1, j*3 + 0) = -to_effector.z; J(1, j*3 + 1) = 0; J(1, j*3 + 2) = to_effector.x;
                                J(2, j*3 + 0) = to_effector.y; J(2, j*3 + 1) = -to_effector.x; J(2, j*3 + 2) = 0;
                            }
                            
                            // Solve J * dq = error_vec using pseudo-inverse with damping
                            auto Jt = xt::transpose(J);
                            auto JJt = xt::linalg::dot(J, Jt);
                            // Add damping
                            for (size_t i = 0; i < 3; ++i)
                                JJt(i, i) += m_damping * m_damping;
                            auto JJt_inv = xt::linalg::inv(JJt);
                            auto J_pinv = xt::linalg::dot(Jt, JJt_inv);
                            
                            // Compute delta q
                            xarray_container<double> err_arr({3});
                            err_arr(0) = error_vec.x; err_arr(1) = error_vec.y; err_arr(2) = error_vec.z;
                            auto dq = xt::linalg::dot(J_pinv, err_arr);
                            
                            // Apply update to joint rotations
                            for (size_t j = 0; j < n_joints; ++j)
                            {
                                double dqx = dq(j*3 + 0);
                                double dqy = dq(j*3 + 1);
                                double dqz = dq(j*3 + 2);
                                double angle = std::sqrt(dqx*dqx + dqy*dqy + dqz*dqz);
                                if (angle > 1e-6)
                                {
                                    double qw = std::cos(angle * 0.5);
                                    double s = std::sin(angle * 0.5) / angle;
                                    double qx = dqx * s, qy = dqy * s, qz = dqz * s;
                                    
                                    // Apply to rotation
                                    double rw = rots(c, j, 0), rx = rots(c, j, 1), ry = rots(c, j, 2), rz = rots(c, j, 3);
                                    double nrw = qw*rw - qx*rx - qy*ry - qz*rz;
                                    double nrx = qw*rx + qx*rw + qy*rz - qz*ry;
                                    double nry = qw*ry - qx*rz + qy*rw + qz*rx;
                                    double nrz = qw*rz + qx*ry - qy*rx + qz*rw;
                                    double len = std::sqrt(nrw*nrw + nrx*nrx + nry*nry + nrz*nrz);
                                    rots(c, j, 0) = nrw/len; rots(c, j, 1) = nrx/len;
                                    rots(c, j, 2) = nry/len; rots(c, j, 3) = nrz/len;
                                }
                            }
                            
                            // Update positions via FK
                            forward_kinematics_single(c, pos, rots);
                            effector = godot::Vector3(pos(c, effector_idx, 0), pos(c, effector_idx, 1), pos(c, effector_idx, 2));
                        }
                    }
                    m_world_positions = pos;
                    m_world_rotations = rots;
                }

                // Forward kinematics for all chains
                void forward_kinematics()
                {
                    if (!m_initialized) return;
                    auto pos = m_world_positions;
                    auto rots = m_world_rotations;
                    auto lengths = m_bone_lengths.is_valid() ? m_bone_lengths->get_tensor_resource()->m_data.to_double_array() : xarray_container<double>();
                    
                    for (size_t c = 0; c < m_num_chains; ++c)
                    {
                        for (size_t j = 0; j < m_num_joints - 1; ++j)
                        {
                            double qw = rots(c, j, 0), qx = rots(c, j, 1), qy = rots(c, j, 2), qz = rots(c, j, 3);
                            godot::Vector3 dir(0, 0, 0);
                            if (lengths.size() > 0 && j < lengths.shape()[1])
                                dir.z = lengths(c, j);
                            else
                                dir.z = 1.0;
                            
                            // Rotate direction
                            double tx = 2.0 * (qy*dir.z - qz*dir.y);
                            double ty = 2.0 * (qz*dir.x - qx*dir.z);
                            double tz = 2.0 * (qx*dir.y - qy*dir.x);
                            godot::Vector3 rotated_dir(
                                dir.x + qw*tx + (qy*tz - qz*ty),
                                dir.y + qw*ty + (qz*tx - qx*tz),
                                dir.z + qw*tz + (qx*ty - qy*tx)
                            );
                            
                            pos(c, j+1, 0) = pos(c, j, 0) + rotated_dir.x;
                            pos(c, j+1, 1) = pos(c, j, 1) + rotated_dir.y;
                            pos(c, j+1, 2) = pos(c, j, 2) + rotated_dir.z;
                        }
                    }
                    m_world_positions = pos;
                    m_joint_positions->set_data(XVariant::from_xarray(pos).variant());
                }

                // Compute current error for all chains
                godot::PackedFloat32Array get_current_error()
                {
                    godot::PackedFloat32Array errors;
                    if (!m_initialized) return errors;
                    errors.resize(static_cast<int>(m_num_chains));
                    auto pos = m_world_positions;
                    auto targets = m_target_positions.is_valid() ? m_target_positions->get_tensor_resource()->m_data.to_double_array() : xarray_container<double>();
                    
                    for (size_t c = 0; c < m_num_chains; ++c)
                    {
                        size_t effector_idx = m_num_joints - 1;
                        if (m_effector_indices.is_valid())
                        {
                            auto eff = m_effector_indices->get_tensor_resource()->m_data.to_double_array();
                            if (c < eff.size()) effector_idx = static_cast<size_t>(eff(c));
                        }
                        godot::Vector3 effector(pos(c, effector_idx, 0), pos(c, effector_idx, 1), pos(c, effector_idx, 2));
                        godot::Vector3 target(targets(c,0), targets(c,1), targets(c,2));
                        errors.set(static_cast<int>(c), effector.distance_to(target));
                    }
                    return errors;
                }

                // Convert to Skeleton3D pose
                void apply_to_skeleton(godot::Skeleton3D* skeleton)
                {
                    if (!skeleton || !m_initialized) return;
                    int bone_count = skeleton->get_bone_count();
                    auto pos = m_world_positions;
                    auto rots = m_world_rotations;
                    
                    for (size_t c = 0; c < std::min(m_num_chains, static_cast<size_t>(1)); ++c)
                    {
                        for (size_t j = 0; j < std::min(m_num_joints, static_cast<size_t>(bone_count)); ++j)
                        {
                            godot::Vector3 position(pos(c, j, 0), pos(c, j, 1), pos(c, j, 2));
                            godot::Quaternion rotation(rots(c, j, 0), rots(c, j, 1), rots(c, j, 2), rots(c, j, 3));
                            skeleton->set_bone_pose_position(static_cast<int>(j), position);
                            skeleton->set_bone_pose_rotation(static_cast<int>(j), rotation);
                            skeleton->set_bone_pose_scale(static_cast<int>(j), godot::Vector3(1, 1, 1));
                        }
                    }
                }

            private:
                godot::Vector3 rotate_vector(const godot::Vector3& v, float qw, float qx, float qy, float qz)
                {
                    float tx = 2.0f * (qy*v.z - qz*v.y);
                    float ty = 2.0f * (qz*v.x - qx*v.z);
                    float tz = 2.0f * (qx*v.y - qy*v.x);
                    return godot::Vector3(
                        v.x + qw*tx + (qy*tz - qz*ty),
                        v.y + qw*ty + (qz*tx - qx*tz),
                        v.z + qw*tz + (qx*ty - qy*tx)
                    );
                }

                void forward_kinematics_single(size_t chain_idx, xarray_container<double>& pos, xarray_container<double>& rots)
                {
                    auto lengths = m_bone_lengths.is_valid() ? m_bone_lengths->get_tensor_resource()->m_data.to_double_array() : xarray_container<double>();
                    for (size_t j = 0; j < m_num_joints - 1; ++j)
                    {
                        double qw = rots(chain_idx, j, 0), qx = rots(chain_idx, j, 1), qy = rots(chain_idx, j, 2), qz = rots(chain_idx, j, 3);
                        godot::Vector3 dir(0, 0, 0);
                        if (lengths.size() > 0 && j < lengths.shape()[1])
                            dir.z = lengths(chain_idx, j);
                        else
                            dir.z = 1.0;
                        
                        float tx = 2.0f * (qy*dir.z - qz*dir.y);
                        float ty = 2.0f * (qz*dir.x - qx*dir.z);
                        float tz = 2.0f * (qx*dir.y - qy*dir.x);
                        godot::Vector3 rotated_dir(
                            dir.x + qw*tx + (qy*tz - qz*ty),
                            dir.y + qw*ty + (qz*tx - qx*tz),
                            dir.z + qw*tz + (qx*ty - qy*tx)
                        );
                        
                        pos(chain_idx, j+1, 0) = pos(chain_idx, j, 0) + rotated_dir.x;
                        pos(chain_idx, j+1, 1) = pos(chain_idx, j, 1) + rotated_dir.y;
                        pos(chain_idx, j+1, 2) = pos(chain_idx, j, 2) + rotated_dir.z;
                    }
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XIK3DRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XIKChain3D>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::XIKChain3D;
        using godot_bridge::XIK3DRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XIK3D_HPP

// godot/xik3d.hpp