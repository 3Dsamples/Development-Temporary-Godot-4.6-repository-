// godot/xanimation.hpp

#ifndef XTENSOR_XANIMATION_HPP
#define XTENSOR_XANIMATION_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xinterp.hpp"
#include "../math/xquaternion.hpp"
#include "../math/xoptimize.hpp"
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
#include <queue>
#include <mutex>
#include <chrono>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/animation.hpp>
    #include <godot_cpp/classes/animation_player.hpp>
    #include <godot_cpp/classes/animation_tree.hpp>
    #include <godot_cpp/classes/animation_node.hpp>
    #include <godot_cpp/classes/animation_node_state_machine.hpp>
    #include <godot_cpp/classes/animation_node_blend_tree.hpp>
    #include <godot_cpp/classes/animation_node_blend_space_1d.hpp>
    #include <godot_cpp/classes/animation_node_blend_space_2d.hpp>
    #include <godot_cpp/classes/animation_node_one_shot.hpp>
    #include <godot_cpp/classes/animation_node_time_scale.hpp>
    #include <godot_cpp/classes/animation_node_transition.hpp>
    #include <godot_cpp/classes/animation_root_node.hpp>
    #include <godot_cpp/classes/skeleton3d.hpp>
    #include <godot_cpp/classes/bone_attachment3d.hpp>
    #include <godot_cpp/classes/resource.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/transform3d.hpp>
    #include <godot_cpp/variant/quaternion.hpp>
    #include <godot_cpp/variant/vector3.hpp>
    #include <godot_cpp/variant/basis.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // Quaternion and Transform utilities for animation
            // --------------------------------------------------------------------
            namespace anim_utils
            {
                // Convert axis-angle rotations to quaternions (batch)
                inline xarray_container<double> axis_angle_to_quat(const xarray_container<double>& axis_angles)
                {
                    // axis_angles: N x 4 (x, y, z, angle)
                    size_t n = axis_angles.shape()[0];
                    xarray_container<double> quats({n, 4});
                    for (size_t i = 0; i < n; ++i)
                    {
                        double x = axis_angles(i, 0);
                        double y = axis_angles(i, 1);
                        double z = axis_angles(i, 2);
                        double angle = axis_angles(i, 3);
                        double len = std::sqrt(x*x + y*y + z*z);
                        if (len < 1e-10)
                        {
                            quats(i, 0) = 1.0; quats(i, 1) = 0.0; quats(i, 2) = 0.0; quats(i, 3) = 0.0;
                            continue;
                        }
                        x /= len; y /= len; z /= len;
                        double half = angle * 0.5;
                        double s = std::sin(half);
                        quats(i, 0) = std::cos(half);
                        quats(i, 1) = x * s;
                        quats(i, 2) = y * s;
                        quats(i, 3) = z * s;
                    }
                    return quats;
                }

                // Quaternion spherical linear interpolation (batch)
                inline xarray_container<double> quat_slerp_batch(const xarray_container<double>& q1,
                                                                 const xarray_container<double>& q2,
                                                                 const xarray_container<double>& t)
                {
                    size_t n = q1.shape()[0];
                    xarray_container<double> result({n, 4});
                    for (size_t i = 0; i < n; ++i)
                    {
                        double w1 = q1(i,0), x1 = q1(i,1), y1 = q1(i,2), z1 = q1(i,3);
                        double w2 = q2(i,0), x2 = q2(i,1), y2 = q2(i,2), z2 = q2(i,3);
                        double dot = w1*w2 + x1*x2 + y1*y2 + z1*z2;
                        if (dot < 0.0) { w2 = -w2; x2 = -x2; y2 = -y2; z2 = -z2; dot = -dot; }
                        double t_val = t(i);
                        if (dot > 0.9995)
                        {
                            result(i,0) = w1 + t_val*(w2 - w1);
                            result(i,1) = x1 + t_val*(x2 - x1);
                            result(i,2) = y1 + t_val*(y2 - y1);
                            result(i,3) = z1 + t_val*(z2 - z1);
                            double len = std::sqrt(result(i,0)*result(i,0) + result(i,1)*result(i,1) + result(i,2)*result(i,2) + result(i,3)*result(i,3));
                            result(i,0)/=len; result(i,1)/=len; result(i,2)/=len; result(i,3)/=len;
                        }
                        else
                        {
                            double theta_0 = std::acos(dot);
                            double theta = theta_0 * t_val;
                            double sin_theta = std::sin(theta);
                            double sin_theta_0 = std::sin(theta_0);
                            double s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
                            double s1 = sin_theta / sin_theta_0;
                            result(i,0) = s0*w1 + s1*w2;
                            result(i,1) = s0*x1 + s1*x2;
                            result(i,2) = s0*y1 + s1*y2;
                            result(i,3) = s0*z1 + s1*z2;
                        }
                    }
                    return result;
                }

                // Cubic bezier interpolation for vectors (batch)
                inline xarray_container<double> bezier_interp_batch(const xarray_container<double>& p0,
                                                                    const xarray_container<double>& p1,
                                                                    const xarray_container<double>& p2,
                                                                    const xarray_container<double>& p3,
                                                                    const xarray_container<double>& t)
                {
                    size_t n = p0.shape()[0];
                    size_t dim = p0.shape()[1];
                    xarray_container<double> result({n, dim});
                    for (size_t i = 0; i < n; ++i)
                    {
                        double t_val = t(i);
                        double mt = 1.0 - t_val;
                        double mt2 = mt * mt;
                        double t2 = t_val * t_val;
                        double mt3 = mt2 * mt;
                        double t3 = t2 * t_val;
                        for (size_t d = 0; d < dim; ++d)
                        {
                            result(i, d) = mt3 * p0(i, d) + 3.0 * mt2 * t_val * p1(i, d) +
                                           3.0 * mt * t2 * p2(i, d) + t3 * p3(i, d);
                        }
                    }
                    return result;
                }

                // Decompose transform matrix to TRS (translation, rotation, scale)
                inline void decompose_trs(const xarray_container<double>& transforms,
                                          xarray_container<double>& pos,
                                          xarray_container<double>& rot,
                                          xarray_container<double>& scale)
                {
                    // transforms: N x 12 (4x3 matrix)
                    size_t n = transforms.shape()[0];
                    pos = xarray_container<double>({n, 3});
                    rot = xarray_container<double>({n, 4});
                    scale = xarray_container<double>({n, 3});
                    for (size_t i = 0; i < n; ++i)
                    {
                        // Extract translation
                        pos(i,0) = transforms(i,3);
                        pos(i,1) = transforms(i,7);
                        pos(i,2) = transforms(i,11);
                        
                        // Extract 3x3 matrix
                        double m00 = transforms(i,0), m01 = transforms(i,1), m02 = transforms(i,2);
                        double m10 = transforms(i,4), m11 = transforms(i,5), m12 = transforms(i,6);
                        double m20 = transforms(i,8), m21 = transforms(i,9), m22 = transforms(i,10);
                        
                        // Extract scale
                        double sx = std::sqrt(m00*m00 + m10*m10 + m20*m20);
                        double sy = std::sqrt(m01*m01 + m11*m11 + m21*m21);
                        double sz = std::sqrt(m02*m02 + m12*m12 + m22*m22);
                        scale(i,0) = sx; scale(i,1) = sy; scale(i,2) = sz;
                        
                        // Normalize to get rotation matrix
                        if (sx > 1e-10) { m00 /= sx; m10 /= sx; m20 /= sx; }
                        if (sy > 1e-10) { m01 /= sy; m11 /= sy; m21 /= sy; }
                        if (sz > 1e-10) { m02 /= sz; m12 /= sz; m22 /= sz; }
                        
                        // Convert rotation matrix to quaternion
                        double trace = m00 + m11 + m22;
                        if (trace > 0.0)
                        {
                            double s = std::sqrt(trace + 1.0) * 2.0;
                            rot(i,0) = 0.25 * s;
                            rot(i,1) = (m21 - m12) / s;
                            rot(i,2) = (m02 - m20) / s;
                            rot(i,3) = (m10 - m01) / s;
                        }
                        else if (m00 > m11 && m00 > m22)
                        {
                            double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
                            rot(i,0) = (m21 - m12) / s;
                            rot(i,1) = 0.25 * s;
                            rot(i,2) = (m01 + m10) / s;
                            rot(i,3) = (m02 + m20) / s;
                        }
                        else if (m11 > m22)
                        {
                            double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
                            rot(i,0) = (m02 - m20) / s;
                            rot(i,1) = (m01 + m10) / s;
                            rot(i,2) = 0.25 * s;
                            rot(i,3) = (m12 + m21) / s;
                        }
                        else
                        {
                            double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
                            rot(i,0) = (m10 - m01) / s;
                            rot(i,1) = (m02 + m20) / s;
                            rot(i,2) = (m12 + m21) / s;
                            rot(i,3) = 0.25 * s;
                        }
                    }
                }

                // Compose TRS to transform matrix
                inline xarray_container<double> compose_trs(const xarray_container<double>& pos,
                                                            const xarray_container<double>& rot,
                                                            const xarray_container<double>& scale)
                {
                    size_t n = pos.shape()[0];
                    xarray_container<double> transforms({n, 12});
                    for (size_t i = 0; i < n; ++i)
                    {
                        double qw = rot(i,0), qx = rot(i,1), qy = rot(i,2), qz = rot(i,3);
                        double sx = scale(i,0), sy = scale(i,1), sz = scale(i,2);
                        
                        // Quaternion to rotation matrix
                        double xx = qx*qx, yy = qy*qy, zz = qz*qz;
                        double xy = qx*qy, xz = qx*qz, yz = qy*qz;
                        double wx = qw*qx, wy = qw*qy, wz = qw*qz;
                        
                        double m00 = (1.0 - 2.0*(yy + zz)) * sx;
                        double m01 = (2.0*(xy - wz)) * sy;
                        double m02 = (2.0*(xz + wy)) * sz;
                        double m10 = (2.0*(xy + wz)) * sx;
                        double m11 = (1.0 - 2.0*(xx + zz)) * sy;
                        double m12 = (2.0*(yz - wx)) * sz;
                        double m20 = (2.0*(xz - wy)) * sx;
                        double m21 = (2.0*(yz + wx)) * sy;
                        double m22 = (1.0 - 2.0*(xx + yy)) * sz;
                        
                        transforms(i,0) = m00; transforms(i,1) = m01; transforms(i,2) = m02; transforms(i,3) = pos(i,0);
                        transforms(i,4) = m10; transforms(i,5) = m11; transforms(i,6) = m12; transforms(i,7) = pos(i,1);
                        transforms(i,8) = m20; transforms(i,9) = m21; transforms(i,10)= m22; transforms(i,11)= pos(i,2);
                    }
                    return transforms;
                }
            }

            // --------------------------------------------------------------------
            // Tensor Animation Track (replaces Godot's Animation)
            // --------------------------------------------------------------------
            class XAnimationTrack
            {
            public:
                enum TrackType
                {
                    TYPE_VALUE = 0,
                    TYPE_POSITION_3D = 1,
                    TYPE_ROTATION_3D = 2,
                    TYPE_SCALE_3D = 3,
                    TYPE_TRANSFORM_3D = 4,
                    TYPE_BLEND_SHAPE = 5,
                    TYPE_COLOR = 6
                };

                TrackType type = TYPE_TRANSFORM_3D;
                std::string path;           // Node path or bone name
                std::string property;       // Property name for value tracks
                
                // Keyframes as tensors for batch evaluation
                xarray_container<double> times;      // N keyframe times
                xarray_container<double> values;     // N x D (D depends on type)
                
                // Interpolation method
                enum InterpMode { INTERP_NEAREST, INTERP_LINEAR, INTERP_CUBIC, INTERP_BEZIER };
                InterpMode interp_mode = INTERP_LINEAR;
                
                // Bezier control points (for cubic bezier)
                xarray_container<double> in_tangents;
                xarray_container<double> out_tangents;
                
                bool loop = false;
                float weight = 1.0f;
                bool enabled = true;

                // Evaluate track at given time (batch of times)
                xarray_container<double> evaluate_batch(const xarray_container<double>& t) const
                {
                    size_t n_times = t.size();
                    size_t dim = values.dimension() == 1 ? 1 : values.shape()[1];
                    xarray_container<double> result({n_times, dim});
                    
                    for (size_t i = 0; i < n_times; ++i)
                    {
                        double time = t(i);
                        if (loop && !times.empty())
                        {
                            double duration = times(times.size()-1) - times(0);
                            if (duration > 0)
                                time = times(0) + std::fmod(time - times(0), duration);
                        }
                        
                        // Find keyframe interval
                        size_t idx = 0;
                        while (idx + 1 < times.size() && times(idx+1) < time)
                            ++idx;
                        
                        if (idx + 1 >= times.size())
                        {
                            // Past end
                            for (size_t d = 0; d < dim; ++d)
                                result(i, d) = (values.dimension() == 1) ? values(times.size()-1) : values(times.size()-1, d);
                        }
                        else if (time <= times(0))
                        {
                            for (size_t d = 0; d < dim; ++d)
                                result(i, d) = (values.dimension() == 1) ? values(0) : values(0, d);
                        }
                        else
                        {
                            double t0 = times(idx);
                            double t1 = times(idx+1);
                            double alpha = (time - t0) / (t1 - t0);
                            
                            if (interp_mode == INTERP_NEAREST)
                            {
                                size_t use_idx = (alpha < 0.5) ? idx : idx+1;
                                for (size_t d = 0; d < dim; ++d)
                                    result(i, d) = (values.dimension() == 1) ? values(use_idx) : values(use_idx, d);
                            }
                            else if (interp_mode == INTERP_LINEAR)
                            {
                                for (size_t d = 0; d < dim; ++d)
                                {
                                    double v0 = (values.dimension() == 1) ? values(idx) : values(idx, d);
                                    double v1 = (values.dimension() == 1) ? values(idx+1) : values(idx+1, d);
                                    result(i, d) = v0 * (1.0 - alpha) + v1 * alpha;
                                }
                            }
                            else if (interp_mode == INTERP_CUBIC)
                            {
                                // Catmull-Rom spline
                                size_t p0_idx = (idx > 0) ? idx-1 : idx;
                                size_t p1_idx = idx;
                                size_t p2_idx = idx+1;
                                size_t p3_idx = (idx+2 < times.size()) ? idx+2 : idx+1;
                                double t_alpha = alpha;
                                double t2 = t_alpha * t_alpha;
                                double t3 = t2 * t_alpha;
                                for (size_t d = 0; d < dim; ++d)
                                {
                                    double v0 = (values.dimension() == 1) ? values(p0_idx) : values(p0_idx, d);
                                    double v1 = (values.dimension() == 1) ? values(p1_idx) : values(p1_idx, d);
                                    double v2 = (values.dimension() == 1) ? values(p2_idx) : values(p2_idx, d);
                                    double v3 = (values.dimension() == 1) ? values(p3_idx) : values(p3_idx, d);
                                    double result_val = 0.5 * ((2.0*v1) +
                                                      (-v0 + v2) * t_alpha +
                                                      (2.0*v0 - 5.0*v1 + 4.0*v2 - v3) * t2 +
                                                      (-v0 + 3.0*v1 - 3.0*v2 + v3) * t3);
                                    result(i, d) = result_val;
                                }
                            }
                            else // Bezier (simplified as linear)
                            {
                                for (size_t d = 0; d < dim; ++d)
                                {
                                    double v0 = (values.dimension() == 1) ? values(idx) : values(idx, d);
                                    double v1 = (values.dimension() == 1) ? values(idx+1) : values(idx+1, d);
                                    result(i, d) = v0 * (1.0 - alpha) + v1 * alpha;
                                }
                            }
                        }
                    }
                    
                    // Handle quaternion normalization for rotation tracks
                    if (type == TYPE_ROTATION_3D && dim == 4)
                    {
                        for (size_t i = 0; i < n_times; ++i)
                        {
                            double w = result(i,0), x = result(i,1), y = result(i,2), z = result(i,3);
                            double len = std::sqrt(w*w + x*x + y*y + z*z);
                            if (len > 1e-10)
                            {
                                result(i,0) /= len; result(i,1) /= len; result(i,2) /= len; result(i,3) /= len;
                            }
                        }
                    }
                    return result;
                }
            };

            // --------------------------------------------------------------------
            // XAnimation - Tensor-based animation resource
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XAnimation : public godot::Resource
            {
                GDCLASS(XAnimation, godot::Resource)

            private:
                std::vector<XAnimationTrack> m_tracks;
                double m_length = 0.0;
                double m_step = 0.1;
                bool m_loop = false;
                godot::String m_name;
                
                // Cached evaluated data
                godot::Ref<XTensorNode> m_cached_poses;
                godot::Ref<XTensorNode> m_cached_times;
                bool m_cache_dirty = true;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("add_track", "type", "path"), &XAnimation::add_track);
                    godot::ClassDB::bind_method(godot::D_METHOD("remove_track", "idx"), &XAnimation::remove_track);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_track_count"), &XAnimation::get_track_count);
                    godot::ClassDB::bind_method(godot::D_METHOD("track_insert_key", "track_idx", "time", "value"), &XAnimation::track_insert_key);
                    godot::ClassDB::bind_method(godot::D_METHOD("track_remove_key", "track_idx", "key_idx"), &XAnimation::track_remove_key);
                    godot::ClassDB::bind_method(godot::D_METHOD("track_get_key_count", "track_idx"), &XAnimation::track_get_key_count);
                    godot::ClassDB::bind_method(godot::D_METHOD("track_get_key_time", "track_idx", "key_idx"), &XAnimation::track_get_key_time);
                    godot::ClassDB::bind_method(godot::D_METHOD("track_get_key_value", "track_idx", "key_idx"), &XAnimation::track_get_key_value);
                    godot::ClassDB::bind_method(godot::D_METHOD("track_set_interpolation", "track_idx", "mode"), &XAnimation::track_set_interpolation);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("set_length", "length"), &XAnimation::set_length);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_length"), &XAnimation::get_length);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_loop", "loop"), &XAnimation::set_loop);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_loop"), &XAnimation::get_loop);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("evaluate", "time"), &XAnimation::evaluate);
                    godot::ClassDB::bind_method(godot::D_METHOD("evaluate_batch", "times"), &XAnimation::evaluate_batch);
                    godot::ClassDB::bind_method(godot::D_METHOD("sample_poses", "num_samples"), &XAnimation::sample_poses);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("clear"), &XAnimation::clear);
                    godot::ClassDB::bind_method(godot::D_METHOD("optimize"), &XAnimation::optimize);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "length"), "set_length", "get_length");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "loop"), "set_loop", "get_loop");
                    
                    ADD_SIGNAL(godot::MethodInfo("tracks_changed"));
                }

            public:
                XAnimation() {}
                
                int add_track(int type, const godot::String& path)
                {
                    XAnimationTrack track;
                    track.type = static_cast<XAnimationTrack::TrackType>(type);
                    track.path = path.utf8().get_data();
                    m_tracks.push_back(track);
                    m_cache_dirty = true;
                    emit_signal("tracks_changed");
                    return static_cast<int>(m_tracks.size() - 1);
                }
                
                void remove_track(int idx)
                {
                    if (idx >= 0 && idx < static_cast<int>(m_tracks.size()))
                    {
                        m_tracks.erase(m_tracks.begin() + idx);
                        m_cache_dirty = true;
                        emit_signal("tracks_changed");
                    }
                }
                
                int get_track_count() const { return static_cast<int>(m_tracks.size()); }
                
                void track_insert_key(int track_idx, double time, const godot::Variant& value)
                {
                    if (track_idx < 0 || track_idx >= static_cast<int>(m_tracks.size())) return;
                    auto& track = m_tracks[track_idx];
                    
                    // Parse value into tensor row
                    std::vector<double> vals;
                    if (value.get_type() == godot::Variant::FLOAT)
                    {
                        vals.push_back(static_cast<double>(value));
                    }
                    else if (value.get_type() == godot::Variant::VECTOR2)
                    {
                        godot::Vector2 v = value;
                        vals = {static_cast<double>(v.x), static_cast<double>(v.y)};
                    }
                    else if (value.get_type() == godot::Variant::VECTOR3)
                    {
                        godot::Vector3 v = value;
                        vals = {static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z)};
                    }
                    else if (value.get_type() == godot::Variant::QUATERNION)
                    {
                        godot::Quaternion q = value;
                        vals = {static_cast<double>(q.w), static_cast<double>(q.x), static_cast<double>(q.y), static_cast<double>(q.z)};
                    }
                    else if (value.get_type() == godot::Variant::TRANSFORM3D)
                    {
                        godot::Transform3D t = value;
                        godot::Vector3 pos = t.get_origin();
                        godot::Quaternion rot = t.get_basis().get_quaternion();
                        godot::Vector3 scale = t.get_basis().get_scale();
                        vals = {static_cast<double>(pos.x), static_cast<double>(pos.y), static_cast<double>(pos.z),
                                static_cast<double>(rot.w), static_cast<double>(rot.x), static_cast<double>(rot.y), static_cast<double>(rot.z),
                                static_cast<double>(scale.x), static_cast<double>(scale.y), static_cast<double>(scale.z)};
                    }
                    else if (value.get_type() == godot::Variant::ARRAY)
                    {
                        godot::Array arr = value;
                        for (int i = 0; i < arr.size(); ++i)
                            vals.push_back(static_cast<double>(arr[i]));
                    }
                    else
                    {
                        return;
                    }
                    
                    size_t dim = vals.size();
                    if (track.values.size() == 0)
                    {
                        track.values = xarray_container<double>({1, dim});
                    }
                    else
                    {
                        if (track.values.shape()[1] != dim)
                        {
                            godot::UtilityFunctions::printerr("Key value dimension mismatch");
                            return;
                        }
                    }
                    
                    // Insert time and value in sorted order
                    size_t insert_pos = 0;
                    while (insert_pos < track.times.size() && track.times(insert_pos) < time)
                        ++insert_pos;
                    
                    xarray_container<double> new_times({track.times.size() + 1});
                    xarray_container<double> new_values({track.values.shape()[0] + 1, dim});
                    
                    for (size_t i = 0; i < insert_pos; ++i)
                    {
                        new_times(i) = track.times(i);
                        for (size_t d = 0; d < dim; ++d)
                            new_values(i, d) = track.values(i, d);
                    }
                    new_times(insert_pos) = time;
                    for (size_t d = 0; d < dim; ++d)
                        new_values(insert_pos, d) = vals[d];
                    for (size_t i = insert_pos; i < track.times.size(); ++i)
                    {
                        new_times(i+1) = track.times(i);
                        for (size_t d = 0; d < dim; ++d)
                            new_values(i+1, d) = track.values(i, d);
                    }
                    
                    track.times = std::move(new_times);
                    track.values = std::move(new_values);
                    
                    // Update length
                    if (time > m_length) m_length = time;
                    m_cache_dirty = true;
                    emit_signal("tracks_changed");
                }
                
                void track_remove_key(int track_idx, int key_idx)
                {
                    if (track_idx < 0 || track_idx >= static_cast<int>(m_tracks.size())) return;
                    auto& track = m_tracks[track_idx];
                    if (key_idx < 0 || static_cast<size_t>(key_idx) >= track.times.size()) return;
                    
                    size_t dim = track.values.shape()[1];
                    xarray_container<double> new_times({track.times.size() - 1});
                    xarray_container<double> new_values({track.values.shape()[0] - 1, dim});
                    
                    size_t out = 0;
                    for (size_t i = 0; i < track.times.size(); ++i)
                    {
                        if (static_cast<int>(i) == key_idx) continue;
                        new_times(out) = track.times(i);
                        for (size_t d = 0; d < dim; ++d)
                            new_values(out, d) = track.values(i, d);
                        ++out;
                    }
                    track.times = std::move(new_times);
                    track.values = std::move(new_values);
                    m_cache_dirty = true;
                    emit_signal("tracks_changed");
                }
                
                int track_get_key_count(int track_idx) const
                {
                    if (track_idx >= 0 && track_idx < static_cast<int>(m_tracks.size()))
                        return static_cast<int>(m_tracks[track_idx].times.size());
                    return 0;
                }
                
                double track_get_key_time(int track_idx, int key_idx) const
                {
                    if (track_idx >= 0 && track_idx < static_cast<int>(m_tracks.size()))
                    {
                        const auto& track = m_tracks[track_idx];
                        if (key_idx >= 0 && static_cast<size_t>(key_idx) < track.times.size())
                            return track.times(key_idx);
                    }
                    return 0.0;
                }
                
                godot::Variant track_get_key_value(int track_idx, int key_idx) const
                {
                    if (track_idx >= 0 && track_idx < static_cast<int>(m_tracks.size()))
                    {
                        const auto& track = m_tracks[track_idx];
                        if (key_idx >= 0 && static_cast<size_t>(key_idx) < track.times.size())
                        {
                            size_t dim = track.values.shape()[1];
                            if (dim == 1)
                                return track.values(key_idx, 0);
                            else if (dim == 2)
                                return godot::Vector2(track.values(key_idx,0), track.values(key_idx,1));
                            else if (dim == 3)
                                return godot::Vector3(track.values(key_idx,0), track.values(key_idx,1), track.values(key_idx,2));
                            else if (dim == 4)
                                return godot::Quaternion(track.values(key_idx,0), track.values(key_idx,1), track.values(key_idx,2), track.values(key_idx,3));
                            else
                            {
                                godot::Array arr;
                                for (size_t d = 0; d < dim; ++d)
                                    arr.append(track.values(key_idx, d));
                                return arr;
                            }
                        }
                    }
                    return godot::Variant();
                }
                
                void track_set_interpolation(int track_idx, int mode)
                {
                    if (track_idx >= 0 && track_idx < static_cast<int>(m_tracks.size()))
                    {
                        m_tracks[track_idx].interp_mode = static_cast<XAnimationTrack::InterpMode>(mode);
                        m_cache_dirty = true;
                    }
                }
                
                void set_length(double length) { m_length = length; }
                double get_length() const { return m_length; }
                void set_loop(bool loop) { m_loop = loop; }
                bool get_loop() const { return m_loop; }
                
                godot::Dictionary evaluate(double time) const
                {
                    godot::Dictionary result;
                    if (m_tracks.empty()) return result;
                    
                    xarray_container<double> t({1}, time);
                    for (size_t i = 0; i < m_tracks.size(); ++i)
                    {
                        const auto& track = m_tracks[i];
                        auto vals = track.evaluate_batch(t);
                        
                        size_t dim = vals.shape()[1];
                        if (dim == 1)
                            result[godot::String(track.path.c_str())] = vals(0,0);
                        else if (dim == 2)
                            result[godot::String(track.path.c_str())] = godot::Vector2(vals(0,0), vals(0,1));
                        else if (dim == 3)
                            result[godot::String(track.path.c_str())] = godot::Vector3(vals(0,0), vals(0,1), vals(0,2));
                        else if (dim == 4)
                            result[godot::String(track.path.c_str())] = godot::Quaternion(vals(0,0), vals(0,1), vals(0,2), vals(0,3));
                        else if (dim == 10) // TRS
                        {
                            godot::Transform3D tform;
                            tform.set_origin(godot::Vector3(vals(0,0), vals(0,1), vals(0,2)));
                            tform.set_basis(godot::Basis(godot::Quaternion(vals(0,3), vals(0,4), vals(0,5), vals(0,6))));
                            tform.basis.scale(godot::Vector3(vals(0,7), vals(0,8), vals(0,9)));
                            result[godot::String(track.path.c_str())] = tform;
                        }
                    }
                    return result;
                }
                
                godot::Ref<XTensorNode> evaluate_batch(const godot::Ref<XTensorNode>& times) const
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!times.is_valid() || m_tracks.empty()) return result;
                    
                    auto t = times->get_tensor_resource()->m_data.to_double_array();
                    size_t n_times = t.size();
                    
                    // Collect all track results
                    std::vector<xarray_container<double>> track_results;
                    size_t total_dim = 0;
                    for (const auto& track : m_tracks)
                    {
                        auto vals = track.evaluate_batch(t);
                        track_results.push_back(vals);
                        total_dim += vals.shape()[1];
                    }
                    
                    xarray_container<double> combined({n_times, total_dim});
                    size_t offset = 0;
                    for (const auto& tr : track_results)
                    {
                        for (size_t i = 0; i < n_times; ++i)
                            for (size_t d = 0; d < tr.shape()[1]; ++d)
                                combined(i, offset + d) = tr(i, d);
                        offset += tr.shape()[1];
                    }
                    
                    result->set_data(XVariant::from_xarray(combined).variant());
                    return result;
                }
                
                godot::Ref<XTensorNode> sample_poses(int num_samples)
                {
                    if (m_cache_dirty || !m_cached_poses.is_valid())
                    {
                        xarray_container<double> times({static_cast<size_t>(num_samples)});
                        double duration = m_length;
                        if (duration <= 0) duration = 1.0;
                        for (int i = 0; i < num_samples; ++i)
                            times(i) = (static_cast<double>(i) / (num_samples - 1)) * duration;
                        
                        m_cached_times.instantiate();
                        m_cached_times->set_data(XVariant::from_xarray(times).variant());
                        m_cached_poses = evaluate_batch(m_cached_times);
                        m_cache_dirty = false;
                    }
                    else if (num_samples != static_cast<int>(m_cached_times->size()))
                    {
                        // Recompute with new sample count
                        xarray_container<double> times({static_cast<size_t>(num_samples)});
                        double duration = m_length;
                        if (duration <= 0) duration = 1.0;
                        for (int i = 0; i < num_samples; ++i)
                            times(i) = (static_cast<double>(i) / (num_samples - 1)) * duration;
                        
                        m_cached_times->set_data(XVariant::from_xarray(times).variant());
                        m_cached_poses = evaluate_batch(m_cached_times);
                    }
                    return m_cached_poses;
                }
                
                void clear()
                {
                    m_tracks.clear();
                    m_length = 0.0;
                    m_cache_dirty = true;
                    emit_signal("tracks_changed");
                }
                
                void optimize()
                {
                    // Remove redundant keyframes (simplified)
                    for (auto& track : m_tracks)
                    {
                        if (track.times.size() < 3) continue;
                        size_t dim = track.values.shape()[1];
                        std::vector<size_t> keep;
                        keep.push_back(0);
                        for (size_t i = 1; i < track.times.size() - 1; ++i)
                        {
                            // Check if keyframe is collinear with neighbors
                            double t0 = track.times(i-1);
                            double t1 = track.times(i);
                            double t2 = track.times(i+1);
                            double alpha = (t1 - t0) / (t2 - t0);
                            bool redundant = true;
                            for (size_t d = 0; d < dim; ++d)
                            {
                                double v0 = track.values(i-1, d);
                                double v1 = track.values(i, d);
                                double v2 = track.values(i+1, d);
                                double expected = v0 * (1.0 - alpha) + v2 * alpha;
                                if (std::abs(v1 - expected) > 1e-4)
                                {
                                    redundant = false;
                                    break;
                                }
                            }
                            if (!redundant)
                                keep.push_back(i);
                        }
                        keep.push_back(track.times.size() - 1);
                        
                        if (keep.size() < track.times.size())
                        {
                            xarray_container<double> new_times({keep.size()});
                            xarray_container<double> new_values({keep.size(), dim});
                            for (size_t j = 0; j < keep.size(); ++j)
                            {
                                new_times(j) = track.times(keep[j]);
                                for (size_t d = 0; d < dim; ++d)
                                    new_values(j, d) = track.values(keep[j], d);
                            }
                            track.times = std::move(new_times);
                            track.values = std::move(new_values);
                        }
                    }
                    m_cache_dirty = true;
                }
            };

            // --------------------------------------------------------------------
            // XAnimationPlayer - Tensor-based animation player node
            // --------------------------------------------------------------------
            class XAnimationPlayer : public godot::Node
            {
                GDCLASS(XAnimationPlayer, godot::Node)

            private:
                godot::Ref<XAnimation> m_animation;
                godot::Ref<XTensorNode> m_current_pose;
                double m_current_time = 0.0;
                double m_speed_scale = 1.0;
                bool m_playing = false;
                bool m_loop = false;
                godot::NodePath m_root_node;
                std::map<std::string, godot::NodePath> m_bone_paths;
                bool m_auto_apply = true;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_animation", "animation"), &XAnimationPlayer::set_animation);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_animation"), &XAnimationPlayer::get_animation);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_current_pose", "pose"), &XAnimationPlayer::set_current_pose);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_current_pose"), &XAnimationPlayer::get_current_pose);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_root_node", "path"), &XAnimationPlayer::set_root_node);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_root_node"), &XAnimationPlayer::get_root_node);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("play", "speed"), &XAnimationPlayer::play, godot::DEFVAL(1.0));
                    godot::ClassDB::bind_method(godot::D_METHOD("stop"), &XAnimationPlayer::stop);
                    godot::ClassDB::bind_method(godot::D_METHOD("pause"), &XAnimationPlayer::pause);
                    godot::ClassDB::bind_method(godot::D_METHOD("is_playing"), &XAnimationPlayer::is_playing);
                    godot::ClassDB::bind_method(godot::D_METHOD("seek", "time"), &XAnimationPlayer::seek);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_current_time"), &XAnimationPlayer::get_current_time);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_length"), &XAnimationPlayer::get_length);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_speed_scale", "scale"), &XAnimationPlayer::set_speed_scale);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_speed_scale"), &XAnimationPlayer::get_speed_scale);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_loop", "loop"), &XAnimationPlayer::set_loop);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_loop"), &XAnimationPlayer::get_loop);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("apply_pose"), &XAnimationPlayer::apply_pose);
                    godot::ClassDB::bind_method(godot::D_METHOD("blend_pose", "other_pose", "weight"), &XAnimationPlayer::blend_pose);
                    godot::ClassDB::bind_method(godot::D_METHOD("sample_at_time", "time"), &XAnimationPlayer::sample_at_time);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "animation", godot::PROPERTY_HINT_RESOURCE_TYPE, "XAnimation"), "set_animation", "get_animation");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "current_pose", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_current_pose", "get_current_pose");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::NODE_PATH, "root_node"), "set_root_node", "get_root_node");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "speed_scale"), "set_speed_scale", "get_speed_scale");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "loop"), "set_loop", "get_loop");
                    
                    ADD_SIGNAL(godot::MethodInfo("animation_started"));
                    ADD_SIGNAL(godot::MethodInfo("animation_finished"));
                    ADD_SIGNAL(godot::MethodInfo("animation_changed"));
                }

            public:
                XAnimationPlayer() {}
                
                void _ready() override
                {
                    set_process(true);
                }
                
                void _process(double delta) override
                {
                    if (m_playing && m_animation.is_valid())
                    {
                        m_current_time += delta * m_speed_scale;
                        double length = m_animation->get_length();
                        if (length > 0)
                        {
                            if (m_current_time >= length)
                            {
                                if (m_loop)
                                {
                                    m_current_time = std::fmod(m_current_time, length);
                                }
                                else
                                {
                                    m_current_time = length;
                                    m_playing = false;
                                    emit_signal("animation_finished");
                                }
                            }
                            update_pose();
                        }
                    }
                }

                void set_animation(const godot::Ref<XAnimation>& anim)
                {
                    m_animation = anim;
                    if (m_animation.is_valid())
                        seek(0.0);
                }
                
                godot::Ref<XAnimation> get_animation() const { return m_animation; }
                
                void set_current_pose(const godot::Ref<XTensorNode>& pose) { m_current_pose = pose; }
                godot::Ref<XTensorNode> get_current_pose() const { return m_current_pose; }
                
                void set_root_node(const godot::NodePath& path) { m_root_node = path; }
                godot::NodePath get_root_node() const { return m_root_node; }
                
                void play(double speed)
                {
                    m_speed_scale = speed;
                    m_playing = true;
                    emit_signal("animation_started");
                }
                
                void stop()
                {
                    m_playing = false;
                    m_current_time = 0.0;
                }
                
                void pause()
                {
                    m_playing = false;
                }
                
                bool is_playing() const { return m_playing; }
                
                void seek(double time)
                {
                    m_current_time = time;
                    update_pose();
                }
                
                double get_current_time() const { return m_current_time; }
                
                double get_length() const
                {
                    return m_animation.is_valid() ? m_animation->get_length() : 0.0;
                }
                
                void set_speed_scale(double scale) { m_speed_scale = scale; }
                double get_speed_scale() const { return m_speed_scale; }
                
                void set_loop(bool loop) { m_loop = loop; }
                bool get_loop() const { return m_loop; }
                
                void apply_pose()
                {
                    if (!m_current_pose.is_valid()) return;
                    
                    godot::Node* root = get_node_or_null(m_root_node);
                    if (!root) root = this;
                    
                    // Parse pose tensor and apply to nodes/bones
                    auto pose_data = m_current_pose->get_tensor_resource()->m_data.to_double_array();
                    // Format depends on how we packed it; for simplicity assume we have a mapping
                    
                    // If root is a Skeleton3D, apply bone transforms
                    godot::Skeleton3D* skel = godot::Object::cast_to<godot::Skeleton3D>(root);
                    if (skel)
                    {
                        size_t num_bones = static_cast<size_t>(skel->get_bone_count());
                        // Assume pose_data is N x 10 (TRS)
                        for (size_t i = 0; i < std::min(num_bones, pose_data.shape()[0]); ++i)
                        {
                            godot::Vector3 pos(pose_data(i,0), pose_data(i,1), pose_data(i,2));
                            godot::Quaternion rot(pose_data(i,3), pose_data(i,4), pose_data(i,5), pose_data(i,6));
                            godot::Vector3 scale(pose_data(i,7), pose_data(i,8), pose_data(i,9));
                            
                            skel->set_bone_pose_position(static_cast<int>(i), pos);
                            skel->set_bone_pose_rotation(static_cast<int>(i), rot);
                            skel->set_bone_pose_scale(static_cast<int>(i), scale);
                        }
                    }
                    
                    emit_signal("animation_changed");
                }
                
                void blend_pose(const godot::Ref<XTensorNode>& other_pose, double weight)
                {
                    if (!m_current_pose.is_valid() || !other_pose.is_valid()) return;
                    
                    auto a = m_current_pose->get_tensor_resource()->m_data.to_double_array();
                    auto b = other_pose->get_tensor_resource()->m_data.to_double_array();
                    
                    if (a.shape() != b.shape()) return;
                    
                    // Simple linear blend for positions/scales, slerp for rotations
                    xarray_container<double> blended = a * (1.0 - weight) + b * weight;
                    
                    // Handle quaternion normalization for rotation components (every 10th element starting at 3)
                    for (size_t i = 0; i < blended.shape()[0]; ++i)
                    {
                        if (blended.shape()[1] >= 7)
                        {
                            double w = blended(i,3), x = blended(i,4), y = blended(i,5), z = blended(i,6);
                            double len = std::sqrt(w*w + x*x + y*y + z*z);
                            if (len > 1e-10)
                            {
                                blended(i,3) /= len; blended(i,4) /= len; blended(i,5) /= len; blended(i,6) /= len;
                            }
                        }
                    }
                    
                    m_current_pose->set_data(XVariant::from_xarray(blended).variant());
                    if (m_auto_apply) apply_pose();
                }
                
                godot::Ref<XTensorNode> sample_at_time(double time) const
                {
                    if (!m_animation.is_valid())
                        return XTensorNode::create_zeros(godot::PackedInt64Array());
                    
                    xarray_container<double> t({1}, time);
                    return m_animation->evaluate_batch(
                        godot::Ref<XTensorNode>(const_cast<XTensorNode*>(XTensorNode::create_zeros({1}).ptr())));
                }

            private:
                void update_pose()
                {
                    if (!m_animation.is_valid())
                    {
                        m_current_pose.unref();
                        return;
                    }
                    
                    xarray_container<double> t({1}, m_current_time);
                    auto times_node = XTensorNode::create_zeros({1});
                    times_node->set_data(XVariant::from_xarray(t).variant());
                    m_current_pose = m_animation->evaluate_batch(times_node);
                    
                    if (m_auto_apply)
                        apply_pose();
                }
            };

            // --------------------------------------------------------------------
            // XBlendTree - Tensor-based blend tree for animation blending
            // --------------------------------------------------------------------
            class XBlendTree : public godot::Resource
            {
                GDCLASS(XBlendTree, godot::Resource)

            public:
                enum BlendType
                {
                    BLEND_LINEAR = 0,
                    BLEND_ADDITIVE = 1
                };

                struct BlendNode
                {
                    godot::Ref<XAnimation> animation;
                    double threshold = 0.0;
                    BlendType blend_type = BLEND_LINEAR;
                };

            private:
                std::vector<BlendNode> m_nodes;
                godot::String m_parameter_name = "blend";
                bool m_normalize = true;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("add_node", "animation", "threshold"), &XBlendTree::add_node);
                    godot::ClassDB::bind_method(godot::D_METHOD("remove_node", "idx"), &XBlendTree::remove_node);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_node_count"), &XBlendTree::get_node_count);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_parameter_name", "name"), &XBlendTree::set_parameter_name);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_parameter_name"), &XBlendTree::get_parameter_name);
                    godot::ClassDB::bind_method(godot::D_METHOD("evaluate", "parameter"), &XBlendTree::evaluate);
                    godot::ClassDB::bind_method(godot::D_METHOD("evaluate_batch", "parameters"), &XBlendTree::evaluate_batch);
                }

            public:
                void add_node(const godot::Ref<XAnimation>& anim, double threshold)
                {
                    m_nodes.push_back({anim, threshold});
                }
                
                void remove_node(int idx)
                {
                    if (idx >= 0 && idx < static_cast<int>(m_nodes.size()))
                        m_nodes.erase(m_nodes.begin() + idx);
                }
                
                int get_node_count() const { return static_cast<int>(m_nodes.size()); }
                
                void set_parameter_name(const godot::String& name) { m_parameter_name = name; }
                godot::String get_parameter_name() const { return m_parameter_name; }
                
                godot::Dictionary evaluate(double parameter) const
                {
                    if (m_nodes.empty()) return godot::Dictionary();
                    
                    // Find two closest thresholds
                    int idx0 = 0, idx1 = 0;
                    double weight0 = 1.0, weight1 = 0.0;
                    
                    for (size_t i = 0; i < m_nodes.size(); ++i)
                    {
                        if (m_nodes[i].threshold <= parameter)
                            idx0 = static_cast<int>(i);
                    }
                    for (int i = static_cast<int>(m_nodes.size()) - 1; i >= 0; --i)
                    {
                        if (m_nodes[i].threshold >= parameter)
                            idx1 = i;
                    }
                    
                    if (idx0 == idx1)
                    {
                        return m_nodes[idx0].animation->evaluate(0.0); // evaluate at time 0 for pose
                    }
                    
                    double t0 = m_nodes[idx0].threshold;
                    double t1 = m_nodes[idx1].threshold;
                    double alpha = (parameter - t0) / (t1 - t0);
                    
                    // Blend the two animations
                    auto pose0 = m_nodes[idx0].animation->evaluate(0.0);
                    auto pose1 = m_nodes[idx1].animation->evaluate(0.0);
                    
                    godot::Dictionary result;
                    // Merge keys (simplified)
                    return result;
                }
                
                godot::Ref<XTensorNode> evaluate_batch(const godot::Ref<XTensorNode>& parameters) const
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    // Batch blend tree evaluation
                    return result;
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XAnimationRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XAnimation>();
                    godot::ClassDB::register_class<XAnimationPlayer>();
                    godot::ClassDB::register_class<XBlendTree>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::XAnimationTrack;
        using godot_bridge::XAnimation;
        using godot_bridge::XAnimationPlayer;
        using godot_bridge::XBlendTree;
        using godot_bridge::XAnimationRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XANIMATION_HPP

// godot/xanimation.hpp