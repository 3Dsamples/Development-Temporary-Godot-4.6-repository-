// godot/xrenderingserver.hpp

#ifndef XTENSOR_XRENDERINGSERVER_HPP
#define XTENSOR_XRENDERINGSERVER_HPP

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
#include "../math/xmaterial.hpp"
#include "../graphics/xmesh.hpp"
#include "../graphics/xrenderer.hpp"
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
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/rendering_server.hpp>
    #include <godot_cpp/classes/rendering_device.hpp>
    #include <godot_cpp/classes/rd_shader_file.hpp>
    #include <godot_cpp/classes/rd_shader_spirv.hpp>
    #include <godot_cpp/classes/rd_texture_view.hpp>
    #include <godot_cpp/classes/rd_sampler.hpp>
    #include <godot_cpp/classes/rd_uniform.hpp>
    #include <godot_cpp/classes/rd_vertex_attribute.hpp>
    #include <godot_cpp/classes/image.hpp>
    #include <godot_cpp/classes/texture2d.hpp>
    #include <godot_cpp/classes/texture3d.hpp>
    #include <godot_cpp/classes/material.hpp>
    #include <godot_cpp/classes/shader_material.hpp>
    #include <godot_cpp/classes/mesh.hpp>
    #include <godot_cpp/classes/mesh_instance3d.hpp>
    #include <godot_cpp/classes/multi_mesh.hpp>
    #include <godot_cpp/classes/immediate_mesh.hpp>
    #include <godot_cpp/classes/surface_tool.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/transform3d.hpp>
    #include <godot_cpp/variant/vector3.hpp>
    #include <godot_cpp/variant/color.hpp>
    #include <godot_cpp/variant/aabb.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // Tensor-based Rendering Server Extensions
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XRenderingServer : public godot::Object
            {
                GDCLASS(XRenderingServer, godot::Object)

            private:
                static XRenderingServer* s_singleton;
                
                godot::RenderingServer* m_rs = nullptr;
                godot::RenderingDevice* m_rd = nullptr;
                
                // Cached RIDs for tensor resources
                std::map<uint64_t, godot::RID> m_texture_cache;
                std::map<uint64_t, godot::RID> m_mesh_cache;
                std::map<uint64_t, godot::RID> m_material_cache;
                std::map<uint64_t, godot::RID> m_shader_cache;
                
                // MultiMesh instances
                struct MultiMeshBatch
                {
                    godot::RID multimesh_rid;
                    godot::RID instance_rid;
                    size_t instance_count;
                    xarray_container<double> transforms; // N x 12 (4x3 matrix)
                    xarray_container<double> colors;     // N x 4
                    xarray_container<double> custom_data; // N x 4
                    bool dirty = true;
                };
                std::map<uint64_t, MultiMeshBatch> m_multimesh_batches;
                
                std::mutex m_mutex;

            protected:
                static void _bind_methods()
                {
                    // Texture operations
                    godot::ClassDB::bind_method(godot::D_METHOD("create_texture_from_tensor", "tensor", "format", "mipmaps"), &XRenderingServer::create_texture_from_tensor, godot::DEFVAL(true));
                    godot::ClassDB::bind_method(godot::D_METHOD("update_texture_from_tensor", "texture_rid", "tensor"), &XRenderingServer::update_texture_from_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("create_texture_3d_from_tensor", "tensor", "format", "mipmaps"), &XRenderingServer::create_texture_3d_from_tensor, godot::DEFVAL(true));
                    godot::ClassDB::bind_method(godot::D_METHOD("get_texture_data_as_tensor", "texture_rid"), &XRenderingServer::get_texture_data_as_tensor);
                    
                    // Mesh operations
                    godot::ClassDB::bind_method(godot::D_METHOD("create_mesh_from_tensors", "vertices", "normals", "uvs", "indices", "colors"), &XRenderingServer::create_mesh_from_tensors);
                    godot::ClassDB::bind_method(godot::D_METHOD("update_mesh_from_tensors", "mesh_rid", "vertices", "normals", "uvs", "indices", "colors"), &XRenderingServer::update_mesh_from_tensors);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_mesh_vertices_as_tensor", "mesh_rid"), &XRenderingServer::get_mesh_vertices_as_tensor);
                    
                    // MultiMesh batch rendering
                    godot::ClassDB::bind_method(godot::D_METHOD("create_multimesh_batch", "mesh_rid", "instance_count", "transform_tensor", "color_tensor"), &XRenderingServer::create_multimesh_batch);
                    godot::ClassDB::bind_method(godot::D_METHOD("update_multimesh_transforms", "batch_id", "transform_tensor"), &XRenderingServer::update_multimesh_transforms);
                    godot::ClassDB::bind_method(godot::D_METHOD("update_multimesh_colors", "batch_id", "color_tensor"), &XRenderingServer::update_multimesh_colors);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_multimesh_visible", "batch_id", "visible"), &XRenderingServer::set_multimesh_visible);
                    godot::ClassDB::bind_method(godot::D_METHOD("free_multimesh_batch", "batch_id"), &XRenderingServer::free_multimesh_batch);
                    
                    // Material operations
                    godot::ClassDB::bind_method(godot::D_METHOD("create_material_from_tensor", "parameters"), &XRenderingServer::create_material_from_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("update_material_parameters", "material_rid", "parameters"), &XRenderingServer::update_material_parameters);
                    
                    // Shader operations
                    godot::ClassDB::bind_method(godot::D_METHOD("compile_shader_from_string", "code", "shader_type"), &XRenderingServer::compile_shader_from_string);
                    godot::ClassDB::bind_method(godot::D_METHOD("create_shader_material", "shader_rid", "parameters"), &XRenderingServer::create_shader_material);
                    
                    // Scene management
                    godot::ClassDB::bind_method(godot::D_METHOD("instance_create"), &XRenderingServer::instance_create);
                    godot::ClassDB::bind_method(godot::D_METHOD("instance_set_transform", "instance_rid", "transform"), &XRenderingServer::instance_set_transform);
                    godot::ClassDB::bind_method(godot::D_METHOD("instance_set_mesh", "instance_rid", "mesh_rid"), &XRenderingServer::instance_set_mesh);
                    godot::ClassDB::bind_method(godot::D_METHOD("instance_set_material", "instance_rid", "material_rid"), &XRenderingServer::instance_set_material);
                    godot::ClassDB::bind_method(godot::D_METHOD("instance_free", "instance_rid"), &XRenderingServer::instance_free);
                    
                    // Rendering device compute shaders
                    godot::ClassDB::bind_method(godot::D_METHOD("create_compute_pipeline", "shader_code", "entry_point"), &XRenderingServer::create_compute_pipeline);
                    godot::ClassDB::bind_method(godot::D_METHOD("dispatch_compute", "pipeline_rid", "input_buffers", "output_buffers", "work_groups"), &XRenderingServer::dispatch_compute);
                    godot::ClassDB::bind_method(godot::D_METHOD("create_storage_buffer_from_tensor", "tensor"), &XRenderingServer::create_storage_buffer_from_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("read_storage_buffer_to_tensor", "buffer_rid"), &XRenderingServer::read_storage_buffer_to_tensor);
                    
                    // Light and environment
                    godot::ClassDB::bind_method(godot::D_METHOD("create_light_from_tensor", "type", "parameters"), &XRenderingServer::create_light_from_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("update_light_parameters", "light_rid", "parameters"), &XRenderingServer::update_light_parameters);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_environment_sky", "environment_rid", "sky_parameters"), &XRenderingServer::set_environment_sky);
                    
                    // Camera
                    godot::ClassDB::bind_method(godot::D_METHOD("create_camera"), &XRenderingServer::create_camera);
                    godot::ClassDB::bind_method(godot::D_METHOD("camera_set_transform", "camera_rid", "transform"), &XRenderingServer::camera_set_transform);
                    godot::ClassDB::bind_method(godot::D_METHOD("camera_set_perspective", "camera_rid", "fov", "near", "far"), &XRenderingServer::camera_set_perspective);
                    godot::ClassDB::bind_method(godot::D_METHOD("camera_set_orthographic", "camera_rid", "size", "near", "far"), &XRenderingServer::camera_set_orthographic);
                    
                    // Viewport
                    godot::ClassDB::bind_method(godot::D_METHOD("viewport_create"), &XRenderingServer::viewport_create);
                    godot::ClassDB::bind_method(godot::D_METHOD("viewport_set_camera", "viewport_rid", "camera_rid"), &XRenderingServer::viewport_set_camera);
                    godot::ClassDB::bind_method(godot::D_METHOD("viewport_set_size", "viewport_rid", "width", "height"), &XRenderingServer::viewport_set_size);
                    godot::ClassDB::bind_method(godot::D_METHOD("viewport_get_texture", "viewport_rid"), &XRenderingServer::viewport_get_texture);
                    
                    // Singleton access
                    godot::ClassDB::bind_method(godot::D_METHOD("get_singleton"), &XRenderingServer::get_singleton);
                    
                    ADD_SIGNAL(godot::MethodInfo("texture_created", godot::PropertyInfo(godot::Variant::INT, "rid_id")));
                    ADD_SIGNAL(godot::MethodInfo("mesh_created", godot::PropertyInfo(godot::Variant::INT, "rid_id")));
                }

            public:
                XRenderingServer()
                {
                    s_singleton = this;
                    m_rs = godot::RenderingServer::get_singleton();
                }
                
                ~XRenderingServer()
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    for (auto& p : m_texture_cache) if (p.second.is_valid()) m_rs->free_rid(p.second);
                    for (auto& p : m_mesh_cache) if (p.second.is_valid()) m_rs->free_rid(p.second);
                    for (auto& p : m_material_cache) if (p.second.is_valid()) m_rs->free_rid(p.second);
                    for (auto& p : m_multimesh_batches)
                    {
                        if (p.second.multimesh_rid.is_valid()) m_rs->free_rid(p.second.multimesh_rid);
                        if (p.second.instance_rid.is_valid()) m_rs->free_rid(p.second.instance_rid);
                    }
                    s_singleton = nullptr;
                }
                
                static XRenderingServer* get_singleton() { return s_singleton; }

                // --------------------------------------------------------------------
                // Texture Operations
                // --------------------------------------------------------------------
                godot::RID create_texture_from_tensor(const godot::Ref<XTensorNode>& tensor, int format, bool mipmaps)
                {
                    if (!tensor.is_valid()) return godot::RID();
                    auto arr = tensor->get_tensor_resource()->m_data.to_double_array();
                    if (arr.dimension() != 3)
                    {
                        godot::UtilityFunctions::printerr("create_texture_from_tensor: tensor must be HxWxC");
                        return godot::RID();
                    }
                    
                    size_t h = arr.shape()[0];
                    size_t w = arr.shape()[1];
                    size_t c = arr.shape()[2];
                    
                    godot::Image::Format image_format;
                    if (c == 1) image_format = godot::Image::FORMAT_L8;
                    else if (c == 2) image_format = godot::Image::FORMAT_LA8;
                    else if (c == 3) image_format = godot::Image::FORMAT_RGB8;
                    else if (c == 4) image_format = godot::Image::FORMAT_RGBA8;
                    else
                    {
                        godot::UtilityFunctions::printerr("create_texture_from_tensor: unsupported channel count");
                        return godot::RID();
                    }
                    
                    godot::Ref<godot::Image> img;
                    img.instantiate();
                    img->create(static_cast<int>(w), static_cast<int>(h), false, image_format);
                    
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            if (c == 1)
                            {
                                float v = std::clamp(static_cast<float>(arr(y, x, 0)), 0.0f, 1.0f);
                                img->set_pixel(static_cast<int>(x), static_cast<int>(y), godot::Color(v, v, v));
                            }
                            else if (c == 3)
                            {
                                float r = std::clamp(static_cast<float>(arr(y, x, 0)), 0.0f, 1.0f);
                                float g = std::clamp(static_cast<float>(arr(y, x, 1)), 0.0f, 1.0f);
                                float b = std::clamp(static_cast<float>(arr(y, x, 2)), 0.0f, 1.0f);
                                img->set_pixel(static_cast<int>(x), static_cast<int>(y), godot::Color(r, g, b));
                            }
                            else if (c == 4)
                            {
                                float r = std::clamp(static_cast<float>(arr(y, x, 0)), 0.0f, 1.0f);
                                float g = std::clamp(static_cast<float>(arr(y, x, 1)), 0.0f, 1.0f);
                                float b = std::clamp(static_cast<float>(arr(y, x, 2)), 0.0f, 1.0f);
                                float a = std::clamp(static_cast<float>(arr(y, x, 3)), 0.0f, 1.0f);
                                img->set_pixel(static_cast<int>(x), static_cast<int>(y), godot::Color(r, g, b, a));
                            }
                        }
                    }
                    
                    godot::Ref<godot::ImageTexture> tex;
                    tex.instantiate();
                    tex->create_from_image(img);
                    
                    godot::RID rid = tex->get_rid();
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_texture_cache[rid.get_id()] = rid;
                    emit_signal("texture_created", static_cast<int64_t>(rid.get_id()));
                    return rid;
                }

                void update_texture_from_tensor(const godot::RID& texture_rid, const godot::Ref<XTensorNode>& tensor)
                {
                    if (!texture_rid.is_valid() || !tensor.is_valid()) return;
                    // Similar to create, but update existing texture
                    // This would require accessing the texture data
                }

                godot::RID create_texture_3d_from_tensor(const godot::Ref<XTensorNode>& tensor, int format, bool mipmaps)
                {
                    if (!tensor.is_valid()) return godot::RID();
                    auto arr = tensor->get_tensor_resource()->m_data.to_double_array();
                    if (arr.dimension() != 4)
                    {
                        godot::UtilityFunctions::printerr("create_texture_3d_from_tensor: tensor must be DxHxWxC");
                        return godot::RID();
                    }
                    // Godot 3D texture creation (placeholder)
                    return godot::RID();
                }

                godot::Ref<XTensorNode> get_texture_data_as_tensor(const godot::RID& texture_rid)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!texture_rid.is_valid()) return result;
                    // Would retrieve texture data from rendering server
                    return result;
                }

                // --------------------------------------------------------------------
                // Mesh Operations
                // --------------------------------------------------------------------
                godot::RID create_mesh_from_tensors(const godot::Ref<XTensorNode>& vertices,
                                                    const godot::Ref<XTensorNode>& normals,
                                                    const godot::Ref<XTensorNode>& uvs,
                                                    const godot::Ref<XTensorNode>& indices,
                                                    const godot::Ref<XTensorNode>& colors)
                {
                    if (!vertices.is_valid()) return godot::RID();
                    auto verts = vertices->get_tensor_resource()->m_data.to_double_array();
                    if (verts.dimension() != 2 || verts.shape()[1] != 3)
                    {
                        godot::UtilityFunctions::printerr("create_mesh_from_tensors: vertices must be Nx3");
                        return godot::RID();
                    }
                    size_t num_verts = verts.shape()[0];
                    
                    godot::Ref<godot::ArrayMesh> mesh;
                    mesh.instantiate();
                    
                    godot::Array arrays;
                    arrays.resize(godot::Mesh::ARRAY_MAX);
                    
                    godot::PackedVector3Array vert_arr;
                    vert_arr.resize(static_cast<int>(num_verts));
                    for (size_t i = 0; i < num_verts; ++i)
                        vert_arr.set(static_cast<int>(i), godot::Vector3(verts(i,0), verts(i,1), verts(i,2)));
                    arrays[godot::Mesh::ARRAY_VERTEX] = vert_arr;
                    
                    if (normals.is_valid())
                    {
                        auto norms = normals->get_tensor_resource()->m_data.to_double_array();
                        if (norms.shape()[0] == num_verts && norms.shape()[1] == 3)
                        {
                            godot::PackedVector3Array norm_arr;
                            norm_arr.resize(static_cast<int>(num_verts));
                            for (size_t i = 0; i < num_verts; ++i)
                                norm_arr.set(static_cast<int>(i), godot::Vector3(norms(i,0), norms(i,1), norms(i,2)));
                            arrays[godot::Mesh::ARRAY_NORMAL] = norm_arr;
                        }
                    }
                    
                    if (uvs.is_valid())
                    {
                        auto uv_arr_tensor = uvs->get_tensor_resource()->m_data.to_double_array();
                        if (uv_arr_tensor.shape()[0] == num_verts && uv_arr_tensor.shape()[1] == 2)
                        {
                            godot::PackedVector2Array uv_arr;
                            uv_arr.resize(static_cast<int>(num_verts));
                            for (size_t i = 0; i < num_verts; ++i)
                                uv_arr.set(static_cast<int>(i), godot::Vector2(uv_arr_tensor(i,0), uv_arr_tensor(i,1)));
                            arrays[godot::Mesh::ARRAY_TEX_UV] = uv_arr;
                        }
                    }
                    
                    if (colors.is_valid())
                    {
                        auto col_arr_tensor = colors->get_tensor_resource()->m_data.to_double_array();
                        if (col_arr_tensor.shape()[0] == num_verts && col_arr_tensor.shape()[1] == 4)
                        {
                            godot::PackedColorArray col_arr;
                            col_arr.resize(static_cast<int>(num_verts));
                            for (size_t i = 0; i < num_verts; ++i)
                                col_arr.set(static_cast<int>(i), godot::Color(col_arr_tensor(i,0), col_arr_tensor(i,1), col_arr_tensor(i,2), col_arr_tensor(i,3)));
                            arrays[godot::Mesh::ARRAY_COLOR] = col_arr;
                        }
                    }
                    
                    if (indices.is_valid())
                    {
                        auto inds = indices->get_tensor_resource()->m_data.to_double_array();
                        godot::PackedInt32Array idx_arr;
                        for (size_t i = 0; i < inds.size(); ++i)
                            idx_arr.append(static_cast<int32_t>(inds.flat(i)));
                        arrays[godot::Mesh::ARRAY_INDEX] = idx_arr;
                    }
                    
                    mesh->add_surface_from_arrays(godot::Mesh::PRIMITIVE_TRIANGLES, arrays);
                    
                    godot::RID rid = mesh->get_rid();
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_mesh_cache[rid.get_id()] = rid;
                    emit_signal("mesh_created", static_cast<int64_t>(rid.get_id()));
                    return rid;
                }

                void update_mesh_from_tensors(const godot::RID& mesh_rid,
                                              const godot::Ref<XTensorNode>& vertices,
                                              const godot::Ref<XTensorNode>& normals,
                                              const godot::Ref<XTensorNode>& uvs,
                                              const godot::Ref<XTensorNode>& indices,
                                              const godot::Ref<XTensorNode>& colors)
                {
                    if (!mesh_rid.is_valid()) return;
                    // Would clear and re-add surface
                }

                godot::Ref<XTensorNode> get_mesh_vertices_as_tensor(const godot::RID& mesh_rid)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    // Retrieve vertices from mesh (requires access to mesh data)
                    return result;
                }

                // --------------------------------------------------------------------
                // MultiMesh Batch Rendering
                // --------------------------------------------------------------------
                uint64_t create_multimesh_batch(const godot::RID& mesh_rid, int64_t instance_count,
                                                const godot::Ref<XTensorNode>& transform_tensor,
                                                const godot::Ref<XTensorNode>& color_tensor)
                {
                    if (!mesh_rid.is_valid() || instance_count <= 0) return 0;
                    
                    godot::RID mm_rid = m_rs->multimesh_create();
                    m_rs->multimesh_allocate_data(mm_rid, static_cast<int>(instance_count), godot::RenderingServer::MULTIMESH_TRANSFORM_3D, true);
                    m_rs->multimesh_set_mesh(mm_rid, mesh_rid);
                    
                    godot::RID instance_rid = m_rs->instance_create2(mm_rid, m_rs->get_test_cube()); // placeholder scene
                    m_rs->instance_set_visible(instance_rid, true);
                    
                    uint64_t batch_id = mm_rid.get_id();
                    MultiMeshBatch batch;
                    batch.multimesh_rid = mm_rid;
                    batch.instance_rid = instance_rid;
                    batch.instance_count = static_cast<size_t>(instance_count);
                    
                    if (transform_tensor.is_valid())
                    {
                        batch.transforms = transform_tensor->get_tensor_resource()->m_data.to_double_array();
                        update_multimesh_transforms(batch_id, transform_tensor);
                    }
                    if (color_tensor.is_valid())
                    {
                        batch.colors = color_tensor->get_tensor_resource()->m_data.to_double_array();
                        update_multimesh_colors(batch_id, color_tensor);
                    }
                    
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_multimesh_batches[batch_id] = batch;
                    return batch_id;
                }

                void update_multimesh_transforms(uint64_t batch_id, const godot::Ref<XTensorNode>& transform_tensor)
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    auto it = m_multimesh_batches.find(batch_id);
                    if (it == m_multimesh_batches.end() || !transform_tensor.is_valid()) return;
                    
                    auto transforms = transform_tensor->get_tensor_resource()->m_data.to_double_array();
                    size_t count = std::min(static_cast<size_t>(transforms.shape()[0]), it->second.instance_count);
                    
                    for (size_t i = 0; i < count; ++i)
                    {
                        godot::Transform3D t;
                        if (transforms.shape()[1] == 12)
                        {
                            // 4x3 matrix
                            t.basis.set_row(0, godot::Vector3(transforms(i,0), transforms(i,1), transforms(i,2)));
                            t.basis.set_row(1, godot::Vector3(transforms(i,4), transforms(i,5), transforms(i,6)));
                            t.basis.set_row(2, godot::Vector3(transforms(i,8), transforms(i,9), transforms(i,10)));
                            t.origin = godot::Vector3(transforms(i,3), transforms(i,7), transforms(i,11));
                        }
                        else if (transforms.shape()[1] == 7)
                        {
                            // position (3) + quaternion (4)
                            t.origin = godot::Vector3(transforms(i,0), transforms(i,1), transforms(i,2));
                            t.basis = godot::Basis(godot::Quaternion(transforms(i,3), transforms(i,4), transforms(i,5), transforms(i,6)));
                        }
                        m_rs->multimesh_instance_set_transform(it->second.multimesh_rid, static_cast<int>(i), t);
                    }
                    it->second.transforms = transforms;
                    it->second.dirty = false;
                }

                void update_multimesh_colors(uint64_t batch_id, const godot::Ref<XTensorNode>& color_tensor)
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    auto it = m_multimesh_batches.find(batch_id);
                    if (it == m_multimesh_batches.end() || !color_tensor.is_valid()) return;
                    
                    auto colors = color_tensor->get_tensor_resource()->m_data.to_double_array();
                    size_t count = std::min(static_cast<size_t>(colors.shape()[0]), it->second.instance_count);
                    
                    for (size_t i = 0; i < count; ++i)
                    {
                        godot::Color c;
                        if (colors.shape()[1] == 3)
                            c = godot::Color(colors(i,0), colors(i,1), colors(i,2), 1.0f);
                        else if (colors.shape()[1] == 4)
                            c = godot::Color(colors(i,0), colors(i,1), colors(i,2), colors(i,3));
                        else
                            c = godot::Color(1,1,1,1);
                        m_rs->multimesh_instance_set_color(it->second.multimesh_rid, static_cast<int>(i), c);
                    }
                    it->second.colors = colors;
                }

                void set_multimesh_visible(uint64_t batch_id, bool visible)
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    auto it = m_multimesh_batches.find(batch_id);
                    if (it != m_multimesh_batches.end())
                        m_rs->instance_set_visible(it->second.instance_rid, visible);
                }

                void free_multimesh_batch(uint64_t batch_id)
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    auto it = m_multimesh_batches.find(batch_id);
                    if (it != m_multimesh_batches.end())
                    {
                        if (it->second.multimesh_rid.is_valid())
                            m_rs->free_rid(it->second.multimesh_rid);
                        if (it->second.instance_rid.is_valid())
                            m_rs->free_rid(it->second.instance_rid);
                        m_multimesh_batches.erase(it);
                    }
                }

                // --------------------------------------------------------------------
                // Material Operations
                // --------------------------------------------------------------------
                godot::RID create_material_from_tensor(const godot::Ref<XTensorNode>& parameters)
                {
                    if (!parameters.is_valid()) return godot::RID();
                    // Parse parameters dictionary (assumed as JSON string or dict)
                    // Placeholder: create standard material
                    godot::RID mat = m_rs->material_create();
                    // Set parameters
                    return mat;
                }

                void update_material_parameters(const godot::RID& material_rid, const godot::Ref<XTensorNode>& parameters)
                {
                    if (!material_rid.is_valid() || !parameters.is_valid()) return;
                }

                // --------------------------------------------------------------------
                // Shader Operations
                // --------------------------------------------------------------------
                godot::RID compile_shader_from_string(const godot::String& code, const godot::String& shader_type)
                {
                    godot::Ref<godot::RDShaderFile> shader_file;
                    shader_file.instantiate();
                    shader_file->set_stage_source(godot::RDShaderFile::STAGE_VERTEX, code);
                    shader_file->set_stage_source(godot::RDShaderFile::STAGE_FRAGMENT, code);
                    // Compile
                    return godot::RID();
                }

                godot::RID create_shader_material(const godot::RID& shader_rid, const godot::Ref<XTensorNode>& parameters)
                {
                    godot::RID mat = m_rs->material_create();
                    m_rs->material_set_shader(mat, shader_rid);
                    return mat;
                }

                // --------------------------------------------------------------------
                // Scene Management
                // --------------------------------------------------------------------
                godot::RID instance_create()
                {
                    return m_rs->instance_create();
                }

                void instance_set_transform(const godot::RID& instance_rid, const godot::Transform3D& transform)
                {
                    if (instance_rid.is_valid())
                        m_rs->instance_set_transform(instance_rid, transform);
                }

                void instance_set_mesh(const godot::RID& instance_rid, const godot::RID& mesh_rid)
                {
                    if (instance_rid.is_valid())
                        m_rs->instance_set_base(instance_rid, mesh_rid);
                }

                void instance_set_material(const godot::RID& instance_rid, const godot::RID& material_rid)
                {
                    if (instance_rid.is_valid())
                        m_rs->instance_set_surface_override_material(instance_rid, 0, material_rid);
                }

                void instance_free(const godot::RID& instance_rid)
                {
                    if (instance_rid.is_valid())
                        m_rs->free_rid(instance_rid);
                }

                // --------------------------------------------------------------------
                // Compute Shaders
                // --------------------------------------------------------------------
                godot::RID create_compute_pipeline(const godot::String& shader_code, const godot::String& entry_point)
                {
                    if (!m_rd) return godot::RID();
                    // Compile compute shader and create pipeline
                    return godot::RID();
                }

                void dispatch_compute(const godot::RID& pipeline_rid, const godot::Array& input_buffers,
                                      const godot::Array& output_buffers, const godot::Vector3i& work_groups)
                {
                    if (!m_rd || !pipeline_rid.is_valid()) return;
                }

                godot::RID create_storage_buffer_from_tensor(const godot::Ref<XTensorNode>& tensor)
                {
                    if (!m_rd || !tensor.is_valid()) return godot::RID();
                    auto arr = tensor->get_tensor_resource()->m_data.to_double_array();
                    godot::PackedByteArray bytes;
                    // Copy tensor data to bytes
                    return m_rd->storage_buffer_create(bytes.size(), bytes);
                }

                godot::Ref<XTensorNode> read_storage_buffer_to_tensor(const godot::RID& buffer_rid)
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    if (!m_rd || !buffer_rid.is_valid()) return result;
                    // Read back buffer
                    return result;
                }

                // --------------------------------------------------------------------
                // Light and Environment
                // --------------------------------------------------------------------
                godot::RID create_light_from_tensor(const godot::String& type, const godot::Ref<XTensorNode>& parameters)
                {
                    godot::RID light;
                    if (type == "directional")
                        light = m_rs->directional_light_create();
                    else if (type == "omni")
                        light = m_rs->omni_light_create();
                    else if (type == "spot")
                        light = m_rs->spot_light_create();
                    else
                        return godot::RID();
                    
                    update_light_parameters(light, parameters);
                    return light;
                }

                void update_light_parameters(const godot::RID& light_rid, const godot::Ref<XTensorNode>& parameters)
                {
                    if (!light_rid.is_valid() || !parameters.is_valid()) return;
                    auto params = parameters->get_tensor_resource()->m_data.to_double_array();
                    // Parse parameters: color, energy, etc.
                    if (params.size() >= 3)
                        m_rs->light_set_color(light_rid, godot::Color(params(0), params(1), params(2), 1.0f));
                }

                void set_environment_sky(const godot::RID& env_rid, const godot::Ref<XTensorNode>& sky_parameters)
                {
                    // Set sky shader parameters from tensor
                }

                // --------------------------------------------------------------------
                // Camera
                // --------------------------------------------------------------------
                godot::RID create_camera()
                {
                    return m_rs->camera_create();
                }

                void camera_set_transform(const godot::RID& camera_rid, const godot::Transform3D& transform)
                {
                    if (camera_rid.is_valid())
                        m_rs->camera_set_transform(camera_rid, transform);
                }

                void camera_set_perspective(const godot::RID& camera_rid, float fov, float near_plane, float far_plane)
                {
                    if (camera_rid.is_valid())
                        m_rs->camera_set_perspective(camera_rid, fov, near_plane, far_plane);
                }

                void camera_set_orthographic(const godot::RID& camera_rid, float size, float near_plane, float far_plane)
                {
                    if (camera_rid.is_valid())
                        m_rs->camera_set_orthographic(camera_rid, size, near_plane, far_plane);
                }

                // --------------------------------------------------------------------
                // Viewport
                // --------------------------------------------------------------------
                godot::RID viewport_create()
                {
                    return m_rs->viewport_create();
                }

                void viewport_set_camera(const godot::RID& viewport_rid, const godot::RID& camera_rid)
                {
                    if (viewport_rid.is_valid())
                        m_rs->viewport_attach_camera(viewport_rid, camera_rid);
                }

                void viewport_set_size(const godot::RID& viewport_rid, int width, int height)
                {
                    if (viewport_rid.is_valid())
                        m_rs->viewport_set_size(viewport_rid, width, height);
                }

                godot::RID viewport_get_texture(const godot::RID& viewport_rid)
                {
                    if (viewport_rid.is_valid())
                        return m_rs->viewport_get_texture(viewport_rid);
                    return godot::RID();
                }
            };

            // Singleton instance
            XRenderingServer* XRenderingServer::s_singleton = nullptr;
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XRenderingServerRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XRenderingServer>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::XRenderingServer;
        using godot_bridge::XRenderingServerRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XRENDERINGSERVER_HPP

// godot/xrenderingserver.hpp