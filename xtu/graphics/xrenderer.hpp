// graphics/xrenderer.hpp

#ifndef XTENSOR_XRENDERER_HPP
#define XTENSOR_XRENDERER_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../math/xnorm.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xstats.hpp"
#include "../math/xrandom.hpp"
#include "../math/xintersection.hpp"
#include "../math/xmaterial.hpp"
#include "xmesh.hpp"
#include "xgraphics.hpp"

#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <map>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <set>
#include <queue>
#include <limits>
#include <tuple>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace renderer
        {
            using namespace mesh;
            using namespace material;
            using namespace intersection;

            // --------------------------------------------------------------------
            // Forward declarations
            // --------------------------------------------------------------------
            class Scene;
            class Camera;
            class Light;
            class Material;
            class Texture;
            class Framebuffer;
            class Shader;
            class Renderer;

            // --------------------------------------------------------------------
            // Basic types
            // --------------------------------------------------------------------
            using Color3f = Vector3<float>;
            using Color4f = Vector4<float>;

            struct Ray
            {
                Vector3f origin;
                Vector3f direction;
                float t_min = 0.001f;
                float t_max = std::numeric_limits<float>::max();
                int depth = 0;
            };

            // --------------------------------------------------------------------
            // Texture class (2D image)
            // --------------------------------------------------------------------
            class Texture
            {
            public:
                enum WrapMode { Repeat, Clamp, Mirror };
                enum FilterMode { Nearest, Bilinear, Trilinear, Anisotropic };
                enum Format { RGB8, RGBA8, RGBA32F, Depth32F };

            private:
                xarray_container<float> m_data; // H x W x C
                size_t m_width = 0;
                size_t m_height = 0;
                size_t m_channels = 4;
                Format m_format = RGBA8;
                WrapMode m_wrap_s = Repeat;
                WrapMode m_wrap_t = Repeat;
                FilterMode m_filter_min = Bilinear;
                FilterMode m_filter_mag = Bilinear;
                bool m_mipmaps = false;
                std::vector<xarray_container<float>> m_mip_levels;

            public:
                Texture() = default;
                Texture(size_t w, size_t h, Format fmt = RGBA8)
                    : m_width(w), m_height(h), m_format(fmt)
                {
                    m_channels = (fmt == RGB8) ? 3 : (fmt == Depth32F ? 1 : 4);
                    m_data = xt::zeros<float>({h, w, m_channels});
                }

                void set_data(const xarray_container<float>& data)
                {
                    if (data.dimension() == 3)
                    {
                        m_data = data;
                        m_height = data.shape()[0];
                        m_width = data.shape()[1];
                        m_channels = data.shape()[2];
                    }
                    else if (data.dimension() == 2)
                    {
                        m_data = xt::view(data, xt::all(), xt::all(), xt::newaxis());
                        m_height = data.shape()[0];
                        m_width = data.shape()[1];
                        m_channels = 1;
                    }
                }

                Color4f sample(const Vector2f& uv, float lod = 0.0f) const
                {
                    if (m_data.size() == 0) return {0,0,0,1};
                    float u = uv.x, v = uv.y;
                    if (m_wrap_s == Repeat) u = u - std::floor(u);
                    else if (m_wrap_s == Clamp) u = std::clamp(u, 0.0f, 1.0f);
                    else if (m_wrap_s == Mirror) { u = std::abs(u); int i = static_cast<int>(u); u = (i%2==0) ? (u-i) : (1-(u-i)); }
                    if (m_wrap_t == Repeat) v = v - std::floor(v);
                    else if (m_wrap_t == Clamp) v = std::clamp(v, 0.0f, 1.0f);
                    else if (m_wrap_t == Mirror) { v = std::abs(v); int i = static_cast<int>(v); v = (i%2==0) ? (v-i) : (1-(v-i)); }

                    float x = u * m_width - 0.5f;
                    float y = v * m_height - 0.5f;
                    if (m_filter_mag == Nearest)
                    {
                        size_t ix = static_cast<size_t>(std::clamp(x+0.5f, 0.0f, (float)m_width-1));
                        size_t iy = static_cast<size_t>(std::clamp(y+0.5f, 0.0f, (float)m_height-1));
                        size_t idx = (iy * m_width + ix) * m_channels;
                        Color4f col(0,0,0,1);
                        if (m_channels >= 1) col.x = m_data.data()[idx];
                        if (m_channels >= 2) col.y = m_data.data()[idx+1];
                        if (m_channels >= 3) col.z = m_data.data()[idx+2];
                        if (m_channels >= 4) col.w = m_data.data()[idx+3];
                        return col;
                    }
                    else // Bilinear
                    {
                        int x0 = static_cast<int>(std::floor(x));
                        int y0 = static_cast<int>(std::floor(y));
                        int x1 = x0 + 1, y1 = y0 + 1;
                        float fx = x - x0, fy = y - y0;
                        x0 = std::clamp(x0, 0, (int)m_width-1);
                        x1 = std::clamp(x1, 0, (int)m_width-1);
                        y0 = std::clamp(y0, 0, (int)m_height-1);
                        y1 = std::clamp(y1, 0, (int)m_height-1);
                        auto get = [&](int px, int py, int c) {
                            size_t idx = (py * m_width + px) * m_channels + c;
                            return m_data.data()[idx];
                        };
                        Color4f c00, c01, c10, c11;
                        for (int c=0; c<4 && c<(int)m_channels; ++c)
                        {
                            float v00 = get(x0,y0,c), v01 = get(x0,y1,c), v10 = get(x1,y0,c), v11 = get(x1,y1,c);
                            float v0 = v00*(1-fx) + v10*fx;
                            float v1 = v01*(1-fx) + v11*fx;
                            (&c00.x)[c] = v0*(1-fy) + v1*fy;
                        }
                        return c00;
                    }
                }

                void generate_mipmaps()
                {
                    m_mipmaps = true;
                    m_mip_levels.clear();
                    m_mip_levels.push_back(m_data);
                    size_t w = m_width, h = m_height;
                    while (w > 1 || h > 1)
                    {
                        w = std::max(1UL, w/2);
                        h = std::max(1UL, h/2);
                        xarray_container<float> level = xt::zeros<float>({h, w, m_channels});
                        // Simple box filter downsampling
                        for (size_t y=0; y<h; ++y)
                        {
                            for (size_t x=0; x<w; ++x)
                            {
                                for (size_t c=0; c<m_channels; ++c)
                                {
                                    float sum = 0; int cnt = 0;
                                    for (int dy=0; dy<2; ++dy)
                                        for (int dx=0; dx<2; ++dx)
                                        {
                                            size_t sx = x*2+dx, sy = y*2+dy;
                                            if (sx < m_mip_levels.back().shape()[1] && sy < m_mip_levels.back().shape()[0])
                                            {
                                                sum += m_mip_levels.back()(sy, sx, c);
                                                cnt++;
                                            }
                                        }
                                    level(y, x, c) = sum / cnt;
                                }
                            }
                        }
                        m_mip_levels.push_back(level);
                    }
                }

                size_t width() const { return m_width; }
                size_t height() const { return m_height; }
                Format format() const { return m_format; }
    };

            // --------------------------------------------------------------------
            // Camera class
            // --------------------------------------------------------------------
            class Camera
            {
            public:
                Vector3f position = {0,0,5};
                Vector3f target = {0,0,0};
                Vector3f up = {0,1,0};
                float fov = 60.0f * M_PI / 180.0f;
                float aspect = 16.0f / 9.0f;
                float near_plane = 0.1f;
                float far_plane = 1000.0f;
                bool orthographic = false;
                float ortho_size = 5.0f;

                Vector3f get_direction() const { return (target - position).normalized(); }
                Vector3f get_right() const { return get_direction().cross(up).normalized(); }
                Vector3f get_up() const { return get_right().cross(get_direction()).normalized(); }

                Ray generate_ray(float ndc_x, float ndc_y) const
                {
                    if (orthographic)
                    {
                        float hw = ortho_size * aspect * 0.5f;
                        float hh = ortho_size * 0.5f;
                        Vector3f origin = position + get_right() * (ndc_x * hw) + get_up() * (ndc_y * hh);
                        return {origin, get_direction()};
                    }
                    else
                    {
                        float tan_fov = std::tan(fov * 0.5f);
                        Vector3f right = get_right() * (tan_fov * aspect);
                        Vector3f up_vec = get_up() * tan_fov;
                        Vector3f dir = (get_direction() + right * ndc_x + up_vec * ndc_y).normalized();
                        return {position, dir};
                    }
                }

                Matrix4 get_view_matrix() const
                {
                    Vector3f z = (position - target).normalized();
                    Vector3f x = up.cross(z).normalized();
                    Vector3f y = z.cross(x);
                    Matrix4 view;
                    view.m[0]=x.x; view.m[1]=y.x; view.m[2]=z.x; view.m[3]=0;
                    view.m[4]=x.y; view.m[5]=y.y; view.m[6]=z.y; view.m[7]=0;
                    view.m[8]=x.z; view.m[9]=y.z; view.m[10]=z.z; view.m[11]=0;
                    view.m[12]= -x.dot(position); view.m[13]= -y.dot(position); view.m[14]= -z.dot(position); view.m[15]=1;
                    return view;
                }

                Matrix4 get_projection_matrix() const
                {
                    Matrix4 proj;
                    if (orthographic)
                    {
                        float h = ortho_size * 0.5f;
                        float w = h * aspect;
                        float inv_w = 1.0f / w;
                        float inv_h = 1.0f / h;
                        float inv_range = 1.0f / (far_plane - near_plane);
                        proj.m[0] = inv_w;
                        proj.m[5] = inv_h;
                        proj.m[10] = -2.0f * inv_range;
                        proj.m[11] = 0;
                        proj.m[14] = -(far_plane + near_plane) * inv_range;
                        proj.m[15] = 1;
                    }
                    else
                    {
                        float tan_half = std::tan(fov*0.5f);
                        float inv_range = 1.0f / (near_plane - far_plane);
                        proj.m[0] = 1.0f / (aspect * tan_half);
                        proj.m[5] = 1.0f / tan_half;
                        proj.m[10] = (far_plane + near_plane) * inv_range;
                        proj.m[11] = -1;
                        proj.m[14] = 2.0f * far_plane * near_plane * inv_range;
                        proj.m[15] = 0;
                    }
                    return proj;
                }
            };

            // --------------------------------------------------------------------
            // Light types
            // --------------------------------------------------------------------
            struct PointLight
            {
                Vector3f position;
                Color3f color = {1,1,1};
                float intensity = 1.0f;
                float radius = 10.0f;
                bool cast_shadows = true;
            };

            struct DirectionalLight
            {
                Vector3f direction = {0,-1,0};
                Color3f color = {1,1,1};
                float intensity = 1.0f;
                bool cast_shadows = true;
            };

            struct SpotLight
            {
                Vector3f position;
                Vector3f direction;
                Color3f color = {1,1,1};
                float intensity = 1.0f;
                float inner_cone = 30.0f * M_PI/180.0f;
                float outer_cone = 45.0f * M_PI/180.0f;
                float radius = 20.0f;
                bool cast_shadows = true;
            };

            struct AmbientLight
            {
                Color3f color = {0.1f,0.1f,0.1f};
                float intensity = 1.0f;
            };

            // --------------------------------------------------------------------
            // Scene object
            // --------------------------------------------------------------------
            struct RenderObject
            {
                std::shared_ptr<Mesh> mesh;
                Matrix4 transform;
                std::shared_ptr<MeshMaterial> material;
                bool visible = true;
                bool cast_shadow = true;
                bool receive_shadow = true;
                size_t id = 0;
            };

            class Scene
            {
            public:
                std::vector<RenderObject> objects;
                std::vector<PointLight> point_lights;
                std::vector<DirectionalLight> dir_lights;
                std::vector<SpotLight> spot_lights;
                AmbientLight ambient;
                Color3f background_color = {0.2f,0.2f,0.2f};
                std::shared_ptr<Texture> environment_map;

                void add_object(std::shared_ptr<Mesh> mesh, const Matrix4& transform,
                                std::shared_ptr<MeshMaterial> material = nullptr)
                {
                    RenderObject obj;
                    obj.mesh = mesh;
                    obj.transform = transform;
                    obj.material = material ? material : std::make_shared<MeshMaterial>();
                    obj.id = objects.size();
                    objects.push_back(obj);
                }

                void build_bvh()
                {
                    // For ray tracing acceleration - simplified
                }

                bool intersect(const Ray& ray, HitInfo<float>& hit) const
                {
                    bool any = false;
                    hit.t = ray.t_max;
                    for (const auto& obj : objects)
                    {
                        if (!obj.visible) continue;
                        // Transform ray to object space
                        Matrix4 inv_transform = obj.transform.inverse();
                        Ray local_ray;
                        local_ray.origin = inv_transform.transform_point(ray.origin);
                        local_ray.direction = inv_transform.transform_vector(ray.direction).normalized();
                        HitInfo<float> local_hit;
                        if (obj.mesh->intersect_ray({local_ray.origin, local_ray.direction}, local_hit, ray.t_min))
                        {
                            if (local_hit.t < hit.t)
                            {
                                hit = local_hit;
                                hit.point = obj.transform.transform_point(local_hit.point);
                                hit.normal = obj.transform.transform_vector(local_hit.normal).normalized();
                                any = true;
                            }
                        }
                    }
                    return any;
                }
            };

            // --------------------------------------------------------------------
            // Shader base class
            // --------------------------------------------------------------------
            class Shader
            {
            public:
                virtual ~Shader() = default;
                virtual Color4f vertex(const Vertex& v, const Matrix4& mvp, const Matrix4& model,
                                       Vector4f& clip_pos) = 0;
                virtual Color4f fragment(const Vector3f& world_pos, const Vector3f& normal,
                                         const Vector2f& uv, const MeshMaterial& mat,
                                         const std::vector<std::shared_ptr<Texture>>& textures) = 0;
            };

            class PBRShader : public Shader
            {
            public:
                Color4f vertex(const Vertex& v, const Matrix4& mvp, const Matrix4& model,
                               Vector4f& clip_pos) override
                {
                    Vector4f pos(v.position.x, v.position.y, v.position.z, 1.0f);
                    clip_pos = mvp * pos;
                    return {1,1,1,1}; // return world pos and normal through varyings (simplified)
                }

                Color4f fragment(const Vector3f& world_pos, const Vector3f& normal,
                                 const Vector2f& uv, const MeshMaterial& mat,
                                 const std::vector<std::shared_ptr<Texture>>& textures) override
                {
                    // Base albedo
                    Color3f albedo = mat.pbr_material.albedo;
                    if (mat.diffuse_texture >= 0 && mat.diffuse_texture < (int)textures.size())
                    {
                        auto tex = textures[mat.diffuse_texture];
                        Color4f tex_col = tex->sample(uv);
                        albedo = Color3f(tex_col.x, tex_col.y, tex_col.z);
                    }
                    float metallic = mat.pbr_material.metallic;
                    float roughness = mat.pbr_material.roughness;
                    // Simplified: return albedo as base color
                    return Color4f(albedo.x, albedo.y, albedo.z, 1.0f);
                }
            };

            // --------------------------------------------------------------------
            // Framebuffer
            // --------------------------------------------------------------------
            class Framebuffer
            {
            public:
                std::shared_ptr<Texture> color_attachment;
                std::shared_ptr<Texture> depth_attachment;
                size_t width, height;

                Framebuffer(size_t w, size_t h) : width(w), height(h)
                {
                    color_attachment = std::make_shared<Texture>(w, h, Texture::RGBA32F);
                    depth_attachment = std::make_shared<Texture>(w, h, Texture::Depth32F);
                    clear({0,0,0,1}, 1.0f);
                }

                void clear(const Color4f& color, float depth = 1.0f)
                {
                    auto& col_data = const_cast<xarray_container<float>&>(color_attachment->m_data);
                    for (size_t i=0; i<col_data.size(); i+=4)
                    {
                        col_data.data()[i] = color.x;
                        col_data.data()[i+1] = color.y;
                        col_data.data()[i+2] = color.z;
                        col_data.data()[i+3] = color.w;
                    }
                    auto& depth_data = const_cast<xarray_container<float>&>(depth_attachment->m_data);
                    std::fill(depth_data.begin(), depth_data.end(), depth);
                }

                void set_pixel(size_t x, size_t y, const Color4f& color, float depth)
                {
                    if (x>=width || y>=height) return;
                    // Depth test
                    float& cur_depth = const_cast<xarray_container<float>&>(depth_attachment->m_data)(y,x,0);
                    if (depth >= cur_depth) return;
                    cur_depth = depth;
                    size_t idx = (y*width + x)*4;
                    auto& data = const_cast<xarray_container<float>&>(color_attachment->m_data);
                    data.data()[idx] = color.x;
                    data.data()[idx+1] = color.y;
                    data.data()[idx+2] = color.z;
                    data.data()[idx+3] = color.w;
                }
            };

            // --------------------------------------------------------------------
            // Rasterization renderer
            // --------------------------------------------------------------------
            class Rasterizer
            {
            public:
                Framebuffer framebuffer;
                std::unique_ptr<Shader> shader;
                Scene* scene = nullptr;
                Camera* camera = nullptr;
                bool backface_culling = true;
                bool depth_test = true;

                Rasterizer(size_t w, size_t h) : framebuffer(w, h)
                {
                    shader = std::make_unique<PBRShader>();
                }

                void render()
                {
                    if (!scene || !camera) return;
                    framebuffer.clear({0,0,0,1});

                    Matrix4 view = camera->get_view_matrix();
                    Matrix4 proj = camera->get_projection_matrix();
                    Matrix4 vp = proj * view;

                    for (const auto& obj : scene->objects)
                    {
                        if (!obj.visible) continue;
                        Matrix4 mvp = vp * obj.transform;

                        for (const auto& face : obj.mesh->faces)
                        {
                            // Get vertices
                            const Vertex& v0 = obj.mesh->vertices[face.indices[0]];
                            const Vertex& v1 = obj.mesh->vertices[face.indices[1]];
                            const Vertex& v2 = obj.mesh->vertices[face.indices[2]];

                            Vector4f clip[3];
                            shader->vertex(v0, mvp, obj.transform, clip[0]);
                            shader->vertex(v1, mvp, obj.transform, clip[1]);
                            shader->vertex(v2, mvp, obj.transform, clip[2]);

                            // Backface culling in NDC
                            if (backface_culling)
                            {
                                Vector3f ndc0(clip[0].x/clip[0].w, clip[0].y/clip[0].w, clip[0].z/clip[0].w);
                                Vector3f ndc1(clip[1].x/clip[1].w, clip[1].y/clip[1].w, clip[1].z/clip[1].w);
                                Vector3f ndc2(clip[2].x/clip[2].w, clip[2].y/clip[2].w, clip[2].z/clip[2].w);
                                Vector3f e1 = ndc1 - ndc0;
                                Vector3f e2 = ndc2 - ndc0;
                                if (e1.x*e2.y - e1.y*e2.x >= 0) continue;
                            }

                            // Viewport transform and rasterize
                            auto to_screen = [&](const Vector4f& c, Vector3f& out) {
                                float inv_w = 1.0f / c.w;
                                out.x = (c.x*inv_w * 0.5f + 0.5f) * framebuffer.width;
                                out.y = (c.y*inv_w * 0.5f + 0.5f) * framebuffer.height;
                                out.z = c.z*inv_w;
                            };
                            Vector3f s0, s1, s2;
                            to_screen(clip[0], s0);
                            to_screen(clip[1], s1);
                            to_screen(clip[2], s2);

                            // Bounding box
                            int min_x = std::max(0, (int)std::floor(std::min({s0.x,s1.x,s2.x})));
                            int max_x = std::min((int)framebuffer.width-1, (int)std::ceil(std::max({s0.x,s1.x,s2.x})));
                            int min_y = std::max(0, (int)std::floor(std::min({s0.y,s1.y,s2.y})));
                            int max_y = std::min((int)framebuffer.height-1, (int)std::ceil(std::max({s0.y,s1.y,s2.y})));

                            for (int y=min_y; y<=max_y; ++y)
                            {
                                for (int x=min_x; x<=max_x; ++x)
                                {
                                    Vector2f p(x+0.5f, y+0.5f);
                                    // Barycentric coordinates
                                    float denom = (s1.y - s2.y)*(s0.x - s2.x) + (s2.x - s1.x)*(s0.y - s2.y);
                                    if (std::abs(denom) < 1e-6f) continue;
                                    float w0 = ((s1.y - s2.y)*(p.x - s2.x) + (s2.x - s1.x)*(p.y - s2.y)) / denom;
                                    float w1 = ((s2.y - s0.y)*(p.x - s2.x) + (s0.x - s2.x)*(p.y - s2.y)) / denom;
                                    float w2 = 1.0f - w0 - w1;
                                    if (w0 < 0 || w1 < 0 || w2 < 0) continue;

                                    // Perspective-correct interpolation
                                    float z = 1.0f / (w0/clip[0].w + w1/clip[1].w + w2/clip[2].w);
                                    if (!depth_test || z < framebuffer.depth_attachment->m_data(y,x,0))
                                    {
                                        // Compute world position and normal for shading (simplified)
                                        Vector3f world_pos = obj.transform.transform_point(
                                            v0.position*w0 + v1.position*w1 + v2.position*w2);
                                        Vector3f normal = (v0.normal*w0 + v1.normal*w1 + v2.normal*w2).normalized();
                                        Vector2f uv = v0.texcoord*w0 + v1.texcoord*w1 + v2.texcoord*w2;
                                        Color4f color = shader->fragment(world_pos, normal, uv, *obj.material, {});
                                        framebuffer.set_pixel(x, y, color, z);
                                    }
                                }
                            }
                        }
                    }
                }
            };

            // --------------------------------------------------------------------
            // Path Tracer (Monte Carlo)
            // --------------------------------------------------------------------
            class PathTracer
            {
            public:
                Scene* scene = nullptr;
                Camera* camera = nullptr;
                size_t max_depth = 5;
                size_t samples_per_pixel = 16;
                Framebuffer framebuffer;
                std::mt19937 rng;
                std::uniform_real_distribution<float> dist;

                PathTracer(size_t w, size_t h) : framebuffer(w, h), rng(42), dist(0.0f,1.0f) {}

                Color3f trace(const Ray& ray, int depth)
                {
                    if (depth >= (int)max_depth) return {0,0,0};
                    HitInfo<float> hit;
                    if (!scene->intersect(ray, hit))
                    {
                        if (scene->environment_map)
                        {
                            Vector3f dir = ray.direction;
                            float u = 0.5f + std::atan2(dir.z, dir.x)/(2.0f*M_PI);
                            float v = 0.5f + std::asin(dir.y)/M_PI;
                            Color4f env = scene->environment_map->sample({u,v});
                            return {env.x, env.y, env.z};
                        }
                        return scene->background_color;
                    }

                    // Get material
                    RenderObject* obj = nullptr;
                    for (auto& o : scene->objects)
                    {
                        if (o.id == hit.primitive_id) { obj = &o; break; }
                    }
                    if (!obj) return {0,0,0};

                    MeshMaterial& mat = *obj->material;
                    Vector3f albedo = mat.pbr_material.albedo;
                    float metallic = mat.pbr_material.metallic;
                    float roughness = mat.pbr_material.roughness;

                    // Direct lighting
                    Color3f direct = {0,0,0};
                    for (const auto& light : scene->point_lights)
                    {
                        Vector3f to_light = light.position - hit.point;
                        float dist = to_light.length();
                        Vector3f L = to_light / dist;
                        // Shadow ray
                        Ray shadow_ray{hit.point + L*0.01f, L, 0.001f, dist};
                        if (!scene->intersect(shadow_ray, hit))
                        {
                            float attenuation = 1.0f / (dist*dist);
                            float NdotL = std::max(0.0f, hit.normal.dot(L));
                            direct = direct + light.color * light.intensity * NdotL * attenuation;
                        }
                    }

                    // Indirect: sample BRDF
                    Vector3f indirect = {0,0,0};
                    if (depth < (int)max_depth-1)
                    {
                        // Sample a direction (simplified cosine-weighted hemisphere)
                        float r1 = dist(rng), r2 = dist(rng);
                        float phi = 2.0f * M_PI * r1;
                        float cos_theta = std::sqrt(1.0f - r2);
                        float sin_theta = std::sqrt(r2);
                        Vector3f tangent, bitangent;
                        orthonormal_basis(hit.normal, tangent, bitangent);
                        Vector3f sample_dir = (tangent * std::cos(phi) * sin_theta +
                                               hit.normal * cos_theta +
                                               bitangent * std::sin(phi) * sin_theta).normalized();
                        Ray bounce{hit.point + sample_dir*0.01f, sample_dir};
                        Color3f bounced = trace(bounce, depth+1);
                        // Simplified BRDF
                        indirect = albedo * bounced;
                    }

                    return (direct + indirect) / M_PI;
                }

                void render()
                {
                    if (!scene || !camera) return;
                    framebuffer.clear({0,0,0,1});

                    #pragma omp parallel for collapse(2)
                    for (size_t y=0; y<framebuffer.height; ++y)
                    {
                        for (size_t x=0; x<framebuffer.width; ++x)
                        {
                            Color3f accum = {0,0,0};
                            for (size_t s=0; s<samples_per_pixel; ++s)
                            {
                                float jx = (x + dist(rng)) / framebuffer.width * 2.0f - 1.0f;
                                float jy = (y + dist(rng)) / framebuffer.height * 2.0f - 1.0f;
                                Ray ray = camera->generate_ray(jx, jy);
                                accum = accum + trace(ray, 0);
                            }
                            accum = accum / (float)samples_per_pixel;
                            framebuffer.set_pixel(x, y, {accum.x, accum.y, accum.z, 1.0f}, 0.0f);
                        }
                    }
                }
            };

            // --------------------------------------------------------------------
            // Main Renderer facade
            // --------------------------------------------------------------------
            class Renderer
            {
            public:
                enum Mode { Rasterize, PathTrace };
                Mode mode = Rasterize;
                std::unique_ptr<Rasterizer> rasterizer;
                std::unique_ptr<PathTracer> pathtracer;
                Scene scene;
                Camera camera;

                Renderer(size_t width, size_t height)
                {
                    rasterizer = std::make_unique<Rasterizer>(width, height);
                    pathtracer = std::make_unique<PathTracer>(width, height);
                }

                void render()
                {
                    if (mode == Rasterize)
                    {
                        rasterizer->scene = &scene;
                        rasterizer->camera = &camera;
                        rasterizer->render();
                    }
                    else
                    {
                        pathtracer->scene = &scene;
                        pathtracer->camera = &camera;
                        pathtracer->render();
                    }
                }

                void save_image(const std::string& filename)
                {
                    // Save framebuffer to file
                    Framebuffer& fb = (mode == Rasterize) ? rasterizer->framebuffer : pathtracer->framebuffer;
                    // Convert to uint8 and save using graphics module or custom PPM
                    std::ofstream out(filename, std::ios::binary);
                    out << "P6\n" << fb.width << " " << fb.height << "\n255\n";
                    auto& img = fb.color_attachment->m_data;
                    for (size_t y=0; y<fb.height; ++y)
                    {
                        for (size_t x=0; x<fb.width; ++x)
                        {
                            float r = img(y,x,0);
                            float g = img(y,x,1);
                            float b = img(y,x,2);
                            out.put(static_cast<char>(std::clamp(r,0.0f,1.0f)*255));
                            out.put(static_cast<char>(std::clamp(g,0.0f,1.0f)*255));
                            out.put(static_cast<char>(std::clamp(b,0.0f,1.0f)*255));
                        }
                    }
                }
            };

        } // namespace renderer

        using renderer::Renderer;
        using renderer::Camera;
        using renderer::Scene;
        using renderer::Texture;
        using renderer::PointLight;
        using renderer::DirectionalLight;
        using renderer::SpotLight;
        using renderer::AmbientLight;
        using renderer::PBRShader;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XRENDERER_HPP

// graphics/xrenderer.hpp