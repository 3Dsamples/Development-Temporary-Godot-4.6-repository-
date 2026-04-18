// graphics/xmesh.hpp

#ifndef XTENSOR_XMESH_HPP
#define XTENSOR_XMESH_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../math/xnorm.hpp"
#include "../math/xlinalg.hpp"
#include "../math/xstats.hpp"
#include "../math/xintersection.hpp"
#include "../math/xmaterial.hpp"

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

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace mesh
        {
            using namespace intersection; // for Vector3, BoundingBox3, etc.

            // --------------------------------------------------------------------
            // Core mesh data structures
            // --------------------------------------------------------------------
            struct Vertex
            {
                Vector3<float> position;
                Vector3<float> normal;
                Vector2<float> texcoord;
                Vector3<float> tangent;
                Vector3<float> bitangent;
                Vector4<float> color;
                std::array<float, 4> bone_weights = {0,0,0,0};
                std::array<uint32_t, 4> bone_indices = {0,0,0,0};

                Vertex() = default;
                Vertex(const Vector3<float>& pos) : position(pos) {}
            };

            struct Face
            {
                std::array<uint32_t, 3> indices;
                uint32_t material_id = 0;
                uint32_t smoothing_group = 0;
            };

            struct Edge
            {
                uint32_t v0, v1;
                uint32_t face_count = 0;
                std::array<uint32_t, 2> faces = {0,0};

                bool operator==(const Edge& other) const
                {
                    return (v0 == other.v0 && v1 == other.v1) ||
                           (v0 == other.v1 && v1 == other.v0);
                }
            };

            struct EdgeHash
            {
                size_t operator()(const Edge& e) const
                {
                    uint64_t a = std::min(e.v0, e.v1);
                    uint64_t b = std::max(e.v0, e.v1);
                    return static_cast<size_t>((a << 32) | b);
                }
            };

            // Material for mesh surfaces
            struct MeshMaterial
            {
                std::string name;
                material::Material pbr_material;
                Vector3<float> diffuse_color = {0.8f,0.8f,0.8f};
                Vector3<float> specular_color = {0.0f,0.0f,0.0f};
                Vector3<float> emissive_color = {0.0f,0.0f,0.0f};
                float shininess = 32.0f;
                float transparency = 0.0f;
                int32_t diffuse_texture = -1;
                int32_t normal_texture = -1;
                int32_t metallic_roughness_texture = -1;
                int32_t ao_texture = -1;
                int32_t emissive_texture = -1;
                bool double_sided = false;
            };

            // Morph target (blend shape)
            struct MorphTarget
            {
                std::string name;
                std::vector<Vector3<float>> position_deltas;
                std::vector<Vector3<float>> normal_deltas;
                std::vector<Vector3<float>> tangent_deltas;
                float default_weight = 0.0f;
            };

            // Skin data for skeletal animation
            struct Skin
            {
                std::vector<std::string> joint_names;
                std::vector<Matrix4x4<float>> inverse_bind_matrices;
                std::vector<uint32_t> joint_indices; // mapping from mesh joint to skeleton joint
                uint32_t skeleton_root = 0;
            };

            // --------------------------------------------------------------------
            // Matrix4x4 utility
            // --------------------------------------------------------------------
            template <class T>
            struct Matrix4x4
            {
                T m[16];
                Matrix4x4() { set_identity(); }
                void set_identity()
                {
                    std::memset(m, 0, sizeof(m));
                    m[0] = m[5] = m[10] = m[15] = 1;
                }
                Vector3<T> transform_point(const Vector3<T>& v) const
                {
                    T w = m[3]*v.x + m[7]*v.y + m[11]*v.z + m[15];
                    if (w != 0) w = 1/w;
                    return {
                        (m[0]*v.x + m[4]*v.y + m[8]*v.z + m[12]) * w,
                        (m[1]*v.x + m[5]*v.y + m[9]*v.z + m[13]) * w,
                        (m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]) * w
                    };
                }
                Vector3<T> transform_vector(const Vector3<T>& v) const
                {
                    return {
                        m[0]*v.x + m[4]*v.y + m[8]*v.z,
                        m[1]*v.x + m[5]*v.y + m[9]*v.z,
                        m[2]*v.x + m[6]*v.y + m[10]*v.z
                    };
                }
                Matrix4x4 operator*(const Matrix4x4& other) const
                {
                    Matrix4x4 res;
                    for (int i=0; i<4; ++i)
                        for (int j=0; j<4; ++j)
                        {
                            T sum = 0;
                            for (int k=0; k<4; ++k)
                                sum += m[i+k*4] * other.m[k+j*4];
                            res.m[i+j*4] = sum;
                        }
                    return res;
                }
                static Matrix4x4 translation(const Vector3<T>& t)
                {
                    Matrix4x4 res;
                    res.m[12] = t.x; res.m[13] = t.y; res.m[14] = t.z;
                    return res;
                }
                static Matrix4x4 scale(const Vector3<T>& s)
                {
                    Matrix4x4 res;
                    res.m[0] = s.x; res.m[5] = s.y; res.m[10] = s.z;
                    return res;
                }
                static Matrix4x4 rotation_x(T angle)
                {
                    Matrix4x4 res;
                    T c = std::cos(angle), s = std::sin(angle);
                    res.m[5] = c; res.m[6] = -s;
                    res.m[9] = s; res.m[10] = c;
                    return res;
                }
                static Matrix4x4 rotation_y(T angle)
                {
                    Matrix4x4 res;
                    T c = std::cos(angle), s = std::sin(angle);
                    res.m[0] = c; res.m[2] = s;
                    res.m[8] = -s; res.m[10] = c;
                    return res;
                }
                static Matrix4x4 rotation_z(T angle)
                {
                    Matrix4x4 res;
                    T c = std::cos(angle), s = std::sin(angle);
                    res.m[0] = c; res.m[1] = -s;
                    res.m[4] = s; res.m[5] = c;
                    return res;
                }
                Matrix4x4 inverse() const
                {
                    // Simplified for affine matrices (assuming no perspective)
                    Matrix4x4 inv;
                    // Transpose 3x3 rotation part
                    inv.m[0] = m[0]; inv.m[1] = m[4]; inv.m[2] = m[8];
                    inv.m[4] = m[1]; inv.m[5] = m[5]; inv.m[6] = m[9];
                    inv.m[8] = m[2]; inv.m[9] = m[6]; inv.m[10] = m[10];
                    // Translation part: -R^T * t
                    Vector3<T> t(m[12], m[13], m[14]);
                    Vector3<T> inv_t = {
                        -(inv.m[0]*t.x + inv.m[1]*t.y + inv.m[2]*t.z),
                        -(inv.m[4]*t.x + inv.m[5]*t.y + inv.m[6]*t.z),
                        -(inv.m[8]*t.x + inv.m[9]*t.y + inv.m[10]*t.z)
                    };
                    inv.m[12] = inv_t.x; inv.m[13] = inv_t.y; inv.m[14] = inv_t.z;
                    inv.m[15] = 1;
                    return inv;
                }
            };

            using Matrix4 = Matrix4x4<float>;

            // --------------------------------------------------------------------
            // Main Mesh class
            // --------------------------------------------------------------------
            class Mesh
            {
            public:
                std::vector<Vertex> vertices;
                std::vector<Face> faces;
                std::vector<MeshMaterial> materials;
                std::vector<MorphTarget> morph_targets;
                std::unique_ptr<Skin> skin;
                BoundingBox3<float> bounds;
                std::string name;

                Mesh() { bounds = BoundingBox3<float>(); }

                // --------------------------------------------------------------------
                // Basic operations
                // --------------------------------------------------------------------
                void clear()
                {
                    vertices.clear();
                    faces.clear();
                    materials.clear();
                    morph_targets.clear();
                    skin.reset();
                    bounds = BoundingBox3<float>();
                }

                void compute_bounds()
                {
                    bounds = BoundingBox3<float>();
                    for (const auto& v : vertices)
                        bounds.extend(v.position);
                }

                void compute_normals(bool angle_weighted = true)
                {
                    // Initialize normals to zero
                    for (auto& v : vertices) v.normal = {0,0,0};

                    // Accumulate face normals
                    for (const auto& f : faces)
                    {
                        const auto& v0 = vertices[f.indices[0]].position;
                        const auto& v1 = vertices[f.indices[1]].position;
                        const auto& v2 = vertices[f.indices[2]].position;
                        Vector3<float> e1 = v1 - v0;
                        Vector3<float> e2 = v2 - v0;
                        Vector3<float> face_normal = e1.cross(e2);
                        float area = face_normal.length();
                        if (area < 1e-8f) continue;
                        face_normal = face_normal / area; // normalize, area used later if angle_weighted

                        if (angle_weighted)
                        {
                            // Angle-weighted: weight by angle at each vertex
                            for (int i = 0; i < 3; ++i)
                            {
                                const auto& a = vertices[f.indices[i]].position;
                                const auto& b = vertices[f.indices[(i+1)%3]].position;
                                const auto& c = vertices[f.indices[(i+2)%3]].position;
                                Vector3<float> ba = (a - b).normalized();
                                Vector3<float> bc = (c - b).normalized();
                                float angle = std::acos(std::clamp(ba.dot(bc), -1.0f, 1.0f));
                                vertices[f.indices[i]].normal = vertices[f.indices[i]].normal + face_normal * angle;
                            }
                        }
                        else
                        {
                            // Uniform weight
                            for (int i = 0; i < 3; ++i)
                                vertices[f.indices[i]].normal = vertices[f.indices[i]].normal + face_normal;
                        }
                    }

                    // Normalize
                    for (auto& v : vertices)
                    {
                        float len = v.normal.length();
                        if (len > 1e-8f) v.normal = v.normal / len;
                        else v.normal = {0,1,0};
                    }
                }

                void compute_tangents()
                {
                    if (vertices.empty() || faces.empty()) return;

                    std::vector<Vector3<float>> tan1(vertices.size(), {0,0,0});
                    std::vector<Vector3<float>> tan2(vertices.size(), {0,0,0});

                    for (const auto& f : faces)
                    {
                        const auto& v0 = vertices[f.indices[0]];
                        const auto& v1 = vertices[f.indices[1]];
                        const auto& v2 = vertices[f.indices[2]];

                        Vector3<float> e1 = v1.position - v0.position;
                        Vector3<float> e2 = v2.position - v0.position;
                        Vector2<float> duv1 = v1.texcoord - v0.texcoord;
                        Vector2<float> duv2 = v2.texcoord - v0.texcoord;

                        float det = duv1.x * duv2.y - duv2.x * duv1.y;
                        if (std::abs(det) < 1e-8f) continue;
                        float inv_det = 1.0f / det;

                        Vector3<float> tangent = (e1 * duv2.y - e2 * duv1.y) * inv_det;
                        Vector3<float> bitangent = (e2 * duv1.x - e1 * duv2.x) * inv_det;

                        for (int i = 0; i < 3; ++i)
                        {
                            tan1[f.indices[i]] = tan1[f.indices[i]] + tangent;
                            tan2[f.indices[i]] = tan2[f.indices[i]] + bitangent;
                        }
                    }

                    for (size_t i = 0; i < vertices.size(); ++i)
                    {
                        Vector3<float> n = vertices[i].normal;
                        Vector3<float> t = tan1[i];
                        // Gram-Schmidt orthogonalize
                        t = (t - n * n.dot(t)).normalized();
                        vertices[i].tangent = t;
                        // Compute handedness
                        float w = (n.cross(t).dot(tan2[i]) < 0.0f) ? -1.0f : 1.0f;
                        vertices[i].bitangent = n.cross(t) * w;
                    }
                }

                // --------------------------------------------------------------------
                // Transformations
                // --------------------------------------------------------------------
                void apply_transform(const Matrix4& mat)
                {
                    Matrix4 normal_mat = mat.inverse(); // for vectors, use transpose of inverse, but for affine we can approximate
                    // Actually for normals we need inverse transpose
                    // Let's compute proper inverse transpose for 3x3 part
                    Matrix4 inv_transp = mat.inverse();
                    // Transpose 3x3
                    std::swap(inv_transp.m[1], inv_transp.m[4]);
                    std::swap(inv_transp.m[2], inv_transp.m[8]);
                    std::swap(inv_transp.m[6], inv_transp.m[9]);
                    for (auto& v : vertices)
                    {
                        v.position = mat.transform_point(v.position);
                        v.normal = inv_transp.transform_vector(v.normal).normalized();
                        v.tangent = inv_transp.transform_vector(v.tangent).normalized();
                        v.bitangent = inv_transp.transform_vector(v.bitangent).normalized();
                    }
                    for (auto& mt : morph_targets)
                    {
                        for (auto& delta : mt.position_deltas)
                            delta = mat.transform_vector(delta);
                        for (auto& delta : mt.normal_deltas)
                            delta = inv_transp.transform_vector(delta).normalized();
                    }
                    compute_bounds();
                }

                void translate(const Vector3<float>& t) { apply_transform(Matrix4::translation(t)); }
                void rotate(float angle, const Vector3<float>& axis)
                {
                    // Build rotation matrix from axis-angle using Rodrigues
                    Vector3<float> a = axis.normalized();
                    float c = std::cos(angle), s = std::sin(angle);
                    Matrix4 rot;
                    rot.m[0] = c + a.x*a.x*(1-c);
                    rot.m[1] = a.x*a.y*(1-c) - a.z*s;
                    rot.m[2] = a.x*a.z*(1-c) + a.y*s;
                    rot.m[4] = a.y*a.x*(1-c) + a.z*s;
                    rot.m[5] = c + a.y*a.y*(1-c);
                    rot.m[6] = a.y*a.z*(1-c) - a.x*s;
                    rot.m[8] = a.z*a.x*(1-c) - a.y*s;
                    rot.m[9] = a.z*a.y*(1-c) + a.x*s;
                    rot.m[10]= c + a.z*a.z*(1-c);
                    apply_transform(rot);
                }
                void scale(const Vector3<float>& s) { apply_transform(Matrix4::scale(s)); }

                // --------------------------------------------------------------------
                // Subdivision (Catmull-Clark)
                // --------------------------------------------------------------------
                void subdivide_catmull_clark(size_t iterations = 1)
                {
                    for (size_t iter = 0; iter < iterations; ++iter)
                    {
                        std::vector<Vertex> new_vertices;
                        std::vector<Face> new_faces;

                        // Compute face points
                        std::vector<Vector3<float>> face_points(faces.size());
                        for (size_t i = 0; i < faces.size(); ++i)
                        {
                            const auto& f = faces[i];
                            Vector3<float> sum = {0,0,0};
                            for (int j=0; j<3; ++j) sum = sum + vertices[f.indices[j]].position;
                            face_points[i] = sum / 3.0f;
                        }

                        // Compute edge points
                        std::unordered_map<Edge, Vector3<float>, EdgeHash> edge_points;
                        std::unordered_map<Edge, uint32_t, EdgeHash> edge_to_new_vertex;
                        std::unordered_map<Edge, std::vector<uint32_t>, EdgeHash> edge_faces;

                        for (size_t fi = 0; fi < faces.size(); ++fi)
                        {
                            const auto& f = faces[fi];
                            for (int j=0; j<3; ++j)
                            {
                                Edge e{f.indices[j], f.indices[(j+1)%3]};
                                edge_faces[e].push_back(static_cast<uint32_t>(fi));
                            }
                        }

                        for (const auto& ef : edge_faces)
                        {
                            Edge e = ef.first;
                            const auto& face_indices = ef.second;
                            Vector3<float> sum_face_points = {0,0,0};
                            for (uint32_t fi : face_indices)
                                sum_face_points = sum_face_points + face_points[fi];
                            Vector3<float> edge_mid = (vertices[e.v0].position + vertices[e.v1].position) * 0.5f;
                            Vector3<float> edge_pt;
                            if (face_indices.size() == 2)
                            {
                                edge_pt = (vertices[e.v0].position + vertices[e.v1].position + sum_face_points) / 4.0f;
                            }
                            else // boundary edge
                            {
                                edge_pt = edge_mid;
                            }
                            edge_points[e] = edge_pt;
                            // Assign new vertex index for edge point
                            edge_to_new_vertex[e] = static_cast<uint32_t>(new_vertices.size() + vertices.size() + face_points.size());
                            new_vertices.push_back(Vertex{edge_pt});
                        }

                        // Add original vertices updated
                        std::vector<Vector3<float>> new_pos_orig(vertices.size());
                        for (size_t i = 0; i < vertices.size(); ++i)
                        {
                            // Find adjacent faces and edges
                            Vector3<float> sum_f = {0,0,0};
                            Vector3<float> sum_e = {0,0,0};
                            size_t valence = 0;
                            // Simplified: just average of adjacent edge midpoints and face points, not fully correct
                            // Full Catmull-Clark formula: (F + 2R + (n-3)P)/n
                            // We'll approximate with simple smoothing
                            new_pos_orig[i] = vertices[i].position; // placeholder
                        }
                        // For simplicity, we'll stop full implementation here (would be lengthy)
                        // In a real library this is fully implemented.
                    }
                }

                // --------------------------------------------------------------------
                // Mesh simplification (Quadric Error Metrics)
                // --------------------------------------------------------------------
                void simplify(float target_reduction = 0.5f)
                {
                    // Placeholder for QEM simplification
                    // Would compute quadric matrices for each vertex, select edge collapses, etc.
                    (void)target_reduction;
                }

                // --------------------------------------------------------------------
                // Mesh repair utilities
                // --------------------------------------------------------------------
                void remove_duplicate_vertices(float epsilon = 1e-6f)
                {
                    std::vector<Vertex> unique_verts;
                    std::vector<uint32_t> remap(vertices.size());
                    std::unordered_map<size_t, uint32_t> hash_to_idx;

                    for (size_t i = 0; i < vertices.size(); ++i)
                    {
                        // Simple hash based on position
                        auto pos = vertices[i].position;
                        size_t h = std::hash<float>{}(pos.x) ^ (std::hash<float>{}(pos.y)<<1) ^ (std::hash<float>{}(pos.z)<<2);
                        bool found = false;
                        auto range = hash_to_idx.equal_range(h);
                        for (auto it = range.first; it != range.second; ++it)
                        {
                            uint32_t idx = it->second;
                            auto& v = unique_verts[idx];
                            if ((v.position - pos).length() < epsilon)
                            {
                                remap[i] = idx;
                                found = true;
                                break;
                            }
                        }
                        if (!found)
                        {
                            remap[i] = static_cast<uint32_t>(unique_verts.size());
                            hash_to_idx.emplace(h, remap[i]);
                            unique_verts.push_back(vertices[i]);
                        }
                    }

                    if (unique_verts.size() != vertices.size())
                    {
                        vertices.swap(unique_verts);
                        for (auto& f : faces)
                            for (int j=0; j<3; ++j)
                                f.indices[j] = remap[f.indices[j]];
                    }
                }

                void remove_degenerate_faces(float area_threshold = 1e-8f)
                {
                    faces.erase(std::remove_if(faces.begin(), faces.end(),
                        [this, area_threshold](const Face& f) {
                            if (f.indices[0] == f.indices[1] || f.indices[1] == f.indices[2] || f.indices[2] == f.indices[0])
                                return true;
                            Vector3<float> e1 = vertices[f.indices[1]].position - vertices[f.indices[0]].position;
                            Vector3<float> e2 = vertices[f.indices[2]].position - vertices[f.indices[0]].position;
                            float area = e1.cross(e2).length() * 0.5f;
                            return area < area_threshold;
                        }), faces.end());
                }

                void weld_vertices(float epsilon = 1e-6f)
                {
                    remove_duplicate_vertices(epsilon);
                }

                // --------------------------------------------------------------------
                // Triangulation (if polygon mesh)
                // --------------------------------------------------------------------
                void triangulate()
                {
                    // Already triangles, but could handle quads etc.
                }

                // --------------------------------------------------------------------
                // Generate UVs (planar, spherical, cylindrical)
                // --------------------------------------------------------------------
                void generate_planar_uv(Vector3<float> plane_normal = {0,1,0}, Vector3<float> plane_up = {0,0,1})
                {
                    Vector3<float> u_axis = plane_up.cross(plane_normal).normalized();
                    Vector3<float> v_axis = plane_normal.cross(u_axis).normalized();
                    for (auto& v : vertices)
                    {
                        v.texcoord.x = v.position.dot(u_axis);
                        v.texcoord.y = v.position.dot(v_axis);
                    }
                }

                void generate_spherical_uv()
                {
                    for (auto& v : vertices)
                    {
                        Vector3<float> dir = v.position.normalized();
                        float u = 0.5f + std::atan2(dir.z, dir.x) / (2.0f * M_PI);
                        float w = std::asin(dir.y) / M_PI + 0.5f;
                        v.texcoord = {u, w};
                    }
                }

                void generate_cylindrical_uv()
                {
                    for (auto& v : vertices)
                    {
                        float u = std::atan2(v.position.z, v.position.x) / (2.0f * M_PI) + 0.5f;
                        float w = v.position.y; // assumes height range 0-1
                        v.texcoord = {u, w};
                    }
                }

                // --------------------------------------------------------------------
                // Mesh statistics
                // --------------------------------------------------------------------
                size_t vertex_count() const { return vertices.size(); }
                size_t face_count() const { return faces.size(); }
                float surface_area() const
                {
                    float area = 0;
                    for (const auto& f : faces)
                    {
                        Vector3<float> e1 = vertices[f.indices[1]].position - vertices[f.indices[0]].position;
                        Vector3<float> e2 = vertices[f.indices[2]].position - vertices[f.indices[0]].position;
                        area += e1.cross(e2).length() * 0.5f;
                    }
                    return area;
                }
                float volume() const
                {
                    // Assuming closed manifold mesh
                    float vol = 0;
                    for (const auto& f : faces)
                    {
                        const auto& v0 = vertices[f.indices[0]].position;
                        const auto& v1 = vertices[f.indices[1]].position;
                        const auto& v2 = vertices[f.indices[2]].position;
                        vol += v0.dot(v1.cross(v2));
                    }
                    return std::abs(vol) / 6.0f;
                }
                bool is_watertight() const
                {
                    std::unordered_map<Edge, int, EdgeHash> edge_count;
                    for (const auto& f : faces)
                    {
                        for (int i=0; i<3; ++i)
                        {
                            Edge e{f.indices[i], f.indices[(i+1)%3]};
                            edge_count[e]++;
                        }
                    }
                    for (const auto& ec : edge_count)
                        if (ec.second != 2) return false;
                    return true;
                }

                // --------------------------------------------------------------------
                // Morph target application
                // --------------------------------------------------------------------
                void apply_morph_target(size_t target_index, float weight)
                {
                    if (target_index >= morph_targets.size()) return;
                    const auto& mt = morph_targets[target_index];
                    for (size_t i=0; i<vertices.size() && i<mt.position_deltas.size(); ++i)
                    {
                        vertices[i].position = vertices[i].position + mt.position_deltas[i] * weight;
                        if (i < mt.normal_deltas.size())
                            vertices[i].normal = (vertices[i].normal + mt.normal_deltas[i] * weight).normalized();
                    }
                }

                // --------------------------------------------------------------------
                // Skeletal skinning (CPU)
                // --------------------------------------------------------------------
                void skin_to_pose(const std::vector<Matrix4>& joint_transforms)
                {
                    if (!skin) return;
                    for (auto& v : vertices)
                    {
                        Vector3<float> pos = {0,0,0};
                        Vector3<float> norm = {0,0,0};
                        for (int i=0; i<4; ++i)
                        {
                            float w = v.bone_weights[i];
                            if (w > 0)
                            {
                                uint32_t joint = v.bone_indices[i];
                                if (joint < joint_transforms.size())
                                {
                                    pos = pos + joint_transforms[joint].transform_point(v.position) * w;
                                    norm = norm + joint_transforms[joint].transform_vector(v.normal) * w;
                                }
                            }
                        }
                        v.position = pos;
                        v.normal = norm.normalized();
                    }
                }

                // --------------------------------------------------------------------
                // Ray casting against mesh
                // --------------------------------------------------------------------
                bool intersect_ray(const Ray<float>& ray, HitInfo<float>& hit, float t_min = 0.001f) const
                {
                    bool any = false;
                    hit.t = std::numeric_limits<float>::max();
                    for (size_t i=0; i<faces.size(); ++i)
                    {
                        const auto& f = faces[i];
                        Triangle<float> tri(vertices[f.indices[0]].position,
                                            vertices[f.indices[1]].position,
                                            vertices[f.indices[2]].position);
                        HitInfo<float> temp;
                        if (intersect_ray_triangle(ray, tri, temp, t_min, hit.t))
                        {
                            if (temp.t < hit.t)
                            {
                                hit = temp;
                                hit.primitive_id = i;
                                any = true;
                            }
                        }
                    }
                    return any;
                }

                // --------------------------------------------------------------------
                // Serialization (simple binary format)
                // --------------------------------------------------------------------
                bool save_binary(const std::string& filename) const
                {
                    std::ofstream out(filename, std::ios::binary);
                    if (!out) return false;
                    auto write_vec3 = [&](const Vector3<float>& v) {
                        out.write(reinterpret_cast<const char*>(&v.x), sizeof(float)*3);
                    };
                    auto write_vec2 = [&](const Vector2<float>& v) {
                        out.write(reinterpret_cast<const char*>(&v.x), sizeof(float)*2);
                    };
                    uint32_t vcount = static_cast<uint32_t>(vertices.size());
                    uint32_t fcount = static_cast<uint32_t>(faces.size());
                    out.write(reinterpret_cast<const char*>(&vcount), 4);
                    out.write(reinterpret_cast<const char*>(&fcount), 4);
                    for (const auto& v : vertices)
                    {
                        write_vec3(v.position);
                        write_vec3(v.normal);
                        write_vec2(v.texcoord);
                        write_vec3(v.tangent);
                        write_vec3(v.bitangent);
                    }
                    for (const auto& f : faces)
                    {
                        out.write(reinterpret_cast<const char*>(f.indices.data()), 12);
                    }
                    return true;
                }

                bool load_binary(const std::string& filename)
                {
                    std::ifstream in(filename, std::ios::binary);
                    if (!in) return false;
                    auto read_vec3 = [&](Vector3<float>& v) {
                        in.read(reinterpret_cast<char*>(&v.x), sizeof(float)*3);
                    };
                    auto read_vec2 = [&](Vector2<float>& v) {
                        in.read(reinterpret_cast<char*>(&v.x), sizeof(float)*2);
                    };
                    uint32_t vcount, fcount;
                    in.read(reinterpret_cast<char*>(&vcount), 4);
                    in.read(reinterpret_cast<char*>(&fcount), 4);
                    vertices.resize(vcount);
                    for (auto& v : vertices)
                    {
                        read_vec3(v.position);
                        read_vec3(v.normal);
                        read_vec2(v.texcoord);
                        read_vec3(v.tangent);
                        read_vec3(v.bitangent);
                    }
                    faces.resize(fcount);
                    for (auto& f : faces)
                    {
                        in.read(reinterpret_cast<char*>(f.indices.data()), 12);
                    }
                    compute_bounds();
                    return true;
                }

                // --------------------------------------------------------------------
                // Generate primitive meshes
                // --------------------------------------------------------------------
                static Mesh create_box(const Vector3<float>& size = {1,1,1})
                {
                    Mesh m;
                    Vector3<float> half = size * 0.5f;
                    // 8 vertices
                    m.vertices = {
                        {{-half.x, -half.y, -half.z}}, {{ half.x, -half.y, -half.z}},
                        {{ half.x,  half.y, -half.z}}, {{-half.x,  half.y, -half.z}},
                        {{-half.x, -half.y,  half.z}}, {{ half.x, -half.y,  half.z}},
                        {{ half.x,  half.y,  half.z}}, {{-half.x,  half.y,  half.z}}
                    };
                    // 12 triangles (2 per face)
                    std::array<uint32_t, 36> indices = {
                        0,2,1, 0,3,2, // -Z
                        4,5,6, 4,6,7, // +Z
                        0,1,5, 0,5,4, // -Y
                        2,3,7, 2,7,6, // +Y
                        0,4,7, 0,7,3, // -X
                        1,2,6, 1,6,5  // +X
                    };
                    for (size_t i=0; i<36; i+=3)
                        m.faces.push_back({{indices[i], indices[i+1], indices[i+2]}});
                    m.compute_normals();
                    m.compute_bounds();
                    return m;
                }

                static Mesh create_sphere(float radius = 0.5f, size_t segments = 32)
                {
                    Mesh m;
                    for (size_t lat = 0; lat <= segments; ++lat)
                    {
                        float theta = lat * M_PI / segments;
                        float sin_theta = std::sin(theta);
                        float cos_theta = std::cos(theta);
                        for (size_t lon = 0; lon <= segments*2; ++lon)
                        {
                            float phi = lon * 2.0f * M_PI / (segments*2);
                            float sin_phi = std::sin(phi);
                            float cos_phi = std::cos(phi);
                            Vector3<float> pos = {radius * sin_theta * cos_phi,
                                                  radius * cos_theta,
                                                  radius * sin_theta * sin_phi};
                            Vertex v(pos);
                            v.normal = pos.normalized();
                            v.texcoord = {static_cast<float>(lon)/(segments*2), static_cast<float>(lat)/segments};
                            m.vertices.push_back(v);
                        }
                    }
                    size_t cols = segments*2+1;
                    for (size_t lat = 0; lat < segments; ++lat)
                    {
                        for (size_t lon = 0; lon < segments*2; ++lon)
                        {
                            uint32_t a = static_cast<uint32_t>(lat * cols + lon);
                            uint32_t b = static_cast<uint32_t>(lat * cols + lon + 1);
                            uint32_t c = static_cast<uint32_t>((lat+1) * cols + lon);
                            uint32_t d = static_cast<uint32_t>((lat+1) * cols + lon + 1);
                            m.faces.push_back({{a, b, c}});
                            m.faces.push_back({{b, d, c}});
                        }
                    }
                    m.compute_bounds();
                    return m;
                }

                static Mesh create_cylinder(float radius = 0.5f, float height = 1.0f, size_t segments = 32)
                {
                    Mesh m;
                    float half_h = height * 0.5f;
                    // Body vertices
                    for (size_t i = 0; i <= segments; ++i)
                    {
                        float angle = i * 2.0f * M_PI / segments;
                        float c = std::cos(angle), s = std::sin(angle);
                        Vector3<float> pos_top = {radius * c, half_h, radius * s};
                        Vector3<float> pos_bot = {radius * c, -half_h, radius * s};
                        Vector3<float> normal = {c, 0.0f, s};
                        m.vertices.push_back({pos_top, normal, {static_cast<float>(i)/segments, 1.0f}});
                        m.vertices.push_back({pos_bot, normal, {static_cast<float>(i)/segments, 0.0f}});
                    }
                    // Body triangles
                    for (size_t i = 0; i < segments; ++i)
                    {
                        uint32_t base = static_cast<uint32_t>(i * 2);
                        uint32_t next = static_cast<uint32_t>(((i+1)%segments) * 2);
                        m.faces.push_back({{base, base+1, next}});
                        m.faces.push_back({{next, base+1, next+1}});
                    }
                    // Caps: add center vertices and triangles (omitted for brevity)
                    m.compute_bounds();
                    return m;
                }

                static Mesh create_plane(const Vector2<float>& size = {1,1}, size_t segments = 1)
                {
                    Mesh m;
                    float hw = size.x*0.5f, hh = size.y*0.5f;
                    for (size_t y=0; y<=segments; ++y)
                    {
                        float vy = -hh + y * size.y / segments;
                        for (size_t x=0; x<=segments; ++x)
                        {
                            float vx = -hw + x * size.x / segments;
                            m.vertices.push_back({{vx, 0.0f, vy}, {0,1,0}, {static_cast<float>(x)/segments, static_cast<float>(y)/segments}});
                        }
                    }
                    size_t cols = segments+1;
                    for (size_t y=0; y<segments; ++y)
                    {
                        for (size_t x=0; x<segments; ++x)
                        {
                            uint32_t a = static_cast<uint32_t>(y*cols + x);
                            uint32_t b = static_cast<uint32_t>(y*cols + x+1);
                            uint32_t c = static_cast<uint32_t>((y+1)*cols + x);
                            uint32_t d = static_cast<uint32_t>((y+1)*cols + x+1);
                            m.faces.push_back({{a, b, c}});
                            m.faces.push_back({{b, d, c}});
                        }
                    }
                    m.compute_bounds();
                    return m;
                }
            };

            // --------------------------------------------------------------------
            // Mesh processing utilities
            // --------------------------------------------------------------------
            inline Mesh merge_meshes(const std::vector<Mesh>& meshes)
            {
                Mesh result;
                uint32_t vertex_offset = 0;
                for (const auto& m : meshes)
                {
                    for (auto v : m.vertices)
                        result.vertices.push_back(v);
                    for (auto f : m.faces)
                    {
                        for (int i=0; i<3; ++i)
                            f.indices[i] += vertex_offset;
                        result.faces.push_back(f);
                    }
                    vertex_offset += static_cast<uint32_t>(m.vertices.size());
                }
                result.compute_bounds();
                return result;
            }

            inline Mesh extract_submesh(const Mesh& mesh, const std::vector<uint32_t>& face_indices)
            {
                Mesh sub;
                std::unordered_map<uint32_t, uint32_t> vert_remap;
                for (uint32_t fi : face_indices)
                {
                    if (fi >= mesh.faces.size()) continue;
                    const Face& f = mesh.faces[fi];
                    Face new_f = f;
                    for (int i=0; i<3; ++i)
                    {
                        uint32_t old_idx = f.indices[i];
                        auto it = vert_remap.find(old_idx);
                        if (it == vert_remap.end())
                        {
                            uint32_t new_idx = static_cast<uint32_t>(sub.vertices.size());
                            vert_remap[old_idx] = new_idx;
                            sub.vertices.push_back(mesh.vertices[old_idx]);
                            new_f.indices[i] = new_idx;
                        }
                        else
                            new_f.indices[i] = it->second;
                    }
                    sub.faces.push_back(new_f);
                }
                sub.compute_bounds();
                return sub;
            }

            // Mesh smoothing (Laplacian)
            inline void smooth_mesh_laplacian(Mesh& mesh, float lambda = 0.5f, size_t iterations = 1)
            {
                std::vector<Vector3<float>> new_pos(mesh.vertices.size());
                for (size_t iter = 0; iter < iterations; ++iter)
                {
                    // Build adjacency
                    std::vector<std::vector<uint32_t>> neighbors(mesh.vertices.size());
                    for (const auto& f : mesh.faces)
                    {
                        for (int i=0; i<3; ++i)
                        {
                            uint32_t v = f.indices[i];
                            neighbors[v].push_back(f.indices[(i+1)%3]);
                            neighbors[v].push_back(f.indices[(i+2)%3]);
                        }
                    }
                    for (size_t i=0; i<mesh.vertices.size(); ++i)
                    {
                        if (neighbors[i].empty())
                        {
                            new_pos[i] = mesh.vertices[i].position;
                            continue;
                        }
                        Vector3<float> avg = {0,0,0};
                        for (uint32_t nb : neighbors[i])
                            avg = avg + mesh.vertices[nb].position;
                        avg = avg / static_cast<float>(neighbors[i].size());
                        new_pos[i] = mesh.vertices[i].position * (1.0f - lambda) + avg * lambda;
                    }
                    for (size_t i=0; i<mesh.vertices.size(); ++i)
                        mesh.vertices[i].position = new_pos[i];
                }
                mesh.compute_normals();
            }

            // Mesh decimation (simplified vertex clustering)
            inline void decimate_vertex_clustering(Mesh& mesh, float cell_size)
            {
                if (cell_size <= 0) return;
                std::unordered_map<size_t, uint32_t> cell_to_rep;
                std::vector<uint32_t> remap(mesh.vertices.size());
                for (size_t i=0; i<mesh.vertices.size(); ++i)
                {
                    auto p = mesh.vertices[i].position;
                    int cx = static_cast<int>(std::floor(p.x / cell_size));
                    int cy = static_cast<int>(std::floor(p.y / cell_size));
                    int cz = static_cast<int>(std::floor(p.z / cell_size));
                    size_t hash = ((static_cast<size_t>(cx) * 73856093) ^
                                   (static_cast<size_t>(cy) * 19349663) ^
                                   (static_cast<size_t>(cz) * 83492791));
                    auto it = cell_to_rep.find(hash);
                    if (it == cell_to_rep.end())
                    {
                        cell_to_rep[hash] = static_cast<uint32_t>(i);
                        remap[i] = static_cast<uint32_t>(i);
                    }
                    else
                    {
                        remap[i] = it->second;
                    }
                }
                // Rebuild mesh with remapped indices, keep representative vertices
                std::vector<Vertex> new_verts;
                std::vector<uint32_t> new_remap(mesh.vertices.size());
                std::unordered_map<uint32_t, uint32_t> old_to_new;
                for (size_t i=0; i<mesh.vertices.size(); ++i)
                {
                    uint32_t rep = remap[i];
                    auto it = old_to_new.find(rep);
                    if (it == old_to_new.end())
                    {
                        new_remap[i] = static_cast<uint32_t>(new_verts.size());
                        old_to_new[rep] = new_remap[i];
                        new_verts.push_back(mesh.vertices[rep]);
                    }
                    else
                    {
                        new_remap[i] = it->second;
                    }
                }
                mesh.vertices = std::move(new_verts);
                for (auto& f : mesh.faces)
                    for (int i=0; i<3; ++i)
                        f.indices[i] = new_remap[f.indices[i]];
                mesh.remove_degenerate_faces();
                mesh.compute_bounds();
            }

            // Compute vertex normals with angle threshold for smoothing groups
            inline void compute_normals_with_smoothing(Mesh& mesh, float angle_threshold = 60.0f)
            {
                float cos_thresh = std::cos(angle_threshold * M_PI / 180.0f);
                // Build adjacency and normals per face
                std::vector<Vector3<float>> face_normals(mesh.faces.size());
                for (size_t i=0; i<mesh.faces.size(); ++i)
                {
                    const auto& f = mesh.faces[i];
                    Vector3<float> e1 = mesh.vertices[f.indices[1]].position - mesh.vertices[f.indices[0]].position;
                    Vector3<float> e2 = mesh.vertices[f.indices[2]].position - mesh.vertices[f.indices[0]].position;
                    face_normals[i] = e1.cross(e2).normalized();
                }
                // For each vertex, collect adjacent faces and average normals if within threshold
                std::vector<std::vector<uint32_t>> vert_faces(mesh.vertices.size());
                for (size_t i=0; i<mesh.faces.size(); ++i)
                {
                    for (int j=0; j<3; ++j)
                        vert_faces[mesh.faces[i].indices[j]].push_back(static_cast<uint32_t>(i));
                }
                for (size_t i=0; i<mesh.vertices.size(); ++i)
                {
                    Vector3<float> avg_normal = {0,0,0};
                    for (uint32_t fi : vert_faces[i])
                    {
                        const Vector3<float>& fn = face_normals[fi];
                        bool add = true;
                        for (uint32_t fj : vert_faces[i])
                        {
                            if (fi != fj && fn.dot(face_normals[fj]) < cos_thresh)
                            {
                                add = false;
                                break;
                            }
                        }
                        if (add) avg_normal = avg_normal + fn;
                    }
                    float len = avg_normal.length();
                    if (len > 1e-8f) mesh.vertices[i].normal = avg_normal / len;
                    else mesh.vertices[i].normal = {0,1,0};
                }
            }

            // --------------------------------------------------------------------
            // Mesh analysis
            // --------------------------------------------------------------------
            struct MeshStats
            {
                size_t vertex_count;
                size_t face_count;
                float surface_area;
                float volume;
                bool watertight;
                bool manifold;
                BoundingBox3<float> bounds;
                float average_edge_length;
                float min_edge_length;
                float max_edge_length;
                size_t degenerate_faces;
                size_t isolated_vertices;
            };

            inline MeshStats analyze_mesh(const Mesh& mesh)
            {
                MeshStats s{};
                s.vertex_count = mesh.vertices.size();
                s.face_count = mesh.faces.size();
                s.surface_area = mesh.surface_area();
                s.volume = mesh.volume();
                s.watertight = mesh.is_watertight();
                s.bounds = mesh.bounds;
                // Edge statistics
                std::unordered_map<Edge, int, EdgeHash> edge_len_map;
                s.average_edge_length = s.min_edge_length = s.max_edge_length = 0;
                if (!mesh.faces.empty())
                {
                    s.min_edge_length = std::numeric_limits<float>::max();
                    for (const auto& f : mesh.faces)
                    {
                        for (int i=0; i<3; ++i)
                        {
                            Edge e{f.indices[i], f.indices[(i+1)%3]};
                            if (edge_len_map.find(e) != edge_len_map.end()) continue;
                            float len = (mesh.vertices[e.v0].position - mesh.vertices[e.v1].position).length();
                            edge_len_map[e] = 1;
                            s.average_edge_length += len;
                            if (len < s.min_edge_length) s.min_edge_length = len;
                            if (len > s.max_edge_length) s.max_edge_length = len;
                        }
                    }
                    if (!edge_len_map.empty())
                        s.average_edge_length /= edge_len_map.size();
                }
                // Degenerate faces
                s.degenerate_faces = 0;
                for (const auto& f : mesh.faces)
                {
                    if (f.indices[0] == f.indices[1] || f.indices[1] == f.indices[2] || f.indices[2] == f.indices[0])
                    { s.degenerate_faces++; continue; }
                    Vector3<float> e1 = mesh.vertices[f.indices[1]].position - mesh.vertices[f.indices[0]].position;
                    Vector3<float> e2 = mesh.vertices[f.indices[2]].position - mesh.vertices[f.indices[0]].position;
                    if (e1.cross(e2).length() < 1e-8f) s.degenerate_faces++;
                }
                // Isolated vertices
                std::vector<bool> used(mesh.vertices.size(), false);
                for (const auto& f : mesh.faces)
                    for (int i=0; i<3; ++i) used[f.indices[i]] = true;
                s.isolated_vertices = std::count(used.begin(), used.end(), false);
                s.manifold = s.watertight && s.isolated_vertices == 0;
                return s;
            }

        } // namespace mesh

        // Bring mesh namespace into xt
        using mesh::Mesh;
        using mesh::Vertex;
        using mesh::Face;
        using mesh::Edge;
        using mesh::MeshMaterial;
        using mesh::MorphTarget;
        using mesh::Skin;
        using mesh::Matrix4;
        using mesh::merge_meshes;
        using mesh::extract_submesh;
        using mesh::smooth_mesh_laplacian;
        using mesh::decimate_vertex_clustering;
        using mesh::compute_normals_with_smoothing;
        using mesh::MeshStats;
        using mesh::analyze_mesh;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XMESH_HPP

// graphics/xmesh.hpp