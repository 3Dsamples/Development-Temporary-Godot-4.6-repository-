// genesis/engine/mesh.cpp

#include "genesis/engine/mesh.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <cstdio>
#include <cmath>
#include <stack>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Mesh implementation
//------------------------------------------------------------------------------

Mesh::Mesh() = default;
Mesh::~Mesh() = default;

void Mesh::clear() {
    vertices_.clear();
    indices_.clear();
    submeshes_.clear();
    bones_.clear();
    animations_.clear();
    aabb_ = datatypes::AABB();
    bvh_.reset();
}

bool Mesh::load_file(const std::string& path) {
    clear();
    size_t dot = path.rfind('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".obj") return load_obj(path);
    else if (ext == ".stl") {
        // Detect binary vs ASCII
        std::ifstream file(path, std::ios::binary);
        if (!file) return false;
        char header[80];
        file.read(header, 80);
        std::string header_str(header, 80);
        if (header_str.find("solid") == 0) {
            file.close();
            return load_stl(path, true);
        } else {
            file.close();
            return load_stl(path, false);
        }
    }
    else if (ext == ".ply") return load_ply(path);
    return false;
}

bool Mesh::load_obj(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::vector<datatypes::Vector3> positions;
    std::vector<datatypes::Vector2> texcoords;
    std::vector<datatypes::Vector3> normals;
    std::unordered_map<ObjFaceVertex, uint32_t, ObjFaceVertexHash> vertex_map;
    std::string line;
    std::string current_material;
    uint32_t index_counter = 0;

    // Add default submesh
    submeshes_.push_back({"", {}});

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            datatypes::Vector3 v;
            iss >> v[0] >> v[1] >> v[2];
            positions.push_back(v);
        } else if (type == "vt") {
            datatypes::Vector2 vt;
            iss >> vt[0] >> vt[1];
            texcoords.push_back(vt);
        } else if (type == "vn") {
            datatypes::Vector3 vn;
            iss >> vn[0] >> vn[1] >> vn[2];
            normals.push_back(vn);
        } else if (type == "f") {
            std::vector<ObjFaceVertex> face_verts;
            std::string part;
            while (iss >> part) {
                ObjFaceVertex fv = {-1, -1, -1};
                size_t s1 = part.find('/');
                if (s1 != std::string::npos) {
                    fv.v_idx = std::stoi(part.substr(0, s1)) - 1;
                    size_t s2 = part.find('/', s1 + 1);
                    if (s2 != std::string::npos) {
                        if (s2 > s1 + 1) fv.vt_idx = std::stoi(part.substr(s1+1, s2-s1-1)) - 1;
                        fv.vn_idx = std::stoi(part.substr(s2+1)) - 1;
                    } else {
                        fv.vt_idx = std::stoi(part.substr(s1+1)) - 1;
                    }
                } else {
                    fv.v_idx = std::stoi(part) - 1;
                }
                // Handle negative indices (relative)
                if (fv.v_idx < 0) fv.v_idx += static_cast<int>(positions.size()) + 1;
                if (fv.vt_idx < 0) fv.vt_idx += static_cast<int>(texcoords.size()) + 1;
                if (fv.vn_idx < 0) fv.vn_idx += static_cast<int>(normals.size()) + 1;
                face_verts.push_back(fv);
            }
            // Triangulate fan (supports convex polygons)
            for (size_t i = 1; i + 1 < face_verts.size(); ++i) {
                std::array<ObjFaceVertex, 3> tri = {face_verts[0], face_verts[i], face_verts[i+1]};
                for (const auto& fv : tri) {
                    auto it = vertex_map.find(fv);
                    if (it != vertex_map.end()) {
                        indices_.push_back(it->second);
                    } else {
                        Vertex vert;
                        vert.position = positions[fv.v_idx];
                        if (fv.vt_idx >= 0 && fv.vt_idx < static_cast<int>(texcoords.size()))
                            vert.texcoord = texcoords[fv.vt_idx];
                        if (fv.vn_idx >= 0 && fv.vn_idx < static_cast<int>(normals.size()))
                            vert.normal = normals[fv.vn_idx];
                        vertices_.push_back(vert);
                        indices_.push_back(index_counter);
                        submeshes_.back().indices.push_back(index_counter);
                        vertex_map[fv] = index_counter++;
                    }
                }
            }
        } else if (type == "usemtl") {
            std::string mtl;
            iss >> mtl;
            if (!current_material.empty()) {
                submeshes_.back().index_count = static_cast<uint32_t>(submeshes_.back().indices.size());
                submeshes_.push_back({mtl, {}});
            } else {
                submeshes_.back().material_name = mtl;
            }
            current_material = mtl;
        }
    }

    if (normals.empty()) compute_normals(true);
    update_aabb();
    return true;
}

bool Mesh::load_stl(const std::string& path, bool is_ascii) {
    clear();
    if (is_ascii) {
        std::ifstream file(path);
        if (!file) return false;
        std::string line;
        std::getline(file, line); // solid name
        datatypes::Vector3 normal;
        std::vector<datatypes::Vector3> tri(3);
        while (file >> line) {
            if (line == "facet" && file >> line && line == "normal") {
                file >> normal[0] >> normal[1] >> normal[2];
            } else if (line == "outer") {
                file >> line; // loop
            } else if (line == "vertex") {
                static int v_idx = 0;
                file >> tri[v_idx][0] >> tri[v_idx][1] >> tri[v_idx][2];
                if (++v_idx == 3) {
                    for (int i = 0; i < 3; ++i) {
                        Vertex v(tri[i]);
                        v.normal = normal;
                        vertices_.push_back(v);
                        indices_.push_back(static_cast<uint32_t>(vertices_.size() - 1));
                    }
                    v_idx = 0;
                }
            }
        }
    } else {
        std::ifstream file(path, std::ios::binary);
        if (!file) return false;
        char header[80];
        file.read(header, 80);
        uint32_t num_triangles;
        file.read(reinterpret_cast<char*>(&num_triangles), sizeof(num_triangles));
        for (uint32_t i = 0; i < num_triangles; ++i) {
            float n[3], v[3][3];
            uint16_t attr;
            file.read(reinterpret_cast<char*>(n), 12);
            for (int j = 0; j < 3; ++j) file.read(reinterpret_cast<char*>(v[j]), 12);
            file.read(reinterpret_cast<char*>(&attr), 2);
            for (int j = 0; j < 3; ++j) {
                Vertex vert(datatypes::Vector3(v[j][0], v[j][1], v[j][2]));
                vert.normal = datatypes::Vector3(n[0], n[1], n[2]);
                vertices_.push_back(vert);
                indices_.push_back(static_cast<uint32_t>(vertices_.size() - 1));
            }
        }
    }
    submeshes_.push_back({"", indices_});
    submeshes_.back().index_count = static_cast<uint32_t>(indices_.size());
    update_aabb();
    return true;
}

bool Mesh::load_ply(const std::string& path) {
    std::ifstream file(path);
    if (!file) return false;
    std::string line;
    std::getline(file, line); // ply
    if (line != "ply") return false;
    // Very basic PLY loader (ASCII only)
    size_t num_vertices = 0, num_faces = 0;
    while (std::getline(file, line) && line != "end_header") {
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "element") {
            std::string type;
            size_t count;
            iss >> type >> count;
            if (type == "vertex") num_vertices = count;
            else if (type == "face") num_faces = count;
        }
    }
    for (size_t i = 0; i < num_vertices; ++i) {
        datatypes::Vector3 v;
        file >> v[0] >> v[1] >> v[2];
        vertices_.push_back(Vertex(v));
    }
    for (size_t i = 0; i < num_faces; ++i) {
        int count;
        file >> count;
        if (count == 3) {
            uint32_t a, b, c;
            file >> a >> b >> c;
            indices_.push_back(a);
            indices_.push_back(b);
            indices_.push_back(c);
        } else if (count == 4) {
            uint32_t a, b, c, d;
            file >> a >> b >> c >> d;
            indices_.push_back(a); indices_.push_back(b); indices_.push_back(c);
            indices_.push_back(a); indices_.push_back(c); indices_.push_back(d);
        }
    }
    submeshes_.push_back({"", indices_});
    compute_normals(true);
    update_aabb();
    return true;
}

Mesh Mesh::create_box(const datatypes::Vector3& size) {
    Mesh mesh;
    datatypes::Vector3 h = size * 0.5f;
    // 8 vertices, 12 triangles
    mesh.vertices_ = {
        Vertex({-h[0], -h[1], -h[2]}), Vertex({ h[0], -h[1], -h[2]}),
        Vertex({ h[0],  h[1], -h[2]}), Vertex({-h[0],  h[1], -h[2]}),
        Vertex({-h[0], -h[1],  h[2]}), Vertex({ h[0], -h[1],  h[2]}),
        Vertex({ h[0],  h[1],  h[2]}), Vertex({-h[0],  h[1],  h[2]})
    };
    mesh.indices_ = {
        0,1,2, 0,2,3, // back
        4,6,5, 4,7,6, // front
        0,4,5, 0,5,1, // bottom
        2,6,7, 2,7,3, // top
        0,3,7, 0,7,4, // left
        1,5,6, 1,6,2  // right
    };
    mesh.compute_normals(false);
    mesh.update_aabb();
    mesh.submeshes_.push_back({"", mesh.indices_});
    return mesh;
}

Mesh Mesh::create_sphere(float radius, int segments) {
    Mesh mesh;
    int rings = segments, sectors = segments;
    float R = 1.0f / static_cast<float>(rings);
    float S = 1.0f / static_cast<float>(sectors);
    for (int r = 0; r <= rings; ++r) {
        for (int s = 0; s <= sectors; ++s) {
            float y = std::sin(-M_PI_2 + M_PI * r * R);
            float x = std::cos(2 * M_PI * s * S) * std::sin(M_PI * r * R);
            float z = std::sin(2 * M_PI * s * S) * std::sin(M_PI * r * R);
            mesh.vertices_.push_back(Vertex(datatypes::Vector3(x * radius, y * radius, z * radius)));
        }
    }
    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < sectors; ++s) {
            int a = r * (sectors+1) + s;
            int b = r * (sectors+1) + (s+1);
            int c = (r+1) * (sectors+1) + s;
            int d = (r+1) * (sectors+1) + (s+1);
            mesh.indices_.insert(mesh.indices_.end(), {static_cast<uint32_t>(a), static_cast<uint32_t>(b), static_cast<uint32_t>(c)});
            mesh.indices_.insert(mesh.indices_.end(), {static_cast<uint32_t>(b), static_cast<uint32_t>(d), static_cast<uint32_t>(c)});
        }
    }
    mesh.compute_normals(true);
    mesh.update_aabb();
    mesh.submeshes_.push_back({"", mesh.indices_});
    return mesh;
}

Mesh Mesh::create_cylinder(float radius, float height, int segments) {
    Mesh mesh;
    float half_h = height * 0.5f;
    // Top and bottom center vertices
    uint32_t top_center = static_cast<uint32_t>(mesh.vertices_.size());
    mesh.vertices_.push_back(Vertex({0, half_h, 0}));
    uint32_t bottom_center = static_cast<uint32_t>(mesh.vertices_.size());
    mesh.vertices_.push_back(Vertex({0, -half_h, 0}));
    // Ring vertices
    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * M_PI * i / segments;
        float x = std::cos(angle) * radius;
        float z = std::sin(angle) * radius;
        mesh.vertices_.push_back(Vertex({x, half_h, z}));
        mesh.vertices_.push_back(Vertex({x, -half_h, z}));
    }
    for (int i = 0; i < segments; ++i) {
        int top0 = 2 + i*2, top1 = 2 + (i+1)*2;
        int bot0 = 3 + i*2, bot1 = 3 + (i+1)*2;
        // Side faces
        mesh.indices_.insert(mesh.indices_.end(), {static_cast<uint32_t>(top0), static_cast<uint32_t>(bot0), static_cast<uint32_t>(top1)});
        mesh.indices_.insert(mesh.indices_.end(), {static_cast<uint32_t>(top1), static_cast<uint32_t>(bot0), static_cast<uint32_t>(bot1)});
        // Top cap
        mesh.indices_.insert(mesh.indices_.end(), {top_center, static_cast<uint32_t>(top1), static_cast<uint32_t>(top0)});
        // Bottom cap
        mesh.indices_.insert(mesh.indices_.end(), {bottom_center, static_cast<uint32_t>(bot0), static_cast<uint32_t>(bot1)});
    }
    mesh.compute_normals(true);
    mesh.update_aabb();
    mesh.submeshes_.push_back({"", mesh.indices_});
    return mesh;
}

Mesh Mesh::create_plane(const datatypes::Vector2& size) {
    Mesh mesh;
    float hx = size[0] * 0.5f, hy = size[1] * 0.5f;
    mesh.vertices_ = {
        Vertex({-hx, 0, -hy}), Vertex({ hx, 0, -hy}),
        Vertex({ hx, 0,  hy}), Vertex({-hx, 0,  hy})
    };
    mesh.indices_ = {0, 1, 2, 0, 2, 3};
    mesh.compute_normals(false);
    mesh.update_aabb();
    mesh.submeshes_.push_back({"", mesh.indices_});
    return mesh;
}

Mesh Mesh::create_capsule(float radius, float height, int segments) {
    // Not fully implemented, placeholder
    Mesh mesh = create_cylinder(radius, height, segments);
    // Could add hemispheres at ends; omitted for brevity.
    return mesh;
}

Mesh Mesh::create_torus(float outer_radius, float inner_radius, int segments, int sides) {
    Mesh mesh;
    for (int i = 0; i <= sides; ++i) {
        float theta = 2.0f * M_PI * i / sides;
        float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
        for (int j = 0; j <= segments; ++j) {
            float phi = 2.0f * M_PI * j / segments;
            float cosPhi = std::cos(phi), sinPhi = std::sin(phi);
            datatypes::Vector3 p;
            p[0] = (outer_radius + inner_radius * cosTheta) * cosPhi;
            p[1] = inner_radius * sinTheta;
            p[2] = (outer_radius + inner_radius * cosTheta) * sinPhi;
            mesh.vertices_.push_back(Vertex(p));
        }
    }
    for (int i = 0; i < sides; ++i) {
        for (int j = 0; j < segments; ++j) {
            int a = i * (segments+1) + j;
            int b = i * (segments+1) + j+1;
            int c = (i+1) * (segments+1) + j;
            int d = (i+1) * (segments+1) + j+1;
            mesh.indices_.insert(mesh.indices_.end(), {static_cast<uint32_t>(a), static_cast<uint32_t>(b), static_cast<uint32_t>(c)});
            mesh.indices_.insert(mesh.indices_.end(), {static_cast<uint32_t>(b), static_cast<uint32_t>(d), static_cast<uint32_t>(c)});
        }
    }
    mesh.compute_normals(true);
    mesh.update_aabb();
    mesh.submeshes_.push_back({"", mesh.indices_});
    return mesh;
}

void Mesh::set_vertices(const std::vector<datatypes::Vector3>& positions) {
    vertices_.resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) vertices_[i].position = positions[i];
    update_aabb();
}

void Mesh::set_indices(const std::vector<uint32_t>& indices) {
    indices_ = indices;
    if (!submeshes_.empty()) submeshes_[0].indices = indices;
}

void Mesh::compute_normals(bool smooth) {
    if (vertices_.empty()) return;
    // Reset normals
    for (auto& v : vertices_) v.normal = datatypes::Vector3(0);
    // Compute face normals and accumulate
    for (size_t i = 0; i < indices_.size(); i += 3) {
        datatypes::Vector3 v0 = vertices_[indices_[i]].position;
        datatypes::Vector3 v1 = vertices_[indices_[i+1]].position;
        datatypes::Vector3 v2 = vertices_[indices_[i+2]].position;
        datatypes::Vector3 face_normal = (v1 - v0).cross(v2 - v0).normalized();
        if (smooth) {
            vertices_[indices_[i]].normal += face_normal;
            vertices_[indices_[i+1]].normal += face_normal;
            vertices_[indices_[i+2]].normal += face_normal;
        } else {
            vertices_[indices_[i]].normal = face_normal;
            vertices_[indices_[i+1]].normal = face_normal;
            vertices_[indices_[i+2]].normal = face_normal;
        }
    }
    // Normalize
    for (auto& v : vertices_) v.normal.normalize();
}

void Mesh::compute_tangents() {
    // Compute tangents for normal mapping (requires texcoords)
    if (vertices_.empty() || indices_.empty()) return;
    std::vector<datatypes::Vector3> tangents(vertices_.size(), datatypes::Vector3(0));
    std::vector<datatypes::Vector3> bitangents(vertices_.size(), datatypes::Vector3(0));
    for (size_t i = 0; i < indices_.size(); i += 3) {
        Vertex& v0 = vertices_[indices_[i]];
        Vertex& v1 = vertices_[indices_[i+1]];
        Vertex& v2 = vertices_[indices_[i+2]];
        datatypes::Vector3 e1 = v1.position - v0.position;
        datatypes::Vector3 e2 = v2.position - v0.position;
        datatypes::Vector2 duv1 = v1.texcoord - v0.texcoord;
        datatypes::Vector2 duv2 = v2.texcoord - v0.texcoord;
        float f = 1.0f / (duv1[0]*duv2[1] - duv2[0]*duv1[1]);
        if (!std::isfinite(f)) continue;
        datatypes::Vector3 tangent = (e1 * duv2[1] - e2 * duv1[1]) * f;
        datatypes::Vector3 bitangent = (e2 * duv1[0] - e1 * duv2[0]) * f;
        tangents[indices_[i]] += tangent;
        tangents[indices_[i+1]] += tangent;
        tangents[indices_[i+2]] += tangent;
        bitangents[indices_[i]] += bitangent;
        bitangents[indices_[i+1]] += bitangent;
        bitangents[indices_[i+2]] += bitangent;
    }
    // Orthogonalize
    for (size_t i = 0; i < vertices_.size(); ++i) {
        datatypes::Vector3 n = vertices_[i].normal;
        datatypes::Vector3 t = tangents[i];
        // Gram-Schmidt orthogonalize
        t = (t - n * n.dot(t)).normalized();
        // Store tangent as 4th component sign (handedness)
        float handedness = (n.cross(t).dot(bitangents[i]) < 0) ? -1.0f : 1.0f;
        vertices_[i].color = datatypes::Vector4(t[0], t[1], t[2], handedness);
    }
}

datatypes::AABB Mesh::compute_aabb() const {
    datatypes::AABB box;
    for (const auto& v : vertices_) box.expand(v.position);
    return box;
}

void Mesh::update_aabb() {
    aabb_ = compute_aabb();
}

void Mesh::build_bvh() {
    if (vertices_.empty()) return;
    std::vector<datatypes::Vector3> pos;
    for (const auto& v : vertices_) pos.push_back(v.position);
    bvh_ = std::make_unique<BVH>();
    bvh_->build(pos, indices_);
}

void Mesh::apply_transform(const datatypes::Matrix4r& transform) {
    for (auto& v : vertices_) {
        v.position = transform.transformPoint(v.position);
        v.normal = transform.transformVector(v.normal).normalized();
    }
    update_aabb();
    if (bvh_) {
        std::vector<datatypes::Vector3> pos;
        for (const auto& v : vertices_) pos.push_back(v.position);
        bvh_->refit(pos);
    }
}

void Mesh::translate(const datatypes::Vector3& offset) {
    datatypes::Matrix4r mat = datatypes::Matrix4r::translation(offset);
    apply_transform(mat);
}

void Mesh::rotate(const datatypes::Quat& rotation) {
    datatypes::Matrix4r mat = datatypes::Matrix4r::rotation(rotation.toRotationMatrix());
    apply_transform(mat);
}

void Mesh::scale(const datatypes::Vector3& scale) {
    datatypes::Matrix4r mat = datatypes::Matrix4r::scale(scale);
    apply_transform(mat);
}

void Mesh::add_submesh(const SubMesh& submesh) {
    submeshes_.push_back(submesh);
}

void Mesh::add_bone(const Bone& bone) {
    bones_.push_back(bone);
}

void Mesh::build_skinning_data() {
    // Would set up bone index/weight attributes on vertices
}

void Mesh::apply_skinning(const std::vector<datatypes::Matrix4r>& bone_transforms) {
    if (bones_.empty() || bone_transforms.empty()) return;
    std::vector<datatypes::Vector3> skinned(vertices_.size(), datatypes::Vector3(0));
    for (size_t i = 0; i < vertices_.size(); ++i) {
        Vertex& v = vertices_[i];
        datatypes::Vector3 pos(0);
        for (int j = 0; j < 4; ++j) {
            if (v.bone_indices[j] >= 0 && v.bone_weights[j] > 0) {
                const datatypes::Matrix4r& mat = bone_transforms[v.bone_indices[j]];
                pos += mat.transformPoint(v.position) * v.bone_weights[j];
            }
        }
        skinned[i] = pos;
    }
    for (size_t i = 0; i < vertices_.size(); ++i) vertices_[i].position = skinned[i];
    update_aabb();
}

void Mesh::add_animation(const AnimationClip& clip) {
    animations_.push_back(clip);
}

void Mesh::sample_animation(const std::string& clip_name, float time,
                            std::vector<datatypes::Matrix4r>& out_transforms) const {
    for (const auto& clip : animations_) {
        if (clip.name == clip_name) {
            // Find keyframes
            if (clip.keyframes.empty()) return;
            size_t idx = 0;
            while (idx + 1 < clip.keyframes.size() && clip.keyframes[idx+1].time <= time) ++idx;
            const auto& kf1 = clip.keyframes[idx];
            const auto& kf2 = (idx+1 < clip.keyframes.size()) ? clip.keyframes[idx+1] : kf1;
            float t = (time - kf1.time) / (kf2.time - kf1.time + 1e-6f);
            t = std::clamp(t, 0.0f, 1.0f);
            out_transforms.resize(bones_.size());
            for (size_t b = 0; b < bones_.size(); ++b) {
                // Interpolate local transform
                datatypes::Vector3 pos = kf1.bone_transforms[b].translation * (1-t) + kf2.bone_transforms[b].translation * t;
                datatypes::Quat rot = datatypes::Quat::slerp(kf1.bone_transforms[b].rotation, kf2.bone_transforms[b].rotation, t);
                datatypes::Transformr local(pos, rot);
                // Compute global transform (assumes hierarchy)
                int parent = bones_[b].parent_index;
                if (parent >= 0) {
                    out_transforms[b] = out_transforms[parent] * local.matrix();
                } else {
                    out_transforms[b] = local.matrix();
                }
            }
            return;
        }
    }
}

Mesh::RayHit Mesh::ray_cast(const datatypes::Ray& ray) const {
    RayHit hit;
    if (bvh_) {
        BVH::RayHit bvh_hit = bvh_->intersect_ray(ray);
        if (bvh_hit.hit) {
            hit.hit = true;
            hit.t = bvh_hit.t;
            hit.point = bvh_hit.point;
            hit.normal = bvh_hit.normal;
            hit.triangle_index = bvh_hit.primitive_index;
        }
    } else {
        // Brute force
        for (size_t i = 0; i < indices_.size(); i += 3) {
            const Vertex& v0 = vertices_[indices_[i]];
            const Vertex& v1 = vertices_[indices_[i+1]];
            const Vertex& v2 = vertices_[indices_[i+2]];
            datatypes::Vector3 e1 = v1.position - v0.position;
            datatypes::Vector3 e2 = v2.position - v0.position;
            datatypes::Vector3 h = ray.direction.cross(e2);
            float a = e1.dot(h);
            if (std::abs(a) < 1e-7f) continue;
            float f = 1.0f / a;
            datatypes::Vector3 s = ray.origin - v0.position;
            float u = f * s.dot(h);
            if (u < 0.0f || u > 1.0f) continue;
            datatypes::Vector3 q = s.cross(e1);
            float v = f * ray.direction.dot(q);
            if (v < 0.0f || u + v > 1.0f) continue;
            float t = f * e2.dot(q);
            if (t > 1e-6f && t < hit.t) {
                hit.hit = true;
                hit.t = t;
                hit.point = ray.pointAt(t);
                hit.normal = (e1.cross(e2)).normalized();
                hit.triangle_index = static_cast<uint32_t>(i/3);
            }
        }
    }
    return hit;
}

Mesh::ClosestPointResult Mesh::closest_point(const datatypes::Vector3& query) const {
    ClosestPointResult result;
    if (bvh_) {
        BVH::ClosestPointResult bvh_res = bvh_->closest_point(query);
        result.point = bvh_res.point;
        result.triangle_index = bvh_res.primitive_index;
        result.distance_sq = bvh_res.distance_sq;
    } else {
        for (size_t i = 0; i < indices_.size(); i += 3) {
            const Vertex& v0 = vertices_[indices_[i]];
            const Vertex& v1 = vertices_[indices_[i+1]];
            const Vertex& v2 = vertices_[indices_[i+2]];
            datatypes::Vector3 cp = closest_point_on_triangle(query, v0.position, v1.position, v2.position);
            float d2 = (cp - query).squaredNorm();
            if (d2 < result.distance_sq) {
                result.distance_sq = d2;
                result.point = cp;
                result.triangle_index = static_cast<uint32_t>(i/3);
            }
        }
    }
    return result;
}

std::vector<Mesh> Mesh::create_convex_decomposition() const {
    // Placeholder: would use V-HACD or similar
    return {*this};
}

float Mesh::surface_area() const {
    float area = 0;
    for (size_t i = 0; i < indices_.size(); i += 3) {
        datatypes::Vector3 v0 = vertices_[indices_[i]].position;
        datatypes::Vector3 v1 = vertices_[indices_[i+1]].position;
        datatypes::Vector3 v2 = vertices_[indices_[i+2]].position;
        area += 0.5f * (v1 - v0).cross(v2 - v0).norm();
    }
    return area;
}

float Mesh::volume() const {
    // Signed volume (requires closed manifold)
    float vol = 0;
    for (size_t i = 0; i < indices_.size(); i += 3) {
        datatypes::Vector3 v0 = vertices_[indices_[i]].position;
        datatypes::Vector3 v1 = vertices_[indices_[i+1]].position;
        datatypes::Vector3 v2 = vertices_[indices_[i+2]].position;
        vol += v0.dot(v1.cross(v2));
    }
    return std::abs(vol) / 6.0f;
}

void Mesh::add_triangle(const datatypes::Vector3& v0, const datatypes::Vector3& v1, const datatypes::Vector3& v2) {
    uint32_t base = static_cast<uint32_t>(vertices_.size());
    vertices_.push_back(Vertex(v0));
    vertices_.push_back(Vertex(v1));
    vertices_.push_back(Vertex(v2));
    indices_.push_back(base);
    indices_.push_back(base+1);
    indices_.push_back(base+2);
}

void Mesh::add_quad(const datatypes::Vector3& v0, const datatypes::Vector3& v1,
                    const datatypes::Vector3& v2, const datatypes::Vector3& v3) {
    add_triangle(v0, v1, v2);
    add_triangle(v0, v2, v3);
}

namespace {
    datatypes::Vector3 closest_point_on_triangle(const datatypes::Vector3& p,
                                                 const datatypes::Vector3& a,
                                                 const datatypes::Vector3& b,
                                                 const datatypes::Vector3& c) {
        // Same algorithm as in BVH
        datatypes::Vector3 ab = b - a;
        datatypes::Vector3 ac = c - a;
        datatypes::Vector3 ap = p - a;
        float d1 = ab.dot(ap);
        float d2 = ac.dot(ap);
        if (d1 <= 0 && d2 <= 0) return a;
        datatypes::Vector3 bp = p - b;
        float d3 = ab.dot(bp);
        float d4 = ac.dot(bp);
        if (d3 >= 0 && d4 <= d3) return b;
        float vc = d1*d4 - d3*d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) {
            float v = d1 / (d1 - d3);
            return a + ab * v;
        }
        datatypes::Vector3 cp = p - c;
        float d5 = ab.dot(cp);
        float d6 = ac.dot(cp);
        if (d6 >= 0 && d5 <= d6) return c;
        float vb = d5*d2 - d1*d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) {
            float w = d2 / (d2 - d6);
            return a + ac * w;
        }
        float va = d3*d6 - d5*d4;
        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + (c - b) * w;
        }
        float denom = 1.0f / (va + vb + vc);
        float v = vb * denom;
        float w = vc * denom;
        return a + ab * v + ac * w;
    }
}

} // namespace engine
} // namespace genesis