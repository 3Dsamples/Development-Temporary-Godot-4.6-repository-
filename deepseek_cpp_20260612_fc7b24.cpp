// Name : lighting enhancement
// File : scene/3d/navigation_obstacle_3d_ext.cpp 52 of 60
// Description : Implementation of NavigationObstacle3DExt with shape (box/sphere/cylinder),
//               carving margin, avoidance radius, priority, debug wireframe mesh,
//               and full NavigationServer3D + RenderingServer sync.
#include "navigation_obstacle_3d_ext.h"
#include "servers/navigation_server_3d.h"
#include "servers/rendering_server.h"
#include "core/math/math_funcs.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/geometry_3d.h"
#include <cmath>
#include <vector>

struct NavigationObstacle3DExt::Impl {
    RID obstacle_rid;                    // NavigationServer3D obstacle ID
    ShapeType shape = SHAPE_BOX;
    Vector3 size = Vector3(1,1,1);       // half extents for box, radius for sphere/cylinder
    float height = 2.0f;                // for cylinder (total height)
    bool carving_enabled = true;
    float carving_margin = 0.2f;
    bool avoidance_enabled = true;
    float avoidance_radius = 1.0f;
    float avoidance_priority = 1.0f;
    bool enabled = true;

    // Debug visualization
    RID debug_mesh_rid;
    RID debug_instance_rid;
    bool debug_visible = false;
    Color debug_color = Color(1, 0, 0);
    Color debug_emissive_color = Color(0,0,0);
    float debug_emissive_intensity = 0.0f;

    int gi_mode = 0;
    float gi_contribution = 1.0f;

    bool dirty = true;

    Impl() {
        obstacle_rid = NavigationServer3D::get_singleton()->obstacle_create();
        debug_mesh_rid = RenderingServer::get_singleton()->mesh_create();
        debug_instance_rid = RenderingServer::get_singleton()->instance_create();
        RenderingServer::get_singleton()->instance_set_base(debug_instance_rid, debug_mesh_rid);
        RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, false);
    }

    ~Impl() {
        if (obstacle_rid.is_valid()) {
            NavigationServer3D::get_singleton()->free(obstacle_rid);
        }
        if (debug_mesh_rid.is_valid()) {
            RenderingServer::get_singleton()->free(debug_mesh_rid);
        }
        if (debug_instance_rid.is_valid()) {
            RenderingServer::get_singleton()->free(debug_instance_rid);
        }
    }

    void update_navigation_server() {
        NavigationServer3D *ns = NavigationServer3D::get_singleton();
        ns->obstacle_set_shape(obstacle_rid, (int)shape);
        if (shape == SHAPE_BOX) {
            ns->obstacle_set_size(obstacle_rid, size);
        } else if (shape == SHAPE_SPHERE) {
            ns->obstacle_set_radius(obstacle_rid, size.x);
        } else if (shape == SHAPE_CYLINDER) {
            ns->obstacle_set_radius(obstacle_rid, size.x);
            ns->obstacle_set_height(obstacle_rid, height);
        }
        ns->obstacle_set_carving_enabled(obstacle_rid, carving_enabled);
        ns->obstacle_set_carving_margin(obstacle_rid, carving_margin);
        ns->obstacle_set_avoidance_enabled(obstacle_rid, avoidance_enabled);
        ns->obstacle_set_avoidance_radius(obstacle_rid, avoidance_radius);
        ns->obstacle_set_avoidance_priority(obstacle_rid, avoidance_priority);
        ns->obstacle_set_enabled(obstacle_rid, enabled);
        // Also set the obstacle's transform (world position/rotation)
        Transform3D global = get_global_transform();  // from Node3D
        ns->obstacle_set_transform(obstacle_rid, global);
    }

    void generate_box_wireframe(Vector<Vector3> &vertices, Vector<int> &indices) {
        Vector3 h = size;
        Vector3 corners[8] = {
            Vector3(-h.x, -h.y, -h.z), Vector3( h.x, -h.y, -h.z),
            Vector3( h.x, -h.y,  h.z), Vector3(-h.x, -h.y,  h.z),
            Vector3(-h.x,  h.y, -h.z), Vector3( h.x,  h.y, -h.z),
            Vector3( h.x,  h.y,  h.z), Vector3(-h.x,  h.y,  h.z)
        };
        int edges[12][2] = {
            {0,1},{1,2},{2,3},{3,0},
            {4,5},{5,6},{6,7},{7,4},
            {0,4},{1,5},{2,6},{3,7}
        };
        for (int i = 0; i < 12; ++i) {
            vertices.push_back(corners[edges[i][0]]);
            vertices.push_back(corners[edges[i][1]]);
            indices.push_back(i*2);
            indices.push_back(i*2+1);
        }
    }

    void generate_sphere_wireframe(Vector<Vector3> &vertices, Vector<int> &indices, float radius) {
        int rings = 16, slices = 16;
        for (int i = 0; i <= rings; ++i) {
            float theta = i * Math_PI / rings;
            float sin_theta = sin(theta);
            float cos_theta = cos(theta);
            for (int j = 0; j <= slices; ++j) {
                float phi = j * 2.0f * Math_PI / slices;
                float sin_phi = sin(phi);
                float cos_phi = cos(phi);
                float x = radius * sin_theta * cos_phi;
                float y = radius * cos_theta;
                float z = radius * sin_theta * sin_phi;
                vertices.push_back(Vector3(x, y, z));
            }
        }
        // Generate line indices (rings and longitudes)
        for (int i = 0; i < rings; ++i) {
            for (int j = 0; j < slices; ++j) {
                int idx = i * (slices + 1) + j;
                int next_idx = i * (slices + 1) + j + 1;
                int down_idx = (i + 1) * (slices + 1) + j;
                int down_next = (i + 1) * (slices + 1) + j + 1;
                indices.push_back(idx); indices.push_back(next_idx);
                indices.push_back(idx); indices.push_back(down_idx);
                if (i == rings-1) {
                    indices.push_back(down_idx); indices.push_back(down_next);
                }
            }
        }
    }

    void generate_cylinder_wireframe(Vector<Vector3> &vertices, Vector<int> &indices, float radius, float height) {
        int radial_segments = 16;
        float half_h = height * 0.5f;
        // bottom and top rings
        for (int i = 0; i <= radial_segments; ++i) {
            float angle = i * 2.0f * Math_PI / radial_segments;
            float x = radius * cos(angle);
            float z = radius * sin(angle);
            vertices.push_back(Vector3(x, -half_h, z));
            vertices.push_back(Vector3(x,  half_h, z));
        }
        // horizontal edges
        for (int i = 0; i < radial_segments; ++i) {
            int i0 = i * 2;
            int i1 = i * 2 + 1;
            int i2 = (i+1) * 2;
            int i3 = (i+1) * 2 + 1;
            indices.push_back(i0); indices.push_back(i2);
            indices.push_back(i1); indices.push_back(i3);
        }
        // vertical edges
        for (int i = 0; i <= radial_segments; ++i) {
            int base = i * 2;
            indices.push_back(base); indices.push_back(base+1);
        }
    }

    void update_debug_mesh() {
        if (!debug_visible) {
            RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, false);
            return;
        }
        Vector<Vector3> vertices;
        Vector<int> indices;
        if (shape == SHAPE_BOX) {
            generate_box_wireframe(vertices, indices);
        } else if (shape == SHAPE_SPHERE) {
            generate_sphere_wireframe(vertices, indices, size.x);
        } else if (shape == SHAPE_CYLINDER) {
            generate_cylinder_wireframe(vertices, indices, size.x, height);
        }
        if (vertices.size() == 0 || indices.size() == 0) return;

        RenderingServer::get_singleton()->mesh_clear(debug_mesh_rid);
        RenderingServer::get_singleton()->mesh_add_surface(debug_mesh_rid, RS::PRIMITIVE_LINES, vertices, indices, Vector<Vector2>(), Vector<Vector3>());
        RID mat = RenderingServer::get_singleton()->material_create();
        RenderingServer::get_singleton()->material_set_param(mat, "albedo", debug_color);
        if (debug_emissive_intensity > 0.0f) {
            RenderingServer::get_singleton()->material_set_param(mat, "emission", debug_emissive_color);
            RenderingServer::get_singleton()->material_set_param(mat, "emission_intensity", debug_emissive_intensity);
        }
        RenderingServer::get_singleton()->mesh_surface_set_material(debug_mesh_rid, 0, mat);
        // Apply global transform (obstacle's world transform)
        Transform3D global = get_global_transform(); // from Node3D
        RenderingServer::get_singleton()->instance_set_transform(debug_instance_rid, global);
        RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, true);
    }

    void sync() {
        if (dirty) {
            update_navigation_server();
            update_debug_mesh();
            dirty = false;
        }
    }
};

NavigationObstacle3DExt::NavigationObstacle3DExt() {
    pimpl = new Impl();
}

NavigationObstacle3DExt::~NavigationObstacle3DExt() {
    delete pimpl;
}

void NavigationObstacle3DExt::set_shape(ShapeType p_shape) {
    pimpl->shape = p_shape;
    pimpl->dirty = true;
    sync_obstacle();
}
NavigationObstacle3DExt::ShapeType NavigationObstacle3DExt::get_shape() const { return pimpl->shape; }

void NavigationObstacle3DExt::set_size(const Vector3 &p_size) {
    pimpl->size = p_size;
    pimpl->dirty = true;
    sync_obstacle();
}
Vector3 NavigationObstacle3DExt::get_size() const { return pimpl->size; }

void NavigationObstacle3DExt::set_height(float p_height) {
    pimpl->height = p_height;
    pimpl->dirty = true;
    sync_obstacle();
}
float NavigationObstacle3DExt::get_height() const { return pimpl->height; }

void NavigationObstacle3DExt::set_carving_enabled(bool p_enabled) {
    pimpl->carving_enabled = p_enabled;
    pimpl->dirty = true;
    sync_obstacle();
}
bool NavigationObstacle3DExt::is_carving_enabled() const { return pimpl->carving_enabled; }

void NavigationObstacle3DExt::set_carving_margin(float p_margin) {
    pimpl->carving_margin = p_margin;
    pimpl->dirty = true;
    sync_obstacle();
}
float NavigationObstacle3DExt::get_carving_margin() const { return pimpl->carving_margin; }

void NavigationObstacle3DExt::set_avoidance_enabled(bool p_enabled) {
    pimpl->avoidance_enabled = p_enabled;
    pimpl->dirty = true;
    sync_obstacle();
}
bool NavigationObstacle3DExt::is_avoidance_enabled() const { return pimpl->avoidance_enabled; }

void NavigationObstacle3DExt::set_avoidance_radius(float p_radius) {
    pimpl->avoidance_radius = p_radius;
    pimpl->dirty = true;
    sync_obstacle();
}
float NavigationObstacle3DExt::get_avoidance_radius() const { return pimpl->avoidance_radius; }

void NavigationObstacle3DExt::set_avoidance_priority(float p_priority) {
    pimpl->avoidance_priority = p_priority;
    pimpl->dirty = true;
    sync_obstacle();
}
float NavigationObstacle3DExt::get_avoidance_priority() const { return pimpl->avoidance_priority; }

void NavigationObstacle3DExt::set_enabled(bool p_enabled) {
    pimpl->enabled = p_enabled;
    pimpl->dirty = true;
    sync_obstacle();
}
bool NavigationObstacle3DExt::is_enabled() const { return pimpl->enabled; }

void NavigationObstacle3DExt::set_debug_visible(bool p_visible) {
    pimpl->debug_visible = p_visible;
    pimpl->dirty = true;
    sync_obstacle();
}
bool NavigationObstacle3DExt::is_debug_visible() const { return pimpl->debug_visible; }

void NavigationObstacle3DExt::set_debug_color(const Color &p_color) {
    pimpl->debug_color = p_color;
    pimpl->dirty = true;
    sync_obstacle();
}
Color NavigationObstacle3DExt::get_debug_color() const { return pimpl->debug_color; }

void NavigationObstacle3DExt::set_debug_emissive(const Color &p_color, float p_intensity) {
    pimpl->debug_emissive_color = p_color;
    pimpl->debug_emissive_intensity = p_intensity;
    pimpl->dirty = true;
    sync_obstacle();
}
void NavigationObstacle3DExt::get_debug_emissive(Color &r_color, float &r_intensity) const {
    r_color = pimpl->debug_emissive_color;
    r_intensity = pimpl->debug_emissive_intensity;
}

void NavigationObstacle3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
}
int NavigationObstacle3DExt::get_gi_mode() const { return pimpl->gi_mode; }

void NavigationObstacle3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
}
float NavigationObstacle3DExt::get_gi_contribution() const { return pimpl->gi_contribution; }

void NavigationObstacle3DExt::sync_obstacle() {
    pimpl->sync();
}