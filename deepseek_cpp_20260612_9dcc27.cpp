// Name : lighting enhancement
// File : scene/3d/navigation_link_3d_ext.cpp 50 of 60
// Description : Implementation of NavigationLink3DExt with bidirectional link,
//               radius, costs, cylinder debug mesh (with optional arrowhead),
//               and full NavigationServer3D + RenderingServer sync.
#include "navigation_link_3d_ext.h"
#include "servers/navigation_server_3d.h"
#include "servers/rendering_server.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/math_funcs.h"
#include <cmath>

struct NavigationLink3DExt::Impl {
    RID link_rid;                       // NavigationServer3D link ID
    Vector3 start;
    Vector3 end;
    float radius = 0.5f;
    bool bidirectional = true;
    float enter_cost = 0.0f;
    float traversal_cost = 0.0f;
    bool enabled = true;

    // Debug visualization
    RID debug_mesh_rid;
    RID debug_instance_rid;
    bool debug_visible = false;
    Color debug_color = Color(0.2, 0.8, 0.2);
    Color debug_emissive_color = Color(0,0,0);
    float debug_emissive_intensity = 0.0f;

    int gi_mode = 0;
    float gi_contribution = 1.0f;

    bool dirty = true;

    Impl() {
        link_rid = NavigationServer3D::get_singleton()->link_create();
        debug_mesh_rid = RenderingServer::get_singleton()->mesh_create();
        debug_instance_rid = RenderingServer::get_singleton()->instance_create();
        RenderingServer::get_singleton()->instance_set_base(debug_instance_rid, debug_mesh_rid);
        RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, false);
    }

    ~Impl() {
        if (link_rid.is_valid()) {
            NavigationServer3D::get_singleton()->free(link_rid);
        }
        if (debug_mesh_rid.is_valid()) {
            RenderingServer::get_singleton()->free(debug_mesh_rid);
        }
        if (debug_instance_rid.is_valid()) {
            RenderingServer::get_singleton()->free(debug_instance_rid);
        }
    }

    void update_navigation_link() {
        NavigationServer3D *ns = NavigationServer3D::get_singleton();
        ns->link_set_start_position(link_rid, start);
        ns->link_set_end_position(link_rid, end);
        ns->link_set_radius(link_rid, radius);
        ns->link_set_bidirectional(link_rid, bidirectional);
        ns->link_set_enter_cost(link_rid, enter_cost);
        ns->link_set_traversal_cost(link_rid, traversal_cost);
        ns->link_set_enabled(link_rid, enabled);
        // NavigationServer uses the link's global transform (if any), but positions are absolute.
        // Also set map if needed (the region will register the link).
    }

    void update_debug_mesh() {
        if (!debug_visible) {
            RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, false);
            return;
        }
        Vector3 dir = end - start;
        float length = dir.length();
        if (length < 0.001f) return;
        Vector3 axis = dir.normalized();
        // Build a cylinder mesh (tube) between start and end with given radius.
        // We'll generate a cylinder aligned with Z, then rotate to align with axis.
        int radial_segments = 12;
        int height_segments = 2;  // only one section, but we need two rings
        std::vector<Vector3> vertices;
        std::vector<int> indices;
        float r = radius;
        // Precompute cylinder points in local Z orientation: bottom ring at z = -length/2, top ring at z = length/2.
        float half_len = length * 0.5f;
        for (int i = 0; i <= radial_segments; ++i) {
            float angle = i * 2.0f * Math_PI / radial_segments;
            float x = r * cos(angle);
            float y = r * sin(angle);
            // bottom ring
            vertices.push_back(Vector3(x, y, -half_len));
            // top ring
            vertices.push_back(Vector3(x, y,  half_len));
        }
        // Create side indices (quads)
        for (int i = 0; i < radial_segments; ++i) {
            int i0 = i * 2;
            int i1 = i * 2 + 1;
            int i2 = (i+1) * 2;
            int i3 = (i+1) * 2 + 1;
            indices.push_back(i0); indices.push_back(i1); indices.push_back(i2);
            indices.push_back(i1); indices.push_back(i3); indices.push_back(i2);
        }
        // Optionally add arrowhead if link is directional (endpoint). Not required for debug.
        // Transform vertices from local orientation to world orientation (aligning Z to axis).
        Quaternion quat = Quaternion().set_from_axis_angle(Vector3(0,0,1), axis);
        Transform3D transform;
        transform.basis = Basis(quat);
        transform.origin = (start + end) * 0.5f;
        for (Vector3 &v : vertices) {
            v = transform * v;
        }
        // Clear and add surface
        RenderingServer::get_singleton()->mesh_clear(debug_mesh_rid);
        if (vertices.size() < 3 || indices.size() < 3) return;
        RenderingServer::get_singleton()->mesh_add_surface(debug_mesh_rid, RS::PRIMITIVE_TRIANGLES, vertices, indices, Vector<Vector2>(), Vector<Vector3>());
        // Create material
        RID mat = RenderingServer::get_singleton()->material_create();
        RenderingServer::get_singleton()->material_set_param(mat, "albedo", debug_color);
        if (debug_emissive_intensity > 0.0f) {
            RenderingServer::get_singleton()->material_set_param(mat, "emission", debug_emissive_color);
            RenderingServer::get_singleton()->material_set_param(mat, "emission_intensity", debug_emissive_intensity);
        }
        RenderingServer::get_singleton()->mesh_surface_set_material(debug_mesh_rid, 0, mat);
        RenderingServer::get_singleton()->instance_set_transform(debug_instance_rid, Transform3D());
        RenderingServer::get_singleton()->instance_set_visible(debug_instance_rid, true);
    }

    void sync() {
        if (dirty) {
            update_navigation_link();
            update_debug_mesh();
            dirty = false;
        }
    }
};

NavigationLink3DExt::NavigationLink3DExt() {
    pimpl = new Impl();
}

NavigationLink3DExt::~NavigationLink3DExt() {
    delete pimpl;
}

void NavigationLink3DExt::set_start_position(const Vector3 &p_position) {
    pimpl->start = p_position;
    pimpl->dirty = true;
    sync_link();
}
Vector3 NavigationLink3DExt::get_start_position() const { return pimpl->start; }

void NavigationLink3DExt::set_end_position(const Vector3 &p_position) {
    pimpl->end = p_position;
    pimpl->dirty = true;
    sync_link();
}
Vector3 NavigationLink3DExt::get_end_position() const { return pimpl->end; }

void NavigationLink3DExt::set_radius(float p_radius) {
    pimpl->radius = p_radius;
    pimpl->dirty = true;
    sync_link();
}
float NavigationLink3DExt::get_radius() const { return pimpl->radius; }

void NavigationLink3DExt::set_bidirectional(bool p_bidirectional) {
    pimpl->bidirectional = p_bidirectional;
    pimpl->dirty = true;
    sync_link();
}
bool NavigationLink3DExt::is_bidirectional() const { return pimpl->bidirectional; }

void NavigationLink3DExt::set_enter_cost(float p_cost) {
    pimpl->enter_cost = p_cost;
    pimpl->dirty = true;
    sync_link();
}
float NavigationLink3DExt::get_enter_cost() const { return pimpl->enter_cost; }

void NavigationLink3DExt::set_traversal_cost(float p_cost) {
    pimpl->traversal_cost = p_cost;
    pimpl->dirty = true;
    sync_link();
}
float NavigationLink3DExt::get_traversal_cost() const { return pimpl->traversal_cost; }

void NavigationLink3DExt::set_enabled(bool p_enabled) {
    pimpl->enabled = p_enabled;
    pimpl->dirty = true;
    sync_link();
}
bool NavigationLink3DExt::is_enabled() const { return pimpl->enabled; }

void NavigationLink3DExt::set_debug_visible(bool p_visible) {
    pimpl->debug_visible = p_visible;
    pimpl->dirty = true;
    sync_link();
}
bool NavigationLink3DExt::is_debug_visible() const { return pimpl->debug_visible; }

void NavigationLink3DExt::set_debug_color(const Color &p_color) {
    pimpl->debug_color = p_color;
    pimpl->dirty = true;
    sync_link();
}
Color NavigationLink3DExt::get_debug_color() const { return pimpl->debug_color; }

void NavigationLink3DExt::set_debug_emissive(const Color &p_color, float p_intensity) {
    pimpl->debug_emissive_color = p_color;
    pimpl->debug_emissive_intensity = p_intensity;
    pimpl->dirty = true;
    sync_link();
}
void NavigationLink3DExt::get_debug_emissive(Color &r_color, float &r_intensity) const {
    r_color = pimpl->debug_emissive_color;
    r_intensity = pimpl->debug_emissive_intensity;
}

void NavigationLink3DExt::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
}
int NavigationLink3DExt::get_gi_mode() const { return pimpl->gi_mode; }

void NavigationLink3DExt::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
}
float NavigationLink3DExt::get_gi_contribution() const { return pimpl->gi_contribution; }

void NavigationLink3DExt::sync_link() {
    pimpl->sync();
}