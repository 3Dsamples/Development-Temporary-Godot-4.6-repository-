// node_3d.cpp
#include "node_3d.h"
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace lighting {

Transform3D::Transform3D() { set_identity(); }

void Transform3D::set_identity() {
    for (int i = 0; i < 3; ++i) {
        origin[i] = 0.0;
        scale[i] = 1.0;
        for (int j = 0; j < 3; ++j)
            basis[i*3 + j] = (i == j) ? 1.0 : 0.0;
    }
}

void Transform3D::translate(double x, double y, double z) {
    origin[0] += x; origin[1] += y; origin[2] += z;
}

void Transform3D::rotate_x(double rad) {
    double c = cos(rad), s = sin(rad);
    double temp[3];
    for (int i = 0; i < 3; ++i) {
        temp[i] = basis[1*3 + i] * c - basis[2*3 + i] * s;
        basis[2*3 + i] = basis[1*3 + i] * s + basis[2*3 + i] * c;
        basis[1*3 + i] = temp[i];
    }
}

// Similar rotate_y, rotate_z omitted for brevity (full code in real impl).

Transform3D Transform3D::inverse() const {
    Transform3D inv;
    // Invert rotation part (orthogonal basis)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            inv.basis[i*3 + j] = basis[j*3 + i]; // transpose
    // Invert translation
    inv.origin[0] = -origin[0]; inv.origin[1] = -origin[1]; inv.origin[2] = -origin[2];
    return inv;
}

struct Node3D::Impl {
    Transform3D local_transform;
    Transform3D global_transform;
    bool transform_dirty = true;
    bool physics_interpolated = false;
    bool visible = true;
    char name[64];
    Node3D* parent = nullptr;
    std::vector<Node3D*> children;
    std::atomic<int> generation_counter = 0;
};

Node3D::Node3D() : pimpl(new Impl) { pimpl->name[0] = 0; }
Node3D::~Node3D() { delete pimpl; }

void Node3D::set_parent(Node3D* parent) {
    if (pimpl->parent) pimpl->parent->remove_child(this);
    pimpl->parent = parent;
    if (parent) parent->add_child(this);
    update_transform();
}

void Node3D::add_child(Node3D* child) { pimpl->children.push_back(child); }
void Node3D::remove_child(Node3D* child) {
    auto it = std::find(pimpl->children.begin(), pimpl->children.end(), child);
    if (it != pimpl->children.end()) pimpl->children.erase(it);
}

void Node3D::set_transform(const Transform3D& local) {
    pimpl->local_transform = local;
    update_transform();
}

const Transform3D& Node3D::get_transform() const { return pimpl->local_transform; }

Transform3D Node3D::get_global_transform() const {
    if (!pimpl->transform_dirty) return pimpl->global_transform;
    if (pimpl->parent) {
        pimpl->global_transform = pimpl->parent->get_global_transform();
        // Compose: global = parent_global * local (not shown fully)
    } else {
        pimpl->global_transform = pimpl->local_transform;
    }
    pimpl->transform_dirty = false;
    return pimpl->global_transform;
}

void Node3D::set_global_transform(const Transform3D& world) {
    if (pimpl->parent) {
        Transform3D parent_global = pimpl->parent->get_global_transform();
        pimpl->local_transform = parent_global.inverse() * world; // composition omitted
    } else {
        pimpl->local_transform = world;
    }
    update_transform();
}

void Node3D::update_transform() {
    pimpl->transform_dirty = true;
    for (Node3D* child : pimpl->children)
        child->update_transform();
    _transform_changed();
}

bool Node3D::is_transform_dirty() const { return pimpl->transform_dirty; }

void Node3D::set_name(const char* name) { strncpy(pimpl->name, name, 63); pimpl->name[63]=0; }
const char* Node3D::get_name() const { return pimpl->name; }
void Node3D::set_visible(bool visible) { pimpl->visible = visible; }
bool Node3D::is_visible() const { return pimpl->visible; }

void Node3D::set_physics_interpolated(bool enabled) { pimpl->physics_interpolated = enabled; }
void Node3D::synchronize_render_server(double delta) { /* stub – tie to render thread */ }
void Node3D::_transform_changed() {}

} // namespace lighting