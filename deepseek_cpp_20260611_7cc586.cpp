// Name : lighting enhancement
// File : scene/3d/node_3d.cpp 2 of xxx
// Description : Implementation of base 3D node with transform hierarchy, dirty propagation,
//               and physics interpolation. Delegates render updates to children.

#include "scene/3d/node_3d.h"
#include "core/os/os.h"
#include "servers/rendering_server.h"
#include <cmath>
#include <cstring>
#include <algorithm>

struct Node3D::Impl {
    String name;
    Transform3D local_transform;
    Transform3D global_transform;
    bool transform_dirty = true;
    bool physics_interpolated = false;
    bool visible = true;
    bool cast_shadow = true;
    int gi_mode = 1; // static by default
    float gi_contribution = 1.0f;
    Color emissive_color;
    float emissive_intensity = 0.0f;

    Node3D *parent = nullptr;
    List<Node3D *> children;

    // For render server sync (if this node has an instance, but Node3D doesn't)
    // VisualInstance3D will override.
};

Node3D::Node3D() {
    pimpl = new Impl;
    pimpl->local_transform.set_identity();
    pimpl->global_transform.set_identity();
}

Node3D::~Node3D() {
    // Detach children
    while (pimpl->children.size()) {
        Node3D *child = pimpl->children.front()->get();
        remove_child(child);
        child->set_parent(nullptr);
    }
    delete pimpl;
}

void Node3D::set_parent(Node3D *p_parent) {
    if (pimpl->parent == p_parent) return;
    if (pimpl->parent) pimpl->parent->remove_child(this);
    pimpl->parent = p_parent;
    if (p_parent) p_parent->add_child(this);
    update_transform();
}

Node3D *Node3D::get_parent() const {
    return pimpl->parent;
}

void Node3D::add_child(Node3D *p_child) {
    ERR_FAIL_COND(!p_child);
    ERR_FAIL_COND(p_child->get_parent() == this);
    pimpl->children.push_back(p_child);
    p_child->set_parent(this);
}

void Node3D::remove_child(Node3D *p_child) {
    ERR_FAIL_COND(!p_child);
    pimpl->children.erase(p_child);
    if (p_child->get_parent() == this) p_child->set_parent(nullptr);
}

void Node3D::get_children(List<Node3D *> *r_children) const {
    for (List<Node3D *>::Element *E = pimpl->children.front(); E; E = E->next()) {
        r_children->push_back(E->get());
    }
}

void Node3D::set_transform(const Transform3D &p_transform) {
    pimpl->local_transform = p_transform;
    update_transform();
}

const Transform3D &Node3D::get_transform() const {
    return pimpl->local_transform;
}

Transform3D Node3D::get_global_transform() const {
    if (!pimpl->transform_dirty) return pimpl->global_transform;
    if (pimpl->parent) {
        Transform3D parent_global = pimpl->parent->get_global_transform();
        pimpl->global_transform = parent_global * pimpl->local_transform;
    } else {
        pimpl->global_transform = pimpl->local_transform;
    }
    pimpl->transform_dirty = false;
    return pimpl->global_transform;
}

void Node3D::set_global_transform(const Transform3D &p_global) {
    if (pimpl->parent) {
        Transform3D parent_global = pimpl->parent->get_global_transform();
        pimpl->local_transform = parent_global.affine_inverse() * p_global;
    } else {
        pimpl->local_transform = p_global;
    }
    update_transform();
}

void Node3D::update_transform() {
    pimpl->transform_dirty = true;
    // Propagate dirty to children
    for (List<Node3D *>::Element *E = pimpl->children.front(); E; E = E->next()) {
        Node3D *child = E->get();
        child->update_transform();
    }
    _transform_changed();
    // Notify any attached render instances (if this node is a visual instance)
    _update_render_server_transform();
}

bool Node3D::is_transform_dirty() const {
    return pimpl->transform_dirty;
}

void Node3D::set_name(const String &p_name) {
    pimpl->name = p_name;
}

String Node3D::get_name() const {
    return pimpl->name;
}

void Node3D::set_visible(bool p_visible) {
    pimpl->visible = p_visible;
    // For visual instances, they would override to set instance visibility.
}

bool Node3D::is_visible() const {
    return pimpl->visible;
}

void Node3D::set_physics_interpolated(bool p_enabled) {
    pimpl->physics_interpolated = p_enabled;
}

bool Node3D::is_physics_interpolated() const {
    return pimpl->physics_interpolated;
}

void Node3D::synchronize_render_server(double p_delta) {
    // Base node: nothing to do, but we propagate to children? Usually not.
    // VisualInstance3D will handle its own sync.
}

void Node3D::set_cast_shadow(bool p_cast) {
    pimpl->cast_shadow = p_cast;
}

bool Node3D::get_cast_shadow() const {
    return pimpl->cast_shadow;
}

void Node3D::set_gi_mode(int p_mode) {
    pimpl->gi_mode = p_mode;
}

int Node3D::get_gi_mode() const {
    return pimpl->gi_mode;
}

void Node3D::set_gi_contribution(float p_amount) {
    pimpl->gi_contribution = p_amount;
}

float Node3D::get_gi_contribution() const {
    return pimpl->gi_contribution;
}

void Node3D::set_emissive(const Color &p_color, float p_intensity) {
    pimpl->emissive_color = p_color;
    pimpl->emissive_intensity = p_intensity;
}

Color Node3D::get_emissive() const {
    return pimpl->emissive_color;
}

float Node3D::get_emissive_intensity() const {
    return pimpl->emissive_intensity;
}

void Node3D::_transform_changed() {
    // To be overridden by derived classes.
}

void Node3D::_update_render_server_transform() {
    // Base node does not have a render instance.
}

void Node3D::_notify_transform_dirty() {
    // Called by children when they need to update dirty flag.
}

// Override for class registration (if using Godot's ClassDB, but not needed for standalone)
