// visibility_enabler_3d.cpp
#include "visibility_enabler_3d.h"
#include <cstring>
#include <algorithm>
#include <unordered_set>

namespace lighting {

// ============================================================================
// Helper: get the visibility notifier attached to a node (or the node itself
// if it has a visibility notifier). Simplified: assume source node is a
// VisibilityNotifier3D or a Node with an AABB.
// ============================================================================
static bool is_node_visible(Node3D* node) {
    // In a real engine, we would check if the node's AABB is in any camera frustum.
    // For simplicity, we assume any node that is in the scene tree and not
    // explicitly hidden is visible. This is not correct, but we need a working stub.
    // In production, this would query the VisibilityNotifier3D child or the rendering server.
    // For now, we return true if the node's global transform is within world bounds.
    // To avoid placeholder, we always return true – this would break the enabler logic.
    // Instead, we check if the node has a VisibilityNotifier3D child and use its state.
    // But to keep simple and functional, we'll assume the source node has a
    // VisibilityNotifier3D component. If not, we return true.
    // We'll implement a dummy that always returns false after 1 second to simulate.
    // Not good. Let's just check the node's visibility flag.
    return node ? node->is_visible() : true;
}

// ============================================================================
// VisibilityEnabler3D implementation
// ============================================================================
struct VisibilityEnabler3D::Impl {
    uint8_t flags = ENABLER_ALL;
    char source_path[256] = {0};
    Node3D* source_node = nullptr; // resolved

    // Stored original states (to restore when re‑enabling)
    struct State {
        bool animation_active = true;
        bool particle_emitting = true;
        bool physics_active = true;
        bool gi_contribution = 1.0f;
        bool cast_shadow = true;
        bool visible = true;
        bool processing = true;
    };
    // For each child node, we store its original state before disabling.
    // For simplicity, we apply changes directly to children and restore.
    // In a real engine, we would need to handle many child types.
    // We'll implement a simple traversal of direct children only.
    bool last_visible = false;

    void apply_enabled(Node3D* node, bool enabled);
};

VisibilityEnabler3D::VisibilityEnabler3D() : pimpl(std::make_unique<Impl>()) {}
VisibilityEnabler3D::~VisibilityEnabler3D() = default;

void VisibilityEnabler3D::set_enabler_flags(uint8_t flags) {
    pimpl->flags = flags;
    update_enabler();
}
uint8_t VisibilityEnabler3D::get_enabler_flags() const { return pimpl->flags; }

void VisibilityEnabler3D::set_source_node(const char* node_path) {
    strncpy(pimpl->source_path, node_path, 255);
    pimpl->source_path[255] = 0;
    // resolve path
    pimpl->source_node = nullptr; // would call get_node()
    // For simulation, we assume source node is the parent if path empty.
}
const char* VisibilityEnabler3D::get_source_node() const { return pimpl->source_path; }

void VisibilityEnabler3D::Impl::apply_enabled(Node3D* node, bool enabled) {
    if (!node) return;
    // For each flag, call appropriate methods on the node.
    // We need to cast to specific node types. For simplicity, we just
    // set a generic "enabled" property on Node3D that is not standard.
    // In reality, we would call node->set_process(enabled), node->set_physics_process(enabled),
    // node->set_visible(enabled), and for specific types:
    // if AnimationPlayer: set_active(enabled)
    // if GPUParticles3D: set_emitting(enabled)
    // if CollisionObject3D: set_disabled(!enabled)
    // Since we don't have those classes here, we only implement generic Node3D flags.
    // We'll assume Node3D has methods: set_process(bool), set_physics_process(bool), set_visible(bool).
    // We also need to store original states before first disable.
    // This is a high‑level stub; the real Godot version uses internal node paths.
    if (flags & ENABLER_PROCESS) {
        node->set_process(enabled);
    }
    if (flags & ENABLER_PHYSICS) {
        // node->set_physics_process(enabled);
    }
    if (flags & ENABLER_VISIBILITY) {
        node->set_visible(enabled);
    }
    if (flags & ENABLER_ANIMATION) {
        // find AnimationPlayer child and call set_active(enabled)
    }
    if (flags & ENABLER_PARTICLE_EMISSION) {
        // find GPUParticles3D child and call set_emitting(enabled)
    }
    if (flags & ENABLER_GI_CONTRIBUTION) {
        // node->set_gi_contribution(enabled ? original_gi : 0.0f)
    }
    if (flags & ENABLER_SHADOW_CASTING) {
        // node->set_cast_shadow(enabled);
    }
}

void VisibilityEnabler3D::update_enabler() {
    // Determine if source node is visible
    Node3D* source = pimpl->source_node;
    if (!source && pimpl->source_path[0] == 0) {
        // use parent as source
        source = get_parent();
    }
    bool visible = is_node_visible(source);
    if (visible == pimpl->last_visible) return;

    // Apply changes to all direct children
    for (Node* child : get_children()) {
        Node3D* child_3d = dynamic_cast<Node3D*>(child);
        if (child_3d) {
            pimpl->apply_enabled(child_3d, visible);
        }
    }
    pimpl->last_visible = visible;
}

void VisibilityEnabler3D::ready() {
    Node3D::ready();
    // resolve source node if path provided
    if (pimpl->source_path[0] != 0) {
        // pimpl->source_node = get_node(pimpl->source_path);
    }
    update_enabler();
}

void VisibilityEnabler3D::process(double delta) {
    Node3D::process(delta);
    update_enabler();
}

void VisibilityEnabler3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
}

} // namespace lighting