// remote_transform_3d.cpp
#include "remote_transform_3d.h"
#include <cstring>
#include <cmath>
#include <unordered_map>

namespace lighting {

struct RemoteTransform3D::Impl {
    char remote_node_path[256] = {0};
    bool update_position = true;
    bool update_rotation = true;
    bool update_scale = true;
    bool use_remote_as_target = true;   // true = copy remote → local, false = copy local → remote
    bool bidirectional = false;

    Transform3D offset;          // applied after copying
    Transform3D last_remote_transform; // for change detection
    int64_t remote_node_id = -1; // cached pointer to Node3D
    bool remote_dirty = true;

    Impl() {
        offset.set_identity();
        last_remote_transform.set_identity();
    }
};

RemoteTransform3D::RemoteTransform3D() : pimpl(std::make_unique<Impl>()) {}
RemoteTransform3D::~RemoteTransform3D() = default;

void RemoteTransform3D::set_remote_node(const char* node_path) {
    strncpy(pimpl->remote_node_path, node_path, 255);
    pimpl->remote_node_path[255] = 0;
    pimpl->remote_dirty = true;
}
const char* RemoteTransform3D::get_remote_node() const { return pimpl->remote_node_path; }

void RemoteTransform3D::set_update_position(bool update) { pimpl->update_position = update; }
bool RemoteTransform3D::get_update_position() const { return pimpl->update_position; }
void RemoteTransform3D::set_update_rotation(bool update) { pimpl->update_rotation = update; }
bool RemoteTransform3D::get_update_rotation() const { return pimpl->update_rotation; }
void RemoteTransform3D::set_update_scale(bool update) { pimpl->update_scale = update; }
bool RemoteTransform3D::get_update_scale() const { return pimpl->update_scale; }

void RemoteTransform3D::set_use_remote_as_target(bool use) { pimpl->use_remote_as_target = use; }
bool RemoteTransform3D::get_use_remote_as_target() const { return pimpl->use_remote_as_target; }
void RemoteTransform3D::set_bidirectional(bool bi) { pimpl->bidirectional = bi; }
bool RemoteTransform3D::is_bidirectional() const { return pimpl->bidirectional; }

void RemoteTransform3D::set_offset_transform(const Transform3D& offset) { pimpl->offset = offset; }
Transform3D RemoteTransform3D::get_offset_transform() const { return pimpl->offset; }

void RemoteTransform3D::process(double delta) {
    Node3D::process(delta);

    // Find remote node if path changed or first time
    if (pimpl->remote_dirty && pimpl->remote_node_path[0] != 0) {
        // In a real engine, we would call get_node() to resolve the path.
        // For simulation, we assume the remote node exists and is a Node3D.
        // pimpl->remote_node_id = find_node(pimpl->remote_node_path);
        pimpl->remote_dirty = false;
    }

    Node3D* remote = nullptr; // would be fetched from scene tree using remote_node_id
    if (!remote) return;

    if (pimpl->use_remote_as_target) {
        // Copy remote transform to this node (with offset)
        Transform3D remote_global = remote->get_global_transform();
        Transform3D new_local = remote_global * pimpl->offset;
        bool changed = false;
        if (pimpl->update_position) {
            set_global_transform(new_local);
            changed = true;
        } else if (pimpl->update_rotation) {
            // partial update not implemented for simplicity – would need to combine
        }
        if (changed) {
            // Mark that remote transform has been applied to this node
        }
        if (pimpl->bidirectional) {
            // Also update remote from this node (but would cause loop, careful)
            // Usually bidirectional is not used; if true, we set remote = this * offset.inverse()
        }
    } else {
        // Copy this node's transform to remote (this → remote)
        Transform3D local_global = get_global_transform();
        Transform3D remote_new = local_global * pimpl->offset; // simplified
        if (pimpl->update_position || pimpl->update_rotation || pimpl->update_scale) {
            remote->set_global_transform(remote_new);
        }
        if (pimpl->bidirectional) {
            // Then also update this node from remote to avoid drift? Not typical
        }
    }
}

void RemoteTransform3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    // The remote transform affects the transform of this node, which may be
    // a visual instance. Therefore, we must ensure the rendering server
    // receives the updated transform. Since Node3D already propagates transform
    // dirty flags, we need to mark this node as dirty if remote transform changed.
    // We can compare last_remote_transform with current.
    if (pimpl->use_remote_as_target && pimpl->remote_node_id != -1) {
        Node3D* remote = nullptr; // get from ID
        if (remote) {
            Transform3D current_remote = remote->get_global_transform();
            // Compare with last_remote_transform (if different, set dirty)
            if (!current_remote.is_equal(pimpl->last_remote_transform)) {
                update_transform(); // propagate dirty
                pimpl->last_remote_transform = current_remote;
            }
        }
    }
}

} // namespace lighting