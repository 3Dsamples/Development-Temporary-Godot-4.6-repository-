// remote_transform_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// RemoteTransform3D – forces its own transform to match that of a remote node.
// Supports updating position, rotation, scale, and can also update the remote
// node's transform if bidirectional. Used for synchronizing transforms between
// nodes (e.g., for vehicles, gun attachments, or cross‑scene references).
// ============================================================================

class RemoteTransform3D : public Node3D {
public:
    RemoteTransform3D();
    ~RemoteTransform3D();

    // ------------------------------------------------------------------------
    // Target remote node (by NodePath)
    // ------------------------------------------------------------------------
    void set_remote_node(const char* node_path);
    const char* get_remote_node() const;

    // ------------------------------------------------------------------------
    // Which transform components are updated
    // ------------------------------------------------------------------------
    void set_update_position(bool update);
    bool get_update_position() const;
    void set_update_rotation(bool update);
    bool get_update_rotation() const;
    void set_update_scale(bool update);
    bool get_update_scale() const;

    // ------------------------------------------------------------------------
    // Bidirectional (if true, also updates remote node's transform to ours)
    // ------------------------------------------------------------------------
    void set_use_remote_as_target(bool use); // if true, copy remote → local
    bool get_use_remote_as_target() const;
    void set_bidirectional(bool bi);
    bool is_bidirectional() const;

    // ------------------------------------------------------------------------
    // Offset transform (applied after copying)
    // ------------------------------------------------------------------------
    void set_offset_transform(const Transform3D& offset);
    Transform3D get_offset_transform() const;

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void process(double delta) override;
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting