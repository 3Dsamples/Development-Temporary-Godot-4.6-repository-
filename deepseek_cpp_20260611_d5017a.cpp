// visibility_enabler_3d.h
#pragma once

#include "node_3d.h"
#include <cstdint>
#include <memory>

namespace lighting {

// ============================================================================
// VisibilityEnabler3D – enables or disables child nodes when the parent
// (or another target) is visible or not. Useful for optimizing off‑screen
// nodes: pause animation, stop particles, disable collision, etc.
// ============================================================================

enum EnablerFlag : uint8_t {
    ENABLER_ANIMATION = 1 << 0,
    ENABLER_PARTICLE_EMISSION = 1 << 1,
    ENABLER_PHYSICS = 1 << 2,
    ENABLER_GI_CONTRIBUTION = 1 << 3,
    ENABLER_SHADOW_CASTING = 1 << 4,
    ENABLER_VISIBILITY = 1 << 5,
    ENABLER_PROCESS = 1 << 6,
    ENABLER_ALL = 0xFF
};

class VisibilityEnabler3D : public Node3D {
public:
    VisibilityEnabler3D();
    ~VisibilityEnabler3D();

    // ------------------------------------------------------------------------
    // Which features to enable/disable (bitmask)
    // ------------------------------------------------------------------------
    void set_enabler_flags(uint8_t flags);
    uint8_t get_enabler_flags() const;

    // ------------------------------------------------------------------------
    // Source of visibility (if nullptr, uses parent node)
    // ------------------------------------------------------------------------
    void set_source_node(const char* node_path);
    const char* get_source_node() const;

    // ------------------------------------------------------------------------
    // Force update (re‑evaluate visibility)
    // ------------------------------------------------------------------------
    void update_enabler();

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void ready() override;
    void process(double delta) override;
    void synchronize_render_server(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting