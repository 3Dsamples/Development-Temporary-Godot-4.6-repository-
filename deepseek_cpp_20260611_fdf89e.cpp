// immediate_mesh_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// ImmediateMesh3D – dynamic mesh that can be drawn every frame.
// Useful for debug drawing, procedurally generated geometry, and particle trails.
// Supports triangles, lines, points; full lighting (shadows, GI, emissive);
// uses CPU‑side vertex buffer updates (not intended for thousands of objects).
// ============================================================================

enum class ImmediatePrimitive : uint8_t {
    TRIANGLES,
    LINES,
    POINTS,
    TRIANGLE_STRIP,
    LINE_STRIP
};

struct ImmediateVertex {
    double pos[3];
    float normal[3];
    float uv[2];
    float color[4]; // RGBA
};

class ImmediateMesh3D : public GeometryInstance3D {
public:
    ImmediateMesh3D();
    ~ImmediateMesh3D();

    // ------------------------------------------------------------------------
    // Begin / end immediate drawing (call once per frame)
    // ------------------------------------------------------------------------
    void begin(ImmediatePrimitive primitive, int material_id = -1);
    void vertex(const double* pos, const float* normal = nullptr,
                const float* uv = nullptr, const float* color = nullptr);
    void end();

    // ------------------------------------------------------------------------
    // Shorthand methods
    // ------------------------------------------------------------------------
    void set_color(float r, float g, float b, float a = 1.0f);
    void set_normal(const float* n);
    void set_uv(const float* uv);

    // ------------------------------------------------------------------------
    // Clear all draw commands (without rendering)
    // ------------------------------------------------------------------------
    void clear();

    // ------------------------------------------------------------------------
    // Lighting & GI flags (same as GeometryInstance3D)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Update mesh (call after end())
    // ------------------------------------------------------------------------
    void update_mesh();

    // ------------------------------------------------------------------------
    // Node overrides
    // ------------------------------------------------------------------------
    void synchronize_render_server(double delta) override;
    void process(double delta) override;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace lighting