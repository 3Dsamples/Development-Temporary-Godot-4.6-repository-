// immediate_mesh_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lighting {

// ============================================================================
// ImmediateMesh3D – a mesh that can be updated every frame with custom geometry.
// Designed for dynamic debug drawing, procedural geometry, and simple effects.
// Supports vertex colors, normals, UVs, and lighting (shadows, GI, emissive).
// Optimized for frequent updates with double‑buffered vertex/index buffers.
// ============================================================================

enum class ImmediatePrimitiveType : uint8_t {
    POINTS,
    LINES,
    LINE_STRIP,
    TRIANGLES,
    TRIANGLE_STRIP
};

class ImmediateMesh3D : public GeometryInstance3D {
public:
    ImmediateMesh3D();
    ~ImmediateMesh3D();

    // ------------------------------------------------------------------------
    // Begin / End (immediate mode API)
    // ------------------------------------------------------------------------
    void surface_begin(ImmediatePrimitiveType primitive);
    void surface_set_vertex(double x, double y, double z);
    void surface_set_normal(double x, double y, double z);
    void surface_set_color(float r, float g, float b, float a = 1.0f);
    void surface_set_uv(float u, float v);
    void surface_add_vertex();      // adds vertex with current attributes
    void surface_end();             // commits the mesh to rendering

    // ------------------------------------------------------------------------
    // Clear all geometry
    // ------------------------------------------------------------------------
    void clear();

    // ------------------------------------------------------------------------
    // Lighting & GI flags (per‑instance)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;         // 0=off,1=static,2=dynamic
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity);
    void get_emissive(float* out_color, float& out_intensity) const;

    // ------------------------------------------------------------------------
    // Force an update (call after modifying vertices manually)
    // ------------------------------------------------------------------------
    void update_mesh();

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