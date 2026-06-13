// immediate_mesh_3d.h
#pragma once

#include "geometry_instance_3d.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>

namespace lighting {

// ============================================================================
// ImmediateMesh3D – allows dynamic mesh generation each frame (or on demand).
// Ideal for debug drawing, procedural geometry, or any mesh that changes
// every frame. Supports vertices, indices, normals, UVs, colors, and tangents.
// Integrated with lighting: casts shadows, receives GI, emissive per vertex.
// ============================================================================

class ImmediateMesh3D : public GeometryInstance3D {
public:
    ImmediateMesh3D();
    ~ImmediateMesh3D();

    // ------------------------------------------------------------------------
    // Begin / End drawing (called from process or draw callback)
    // ------------------------------------------------------------------------
    void begin_mesh(int primitive_type); // 0=points,1=lines,2=triangles,3=triangle strip
    void end_mesh();

    // ------------------------------------------------------------------------
    // Vertex data submission (must be called between begin/end)
    // ------------------------------------------------------------------------
    void set_vertex(double x, double y, double z);
    void set_normal(float nx, float ny, float nz);
    void set_color(float r, float g, float b, float a = 1.0f);
    void set_uv(float u, float v);
    void set_tangent(float tx, float ty, float tz, float tw = 1.0f);
    void add_vertex(); // commits current vertex (with previously set attributes)

    // ------------------------------------------------------------------------
    // Convenience: add vertex with all attributes at once
    // ------------------------------------------------------------------------
    void add_vertex_full(double x, double y, double z,
                         float nx, float ny, float nz,
                         float r, float g, float b, float a = 1.0f,
                         float u = 0.0f, float v = 0.0f);

    // ------------------------------------------------------------------------
    // Indexed drawing (optional)
    // ------------------------------------------------------------------------
    void set_index(int index); // for indexed primitives (add after vertices)
    void add_index(int index); // push index

    // ------------------------------------------------------------------------
    // Clear mesh
    // ------------------------------------------------------------------------
    void clear();

    // ------------------------------------------------------------------------
    // Automatic rebuild every frame (if true, calls begin/end automatically)
    // ------------------------------------------------------------------------
    void set_auto_redraw(bool enable);
    bool is_auto_redraw() const;
    void set_draw_callback(std::function<void(ImmediateMesh3D*)> callback);

    // ------------------------------------------------------------------------
    // Lighting & GI (each vertex can be emissive, but mesh can override)
    // ------------------------------------------------------------------------
    void set_cast_shadow(bool cast) override;
    void set_receive_shadow(bool receive) override;
    void set_gi_mode(int mode) override;
    void set_gi_contribution(float amount) override;
    void set_emissive(const float* color, float intensity) override;

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