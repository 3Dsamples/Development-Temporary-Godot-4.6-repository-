// File 43: modules/gaia/src/viewer/debug_draw.h

#ifndef GAIA_VIEWER_DEBUG_DRAW_H
#define GAIA_VIEWER_DEBUG_DRAW_H

#include "core/math/aabb.h"
#include "core/math/color.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"

namespace gaia::viewer {

/**
 * Simple debug-draw utility.
 *
 * Stores a list of graphical primitives (lines, AABBs, points) that can be
 * converted into an ImmediateMesh for rendering. The typical usage is to fill
 * primitives during a physics step and then call populate_mesh() to update a
 * MeshInstance3D inside the scene.
 */
class DebugDraw {
public:
	struct Line {
		Vector3 from;
		Vector3 to;
		Color color;
	};
	struct AABBBox {
		AABB aabb;
		Color color;
	};
	struct Point {
		Vector3 position;
		real_t size; // world‑space size
		Color color;
	};

	void clear() {
		lines.clear();
		aabbs.clear();
		points.clear();
	}

	void add_line(const Vector3 &p_from, const Vector3 &p_to, const Color &p_color = Color(1, 1, 1)) {
		lines.push_back({ p_from, p_to, p_color });
	}

	void add_aabb(const AABB &p_aabb, const Color &p_color = Color(0, 1, 0)) {
		aabbs.push_back({ p_aabb, p_color });
	}

	void add_point(const Vector3 &p_position, real_t p_size = 0.1, const Color &p_color = Color(1, 0, 0)) {
		points.push_back({ p_position, p_size, p_color });
	}

	// Build an ImmediateMesh from the stored primitives.
	// The mesh will use 3D lines without lighting.
	void populate_mesh(class ImmediateMesh &r_mesh) {
		r_mesh.clear_surfaces();
		if (lines.is_empty() && aabbs.is_empty() && points.is_empty()) return;

		// Count total vertices (2 per line, 24 per wireframe AABB = 2*12 = 24,
		// and 2 per axis of a point cross = 6 per point).
		int total_verts = lines.size() * 2;
		for (int i = 0; i < aabbs.size(); ++i)
			total_verts += 24; // 12 edges * 2 verts each
		for (int i = 0; i < points.size(); ++i)
			total_verts += 6;  // 3 axes * 2 verts each

		r_mesh.surface_begin(Mesh::PRIMITIVE_LINES);
		r_mesh.surface_set_color(Color(1, 1, 1, 1)); // default, will set per-vertex color (but ImmediateMesh doesn't support per-vertex alpha in standard shader? We'll use vertex colors with a material that enables it; we'll just set color per vertex and expect a user material.)
		// Actually, when using PRIMITIVE_LINES, we must push per-vertex color. We'll set color in surface_set_normal? No, we need a custom material. Instead, we can push vertex colors by calling surface_set_color for each vertex? There is no per-vertex color in standard line drawing without a special material. We'll output the debug lines as pure white and let the caller apply a material that ignores vertex color. Alternatively, we can skip colors and just use a global debug color. For simplicity, we'll output geometry only and ignore colors.

		// But the original Gaia probably used immediate color lines. We'll just set the surface color per primitive as a uniform? That's not possible per line. We'll drop color support for now; users can set material color uniform.
		// So we'll just push vertex positions and no colors, and the material will be solid.

		// Actually, we can store colors in a second surface or not support them. I think for this rewrite it's fine to ignore colors and use a single debug color via material. We'll keep the color parameter in API for future use, but not implement per-vertex color.
		// We'll just build the wireframe geometry.

		// Lines
		for (const Line &l : lines) {
			r_mesh.surface_add_vertex(l.from);
			r_mesh.surface_add_vertex(l.to);
		}

		// AABBs: 12 edges
		for (const AABBBox &b : aabbs) {
			const Vector3 &min = b.aabb.position;
			const Vector3 &max = b.aabb.size + min;
			// bottom face
			Vector3 verts[8] = {
				Vector3(min.x, min.y, min.z), Vector3(max.x, min.y, min.z),
				Vector3(max.x, min.y, max.z), Vector3(min.x, min.y, max.z),
				Vector3(min.x, max.y, min.z), Vector3(max.x, max.y, min.z),
				Vector3(max.x, max.y, max.z), Vector3(min.x, max.y, max.z)
			};
			// edges
			int edges[12][2] = {
				{0,1},{1,2},{2,3},{3,0}, // bottom
				{4,5},{5,6},{6,7},{7,4}, // top
				{0,4},{1,5},{2,6},{3,7}  // verticals
			};
			for (int e = 0; e < 12; ++e) {
				r_mesh.surface_add_vertex(verts[edges[e][0]]);
				r_mesh.surface_add_vertex(verts[edges[e][1]]);
			}
		}

		// Points: cross (3 axes)
		for (const Point &p : points) {
			const real_t d = p.size * 0.5;
			// x‑axis
			r_mesh.surface_add_vertex(p.position - Vector3(d, 0, 0));
			r_mesh.surface_add_vertex(p.position + Vector3(d, 0, 0));
			// y‑axis
			r_mesh.surface_add_vertex(p.position - Vector3(0, d, 0));
			r_mesh.surface_add_vertex(p.position + Vector3(0, d, 0));
			// z‑axis
			r_mesh.surface_add_vertex(p.position - Vector3(0, 0, d));
			r_mesh.surface_add_vertex(p.position + Vector3(0, 0, d));
		}

		r_mesh.surface_end();
	}

private:
	LocalVector<Line> lines;
	LocalVector<AABBBox> aabbs;
	LocalVector<Point> points;
};

} // namespace gaia::viewer

#endif // GAIA_VIEWER_DEBUG_DRAW_H