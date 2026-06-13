// File 231: modules/newton/src/collision/newton_heightfield_collision.cpp
// Heightfield collision shape implementation: support point, inertia, AABB,
// and binding methods for Godot editor integration.

#include "newton_heightfield_collision.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/aabb.h"
#include "core/math/basis.h"
#include "core/typedefs.h"

namespace newton {

// Support point: iterate over all grid cells and test their four corners.
vec3 NewtonHeightfieldCollision::get_support(const vec3 &p_dir, const mat4 &p_transform) const {
	vec3 local_dir = p_transform.basis.xform_inv(p_dir);
	// Origin of the heightfield in local space is at (0,0,0) in the centre of the base.
	vec3 offset = vec3(-width * cell_size * 0.5f, 0.0f, -depth * cell_size * 0.5f);
	real_t best_dot = -INFINITY;
	vec3 best_vertex(0.0f, 0.0f, 0.0f);
	// Walk every cell.
	for (int z = 0; z < depth; ++z) {
		for (int x = 0; x < width; ++x) {
			// Heights of the four corners (clamp to neighbour if at border).
			real_t h00 = heights[z * width + x];
			real_t h10 = (x + 1 < width) ? heights[z * width + (x + 1)] : h00;
			real_t h01 = (z + 1 < depth) ? heights[(z + 1) * width + x] : h00;
			real_t h11 = (x + 1 < width && z + 1 < depth) ? heights[(z + 1) * width + (x + 1)] : h00;

			real_t lx = offset.x + x * cell_size;
			real_t lz = offset.z + z * cell_size;

			vec3 corners[4] = {
				vec3(lx,              h00, lz),
				vec3(lx + cell_size,  h10, lz),
				vec3(lx,              h01, lz + cell_size),
				vec3(lx + cell_size,  h11, lz + cell_size)
			};

			for (int c = 0; c < 4; ++c) {
				real_t d = corners[c].dot(local_dir);
				if (d > best_dot) {
					best_dot = d;
					best_vertex = corners[c];
				}
			}
		}
	}
	return p_transform.xform(best_vertex);
}

// Inertia tensor approximated by a bounding box of the heightfield.
mat3 NewtonHeightfieldCollision::compute_inertia(real_t p_mass) const {
	aabb box = get_local_aabb();
	vec3 size = box.size;
	real_t x = size.x;
	real_t y = size.y;
	real_t z = size.z;
	real_t Ix = (1.0f / 12.0f) * p_mass * (y * y + z * z);
	real_t Iy = (1.0f / 12.0f) * p_mass * (x * x + z * z);
	real_t Iz = (1.0f / 12.0f) * p_mass * (x * x + y * y);
	return mat3().scaled(vec3(Ix, Iy, Iz));
}

// Local AABB from the stored min/max heights and grid extent.
aabb NewtonHeightfieldCollision::get_local_aabb() const {
	vec3 half_extents(width * cell_size * 0.5f, (max_height - min_height) * 0.5f, depth * cell_size * 0.5f);
	vec3 center(0.0f, (min_height + max_height) * 0.5f, 0.0f);
	return aabb(center - half_extents, half_extents * 2.0f);
}

// Bind methods for Godot editor.
void NewtonHeightfieldCollision::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_grid", "width", "depth", "cell_size"), &NewtonHeightfieldCollision::set_grid, DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("set_height", "x", "z", "height"), &NewtonHeightfieldCollision::set_height);
	ClassDB::bind_method(D_METHOD("get_height", "x", "z"), &NewtonHeightfieldCollision::get_height);
	ClassDB::bind_method(D_METHOD("get_width"), &NewtonHeightfieldCollision::get_width);
	ClassDB::bind_method(D_METHOD("get_depth"), &NewtonHeightfieldCollision::get_depth);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &NewtonHeightfieldCollision::get_cell_size);
	ClassDB::bind_method(D_METHOD("get_min_height"), &NewtonHeightfieldCollision::get_min_height);
	ClassDB::bind_method(D_METHOD("get_max_height"), &NewtonHeightfieldCollision::get_max_height);
}

} // namespace newton