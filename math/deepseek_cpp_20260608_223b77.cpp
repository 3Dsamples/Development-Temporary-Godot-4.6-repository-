// File 212: modules/newton/src/collision/newton_heightfield_collision.h
// Heightfield collision shape – a grid of heights used for terrain.
// Support point finds the local vertex that maximizes dot(direction, vertex).
// Inertia is approximated as a box bounding the heightfield extent.

#ifndef NEWTON_COLLISION_HEIGHTFIELD_H
#define NEWTON_COLLISION_HEIGHTFIELD_H

#include "newton_collision.h"
#include "core/templates/local_vector.h"
#include "../core/newton_types.h"
#include "../core/newton_constants.h"

namespace newton {

class NewtonHeightfieldCollision : public NewtonCollision {
	GDCLASS(NewtonHeightfieldCollision, NewtonCollision);

public:
	NewtonHeightfieldCollision() : width(1), depth(1), cell_size(1.0), min_height(0.0), max_height(0.0) {}

	// Set grid dimensions and allocate the height array.
	void set_grid(int p_width, int p_depth, real_t p_cell_size = 1.0) {
		width = MAX(p_width, 2);
		depth = MAX(p_depth, 2);
		cell_size = MAX(p_cell_size, 1e-6);
		heights.resize(width * depth);
		for (int i = 0; i < heights.size(); ++i) heights[i] = 0.0;
		min_height = 0.0;
		max_height = 0.0;
	}

	// Access a height value.
	void set_height(int x, int z, real_t p_height) {
		ERR_FAIL_INDEX(x, width);
		ERR_FAIL_INDEX(z, depth);
		heights[z * width + x] = p_height;
		// Update bounds.
		if (p_height < min_height) min_height = p_height;
		if (p_height > max_height) max_height = p_height;
	}
	real_t get_height(int x, int z) const {
		ERR_FAIL_INDEX_V(x, width, 0.0);
		ERR_FAIL_INDEX_V(z, depth, 0.0);
		return heights[z * width + x];
	}

	// Derived information.
	int get_width() const { return width; }
	int get_depth() const { return depth; }
	real_t get_cell_size() const { return cell_size; }
	real_t get_min_height() const { return min_height; }
	real_t get_max_height() const { return max_height; }

	virtual ShapeType get_shape_type() const override { return ShapeType::HEIGHTFIELD; }

	// Support point: brute‑force search over all heightfield vertices that
	// are inside the direction's projection.  For each cell, we consider the
	// four corners (or just the minimum / maximum of the cell) but we use
	// the actual heights of the four corners and choose the one that gives
	// the largest dot product.
	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir);
		// Local origin of the heightfield is at (0,0,0) in the centre of the base plane.
		vec3 offset = vec3(-width * cell_size * 0.5, 0.0, -depth * cell_size * 0.5);
		real_t best_dot = -INFINITY;
		vec3 best_vertex(0, 0, 0);
		// Iterate over every cell and test its four corners.
		for (int z = 0; z < depth; ++z) {
			for (int x = 0; x < width; ++x) {
				// Cell corners in local space.
				real_t h00 = heights[z * width + x];
				real_t h10 = (x + 1 < width) ? heights[z * width + (x + 1)] : h00;
				real_t h01 = (z + 1 < depth) ? heights[(z + 1) * width + x] : h00;
				real_t h11 = (x + 1 < width && z + 1 < depth) ? heights[(z + 1) * width + (x + 1)] : h00;
				real_t lx = offset.x + x * cell_size;
				real_t lz = offset.z + z * cell_size;
				vec3 corners[4] = {
					vec3(lx,           h00, lz),
					vec3(lx + cell_size, h10, lz),
					vec3(lx,           h01, lz + cell_size),
					vec3(lx + cell_size, h11, lz + cell_size)
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

	// Inertia: approximate as a box with extents derived from the bounding AABB.
	virtual mat3 compute_inertia(real_t p_mass) const override {
		aabb box = get_local_aabb();
		vec3 size = box.size;
		real_t x = size.x, y = size.y, z = size.z;
		real_t Ix = (1.0 / 12.0) * p_mass * (y * y + z * z);
		real_t Iy = (1.0 / 12.0) * p_mass * (x * x + z * z);
		real_t Iz = (1.0 / 12.0) * p_mass * (x * x + y * y);
		return mat3().scaled(vec3(Ix, Iy, Iz));
	}

	// Local AABB from min/max heights and grid extents.
	virtual aabb get_local_aabb() const override {
		vec3 half_extents(width * cell_size * 0.5, (max_height - min_height) * 0.5, depth * cell_size * 0.5);
		vec3 center(0.0, (min_height + max_height) * 0.5, 0.0);
		return aabb(center - half_extents, half_extents * 2.0);
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_grid", "width", "depth", "cell_size"), &NewtonHeightfieldCollision::set_grid, DEFVAL(1.0));
		ClassDB::bind_method(D_METHOD("set_height", "x", "z", "height"), &NewtonHeightfieldCollision::set_height);
		ClassDB::bind_method(D_METHOD("get_height", "x", "z"), &NewtonHeightfieldCollision::get_height);
		ClassDB::bind_method(D_METHOD("get_width"), &NewtonHeightfieldCollision::get_width);
		ClassDB::bind_method(D_METHOD("get_depth"), &NewtonHeightfieldCollision::get_depth);
		ClassDB::bind_method(D_METHOD("get_cell_size"), &NewtonHeightfieldCollision::get_cell_size);
		ClassDB::bind_method(D_METHOD("get_min_height"), &NewtonHeightfieldCollision::get_min_height);
		ClassDB::bind_method(D_METHOD("get_max_height"), &NewtonHeightfieldCollision::get_max_height);
		ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "grid_dimensions"), "set_grid", "");
	}

private:
	int width;
	int depth;
	real_t cell_size;
	LocalVector<real_t> heights;
	real_t min_height;
	real_t max_height;
};

} // namespace newton

#endif // NEWTON_COLLISION_HEIGHTFIELD_H