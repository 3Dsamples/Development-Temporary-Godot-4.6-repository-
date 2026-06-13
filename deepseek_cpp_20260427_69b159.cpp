// File 276: modules/vienna/src/collision/vienna_heightfield.h
// Vienna Heightfield collision shape – a grid of heights for terrain.
// Support point finds the vertex that maximizes dot(direction, vertex).
// Inertia is approximated as a box bounding the heightfield extent.

#ifndef VIENNA_COLLISION_HEIGHTFIELD_H
#define VIENNA_COLLISION_HEIGHTFIELD_H

#include "vienna_shape.h"
#include "core/templates/local_vector.h"

namespace vienna {

class ViennaHeightfield : public ViennaShape {
	GDCLASS(ViennaHeightfield, ViennaShape);

public:
	ViennaHeightfield() : width(1), depth(1), cell_size(1.0), min_height(0.0), max_height(0.0) {}

	void set_grid(int p_width, int p_depth, real_t p_cell_size = 1.0) {
		width = MAX(p_width, 2);
		depth = MAX(p_depth, 2);
		cell_size = MAX(p_cell_size, 1e-6);
		heights.resize(width * depth);
		for (int i = 0; i < heights.size(); ++i) heights[i] = 0.0;
		min_height = 0.0;
		max_height = 0.0;
	}

	void set_height(int x, int z, real_t p_height) {
		ERR_FAIL_INDEX(x, width);
		ERR_FAIL_INDEX(z, depth);
		heights[z * width + x] = p_height;
		if (p_height < min_height) min_height = p_height;
		if (p_height > max_height) max_height = p_height;
	}
	real_t get_height(int x, int z) const {
		ERR_FAIL_INDEX_V(x, width, 0.0);
		ERR_FAIL_INDEX_V(z, depth, 0.0);
		return heights[z * width + x];
	}

	int get_width() const { return width; }
	int get_depth() const { return depth; }
	real_t get_cell_size() const { return cell_size; }
	real_t get_min_height() const { return min_height; }
	real_t get_max_height() const { return max_height; }

	virtual ShapeType get_shape_type() const override { return ShapeType::HEIGHTFIELD; }

	virtual vec3 get_support(const vec3 &p_dir, const mat4 &p_transform) const override {
		vec3 local_dir = p_transform.basis.xform_inv(p_dir);
		vec3 offset = vec3(-width * cell_size * 0.5, 0.0, -depth * cell_size * 0.5);
		real_t best_dot = -INFINITY;
		vec3 best_vertex(0, 0, 0);
		for (int z = 0; z < depth; ++z) {
			for (int x = 0; x < width; ++x) {
				real_t h00 = heights[z * width + x];
				real_t h10 = (x + 1 < width) ? heights[z * width + (x + 1)] : h00;
				real_t h01 = (z + 1 < depth) ? heights[(z + 1) * width + x] : h00;
				real_t h11 = (x + 1 < width && z + 1 < depth) ? heights[(z + 1) * width + (x + 1)] : h00;
				real_t lx = offset.x + x * cell_size;
				real_t lz = offset.z + z * cell_size;
				vec3 corners[4] = {
					vec3(lx, h00, lz),
					vec3(lx + cell_size, h10, lz),
					vec3(lx, h01, lz + cell_size),
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

	virtual mat3 compute_inertia(real_t p_mass) const override {
		aabb box = get_local_aabb();
		vec3 size = box.size;
		real_t x = size.x, y = size.y, z = size.z;
		real_t Ix = (1.0 / 12.0) * p_mass * (y * y + z * z);
		real_t Iy = (1.0 / 12.0) * p_mass * (x * x + z * z);
		real_t Iz = (1.0 / 12.0) * p_mass * (x * x + y * y);
		return mat3().scaled(vec3(Ix, Iy, Iz));
	}

	virtual aabb get_local_aabb() const override {
		vec3 half_extents(width * cell_size * 0.5f, (max_height - min_height) * 0.5f, depth * cell_size * 0.5f);
		vec3 center(0.0f, (min_height + max_height) * 0.5f, 0.0f);
		return aabb(center - half_extents, half_extents * 2.0f);
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_grid", "width", "depth", "cell_size"), &ViennaHeightfield::set_grid, DEFVAL(1.0));
		ClassDB::bind_method(D_METHOD("set_height", "x", "z", "height"), &ViennaHeightfield::set_height);
		ClassDB::bind_method(D_METHOD("get_height", "x", "z"), &ViennaHeightfield::get_height);
		ClassDB::bind_method(D_METHOD("get_width"), &ViennaHeightfield::get_width);
		ClassDB::bind_method(D_METHOD("get_depth"), &ViennaHeightfield::get_depth);
		ClassDB::bind_method(D_METHOD("get_cell_size"), &ViennaHeightfield::get_cell_size);
		ClassDB::bind_method(D_METHOD("get_min_height"), &ViennaHeightfield::get_min_height);
		ClassDB::bind_method(D_METHOD("get_max_height"), &ViennaHeightfield::get_max_height);
	}

private:
	int width;
	int depth;
	real_t cell_size;
	LocalVector<real_t> heights;
	real_t min_height;
	real_t max_height;
};

} // namespace vienna

#endif // VIENNA_COLLISION_HEIGHTFIELD_H