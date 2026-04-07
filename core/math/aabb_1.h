--- START OF FILE core/math/aabb.h ---

#ifndef AABB_H
#define AABB_H

#include "core/math/vector3.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * AABB Template
 * 
 * 32-byte aligned 3D bounding box for deterministic spatial partitioning.
 * Engineered for EnTT SoA streams and parallel Warp kernel sweeps.
 */
template <typename T>
struct ET_ALIGN_32 AABB {
	Vector3<T> position;
	Vector3<T> size;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ AABB() {}

	_FORCE_INLINE_ AABB(const Vector3<T> &p_pos, const Vector3<T> &p_size) :
			position(p_pos),
			size(p_size) {}

	// ------------------------------------------------------------------------
	// Accessors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector3<T> get_position() const { return position; }
	_FORCE_INLINE_ void set_position(const Vector3<T> &p_pos) { position = p_pos; }

	_FORCE_INLINE_ Vector3<T> get_size() const { return size; }
	_FORCE_INLINE_ void set_size(const Vector3<T> &p_size) { size = p_size; }

	_FORCE_INLINE_ Vector3<T> get_end() const { return position + size; }
	_FORCE_INLINE_ void set_end(const Vector3<T> &p_end) { size = p_end - position; }

	_FORCE_INLINE_ Vector3<T> get_center() const { return position + (size * MathConstants<T>::half()); }

	// ------------------------------------------------------------------------
	// Deterministic Intersection API
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ bool operator==(const AABB &p_r) const { return position == p_r.position && size == p_r.size; }
	_FORCE_INLINE_ bool operator!=(const AABB &p_r) const { return position != p_r.position || size != p_r.size; }

	_FORCE_INLINE_ bool intersects(const AABB &p_aabb) const {
		if (position.x >= (p_aabb.position.x + p_aabb.size.x)) return false;
		if ((position.x + size.x) <= p_aabb.position.x) return false;
		if (position.y >= (p_aabb.position.y + p_aabb.size.y)) return false;
		if ((position.y + size.y) <= p_aabb.position.y) return false;
		if (position.z >= (p_aabb.position.z + p_aabb.size.z)) return false;
		if ((position.z + size.z) <= p_aabb.position.z) return false;
		return true;
	}

	_FORCE_INLINE_ bool encloses(const AABB &p_aabb) const {
		Vector3<T> src_end = position + size;
		Vector3<T> dst_end = p_aabb.position + p_aabb.size;
		return (p_aabb.position.x >= position.x) &&
				(p_aabb.position.y >= position.y) &&
				(p_aabb.position.z >= position.z) &&
				(dst_end.x <= src_end.x) &&
				(dst_end.y <= src_end.y) &&
				(dst_end.z <= src_end.z);
	}

	_FORCE_INLINE_ bool has_point(const Vector3<T> &p_point) const {
		if (p_point.x < position.x) return false;
		if (p_point.y < position.y) return false;
		if (p_point.z < position.z) return false;
		if (p_point.x > (position.x + size.x)) return false;
		if (p_point.y > (position.y + size.y)) return false;
		if (p_point.z > (position.z + size.z)) return false;
		return true;
	}

	// ------------------------------------------------------------------------
	// Volumetric & Physics Interaction
	// ------------------------------------------------------------------------

	/**
	 * get_volume()
	 * Essential for "Balloon Effect" and "Flesh" volume preservation kernels.
	 */
	_FORCE_INLINE_ T get_volume() const {
		return size.x * size.y * size.z;
	}

	/**
	 * get_support()
	 * Returns the vertex furthest in a given direction for GJK Narrow-phase checks.
	 */
	_FORCE_INLINE_ Vector3<T> get_support(const Vector3<T> &p_direction) const {
		Vector3<T> support = position;
		if (p_direction.x > MathConstants<T>::zero()) support.x += size.x;
		if (p_direction.y > MathConstants<T>::zero()) support.y += size.y;
		if (p_direction.z > MathConstants<T>::zero()) support.z += size.z;
		return support;
	}

	/**
	 * intersection()
	 * Returns the bit-perfect overlapping volume AABB.
	 */
	AABB intersection(const AABB &p_aabb) const {
		Vector3<T> src_min = position;
		Vector3<T> src_max = position + size;
		Vector3<T> dst_min = p_aabb.position;
		Vector3<T> dst_max = p_aabb.position + p_aabb.size;

		Vector3<T> min_v, max_v;
		min_v.x = MAX(src_min.x, dst_min.x);
		min_v.y = MAX(src_min.y, dst_min.y);
		min_v.z = MAX(src_min.z, dst_min.z);
		max_v.x = MIN(src_max.x, dst_max.x);
		max_v.y = MIN(src_max.y, dst_max.y);
		max_v.z = MIN(src_max.z, dst_max.z);

		if (min_v.x >= max_v.x || min_v.y >= max_v.y || min_v.z >= max_v.z) {
			return AABB();
		}
		return AABB(min_v, max_v - min_v);
	}

	// ------------------------------------------------------------------------
	// Modification API (Batch Oriented)
	// ------------------------------------------------------------------------
	void merge_with(const AABB &p_aabb) {
		Vector3<T> min_v, max_v;
		min_v.x = MIN(position.x, p_aabb.position.x);
		min_v.y = MIN(position.y, p_aabb.position.y);
		min_v.z = MIN(position.z, p_aabb.position.z);
		max_v.x = MAX(position.x + size.x, p_aabb.position.x + p_aabb.size.x);
		max_v.y = MAX(position.y + size.y, p_aabb.position.y + p_aabb.size.y);
		max_v.z = MAX(position.z + size.z, p_aabb.position.z + p_aabb.size.z);
		position = min_v;
		size = max_v - min_v;
	}

	_FORCE_INLINE_ void expand_to(const Vector3<T> &p_vector) {
		Vector3<T> end = position + size;
		if (p_vector.x < position.x) position.x = p_vector.x;
		if (p_vector.y < position.y) position.y = p_vector.y;
		if (p_vector.z < position.z) position.z = p_vector.z;
		if (p_vector.x > end.x) end.x = p_vector.x;
		if (p_vector.y > end.y) end.y = p_vector.y;
		if (p_vector.z > end.z) end.z = p_vector.z;
		size = end - position;
	}

	_FORCE_INLINE_ bool is_equal_approx(const AABB &p_aabb) const {
		return position.is_equal_approx(p_aabb.position) && size.is_equal_approx(p_aabb.size);
	}

	operator String() const {
		return "[P: " + (String)position + ", S: " + (String)size + "]";
	}
};

// Simulation Tier Typedefs
typedef AABB<FixedMathCore> AABBf; // Bit-perfect Broadphase / Triggers
typedef AABB<BigIntCore> AABBb;    // Discrete Macro-Volume Sectoring

#endif // AABB_H

--- END OF FILE core/math/aabb.h ---
