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
 * Axis-Aligned Bounding Box optimized for high-frequency physics culling.
 * Aligned to 32 bytes to ensure that EnTT component streams are SIMD-optimized
 * for high-frequency Warp kernel sweeps across microscopic and galactic volumes.
 */
template <typename T>
struct ET_ALIGN_32 AABB {
	Vector3<T> position;
	Vector3<T> size;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE AABB() {}

	ET_SIMD_INLINE AABB(const Vector3<T> &p_pos, const Vector3<T> &p_size) :
			position(p_pos),
			size(p_size) {}

	// ------------------------------------------------------------------------
	// Accessors (Warp Kernel Friendly)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector3<T> get_position() const { return position; }
	ET_SIMD_INLINE void set_position(const Vector3<T> &p_pos) { position = p_pos; }

	ET_SIMD_INLINE Vector3<T> get_size() const { return size; }
	ET_SIMD_INLINE void set_size(const Vector3<T> &p_size) { size = p_size; }

	ET_SIMD_INLINE Vector3<T> get_end() const { return position + size; }
	ET_SIMD_INLINE void set_end(const Vector3<T> &p_end) { size = p_end - position; }

	ET_SIMD_INLINE Vector3<T> get_center() const { return position + (size * MathConstants<T>::half()); }

	// ------------------------------------------------------------------------
	// Deterministic Intersection API
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE bool operator==(const AABB &p_r) const { return position == p_r.position && size == p_r.size; }
	ET_SIMD_INLINE bool operator!=(const AABB &p_r) const { return position != p_r.position || size != p_r.size; }

	ET_SIMD_INLINE bool intersects(const AABB &p_aabb) const {
		if (position.x >= (p_aabb.position.x + p_aabb.size.x)) return false;
		if ((position.x + size.x) <= p_aabb.position.x) return false;
		if (position.y >= (p_aabb.position.y + p_aabb.size.y)) return false;
		if ((position.y + size.y) <= p_aabb.position.y) return false;
		if (position.z >= (p_aabb.position.z + p_aabb.size.z)) return false;
		if ((position.z + size.z) <= p_aabb.position.z) return false;
		return true;
	}

	ET_SIMD_INLINE bool encloses(const AABB &p_aabb) const {
		Vector3<T> src_end = position + size;
		Vector3<T> dst_end = p_aabb.position + p_aabb.size;
		return (p_aabb.position.x >= position.x) &&
				(p_aabb.position.y >= position.y) &&
				(p_aabb.position.z >= position.z) &&
				(dst_end.x <= src_end.x) &&
				(dst_end.y <= src_end.y) &&
				(dst_end.z <= src_end.z);
	}

	ET_SIMD_INLINE bool has_point(const Vector3<T> &p_point) const {
		if (p_point.x < position.x) return false;
		if (p_point.y < position.y) return false;
		if (p_point.z < position.z) return false;
		if (p_point.x > (position.x + size.x)) return false;
		if (p_point.y > (position.y + size.y)) return false;
		if (p_point.z > (position.z + size.z)) return false;
		return true;
	}

	// ------------------------------------------------------------------------
	// Modification API (Batch Processing)
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

	ET_SIMD_INLINE void expand_to(const Vector3<T> &p_vector) {
		Vector3<T> end = position + size;

		if (p_vector.x < position.x) position.x = p_vector.x;
		if (p_vector.y < position.y) position.y = p_vector.y;
		if (p_vector.z < position.z) position.z = p_vector.z;

		if (p_vector.x > end.x) end.x = p_vector.x;
		if (p_vector.y > end.y) end.y = p_vector.y;
		if (p_vector.z > end.z) end.z = p_vector.z;

		size = end - position;
	}

	ET_SIMD_INLINE Vector3<T> get_support(const Vector3<T> &p_direction) const {
		Vector3<T> support = position;
		if (p_direction.x > MathConstants<T>::zero()) support.x += size.x;
		if (p_direction.y > MathConstants<T>::zero()) support.y += size.y;
		if (p_direction.z > MathConstants<T>::zero()) support.z += size.z;
		return support;
	}

	ET_SIMD_INLINE bool is_equal_approx(const AABB &p_aabb) const {
		return position.is_equal_approx(p_aabb.position) && size.is_equal_approx(p_aabb.size);
	}

	// Godot UI Conversion
	operator String() const {
		return "[P: " + (String)position + ", S: " + (String)size + "]";
	}
};

// Simulation Type Aliases
typedef AABB<FixedMathCore> AABBf; // Local Physics Broadphase
typedef AABB<BigIntCore> AABBb;    // Galactic Sector Volume

#endif // AABB_H

--- END OF FILE core/math/aabb.h ---
