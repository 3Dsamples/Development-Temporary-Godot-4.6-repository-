--- START OF FILE core/math/rect2.h ---

#ifndef RECT2_H
#define RECT2_H

#include "core/math/vector2.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Rect2 Template
 * 
 * 2D Axis-Aligned Bounding Box for the Universal Solver.
 * Aligned to 32 bytes to ensure that EnTT component streams are SIMD-optimized
 * for high-frequency Warp kernel sweeps across 2D simulation layers.
 */
template <typename T>
struct ET_ALIGN_32 Rect2 {
	Vector2<T> position;
	Vector2<T> size;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Rect2() {}

	ET_SIMD_INLINE Rect2(T p_x, T p_y, T p_width, T p_height) :
			position(p_x, p_y),
			size(p_width, p_height) {}

	ET_SIMD_INLINE Rect2(const Vector2<T> &p_pos, const Vector2<T> &p_size) :
			position(p_pos),
			size(p_size) {}

	// ------------------------------------------------------------------------
	// Accessors (Warp Kernel Friendly)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector2<T> get_position() const { return position; }
	ET_SIMD_INLINE void set_position(const Vector2<T> &p_pos) { position = p_pos; }

	ET_SIMD_INLINE Vector2<T> get_size() const { return size; }
	ET_SIMD_INLINE void set_size(const Vector2<T> &p_size) { size = p_size; }

	ET_SIMD_INLINE Vector2<T> get_end() const { return position + size; }
	ET_SIMD_INLINE void set_end(const Vector2<T> &p_end) { size = p_end - position; }

	ET_SIMD_INLINE Vector2<T> get_center() const { return position + (size * MathConstants<T>::half()); }

	// ------------------------------------------------------------------------
	// Deterministic Intersection API
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE bool operator==(const Rect2 &p_rect) const { return position == p_rect.position && size == p_rect.size; }
	ET_SIMD_INLINE bool operator!=(const Rect2 &p_rect) const { return position != p_rect.position || size != p_rect.size; }

	ET_SIMD_INLINE bool intersects(const Rect2 &p_rect) const {
		if (position.x >= (p_rect.position.x + p_rect.size.x)) return false;
		if ((position.x + size.x) <= p_rect.position.x) return false;
		if (position.y >= (p_rect.position.y + p_rect.size.y)) return false;
		if ((position.y + size.y) <= p_rect.position.y) return false;
		return true;
	}

	ET_SIMD_INLINE bool encloses(const Rect2 &p_rect) const {
		return (p_rect.position.x >= position.x) && (p_rect.position.y >= position.y) &&
				((p_rect.position.x + p_rect.size.x) <= (position.x + size.x)) &&
				((p_rect.position.y + p_rect.size.y) <= (position.y + size.y));
	}

	ET_SIMD_INLINE bool has_point(const Vector2<T> &p_point) const {
		if (p_point.x < position.x) return false;
		if (p_point.y < position.y) return false;
		if (p_point.x >= (position.x + size.size.x)) return false; // Note: Godot usually uses >= for endpoint check in Rect2
		if (p_point.y >= (position.y + size.size.y)) return false;
		return true;
	}

	// ------------------------------------------------------------------------
	// Modification API (Batch Processing)
	// ------------------------------------------------------------------------
	void merge_with(const Rect2 &p_rect) {
		Vector2<T> min_v, max_v;

		min_v.x = MIN(position.x, p_rect.position.x);
		min_v.y = MIN(position.y, p_rect.position.y);

		max_v.x = MAX(position.x + size.x, p_rect.position.x + p_rect.size.x);
		max_v.y = MAX(position.y + size.y, p_rect.position.y + p_rect.size.y);

		position = min_v;
		size = max_v - min_v;
	}

	ET_SIMD_INLINE void expand_to(const Vector2<T> &p_vector) {
		Vector2<T> end = position + size;

		if (p_vector.x < position.x) position.x = p_vector.x;
		if (p_vector.y < position.y) position.y = p_vector.y;

		if (p_vector.x > end.x) end.x = p_vector.x;
		if (p_vector.y > end.y) end.y = p_vector.y;

		size = end - position;
	}

	ET_SIMD_INLINE bool is_equal_approx(const Rect2 &p_rect) const {
		return position.is_equal_approx(p_rect.position) && size.is_equal_approx(p_rect.size);
	}

	// Godot UI Conversion
	operator String() const {
		return "[P: " + (String)position + ", S: " + (String)size + "]";
	}
};

// Simulation Type Aliases
typedef Rect2<FixedMathCore> Rect2f; // Local 2D Physics Broadphase
typedef Rect2<BigIntCore> Rect2b;    // Macro-Scale 2D Map Bounds

#endif // RECT2_H

--- END OF FILE core/math/rect2.h ---
