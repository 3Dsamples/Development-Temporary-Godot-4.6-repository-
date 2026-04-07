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
 * 32-byte aligned 2D bounding box for deterministic spatial logic.
 * Replaces standard floating-point rectangles to eliminate rounding jitter.
 * Engineered for high-speed EnTT component batch processing.
 */
template <typename T>
struct ET_ALIGN_32 Rect2 {
	Vector2<T> position;
	Vector2<T> size;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Rect2() {}

	_FORCE_INLINE_ Rect2(T p_x, T p_y, T p_width, T p_height) :
			position(p_x, p_y),
			size(p_width, p_height) {}

	_FORCE_INLINE_ Rect2(const Vector2<T> &p_pos, const Vector2<T> &p_size) :
			position(p_pos),
			size(p_size) {}

	// ------------------------------------------------------------------------
	// Accessors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector2<T> get_position() const { return position; }
	_FORCE_INLINE_ void set_position(const Vector2<T> &p_pos) { position = p_pos; }

	_FORCE_INLINE_ Vector2<T> get_size() const { return size; }
	_FORCE_INLINE_ void set_size(const Vector2<T> &p_size) { size = p_size; }

	_FORCE_INLINE_ Vector2<T> get_end() const { return position + size; }
	_FORCE_INLINE_ void set_end(const Vector2<T> &p_end) { size = p_end - position; }

	_FORCE_INLINE_ Vector2<T> get_center() const { return position + (size * MathConstants<T>::half()); }

	// ------------------------------------------------------------------------
	// Deterministic Intersection API
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ bool operator==(const Rect2 &p_rect) const { return position == p_rect.position && size == p_rect.size; }
	_FORCE_INLINE_ bool operator!=(const Rect2 &p_rect) const { return position != p_rect.position || size != p_rect.size; }

	/**
	 * intersects()
	 * Checks if another rectangle overlaps with this one.
	 * Bit-perfect implementation for 120 FPS 2D broadphase.
	 */
	_FORCE_INLINE_ bool intersects(const Rect2 &p_rect) const {
		if (position.x >= (p_rect.position.x + p_rect.size.x)) return false;
		if ((position.x + size.x) <= p_rect.position.x) return false;
		if (position.y >= (p_rect.position.y + p_rect.size.y)) return false;
		if ((position.y + size.y) <= p_rect.position.y) return false;
		return true;
	}

	_FORCE_INLINE_ bool encloses(const Rect2 &p_rect) const {
		return (p_rect.position.x >= position.x) && (p_rect.position.y >= position.y) &&
				((p_rect.position.x + p_rect.size.x) <= (position.x + size.x)) &&
				((p_rect.position.y + p_rect.size.y) <= (position.y + size.y));
	}

	_FORCE_INLINE_ bool has_point(const Vector2<T> &p_point) const {
		if (p_point.x < position.x) return false;
		if (p_point.y < position.y) return false;
		if (p_point.x >= (position.x + size.x)) return false;
		if (p_point.y >= (position.y + size.y)) return false;
		return true;
	}

	// ------------------------------------------------------------------------
	// Sophisticated 2D Behaviors
	// ------------------------------------------------------------------------

	/**
	 * get_area()
	 * Returns the surface area. Essential for 2D soft-body volume pressure simulation.
	 */
	_FORCE_INLINE_ T get_area() const {
		return size.x * size.y;
	}

	/**
	 * intersection()
	 * Returns the bit-perfect intersection rectangle.
	 */
	_FORCE_INLINE_ Rect2 intersection(const Rect2 &p_rect) const {
		Rect2 res;
		if (!intersects(p_rect)) return res;

		res.position.x = MAX(position.x, p_rect.position.x);
		res.position.y = MAX(position.y, p_rect.position.y);

		Vector2<T> res_end;
		res_end.x = MIN(position.x + size.x, p_rect.position.x + p_rect.size.x);
		res_end.y = MIN(position.y + size.y, p_rect.position.y + p_rect.size.y);

		res.size = res_end - res.position;
		return res;
	}

	// ------------------------------------------------------------------------
	// Modification API (Warp Batch Friendly)
	// ------------------------------------------------------------------------
	
	_FORCE_INLINE_ void merge(const Rect2 &p_rect) {
		Vector2<T> min_v, max_v;
		min_v.x = MIN(position.x, p_rect.position.x);
		min_v.y = MIN(position.y, p_rect.position.y);
		max_v.x = MAX(position.x + size.x, p_rect.position.x + p_rect.size.x);
		max_v.y = MAX(position.y + size.y, p_rect.position.y + p_rect.size.y);

		position = min_v;
		size = max_v - min_v;
	}

	_FORCE_INLINE_ Rect2 expand(const Vector2<T> &p_vector) const {
		Rect2 r = *this;
		Vector2<T> end = r.position + r.size;

		if (p_vector.x < r.position.x) r.position.x = p_vector.x;
		if (p_vector.y < r.position.y) r.position.y = p_vector.y;
		if (p_vector.x > end.x) end.x = p_vector.x;
		if (p_vector.y > end.y) end.y = p_vector.y;

		r.size = end - r.position;
		return r;
	}

	_FORCE_INLINE_ Rect2 grow(T p_amount) const {
		Rect2 g = *this;
		g.position.x -= p_amount;
		g.position.y -= p_amount;
		g.size.x += p_amount * T(2LL);
		g.size.y += p_amount * T(2LL);
		return g;
	}

	_FORCE_INLINE_ Rect2 snapped(const Vector2<T> &p_step) const {
		return Rect2(position.snapped(p_step), size.snapped(p_step));
	}

	operator String() const {
		return "[P: " + (String)position + ", S: " + (String)size + "]";
	}
};

// Simulation Tier Typedefs
typedef Rect2<FixedMathCore> Rect2f; // Bit-perfect 2D collision volumes
typedef Rect2<BigIntCore> Rect2b;    // Discrete macro-grid sector bounds

#endif // RECT2_H

--- END OF FILE core/math/rect2.h ---
