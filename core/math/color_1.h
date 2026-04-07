--- START OF FILE core/math/color.h ---

#ifndef COLOR_H
#define COLOR_H

#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Color Template
 * 
 * 32-byte aligned RGBA structure for deterministic color science.
 * Strictly uses Software-Defined Arithmetic to eliminate spectral drift.
 * Optimized for Warp-style parallel shading kernels.
 */
template <typename T>
struct ET_ALIGN_32 Color {
	T r, g, b, a;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Color() : r(MathConstants<T>::zero()), g(MathConstants<T>::zero()), b(MathConstants<T>::zero()), a(MathConstants<T>::one()) {}
	_FORCE_INLINE_ Color(T p_r, T p_g, T p_b, T p_a = MathConstants<T>::one()) : r(p_r), g(p_g), b(p_b), a(p_a) {}

	// ------------------------------------------------------------------------
	// Operators (Deterministic Batch Logic)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ bool operator==(const Color &p_color) const { return (r == p_color.r && g == p_color.g && b == p_color.b && a == p_color.a); }
	_FORCE_INLINE_ bool operator!=(const Color &p_color) const { return !(*this == p_color); }

	_FORCE_INLINE_ Color operator+(const Color &p_color) const { return Color(r + p_color.r, g + p_color.g, b + p_color.b, a + p_color.a); }
	_FORCE_INLINE_ void operator+=(const Color &p_color) { r += p_color.r; g += p_color.g; b += p_color.b; a += p_color.a; }
	_FORCE_INLINE_ Color operator-(const Color &p_color) const { return Color(r - p_color.r, g - p_color.g, b - p_color.b, a - p_color.a); }
	_FORCE_INLINE_ void operator-=(const Color &p_color) { r -= p_color.r; g -= p_color.g; b -= p_color.b; a -= p_color.a; }

	_FORCE_INLINE_ Color operator*(const Color &p_color) const { return Color(r * p_color.r, g * p_color.g, b * p_color.b, a * p_color.a); }
	_FORCE_INLINE_ Color operator*(T p_scalar) const { return Color(r * p_scalar, g * p_scalar, b * p_scalar, a * p_scalar); }
	_FORCE_INLINE_ void operator*=(T p_scalar) { r *= p_scalar; g *= p_scalar; b *= p_scalar; a *= p_scalar; }

	_FORCE_INLINE_ Color operator/(const Color &p_color) const { return Color(r / p_color.r, g / p_color.g, b / p_color.b, a / p_color.a); }
	_FORCE_INLINE_ Color operator/(T p_scalar) const { return Color(r / p_scalar, g / p_scalar, b / p_scalar, a / p_scalar); }
	_FORCE_INLINE_ void operator/=(T p_scalar) { r /= p_scalar; g /= p_scalar; b /= p_scalar; a /= p_scalar; }

	_FORCE_INLINE_ Color operator-() const { return Color(-r, -g, -b, -a); }

	_FORCE_INLINE_ T &operator[](int p_idx) { return (&r)[p_idx]; }
	_FORCE_INLINE_ const T &operator[](int p_idx) const { return (&r)[p_idx]; }

	// ------------------------------------------------------------------------
	// Sophisticated Spectral API
	// ------------------------------------------------------------------------

	/**
	 * get_luminance()
	 * Returns the bit-perfect Rec. 709 luminance tensor.
	 * Coeffs: R: 0.2126, G: 0.7152, B: 0.0722.
	 */
	_FORCE_INLINE_ T get_luminance() const {
		if constexpr (std::is_same<T, FixedMathCore>::value) {
			return r * FixedMathCore(913142732LL, true) + 
			       g * FixedMathCore(3071746048LL, true) + 
			       b * FixedMathCore(310118991LL, true);
		}
		return (r + g + b) / T(3LL); // Fallback for BigIntCore macro-scaling
	}

	/**
	 * apply_anime_banding()
	 * Specialized behavior: Snaps color intensities to discrete bands.
	 * Essential for cel-shaded 120 FPS high-dynamic-range visuals.
	 */
	_FORCE_INLINE_ Color apply_anime_banding(int p_levels) const {
		T levels_f(static_cast<int64_t>(p_levels));
		T inv_levels = MathConstants<T>::one() / levels_f;
		return Color(
				Math::floor(r * levels_f) * inv_levels,
				Math::floor(g * levels_f) * inv_levels,
				Math::floor(b * levels_f) * inv_levels,
				a);
	}

	_FORCE_INLINE_ Color lerp(const Color &p_to, T p_weight) const {
		return Color(
				Math::lerp(r, p_to.r, p_weight),
				Math::lerp(g, p_to.g, p_weight),
				Math::lerp(b, p_to.b, p_weight),
				Math::lerp(a, p_to.a, p_weight));
	}

	_FORCE_INLINE_ Color blend(const Color &p_over) const {
		T res_a = p_over.a + a * (MathConstants<T>::one() - p_over.a);
		if (res_a == MathConstants<T>::zero()) return Color(MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::zero());
		return Color(
				(p_over.r * p_over.a + r * a * (MathConstants<T>::one() - p_over.a)) / res_a,
				(p_over.g * p_over.a + g * a * (MathConstants<T>::one() - p_over.a)) / res_a,
				(p_over.b * p_over.a + b * a * (MathConstants<T>::one() - p_over.a)) / res_a,
				res_a);
	}

	_FORCE_INLINE_ void invert() {
		r = MathConstants<T>::one() - r;
		g = MathConstants<T>::one() - g;
		b = MathConstants<T>::one() - b;
	}

	_FORCE_INLINE_ Color inverted() const {
		Color c = *this;
		c.invert();
		return c;
	}

	// ------------------------------------------------------------------------
	// Conversion & UI
	// ------------------------------------------------------------------------
	String to_html(bool p_alpha = true) const;

	operator String() const {
		return "(" + String(r.to_string().c_str()) + ", " + 
		             String(g.to_string().c_str()) + ", " + 
		             String(b.to_string().c_str()) + ", " + 
		             String(a.to_string().c_str()) + ")";
	}
};

// Simulation Tier Typedefs
typedef Color<FixedMathCore> Colorf; // Bit-perfect spectral color/HDR
typedef Color<BigIntCore> Colorb;    // Discrete macro-spectral categories

#endif // COLOR_H

--- END OF FILE core/math/color.h ---
