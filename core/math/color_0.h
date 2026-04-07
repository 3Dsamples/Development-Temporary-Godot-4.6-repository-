--- START OF FILE core/math/color.h ---

#ifndef COLOR_H
#define COLOR_H

#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Color Template
 * 
 * Deterministic RGBA color structure for high-fidelity simulation.
 * Aligned to 32 bytes to ensure that EnTT color component streams are
 * SIMD-ready for Warp-style parallel rendering or spectral math kernels.
 */
template <typename T>
struct ET_ALIGN_32 Color {
	T r, g, b, a;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Color() : r(T()), g(T()), b(T()), a(MathConstants<T>::one()) {}
	ET_SIMD_INLINE Color(T p_r, T p_g, T p_b, T p_a = MathConstants<T>::one()) : r(p_r), g(p_g), b(p_b), a(p_a) {}

	// ------------------------------------------------------------------------
	// Operators (Deterministic Batch Logic)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE bool operator==(const Color &p_color) const { return (r == p_color.r && g == p_color.g && b == p_color.b && a == p_color.a); }
	ET_SIMD_INLINE bool operator!=(const Color &p_color) const { return (r != p_color.r || g != p_color.g || b != p_color.b || a != p_color.a); }

	ET_SIMD_INLINE Color operator+(const Color &p_color) const { return Color(r + p_color.r, g + p_color.g, b + p_color.b, a + p_color.a); }
	ET_SIMD_INLINE void operator+=(const Color &p_color) { r += p_color.r; g += p_color.g; b += p_color.b; a += p_color.a; }
	ET_SIMD_INLINE Color operator-(const Color &p_color) const { return Color(r - p_color.r, g - p_color.g, b - p_color.b, a - p_color.a); }
	ET_SIMD_INLINE void operator-=(const Color &p_color) { r -= p_color.r; g -= p_color.g; b -= p_color.b; a -= p_color.a; }

	ET_SIMD_INLINE Color operator*(const Color &p_color) const { return Color(r * p_color.r, g * p_color.g, b * p_color.b, a * p_color.a); }
	ET_SIMD_INLINE Color operator*(T p_scalar) const { return Color(r * p_scalar, g * p_scalar, b * p_scalar, a * p_scalar); }
	ET_SIMD_INLINE void operator*=(T p_scalar) { r *= p_scalar; g *= p_scalar; b *= p_scalar; a *= p_scalar; }
	ET_SIMD_INLINE Color operator/(const Color &p_color) const { return Color(r / p_color.r, g / p_color.g, b / p_color.b, a / p_color.a); }
	ET_SIMD_INLINE Color operator/(T p_scalar) const { return Color(r / p_scalar, g / p_scalar, b / p_scalar, a / p_scalar); }
	ET_SIMD_INLINE void operator/=(T p_scalar) { r /= p_scalar; g /= p_scalar; b /= p_scalar; a /= p_scalar; }
	ET_SIMD_INLINE Color operator-() const { return Color(-r, -g, -b, -a); }

	ET_SIMD_INLINE T &operator[](int p_idx) { return (&r)[p_idx]; }
	ET_SIMD_INLINE const T &operator[](int p_idx) const { return (&r)[p_idx]; }

	// ------------------------------------------------------------------------
	// Deterministic Simulation Math
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE T get_luminance() const {
		// Rec. 709 deterministic coefficients: 0.2126R + 0.7152G + 0.0722B
		// Represented as bit-perfect FixedMathCore values.
		if constexpr (std::is_same<T, FixedMathCore>::value) {
			return r * FixedMathCore(913142732LL, true) + g * g * FixedMathCore(3071746048LL, true) + b * FixedMathCore(310118991LL, true);
		} else {
			return (r * T(2126) + g * T(7152) + b * T(722)) / T(10000);
		}
	}

	ET_SIMD_INLINE Color lerp(const Color &p_to, T p_weight) const {
		return Color(
				Math::lerp(r, p_to.r, p_weight),
				Math::lerp(g, p_to.g, p_weight),
				Math::lerp(b, p_to.b, p_weight),
				Math::lerp(a, p_to.a, p_weight));
	}

	ET_SIMD_INLINE Color clamped(T p_min = MathConstants<T>::zero(), T p_max = MathConstants<T>::one()) const {
		return Color(
				CLAMP(r, p_min, p_max),
				CLAMP(g, p_min, p_max),
				CLAMP(b, p_min, p_max),
				CLAMP(a, p_min, p_max));
	}

	// Godot UI/HTML Conversion
	String to_html(bool p_alpha = true) const {
		auto to_hex = [](T v) {
			if constexpr (std::is_same<T, BigIntCore>::value) return String("00"); // Discrete scale fallback
			int64_t val = (v * FixedMathCore(255LL, false)).to_int();
			val = CLAMP(val, 0, 255);
			return String::num(val, 16).to_upper();
		};
		String txt = to_hex(r) + to_hex(g) + to_hex(b);
		if (p_alpha) txt += to_hex(a);
		return txt;
	}

	operator String() const {
		return "(" + String(r.to_string().c_str()) + ", " + String(g.to_string().c_str()) + ", " + String(b.to_string().c_str()) + ", " + String(a.to_string().c_str()) + ")";
	}
};

typedef Color<FixedMathCore> Colorf; // Deterministic Simulation/HDR
typedef Color<BigIntCore> Colorb;    // Discrete Spectral Counts

#endif // COLOR_H

--- END OF FILE core/math/color.h ---
