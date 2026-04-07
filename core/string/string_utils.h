--- START OF FILE core/string/string_utils.h ---

#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * StringUtils
 * 
 * Utility class for deterministic string formatting and manipulation.
 * Replaces standard floating-point text conversion with Scale-Aware logic.
 * Optimized for 120 FPS UI updates in massive-scale simulation views.
 */
class StringUtils {
public:
	// ------------------------------------------------------------------------
	// Naming Convention Utilities
	// ------------------------------------------------------------------------

	static String to_pascal_case(const String &p_str);
	static String to_snake_case(const String &p_str);
	static String to_camel_case(const String &p_str);

	// ------------------------------------------------------------------------
	// Deterministic Formatting (BigIntCore / FixedMathCore)
	// ------------------------------------------------------------------------

	/**
	 * format_bigint()
	 * Converts an arbitrary-precision integer to a human-readable string.
	 * Supports Scientific, AA, and Metric notations for macro-economies.
	 */
	static String format_bigint(const BigIntCore &p_val, int p_notation = 0);

	/**
	 * format_fixed()
	 * Converts a deterministic fixed-point number to a decimal string.
	 * Guaranteed identical output on all hardware for physics debugging.
	 */
	static String format_fixed(const FixedMathCore &p_val, int p_precision = 6);

	/**
	 * humanize_byte_size()
	 * Uses BigIntCore to describe massive galactic data archives (PB, EB, ZB).
	 */
	static String humanize_byte_size(const BigIntCore &p_bytes);

	// ------------------------------------------------------------------------
	// Search & Split logic
	// ------------------------------------------------------------------------

	static Vector<String> split(const String &p_str, const String &p_delimiter, bool p_allow_empty = true);
	static String join(const Vector<String> &p_parts, const String &p_delimiter);
	
	static bool fuzzy_match(const String &p_pattern, const String &p_str);
	static int64_t find_closest_match(const String &p_str, const Vector<String> &p_options);

	// ------------------------------------------------------------------------
	// Validation
	// ------------------------------------------------------------------------

	static bool is_valid_bigint(const String &p_str);
	static bool is_valid_fixed(const String &p_str);
};

#endif // STRING_UTILS_H

--- END OF FILE core/string/string_utils.h ---
