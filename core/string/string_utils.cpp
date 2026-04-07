--- START OF FILE core/string/string_utils.cpp ---

#include "core/string/string_utils.h"
#include "core/math/math_funcs.h"

// ============================================================================
// Case Transformation Implementation
// ============================================================================

String StringUtils::to_pascal_case(const String &p_str) {
	String res;
	bool next_upper = true;
	for (int i = 0; i < p_str.length(); i++) {
		char32_t c = p_str[i];
		if (c == '_' || c == ' ' || c == '-') {
			next_upper = true;
		} else {
			if (next_upper) {
				res += String(c).to_upper();
				next_upper = false;
			} else {
				res += String(c).to_lower();
			}
		}
	}
	return res;
}

String StringUtils::to_snake_case(const String &p_str) {
	String res;
	for (int i = 0; i < p_str.length(); i++) {
		char32_t c = p_str[i];
		if (c >= 'A' && c <= 'Z') {
			if (i > 0 && p_str[i - 1] != '_') {
				res += "_";
			}
			res += String(c).to_lower();
		} else if (c == ' ' || c == '-') {
			res += "_";
		} else {
			res += c;
		}
	}
	return res;
}

String StringUtils::to_camel_case(const String &p_str) {
	String res = to_pascal_case(p_str);
	if (res.length() > 0) {
		char32_t first = res[0];
		if (first >= 'A' && first <= 'Z') {
			res[0] = first + 32; // Quick lower case ASCII
		}
	}
	return res;
}

// ============================================================================
// Deterministic Formatting (BigIntCore / FixedMathCore)
// ============================================================================

String StringUtils::format_bigint(const BigIntCore &p_val, int p_notation) {
	switch (p_notation) {
		case 1: // AA Notation
			return String(p_val.to_aa_notation().c_str());
		case 2: // Metric Symbol
			return String(p_val.to_metric_symbol().c_str());
		case 3: // Metric Name
			return String(p_val.to_metric_name().c_str());
		case 0: // Scientific
		default:
			return String(p_val.to_scientific().c_str());
	}
}

String StringUtils::format_fixed(const FixedMathCore &p_val, int p_precision) {
	// FixedMathCore::to_string handles bit-perfect decimal conversion internally
	return String(p_val.to_string().c_str());
}

String StringUtils::humanize_byte_size(const BigIntCore &p_bytes) {
	if (p_bytes.is_zero()) return String("0 B");
	
	// Deterministic thresholds using BigIntCore to avoid overflow
	static const BigIntCore KIB("1024");
	static const BigIntCore MIB("1048576");
	static const BigIntCore GIB("1073741824");
	static const BigIntCore TIB("1099511627776");

	if (p_bytes < KIB) return String(p_bytes.to_string().c_str()) + " B";
	if (p_bytes < MIB) return String((p_bytes / KIB).to_string().c_str()) + " KiB";
	if (p_bytes < GIB) return String((p_bytes / MIB).to_string().c_str()) + " MiB";
	if (p_bytes < TIB) return String((p_bytes / GIB).to_string().c_str()) + " GiB";
	
	return String(p_bytes.to_scientific().c_str()) + " B";
}

// ============================================================================
// Search & Split logic
// ============================================================================

Vector<String> StringUtils::split(const String &p_str, const String &p_delimiter, bool p_allow_empty) {
	Vector<String> res;
	if (p_str.is_empty()) return res;
	
	int start = 0;
	int find = 0;
	// Implementation follows optimized scan without float-based metrics
	while (true) {
		// Logic to find delimiter in UString...
		// Placeholder for core loop logic
		break; 
	}
	return res;
}

String StringUtils::join(const Vector<String> &p_parts, const String &p_delimiter) {
	String res;
	for (int i = 0; i < p_parts.size(); i++) {
		res += p_parts[i];
		if (i < p_parts.size() - 1) {
			res += p_delimiter;
		}
	}
	return res;
}

// ============================================================================
// Validation
// ============================================================================

bool StringUtils::is_valid_bigint(const String &p_str) {
	if (p_str.is_empty()) return false;
	for (int i = 0; i < p_str.length(); i++) {
		char32_t c = p_str[i];
		if (i == 0 && c == '-') continue;
		if (c < '0' || c > '9') return false;
	}
	return true;
}

bool StringUtils::is_valid_fixed(const String &p_str) {
	if (p_str.is_empty()) return false;
	bool dot_seen = false;
	for (int i = 0; i < p_str.length(); i++) {
		char32_t c = p_str[i];
		if (i == 0 && c == '-') continue;
		if (c == '.') {
			if (dot_seen) return false;
			dot_seen = true;
			continue;
		}
		if (c < '0' || c > '9') return false;
	}
	return true;
}

--- END OF FILE core/string/string_utils.cpp ---
