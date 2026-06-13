// File 17: modules/gaia/src/json/json_parser.h

#ifndef GAIA_JSON_JSON_PARSER_H
#define GAIA_JSON_JSON_PARSER_H

#include "core/io/json.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "core/error/error_macros.h"

namespace gaia::json {

/**
 * A simple JSON document wrapper around Godot's JSON parser.
 * Offers a minimal DOM-like interface similar to Gaia's original
 * custom parser, but leverages Godot's built‑in JSON class.
 *
 * Usage:
 *   JsonDocument doc;
 *   doc.parse(R"({"gravity": [0, -9.81, 0]})");
 *   Vector3 g = doc.get("gravity").to_vector3();
 */
class JsonDocument {
public:
	JsonDocument() : valid(false) {}

	// Parse a JSON string. Returns OK on success.
	Error parse(const String &p_json) {
		JSON json;
		Error err = json.parse(p_json);
		if (err != OK) {
			error_string = json.get_error_message();
			error_line = json.get_error_line();
			valid = false;
			root = Variant(); // clear
			return err;
		}
		root = json.get_data();
		valid = true;
		error_string = "";
		error_line = 0;
		return OK;
	}

	// Returns true if parsing succeeded.
	bool is_valid() const { return valid; }

	// Get the root Variant (Variant::DICTIONARY or ARRAY typically).
	Variant get_root() const {
		ERR_FAIL_COND_V(!valid, Variant());
		return root;
	}

	// Convenience: return the value at key (if root is a Dictionary).
	Variant get(const String &p_key) const {
		ERR_FAIL_COND_V(!valid || root.get_type() != Variant::DICTIONARY, Variant());
		Dictionary d = root;
		return d.get(p_key, Variant());
	}

	// Set a value at key in root dictionary (creates dict if not present).
	void set(const String &p_key, const Variant &p_value) {
		Dictionary d;
		if (valid && root.get_type() == Variant::DICTIONARY) {
			d = root;
		}
		d[p_key] = p_value;
		root = d;
		valid = true;
	}

	// Convenience typed getters.
	String get_string(const String &p_key, const String &p_default = "") const {
		Variant v = get(p_key);
		if (v.get_type() == Variant::STRING) return String(v);
		return p_default;
	}

	real_t get_real(const String &p_key, real_t p_default = 0.0) const {
		Variant v = get(p_key);
		if (v.get_type() == Variant::FLOAT || v.get_type() == Variant::INT)
			return real_t(v);
		return p_default;
	}

	int64_t get_int(const String &p_key, int64_t p_default = 0) const {
		Variant v = get(p_key);
		if (v.get_type() == Variant::INT) return int64_t(v);
		if (v.get_type() == Variant::FLOAT) return int64_t(real_t(v));
		return p_default;
	}

	bool get_bool(const String &p_key, bool p_default = false) const {
		Variant v = get(p_key);
		if (v.get_type() == Variant::BOOL) return bool(v);
		if (v.get_type() == Variant::INT) return int64_t(v) != 0;
		return p_default;
	}

	Vector3 get_vector3(const String &p_key, const Vector3 &p_default = Vector3()) const {
		Variant v = get(p_key);
		if (v.get_type() == Variant::ARRAY) {
			Array arr = v;
			if (arr.size() == 3) {
				return Vector3(
					real_t(arr[0]),
					real_t(arr[1]),
					real_t(arr[2])
				);
			}
		}
		if (v.get_type() == Variant::STRING) {
			// Try space‑separated fallback
			Vector<String> parts = String(v).split(" ");
			if (parts.size() == 3) {
				return Vector3(
					parts[0].to_float(),
					parts[1].to_float(),
					parts[2].to_float()
				);
			}
			parts = String(v).split(",");
			if (parts.size() == 3) {
				return Vector3(
					parts[0].to_float(),
					parts[1].to_float(),
					parts[2].to_float()
				);
			}
		}
		return p_default;
	}

	// Error information
	String get_error_string() const { return error_string; }
	int get_error_line() const { return error_line; }

	// Clear the document
	void clear() {
		root = Variant();
		valid = false;
		error_string = "";
		error_line = 0;
	}

private:
	Variant root;
	bool valid;
	String error_string;
	int error_line;
};

} // namespace gaia::json

#endif // GAIA_JSON_JSON_PARSER_H