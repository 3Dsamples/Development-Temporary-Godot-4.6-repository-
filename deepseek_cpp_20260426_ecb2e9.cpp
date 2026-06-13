// File 109: modules/gaia/src/parser/parser.h
// Generic text parser for Gaia simulation input files.
// Reads key-value pairs, vectors, and matrices from human-readable scripts.
// Adapted to Godot's FileAccess and String utilities.

#ifndef GAIA_PARSER_PARSER_H
#define GAIA_PARSER_PARSER_H

#include "core/io/file_access.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "core/templates/hash_map.h"

namespace gaia::parser {

class Parser {
public:
	Parser() {}

	// Load and parse a Gaia simulation file.
	// Comments start with '#'; empty lines are skipped.
	// Lines have the form "key value(s)".
	Error load(const String &p_path) {
		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
		ERR_FAIL_COND_V(f.is_null(), ERR_FILE_CANT_OPEN);

		data.clear();

		while (!f->eof_reached()) {
			String line = f->get_line().strip_edges();
			if (line.is_empty() || line.begins_with("#")) continue;

			// Split line by whitespace
			Vector<String> tokens = line.split(" ", false);
			if (tokens.size() < 2) continue;

			String key = tokens[0];

			// Re-join remaining tokens as value string
			String value_str;
			for (int i = 1; i < tokens.size(); ++i) {
				if (i > 1) value_str += " ";
				value_str += tokens[i];
			}

			data[key] = value_str;
		}

		return OK;
	}

	// Check if a key was present.
	bool has_key(const String &p_key) const {
		return data.has(p_key);
	}

	// Get raw value string for a key.
	String get_value_string(const String &p_key, const String &p_default = "") const {
		HashMap<String, String>::ConstIterator it = data.find(p_key);
		if (it) return it->value;
		return p_default;
	}

	// Interpret value as real.
	real_t get_real(const String &p_key, real_t p_default = 0.0) const {
		String val = get_value_string(p_key);
		if (val.is_empty()) return p_default;
		return val.to_float();
	}

	// Interpret value as integer.
	int64_t get_int(const String &p_key, int64_t p_default = 0) const {
		String val = get_value_string(p_key);
		if (val.is_empty()) return p_default;
		return val.to_int();
	}

	// Interpret value as Vector2 ("x y").
	Vector2 get_vector2(const String &p_key, const Vector2 &p_default = Vector2()) const {
		Vector<String> parts = get_value_string(p_key).split(" ");
		if (parts.size() >= 2) {
			return Vector2(parts[0].to_float(), parts[1].to_float());
		}
		return p_default;
	}

	// Interpret value as Vector3 ("x y z").
	Vector3 get_vector3(const String &p_key, const Vector3 &p_default = Vector3()) const {
		Vector<String> parts = get_value_string(p_key).split(" ");
		if (parts.size() >= 3) {
			return Vector3(parts[0].to_float(), parts[1].to_float(), parts[2].to_float());
		}
		return p_default;
	}

	// Interpret value as Vector4 ("x y z w").
	Vector4 get_vector4(const String &p_key, const Vector4 &p_default = Vector4()) const {
		Vector<String> parts = get_value_string(p_key).split(" ");
		if (parts.size() >= 4) {
			return Vector4(parts[0].to_float(), parts[1].to_float(), parts[2].to_float(), parts[3].to_float());
		}
		return p_default;
	}

	// Return all keys.
	List<String> get_keys() const {
		List<String> keys;
		for (const KeyValue<String, String> &kv : data) {
			keys.push_back(kv.key);
		}
		return keys;
	}

private:
	HashMap<String, String> data;
};

} // namespace gaia::parser

#endif // GAIA_PARSER_PARSER_H