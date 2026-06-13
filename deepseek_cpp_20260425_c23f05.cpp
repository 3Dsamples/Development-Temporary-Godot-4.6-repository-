// File 15: modules/gaia/src/io/parameter_reader.h

#ifndef GAIA_IO_PARAMETER_READER_H
#define GAIA_IO_PARAMETER_READER_H

#include "core/io/file_access.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "core/templates/hash_map.h"
#include "core/error/error_macros.h"

namespace gaia::io {

/**
 * Simple parameter reader for simulation configuration files.
 *
 * The file format is line‑based: each line contains either a comment
 * (starting with #), a blank line, or a key = value pair.
 * Supported value types: real, integer, string (double‑quoted), and
 * Vector3 (written as three space‑separated reals or "x y z").
 *
 * Usage:
 *    ParameterReader reader;
 *    reader.load("res://simulation.cfg");
 *    real_t stiffness = reader.get_real("cloth.stiffness", 1000.0);
 */
class ParameterReader {
public:
	ParameterReader() {}

	// Load and parse a file. Returns OK or error.
	Error load(const String &p_path) {
		Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
		ERR_FAIL_COND_V_MSG(file.is_null(), ERR_FILE_CANT_OPEN,
				"Cannot open parameter file: " + p_path);

		params.clear();

		while (!file->eof_reached()) {
			String line = file->get_line().strip_edges();
			if (line.is_empty() || line.begins_with("#")) {
				continue;
			}

			int eq = line.find("=");
			ERR_CONTINUE_MSG(eq == -1, "Malformed line (no '='): " + line);

			String key = line.substr(0, eq).strip_edges();
			String value = line.substr(eq + 1, line.length()).strip_edges();

			params[key] = value;
		}

		return OK;
	}

	// Check if a parameter exists.
	bool has(const String &p_key) const {
		return params.has(p_key);
	}

	// Retrieve a string value.
	String get_string(const String &p_key, const String &p_default = "") const {
		HashMap<String, String>::ConstIterator it = params.find(p_key);
		if (!it) return p_default;
		// If the string is quoted, remove the quotes.
		String val = it->value;
		if (val.begins_with("\"") && val.ends_with("\"")) {
			val = val.substr(1, val.length() - 2);
		}
		return val;
	}

	// Retrieve a real number.
	real_t get_real(const String &p_key, real_t p_default = 0.0) const {
		HashMap<String, String>::ConstIterator it = params.find(p_key);
		if (!it) return p_default;
		return it->value.to_float();
	}

	// Retrieve an integer.
	int64_t get_int(const String &p_key, int64_t p_default = 0) const {
		HashMap<String, String>::ConstIterator it = params.find(p_key);
		if (!it) return p_default;
		return it->value.to_int();
	}

	// Retrieve a Vector3 (values separated by spaces or commas).
	Vector3 get_vector3(const String &p_key, const Vector3 &p_default = Vector3()) const {
		HashMap<String, String>::ConstIterator it = params.find(p_key);
		if (!it) return p_default;
		Vector<String> parts = it->value.split(" ");
		if (parts.size() < 3) {
			parts = it->value.split(","); // try comma
		}
		if (parts.size() >= 3) {
			return Vector3(
				parts[0].to_float(),
				parts[1].to_float(),
				parts[2].to_float()
			);
		}
		return p_default;
	}

	// Retrieve a boolean ("true"/"yes"/"1" -> true).
	bool get_bool(const String &p_key, bool p_default = false) const {
		HashMap<String, String>::ConstIterator it = params.find(p_key);
		if (!it) return p_default;
		String v = it->value.to_lower();
		return (v == "true" || v == "yes" || v == "1");
	}

	// Return all parameter keys.
	List<String> keys() const {
		List<String> out;
		for (const KeyValue<String, String> &kv : params) {
			out.push_back(kv.key);
		}
		return out;
	}

	// Clear all loaded parameters.
	void clear() { params.clear(); }

private:
	HashMap<String, String> params;
};

} // namespace gaia::io

#endif // GAIA_IO_PARAMETER_READER_H