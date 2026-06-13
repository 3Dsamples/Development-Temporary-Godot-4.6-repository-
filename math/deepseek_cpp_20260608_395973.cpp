// File 16: modules/gaia/src/io/parameter_writer.h

#ifndef GAIA_IO_PARAMETER_WRITER_H
#define GAIA_IO_PARAMETER_WRITER_H

#include "core/io/file_access.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/variant/variant.h"

namespace gaia::io {

/**
 * Parameter writer: stores key‑value pairs and writes them to a file
 * in a human‑readable format (compatible with ParameterReader).
 */
class ParameterWriter {
public:
	ParameterWriter() {}

	// Set a parameter (overwrites existing).
	void set_string(const String &p_key, const String &p_value) {
		params[p_key] = Variant(p_value);
	}
	void set_real(const String &p_key, real_t p_value) {
		params[p_key] = Variant(p_value);
	}
	void set_int(const String &p_key, int64_t p_value) {
		params[p_key] = Variant(p_value);
	}
	void set_bool(const String &p_key, bool p_value) {
		params[p_key] = Variant(p_value);
	}
	void set_vector3(const String &p_key, const Vector3 &p_value) {
		String s = vformat("%s %s %s",
			rtos(p_value.x),
			rtos(p_value.y),
			rtos(p_value.z));
		params[p_key] = Variant(s);
	}

	// Remove a parameter.
	void erase(const String &p_key) {
		params.erase(p_key);
	}

	// Write all parameters to a file (overwrites if exists).
	Error write(const String &p_path) const {
		Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
		ERR_FAIL_COND_V_MSG(file.is_null(), ERR_FILE_CANT_OPEN,
				"ParameterWriter: Cannot open file for writing: " + p_path);

		for (const KeyValue<String, Variant> &kv : params) {
			String line;
			Variant::Type t = kv.value.get_type();
			switch (t) {
				case Variant::STRING:
					line = vformat("%s = \"%s\"", kv.key, kv.value);
					break;
				case Variant::FLOAT:
					line = vformat("%s = %s", kv.key, rtos(kv.value));
					break;
				case Variant::INT:
					line = vformat("%s = %d", kv.key, (int64_t)kv.value);
					break;
				case Variant::BOOL:
					line = vformat("%s = %s", kv.key,
						(bool(kv.value) ? "true" : "false"));
					break;
				case Variant::VECTOR3: {
					Vector3 v = kv.value;
					line = vformat("%s = %s %s %s",
						kv.key, rtos(v.x), rtos(v.y), rtos(v.z));
				} break;
				default:
					// Fallback to string representation.
					line = vformat("%s = %s", kv.key, kv.value);
					break;
			}
			file->store_line(line);
		}
		return OK;
	}

	// Clear all stored parameters.
	void clear() { params.clear(); }

private:
	HashMap<String, Variant> params;
};

} // namespace gaia::io

#endif // GAIA_IO_PARAMETER_WRITER_H