// File 138: modules/gaia/src/parser/input_handler.h
// InputHandler: processes command‑line arguments, menu selections, and
// configuration file paths to set up a Gaia simulation scene.
// Replaces Gaia's `InputHandler.h` with a Godot‑style argument parser.

#ifndef GAIA_PARSER_INPUT_HANDLER_H
#define GAIA_PARSER_INPUT_HANDLER_H

#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/variant/variant.h"
#include "core/os/main_loop.h"           // for OS argument access
#include "parser.h"                      // the Gaia script parser

namespace gaia::parser {

class InputHandler {
public:
	InputHandler() {}

	// Parse command‑line arguments into a dictionary.
	// Supports: --key value  or --key=value  or --flag
	void parse_arguments(int argc, char *argv[]) {
		args.clear();
		for (int i = 1; i < argc; ++i) {
			String arg(argv[i]);
			if (arg.begins_with("--")) {
				String key = arg.replace_first("--", "");
				// check for '='
				int eq = key.find("=");
				String value;
				if (eq != -1) {
					value = key.substr(eq + 1, key.length() - eq - 1);
					key = key.substr(0, eq);
				} else if (i + 1 < argc && !String(argv[i + 1]).begins_with("--")) {
					value = argv[++i];
				} else {
					value = "true"; // flag = true
				}
				args[key] = value;
			}
		}
	}

	// Get an argument as a string, or default.
	String get_arg(const String &p_key, const String &p_default = "") const {
		HashMap<String, String>::ConstIterator it = args.find(p_key);
		return it ? it->value : p_default;
	}

	// Typed accessors.
	bool get_bool(const String &p_key, bool p_default = false) const {
		String v = get_arg(p_key);
		if (v.is_empty()) return p_default;
		return (v == "true" || v == "1" || v == "yes");
	}
	real_t get_real(const String &p_key, real_t p_default = 0.0) const {
		String v = get_arg(p_key);
		if (v.is_empty()) return p_default;
		return v.to_float();
	}
	int64_t get_int(const String &p_key, int64_t p_default = 0) const {
		String v = get_arg(p_key);
		if (v.is_empty()) return p_default;
		return v.to_int();
	}

	// Optional: parse a Gaia script path and load its content into a Parser.
	void load_script(const String &p_path, Parser &r_parser) {
		r_parser.load(p_path);
	}

private:
	HashMap<String, String> args;
};

} // namespace gaia::parser

#endif // GAIA_PARSER_INPUT_HANDLER_H