// File 53: modules/genesis/src/options/options_system.h
// Genesis-style hierarchical option tree, adapted to Godot's Variant and Dictionary.

#ifndef GENESIS_OPTIONS_OPTIONS_SYSTEM_H
#define GENESIS_OPTIONS_OPTIONS_SYSTEM_H

#include "core/variant/variant.h"
#include "core/variant/dictionary.h"
#include "core/string/ustring.h"
#include "core/io/json.h"
#include "core/error/error_macros.h"

namespace genesis::options {

/**
 * The Options class holds a nested dictionary of parameters that configure
 * solvers, entities, materials, sensors, etc. It provides type-safe accessors
 * and hierarchical merging, mimicking Genesis' `gs.Options`.
 */
class Options {
private:
	Dictionary data;

public:
	Options() {}
	explicit Options(const Dictionary &p_dict) : data(p_dict) {}

	// --- Load / save from JSON (compatible with Genesis .yaml) ---
	Error load_from_json(const String &p_path) {
		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
		ERR_FAIL_COND_V(f.is_null(), ERR_FILE_CANT_OPEN);
		String text = f->get_as_utf8_string();
		JSON json;
		Error err = json.parse(text);
		if (err != OK) return err;
		Variant res = json.get_data();
		if (res.get_type() != Variant::DICTIONARY) return ERR_PARSE_ERROR;
		data = res;
		return OK;
	}

	Error save_to_json(const String &p_path) const {
		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
		ERR_FAIL_COND_V(f.is_null(), ERR_FILE_CANT_OPEN);
		JSON json;
		String text = json.stringify(data, "\t");
		f->store_string(text);
		return OK;
	}

	// --- Hierarchical access via dot-separated keys ---
	Variant get(const String &p_key, const Variant &p_default = Variant()) const {
		Vector<String> parts = p_key.split(".");
		Variant current = data;
		for (const String &part : parts) {
			if (current.get_type() != Variant::DICTIONARY) return p_default;
			Dictionary d = current;
			if (!d.has(part)) return p_default;
			current = d[part];
		}
		return current;
	}

	void set(const String &p_key, const Variant &p_value) {
		Vector<String> parts = p_key.split(".");
		if (parts.size() == 1) {
			data[parts[0]] = p_value;
			return;
		}
		// Walk / create nested dicts
		Dictionary *current = &data;
		for (int i = 0; i < parts.size() - 1; ++i) {
			if (!current->has(parts[i]) || (*current)[parts[i]].get_type() != Variant::DICTIONARY) {
				(*current)[parts[i]] = Dictionary();
			}
			Dictionary next = (*current)[parts[i]];
			current = &next; // pointer to local copy is invalid; need persistent reference.
		}
		// Fix: we need to modify the actual dict chain; use recursive helper.
		_set_recursive(data, parts, 0, p_value);
	}

	// Merge another Options object (shallow merge for leaves)
	void merge(const Options &p_other) {
		_merge_dict(data, p_other.data);
	}

	// Typed getters with defaults
	real_t get_real(const String &p_key, real_t p_default = 0.0) const {
		Variant v = get(p_key);
		if (v.get_type() == Variant::FLOAT || v.get_type() == Variant::INT) return v;
		return p_default;
	}
	int64_t get_int(const String &p_key, int64_t p_default = 0) const {
		Variant v = get(p_key);
		if (v.get_type() == Variant::INT) return int64_t(v);
		return p_default;
	}
	bool get_bool(const String &p_key, bool p_default = false) const {
		Variant v = get(p_key);
		if (v.get_type() == Variant::BOOL) return v;
		return p_default;
	}
	Vector3 get_vector3(const String &p_key, const Vector3 &p_default = Vector3()) const {
		Variant v = get(p_key);
		if (v.get_type() == Variant::ARRAY) {
			Array arr = v;
			if (arr.size() == 3) return Vector3(arr[0], arr[1], arr[2]);
		}
		return p_default;
	}
	String get_string(const String &p_key, const String &p_default = "") const {
		Variant v = get(p_key);
		if (v.get_type() == Variant::STRING) return v;
		return p_default;
	}
	Dictionary get_dict(const String &p_key, const Dictionary &p_default = Dictionary()) const {
		Variant v = get(p_key);
		if (v.get_type() == Variant::DICTIONARY) return v;
		return p_default;
	}

	// Return raw dictionary for low-level ops
	const Dictionary &get_data() const { return data; }

private:
	void _set_recursive(Variant &current, const Vector<String> &parts, int idx, const Variant &value) {
		if (idx == parts.size() - 1) {
			Dictionary d = current;
			d[parts[idx]] = value;
			current = d;
			return;
		}
		Dictionary d = current;
		if (!d.has(parts[idx]) || d[parts[idx]].get_type() != Variant::DICTIONARY) {
			d[parts[idx]] = Dictionary();
		}
		Variant next = d[parts[idx]];
		_set_recursive(next, parts, idx + 1, value);
		d[parts[idx]] = next;
		current = d;
	}

	void _merge_dict(Dictionary &target, const Dictionary &source) {
		for (const KeyValue<String, Variant> &kv : source) {
			if (target.has(kv.key) && target[kv.key].get_type() == Variant::DICTIONARY &&
				kv.value.get_type() == Variant::DICTIONARY) {
				Dictionary t = target[kv.key];
				_merge_dict(t, kv.value);
				target[kv.key] = t;
			} else {
				target[kv.key] = kv.value;
			}
		}
	}
};

} // namespace genesis::options

#endif // GENESIS_OPTIONS_OPTIONS_SYSTEM_H