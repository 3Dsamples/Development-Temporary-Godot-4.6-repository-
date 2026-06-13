// File 133: modules/genesis/src/recorders/file_writers.h
// FileWriter – a recorder that persists channels to disk as CSV or gzipped NPZ.
// It extends BaseRecorder and implements flush() to produce a human‑readable
// CSV file (one column per channel, header row with names) and an optional
// binary NPZ file for efficient numerical storage.

#ifndef GENESIS_RECORDERS_FILE_WRITERS_H
#define GENESIS_RECORDERS_FILE_WRITERS_H

#include "base_recorder.h"
#include "core/io/file_access.h"
#include "core/variant/variant.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"

namespace genesis {

class FileRecorder : public BaseRecorder {
	GDCLASS(FileRecorder, BaseRecorder);

public:
	enum Format {
		FORMAT_CSV,
		FORMAT_NPZ,
		FORMAT_BOTH
	};

	FileRecorder() : format(FORMAT_CSV), compression(false) {}

	void set_format(Format p_format) { format = p_format; }
	Format get_format() const { return format; }

	void set_compression(bool p_comp) { compression = p_comp; }
	bool get_compression() const { return compression; }

	// Implement flush: write all recorded frames to file(s).
	virtual Error flush(const String &p_path) override {
		ERR_FAIL_COND_V(channels.is_empty(), ERR_UNCONFIGURED);

		if (format == FORMAT_CSV || format == FORMAT_BOTH) {
			Error err = write_csv(p_path);
			if (err != OK) return err;
		}
		if (format == FORMAT_NPZ || format == FORMAT_BOTH) {
			Error err = write_npz(p_path);
			if (err != OK) return err;
		}
		return OK;
	}

private:
	Error write_csv(const String &p_path) {
		String csv_path = p_path;
		if (!csv_path.ends_with(".csv")) csv_path += ".csv";

		Ref<FileAccess> f = FileAccess::open(csv_path, FileAccess::WRITE);
		ERR_FAIL_COND_V(f.is_null(), ERR_FILE_CANT_WRITE);

		// Header: channel names
		LocalVector<String> names;
		for (const KeyValue<String, DataChannel> &kv : channels) {
			names.push_back(kv.key);
		}
		for (int i = 0; i < names.size(); ++i) {
			if (i > 0) f->store_string(",");
			f->store_string(names[i]);
		}
		f->store_string("\n");

		// Data rows (up to current_frame)
		for (int frame = 0; frame < current_frame; ++frame) {
			for (int i = 0; i < names.size(); ++i) {
				if (i > 0) f->store_string(",");
				const DataChannel &ch = channels[names[i]];
				const Variant &val = ch.data[frame];
				f->store_string(variant_to_string(val));
			}
			f->store_string("\n");
		}
		return OK;
	}

	Error write_npz(const String &p_path) {
		String npz_path = p_path;
		if (!npz_path.ends_with(".npz")) npz_path += ".npz";

		// Simple custom binary format: header, then per‑channel binary blocks.
		Ref<FileAccess> f = FileAccess::open(npz_path, FileAccess::WRITE);
		ERR_FAIL_COND_V(f.is_null(), ERR_FILE_CANT_WRITE);

		// Magic bytes "GNPZ"
		f->store_buffer((const uint8_t *)"GNPZ", 4);
		// Number of channels (uint32)
		int num_channels = channels.size();
		f->store_buffer((const uint8_t *)&num_channels, 4);
		// Number of frames (uint32)
		f->store_buffer((const uint8_t *)&current_frame, 4);

		// Per channel metadata
		for (const KeyValue<String, DataChannel> &kv : channels) {
			// Channel name length (uint16) + name bytes
			CharString cs = kv.key.utf8();
			uint16_t name_len = cs.length();
			f->store_buffer((const uint8_t *)&name_len, 2);
			f->store_buffer((const uint8_t *)cs.ptr(), name_len);
			// Variant type (uint8)
			uint8_t vtype = (uint8_t)kv.value.type;
			f->store_buffer((const uint8_t *)&vtype, 1);
		}

		// Data: for each frame, write each channel's data as raw float/int/etc.
		for (int frame = 0; frame < current_frame; ++frame) {
			for (const KeyValue<String, DataChannel> &kv : channels) {
				const Variant &val = kv.value.data[frame];
				write_variant_binary(f, val);
			}
		}

		return OK;
	}

	void write_variant_binary(const Ref<FileAccess> &f, const Variant &val) {
		switch (val.get_type()) {
			case Variant::FLOAT: {
				float v = val;
				f->store_buffer((const uint8_t *)&v, sizeof(float));
				break;
			}
			case Variant::VECTOR3: {
				Vector3 v = val;
				float buf[3] = {v.x, v.y, v.z};
				f->store_buffer((const uint8_t *)buf, sizeof(float)*3);
				break;
			}
			case Variant::INT: {
				int64_t v = val;
				f->store_buffer((const uint8_t *)&v, sizeof(int64_t));
				break;
			}
			default: // store as zero float
			{
				float zero = 0.0f;
				f->store_buffer((const uint8_t *)&zero, sizeof(float));
			}
		}
	}

	String variant_to_string(const Variant &val) {
		switch (val.get_type()) {
			case Variant::FLOAT: return rtos(val);
			case Variant::VECTOR3: {
				Vector3 v = val;
				return vformat("%s,%s,%s", rtos(v.x), rtos(v.y), rtos(v.z));
			}
			case Variant::INT: return itos((int64_t)val);
			default: return "0";
		}
	}

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_format", "format"), &FileRecorder::set_format);
		ClassDB::bind_method(D_METHOD("get_format"), &FileRecorder::get_format);
		ClassDB::bind_method(D_METHOD("set_compression", "comp"), &FileRecorder::set_compression);
		ClassDB::bind_method(D_METHOD("get_compression"), &FileRecorder::get_compression);
		BIND_ENUM_CONSTANT(FORMAT_CSV);
		BIND_ENUM_CONSTANT(FORMAT_NPZ);
		BIND_ENUM_CONSTANT(FORMAT_BOTH);
		ADD_PROPERTY(PropertyInfo(Variant::INT, "format", PROPERTY_HINT_ENUM, "CSV,NPZ,Both"), "set_format", "get_format");
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "compression"), "set_compression", "get_compression");
	}

	Format format;
	bool compression;
};

} // namespace genesis

#endif // GENESIS_RECORDERS_FILE_WRITERS_H