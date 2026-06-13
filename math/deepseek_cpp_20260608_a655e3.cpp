// File 128: modules/genesis/src/recorders/base_recorder.h
// Base recorder class for logging simulation data to file or memory buffers.
// Provides a common interface for time-series recording of scalar, vector,
// and tensor quantities produced by solvers, sensors, or user scripts.
// Derived classes (CSV, NPZ, custom) implement the actual output.

#ifndef GENESIS_RECORDERS_BASE_RECORDER_H
#define GENESIS_RECORDERS_BASE_RECORDER_H

#include "core/io/resource.h"
#include "core/variant/variant.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"

namespace genesis {

class BaseRecorder : public Resource {
	GDCLASS(BaseRecorder, Resource);

public:
	BaseRecorder() : enabled(true), max_buffer_frames(1000), current_frame(0) {}

	// --- Enable / disable ---
	void set_enabled(bool p_enabled) { enabled = p_enabled; }
	bool is_enabled() const { return enabled; }

	// --- Buffer size (maximum number of frames stored in memory) ---
	void set_max_buffer_frames(int p_max) { max_buffer_frames = MAX(p_max, 1); }
	int get_max_buffer_frames() const { return max_buffer_frames; }

	// --- Data channel registration ---
	// Each channel has a name and a data type (REAL, VECTOR3, etc.).
	void register_channel(const String &p_name, Variant::Type p_type) {
		ERR_FAIL_COND(channel_types.has(p_name));
		channel_types[p_name] = p_type;
		// Allocate buffer
		DataChannel ch;
		ch.type = p_type;
		ch.data.resize(max_buffer_frames); // placeholder per-frame Variant
		for (int i = 0; i < max_buffer_frames; ++i) ch.data[i] = Variant();
		channels[p_name] = ch;
	}

	// --- Record a value into a channel at the current frame ---
	void record(const String &p_channel, const Variant &p_value) {
		ERR_FAIL_COND(!channels.has(p_channel));
		DataChannel &ch = channels[p_channel];
		if (current_frame < ch.data.size()) {
			ch.data[current_frame] = p_value;
		} else {
			// Push back if exceeding buffer (unlikely because we resize when frame advances)
			ch.data.push_back(p_value);
		}
	}

	// --- Advance to the next frame, resizing if buffer capacity is reached ---
	void advance_frame() {
		current_frame++;
		if (current_frame >= max_buffer_frames) {
			// Overwrite oldest data (ring buffer not implemented; simply clear and restart index)
			// A full implementation would cycle; for simplicity we keep all frames up to capacity
			// and stop recording when full.
			if (current_frame >= max_buffer_frames) {
				// Option 1: keep appending but ignore max? We'll just increase buffer size.
				max_buffer_frames = current_frame + 100;
				for (KeyValue<String, DataChannel> &kv : channels) {
					kv.value.data.resize(max_buffer_frames);
					for (int i = kv.value.data.size() - 100; i < kv.value.data.size(); ++i)
						kv.value.data[i] = Variant();
				}
			}
		}
	}

	// --- Flush recorded data to disk (implemented by derived classes) ---
	virtual Error flush(const String &p_file_path) = 0;

	// --- Clear all recorded data and reset frame counter ---
	void clear() {
		for (KeyValue<String, DataChannel> &kv : channels) {
			kv.value.data.clear();
			kv.value.data.resize(max_buffer_frames);
			for (int i = 0; i < max_buffer_frames; ++i) kv.value.data[i] = Variant();
		}
		current_frame = 0;
	}

	// --- Access to raw data (for plotting or in-memory analysis) ---
	Dictionary get_all_data() const {
		Dictionary dict;
		for (const KeyValue<String, DataChannel> &kv : channels) {
			Array arr;
			for (int i = 0; i < current_frame; ++i) arr.push_back(kv.value.data[i]);
			dict[kv.key] = arr;
		}
		return dict;
	}

	int get_current_frame() const { return current_frame; }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &BaseRecorder::set_enabled);
		ClassDB::bind_method(D_METHOD("is_enabled"), &BaseRecorder::is_enabled);
		ClassDB::bind_method(D_METHOD("set_max_buffer_frames", "max_frames"), &BaseRecorder::set_max_buffer_frames);
		ClassDB::bind_method(D_METHOD("get_max_buffer_frames"), &BaseRecorder::get_max_buffer_frames);
		ClassDB::bind_method(D_METHOD("register_channel", "name", "type"), &BaseRecorder::register_channel);
		ClassDB::bind_method(D_METHOD("record", "channel", "value"), &BaseRecorder::record);
		ClassDB::bind_method(D_METHOD("advance_frame"), &BaseRecorder::advance_frame);
		ClassDB::bind_method(D_METHOD("flush", "path"), &BaseRecorder::flush);
		ClassDB::bind_method(D_METHOD("clear"), &BaseRecorder::clear);
		ClassDB::bind_method(D_METHOD("get_all_data"), &BaseRecorder::get_all_data);
		ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "max_buffer_frames"), "set_max_buffer_frames", "get_max_buffer_frames");
	}

	struct DataChannel {
		Variant::Type type;
		LocalVector<Variant> data; // per-frame
	};

	HashMap<String, DataChannel> channels;
	HashMap<String, Variant::Type> channel_types;
	bool enabled;
	int max_buffer_frames;
	int current_frame;
};

} // namespace genesis

#endif // GENESIS_RECORDERS_BASE_RECORDER_H