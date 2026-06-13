// File 134: modules/genesis/src/recorders/recorder_manager.h
// RecorderManager – owns a collection of BaseRecorder instances and advances
// them each frame. It provides a single flush_all() to write all recorders
// to disk.

#ifndef GENESIS_RECORDERS_RECORDER_MANAGER_H
#define GENESIS_RECORDERS_RECORDER_MANAGER_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "base_recorder.h"

namespace genesis {

class RecorderManager : public RefCounted {
	GDCLASS(RecorderManager, RefCounted);

public:
	RecorderManager() {}

	// Add a recorder. Returns its index.
	int add_recorder(const Ref<BaseRecorder> &p_recorder) {
		ERR_FAIL_COND_V(p_recorder.is_null(), -1);
		recorders.push_back(p_recorder);
		return recorders.size() - 1;
	}

	// Remove a recorder by index.
	void remove_recorder(int p_idx) {
		ERR_FAIL_INDEX(p_idx, recorders.size());
		recorders.remove_at(p_idx);
	}

	// Clear all recorders.
	void clear() { recorders.clear(); }

	// Advance all recorders one frame.
	void advance_frame() {
		for (Ref<BaseRecorder> &rec : recorders) {
			if (rec.is_valid() && rec->is_enabled()) rec->advance_frame();
		}
	}

	// Flush all recorders to a directory (each recorder appends its own file name).
	Error flush_all(const String &p_directory) {
		for (Ref<BaseRecorder> &rec : recorders) {
			if (rec.is_valid() && rec->is_enabled()) {
				// Build path: directory / "recorder_<index>.csv" (or appropriate)
				String name = vformat("recorder_%d", rec->get_instance_id());
				String path = p_directory.path_join(name);
				Error err = rec->flush(path);
				if (err != OK) return err;
			}
		}
		return OK;
	}

	// Return a recorder by index.
	Ref<BaseRecorder> get_recorder(int p_idx) const {
		ERR_FAIL_INDEX_V(p_idx, recorders.size(), Ref<BaseRecorder>());
		return recorders[p_idx];
	}

	int get_recorder_count() const { return recorders.size(); }

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("add_recorder", "recorder"), &RecorderManager::add_recorder);
		ClassDB::bind_method(D_METHOD("remove_recorder", "index"), &RecorderManager::remove_recorder);
		ClassDB::bind_method(D_METHOD("clear"), &RecorderManager::clear);
		ClassDB::bind_method(D_METHOD("advance_frame"), &RecorderManager::advance_frame);
		ClassDB::bind_method(D_METHOD("flush_all", "directory"), &RecorderManager::flush_all);
		ClassDB::bind_method(D_METHOD("get_recorder", "index"), &RecorderManager::get_recorder);
		ClassDB::bind_method(D_METHOD("get_recorder_count"), &RecorderManager::get_recorder_count);
	}

private:
	LocalVector<Ref<BaseRecorder>> recorders;
};

} // namespace genesis

#endif // GENESIS_RECORDERS_RECORDER_MANAGER_H