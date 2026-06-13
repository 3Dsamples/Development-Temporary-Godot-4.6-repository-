// File 143: modules/gaia/src/version_tracker/version_tracker.h
// VersionTracker – stores git commit hash and build metadata for reproducible
// simulation runs. When integrated with Godot's build system, the commit hash
// is injected as a compile‑time define. Otherwise fallback to "unknown".

#ifndef GAIA_VERSION_TRACKER_H
#define GAIA_VERSION_TRACKER_H

#include "core/string/ustring.h"

namespace gaia {

class VersionTracker {
public:
	// The commit hash is set via the SCons build system:
	// env.Append(CPPDEFINES=['GAIA_GIT_HASH="'+git_hash+'"'])
	static String get_git_hash() {
#ifdef GAIA_GIT_HASH
		return String(GAIA_GIT_HASH);
#else
		return "unknown";
#endif
	}

	// Retrieve the Godot engine version for completeness.
	static String get_godot_version() {
		return VERSION_FULL_CONFIG;
	}

	// Build a full identification string.
	static String get_full_version() {
		return vformat("Gaia-%s / Godot %s", get_git_hash(), get_godot_version());
	}
};

} // namespace gaia

#endif // GAIA_VERSION_TRACKER_H