// genesis/version.h

#pragma once

#include <string>

namespace genesis {

//------------------------------------------------------------------------------
// Version information for the Genesis library.
//------------------------------------------------------------------------------

// Major version: breaking changes
constexpr int VERSION_MAJOR = 0;

// Minor version: new features, non-breaking
constexpr int VERSION_MINOR = 4;

// Patch version: bug fixes
constexpr int VERSION_PATCH = 6;

// Full version string
constexpr const char* VERSION = "0.4.6";

// Version as a packed integer (e.g., 0x000406 for 0.4.6)
constexpr int VERSION_INT = (VERSION_MAJOR << 16) | (VERSION_MINOR << 8) | VERSION_PATCH;

// Git commit hash (set during build)
extern const std::string GIT_COMMIT_HASH;

// Build date and time (set during build)
extern const std::string BUILD_DATE;
extern const std::string BUILD_TIME;

// Full version description including build info
std::string full_version_string();

// Check if version is at least the given major.minor.patch
constexpr bool version_at_least(int major, int minor, int patch) {
    return VERSION_INT >= ((major << 16) | (minor << 8) | patch);
}

} // namespace genesis