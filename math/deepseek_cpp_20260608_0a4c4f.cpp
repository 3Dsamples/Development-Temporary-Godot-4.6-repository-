// File 41: modules/gaia/src/utility/logger.h

#ifndef GAIA_UTILITY_LOGGER_H
#define GAIA_UTILITY_LOGGER_H

#include "core/print_string.h"   // print_line, print_verbose, WARN_PRINT, ERR_PRINT

namespace gaia::logger {

/**
 * Simple logging utility that wraps Godot's built-in print macros.
 * Replaces Gaia's original debug print routines.
 */
inline void info(const String &p_message) {
	print_line(String("[Gaia][INFO] ") + p_message);
}

inline void debug(const String &p_message) {
	print_verbose(String("[Gaia][DEBUG] ") + p_message);
}

inline void warn(const String &p_message) {
	WARN_PRINT(String("[Gaia][WARN] ") + p_message);
}

inline void error(const String &p_message) {
	ERR_PRINT(String("[Gaia][ERROR] ") + p_message);
}

} // namespace gaia::logger

#endif // GAIA_UTILITY_LOGGER_H