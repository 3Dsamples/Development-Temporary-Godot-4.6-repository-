// File 14: modules/gaia/src/io/io.h

#ifndef GAIA_IO_IO_H
#define GAIA_IO_IO_H

#include "core/io/file_access.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

namespace gaia::io {

// ---------------------------------------------------------------------------
// Low‑level binary I/O helpers, wrapping Godot’s FileAccess.
// Used to serialise simulation state, mesh data and parameters.
// ---------------------------------------------------------------------------

// Write a single value of type T (must have fixed size).
template <typename T>
void write_binary(Ref<FileAccess> f, const T &value) {
	ERR_FAIL_COND(f.is_null());
	f->store_buffer((const uint8_t *)&value, sizeof(T));
}

// Read a single value of type T.
template <typename T>
T read_binary(Ref<FileAccess> f) {
	ERR_FAIL_COND_V(f.is_null(), T());
	T value;
	f->get_buffer((uint8_t *)&value, sizeof(T));
	return value;
}

// Convenience overloads for common types.
inline void write_uint32(Ref<FileAccess> f, uint32_t v) { write_binary(f, v); }
inline uint32_t read_uint32(Ref<FileAccess> f) { return read_binary<uint32_t>(f); }

inline void write_real(Ref<FileAccess> f, real_t v) { write_binary(f, v); }
inline real_t read_real(Ref<FileAccess> f) { return read_binary<real_t>(f); }

inline void write_vector3(Ref<FileAccess> f, const Vector3 &v) {
	write_real(f, v.x);
	write_real(f, v.y);
	write_real(f, v.z);
}
inline Vector3 read_vector3(Ref<FileAccess> f) {
	Vector3 v;
	v.x = read_real(f);
	v.y = read_real(f);
	v.z = read_real(f);
	return v;
}

// Write / read a zero‑terminated ASCII string.
inline void write_string(Ref<FileAccess> f, const String &s) {
	CharString cs = s.utf8();
	f->store_buffer((const uint8_t *)cs.ptr(), cs.length() + 1); // include null terminator
}
inline String read_string(Ref<FileAccess> f) {
	String s;
	char ch;
	while ((ch = f->get_8()) != '\0') {
		s += ch;
	}
	return s;
}

// Read entire file content as a String.
inline String read_file_as_string(const String &path) {
	Ref<FileAccess> f = FileAccess::open(path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(f.is_null(), String(), "Cannot open file: " + path);
	String content;
	while (!f->eof_reached()) {
		content += f->get_line() + "\n";
	}
	return content;
}

// Write a string to a file, overwriting existing content.
inline void write_string_to_file(const String &path, const String &content) {
	Ref<FileAccess> f = FileAccess::open(path, FileAccess::WRITE);
	ERR_FAIL_COND_MSG(f.is_null(), "Cannot open file for writing: " + path);
	f->store_string(content);
}

} // namespace gaia::io

#endif // GAIA_IO_IO_H