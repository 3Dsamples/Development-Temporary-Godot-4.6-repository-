--- START OF FILE core/io/file_access.cpp ---

#include "core/io/file_access.h"
#include "core/object/class_db.h"
#include "core/os/os.h"

FileAccess *(*FileAccess::create_func)() = nullptr;

/**
 * open()
 * 
 * Static factory method to instantiate the platform-specific FileAccess implementation.
 * Uses BigIntCore compatible 64-bit offsets to handle massive simulation databases.
 */
Ref<FileAccess> FileAccess::open(const String &p_path, int p_mode_flags, Error *r_error) {
	if (!create_func) {
		if (r_error) *r_error = ERR_UNAVAILABLE;
		return Ref<FileAccess>();
	}

	FileAccess *fa = create_func();
	Error err = fa->_open(p_path, p_mode_flags);
	if (r_error) *r_error = err;

	if (err != OK) {
		memdelete(fa);
		return Ref<FileAccess>();
	}

	return Ref<FileAccess>(fa);
}

/**
 * exists()
 * 
 * Verifies the existence of a file on the persistent storage layer.
 */
bool FileAccess::exists(const String &p_path) {
	Ref<FileAccess> fa = open(p_path, READ);
	if (fa.is_null()) return false;
	return true;
}

/**
 * _bind_methods()
 * 
 * Exposes the bit-perfect I/O API to Godot's reflection system.
 * Replaces standard floating-point I/O with FixedMathCore.
 */
void FileAccess::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open", "path", "flags"), &FileAccess::open);
	ClassDB::bind_method(D_METHOD("exists", "path"), &FileAccess::exists);
	ClassDB::bind_method(D_METHOD("close"), &FileAccess::close);
	ClassDB::bind_method(D_METHOD("is_open"), &FileAccess::is_open);

	ClassDB::bind_method(D_METHOD("get_position"), &FileAccess::get_position);
	ClassDB::bind_method(D_METHOD("seek", "position"), &FileAccess::seek);
	ClassDB::bind_method(D_METHOD("get_length"), &FileAccess::get_length);
	ClassDB::bind_method(D_METHOD("eof_reached"), &FileAccess::eof_reached);

	ClassDB::bind_method(D_METHOD("get_8"), &FileAccess::get_8);
	ClassDB::bind_method(D_METHOD("get_16"), &FileAccess::get_16);
	ClassDB::bind_method(D_METHOD("get_32"), &FileAccess::get_32);
	ClassDB::bind_method(D_METHOD("get_64"), &FileAccess::get_64);

	ClassDB::bind_method(D_METHOD("store_8", "value"), &FileAccess::store_8);
	ClassDB::bind_method(D_METHOD("store_16", "value"), &FileAccess::store_16);
	ClassDB::bind_method(D_METHOD("store_32", "value"), &FileAccess::store_32);
	ClassDB::bind_method(D_METHOD("store_64", "value"), &FileAccess::store_64);

	// Hyper-Simulation Specific Bindings
	ClassDB::bind_method(D_METHOD("store_fixed", "value"), &FileAccess::store_fixed);
	ClassDB::bind_method(D_METHOD("get_fixed"), &FileAccess::get_fixed);
	ClassDB::bind_method(D_METHOD("store_bigint", "value"), &FileAccess::store_bigint);
	ClassDB::bind_method(D_METHOD("get_bigint"), &FileAccess::get_bigint);
	ClassDB::bind_method(D_METHOD("store_string", "value"), &FileAccess::store_string);
	ClassDB::bind_method(D_METHOD("get_string"), &FileAccess::get_string);

	BIND_ENUM_CONSTANT(READ);
	BIND_ENUM_CONSTANT(WRITE);
	BIND_ENUM_CONSTANT(READ_WRITE);
	BIND_ENUM_CONSTANT(WRITE_READ);
}

--- END OF FILE core/io/file_access.cpp ---
