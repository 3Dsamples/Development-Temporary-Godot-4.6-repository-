--- START OF FILE core/io/file_access.h ---

#ifndef FILE_ACCESS_H
#define FILE_ACCESS_H

#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "core/typedefs.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * FileAccess
 * 
 * Abstract base class for bit-perfect file system interaction.
 * Optimized for Universal Solver data streams.
 */
class FileAccess : public RefCounted {
	GDCLASS(FileAccess, RefCounted);

public:
	enum ModeFlags {
		READ = 1,
		WRITE = 2,
		READ_WRITE = 3,
		WRITE_READ = 7,
	};

protected:
	static FileAccess *(*create_func)();
	static void _bind_methods();

public:
	static Ref<FileAccess> open(const String &p_path, int p_mode_flags, Error *r_error = nullptr);
	static bool exists(const String &p_path);

	virtual Error _open(const String &p_path, int p_mode_flags) = 0;
	virtual void close() = 0;
	virtual bool is_open() const = 0;

	// Use BigIntCore for 64-bit offsets to support multi-terabyte galactic datasets
	virtual uint64_t get_position() const = 0;
	virtual void seek(uint64_t p_position) = 0;
	virtual void seek_end(int64_t p_relative_offset = 0) = 0;
	virtual uint64_t get_length() const = 0;
	virtual bool eof_reached() const = 0;

	// ------------------------------------------------------------------------
	// Deterministic Data I/O (Atomic)
	// ------------------------------------------------------------------------

	virtual uint8_t get_8() const = 0;
	virtual uint16_t get_16() const = 0;
	virtual uint32_t get_32() const = 0;
	virtual uint64_t get_64() const = 0;

	virtual void store_8(uint8_t p_dest) = 0;
	virtual void store_16(uint16_t p_dest) = 0;
	virtual void store_32(uint32_t p_dest) = 0;
	virtual void store_64(uint64_t p_dest) = 0;

	virtual void get_buffer(uint8_t *p_dst, uint64_t p_length) const = 0;
	virtual void store_buffer(const uint8_t *p_src, uint64_t p_length) = 0;

	// ------------------------------------------------------------------------
	// Hyper-Simulation Specialized Streaming
	// ------------------------------------------------------------------------

	/**
	 * store_fixed()
	 * Writes FixedMathCore raw bits directly to ensure 100% cross-platform determinism.
	 */
	ET_SIMD_INLINE void store_fixed(const FixedMathCore &p_val) {
		store_64(static_cast<uint64_t>(p_val.get_raw()));
	}

	/**
	 * get_fixed()
	 * Re-inflates bit-perfect FixedMathCore data.
	 */
	ET_SIMD_INLINE FixedMathCore get_fixed() const {
		return FixedMathCore(static_cast<int64_t>(get_64()), true);
	}

	/**
	 * store_bigint()
	 * Serializes BigIntCore by writing its length followed by its base-10^9 chunks.
	 */
	void store_bigint(const BigIntCore &p_val) {
		std::string s = p_val.to_string();
		uint32_t len = static_cast<uint32_t>(s.length());
		store_32(len);
		store_buffer(reinterpret_cast<const uint8_t *>(s.c_str()), len);
	}

	/**
	 * get_bigint()
	 */
	BigIntCore get_bigint() const {
		uint32_t len = get_32();
		if (len == 0) return BigIntCore(0LL);
		Vector<uint8_t> buf;
		buf.resize(len);
		get_buffer(buf.ptrw(), len);
		std::string s(reinterpret_cast<const char *>(buf.ptr()), len);
		return BigIntCore(s);
	}

	/**
	 * store_string()
	 * UTF-8 encoded string storage with 32-bit length prefix.
	 */
	void store_string(const String &p_string) {
		std::string s = p_string.utf8().get_data();
		store_32(static_cast<uint32_t>(s.length()));
		store_buffer(reinterpret_cast<const uint8_t *>(s.c_str()), s.length());
	}

	String get_string() const {
		uint32_t len = get_32();
		if (len == 0) return String();
		Vector<uint8_t> buf;
		buf.resize(len);
		get_buffer(buf.ptrw(), len);
		return String::utf8(reinterpret_cast<const char *>(buf.ptr()), len);
	}

	FileAccess() {}
	virtual ~FileAccess() {}
};

#endif // FILE_ACCESS_H

--- END OF FILE core/io/file_access.h ---
