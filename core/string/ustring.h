--- START OF FILE core/string/ustring.h ---

#ifndef USTRING_H
#define USTRING_H

#include "core/typedefs.h"
#include "core/templates/cowdata.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * String (UString)
 * 
 * Unicode string class with COW (Copy-On-Write) safety.
 * Optimized for high-frequency simulation UI updates.
 * Strictly utilizes FixedMathCore and BigIntCore for all numeric string logic.
 */
class ET_ALIGN_32 String {
	struct StringData {
		SafeRefCount refcount;
		uint32_t length;
		char32_t *data;
	};

	CowData<char32_t> _cowdata;

public:
	// ------------------------------------------------------------------------
	// Constructors & Destructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE String() {}
	String(const String &p_from);
	String(const char *p_contents);
	String(const char32_t *p_contents);
	
	// Simulation Type Constructors
	String(const BigIntCore &p_big_int);
	String(const FixedMathCore &p_fixed);
	String(int64_t p_int);

	~String();

	// ------------------------------------------------------------------------
	// Static Converters (Scale-Aware UI)
	// ------------------------------------------------------------------------
	static String num_bigint(const BigIntCore &p_num);
	static String num_fixed(const FixedMathCore &p_num, int p_decimals = 6);
	static String num_int64(int64_t p_num, int p_base = 10);
	
	// ------------------------------------------------------------------------
	// Logic & Accessors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ int length() const { return _cowdata.size(); }
	_FORCE_INLINE_ bool is_empty() const { return _cowdata.size() == 0; }

	char32_t operator[](int p_idx) const;
	char32_t &operator[](int p_idx);

	const char32_t *ptr() const { return _cowdata.ptr(); }
	char32_t *ptrw() { return _cowdata.ptrw(); }

	// ------------------------------------------------------------------------
	// Operators
	// ------------------------------------------------------------------------
	void operator=(const String &p_from);
	bool operator==(const String &p_other) const;
	bool operator!=(const String &p_other) const;
	bool operator<(const String &p_other) const;

	String operator+(const String &p_other) const;
	void operator+=(const String &p_other);
	void operator+=(char32_t p_char);

	// ------------------------------------------------------------------------
	// Search & Manipulation
	// ------------------------------------------------------------------------
	bool begins_with(const String &p_string) const;
	bool ends_with(const String &p_string) const;
	bool contains(const String &p_string) const;
	
	String to_lower() const;
	String to_upper() const;
	String strip_edges() const;
	String get_extension() const;
	String get_basename() const;

	// ------------------------------------------------------------------------
	// Simulation Data Extraction
	// ------------------------------------------------------------------------
	int64_t to_int() const;
	BigIntCore to_bigint() const;
	FixedMathCore to_fixed() const;

	// ------------------------------------------------------------------------
	// UTF-8 Bridge (For OS and Logging)
	// ------------------------------------------------------------------------
	struct CharString {
		Vector<char> data;
		_FORCE_INLINE_ const char *get_data() const { return data.ptr(); }
		_FORCE_INLINE_ int length() const { return data.size() - 1; }
	};

	CharString utf8() const;
	static String utf8(const char *p_utf8, int p_len = -1);

	// Hashing for ClassDB and EnTT registry keys
	uint32_t hash() const;
};

// Global operators for UI assembly
inline String operator+(const char *p_left, const String &p_right) { return String(p_left) + p_right; }

#endif // USTRING_H

--- END OF FILE core/string/ustring.h ---
