--- START OF FILE core/string/ustring.cpp ---

#include "core/string/ustring.h"
#include "core/os/memory.h"
#include "core/math/math_funcs.h"
#include <cstring>

// ============================================================================
// Constructors & Destructors
// ============================================================================

String::String(const String &p_from) {
	_cowdata = p_from._cowdata;
}

String::String(const char *p_contents) {
	if (!p_contents) return;
	int len = 0;
	while (p_contents[len]) len++;
	if (len == 0) return;

	_cowdata.resize(len);
	char32_t *w = _cowdata.ptrw();
	for (int i = 0; i < len; i++) {
		w[i] = (char32_t)(uint8_t)p_contents[i];
	}
}

String::String(const char32_t *p_contents) {
	if (!p_contents) return;
	int len = 0;
	while (p_contents[len]) len++;
	if (len == 0) return;

	_cowdata.resize(len);
	char32_t *w = _cowdata.ptrw();
	memcpy(w, p_contents, len * sizeof(char32_t));
}

String::String(const BigIntCore &p_big_int) {
	*this = String(p_big_int.to_string().c_str());
}

String::String(const FixedMathCore &p_fixed) {
	*this = String(p_fixed.to_string().c_str());
}

String::String(int64_t p_int) {
	*this = num_int64(p_int);
}

String::~String() {
	// CowData handles its own unreferencing
}

// ============================================================================
// Static Converters (Deterministic - No Floats)
// ============================================================================

String String::num_bigint(const BigIntCore &p_num) {
	return String(p_num.to_string().c_str());
}

String String::num_fixed(const FixedMathCore &p_num, int p_decimals) {
	// FixedMathCore::to_string() is already bit-perfect and FPU-free
	return String(p_num.to_string().c_str());
}

String String::num_int64(int64_t p_num, int p_base) {
	if (p_num == 0) return String("0");
	char buf[65];
	int i = 64;
	buf[i--] = 0;
	bool neg = p_num < 0;
	uint64_t n = neg ? (uint64_t)-p_num : (uint64_t)p_num;
	while (n > 0) {
		uint32_t digit = n % p_base;
		buf[i--] = (digit < 10 ? '0' + digit : 'a' + digit - 10);
		n /= p_base;
	}
	if (neg) buf[i--] = '-';
	return String(&buf[i + 1]);
}

// ============================================================================
// Core Logic & Manipulation
// ============================================================================

char32_t String::operator[](int p_idx) const {
	CRASH_BAD_INDEX(p_idx, length());
	return ptr()[p_idx];
}

char32_t &String::operator[](int p_idx) {
	CRASH_BAD_INDEX(p_idx, length());
	return ptrw()[p_idx];
}

void String::operator=(const String &p_from) {
	_cowdata = p_from._cowdata;
}

bool String::operator==(const String &p_other) const {
	if (_cowdata.ptr() == p_other._cowdata.ptr()) return true;
	if (length() != p_other.length()) return false;
	return memcmp(ptr(), p_other.ptr(), length() * sizeof(char32_t)) == 0;
}

bool String::operator!=(const String &p_other) const {
	return !(*this == p_other);
}

String String::operator+(const String &p_other) const {
	String s = *this;
	s += p_other;
	return s;
}

void String::operator+=(const String &p_other) {
	int l = length();
	int ol = p_other.length();
	if (ol == 0) return;
	_cowdata.resize(l + ol);
	char32_t *w = _cowdata.ptrw();
	memcpy(w + l, p_other.ptr(), ol * sizeof(char32_t));
}

void String::operator+=(char32_t p_char) {
	int l = length();
	_cowdata.resize(l + 1);
	_cowdata.ptrw()[l] = p_char;
}

// ============================================================================
// Simulation Data Extraction
// ============================================================================

int64_t String::to_int() const {
	CharString cs = utf8();
	return (int64_t)std::atoll(cs.get_data());
}

BigIntCore String::to_bigint() const {
	CharString cs = utf8();
	return BigIntCore(cs.get_data());
}

FixedMathCore String::to_fixed() const {
	CharString cs = utf8();
	return FixedMathCore(std::string(cs.get_data()));
}

// ============================================================================
// UTF-8 Bridge
// ============================================================================

String::CharString String::utf8() const {
	CharString cs;
	int len = length();
	if (len == 0) {
		cs.data.push_back(0);
		return cs;
	}
	const char32_t *p = ptr();
	// Simplified UTF-8 encoding for core simulation logs
	for (int i = 0; i < len; i++) {
		char32_t c = p[i];
		if (c < 0x80) {
			cs.data.push_back((char)c);
		} else if (c < 0x800) {
			cs.data.push_back((char)(0xC0 | (c >> 6)));
			cs.data.push_back((char)(0x80 | (c & 0x3F)));
		} else {
			cs.data.push_back((char)(0xE0 | (c >> 12)));
			cs.data.push_back((char)(0x80 | ((c >> 6) & 0x3F)));
			cs.data.push_back((char)(0x80 | (c & 0x3F)));
		}
	}
	cs.data.push_back(0);
	return cs;
}

String String::utf8(const char *p_utf8, int p_len) {
	if (!p_utf8) return String();
	int len = (p_len < 0) ? (int)strlen(p_utf8) : p_len;
	if (len == 0) return String();

	String s;
	s._cowdata.resize(len); // Upper bound
	char32_t *w = s._cowdata.ptrw();
	int actual_len = 0;

	for (int i = 0; i < len; i++) {
		uint8_t c = (uint8_t)p_utf8[i];
		if (c < 0x80) {
			w[actual_len++] = c;
		} else if ((c & 0xE0) == 0xC0) {
			w[actual_len++] = ((c & 0x1F) << 6) | (p_utf8[++i] & 0x3F);
		} else if ((c & 0xF0) == 0xE0) {
			w[actual_len++] = ((c & 0x0F) << 12) | ((p_utf8[i + 1] & 0x3F) << 6) | (p_utf8[i + 2] & 0x3F);
			i += 2;
		}
	}
	s._cowdata.resize(actual_len);
	return s;
}

uint32_t String::hash() const {
	int len = length();
	if (len == 0) return 0;
	// DJB2 Hash implementation for 120 FPS StringName interning
	uint32_t hash = 5381;
	const char32_t *p = ptr();
	for (int i = 0; i < len; i++) {
		hash = ((hash << 5) + hash) + p[i];
	}
	return hash;
}

--- END OF FILE core/string/ustring.cpp ---
