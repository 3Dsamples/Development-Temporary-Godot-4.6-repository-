--- START OF FILE src/big_int_core.cpp ---

#include "src/big_int_core.h"
#include <iomanip>
#include <sstream>
#include <algorithm>

/**
 * _trim()
 * 
 * Removes leading zero chunks to maintain canonical representation.
 * Essential for O(1) comparison and efficient Warp kernel sweeps.
 */
_FORCE_INLINE_ void BigIntCore::_trim() {
	while (chunks.size() > 1 && chunks.back() == 0) {
		chunks.pop_back();
	}
	if (chunks.size() == 1 && chunks[0] == 0) {
		is_negative = false;
	}
}

/**
 * _divide_by_2()
 * 
 * High-speed bitwise-style division for use in the exponentiation kernel.
 */
_FORCE_INLINE_ void BigIntCore::_divide_by_2() {
	uint32_t carry = 0;
	for (int i = chunks.size() - 1; i >= 0; i--) {
		uint64_t current = chunks[i] + static_cast<uint64_t>(carry) * BASE;
		chunks.ptrw()[i] = static_cast<uint32_t>(current / 2);
		carry = static_cast<uint32_t>(current % 2);
	}
	_trim();
}

BigIntCore::BigIntCore(int64_t p_value) {
	is_negative = p_value < 0;
	uint64_t v = is_negative ? static_cast<uint64_t>(-p_value) : static_cast<uint64_t>(p_value);
	if (v == 0) {
		chunks.push_back(0);
	} else {
		while (v > 0) {
			chunks.push_back(static_cast<uint32_t>(v % BASE));
			v /= BASE;
		}
	}
}

BigIntCore::BigIntCore(const char *p_str) {
	is_negative = false;
	if (!p_str) {
		chunks.push_back(0);
		return;
	}
	std::string s = p_str;
	if (s.empty()) {
		chunks.push_back(0);
		return;
	}
	if (s[0] == '-') {
		is_negative = true;
		s = s.substr(1);
	}
	for (int i = s.length(); i > 0; i -= 9) {
		if (i < 9) {
			chunks.push_back(static_cast<uint32_t>(std::stoul(s.substr(0, i))));
		} else {
			chunks.push_back(static_cast<uint32_t>(std::stoul(s.substr(i - 9, 9))));
		}
	}
	_trim();
}

BigIntCore &BigIntCore::operator=(const BigIntCore &p_other) {
	chunks = p_other.chunks;
	is_negative = p_other.is_negative;
	return *this;
}

BigIntCore &BigIntCore::operator=(int64_t p_value) {
	*this = BigIntCore(p_value);
	return *this;
}

bool BigIntCore::operator==(const BigIntCore &p_other) const {
	return is_negative == p_other.is_negative && chunks == p_other.chunks;
}

bool BigIntCore::operator<(const BigIntCore &p_other) const {
	if (is_negative != p_other.is_negative) {
		return is_negative;
	}
	if (chunks.size() != p_other.chunks.size()) {
		return (chunks.size() < p_other.chunks.size()) ^ is_negative;
	}
	for (int i = chunks.size() - 1; i >= 0; i--) {
		if (chunks[i] != p_other.chunks[i]) {
			return (chunks[i] < p_other.chunks[i]) ^ is_negative;
		}
	}
	return false;
}

bool BigIntCore::operator<=(const BigIntCore &p_other) const { return *this < p_other || *this == p_other; }
bool BigIntCore::operator>(const BigIntCore &p_other) const { return !(*this <= p_other); }
bool BigIntCore::operator>=(const BigIntCore &p_other) const { return !(*this < p_other); }

BigIntCore BigIntCore::operator+(const BigIntCore &p_other) const {
	if (is_negative != p_other.is_negative) {
		BigIntCore tmp = p_other;
		tmp.is_negative = !tmp.is_negative;
		return *this - tmp;
	}
	BigIntCore res;
	res.is_negative = is_negative;
	res.chunks.clear();
	uint32_t carry = 0;
	int n = std::max(chunks.size(), p_other.chunks.size());
	for (int i = 0; i < n || carry; i++) {
		uint64_t sum = carry + (i < chunks.size() ? chunks[i] : 0) + (i < p_other.chunks.size() ? p_other.chunks[i] : 0);
		res.chunks.push_back(static_cast<uint32_t>(sum % BASE));
		carry = static_cast<uint32_t>(sum / BASE);
	}
	return res;
}

BigIntCore BigIntCore::operator-(const BigIntCore &p_other) const {
	if (is_negative != p_other.is_negative) {
		BigIntCore tmp = p_other;
		tmp.is_negative = !tmp.is_negative;
		return *this + tmp;
	}
	if (absolute() < p_other.absolute()) {
		BigIntCore res = p_other - *this;
		res.is_negative = !is_negative;
		return res;
	}
	BigIntCore res;
	res.is_negative = is_negative;
	res.chunks.clear();
	int32_t borrow = 0;
	for (int i = 0; i < chunks.size(); i++) {
		int64_t sub = static_cast<int64_t>(chunks[i]) - borrow - (i < p_other.chunks.size() ? p_other.chunks[i] : 0);
		if (sub < 0) {
			sub += BASE;
			borrow = 1;
		} else {
			borrow = 0;
		}
		res.chunks.push_back(static_cast<uint32_t>(sub));
	}
	res._trim();
	return res;
}

BigIntCore BigIntCore::operator*(const BigIntCore &p_other) const {
	BigIntCore res;
	res.is_negative = is_negative != p_other.is_negative;
	res.chunks.resize(chunks.size() + p_other.chunks.size());
	res.chunks.fill(0);
	for (int i = 0; i < chunks.size(); i++) {
		uint32_t carry = 0;
		for (int j = 0; j < p_other.chunks.size() || carry; j++) {
			uint64_t cur = res.chunks[i + j] + chunks[i] * 1ULL * (j < p_other.chunks.size() ? p_other.chunks[j] : 0) + carry;
			res.chunks.ptrw()[i + j] = static_cast<uint32_t>(cur % BASE);
			carry = static_cast<uint32_t>(cur / BASE);
		}
	}
	res._trim();
	return res;
}

BigIntCore BigIntCore::operator/(const BigIntCore &p_other) const {
	CRASH_COND_MSG(p_other.is_zero(), "BigIntCore: Division by zero.");
	BigIntCore b = p_other.absolute();
	BigIntCore q, r;
	q.chunks.resize(chunks.size());
	q.chunks.fill(0);
	for (int i = chunks.size() - 1; i >= 0; i--) {
		r = r * BASE;
		r = r + chunks[i];
		uint32_t low = 0, high = BASE - 1, m = 0;
		while (low <= high) {
			uint32_t mid = (low + high) / 2;
			if (b * mid <= r) {
				m = mid;
				low = mid + 1;
			} else {
				high = mid - 1;
			}
		}
		q.chunks.ptrw()[i] = m;
		r = r - b * m;
	}
	q.is_negative = is_negative != p_other.is_negative;
	q._trim();
	return q;
}

BigIntCore BigIntCore::operator%(const BigIntCore &p_other) const {
	CRASH_COND_MSG(p_other.is_zero(), "BigIntCore: Modulo by zero.");
	BigIntCore b = p_other.absolute();
	BigIntCore r;
	for (int i = chunks.size() - 1; i >= 0; i--) {
		r = r * BASE;
		r = r + chunks[i];
		uint32_t low = 0, high = BASE - 1, m = 0;
		while (low <= high) {
			uint32_t mid = (low + high) / 2;
			if (b * mid <= r) {
				m = mid;
				low = mid + 1;
			} else {
				high = mid - 1;
			}
		}
		r = r - b * m;
	}
	r.is_negative = is_negative;
	r._trim();
	return r;
}

BigIntCore &BigIntCore::operator+=(const BigIntCore &p_other) { return *this = *this + p_other; }
BigIntCore &BigIntCore::operator-=(const BigIntCore &p_other) { return *this = *this - p_other; }
BigIntCore &BigIntCore::operator*=(const BigIntCore &p_other) { return *this = *this * p_other; }
BigIntCore &BigIntCore::operator/=(const BigIntCore &p_other) { return *this = *this / p_other; }

BigIntCore BigIntCore::power(const BigIntCore &p_exponent) const {
	BigIntCore res(1);
	BigIntCore b = *this;
	BigIntCore e = p_exponent;
	while (!e.is_zero()) {
		if (e.chunks[0] % 2 == 1) res *= b;
		b *= b;
		e._divide_by_2();
	}
	return res;
}

BigIntCore BigIntCore::square_root() const {
	CRASH_COND_MSG(is_negative, "BigIntCore: Square root of negative.");
	if (is_zero()) return BigIntCore(0);
	BigIntCore low(1), high = *this, res(1);
	while (low <= high) {
		BigIntCore mid = (low + high) / 2;
		if (mid * mid <= *this) {
			res = mid;
			low = mid + 1;
		} else {
			high = mid - 1;
		}
	}
	return res;
}

std::string BigIntCore::to_string() const {
	if (is_zero()) return "0";
	std::ostringstream oss;
	if (is_negative) oss << "-";
	oss << chunks.back();
	for (int i = chunks.size() - 2; i >= 0; i--) {
		oss << std::setfill('0') << std::setw(9) << chunks[i];
	}
	return oss.str();
}

std::string BigIntCore::to_scientific() const {
	std::string s = to_string();
	if (s == "0") return s;
	size_t off = is_negative ? 1 : 0;
	if (s.length() - off <= 6) return s;
	std::string res = (is_negative ? "-" : "") + s.substr(off, 1) + "." + s.substr(off + 1, 2) + "e" + std::to_string(s.length() - off - 1);
	return res;
}

std::string BigIntCore::to_aa_notation() const {
	std::string s = to_string();
	size_t off = is_negative ? 1 : 0;
	size_t len = s.length() - off;
	if (len <= 3) return s;
	size_t exp = (len - 1) / 3;
	std::string prefix = s.substr(off, len - exp * 3);
	if (prefix.length() < 3) prefix += "." + s.substr(off + prefix.length(), 3 - prefix.length());
	static const char *suffixes[] = { "", "k", "m", "b", "t" };
	if (exp < 5) return (is_negative ? "-" : "") + prefix + suffixes[exp];
	int aa = exp - 5;
	char c1 = 'a' + (aa / 26);
	char c2 = 'a' + (aa % 26);
	return (is_negative ? "-" : "") + prefix + c1 + c2;
}

std::string BigIntCore::to_metric_symbol() const {
	static const char *sym[] = { "", "k", "M", "G", "T", "P", "E", "Z", "Y" };
	std::string s = to_string();
	size_t off = is_negative ? 1 : 0;
	size_t exp = (s.length() - off - 1) / 3;
	if (exp < 9) {
		std::string pre = s.substr(off, s.length() - off - exp * 3);
		return (is_negative ? "-" : "") + pre + sym[exp];
	}
	return to_scientific();
}

std::string BigIntCore::to_metric_name() const {
	static const char *names[] = { "", " kilo", " mega", " giga", " tera", " peta", " exa", " zetta", " yotta" };
	std::string s = to_string();
	size_t off = is_negative ? 1 : 0;
	size_t exp = (s.length() - off - 1) / 3;
	if (exp < 9) {
		std::string pre = s.substr(off, s.length() - off - exp * 3);
		return (is_negative ? "-" : "") + pre + names[exp];
	}
	return to_scientific();
}

--- END OF FILE src/big_int_core.cpp ---
