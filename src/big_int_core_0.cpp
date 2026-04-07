--- START OF FILE src/big_int_core.cpp ---

#include "big_int_core.h"
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// Internal SIMD/Warp Cache Alignment and Trimming
// ============================================================================

ET_SIMD_INLINE void BigIntCore::trim() {
    while (chunks.size() > 1 && chunks.back() == 0) {
        chunks.pop_back();
    }
    if (chunks.size() == 1 && chunks[0] == 0) {
        is_negative = false;
    }
}

ET_SIMD_INLINE void BigIntCore::divide_by_2() {
    uint32_t carry = 0;
    for (int i = (int)chunks.size() - 1; i >= 0; --i) {
        uint64_t current = chunks[i] + carry * 1ULL * BASE;
        chunks[i] = current / 2;
        carry = current % 2;
    }
    trim();
}

// ============================================================================
// Constructors (Warp-Kernel Friendly)
// ============================================================================

ET_SIMD_INLINE BigIntCore::BigIntCore() : is_negative(false) {
    chunks.push_back(0);
}

ET_SIMD_INLINE BigIntCore::BigIntCore(int64_t value) {
    is_negative = value < 0;
    uint64_t abs_val = is_negative ? -value : value;
    if (abs_val == 0) {
        chunks.push_back(0);
    } else {
        while (abs_val > 0) {
            chunks.push_back(abs_val % BASE);
            abs_val /= BASE;
        }
    }
}

BigIntCore::BigIntCore(const std::string& value) {
    is_negative = false;
    std::string str = value;
    if (str.empty()) {
        chunks.push_back(0);
        return;
    }
    if (str[0] == '-') {
        is_negative = true;
        str = str.substr(1);
    } else if (str[0] == '+') {
        str = str.substr(1);
    }

    if (str.empty() || str.find_first_not_of("0123456789") != std::string::npos) {
        chunks.push_back(0);
        is_negative = false;
        return;
    }

    // Process blocks of 9 digits for Base 10^9 storage optimization
    for (int i = (int)str.length(); i > 0; i -= 9) {
        if (i < 9) {
            chunks.push_back(std::stoi(str.substr(0, i)));
        } else {
            chunks.push_back(std::stoi(str.substr(i - 9, 9)));
        }
    }
    trim();
}

ET_SIMD_INLINE BigIntCore::BigIntCore(const BigIntCore& other) {
    chunks = other.chunks;
    is_negative = other.is_negative;
}

// ============================================================================
// Assignment
// ============================================================================

ET_SIMD_INLINE BigIntCore& BigIntCore::operator=(const BigIntCore& other) {
    if (this != &other) {
        chunks = other.chunks;
        is_negative = other.is_negative;
    }
    return *this;
}

ET_SIMD_INLINE BigIntCore& BigIntCore::operator=(int64_t value) {
    *this = BigIntCore(value);
    return *this;
}

BigIntCore& BigIntCore::operator=(const std::string& value) {
    *this = BigIntCore(value);
    return *this;
}

// ============================================================================
// Comparison Operators (Constant time logical checks for EnTT view filtering)
// ============================================================================

bool BigIntCore::operator==(const BigIntCore& other) const {
    return is_negative == other.is_negative && chunks == other.chunks;
}

bool BigIntCore::operator!=(const BigIntCore& other) const {
    return !(*this == other);
}

bool BigIntCore::operator<(const BigIntCore& other) const {
    if (is_negative != other.is_negative) return is_negative;
    if (chunks.size() != other.chunks.size()) {
        return (chunks.size() < other.chunks.size()) ^ is_negative;
    }
    for (int i = (int)chunks.size() - 1; i >= 0; --i) {
        if (chunks[i] != other.chunks[i]) {
            return (chunks[i] < other.chunks[i]) ^ is_negative;
        }
    }
    return false;
}

bool BigIntCore::operator<=(const BigIntCore& other) const {
    return *this < other || *this == other;
}

bool BigIntCore::operator>(const BigIntCore& other) const {
    return !(*this <= other);
}

bool BigIntCore::operator>=(const BigIntCore& other) const {
    return !(*this < other);
}

// ============================================================================
// Core Arithmetic (Zero-Copy and Deterministic)
// ============================================================================

BigIntCore BigIntCore::operator+(const BigIntCore& other) const {
    if (is_negative == other.is_negative) {
        BigIntCore res;
        res.is_negative = is_negative;
        res.chunks.clear();
        uint32_t carry = 0;
        size_t max_size = std::max(chunks.size(), other.chunks.size());
        for (size_t i = 0; i < max_size || carry; ++i) {
            uint64_t sum = carry;
            if (i < chunks.size()) sum += chunks[i];
            if (i < other.chunks.size()) sum += other.chunks[i];
            res.chunks.push_back(sum % BASE);
            carry = sum / BASE;
        }
        res.trim();
        return res;
    } else {
        BigIntCore abs_other = other;
        abs_other.is_negative = !abs_other.is_negative;
        return *this - abs_other;
    }
}

BigIntCore BigIntCore::operator-(const BigIntCore& other) const {
    if (is_negative != other.is_negative) {
        BigIntCore abs_other = other;
        abs_other.is_negative = !abs_other.is_negative;
        return *this + abs_other;
    }

    const BigIntCore* larger = this;
    const BigIntCore* smaller = &other;
    bool res_negative = is_negative;

    BigIntCore abs_this = *this;
    abs_this.is_negative = false;
    BigIntCore abs_other = other;
    abs_other.is_negative = false;

    if (abs_this < abs_other) {
        larger = &other;
        smaller = this;
        res_negative = !is_negative;
    }

    BigIntCore res;
    res.is_negative = res_negative;
    res.chunks.clear();
    int32_t borrow = 0;

    for (size_t i = 0; i < larger->chunks.size(); ++i) {
        int64_t diff = larger->chunks[i] - borrow;
        if (i < smaller->chunks.size()) diff -= smaller->chunks[i];
        if (diff < 0) {
            diff += BASE;
            borrow = 1;
        } else {
            borrow = 0;
        }
        res.chunks.push_back((uint32_t)diff);
    }
    res.trim();
    return res;
}

BigIntCore BigIntCore::operator*(const BigIntCore& other) const {
    BigIntCore res;
    res.chunks.assign(chunks.size() + other.chunks.size(), 0);
    res.is_negative = is_negative != other.is_negative;

    for (size_t i = 0; i < chunks.size(); ++i) {
        uint64_t carry = 0;
        for (size_t j = 0; j < other.chunks.size() || carry; ++j) {
            uint64_t current = res.chunks[i + j] + chunks[i] * 1ULL * (j < other.chunks.size() ? other.chunks[j] : 0) + carry;
            res.chunks[i + j] = current % BASE;
            carry = current / BASE;
        }
    }
    res.trim();
    return res;
}

BigIntCore BigIntCore::operator/(const BigIntCore& other) const {
    if (other.is_zero()) {
        throw std::runtime_error("UniversalSolver Error: Division by zero in BigIntCore.");
    }

    BigIntCore abs_this = absolute();
    BigIntCore abs_other = other.absolute();

    if (abs_this < abs_other) {
        return BigIntCore(0);
    }

    BigIntCore res;
    res.chunks.clear();
    BigIntCore current;
    
    // Chunk-wise binary search division ensuring absolute determinism
    for (int i = (int)chunks.size() - 1; i >= 0; --i) {
        current.chunks.insert(current.chunks.begin(), chunks[i]);
        current.trim();
        
        uint32_t left = 0, right = BASE - 1, chunk_res = 0;
        while (left <= right) {
            uint32_t mid = left + (right - left) / 2;
            BigIntCore mid_val(mid);
            if ((abs_other * mid_val) <= current) {
                chunk_res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        res.chunks.insert(res.chunks.begin(), chunk_res);
        current = current - (abs_other * BigIntCore(chunk_res));
    }
    res.is_negative = is_negative != other.is_negative;
    res.trim();
    return res;
}

BigIntCore BigIntCore::operator%(const BigIntCore& other) const {
    if (other.is_zero()) {
        throw std::runtime_error("UniversalSolver Error: Modulo by zero in BigIntCore.");
    }
    BigIntCore res = *this - ((*this / other) * other);
    if (res.is_negative && !res.is_zero()) {
        res = res + other.absolute();
    }
    return res;
}

// ============================================================================
// Compound Assignment
// ============================================================================

ET_SIMD_INLINE BigIntCore& BigIntCore::operator+=(const BigIntCore& other) { *this = *this + other; return *this; }
ET_SIMD_INLINE BigIntCore& BigIntCore::operator-=(const BigIntCore& other) { *this = *this - other; return *this; }
ET_SIMD_INLINE BigIntCore& BigIntCore::operator*=(const BigIntCore& other) { *this = *this * other; return *this; }
ET_SIMD_INLINE BigIntCore& BigIntCore::operator/=(const BigIntCore& other) { *this = *this / other; return *this; }
ET_SIMD_INLINE BigIntCore& BigIntCore::operator%=(const BigIntCore& other) { *this = *this % other; return *this; }

// ============================================================================
// Advanced Deterministic Math
// ============================================================================

BigIntCore BigIntCore::absolute() const {
    BigIntCore res = *this;
    res.is_negative = false;
    return res;
}

BigIntCore BigIntCore::power(const BigIntCore& exponent) const {
    BigIntCore res(1);
    BigIntCore base = *this;
    BigIntCore exp = exponent;

    // Exponentiation by squaring
    while (!exp.is_zero()) {
        if (exp.chunks[0] % 2 == 1) {
            res = res * base;
        }
        base = base * base;
        exp.divide_by_2();
    }
    return res;
}

BigIntCore BigIntCore::square_root() const {
    if (is_negative) throw std::runtime_error("UniversalSolver Error: Square root of negative number in BigIntCore.");
    if (is_zero()) return BigIntCore(0);

    BigIntCore left(1), right = *this, ans(1);
    while (left <= right) {
        BigIntCore mid = (left + right) / BigIntCore(2);
        if (mid * mid <= *this) {
            ans = mid;
            left = mid + BigIntCore(1);
        } else {
            right = mid - BigIntCore(1);
        }
    }
    return ans;
}

// ============================================================================
// Hashing (For EnTT Sparse Set Indexing & ECS Integration)
// ============================================================================

uint32_t BigIntCore::hash() const {
    // High-entropy DJB2-style hash on internal chunks for quick ECS lookup
    uint32_t h = 5381;
    h = ((h << 5) + h) + (is_negative ? 1 : 0);
    for (size_t i = 0; i < chunks.size(); ++i) {
        h = ((h << 5) + h) + chunks[i];
    }
    return h;
}

// ============================================================================
// Utility
// ============================================================================

ET_SIMD_INLINE bool BigIntCore::is_zero() const {
    return chunks.size() == 1 && chunks[0] == 0;
}

ET_SIMD_INLINE int BigIntCore::sign() const {
    if (is_zero()) return 0;
    return is_negative ? -1 : 1;
}

// ============================================================================
// String Formatting (For Idle/Incremental Games UI)
// ============================================================================

std::string BigIntCore::to_string() const {
    if (is_zero()) return "0";
    std::ostringstream oss;
    if (is_negative) oss << "-";
    oss << chunks.back();
    for (int i = (int)chunks.size() - 2; i >= 0; --i) {
        oss << std::setfill('0') << std::setw(9) << chunks[i];
    }
    return oss.str();
}

std::string BigIntCore::to_scientific() const {
    std::string full = to_string();
    if (full == "0") return "0";
    
    size_t offset = is_negative ? 1 : 0;
    size_t length = full.length() - offset;
    
    if (length <= 5) return full;
    
    std::string coeff = full.substr(offset, 1) + "." + full.substr(offset + 1, 2);
    return (is_negative ? "-" : "") + coeff + "e" + std::to_string(length - 1);
}

std::string BigIntCore::to_aa_notation() const {
    std::string full = to_string();
    if (full == "0") return "0";

    size_t offset = is_negative ? 1 : 0;
    size_t total_digits = full.length() - offset;
    
    if (total_digits < 4) return full;
    
    size_t exponent = total_digits - 1;
    size_t power_of_thousand = exponent / 3;
    
    std::string prefix;
    size_t display_digits = total_digits - (power_of_thousand * 3);
    prefix = full.substr(offset, display_digits);
    if (full.length() > offset + display_digits + 2) {
        prefix += "." + full.substr(offset + display_digits, 2);
    }
    
    std::string suffix = "";
    if (power_of_thousand == 1) suffix = "K";
    else if (power_of_thousand == 2) suffix = "M";
    else if (power_of_thousand == 3) suffix = "B";
    else if (power_of_thousand == 4) suffix = "T";
    else {
        size_t aa_index = power_of_thousand - 5; // 5th power of thousand starts 'aa'
        char first_char = 'a' + (aa_index / 26);
        char second_char = 'a' + (aa_index % 26);
        suffix = std::string(1, first_char) + std::string(1, second_char);
    }
    
    return (is_negative ? "-" : "") + prefix + suffix;
}

std::string BigIntCore::to_metric_symbol() const {
    std::string full = to_string();
    size_t offset = is_negative ? 1 : 0;
    size_t total_digits = full.length() - offset;
    if (total_digits < 4) return full;

    size_t power_of_thousand = (total_digits - 1) / 3;
    std::string symbols[] = {"", "k", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q"};
    
    std::string prefix = full.substr(offset, total_digits - (power_of_thousand * 3));
    prefix += "." + full.substr(offset + prefix.length(), 2);
    
    std::string suffix = (power_of_thousand < 11) ? symbols[power_of_thousand] : "e" + std::to_string(power_of_thousand * 3);
    return (is_negative ? "-" : "") + prefix + suffix;
}

std::string BigIntCore::to_metric_name() const {
    std::string full = to_string();
    size_t offset = is_negative ? 1 : 0;
    size_t total_digits = full.length() - offset;
    if (total_digits < 4) return full;

    size_t power_of_thousand = (total_digits - 1) / 3;
    std::string names[] = {"", "kilo", "mega", "giga", "tera", "peta", "exa", "zetta", "yotta", "ronna", "quetta"};
    
    std::string prefix = full.substr(offset, total_digits - (power_of_thousand * 3));
    prefix += "." + full.substr(offset + prefix.length(), 2);
    
    std::string suffix = (power_of_thousand < 11) ? " " + names[power_of_thousand] : "e" + std::to_string(power_of_thousand * 3);
    return (is_negative ? "-" : "") + prefix + suffix;
}

--- END OF FILE src/big_int_core.cpp ---
