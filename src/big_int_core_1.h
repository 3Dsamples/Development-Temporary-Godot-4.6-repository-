--- START OF FILE src/big_int_core.h ---

#ifndef BIG_INT_CORE_H
#define BIG_INT_CORE_H

#include "core/typedefs.h"
#include "core/templates/vector.h"

/**
 * BigIntCore
 * 
 * The foundational arbitrary-precision integer for the Scale-Aware pipeline.
 * Engineered for EnTT Sparse-Set integration (Who) and Warp-Kernel execution (How Fast).
 * 
 * Features:
 * - Base-10^9 optimized chunking.
 * - SIMD-aligned to 32 bytes for cache-efficiency.
 * - Deterministic DJB2/Murmur-hybrid hashing for EnTT indexing.
 * - Zero-copy memory access for Warp batch math.
 */
struct ET_ALIGN_32 BigIntCore {
private:
	// Contiguous chunk storage compatible with ETEngine memory telemetry
	Vector<uint32_t> chunks;
	bool is_negative;
	static const uint32_t BASE = 1000000000;

	_FORCE_INLINE_ void _trim();
	_FORCE_INLINE_ void _divide_by_2();

public:
	// ------------------------------------------------------------------------
	// Constructors (Bit-Perfect Initialization)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ BigIntCore() : is_negative(false) { chunks.push_back(0); }
	_FORCE_INLINE_ BigIntCore(int64_t p_value);
	BigIntCore(const char *p_str);
	_FORCE_INLINE_ BigIntCore(const BigIntCore &p_other) : chunks(p_other.chunks), is_negative(p_other.is_negative) {}

	// ------------------------------------------------------------------------
	// Assignment API
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ BigIntCore &operator=(const BigIntCore &p_other);
	_FORCE_INLINE_ BigIntCore &operator=(int64_t p_value);

	// ------------------------------------------------------------------------
	// Arithmetic Operators (Zero-Copy Data Flow)
	// ------------------------------------------------------------------------
	BigIntCore operator+(const BigIntCore &p_other) const;
	BigIntCore operator-(const BigIntCore &p_other) const;
	BigIntCore operator*(const BigIntCore &p_other) const;
	BigIntCore operator/(const BigIntCore &p_other) const;
	BigIntCore operator%(const BigIntCore &p_other) const;

	_FORCE_INLINE_ BigIntCore &operator+=(const BigIntCore &p_other);
	_FORCE_INLINE_ BigIntCore &operator-=(const BigIntCore &p_other);
	_FORCE_INLINE_ BigIntCore &operator*=(const BigIntCore &p_other);
	_FORCE_INLINE_ BigIntCore &operator/=(const BigIntCore &p_other);
	_FORCE_INLINE_ BigIntCore &operator%=(const BigIntCore &p_other) {
		*this = *this % p_other;
		return *this;
	}

	// ------------------------------------------------------------------------
	// Comparison API (Deterministic Logic)
	// ------------------------------------------------------------------------
	bool operator==(const BigIntCore &p_other) const;
	bool operator!=(const BigIntCore &p_other) const { return !(*this == p_other); }
	bool operator<(const BigIntCore &p_other) const;
	bool operator<=(const BigIntCore &p_other) const;
	bool operator>(const BigIntCore &p_other) const;
	bool operator>=(const BigIntCore &p_other) const;

	// ------------------------------------------------------------------------
	// Advanced Math Features (Warp-Kernel Logic)
	// ------------------------------------------------------------------------
	BigIntCore power(const BigIntCore &p_exponent) const;
	BigIntCore square_root() const;
	_FORCE_INLINE_ BigIntCore absolute() const {
		BigIntCore res = *this;
		res.is_negative = false;
		return res;
	}

	// ------------------------------------------------------------------------
	// Scale-Aware Utilities
	// ------------------------------------------------------------------------
	std::string to_string() const;
	std::string to_scientific() const;
	std::string to_aa_notation() const;
	std::string to_metric_symbol() const;
	std::string to_metric_name() const;

	_FORCE_INLINE_ bool is_zero() const { return chunks.size() == 1 && chunks[0] == 0; }
	_FORCE_INLINE_ int sign() const { return is_zero() ? 0 : (is_negative ? -1 : 1); }

	/**
	 * hash()
	 * Essential for EnTT Sparse-Set entity handles.
	 * Provides O(1) identification for trillion-entity simulations.
	 */
	_FORCE_INLINE_ uint32_t hash() const {
		uint32_t h = 5381;
		for (int i = 0; i < chunks.size(); i++) {
			h = ((h << 5) + h) + chunks[i];
		}
		return h ^ (is_negative ? 0xFFFFFFFF : 0);
	}

	/**
	 * get_raw_chunks()
	 * Direct access for Warp Kernels to stream data into SIMD registers.
	 */
	_FORCE_INLINE_ const uint32_t *get_raw_chunks() const { return chunks.ptr(); }
	_FORCE_INLINE_ int get_chunk_count() const { return chunks.size(); }
};

#endif // BIG_INT_CORE_H

--- END OF FILE src/big_int_core.h ---
