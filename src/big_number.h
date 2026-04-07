--- START OF FILE src/big_number.h ---

#ifndef BIG_NUMBER_H
#define BIG_NUMBER_H

#include "core/object/ref_counted.h"
#include "core/object/class_db.h"
#include "core/string/ustring.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

// ============================================================================
// BigNumber Class (Macro-Scale Economy / Idle Game Integers)
// ============================================================================

class BigNumber : public RefCounted {
    GDCLASS(BigNumber, RefCounted);

private:
    BigIntCore core;

protected:
    static void _bind_methods();

public:
    BigNumber();
    ~BigNumber();

    // Initialization
    void set_value_from_string(const String& p_value);
    void set_value_from_int(int64_t p_value);

    // Arithmetic Operations (Returns a new Ref<BigNumber> to keep variables immutable in GDScript)
    Ref<BigNumber> add(const Ref<BigNumber>& p_other) const;
    Ref<BigNumber> subtract(const Ref<BigNumber>& p_other) const;
    Ref<BigNumber> multiply(const Ref<BigNumber>& p_other) const;
    Ref<BigNumber> divide(const Ref<BigNumber>& p_other) const;
    Ref<BigNumber> modulo(const Ref<BigNumber>& p_other) const;

    // Advanced Math Features (EnTT / Warp Backend Hooks)
    Ref<BigNumber> power(const Ref<BigNumber>& p_exponent) const;
    Ref<BigNumber> square_root() const;
    Ref<BigNumber> absolute() const;

    // Comparison Operations
    bool is_equal(const Ref<BigNumber>& p_other) const;
    bool is_not_equal(const Ref<BigNumber>& p_other) const;
    bool is_less_than(const Ref<BigNumber>& p_other) const;
    bool is_less_than_or_equal(const Ref<BigNumber>& p_other) const;
    bool is_greater_than(const Ref<BigNumber>& p_other) const;
    bool is_greater_than_or_equal(const Ref<BigNumber>& p_other) const;
    bool is_zero() const;

    // String Formatting / UI Conversions
    String to_string() const;
    String to_scientific() const;
    String to_aa_notation() const;
    String to_metric_symbol() const;
    String to_metric_name() const;

    // Internal getters/setters for Zero-Copy Kernel executions
    ET_SIMD_INLINE BigIntCore get_core() const { return core; }
    ET_SIMD_INLINE void set_core(const BigIntCore& p_core) { core = p_core; }
};

// ============================================================================
// FixedNumber Class (Macro-Scale Deterministic Scientific Physics)
// ============================================================================

class FixedNumber : public RefCounted {
    GDCLASS(FixedNumber, RefCounted);

private:
    FixedMathCore core;

protected:
    static void _bind_methods();

public:
    FixedNumber();
    ~FixedNumber();

    // Initialization
    void set_value_from_string(const String& p_value);
    void set_value_from_int(int64_t p_value);
    void set_value_from_float(double p_value);

    // Arithmetic Operations
    Ref<FixedNumber> add(const Ref<FixedNumber>& p_other) const;
    Ref<FixedNumber> subtract(const Ref<FixedNumber>& p_other) const;
    Ref<FixedNumber> multiply(const Ref<FixedNumber>& p_other) const;
    Ref<FixedNumber> divide(const Ref<FixedNumber>& p_other) const;
    Ref<FixedNumber> modulo(const Ref<FixedNumber>& p_other) const;

    // Deterministic Scientific Math
    Ref<FixedNumber> absolute() const;
    Ref<FixedNumber> square_root() const;
    Ref<FixedNumber> power(int32_t exponent) const;
    Ref<FixedNumber> sin() const;
    Ref<FixedNumber> cos() const;
    Ref<FixedNumber> tan() const;
    Ref<FixedNumber> atan2(const Ref<FixedNumber>& p_other) const;

    // Constants
    static Ref<FixedNumber> pi();
    static Ref<FixedNumber> e();

    // Comparison Operations
    bool is_equal(const Ref<FixedNumber>& p_other) const;
    bool is_not_equal(const Ref<FixedNumber>& p_other) const;
    bool is_less_than(const Ref<FixedNumber>& p_other) const;
    bool is_less_than_or_equal(const Ref<FixedNumber>& p_other) const;
    bool is_greater_than(const Ref<FixedNumber>& p_other) const;
    bool is_greater_than_or_equal(const Ref<FixedNumber>& p_other) const;

    // Output and conversion
    String to_string() const;
    double to_float() const;
    int64_t to_int() const;

    // Internal getters/setters for Zero-Copy Kernel executions
    ET_SIMD_INLINE FixedMathCore get_core() const { return core; }
    ET_SIMD_INLINE void set_core(const FixedMathCore& p_core) { core = p_core; }
};

#endif // BIG_NUMBER_H

--- END OF FILE src/big_number.h ---
