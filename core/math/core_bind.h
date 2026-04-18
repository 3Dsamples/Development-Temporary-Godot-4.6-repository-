// core/core_bind.h
#ifndef CORE_BIND_H
#define CORE_BIND_H

#include "core/object/class_db.h"
#include "core/object/ref_counted.h"
#include "core/variant/variant.h"
#include "../big_number.h"

// ----------------------------------------------------------------------------
// CoreBind: High-level C++ API bridge for BigNumber operations.
// Provides a static interface for servers and core systems to access
// BigNumber component pools managed by EnTT (or other ECS).
// This class is exposed to GDScript as "UEPCore".
// ----------------------------------------------------------------------------

class UEPCore : public Object {
    GDCLASS(UEPCore, Object);

    static UEPCore* singleton;

protected:
    static void _bind_methods();

public:
    static UEPCore* get_singleton() { return singleton; }
    
    UEPCore();
    ~UEPCore();

    // ------------------------------------------------------------------------
    // BigNumber pool management (for ECS integration)
    // ------------------------------------------------------------------------
    
    // Create a new BigNumber instance in the pool and return its ID (handle).
    uint64_t create_bignumber();
    
    // Destroy a BigNumber instance by ID.
    void destroy_bignumber(uint64_t id);
    
    // Check if an ID is valid.
    bool is_bignumber_valid(uint64_t id) const;
    
    // Set the value of a pooled BigNumber from string.
    void bignumber_set_string(uint64_t id, const String& str);
    
    // Get the string representation of a pooled BigNumber.
    String bignumber_get_string(uint64_t id) const;
    
    // Set from integer.
    void bignumber_set_int(uint64_t id, int64_t val);
    
    // Get as integer (if within range).
    int64_t bignumber_get_int(uint64_t id) const;
    
    // Set from double.
    void bignumber_set_float(uint64_t id, double val);
    
    // Get as double.
    double bignumber_get_float(uint64_t id) const;
    
    // Arithmetic operations between pooled numbers, storing result in another.
    void bignumber_add(uint64_t result_id, uint64_t a_id, uint64_t b_id);
    void bignumber_sub(uint64_t result_id, uint64_t a_id, uint64_t b_id);
    void bignumber_mul(uint64_t result_id, uint64_t a_id, uint64_t b_id);
    void bignumber_div(uint64_t result_id, uint64_t a_id, uint64_t b_id);
    void bignumber_mod(uint64_t result_id, uint64_t a_id, uint64_t b_id);
    
    // Unary operations.
    void bignumber_negate(uint64_t result_id, uint64_t src_id);
    void bignumber_abs(uint64_t result_id, uint64_t src_id);
    
    // Comparison.
    int bignumber_compare(uint64_t a_id, uint64_t b_id) const;
    bool bignumber_is_zero(uint64_t id) const;
    bool bignumber_is_negative(uint64_t id) const;
    
    // Math functions.
    void bignumber_sin(uint64_t result_id, uint64_t src_id);
    void bignumber_cos(uint64_t result_id, uint64_t src_id);
    void bignumber_tan(uint64_t result_id, uint64_t src_id);
    void bignumber_asin(uint64_t result_id, uint64_t src_id);
    void bignumber_acos(uint64_t result_id, uint64_t src_id);
    void bignumber_atan(uint64_t result_id, uint64_t src_id);
    void bignumber_atan2(uint64_t result_id, uint64_t y_id, uint64_t x_id);
    void bignumber_exp(uint64_t result_id, uint64_t src_id);
    void bignumber_log(uint64_t result_id, uint64_t src_id);
    void bignumber_log10(uint64_t result_id, uint64_t src_id);
    void bignumber_pow(uint64_t result_id, uint64_t base_id, uint64_t exp_id);
    void bignumber_sqrt(uint64_t result_id, uint64_t src_id);
    
    // Floor/Ceil/Round.
    void bignumber_floor(uint64_t result_id, uint64_t src_id);
    void bignumber_ceil(uint64_t result_id, uint64_t src_id);
    void bignumber_round(uint64_t result_id, uint64_t src_id);
    void bignumber_frac(uint64_t result_id, uint64_t src_id);
    
    // Copy operation.
    void bignumber_copy(uint64_t dst_id, uint64_t src_id);
    
    // ------------------------------------------------------------------------
    // Batch operations (SIMD-accelerated on arrays of BigNumbers)
    // These operate on contiguous arrays of pooled IDs.
    // ------------------------------------------------------------------------
    void bignumber_batch_add(const Vector<uint64_t>& result_ids, const Vector<uint64_t>& a_ids, const Vector<uint64_t>& b_ids);
    void bignumber_batch_mul(const Vector<uint64_t>& result_ids, const Vector<uint64_t>& a_ids, const Vector<uint64_t>& b_ids);
    
    // ------------------------------------------------------------------------
    // Xtensor expression evaluation (if enabled)
    // Evaluate an xtensor expression involving BigNumbers.
    // ------------------------------------------------------------------------
#ifdef UEP_USE_XTENSOR
    void evaluate_xtensor_expression(const String& expr_str, uint64_t result_id);
#endif

    // ------------------------------------------------------------------------
    // Memory management
    // ------------------------------------------------------------------------
    void clear_pool(); // Release all pooled numbers.
    size_t get_pool_size() const;
    size_t get_pool_capacity() const;
    void reserve_pool(size_t capacity);

private:
    // Internal pool using EnTT or a simple map (for simplicity, we use unordered_map)
    struct PoolEntry {
        uep::BigNumber value;
        uint64_t generation; // for handle validation
        bool active;
    };
    
    std::unordered_map<uint64_t, PoolEntry> pool;
    uint64_t next_id = 1;
    
    PoolEntry* get_entry(uint64_t id);
    const PoolEntry* get_entry(uint64_t id) const;
};

// ----------------------------------------------------------------------------
// Additional bindings for core math utilities
// ----------------------------------------------------------------------------
class UEPMathBindings : public Object {
    GDCLASS(UEPMathBindings, Object);
    
protected:
    static void _bind_methods();
    
public:
    // Constants
    static uep::BigNumber pi() { return uep::BigNumber::pi(); }
    static uep::BigNumber e() { return uep::BigNumber::e(); }
    static uep::BigNumber ln2() { return uep::BigNumber::ln2(); }
    
    // Conversion helpers
    static String bignumber_to_string(const uep::BigNumber& val) { return String(val.to_string().c_str()); }
    static uep::BigNumber bignumber_from_string(const String& str) { return uep::BigNumber::from_string(str.utf8().get_data()); }
};

#endif // CORE_BIND_H
// Ending of File 9 of 15 (core/core_bind.h)