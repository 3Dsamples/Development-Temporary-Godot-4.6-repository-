// core/core_bind.cpp
#include "core_bind.h"
#include "../big_number.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "core/object/class_db.h"
#include "core/object/ref_counted.h"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstring>

// ----------------------------------------------------------------------------
// UEPCore Singleton Implementation
// ----------------------------------------------------------------------------
UEPCore* UEPCore::singleton = nullptr;

UEPCore::UEPCore() {
    singleton = this;
    pool.reserve(256); // Pre-allocate some space
}

UEPCore::~UEPCore() {
    clear_pool();
    singleton = nullptr;
}

void UEPCore::_bind_methods() {
    // Pool management
    ClassDB::bind_method(D_METHOD("create_bignumber"), &UEPCore::create_bignumber);
    ClassDB::bind_method(D_METHOD("destroy_bignumber", "id"), &UEPCore::destroy_bignumber);
    ClassDB::bind_method(D_METHOD("is_bignumber_valid", "id"), &UEPCore::is_bignumber_valid);
    ClassDB::bind_method(D_METHOD("clear_pool"), &UEPCore::clear_pool);
    ClassDB::bind_method(D_METHOD("get_pool_size"), &UEPCore::get_pool_size);
    ClassDB::bind_method(D_METHOD("get_pool_capacity"), &UEPCore::get_pool_capacity);
    ClassDB::bind_method(D_METHOD("reserve_pool", "capacity"), &UEPCore::reserve_pool);
    
    // Setters/Getters
    ClassDB::bind_method(D_METHOD("bignumber_set_string", "id", "value"), &UEPCore::bignumber_set_string);
    ClassDB::bind_method(D_METHOD("bignumber_get_string", "id"), &UEPCore::bignumber_get_string);
    ClassDB::bind_method(D_METHOD("bignumber_set_int", "id", "value"), &UEPCore::bignumber_set_int);
    ClassDB::bind_method(D_METHOD("bignumber_get_int", "id"), &UEPCore::bignumber_get_int);
    ClassDB::bind_method(D_METHOD("bignumber_set_float", "id", "value"), &UEPCore::bignumber_set_float);
    ClassDB::bind_method(D_METHOD("bignumber_get_float", "id"), &UEPCore::bignumber_get_float);
    
    // Arithmetic
    ClassDB::bind_method(D_METHOD("bignumber_add", "result_id", "a_id", "b_id"), &UEPCore::bignumber_add);
    ClassDB::bind_method(D_METHOD("bignumber_sub", "result_id", "a_id", "b_id"), &UEPCore::bignumber_sub);
    ClassDB::bind_method(D_METHOD("bignumber_mul", "result_id", "a_id", "b_id"), &UEPCore::bignumber_mul);
    ClassDB::bind_method(D_METHOD("bignumber_div", "result_id", "a_id", "b_id"), &UEPCore::bignumber_div);
    ClassDB::bind_method(D_METHOD("bignumber_mod", "result_id", "a_id", "b_id"), &UEPCore::bignumber_mod);
    
    // Unary
    ClassDB::bind_method(D_METHOD("bignumber_negate", "result_id", "src_id"), &UEPCore::bignumber_negate);
    ClassDB::bind_method(D_METHOD("bignumber_abs", "result_id", "src_id"), &UEPCore::bignumber_abs);
    
    // Comparison
    ClassDB::bind_method(D_METHOD("bignumber_compare", "a_id", "b_id"), &UEPCore::bignumber_compare);
    ClassDB::bind_method(D_METHOD("bignumber_is_zero", "id"), &UEPCore::bignumber_is_zero);
    ClassDB::bind_method(D_METHOD("bignumber_is_negative", "id"), &UEPCore::bignumber_is_negative);
    
    // Math functions
    ClassDB::bind_method(D_METHOD("bignumber_sin", "result_id", "src_id"), &UEPCore::bignumber_sin);
    ClassDB::bind_method(D_METHOD("bignumber_cos", "result_id", "src_id"), &UEPCore::bignumber_cos);
    ClassDB::bind_method(D_METHOD("bignumber_tan", "result_id", "src_id"), &UEPCore::bignumber_tan);
    ClassDB::bind_method(D_METHOD("bignumber_asin", "result_id", "src_id"), &UEPCore::bignumber_asin);
    ClassDB::bind_method(D_METHOD("bignumber_acos", "result_id", "src_id"), &UEPCore::bignumber_acos);
    ClassDB::bind_method(D_METHOD("bignumber_atan", "result_id", "src_id"), &UEPCore::bignumber_atan);
    ClassDB::bind_method(D_METHOD("bignumber_atan2", "result_id", "y_id", "x_id"), &UEPCore::bignumber_atan2);
    ClassDB::bind_method(D_METHOD("bignumber_exp", "result_id", "src_id"), &UEPCore::bignumber_exp);
    ClassDB::bind_method(D_METHOD("bignumber_log", "result_id", "src_id"), &UEPCore::bignumber_log);
    ClassDB::bind_method(D_METHOD("bignumber_log10", "result_id", "src_id"), &UEPCore::bignumber_log10);
    ClassDB::bind_method(D_METHOD("bignumber_pow", "result_id", "base_id", "exp_id"), &UEPCore::bignumber_pow);
    ClassDB::bind_method(D_METHOD("bignumber_sqrt", "result_id", "src_id"), &UEPCore::bignumber_sqrt);
    
    // Floor/Ceil/Round
    ClassDB::bind_method(D_METHOD("bignumber_floor", "result_id", "src_id"), &UEPCore::bignumber_floor);
    ClassDB::bind_method(D_METHOD("bignumber_ceil", "result_id", "src_id"), &UEPCore::bignumber_ceil);
    ClassDB::bind_method(D_METHOD("bignumber_round", "result_id", "src_id"), &UEPCore::bignumber_round);
    ClassDB::bind_method(D_METHOD("bignumber_frac", "result_id", "src_id"), &UEPCore::bignumber_frac);
    
    // Copy
    ClassDB::bind_method(D_METHOD("bignumber_copy", "dst_id", "src_id"), &UEPCore::bignumber_copy);
    
    // Batch operations
    ClassDB::bind_method(D_METHOD("bignumber_batch_add", "result_ids", "a_ids", "b_ids"), &UEPCore::bignumber_batch_add);
    ClassDB::bind_method(D_METHOD("bignumber_batch_mul", "result_ids", "a_ids", "b_ids"), &UEPCore::bignumber_batch_mul);
    
#ifdef UEP_USE_XTENSOR
    ClassDB::bind_method(D_METHOD("evaluate_xtensor_expression", "expr", "result_id"), &UEPCore::evaluate_xtensor_expression);
#endif
}

// ----------------------------------------------------------------------------
// Pool management implementation
// ----------------------------------------------------------------------------
uint64_t UEPCore::create_bignumber() {
    uint64_t id = next_id++;
    pool[id] = {uep::BigNumber(), 0, true};
    return id;
}

void UEPCore::destroy_bignumber(uint64_t id) {
    auto it = pool.find(id);
    if (it != pool.end()) {
        it->second.active = false;
        it->second.generation++; // Invalidate any stale handles
        pool.erase(it);
    }
}

bool UEPCore::is_bignumber_valid(uint64_t id) const {
    auto it = pool.find(id);
    return it != pool.end() && it->second.active;
}

void UEPCore::clear_pool() {
    pool.clear();
    next_id = 1;
}

size_t UEPCore::get_pool_size() const {
    return pool.size();
}

size_t UEPCore::get_pool_capacity() const {
    return pool.bucket_count() * 1; // approximate
}

void UEPCore::reserve_pool(size_t capacity) {
    pool.reserve(capacity);
}

// ----------------------------------------------------------------------------
// Internal helper: get entry with validation
// ----------------------------------------------------------------------------
UEPCore::PoolEntry* UEPCore::get_entry(uint64_t id) {
    auto it = pool.find(id);
    if (it != pool.end() && it->second.active) {
        return &it->second;
    }
    return nullptr;
}

const UEPCore::PoolEntry* UEPCore::get_entry(uint64_t id) const {
    auto it = pool.find(id);
    if (it != pool.end() && it->second.active) {
        return &it->second;
    }
    return nullptr;
}

// ----------------------------------------------------------------------------
// Individual operations on pooled BigNumbers
// ----------------------------------------------------------------------------
void UEPCore::bignumber_set_string(uint64_t id, const String& str) {
    PoolEntry* entry = get_entry(id);
    if (entry) {
        entry->value = uep::BigNumber::from_string(str.utf8().get_data());
    }
}

String UEPCore::bignumber_get_string(uint64_t id) const {
    const PoolEntry* entry = get_entry(id);
    if (entry) {
        return String(entry->value.to_string().c_str());
    }
    return String("0");
}

void UEPCore::bignumber_set_int(uint64_t id, int64_t val) {
    PoolEntry* entry = get_entry(id);
    if (entry) {
        entry->value = uep::BigNumber(val);
    }
}

int64_t UEPCore::bignumber_get_int(uint64_t id) const {
    const PoolEntry* entry = get_entry(id);
    if (entry) {
        return entry->value.to_int64();
    }
    return 0;
}

void UEPCore::bignumber_set_float(uint64_t id, double val) {
    PoolEntry* entry = get_entry(id);
    if (entry) {
        entry->value = uep::BigNumber(val);
    }
}

double UEPCore::bignumber_get_float(uint64_t id) const {
    const PoolEntry* entry = get_entry(id);
    if (entry) {
        return entry->value.to_double();
    }
    return 0.0;
}

void UEPCore::bignumber_add(uint64_t result_id, uint64_t a_id, uint64_t b_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* a = get_entry(a_id);
    const PoolEntry* b = get_entry(b_id);
    if (res && a && b) {
        res->value = a->value + b->value;
    }
}

void UEPCore::bignumber_sub(uint64_t result_id, uint64_t a_id, uint64_t b_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* a = get_entry(a_id);
    const PoolEntry* b = get_entry(b_id);
    if (res && a && b) {
        res->value = a->value - b->value;
    }
}

void UEPCore::bignumber_mul(uint64_t result_id, uint64_t a_id, uint64_t b_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* a = get_entry(a_id);
    const PoolEntry* b = get_entry(b_id);
    if (res && a && b) {
        res->value = a->value * b->value;
    }
}

void UEPCore::bignumber_div(uint64_t result_id, uint64_t a_id, uint64_t b_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* a = get_entry(a_id);
    const PoolEntry* b = get_entry(b_id);
    if (res && a && b) {
        if (!b->value.is_zero()) {
            res->value = a->value / b->value;
        }
    }
}

void UEPCore::bignumber_mod(uint64_t result_id, uint64_t a_id, uint64_t b_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* a = get_entry(a_id);
    const PoolEntry* b = get_entry(b_id);
    if (res && a && b) {
        if (!b->value.is_zero()) {
            res->value = a->value % b->value;
        }
    }
}

void UEPCore::bignumber_negate(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = -src->value;
    }
}

void UEPCore::bignumber_abs(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = src->value.abs();
    }
}

int UEPCore::bignumber_compare(uint64_t a_id, uint64_t b_id) const {
    const PoolEntry* a = get_entry(a_id);
    const PoolEntry* b = get_entry(b_id);
    if (a && b) {
        if (a->value < b->value) return -1;
        if (a->value > b->value) return 1;
        return 0;
    }
    return 0;
}

bool UEPCore::bignumber_is_zero(uint64_t id) const {
    const PoolEntry* entry = get_entry(id);
    return entry ? entry->value.is_zero() : true;
}

bool UEPCore::bignumber_is_negative(uint64_t id) const {
    const PoolEntry* entry = get_entry(id);
    return entry ? entry->value.is_negative() : false;
}

void UEPCore::bignumber_sin(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = uep::BigNumber::sin(src->value);
    }
}

void UEPCore::bignumber_cos(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = uep::BigNumber::cos(src->value);
    }
}

void UEPCore::bignumber_tan(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = uep::BigNumber::tan(src->value);
    }
}

void UEPCore::bignumber_asin(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = uep::BigNumber::asin(src->value);
    }
}

void UEPCore::bignumber_acos(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = uep::BigNumber::acos(src->value);
    }
}

void UEPCore::bignumber_atan(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = uep::BigNumber::atan(src->value);
    }
}

void UEPCore::bignumber_atan2(uint64_t result_id, uint64_t y_id, uint64_t x_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* y = get_entry(y_id);
    const PoolEntry* x = get_entry(x_id);
    if (res && y && x) {
        res->value = uep::BigNumber::atan2(y->value, x->value);
    }
}

void UEPCore::bignumber_exp(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = uep::BigNumber::exp(src->value);
    }
}

void UEPCore::bignumber_log(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = uep::BigNumber::log(src->value);
    }
}

void UEPCore::bignumber_log10(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = uep::BigNumber::log10(src->value);
    }
}

void UEPCore::bignumber_pow(uint64_t result_id, uint64_t base_id, uint64_t exp_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* base = get_entry(base_id);
    const PoolEntry* exp = get_entry(exp_id);
    if (res && base && exp) {
        res->value = uep::BigNumber::pow(base->value, exp->value);
    }
}

void UEPCore::bignumber_sqrt(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = uep::BigNumber::sqrt(src->value);
    }
}

void UEPCore::bignumber_floor(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = src->value.floor();
    }
}

void UEPCore::bignumber_ceil(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = src->value.ceil();
    }
}

void UEPCore::bignumber_round(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = src->value.round();
    }
}

void UEPCore::bignumber_frac(uint64_t result_id, uint64_t src_id) {
    PoolEntry* res = get_entry(result_id);
    const PoolEntry* src = get_entry(src_id);
    if (res && src) {
        res->value = src->value.frac();
    }
}

void UEPCore::bignumber_copy(uint64_t dst_id, uint64_t src_id) {
    PoolEntry* dst = get_entry(dst_id);
    const PoolEntry* src = get_entry(src_id);
    if (dst && src) {
        dst->value = src->value;
    }
}

// ----------------------------------------------------------------------------
// Batch operations using SIMD (if available) or simple loop
// ----------------------------------------------------------------------------
void UEPCore::bignumber_batch_add(const Vector<uint64_t>& result_ids, const Vector<uint64_t>& a_ids, const Vector<uint64_t>& b_ids) {
    int count = MIN(result_ids.size(), MIN(a_ids.size(), b_ids.size()));
    for (int i = 0; i < count; i++) {
        PoolEntry* res = get_entry(result_ids[i]);
        const PoolEntry* a = get_entry(a_ids[i]);
        const PoolEntry* b = get_entry(b_ids[i]);
        if (res && a && b) {
            res->value = a->value + b->value;
        }
    }
}

void UEPCore::bignumber_batch_mul(const Vector<uint64_t>& result_ids, const Vector<uint64_t>& a_ids, const Vector<uint64_t>& b_ids) {
    int count = MIN(result_ids.size(), MIN(a_ids.size(), b_ids.size()));
    for (int i = 0; i < count; i++) {
        PoolEntry* res = get_entry(result_ids[i]);
        const PoolEntry* a = get_entry(a_ids[i]);
        const PoolEntry* b = get_entry(b_ids[i]);
        if (res && a && b) {
            res->value = a->value * b->value;
        }
    }
}

#ifdef UEP_USE_XTENSOR
void UEPCore::evaluate_xtensor_expression(const String& expr_str, uint64_t result_id) {
    // Placeholder for xtensor expression evaluation
    // Full implementation would parse expr_str and execute fused kernel.
    PoolEntry* res = get_entry(result_id);
    if (!res) return;
    // For now, just set to zero.
    res->value = uep::BigNumber();
}
#endif

// ----------------------------------------------------------------------------
// UEPMathBindings implementation
// ----------------------------------------------------------------------------
void UEPMathBindings::_bind_methods() {
    ClassDB::bind_static_method("UEPMathBindings", D_METHOD("pi"), &UEPMathBindings::pi);
    ClassDB::bind_static_method("UEPMathBindings", D_METHOD("e"), &UEPMathBindings::e);
    ClassDB::bind_static_method("UEPMathBindings", D_METHOD("ln2"), &UEPMathBindings::ln2);
    ClassDB::bind_static_method("UEPMathBindings", D_METHOD("bignumber_to_string", "val"), &UEPMathBindings::bignumber_to_string);
    ClassDB::bind_static_method("UEPMathBindings", D_METHOD("bignumber_from_string", "str"), &UEPMathBindings::bignumber_from_string);
}

// ----------------------------------------------------------------------------
// Module initialization for core_bind (called from register_types)
// ----------------------------------------------------------------------------
void register_core_bind_classes() {
    GDREGISTER_CLASS(UEPCore);
    GDREGISTER_CLASS(UEPMathBindings);
    Engine::get_singleton()->add_singleton(Engine::Singleton("UEPCore", UEPCore::get_singleton()));
}

// Ending of File 10 of 15 (core/core_bind.cpp)