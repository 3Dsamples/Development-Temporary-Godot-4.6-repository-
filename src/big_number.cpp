--- START OF FILE src/big_number.cpp ---

#include "src/big_number.h"
#include "core/object/class_db.h"
#include "core/error/error_macros.h"
#include "core/variant/variant_utility.h"

// ============================================================================
// BigNumber Implementation (Macro-Scale Economy / Idle Game Integers)
// ============================================================================

void BigNumber::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_value_from_string", "value"), &BigNumber::set_value_from_string);
    ClassDB::bind_method(D_METHOD("set_value_from_int", "value"), &BigNumber::set_value_from_int);

    ClassDB::bind_method(D_METHOD("add", "other"), &BigNumber::add);
    ClassDB::bind_method(D_METHOD("subtract", "other"), &BigNumber::subtract);
    ClassDB::bind_method(D_METHOD("multiply", "other"), &BigNumber::multiply);
    ClassDB::bind_method(D_METHOD("divide", "other"), &BigNumber::divide);
    ClassDB::bind_method(D_METHOD("modulo", "other"), &BigNumber::modulo);

    ClassDB::bind_method(D_METHOD("power", "exponent"), &BigNumber::power);
    ClassDB::bind_method(D_METHOD("square_root"), &BigNumber::square_root);
    ClassDB::bind_method(D_METHOD("absolute"), &BigNumber::absolute);

    ClassDB::bind_method(D_METHOD("is_equal", "other"), &BigNumber::is_equal);
    ClassDB::bind_method(D_METHOD("is_not_equal", "other"), &BigNumber::is_not_equal);
    ClassDB::bind_method(D_METHOD("is_less_than", "other"), &BigNumber::is_less_than);
    ClassDB::bind_method(D_METHOD("is_less_than_or_equal", "other"), &BigNumber::is_less_than_or_equal);
    ClassDB::bind_method(D_METHOD("is_greater_than", "other"), &BigNumber::is_greater_than);
    ClassDB::bind_method(D_METHOD("is_greater_than_or_equal", "other"), &BigNumber::is_greater_than_or_equal);
    ClassDB::bind_method(D_METHOD("is_zero"), &BigNumber::is_zero);

    ClassDB::bind_method(D_METHOD("to_string"), &BigNumber::to_string);
    ClassDB::bind_method(D_METHOD("to_scientific"), &BigNumber::to_scientific);
    ClassDB::bind_method(D_METHOD("to_aa_notation"), &BigNumber::to_aa_notation);
    ClassDB::bind_method(D_METHOD("to_metric_symbol"), &BigNumber::to_metric_symbol);
    ClassDB::bind_method(D_METHOD("to_metric_name"), &BigNumber::to_metric_name);
}

BigNumber::BigNumber() {
    core = BigIntCore(0);
}

BigNumber::~BigNumber() {}

void BigNumber::set_value_from_string(const String& p_value) {
    core = BigIntCore(p_value.utf8().get_data());
}

void BigNumber::set_value_from_int(int64_t p_value) {
    core = BigIntCore(p_value);
}

Ref<BigNumber> BigNumber::add(const Ref<BigNumber>& p_other) const {
    ERR_FAIL_COND_V_MSG(p_other.is_null(), Ref<BigNumber>(), "UniversalSolver Error: Other BigNumber is null");
    Ref<BigNumber> result; result.instantiate();
    result->set_core(core + p_other->get_core()); return result;
}

Ref<BigNumber> BigNumber::subtract(const Ref<BigNumber>& p_other) const {
    ERR_FAIL_COND_V_MSG(p_other.is_null(), Ref<BigNumber>(), "UniversalSolver Error: Other BigNumber is null");
    Ref<BigNumber> result; result.instantiate();
    result->set_core(core - p_other->get_core()); return result;
}

Ref<BigNumber> BigNumber::multiply(const Ref<BigNumber>& p_other) const {
    ERR_FAIL_COND_V_MSG(p_other.is_null(), Ref<BigNumber>(), "UniversalSolver Error: Other BigNumber is null");
    Ref<BigNumber> result; result.instantiate();
    result->set_core(core * p_other->get_core()); return result;
}

Ref<BigNumber> BigNumber::divide(const Ref<BigNumber>& p_other) const {
    ERR_FAIL_COND_V_MSG(p_other.is_null(), Ref<BigNumber>(), "UniversalSolver Error: Other BigNumber is null");
    if (p_other->is_zero()) { ERR_PRINT("UniversalSolver Error: BigNumber Division by zero."); return Ref<BigNumber>(); }
    Ref<BigNumber> result; result.instantiate();
    result->set_core(core / p_other->get_core()); return result;
}

Ref<BigNumber> BigNumber::modulo(const Ref<BigNumber>& p_other) const {
    ERR_FAIL_COND_V_MSG(p_other.is_null(), Ref<BigNumber>(), "UniversalSolver Error: Other BigNumber is null");
    if (p_other->is_zero()) { ERR_PRINT("UniversalSolver Error: BigNumber Modulo by zero."); return Ref<BigNumber>(); }
    Ref<BigNumber> result; result.instantiate();
    result->set_core(core % p_other->get_core()); return result;
}

Ref<BigNumber> BigNumber::power(const Ref<BigNumber>& p_exponent) const {
    ERR_FAIL_COND_V_MSG(p_exponent.is_null(), Ref<BigNumber>(), "UniversalSolver Error: Exponent BigNumber is null");
    Ref<BigNumber> result; result.instantiate();
    result->set_core(core.power(p_exponent->get_core())); return result;
}

Ref<BigNumber> BigNumber::square_root() const {
    Ref<BigNumber> result; result.instantiate();
    try { result->set_core(core.square_root()); } 
    catch (const std::exception& e) { ERR_PRINT(String("UniversalSolver Error: ") + String(e.what())); }
    return result;
}

Ref<BigNumber> BigNumber::absolute() const {
    Ref<BigNumber> result; result.instantiate();
    result->set_core(core.absolute()); return result;
}

bool BigNumber::is_equal(const Ref<BigNumber>& p_other) const {
    if (p_other.is_null()) return false; return core == p_other->get_core();
}
bool BigNumber::is_not_equal(const Ref<BigNumber>& p_other) const {
    if (p_other.is_null()) return true; return core != p_other->get_core();
}
bool BigNumber::is_less_than(const Ref<BigNumber>& p_other) const {
    if (p_other.is_null()) return false; return core < p_other->get_core();
}
bool BigNumber::is_less_than_or_equal(const Ref<BigNumber>& p_other) const {
    if (p_other.is_null()) return false; return core <= p_other->get_core();
}
bool BigNumber::is_greater_than(const Ref<BigNumber>& p_other) const {
    if (p_other.is_null()) return false; return core > p_other->get_core();
}
bool BigNumber::is_greater_than_or_equal(const Ref<BigNumber>& p_other) const {
    if (p_other.is_null()) return false; return core >= p_other->get_core();
}
bool BigNumber::is_zero() const { return core.is_zero(); }
String BigNumber::to_string() const { return String(core.to_string().c_str()); }
String BigNumber::to_scientific() const { return String(core.to_scientific().c_str()); }
String BigNumber::to_aa_notation() const { return String(core.to_aa_notation().c_str()); }
String BigNumber::to_metric_symbol() const { return String(core.to_metric_symbol().c_str()); }
String BigNumber::to_metric_name() const { return String(core.to_metric_name().c_str()); }


// ============================================================================
// FixedNumber Implementation (Macro-Scale Deterministic Scientific Physics)
// ============================================================================

void FixedNumber::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_value_from_string", "value"), &FixedNumber::set_value_from_string);
    ClassDB::bind_method(D_METHOD("set_value_from_int", "value"), &FixedNumber::set_value_from_int);
    ClassDB::bind_method(D_METHOD("set_value_from_float", "value"), &FixedNumber::set_value_from_float);

    ClassDB::bind_method(D_METHOD("add", "other"), &FixedNumber::add);
    ClassDB::bind_method(D_METHOD("subtract", "other"), &FixedNumber::subtract);
    ClassDB::bind_method(D_METHOD("multiply", "other"), &FixedNumber::multiply);
    ClassDB::bind_method(D_METHOD("divide", "other"), &FixedNumber::divide);
    ClassDB::bind_method(D_METHOD("modulo", "other"), &FixedNumber::modulo);

    ClassDB::bind_method(D_METHOD("absolute"), &FixedNumber::absolute);
    ClassDB::bind_method(D_METHOD("square_root"), &FixedNumber::square_root);
    ClassDB::bind_method(D_METHOD("power", "exponent"), &FixedNumber::power);
    ClassDB::bind_method(D_METHOD("sin"), &FixedNumber::sin);
    ClassDB::bind_method(D_METHOD("cos"), &FixedNumber::cos);
    ClassDB::bind_method(D_METHOD("tan"), &FixedNumber::tan);
    ClassDB::bind_method(D_METHOD("atan2", "other"), &FixedNumber::atan2);

    ClassDB::bind_static_method("FixedNumber", D_METHOD("pi"), &FixedNumber::pi);
    ClassDB::bind_static_method("FixedNumber", D_METHOD("e"), &FixedNumber::e);

    ClassDB::bind_method(D_METHOD("is_equal", "other"), &FixedNumber::is_equal);
    ClassDB::bind_method(D_METHOD("is_not_equal", "other"), &FixedNumber::is_not_equal);
    ClassDB::bind_method(D_METHOD("is_less_than", "other"), &FixedNumber::is_less_than);
    ClassDB::bind_method(D_METHOD("is_less_than_or_equal", "other"), &FixedNumber::is_less_than_or_equal);
    ClassDB::bind_method(D_METHOD("is_greater_than", "other"), &FixedNumber::is_greater_than);
    ClassDB::bind_method(D_METHOD("is_greater_than_or_equal", "other"), &FixedNumber::is_greater_than_or_equal);

    ClassDB::bind_method(D_METHOD("to_string"), &FixedNumber::to_string);
    ClassDB::bind_method(D_METHOD("to_float"), &FixedNumber::to_float);
    ClassDB::bind_method(D_METHOD("to_int"), &FixedNumber::to_int);
}

FixedNumber::FixedNumber() { core = FixedMathCore(); }
FixedNumber::~FixedNumber() {}

void FixedNumber::set_value_from_string(const String& p_value) { core = FixedMathCore(p_value.utf8().get_data()); }
void FixedNumber::set_value_from_int(int64_t p_value) { core = FixedMathCore(p_value); }
void FixedNumber::set_value_from_float(double p_value) { core = FixedMathCore(p_value); }

Ref<FixedNumber> FixedNumber::add(const Ref<FixedNumber>& p_other) const {
    ERR_FAIL_COND_V_MSG(p_other.is_null(), Ref<FixedNumber>(), "UniversalSolver Error: Other FixedNumber is null");
    Ref<FixedNumber> res; res.instantiate(); res->set_core(core + p_other->get_core()); return res;
}
Ref<FixedNumber> FixedNumber::subtract(const Ref<FixedNumber>& p_other) const {
    ERR_FAIL_COND_V_MSG(p_other.is_null(), Ref<FixedNumber>(), "UniversalSolver Error: Other FixedNumber is null");
    Ref<FixedNumber> res; res.instantiate(); res->set_core(core - p_other->get_core()); return res;
}
Ref<FixedNumber> FixedNumber::multiply(const Ref<FixedNumber>& p_other) const {
    ERR_FAIL_COND_V_MSG(p_other.is_null(), Ref<FixedNumber>(), "UniversalSolver Error: Other FixedNumber is null");
    Ref<FixedNumber> res; res.instantiate(); res->set_core(core * p_other->get_core()); return res;
}
Ref<FixedNumber> FixedNumber::divide(const Ref<FixedNumber>& p_other) const {
    ERR_FAIL_COND_V_MSG(p_other.is_null(), Ref<FixedNumber>(), "UniversalSolver Error: Other FixedNumber is null");
    Ref<FixedNumber> res; res.instantiate();
    try { res->set_core(core / p_other->get_core()); }
    catch (const std::exception& e) { ERR_PRINT(e.what()); } return res;
}
Ref<FixedNumber> FixedNumber::modulo(const Ref<FixedNumber>& p_other) const {
    ERR_FAIL_COND_V_MSG(p_other.is_null(), Ref<FixedNumber>(), "UniversalSolver Error: Other FixedNumber is null");
    Ref<FixedNumber> res; res.instantiate();
    try { res->set_core(core % p_other->get_core()); }
    catch (const std::exception& e) { ERR_PRINT(e.what()); } return res;
}

Ref<FixedNumber> FixedNumber::absolute() const {
    Ref<FixedNumber> res; res.instantiate(); res->set_core(core.absolute()); return res;
}
Ref<FixedNumber> FixedNumber::square_root() const {
    Ref<FixedNumber> res; res.instantiate();
    try { res->set_core(core.square_root()); }
    catch (const std::exception& e) { ERR_PRINT(e.what()); } return res;
}
Ref<FixedNumber> FixedNumber::power(int32_t exponent) const {
    Ref<FixedNumber> res; res.instantiate(); res->set_core(core.power(exponent)); return res;
}
Ref<FixedNumber> FixedNumber::sin() const {
    Ref<FixedNumber> res; res.instantiate(); res->set_core(core.sin()); return res;
}
Ref<FixedNumber> FixedNumber::cos() const {
    Ref<FixedNumber> res; res.instantiate(); res->set_core(core.cos()); return res;
}
Ref<FixedNumber> FixedNumber::tan() const {
    Ref<FixedNumber> res; res.instantiate();
    try { res->set_core(core.tan()); }
    catch (const std::exception& e) { ERR_PRINT(e.what()); } return res;
}
Ref<FixedNumber> FixedNumber::atan2(const Ref<FixedNumber>& p_other) const {
    ERR_FAIL_COND_V_MSG(p_other.is_null(), Ref<FixedNumber>(), "UniversalSolver Error: Other FixedNumber is null");
    Ref<FixedNumber> res; res.instantiate(); res->set_core(core.atan2(p_other->get_core())); return res;
}

Ref<FixedNumber> FixedNumber::pi() {
    Ref<FixedNumber> res; res.instantiate(); res->set_core(FixedMathCore::pi()); return res;
}
Ref<FixedNumber> FixedNumber::e() {
    Ref<FixedNumber> res; res.instantiate(); res->set_core(FixedMathCore::e()); return res;
}

bool FixedNumber::is_equal(const Ref<FixedNumber>& p_other) const {
    if (p_other.is_null()) return false; return core == p_other->get_core();
}
bool FixedNumber::is_not_equal(const Ref<FixedNumber>& p_other) const {
    if (p_other.is_null()) return true; return core != p_other->get_core();
}
bool FixedNumber::is_less_than(const Ref<FixedNumber>& p_other) const {
    if (p_other.is_null()) return false; return core < p_other->get_core();
}
bool FixedNumber::is_less_than_or_equal(const Ref<FixedNumber>& p_other) const {
    if (p_other.is_null()) return false; return core <= p_other->get_core();
}
bool FixedNumber::is_greater_than(const Ref<FixedNumber>& p_other) const {
    if (p_other.is_null()) return false; return core > p_other->get_core();
}
bool FixedNumber::is_greater_than_or_equal(const Ref<FixedNumber>& p_other) const {
    if (p_other.is_null()) return false; return core >= p_other->get_core();
}

String FixedNumber::to_string() const { return String(core.to_string().c_str()); }
double FixedNumber::to_float() const { return core.to_double(); }
int64_t FixedNumber::to_int() const { return core.to_int(); }

--- END OF FILE src/big_number.cpp ---
