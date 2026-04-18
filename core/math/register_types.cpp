// src/register_types.cpp
#include "register_types.h"
#include "big_int_core.h"
#include "fixed_math_core.h"
#include "big_number.h"

#include "core/object/class_db.h"
#include "core/variant/variant.h"
#include "core/string/ustring.h"
#include "core/io/resource.h"
#include "core/object/ref_counted.h"
#include "core/object/object.h"
#include "core/variant/variant_parser.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#endif

// ----------------------------------------------------------------------------
// Variant type IDs (global)
// ----------------------------------------------------------------------------
int VARIANT_TYPE_BIGINTCORE = 0;
int VARIANT_TYPE_FIXEDMATHCORE = 0;
int VARIANT_TYPE_BIGNUMBER = 0;

// ----------------------------------------------------------------------------
// Forward declarations of custom Variant functions
// ----------------------------------------------------------------------------
struct VariantBigIntCore {
    static _FORCE_INLINE_ uep::BigIntCore* get(Variant *v) {
        return reinterpret_cast<uep::BigIntCore*>(v->_data._mem);
    }
    static _FORCE_INLINE_ const uep::BigIntCore* get(const Variant *v) {
        return reinterpret_cast<const uep::BigIntCore*>(v->_data._mem);
    }
    static void construct(Variant *v, const uep::BigIntCore& val) {
        memnew_placement(v->_data._mem, uep::BigIntCore(val));
    }
    static void default_construct(Variant *v) {
        memnew_placement(v->_data._mem, uep::BigIntCore);
    }
    static void destruct(Variant *v) {
        get(v)->~BigIntCore();
    }
};

struct VariantFixedMathCore {
    static _FORCE_INLINE_ uep::FixedMathCore* get(Variant *v) {
        return reinterpret_cast<uep::FixedMathCore*>(v->_data._mem);
    }
    static _FORCE_INLINE_ const uep::FixedMathCore* get(const Variant *v) {
        return reinterpret_cast<const uep::FixedMathCore*>(v->_data._mem);
    }
    static void construct(Variant *v, const uep::FixedMathCore& val) {
        memnew_placement(v->_data._mem, uep::FixedMathCore(val));
    }
    static void default_construct(Variant *v) {
        memnew_placement(v->_data._mem, uep::FixedMathCore);
    }
    static void destruct(Variant *v) {
        get(v)->~FixedMathCore();
    }
};

struct VariantBigNumber {
    static _FORCE_INLINE_ uep::BigNumber* get(Variant *v) {
        return reinterpret_cast<uep::BigNumber*>(v->_data._mem);
    }
    static _FORCE_INLINE_ const uep::BigNumber* get(const Variant *v) {
        return reinterpret_cast<const uep::BigNumber*>(v->_data._mem);
    }
    static void construct(Variant *v, const uep::BigNumber& val) {
        memnew_placement(v->_data._mem, uep::BigNumber(val));
    }
    static void default_construct(Variant *v) {
        memnew_placement(v->_data._mem, uep::BigNumber);
    }
    static void destruct(Variant *v) {
        get(v)->~BigNumber();
    }
};

// ----------------------------------------------------------------------------
// Variant operator function prototypes
// ----------------------------------------------------------------------------
static void bigintcore_get(const Variant *v, Variant *r_ret);
static void bigintcore_set(Variant *v, const Variant *p_val, bool *r_valid);
static bool bigintcore_has(const Variant *v, const StringName &p_member);
static void bigintcore_call(const Variant *v, const StringName &p_method, const Variant **p_args, int p_argcount, Variant *r_ret, Variant::CallError &r_error);
static String bigintcore_to_string(const Variant *v);
static uint32_t bigintcore_hash(const Variant *v);
static bool bigintcore_iter_init(const Variant *v, Variant::iter &r_iter, bool &r_valid);
static bool bigintcore_iter_next(const Variant *v, Variant::iter &r_iter, bool &r_valid);
static void bigintcore_iter_get(const Variant *v, Variant::iter &r_iter, Variant *r_ret, bool &r_valid);

static void fixedmathcore_get(const Variant *v, Variant *r_ret);
static void fixedmathcore_set(Variant *v, const Variant *p_val, bool *r_valid);
static bool fixedmathcore_has(const Variant *v, const StringName &p_member);
static void fixedmathcore_call(const Variant *v, const StringName &p_method, const Variant **p_args, int p_argcount, Variant *r_ret, Variant::CallError &r_error);
static String fixedmathcore_to_string(const Variant *v);
static uint32_t fixedmathcore_hash(const Variant *v);
static bool fixedmathcore_iter_init(const Variant *v, Variant::iter &r_iter, bool &r_valid);
static bool fixedmathcore_iter_next(const Variant *v, Variant::iter &r_iter, bool &r_valid);
static void fixedmathcore_iter_get(const Variant *v, Variant::iter &r_iter, Variant *r_ret, bool &r_valid);

static void bignumber_get(const Variant *v, Variant *r_ret);
static void bignumber_set(Variant *v, const Variant *p_val, bool *r_valid);
static bool bignumber_has(const Variant *v, const StringName &p_member);
static void bignumber_call(const Variant *v, const StringName &p_method, const Variant **p_args, int p_argcount, Variant *r_ret, Variant::CallError &r_error);
static String bignumber_to_string(const Variant *v);
static uint32_t bignumber_hash(const Variant *v);
static bool bignumber_iter_init(const Variant *v, Variant::iter &r_iter, bool &r_valid);
static bool bignumber_iter_next(const Variant *v, Variant::iter &r_iter, bool &r_valid);
static void bignumber_iter_get(const Variant *v, Variant::iter &r_iter, Variant *r_ret, bool &r_valid);

// ----------------------------------------------------------------------------
// Variant operator tables
// ----------------------------------------------------------------------------
static const Variant::OperatorEvaluator bigintcore_operator_evaluators[Variant::OP_MAX];
static const Variant::BuiltInMethod bigintcore_builtin_methods[] = {
    // No built-in methods beyond operators
    { StringName(), nullptr, 0, Variant::NIL }
};

static const Variant::OperatorEvaluator fixedmathcore_operator_evaluators[Variant::OP_MAX];
static const Variant::BuiltInMethod fixedmathcore_builtin_methods[] = {
    { StringName(), nullptr, 0, Variant::NIL }
};

static const Variant::OperatorEvaluator bignumber_operator_evaluators[Variant::OP_MAX];
static const Variant::BuiltInMethod bignumber_builtin_methods[] = {
    { StringName(), nullptr, 0, Variant::NIL }
};

// ----------------------------------------------------------------------------
// UEPMath singleton implementation
// ----------------------------------------------------------------------------
UEPMath* UEPMath::singleton = nullptr;

UEPMath::UEPMath() {
    singleton = this;
}

UEPMath::~UEPMath() {
    singleton = nullptr;
}

void UEPMath::_bind_methods() {
    // Bind static math functions
    ClassDB::bind_static_method("UEPMath", D_METHOD("sin", "x"), &UEPMath::sin);
    ClassDB::bind_static_method("UEPMath", D_METHOD("cos", "x"), &UEPMath::cos);
    ClassDB::bind_static_method("UEPMath", D_METHOD("tan", "x"), &UEPMath::tan);
    ClassDB::bind_static_method("UEPMath", D_METHOD("asin", "x"), &UEPMath::asin);
    ClassDB::bind_static_method("UEPMath", D_METHOD("acos", "x"), &UEPMath::acos);
    ClassDB::bind_static_method("UEPMath", D_METHOD("atan", "x"), &UEPMath::atan);
    ClassDB::bind_static_method("UEPMath", D_METHOD("atan2", "y", "x"), &UEPMath::atan2);
    ClassDB::bind_static_method("UEPMath", D_METHOD("exp", "x"), &UEPMath::exp);
    ClassDB::bind_static_method("UEPMath", D_METHOD("log", "x"), &UEPMath::log);
    ClassDB::bind_static_method("UEPMath", D_METHOD("log10", "x"), &UEPMath::log10);
    ClassDB::bind_static_method("UEPMath", D_METHOD("pow", "base", "exp"), &UEPMath::pow);
    ClassDB::bind_static_method("UEPMath", D_METHOD("sqrt", "x"), &UEPMath::sqrt);
    
    ClassDB::bind_static_method("UEPMath", D_METHOD("pi"), &UEPMath::pi);
    ClassDB::bind_static_method("UEPMath", D_METHOD("e"), &UEPMath::e);
    ClassDB::bind_static_method("UEPMath", D_METHOD("ln2"), &UEPMath::ln2);
    
    ClassDB::bind_static_method("UEPMath", D_METHOD("abs", "x"), &UEPMath::abs);
    ClassDB::bind_static_method("UEPMath", D_METHOD("floor", "x"), &UEPMath::floor);
    ClassDB::bind_static_method("UEPMath", D_METHOD("ceil", "x"), &UEPMath::ceil);
    ClassDB::bind_static_method("UEPMath", D_METHOD("round", "x"), &UEPMath::round);
    ClassDB::bind_static_method("UEPMath", D_METHOD("frac", "x"), &UEPMath::frac);
    ClassDB::bind_static_method("UEPMath", D_METHOD("sign", "x"), &UEPMath::sign);
    ClassDB::bind_static_method("UEPMath", D_METHOD("min", "a", "b"), &UEPMath::min);
    ClassDB::bind_static_method("UEPMath", D_METHOD("max", "a", "b"), &UEPMath::max);
    ClassDB::bind_static_method("UEPMath", D_METHOD("clamp", "val", "lo", "hi"), &UEPMath::clamp);
    ClassDB::bind_static_method("UEPMath", D_METHOD("lerp", "a", "b", "t"), &UEPMath::lerp);
    
    ClassDB::bind_static_method("UEPMath", D_METHOD("to_string", "x"), &UEPMath::to_string);
    ClassDB::bind_static_method("UEPMath", D_METHOD("from_string", "str"), &UEPMath::from_string);
    ClassDB::bind_static_method("UEPMath", D_METHOD("from_int", "val"), &UEPMath::from_int);
    ClassDB::bind_static_method("UEPMath", D_METHOD("from_float", "val"), &UEPMath::from_float);
    ClassDB::bind_static_method("UEPMath", D_METHOD("to_int", "x"), &UEPMath::to_int);
    ClassDB::bind_static_method("UEPMath", D_METHOD("to_float", "x"), &UEPMath::to_float);
}

// ----------------------------------------------------------------------------
// Register BigIntCore class with Godot
// ----------------------------------------------------------------------------
class BigIntCoreObject : public RefCounted {
    GDCLASS(BigIntCoreObject, RefCounted);
    
    uep::BigIntCore value;
    
protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("add", "other"), &BigIntCoreObject::add);
        ClassDB::bind_method(D_METHOD("sub", "other"), &BigIntCoreObject::sub);
        ClassDB::bind_method(D_METHOD("mul", "other"), &BigIntCoreObject::mul);
        ClassDB::bind_method(D_METHOD("div", "other"), &BigIntCoreObject::div);
        ClassDB::bind_method(D_METHOD("mod", "other"), &BigIntCoreObject::mod);
        ClassDB::bind_method(D_METHOD("negate"), &BigIntCoreObject::negate);
        ClassDB::bind_method(D_METHOD("abs"), &BigIntCoreObject::abs);
        ClassDB::bind_method(D_METHOD("compare", "other"), &BigIntCoreObject::compare);
        ClassDB::bind_method(D_METHOD("is_zero"), &BigIntCoreObject::is_zero);
        ClassDB::bind_method(D_METHOD("is_negative"), &BigIntCoreObject::is_negative);
        ClassDB::bind_method(D_METHOD("to_string"), &BigIntCoreObject::to_string);
        ClassDB::bind_method(D_METHOD("from_string", "str"), &BigIntCoreObject::from_string);
        ClassDB::bind_method(D_METHOD("to_int"), &BigIntCoreObject::to_int);
        ClassDB::bind_method(D_METHOD("from_int", "val"), &BigIntCoreObject::from_int);
        
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "value"), "set_value", "get_value");
    }
    
public:
    BigIntCoreObject() {}
    BigIntCoreObject(const uep::BigIntCore& p_val) : value(p_val) {}
    
    void set_value(const String& p_str) { value = uep::BigIntCore::from_string(p_str.utf8().get_data()); }
    String get_value() const { return String(value.to_string().c_str()); }
    
    Ref<BigIntCoreObject> add(const Ref<BigIntCoreObject>& other) const {
        return memnew(BigIntCoreObject(value + other->value));
    }
    Ref<BigIntCoreObject> sub(const Ref<BigIntCoreObject>& other) const {
        return memnew(BigIntCoreObject(value - other->value));
    }
    Ref<BigIntCoreObject> mul(const Ref<BigIntCoreObject>& other) const {
        return memnew(BigIntCoreObject(value * other->value));
    }
    Ref<BigIntCoreObject> div(const Ref<BigIntCoreObject>& other) const {
        return memnew(BigIntCoreObject(value / other->value));
    }
    Ref<BigIntCoreObject> mod(const Ref<BigIntCoreObject>& other) const {
        return memnew(BigIntCoreObject(value % other->value));
    }
    Ref<BigIntCoreObject> negate() const {
        return memnew(BigIntCoreObject(-value));
    }
    Ref<BigIntCoreObject> abs() const {
        uep::BigIntCore copy = value;
        if (copy.is_negative()) copy = -copy;
        return memnew(BigIntCoreObject(copy));
    }
    int compare(const Ref<BigIntCoreObject>& other) const {
        if (value < other->value) return -1;
        if (value > other->value) return 1;
        return 0;
    }
    bool is_zero() const { return value.is_zero(); }
    bool is_negative() const { return value.is_negative(); }
    String to_string() const { return String(value.to_string().c_str()); }
    void from_string(const String& str) { value = uep::BigIntCore::from_string(str.utf8().get_data()); }
    int64_t to_int() const { return value.to_int64(); }
    void from_int(int64_t val) { value = uep::BigIntCore(val); }
};

// ----------------------------------------------------------------------------
// Register FixedMathCore class with Godot
// ----------------------------------------------------------------------------
class FixedMathCoreObject : public RefCounted {
    GDCLASS(FixedMathCoreObject, RefCounted);
    
    uep::FixedMathCore value;
    
protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("add", "other"), &FixedMathCoreObject::add);
        ClassDB::bind_method(D_METHOD("sub", "other"), &FixedMathCoreObject::sub);
        ClassDB::bind_method(D_METHOD("mul", "other"), &FixedMathCoreObject::mul);
        ClassDB::bind_method(D_METHOD("div", "other"), &FixedMathCoreObject::div);
        ClassDB::bind_method(D_METHOD("negate"), &FixedMathCoreObject::negate);
        ClassDB::bind_method(D_METHOD("abs"), &FixedMathCoreObject::abs);
        ClassDB::bind_method(D_METHOD("floor"), &FixedMathCoreObject::floor);
        ClassDB::bind_method(D_METHOD("ceil"), &FixedMathCoreObject::ceil);
        ClassDB::bind_method(D_METHOD("round"), &FixedMathCoreObject::round);
        ClassDB::bind_method(D_METHOD("frac"), &FixedMathCoreObject::frac);
        ClassDB::bind_method(D_METHOD("sin"), &FixedMathCoreObject::sin);
        ClassDB::bind_method(D_METHOD("cos"), &FixedMathCoreObject::cos);
        ClassDB::bind_method(D_METHOD("tan"), &FixedMathCoreObject::tan);
        ClassDB::bind_method(D_METHOD("asin"), &FixedMathCoreObject::asin);
        ClassDB::bind_method(D_METHOD("acos"), &FixedMathCoreObject::acos);
        ClassDB::bind_method(D_METHOD("atan"), &FixedMathCoreObject::atan);
        ClassDB::bind_method(D_METHOD("atan2", "x"), &FixedMathCoreObject::atan2);
        ClassDB::bind_method(D_METHOD("exp"), &FixedMathCoreObject::exp);
        ClassDB::bind_method(D_METHOD("log"), &FixedMathCoreObject::log);
        ClassDB::bind_method(D_METHOD("sqrt"), &FixedMathCoreObject::sqrt);
        ClassDB::bind_method(D_METHOD("to_float"), &FixedMathCoreObject::to_float);
        ClassDB::bind_method(D_METHOD("from_float", "val"), &FixedMathCoreObject::from_float);
        ClassDB::bind_method(D_METHOD("to_string"), &FixedMathCoreObject::to_string);
        
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "value"), "set_value", "get_value");
    }
    
public:
    FixedMathCoreObject() {}
    FixedMathCoreObject(const uep::FixedMathCore& p_val) : value(p_val) {}
    
    void set_value(double p_val) { value = uep::FixedMathCore(p_val); }
    double get_value() const { return value.to_double(); }
    
    Ref<FixedMathCoreObject> add(const Ref<FixedMathCoreObject>& other) const {
        return memnew(FixedMathCoreObject(value + other->value));
    }
    Ref<FixedMathCoreObject> sub(const Ref<FixedMathCoreObject>& other) const {
        return memnew(FixedMathCoreObject(value - other->value));
    }
    Ref<FixedMathCoreObject> mul(const Ref<FixedMathCoreObject>& other) const {
        return memnew(FixedMathCoreObject(value * other->value));
    }
    Ref<FixedMathCoreObject> div(const Ref<FixedMathCoreObject>& other) const {
        return memnew(FixedMathCoreObject(value / other->value));
    }
    Ref<FixedMathCoreObject> negate() const {
        return memnew(FixedMathCoreObject(-value));
    }
    Ref<FixedMathCoreObject> abs() const {
        return memnew(FixedMathCoreObject(value.abs()));
    }
    Ref<FixedMathCoreObject> floor() const {
        return memnew(FixedMathCoreObject(value.floor()));
    }
    Ref<FixedMathCoreObject> ceil() const {
        return memnew(FixedMathCoreObject(value.ceil()));
    }
    Ref<FixedMathCoreObject> round() const {
        return memnew(FixedMathCoreObject(value.round()));
    }
    Ref<FixedMathCoreObject> frac() const {
        return memnew(FixedMathCoreObject(value.frac()));
    }
    Ref<FixedMathCoreObject> sin() const {
        return memnew(FixedMathCoreObject(uep::FixedMathCore::sin(value)));
    }
    Ref<FixedMathCoreObject> cos() const {
        return memnew(FixedMathCoreObject(uep::FixedMathCore::cos(value)));
    }
    Ref<FixedMathCoreObject> tan() const {
        return memnew(FixedMathCoreObject(uep::FixedMathCore::tan(value)));
    }
    Ref<FixedMathCoreObject> asin() const {
        return memnew(FixedMathCoreObject(uep::FixedMathCore::asin(value)));
    }
    Ref<FixedMathCoreObject> acos() const {
        return memnew(FixedMathCoreObject(uep::FixedMathCore::acos(value)));
    }
    Ref<FixedMathCoreObject> atan() const {
        return memnew(FixedMathCoreObject(uep::FixedMathCore::atan(value)));
    }
    Ref<FixedMathCoreObject> atan2(const Ref<FixedMathCoreObject>& x) const {
        return memnew(FixedMathCoreObject(uep::FixedMathCore::atan2(value, x->value)));
    }
    Ref<FixedMathCoreObject> exp() const {
        return memnew(FixedMathCoreObject(uep::FixedMathCore::exp(value)));
    }
    Ref<FixedMathCoreObject> log() const {
        return memnew(FixedMathCoreObject(uep::FixedMathCore::log(value)));
    }
    Ref<FixedMathCoreObject> sqrt() const {
        return memnew(FixedMathCoreObject(uep::FixedMathCore::sqrt(value)));
    }
    double to_float() const { return value.to_double(); }
    void from_float(double val) { value = uep::FixedMathCore(val); }
    String to_string() const { return String(value.to_string().c_str()); }
};

// ----------------------------------------------------------------------------
// Register BigNumber class with Godot
// ----------------------------------------------------------------------------
class BigNumberObject : public RefCounted {
    GDCLASS(BigNumberObject, RefCounted);
    
    uep::BigNumber value;
    
protected:
    static void _bind_methods() {
        // Arithmetic
        ClassDB::bind_method(D_METHOD("add", "other"), &BigNumberObject::add);
        ClassDB::bind_method(D_METHOD("sub", "other"), &BigNumberObject::sub);
        ClassDB::bind_method(D_METHOD("mul", "other"), &BigNumberObject::mul);
        ClassDB::bind_method(D_METHOD("div", "other"), &BigNumberObject::div);
        ClassDB::bind_method(D_METHOD("mod", "other"), &BigNumberObject::mod);
        ClassDB::bind_method(D_METHOD("negate"), &BigNumberObject::negate);
        // Math
        ClassDB::bind_method(D_METHOD("abs"), &BigNumberObject::abs);
        ClassDB::bind_method(D_METHOD("floor"), &BigNumberObject::floor);
        ClassDB::bind_method(D_METHOD("ceil"), &BigNumberObject::ceil);
        ClassDB::bind_method(D_METHOD("round"), &BigNumberObject::round);
        ClassDB::bind_method(D_METHOD("frac"), &BigNumberObject::frac);
        ClassDB::bind_method(D_METHOD("sin"), &BigNumberObject::sin);
        ClassDB::bind_method(D_METHOD("cos"), &BigNumberObject::cos);
        ClassDB::bind_method(D_METHOD("tan"), &BigNumberObject::tan);
        ClassDB::bind_method(D_METHOD("asin"), &BigNumberObject::asin);
        ClassDB::bind_method(D_METHOD("acos"), &BigNumberObject::acos);
        ClassDB::bind_method(D_METHOD("atan"), &BigNumberObject::atan);
        ClassDB::bind_method(D_METHOD("atan2", "x"), &BigNumberObject::atan2);
        ClassDB::bind_method(D_METHOD("exp"), &BigNumberObject::exp);
        ClassDB::bind_method(D_METHOD("log"), &BigNumberObject::log);
        ClassDB::bind_method(D_METHOD("log10"), &BigNumberObject::log10);
        ClassDB::bind_method(D_METHOD("pow", "exp"), &BigNumberObject::pow);
        ClassDB::bind_method(D_METHOD("sqrt"), &BigNumberObject::sqrt);
        // Comparison
        ClassDB::bind_method(D_METHOD("compare", "other"), &BigNumberObject::compare);
        ClassDB::bind_method(D_METHOD("is_zero"), &BigNumberObject::is_zero);
        ClassDB::bind_method(D_METHOD("is_negative"), &BigNumberObject::is_negative);
        // Conversion
        ClassDB::bind_method(D_METHOD("to_string"), &BigNumberObject::to_string);
        ClassDB::bind_method(D_METHOD("from_string", "str"), &BigNumberObject::from_string);
        ClassDB::bind_method(D_METHOD("to_float"), &BigNumberObject::to_float);
        ClassDB::bind_method(D_METHOD("from_float", "val"), &BigNumberObject::from_float);
        ClassDB::bind_method(D_METHOD("to_int"), &BigNumberObject::to_int);
        ClassDB::bind_method(D_METHOD("from_int", "val"), &BigNumberObject::from_int);
        
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "value"), "set_value", "get_value");
    }
    
public:
    BigNumberObject() {}
    BigNumberObject(const uep::BigNumber& p_val) : value(p_val) {}
    
    void set_value(const String& p_str) { value = uep::BigNumber::from_string(p_str.utf8().get_data()); }
    String get_value() const { return String(value.to_string().c_str()); }
    
    Ref<BigNumberObject> add(const Ref<BigNumberObject>& other) const { return memnew(BigNumberObject(value + other->value)); }
    Ref<BigNumberObject> sub(const Ref<BigNumberObject>& other) const { return memnew(BigNumberObject(value - other->value)); }
    Ref<BigNumberObject> mul(const Ref<BigNumberObject>& other) const { return memnew(BigNumberObject(value * other->value)); }
    Ref<BigNumberObject> div(const Ref<BigNumberObject>& other) const { return memnew(BigNumberObject(value / other->value)); }
    Ref<BigNumberObject> mod(const Ref<BigNumberObject>& other) const { return memnew(BigNumberObject(value % other->value)); }
    Ref<BigNumberObject> negate() const { return memnew(BigNumberObject(-value)); }
    
    Ref<BigNumberObject> abs() const { return memnew(BigNumberObject(value.abs())); }
    Ref<BigNumberObject> floor() const { return memnew(BigNumberObject(value.floor())); }
    Ref<BigNumberObject> ceil() const { return memnew(BigNumberObject(value.ceil())); }
    Ref<BigNumberObject> round() const { return memnew(BigNumberObject(value.round())); }
    Ref<BigNumberObject> frac() const { return memnew(BigNumberObject(value.frac())); }
    Ref<BigNumberObject> sin() const { return memnew(BigNumberObject(uep::BigNumber::sin(value))); }
    Ref<BigNumberObject> cos() const { return memnew(BigNumberObject(uep::BigNumber::cos(value))); }
    Ref<BigNumberObject> tan() const { return memnew(BigNumberObject(uep::BigNumber::tan(value))); }
    Ref<BigNumberObject> asin() const { return memnew(BigNumberObject(uep::BigNumber::asin(value))); }
    Ref<BigNumberObject> acos() const { return memnew(BigNumberObject(uep::BigNumber::acos(value))); }
    Ref<BigNumberObject> atan() const { return memnew(BigNumberObject(uep::BigNumber::atan(value))); }
    Ref<BigNumberObject> atan2(const Ref<BigNumberObject>& x) const { return memnew(BigNumberObject(uep::BigNumber::atan2(value, x->value))); }
    Ref<BigNumberObject> exp() const { return memnew(BigNumberObject(uep::BigNumber::exp(value))); }
    Ref<BigNumberObject> log() const { return memnew(BigNumberObject(uep::BigNumber::log(value))); }
    Ref<BigNumberObject> log10() const { return memnew(BigNumberObject(uep::BigNumber::log10(value))); }
    Ref<BigNumberObject> pow(const Ref<BigNumberObject>& exp) const { return memnew(BigNumberObject(uep::BigNumber::pow(value, exp->value))); }
    Ref<BigNumberObject> sqrt() const { return memnew(BigNumberObject(uep::BigNumber::sqrt(value))); }
    
    int compare(const Ref<BigNumberObject>& other) const {
        if (value < other->value) return -1;
        if (value > other->value) return 1;
        return 0;
    }
    bool is_zero() const { return value.is_zero(); }
    bool is_negative() const { return value.is_negative(); }
    String to_string() const { return String(value.to_string().c_str()); }
    void from_string(const String& str) { value = uep::BigNumber::from_string(str.utf8().get_data()); }
    double to_float() const { return value.to_double(); }
    void from_float(double val) { value = uep::BigNumber(val); }
    int64_t to_int() const { return value.to_int64(); }
    void from_int(int64_t val) { value = uep::BigNumber(val); }
};

// ----------------------------------------------------------------------------
// Registration functions implementation
// ----------------------------------------------------------------------------
namespace uep {
namespace registration {

void register_bigintcore_class() {
    GDREGISTER_CLASS(BigIntCoreObject);
}

void register_fixedmathcore_class() {
    GDREGISTER_CLASS(FixedMathCoreObject);
}

void register_bignumber_class() {
    GDREGISTER_CLASS(BigNumberObject);
}

void register_classes() {
    register_bigintcore_class();
    register_fixedmathcore_class();
    register_bignumber_class();
    
    // Register the UEPMath singleton
    GDREGISTER_CLASS(UEPMath);
    Engine::get_singleton()->add_singleton(Engine::Singleton("UEPMath", UEPMath::get_singleton()));
}

void register_variant_types() {
    // Register BigIntCore as a Variant type
    Variant::Type bigintcore_type = Variant::VARIANT_MAX;
    for (int i = 0; i < Variant::VARIANT_MAX; i++) {
        if (Variant::get_type_name(Variant::Type(i)) == StringName()) {
            bigintcore_type = Variant::Type(i);
            break;
        }
    }
    if (bigintcore_type == Variant::VARIANT_MAX) {
        bigintcore_type = Variant::Type(Variant::VARIANT_MAX);
    }
    VARIANT_TYPE_BIGINTCORE = bigintcore_type;
    
    Variant::register_type(Variant::Type(bigintcore_type), "BigIntCore",
        bigintcore_get, bigintcore_set,
        bigintcore_has, bigintcore_call,
        bigintcore_to_string, bigintcore_hash,
        bigintcore_iter_init, bigintcore_iter_next, bigintcore_iter_get,
        bigintcore_operator_evaluators, bigintcore_builtin_methods,
        sizeof(uep::BigIntCore), Variant::OBJECT_TYPE_NONE,
        &VariantBigIntCore::construct, &VariantBigIntCore::default_construct, &VariantBigIntCore::destruct);
    
    // Register FixedMathCore
    Variant::Type fixed_type = Variant::VARIANT_MAX;
    for (int i = 0; i < Variant::VARIANT_MAX; i++) {
        if (Variant::get_type_name(Variant::Type(i)) == StringName()) {
            fixed_type = Variant::Type(i);
            break;
        }
    }
    VARIANT_TYPE_FIXEDMATHCORE = fixed_type;
    Variant::register_type(Variant::Type(fixed_type), "FixedMathCore",
        fixedmathcore_get, fixedmathcore_set,
        fixedmathcore_has, fixedmathcore_call,
        fixedmathcore_to_string, fixedmathcore_hash,
        fixedmathcore_iter_init, fixedmathcore_iter_next, fixedmathcore_iter_get,
        fixedmathcore_operator_evaluators, fixedmathcore_builtin_methods,
        sizeof(uep::FixedMathCore), Variant::OBJECT_TYPE_NONE,
        &VariantFixedMathCore::construct, &VariantFixedMathCore::default_construct, &VariantFixedMathCore::destruct);
    
    // Register BigNumber
    Variant::Type bn_type = Variant::VARIANT_MAX;
    for (int i = 0; i < Variant::VARIANT_MAX; i++) {
        if (Variant::get_type_name(Variant::Type(i)) == StringName()) {
            bn_type = Variant::Type(i);
            break;
        }
    }
    VARIANT_TYPE_BIGNUMBER = bn_type;
    Variant::register_type(Variant::Type(bn_type), "BigNumber",
        bignumber_get, bignumber_set,
        bignumber_has, bignumber_call,
        bignumber_to_string, bignumber_hash,
        bignumber_iter_init, bignumber_iter_next, bignumber_iter_get,
        bignumber_operator_evaluators, bignumber_builtin_methods,
        sizeof(uep::BigNumber), Variant::OBJECT_TYPE_NONE,
        &VariantBigNumber::construct, &VariantBigNumber::default_construct, &VariantBigNumber::destruct);
}

void register_core_constants() {
    // Constants are registered via core_constants.cpp, but we can also add them here.
    // We'll rely on core_constants registration.
}

void register_math_functions() {
    // Already done via UEPMath singleton binding.
}

void initialize_math_core() {
    // Initialize any global tables (like CORDIC tables, factorial tables)
    // These are statically initialized in the respective .cpp files, so nothing needed here.
}

void cleanup_math_core() {
    // Cleanup if necessary.
}

} // namespace registration
} // namespace uep

// ----------------------------------------------------------------------------
// Variant operator implementations (stubs, full implementation would be lengthy)
// These provide get/set/call for the custom types. For brevity, we implement basic functionality.
// ----------------------------------------------------------------------------
static void bigintcore_get(const Variant *v, Variant *r_ret) {
    const uep::BigIntCore* ptr = VariantBigIntCore::get(v);
    // For simplicity, return the object wrapper or the string representation
    *r_ret = String(ptr->to_string().c_str());
}

static void bigintcore_set(Variant *v, const Variant *p_val, bool *r_valid) {
    uep::BigIntCore* ptr = VariantBigIntCore::get(v);
    if (p_val->get_type() == Variant::STRING) {
        *ptr = uep::BigIntCore::from_string(p_val->operator String().utf8().get_data());
        *r_valid = true;
    } else if (p_val->get_type() == Variant::INT) {
        *ptr = uep::BigIntCore(p_val->operator int64_t());
        *r_valid = true;
    } else {
        *r_valid = false;
    }
}

static bool bigintcore_has(const Variant *v, const StringName &p_member) { return false; }
static void bigintcore_call(const Variant *v, const StringName &p_method, const Variant **p_args, int p_argcount, Variant *r_ret, Variant::CallError &r_error) {
    r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
}
static String bigintcore_to_string(const Variant *v) {
    return String(VariantBigIntCore::get(v)->to_string().c_str());
}
static uint32_t bigintcore_hash(const Variant *v) {
    // Hash based on limbs
    const uep::BigIntCore* ptr = VariantBigIntCore::get(v);
    uint32_t h = ptr->size();
    for (size_t i = 0; i < ptr->size(); i++) {
        h = hash_murmur3_one_64(ptr->data()[i], h);
    }
    return h;
}
static bool bigintcore_iter_init(const Variant *v, Variant::iter &r_iter, bool &r_valid) { r_valid = false; return false; }
static bool bigintcore_iter_next(const Variant *v, Variant::iter &r_iter, bool &r_valid) { r_valid = false; return false; }
static void bigintcore_iter_get(const Variant *v, Variant::iter &r_iter, Variant *r_ret, bool &r_valid) { r_valid = false; }

// FixedMathCore variants
static void fixedmathcore_get(const Variant *v, Variant *r_ret) {
    *r_ret = VariantBigMathCore::get(v)->to_double();
}
static void fixedmathcore_set(Variant *v, const Variant *p_val, bool *r_valid) {
    uep::FixedMathCore* ptr = VariantFixedMathCore::get(v);
    if (p_val->get_type() == Variant::FLOAT || p_val->get_type() == Variant::INT) {
        *ptr = uep::FixedMathCore(p_val->operator double());
        *r_valid = true;
    } else {
        *r_valid = false;
    }
}
static bool fixedmathcore_has(const Variant *v, const StringName &p_member) { return false; }
static void fixedmathcore_call(const Variant *v, const StringName &p_method, const Variant **p_args, int p_argcount, Variant *r_ret, Variant::CallError &r_error) {
    r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
}
static String fixedmathcore_to_string(const Variant *v) {
    return String(VariantFixedMathCore::get(v)->to_string().c_str());
}
static uint32_t fixedmathcore_hash(const Variant *v) {
    return hash_murmur3_one_64(VariantFixedMathCore::get(v)->raw());
}
static bool fixedmathcore_iter_init(const Variant *v, Variant::iter &r_iter, bool &r_valid) { r_valid = false; return false; }
static bool fixedmathcore_iter_next(const Variant *v, Variant::iter &r_iter, bool &r_valid) { r_valid = false; return false; }
static void fixedmathcore_iter_get(const Variant *v, Variant::iter &r_iter, Variant *r_ret, bool &r_valid) { r_valid = false; }

// BigNumber variants
static void bignumber_get(const Variant *v, Variant *r_ret) {
    *r_ret = String(VariantBigNumber::get(v)->to_string().c_str());
}
static void bignumber_set(Variant *v, const Variant *p_val, bool *r_valid) {
    uep::BigNumber* ptr = VariantBigNumber::get(v);
    if (p_val->get_type() == Variant::STRING) {
        *ptr = uep::BigNumber::from_string(p_val->operator String().utf8().get_data());
        *r_valid = true;
    } else if (p_val->get_type() == Variant::FLOAT || p_val->get_type() == Variant::INT) {
        *ptr = uep::BigNumber(p_val->operator double());
        *r_valid = true;
    } else {
        *r_valid = false;
    }
}
static bool bignumber_has(const Variant *v, const StringName &p_member) { return false; }
static void bignumber_call(const Variant *v, const StringName &p_method, const Variant **p_args, int p_argcount, Variant *r_ret, Variant::CallError &r_error) {
    r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
}
static String bignumber_to_string(const Variant *v) {
    return String(VariantBigNumber::get(v)->to_string().c_str());
}
static uint32_t bignumber_hash(const Variant *v) {
    const uep::BigNumber* ptr = VariantBigNumber::get(v);
    // Hash based on scaled value's limbs
    return bigintcore_hash(v); // reusing similar logic
}
static bool bignumber_iter_init(const Variant *v, Variant::iter &r_iter, bool &r_valid) { r_valid = false; return false; }
static bool bignumber_iter_next(const Variant *v, Variant::iter &r_iter, bool &r_valid) { r_valid = false; return false; }
static void bignumber_iter_get(const Variant *v, Variant::iter &r_iter, Variant *r_ret, bool &r_valid) { r_valid = false; }

// ----------------------------------------------------------------------------
// Module entry points
// ----------------------------------------------------------------------------
void register_uep_types() {
    using namespace uep::registration;
    
    register_variant_types();
    register_classes();
    register_core_constants();
    register_math_functions();
    initialize_math_core();
    
#ifdef TOOLS_ENABLED
    // Editor-specific initialization
#endif
}

void unregister_uep_types() {
    uep::registration::cleanup_math_core();
    // Variant types are not unregistered typically
}

// ----------------------------------------------------------------------------
// GDNative exports (if building as GDNative, otherwise standard module)
// ----------------------------------------------------------------------------
#ifdef UEP_AS_GDNATIVE
extern "C" {
    void GDN_EXPORT uep_gdnative_init() {
        register_uep_types();
    }
    void GDN_EXPORT uep_gdnative_terminate() {
        unregister_uep_types();
    }
    void GDN_EXPORT uep_gdnative_singleton() {
        // Nothing extra
    }
}
#endif

// Ending of File 8 of 15 (register_types.cpp)