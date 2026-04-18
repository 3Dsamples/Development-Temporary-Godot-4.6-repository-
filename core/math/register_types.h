// src/register_types.h
#ifndef UEP_REGISTER_TYPES_H
#define UEP_REGISTER_TYPES_H

#include "core/object/class_db.h"
#include "core/variant/variant.h"
#include "core/string/ustring.h"
#include "core/io/resource.h"

// Forward declarations of our math classes
namespace uep {
    class BigIntCore;
    class FixedMathCore;
    class BigNumber;
}

// ----------------------------------------------------------------------------
// Module initialization and cleanup functions
// These are called by Godot during module registration.
// ----------------------------------------------------------------------------
void register_uep_types();
void unregister_uep_types();

// ----------------------------------------------------------------------------
// Internal registration helpers
// ----------------------------------------------------------------------------
namespace uep {
namespace registration {

// Register all classes with ClassDB
void register_classes();

// Register variant types for use in GDScript and Variant system
void register_variant_types();

// Register core constants (PI, E, etc.) as global constants
void register_core_constants();

// Register math functions as global utilities (sin, cos, etc.)
void register_math_functions();

// Initialize the UEP math core (allocates global tables, etc.)
void initialize_math_core();

// Cleanup the UEP math core
void cleanup_math_core();

// Register BigIntCore with Godot's ClassDB
void register_bigintcore_class();

// Register FixedMathCore with Godot's ClassDB
void register_fixedmathcore_class();

// Register BigNumber with Godot's ClassDB
void register_bignumber_class();

// Helper to bind a static math function to GDScript
template<typename Ret, typename... Args>
void bind_static_math_function(const char* p_name, Ret(*p_func)(Args...));

// Helper to bind a method to a class
template<typename T, typename Ret, typename... Args>
void bind_method(const char* p_name, Ret(T::*p_method)(Args...));

// Helper to bind a const method
template<typename T, typename Ret, typename... Args>
void bind_const_method(const char* p_name, Ret(T::*p_method)(Args...) const);

} // namespace registration
} // namespace uep

// ----------------------------------------------------------------------------
// Xtensor integration registration (if enabled)
// ----------------------------------------------------------------------------
#ifdef UEP_USE_XTENSOR
void register_xtensor_bindings();
#endif

// ----------------------------------------------------------------------------
// Module entry points (called by Godot engine)
// ----------------------------------------------------------------------------
extern "C" {
    void GDN_EXPORT uep_gdnative_init();
    void GDN_EXPORT uep_gdnative_terminate();
    void GDN_EXPORT uep_gdnative_singleton();
}

// ----------------------------------------------------------------------------
// Registration implementation details (inline helper functions)
// ----------------------------------------------------------------------------
namespace uep {
namespace registration {

template<typename Ret, typename... Args>
void bind_static_math_function(const char* p_name, Ret(*p_func)(Args...)) {
    // Create a callable wrapper that can be bound to ClassDB
    // Since ClassDB expects methods bound to a class, we'll bind them as
    // static methods of a utility singleton class.
    // This will be implemented in register_types.cpp.
}

template<typename T, typename Ret, typename... Args>
void bind_method(const char* p_name, Ret(T::*p_method)(Args...)) {
    ClassDB::bind_method(p_name, p_method);
}

template<typename T, typename Ret, typename... Args>
void bind_const_method(const char* p_name, Ret(T::*p_method)(Args...) const) {
    ClassDB::bind_method(p_name, p_method);
}

} // namespace registration
} // namespace uep

// ----------------------------------------------------------------------------
// Macro to simplify binding of math functions
// ----------------------------------------------------------------------------
#define UEP_BIND_MATH_FUNC(m_func) \
    ClassDB::bind_static_method("UEPMath", D_METHOD(#m_func), &uep::BigNumber::m_func);

// ----------------------------------------------------------------------------
// Variant type IDs (assigned by Godot during registration)
// ----------------------------------------------------------------------------
extern int VARIANT_TYPE_BIGINTCORE;
extern int VARIANT_TYPE_FIXEDMATHCORE;
extern int VARIANT_TYPE_BIGNUMBER;

// ----------------------------------------------------------------------------
// Singleton class that provides global access to UEP math functions
// ----------------------------------------------------------------------------
class UEPMath : public Object {
    GDCLASS(UEPMath, Object);

protected:
    static UEPMath* singleton;
    static void _bind_methods();

public:
    static UEPMath* get_singleton() { return singleton; }
    UEPMath();
    ~UEPMath();

    // These methods forward to the BigNumber static functions
    static BigNumber sin(const BigNumber& x) { return BigNumber::sin(x); }
    static BigNumber cos(const BigNumber& x) { return BigNumber::cos(x); }
    static BigNumber tan(const BigNumber& x) { return BigNumber::tan(x); }
    static BigNumber asin(const BigNumber& x) { return BigNumber::asin(x); }
    static BigNumber acos(const BigNumber& x) { return BigNumber::acos(x); }
    static BigNumber atan(const BigNumber& x) { return BigNumber::atan(x); }
    static BigNumber atan2(const BigNumber& y, const BigNumber& x) { return BigNumber::atan2(y, x); }
    static BigNumber exp(const BigNumber& x) { return BigNumber::exp(x); }
    static BigNumber log(const BigNumber& x) { return BigNumber::log(x); }
    static BigNumber log10(const BigNumber& x) { return BigNumber::log10(x); }
    static BigNumber pow(const BigNumber& base, const BigNumber& exp) { return BigNumber::pow(base, exp); }
    static BigNumber sqrt(const BigNumber& x) { return BigNumber::sqrt(x); }
    
    // Constants
    static BigNumber pi() { return BigNumber::pi(); }
    static BigNumber e() { return BigNumber::e(); }
    static BigNumber ln2() { return BigNumber::ln2(); }
    
    // Utility functions
    static BigNumber abs(const BigNumber& x) { return x.abs(); }
    static BigNumber floor(const BigNumber& x) { return x.floor(); }
    static BigNumber ceil(const BigNumber& x) { return x.ceil(); }
    static BigNumber round(const BigNumber& x) { return x.round(); }
    static BigNumber frac(const BigNumber& x) { return x.frac(); }
    static BigNumber sign(const BigNumber& x) { return uep::sign(x); }
    static BigNumber min(const BigNumber& a, const BigNumber& b) { return uep::min(a, b); }
    static BigNumber max(const BigNumber& a, const BigNumber& b) { return uep::max(a, b); }
    static BigNumber clamp(const BigNumber& val, const BigNumber& lo, const BigNumber& hi) { return uep::clamp(val, lo, hi); }
    static BigNumber lerp(const BigNumber& a, const BigNumber& b, const BigNumber& t) { return uep::lerp(a, b, t); }
    
    // Conversion
    static String to_string(const BigNumber& x) { return String(x.to_string().c_str()); }
    static BigNumber from_string(const String& str) { return BigNumber::from_string(str.utf8().get_data()); }
    static BigNumber from_int(int64_t val) { return BigNumber(val); }
    static BigNumber from_float(double val) { return BigNumber(val); }
    static int64_t to_int(const BigNumber& x) { return x.to_int64(); }
    static double to_float(const BigNumber& x) { return x.to_double(); }
};

#endif // UEP_REGISTER_TYPES_H
// Ending of File 7 of 15 (register_types.h)