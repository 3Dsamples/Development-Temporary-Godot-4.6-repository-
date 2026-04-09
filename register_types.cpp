#include "register_types.h"

#include "core/object/class_db.h"
#include "src/big_int_core.h"
#include "src/big_number.h"
#include "src/fixed_math_core.h"

void initialize_big_math_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_CORE) {
		// Register BigIntCore
		ClassDB::register_class<BigIntCore>();
		
		// Arithmetic Bindings
		ClassDB::bind_method(D_METHOD("add", "other"), &BigIntCore::operator+);
		ClassDB::bind_method(D_METHOD("sub", "other"), &BigIntCore::operator-);
		ClassDB::bind_method(D_METHOD("mul", "other"), &BigIntCore::operator*);
		ClassDB::bind_method(D_METHOD("div", "other"), &BigIntCore::operator/);
		ClassDB::bind_method(D_METHOD("mod", "other"), &BigIntCore::operator%);
		
		// Comparison Bindings
		ClassDB::bind_method(D_METHOD("is_equal", "other"), &BigIntCore::operator==);
		ClassDB::bind_method(D_METHOD("is_less", "other"), &BigIntCore::operator<);
		
		// Utility Bindings
		ClassDB::bind_method(D_METHOD("to_string"), &BigIntCore::to_string);
		ClassDB::bind_method(D_METHOD("to_int"), &BigIntCore::to_int64);
		ClassDB::bind_method(D_METHOD("is_zero"), &BigIntCore::is_zero);
		ClassDB::bind_method(D_METHOD("is_negative"), &BigIntCore::is_negative);
		ClassDB::bind_method(D_METHOD("get_bit_length"), &BigIntCore::get_bit_length);

		// Register BigNumber
		ClassDB::register_class<BigNumber>();
		
		// BigNumber Arithmetic
		ClassDB::bind_method(D_METHOD("add", "other"), &BigNumber::operator+);
		ClassDB::bind_method(D_METHOD("sub", "other"), &BigNumber::operator-);
		ClassDB::bind_method(D_METHOD("mul", "other"), &BigNumber::operator*);
		ClassDB::bind_method(D_METHOD("div", "other"), &BigNumber::operator/);
		
		// BigNumber Comparison
		ClassDB::bind_method(D_METHOD("is_equal", "other"), &BigNumber::operator==);
		ClassDB::bind_method(D_METHOD("is_less", "other"), &BigNumber::operator<);
		
		// BigNumber Utilities
		ClassDB::bind_method(D_METHOD("to_string"), &BigNumber::to_string);
		ClassDB::bind_method(D_METHOD("get_integer"), &BigNumber::get_integer);
		ClassDB::bind_method(D_METHOD("get_fractional"), &BigNumber::get_fractional);
		
		// Static Math methods for BigNumber
		ClassDB::bind_static_method("BigNumber", D_METHOD("abs", "val"), &BigNumber::abs);
		ClassDB::bind_static_method("BigNumber", D_METHOD("floor", "val"), &BigNumber::floor);
		ClassDB::bind_static_method("BigNumber", D_METHOD("ceil", "val"), &BigNumber::ceil);
		ClassDB::bind_static_method("BigNumber", D_METHOD("round", "val"), &BigNumber::round);
	}
}

void uninitialize_big_math_module(ModuleInitializationLevel p_level) {
	// Logic for cleaning up memory-pooled math structures if applicable
}