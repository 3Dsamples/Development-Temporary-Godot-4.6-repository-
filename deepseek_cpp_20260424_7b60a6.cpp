// genesis/repr_base.cpp

#include "genesis/repr_base.h"

// This file contains non-inline implementations and explicit instantiations
// for the ReprBase system. Since most logic is template/inline, we add
// a small utility function for demangling that might be needed externally.

namespace genesis {

// Explicit instantiation of repr_value for common types
namespace detail {
    template std::string repr_value<int>(const int&);
    template std::string repr_value<long>(const long&);
    template std::string repr_value<long long>(const long long&);
    template std::string repr_value<unsigned int>(const unsigned int&);
    template std::string repr_value<unsigned long>(const unsigned long&);
    template std::string repr_value<unsigned long long>(const unsigned long long&);
    template std::string repr_value<float>(const float&);
    template std::string repr_value<double>(const double&);
    template std::string repr_value<bool>(const bool&);
    template std::string repr_value<std::string>(const std::string&);
    template std::string repr_value<const char*>(const char* const&);
}

// A convenience function to get demangled name of any type (not just *this)
std::string demangle_type_name(const std::type_info& ti) {
    int status = -1;
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(ti.name(), nullptr, nullptr, &status),
        std::free
    };
    return (status == 0) ? res.get() : ti.name();
}

// For MSVC compatibility, we could add an alternative implementation here
// but we keep it simple for GCC/Clang.

} // namespace genesis