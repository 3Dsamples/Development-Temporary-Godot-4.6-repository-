// genesis/typing.cpp

#include "genesis/typing.h"
#include <cxxabi.h>
#include <memory>
#include <cstdlib>

namespace genesis {
namespace typing {

//------------------------------------------------------------------------------
// Demangle implementation (used by type_name<T>())
//------------------------------------------------------------------------------
std::string demangle_type_name(const std::type_info& ti) {
    int status = -1;
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(ti.name(), nullptr, nullptr, &status),
        std::free
    };
    return (status == 0) ? res.get() : ti.name();
}

// Explicit instantiation of demangle_type_name for common types (not needed,
// but ensures function is compiled into library).
template std::string type_name<int>();
template std::string type_name<float>();
template std::string type_name<double>();
template std::string type_name<std::string>();

//------------------------------------------------------------------------------
// Range iterator implementation details (mostly inline, but any non-inline
// could go here - all are defined inline in header)
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Enumerate/Zip implementation (all inline templates, nothing to define here)
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Optional: Provide a runtime type check helper that doesn't require RTTI
// if disabled (not implemented here as we assume RTTI enabled)
//------------------------------------------------------------------------------

} // namespace typing
} // namespace genesis