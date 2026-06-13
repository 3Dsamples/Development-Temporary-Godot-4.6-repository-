// genesis/repr_base.h

#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <typeinfo>
#include <cxxabi.h>
#include <memory>
#include <iostream>

namespace genesis {

//------------------------------------------------------------------------------
// ReprBase - Base class for objects that support rich string representation.
// Mirrors the Python __repr__ / __str__ conventions.
//------------------------------------------------------------------------------

class ReprBase {
public:
    virtual ~ReprBase() = default;

    // Return a string that ideally can be used to reconstruct the object (like Python __repr__).
    // Default implementation returns the class name and address.
    virtual std::string repr() const {
        std::ostringstream oss;
        oss << "<" << demangled_class_name() << " at 0x"
            << std::hex << std::noshowbase << reinterpret_cast<uintptr_t>(this)
            << std::dec << ">";
        return oss.str();
    }

    // Return a human-readable string (like Python __str__).
    // By default, falls back to repr().
    virtual std::string str() const {
        return repr();
    }

    // Helper to get demangled class name.
    std::string demangled_class_name() const {
        const char* name = typeid(*this).name();
        int status = -1;
        std::unique_ptr<char, void(*)(void*)> res {
            abi::__cxa_demangle(name, nullptr, nullptr, &status),
            std::free
        };
        return (status == 0) ? res.get() : name;
    }

    // Overload stream insertion to use str().
    friend std::ostream& operator<<(std::ostream& os, const ReprBase& obj) {
        os << obj.str();
        return os;
    }
};

//------------------------------------------------------------------------------
// A helper macro to easily implement repr() for derived classes.
// Usage: inside class definition, add:
//   GENESIS_REPR_DECL
// and in .cpp:
//   GENESIS_REPR_IMPL(ClassName, "format string", member1, member2, ...)
//------------------------------------------------------------------------------

#define GENESIS_REPR_DECL \
    virtual std::string repr() const override; \
    virtual std::string str() const override;

// Helper function to format a value with proper quoting for strings.
namespace detail {
    template<typename T>
    inline std::string repr_value(const T& val) {
        std::ostringstream oss;
        oss << val;
        return oss.str();
    }

    // Specialization for strings: add quotes and escape.
    template<>
    inline std::string repr_value(const std::string& val) {
        std::ostringstream oss;
        oss << '"';
        for (char c : val) {
            if (c == '"') oss << "\\\"";
            else if (c == '\\') oss << "\\\\";
            else if (c == '\n') oss << "\\n";
            else if (c == '\t') oss << "\\t";
            else if (c >= 32 && c < 127) oss << c;
            else oss << "\\x" << std::hex << std::setfill('0') << std::setw(2) << (int)(unsigned char)c << std::dec;
        }
        oss << '"';
        return oss.str();
    }

    template<>
    inline std::string repr_value(const char* const& val) {
        return repr_value(std::string(val));
    }
}

// Macro implementation helper: build the repr string.
// Example: GENESIS_REPR_IMPL(Vector3, "Vector3({}, {}, {})", x, y, z)
// Note: This macro is intended to be placed in a .cpp file.

#define GENESIS_REPR_IMPL(ClassName, Format, ...) \
    std::string ClassName::repr() const { \
        std::ostringstream oss; \
        oss << #ClassName << "("; \
        bool first = true; \
        (void)std::initializer_list<int>{ ( \
            (oss << (first ? (first = false, "") : ", ") << ::genesis::detail::repr_value(__VA_ARGS__)), 0)... \
        }; \
        oss << ")"; \
        return oss.str(); \
    } \
    std::string ClassName::str() const { return repr(); }

// Overload for classes that need custom formatting, we also provide a manual option.
#define GENESIS_REPR_MANUAL_DECL \
    virtual std::string repr() const override; \
    virtual std::string str() const override;

} // namespace genesis