// include/xtu/godot/xerror.hpp
// xtensor-unified - Error handling and diagnostics for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XERROR_HPP
#define XTU_GODOT_XERROR_HPP

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xcore.hpp"

#ifdef XTU_OS_WINDOWS
#include <windows.h>
#include <dbghelp.h>
#else
#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <signal.h>
#include <unistd.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {
namespace error {

// #############################################################################
// Forward declarations
// #############################################################################
class ErrorHandler;
class CrashHandler;
class CallStack;

// #############################################################################
// Error codes (aligned with Godot's Error enum)
// #############################################################################
enum class ErrorCode : int32_t {
    OK = 0,
    FAILED = 1,
    ERR_UNAVAILABLE = 2,
    ERR_UNCONFIGURED = 3,
    ERR_UNAUTHORIZED = 4,
    ERR_PARAMETER_RANGE_ERROR = 5,
    ERR_OUT_OF_MEMORY = 6,
    ERR_FILE_NOT_FOUND = 7,
    ERR_FILE_BAD_DRIVE = 8,
    ERR_FILE_BAD_PATH = 9,
    ERR_FILE_NO_PERMISSION = 10,
    ERR_FILE_ALREADY_IN_USE = 11,
    ERR_FILE_CANT_OPEN = 12,
    ERR_FILE_CANT_WRITE = 13,
    ERR_FILE_CANT_READ = 14,
    ERR_FILE_UNRECOGNIZED = 15,
    ERR_FILE_CORRUPT = 16,
    ERR_FILE_MISSING_DEPENDENCIES = 17,
    ERR_FILE_EOF = 18,
    ERR_CANT_OPEN = 19,
    ERR_CANT_CREATE = 20,
    ERR_PARSE_ERROR = 21,
    ERR_QUERY_FAILED = 22,
    ERR_ALREADY_IN_USE = 23,
    ERR_LOCKED = 24,
    ERR_TIMEOUT = 25,
    ERR_CANT_CONNECT = 26,
    ERR_CANT_RESOLVE = 27,
    ERR_CONNECTION_ERROR = 28,
    ERR_CANT_ACQUIRE_RESOURCE = 29,
    ERR_INVALID_DATA = 30,
    ERR_INVALID_PARAMETER = 31,
    ERR_ALREADY_EXISTS = 32,
    ERR_DOES_NOT_EXIST = 33,
    ERR_DATABASE_CANT_READ = 34,
    ERR_DATABASE_CANT_WRITE = 35,
    ERR_COMPILATION_FAILED = 36,
    ERR_METHOD_NOT_FOUND = 37,
    ERR_LINK_FAILED = 38,
    ERR_SCRIPT_FAILED = 39,
    ERR_CYCLIC_LINK = 40,
    ERR_INVALID_DECLARATION = 41,
    ERR_DUPLICATE_SYMBOL = 42,
    ERR_PARSE_ERROR_TOKEN = 43,
    ERR_BUSY = 44,
    ERR_HELP = 45,
    ERR_BUG = 46,
    ERR_PRINTER_ON_FIRE = 47,
    ERR_MAX = 48
};

// #############################################################################
// Error category for grouping related errors
// #############################################################################
enum class ErrorCategory : uint8_t {
    CATEGORY_NONE = 0,
    CATEGORY_IO = 1,
    CATEGORY_MEMORY = 2,
    CATEGORY_PARAMETER = 3,
    CATEGORY_NETWORK = 4,
    CATEGORY_PARSE = 5,
    CATEGORY_COMPILATION = 6,
    CATEGORY_SCRIPT = 7,
    CATEGORY_INTERNAL = 8
};

// #############################################################################
// Error severity levels
// #############################################################################
enum class ErrorSeverity : uint8_t {
    SEVERITY_DEBUG = 0,
    SEVERITY_INFO = 1,
    SEVERITY_WARNING = 2,
    SEVERITY_ERROR = 3,
    SEVERITY_FATAL = 4
};

// #############################################################################
// Error information structure
// #############################################################################
struct ErrorInfo {
    ErrorCode code = ErrorCode::OK;
    ErrorCategory category = ErrorCategory::CATEGORY_NONE;
    ErrorSeverity severity = ErrorSeverity::SEVERITY_ERROR;
    String message;
    String file;
    String function;
    int line = 0;
    uint64_t timestamp = 0;
    std::vector<String> call_stack;
    std::map<String, Variant> context;
};

// #############################################################################
// ErrorHandler - Global error management singleton
// #############################################################################
class ErrorHandler : public Object {
    XTU_GODOT_REGISTER_CLASS(ErrorHandler, Object)

private:
    static ErrorHandler* s_singleton;
    std::vector<ErrorInfo> m_error_history;
    std::vector<std::function<void(const ErrorInfo&)>> m_error_callbacks;
    bool m_break_on_error = false;
    bool m_abort_on_fatal = true;
    size_t m_max_history = 100;
    std::mutex m_mutex;
    std::unordered_map<ErrorCode, String> m_error_messages;

public:
    static ErrorHandler* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("ErrorHandler"); }

    ErrorHandler() {
        s_singleton = this;
        initialize_error_messages();
    }

    ~ErrorHandler() { s_singleton = nullptr; }

    void set_break_on_error(bool enabled) { m_break_on_error = enabled; }
    bool get_break_on_error() const { return m_break_on_error; }

    void set_abort_on_fatal(bool enabled) { m_abort_on_fatal = enabled; }
    bool get_abort_on_fatal() const { return m_abort_on_fatal; }

    void add_error_callback(std::function<void(const ErrorInfo&)> callback) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_error_callbacks.push_back(callback);
    }

    void clear_error_callbacks() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_error_callbacks.clear();
    }

    void report_error(ErrorCode code, const String& message,
                      const char* file = nullptr, int line = 0,
                      const char* function = nullptr,
                      ErrorSeverity severity = ErrorSeverity::SEVERITY_ERROR) {
        ErrorInfo info;
        info.code = code;
        info.category = get_category_for_code(code);
        info.severity = severity;
        info.message = message;
        info.file = file ? String(file) : String();
        info.function = function ? String(function) : String();
        info.line = line;
        info.timestamp = OS::get_singleton()->get_ticks_msec();
        info.call_stack = CallStack::capture(2); // Skip the error reporting frames

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_error_history.push_back(info);
            if (m_error_history.size() > m_max_history) {
                m_error_history.erase(m_error_history.begin());
            }
        }

        // Log to console
        log_error(info);

        // Invoke callbacks
        for (const auto& cb : m_error_callbacks) {
            cb(info);
        }

        // Break if enabled
        if (m_break_on_error && severity >= ErrorSeverity::SEVERITY_ERROR) {
            debug_break();
        }

        // Abort on fatal
        if (m_abort_on_fatal && severity == ErrorSeverity::SEVERITY_FATAL) {
            std::abort();
        }
    }

    std::vector<ErrorInfo> get_error_history() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_error_history;
    }

    void clear_history() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_error_history.clear();
    }

    String get_error_message(ErrorCode code) const {
        auto it = m_error_messages.find(code);
        return it != m_error_messages.end() ? it->second : "Unknown error";
    }

    static String format_error(const ErrorInfo& info) {
        String result;
        switch (info.severity) {
            case ErrorSeverity::SEVERITY_DEBUG: result = "DEBUG: "; break;
            case ErrorSeverity::SEVERITY_INFO: result = "INFO: "; break;
            case ErrorSeverity::SEVERITY_WARNING: result = "WARNING: "; break;
            case ErrorSeverity::SEVERITY_ERROR: result = "ERROR: "; break;
            case ErrorSeverity::SEVERITY_FATAL: result = "FATAL: "; break;
        }
        result += info.message;
        if (!info.file.empty()) {
            result += " [" + info.file + ":" + String::num(info.line) + "]";
        }
        if (!info.function.empty()) {
            result += " in " + info.function + "()";
        }
        return result;
    }

    static void debug_break() {
#ifdef XTU_OS_WINDOWS
        DebugBreak();
#else
        raise(SIGTRAP);
#endif
    }

private:
    void initialize_error_messages() {
        m_error_messages[ErrorCode::OK] = "OK";
        m_error_messages[ErrorCode::FAILED] = "Generic error";
        m_error_messages[ErrorCode::ERR_UNAVAILABLE] = "Resource unavailable";
        m_error_messages[ErrorCode::ERR_UNCONFIGURED] = "Resource unconfigured";
        m_error_messages[ErrorCode::ERR_UNAUTHORIZED] = "Unauthorized";
        m_error_messages[ErrorCode::ERR_PARAMETER_RANGE_ERROR] = "Parameter out of range";
        m_error_messages[ErrorCode::ERR_OUT_OF_MEMORY] = "Out of memory";
        m_error_messages[ErrorCode::ERR_FILE_NOT_FOUND] = "File not found";
        m_error_messages[ErrorCode::ERR_FILE_CANT_OPEN] = "Cannot open file";
        m_error_messages[ErrorCode::ERR_FILE_CANT_WRITE] = "Cannot write file";
        m_error_messages[ErrorCode::ERR_PARSE_ERROR] = "Parse error";
        m_error_messages[ErrorCode::ERR_METHOD_NOT_FOUND] = "Method not found";
        m_error_messages[ErrorCode::ERR_COMPILATION_FAILED] = "Compilation failed";
        m_error_messages[ErrorCode::ERR_SCRIPT_FAILED] = "Script failed";
        m_error_messages[ErrorCode::ERR_INVALID_DATA] = "Invalid data";
        m_error_messages[ErrorCode::ERR_INVALID_PARAMETER] = "Invalid parameter";
        m_error_messages[ErrorCode::ERR_TIMEOUT] = "Operation timed out";
        m_error_messages[ErrorCode::ERR_BUG] = "Internal bug";
    }

    static ErrorCategory get_category_for_code(ErrorCode code) {
        int c = static_cast<int>(code);
        if (c >= static_cast<int>(ErrorCode::ERR_FILE_NOT_FOUND) &&
            c <= static_cast<int>(ErrorCode::ERR_FILE_EOF)) {
            return ErrorCategory::CATEGORY_IO;
        }
        if (c == static_cast<int>(ErrorCode::ERR_OUT_OF_MEMORY)) {
            return ErrorCategory::CATEGORY_MEMORY;
        }
        if (c >= static_cast<int>(ErrorCode::ERR_CANT_CONNECT) &&
            c <= static_cast<int>(ErrorCode::ERR_CONNECTION_ERROR)) {
            return ErrorCategory::CATEGORY_NETWORK;
        }
        if (c == static_cast<int>(ErrorCode::ERR_PARSE_ERROR) ||
            c == static_cast<int>(ErrorCode::ERR_PARSE_ERROR_TOKEN)) {
            return ErrorCategory::CATEGORY_PARSE;
        }
        if (c >= static_cast<int>(ErrorCode::ERR_COMPILATION_FAILED) &&
            c <= static_cast<int>(ErrorCode::ERR_DUPLICATE_SYMBOL)) {
            return ErrorCategory::CATEGORY_COMPILATION;
        }
        if (c >= static_cast<int>(ErrorCode::ERR_SCRIPT_FAILED) &&
            c <= static_cast<int>(ErrorCode::ERR_INVALID_DECLARATION)) {
            return ErrorCategory::CATEGORY_SCRIPT;
        }
        return ErrorCategory::CATEGORY_NONE;
    }

    void log_error(const ErrorInfo& info) {
        String formatted = format_error(info);
        fprintf(stderr, "%s\n", formatted.utf8());
        if (!info.call_stack.empty()) {
            fprintf(stderr, "Call Stack:\n");
            for (const auto& frame : info.call_stack) {
                fprintf(stderr, "  %s\n", frame.utf8());
            }
        }
    }
};

// #############################################################################
// CallStack - Stack trace capture and formatting
// #############################################################################
class CallStack {
public:
    struct Frame {
        String function;
        String file;
        int line = 0;
        void* address = nullptr;
    };

    static std::vector<String> capture(int skip_frames = 0) {
        std::vector<String> result;
        std::vector<Frame> frames = get_frames(skip_frames + 1);
        for (const auto& frame : frames) {
            String str = frame.function;
            if (!frame.file.empty()) {
                str += " (" + frame.file + ":" + String::num(frame.line) + ")";
            }
            result.push_back(str);
        }
        return result;
    }

    static std::vector<Frame> get_frames(int skip_frames = 0) {
        std::vector<Frame> result;
#ifdef XTU_OS_WINDOWS
        void* stack[64];
        HANDLE process = GetCurrentProcess();
        SymInitialize(process, nullptr, TRUE);
        USHORT frames = CaptureStackBackTrace(skip_frames, 64, stack, nullptr);

        for (USHORT i = 0; i < frames; ++i) {
            Frame frame;
            frame.address = stack[i];

            char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
            SYMBOL_INFO* symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = MAX_SYM_NAME;

            if (SymFromAddr(process, (DWORD64)stack[i], nullptr, symbol)) {
                frame.function = String(symbol->Name);
            } else {
                frame.function = "<unknown>";
            }

            IMAGEHLP_LINE64 line;
            line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
            DWORD displacement;
            if (SymGetLineFromAddr64(process, (DWORD64)stack[i], &displacement, &line)) {
                frame.file = String(line.FileName);
                frame.line = static_cast<int>(line.LineNumber);
            }

            result.push_back(frame);
        }
        SymCleanup(process);
#else
        void* buffer[64];
        int frames = backtrace(buffer, 64);
        char** symbols = backtrace_symbols(buffer, frames);

        for (int i = skip_frames; i < frames; ++i) {
            Frame frame;
            frame.address = buffer[i];

            if (symbols) {
                String sym(symbols[i]);
                frame.function = demangle(sym);
                result.push_back(frame);
            }
        }
        free(symbols);
#endif
        return result;
    }

    static String demangle(const String& symbol) {
#ifdef XTU_OS_WINDOWS
        return symbol;
#else
        // Parse and demangle C++ symbol
        size_t start = symbol.find('(');
        size_t end = symbol.find('+');
        if (start != String::npos && end != String::npos) {
            String mangled = symbol.substr(start + 1, end - start - 1);
            int status;
            char* demangled = abi::__cxa_demangle(mangled.utf8(), nullptr, nullptr, &status);
            if (status == 0 && demangled) {
                String result = symbol.substr(0, start + 1) + String(demangled) + symbol.substr(end);
                free(demangled);
                return result;
            }
        }
        return symbol;
#endif
    }
};

// #############################################################################
// CrashHandler - Native crash handling and minidump generation
// #############################################################################
class CrashHandler {
private:
    static CrashHandler* s_instance;
    std::function<void(const String&)> m_crash_callback;
    String m_dump_path;
    bool m_installed = false;
    std::mutex m_mutex;

#ifdef XTU_OS_WINDOWS
    static LONG WINAPI exception_handler(EXCEPTION_POINTERS* exception_info) {
        return s_instance->handle_exception(exception_info);
    }
#else
    static void signal_handler(int sig, siginfo_t* info, void* context) {
        s_instance->handle_signal(sig, info, context);
    }
#endif

public:
    static CrashHandler* get_singleton() {
        if (!s_instance) s_instance = new CrashHandler();
        return s_instance;
    }

    void install(const String& dump_path = "") {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_installed) return;

        m_dump_path = dump_path.empty() ? OS::get_singleton()->get_user_data_dir() : dump_path;

#ifdef XTU_OS_WINDOWS
        SetUnhandledExceptionFilter(exception_handler);
#else
        struct sigaction sa;
        sa.sa_sigaction = signal_handler;
        sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
        sigemptyset(&sa.sa_mask);

        sigaction(SIGSEGV, &sa, nullptr);
        sigaction(SIGABRT, &sa, nullptr);
        sigaction(SIGFPE, &sa, nullptr);
        sigaction(SIGILL, &sa, nullptr);
        sigaction(SIGBUS, &sa, nullptr);
#endif
        m_installed = true;
    }

    void set_crash_callback(std::function<void(const String&)> callback) {
        m_crash_callback = callback;
    }

    String generate_crash_report(const String& context = "") {
        String report;
        report += "=== Godot Crash Report ===\n";
        report += "Version: 4.6.0\n";
        report += "OS: " + OS::get_singleton()->get_name() + " " + OS::get_singleton()->get_version() + "\n";
        report += "Time: " + String::num(OS::get_singleton()->get_unix_time()) + "\n";
        report += "Context: " + context + "\n\n";

        report += "Call Stack:\n";
        auto frames = CallStack::capture();
        for (const auto& frame : frames) {
            report += "  " + frame + "\n";
        }

        report += "\nError History:\n";
        auto history = ErrorHandler::get_singleton()->get_error_history();
        for (const auto& err : history) {
            report += "  " + ErrorHandler::format_error(err) + "\n";
        }

        return report;
    }

    void save_minidump(void* exception_pointers = nullptr) {
#ifdef XTU_OS_WINDOWS
        if (!exception_pointers) return;

        String dump_file = m_dump_path + "/crash_" +
                          String::num(OS::get_singleton()->get_unix_time()) + ".dmp";

        HANDLE file = CreateFileA(dump_file.utf8(), GENERIC_WRITE, 0, nullptr,
                                   CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file != INVALID_HANDLE_VALUE) {
            MINIDUMP_EXCEPTION_INFORMATION mei;
            mei.ThreadId = GetCurrentThreadId();
            mei.ExceptionPointers = (EXCEPTION_POINTERS*)exception_pointers;
            mei.ClientPointers = TRUE;

            MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), file,
                              MiniDumpNormal, &mei, nullptr, nullptr);
            CloseHandle(file);
        }
#endif
    }

private:
#ifdef XTU_OS_WINDOWS
    LONG handle_exception(EXCEPTION_POINTERS* exception_info) {
        String context = "Exception code: 0x" + String::num_hex(exception_info->ExceptionRecord->ExceptionCode);
        String report = generate_crash_report(context);
        save_minidump(exception_info);

        if (m_crash_callback) m_crash_callback(report);

        fprintf(stderr, "%s\n", report.utf8());
        return EXCEPTION_EXECUTE_HANDLER;
    }
#else
    void handle_signal(int sig, siginfo_t* info, void* context) {
        String sig_name;
        switch (sig) {
            case SIGSEGV: sig_name = "SIGSEGV (Segmentation fault)"; break;
            case SIGABRT: sig_name = "SIGABRT (Abort)"; break;
            case SIGFPE: sig_name = "SIGFPE (Floating point exception)"; break;
            case SIGILL: sig_name = "SIGILL (Illegal instruction)"; break;
            case SIGBUS: sig_name = "SIGBUS (Bus error)"; break;
            default: sig_name = "Unknown signal " + String::num(sig);
        }

        String report = generate_crash_report(sig_name);
        if (m_crash_callback) m_crash_callback(report);
        fprintf(stderr, "%s\n", report.utf8());

        // Re-raise signal with default handler
        signal(sig, SIG_DFL);
        raise(sig);
    }
#endif
};

CrashHandler* CrashHandler::s_instance = nullptr;

// #############################################################################
// Error macros for convenient usage
// #############################################################################
#define XTU_ERR_FAIL() \
    return ::xtu::godot::error::ErrorCode::FAILED

#define XTU_ERR_FAIL_V(value) \
    return value

#define XTU_ERR_FAIL_MSG(msg) \
    do { \
        ::xtu::godot::error::ErrorHandler::get_singleton()->report_error( \
            ::xtu::godot::error::ErrorCode::FAILED, msg, __FILE__, __LINE__, __FUNCTION__); \
        return ::xtu::godot::error::ErrorCode::FAILED; \
    } while(0)

#define XTU_ERR_FAIL_COND(cond) \
    do { \
        if (cond) { \
            ::xtu::godot::error::ErrorHandler::get_singleton()->report_error( \
                ::xtu::godot::error::ErrorCode::FAILED, "Condition failed: " #cond, \
                __FILE__, __LINE__, __FUNCTION__); \
            return ::xtu::godot::error::ErrorCode::FAILED; \
        } \
    } while(0)

#define XTU_ERR_FAIL_COND_V(cond, ret) \
    do { \
        if (cond) { \
            ::xtu::godot::error::ErrorHandler::get_singleton()->report_error( \
                ::xtu::godot::error::ErrorCode::FAILED, "Condition failed: " #cond, \
                __FILE__, __LINE__, __FUNCTION__); \
            return ret; \
        } \
    } while(0)

#define XTU_ERR_FAIL_NULL(cond) \
    do { \
        if (cond) { \
            ::xtu::godot::error::ErrorHandler::get_singleton()->report_error( \
                ::xtu::godot::error::ErrorCode::FAILED, "Condition failed: " #cond, \
                __FILE__, __LINE__, __FUNCTION__); \
            return nullptr; \
        } \
    } while(0)

#define XTU_ERR_FAIL_INDEX(idx, size) \
    do { \
        if (idx < 0 || idx >= size) { \
            ::xtu::godot::error::ErrorHandler::get_singleton()->report_error( \
                ::xtu::godot::error::ErrorCode::ERR_PARAMETER_RANGE_ERROR, \
                "Index out of range", __FILE__, __LINE__, __FUNCTION__); \
            return ::xtu::godot::error::ErrorCode::ERR_PARAMETER_RANGE_ERROR; \
        } \
    } while(0)

#define XTU_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            ::xtu::godot::error::ErrorHandler::get_singleton()->report_error( \
                ::xtu::godot::error::ErrorCode::ERR_BUG, \
                "Assertion failed: " #cond, __FILE__, __LINE__, __FUNCTION__, \
                ::xtu::godot::error::ErrorSeverity::SEVERITY_FATAL); \
        } \
    } while(0)

#define XTU_ASSERT_MSG(cond, msg) \
    do { \
        if (!(cond)) { \
            ::xtu::godot::error::ErrorHandler::get_singleton()->report_error( \
                ::xtu::godot::error::ErrorCode::ERR_BUG, \
                "Assertion failed: " #cond " - " msg, __FILE__, __LINE__, __FUNCTION__, \
                ::xtu::godot::error::ErrorSeverity::SEVERITY_FATAL); \
        } \
    } while(0)

#define XTU_WARN(cond) \
    do { \
        if (!(cond)) { \
            ::xtu::godot::error::ErrorHandler::get_singleton()->report_error( \
                ::xtu::godot::error::ErrorCode::OK, \
                "Warning: " #cond, __FILE__, __LINE__, __FUNCTION__, \
                ::xtu::godot::error::ErrorSeverity::SEVERITY_WARNING); \
        } \
    } while(0)

#define XTU_CRASH_NOW() \
    do { \
        ::xtu::godot::error::ErrorHandler::get_singleton()->report_error( \
            ::xtu::godot::error::ErrorCode::ERR_BUG, \
            "Intentional crash", __FILE__, __LINE__, __FUNCTION__, \
            ::xtu::godot::error::ErrorSeverity::SEVERITY_FATAL); \
        ::xtu::godot::error::ErrorHandler::debug_break(); \
    } while(0)

} // namespace error

// Bring into main namespace
using error::ErrorHandler;
using error::CrashHandler;
using error::CallStack;
using error::ErrorCode;
using error::ErrorCategory;
using error::ErrorSeverity;
using error::ErrorInfo;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XERROR_HPP