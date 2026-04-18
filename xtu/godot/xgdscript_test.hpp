// include/xtu/godot/xgdscript_test.hpp
// xtensor-unified - GDScript Testing Framework for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XGDSCRIPT_TEST_HPP
#define XTU_GODOT_XGDSCRIPT_TEST_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/godot/xvariant.hpp"
#include "xtu/godot/xclassdb.hpp"
#include "xtu/godot/xgdscript.hpp"
#include "xtu/io/xio_json.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace gdscript {

// #############################################################################
// Forward declarations
// #############################################################################
class GDScriptTestRunner;
class GDScriptTestSuite;
class GDScriptTestCase;
class GDScriptAssertion;

// #############################################################################
// Test status
// #############################################################################
enum class TestStatus : uint8_t {
    STATUS_PENDING = 0,
    STATUS_RUNNING = 1,
    STATUS_PASSED = 2,
    STATUS_FAILED = 3,
    STATUS_SKIPPED = 4,
    STATUS_ERROR = 5
};

// #############################################################################
// Test assertion result
// #############################################################################
struct TestAssertionResult {
    bool passed = false;
    String message;
    String expected;
    String actual;
    String file;
    int line = 0;
};

// #############################################################################
// Test case result
// #############################################################################
struct TestCaseResult {
    String name;
    TestStatus status = TestStatus::STATUS_PENDING;
    double duration_ms = 0.0;
    std::vector<TestAssertionResult> assertions;
    String error_message;
    String error_stack;
};

// #############################################################################
// Test suite result
// #############################################################################
struct TestSuiteResult {
    String name;
    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;
    int skipped_tests = 0;
    double duration_ms = 0.0;
    std::vector<TestCaseResult> test_cases;
};

// #############################################################################
// GDScriptTestCase - Individual test case
// #############################################################################
class GDScriptTestCase : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(GDScriptTestCase, RefCounted)

private:
    String m_name;
    std::function<void()> m_test_func;
    bool m_skip = false;
    String m_skip_reason;
    int m_timeout_seconds = 30;

public:
    static StringName get_class_static() { return StringName("GDScriptTestCase"); }

    void set_name(const String& name) { m_name = name; }
    String get_name() const { return m_name; }

    void set_test_func(std::function<void()> func) { m_test_func = func; }
    void set_skip(bool skip, const String& reason = "") { m_skip = skip; m_skip_reason = reason; }
    bool is_skipped() const { return m_skip; }
    String get_skip_reason() const { return m_skip_reason; }

    void set_timeout(int seconds) { m_timeout_seconds = seconds; }
    int get_timeout() const { return m_timeout_seconds; }

    TestCaseResult run() {
        TestCaseResult result;
        result.name = m_name;

        if (m_skip) {
            result.status = TestStatus::STATUS_SKIPPED;
            return result;
        }

        auto start = std::chrono::high_resolution_clock::now();
        result.status = TestStatus::STATUS_RUNNING;

        try {
            if (m_test_func) {
                m_test_func();
            }
            result.status = TestStatus::STATUS_PASSED;
        } catch (const std::exception& e) {
            result.status = TestStatus::STATUS_ERROR;
            result.error_message = String(e.what());
        } catch (...) {
            result.status = TestStatus::STATUS_ERROR;
            result.error_message = "Unknown error";
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Collect assertions from global assertion tracker
        result.assertions = GDScriptAssertion::get_singleton()->flush_assertions();

        // Count failures
        for (const auto& a : result.assertions) {
            if (!a.passed && result.status == TestStatus::STATUS_PASSED) {
                result.status = TestStatus::STATUS_FAILED;
            }
        }

        return result;
    }
};

// #############################################################################
// GDScriptTestSuite - Collection of test cases
// #############################################################################
class GDScriptTestSuite : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(GDScriptTestSuite, RefCounted)

private:
    String m_name;
    String m_description;
    std::vector<Ref<GDScriptTestCase>> m_test_cases;
    std::unordered_map<String, size_t> m_test_index;

public:
    static StringName get_class_static() { return StringName("GDScriptTestSuite"); }

    void set_name(const String& name) { m_name = name; }
    String get_name() const { return m_name; }

    void set_description(const String& desc) { m_description = desc; }
    String get_description() const { return m_description; }

    void add_test_case(const Ref<GDScriptTestCase>& test_case) {
        m_test_index[test_case->get_name()] = m_test_cases.size();
        m_test_cases.push_back(test_case);
    }

    Ref<GDScriptTestCase> get_test_case(const String& name) const {
        auto it = m_test_index.find(name);
        return it != m_test_index.end() ? m_test_cases[it->second] : Ref<GDScriptTestCase>();
    }

    const std::vector<Ref<GDScriptTestCase>>& get_test_cases() const { return m_test_cases; }
    size_t get_test_count() const { return m_test_cases.size(); }

    TestSuiteResult run(bool parallel = true) {
        TestSuiteResult result;
        result.name = m_name;
        result.total_tests = static_cast<int>(m_test_cases.size());

        auto start = std::chrono::high_resolution_clock::now();

        if (parallel) {
            std::vector<TestCaseResult> results(m_test_cases.size());
            parallel::parallel_for(0, m_test_cases.size(), [&](size_t i) {
                results[i] = m_test_cases[i]->run();
            });
            result.test_cases = std::move(results);
        } else {
            for (const auto& tc : m_test_cases) {
                result.test_cases.push_back(tc->run());
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Count results
        for (const auto& tc : result.test_cases) {
            switch (tc.status) {
                case TestStatus::STATUS_PASSED: ++result.passed_tests; break;
                case TestStatus::STATUS_FAILED: ++result.failed_tests; break;
                case TestStatus::STATUS_SKIPPED: ++result.skipped_tests; break;
                default: break;
            }
        }

        return result;
    }
};

// #############################################################################
// GDScriptAssertion - Assertion tracking singleton
// #############################################################################
class GDScriptAssertion : public Object {
    XTU_GODOT_REGISTER_CLASS(GDScriptAssertion, Object)

private:
    static GDScriptAssertion* s_singleton;
    std::vector<TestAssertionResult> m_assertions;
    mutable std::mutex m_mutex;

public:
    static GDScriptAssertion* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("GDScriptAssertion"); }

    GDScriptAssertion() { s_singleton = this; }
    ~GDScriptAssertion() { s_singleton = nullptr; }

    void assert_true(bool condition, const String& message = "",
                     const char* file = nullptr, int line = 0) {
        std::lock_guard<std::mutex> lock(m_mutex);
        TestAssertionResult result;
        result.passed = condition;
        result.message = message.empty() ? "Expected true, got false" : message;
        result.expected = "true";
        result.actual = "false";
        result.file = file ? String(file) : String();
        result.line = line;
        m_assertions.push_back(result);

        if (!condition) {
            emit_signal("assertion_failed", result.message, result.file, result.line);
        }
    }

    void assert_false(bool condition, const String& message = "",
                      const char* file = nullptr, int line = 0) {
        assert_true(!condition, message.empty() ? "Expected false, got true" : message, file, line);
    }

    void assert_equal(const Variant& expected, const Variant& actual,
                      const String& message = "", const char* file = nullptr, int line = 0) {
        bool equal = (expected == actual);
        std::lock_guard<std::mutex> lock(m_mutex);
        TestAssertionResult result;
        result.passed = equal;
        result.message = message.empty() ? "Values not equal" : message;
        result.expected = variant_to_string(expected);
        result.actual = variant_to_string(actual);
        result.file = file ? String(file) : String();
        result.line = line;
        m_assertions.push_back(result);
    }

    void assert_not_equal(const Variant& expected, const Variant& actual,
                          const String& message = "", const char* file = nullptr, int line = 0) {
        bool not_equal = (expected != actual);
        std::lock_guard<std::mutex> lock(m_mutex);
        TestAssertionResult result;
        result.passed = not_equal;
        result.message = message.empty() ? "Values are equal" : message;
        result.expected = "not " + variant_to_string(expected);
        result.actual = variant_to_string(actual);
        result.file = file ? String(file) : String();
        result.line = line;
        m_assertions.push_back(result);
    }

    void assert_null(const Variant& value, const String& message = "",
                     const char* file = nullptr, int line = 0) {
        assert_true(value.is_nil(), message.empty() ? "Expected null" : message, file, line);
    }

    void assert_not_null(const Variant& value, const String& message = "",
                         const char* file = nullptr, int line = 0) {
        assert_true(!value.is_nil(), message.empty() ? "Expected not null" : message, file, line);
    }

    void assert_approx(float expected, float actual, float tolerance = 0.0001f,
                       const String& message = "", const char* file = nullptr, int line = 0) {
        bool approx = std::abs(expected - actual) <= tolerance;
        std::lock_guard<std::mutex> lock(m_mutex);
        TestAssertionResult result;
        result.passed = approx;
        result.message = message.empty() ? "Values not approximately equal" : message;
        result.expected = String::num(expected) + " ± " + String::num(tolerance);
        result.actual = String::num(actual);
        result.file = file ? String(file) : String();
        result.line = line;
        m_assertions.push_back(result);
    }

    void assert_throws(const std::function<void()>& func, const String& message = "",
                       const char* file = nullptr, int line = 0) {
        bool threw = false;
        try {
            func();
        } catch (...) {
            threw = true;
        }
        assert_true(threw, message.empty() ? "Expected exception was not thrown" : message, file, line);
    }

    std::vector<TestAssertionResult> flush_assertions() {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto result = std::move(m_assertions);
        m_assertions.clear();
        return result;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_assertions.clear();
    }

private:
    static String variant_to_string(const Variant& v) {
        if (v.is_nil()) return "null";
        if (v.is_bool()) return v.as<bool>() ? "true" : "false";
        if (v.is_num()) return String::num(v.as<double>());
        if (v.is_string()) return "\"" + v.as<String>() + "\"";
        return v.as<String>();
    }
};

// #############################################################################
// GDScriptTestRunner - Main test runner
// #############################################################################
class GDScriptTestRunner : public Object {
    XTU_GODOT_REGISTER_CLASS(GDScriptTestRunner, Object)

private:
    static GDScriptTestRunner* s_singleton;
    std::vector<Ref<GDScriptTestSuite>> m_suites;
    std::unordered_map<String, Ref<GDScriptTestSuite>> m_suite_map;
    bool m_running = false;
    std::mutex m_mutex;

public:
    static GDScriptTestRunner* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("GDScriptTestRunner"); }

    GDScriptTestRunner() { s_singleton = this; }
    ~GDScriptTestRunner() { s_singleton = nullptr; }

    void add_suite(const Ref<GDScriptTestSuite>& suite) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_suite_map[suite->get_name()] = suite;
        m_suites.push_back(suite);
    }

    Ref<GDScriptTestSuite> get_suite(const String& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_suite_map.find(name);
        return it != m_suite_map.end() ? it->second : Ref<GDScriptTestSuite>();
    }

    std::vector<Ref<GDScriptTestSuite>> get_suites() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_suites;
    }

    std::vector<TestSuiteResult> run_all(bool parallel = true) {
        m_running = true;
        std::vector<TestSuiteResult> results;
        results.reserve(m_suites.size());

        for (const auto& suite : m_suites) {
            results.push_back(suite->run(parallel));
            emit_signal("suite_completed", suite->get_name());
        }

        m_running = false;
        emit_signal("all_completed");
        return results;
    }

    TestSuiteResult run_suite(const String& name, bool parallel = true) {
        auto suite = get_suite(name);
        if (suite.is_valid()) {
            return suite->run(parallel);
        }
        return TestSuiteResult();
    }

    bool is_running() const { return m_running; }

    String generate_report(const std::vector<TestSuiteResult>& results) const {
        io::json::JsonValue json;
        io::json::JsonValue suites_arr;

        int total_tests = 0;
        int total_passed = 0;
        int total_failed = 0;
        int total_skipped = 0;
        double total_duration = 0.0;

        for (const auto& suite : results) {
            io::json::JsonValue suite_json;
            suite_json["name"] = io::json::JsonValue(suite.name.to_std_string());
            suite_json["total"] = io::json::JsonValue(suite.total_tests);
            suite_json["passed"] = io::json::JsonValue(suite.passed_tests);
            suite_json["failed"] = io::json::JsonValue(suite.failed_tests);
            suite_json["skipped"] = io::json::JsonValue(suite.skipped_tests);
            suite_json["duration_ms"] = io::json::JsonValue(suite.duration_ms);

            io::json::JsonValue cases_arr;
            for (const auto& tc : suite.test_cases) {
                io::json::JsonValue case_json;
                case_json["name"] = io::json::JsonValue(tc.name.to_std_string());
                case_json["status"] = io::json::JsonValue(static_cast<int>(tc.status));
                case_json["duration_ms"] = io::json::JsonValue(tc.duration_ms);
                if (!tc.error_message.empty()) {
                    case_json["error"] = io::json::JsonValue(tc.error_message.to_std_string());
                }
                cases_arr.as_array().push_back(case_json);
            }
            suite_json["cases"] = cases_arr;
            suites_arr.as_array().push_back(suite_json);

            total_tests += suite.total_tests;
            total_passed += suite.passed_tests;
            total_failed += suite.failed_tests;
            total_skipped += suite.skipped_tests;
            total_duration += suite.duration_ms;
        }

        json["total_tests"] = io::json::JsonValue(total_tests);
        json["total_passed"] = io::json::JsonValue(total_passed);
        json["total_failed"] = io::json::JsonValue(total_failed);
        json["total_skipped"] = io::json::JsonValue(total_skipped);
        json["total_duration_ms"] = io::json::JsonValue(total_duration);
        json["suites"] = suites_arr;

        return json.dump(2).c_str();
    }

    void discover_tests(const String& path) {
        // Scan directory for test files (*.test.gd)
        Ref<DirAccess> dir = DirAccess::open(path);
        if (!dir.is_valid()) return;

        dir->list_dir_begin();
        String item;
        while (!(item = dir->get_next()).empty()) {
            if (item.ends_with(".test.gd") || item.ends_with("_test.gd")) {
                load_test_file(path + "/" + item);
            }
        }
        dir->list_dir_end();
    }

private:
    void load_test_file(const String& path) {
        Ref<GDScript> script = ResourceLoader::load(path);
        if (!script.is_valid()) return;

        // Extract test suite from script
        // Tests are functions starting with "test_"
        Ref<GDScriptTestSuite> suite;
        suite.instance();
        suite->set_name(path.get_file().get_basename());

        // This would use reflection to find test methods
        // For now, placeholder
        add_suite(suite);
    }
};

// #############################################################################
// Test macros for GDScript
// #############################################################################
#define XTU_TEST_ASSERT(condition) \
    ::xtu::godot::gdscript::GDScriptAssertion::get_singleton()->assert_true(condition, #condition, __FILE__, __LINE__)

#define XTU_TEST_ASSERT_MSG(condition, message) \
    ::xtu::godot::gdscript::GDScriptAssertion::get_singleton()->assert_true(condition, message, __FILE__, __LINE__)

#define XTU_TEST_ASSERT_FALSE(condition) \
    ::xtu::godot::gdscript::GDScriptAssertion::get_singleton()->assert_false(condition, #condition, __FILE__, __LINE__)

#define XTU_TEST_ASSERT_EQ(expected, actual) \
    ::xtu::godot::gdscript::GDScriptAssertion::get_singleton()->assert_equal(expected, actual, #expected " == " #actual, __FILE__, __LINE__)

#define XTU_TEST_ASSERT_NE(expected, actual) \
    ::xtu::godot::gdscript::GDScriptAssertion::get_singleton()->assert_not_equal(expected, actual, #expected " != " #actual, __FILE__, __LINE__)

#define XTU_TEST_ASSERT_NULL(value) \
    ::xtu::godot::gdscript::GDScriptAssertion::get_singleton()->assert_null(value, #value " is null", __FILE__, __LINE__)

#define XTU_TEST_ASSERT_NOT_NULL(value) \
    ::xtu::godot::gdscript::GDScriptAssertion::get_singleton()->assert_not_null(value, #value " is not null", __FILE__, __LINE__)

#define XTU_TEST_ASSERT_APPROX(expected, actual, tolerance) \
    ::xtu::godot::gdscript::GDScriptAssertion::get_singleton()->assert_approx(expected, actual, tolerance, #expected " ≈ " #actual, __FILE__, __LINE__)

} // namespace gdscript

// Bring into main namespace
using gdscript::GDScriptTestRunner;
using gdscript::GDScriptTestSuite;
using gdscript::GDScriptTestCase;
using gdscript::GDScriptAssertion;
using gdscript::TestStatus;
using gdscript::TestCaseResult;
using gdscript::TestSuiteResult;
using gdscript::TestAssertionResult;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XGDSCRIPT_TEST_HPP