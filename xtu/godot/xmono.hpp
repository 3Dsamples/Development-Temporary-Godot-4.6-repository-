// include/xtu/godot/xmono.hpp
// xtensor-unified - Mono/.NET Integration for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XMONO_HPP
#define XTU_GODOT_XMONO_HPP

#include <atomic>
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
#include "xtu/godot/xresource.hpp"
#include "xtu/godot/xcore.hpp"

#ifdef XTU_USE_MONO
#include <mono/jit/jit.h>
#include <mono/metadata/assembly.h>
#include <mono/metadata/mono-config.h>
#include <mono/metadata/debug-helpers.h>
#include <mono/metadata/threads.h>
#endif

XTU_NAMESPACE_BEGIN
namespace godot {
namespace mono {

// #############################################################################
// Forward declarations
// #############################################################################
class GDMono;
class CSharpScript;
class CSharpInstance;
class MonoGCHandle;
class MonoObject;
class MonoClass;
class MonoMethod;
class MonoAssembly;

// #############################################################################
// Mono GC handle types
// #############################################################################
enum class MonoGCHandleType : uint8_t {
    HANDLE_NORMAL = 0,
    HANDLE_WEAK = 1,
    HANDLE_WEAK_TRACK = 2,
    HANDLE_PINNED = 3
};

// #############################################################################
// Mono script property access
// #############################################################################
enum class MonoPropertyAccess : uint8_t {
    ACCESS_READ = 0,
    ACCESS_WRITE = 1,
    ACCESS_READ_WRITE = 2
};

// #############################################################################
// C# build configuration
// #############################################################################
enum class CSBuildConfiguration : uint8_t {
    CONFIG_DEBUG = 0,
    CONFIG_RELEASE = 1,
    CONFIG_EXPORT_DEBUG = 2,
    CONFIG_EXPORT_RELEASE = 3
};

// #############################################################################
// MonoGCHandle - Safe GC handle wrapper
// #############################################################################
class MonoGCHandle : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(MonoGCHandle, RefCounted)

private:
#ifdef XTU_USE_MONO
    uint32_t m_handle = 0;
    MonoGCHandleType m_type = MonoGCHandleType::HANDLE_NORMAL;
#endif
    bool m_valid = false;

public:
    static StringName get_class_static() { return StringName("MonoGCHandle"); }

    MonoGCHandle() = default;
#ifdef XTU_USE_MONO
    MonoGCHandle(MonoObject* obj, MonoGCHandleType type = MonoGCHandleType::HANDLE_NORMAL) {
        if (obj) {
            m_handle = mono_gchandle_new(obj, type == MonoGCHandleType::HANDLE_PINNED);
            m_type = type;
            m_valid = true;
        }
    }

    ~MonoGCHandle() {
        if (m_valid) {
            mono_gchandle_free(m_handle);
        }
    }

    MonoObject* get_target() const {
        return m_valid ? mono_gchandle_get_target(m_handle) : nullptr;
    }

    void set_target(MonoObject* obj) {
        if (m_valid) {
            mono_gchandle_free(m_handle);
        }
        if (obj) {
            m_handle = mono_gchandle_new(obj, m_type == MonoGCHandleType::HANDLE_PINNED);
            m_valid = true;
        } else {
            m_valid = false;
        }
    }

    bool is_valid() const { return m_valid; }
#endif
};

// #############################################################################
// CSharpInstance - Instance of a C# script attached to a Godot Object
// #############################################################################
class CSharpInstance : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(CSharpInstance, RefCounted)

private:
    Object* m_owner = nullptr;
    Ref<MonoGCHandle> m_gchandle;
    bool m_initialized = false;
    std::unordered_map<StringName, Variant> m_exported_properties;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("CSharpInstance"); }

    void set_owner(Object* owner) { m_owner = owner; }
    Object* get_owner() const { return m_owner; }

    void set_gchandle(const Ref<MonoGCHandle>& handle) { m_gchandle = handle; }
    Ref<MonoGCHandle> get_gchandle() const { return m_gchandle; }

    void set_initialized(bool initialized) { m_initialized = initialized; }
    bool is_initialized() const { return m_initialized; }

    void set_exported_property(const StringName& name, const Variant& value) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_exported_properties[name] = value;
    }

    Variant get_exported_property(const StringName& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_exported_properties.find(name);
        return it != m_exported_properties.end() ? it->second : Variant();
    }

    bool has_exported_property(const StringName& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_exported_properties.find(name) != m_exported_properties.end();
    }

    Variant call_method(const StringName& method, const std::vector<Variant>& args) {
        // Invoke C# method via Mono
        return Variant();
    }
};

// #############################################################################
// CSharpScript - C# script resource
// #############################################################################
class CSharpScript : public Resource {
    XTU_GODOT_REGISTER_CLASS(CSharpScript, Resource)

private:
    String m_source_path;
    String m_class_name;
    String m_namespace;
    String m_assembly_name;
    Ref<CSharpScript> m_base_script;
    std::vector<StringName> m_exported_properties;
    std::vector<StringName> m_methods;
    std::vector<StringName> m_signals;
    bool m_valid = false;
    bool m_tool = false;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("CSharpScript"); }

    void set_source_path(const String& path) { m_source_path = path; }
    String get_source_path() const { return m_source_path; }

    void set_class_name(const String& name) { m_class_name = name; }
    String get_class_name() const { return m_class_name; }

    void set_namespace(const String& ns) { m_namespace = ns; }
    String get_namespace() const { return m_namespace; }

    void set_assembly_name(const String& name) { m_assembly_name = name; }
    String get_assembly_name() const { return m_assembly_name; }

    void set_base_script(const Ref<CSharpScript>& base) { m_base_script = base; }
    Ref<CSharpScript> get_base_script() const { return m_base_script; }

    void set_valid(bool valid) { m_valid = valid; }
    bool is_valid() const { return m_valid; }

    void set_tool(bool tool) { m_tool = tool; }
    bool is_tool() const { return m_tool; }

    std::vector<StringName> get_exported_properties() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_exported_properties;
    }

    std::vector<StringName> get_methods() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_methods;
    }

    bool has_method(const StringName& method) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return std::find(m_methods.begin(), m_methods.end(), method) != m_methods.end();
    }

    Ref<CSharpInstance> instance_create(Object* owner) {
        if (!m_valid) return Ref<CSharpInstance>();

        Ref<CSharpInstance> instance;
        instance.instance();
        instance->set_owner(owner);

        // Create C# object via Mono
        return instance;
    }

    void load_from_file(const String& path) {
        m_source_path = path;
        // Parse and compile C# script
    }
};

// #############################################################################
// GDMono - Global Mono runtime manager
// #############################################################################
class GDMono : public Object {
    XTU_GODOT_REGISTER_CLASS(GDMono, Object)

public:
    enum ApiHashType {
        API_HASH_CORE,
        API_HASH_EDITOR,
        API_HASH_MAX
    };

private:
    static GDMono* s_singleton;
#ifdef XTU_USE_MONO
    MonoDomain* m_root_domain = nullptr;
    MonoDomain* m_scripts_domain = nullptr;
    MonoDomain* m_tools_domain = nullptr;
    MonoAssembly* m_core_assembly = nullptr;
    MonoAssembly* m_editor_assembly = nullptr;
    MonoAssembly* m_project_assembly = nullptr;
#endif
    bool m_initialized = false;
    bool m_runtime_initialized = false;
    String m_project_assembly_path;
    String m_core_assembly_path;
    String m_editor_assembly_path;
    std::unordered_map<uint64_t, Ref<MonoGCHandle>> m_gc_handles;
    std::unordered_map<String, Ref<CSharpScript>> m_script_cache;
    std::unordered_map<String, uint64_t> m_api_hashes;
    mutable std::mutex m_mutex;

public:
    static GDMono* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("GDMono"); }

    GDMono() { s_singleton = this; }
    ~GDMono() { shutdown(); s_singleton = nullptr; }

    bool initialize() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized) return true;

#ifdef XTU_USE_MONO
        // Set Mono configuration
        mono_config_parse(nullptr);

        // Initialize root domain
        m_root_domain = mono_jit_init_version("Godot", "v4.0.0");
        if (!m_root_domain) {
            return false;
        }

        // Create script domain
        m_scripts_domain = mono_domain_create_appdomain("GodotScripts", nullptr);
        mono_domain_set(m_scripts_domain, false);

        // Load core assembly
        if (!m_core_assembly_path.empty()) {
            m_core_assembly = mono_domain_assembly_open(m_scripts_domain, m_core_assembly_path.utf8());
        }

        m_runtime_initialized = true;
        m_initialized = true;
        return true;
#else
        return false;
#endif
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(m_mutex);
#ifdef XTU_USE_MONO
        if (m_runtime_initialized) {
            mono_jit_cleanup(m_root_domain);
            m_root_domain = nullptr;
            m_scripts_domain = nullptr;
            m_runtime_initialized = false;
        }
#endif
        m_initialized = false;
    }

    bool is_initialized() const { return m_initialized; }

    void set_core_assembly_path(const String& path) { m_core_assembly_path = path; }
    String get_core_assembly_path() const { return m_core_assembly_path; }

    void set_editor_assembly_path(const String& path) { m_editor_assembly_path = path; }
    String get_editor_assembly_path() const { return m_editor_assembly_path; }

    void set_project_assembly_path(const String& path) { m_project_assembly_path = path; }
    String get_project_assembly_path() const { return m_project_assembly_path; }

    Ref<CSharpScript> load_script(const String& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_script_cache.find(path);
        if (it != m_script_cache.end()) {
            return it->second;
        }

        Ref<CSharpScript> script;
        script.instance();
        script->load_from_file(path);

        if (script->is_valid()) {
            m_script_cache[path] = script;
        }
        return script;
    }

    void reload_script(const String& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_script_cache.erase(path);
    }

    String get_api_hash(ApiHashType type) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_api_hashes.find(String::num(static_cast<int>(type)));
        return it != m_api_hashes.end() ? String::num_hex(it->second) : String();
    }

    void set_api_hash(ApiHashType type, uint64_t hash) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_api_hashes[String::num(static_cast<int>(type))] = hash;
    }

    Ref<MonoGCHandle> create_gc_handle(MonoObject* obj, MonoGCHandleType type = MonoGCHandleType::HANDLE_NORMAL) {
#ifdef XTU_USE_MONO
        Ref<MonoGCHandle> handle;
        handle.instance();
        handle->set_target(obj);
        return handle;
#else
        return Ref<MonoGCHandle>();
#endif
    }

    String call_native_method(MonoObject* obj, const String& method, const std::vector<Variant>& args) {
        // Marshal and call C# method
        return String();
    }

    Variant mono_object_to_variant(MonoObject* obj) {
        // Convert Mono object to Godot Variant
        return Variant();
    }

    MonoObject* variant_to_mono_object(const Variant& v) {
        // Convert Godot Variant to Mono object
        return nullptr;
    }

    // Build management
    bool build_project_csproj(const String& csproj_path, CSBuildConfiguration config) {
        // Invoke MSBuild or dotnet build
        return true;
    }

    bool build_scripts(const std::vector<String>& paths) {
        // Compile C# scripts to assembly
        return true;
    }
};

// #############################################################################
// CSharpLanguage - Script language implementation for C#
// #############################################################################
class CSharpLanguage : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(CSharpLanguage, RefCounted)

private:
    static CSharpLanguage* s_singleton;
    bool m_debugging_enabled = false;
    int m_debugger_port = 55556;

public:
    static CSharpLanguage* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("CSharpLanguage"); }

    CSharpLanguage() { s_singleton = this; }
    ~CSharpLanguage() { s_singleton = nullptr; }

    String get_name() const { return "C#"; }
    String get_type() const { return "CSharpScript"; }
    String get_extension() const { return "cs"; }

    void set_debugging_enabled(bool enabled) { m_debugging_enabled = enabled; }
    bool is_debugging_enabled() const { return m_debugging_enabled; }

    void set_debugger_port(int port) { m_debugger_port = port; }
    int get_debugger_port() const { return m_debugger_port; }

    Ref<CSharpScript> create_script() {
        Ref<CSharpScript> script;
        script.instance();
        return script;
    }

    void reload_tool_script(const Ref<CSharpScript>& script, bool soft_reload) {
        // Reload script in editor
    }

    void frame() {
        // Process debugger events
    }
};

// #############################################################################
// MonoUtils - Utility functions for Mono interop
// #############################################################################
class MonoUtils {
public:
    static String mono_string_to_godot(MonoString* str) {
#ifdef XTU_USE_MONO
        if (!str) return String();
        char* utf8 = mono_string_to_utf8(str);
        String result(utf8);
        mono_free(utf8);
        return result;
#else
        return String();
#endif
    }

    static MonoString* godot_to_mono_string(const String& str) {
#ifdef XTU_USE_MONO
        return mono_string_new(mono_domain_get(), str.utf8());
#else
        return nullptr;
#endif
    }

    static MonoObject* variant_to_mono(const Variant& v) {
        switch (v.get_type()) {
            case VariantType::BOOL:
                return mono_value_box(mono_domain_get(), mono_get_boolean_class(), &v);
            case VariantType::INT:
                return mono_value_box(mono_domain_get(), mono_get_int64_class(), &v);
            case VariantType::FLOAT:
                return mono_value_box(mono_domain_get(), mono_get_double_class(), &v);
            case VariantType::STRING:
                return (MonoObject*)godot_to_mono_string(v.as<String>());
            default:
                return nullptr;
        }
    }

    static Variant mono_to_variant(MonoObject* obj) {
        if (!obj) return Variant();
#ifdef XTU_USE_MONO
        MonoClass* klass = mono_object_get_class(obj);
        MonoType* type = mono_class_get_type(klass);

        switch (mono_type_get_type(type)) {
            case MONO_TYPE_BOOLEAN: {
                bool* val = (bool*)mono_object_unbox(obj);
                return Variant(*val);
            }
            case MONO_TYPE_I8:
            case MONO_TYPE_U8:
            case MONO_TYPE_I4:
            case MONO_TYPE_U4:
            case MONO_TYPE_I2:
            case MONO_TYPE_U2: {
                int64_t val = *(int64_t*)mono_object_unbox(obj);
                return Variant(val);
            }
            case MONO_TYPE_R4:
            case MONO_TYPE_R8: {
                double val = *(double*)mono_object_unbox(obj);
                return Variant(val);
            }
            case MONO_TYPE_STRING: {
                return Variant(mono_string_to_godot((MonoString*)obj));
            }
            default:
                break;
        }
#endif
        return Variant();
    }
};

} // namespace mono

// Bring into main namespace
using mono::GDMono;
using mono::CSharpScript;
using mono::CSharpInstance;
using mono::CSharpLanguage;
using mono::MonoGCHandle;
using mono::MonoUtils;
using mono::MonoGCHandleType;
using mono::MonoPropertyAccess;
using mono::CSBuildConfiguration;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XMONO_HPP