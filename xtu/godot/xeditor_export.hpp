// include/xtu/godot/xeditor_export.hpp
// xtensor-unified - Editor export system for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XEDITOR_EXPORT_HPP
#define XTU_GODOT_XEDITOR_EXPORT_HPP

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
#include "xtu/godot/xeditor.hpp"
#include "xtu/io/xio_json.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace editor {

// #############################################################################
// Forward declarations
// #############################################################################
class EditorExportPlatform;
class EditorExportPreset;
class EditorExportPlugin;
class EditorExportPlatformPC;
class EditorExportPlatformWindows;
class EditorExportPlatformLinux;
class EditorExportPlatformMacOS;
class EditorExportPlatformAndroid;
class EditorExportPlatformIOS;
class EditorExportPlatformWeb;

// #############################################################################
// Export feature tags
// #############################################################################
enum class ExportFeature : uint32_t {
    FEATURE_MOBILE = 1 << 0,
    FEATURE_PC = 1 << 1,
    FEATURE_WEB = 1 << 2,
    FEATURE_CONSOLE = 1 << 3,
    FEATURE_VR = 1 << 4,
    FEATURE_AR = 1 << 5,
    FEATURE_64_BIT = 1 << 6,
    FEATURE_32_BIT = 1 << 7,
    FEATURE_DEBUG = 1 << 8,
    FEATURE_RELEASE = 1 << 9
};

// #############################################################################
// Export debug level
// #############################################################################
enum class ExportDebugLevel : uint8_t {
    DEBUG_DISABLED = 0,
    DEBUG_ENABLED = 1,
    DEBUG_DEEP = 2
};

// #############################################################################
// Export texture format
// #############################################################################
enum class ExportTextureFormat : uint8_t {
    FORMAT_ETC2 = 0,
    FORMAT_ASTC = 1,
    FORMAT_S3TC = 2,
    FORMAT_PVRTC = 3,
    FORMAT_BPTC = 4
};

// #############################################################################
// EditorExportPreset - Export configuration preset
// #############################################################################
class EditorExportPreset : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(EditorExportPreset, RefCounted)

private:
    String m_name;
    Ref<EditorExportPlatform> m_platform;
    std::map<String, Variant> m_options;
    std::vector<String> m_include_files;
    std::vector<String> m_exclude_files;
    std::vector<String> m_include_filters;
    std::vector<String> m_exclude_filters;
    String m_export_path;
    String m_custom_features;
    bool m_runnable = true;
    bool m_dedicated_server = false;

public:
    static StringName get_class_static() { return StringName("EditorExportPreset"); }

    void set_name(const String& name) { m_name = name; }
    String get_name() const { return m_name; }

    void set_platform(const Ref<EditorExportPlatform>& platform) { m_platform = platform; }
    Ref<EditorExportPlatform> get_platform() const { return m_platform; }

    void set_option(const String& name, const Variant& value) { m_options[name] = value; }
    Variant get_option(const String& name, const Variant& default_val = Variant()) const {
        auto it = m_options.find(name);
        return it != m_options.end() ? it->second : default_val;
    }
    bool has_option(const String& name) const { return m_options.find(name) != m_options.end(); }

    void set_include_files(const std::vector<String>& files) { m_include_files = files; }
    const std::vector<String>& get_include_files() const { return m_include_files; }

    void set_exclude_files(const std::vector<String>& files) { m_exclude_files = files; }
    const std::vector<String>& get_exclude_files() const { return m_exclude_files; }

    void set_export_path(const String& path) { m_export_path = path; }
    String get_export_path() const { return m_export_path; }

    void set_runnable(bool runnable) { m_runnable = runnable; }
    bool is_runnable() const { return m_runnable; }

    void set_dedicated_server(bool server) { m_dedicated_server = server; }
    bool is_dedicated_server() const { return m_dedicated_server; }

    void set_custom_features(const String& features) { m_custom_features = features; }
    String get_custom_features() const { return m_custom_features; }
};

// #############################################################################
// EditorExportPlugin - Hook into export process
// #############################################################################
class EditorExportPlugin : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(EditorExportPlugin, RefCounted)

public:
    static StringName get_class_static() { return StringName("EditorExportPlugin"); }

    virtual String get_name() const { return String(); }
    virtual bool supports_platform(const Ref<EditorExportPlatform>& platform) const { return true; }

    virtual void _export_begin(const std::vector<String>& features, bool is_debug, const String& path, uint32_t flags) {}
    virtual void _export_file(const String& path, const String& type, const std::vector<String>& features) {}
    virtual void _export_end() {}

    void add_file(const String& path, const std::vector<uint8_t>& data, bool remap) {
        // Add file to export
    }

    void add_shared_object(const String& path, const std::vector<String>& tags) {
        // Add shared library
    }

    void add_ios_framework(const String& path) {}
    void add_ios_embedded_framework(const String& path) {}
    void add_ios_bundle_file(const String& path) {}
    void add_ios_plist_content(const String& content) {}
    void add_ios_linker_flags(const String& flags) {}
    void add_ios_cpp_code(const String& code) {}
    void skip() {}
};

// #############################################################################
// EditorExportPlatform - Base class for export platforms
// #############################################################################
class EditorExportPlatform : public RefCounted {
    XTU_GODOT_REGISTER_CLASS(EditorExportPlatform, RefCounted)

protected:
    String m_name;
    String m_os_name;

public:
    static StringName get_class_static() { return StringName("EditorExportPlatform"); }

    virtual String get_name() const { return m_name; }
    virtual String get_os_name() const { return m_os_name; }

    virtual std::vector<String> get_binary_extensions() const = 0;
    virtual std::vector<String> get_platform_features() const { return {}; }

    virtual void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const = 0;
    virtual std::vector<PropertyInfo> get_export_options() const = 0;
    virtual bool should_update_export_options() { return false; }

    virtual bool can_export(const Ref<EditorExportPreset>& preset, String& error) const { return true; }
    virtual bool has_valid_export_configuration(const Ref<EditorExportPreset>& preset, String& error) const { return true; }
    virtual bool has_valid_project_configuration(const Ref<EditorExportPreset>& preset, String& error) const { return true; }

    virtual Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags = 0) = 0;
    virtual Error export_pack(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags = 0);
    virtual Error export_zip(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags = 0);

    virtual String get_template_file_name(const String& target, const String& arch) const { return String(); }

    virtual bool is_executable(const String& path) const { return false; }
};

// #############################################################################
// EditorExportPlatformPC - Base for PC platforms
// #############################################################################
class EditorExportPlatformPC : public EditorExportPlatform {
    XTU_GODOT_REGISTER_CLASS(EditorExportPlatformPC, EditorExportPlatform)

protected:
    std::vector<String> m_binary_extensions;

public:
    static StringName get_class_static() { return StringName("EditorExportPlatformPC"); }

    std::vector<String> get_binary_extensions() const override { return m_binary_extensions; }
    bool is_executable(const String& path) const override { return true; }
};

// #############################################################################
// EditorExportPlatformWindows - Windows exporter
// #############################################################################
class EditorExportPlatformWindows : public EditorExportPlatformPC {
    XTU_GODOT_REGISTER_CLASS(EditorExportPlatformWindows, EditorExportPlatformPC)

public:
    static StringName get_class_static() { return StringName("EditorExportPlatformWindows"); }

    EditorExportPlatformWindows() {
        m_name = "Windows Desktop";
        m_os_name = "Windows";
        m_binary_extensions = {"exe", "console.exe"};
    }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back(PropertyInfo{VariantType::STRING, "custom_template/debug"});
        options.push_back(PropertyInfo{VariantType::STRING, "custom_template/release"});
        options.push_back(PropertyInfo{VariantType::BOOL, "binary_format/embed_pck"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/icon"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/file_version"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/product_version"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/company_name"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/product_name"});
        options.push_back(PropertyInfo{VariantType::BOOL, "application/console_wrapper_icon"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("windows");
        features.push_back("pc");
        if (preset->get_option("binary_format/64_bits", true).as<bool>()) {
            features.push_back("x86_64");
        } else {
            features.push_back("x86_32");
        }
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        // Windows export implementation
        return OK;
    }
};

// #############################################################################
// EditorExportPlatformLinux - Linux exporter
// #############################################################################
class EditorExportPlatformLinux : public EditorExportPlatformPC {
    XTU_GODOT_REGISTER_CLASS(EditorExportPlatformLinux, EditorExportPlatformPC)

public:
    static StringName get_class_static() { return StringName("EditorExportPlatformLinux"); }

    EditorExportPlatformLinux() {
        m_name = "Linux/X11";
        m_os_name = "Linux";
        m_binary_extensions = {"x86_32", "x86_64", "arm32", "arm64"};
    }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back(PropertyInfo{VariantType::STRING, "custom_template/debug"});
        options.push_back(PropertyInfo{VariantType::STRING, "custom_template/release"});
        options.push_back(PropertyInfo{VariantType::BOOL, "binary_format/embed_pck"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/icon"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("linux");
        features.push_back("pc");
        features.push_back("x11");
        String arch = preset->get_option("binary_format/architecture", "x86_64").as<String>();
        features.push_back(arch);
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        return OK;
    }
};

// #############################################################################
// EditorExportPlatformMacOS - macOS exporter
// #############################################################################
class EditorExportPlatformMacOS : public EditorExportPlatformPC {
    XTU_GODOT_REGISTER_CLASS(EditorExportPlatformMacOS, EditorExportPlatformPC)

public:
    static StringName get_class_static() { return StringName("EditorExportPlatformMacOS"); }

    EditorExportPlatformMacOS() {
        m_name = "macOS";
        m_os_name = "macOS";
        m_binary_extensions = {"zip", "dmg"};
    }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back(PropertyInfo{VariantType::STRING, "custom_template/debug"});
        options.push_back(PropertyInfo{VariantType::STRING, "custom_template/release"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/icon"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/bundle_identifier"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/signature"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/short_version"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/version"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/copyright"});
        options.push_back(PropertyInfo{VariantType::BOOL, "application/codesign/enable"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/codesign/identity"});
        options.push_back(PropertyInfo{VariantType::BOOL, "application/notarization/enable"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("macos");
        features.push_back("pc");
        features.push_back("x86_64");
        features.push_back("arm64");
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        return OK;
    }
};

// #############################################################################
// EditorExportPlatformAndroid - Android exporter
// #############################################################################
class EditorExportPlatformAndroid : public EditorExportPlatform {
    XTU_GODOT_REGISTER_CLASS(EditorExportPlatformAndroid, EditorExportPlatform)

public:
    static StringName get_class_static() { return StringName("EditorExportPlatformAndroid"); }

    EditorExportPlatformAndroid() {
        m_name = "Android";
        m_os_name = "Android";
    }

    std::vector<String> get_binary_extensions() const override { return {"apk", "aab"}; }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back(PropertyInfo{VariantType::STRING, "custom_template/debug"});
        options.push_back(PropertyInfo{VariantType::STRING, "custom_template/release"});
        options.push_back(PropertyInfo{VariantType::STRING, "package/unique_name"});
        options.push_back(PropertyInfo{VariantType::STRING, "package/name"});
        options.push_back(PropertyInfo{VariantType::BOOL, "package/signed"});
        options.push_back(PropertyInfo{VariantType::STRING, "package/keystore/debug"});
        options.push_back(PropertyInfo{VariantType::STRING, "package/keystore/release"});
        options.push_back(PropertyInfo{VariantType::STRING, "package/keystore/release_user"});
        options.push_back(PropertyInfo{VariantType::STRING, "package/keystore/release_password"});
        options.push_back(PropertyInfo{VariantType::BOOL, "screen/immersive_mode"});
        options.push_back(PropertyInfo{VariantType::BOOL, "screen/support_large"});
        options.push_back(PropertyInfo{VariantType::BOOL, "screen/support_xlarge"});
        options.push_back(PropertyInfo{VariantType::INT, "version/code"});
        options.push_back(PropertyInfo{VariantType::STRING, "version/name"});
        options.push_back(PropertyInfo{VariantType::BOOL, "gradle_build/use_gradle"});
        options.push_back(PropertyInfo{VariantType::INT, "gradle_build/min_sdk"});
        options.push_back(PropertyInfo{VariantType::INT, "gradle_build/target_sdk"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("android");
        features.push_back("mobile");
        String arch = preset->get_option("architectures/armeabi-v7a", false).as<bool>() ? "armv7" : "arm64";
        features.push_back(arch);
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        return OK;
    }
};

// #############################################################################
// EditorExportPlatformIOS - iOS exporter
// #############################################################################
class EditorExportPlatformIOS : public EditorExportPlatform {
    XTU_GODOT_REGISTER_CLASS(EditorExportPlatformIOS, EditorExportPlatform)

public:
    static StringName get_class_static() { return StringName("EditorExportPlatformIOS"); }

    EditorExportPlatformIOS() {
        m_name = "iOS";
        m_os_name = "iOS";
    }

    std::vector<String> get_binary_extensions() const override { return {"ipa", "app"}; }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back(PropertyInfo{VariantType::STRING, "custom_template/debug"});
        options.push_back(PropertyInfo{VariantType::STRING, "custom_template/release"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/app_store_team_id"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/provisioning_profile_uuid_debug"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/provisioning_profile_uuid_release"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/bundle_identifier"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/short_version"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/version"});
        options.push_back(PropertyInfo{VariantType::STRING, "application/copyright"});
        options.push_back(PropertyInfo{VariantType::BOOL, "capabilities/access_wifi"});
        options.push_back(PropertyInfo{VariantType::BOOL, "capabilities/arkit"});
        options.push_back(PropertyInfo{VariantType::BOOL, "capabilities/camera"});
        options.push_back(PropertyInfo{VariantType::BOOL, "capabilities/microphone"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("ios");
        features.push_back("mobile");
        features.push_back("arm64");
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        return OK;
    }
};

// #############################################################################
// EditorExportPlatformWeb - Web exporter
// #############################################################################
class EditorExportPlatformWeb : public EditorExportPlatform {
    XTU_GODOT_REGISTER_CLASS(EditorExportPlatformWeb, EditorExportPlatform)

public:
    static StringName get_class_static() { return StringName("EditorExportPlatformWeb"); }

    EditorExportPlatformWeb() {
        m_name = "Web";
        m_os_name = "Web";
    }

    std::vector<String> get_binary_extensions() const override { return {"html"}; }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back(PropertyInfo{VariantType::STRING, "custom_template/debug"});
        options.push_back(PropertyInfo{VariantType::STRING, "custom_template/release"});
        options.push_back(PropertyInfo{VariantType::STRING, "html/head_include"});
        options.push_back(PropertyInfo{VariantType::STRING, "html/canvas_resize_policy"});
        options.push_back(PropertyInfo{VariantType::STRING, "html/focus_canvas_on_start"});
        options.push_back(PropertyInfo{VariantType::BOOL, "html/export_icon"});
        options.push_back(PropertyInfo{VariantType::BOOL, "progressive_web_app/enabled"});
        options.push_back(PropertyInfo{VariantType::STRING, "progressive_web_app/offline_page"});
        options.push_back(PropertyInfo{VariantType::BOOL, "progressive_web_app/display"});
        options.push_back(PropertyInfo{VariantType::STRING, "progressive_web_app/orientation"});
        options.push_back(PropertyInfo{VariantType::STRING, "progressive_web_app/background_color"});
        options.push_back(PropertyInfo{VariantType::STRING, "progressive_web_app/theme_color"});
        options.push_back(PropertyInfo{VariantType::STRING, "progressive_web_app/icon_144x144"});
        options.push_back(PropertyInfo{VariantType::STRING, "progressive_web_app/icon_180x180"});
        options.push_back(PropertyInfo{VariantType::STRING, "progressive_web_app/icon_512x512"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("web");
        features.push_back("wasm");
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        return OK;
    }
};

// #############################################################################
// EditorExport - Export manager singleton
// #############################################################################
class EditorExport : public Object {
    XTU_GODOT_REGISTER_CLASS(EditorExport, Object)

private:
    static EditorExport* s_singleton;
    std::vector<Ref<EditorExportPlatform>> m_platforms;
    std::vector<Ref<EditorExportPreset>> m_presets;
    std::vector<Ref<EditorExportPlugin>> m_plugins;
    std::mutex m_mutex;

public:
    static EditorExport* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("EditorExport"); }

    EditorExport() {
        s_singleton = this;
        register_default_platforms();
    }

    ~EditorExport() { s_singleton = nullptr; }

    void add_export_platform(const Ref<EditorExportPlatform>& platform) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (std::find(m_platforms.begin(), m_platforms.end(), platform) == m_platforms.end()) {
            m_platforms.push_back(platform);
        }
    }

    std::vector<Ref<EditorExportPlatform>> get_export_platforms() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_platforms;
    }

    Ref<EditorExportPlatform> get_export_platform(const String& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& p : m_platforms) {
            if (p->get_name() == name) return p;
        }
        return Ref<EditorExportPlatform>();
    }

    void add_export_preset(const Ref<EditorExportPreset>& preset) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_presets.push_back(preset);
    }

    void remove_export_preset(int idx) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (idx >= 0 && idx < static_cast<int>(m_presets.size())) {
            m_presets.erase(m_presets.begin() + idx);
        }
    }

    std::vector<Ref<EditorExportPreset>> get_export_presets() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_presets;
    }

    void add_export_plugin(const Ref<EditorExportPlugin>& plugin) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_plugins.push_back(plugin);
    }

    void remove_export_plugin(const Ref<EditorExportPlugin>& plugin) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = std::find(m_plugins.begin(), m_plugins.end(), plugin);
        if (it != m_plugins.end()) m_plugins.erase(it);
    }

    Error export_preset(const Ref<EditorExportPreset>& preset, const String& path, bool debug, uint32_t flags = 0) {
        if (!preset.is_valid()) return ERR_INVALID_PARAMETER;
        Ref<EditorExportPlatform> platform = preset->get_platform();
        if (!platform.is_valid()) return ERR_UNAVAILABLE;

        String error;
        if (!platform->can_export(preset, error)) {
            return ERR_UNAUTHORIZED;
        }

        // Notify plugins
        std::vector<String> features;
        platform->get_preset_features(preset, features);
        for (auto& plugin : m_plugins) {
            if (plugin->supports_platform(platform)) {
                plugin->_export_begin(features, debug, path, flags);
            }
        }

        Error err = platform->export_project(preset, debug, path, flags);

        for (auto& plugin : m_plugins) {
            if (plugin->supports_platform(platform)) {
                plugin->_export_end();
            }
        }

        return err;
    }

private:
    void register_default_platforms() {
        add_export_platform(Ref<EditorExportPlatform>(new EditorExportPlatformWindows()));
        add_export_platform(Ref<EditorExportPlatform>(new EditorExportPlatformLinux()));
        add_export_platform(Ref<EditorExportPlatform>(new EditorExportPlatformMacOS()));
        add_export_platform(Ref<EditorExportPlatform>(new EditorExportPlatformAndroid()));
        add_export_platform(Ref<EditorExportPlatform>(new EditorExportPlatformIOS()));
        add_export_platform(Ref<EditorExportPlatform>(new EditorExportPlatformWeb()));
    }
};

} // namespace editor

using editor::EditorExport;
using editor::EditorExportPlatform;
using editor::EditorExportPreset;
using editor::EditorExportPlugin;
using editor::EditorExportPlatformWindows;
using editor::EditorExportPlatformLinux;
using editor::EditorExportPlatformMacOS;
using editor::EditorExportPlatformAndroid;
using editor::EditorExportPlatformIOS;
using editor::EditorExportPlatformWeb;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XEDITOR_EXPORT_HPP