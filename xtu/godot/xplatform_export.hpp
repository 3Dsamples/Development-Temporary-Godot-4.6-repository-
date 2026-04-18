// include/xtu/godot/xplatform_export.hpp
// xtensor-unified - Platform Export System for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XPLATFORM_EXPORT_HPP
#define XTU_GODOT_XPLATFORM_EXPORT_HPP

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
#include "xtu/godot/xeditor_export.hpp"
#include "xtu/io/xio_json.hpp"
#include "xtu/parallel/xparallel.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {
namespace platform_export {

// #############################################################################
// Platform export template
// #############################################################################
struct ExportTemplate {
    String path;
    String version;
    String arch;
    bool debug = false;
    bool valid = false;
};

// #############################################################################
// Export archive types
// #############################################################################
enum class ExportArchiveType : uint8_t {
    ARCHIVE_ZIP = 0,
    ARCHIVE_TAR = 1,
    ARCHIVE_DMG = 2,
    ARCHIVE_APP = 3,
    ARCHIVE_APK = 4,
    ARCHIVE_AAB = 5,
    ARCHIVE_IPA = 6
};

// #############################################################################
// Code signing configuration
// #############################################################################
struct CodeSignConfig {
    bool enabled = false;
    String identity;
    String certificate_path;
    String private_key_path;
    String password;
    String timestamp_url;
    String entitlements_path;
    bool hardened_runtime = true;
};

// #############################################################################
// Base platform exporter - provides common export functionality
// #############################################################################
class PlatformExporter : public EditorExportPlatform {
    XTU_GODOT_REGISTER_CLASS(PlatformExporter, EditorExportPlatform)

protected:
    std::map<String, String> m_template_paths;
    std::map<String, ExportTemplate> m_cached_templates;
    CodeSignConfig m_code_sign;
    mutable std::mutex m_mutex;

public:
    static StringName get_class_static() { return StringName("PlatformExporter"); }

    virtual String get_platform_name() const = 0;
    virtual String get_platform_arch() const = 0;

    // Template management
    void set_template_path(const String& key, const String& path) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_template_paths[key] = path;
        m_cached_templates.erase(key);
    }

    String get_template_path(const String& key) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_template_paths.find(key);
        return it != m_template_paths.end() ? it->second : String();
    }

    virtual ExportTemplate get_template(bool debug, const String& arch = "") {
        String key = debug ? "debug" : "release";
        if (!arch.empty()) key += "_" + arch;

        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cached_templates.find(key);
        if (it != m_cached_templates.end()) {
            return it->second;
        }

        ExportTemplate tmpl = find_template(debug, arch);
        m_cached_templates[key] = tmpl;
        return tmpl;
    }

    // Code signing
    void set_code_sign_enabled(bool enabled) { m_code_sign.enabled = enabled; }
    bool is_code_sign_enabled() const { return m_code_sign.enabled; }

    void set_code_sign_identity(const String& identity) { m_code_sign.identity = identity; }
    String get_code_sign_identity() const { return m_code_sign.identity; }

    void set_code_sign_certificate(const String& path) { m_code_sign.certificate_path = path; }
    void set_code_sign_private_key(const String& path) { m_code_sign.private_key_path = path; }
    void set_code_sign_password(const String& password) { m_code_sign.password = password; }

    // PCK embedding
    Error embed_pck(const String& exe_path, const String& pck_path) {
        Ref<FileAccess> exe = FileAccess::open(exe_path, FileAccess::READ_WRITE);
        Ref<FileAccess> pck = FileAccess::open(pck_path, FileAccess::READ);
        if (!exe.is_valid() || !pck.is_valid()) return ERR_FILE_CANT_OPEN;

        exe->seek_end();
        std::vector<uint8_t> pck_data = pck->get_buffer(pck->get_length());

        const uint8_t marker[] = {'G', 'D', 'P', 'C', 'K', 0, 0, 0};
        exe->store_buffer({marker, marker + 8});
        exe->store_64(pck_data.size());
        exe->store_buffer(pck_data);

        return OK;
    }

    // Icon replacement
    virtual Error set_icon(const String& target_path, const String& icon_path) {
        return ERR_UNAVAILABLE;
    }

    // Version info embedding (Windows)
    virtual Error set_version_info(const String& target_path, const std::map<String, String>& info) {
        return ERR_UNAVAILABLE;
    }

    // Archive creation
    Error create_zip(const String& source_dir, const String& output_path) {
        String cmd = "cd \"" + source_dir.to_std_string() + "\" && zip -r \"" + output_path.to_std_string() + "\" .";
        int ret = system(cmd.utf8());
        return ret == 0 ? OK : ERR_COMPILATION_FAILED;
    }

protected:
    virtual ExportTemplate find_template(bool debug, const String& arch) {
        ExportTemplate tmpl;
        String key = debug ? "debug" : "release";
        if (!arch.empty()) key += "_" + arch;

        String path = get_template_path(key);
        if (!path.empty() && FileAccess::file_exists(path)) {
            tmpl.path = path;
            tmpl.debug = debug;
            tmpl.arch = arch;
            tmpl.valid = true;
        }
        return tmpl;
    }
};

// #############################################################################
// WindowsExporter - Windows platform exporter
// #############################################################################
class WindowsExporter : public PlatformExporter {
    XTU_GODOT_REGISTER_CLASS(WindowsExporter, PlatformExporter)

private:
    std::vector<String> m_binary_extensions = {"exe", "console.exe"};

public:
    static StringName get_class_static() { return StringName("WindowsExporter"); }

    WindowsExporter() {
        m_name = "Windows Desktop";
        m_os_name = "Windows";
    }

    String get_platform_name() const override { return "Windows"; }
    String get_platform_arch() const override { return "x86_64"; }

    std::vector<String> get_binary_extensions() const override { return m_binary_extensions; }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::STRING, "custom_template/debug"});
        options.push_back({VariantType::STRING, "custom_template/release"});
        options.push_back({VariantType::BOOL, "binary_format/embed_pck"});
        options.push_back({VariantType::BOOL, "binary_format/console_wrapper"});
        options.push_back({VariantType::STRING, "application/icon"});
        options.push_back({VariantType::STRING, "application/file_version"});
        options.push_back({VariantType::STRING, "application/product_version"});
        options.push_back({VariantType::STRING, "application/company_name"});
        options.push_back({VariantType::STRING, "application/product_name"});
        options.push_back({VariantType::STRING, "application/file_description"});
        options.push_back({VariantType::STRING, "application/copyright"});
        options.push_back({VariantType::BOOL, "codesign/enable"});
        options.push_back({VariantType::STRING, "codesign/identity"});
        options.push_back({VariantType::STRING, "codesign/password"});
        options.push_back({VariantType::STRING, "codesign/timestamp_url"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("windows");
        features.push_back("pc");
        features.push_back("x86_64");
        if (preset.is_valid() && preset->get_option("binary_format/console_wrapper", false).as<bool>()) {
            features.push_back("console");
        }
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        // Get template
        ExportTemplate tmpl = get_template(debug);
        if (!tmpl.valid) return ERR_FILE_NOT_FOUND;

        // Build PCK
        String pck_path = path.get_basename() + ".pck";
        Error err = export_pack(preset, debug, pck_path, flags);
        if (err != OK) return err;

        // Copy template
        Ref<FileAccess> src = FileAccess::open(tmpl.path, FileAccess::READ);
        Ref<FileAccess> dst = FileAccess::open(path, FileAccess::WRITE);
        if (!src.is_valid() || !dst.is_valid()) return ERR_FILE_CANT_WRITE;

        dst->store_buffer(src->get_buffer(src->get_length()));

        // Embed PCK if requested
        if (preset->get_option("binary_format/embed_pck", true).as<bool>()) {
            embed_pck(path, pck_path);
            DirAccess::remove(pck_path);
        }

        // Apply icon
        if (preset->has_option("application/icon")) {
            String icon_path = preset->get_option("application/icon").as<String>();
            if (!icon_path.empty()) {
                set_icon(path, icon_path);
            }
        }

        // Apply version info
        std::map<String, String> version_info;
        version_info["FileVersion"] = preset->get_option("application/file_version", "1.0.0.0").as<String>();
        version_info["ProductVersion"] = preset->get_option("application/product_version", "1.0.0.0").as<String>();
        version_info["CompanyName"] = preset->get_option("application/company_name", "").as<String>();
        version_info["ProductName"] = preset->get_option("application/product_name", "Godot App").as<String>();
        version_info["FileDescription"] = preset->get_option("application/file_description", "").as<String>();
        version_info["LegalCopyright"] = preset->get_option("application/copyright", "").as<String>();
        set_version_info(path, version_info);

        // Code signing
        if (preset->get_option("codesign/enable", false).as<bool>()) {
            sign_executable(path, preset);
        }

        return OK;
    }

    Error set_icon(const String& target_path, const String& icon_path) override {
        // Windows: UpdateResource with RT_GROUP_ICON and RT_ICON
        return OK;
    }

    Error set_version_info(const String& target_path, const std::map<String, String>& info) override {
        // Windows: UpdateResource with VS_VERSION_INFO
        return OK;
    }

private:
    void sign_executable(const String& path, const Ref<EditorExportPreset>& preset) {
        String identity = preset->get_option("codesign/identity").as<String>();
        String password = preset->get_option("codesign/password").as<String>();
        String timestamp = preset->get_option("codesign/timestamp_url", "http://timestamp.digicert.com").as<String>();

        String cmd = "signtool sign /fd SHA256";
        if (!identity.empty()) cmd += " /f \"" + identity + "\"";
        if (!password.empty()) cmd += " /p \"" + password + "\"";
        if (!timestamp.empty()) cmd += " /tr \"" + timestamp + "\" /td SHA256";
        cmd += " /v \"" + path.to_std_string() + "\"";

        system(cmd.utf8());
    }
};

// #############################################################################
// MacOSExporter - macOS platform exporter
// #############################################################################
class MacOSExporter : public PlatformExporter {
    XTU_GODOT_REGISTER_CLASS(MacOSExporter, PlatformExporter)

private:
    std::vector<String> m_binary_extensions = {"zip", "dmg"};

public:
    static StringName get_class_static() { return StringName("MacOSExporter"); }

    MacOSExporter() {
        m_name = "macOS";
        m_os_name = "macOS";
    }

    String get_platform_name() const override { return "macOS"; }
    String get_platform_arch() const override { return "universal"; }

    std::vector<String> get_binary_extensions() const override { return m_binary_extensions; }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::STRING, "custom_template/debug"});
        options.push_back({VariantType::STRING, "custom_template/release"});
        options.push_back({VariantType::STRING, "application/icon"});
        options.push_back({VariantType::STRING, "application/bundle_identifier"});
        options.push_back({VariantType::STRING, "application/short_version"});
        options.push_back({VariantType::STRING, "application/version"});
        options.push_back({VariantType::STRING, "application/copyright"});
        options.push_back({VariantType::BOOL, "application/codesign/enable"});
        options.push_back({VariantType::STRING, "application/codesign/identity"});
        options.push_back({VariantType::BOOL, "application/notarization/enable"});
        options.push_back({VariantType::STRING, "application/notarization/apple_id"});
        options.push_back({VariantType::STRING, "application/notarization/team_id"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("macos");
        features.push_back("pc");
        features.push_back("x86_64");
        features.push_back("arm64");
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        String temp_dir = OS::get_singleton()->get_cache_dir() + "/macos_export";
        DirAccess::make_dir_recursive(temp_dir);

        // Create app bundle structure
        String app_name = preset->get_option("application/name", "GodotApp").as<String>();
        String bundle_id = preset->get_option("application/bundle_identifier", "com.example.game").as<String>();
        String version = preset->get_option("application/version", "1.0").as<String>();

        String app_dir = temp_dir + "/" + app_name + ".app";
        String contents_dir = app_dir + "/Contents";
        String macos_dir = contents_dir + "/MacOS";
        String resources_dir = contents_dir + "/Resources";

        DirAccess::make_dir_recursive(macos_dir);
        DirAccess::make_dir_recursive(resources_dir);

        // Copy template executable
        ExportTemplate tmpl = get_template(debug);
        if (!tmpl.valid) return ERR_FILE_NOT_FOUND;

        String exe_path = macos_dir + "/" + app_name;
        Ref<FileAccess> src = FileAccess::open(tmpl.path, FileAccess::READ);
        Ref<FileAccess> dst = FileAccess::open(exe_path, FileAccess::WRITE);
        if (!src.is_valid() || !dst.is_valid()) return ERR_FILE_CANT_WRITE;

        dst->store_buffer(src->get_buffer(src->get_length()));
        chmod(exe_path.utf8(), 0755);

        // Build PCK
        String pck_path = resources_dir + "/" + app_name + ".pck";
        export_pack(preset, debug, pck_path, flags);

        // Create Info.plist
        String plist = generate_info_plist(app_name, bundle_id, version, preset);
        Ref<FileAccess> plist_file = FileAccess::open(contents_dir + "/Info.plist", FileAccess::WRITE);
        if (plist_file.is_valid()) plist_file->store_string(plist);

        // Copy icon
        String icon_path = preset->get_option("application/icon").as<String>();
        if (!icon_path.empty()) {
            DirAccess::copy(icon_path, resources_dir + "/Icon.icns");
        }

        // Code signing
        if (preset->get_option("application/codesign/enable", false).as<bool>()) {
            String identity = preset->get_option("application/codesign/identity").as<String>();
            sign_app_bundle(app_dir, identity);
        }

        // Package
        if (path.ends_with(".zip")) {
            create_zip(temp_dir, path);
        } else if (path.ends_with(".dmg")) {
            create_dmg(app_dir, path);
        }

        DirAccess::remove(temp_dir);
        return OK;
    }

private:
    String generate_info_plist(const String& name, const String& bundle_id, const String& version,
                                const Ref<EditorExportPreset>& preset) {
        String copyright = preset->get_option("application/copyright", "").as<String>();
        String short_version = preset->get_option("application/short_version", version).as<String>();

        return R"(<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>)" + name + R"(</string>
    <key>CFBundleIdentifier</key>
    <string>)" + bundle_id + R"(</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>)" + name + R"(</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>)" + short_version + R"(</string>
    <key>CFBundleVersion</key>
    <string>)" + version + R"(</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHumanReadableCopyright</key>
    <string>)" + copyright + R"(</string>
</dict>
</plist>)";
    }

    void sign_app_bundle(const String& app_path, const String& identity) {
        String cmd = "codesign --deep --force --sign \"" + identity.to_std_string() + "\" \"" + app_path.to_std_string() + "\"";
        system(cmd.utf8());
    }

    void create_dmg(const String& app_path, const String& dmg_path) {
        String cmd = "hdiutil create -volname \"Godot App\" -srcfolder \"" + app_path.to_std_string() + "\" -ov -format UDZO \"" + dmg_path.to_std_string() + "\"";
        system(cmd.utf8());
    }
};

// #############################################################################
// AndroidExporter - Android platform exporter
// #############################################################################
class AndroidExporter : public PlatformExporter {
    XTU_GODOT_REGISTER_CLASS(AndroidExporter, PlatformExporter)

private:
    std::vector<String> m_binary_extensions = {"apk", "aab"};

public:
    static StringName get_class_static() { return StringName("AndroidExporter"); }

    AndroidExporter() {
        m_name = "Android";
        m_os_name = "Android";
    }

    String get_platform_name() const override { return "Android"; }
    String get_platform_arch() const override { return "arm64"; }

    std::vector<String> get_binary_extensions() const override { return m_binary_extensions; }

    std::vector<PropertyInfo> get_export_options() const override {
        std::vector<PropertyInfo> options;
        options.push_back({VariantType::STRING, "custom_template/debug"});
        options.push_back({VariantType::STRING, "custom_template/release"});
        options.push_back({VariantType::STRING, "package/unique_name"});
        options.push_back({VariantType::STRING, "package/name"});
        options.push_back({VariantType::BOOL, "package/signed"});
        options.push_back({VariantType::STRING, "package/keystore/debug"});
        options.push_back({VariantType::STRING, "package/keystore/release"});
        options.push_back({VariantType::STRING, "package/keystore/release_user"});
        options.push_back({VariantType::STRING, "package/keystore/release_password"});
        options.push_back({VariantType::INT, "version/code"});
        options.push_back({VariantType::STRING, "version/name"});
        options.push_back({VariantType::BOOL, "gradle_build/use_gradle"});
        options.push_back({VariantType::INT, "gradle_build/min_sdk"});
        options.push_back({VariantType::INT, "gradle_build/target_sdk"});
        return options;
    }

    void get_preset_features(const Ref<EditorExportPreset>& preset, std::vector<String>& features) const override {
        features.push_back("android");
        features.push_back("mobile");
        features.push_back("arm64");
    }

    Error export_project(const Ref<EditorExportPreset>& preset, bool debug, const String& path, uint32_t flags) override {
        String build_dir = OS::get_singleton()->get_cache_dir() + "/android_export";
        DirAccess::make_dir_recursive(build_dir);

        // Prepare Gradle project
        prepare_gradle_project(build_dir, preset, debug);

        // Build PCK
        String pck_path = build_dir + "/assets/godot.pck";
        export_pack(preset, debug, pck_path, flags);

        // Run Gradle
        String cmd = "cd \"" + build_dir.to_std_string() + "\" && ./gradlew ";
        cmd += debug ? "assembleDebug" : "assembleRelease";

        int result = system(cmd.utf8());
        if (result != 0) return ERR_COMPILATION_FAILED;

        // Copy output
        String apk_path = build_dir + "/app/build/outputs/apk/" + (debug ? "debug" : "release") +
                          "/app-" + (debug ? "debug" : "release") + ".apk";
        DirAccess::copy(apk_path, path);

        DirAccess::remove(build_dir);
        return OK;
    }

private:
    void prepare_gradle_project(const String& dir, const Ref<EditorExportPreset>& preset, bool debug) {
        // Implementation similar to xexporter.hpp
    }
};

// #############################################################################
// PlatformExportManager - Central export registration
// #############################################################################
class PlatformExportManager : public Object {
    XTU_GODOT_REGISTER_CLASS(PlatformExportManager, Object)

private:
    static PlatformExportManager* s_singleton;
    std::vector<Ref<PlatformExporter>> m_exporters;
    mutable std::mutex m_mutex;

public:
    static PlatformExportManager* get_singleton() { return s_singleton; }
    static StringName get_class_static() { return StringName("PlatformExportManager"); }

    PlatformExportManager() {
        s_singleton = this;
        register_default_exporters();
    }

    ~PlatformExportManager() { s_singleton = nullptr; }

    void register_exporter(const Ref<PlatformExporter>& exporter) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_exporters.push_back(exporter);
        EditorExport::get_singleton()->add_export_platform(exporter);
    }

    Ref<PlatformExporter> get_exporter(const String& platform) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& exp : m_exporters) {
            if (exp->get_platform_name() == platform) return exp;
        }
        return Ref<PlatformExporter>();
    }

private:
    void register_default_exporters() {
        register_exporter(Ref<PlatformExporter>(new WindowsExporter()));
        register_exporter(Ref<PlatformExporter>(new MacOSExporter()));
        register_exporter(Ref<PlatformExporter>(new AndroidExporter()));
    }
};

} // namespace platform_export

// Bring into main namespace
using platform_export::PlatformExporter;
using platform_export::WindowsExporter;
using platform_export::MacOSExporter;
using platform_export::AndroidExporter;
using platform_export::PlatformExportManager;
using platform_export::ExportTemplate;
using platform_export::ExportArchiveType;
using platform_export::CodeSignConfig;

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XPLATFORM_EXPORT_HPP